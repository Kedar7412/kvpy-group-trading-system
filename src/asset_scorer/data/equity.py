"""Equity & commodity data provider (Yahoo Finance via yfinance).

Returns the exact same ``AssetData`` shape as the crypto provider so the entire
scoring engine is asset-class agnostic. For each symbol it pulls:

  * OHLCV history (daily, adjusted).
  * Real fundamentals from ``Ticker.info`` (valuation, profitability, growth,
    leverage, FCF yield), condensed into the same ``snapshot_score`` the
    Fundamentals factor consumes.
  * Recent news headlines from ``Ticker.news``.

Equities have no free L2 order book, so ``orderbook`` is left empty and the
orderflow factor falls back to its OHLCV-derived flow proxy. Commodity futures
(e.g. ``GC=F``) work through the same path; they simply have sparse
fundamentals, so their fundamentals score leans on the OHLCV proxy.

Any failure per symbol falls back to deterministic synthetic data so the
pipeline always runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from .provider import AssetData, MarketDataProvider, OrderBook


class EquityProvider:
    def __init__(self, config: DataConfig):
        self.config = config
        self._synth = MarketDataProvider(config)  # for deterministic fallback
        self.errors: list[str] = []

    # -- public API -------------------------------------------------------
    def fetch(self, symbol: str) -> AssetData:
        data = self._fetch_real(symbol)
        if data is not None:
            return data
        if not self.config.use_synthetic_fallback:
            raise RuntimeError(f"Failed to fetch {symbol} and synthetic disabled.")
        return self._synth._synthesize(symbol)

    def fetch_universe(self, symbols: list[str]) -> dict[str, AssetData]:
        return {s: self.fetch(s) for s in symbols}

    # -- real fetch -------------------------------------------------------
    def _period_for(self) -> str:
        years = max(1, min(10, int(np.ceil(self.config.history_limit / 252 * 1.5))))
        return f"{years}y"

    def _fetch_real(self, symbol: str) -> AssetData | None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=self._period_for(), interval="1d", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            df = hist.rename(
                columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                }
            )[["open", "high", "low", "close", "volume"]]
            df = df.tail(self.config.history_limit)
            df.index = pd.to_datetime(df.index).tz_localize(None)

            fundamentals = self._extract_fundamentals(ticker)
            headlines = self._extract_news(ticker)

            meta = {"synthetic": False, "source": "yahoo"}
            if fundamentals:
                meta["fundamentals_source"] = "yfinance"
            if headlines:
                meta["news_source"] = "yfinance"

            return AssetData(
                symbol=symbol,
                ohlcv=df,
                orderbook=OrderBook(),  # no free L2 for equities/commodities
                news_headlines=headlines,
                fundamentals=fundamentals,
                meta=meta,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            self.errors.append(f"{symbol}:{type(exc).__name__}")
            return None

    # -- fundamentals -----------------------------------------------------
    def _extract_fundamentals(self, ticker) -> dict:
        try:
            info = ticker.info or {}
        except Exception:
            return {}
        if not info:
            return {}

        pe = info.get("forwardPE") or info.get("trailingPE")
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        margins = info.get("profitMargins")
        roe = info.get("returnOnEquity")
        rev_growth = info.get("revenueGrowth")
        earn_growth = info.get("earningsGrowth")
        d_to_e = info.get("debtToEquity")

        snap = {
            "pe": pe,
            "free_cash_flow": fcf,
            "market_cap_usd": mcap,
            "profit_margins": margins,
            "return_on_equity": roe,
            "revenue_growth": rev_growth,
            "earnings_growth": earn_growth,
            "debt_to_equity": d_to_e,
            "sector": info.get("sector"),
        }
        snap["snapshot_score"] = self._equity_snapshot_score(snap)
        return snap

    @staticmethod
    def _equity_snapshot_score(s: dict) -> float:
        """Condense equity fundamentals into ~z-units (higher = more real value).

        Rewards cheap valuation, strong cash generation, profitability, healthy
        (not hype) growth, and low leverage.
        """
        parts: list[float] = []

        pe = s.get("pe")
        if pe and pe > 0:
            ey = 1.0 / pe  # earnings yield; 5% neutral, 8%+ cheap
            parts.append(float(np.clip((ey - 0.05) / 0.05, -1.0, 1.0)))

        fcf, mcap = s.get("free_cash_flow"), s.get("market_cap_usd")
        if fcf and mcap and mcap > 0:
            fcf_yield = fcf / mcap
            parts.append(float(np.clip((fcf_yield - 0.03) / 0.04, -1.0, 1.0)))

        margins = s.get("profit_margins")
        if margins is not None:
            parts.append(float(np.clip((margins - 0.10) / 0.15, -1.0, 1.0)))

        roe = s.get("return_on_equity")
        if roe is not None:
            parts.append(float(np.clip((roe - 0.10) / 0.20, -1.0, 1.0)))

        growth = s.get("revenue_growth")
        if growth is None:
            growth = s.get("earnings_growth")
        if growth is not None:
            # Healthy growth is good; cap so hyper-growth hype can't dominate.
            parts.append(float(np.clip((growth - 0.05) / 0.15, -1.0, 1.0)))

        d_to_e = s.get("debt_to_equity")
        if d_to_e is not None:
            # yfinance reports D/E as a percentage (e.g. 80 == 0.8x).
            ratio = d_to_e / 100.0
            parts.append(float(np.clip((1.0 - ratio) / 1.0, -1.0, 1.0)))

        if not parts:
            return 0.0
        return float(np.mean(parts))

    # -- news -------------------------------------------------------------
    @staticmethod
    def _extract_news(ticker, limit: int = 12) -> list[str]:
        try:
            items = ticker.news or []
        except Exception:
            return []
        titles: list[str] = []
        for it in items:
            # yfinance changed the schema: title may be top-level or under content.
            title = it.get("title")
            if not title and isinstance(it.get("content"), dict):
                title = it["content"].get("title")
            if title:
                titles.append(title)
        return titles[:limit]
