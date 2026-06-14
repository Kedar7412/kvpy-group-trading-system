"""Market data provider.

Pulls daily OHLCV history and a current order-book snapshot per symbol via
``ccxt``. If the network/exchange is unavailable, it deterministically
synthesizes plausible data so the whole scoring pipeline remains runnable and
reproducible (seeded per symbol).

The synthetic generator is intentionally *not* random noise: it builds assets
with distinct regimes (steady uptrend, parabolic blow-off, decaying downtrend,
range) so the scoring and anti-bubble logic can be exercised meaningfully.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DataConfig


@dataclass
class OrderBook:
    """A lightweight order-book snapshot (price, amount) levels."""

    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.bids or not self.asks

    @property
    def mid(self) -> float:
        if self.is_empty:
            return float("nan")
        return (self.bids[0][0] + self.asks[0][0]) / 2.0


@dataclass
class AssetData:
    """All raw inputs for one asset.

    ``ohlcv`` columns: open, high, low, close, volume, indexed by timestamp.
    ``meta`` carries provider-level info (e.g. whether data is synthetic).
    """

    symbol: str
    ohlcv: pd.DataFrame
    orderbook: OrderBook
    news_headlines: list[str] = field(default_factory=list)
    fundamentals: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))


def _seed_from_symbol(symbol: str) -> int:
    digest = hashlib.sha256(symbol.encode()).hexdigest()
    return int(digest[:8], 16)


class MarketDataProvider:
    """Fetches data via ccxt with a deterministic synthetic fallback."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._exchange = None
        self._init_error: str | None = None
        self.enrich_errors: list[str] = []

    # -- ccxt lifecycle ---------------------------------------------------
    def _get_exchange(self):
        if self._exchange is not None or self._init_error is not None:
            return self._exchange
        try:
            import ccxt  # imported lazily so the package works without network

            klass = getattr(ccxt, self.config.exchange)
            self._exchange = klass(
                {
                    "enableRateLimit": True,
                    "timeout": self.config.request_timeout_ms,
                }
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self._init_error = str(exc)
            self._exchange = None
        return self._exchange

    # -- public API -------------------------------------------------------
    def fetch(self, symbol: str) -> AssetData:
        """Fetch one asset, falling back to synthetic data on any failure."""
        data = self._fetch_real(symbol)
        if data is not None:
            return data
        if not self.config.use_synthetic_fallback:
            raise RuntimeError(
                f"Failed to fetch {symbol} and synthetic fallback disabled."
            )
        return self._synthesize(symbol)

    def fetch_universe(self, symbols: list[str]) -> dict[str, AssetData]:
        assets = {s: self.fetch(s) for s in symbols}
        # Skip network enrichment entirely if we're clearly offline (all synthetic).
        if assets and not all(a.synthetic for a in assets.values()):
            self._enrich(assets)
        return assets

    def _enrich(self, assets: dict[str, AssetData]) -> None:
        if not (self.config.enrich_fundamentals or self.config.enrich_news):
            return
        try:
            from .enrich import CryptoEnricher

            enr = CryptoEnricher(timeout=max(5, self.config.request_timeout_ms // 1000))
            enr.prepare([s.split("/")[0] for s in assets])
            for symbol, ad in assets.items():
                base = symbol.split("/")[0]
                try:
                    enr.enrich(
                        ad,
                        base,
                        want_fundamentals=self.config.enrich_fundamentals,
                        want_news=self.config.enrich_news,
                    )
                except Exception:
                    continue
            self.enrich_errors = enr.errors
        except Exception:
            # Enrichment is best-effort; never break the core pipeline.
            return

    # -- real fetch -------------------------------------------------------
    def _fetch_real(self, symbol: str) -> AssetData | None:
        ex = self._get_exchange()
        if ex is None:
            return None
        try:
            raw = ex.fetch_ohlcv(
                symbol,
                timeframe=self.config.timeframe,
                limit=self.config.history_limit,
            )
            if not raw:
                return None
            df = pd.DataFrame(
                raw, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")

            ob = OrderBook()
            try:
                book = ex.fetch_order_book(
                    symbol, limit=self.config.orderbook_depth_levels
                    if hasattr(self.config, "orderbook_depth_levels")
                    else 25,
                )
                ob = OrderBook(
                    bids=[(float(p), float(a)) for p, a, *_ in book.get("bids", [])],
                    asks=[(float(p), float(a)) for p, a, *_ in book.get("asks", [])],
                )
            except Exception:
                ob = self._synthetic_orderbook(symbol, float(df["close"].iloc[-1]))

            return AssetData(
                symbol=symbol,
                ohlcv=df,
                orderbook=ob,
                news_headlines=[],
                meta={"synthetic": False, "source": self.config.exchange,
                      "exchange": self.config.exchange},
            )
        except Exception:
            return None

    # -- synthetic fallback ----------------------------------------------
    def _synthesize(self, symbol: str) -> AssetData:
        rng = np.random.default_rng(_seed_from_symbol(symbol))
        n = self.config.history_limit

        # Pick a regime deterministically per symbol.
        regimes = ["uptrend", "bubble", "downtrend", "range", "recovery"]
        regime = regimes[_seed_from_symbol(symbol) % len(regimes)]

        drift, vol, shape = self._regime_params(regime)
        base_price = 10.0 + (_seed_from_symbol(symbol) % 5000) / 50.0

        # Build a log-price path = noisy random walk (drift) + smooth regime trend.
        t = np.linspace(0, 1, n)
        trend = shape(t)
        noise = rng.normal(0, vol, n)
        log_ret = drift / n + noise            # per-bar log returns
        log_path = np.cumsum(log_ret) + trend  # add deterministic regime shape
        close = base_price * np.exp(log_path - log_path[0])

        # Construct OHLC around close.
        intraday = np.abs(rng.normal(0, vol, n)) + 0.002
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = self._synthetic_volume(rng, close, regime)

        idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=n, freq="D")
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )

        ob = self._synthetic_orderbook(symbol, float(close[-1]), regime=regime)
        return AssetData(
            symbol=symbol,
            ohlcv=df,
            orderbook=ob,
            news_headlines=self._synthetic_news(symbol, regime),
            meta={"synthetic": True, "regime": regime},
        )

    @staticmethod
    def _regime_params(regime: str):
        """Return (drift, vol, shape_fn) for a regime."""
        if regime == "uptrend":
            return 0.6, 0.018, (lambda t: 0.5 * t)
        if regime == "bubble":
            # Accelerating parabolic rise then early roll-over.
            return 0.2, 0.035, (lambda t: 1.6 * np.power(t, 3))
        if regime == "downtrend":
            return -0.5, 0.025, (lambda t: -0.6 * t)
        if regime == "range":
            return 0.0, 0.02, (lambda t: 0.12 * np.sin(2 * np.pi * 3 * t))
        # recovery: down then up
        return 0.1, 0.022, (lambda t: 0.4 * (t - 0.5) ** 2 * np.sign(t - 0.4))

    @staticmethod
    def _synthetic_volume(rng, close, regime):
        n = len(close)
        base = rng.lognormal(mean=12, sigma=0.4, size=n)
        ret = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
        vol = base * (1 + 4 * ret)
        if regime == "bubble":
            # Volume climaxes near the top.
            vol *= 1 + 2 * np.linspace(0, 1, n) ** 2
        return vol

    def _synthetic_orderbook(
        self, symbol: str, mid: float, regime: str = "range"
    ) -> OrderBook:
        rng = np.random.default_rng(_seed_from_symbol(symbol) + 99)
        levels = getattr(self.config, "orderbook_depth_levels", 25)
        spread = mid * 0.0005
        # Imbalance reflects regime: uptrend/recovery = bid-heavy, bubble/down = ask-heavy.
        if regime in ("uptrend", "recovery"):
            bid_mult, ask_mult = 1.4, 0.9
        elif regime in ("bubble", "downtrend"):
            bid_mult, ask_mult = 0.85, 1.35
        else:
            bid_mult, ask_mult = 1.0, 1.0

        bids, asks = [], []
        for i in range(levels):
            bp = mid - spread / 2 - i * spread
            ap = mid + spread / 2 + i * spread
            decay = np.exp(-i / 8)
            bids.append((bp, float(rng.lognormal(3, 0.5) * bid_mult * decay)))
            asks.append((ap, float(rng.lognormal(3, 0.5) * ask_mult * decay)))
        return OrderBook(bids=bids, asks=asks)

    @staticmethod
    def _synthetic_news(symbol: str, regime: str) -> list[str]:
        base = symbol.split("/")[0]
        catalog = {
            "uptrend": [
                f"{base} adoption rises as network activity grows",
                f"Analysts upgrade {base} on improving fundamentals",
            ],
            "bubble": [
                f"{base} goes parabolic as retail FOMO surges",
                f"Social mentions of {base} hit all-time high amid hype",
            ],
            "downtrend": [
                f"{base} slides amid weakening demand",
                f"Outflows continue for {base} as sentiment sours",
            ],
            "range": [
                f"{base} trades sideways awaiting catalyst",
                f"{base} volatility compresses in tight range",
            ],
            "recovery": [
                f"{base} stabilizes after sell-off, builders return",
                f"On-chain metrics for {base} quietly improve",
            ],
        }
        return catalog.get(regime, [f"{base} market update"])
