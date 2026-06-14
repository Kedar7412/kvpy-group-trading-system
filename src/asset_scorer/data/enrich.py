"""Real-data enrichment for crypto assets (free, keyless sources).

Two enrichments, both best-effort with graceful fallback (any failure leaves
the asset's OHLCV-derived signals untouched):

  * Fundamentals  - CoinGecko market/supply data, condensed into a single
                    ``snapshot_score`` (roughly z-units, higher = more real
                    value): liquidity (volume/mcap turnover) and low future
                    dilution (circulating/max supply). Fetched for the whole
                    universe in ONE batched ``/coins/markets`` call to respect
                    the free-tier rate limit.
  * News          - real headlines pulled from crypto RSS feeds, matched per
                    asset with word-boundary keywords (so "sol" doesn't match
                    "con-sol-idation"), then scored by the News factor lexicon.

Only the standard library is used for HTTP/XML so there is no extra dependency
and the package keeps working offline.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

_UA = {"User-Agent": "asset-scorer/0.1 (+research)"}
_COINGECKO = "https://api.coingecko.com/api/v3"

# Curated base-symbol -> CoinGecko id. Avoids ambiguous symbol collisions.
SYMBOL_TO_ID: dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche-2",
    "LINK": "chainlink", "MATIC": "matic-network", "POL": "polygon-ecosystem-token",
    "DOT": "polkadot", "TRX": "tron", "LTC": "litecoin", "BCH": "bitcoin-cash",
    "ATOM": "cosmos", "UNI": "uniswap", "XLM": "stellar", "ETC": "ethereum-classic",
    "NEAR": "near", "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
    "FIL": "filecoin", "ICP": "internet-computer", "HBAR": "hedera-hashgraph",
    "VET": "vechain", "ALGO": "algorand", "AAVE": "aave", "INJ": "injective-protocol",
    "SUI": "sui", "TON": "the-open-network", "SHIB": "shiba-inu", "PEPE": "pepe",
}

# Base-symbol -> keywords used to filter news headlines.
NEWS_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"], "ETH": ["ethereum", "ether", "eth"],
    "SOL": ["solana", "sol"], "BNB": ["bnb", "binance coin"], "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"], "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"], "LINK": ["chainlink", "link"],
    "MATIC": ["polygon", "matic"], "POL": ["polygon", "pol"], "DOT": ["polkadot", "dot"],
    "LTC": ["litecoin", "ltc"], "TRX": ["tron", "trx"], "ATOM": ["cosmos", "atom"],
    "UNI": ["uniswap", "uni"], "NEAR": ["near protocol"], "ARB": ["arbitrum", "arb"],
    "OP": ["optimism"], "SUI": ["sui network", "sui"], "TON": ["toncoin", "ton"],
    "SHIB": ["shiba inu", "shib"], "PEPE": ["pepe"], "APT": ["aptos", "apt"],
}

DEFAULT_NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]


def _http_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


@dataclass
class _NewsItem:
    title: str
    summary: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.summary}".lower()


class CryptoEnricher:
    """Fetches and caches real fundamentals + news, then enriches assets."""

    def __init__(self, timeout: int = 15, news_feeds: list[str] | None = None):
        self.timeout = timeout
        self.news_feeds = news_feeds or DEFAULT_NEWS_FEEDS
        self._news_cache: list[_NewsItem] | None = None
        self._markets: dict[str, dict] = {}  # coin_id -> markets row
        self._markets_loaded = False
        self.errors: list[str] = []

    # -- preparation ------------------------------------------------------
    def prepare(self, bases: list[str]) -> None:
        """Batch-load market fundamentals for the whole universe in one call."""
        self._load_markets(bases)

    def _load_markets(self, bases: list[str]) -> None:
        if self._markets_loaded:
            return
        self._markets_loaded = True
        ids = sorted({SYMBOL_TO_ID[b.upper()] for b in bases if b.upper() in SYMBOL_TO_ID})
        if not ids:
            return
        params = urllib.parse.urlencode(
            {
                "vs_currency": "usd",
                "ids": ",".join(ids),
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "",
            }
        )
        url = f"{_COINGECKO}/coins/markets?{params}"
        try:
            import json

            rows = json.loads(_http_get(url, self.timeout))
            for row in rows:
                self._markets[row["id"]] = row
        except Exception as exc:  # pragma: no cover - network dependent
            self.errors.append(f"markets:{type(exc).__name__}")

    # -- news -------------------------------------------------------------
    def _load_news(self) -> list[_NewsItem]:
        if self._news_cache is not None:
            return self._news_cache
        items: list[_NewsItem] = []
        for feed in self.news_feeds:
            try:
                raw = _http_get(feed, self.timeout)
                items.extend(self._parse_rss(raw))
            except Exception as exc:  # pragma: no cover - network dependent
                self.errors.append(f"news:{feed}:{type(exc).__name__}")
        self._news_cache = items
        return items

    @staticmethod
    def _parse_rss(raw: bytes) -> list[_NewsItem]:
        out: list[_NewsItem] = []
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return out
        # RSS 2.0: channel/item with title + description.
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            if title:
                out.append(_NewsItem(title=title, summary=desc))
        # Atom fallback: entry/title + summary.
        if not out:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
                summ = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
                if title:
                    out.append(_NewsItem(title=title, summary=summ))
        return out

    def headlines_for(self, base: str, limit: int = 12) -> list[str]:
        keywords = NEWS_KEYWORDS.get(base.upper(), [base.lower()])
        # Word-boundary match so short tickers (sol, op, ton) don't match
        # substrings inside unrelated words ("con-sol-idation").
        patterns = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords]
        items = self._load_news()
        matched = [it.title for it in items if any(p.search(it.text) for p in patterns)]
        return matched[:limit]

    # -- fundamentals -----------------------------------------------------
    def fundamentals_for(self, base: str) -> dict:
        coin_id = SYMBOL_TO_ID.get(base.upper())
        if coin_id is None:
            return {}
        if not self._markets_loaded:
            self._load_markets([base])
        row = self._markets.get(coin_id)
        if not row:
            return {}

        snapshot = {
            "coin_id": coin_id,
            "market_cap_usd": row.get("market_cap"),
            "total_volume_usd": row.get("total_volume"),
            "circulating_supply": row.get("circulating_supply"),
            "total_supply": row.get("total_supply"),
            "max_supply": row.get("max_supply"),
            "market_cap_rank": row.get("market_cap_rank"),
            "ath_change_pct": row.get("ath_change_percentage"),
        }
        snapshot["snapshot_score"] = self._fundamental_snapshot_score(snapshot)
        return snapshot

    @staticmethod
    def _fundamental_snapshot_score(s: dict) -> float:
        """Condense fundamentals into ~z-units (higher = more real value)."""
        parts: list[float] = []

        mc, vol = s.get("market_cap_usd"), s.get("total_volume_usd")
        if mc and vol and mc > 0:
            liq = vol / mc  # turnover; healthy markets trade a real fraction daily
            # ~0.02 typical, 0.2 very active. log10 maps to roughly [-1, 1].
            parts.append(float(np.clip((np.log10(liq + 1e-9) + 1.7) / 1.0, -1.0, 1.0)))

        circ = s.get("circulating_supply")
        denom = s.get("max_supply") or s.get("total_supply")
        if circ and denom and denom > 0:
            float_ratio = min(circ / denom, 1.0)  # high => low future dilution
            parts.append(float(np.clip((float_ratio - 0.5) * 2.0, -1.0, 1.0)))

        if not parts:
            return 0.0
        return float(np.mean(parts))

    # -- orchestration ----------------------------------------------------
    def enrich(self, asset, base: str, want_fundamentals=True, want_news=True) -> None:
        if want_news:
            heads = self.headlines_for(base)
            if heads:
                asset.news_headlines = heads
                asset.meta["news_source"] = "rss"
        if want_fundamentals:
            fund = self.fundamentals_for(base)
            if fund:
                asset.fundamentals = fund
                asset.meta["fundamentals_source"] = "coingecko"
