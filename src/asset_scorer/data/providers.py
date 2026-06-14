"""Provider factory: choose a data source by asset class.

All providers expose ``fetch_universe(symbols) -> dict[str, AssetData]`` so the
scoring engine never needs to know which asset class it is scoring.
"""

from __future__ import annotations

from typing import Protocol

from ..config import DataConfig
from .equity import EquityProvider
from .provider import AssetData, MarketDataProvider


class Provider(Protocol):
    def fetch_universe(self, symbols: list[str]) -> dict[str, AssetData]: ...


def get_provider(asset_class: str, config: DataConfig) -> Provider:
    """Return the right provider for the asset class.

    crypto             -> ccxt exchanges + CoinGecko/RSS enrichment
    equity / commodity -> Yahoo Finance (yfinance)
    """
    ac = (asset_class or "crypto").lower()
    if ac == "crypto":
        return MarketDataProvider(config)
    if ac in ("equity", "stock", "stocks", "commodity", "commodities", "futures"):
        return EquityProvider(config)
    raise ValueError(f"Unknown asset_class: {asset_class!r}")
