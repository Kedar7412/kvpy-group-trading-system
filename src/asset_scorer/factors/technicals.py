"""Technicals factor: trend structure + momentum, penalized by volatility.

Rewards assets in healthy uptrends with positive momentum and contained
volatility. Exposes ``momentum_z`` (parabolic detector) for the anti-bubble
logic.
"""

from __future__ import annotations

import pandas as pd

from ..config import FactorConfig
from ..data import AssetData
from . import ta

Z_WIN = 90


class Technicals:
    name = "technicals"

    def compute(self, data: AssetData, cfg: FactorConfig) -> dict[str, pd.Series]:
        close = data.ohlcv["close"]
        sma_fast = ta.sma(close, cfg.trend_fast)
        sma_slow = ta.sma(close, cfg.trend_slow)

        trend = (close / sma_slow - 1.0) + (sma_fast / sma_slow - 1.0)
        momentum = ta.roc(close, cfg.momentum_lookback)
        vol = ta.realized_vol(close, cfg.vol_lookback)

        trend_z = ta.rolling_zscore(trend, Z_WIN)
        mom_z = ta.rolling_zscore(momentum, Z_WIN)
        low_vol_z = -ta.rolling_zscore(vol, Z_WIN)

        score = 0.40 * trend_z + 0.40 * mom_z + 0.20 * low_vol_z

        return {
            "score": score,
            "momentum_z": ta.rolling_zscore(momentum, cfg.vol_lookback),
            "trend": trend,
        }
