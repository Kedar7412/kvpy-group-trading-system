"""Technicals factor: trend + quality of trend, with a mean-reversion guard.

The key fix: pure momentum (ROC) chases what already moved and buys tops.
Instead we reward:
  * Trend *direction* (above SMA) but not magnitude,
  * Low volatility (contained, clean trend) not a chaotic parabolic,
  * A **mean-reversion guard**: heavily penalize extreme positive momentum
    (the exhaustion zone) and reward moderate pullbacks within uptrends.

Exposes ``momentum_z`` for the anti-bubble detector.
"""

from __future__ import annotations

import numpy as np
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

        # Trend direction: is the asset above its slow SMA? Binary signal.
        trend_dir = (close > sma_slow).astype(float) + (sma_fast > sma_slow).astype(float)
        trend_dir = trend_dir / 2.0  # [0, 1]: 0 = below both, 1 = above both

        # Momentum: ROC over lookback.
        momentum = ta.roc(close, cfg.momentum_lookback)

        # Mean-reversion guard: extreme positive momentum is DANGEROUS, not good.
        # Reward moderate positive (0-3% over lookback); penalize >5% as exhausted.
        # Moderate negatives within a trend are potential entries.
        mom_quality = -((momentum - 0.02).clip(lower=0.0)) * 3.0  # exhaustion penalty
        mom_quality = mom_quality + ((-momentum).clip(lower=0.0, upper=0.05)) * 2.0  # pullback reward

        vol = ta.realized_vol(close, cfg.vol_lookback)

        trend_z = ta.rolling_zscore(trend_dir, Z_WIN)
        mom_q_z = ta.rolling_zscore(mom_quality, Z_WIN)
        low_vol_z = -ta.rolling_zscore(vol, Z_WIN)

        # Removed pure momentum; replaced with momentum-quality (anti-chasing).
        score = 0.35 * trend_z + 0.40 * mom_q_z + 0.25 * low_vol_z

        return {
            "score": score,
            "momentum_z": ta.rolling_zscore(momentum, cfg.vol_lookback),
            "trend": trend_dir,
        }
