"""Indicators factor: classic TA oscillators (RSI, MACD, Bollinger).

Rewards positive MACD momentum and constructive RSI, but explicitly penalizes
overbought extremes. Exposes raw ``rsi`` for the anti-bubble logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FactorConfig
from ..data import AssetData
from . import ta

Z_WIN = 90


class Indicators:
    name = "indicators"

    def compute(self, data: AssetData, cfg: FactorConfig) -> dict[str, pd.Series]:
        close = data.ohlcv["close"]

        rsi = ta.rsi(close, cfg.rsi_period)
        macd_line, _, hist = ta.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        bb_pos = ta.bollinger_position(close, cfg.bb_period, cfg.bb_std)

        # RSI signal: reward the 50-70 zone, penalize >70 (overbought) and <30.
        rsi_signal = (rsi - 50.0) / 50.0
        overbought = (rsi - 70.0).clip(lower=0.0) / 30.0
        rsi_score = rsi_signal - overbought

        macd_norm = hist / close  # scale-free
        # Bollinger: above the upper band (bb_pos > 1) is a stretch -> penalize.
        bb_stretch = (bb_pos - 1.0).clip(lower=0.0)

        macd_z = ta.rolling_zscore(macd_norm, Z_WIN)
        rsi_z = ta.rolling_zscore(rsi_score, Z_WIN)
        stretch_z = -ta.rolling_zscore(bb_stretch.fillna(0.0), Z_WIN)

        score = 0.45 * macd_z + 0.35 * rsi_z + 0.20 * stretch_z

        return {
            "score": score,
            "rsi": rsi,
            "bb_pos": bb_pos,
        }
