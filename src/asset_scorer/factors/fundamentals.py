"""Fundamentals factor: real-value / quality, anti-momentum tilt.

This factor is deliberately **contrarian to momentum**: it rewards assets that
are cheap relative to their history (drawdown from highs), have organic/steady
volume (not FOMO spikes), and penalizes overextension (price stretched far above
anchor). The goal is to find things with real value that *haven't already run*.

When real fundamentals (CoinGecko snapshot or yfinance financials) are present,
they directly contribute to today's score with heavy weight.

The key insight from the live scorecard failure: the old version rewarded
liquidity + accumulation, which are *momentum proxies* (things going up).
This made FAVORED = "buy what already ran" = guaranteed mean-reversion trap.
Now FAVORED = "real value at a fair/cheap price."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FactorConfig
from ..data import AssetData
from . import ta

Z_WIN = 120


class Fundamentals:
    name = "fundamentals"

    def compute(self, data: AssetData, cfg: FactorConfig) -> dict[str, pd.Series]:
        o = data.ohlcv
        close, volume = o["close"], o["volume"]

        # 1. VALUATION: drawdown from rolling high = how "cheap" vs recent peak.
        #    Deep drawdown = potential value; at-or-above high = extended.
        rolling_high = close.rolling(200, min_periods=40).max()
        drawdown = close / rolling_high - 1.0  # in [-1, 0], deeper = cheaper
        # Invert: deeper drawdown -> higher score (undervalued).
        cheapness = -drawdown  # in [0, 1], higher = more undervalued

        # 2. OVEREXTENSION PENALTY: price stretched above a long-run anchor.
        anchor = ta.sma(close, min(200, max(50, cfg.liquidity_lookback * 4)))
        premium = (close / anchor - 1.0).clip(lower=0.0)
        overextension = premium

        # 3. ORGANIC VOLUME: penalize volume bursts (FOMO spikes).
        vol_mean = volume.rolling(cfg.liquidity_lookback, min_periods=5).mean()
        vol_std = volume.rolling(cfg.liquidity_lookback, min_periods=5).std(ddof=0)
        burstiness = vol_std / vol_mean.replace(0, np.nan)

        # 4. MEAN-REVERSION SETUP: RSI-like oversold detection.
        roc_20 = ta.roc(close, 20)
        oversold = (-roc_20).clip(lower=0.0)  # positive when price fell recently

        # Combine: reward cheap + organic + not overextended + oversold.
        cheap_z = ta.rolling_zscore(cheapness, Z_WIN)
        overext_z = -ta.rolling_zscore(overextension.fillna(0.0), Z_WIN)
        organic_z = -ta.rolling_zscore(burstiness.fillna(0.0), Z_WIN)
        oversold_z = ta.rolling_zscore(oversold.fillna(0.0), Z_WIN)

        score = (
            0.35 * cheap_z +        # undervalued vs history
            0.30 * overext_z +       # not stretched above anchor
            0.15 * organic_z +       # steady, not spiky volume
            0.20 * oversold_z        # recent pullback = potential entry
        )

        # Blend the real fundamentals snapshot (CoinGecko/yfinance) into today.
        # Real fundamentals get HEAVY weight because they're the actual signal.
        snap = (data.fundamentals or {}).get("snapshot_score")
        if snap is not None and len(score):
            valid = score.dropna()
            if len(valid):
                score = score.copy()
                # Real fundamentals dominate: weight 1.5x the OHLCV proxy.
                score.iloc[-1] = 0.4 * valid.iloc[-1] + 1.5 * float(snap)

        return {
            "score": score,
            "overextension": overextension,
            "premium": premium,
        }
