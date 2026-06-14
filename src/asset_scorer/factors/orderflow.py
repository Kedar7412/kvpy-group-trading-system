"""Orderflow factor: who is actually buying.

Historical order books are not freely available, so the backtestable time
series is a flow *proxy* built from OHLCV: Chaikin Money Flow (intrabar
accumulation/distribution) plus the slope of On-Balance-Volume. When a live
order-book snapshot is present, its top-of-book imbalance nudges the latest
value so today's score reflects real depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FactorConfig
from ..data import AssetData
from . import ta

Z_WIN = 90


class Orderflow:
    name = "orderflow"

    def compute(self, data: AssetData, cfg: FactorConfig) -> dict[str, pd.Series]:
        o = data.ohlcv
        cmf = ta.chaikin_money_flow(o["high"], o["low"], o["close"], o["volume"], 20)
        obv = ta.on_balance_volume(o["close"], o["volume"])
        obv_slope = obv.diff(10) / o["volume"].rolling(10, min_periods=3).mean()

        cmf_z = ta.rolling_zscore(cmf, Z_WIN)
        obv_z = ta.rolling_zscore(obv_slope, Z_WIN)
        score = 0.5 * cmf_z + 0.5 * obv_z

        imbalance = self._book_imbalance(data, cfg)
        if not np.isnan(imbalance) and len(score) > 0:
            # Blend live depth imbalance (in z-like units) into today's value.
            score = score.copy()
            last_valid = score.dropna()
            if len(last_valid):
                score.iloc[-1] = last_valid.iloc[-1] + 0.75 * imbalance

        return {
            "score": score,
            "cmf": cmf,
            "book_imbalance": pd.Series(imbalance, index=[o.index[-1]]),
        }

    @staticmethod
    def _book_imbalance(data: AssetData, cfg: FactorConfig) -> float:
        ob = data.orderbook
        if ob is None or ob.is_empty:
            return float("nan")
        levels = cfg.orderbook_depth_levels
        bid_vol = sum(a for _, a in ob.bids[:levels])
        ask_vol = sum(a for _, a in ob.asks[:levels])
        total = bid_vol + ask_vol
        if total <= 0:
            return float("nan")
        # In [-1, 1]: +1 fully bid-supported, -1 fully ask-heavy.
        return (bid_vol - ask_vol) / total
