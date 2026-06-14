"""Fundamentals factor (crypto proxy): real-value / quality, anti-bubble tilt.

True crypto fundamentals need on-chain & protocol-revenue data (pluggable via a
provider). The MVP proxies "realness" from market structure:

  * Liquidity / depth  -> deep, tradable markets are higher quality.
  * Organic volume     -> steady volume is healthier than FOMO volume spikes.
  * Valuation anchor   -> price far above its long-run anchor (e.g. 200-bar SMA)
                          is an overextension penalty (the core "real not bubble"
                          tilt). Reasonable premiums are fine; blow-offs are not.
  * Accumulation       -> sustained money flow confirms genuine demand.

Exposes ``overextension`` so reporting can show *why* a hot asset is discounted.
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

        # Liquidity: traded value in quote currency.
        quote_volume = close * volume
        liquidity = np.log1p(quote_volume).rolling(
            cfg.liquidity_lookback, min_periods=5
        ).mean()

        # Organic volume: penalize bursts (std/mean of volume); lower = healthier.
        vol_mean = volume.rolling(cfg.liquidity_lookback, min_periods=5).mean()
        vol_std = volume.rolling(cfg.liquidity_lookback, min_periods=5).std(ddof=0)
        burstiness = vol_std / vol_mean.replace(0, np.nan)

        # Valuation anchor: premium of price over a long-run mean.
        anchor = ta.sma(close, min(200, max(50, cfg.liquidity_lookback * 4)))
        premium = close / anchor - 1.0
        overextension = premium.clip(lower=0.0)  # only premiums are penalized

        # Accumulation: sustained money flow.
        cmf = ta.chaikin_money_flow(o["high"], o["low"], close, volume, 30)

        liq_z = ta.rolling_zscore(liquidity, Z_WIN)
        organic_z = -ta.rolling_zscore(burstiness.fillna(0.0), Z_WIN)
        overext_z = -ta.rolling_zscore(overextension.fillna(0.0), Z_WIN)
        accum_z = ta.rolling_zscore(cmf, Z_WIN)

        score = 0.30 * liq_z + 0.20 * organic_z + 0.30 * overext_z + 0.20 * accum_z

        # Blend the real fundamentals snapshot (CoinGecko) into today's value.
        snap = (data.fundamentals or {}).get("snapshot_score")
        if snap is not None and len(score):
            valid = score.dropna()
            if len(valid):
                score = score.copy()
                score.iloc[-1] = valid.iloc[-1] + 0.75 * float(snap)

        return {
            "score": score,
            "overextension": overextension,
            "premium": premium,
        }
