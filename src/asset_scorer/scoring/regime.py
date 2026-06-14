"""Market regime detection -- the alpha-saving filter.

The biggest reason a long-biased scorer bleeds is going long into a falling or
chaotic market. This module reads the *market itself* (the universe as an
equal-weight index) and labels each day:

    risk_on   - index in an uptrend, broad participation, calm volatility
    risk_off  - index below its trend, weak breadth, or a volatility spike
    neutral   - in between

Everything is trailing/rolling, so the label at date t uses only information
available at t (no look-ahead). FAVORED longs are gated to risk_on/neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import RegimeConfig

RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"


@dataclass
class RegimeState:
    label: str
    index_above_trend: bool
    breadth: float
    vol_z: float


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(3, n // 2)).mean()


def compute_regime(close: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    """Return a (dates) frame with columns: label, index, breadth, vol_z."""
    if close is None or close.empty:
        return pd.DataFrame(columns=["label", "index", "breadth", "vol_z"])

    # Equal-weight index from per-asset daily returns (robust to price scale).
    rets = close.pct_change()
    idx_ret = rets.mean(axis=1)
    index = (1.0 + idx_ret.fillna(0.0)).cumprod()

    sma_fast = _sma(index, cfg.index_fast)
    sma_slow = _sma(index, cfg.index_slow)
    above_trend = (index > sma_slow) & (sma_fast >= sma_slow)

    # Breadth: fraction of assets above their own breadth SMA.
    asset_sma = close.rolling(cfg.breadth_lookback, min_periods=cfg.breadth_lookback // 2).mean()
    breadth = (close > asset_sma).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)

    # Volatility stress on the index.
    rvol = idx_ret.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback // 2).std(ddof=0)
    vmean = rvol.rolling(cfg.vol_z_window, min_periods=cfg.vol_z_window // 2).mean()
    vstd = rvol.rolling(cfg.vol_z_window, min_periods=cfg.vol_z_window // 2).std(ddof=0)
    vol_z = ((rvol - vmean) / vstd.replace(0, np.nan)).fillna(0.0)

    out = pd.DataFrame({
        "index": index, "breadth": breadth.fillna(0.5), "vol_z": vol_z,
        "above_trend": above_trend.fillna(False),
    })

    def _label(row) -> str:
        if (not row["above_trend"]) or row["breadth"] < cfg.breadth_weak or row["vol_z"] > cfg.vol_hot_z:
            return RISK_OFF
        if row["above_trend"] and row["breadth"] > cfg.breadth_strong and row["vol_z"] <= cfg.vol_hot_z:
            return RISK_ON
        return NEUTRAL

    out["label"] = out.apply(_label, axis=1)
    return out[["label", "index", "breadth", "vol_z"]]


def regime_state_at(regime_df: pd.DataFrame, date) -> RegimeState:
    if regime_df is None or regime_df.empty:
        return RegimeState(NEUTRAL, True, 0.5, 0.0)
    if date in regime_df.index:
        row = regime_df.loc[date]
    else:
        valid = regime_df.loc[:date]
        if len(valid) == 0:
            return RegimeState(NEUTRAL, True, 0.5, 0.0)
        row = valid.iloc[-1]
    return RegimeState(
        label=str(row["label"]),
        index_above_trend=bool(regime_df.get("index") is not None),
        breadth=float(row["breadth"]),
        vol_z=float(row["vol_z"]),
    )


def longs_allowed(label: str) -> bool:
    """FAVORED longs are only sensible in risk_on / neutral regimes."""
    return label in (RISK_ON, NEUTRAL)


def same_regime_dates(
    regime_df: pd.DataFrame,
    label: str,
    upto=None,
    min_samples: int = 50,
):
    """Dates sharing the given regime label (optionally up to a cutoff).

    Returns ``None`` when there are too few same-regime observations, signalling
    the caller to fall back to all-history weighting. This is what makes the
    weights *regime-conditional*: in a risk-off market we learn factor weights
    from past risk-off markets, not from the whole history.
    """
    if regime_df is None or regime_df.empty or "label" not in regime_df:
        return None
    df = regime_df if upto is None else regime_df.loc[:upto]
    dates = df.index[df["label"] == label]
    if len(dates) < min_samples:
        return None
    return dates

