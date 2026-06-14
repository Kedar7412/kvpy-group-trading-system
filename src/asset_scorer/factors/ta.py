"""Vectorized technical-analysis primitives (pandas/numpy only).

Implemented by hand so there is no dependency on TA-Lib / pandas-ta and so the
exact, point-in-time (no lookahead) semantics are explicit and testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=max(2, period // 2)).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=max(2, period // 2)).mean()


def roc(s: pd.Series, period: int) -> pd.Series:
    """Rate of change over `period` bars (fractional)."""
    return s.pct_change(period)


def log_returns(s: pd.Series) -> pd.Series:
    return np.log(s).diff()


def rolling_zscore(s: pd.Series, period: int) -> pd.Series:
    mean = s.rolling(period, min_periods=max(2, period // 2)).mean()
    std = s.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When there are no losses RSI is 100; when no gains it is 0.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss != 0, 50.0))
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    close: pd.Series, period: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return lower, mid, upper


def bollinger_position(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """Where price sits within the bands: 0 = lower band, 1 = upper band."""
    lower, _, upper = bollinger(close, period, n_std)
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def realized_vol(close: pd.Series, period: int = 30) -> pd.Series:
    return log_returns(close).rolling(period, min_periods=max(2, period // 2)).std(ddof=0)


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """CMF: volume-weighted close location within the bar's range."""
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm * volume
    return mfv.rolling(period, min_periods=max(2, period // 2)).sum() / volume.rolling(
        period, min_periods=max(2, period // 2)
    ).sum()
