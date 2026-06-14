"""Anti-bubble penalty: the "real assets, not FOMO" tilt.

A high composite score should require genuine confirmation. This module
multiplies the composite down when an asset shows the bubble signature -
overbought RSI, parabolic momentum, and/or extreme news heat - WITHOUT
fundamentals and orderflow backing it up.

penalty = 1 - strength * heat_intensity * confirmation_deficit * (1 - floor)

so the penalty only bites when something is hot AND unconfirmed; a hot move
that fundamentals + flow support is left essentially untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BubbleConfig


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _heat_intensity(
    rsi: pd.DataFrame, momentum_z: pd.DataFrame, heat: pd.DataFrame, cfg: BubbleConfig
) -> pd.DataFrame:
    h_rsi = _clip01((rsi - cfg.rsi_hot) / max(1e-9, 100.0 - cfg.rsi_hot))
    h_mom = _clip01((momentum_z - cfg.momentum_hot_z) / 2.0)
    h_hype = _clip01((heat - cfg.hype_hot) / max(1e-9, 100.0 - cfg.hype_hot))
    # Any single extreme can flag overheating -> take the worst signal.
    return _elementwise_max(h_rsi, h_mom, h_hype)


def _elementwise_max(*frames: pd.DataFrame) -> pd.DataFrame:
    out = frames[0].fillna(0.0)
    for f in frames[1:]:
        out = np.maximum(out, f.fillna(0.0))
    return out


def _confirmation_deficit(
    fundamentals_norm: pd.DataFrame, orderflow_norm: pd.DataFrame, cfg: BubbleConfig
) -> pd.DataFrame:
    confirm = (fundamentals_norm.fillna(50.0) + orderflow_norm.fillna(50.0)) / 2.0
    return _clip01((cfg.confirm_threshold - confirm) / max(1e-9, cfg.confirm_threshold))


def bubble_penalty_panel(
    rsi: pd.DataFrame,
    momentum_z: pd.DataFrame,
    heat: pd.DataFrame,
    fundamentals_norm: pd.DataFrame,
    orderflow_norm: pd.DataFrame,
    cfg: BubbleConfig,
) -> pd.DataFrame:
    """Return a (dates x symbols) multiplicative penalty in [floor, 1]."""
    intensity = _heat_intensity(rsi, momentum_z, heat, cfg)
    deficit = _confirmation_deficit(fundamentals_norm, orderflow_norm, cfg)
    intensity, deficit = intensity.align(deficit, join="outer")
    penalty = 1.0 - cfg.strength * intensity.fillna(0.0) * deficit.fillna(0.0) * (
        1.0 - cfg.penalty_floor
    )
    return penalty.clip(cfg.penalty_floor, 1.0)


@dataclass
class BubbleResult:
    symbol: str
    penalty: float
    heat_intensity: float
    confirmation_deficit: float
    rsi: float
    momentum_z: float
    news_heat: float
    label: str


def latest_bubble_breakdown(
    symbol: str,
    rsi: float,
    momentum_z: float,
    news_heat: float,
    fundamentals_norm: float,
    orderflow_norm: float,
    cfg: BubbleConfig,
) -> BubbleResult:
    """Explainable per-asset bubble assessment for 'today'."""
    h_rsi = _clip01((rsi - cfg.rsi_hot) / max(1e-9, 100.0 - cfg.rsi_hot))
    h_mom = _clip01((momentum_z - cfg.momentum_hot_z) / 2.0)
    h_hype = _clip01((news_heat - cfg.hype_hot) / max(1e-9, 100.0 - cfg.hype_hot))
    intensity = float(max(h_rsi, h_mom, h_hype))

    confirm = (np.nan_to_num(fundamentals_norm, nan=50.0) + np.nan_to_num(orderflow_norm, nan=50.0)) / 2.0
    deficit = float(_clip01((cfg.confirm_threshold - confirm) / max(1e-9, cfg.confirm_threshold)))

    penalty = float(
        np.clip(1.0 - cfg.strength * intensity * deficit * (1.0 - cfg.penalty_floor),
                cfg.penalty_floor, 1.0)
    )

    if intensity > 0.5 and deficit > 0.5:
        label = "bubble-risk: hot & unconfirmed"
    elif intensity > 0.5 and deficit <= 0.5:
        label = "hot but confirmed"
    elif intensity > 0.2:
        label = "warming"
    else:
        label = "normal"

    return BubbleResult(
        symbol=symbol,
        penalty=penalty,
        heat_intensity=intensity,
        confirmation_deficit=deficit,
        rsi=float(rsi),
        momentum_z=float(momentum_z),
        news_heat=float(news_heat),
        label=label,
    )
