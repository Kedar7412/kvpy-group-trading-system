"""Flexible, IC-driven factor weights.

The weight a factor receives is proportional to its measured predictive power
(Information Coefficient = rank correlation between the factor score and the
forward return), blended toward neutral priors so weights stay stable when
history is short.

Two IC views are combined so weights are *per-asset flexible*:
  * global IC  - how well the factor works across the whole universe.
  * asset IC   - how well the factor works for that specific asset.

This is why news can dominate a hype-driven coin while fundamentals dominate a
mature one: each asset's weight vector reflects what has actually predicted its
returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import WeightConfig


@dataclass
class FlexibleWeights:
    """Per-(factor, symbol) weights plus the IC diagnostics behind them."""

    weights: pd.DataFrame          # index=factors, columns=symbols, sums to 1 per col
    global_ic: pd.Series           # index=factors
    asset_ic: pd.DataFrame         # index=factors, columns=symbols


def _spearman(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 8 or df.iloc[:, 0].nunique() < 3 or df.iloc[:, 1].nunique() < 3:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def information_coefficients(
    factor_scores: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute global and per-asset IC for each factor.

    ``factor_scores[name]`` and ``forward_returns`` are both (dates x symbols),
    already time-aligned (factor at t, return realized over t -> t+H).
    """
    factors = list(factor_scores)
    symbols = list(forward_returns.columns)

    asset_ic = pd.DataFrame(index=factors, columns=symbols, dtype=float)
    global_ic = pd.Series(index=factors, dtype=float)

    for f in factors:
        panel = factor_scores[f].reindex(columns=symbols)
        # Per-asset IC: time-series rank corr of the factor vs that asset's return.
        for s in symbols:
            asset_ic.loc[f, s] = _spearman(panel[s], forward_returns[s])
        # Global IC: pool every (date, symbol) observation.
        flat_f = panel.stack(future_stack=True)
        flat_r = forward_returns.reindex(columns=symbols).stack(future_stack=True)
        global_ic[f] = _spearman(flat_f, flat_r)

    return global_ic, asset_ic


def compute_flexible_weights(
    factor_scores: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    cfg: WeightConfig,
) -> FlexibleWeights:
    factors = list(factor_scores)
    symbols = list(forward_returns.columns)
    global_ic, asset_ic = information_coefficients(factor_scores, forward_returns)

    priors = pd.Series({f: cfg.priors.get(f, 1.0) for f in factors}, dtype=float)
    priors = priors / priors.sum()

    weights = pd.DataFrame(index=factors, columns=symbols, dtype=float)
    for s in symbols:
        # Blend asset IC with global IC (asset IC is noisier, so down-weight it
        # when missing); clamp negatives (don't actively bet against a factor).
        a_ic = asset_ic[s]
        blended = pd.Series(index=factors, dtype=float)
        for f in factors:
            g = global_ic[f] if np.isfinite(global_ic[f]) else 0.0
            a = a_ic[f] if np.isfinite(a_ic[f]) else g
            blended[f] = 0.5 * a + 0.5 * g
        signal = blended.clip(lower=cfg.ic_floor).fillna(0.0)

        # prior_strength acts as a pseudo-count anchoring to neutral priors.
        raw = cfg.prior_strength * priors + signal
        if raw.sum() <= 0:
            raw = priors.copy()
        w = raw / raw.sum()
        w = w.clip(lower=cfg.min_weight)
        w = w / w.sum()
        weights[s] = w

    return FlexibleWeights(weights=weights, global_ic=global_ic, asset_ic=asset_ic)
