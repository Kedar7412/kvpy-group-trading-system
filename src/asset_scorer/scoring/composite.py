"""Composite scoring: weighted blend of normalized factors + anti-bubble.

base_score(date, symbol) = sum_f weight(f, symbol) * normalized_factor(f)
final_score             = base_score * bubble_penalty   (still 0-100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def composite_panel(
    normalized_factors: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    bubble_penalty: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (base_score_panel, final_score_panel), both (dates x symbols).

    ``weights`` is (factors x symbols); each column sums to 1. Per-asset weights
    are broadcast across all dates (weights are estimated from full history and
    applied uniformly within a run; re-running daily re-estimates them).
    """
    factors = list(normalized_factors)
    # Reference index/columns from the first factor panel.
    ref = normalized_factors[factors[0]]
    base = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    weight_sum = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)

    for f in factors:
        panel = normalized_factors[f].reindex(index=ref.index, columns=ref.columns)
        w_row = weights.loc[f].reindex(ref.columns)
        # Broadcast per-symbol weight across dates; ignore NaN factor cells.
        contrib = panel.mul(w_row, axis=1)
        mask = panel.notna()
        base = base.add(contrib.fillna(0.0))
        weight_sum = weight_sum.add(mask.mul(w_row, axis=1).fillna(0.0))

    # Renormalize by the weight actually present (handles missing factor data).
    base = base.div(weight_sum.replace(0, np.nan))

    if bubble_penalty is None:
        final = base.copy()
    else:
        pen = bubble_penalty.reindex(index=base.index, columns=base.columns).fillna(1.0)
        final = base * pen

    return base, final
