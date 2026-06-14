"""Cross-sectional normalization.

Turns each factor's raw signal into a comparable 0-100 score by ranking assets
*against each other on the same date*. We use a robust cross-sectional z-score
(clipped to +/-3 to tame outliers) squashed through a logistic to land in
[0, 100]. 50 means "average asset today"; 100 means "best in the universe".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_score(panel: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Normalize a (dates x symbols) factor panel to 0-100 per row.

    Rows with fewer than 2 valid values are left as NaN (cannot rank a single
    asset against the cross-section).
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0)
    z = panel.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    z = z.clip(-clip, clip)
    scored = 100.0 / (1.0 + np.exp(-z))

    # Where the cross-section is degenerate (0/1 assets or zero variance), use 50.
    valid_counts = panel.notna().sum(axis=1)
    degenerate = (valid_counts < 2) | std.isna() | (std == 0)
    scored.loc[degenerate] = scored.loc[degenerate].fillna(50.0)
    scored = scored.mask(panel.isna())  # keep NaN where the factor itself is NaN
    return scored
