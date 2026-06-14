"""Score-quality metrics.

Validates the core promise - higher score => more desirable asset - by relating
historical scores to realized forward returns:

  * IC          : pooled Spearman rank corr(score, forward return). >0 = the
                  ranking has predictive sign; ~0.03-0.08 is typical/useful.
  * top_minus_bottom : mean forward return of the top-scoring half minus the
                  bottom-scoring half (per date, then averaged). The economic
                  payoff of trusting the score.
  * hit_rate    : share of dates where the top half beat the bottom half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    n_observations: int
    n_dates: int
    ic: float
    top_minus_bottom: float
    hit_rate: float
    mean_forward_return: float

    def as_dict(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "n_dates": self.n_dates,
            "information_coefficient": round(self.ic, 5)
            if np.isfinite(self.ic)
            else None,
            "top_minus_bottom_return": round(self.top_minus_bottom, 6)
            if np.isfinite(self.top_minus_bottom)
            else None,
            "hit_rate": round(self.hit_rate, 4) if np.isfinite(self.hit_rate) else None,
            "mean_forward_return": round(self.mean_forward_return, 6)
            if np.isfinite(self.mean_forward_return)
            else None,
        }


def evaluate_scores(
    score_panel: pd.DataFrame, forward_returns: pd.DataFrame
) -> BacktestSummary:
    """Both args are (dates x symbols); score at t vs return realized t -> t+H."""
    scores = score_panel.reindex_like(forward_returns)
    pairs = pd.concat(
        [scores.stack(future_stack=True), forward_returns.stack(future_stack=True)],
        axis=1,
        keys=["score", "ret"],
    ).dropna()

    if len(pairs) < 10:
        return BacktestSummary(len(pairs), 0, np.nan, np.nan, np.nan, np.nan)

    ic = float(pairs["score"].corr(pairs["ret"], method="spearman"))
    mean_fwd = float(pairs["ret"].mean())

    # Per-date top-half minus bottom-half forward return.
    diffs, wins, n_dates = [], 0, 0
    for _, row in scores.iterrows():
        valid = row.dropna()
        if len(valid) < 4:
            continue
        date = row.name
        rets = forward_returns.loc[date, valid.index].dropna()
        common = valid.index.intersection(rets.index)
        if len(common) < 4:
            continue
        med = valid[common].median()
        top = rets[valid[common] >= med]
        bottom = rets[valid[common] < med]
        if len(top) == 0 or len(bottom) == 0:
            continue
        d = float(top.mean() - bottom.mean())
        diffs.append(d)
        wins += int(d > 0)
        n_dates += 1

    top_minus_bottom = float(np.mean(diffs)) if diffs else np.nan
    hit_rate = float(wins / n_dates) if n_dates else np.nan

    return BacktestSummary(
        n_observations=len(pairs),
        n_dates=n_dates,
        ic=ic,
        top_minus_bottom=top_minus_bottom,
        hit_rate=hit_rate,
        mean_forward_return=mean_fwd,
    )
