"""Public live scorecard: did the calls we actually made come true?

This is the trust artifact. It takes the calls we *sealed* into the ledger
(stored in the ``scores`` table) and grades them against what actually happened
to prices afterwards -- no cherry-picking, no look-ahead, and with the ledger
integrity proof attached.

For each stored call (made on date D for symbol S over horizon H):
  * entry  = close on D, exit = close on D+H trading bars;
  * matured calls get a realized forward return and a realized max drawdown;
  * a FAVORED call is correct if the forward return is positive, AVOID if it is
    not, AVOID-BUBBLE if a real drawdown followed; NO-EDGE/NEUTRAL are tracked
    as abstentions (the honesty coverage).

Aggregates: accuracy by call type, calibration by confidence bucket, the
FAVORED-vs-AVOID spread vs an equal-weight benchmark, a bullshit-detector check
(do flagged names fall more?), and a "follow the FAVORED calls" equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACTIONABLE = ("FAVORED", "AVOID")


@dataclass
class ScorecardResult:
    asset_class: str
    horizon: int
    generated_at: str
    as_of_first: str | None
    as_of_last: str | None
    n_dates: int
    n_calls: int
    n_matured: int
    n_pending: int
    actionable_accuracy: float | None
    favored_mean_return: float | None
    avoid_mean_return: float | None
    benchmark_mean_return: float | None
    favored_minus_benchmark: float | None
    abstention_rate: float | None
    by_call: list[dict]
    by_confidence: list[dict]
    detector: dict
    follow_favored: dict
    ledger: dict
    note: str = ""
    equity_curve: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        def r(x, n=4):
            return round(x, n) if isinstance(x, (int, float)) and x == x else None
        return {
            "asset_class": self.asset_class,
            "horizon": self.horizon,
            "generated_at": self.generated_at,
            "as_of_first": self.as_of_first,
            "as_of_last": self.as_of_last,
            "n_dates": self.n_dates,
            "n_calls": self.n_calls,
            "n_matured": self.n_matured,
            "n_pending": self.n_pending,
            "actionable_accuracy": r(self.actionable_accuracy),
            "favored_mean_return": r(self.favored_mean_return, 5),
            "avoid_mean_return": r(self.avoid_mean_return, 5),
            "benchmark_mean_return": r(self.benchmark_mean_return, 5),
            "favored_minus_benchmark": r(self.favored_minus_benchmark, 5),
            "abstention_rate": r(self.abstention_rate),
            "by_call": self.by_call,
            "by_confidence": self.by_confidence,
            "detector": self.detector,
            "follow_favored": self.follow_favored,
            "ledger": self.ledger,
            "equity_curve": self.equity_curve,
            "note": self.note,
        }


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def evaluate_scorecard(
    store,
    asset_class: str,
    close_panel: pd.DataFrame,
    horizon: int,
    favorable_threshold: float = 0.0,
    crash_drawdown: float = 0.15,
) -> ScorecardResult:
    generated_at = pd.Timestamp.utcnow().tz_localize(None).isoformat(timespec="seconds")
    ledger = store.verify_chain()
    calls = store.all_scores(asset_class)

    if close_panel is not None and not close_panel.empty:
        if close_panel.index.tz is not None:
            close_panel = close_panel.copy()
            close_panel.index = close_panel.index.tz_localize(None)
        idx = close_panel.index
    else:
        idx = pd.DatetimeIndex([])

    matured = []  # enriched call dicts with realized outcomes
    n_pending = 0
    for c in calls:
        as_of = c.get("as_of")
        sym = c.get("symbol")
        if not as_of or sym not in (close_panel.columns if len(idx) else []):
            n_pending += 1
            continue
        pos = idx.searchsorted(_ts(as_of), side="right") - 1
        if pos < 0:
            n_pending += 1
            continue
        exit_pos = pos + horizon
        if exit_pos >= len(idx):
            n_pending += 1
            continue
        entry = close_panel.iloc[pos][sym]
        exit_px = close_panel.iloc[exit_pos][sym]
        if pd.isna(entry) or pd.isna(exit_px) or entry <= 0:
            n_pending += 1
            continue
        window = close_panel.iloc[pos + 1: exit_pos + 1][sym]
        fwd = float(exit_px / entry - 1.0)
        dd = float(window.min() / entry - 1.0) if len(window) else 0.0
        matured.append({
            "as_of": as_of, "symbol": sym, "call": c.get("call"),
            "score": c.get("final_score"), "confidence": c.get("confidence"),
            "prob": c.get("bubble_probability"), "fwd": fwd, "drawdown": dd,
        })

    n_calls = len(calls)
    n_matured = len(matured)
    dates = sorted({c.get("as_of") for c in calls if c.get("as_of")})

    if n_matured == 0:
        return ScorecardResult(
            asset_class=asset_class, horizon=horizon, generated_at=generated_at,
            as_of_first=dates[0] if dates else None,
            as_of_last=dates[-1] if dates else None,
            n_dates=len(dates), n_calls=n_calls, n_matured=0, n_pending=n_pending,
            actionable_accuracy=None, favored_mean_return=None, avoid_mean_return=None,
            benchmark_mean_return=None, favored_minus_benchmark=None,
            abstention_rate=_abstention_rate(calls),
            by_call=[], by_confidence=[], detector={}, follow_favored={},
            ledger=ledger,
            note="No matured calls yet -- run `daily`/`backfill` and let the horizon pass.",
        )

    m = pd.DataFrame(matured)
    benchmark = float(m["fwd"].mean())

    # Accuracy of actionable calls.
    fav = m[m["call"] == "FAVORED"]
    avo = m[m["call"] == "AVOID"]
    act = m[m["call"].isin(ACTIONABLE)]
    correct = (
        ((act["call"] == "FAVORED") & (act["fwd"] > favorable_threshold))
        | ((act["call"] == "AVOID") & (act["fwd"] <= favorable_threshold))
    )
    accuracy = float(correct.mean()) if len(act) else None

    by_call = []
    for call in ["FAVORED", "NEUTRAL", "AVOID", "AVOID-BUBBLE", "NO-EDGE"]:
        g = m[m["call"] == call]
        if len(g) == 0:
            continue
        by_call.append({
            "call": call, "n": int(len(g)),
            "mean_forward_return": round(float(g["fwd"].mean()), 5),
            "win_rate": round(float((g["fwd"] > 0).mean()), 4),
        })

    by_conf = _calibration_by_confidence(m, favorable_threshold)
    detector = _detector_check(m, crash_drawdown)
    follow, curve = _follow_favored(m)

    return ScorecardResult(
        asset_class=asset_class, horizon=horizon, generated_at=generated_at,
        as_of_first=dates[0], as_of_last=dates[-1], n_dates=len(dates),
        n_calls=n_calls, n_matured=n_matured, n_pending=n_pending,
        actionable_accuracy=accuracy,
        favored_mean_return=float(fav["fwd"].mean()) if len(fav) else None,
        avoid_mean_return=float(avo["fwd"].mean()) if len(avo) else None,
        benchmark_mean_return=benchmark,
        favored_minus_benchmark=(float(fav["fwd"].mean()) - benchmark) if len(fav) else None,
        abstention_rate=_abstention_rate(calls),
        by_call=by_call, by_confidence=by_conf, detector=detector,
        follow_favored=follow, ledger=ledger, equity_curve=curve,
    )


def _abstention_rate(calls: list[dict]) -> float | None:
    if not calls:
        return None
    n_abstain = sum(1 for c in calls if c.get("call") in ("NO-EDGE", "NEUTRAL"))
    return round(n_abstain / len(calls), 4)


def _calibration_by_confidence(m: pd.DataFrame, thr: float) -> list[dict]:
    bins = [(0.0, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01)]
    conf = m["confidence"]
    bullish = m["score"] >= 50.0
    correct = (bullish & (m["fwd"] > thr)) | (~bullish & (m["fwd"] <= thr))
    rows = []
    for lo, hi in bins:
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "confidence_bucket": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "n": n,
            "hit_rate": round(float(correct[mask].mean()), 4),
            "mean_forward_return": round(float(m.loc[mask, "fwd"].mean()), 5),
        })
    return rows


def _detector_check(m: pd.DataFrame, crash_dd: float) -> dict:
    prob = m["prob"]
    if prob.notna().sum() < 5:
        return {"note": "insufficient probability data"}
    flagged = m[prob >= 0.6]
    calm = m[prob < 0.6]
    out = {
        "flagged_n": int(len(flagged)),
        "flagged_mean_forward_return": round(float(flagged["fwd"].mean()), 5) if len(flagged) else None,
        "calm_mean_forward_return": round(float(calm["fwd"].mean()), 5) if len(calm) else None,
    }
    if len(flagged):
        out["flagged_crash_rate"] = round(float((flagged["drawdown"] <= -crash_dd).mean()), 4)
    if len(calm):
        out["calm_crash_rate"] = round(float((calm["drawdown"] <= -crash_dd).mean()), 4)
    return out


def _follow_favored(m: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Equity curve of acting on FAVORED calls: per date, mean fwd return of
    FAVORED names (cash if none). Overlapping windows -> illustrative, not a
    live P&L, but built only from sealed calls.
    """
    by_date = []
    for as_of, g in m.groupby("as_of"):
        fav = g[g["call"] == "FAVORED"]
        r = float(fav["fwd"].mean()) if len(fav) else 0.0
        by_date.append((as_of, r))
    by_date.sort(key=lambda x: x[0])
    if not by_date:
        return {}, []
    equity = 1.0
    curve = []
    rets = []
    for as_of, r in by_date:
        equity *= (1.0 + r)
        rets.append(r)
        curve.append({"as_of": as_of, "equity": round(equity, 5), "period_return": round(r, 5)})
    arr = np.array(rets)
    sharpe = float(arr.mean() / arr.std(ddof=0) * np.sqrt(len(arr))) if arr.std(ddof=0) > 0 else 0.0
    summary = {
        "n_periods": len(rets),
        "total_return": round(float(equity - 1.0), 4),
        "mean_period_return": round(float(arr.mean()), 5),
        "sharpe_like": round(sharpe, 3),
    }
    return summary, curve
