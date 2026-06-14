"""Walk-forward (out-of-sample) backtest -- the edge gate.

This is the honest test: at each rebalance date we fit the weights, the
confidence model, and the bubble detector using ONLY past data, then score the
(unseen) current date and record the realized forward return. Nothing about the
future leaks into a decision.

From the out-of-sample records we build:
  * OOS Information Coefficient (does score rank predict forward return?),
  * an equal-weight benchmark, a long-only top-quantile book, a long-short
    book, and a *selective* book that only trades when confidence clears a bar
    (and holds cash otherwise) -- all net of transaction costs,
  * a calibration-by-confidence table (the trust artifact: do higher-confidence
    calls actually win more often?),
  * a detector check (do high crash-probability flags precede worse returns?).

If the selective book doesn't beat the benchmark after costs, that's the signal
to fix the model before trusting it -- which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..scoring import (
    BubbleDetector, compute_flexible_weights, compute_regime, longs_allowed,
    recommend, regime_state_at, same_regime_dates,
)
from ..calibration import ConfidenceModel as _ConfidenceModel


@dataclass
class PortfolioStats:
    name: str
    n_periods: int
    total_return: float
    annualized: float
    sharpe: float
    hit_rate: float
    max_drawdown: float
    coverage: float = 1.0  # fraction of periods with a live position
    equity_curve: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_periods": self.n_periods,
            "total_return": round(self.total_return, 4),
            "annualized": round(self.annualized, 4),
            "sharpe": round(self.sharpe, 3),
            "hit_rate": round(self.hit_rate, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "coverage": round(self.coverage, 4),
        }


@dataclass
class WalkForwardResult:
    asset_class: str
    horizon: int
    n_rebalances: int
    n_observations: int
    oos_ic: float
    portfolios: dict[str, PortfolioStats]
    by_confidence: list[dict]
    detector: dict
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "asset_class": self.asset_class,
            "horizon": self.horizon,
            "n_rebalances": self.n_rebalances,
            "n_observations": self.n_observations,
            "oos_information_coefficient": round(self.oos_ic, 5)
            if np.isfinite(self.oos_ic) else None,
            "portfolios": {k: v.as_dict() for k, v in self.portfolios.items()},
            "calibration_by_confidence": self.by_confidence,
            "detector_check": self.detector,
            "note": self.note,
        }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 8:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def _stats(name, rets: list[float], ppy: float, coverage: float = 1.0) -> PortfolioStats:
    r = np.array([x for x in rets if x is not None], dtype=float)
    if len(r) == 0:
        return PortfolioStats(name, 0, 0, 0, 0, 0, 0, coverage, [1.0])
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1] - 1.0)
    n = len(r)
    annualized = float((equity[-1]) ** (ppy / n) - 1.0) if equity[-1] > 0 else -1.0
    sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(ppy)) if r.std(ddof=0) > 0 else 0.0
    hit = float((r > 0).mean())
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())
    return PortfolioStats(
        name, n, total, annualized, sharpe, hit, max_dd, coverage,
        [1.0] + equity.tolist(),
    )


class WalkForwardBacktester:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self, engine, data, store=None, persist: bool = False,
            source: str = "backfill") -> WalkForwardResult:
        cfg = self.config.backtest
        H = self.config.calibration.forward_horizon
        step = cfg.step or H
        synthetic = bool(data) and all(ad.synthetic for ad in data.values())

        panels, _ = engine.build_panels(data)
        norm = panels.norm_panels
        fwd = panels.forward_returns
        close = panels.close_panel
        factors = panels.factor_order
        dates = list(panels.index)

        # Point-in-time bubble signals (no leak); the detector is refit per step.
        seed_det = BubbleDetector(self.config.bubble)
        signals = seed_det.signals(
            panels.rsi, panels.momentum_z, panels.heat, panels.bb_pos,
            panels.premium, close,
        )
        regime_df = compute_regime(close, self.config.regime)

        start = max(cfg.min_train, 1)
        rebal = [i for i in range(start, len(dates) - H, step)]
        records = []  # dicts: date, symbol, score, confidence, prob, fwd
        last_train_end = None
        weights = conf_model = prob_at = None

        for i in rebal:
            t = dates[i]
            train_end = dates[i - H]  # last date whose forward return is realized
            label_t = regime_state_at(regime_df, t).label

            # Regime-conditional weights: learn from same-regime history only.
            sel = None
            if self.config.weights.regime_conditional:
                sel = same_regime_dates(
                    regime_df, label_t, upto=train_end,
                    min_samples=self.config.weights.regime_min_samples,
                )
            if sel is not None:
                sel = sel[sel <= train_end]
                nf_train = {f: norm[f].reindex(sel) for f in factors}
                fr_train = fwd.reindex(sel)
            else:
                nf_train = {f: norm[f].loc[:train_end] for f in factors}
                fr_train = fwd.loc[:train_end]
            flex = compute_flexible_weights(nf_train, fr_train, self.config.weights)
            weights = flex.weights

            # Bubble probability: fit detector on past, predict at all dates.
            det = BubbleDetector(self.config.bubble)
            sig_train = {k: v.loc[:train_end] for k, v in signals.items()}
            det.fit(sig_train, close.loc[:train_end])
            prob_panel = det.probability_panel(signals)

            # Confidence model fit on past (optional, else heuristic).
            conf_model = _ConfidenceModel(self.config.calibration)
            if cfg.retrain_calibration:
                feat_train = self._stack(nf_train, factors)
                tgt = fr_train.stack(future_stack=True).reindex(feat_train.index)
                conf_model.fit(feat_train, tgt)

            # Score the unseen date t.
            feat_t = pd.DataFrame({f: norm[f].loc[t] for f in factors})
            base_t = self._composite_row(feat_t, weights, factors)
            deficit = det.confirmation_deficit_panel(
                norm.get("fundamentals", pd.DataFrame()),
                norm.get("orderflow", pd.DataFrame()),
            )
            prob_t = prob_panel.loc[t] if t in prob_panel.index else pd.Series(0.0, index=feat_t.index)
            def_t = deficit.loc[t] if t in deficit.index else pd.Series(0.5, index=feat_t.index)
            pen_t = (1.0 - self.config.bubble.strength * prob_t.fillna(0.0)
                     * (0.4 + 0.6 * def_t.fillna(0.5)) * (1.0 - self.config.bubble.penalty_floor)
                     ).clip(self.config.bubble.penalty_floor, 1.0)
            score_t = (base_t * pen_t).reindex(feat_t.index)
            conf_t = conf_model.confidence(feat_t, score_t)
            pfav_t = pd.Series(
                conf_model.predict_favorable_proba(feat_t), index=feat_t.index
            )
            fwd_t = fwd.loc[t]
            ranks = score_t.rank(ascending=False, method="min")
            longs_ok = longs_allowed(label_t)
            regime_label = label_t

            date_rows = []
            for sym in feat_t.index:
                s = score_t.get(sym)
                if s is not None and not pd.isna(s):
                    fr = fwd_t.get(sym)
                    if fr is not None and not pd.isna(fr):
                        records.append({
                            "date": t, "symbol": sym, "score": float(s),
                            "confidence": float(conf_t.get(sym, np.nan)),
                            "prob": float(prob_t.get(sym, np.nan)),
                            "fwd": float(fr), "regime": regime_label,
                        })
                    if persist:
                        prob = float(prob_t.get(sym, 0.0))
                        conf = float(conf_t.get(sym, np.nan))
                        rec_call = recommend(
                            float(s), conf, prob, self.config.abstention, longs_ok=longs_ok
                        )
                        deficit_v = float(def_t.get(sym, 0.5))
                        date_rows.append({
                            "symbol": sym,
                            "rank": int(ranks.get(sym, 0)),
                            "final_score": float(s),
                            "base_score": float(base_t.get(sym, np.nan)),
                            "confidence": conf,
                            "favorable_probability": float(pfav_t.get(sym, np.nan)),
                            "news": float(feat_t.loc[sym, "news"]) if "news" in feat_t else None,
                            "technicals": float(feat_t.loc[sym, "technicals"]) if "technicals" in feat_t else None,
                            "fundamentals": float(feat_t.loc[sym, "fundamentals"]) if "fundamentals" in feat_t else None,
                            "orderflow": float(feat_t.loc[sym, "orderflow"]) if "orderflow" in feat_t else None,
                            "indicators": float(feat_t.loc[sym, "indicators"]) if "indicators" in feat_t else None,
                            "bubble_probability": prob,
                            "bubble_label": self._bubble_label(prob, deficit_v),
                            "bubble_penalty": float(pen_t.get(sym, 1.0)),
                            "call": rec_call.call,
                            "rationale": rec_call.rationale,
                            "last_price": float(close.loc[t, sym]) if sym in close.columns else None,
                        })

            if persist and store is not None and date_rows:
                as_of = str(t.date()) if hasattr(t, "date") else str(t)
                store.save_calls(
                    as_of, self.config.asset_class, date_rows,
                    source=source, synthetic=synthetic, horizon=H,
                )

        if not records:
            return WalkForwardResult(
                self.config.asset_class, H, 0, 0, float("nan"), {}, [], {},
                note="Not enough history for a walk-forward test. Try --history 700.",
            )

        rec = pd.DataFrame(records)
        ppy = 252.0 / H
        result = self._build_portfolios(rec, ppy, cfg)
        result.note = (
            f"{len(rebal)} rebalances, step {step}, costs {cfg.cost_bps}bps, "
            f"selective>= {cfg.selective_confidence:.0%} confidence."
        )
        return result

    # -- portfolio + diagnostics -----------------------------------------
    def _build_portfolios(self, rec: pd.DataFrame, ppy: float, cfg) -> WalkForwardResult:
        q = cfg.quantile
        cost = cfg.cost_bps / 1e4
        bench, longonly, longshort, selective, regime_long = [], [], [], [], []
        sel_live = 0
        reg_live = 0

        for _, g in rec.groupby("date"):
            g = g.dropna(subset=["score", "fwd"])
            if len(g) < 4:
                continue
            hi = g["score"].quantile(1 - q)
            lo = g["score"].quantile(q)
            top = g[g["score"] >= hi]
            bot = g[g["score"] <= lo]
            bench.append(g["fwd"].mean())
            longonly.append(top["fwd"].mean() - cost)
            if len(bot):
                longshort.append(top["fwd"].mean() - bot["fwd"].mean() - 2 * cost)
            # selective: only top names that also clear the confidence bar
            sel = top[top["confidence"] >= cfg.selective_confidence]
            if len(sel):
                selective.append(sel["fwd"].mean() - cost)
                sel_live += 1
            else:
                selective.append(0.0)  # cash
            # regime-gated long: only hold the top book when the market allows longs
            regime_ok = "regime" in g.columns and longs_allowed(str(g["regime"].iloc[0]))
            if regime_ok:
                regime_long.append(top["fwd"].mean() - cost)
                reg_live += 1
            else:
                regime_long.append(0.0)  # cash in risk-off

        n_dates = rec["date"].nunique()
        n_eligible = len([1 for _, g in rec.groupby("date") if len(g) >= 4])
        coverage = sel_live / max(1, n_eligible)
        reg_coverage = reg_live / max(1, n_eligible)

        portfolios = {
            "benchmark": _stats("Equal-weight benchmark", bench, ppy),
            "long_only_top": _stats("Long top-quantile", longonly, ppy),
            "long_short": _stats("Long-short", longshort, ppy),
            "selective": _stats("Selective (high-confidence only)", selective, ppy, coverage),
            "regime_long": _stats("Regime-gated long (cash in risk-off)", regime_long, ppy, reg_coverage),
        }

        oos_ic = _spearman(rec["score"], rec["fwd"])
        by_conf = self._calibration_table(rec)
        detector = self._detector_check(rec)

        return WalkForwardResult(
            asset_class=self.config.asset_class,
            horizon=self.config.calibration.forward_horizon,
            n_rebalances=n_dates,
            n_observations=len(rec),
            oos_ic=oos_ic,
            portfolios=portfolios,
            by_confidence=by_conf,
            detector=detector,
        )

    @staticmethod
    def _calibration_table(rec: pd.DataFrame) -> list[dict]:
        """Do higher-confidence calls win more often? (the trust artifact)."""
        bins = [(0.0, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.01)]
        rows = []
        # "Correct" = the directional call (bullish if score>=50) matched the move.
        bullish = rec["score"] >= 50.0
        correct = ((bullish & (rec["fwd"] > 0)) | (~bullish & (rec["fwd"] <= 0)))
        for lo, hi in bins:
            m = (rec["confidence"] >= lo) & (rec["confidence"] < hi)
            n = int(m.sum())
            if n == 0:
                continue
            rows.append({
                "confidence_bucket": f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}",
                "n": n,
                "hit_rate": round(float(correct[m].mean()), 4),
                "mean_forward_return": round(float(rec.loc[m, "fwd"].mean()), 5),
            })
        return rows

    @staticmethod
    def _detector_check(rec: pd.DataFrame) -> dict:
        """Do high crash-probability flags precede worse returns?"""
        prob = rec["prob"]
        if prob.notna().sum() < 10:
            return {"note": "insufficient probability data"}
        flagged = rec[prob >= 0.6]
        calm = rec[prob < 0.6]
        return {
            "flagged_n": int(len(flagged)),
            "flagged_mean_forward_return": round(float(flagged["fwd"].mean()), 5)
            if len(flagged) else None,
            "calm_mean_forward_return": round(float(calm["fwd"].mean()), 5)
            if len(calm) else None,
            "flagged_negative_rate": round(float((flagged["fwd"] < 0).mean()), 4)
            if len(flagged) else None,
        }

    @staticmethod
    def _bubble_label(prob: float, deficit: float) -> str:
        if prob >= 0.6 and deficit > 0.5:
            return "bubble-risk: hot & unconfirmed"
        if prob >= 0.6:
            return "hot but confirmed"
        if prob >= 0.35:
            return "warming"
        return "normal"

    @staticmethod
    def _stack(norm_panels, factors) -> pd.DataFrame:
        cols = {f: norm_panels[f].stack(future_stack=True) for f in factors}
        return pd.concat(cols, axis=1)

    @staticmethod
    def _composite_row(feat_t: pd.DataFrame, weights: pd.DataFrame, factors) -> pd.Series:
        """base score at one date = sum_f weight(f, sym) * normalized(f)."""
        base = pd.Series(0.0, index=feat_t.index)
        wsum = pd.Series(0.0, index=feat_t.index)
        for f in factors:
            if f not in weights.index:
                continue
            w = weights.loc[f].reindex(feat_t.index)
            col = feat_t[f]
            mask = col.notna() & w.notna()
            base = base.add((col * w).where(mask, 0.0), fill_value=0.0)
            wsum = wsum.add(w.where(mask, 0.0), fill_value=0.0)
        return base.div(wsum.replace(0, np.nan))
