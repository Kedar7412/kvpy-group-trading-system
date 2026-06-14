"""Confidence model = P(the score's directional call is correct).

Definition of "accurate":
    A score is a *directional call*. A high score (>50) says "favorable forward
    return is likely"; a low score (<50) says "unfavorable". The score is
    accurate if the realized forward return matches that call.

Method (quant math):
    1. Label each historical observation y = 1 if the H-bar forward return is
       favorable (> threshold), else 0.
    2. Fit a logistic model on the normalized factor scores, wrapped in
       ``CalibratedClassifierCV`` (isotonic or Platt) so the output probability
       p = P(favorable) is *calibrated* - i.e. among samples we call "70%",
       about 70% truly are favorable.
    3. Confidence for an asset today = p if its score is bullish (>=50) else
       (1 - p). This is the probability the score's stance is right.
    4. Quality is reported via out-of-fold Brier score + a reliability curve.

If history is too short to fit, we fall back to a transparent heuristic based on
factor agreement and conviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

from ..config import CalibrationConfig


@dataclass
class ReliabilityReport:
    fitted: bool
    method: str
    n_samples: int
    base_rate: float                       # share of favorable outcomes
    brier_score: float | None = None       # lower is better (0 = perfect)
    brier_baseline: float | None = None    # Brier of always predicting base_rate
    skill_score: float | None = None       # 1 - brier/baseline (>0 = useful)
    bin_confidence: list[float] = field(default_factory=list)
    bin_observed: list[float] = field(default_factory=list)
    note: str = ""


class ConfidenceModel:
    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.model: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self.base_rate: float = 0.5
        self.report: ReliabilityReport | None = None

    # -- training ---------------------------------------------------------
    def fit(self, features: pd.DataFrame, forward_return: pd.Series) -> "ConfidenceModel":
        df = features.copy()
        df["__y__"] = (forward_return > self.cfg.favorable_threshold).astype(float)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        self.feature_names = list(features.columns)
        y = df["__y__"].to_numpy()
        X = df[self.feature_names].to_numpy()
        n = len(df)
        base_rate = float(y.mean()) if n else 0.5
        self.base_rate = base_rate

        # Need both classes and enough samples to calibrate honestly.
        if n < self.cfg.min_train_samples or len(np.unique(y)) < 2:
            self.model = None
            self.report = ReliabilityReport(
                fitted=False,
                method="heuristic",
                n_samples=n,
                base_rate=base_rate,
                note=(
                    f"Insufficient/skewed history (n={n}, "
                    f"need >= {self.cfg.min_train_samples}); using heuristic confidence."
                ),
            )
            return self

        method = self.cfg.calibration_method
        folds = max(2, min(self.cfg.cv_folds, int(min(np.bincount(y.astype(int))))))

        def make_estimator():
            base = LogisticRegression(max_iter=1000, C=1.0)
            return CalibratedClassifierCV(base, method=method, cv=folds)

        # Honest, out-of-fold probabilities for scoring the calibration quality.
        try:
            oof = cross_val_predict(
                make_estimator(), X, y, cv=folds, method="predict_proba"
            )[:, 1]
            brier = float(brier_score_loss(y, oof))
            baseline = float(brier_score_loss(y, np.full_like(y, base_rate)))
            skill = 1.0 - brier / baseline if baseline > 0 else 0.0
            frac_pos, mean_pred = calibration_curve(
                y, oof, n_bins=10, strategy="quantile"
            )
        except Exception as exc:  # pragma: no cover - degenerate CV splits
            brier = baseline = skill = None
            frac_pos = mean_pred = np.array([])
            note = f"Calibration metrics unavailable: {exc}"
        else:
            note = ""

        # Final model trained on all data.
        self.model = make_estimator().fit(X, y)
        self.report = ReliabilityReport(
            fitted=True,
            method=method,
            n_samples=n,
            base_rate=base_rate,
            brier_score=brier,
            brier_baseline=baseline,
            skill_score=skill,
            bin_confidence=list(map(float, mean_pred)),
            bin_observed=list(map(float, frac_pos)),
            note=note,
        )
        return self

    # -- inference --------------------------------------------------------
    def predict_favorable_proba(self, features: pd.DataFrame) -> np.ndarray:
        """P(favorable forward return) per row, calibrated when fitted."""
        if self.model is None:
            return self._heuristic_favorable(features)
        X = features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean()).fillna(0.0).to_numpy()
        return self.model.predict_proba(X)[:, 1]

    def confidence(self, features: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """Probability the score's directional call is correct.

        Bullish score (>=50): confidence = P(favorable).
        Bearish score (<50) : confidence = 1 - P(favorable).
        """
        p_fav = self.predict_favorable_proba(features)
        p_fav = pd.Series(p_fav, index=features.index)
        bullish = scores.reindex(features.index) >= 50.0
        conf = p_fav.where(bullish, 1.0 - p_fav)
        return conf.clip(0.0, 1.0)

    # -- fallback ---------------------------------------------------------
    def _heuristic_favorable(self, features: pd.DataFrame) -> np.ndarray:
        """Agreement/conviction heuristic mapped to a favorable probability.

        When the factor scores agree (low dispersion) and lean bullish, the
        favorable probability rises above the base rate; strong disagreement
        pulls it back toward 0.5.
        """
        f = features[self.feature_names] if self.feature_names else features
        # Factor scores are 0-100; center at 50.
        centered = (f - 50.0) / 50.0
        mean_lean = centered.mean(axis=1)
        dispersion = centered.std(axis=1, ddof=0).fillna(1.0)
        agreement = (1.0 - dispersion.clip(0, 1))
        lean = (mean_lean * agreement).clip(-1, 1)
        # Map lean in [-1,1] to probability around the base rate.
        p = self.base_rate + (0.45 * lean)
        return p.clip(0.02, 0.98).to_numpy()
