"""The bullshit detector: a predictive, calibrated bubble/crash model.

Instead of a fixed heuristic penalty, this blends several "overheating"
signals and (when enough history exists) trains a calibrated classifier to
predict a forward **drawdown** -- i.e. the probability the price craters within
the next ``crash_lookahead`` bars. That probability:

  * is reported per asset as ``bubble_probability`` (0-1),
  * drives a multiplicative penalty on the score (strong when an asset is hot
    AND unconfirmed by fundamentals + orderflow), and
  * comes with human-readable ``reasons`` so the call is explainable.

Signals (each mapped to 0..1, where 1 = extreme):
  RSI overbought · parabolic momentum · news/attention hype · stretch above the
  upper Bollinger band · premium over a long-run price anchor · volatility
  climax · vertical (blow-off) price move.

If history is too short to train, a transparent soft-OR heuristic is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from ..config import BubbleConfig

SIGNAL_NAMES = (
    "rsi", "momentum", "hype", "bollinger", "overextension", "vol_climax", "vertical",
)

_REASON = {
    "rsi": "overbought RSI",
    "momentum": "parabolic momentum",
    "hype": "news/attention hype",
    "bollinger": "stretched above upper band",
    "overextension": "far above long-run price anchor",
    "vol_climax": "volatility climax",
    "vertical": "vertical blow-off move",
}


@dataclass
class BubbleAssessment:
    symbol: str
    probability: float            # P(crash) in [0, 1]
    penalty: float                # multiplicative score penalty in [floor, 1]
    confirmation_deficit: float   # 0 = fully confirmed, 1 = no confirmation
    heat_intensity: float         # max raw signal (0..1)
    label: str
    reasons: list[str] = field(default_factory=list)
    # kept for backward-compatible reporting/storage
    rsi: float = float("nan")
    momentum_z: float = float("nan")
    news_heat: float = float("nan")
    overextension: float = float("nan")

    @property
    def news_heat_val(self) -> float:  # pragma: no cover
        return self.news_heat


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(3, win // 2)).mean()
    sd = s.rolling(win, min_periods=max(3, win // 2)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


class BubbleDetector:
    def __init__(self, cfg: BubbleConfig):
        self.cfg = cfg
        self.model: CalibratedClassifierCV | None = None
        self.fitted = False
        self.note = ""
        self.signal_weights = {
            "rsi": 0.18, "momentum": 0.20, "hype": 0.12, "bollinger": 0.12,
            "overextension": 0.20, "vol_climax": 0.08, "vertical": 0.10,
        }

    # -- signal construction ---------------------------------------------
    def signals(
        self,
        rsi: pd.DataFrame,
        momentum_z: pd.DataFrame,
        heat: pd.DataFrame,
        bb_pos: pd.DataFrame,
        premium: pd.DataFrame,
        close: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Return one 0..1 panel per signal, all aligned to ``close``'s index."""
        idx, cols = close.index, close.columns

        def align(df):
            return df.reindex(index=idx, columns=cols) if not df.empty else pd.DataFrame(
                index=idx, columns=cols, dtype=float
            )

        rsi, momentum_z = align(rsi), align(momentum_z)
        heat, bb_pos, premium = align(heat), align(bb_pos), align(premium)

        # Volatility climax & vertical move computed from prices directly.
        logret = np.log(close).diff()
        rvol = logret.rolling(20, min_periods=8).std(ddof=0)
        vol_z = rvol.apply(lambda c: _rolling_z(c, 90))
        roc5 = close.pct_change(5)
        vert_z = roc5.apply(lambda c: _rolling_z(c, 90))

        sig = {
            "rsi": _clip01((rsi - 70.0) / 30.0),
            "momentum": _clip01((momentum_z - 1.0) / 2.0),
            "hype": _clip01((heat - 70.0) / 30.0),
            "bollinger": _clip01((bb_pos - 1.0) / 0.5),
            "overextension": _clip01(premium / 0.6),
            "vol_climax": _clip01(vol_z / 2.5),
            "vertical": _clip01(vert_z / 2.5),
        }
        return {k: v.fillna(0.0) for k, v in sig.items()}

    # -- labels -----------------------------------------------------------
    def crash_labels(self, close: pd.DataFrame) -> pd.DataFrame:
        """1 if price draws down >= threshold within the next lookahead bars."""
        L = self.cfg.crash_lookahead
        arr = close.to_numpy(dtype=float)
        T = arr.shape[0]
        fwd_min = np.full_like(arr, np.nan)
        for t in range(T):
            end = min(T, t + 1 + L)
            if t + 1 < end:
                fwd_min[t] = np.nanmin(arr[t + 1:end], axis=0)
        drawdown = fwd_min / arr - 1.0
        labels = (drawdown <= -self.cfg.crash_drawdown).astype(float)
        labels[np.isnan(drawdown)] = np.nan
        return pd.DataFrame(labels, index=close.index, columns=close.columns)

    # -- heuristic probability -------------------------------------------
    def _heuristic_prob(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Soft-OR blend: p = 1 - prod(1 - w_i * s_i)."""
        ref = next(iter(signals.values()))
        keep = pd.DataFrame(1.0, index=ref.index, columns=ref.columns)
        for name, s in signals.items():
            w = self.signal_weights.get(name, 0.1)
            keep = keep * (1.0 - w * s.fillna(0.0))
        return (1.0 - keep).clip(0.0, 1.0)

    # -- fit / predict ----------------------------------------------------
    def fit(self, signals: dict[str, pd.DataFrame], close: pd.DataFrame) -> "BubbleDetector":
        labels = self.crash_labels(close)
        feat = self._stack(signals)
        y = labels.stack(future_stack=True).reindex(feat.index)
        df = feat.copy()
        df["__y__"] = y
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < self.cfg.min_train_samples or df["__y__"].nunique() < 2:
            self.fitted = False
            self.note = (
                f"heuristic (n={len(df)}, need >= {self.cfg.min_train_samples})"
            )
            return self
        X = df[list(SIGNAL_NAMES)].to_numpy()
        yv = df["__y__"].to_numpy()
        folds = max(2, min(3, int(min(np.bincount(yv.astype(int))))))
        try:
            base = LogisticRegression(max_iter=1000)
            self.model = CalibratedClassifierCV(base, method="isotonic", cv=folds).fit(X, yv)
            self.fitted = True
            self.note = f"calibrated model (n={len(df)})"
        except Exception as exc:  # pragma: no cover
            self.fitted = False
            self.note = f"heuristic (fit failed: {exc})"
        return self

    def probability_panel(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            return self._heuristic_prob(signals)
        feat = self._stack(signals).fillna(0.0)
        p = self.model.predict_proba(feat[list(SIGNAL_NAMES)].to_numpy())[:, 1]
        out = pd.Series(p, index=feat.index).unstack()
        ref = next(iter(signals.values()))
        return out.reindex(index=ref.index, columns=ref.columns)

    @staticmethod
    def _stack(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        cols = {name: signals[name].stack(future_stack=True) for name in SIGNAL_NAMES}
        return pd.concat(cols, axis=1)

    # -- penalty + per-asset assessment ----------------------------------
    def penalty_from_prob(self, prob: float, deficit: float) -> float:
        # Confirmed hot moves keep ~40% of the penalty weight; unconfirmed get all.
        intensity = prob * (0.4 + 0.6 * deficit)
        pen = 1.0 - self.cfg.strength * intensity * (1.0 - self.cfg.penalty_floor)
        return float(np.clip(pen, self.cfg.penalty_floor, 1.0))

    def confirmation_deficit_panel(
        self, fundamentals_norm: pd.DataFrame, orderflow_norm: pd.DataFrame
    ) -> pd.DataFrame:
        confirm = (fundamentals_norm.fillna(50.0) + orderflow_norm.fillna(50.0)) / 2.0
        return ((self.cfg.confirm_threshold - confirm)
                / max(1e-9, self.cfg.confirm_threshold)).clip(0.0, 1.0)

    def penalty_panel(
        self,
        prob: pd.DataFrame,
        fundamentals_norm: pd.DataFrame,
        orderflow_norm: pd.DataFrame,
    ) -> pd.DataFrame:
        """Multiplicative score penalty in [floor, 1] for the whole panel."""
        deficit = self.confirmation_deficit_panel(
            fundamentals_norm.reindex_like(prob),
            orderflow_norm.reindex_like(prob),
        )
        intensity = prob.fillna(0.0) * (0.4 + 0.6 * deficit.fillna(0.0))
        pen = 1.0 - self.cfg.strength * intensity * (1.0 - self.cfg.penalty_floor)
        return pen.clip(self.cfg.penalty_floor, 1.0)

    def assess_latest(
        self,
        symbol: str,
        signals_latest: dict[str, float],
        probability: float,
        fundamentals_norm: float,
        orderflow_norm: float,
        rsi: float,
        momentum_z: float,
        news_heat: float,
        overextension: float,
    ) -> BubbleAssessment:
        confirm = (np.nan_to_num(fundamentals_norm, nan=50.0)
                   + np.nan_to_num(orderflow_norm, nan=50.0)) / 2.0
        deficit = float(_clip01((self.cfg.confirm_threshold - confirm)
                                 / max(1e-9, self.cfg.confirm_threshold)))
        prob = float(np.clip(probability, 0.0, 1.0))
        penalty = self.penalty_from_prob(prob, deficit)
        heat_intensity = float(max((v for v in signals_latest.values()), default=0.0))

        reasons = [
            _REASON[k] for k, v in signals_latest.items()
            if v is not None and v >= 0.5 and k in _REASON
        ]

        if prob >= self.cfg.bubble_flag_prob and deficit > 0.5:
            label = "bubble-risk: hot & unconfirmed"
        elif prob >= self.cfg.bubble_flag_prob:
            label = "hot but confirmed"
        elif prob >= 0.35 or heat_intensity >= 0.5:
            label = "warming"
        else:
            label = "normal"

        return BubbleAssessment(
            symbol=symbol,
            probability=prob,
            penalty=penalty,
            confirmation_deficit=deficit,
            heat_intensity=heat_intensity,
            label=label,
            reasons=reasons,
            rsi=float(rsi) if rsi == rsi else float("nan"),
            momentum_z=float(momentum_z) if momentum_z == momentum_z else float("nan"),
            news_heat=float(news_heat) if news_heat == news_heat else float("nan"),
            overextension=float(overextension) if overextension == overextension else float("nan"),
        )
