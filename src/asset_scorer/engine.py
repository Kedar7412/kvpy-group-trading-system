"""Scoring engine: the orchestrator that produces daily scores + confidence.

Pipeline per run:
    data -> factor series -> cross-sectional normalization (0-100)
         -> flexible IC weights -> predictive bubble penalty -> composite score
         -> calibrated confidence -> abstention/recommendation

Point-in-time panels are built once and reused by both the live run and the
walk-forward backtester, so there is a single source of truth and no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import FACTORS
from .backtest import BacktestSummary, evaluate_scores
from .calibration import ConfidenceModel, ReliabilityReport
from .config import AppConfig
from .data import AssetData
from .factors import default_factors
from .scoring import (
    BubbleAssessment,
    BubbleDetector,
    Recommendation,
    SIGNAL_NAMES,
    composite_panel,
    compute_flexible_weights,
    compute_regime,
    cross_sectional_score,
    longs_allowed,
    recommend,
    regime_state_at,
)


@dataclass
class Panels:
    """Reusable, point-in-time (no-leak) panels for one universe."""

    norm_panels: dict[str, pd.DataFrame]   # factor -> (dates x symbols), 0-100
    forward_returns: pd.DataFrame
    close_panel: pd.DataFrame
    factor_order: list[str]
    rsi: pd.DataFrame
    momentum_z: pd.DataFrame
    heat: pd.DataFrame
    bb_pos: pd.DataFrame
    premium: pd.DataFrame

    @property
    def index(self) -> pd.Index:
        return self.norm_panels[self.factor_order[0]].index


@dataclass
class AssetScore:
    symbol: str
    rank: int
    final_score: float
    base_score: float
    confidence: float
    factor_scores: dict[str, float]
    weights: dict[str, float]
    bubble: BubbleAssessment
    recommendation: Recommendation
    last_price: float
    favorable_probability: float
    synthetic: bool
    regime: str | None = None
    fundamentals: dict = field(default_factory=dict)
    news_count: int = 0

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "call": self.recommendation.call,
            "rationale": self.recommendation.rationale,
            "final_score": round(self.final_score, 2),
            "base_score": round(self.base_score, 2),
            "confidence": round(self.confidence, 4),
            "favorable_probability": round(self.favorable_probability, 4),
            "bubble_probability": round(self.bubble.probability, 4),
            "factor_scores": {k: round(v, 2) for k, v in self.factor_scores.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "bubble": {
                "label": self.bubble.label,
                "probability": round(self.bubble.probability, 4),
                "penalty": round(self.bubble.penalty, 4),
                "confirmation_deficit": round(self.bubble.confirmation_deficit, 4),
                "reasons": self.bubble.reasons,
                "rsi": round(self.bubble.rsi, 2) if self.bubble.rsi == self.bubble.rsi else None,
                "momentum_z": round(self.bubble.momentum_z, 3) if self.bubble.momentum_z == self.bubble.momentum_z else None,
                "news_heat": round(self.bubble.news_heat, 2) if self.bubble.news_heat == self.bubble.news_heat else None,
            },
            "last_price": self.last_price,
            "synthetic": self.synthetic,
            "regime": self.regime,
            "fundamentals": {k: v for k, v in self.fundamentals.items() if k != "coin_id"},
            "news_count": self.news_count,
        }


@dataclass
class EngineResult:
    as_of: str
    horizon: int
    scores: list[AssetScore]
    reliability: ReliabilityReport
    backtest: BacktestSummary
    global_ic: dict[str, float]
    data_source: str
    asset_class: str = "crypto"
    regime: str = "neutral"
    n_synthetic: int = 0
    n_real_news: int = 0
    n_real_fundamentals: int = 0
    bubble_model_note: str = ""
    weights_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "forward_horizon": self.horizon,
            "asset_class": self.asset_class,
            "data_source": self.data_source,
            "regime": self.regime,
            "n_assets": len(self.scores),
            "n_synthetic": self.n_synthetic,
            "n_real_news": self.n_real_news,
            "n_real_fundamentals": self.n_real_fundamentals,
            "bubble_detector": self.bubble_model_note,
            "global_information_coefficients": {
                k: (round(v, 5) if v is not None and np.isfinite(v) else None)
                for k, v in self.global_ic.items()
            },
            "backtest": self.backtest.as_dict(),
            "calibration": {
                "fitted": self.reliability.fitted,
                "method": self.reliability.method,
                "n_samples": self.reliability.n_samples,
                "base_rate": round(self.reliability.base_rate, 4),
                "brier_score": (
                    round(self.reliability.brier_score, 5)
                    if self.reliability.brier_score is not None else None
                ),
                "skill_score": (
                    round(self.reliability.skill_score, 5)
                    if self.reliability.skill_score is not None else None
                ),
                "note": self.reliability.note,
            },
            "scores": [s.as_dict() for s in self.scores],
        }


def _panel_from(outputs, factor: str, key: str = "score") -> pd.DataFrame:
    cols = {}
    for symbol, fout in outputs.items():
        series = fout.get(factor, {}).get(key)
        if series is not None:
            cols[symbol] = series
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def _latest_valid(panel: pd.DataFrame, symbol: str, as_of) -> float:
    if panel is None or panel.empty or symbol not in panel.columns:
        return float("nan")
    col = panel[symbol]
    if as_of in col.index and pd.notna(col.loc[as_of]):
        return float(col.loc[as_of])
    valid = col.dropna()
    return float(valid.iloc[-1]) if len(valid) else float("nan")


class ScoringEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.factors = default_factors()
        self.confidence_model = ConfidenceModel(config.calibration)

    # -- panel construction (shared with the backtester) ------------------
    def build_panels(self, data: dict[str, AssetData]) -> tuple[Panels, dict]:
        symbols = list(data)
        factor_order = [f for f in FACTORS if f in self.factors]
        H = self.config.calibration.forward_horizon

        for ad in data.values():
            idx = ad.ohlcv.index
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
                ad.ohlcv.index = idx.tz_localize(None)

        outputs = {
            symbol: {
                name: factor.compute(ad, self.config.factor)
                for name, factor in self.factors.items()
            }
            for symbol, ad in data.items()
        }

        raw_panels = {f: _panel_from(outputs, f) for f in factor_order}
        norm_panels = {f: cross_sectional_score(raw_panels[f]) for f in factor_order}

        close_panel = pd.concat(
            {s: data[s].ohlcv["close"] for s in symbols}, axis=1
        ).sort_index()
        ref_index = norm_panels[factor_order[0]].index
        close_panel = close_panel.reindex(ref_index)
        forward_returns = close_panel.pct_change(H).shift(-H)

        panels = Panels(
            norm_panels=norm_panels,
            forward_returns=forward_returns,
            close_panel=close_panel,
            factor_order=factor_order,
            rsi=_panel_from(outputs, "indicators", "rsi").reindex(ref_index),
            momentum_z=_panel_from(outputs, "technicals", "momentum_z").reindex(ref_index),
            heat=_panel_from(outputs, "news", "heat").reindex(ref_index),
            bb_pos=_panel_from(outputs, "indicators", "bb_pos").reindex(ref_index),
            premium=_panel_from(outputs, "fundamentals", "premium").reindex(ref_index),
        )
        return panels, outputs

    # -- main run ---------------------------------------------------------
    def run(self, data: dict[str, AssetData]) -> EngineResult:
        p, outputs = self.build_panels(data)
        H = self.config.calibration.forward_horizon
        factor_order = p.factor_order

        # Flexible IC weights.
        flex = compute_flexible_weights(p.norm_panels, p.forward_returns, self.config.weights)

        # Predictive bubble detector (the bullshit detector).
        detector = BubbleDetector(self.config.bubble)
        signals = detector.signals(p.rsi, p.momentum_z, p.heat, p.bb_pos, p.premium, p.close_panel)
        detector.fit(signals, p.close_panel)
        prob_panel = detector.probability_panel(signals)
        penalty = detector.penalty_panel(
            prob_panel,
            p.norm_panels.get("fundamentals", pd.DataFrame()),
            p.norm_panels.get("orderflow", pd.DataFrame()),
        )

        base_panel, final_panel = composite_panel(p.norm_panels, flex.weights, penalty)

        # Calibrated confidence.
        features = self._stack_features(p.norm_panels, factor_order)
        target = p.forward_returns.stack(future_stack=True).reindex(features.index)
        self.confidence_model.fit(features, target)

        bt = evaluate_scores(final_panel, p.forward_returns)
        as_of = self._resolve_as_of(final_panel)

        # Market regime (point-in-time) gates FAVORED longs.
        regime_df = compute_regime(p.close_panel, self.config.regime)
        regime_label = regime_state_at(regime_df, as_of).label
        longs_ok = longs_allowed(regime_label)

        asset_scores = self._build_asset_scores(
            data, p, signals, prob_panel, final_panel, base_panel,
            flex.weights, factor_order, as_of, detector, longs_ok,
        )

        n_synth = sum(1 for ad in data.values() if ad.synthetic)
        n_news = sum(1 for ad in data.values() if ad.meta.get("news_source"))
        n_fund = sum(1 for ad in data.values() if ad.meta.get("fundamentals_source"))
        real_sources = {
            ad.meta.get("source") for ad in data.values()
            if not ad.synthetic and ad.meta.get("source")
        }
        if n_synth == len(data):
            source = "synthetic"
        elif n_synth:
            source = "mixed"
        elif len(real_sources) == 1:
            source = real_sources.pop()
        elif real_sources:
            source = "+".join(sorted(real_sources))
        else:
            source = self.config.data.exchange

        return EngineResult(
            as_of=str(as_of.date()) if hasattr(as_of, "date") else str(as_of),
            horizon=H,
            scores=asset_scores,
            reliability=self.confidence_model.report,
            backtest=bt,
            global_ic={f: float(flex.global_ic.get(f, np.nan)) for f in factor_order},
            data_source=source,
            asset_class=self.config.asset_class,
            regime=regime_label,
            n_synthetic=n_synth,
            n_real_news=n_news,
            n_real_fundamentals=n_fund,
            bubble_model_note=detector.note,
            weights_matrix=flex.weights,
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _stack_features(norm_panels, factor_order):
        cols = {f: norm_panels[f].stack(future_stack=True) for f in factor_order}
        return pd.concat(cols, axis=1)

    @staticmethod
    def _resolve_as_of(final_panel):
        non_empty = final_panel.dropna(how="all")
        return non_empty.index[-1] if len(non_empty) else final_panel.index[-1]

    def _build_asset_scores(
        self, data, p, signals, prob_panel, final_panel, base_panel,
        weights, factor_order, as_of, detector, longs_ok=True,
    ) -> list[AssetScore]:
        rows: list[AssetScore] = []
        for symbol in data:
            final = _latest_valid(final_panel, symbol, as_of)
            base = _latest_valid(base_panel, symbol, as_of)
            fscores = {f: _latest_valid(p.norm_panels[f], symbol, as_of) for f in factor_order}
            wcol = weights[symbol].to_dict() if symbol in weights.columns else {}

            sig_latest = {
                name: _latest_valid(signals[name], symbol, as_of) for name in SIGNAL_NAMES
            }
            prob = _latest_valid(prob_panel, symbol, as_of)
            premium = _latest_valid(p.premium, symbol, as_of)
            bubble = detector.assess_latest(
                symbol=symbol,
                signals_latest=sig_latest,
                probability=prob if prob == prob else 0.0,
                fundamentals_norm=fscores.get("fundamentals", np.nan),
                orderflow_norm=fscores.get("orderflow", np.nan),
                rsi=_latest_valid(p.rsi, symbol, as_of),
                momentum_z=_latest_valid(p.momentum_z, symbol, as_of),
                news_heat=_latest_valid(p.heat, symbol, as_of),
                overextension=max(premium, 0.0) if premium == premium else float("nan"),
            )
            rows.append(
                AssetScore(
                    symbol=symbol, rank=0, final_score=final, base_score=base,
                    confidence=float("nan"), factor_scores=fscores, weights=wcol,
                    bubble=bubble,
                    recommendation=Recommendation("NO-EDGE", "pending"),
                    last_price=float(data[symbol].ohlcv["close"].iloc[-1]),
                    favorable_probability=float("nan"),
                    synthetic=data[symbol].synthetic,
                    regime=data[symbol].meta.get("regime"),
                    fundamentals=dict(data[symbol].fundamentals or {}),
                    news_count=len(data[symbol].news_headlines or []),
                )
            )

        feat_today = pd.DataFrame(
            {f: {r.symbol: r.factor_scores[f] for r in rows} for f in factor_order}
        )
        final_series = pd.Series({r.symbol: r.final_score for r in rows})
        p_fav = pd.Series(
            self.confidence_model.predict_favorable_proba(feat_today), index=feat_today.index
        )
        conf = self.confidence_model.confidence(feat_today, final_series)
        for r in rows:
            r.favorable_probability = float(p_fav.get(r.symbol, np.nan))
            r.confidence = float(conf.get(r.symbol, np.nan))
            r.recommendation = recommend(
                r.final_score, r.confidence, r.bubble.probability,
                self.config.abstention, longs_ok=longs_ok,
            )

        rows.sort(key=lambda r: (np.isnan(r.final_score), -np.nan_to_num(r.final_score, nan=-1e9)))
        for i, r in enumerate(rows, start=1):
            r.rank = i
        return rows
