"""Scoring engine: the orchestrator that produces daily scores + confidence.

Pipeline per run:
    data -> factor series -> cross-sectional normalization (0-100)
         -> flexible IC weights -> anti-bubble penalty -> composite score
         -> calibrated confidence -> backtest validation

Everything is point-in-time and re-estimated each run, so calling it daily
yields a fresh score, fresh per-asset weights, and a fresh confidence.
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
    bubble_penalty_panel,
    composite_panel,
    compute_flexible_weights,
    cross_sectional_score,
    latest_bubble_breakdown,
)
from .scoring.antibubble import BubbleResult


@dataclass
class AssetScore:
    symbol: str
    rank: int
    final_score: float
    base_score: float
    confidence: float
    factor_scores: dict[str, float]      # normalized 0-100 per factor
    weights: dict[str, float]            # per-asset factor weights
    bubble: BubbleResult
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
            "final_score": round(self.final_score, 2),
            "base_score": round(self.base_score, 2),
            "confidence": round(self.confidence, 4),
            "favorable_probability": round(self.favorable_probability, 4),
            "factor_scores": {k: round(v, 2) for k, v in self.factor_scores.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "bubble": {
                "label": self.bubble.label,
                "penalty": round(self.bubble.penalty, 4),
                "heat_intensity": round(self.bubble.heat_intensity, 4),
                "confirmation_deficit": round(self.bubble.confirmation_deficit, 4),
                "rsi": round(self.bubble.rsi, 2),
                "momentum_z": round(self.bubble.momentum_z, 3),
                "news_heat": round(self.bubble.news_heat, 2),
            },
            "last_price": self.last_price,
            "synthetic": self.synthetic,
            "regime": self.regime,
            "fundamentals": {
                k: v for k, v in self.fundamentals.items() if k != "coin_id"
            },
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
    n_synthetic: int = 0
    n_real_news: int = 0
    n_real_fundamentals: int = 0
    weights_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "forward_horizon": self.horizon,
            "asset_class": self.asset_class,
            "data_source": self.data_source,
            "n_assets": len(self.scores),
            "n_synthetic": self.n_synthetic,
            "n_real_news": self.n_real_news,
            "n_real_fundamentals": self.n_real_fundamentals,
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
                    if self.reliability.brier_score is not None
                    else None
                ),
                "brier_baseline": (
                    round(self.reliability.brier_baseline, 5)
                    if self.reliability.brier_baseline is not None
                    else None
                ),
                "skill_score": (
                    round(self.reliability.skill_score, 5)
                    if self.reliability.skill_score is not None
                    else None
                ),
                "note": self.reliability.note,
            },
            "scores": [s.as_dict() for s in self.scores],
        }


def _panel_from(outputs: dict[str, dict[str, dict[str, pd.Series]]], factor: str,
                key: str = "score") -> pd.DataFrame:
    """Assemble a (dates x symbols) panel for one factor/diagnostic key."""
    cols = {}
    for symbol, fout in outputs.items():
        series = fout.get(factor, {}).get(key)
        if series is not None:
            cols[symbol] = series
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def _latest_valid(panel: pd.DataFrame, symbol: str, as_of) -> float:
    """Value at as_of, falling back to the symbol's most recent valid value."""
    if symbol not in panel.columns:
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

    def run(self, data: dict[str, AssetData]) -> EngineResult:
        symbols = list(data)
        factor_order = [f for f in FACTORS if f in self.factors]
        H = self.config.calibration.forward_horizon

        # Normalize all OHLCV indices to timezone-naive so real and synthetic
        # assets (which may differ in tz-awareness) can always be merged.
        for ad in data.values():
            idx = ad.ohlcv.index
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
                ad.ohlcv.index = idx.tz_localize(None)

        # 1. Run every factor on every asset.
        outputs: dict[str, dict[str, dict[str, pd.Series]]] = {}
        for symbol, ad in data.items():
            outputs[symbol] = {
                name: factor.compute(ad, self.config.factor)
                for name, factor in self.factors.items()
            }

        # 2. Raw factor panels -> cross-sectional 0-100 normalization.
        raw_panels = {f: _panel_from(outputs, f) for f in factor_order}
        norm_panels = {f: cross_sectional_score(raw_panels[f]) for f in factor_order}

        # 3. Forward returns (close at t -> t+H).
        close_panel = pd.concat(
            {s: data[s].ohlcv["close"] for s in symbols}, axis=1
        ).sort_index()
        forward_returns = close_panel.pct_change(H).shift(-H)

        # Align forward returns to the normalization index.
        ref_index = norm_panels[factor_order[0]].index
        forward_returns = forward_returns.reindex(ref_index)

        # 4. Flexible IC-driven weights.
        flex = compute_flexible_weights(
            norm_panels, forward_returns, self.config.weights
        )

        # 5. Anti-bubble penalty panel.
        rsi_panel = _panel_from(outputs, "indicators", "rsi")
        mom_panel = _panel_from(outputs, "technicals", "momentum_z")
        heat_panel = _panel_from(outputs, "news", "heat")
        penalty = bubble_penalty_panel(
            rsi=rsi_panel,
            momentum_z=mom_panel,
            heat=heat_panel,
            fundamentals_norm=norm_panels.get("fundamentals", pd.DataFrame()),
            orderflow_norm=norm_panels.get("orderflow", pd.DataFrame()),
            cfg=self.config.bubble,
        )

        # 6. Composite scores.
        base_panel, final_panel = composite_panel(norm_panels, flex.weights, penalty)

        # 7. Calibrated confidence model (trained on full history).
        features = self._stack_features(norm_panels, factor_order)
        target = forward_returns.stack(future_stack=True).reindex(features.index)
        self.confidence_model.fit(features, target)

        # 8. Backtest validation of the final score.
        bt = evaluate_scores(final_panel, forward_returns)

        # 9. Build the "today" snapshot.
        as_of = self._resolve_as_of(final_panel)
        asset_scores = self._build_asset_scores(
            data, outputs, norm_panels, final_panel, base_panel,
            flex.weights, factor_order, as_of,
        )

        n_synth = sum(1 for ad in data.values() if ad.synthetic)
        n_news = sum(1 for ad in data.values() if ad.meta.get("news_source"))
        n_fund = sum(1 for ad in data.values() if ad.meta.get("fundamentals_source"))
        # Source label comes from the real assets' metadata when available.
        real_sources = {
            ad.meta.get("source")
            for ad in data.values()
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
            n_synthetic=n_synth,
            n_real_news=n_news,
            n_real_fundamentals=n_fund,
            weights_matrix=flex.weights,
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _stack_features(norm_panels: dict[str, pd.DataFrame],
                        factor_order: list[str]) -> pd.DataFrame:
        cols = {f: norm_panels[f].stack(future_stack=True) for f in factor_order}
        return pd.concat(cols, axis=1)

    @staticmethod
    def _resolve_as_of(final_panel: pd.DataFrame):
        non_empty = final_panel.dropna(how="all")
        return non_empty.index[-1] if len(non_empty) else final_panel.index[-1]

    def _build_asset_scores(
        self, data, outputs, norm_panels, final_panel, base_panel,
        weights, factor_order, as_of,
    ) -> list[AssetScore]:
        rows: list[AssetScore] = []
        for symbol in data:
            final = _latest_valid(final_panel, symbol, as_of)
            base = _latest_valid(base_panel, symbol, as_of)
            fscores = {
                f: _latest_valid(norm_panels[f], symbol, as_of) for f in factor_order
            }
            wcol = (
                weights[symbol].to_dict() if symbol in weights.columns else {}
            )
            bubble = latest_bubble_breakdown(
                symbol=symbol,
                rsi=_latest_valid(_panel_from(outputs, "indicators", "rsi"), symbol, as_of),
                momentum_z=_latest_valid(
                    _panel_from(outputs, "technicals", "momentum_z"), symbol, as_of
                ),
                news_heat=_latest_valid(_panel_from(outputs, "news", "heat"), symbol, as_of),
                fundamentals_norm=fscores.get("fundamentals", np.nan),
                orderflow_norm=fscores.get("orderflow", np.nan),
                cfg=self.config.bubble,
            )
            last_price = float(data[symbol].ohlcv["close"].iloc[-1])
            rows.append(
                AssetScore(
                    symbol=symbol,
                    rank=0,
                    final_score=final,
                    base_score=base,
                    confidence=float("nan"),
                    factor_scores=fscores,
                    weights=wcol,
                    bubble=bubble,
                    last_price=last_price,
                    favorable_probability=float("nan"),
                    synthetic=data[symbol].synthetic,
                    regime=data[symbol].meta.get("regime"),
                    fundamentals=dict(data[symbol].fundamentals or {}),
                    news_count=len(data[symbol].news_headlines or []),
                )
            )

        # Confidence for today (vectorized over the snapshot).
        feat_today = pd.DataFrame(
            {f: {r.symbol: r.factor_scores[f] for r in rows} for f in factor_order}
        )
        final_series = pd.Series({r.symbol: r.final_score for r in rows})
        p_fav = pd.Series(
            self.confidence_model.predict_favorable_proba(feat_today),
            index=feat_today.index,
        )
        conf = self.confidence_model.confidence(feat_today, final_series)
        for r in rows:
            r.favorable_probability = float(p_fav.get(r.symbol, np.nan))
            r.confidence = float(conf.get(r.symbol, np.nan))

        # Rank by final score (desc); NaNs sink to the bottom.
        rows.sort(key=lambda r: (np.isnan(r.final_score), -np.nan_to_num(r.final_score, nan=-1e9)))
        for i, r in enumerate(rows, start=1):
            r.rank = i
        return rows
