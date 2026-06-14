"""Scoring math: normalization, flexible weighting, anti-bubble, composite."""

from .antibubble import BubbleResult, bubble_penalty_panel, latest_bubble_breakdown
from .bubble_detector import BubbleAssessment, BubbleDetector, SIGNAL_NAMES
from .composite import composite_panel
from .normalize import cross_sectional_score
from .recommend import Recommendation, recommend
from .regime import (
    NEUTRAL, RISK_OFF, RISK_ON, RegimeState, compute_regime,
    longs_allowed, regime_state_at, same_regime_dates,
)
from .weights import FlexibleWeights, compute_flexible_weights, information_coefficients

__all__ = [
    "cross_sectional_score",
    "information_coefficients",
    "compute_flexible_weights",
    "FlexibleWeights",
    "bubble_penalty_panel",
    "latest_bubble_breakdown",
    "BubbleResult",
    "BubbleDetector",
    "BubbleAssessment",
    "SIGNAL_NAMES",
    "composite_panel",
    "Recommendation",
    "recommend",
    "compute_regime",
    "regime_state_at",
    "longs_allowed",
    "same_regime_dates",
    "RegimeState",
    "RISK_ON",
    "NEUTRAL",
    "RISK_OFF",
]
