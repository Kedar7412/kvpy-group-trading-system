"""Scoring math: normalization, flexible weighting, anti-bubble, composite."""

from .antibubble import BubbleResult, bubble_penalty_panel, latest_bubble_breakdown
from .composite import composite_panel
from .normalize import cross_sectional_score
from .weights import FlexibleWeights, compute_flexible_weights, information_coefficients

__all__ = [
    "cross_sectional_score",
    "information_coefficients",
    "compute_flexible_weights",
    "FlexibleWeights",
    "bubble_penalty_panel",
    "latest_bubble_breakdown",
    "BubbleResult",
    "composite_panel",
]
