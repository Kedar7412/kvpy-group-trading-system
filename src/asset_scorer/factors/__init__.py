"""Factor package: the five scoring factors and a registry helper."""

from .base import Factor
from .fundamentals import Fundamentals
from .indicators import Indicators
from .news import News
from .orderflow import Orderflow
from .technicals import Technicals


def default_factors() -> dict[str, Factor]:
    """Return one instance of each factor keyed by name (order = FACTORS)."""
    factors = [News(), Technicals(), Fundamentals(), Orderflow(), Indicators()]
    return {f.name: f for f in factors}


__all__ = [
    "Factor",
    "News",
    "Technicals",
    "Fundamentals",
    "Orderflow",
    "Indicators",
    "default_factors",
]
