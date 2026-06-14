"""Backtest evaluation: does a higher score actually mean a better asset?"""

from .metrics import BacktestSummary, evaluate_scores

__all__ = ["BacktestSummary", "evaluate_scores"]
