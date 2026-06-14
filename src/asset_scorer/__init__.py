"""Systematic multi-factor scoring engine for tradable assets.

Produces a daily attractiveness score (0-100) per asset from five factors
(news, technicals, fundamentals, orderflow, indicators), with flexible
per-asset weights and a calibrated confidence = probability the score is
accurate.
"""

__version__ = "0.1.0"

FACTORS = ("news", "technicals", "fundamentals", "orderflow", "indicators")
