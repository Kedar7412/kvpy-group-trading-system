"""News factor: narrative sentiment + an attention/heat gauge.

Free historical news is not available, so the backtestable series is a
narrative *proxy* (medium-term drift, lightly de-rated when attention is
extreme). When live headlines are present, a lexicon sentiment score blends
into today's value. ``heat`` (0-100) measures how much attention an asset is
getting and feeds the anti-bubble FOMO detector.

Swap in a real provider by implementing ``score_headlines`` against an API
that returns dated, scored articles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FactorConfig
from ..data import AssetData
from . import ta

Z_WIN = 90

_POSITIVE = {
    "adoption", "upgrade", "growth", "rises", "improving", "improve", "improves",
    "bullish", "partnership", "record", "strong", "surge", "rally", "gains",
    "stabilizes", "builders", "returns", "approval", "inflows",
}
_NEGATIVE = {
    "slides", "weakening", "outflows", "sours", "selloff", "sell-off", "hack",
    "lawsuit", "ban", "bearish", "crash", "dump", "fraud", "fear", "fud",
    "liquidation", "decline", "drops",
}
_HYPE = {"parabolic", "fomo", "hype", "moon", "all-time", "ath", "retail", "frenzy"}


class News:
    name = "news"

    def compute(self, data: AssetData, cfg: FactorConfig) -> dict[str, pd.Series]:
        close = data.ohlcv["close"]
        volume = data.ohlcv["volume"]

        # Attention/heat proxy: |returns| * volume, standardized then mapped 0-100.
        attention = ta.log_returns(close).abs() * np.log1p(volume)
        heat_z = ta.rolling_zscore(attention, Z_WIN)
        heat = (1 / (1 + np.exp(-heat_z)) * 100).clip(0, 100)

        # Narrative proxy: medium-term drift, de-rated when attention is extreme.
        narrative = ta.roc(close, 30)
        narrative_z = ta.rolling_zscore(narrative, Z_WIN)
        score = narrative_z - 0.25 * heat_z.clip(lower=0.0)

        # Blend real headline sentiment into today's value when available.
        sentiment = self.score_headlines(data.news_headlines)
        if sentiment is not None and len(score):
            valid = score.dropna()
            if len(valid):
                score = score.copy()
                score.iloc[-1] = valid.iloc[-1] + sentiment

        return {
            "score": score,
            "heat": heat,
            "sentiment_today": pd.Series(
                sentiment if sentiment is not None else np.nan,
                index=[close.index[-1]],
            ),
        }

    @staticmethod
    def score_headlines(headlines: list[str]) -> float | None:
        """Lexicon sentiment in roughly [-1, 1]; None if no headlines."""
        if not headlines:
            return None
        pos = neg = hype = 0
        for h in headlines:
            tokens = h.lower().replace(",", " ").replace(".", " ").split()
            pos += sum(t in _POSITIVE for t in tokens)
            neg += sum(t in _NEGATIVE for t in tokens)
            hype += sum(t in _HYPE for t in tokens)
        total = pos + neg + hype
        if total == 0:
            return 0.0
        # Hype counts as mildly negative for a "real value" tilt.
        raw = (pos - neg - 0.5 * hype) / total
        return float(np.clip(raw, -1.0, 1.0))
