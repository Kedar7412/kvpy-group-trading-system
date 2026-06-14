"""Factor abstraction.

Every factor computes, from one asset's raw data, a dict of point-in-time
(no lookahead) time series. The mandatory ``"score"`` series is the factor's
raw attractiveness signal in natural units (higher = more attractive); it is
normalized cross-sectionally later by the scoring engine. Additional keys are
diagnostics that the anti-bubble logic and reporting consume.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..config import FactorConfig
from ..data import AssetData


@runtime_checkable
class Factor(Protocol):
    name: str

    def compute(self, data: AssetData, cfg: FactorConfig) -> dict[str, pd.Series]:
        """Return {'score': series, ...diagnostics}. All aligned to ohlcv index."""
        ...
