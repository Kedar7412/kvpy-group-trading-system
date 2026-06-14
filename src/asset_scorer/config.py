"""Configuration for the asset scoring engine.

Everything tunable lives here so the math modules stay clean. Defaults are
chosen to be sensible for a crypto MVP but the structures generalize to
stocks/commodities by swapping the data provider and the fundamentals factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import FACTORS


@dataclass(frozen=True)
class FactorConfig:
    """Lookback / smoothing parameters shared across factor calculations."""

    # Technicals
    trend_fast: int = 20
    trend_slow: int = 50
    momentum_lookback: int = 20
    vol_lookback: int = 30

    # Indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # Fundamentals proxy (liquidity / quality)
    liquidity_lookback: int = 30

    # Orderflow
    orderbook_depth_levels: int = 25


@dataclass(frozen=True)
class WeightConfig:
    """Controls the IC-driven flexible weighting scheme.

    Weights start from `priors` and are blended toward each factor's measured
    Information Coefficient (predictive power). `prior_strength` is the
    pseudo-count that keeps weights stable when history is short.
    """

    # Neutral starting priors (renormalized internally).
    priors: dict[str, float] = field(
        default_factory=lambda: {f: 1.0 for f in FACTORS}
    )
    prior_strength: float = 0.5  # 0 => pure IC, large => stick to priors
    ic_floor: float = 0.0  # negative ICs are clamped (don't bet against a factor)
    min_weight: float = 0.02  # never let a factor go fully to zero


@dataclass(frozen=True)
class BubbleConfig:
    """Anti-bubble penalty parameters.

    The penalty fires when momentum/indicator/news heat is high but
    fundamentals + orderflow do not confirm. It scales the composite score
    down toward `max_penalty` (a multiplicative factor in [floor, 1]).
    """

    rsi_hot: float = 75.0          # RSI above this is "overheated"
    momentum_hot_z: float = 1.5    # momentum z-score considered parabolic
    hype_hot: float = 70.0         # news/indicator heat threshold (0-100)
    confirm_threshold: float = 50.0  # fundamentals/orderflow below = no confirm
    penalty_floor: float = 0.55    # strongest possible multiplicative penalty
    strength: float = 1.0          # global multiplier on penalty intensity


@dataclass(frozen=True)
class CalibrationConfig:
    """Confidence model = probability the score is 'accurate'.

    Accuracy is defined as: a top-ranked (high) score is followed by a
    favorable forward return, and a low score by an unfavorable one. We train a
    probabilistic classifier on factor scores and isotonically calibrate it.
    """

    forward_horizon: int = 5        # bars ahead used to judge accuracy
    favorable_threshold: float = 0.0  # forward return > this == favorable
    min_train_samples: int = 200
    calibration_method: str = "isotonic"  # or "sigmoid" (Platt)
    cv_folds: int = 3
    random_state: int = 7


@dataclass(frozen=True)
class DataConfig:
    exchange: str = "kraken"
    timeframe: str = "1d"
    history_limit: int = 400         # bars of OHLCV history to pull
    quote: str = "USDT"
    request_timeout_ms: int = 15000
    use_synthetic_fallback: bool = True
    enrich_fundamentals: bool = True  # CoinGecko market/supply/dev data
    enrich_news: bool = True          # real headlines from crypto RSS feeds


@dataclass(frozen=True)
class AppConfig:
    asset_class: str = "crypto"  # crypto | equity | commodity
    universe: list[str] = field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
            "LINK/USDT",
            "MATIC/USDT",
        ]
    )
    factor: FactorConfig = field(default_factory=FactorConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    bubble: BubbleConfig = field(default_factory=BubbleConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    data: DataConfig = field(default_factory=DataConfig)


DEFAULT_CONFIG = AppConfig()

# Default universes per asset class (used by the CLI when --symbols is omitted).
DEFAULT_UNIVERSES: dict[str, list[str]] = {
    "crypto": list(DEFAULT_CONFIG.universe),
    "equity": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
        "TSLA", "JPM", "XOM", "JNJ", "WMT", "KO",
    ],
    "commodity": [
        "GC=F",  # gold
        "SI=F",  # silver
        "CL=F",  # crude oil (WTI)
        "NG=F",  # natural gas
        "HG=F",  # copper
        "ZC=F",  # corn
        "ZW=F",  # wheat
        "ZS=F",  # soybeans
    ],
}

