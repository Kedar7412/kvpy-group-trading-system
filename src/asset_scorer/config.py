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

    The priors are deliberately tilted toward fundamentals + orderflow (the
    'real value' factors) and away from pure momentum/indicators (which tend to
    chase what already moved). This prevents the model from becoming a
    momentum-chasing buy-the-top system.
    """

    # Neutral starting priors -- tilted toward value/flow, away from momentum.
    priors: dict[str, float] = field(
        default_factory=lambda: {
            "news": 0.12,
            "technicals": 0.10,      # de-emphasized: avoids buying the top
            "fundamentals": 0.35,    # the core 'real value' signal
            "orderflow": 0.30,       # genuine demand confirmation
            "indicators": 0.13,
        }
    )
    prior_strength: float = 0.6  # stronger prior anchoring (was 0.5)
    ic_floor: float = 0.0  # negative ICs are clamped (don't bet against a factor)
    min_weight: float = 0.02  # never let a factor go fully to zero
    regime_conditional: bool = True   # estimate weights from same-regime history
    regime_min_samples: int = 50      # below this, fall back to all-history weights


@dataclass(frozen=True)
class BubbleConfig:
    """Anti-bubble / "bullshit detector" parameters.

    The detector blends several overheating signals into a crash probability and
    discounts the score when an asset is hot AND unconfirmed by fundamentals +
    orderflow. When enough history exists, the probability is a *calibrated
    model* trained to predict forward drawdowns; otherwise a transparent
    heuristic blend is used.
    """

    rsi_hot: float = 75.0          # RSI above this is "overheated"
    momentum_hot_z: float = 1.5    # momentum z-score considered parabolic
    hype_hot: float = 70.0         # news/indicator heat threshold (0-100)
    confirm_threshold: float = 50.0  # fundamentals/orderflow below = no confirm
    penalty_floor: float = 0.45    # strongest possible multiplicative penalty
    strength: float = 1.0          # global multiplier on penalty intensity

    # Predictive crash model
    crash_lookahead: int = 20      # bars ahead to look for a drawdown
    crash_drawdown: float = 0.10   # lowered from 0.15: catch smaller drops too
    min_train_samples: int = 250   # below this -> heuristic probability
    bubble_flag_prob: float = 0.35  # lowered from 0.60: was never triggering on real data


@dataclass(frozen=True)
class RegimeConfig:
    """Market-regime detection (point-in-time) from the universe itself.

    The idea: don't go long into a falling, choppy, or panicking market. We
    build an equal-weight index from the universe and read its trend, breadth,
    and volatility to label each day risk_on / neutral / risk_off. FAVORED longs
    are only allowed in risk_on/neutral.
    """

    index_fast: int = 30          # tighter fast SMA (was 50)
    index_slow: int = 80          # tighter slow SMA (was 100)
    breadth_lookback: int = 40    # SMA each asset is compared against for breadth
    vol_lookback: int = 20        # realized-vol window on index returns
    vol_z_window: int = 90        # window for the vol z-score
    vol_hot_z: float = 1.0        # lowered from 1.25 -- trigger earlier
    breadth_weak: float = 0.45    # raised from 0.40 -- less tolerant
    breadth_strong: float = 0.60  # raised from 0.55 -- only trust broad rallies


@dataclass(frozen=True)
class AbstentionConfig:
    """Selective prediction: the courage to say 'no edge'.

    A high-quality call requires real conviction. Below the confidence floor the
    engine abstains ('NO-EDGE') rather than emit a noisy score. High crash
    probability forces an AVOID regardless of score.

    The AVOID side is the stronger signal (assets to stay away from are more
    predictable than assets to buy). The FAVORED threshold is deliberately
    cautious while AVOID fires more readily.
    """

    min_confidence: float = 0.50     # lowered from 0.55 -- was over-abstaining
    favored_score: float = 55.0      # lowered from 58 -- give it room to call
    avoid_score: float = 45.0        # widened from 42 -- AVOID is the strong side
    bubble_avoid_prob: float = 0.35  # lowered from 0.60 -- was never triggering


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward (out-of-sample) backtest parameters."""

    min_train: int = 150          # bars of history before the first OOS call
    step: int | None = None       # rebalance spacing in bars (default = horizon)
    quantile: float = 0.34        # top/bottom fraction for long-short baskets
    cost_bps: float = 10.0        # round-trip transaction cost per rebalance leg
    selective_confidence: float = 0.55  # min confidence for the selective book
    retrain_calibration: bool = True     # refit confidence each rebalance


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
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
            "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT",
            "ATOM/USDT", "UNI/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT",
            "INJ/USDT", "APT/USDT", "SUI/USDT",
        ]
    )
    factor: FactorConfig = field(default_factory=FactorConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    bubble: BubbleConfig = field(default_factory=BubbleConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    data: DataConfig = field(default_factory=DataConfig)


DEFAULT_CONFIG = AppConfig()

# Default universes per asset class (used by the CLI when --symbols is omitted).
DEFAULT_UNIVERSES: dict[str, list[str]] = {
    "crypto": list(DEFAULT_CONFIG.universe),
    "equity": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
        "JPM", "XOM", "JNJ", "WMT", "KO", "CRM", "UBER", "PLTR", "COIN", "NFLX",
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

