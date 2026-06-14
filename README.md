# asset-scorer

A systematic, daily multi-factor **scoring engine** for tradable assets
(crypto-first MVP) with a **calibrated confidence** = the probability that each
score's directional call is actually correct.

A higher score means *more desirable*; a lower score means *ignore*. Scores are
deliberately tilted toward **real value, not bubbles/FOMO**.

## How a score is built

```
data -> 5 factor signals -> cross-sectional normalization (0-100)
     -> flexible IC weights -> anti-bubble penalty -> composite score (0-100)
     -> calibrated confidence  + backtest validation
```

### The 5 factors
| Factor | Measures | Source (MVP) |
|---|---|---|
| **news** | narrative + attention/heat | lexicon on headlines; OHLCV attention proxy |
| **technicals** | trend + momentum, vol-penalized | OHLCV |
| **fundamentals** | real value / quality (anti-bubble) | liquidity, organic volume, valuation premium, accumulation |
| **orderflow** | who is actually buying | live book imbalance + CMF/OBV proxy |
| **indicators** | RSI, MACD, Bollinger | OHLCV |

Each factor's raw signal is normalized **cross-sectionally** (ranked against the
universe each day) into 0-100, where 50 = average asset, 100 = best today.

### Flexible per-asset weights
Weights are driven by each factor's **Information Coefficient** (rank
correlation between the factor and the forward return), blended from a *global*
view and a *per-asset* view, then anchored to neutral priors. So news can
dominate a hype-driven coin while fundamentals dominate a mature one.

### Anti-bubble tilt
A multiplicative penalty fires when an asset is *hot* (overbought RSI, parabolic
momentum, or extreme news heat) but **unconfirmed** by fundamentals + orderflow:
```
penalty = 1 - strength * heat_intensity * confirmation_deficit * (1 - floor)
```
Hot-and-confirmed moves are left essentially untouched; hot-and-unconfirmed FOMO
gets discounted.

### Confidence = "probability the score is accurate"
The score is a directional call (>=50 bullish). We label history (favorable
forward return or not), fit a logistic model on the factor scores, and
**calibrate** it (isotonic / Platt) so that "70% confidence" really wins ~70% of
the time. Quality is reported via out-of-fold **Brier score** and a **skill
score** (vs. always predicting the base rate). Confidence = P(favorable) for
bullish scores, 1 - P(favorable) for bearish.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

## Use

```bash
# Crypto (default) — kraken OHLCV + CoinGecko fundamentals + RSS news
asset-scorer
asset-scorer --symbols BTC/USDT ETH/USDT
asset-scorer --exchange coinbase --horizon 10

# Equities — Yahoo Finance OHLCV + real financials (P/E, FCF, ROE, growth) + news
asset-scorer --asset-class equity
asset-scorer --asset-class equity --symbols AAPL MSFT NVDA JPM

# Commodities — Yahoo futures (gold, oil, copper, grains)
asset-scorer --asset-class commodity

asset-scorer --synthetic           # offline deterministic data
asset-scorer --no-enrich           # skip fundamentals/news enrichment
asset-scorer --json report.json    # also write a JSON report
```

If a data source is unreachable, the engine falls back to **deterministic
synthetic data** (marked with `*`) so the pipeline always runs.

## Persistence & daily updates
Scores can be persisted to a SQLite database and tracked over time.

```bash
# Score and save to the DB (idempotent: re-running a day UPSERTs)
asset-scorer score --asset-class equity --save --db asset_scores.db

# Daily job: score all asset classes and persist (for cron/schedulers)
asset-scorer daily --db asset_scores.db

# Inspect what's stored
asset-scorer runs --db asset_scores.db
asset-scorer history AAPL --asset-class equity --db asset_scores.db
```

Schedule the daily job with cron (e.g. every day at 23:30):

```cron
30 23 * * *  /path/to/.venv/bin/asset-scorer daily --db /path/to/asset_scores.db
```

The `scores` table accumulates a per-asset time series (score, confidence, the
five factor scores, bubble label) that powers the history view and dashboard.

## Web dashboard
A FastAPI + single-page dashboard visualizes the stored scores.

```bash
asset-scorer daily --db asset_scores.db   # populate some data first
asset-scorer serve --db asset_scores.db   # http://127.0.0.1:8000
```

The dashboard shows a ranked, color-coded score table with run diagnostics
(calibration skill, Brier, IC, hit rate), a per-asset **score-history line
chart** (score + confidence over time), and a **factor-breakdown bar chart**.
Buttons trigger a live refresh (fetch + score + save) or a synthetic demo run.
API routes live under `/api/*` (`scores`, `history`, `runs`, `refresh`).

## Asset classes
The engine is asset-class agnostic — all providers return the same `AssetData`
shape, so the same five factors, weighting, anti-bubble, and calibration apply
to everything.

| Class | OHLCV | Fundamentals | News | Orderflow |
|---|---|---|---|---|
| **crypto** | ccxt (kraken) | CoinGecko (liquidity, float/dilution) | crypto RSS feeds | live book imbalance + OHLCV proxy |
| **equity** | yfinance | yfinance (P/E, FCF yield, ROE, margins, growth, leverage) | yfinance news | OHLCV proxy |
| **commodity** | yfinance futures | sparse → OHLCV proxy | yfinance news | OHLCV proxy |

## Output
- A ranked table: score, confidence, P(favorable), per-factor scores, anti-bubble label.
- Diagnostics: calibration quality (Brier/skill), backtest (IC, top-minus-bottom, hit rate), per-factor global IC.
- Optional JSON report with the full breakdown including per-asset weights.

## Extending further
- Wire a dedicated **news API** (dated, scored articles) into the providers for stronger, historical news signals.
- Add on-chain protocol revenue / active-address data to the crypto fundamentals snapshot.
- Plug in a paid **orderflow** feed (L2/tick) to replace the OHLCV flow proxy.

## Caveats (MVP)
- `orderflow` and `news` historical series are **proxies** from OHLCV; live order
  book and headlines only adjust *today's* value. Real historical L2/news data
  would strengthen their IC and the calibration.
- This is research tooling, **not investment advice**.
