# asset-scorer

![CI](https://github.com/Kedar7412/kvpy-group-trading-system/actions/workflows/ci.yml/badge.svg)
![daily-scorecard](https://github.com/Kedar7412/kvpy-group-trading-system/actions/workflows/scorecard.yml/badge.svg)

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

## Proving edge: the walk-forward backtest

A score product is only worth trusting if it's tested **out of sample**. The
`backtest` command does an honest walk-forward: at each rebalance it fits the
weights, confidence model, and bubble detector on **past data only**, scores the
unseen date, and records the realized forward return — net of transaction costs.

```bash
asset-scorer backtest --asset-class equity --history 650
asset-scorer backtest --symbols BTC/USDT ETH/USDT SOL/USDT --history 700
```

It reports an equal-weight benchmark, a long-only top-quantile book, a
long-short book, and a **selective** book (trades only high-confidence calls,
holds cash otherwise) — plus a **calibration-by-confidence table**: do
higher-confidence calls actually win more often? That table is the trust
artifact. If the selective book doesn't beat the benchmark after costs, the tool
says so honestly rather than overfitting.

## Calls & abstention (selective prediction)

Every asset gets an explicit **Call**, including the option to stay silent:

| Call | Meaning |
|---|---|
| `FAVORED` | high score with sufficient confidence |
| `AVOID` | low score with sufficient confidence |
| `AVOID-BUBBLE` | high crash probability — looks like hype, not value |
| `NEUTRAL` | nothing decisive |
| `NO-EDGE` | confidence below the floor — we abstain rather than guess |

Saying "no edge" most of the time is a feature, not a bug.

## Market regime filter

A long-biased scorer bleeds most by going long into a falling, choppy market.
The engine builds an equal-weight index from the universe and reads its trend,
breadth, and volatility to label each day **risk_on / neutral / risk_off**
(point-in-time, no look-ahead). FAVORED longs are suppressed in `risk_off`, and
the backtest adds a **regime-gated long book** that holds cash in risk-off. On a
wider 18-asset universe this lifted the out-of-sample IC and gave the
regime-gated book a materially better Sharpe than the naive long book.

Weights are also **regime-conditional**: factor weights are estimated from
*same-regime* history (with an all-history fallback when samples are thin), so
the model leans on whatever has actually worked in the current kind of market.

## Deployment (public scorecard URL)
The app serves the dashboard at `/` and a public, verifiable scorecard at
`/scorecard`. It's container- and PaaS-ready and **auto-seeds** a demo on first
boot so the page is never empty:

```bash
docker build -t asset-scorer . && docker run -p 8000:8000 -v data:/data asset-scorer
# or Render (render.yaml) / Fly (fly.toml) / Procfile hosts
```

`serve` reads `PORT`, `ASSET_SCORER_HOST`, and `ASSET_SCORER_DB` from the
environment. Schedule `asset-scorer daily` to append real calls so the record
compounds. Full instructions in [DEPLOY.md](DEPLOY.md).

## The bullshit detector (predictive bubble model)

The anti-bubble logic is a **calibrated model** that predicts the probability of
a forward drawdown from seven overheating signals (overbought RSI, parabolic
momentum, news hype, Bollinger stretch, premium over a long-run anchor,
volatility climax, vertical blow-off). It reports `P(crash)` per asset with
human-readable reasons, and discounts the score when an asset is hot **and**
unconfirmed by fundamentals + orderflow.

## Verifiable track record

Every saved run is sealed into a hash-chained, append-only **ledger**. `verify`
recomputes the chain *and* the score-content hashes, so any after-the-fact edit
to the history is detectable.

```bash
asset-scorer verify --db asset_scores.db
```

## Public live scorecard

The scorecard grades the calls we actually **sealed in the ledger** against what
prices really did — no cherry-picking, no look-ahead — with the integrity proof
attached. Populate history instantly with a point-in-time `backfill` (replays
the walk-forward and seals each date's calls), then grade them:

```bash
asset-scorer backfill --asset-class crypto --history 400 --db asset_scores.db
asset-scorer scorecard --asset-class crypto --db asset_scores.db
```

It reports: actionable accuracy, abstention rate, FAVORED-vs-benchmark spread,
accuracy by call type, **calibration by confidence** (do higher-confidence calls
win more?), a bullshit-detector check (do flagged names fall more?), and a
follow-the-FAVORED equity curve — alongside the ledger verification badge. Going
forward, `daily` keeps appending, so the record compounds. The same view is on
the dashboard (`/api/scorecard`).

## Web dashboard
A premium single-page dashboard (FastAPI + Three.js) with an interactive 3D
object that follows your cursor.

```bash
asset-scorer daily --db asset_scores.db   # populate some data first
asset-scorer serve --db asset_scores.db   # http://127.0.0.1:8000
```

It shows the live **regime badge**, color-coded scores with the **Call**
(FAVORED/AVOID/NO-EDGE…) and crash probability, run diagnostics, score-history
and factor-breakdown charts, and an in-page **Grade calls** scorecard. A
separate **public, read-only scorecard page** at `/scorecard` grades the sealed
calls and publishes the **ledger head hash** so anyone can verify the record.
API routes live under `/api/*` (`scores`, `history`, `runs`, `refresh`,
`scorecard`, `verify`).

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
