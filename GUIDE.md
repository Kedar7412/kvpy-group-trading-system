# Asset Scorer — The Complete Guide (A to Z, in plain English)

This guide explains **what this tool is, how it thinks, and how to use it** —
with no assumptions about prior knowledge. If a word looks technical, check the
**Glossary** at the bottom.

> ⚠️ This is a research/learning tool. It is **not financial advice**. Do not
> trade real money based on it.

---

## A. What is this, in one sentence?

It looks at many tradable things (crypto coins, stocks, commodities like gold or
oil), gives each one a **score from 0 to 100** for how attractive it looks
today, and also tells you **how much to trust that score**.

- **High score (closer to 100)** = looks attractive right now.
- **Low score (closer to 0)** = looks weak, probably ignore.
- **Confidence %** = how reliable today's score is (more on this below).

It updates whenever you run it (e.g. once a day), and it is built to favor
**real strength over hype/FOMO bubbles**.

---

## B. The big idea: two different numbers

Most tools give you one number and call it a day. This one gives you **two**,
because they answer different questions:

1. **Score (0–100): "How good does this look?"**
   A ranking. Compares each asset against the others *today*.
2. **Confidence (0–100%): "How much should I trust that score?"**
   Some days the signal is clear; some days it's noise. Confidence tells you
   which is which.

Think of a weather forecast: the **score** is "it will be sunny," and the
**confidence** is "70% chance." Both matter.

---

## C. How a score is built (the 5 factors)

Each asset is examined through **five lenses** ("factors"). Each lens produces
its own mini-score from 0 to 100.

| Factor | Plain-English question | Examples of what it looks at |
|---|---|---|
| **News** | Is the story positive, and is it overheating? | recent headlines, how much buzz/attention |
| **Technicals** | Is the price trending up healthily? | trend direction, momentum, how wild the swings are |
| **Fundamentals** | Is there real value here (not just hype)? | crypto: liquidity, token supply/dilution · stocks: P/E, cash flow, profit, growth, debt |
| **Orderflow** | Are buyers actually stepping in? | buy vs sell pressure, accumulation |
| **Indicators** | What do classic trading signals say? | RSI, MACD, Bollinger Bands |

A column in the output shows each factor's mini-score, so you can always see
**why** an asset got its overall score.

---

## D. How the 5 factors are combined (flexible weights)

The five mini-scores are blended into one final score. But they are **not**
weighted equally, and the weights are **not the same for every asset**.

The tool measures, from history, **which factors actually predicted that
asset's future moves** (this measurement is called the *Information
Coefficient*, or IC). Factors that worked get more weight; factors that didn't
get less.

Plain example: for a meme coin driven by hype, **news** may earn a big weight.
For a stable blue-chip stock, **fundamentals** may dominate. The tool figures
this out per asset, automatically.

---

## E. The anti-bubble rule (why hype gets punished)

A big problem with simple scoring: things that are mooning on pure hype look
"great" right when they're most dangerous.

So there's a deliberate **anti-bubble penalty**. If an asset is **hot**
(overbought, going parabolic, or drowning in hype) **but** its fundamentals and
real buying don't back it up, its score gets pushed **down**.

- Hot **and** backed by real value/buying → barely touched.
- Hot **and** unconfirmed (pure FOMO) → discounted.

The "Bubble" column tells you the verdict in words:
- `normal` — nothing alarming.
- `warming` — getting a little hot.
- `hot but confirmed` — hot, but real strength supports it.
- `bubble-risk: hot & unconfirmed` — careful, this looks like FOMO.

---

## F. The confidence number (the "quant math")

This answers: **"What's the probability today's score is right?"**

How it's built, simply:
1. Look back through history. Each past day, did a high score actually lead to a
   good result, and a low score to a bad one?
2. Train a small statistical model to learn that relationship.
3. **Calibrate** it so the percentages mean what they say — i.e. when it says
   "70%," it really wins about 70% of the time (not 50%, not 90%).
4. Grade the model honestly on data it didn't train on, using a **Brier score**
   and a **skill score** (explained in the Glossary).

Bottom line: **confidence is a calibrated probability**, not a vibe.

---

## G. Install (one time)

You need Python 3.10 or newer.

```bash
# from the project folder
python3 -m venv .venv
.venv/bin/pip install -e .
```

That installs the `asset-scorer` command. (On Windows the path is
`.venv\Scripts\asset-scorer`.)

---

## H. Your first run

```bash
asset-scorer
```

This scores a default basket of crypto coins using live data and prints a table.
If the internet or data source is unavailable, it automatically uses **demo
(synthetic) data** so you still see it work — those rows are marked with `*`.

Want to see it instantly without any network?

```bash
asset-scorer --synthetic
```

---

## I. Reading the output table

```
#  Symbol   Score  Conf  P(fav)  News Tech Fund Flow  Ind  Bubble
1  ADA/USDT  72.1   51%    51%     85   81   50   74   74  normal
```

- **#** — rank (1 = most attractive today).
- **Symbol** — the asset (a `*` means demo/synthetic data).
- **Score** — the final 0–100 score (color-coded: green good, red weak).
- **Conf** — confidence: how much to trust this score.
- **P(fav)** — the model's probability of a favorable move.
- **News / Tech / Fund / Flow / Ind** — the five factor mini-scores (0–100).
- **Bubble** — the anti-bubble verdict (see section E).

Below the table you'll see **diagnostics**: calibration quality and a backtest
(does a higher score really tend to mean a better result?).

---

## J. The commands (what you can do)

The tool has five sub-commands. `score` is the default.

### 1) `score` — score a universe right now
```bash
asset-scorer score --asset-class equity --symbols AAPL MSFT NVDA JPM
asset-scorer score --top 5                 # show only the top 5
asset-scorer score --json report.json      # also save a JSON report
asset-scorer score --save --db scores.db   # save results to the database
```

### 2) `daily` — score everything and save (for automation)
```bash
asset-scorer daily --db scores.db          # crypto + equity + commodity
asset-scorer daily --classes crypto --db scores.db
```

### 3) `history` — see an asset's score over time (needs saved data)
```bash
asset-scorer history AAPL --asset-class equity --db scores.db
asset-scorer history BTC/USDT --db scores.db
```

### 4) `runs` — list past scoring runs and their quality
```bash
asset-scorer runs --db scores.db
```

### 5) `serve` — open the web dashboard
```bash
asset-scorer serve --db scores.db
# then open http://127.0.0.1:8000 in your browser
```
(Press `Ctrl+C` in the terminal to stop the dashboard.)

---

## K. The three asset classes

Pick with `--asset-class`:

| Class | What | Where the data comes from |
|---|---|---|
| `crypto` (default) | coins like BTC, ETH | kraken (prices) + CoinGecko (fundamentals) + crypto news sites |
| `equity` | stocks like AAPL, JPM | Yahoo Finance (prices, financials, news) |
| `commodity` | futures like gold (`GC=F`), oil (`CL=F`) | Yahoo Finance |

If you don't pass `--symbols`, a sensible default list is used for each class.

---

## L. Useful options (cheat sheet)

| Option | What it does |
|---|---|
| `--asset-class crypto\|equity\|commodity` | choose the market |
| `--symbols A B C` | score specific assets |
| `--horizon N` | judge results N days ahead (default 5) |
| `--history N` | how many days of past data to pull (default 400) |
| `--top N` | only show the top N rows |
| `--synthetic` | use offline demo data |
| `--no-enrich` | skip the extra fundamentals/news fetch (faster) |
| `--save --db FILE` | save results to a database |
| `--json FILE` | write a full JSON report |

---

## M. Saving history and running it daily

The database (a single `.db` file) remembers every run, so you can watch scores
change over time.

**Save today's scores:**
```bash
asset-scorer daily --db scores.db
```

**Run it automatically every day (Linux/Mac, using cron):**
```cron
30 23 * * *  /full/path/.venv/bin/asset-scorer daily --db /full/path/scores.db
```
That line means "every day at 23:30, score everything and save it."

Re-running on the same day **updates** that day's row instead of duplicating it,
so you can run it as often as you like.

---

## N. The web dashboard

```bash
asset-scorer daily --db scores.db     # 1) put some data in
asset-scorer serve --db scores.db     # 2) start the dashboard
```
Open **http://127.0.0.1:8000**. You get:
- a ranked, color-coded score table,
- run quality stats at the top,
- a **line chart** of a chosen asset's score and confidence over time,
- a **bar chart** of its five factor scores,
- buttons to refresh with live or demo data.

Click any row to load its charts. Stop the server with `Ctrl+C`.

---

## O. Troubleshooting (common issues)

- **"It says synthetic / shows `*`."** The live source was unreachable, so it
  used demo data. Check your internet, or just try again.
- **Binance errors / "451".** Binance is blocked in some regions. This tool
  defaults to **kraken** for crypto. You can try `--exchange coinbase`.
- **A stock/coin shows no news.** There may simply be no recent matching
  headlines — that's normal, not an error.
- **The dashboard "hangs" the terminal.** That's expected — `serve` runs until
  you stop it with `Ctrl+C`. Open the browser in the meantime.
- **First run is slow.** It downloads market data and fundamentals. Use
  `--no-enrich` or `--synthetic` to go faster while testing.

---

## P. Honest limitations (please read)

- **Orderflow and news history are approximations.** Truly detailed historical
  order-book and news data isn't free, so those two factors use price/volume
  stand-ins for the past; only *today's* value uses live data.
- **The confidence is only as good as the data and history.** On short or noisy
  data, the model honestly reports low skill rather than pretending. It improves
  with more assets, longer history, and richer data.
- **This does not place trades** and is **not advice.** It's a thinking tool.

---

## Glossary (every technical term, in plain words)

- **Asset** — anything tradable (a coin, a stock, a commodity).
- **Factor** — one "lens" used to judge an asset (news, technicals, etc.).
- **Score** — the final 0–100 attractiveness number.
- **Confidence** — calibrated probability that the score's call is correct.
- **Normalization / z-score** — rescaling so different things can be compared
  fairly (like grading on a curve).
- **Cross-sectional** — comparing all assets against each other on the same day.
- **Information Coefficient (IC)** — a measure (roughly −1 to +1) of how well a
  factor predicted future moves. Positive = useful. Used to set weights.
- **Weights** — how much each factor counts toward the final score; set per
  asset based on what actually worked.
- **Anti-bubble penalty** — a rule that lowers the score of hot-but-unsupported
  (FOMO) assets.
- **RSI / MACD / Bollinger Bands** — classic technical indicators (momentum and
  price-range tools).
- **Forward return / horizon** — how the price moved over the next N days; used
  to check if scores were right.
- **Calibration** — adjusting probabilities so "70%" really means 70%.
- **Brier score** — a grade for probability forecasts; **lower is better**
  (0 = perfect).
- **Skill score** — how much better the model is than a naive guess; **above 0
  is good**, below 0 means no edge.
- **Backtest** — testing the scoring against historical data.
- **Synthetic data** — fake but realistic offline data so the tool always runs.
- **Enrichment** — the extra step that fetches real fundamentals and news.

---

## One-minute quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .   # install
asset-scorer --synthetic                              # see it work offline
asset-scorer --asset-class equity --symbols AAPL MSFT JPM   # real stocks
asset-scorer daily --db scores.db                     # save a run
asset-scorer serve --db scores.db                     # dashboard at :8000
```

That's the whole tool, A to Z. Have fun exploring — and remember, it's for
learning, not for betting the rent. 🙂
