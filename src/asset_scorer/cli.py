"""Command-line interface for the asset scoring engine.

Usage examples:
    asset-scorer                              # score the default crypto universe
    asset-scorer --symbols BTC/USDT ETH/USDT  # custom universe
    asset-scorer --exchange kraken --horizon 10
    asset-scorer --synthetic                  # force offline synthetic data
    asset-scorer --json report.json           # also write a JSON report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from rich.console import Console
from rich.table import Table

from .config import (
    AppConfig,
    DEFAULT_UNIVERSES,
)
from .data.provider import MarketDataProvider
from .data.providers import get_provider
from .engine import EngineResult, ScoringEngine
from .storage import DEFAULT_DB_PATH


def _build_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    asset_class = (args.asset_class or base.asset_class).lower()
    data = dataclasses.replace(
        base.data,
        exchange=args.exchange or base.data.exchange,
        timeframe=args.timeframe or base.data.timeframe,
        use_synthetic_fallback=True,
        history_limit=args.history or base.data.history_limit,
    )
    calib = base.calibration
    if args.horizon:
        calib = dataclasses.replace(base.calibration, forward_horizon=args.horizon)
    if args.no_enrich:
        data = dataclasses.replace(data, enrich_fundamentals=False, enrich_news=False)
    universe = getattr(args, "symbols", None) or DEFAULT_UNIVERSES.get(
        asset_class, base.universe
    )
    return dataclasses.replace(
        base, asset_class=asset_class, universe=universe, data=data, calibration=calib
    )


def _fetch(config: AppConfig, force_synthetic: bool) -> dict:
    if force_synthetic:
        synth = MarketDataProvider(config.data)
        return {s: synth._synthesize(s) for s in config.universe}
    provider = get_provider(config.asset_class, config.data)
    return provider.fetch_universe(config.universe)


def _color_score(value: float) -> str:
    if value != value:  # NaN
        return "dim"
    if value >= 65:
        return "bold green"
    if value >= 50:
        return "green"
    if value >= 35:
        return "yellow"
    return "red"


_CALL_STYLE = {
    "FAVORED": "bold green",
    "AVOID": "red",
    "AVOID-BUBBLE": "bold red",
    "NEUTRAL": "yellow",
    "NO-EDGE": "dim",
}


def _render_table(result: EngineResult, console: Console, top: int | None) -> None:
    title = (
        f"Asset Scores  |  {result.asset_class}  |  as of {result.as_of}  |  "
        f"horizon {result.horizon} bars  |  source: {result.data_source}"
    )
    table = Table(title=title, header_style="bold cyan", expand=False)
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    table.add_column("Call")
    table.add_column("Score", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Crash", justify="right")
    table.add_column("News", justify="right")
    table.add_column("Tech", justify="right")
    table.add_column("Fund", justify="right")
    table.add_column("Flow", justify="right")
    table.add_column("Ind", justify="right")

    rows = result.scores if top is None else result.scores[:top]
    for s in rows:
        fs = s.factor_scores
        call = s.recommendation.call
        crash = s.bubble.probability
        crash_style = "red" if crash >= 0.6 else ("yellow" if crash >= 0.35 else "dim")
        table.add_row(
            str(s.rank),
            s.symbol + (" *" if s.synthetic else ""),
            f"[{_CALL_STYLE.get(call, 'white')}]{call}[/]",
            f"[{_color_score(s.final_score)}]{s.final_score:5.1f}[/]",
            f"{s.confidence * 100:4.0f}%",
            f"[{crash_style}]{crash * 100:3.0f}%[/]",
            f"{fs.get('news', float('nan')):4.0f}",
            f"{fs.get('technicals', float('nan')):4.0f}",
            f"{fs.get('fundamentals', float('nan')):4.0f}",
            f"{fs.get('orderflow', float('nan')):4.0f}",
            f"{fs.get('indicators', float('nan')):4.0f}",
        )
    console.print(table)
    console.print(
        "[dim]Call: FAVORED / NEUTRAL / AVOID / AVOID-BUBBLE / NO-EDGE (abstain). "
        "Crash = P(drawdown ahead).[/]"
    )

    # Showcase the bullshit detector: list flagged assets and why.
    flagged = [s for s in rows if s.bubble.reasons and s.bubble.probability >= 0.35]
    if flagged:
        console.print("[bold]Bubble watch:[/]")
        for s in flagged[:6]:
            console.print(
                f"  [yellow]{s.symbol}[/] P(crash) {s.bubble.probability:.0%} "
                f"[dim]{s.bubble.label}[/] — {', '.join(s.bubble.reasons)}"
            )


def _render_diagnostics(result: EngineResult, console: Console) -> None:
    r = result.reliability
    bt = result.backtest

    cal = Table(title="Calibration (confidence quality)", header_style="bold magenta")
    cal.add_column("Metric")
    cal.add_column("Value", justify="right")
    cal.add_row("Fitted model", "yes" if r.fitted else "no (heuristic)")
    cal.add_row("Method", r.method)
    cal.add_row("Training samples", str(r.n_samples))
    cal.add_row("Base rate (favorable)", f"{r.base_rate:.3f}")
    if r.brier_score is not None:
        cal.add_row("Brier score", f"{r.brier_score:.4f}")
        cal.add_row("Brier baseline", f"{r.brier_baseline:.4f}")
        cal.add_row(
            "Skill score",
            f"[{'green' if (r.skill_score or 0) > 0 else 'red'}]{r.skill_score:.4f}[/]",
        )
    if r.note:
        cal.add_row("Note", r.note)
    console.print(cal)

    val = Table(title="Backtest (does higher score = better asset?)", header_style="bold magenta")
    val.add_column("Metric")
    val.add_column("Value", justify="right")
    d = bt.as_dict()
    val.add_row("Observations", str(d["n_observations"]))
    val.add_row("Information Coefficient", str(d["information_coefficient"]))
    val.add_row("Top-minus-bottom return", str(d["top_minus_bottom_return"]))
    val.add_row("Hit rate", str(d["hit_rate"]))
    console.print(val)

    ic = Table(title="Global factor IC (predictive power)", header_style="bold magenta")
    ic.add_column("Factor")
    ic.add_column("IC", justify="right")
    for f, v in result.global_ic.items():
        shown = f"{v:.4f}" if v == v else "n/a"
        ic.add_row(f, shown)
    console.print(ic)
    if result.bubble_model_note:
        console.print(f"[dim]bullshit detector: {result.bubble_model_note}[/]")


def _render_history(rows: list[dict], symbol: str, asset_class: str,
                    console: Console) -> None:
    if not rows:
        console.print(f"[yellow]No stored history for {symbol} ({asset_class}).[/]")
        return
    table = Table(
        title=f"Score history: {symbol} ({asset_class})",
        header_style="bold cyan",
    )
    table.add_column("as_of")
    table.add_column("Score", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("News", justify="right")
    table.add_column("Tech", justify="right")
    table.add_column("Fund", justify="right")
    table.add_column("Flow", justify="right")
    table.add_column("Ind", justify="right")
    table.add_column("Bubble")

    def _fmt(v):
        return f"{v:.0f}" if isinstance(v, (int, float)) else "-"

    for r in rows:
        table.add_row(
            r["as_of"],
            f"[{_color_score(r.get('final_score') or float('nan'))}]"
            f"{(r.get('final_score') or float('nan')):.1f}[/]",
            f"{(r.get('confidence') or 0) * 100:.0f}%",
            _fmt(r.get("news")), _fmt(r.get("technicals")),
            _fmt(r.get("fundamentals")), _fmt(r.get("orderflow")),
            _fmt(r.get("indicators")),
            r.get("bubble_label") or "-",
        )
    console.print(table)


def _render_runs(rows: list[dict], console: Console) -> None:
    if not rows:
        console.print("[yellow]No stored runs yet. Run `asset-scorer score --save`.[/]")
        return
    table = Table(title="Stored runs", header_style="bold cyan")
    for col in ("as_of", "class", "source", "assets", "skill", "IC", "hit", "saved_at"):
        table.add_column(col)
    for r in rows:
        def _num(v, fmt="{:.4f}"):
            return fmt.format(v) if isinstance(v, (int, float)) else "-"
        table.add_row(
            r["as_of"], r["asset_class"], r.get("data_source") or "-",
            str(r.get("n_assets") or "-"),
            _num(r.get("skill_score")),
            _num(r.get("information_coefficient")),
            _num(r.get("hit_rate"), "{:.2f}"),
            (r.get("created_at") or "")[:19],
        )
    console.print(table)


def _run_engine(config: AppConfig, force_synthetic: bool, console: Console) -> EngineResult:
    with console.status(f"[cyan]Fetching {config.asset_class} data..."):
        data = _fetch(config, force_synthetic)
    engine = ScoringEngine(config)
    with console.status("[cyan]Scoring assets..."):
        return engine.run(data)


def _cmd_score(args: argparse.Namespace, console: Console) -> int:
    config = _build_config(args)
    result = _run_engine(config, args.synthetic, console)

    _render_table(result, console, args.top)
    if not args.quiet:
        _render_diagnostics(result, console)
    if any(s.synthetic for s in result.scores):
        console.print(
            "[dim]* synthetic/offline data (source unreachable or --synthetic). "
            "Scores are illustrative, not live.[/]"
        )
    console.print(
        f"[dim]enrichment: real fundamentals for {result.n_real_fundamentals}/"
        f"{len(result.scores)} assets, real news for {result.n_real_news}/"
        f"{len(result.scores)} assets.[/]"
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result.as_dict(), fh, indent=2)
        console.print(f"[green]Wrote JSON report to {args.json}[/]")

    if args.save:
        from .storage import ScoreStore

        store = ScoreStore(args.db)
        n = store.save_result(result)
        console.print(f"[green]Saved {n} scores for {result.as_of} to {args.db}[/]")
    return 0


def _cmd_daily(args: argparse.Namespace, console: Console) -> int:
    """Score one or more asset classes and persist them (for cron/schedulers)."""
    from .storage import ScoreStore

    store = ScoreStore(args.db)
    classes = args.classes or ["crypto", "equity", "commodity"]
    total = 0
    for ac in classes:
        sub = argparse.Namespace(**vars(args))
        sub.asset_class = ac
        sub.symbols = None
        config = _build_config(sub)
        try:
            result = _run_engine(config, args.synthetic, console)
        except Exception as exc:
            console.print(f"[red]{ac}: failed ({type(exc).__name__}: {exc})[/]")
            continue
        n = store.save_result(result)
        total += n
        console.print(
            f"[green]{ac}: saved {n} scores for {result.as_of} "
            f"(source: {result.data_source})[/]"
        )
        if not args.quiet:
            _render_table(result, console, args.top)
    console.print(f"[bold green]Daily run complete: {total} scores saved to {args.db}[/]")
    return 0


def _cmd_history(args: argparse.Namespace, console: Console) -> int:
    from .storage import ScoreStore

    store = ScoreStore(args.db)
    rows = store.score_history(args.symbol, args.asset_class, args.limit)
    _render_history(rows, args.symbol, args.asset_class, console)
    return 0


def _cmd_runs(args: argparse.Namespace, console: Console) -> int:
    from .storage import ScoreStore

    store = ScoreStore(args.db)
    rows = store.list_runs(args.limit, args.asset_class if args.asset_class != "all" else None)
    _render_runs(rows, console)
    return 0


def _render_backtest(wf, console: Console) -> None:
    d = wf.as_dict()
    console.print(
        f"[bold cyan]Walk-forward backtest[/]  |  {d['asset_class']}  |  "
        f"horizon {d['horizon']}  |  {d['n_rebalances']} rebalances, "
        f"{d['n_observations']} obs  |  OOS IC: {d['oos_information_coefficient']}"
    )
    pt = Table(title="Out-of-sample portfolios (net of costs)", header_style="bold magenta")
    for c in ("Strategy", "Ann.return", "Sharpe", "Hit", "MaxDD", "Coverage"):
        pt.add_column(c, justify="right" if c != "Strategy" else "left")
    for v in d["portfolios"].values():
        ann = v["annualized"]
        pt.add_row(
            v["name"],
            f"[{'green' if ann > 0 else 'red'}]{ann:+.1%}[/]",
            f"{v['sharpe']:+.2f}",
            f"{v['hit_rate']:.0%}",
            f"{v['max_drawdown']:.1%}",
            f"{v['coverage']:.0%}",
        )
    console.print(pt)

    if d["calibration_by_confidence"]:
        ct = Table(
            title="Calibration: do higher-confidence calls win more? (trust artifact)",
            header_style="bold magenta",
        )
        for c in ("Confidence", "N", "Hit rate", "Mean fwd ret"):
            ct.add_column(c, justify="right" if c != "Confidence" else "left")
        for b in d["calibration_by_confidence"]:
            ct.add_row(
                b["confidence_bucket"], str(b["n"]),
                f"{b['hit_rate']:.0%}", f"{b['mean_forward_return']:+.4f}",
            )
        console.print(ct)

    det = d["detector_check"]
    if det.get("flagged_mean_forward_return") is not None:
        console.print(
            f"[bold]Bullshit-detector check:[/] flagged {det['flagged_n']} obs · "
            f"flagged avg fwd return [red]{det['flagged_mean_forward_return']:+.4f}[/] "
            f"vs calm [green]{det['calm_mean_forward_return']:+.4f}[/] "
            f"(flagged negative {det['flagged_negative_rate']:.0%} of the time)"
        )
    console.print(f"[dim]{d['note']}[/]")
    console.print(
        "[dim]Read: if the Selective book beats the benchmark with a positive "
        "Sharpe, there is real, tradeable edge. If not, the model says so honestly.[/]"
    )


def _cmd_backtest(args: argparse.Namespace, console: Console) -> int:
    from .backtest import WalkForwardBacktester

    config = _build_config(args)
    bt = config.backtest
    bt = dataclasses.replace(
        bt,
        min_train=args.min_train or bt.min_train,
        cost_bps=args.cost_bps if args.cost_bps is not None else bt.cost_bps,
        selective_confidence=args.selective_confidence or bt.selective_confidence,
    )
    config = dataclasses.replace(config, backtest=bt)

    with console.status(f"[cyan]Fetching {config.asset_class} data..."):
        data = _fetch(config, args.synthetic)
    engine = ScoringEngine(config)
    with console.status("[cyan]Running walk-forward backtest (fits per rebalance)..."):
        wf = WalkForwardBacktester(config).run(engine, data)
    _render_backtest(wf, console)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(wf.as_dict(), fh, indent=2)
        console.print(f"[green]Wrote backtest JSON to {args.json}[/]")
    return 0


def _close_panel(data):
    import pandas as pd
    return pd.concat({s: d.ohlcv["close"] for s, d in data.items()}, axis=1).sort_index()


def _cmd_backfill(args: argparse.Namespace, console: Console) -> int:
    """Replay history point-in-time and persist the calls into the ledger."""
    from .backtest import WalkForwardBacktester
    from .storage import ScoreStore

    config = _build_config(args)
    if args.min_train:
        config = dataclasses.replace(
            config, backtest=dataclasses.replace(config.backtest, min_train=args.min_train)
        )
    with console.status(f"[cyan]Fetching {config.asset_class} data..."):
        data = _fetch(config, args.synthetic)
    engine = ScoringEngine(config)
    store = ScoreStore(args.db)
    with console.status("[cyan]Replaying history & sealing point-in-time calls..."):
        wf = WalkForwardBacktester(config).run(
            engine, data, store=store, persist=True, source="backfill"
        )
    console.print(
        f"[green]Backfilled {wf.n_observations} calls across {wf.n_rebalances} "
        f"dates into {args.db}.[/]"
    )
    chain = store.verify_chain()
    console.print(
        f"[dim]Ledger now holds {chain['entries']} sealed entries "
        f"({'verified' if chain['ok'] else 'BROKEN'}).[/]"
    )
    console.print("[dim]Now run:  asset-scorer scorecard --db "
                  f"{args.db} --asset-class {config.asset_class}[/]")
    return 0


def _render_scorecard(d: dict, console: Console) -> None:
    led = d["ledger"]
    badge = ("[bold green]verified[/]" if led.get("ok") else "[bold red]TAMPERED[/]")
    console.print(
        f"[bold cyan]Live Scorecard[/]  |  {d['asset_class']}  |  horizon "
        f"{d['horizon']}  |  ledger: {badge} ({led.get('entries', 0)} entries)"
    )
    console.print(
        f"[dim]calls: {d['n_calls']} sealed · {d['n_matured']} matured · "
        f"{d['n_pending']} pending · dates {d['as_of_first']} -> {d['as_of_last']}[/]"
    )
    if d["n_matured"] == 0:
        console.print(f"[yellow]{d['note']}[/]")
        return

    acc = d["actionable_accuracy"]
    fmb = d["favored_minus_benchmark"]
    head = Table(title="Headline (matured calls only)", header_style="bold magenta")
    head.add_column("Metric")
    head.add_column("Value", justify="right")
    head.add_row("Actionable accuracy", f"{acc:.0%}" if acc is not None else "-")
    head.add_row("Abstention rate (NO-EDGE/NEUTRAL)",
                 f"{d['abstention_rate']:.0%}" if d["abstention_rate"] is not None else "-")
    head.add_row("FAVORED mean return",
                 f"{d['favored_mean_return']:+.4f}" if d["favored_mean_return"] is not None else "-")
    head.add_row("Benchmark mean return",
                 f"{d['benchmark_mean_return']:+.4f}" if d["benchmark_mean_return"] is not None else "-")
    if fmb is not None:
        head.add_row("FAVORED minus benchmark",
                     f"[{'green' if fmb > 0 else 'red'}]{fmb:+.4f}[/]")
    console.print(head)

    if d["by_call"]:
        ct = Table(title="By call type", header_style="bold magenta")
        for c in ("Call", "N", "Win rate", "Mean fwd ret"):
            ct.add_column(c, justify="right" if c != "Call" else "left")
        for b in d["by_call"]:
            ct.add_row(b["call"], str(b["n"]), f"{b['win_rate']:.0%}",
                       f"{b['mean_forward_return']:+.4f}")
        console.print(ct)

    if d["by_confidence"]:
        cf = Table(title="Calibration: higher confidence -> higher hit rate?",
                   header_style="bold magenta")
        for c in ("Confidence", "N", "Hit rate", "Mean fwd ret"):
            cf.add_column(c, justify="right" if c != "Confidence" else "left")
        for b in d["by_confidence"]:
            cf.add_row(b["confidence_bucket"], str(b["n"]), f"{b['hit_rate']:.0%}",
                       f"{b['mean_forward_return']:+.4f}")
        console.print(cf)

    det = d["detector"]
    if det.get("flagged_mean_forward_return") is not None:
        line = (f"[bold]Bullshit-detector:[/] {det['flagged_n']} flagged · "
                f"flagged avg fwd [red]{det['flagged_mean_forward_return']:+.4f}[/] "
                f"vs calm [green]{det['calm_mean_forward_return']:+.4f}[/]")
        if det.get("flagged_crash_rate") is not None:
            line += f" · flagged crash rate {det['flagged_crash_rate']:.0%}"
        console.print(line)

    ff = d["follow_favored"]
    if ff:
        console.print(
            f"[bold]Follow-the-FAVORED:[/] {ff['n_periods']} periods · "
            f"total {ff['total_return']:+.1%} · Sharpe-like {ff['sharpe_like']:+.2f} "
            f"[dim](illustrative, overlapping windows)[/]"
        )


def _cmd_scorecard(args: argparse.Namespace, console: Console) -> int:
    from .backtest import evaluate_scorecard
    from .storage import ScoreStore

    config = _build_config(args)
    store = ScoreStore(args.db)
    syms = store.symbols(config.asset_class) or config.universe
    config = dataclasses.replace(config, universe=syms)

    with console.status(f"[cyan]Fetching prices to grade {config.asset_class} calls..."):
        data = _fetch(config, args.synthetic)
    close = _close_panel(data)
    res = evaluate_scorecard(
        store, config.asset_class, close,
        horizon=config.calibration.forward_horizon,
        crash_drawdown=config.bubble.crash_drawdown,
    )
    _render_scorecard(res.as_dict(), console)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res.as_dict(), fh, indent=2)
        console.print(f"[green]Wrote scorecard JSON to {args.json}[/]")
    return 0


def _cmd_verify(args: argparse.Namespace, console: Console) -> int:
    from .storage import ScoreStore

    store = ScoreStore(args.db)
    res = store.verify_chain()
    if res["entries"] == 0:
        console.print("[yellow]Ledger is empty. Save some runs first (score --save / daily).[/]")
        return 0
    if res["ok"]:
        console.print(
            f"[bold green]Track record verified.[/] {res['entries']} sealed "
            f"entries, chain intact.\n[dim]head: {res['head']}[/]"
        )
    else:
        reason = {"chain": "ledger chain", "content": "score data"}.get(
            res.get("reason"), "history"
        )
        console.print(
            f"[bold red]Tamper detected[/] ({reason}) at ledger seq "
            f"{res['broken_at_seq']} ({res['as_of']} / {res['asset_class']}). "
            f"The saved record was altered after the fact."
        )
    return 0


def _cmd_serve(args: argparse.Namespace, console: Console) -> int:
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn/fastapi not installed. Run: pip install -e .[/]")
        return 1
    from .web import create_app

    app = create_app(args.db)
    console.print(
        f"[green]Asset Scorer dashboard -> http://{args.host}:{args.port}[/]  "
        f"[dim](db: {args.db})[/]"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--asset-class", choices=["crypto", "equity", "commodity"],
                   default="crypto", help="Asset class (data source + default universe)")
    p.add_argument("--exchange", help="ccxt exchange id for crypto (default: kraken)")
    p.add_argument("--timeframe", help="OHLCV timeframe (default: 1d)")
    p.add_argument("--horizon", type=int, help="Forward horizon in bars (default: 5)")
    p.add_argument("--history", type=int, help="Bars of history to pull (default: 400)")
    p.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    p.add_argument("--no-enrich", action="store_true",
                   help="Disable fundamentals + news enrichment")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset-scorer",
        description="Systematic multi-factor asset scoring with calibrated confidence.",
    )
    sub = parser.add_subparsers(dest="command")

    p_score = sub.add_parser("score", help="Score a universe now (default command)")
    _add_data_args(p_score)
    p_score.add_argument("--symbols", nargs="+", help="Symbols, e.g. BTC/USDT ETH/USDT")
    p_score.add_argument("--top", type=int, help="Only show the top N assets")
    p_score.add_argument("--json", metavar="PATH", help="Write a JSON report to PATH")
    p_score.add_argument("--quiet", action="store_true", help="Suppress diagnostics")
    p_score.add_argument("--save", action="store_true", help="Persist scores to the DB")

    p_daily = sub.add_parser("daily", help="Score asset classes and persist (for cron)")
    _add_data_args(p_daily)
    p_daily.add_argument("--classes", nargs="+",
                         choices=["crypto", "equity", "commodity"],
                         help="Asset classes to run (default: all three)")
    p_daily.add_argument("--top", type=int, help="Rows to show per class")
    p_daily.add_argument("--quiet", action="store_true", help="Don't print tables")

    p_hist = sub.add_parser("history", help="Show stored score history for a symbol")
    p_hist.add_argument("symbol", help="Symbol, e.g. BTC/USDT or AAPL")
    p_hist.add_argument("--asset-class", choices=["crypto", "equity", "commodity"],
                        default="crypto")
    p_hist.add_argument("--limit", type=int, default=60, help="Max rows")
    p_hist.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")

    p_runs = sub.add_parser("runs", help="List stored scoring runs")
    p_runs.add_argument("--asset-class", default="all")
    p_runs.add_argument("--limit", type=int, default=50)
    p_runs.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")

    p_bt = sub.add_parser("backtest", help="Walk-forward (out-of-sample) edge test")
    _add_data_args(p_bt)
    p_bt.add_argument("--symbols", nargs="+", help="Symbols to test")
    p_bt.add_argument("--min-train", type=int, help="Bars of history before first OOS call")
    p_bt.add_argument("--cost-bps", type=float, help="Transaction cost per leg (bps)")
    p_bt.add_argument("--selective-confidence", type=float,
                      help="Min confidence for the selective book")
    p_bt.add_argument("--json", metavar="PATH", help="Write backtest JSON to PATH")

    p_verify = sub.add_parser("verify", help="Verify the tamper-evident track record")
    p_verify.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")

    p_bf = sub.add_parser("backfill", help="Replay history & seal point-in-time calls")
    _add_data_args(p_bf)
    p_bf.add_argument("--symbols", nargs="+", help="Symbols to backfill")
    p_bf.add_argument("--min-train", type=int, help="Bars before first sealed call")

    p_sc = sub.add_parser("scorecard", help="Public scorecard: did our calls come true?")
    _add_data_args(p_sc)
    p_sc.add_argument("--symbols", nargs="+", help="Restrict to these symbols")
    p_sc.add_argument("--json", metavar="PATH", help="Write scorecard JSON to PATH")

    p_serve = sub.add_parser("serve", help="Launch the web dashboard")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path")

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Default to the `score` subcommand for backward-compatible flat usage
    # (e.g. `asset-scorer --asset-class equity`).
    known = {"score", "daily", "history", "runs", "backtest", "verify",
             "backfill", "scorecard", "serve"}
    help_flags = {"-h", "--help"}
    if not argv or (argv[0] not in known and argv[0] not in help_flags):
        argv = ["score"] + list(argv)

    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    handlers = {
        "score": _cmd_score,
        "daily": _cmd_daily,
        "history": _cmd_history,
        "runs": _cmd_runs,
        "backtest": _cmd_backtest,
        "verify": _cmd_verify,
        "backfill": _cmd_backfill,
        "scorecard": _cmd_scorecard,
        "serve": _cmd_serve,
    }
    return handlers[args.command](args, console)


if __name__ == "__main__":
    sys.exit(main())
