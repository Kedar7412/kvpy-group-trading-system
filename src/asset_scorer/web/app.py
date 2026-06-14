"""FastAPI backend for the asset-scorer dashboard.

Read endpoints serve persisted data from the SQLite ``ScoreStore``; a refresh
endpoint runs the engine live for one asset class and saves the result.

Endpoints
---------
GET  /                         -> the single-page dashboard
GET  /api/asset-classes        -> classes that have stored scores
GET  /api/scores?asset_class=  -> latest scores for a class (ranked)
GET  /api/runs?asset_class=    -> stored run diagnostics
GET  /api/history?symbol=&asset_class=  -> per-asset score time series
POST /api/refresh?asset_class= -> score + persist now (live data)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ..storage import DEFAULT_DB_PATH, ScoreStore

_STATIC = Path(__file__).parent / "static"


def _parse_json_fields(row: dict) -> dict:
    """Expand the *_json text columns into real objects for the API."""
    out = dict(row)
    for key in ("weights_json", "fundamentals_json", "global_ic_json"):
        if key in out and isinstance(out[key], str):
            try:
                out[key.replace("_json", "")] = json.loads(out[key])
            except (ValueError, TypeError):
                out[key.replace("_json", "")] = None
            out.pop(key, None)
    return out


def create_app(db_path: str = DEFAULT_DB_PATH) -> FastAPI:
    app = FastAPI(title="Asset Scorer", version="0.1.0")
    store = ScoreStore(db_path)
    app.state.db_path = db_path

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = _STATIC / "index.html"
        if not html.exists():
            return "<h1>Dashboard asset missing</h1>"
        return html.read_text(encoding="utf-8")

    @app.get("/api/asset-classes")
    def asset_classes() -> list[str]:
        classes = store.asset_classes()
        return classes or ["crypto", "equity", "commodity"]

    @app.get("/api/scores")
    def scores(asset_class: str = Query("crypto")) -> JSONResponse:
        rows = [_parse_json_fields(r) for r in store.latest_scores(asset_class)]
        as_of = store.latest_as_of(asset_class)
        runs = store.list_runs(1, asset_class)
        return JSONResponse(
            {
                "asset_class": asset_class,
                "as_of": as_of,
                "run": runs[0] if runs else None,
                "scores": rows,
            }
        )

    @app.get("/api/runs")
    def runs(asset_class: str | None = Query(None), limit: int = Query(50)) -> list:
        return store.list_runs(limit, asset_class)

    @app.get("/api/verify")
    def verify() -> JSONResponse:
        return JSONResponse(store.verify_chain())

    @app.get("/api/scorecard")
    def scorecard(
        asset_class: str = Query("crypto"),
        synthetic: bool = Query(False),
    ) -> JSONResponse:
        """Grade the sealed calls against realized prices. Sync -> threadpool."""
        import dataclasses

        import pandas as pd

        from ..backtest import evaluate_scorecard
        from ..config import DEFAULT_UNIVERSES, AppConfig
        from ..data.provider import MarketDataProvider
        from ..data.providers import get_provider

        syms = store.symbols(asset_class) or DEFAULT_UNIVERSES.get(asset_class, [])
        if not syms:
            return JSONResponse({"asset_class": asset_class, "n_calls": 0,
                                 "note": "No stored calls. Run backfill or daily first."})
        base = AppConfig()
        config = dataclasses.replace(base, asset_class=asset_class, universe=syms)
        try:
            if synthetic:
                prov = MarketDataProvider(config.data)
                data = {s: prov._synthesize(s) for s in syms}
            else:
                data = get_provider(asset_class, config.data).fetch_universe(syms)
            close = pd.concat(
                {s: d.ohlcv["close"] for s, d in data.items()}, axis=1
            ).sort_index()
            res = evaluate_scorecard(
                store, asset_class, close,
                horizon=config.calibration.forward_horizon,
                crash_drawdown=config.bubble.crash_drawdown,
            )
            return JSONResponse(res.as_dict())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    @app.get("/api/history")
    def history(
        symbol: str = Query(...),
        asset_class: str = Query("crypto"),
        limit: int = Query(365),
    ) -> JSONResponse:
        rows = store.score_history(symbol, asset_class, limit)
        return JSONResponse({"symbol": symbol, "asset_class": asset_class, "history": rows})

    @app.get("/api/symbols")
    def symbols(asset_class: str = Query("crypto")) -> list[str]:
        return store.symbols(asset_class)

    @app.post("/api/refresh")
    def refresh(
        asset_class: str = Query("crypto"),
        synthetic: bool = Query(False),
    ) -> JSONResponse:
        """Run the engine live for one asset class and persist the result.

        Defined as a sync function so FastAPI runs it in a worker thread and the
        event loop stays responsive during the (slow) data fetch.
        """
        try:
            from ..config import DEFAULT_UNIVERSES, AppConfig
            from ..data.provider import MarketDataProvider
            from ..data.providers import get_provider
            from ..engine import ScoringEngine
            import dataclasses

            base = AppConfig()
            universe = DEFAULT_UNIVERSES.get(asset_class, base.universe)
            config = dataclasses.replace(
                base, asset_class=asset_class, universe=universe
            )
            if synthetic:
                synth = MarketDataProvider(config.data)
                data = {s: synth._synthesize(s) for s in universe}
            else:
                data = get_provider(asset_class, config.data).fetch_universe(universe)
            result = ScoringEngine(config).run(data)
            n = store.save_result(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
        return JSONResponse(
            {"asset_class": asset_class, "as_of": result.as_of, "saved": n}
        )

    return app
