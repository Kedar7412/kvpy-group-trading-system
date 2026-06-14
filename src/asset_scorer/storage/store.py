"""SQLite persistence for scoring runs and per-asset score history.

Two tables:
  * ``runs``   - one row per (as_of, asset_class) scoring run with the
                 run-level diagnostics (calibration + backtest).
  * ``scores`` - one row per (as_of, asset_class, symbol) with the final score,
                 confidence, per-factor scores, weights, and bubble info.

Writes are idempotent: re-running the same day UPSERTs (so scheduled daily runs
never create duplicates but always reflect the latest data). The ``scores``
table is the time series that powers history queries and the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycle at runtime
    from ..engine import EngineResult

DEFAULT_DB_PATH = "asset_scores.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    as_of            TEXT NOT NULL,
    asset_class      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    horizon          INTEGER,
    data_source      TEXT,
    n_assets         INTEGER,
    n_synthetic      INTEGER,
    calib_fitted     INTEGER,
    calib_method     TEXT,
    base_rate        REAL,
    brier_score      REAL,
    skill_score      REAL,
    information_coefficient REAL,
    top_minus_bottom REAL,
    hit_rate         REAL,
    global_ic_json   TEXT,
    PRIMARY KEY (as_of, asset_class)
);

CREATE TABLE IF NOT EXISTS scores (
    as_of                 TEXT NOT NULL,
    asset_class           TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    rank                  INTEGER,
    final_score           REAL,
    base_score            REAL,
    confidence            REAL,
    favorable_probability REAL,
    news                  REAL,
    technicals            REAL,
    fundamentals          REAL,
    orderflow             REAL,
    indicators            REAL,
    bubble_label          TEXT,
    bubble_penalty        REAL,
    last_price            REAL,
    synthetic             INTEGER,
    weights_json          TEXT,
    fundamentals_json     TEXT,
    PRIMARY KEY (as_of, asset_class, symbol)
);

CREATE INDEX IF NOT EXISTS idx_scores_symbol
    ON scores (asset_class, symbol, as_of);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScoreStore:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = str(path)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- writes -----------------------------------------------------------
    def save_result(self, result: "EngineResult") -> int:
        """Persist one EngineResult. Returns the number of asset rows written."""
        now = _now_iso()
        r = result.reliability
        bt = result.backtest
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    as_of, asset_class, created_at, horizon, data_source,
                    n_assets, n_synthetic, calib_fitted, calib_method, base_rate,
                    brier_score, skill_score, information_coefficient,
                    top_minus_bottom, hit_rate, global_ic_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(as_of, asset_class) DO UPDATE SET
                    created_at=excluded.created_at,
                    horizon=excluded.horizon,
                    data_source=excluded.data_source,
                    n_assets=excluded.n_assets,
                    n_synthetic=excluded.n_synthetic,
                    calib_fitted=excluded.calib_fitted,
                    calib_method=excluded.calib_method,
                    base_rate=excluded.base_rate,
                    brier_score=excluded.brier_score,
                    skill_score=excluded.skill_score,
                    information_coefficient=excluded.information_coefficient,
                    top_minus_bottom=excluded.top_minus_bottom,
                    hit_rate=excluded.hit_rate,
                    global_ic_json=excluded.global_ic_json
                """,
                (
                    result.as_of, result.asset_class, now, result.horizon,
                    result.data_source, len(result.scores), result.n_synthetic,
                    int(r.fitted), r.method, r.base_rate, r.brier_score,
                    r.skill_score, bt.ic if bt.ic == bt.ic else None,
                    bt.top_minus_bottom if bt.top_minus_bottom == bt.top_minus_bottom else None,
                    bt.hit_rate if bt.hit_rate == bt.hit_rate else None,
                    json.dumps(result.global_ic),
                ),
            )

            rows = 0
            for s in result.scores:
                fs = s.factor_scores
                conn.execute(
                    """
                    INSERT INTO scores (
                        as_of, asset_class, symbol, created_at, rank,
                        final_score, base_score, confidence, favorable_probability,
                        news, technicals, fundamentals, orderflow, indicators,
                        bubble_label, bubble_penalty, last_price, synthetic,
                        weights_json, fundamentals_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(as_of, asset_class, symbol) DO UPDATE SET
                        created_at=excluded.created_at,
                        rank=excluded.rank,
                        final_score=excluded.final_score,
                        base_score=excluded.base_score,
                        confidence=excluded.confidence,
                        favorable_probability=excluded.favorable_probability,
                        news=excluded.news, technicals=excluded.technicals,
                        fundamentals=excluded.fundamentals, orderflow=excluded.orderflow,
                        indicators=excluded.indicators,
                        bubble_label=excluded.bubble_label,
                        bubble_penalty=excluded.bubble_penalty,
                        last_price=excluded.last_price, synthetic=excluded.synthetic,
                        weights_json=excluded.weights_json,
                        fundamentals_json=excluded.fundamentals_json
                    """,
                    (
                        result.as_of, result.asset_class, s.symbol, now, s.rank,
                        _f(s.final_score), _f(s.base_score), _f(s.confidence),
                        _f(s.favorable_probability),
                        _f(fs.get("news")), _f(fs.get("technicals")),
                        _f(fs.get("fundamentals")), _f(fs.get("orderflow")),
                        _f(fs.get("indicators")),
                        s.bubble.label, _f(s.bubble.penalty), _f(s.last_price),
                        int(s.synthetic), json.dumps(s.weights),
                        json.dumps(s.fundamentals),
                    ),
                )
                rows += 1
            return rows

    # -- reads ------------------------------------------------------------
    def list_runs(self, limit: int = 50, asset_class: str | None = None) -> list[dict]:
        q = "SELECT * FROM runs"
        params: list = []
        if asset_class:
            q += " WHERE asset_class=?"
            params.append(asset_class)
        q += " ORDER BY as_of DESC, asset_class LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    def latest_as_of(self, asset_class: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(as_of) AS m FROM scores WHERE asset_class=?",
                (asset_class,),
            ).fetchone()
        return row["m"] if row and row["m"] else None

    def latest_scores(self, asset_class: str) -> list[dict]:
        as_of = self.latest_as_of(asset_class)
        if as_of is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scores WHERE asset_class=? AND as_of=? ORDER BY rank",
                (asset_class, as_of),
            ).fetchall()
        return [dict(r) for r in rows]

    def score_history(
        self, symbol: str, asset_class: str, limit: int = 365
    ) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scores
                WHERE asset_class=? AND symbol=?
                ORDER BY as_of DESC LIMIT ?
                """,
                (asset_class, symbol, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def symbols(self, asset_class: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM scores WHERE asset_class=? ORDER BY symbol",
                (asset_class,),
            ).fetchall()
        return [r["symbol"] for r in rows]

    def asset_classes(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT asset_class FROM scores ORDER BY asset_class"
            ).fetchall()
        return [r["asset_class"] for r in rows]


def _f(value) -> float | None:
    """Coerce to a float SQLite can store; NaN/None become NULL."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN
