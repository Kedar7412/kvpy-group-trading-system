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

import hashlib
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
    bubble_probability    REAL,
    call                  TEXT,
    rationale             TEXT,
    last_price            REAL,
    synthetic             INTEGER,
    weights_json          TEXT,
    fundamentals_json     TEXT,
    PRIMARY KEY (as_of, asset_class, symbol)
);

CREATE INDEX IF NOT EXISTS idx_scores_symbol
    ON scores (asset_class, symbol, as_of);

-- Tamper-evident append-only audit log. Each row is hash-chained to the
-- previous one, so the entire history of saved runs can be verified and any
-- after-the-fact edit is detectable. This is the seed of a public, auditable
-- track record.
CREATE TABLE IF NOT EXISTS ledger (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of        TEXT NOT NULL,
    asset_class  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    n_scores     INTEGER,
    payload_hash TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    entry_hash   TEXT NOT NULL
);
"""

# Columns added after v0.1; applied to pre-existing databases by _migrate().
_MIGRATIONS = {
    "scores": {
        "bubble_probability": "REAL",
        "call": "TEXT",
        "rationale": "TEXT",
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_payload(as_of: str, asset_class: str, triples: list) -> str:
    """Canonical JSON of a run's scores -> the thing we hash.

    triples: iterable of (symbol, final_score_or_None, call). Sorted so the
    hash is independent of row order.
    """
    rows = sorted(
        [
            [t[0], (round(float(t[1]), 4) if t[1] is not None and t[1] == t[1] else None), t[2]]
            for t in triples
        ]
    )
    return json.dumps(
        {"as_of": as_of, "asset_class": asset_class, "scores": rows}, sort_keys=True
    )


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
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created."""
        with self._connect() as conn:
            for table, cols in _MIGRATIONS.items():
                existing = {
                    r["name"] for r in conn.execute(f"PRAGMA table_info({table})")
                }
                for col, decl in cols.items():
                    if col not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

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
                        bubble_label, bubble_penalty, bubble_probability, call,
                        rationale, last_price, synthetic, weights_json, fundamentals_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        bubble_probability=excluded.bubble_probability,
                        call=excluded.call, rationale=excluded.rationale,
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
                        s.bubble.label, _f(s.bubble.penalty), _f(s.bubble.probability),
                        s.recommendation.call, s.recommendation.rationale,
                        _f(s.last_price), int(s.synthetic), json.dumps(s.weights),
                        json.dumps(s.fundamentals),
                    ),
                )
                rows += 1

            self._append_ledger(conn, result, now)
            return rows

    def _append_ledger(self, conn, result: "EngineResult", now: str) -> None:
        """Append a hash-chained, tamper-evident entry for an EngineResult."""
        triples = [
            (s.symbol,
             s.final_score if s.final_score == s.final_score else None,
             s.recommendation.call)
            for s in result.scores
        ]
        self._append_ledger_triples(
            conn, result.as_of, result.asset_class, triples, now
        )

    def _append_ledger_triples(self, conn, as_of, asset_class, triples, now) -> None:
        payload = _canonical_payload(as_of, asset_class, triples)
        payload_hash = hashlib.sha256(payload.encode()).hexdigest()
        prev = conn.execute(
            "SELECT entry_hash FROM ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_hash = prev["entry_hash"] if prev else "0" * 64
        entry_hash = hashlib.sha256((prev_hash + payload_hash).encode()).hexdigest()
        conn.execute(
            """
            INSERT INTO ledger (
                as_of, asset_class, created_at, n_scores,
                payload_hash, prev_hash, entry_hash
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (as_of, asset_class, now, len(triples), payload_hash, prev_hash, entry_hash),
        )

    def save_calls(
        self, as_of: str, asset_class: str, rows: list[dict], *,
        source: str = "backfill", synthetic: bool = False, horizon: int | None = None,
    ) -> int:
        """Persist a batch of point-in-time calls (used by backfill).

        ``rows`` items are plain dicts with keys: symbol, rank, final_score,
        base_score, confidence, favorable_probability, news, technicals,
        fundamentals, orderflow, indicators, bubble_probability, bubble_label,
        bubble_penalty, call, rationale, last_price. Inserts a minimal run row,
        the score rows (UPSERT), and one hash-chained ledger entry.
        """
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (as_of, asset_class, created_at, horizon,
                    data_source, n_assets, n_synthetic)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(as_of, asset_class) DO UPDATE SET
                    created_at=excluded.created_at, horizon=excluded.horizon,
                    data_source=excluded.data_source, n_assets=excluded.n_assets,
                    n_synthetic=excluded.n_synthetic
                """,
                (as_of, asset_class, now, horizon, source, len(rows),
                 len(rows) if synthetic else 0),
            )
            for d in rows:
                conn.execute(
                    """
                    INSERT INTO scores (
                        as_of, asset_class, symbol, created_at, rank,
                        final_score, base_score, confidence, favorable_probability,
                        news, technicals, fundamentals, orderflow, indicators,
                        bubble_label, bubble_penalty, bubble_probability, call,
                        rationale, last_price, synthetic, weights_json, fundamentals_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(as_of, asset_class, symbol) DO UPDATE SET
                        created_at=excluded.created_at, rank=excluded.rank,
                        final_score=excluded.final_score, base_score=excluded.base_score,
                        confidence=excluded.confidence,
                        favorable_probability=excluded.favorable_probability,
                        news=excluded.news, technicals=excluded.technicals,
                        fundamentals=excluded.fundamentals, orderflow=excluded.orderflow,
                        indicators=excluded.indicators, bubble_label=excluded.bubble_label,
                        bubble_penalty=excluded.bubble_penalty,
                        bubble_probability=excluded.bubble_probability,
                        call=excluded.call, rationale=excluded.rationale,
                        last_price=excluded.last_price, synthetic=excluded.synthetic
                    """,
                    (
                        as_of, asset_class, d["symbol"], now, d.get("rank"),
                        _f(d.get("final_score")), _f(d.get("base_score")),
                        _f(d.get("confidence")), _f(d.get("favorable_probability")),
                        _f(d.get("news")), _f(d.get("technicals")),
                        _f(d.get("fundamentals")), _f(d.get("orderflow")),
                        _f(d.get("indicators")),
                        d.get("bubble_label"), _f(d.get("bubble_penalty")),
                        _f(d.get("bubble_probability")), d.get("call"),
                        d.get("rationale"), _f(d.get("last_price")),
                        int(synthetic), None, None,
                    ),
                )
            triples = [
                (d["symbol"],
                 d.get("final_score") if d.get("final_score") is not None
                 and d.get("final_score") == d.get("final_score") else None,
                 d.get("call"))
                for d in rows
            ]
            self._append_ledger_triples(conn, as_of, asset_class, triples, now)
            return len(rows)

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

    def distinct_as_of(self, asset_class: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT as_of FROM scores WHERE asset_class=? ORDER BY as_of",
                (asset_class,),
            ).fetchall()
        return [r["as_of"] for r in rows]

    def scores_on(self, as_of: str, asset_class: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scores WHERE as_of=? AND asset_class=? ORDER BY rank",
                (as_of, asset_class),
            ).fetchall()
        return [dict(r) for r in rows]

    def all_scores(self, asset_class: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scores WHERE asset_class=? ORDER BY as_of, rank",
                (asset_class,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- verifiable track record -----------------------------------------
    def ledger_entries(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ledger ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def verify_chain(self) -> dict:
        """Verify both the hash chain AND that the scores match what was sealed.

        Two checks:
          1. chain integrity: each entry's prev/entry hashes are consistent;
          2. content integrity: recomputing the score hash from the current
             `scores` table matches the latest sealed payload for that run.
        Either failing means the history was altered after the fact.
        """
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM ledger ORDER BY seq ASC"
            ).fetchall()]
        prev = "0" * 64
        latest_for_key: dict[tuple, dict] = {}
        for row in rows:
            expected = hashlib.sha256(
                (prev + row["payload_hash"]).encode()
            ).hexdigest()
            if row["prev_hash"] != prev or row["entry_hash"] != expected:
                return {
                    "ok": False, "reason": "chain", "entries": len(rows),
                    "broken_at_seq": row["seq"],
                    "as_of": row["as_of"], "asset_class": row["asset_class"],
                }
            prev = row["entry_hash"]
            latest_for_key[(row["as_of"], row["asset_class"])] = row

        # Content integrity against the current scores table.
        for (as_of, ac), row in latest_for_key.items():
            triples = [
                (r["symbol"], r.get("final_score"), r.get("call"))
                for r in self.scores_on(as_of, ac)
            ]
            recomputed = hashlib.sha256(
                _canonical_payload(as_of, ac, triples).encode()
            ).hexdigest()
            if recomputed != row["payload_hash"]:
                return {
                    "ok": False, "reason": "content", "entries": len(rows),
                    "broken_at_seq": row["seq"], "as_of": as_of, "asset_class": ac,
                }

        return {"ok": True, "entries": len(rows), "head": prev if rows else None}


def _f(value) -> float | None:
    """Coerce to a float SQLite can store; NaN/None become NULL."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN
