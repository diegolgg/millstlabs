"""Cumulative spend guard for a hosted (paid) OpenAI-compatible provider, modelled on Diego's cost guard on `main`
(src/millstlabs/flags/provider.py): reserve a conservative estimate before sending, settle the real cost after, and keep
the cap cumulative across restarts, runs and processes by keeping it in one sqlite file.

Table `calls(key TEXT PRIMARY KEY, reserved REAL, cost REAL, state TEXT, created REAL)`. States:

- `pending`: reserved, request in flight (or the process died mid-request: treated as uncertain).
- `complete`: settled; `cost` is the actual dollar cost from the provider's usage.
- `uncertain`: the request may have been billed (5xx, timeout mid-request, malformed body). The reservation is kept.
- `released`: the provider certainly did not bill (401/403/404/400, 429, connection refused); counts nothing.

Committed money = sum(cost of complete) + sum(reserved of pending and uncertain). `reserve` refuses (BudgetExhausted)
when committed + estimate > max_usd. Every read-check-write runs in a `BEGIN IMMEDIATE` transaction with a 30 s busy
timeout, so concurrent processes sharing the file can never jointly overshoot the cap. A connection is opened per
operation, so one guard is also safe to use from several threads.

A key whose earlier attempt is `pending` or `uncertain` is refused (UncertainOutcome) unless the caller explicitly
allows the retry; then the old row is moved to `<key>:uncertain:<ns>` with its reservation intact and a new row is
reserved. A `complete` or `released` row under the same key (a deliberate re-send, e.g. cache mode `off`) is moved to
`<key>:<state>:<ns>` for the audit trail.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .backend import approx_tokens

BUSY_TIMEOUT_S = 30.0
STATES = ("pending", "complete", "uncertain", "released")


class BudgetExhausted(RuntimeError):
    """The cumulative cap would be exceeded by this call's estimate. No request was sent."""


class UncertainOutcome(RuntimeError):
    """An earlier attempt of this request may have been billed; re-sending needs an explicit retry_uncertain."""


def estimate_usd(prompt_text: str, max_tokens: int, price_in_per_mtok: float, price_out_per_mtok: float) -> float:
    """Conservative pre-send estimate: (approx prompt tokens * 1.1 + 256 framing tokens) at the input price plus the full
    output allowance at the output price."""
    return ((approx_tokens(prompt_text) * 1.1 + 256) * price_in_per_mtok + max_tokens * price_out_per_mtok) / 1e6


def usage_cost_usd(prompt_tokens: int, completion_tokens: int, price_in_per_mtok: float,
                   price_out_per_mtok: float) -> float:
    """Actual cost from the provider's usage. `prompt_tokens` includes any provider-cached prefix, charged at the full
    input rate (never under-counts)."""
    return (prompt_tokens * price_in_per_mtok + completion_tokens * price_out_per_mtok) / 1e6


class SpendGuard:
    def __init__(self, path: str | Path, max_usd: float):
        self.path = Path(path)
        self.max_usd = float(max_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tx() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, reserved REAL, cost REAL, "
                       "state TEXT, created REAL)")

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_S, isolation_level=None)
        db.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_S * 1000)}")
        return db

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
        finally:
            db.close()

    @staticmethod
    def _committed(db: sqlite3.Connection) -> float:
        return db.execute("SELECT COALESCE(SUM(CASE WHEN state='complete' THEN cost "
                          "WHEN state IN ('pending','uncertain') THEN reserved ELSE 0 END), 0) FROM calls").fetchone()[0]

    # ------------------------------------------------------------------ API
    def spent_or_reserved(self) -> float:
        db = self._connect()
        try:
            return float(self._committed(db))
        finally:
            db.close()

    def state(self, key: str) -> str | None:
        db = self._connect()
        try:
            row = db.execute("SELECT state FROM calls WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            db.close()

    def reserve(self, key: str, estimate_usd: float, retry_uncertain: bool = False) -> None:
        """Reserve `estimate_usd` under `key` or raise BudgetExhausted / UncertainOutcome (nothing reserved then)."""
        with self._tx() as db:
            row = db.execute("SELECT state FROM calls WHERE key=?", (key,)).fetchone()
            if row is not None:
                st = row[0]
                if st in ("pending", "uncertain"):
                    if not retry_uncertain:
                        raise UncertainOutcome(
                            f"an earlier attempt of this request is '{st}' in {self.path}: it may have been billed. "
                            "No request was sent. Set llm.retry_uncertain: true to re-send it (the old reservation "
                            "is kept); there is no automatic paid retry.")
                    st = "uncertain"  # a pending row whose process died is an uncertain outcome
                db.execute("UPDATE calls SET key=?, state=? WHERE key=?", (f"{key}:{st}:{time.time_ns()}", st, key))
            committed = self._committed(db)
            if committed + estimate_usd > self.max_usd:
                raise BudgetExhausted(
                    f"spend cap ${self.max_usd:.4f} would be exceeded: ${committed:.4f} spent or reserved in "
                    f"{self.path} + this call's estimate ${estimate_usd:.6f}. No request was sent. Raise llm.max_usd "
                    "explicitly to spend more.")
            db.execute("INSERT INTO calls (key, reserved, cost, state, created) VALUES (?, ?, NULL, 'pending', ?)",
                       (key, float(estimate_usd), time.time()))

    def _set(self, key: str, state: str, cost: float | None = None) -> None:
        with self._tx() as db:
            cur = db.execute("UPDATE calls SET state=?, cost=COALESCE(?, cost) WHERE key=? AND state='pending'",
                             (state, cost, key))
            if cur.rowcount != 1:
                raise KeyError(f"no pending reservation {key!r} in {self.path}")

    def settle(self, key: str, cost_usd: float) -> None:
        self._set(key, "complete", float(cost_usd))

    def release(self, key: str) -> None:
        self._set(key, "released")

    def mark_uncertain(self, key: str) -> None:
        self._set(key, "uncertain")

    def totals(self) -> dict[str, Any]:
        db = self._connect()
        try:
            spent, calls = db.execute("SELECT COALESCE(SUM(cost),0), COUNT(*) FROM calls WHERE state='complete'").fetchone()
            reserved = db.execute("SELECT COALESCE(SUM(reserved),0) FROM calls WHERE state='pending'").fetchone()[0]
            uncertain, n_unc = db.execute("SELECT COALESCE(SUM(reserved),0), COUNT(*) FROM calls "
                                          "WHERE state='uncertain'").fetchone()
            released = db.execute("SELECT COUNT(*) FROM calls WHERE state='released'").fetchone()[0]
        finally:
            db.close()
        return {"spent": float(spent), "reserved": float(reserved), "uncertain": float(uncertain), "calls": int(calls),
                "uncertain_calls": int(n_unc), "released_calls": int(released), "cap": self.max_usd,
                "committed": float(spent) + float(reserved) + float(uncertain)}
