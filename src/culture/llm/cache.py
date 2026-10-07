"""Record / replay cache for LLM calls, keyed by sha256(backend, model, effort, system, messages, max_tokens, seed_tag,
and the backend's sampling parameters such as the seed and temperature).

Modes:
- `record`: always call the backend and store the response (overwrites).
- `replay`: serve hits, call and store on a miss (the default for runs; makes resume free).
- `replay_strict`: serve hits, raise `CacheMiss` on a miss (CI and re-analysis).
- `off`: no cache.

A hit recorded by the same run (same `meta.run`) is that run's own call being replayed after a crash. It is ledgered as
spend, flagged `replayed_within_run`, so a resumed run's cost curve equals the uninterrupted one. Hits from other runs
are ledgered as cached with zero new spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .backend import Backend, CostLedger, Request, Response

MODES = ("record", "replay", "replay_strict", "off")


class CacheMiss(KeyError):
    pass


def request_key(backend_name: str, req: Request, extra: dict[str, Any] | None = None) -> str:
    """`extra` holds backend sampling parameters (seed, temperature, ...); the stub has none, so its keys are unchanged."""
    material = req.key_material(backend_name)
    if extra:
        material["sampling"] = extra
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class CallCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f, sort_keys=True)
        os.replace(tmp, p)

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*.json"))


class CachedBackend:
    """Wraps a backend with the call cache and the cost ledger. This is the only object the agent layer calls."""

    def __init__(self, inner: Backend, cache: CallCache | None, mode: str = "replay", ledger: CostLedger | None = None):
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}")
        self.inner, self.cache, self.mode = inner, cache, mode
        self.ledger = ledger or CostLedger()
        self.name = inner.name
        self.hits = self.misses = 0

    def _serve(self, req: Request, key: str, entry: dict[str, Any]) -> Response:
        resp = Response.from_dict(entry["response"])
        resp.request_key = key
        same_run = entry.get("run") is not None and entry.get("run") == req.meta.get("run")
        resp.cached = not same_run
        if hasattr(self.inner, "observe"):
            self.inner.observe(req)  # keep simulated provider-side state (prompt-cache warmth) in step
        self.hits += 1
        self.ledger.record(req, resp, extra={"replayed_within_run": True} if same_run else None)
        return resp

    def _key(self, req: Request) -> str:
        return request_key(self.name, req, self.inner.key_extra() if hasattr(self.inner, "key_extra") else None)

    def _lookup(self, req: Request, key: str) -> dict[str, Any] | None:
        if self.cache is not None and self.mode in ("replay", "replay_strict"):
            entry = self.cache.get(key)
            if entry is not None:
                return entry
            if self.mode == "replay_strict":
                raise CacheMiss(f"replay_strict: no cached response for {req.tag} {req.seed_tag} ({key[:12]})")
        return None

    def _timed(self, req: Request) -> tuple[Response, float]:
        t0 = time.perf_counter()
        resp = self.inner.complete(req)
        return resp, round(time.perf_counter() - t0, 3)

    def _store(self, req: Request, key: str, resp: Response, latency: float) -> Response:
        resp.request_key = key
        self.misses += 1
        if self.cache is not None and self.mode != "off":
            self.cache.put(key, {"response": resp.to_dict(), "run": req.meta.get("run"), "tag": req.tag,
                                 "seed_tag": req.seed_tag})
        self.ledger.record(req, resp, extra={"latency_s": latency})  # wall time of the backend call (volatile)
        return resp

    def complete(self, req: Request) -> Response:
        key = self._key(req)
        entry = self._lookup(req, key)
        if entry is not None:
            return self._serve(req, key, entry)
        resp, latency = self._timed(req)
        return self._store(req, key, resp, latency)

    def complete_batch(self, reqs: list[Request]) -> list[Response]:
        """Serial unless the inner backend is configured with concurrency > 1 (a hosted provider). Then the cache misses
        are sent on a thread pool; cache writes and ledger rows are made afterwards on this thread, in request order,
        for every call that succeeded, before the first error (if any) is raised. A request repeated inside the batch
        is sent once and the repeat is served afterwards, exactly as in the serial path."""
        workers = int(getattr(getattr(self.inner, "config", None), "concurrency", 1) or 1)
        if workers <= 1 or len(reqs) <= 1:
            return [self.complete(r) for r in reqs]
        keys = [self._key(r) for r in reqs]
        entries = [self._lookup(r, k) for r, k in zip(reqs, keys)]
        first_of: dict[str, int] = {}
        send = [i for i, (k, e) in enumerate(zip(keys, entries)) if e is None and first_of.setdefault(k, i) == i]
        with ThreadPoolExecutor(max_workers=min(workers, len(send) or 1)) as ex:
            futs = {i: ex.submit(self._timed, reqs[i]) for i in send}
        out: list[Response | None] = [None] * len(reqs)
        error: BaseException | None = None
        for i, r in enumerate(reqs):
            if entries[i] is not None:
                out[i] = self._serve(r, keys[i], entries[i])
            elif i in futs:
                if futs[i].exception() is not None:
                    error = error or futs[i].exception()
                    continue
                out[i] = self._store(r, keys[i], *futs[i].result())
            elif error is None:  # a repeat of an earlier request in this batch
                out[i] = self.complete(r)
        if error is not None:
            raise error
        return out  # type: ignore[return-value]

    # state that must survive a resume (simulated provider cache warmth for the stub)
    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict() if hasattr(self.inner, "state_dict") else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(state)
