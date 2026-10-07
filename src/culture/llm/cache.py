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

    def complete(self, req: Request) -> Response:
        key = request_key(self.name, req, self.inner.key_extra() if hasattr(self.inner, "key_extra") else None)
        if self.cache is not None and self.mode in ("replay", "replay_strict"):
            entry = self.cache.get(key)
            if entry is not None:
                return self._serve(req, key, entry)
            if self.mode == "replay_strict":
                raise CacheMiss(f"replay_strict: no cached response for {req.tag} {req.seed_tag} ({key[:12]})")
        t0 = time.perf_counter()
        resp = self.inner.complete(req)
        latency = round(time.perf_counter() - t0, 3)
        resp.request_key = key
        self.misses += 1
        if self.cache is not None and self.mode != "off":
            self.cache.put(key, {"response": resp.to_dict(), "run": req.meta.get("run"), "tag": req.tag,
                                 "seed_tag": req.seed_tag})
        self.ledger.record(req, resp, extra={"latency_s": latency})  # wall time of the backend call (volatile)
        return resp

    def complete_batch(self, reqs: list[Request]) -> list[Response]:
        return [self.complete(r) for r in reqs]

    # state that must survive a resume (simulated provider cache warmth for the stub)
    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict() if hasattr(self.inner, "state_dict") else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(state)
