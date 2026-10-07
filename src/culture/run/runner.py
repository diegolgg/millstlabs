"""Run loop: config freeze, deployment lock, atomic checkpoints with log byte offsets, resume, wall-clock budget.

Resume contract: every generation ends with an atomic checkpoint holding the bounded run state and the byte length of
each JSONL log. A crash mid-generation leaves partial log lines; on restart the logs are truncated to the checkpointed
offsets, state is reloaded, and the generation re-runs. LLM calls made before the crash are replayed from the cache, so
the resumed run ends in the identical state (tested in test_runner.py)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, from_dict
from .context import LOGS, RunContext
from .generation import run_generation
from .manifest import build_manifest


class ConfigMismatch(RuntimeError):
    pass


class RunLocked(RuntimeError):
    pass


def _atomic_json(path: Path, obj: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, sort_keys=True, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Runner:
    def __init__(self, cfg: ExperimentConfig, out: str | Path):
        self.cfg = cfg
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(self.out / ".lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(self._lock_fd)
            raise RunLocked(f"{self.out} is in use by another runner") from e
        try:
            self._freeze_config()
            ckpt = self.out / "checkpoint.json"
            state = None
            if ckpt.exists():
                with open(ckpt) as f:
                    state = json.load(f)
                self._truncate(state["offsets"])
            else:
                self._truncate({n: 0 for n in LOGS})
            self.ctx = RunContext(cfg, self.out)
            if state is not None:
                self.ctx.load_state_dict(state["state"])
            self.resumed = state is not None
        except BaseException:
            self.close()
            raise

    def _freeze_config(self) -> None:
        cpath = self.out / "config.json"
        if cpath.exists():
            with open(cpath) as f:
                saved = json.load(f)
            if saved["digest"] != self.cfg.digest():
                raise ConfigMismatch(f"{self.out} was created with config {saved['digest']}; this config is "
                                     f"{self.cfg.digest()}. Use a new output directory.")
        else:
            _atomic_json(cpath, {"digest": self.cfg.digest(), "config": self.cfg.to_dict()})
            _atomic_json(self.out / "manifest.json", build_manifest(self.cfg))

    def _truncate(self, offsets: dict[str, int]) -> None:
        for n in LOGS:
            p = self.out / f"{n}.jsonl"
            off = offsets.get(n, 0)
            if p.exists() and p.stat().st_size > off:
                with open(p, "r+b") as f:
                    f.truncate(off)

    def checkpoint(self) -> None:
        _atomic_json(self.out / "checkpoint.json", {"offsets": self.ctx.offsets(), "state": self.ctx.state_dict()})

    def run(self, generations: int | None = None, progress: bool = False) -> list[dict[str, Any]]:
        total = generations if generations is not None else self.cfg.runner.generations
        budget = self.cfg.runner.wall_clock_budget_s
        t0 = time.time()
        recs = []
        g = self.ctx.generation + 1
        while g < total:
            rec = run_generation(self.ctx, g)
            if g == 0:
                self._record_initial(rec)
            self.checkpoint()
            recs.append(rec)
            if progress:
                print(f"[{self.cfg.name}/{self.cfg.condition}] gen {g}: best {rec['population_best']:.2f} "
                      f"mean {rec['population_mean']:.2f} tokens {rec['cost']['tokens']} "
                      f"({rec['volatile']['wall_seconds']:.1f}s)", flush=True)
            g += 1
            if budget is not None and time.time() - t0 > budget:
                break
        return recs

    def _record_initial(self, rec: dict[str, Any]) -> None:
        """Initial (generation-0) evaluation and warm-start hashes, written once and never back-filled."""
        p = self.out / "initial_evaluation.json"
        if p.exists():
            return
        _atomic_json(p, {"warm_start": {a: v["incumbent"] for a, v in rec["agents"].items()},
                         "scores": {a: v["score"] for a, v in rec["agents"].items()},
                         "anchor_scores": {a: v["anchor_score"] for a, v in rec["agents"].items()}})
        man = json.loads((self.out / "manifest.json").read_text())
        man["warm_start_hashes"] = {a: v["incumbent"] for a, v in rec["agents"].items()}
        _atomic_json(self.out / "manifest.json", man)

    def close(self) -> None:
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            ctx.close()
        if getattr(self, "_lock_fd", None) is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


VOLATILE_KEYS = {"volatile", "wall_seconds", "replayed_within_run", "cached", "new_spend_usd", "real_spend_usd"}


def _strip(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE_KEYS}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def state_digest(out: str | Path) -> dict[str, str]:
    """Hash of everything a run produced, minus wall-clock and cache-provenance fields. Equal digests mean an
    interrupted-and-resumed run ended in the same state as an uninterrupted one."""
    out = Path(out)
    parts = {}
    with open(out / "checkpoint.json") as f:
        ck = json.load(f)
    parts["state"] = json.dumps(_strip(ck["state"]), sort_keys=True)
    for n in LOGS:
        p = out / f"{n}.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
        parts[n] = json.dumps(_strip(rows), sort_keys=True)
    arts = sorted(x.name for x in (out / "artifacts").glob("*/*.json")) if (out / "artifacts").exists() else []
    parts["artifact_files"] = json.dumps(arts)
    return {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in parts.items()}


def run_config(cfg: ExperimentConfig | dict, out: str | Path, generations: int | None = None,
               progress: bool = False) -> list[dict[str, Any]]:
    if isinstance(cfg, dict):
        cfg = from_dict(cfg)
    with Runner(cfg, out) as r:
        return r.run(generations, progress)
