"""Content-addressed artifact store, harness evidence registry, per-group corpus with a touch log.

Corpus verbs follow the shared grammar agreed with Diego's sandbox: `deposit(item, evidence)` with mechanical,
ownership-bound validation; `retrieve(policy)` that writes a touch log. Prose is stored, never certified.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Artifact, Evaluation


class ArtifactStore:
    """Write-once JSON files keyed by artifact id (identical content -> identical file, so re-runs are idempotent)."""

    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        self._mem: dict[str, Artifact] = {}
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, aid: str) -> Path:
        return self.root / aid[:2] / f"{aid}.json"

    def put(self, a: Artifact) -> None:
        self._mem[a.id] = a
        if len(self._mem) > 4096:
            self._mem.pop(next(iter(self._mem)))
        if self.root:
            p = self._path(a.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(a.to_dict(), f, sort_keys=True)
            os.replace(tmp, p)

    def get(self, aid: str) -> Artifact:
        if aid in self._mem:
            return self._mem[aid]
        if self.root and self._path(aid).exists():
            with open(self._path(aid)) as f:
                a = Artifact.from_dict(json.load(f))
            self._mem[aid] = a
            return a
        raise KeyError(aid)

    def __contains__(self, aid: str) -> bool:
        return aid in self._mem or bool(self.root and self._path(aid).exists())


class EvidenceRegistry:
    """Digests of every Evaluation the harness produced. Only registered evidence can be deposited or attached.
    Entries older than `horizon` generations are pruned (bounded state)."""

    def __init__(self, horizon: int = 3):
        self.horizon = horizon
        self.issued: dict[str, int] = {}

    def issue(self, ev: Evaluation) -> str:
        d = ev.digest()
        self.issued[d] = ev.generation
        return d

    def check(self, ev: Evaluation | None) -> bool:
        return ev is not None and self.issued.get(ev.digest()) is not None

    def prune(self, generation: int) -> None:
        self.issued = {d: g for d, g in self.issued.items() if g > generation - self.horizon}

    def state_dict(self) -> dict:
        return {"issued": self.issued}

    def load_state_dict(self, s: dict) -> None:
        self.issued = {k: int(v) for k, v in s["issued"].items()}


class EvidenceRejected(ValueError):
    pass


@dataclass
class CorpusEntry:
    artifact_id: str
    depositor: str
    generation: int
    score: float
    evidence_digest: str
    summary: str


@dataclass
class Corpus:
    group: str
    capacity: int = 20
    entries: list[CorpusEntry] = field(default_factory=list)

    def deposit(self, artifact: Artifact, evidence: Evaluation, depositor: str, registry: EvidenceRegistry,
                holders: dict[str, str]) -> CorpusEntry:
        """Validate and store. Evidence must be a harness-issued evaluation of this very artifact, and the depositor
        must currently hold the artifact (ownership-bound)."""
        if not registry.check(evidence):
            raise EvidenceRejected("evidence was not produced by the harness")
        if evidence.artifact_id != artifact.id:
            raise EvidenceRejected("evidence is for a different artifact")
        if holders.get(depositor) != artifact.id:
            raise EvidenceRejected(f"{depositor} does not hold {artifact.id}")
        self.entries = [e for e in self.entries if e.artifact_id != artifact.id]
        entry = CorpusEntry(artifact.id, depositor, evidence.generation, evidence.selfplay_mean, evidence.digest(),
                            evidence.summary())
        self.entries.append(entry)
        if len(self.entries) > self.capacity:  # keep the best, ties broken by recency
            self.entries.sort(key=lambda e: (e.score, e.generation), reverse=True)
            self.entries = self.entries[: self.capacity]
        return entry

    def retrieve(self, receiver: str, generation: int, k: int, touch: "TouchLog", exclude: set[str] = frozenset(),
                 policy: str = "top_k") -> list[CorpusEntry]:
        pool = [e for e in self.entries if e.artifact_id not in exclude]
        if policy == "recent":
            pool.sort(key=lambda e: (e.generation, e.score), reverse=True)
        else:
            pool.sort(key=lambda e: (e.score, e.generation), reverse=True)
        out = pool[:k]
        for e in out:
            touch.add(receiver, e.artifact_id, "corpus", generation, used=True)
        return out

    def state_dict(self) -> dict:
        return {"group": self.group, "capacity": self.capacity, "entries": [e.__dict__ for e in self.entries]}

    @staticmethod
    def from_state(s: dict) -> "Corpus":
        return Corpus(s["group"], s["capacity"], [CorpusEntry(**e) for e in s["entries"]])


class TouchLog:
    """(receiver, item id, kind, generation, used) rows; the data credit assignment needs."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.pending: list[dict[str, Any]] = []

    def add(self, receiver: str, item_id: str, kind: str, generation: int, used: bool) -> None:
        self.pending.append({"receiver": receiver, "item": item_id, "kind": kind, "generation": generation,
                             "used": bool(used)})

    def flush(self) -> list[dict[str, Any]]:
        rows, self.pending = self.pending, []
        if self.path and rows:
            with open(self.path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
        return rows
