"""Selection: which artifact an agent builds on (parent choice) and whether a revision replaces the incumbent.

Each group keeps an archive of (artifact id, score, parent, generation). Formulas for the phase-4 rules are in spec
section 15."""

from __future__ import annotations

import random
from typing import Any

from .base import AgentState, Policy


class Selection(Policy):
    kind = "selection"

    def __init__(self):
        self.archive: dict[str, list[dict[str, Any]]] = {}  # group -> entries

    def record(self, group: str, aid: str, score: float, parent: str | None, generation: int) -> None:
        arch = self.archive.setdefault(group, [])
        for e in arch:
            if e["id"] == aid:
                e["score"] = score
                return
        arch.append({"id": aid, "score": score, "parent": parent, "generation": generation, "children": 0})
        for e in arch:
            if e["id"] == parent:
                e["children"] += 1
        self.prune(group)

    def prune(self, group: str) -> None:
        pass

    def choose_parent(self, agent: AgentState, rng: random.Random) -> str | None:
        return agent.incumbent

    def accept(self, candidate_score: float, incumbent_score: float) -> bool:
        return candidate_score >= incumbent_score

    def wants_revision(self, agent: AgentState, rng: random.Random) -> bool:
        return True

    def state_dict(self):
        return {"archive": self.archive}

    def load_state_dict(self, state):
        self.archive = {g: [dict(e) for e in es] for g, es in state.get("archive", {}).items()}


class KeepBestK(Selection):
    """Hill climbing: revise your own incumbent; keep the revision only if it scores at least as well on the same
    seeds. The group archive keeps the best k artifacts (bounded state)."""

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def prune(self, group):
        arch = self.archive[group]
        if len(arch) > self.k:
            arch.sort(key=lambda e: (e["score"], e["generation"]), reverse=True)
            del arch[self.k:]


REGISTRY = {"keep_best_k": KeepBestK}
