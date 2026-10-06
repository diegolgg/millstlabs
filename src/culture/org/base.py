"""Shared types for organization policies. Game-agnostic: policies see artifact ids, scores and a `Judge` callback,
never the game (nothing in org/ imports game/)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Judge(Protocol):
    """Engine-time measurements the run layer provides to policies (all on explicit seed lists, so paired)."""

    def selfplay(self, aid: str, seeds: list[int]) -> tuple[float, float]: ...  # (mean score, illegal rate)

    def crossplay(self, a: str, b: str, seeds: list[int]) -> float: ...

    def anchors(self, aid: str, seeds: list[int]) -> dict[str, float]: ...


@dataclass
class AgentState:
    id: str
    group: str
    incumbent: str | None = None
    score: float = 0.0  # incumbent self-play mean on the latest generation's shared seeds
    anchor_score: float = 0.0  # incumbent mean with the anchor set
    credit: float = 0.0
    budget_left: float = 0.0
    previous: str | None = None  # incumbent before the latest adoption (critical social learning reverts here)
    pending_check: dict[str, Any] | None = None  # critical social learning: adopted, re-verify after a revision
    counters: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n


class Policy:
    """Base: a policy is a small object built from config params, optionally with state that survives resume."""

    kind = ""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass

    def describe(self) -> str:
        return type(self).__name__
