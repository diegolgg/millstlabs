"""Artifact, Evaluation, Verification, TeachingMessage (spec section 5). Plain dataclasses with JSON round trips."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def artifact_id(code: str, conventions: str) -> str:
    return hashlib.sha256((code + "\0" + conventions).encode()).hexdigest()[:16]


def _from(cls, d: dict | None):
    if d is None:
        return None
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Evaluation:
    artifact_id: str
    generation: int
    seed_base: int
    n_games: int
    selfplay_scores: list[int]
    selfplay_iqm: float
    selfplay_mean: float
    selfplay_ci: tuple[float, float]
    crossplay: dict[str, float]  # artifact id -> mean score on shared seeds
    anchors: dict[str, float]  # anchor name -> mean score paired with this bot
    illegal_rate: float
    lines: int
    complexity: int
    wall_seconds: float
    error: str | None = None

    def digest(self) -> str:
        """Content hash used as the evidence certificate (wall time excluded so it is reproducible)."""
        d = asdict(self)
        d.pop("wall_seconds")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "Evaluation | None":
        if d is not None and "selfplay_scores" not in d:
            d = dict(d, selfplay_scores=[])
        e = _from(Evaluation, d)
        if e is not None:
            e.selfplay_ci = tuple(e.selfplay_ci)
        return e

    def summary(self) -> str:
        # sorted: dict order must not depend on whether the evaluation went through a JSON checkpoint
        anchors = ", ".join(f"{k} {self.anchors[k]:.2f}" for k in sorted(self.anchors))
        cross = ", ".join(f"{k[:8]} {self.crossplay[k]:.2f}" for k in sorted(self.crossplay)[:6])
        return (f"self-play mean {self.selfplay_mean:.2f}, IQM {self.selfplay_iqm:.2f} "
                f"(95% CI {self.selfplay_ci[0]:.2f}-{self.selfplay_ci[1]:.2f}) over {self.n_games} games; "
                f"with anchors: {anchors or 'n/a'}; cross-play: {cross or 'n/a'}; illegal-move rate "
                f"{self.illegal_rate:.3f}; {self.lines} lines")


@dataclass
class Verification:
    """The receiver's engine-time check of a delivered artifact (the verification log)."""

    message_id: str
    receiver: str
    generation: int
    policy: str
    n_games: int
    payload_score: float  # payload self-play mean on the receiver's verification seeds
    incumbent_score: float  # receiver's incumbent on the same seeds
    crossplay_with_incumbent: float | None
    anchors: dict[str, float]
    illegal_rate: float
    passed: bool
    decision: str = ""  # adopted | rejected | merged | reverted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "Verification | None":
        return _from(Verification, d)


@dataclass
class Artifact:
    id: str
    code: str
    conventions: str
    author: str
    group: str
    generation: int
    parents: list[str] = field(default_factory=list)
    teaching_sources: list[str] = field(default_factory=list)
    evaluation: Evaluation | None = None
    verification: Verification | None = None
    origin: str = "revise"  # seed | author | revise | merge | oracle
    edit_mode: str = "full"

    @staticmethod
    def make(code: str, conventions: str, **kw) -> "Artifact":
        return Artifact(id=artifact_id(code, conventions), code=code, conventions=conventions, **kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evaluation"] = self.evaluation.to_dict() if self.evaluation else None
        d["verification"] = self.verification.to_dict() if self.verification else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "TeachingMessage":
        m = _from(TeachingMessage, d)
        m.evidence = Evaluation.from_dict(d.get("evidence")) if d.get("evidence") else None
        if m.evidence is not None and "selfplay_scores" not in d["evidence"]:
            m.evidence.selfplay_scores = []
        m.verification = Verification.from_dict(d.get("verification"))
        return m

    @staticmethod
    def from_dict(d: dict) -> "Artifact":
        a = _from(Artifact, d)
        a.evaluation = Evaluation.from_dict(d.get("evaluation"))
        a.verification = Verification.from_dict(d.get("verification"))
        return a

    def meta(self) -> dict[str, Any]:
        """Everything but the code (for logs)."""
        d = self.to_dict()
        d.pop("code")
        d.pop("conventions")
        if d["evaluation"]:
            d["evaluation"].pop("selfplay_scores", None)
        return d


@dataclass
class TeachingMessage:
    id: str
    sender: str
    receiver: str
    generation: int
    artifact_id: str  # the payload
    delta_text: str  # what changed and why, written by the sender
    evidence: Evaluation | None  # the sender's own evaluation of the payload, attached by the harness
    delivered: bool = True  # set by the delivery policy
    touched_at: int | None = None  # generation at which the receiver actually read it
    decision: str | None = None
    verification: Verification | None = None
    sender_group: str = ""
    receiver_group: str = ""
    sabotaged: bool = False  # payload replaced by the sabotage treatment (D1); set by the harness, never by agents

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.to_dict() if self.evidence else None
        if d["evidence"]:
            d["evidence"].pop("selfplay_scores", None)
        d["verification"] = self.verification.to_dict() if self.verification else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "TeachingMessage":
        m = _from(TeachingMessage, d)
        m.evidence = Evaluation.from_dict(d.get("evidence")) if d.get("evidence") else None
        if m.evidence is not None and "selfplay_scores" not in d["evidence"]:
            m.evidence.selfplay_scores = []
        m.verification = Verification.from_dict(d.get("verification"))
        return m
