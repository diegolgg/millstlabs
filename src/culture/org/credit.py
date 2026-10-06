"""Credit for students' improvement (spec section 11). v0 `paired_delta`; the rest arrive in phase 4.

The generation loop hands each policy a list of `Window`s: one per (student, generation) with the student's score
against the anchor set before and after the window on identical seeds, and every message the student could have
received (delivered or not, adopted or not)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Policy


@dataclass
class Offer:
    message_id: str
    sender: str
    delivered: bool
    adopted: bool
    source_artifact: str = ""


@dataclass
class Window:
    student: str
    generation: int
    before: float  # f(s_before; A, seeds_g)
    after: float  # f(s_after; A, seeds_g)
    offers: list[Offer] = field(default_factory=list)
    self_revised: bool = False

    @property
    def delta(self) -> float:
        return self.after - self.before


class Credit(Policy):
    kind = "credit"

    def assign(self, windows: list[Window], provenance=None) -> tuple[dict[str, float], list[dict]]:
        """Return (credit increment per teacher, flags/notes per window)."""
        raise NotImplementedError


class NoCredit(Credit):
    def assign(self, windows, provenance=None):
        return {}, []


class PairedDelta(Credit):
    """credit[T] += delta of the student over the window, split equally across the adopted messages in it.
    Windows with several adoptions or a self-revision are flagged (additivity assumption, self-revision bias)."""

    def assign(self, windows, provenance=None):
        out: dict[str, float] = {}
        notes = []
        for w in windows:
            adopted = [o for o in w.offers if o.delivered and o.adopted]
            if not adopted:
                continue
            share = w.delta / len(adopted)
            for o in adopted:
                out[o.sender] = out.get(o.sender, 0.0) + share
            notes.append({"student": w.student, "generation": w.generation, "delta": w.delta,
                          "teachers": [o.sender for o in adopted], "split": len(adopted) > 1,
                          "self_revised": w.self_revised})
        return out, notes


REGISTRY = {"none": NoCredit, "paired_delta": PairedDelta}
