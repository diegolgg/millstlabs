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


class LineageDecay(Credit):
    """paired_delta for the direct teachers, plus gamma^d of the same share for teachers d hops up the lineage of the
    adopted payload (its own teachers, their teachers, ...). Pools evidence across a lineage; over-credits by design."""

    def __init__(self, gamma: float = 0.5, max_depth: int = 5):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma, self.max_depth = gamma, max_depth

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
                if provenance is not None and o.source_artifact:
                    for t, dist in provenance.teachers_of(o.source_artifact, self.max_depth).items():
                        if t in (o.sender, w.student):
                            continue
                        out[t] = out.get(t, 0.0) + share * self.gamma ** (dist + 1)
            notes.append({"student": w.student, "generation": w.generation, "delta": w.delta,
                          "teachers": [o.sender for o in adopted]})
        return out, notes


class DatamodelRegression(Credit):
    """Datamodel credit over randomized delivery (spec section 11, v1).

    Each window (student, generation) contributes one row: y = the student's paired delta, x_T = number of messages
    from teacher T *delivered* to the student in that window. Delivery is Bernoulli and logged, so x is randomized
    and the ridge coefficient beta_T estimates the causal effect of one delivered message from T (intent to treat:
    it includes the receiver's adoption decision). Rows are pooled across students and generations over the last
    `window` generations; credit[T] = beta_T * (messages from T delivered in the window). `assign` returns the change
    in that estimate since the last call so that cumulative agent credit tracks the current estimate."""

    def __init__(self, lam: float = 1.0, window: int = 50, intercept: bool = True):
        self.lam, self.window, self.intercept = lam, window, intercept
        self.rows: list[dict] = []  # {"g", "y", "x": {teacher: count}}
        self.last: dict[str, float] = {}
        self.beta: dict[str, float] = {}

    def fit(self) -> tuple[dict[str, float], dict[str, int]]:
        import numpy as np

        teachers = sorted({t for r in self.rows for t in r["x"]})
        if not teachers or len(self.rows) < 2:
            return {}, {}
        X = np.array([[r["x"].get(t, 0) for t in teachers] for r in self.rows], float)
        y = np.array([r["y"] for r in self.rows], float)
        if self.intercept:
            xm, ym = X.mean(axis=0), y.mean()
            Xc, yc = X - xm, y - ym
        else:
            Xc, yc = X, y
        beta = np.linalg.solve(Xc.T @ Xc + self.lam * np.eye(len(teachers)), Xc.T @ yc)
        counts = X.sum(axis=0)
        return dict(zip(teachers, beta.tolist())), dict(zip(teachers, counts.astype(int).tolist()))

    def assign(self, windows, provenance=None):
        for w in windows:
            x: dict[str, int] = {}
            for o in w.offers:
                if o.delivered:
                    x[o.sender] = x.get(o.sender, 0) + 1
            for o in w.offers:  # teachers who could have been delivered appear as explicit zeros
                x.setdefault(o.sender, 0)
            self.rows.append({"g": w.generation, "y": w.delta, "x": x})
        if windows:
            g = max(w.generation for w in windows)
            self.rows = [r for r in self.rows if r["g"] > g - self.window]
        self.beta, counts = self.fit()
        est = {t: self.beta[t] * counts[t] for t in self.beta}
        inc = {t: est[t] - self.last.get(t, 0.0) for t in est}
        for t in self.last:
            if t not in est:
                inc[t] = -self.last[t]
        self.last = est
        return inc, [{"beta": {t: round(b, 4) for t, b in self.beta.items()}, "rows": len(self.rows)}]

    def state_dict(self):
        return {"rows": self.rows, "last": self.last, "beta": self.beta}

    def load_state_dict(self, state):
        self.rows, self.last, self.beta = state.get("rows", []), state.get("last", {}), state.get("beta", {})


REGISTRY = {"none": NoCredit, "paired_delta": PairedDelta, "lineage_decay": LineageDecay,
            "datamodel_regression": DatamodelRegression}
