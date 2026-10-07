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

    def record(self, group: str, aid: str, score: float, parent: str | None, generation: int,
               parent_score: float | None = None) -> None:
        """`parent_score` is the parent's score on the same deals as `score` (paired); rules that label a child as
        better or worse than its parent use it, never a parent score from another generation's deals."""
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


class ShinkaWeighted(Selection):
    """ShinkaEvolve's weighted parent rule over the group archive: s_i = sigmoid(lambda (score_i - median) / MAD),
    h_i = 1 / (1 + children_i), p_i proportional to s_i h_i (lambda = 10). Every evaluated child enters the archive
    and becomes the agent's incumbent (no hill-climbing gate); the archive is capped at `cap` by score."""

    def __init__(self, lam: float = 10.0, cap: int = 50):
        super().__init__()
        self.lam, self.cap = lam, cap

    def prune(self, group):
        arch = self.archive[group]
        if len(arch) > self.cap:
            arch.sort(key=lambda e: (e["score"], e["generation"]), reverse=True)
            del arch[self.cap:]

    def weights(self, group: str) -> list[tuple[str, float]]:
        import math
        import statistics

        arch = self.archive.get(group, [])
        if not arch:
            return []
        sc = [e["score"] for e in arch]
        med = statistics.median(sc)
        mad = statistics.median([abs(x - med) for x in sc]) or 1.0
        out = []
        for e in arch:
            z = max(-50.0, min(50.0, self.lam * (e["score"] - med) / mad))
            out.append((e["id"], (1 / (1 + math.exp(-z))) / (1 + e["children"])))
        return out

    def choose_parent(self, agent, rng):
        w = self.weights(agent.group)
        if not w:
            return agent.incumbent
        tot = sum(x for _, x in w)
        r, acc = rng.random() * tot, 0.0
        for aid, x in w:
            acc += x
            if r <= acc:
                return aid
        return w[-1][0]

    def accept(self, candidate_score, incumbent_score):
        return True


class HGMCladeTS(Selection):
    """Huxley-Godel Machine parent rule. Each group keeps a tree of artifacts; a node succeeds if it scored above its
    parent on the same deals (paired: the parent is re-scored on the generation's seeds the child was evaluated on).
    Label rule: success if child - parent > margin, failure if < -margin, and no label (neither) on a tie within the
    margin, so a behaviourally identical revision never counts either way and the label cannot flip with seed noise.
    Clade metaproductivity uses success/failure counts over the node's whole subtree including the node itself,
    CMP = nC_s / (nC_s + nC_f); the parent is chosen by Thompson sampling Beta(tau(1 + nC_s), tau(1 + nC_f)).
    UCB-Air widening: an agent expands (makes an LLM revision) only if n^alpha >= |T|, where n counts the group's
    agent-steps and |T| its tree size; otherwise it spends the step re-evaluating (free in this harness), i.e. it skips
    the LLM call. Revisions always join the tree; the agent's incumbent becomes the child (judged by descendants)."""

    def __init__(self, alpha: float = 0.6, tau: float = 1.0, margin: float = 0.0):
        super().__init__()
        self.alpha, self.tau, self.margin = alpha, tau, margin
        self.tree: dict[str, dict[str, dict]] = {}  # group -> id -> {parent, score, ok}
        self.steps: dict[str, int] = {}

    def label(self, score: float, parent_score: float | None) -> bool | None:
        if parent_score is None:
            return None  # no paired parent score: unlabelled rather than compared across different deals
        d = score - parent_score
        return True if d > self.margin else False if d < -self.margin else None

    def record(self, group, aid, score, parent, generation, parent_score=None):
        t = self.tree.setdefault(group, {})
        if aid in t:
            t[aid]["score"] = score
            return
        linked = parent in t
        t[aid] = {"parent": parent if linked else None, "score": score,
                  "ok": self.label(score, parent_score) if linked else None}

    def clade_counts(self, group: str, aid: str) -> tuple[int, int]:
        t = self.tree.get(group, {})
        kids: dict[str, list[str]] = {}
        for k, v in t.items():
            if v["parent"]:
                kids.setdefault(v["parent"], []).append(k)
        s = f = 0
        stack = [aid] if aid in t else []  # the node's own outcome counts towards its clade
        while stack:
            k = stack.pop()
            if t[k]["ok"]:
                s += 1
            elif t[k]["ok"] is False:
                f += 1
            stack += kids.get(k, [])
        return s, f

    def wants_revision(self, agent, rng):
        g = agent.group
        self.steps[g] = self.steps.get(g, 0) + 1
        return self.steps[g] ** self.alpha >= len(self.tree.get(g, {}))

    def choose_parent(self, agent, rng):
        t = self.tree.get(agent.group, {})
        if not t:
            return agent.incumbent
        best, best_draw = None, -1.0
        for aid in sorted(t):
            s, f = self.clade_counts(agent.group, aid)
            draw = rng.betavariate(self.tau * (1 + s), self.tau * (1 + f))
            if draw > best_draw:
                best, best_draw = aid, draw
        return best

    def accept(self, candidate_score, incumbent_score):
        return True

    def state_dict(self):
        return {"archive": self.archive, "tree": self.tree, "steps": self.steps}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.tree = {g: {k: dict(v) for k, v in t.items()} for g, t in state.get("tree", {}).items()}
        self.steps = dict(state.get("steps", {}))


REGISTRY = {"keep_best_k": KeepBestK, "shinka_weighted": ShinkaWeighted, "hgm_clade_ts": HGMCladeTS}
