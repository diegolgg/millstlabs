"""Allocation of the per-generation budget across groups, plus the teaching-cost rule."""

from __future__ import annotations

import math

from .base import Policy


class Allocation(Policy):
    kind = "allocation"

    def allocate(self, groups: list[str], stats: dict[str, dict], total: float) -> dict[str, float]:
        """stats[group] has 'credit', 'score' (group best), 'mean', 'size'. Returns budget per group summing to total."""
        raise NotImplementedError


class Uniform(Allocation):
    def allocate(self, groups, stats, total):
        return {g: total / len(groups) for g in groups}


class Proportional(Allocation):
    """Proportional to a non-negative group statistic (default: cumulative credit; uniform when all zero)."""

    def __init__(self, by: str = "credit"):
        self.by = by

    def allocate(self, groups, stats, total):
        w = {g: max(0.0, float(stats[g].get(self.by, 0.0))) for g in groups}
        s = sum(w.values())
        if s <= 0:
            return Uniform().allocate(groups, stats, total)
        return {g: total * w[g] / s for g in groups}


class SoftmaxFloor(Allocation):
    """softmax(stat / T) mixed with a uniform floor: share_g = eps/G + (1 - eps) softmax_g. The eps/G floor keeps
    every group sampleable (same role as the uniform floor in Online TASS)."""

    def __init__(self, T: float = 1.0, eps: float = 0.1, by: str = "score"):
        if T <= 0 or not 0 <= eps <= 1:
            raise ValueError("softmax_floor needs T > 0 and eps in [0, 1]")
        self.T, self.eps, self.by = T, eps, by

    def allocate(self, groups, stats, total):
        x = [float(stats[g].get(self.by, 0.0)) / self.T for g in groups]
        m = max(x)
        e = [math.exp(v - m) for v in x]
        z = sum(e)
        G = len(groups)
        return {g: total * (self.eps / G + (1 - self.eps) * ei / z) for g, ei in zip(groups, e)}


def maxent_nash(A, iters: int = 400):
    """Maximum-entropy Nash equilibrium of the symmetric zero-sum game with antisymmetric payoff A (Balduzzi et al.
    2018, Nash averaging). Value is 0; we maximize entropy subject to (p^T A)_j >= 0 for every column j."""
    import numpy as np
    from scipy.optimize import linprog, minimize

    A = np.asarray(A, float)
    n = A.shape[0]
    # a feasible Nash point first (LP), then the max-entropy point inside the equilibrium polytope
    res = linprog(np.zeros(n), A_ub=-A.T, b_ub=np.zeros(n), A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, 1)] * n)
    p0 = res.x if res.success else np.full(n, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda p: p.sum() - 1.0}, {"type": "ineq", "fun": lambda p: A.T @ p + 1e-9}]
    obj = lambda p: float(np.sum(p * np.log(np.clip(p, 1e-12, None))))  # noqa: E731 (negative entropy)
    out = minimize(obj, np.clip(p0, 1e-6, None) / np.clip(p0, 1e-6, None).sum(), method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=cons, options={"maxiter": iters, "ftol": 1e-12})
    p = out.x if out.success else p0
    p = np.clip(p, 0, None)
    return p / p.sum()


class NashRelative(Allocation):
    """Nash averaging over groups. The meta-game payoff A[i, j] is the mean over this generation's shared deals of
    sign(score_i - score_j) for the groups' best artifacts (antisymmetric). Budget share = eps/G + (1 - eps) * p_i
    with p the maximum-entropy Nash mixture. A cloned group duplicates a row/column of A; max-entropy splits the
    original's mass between the copies, so cloning does not buy budget (the property queue item 8 tests)."""

    def __init__(self, eps: float = 0.1):
        if not 0 <= eps <= 1:
            raise ValueError("eps must be in [0, 1]")
        self.eps = eps

    def allocate(self, groups, stats, total):
        import numpy as np

        scores = [np.asarray(stats[g].get("scores", [stats[g].get("score", 0.0)]), float) for g in groups]
        m = min(len(x) for x in scores)
        G = len(groups)
        A = np.zeros((G, G))
        for i in range(G):
            for j in range(G):
                A[i, j] = float(np.mean(np.sign(scores[i][:m] - scores[j][:m]))) if m else 0.0
        p = maxent_nash(A)
        return {g: total * (self.eps / G + (1 - self.eps) * float(pi)) for g, pi in zip(groups, p)}


REGISTRY = {"uniform": Uniform, "proportional": Proportional, "softmax_floor": SoftmaxFloor,
            "nash_relative": NashRelative}


class TeachingCost(Policy):
    kind = "teaching_cost"

    def cost(self) -> float:
        return 0.0

    def credit_share(self) -> float:
        return 0.0


class Free(TeachingCost):
    pass


class Costly(TeachingCost):
    """Teaching debits c budget units from the sender; with credit_share alpha the teacher receives alpha of its
    credited student improvement back as budget (queue item 7)."""

    def __init__(self, c: float = 1.0, credit_share: float = 0.0):
        self.c, self.alpha = c, credit_share

    def cost(self):
        return self.c

    def credit_share(self):
        return self.alpha


TEACHING_COST_REGISTRY = {"free": Free, "costly": Costly}
