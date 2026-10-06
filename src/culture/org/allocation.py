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


REGISTRY = {"uniform": Uniform, "proportional": Proportional, "softmax_floor": SoftmaxFloor}


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
