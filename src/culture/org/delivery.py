"""Delivery: which routed messages actually arrive. Bernoulli delivery is what makes datamodel credit identifiable."""

from __future__ import annotations

import random

from .base import Policy


class Delivery(Policy):
    kind = "delivery"

    def deliver(self, n: int, rng: random.Random) -> list[bool]:
        raise NotImplementedError


class Deterministic(Delivery):
    def deliver(self, n, rng):
        return [True] * n


class Bernoulli(Delivery):
    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError("bernoulli p must be in [0, 1]")
        self.p = p

    def deliver(self, n, rng):
        return [rng.random() < self.p for _ in range(n)]


REGISTRY = {"deterministic": Deterministic, "bernoulli": Bernoulli}
