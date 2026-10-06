"""Migration of agents between groups."""

from __future__ import annotations

import random

from .base import AgentState, Policy
from .population import Population


class Migration(Policy):
    kind = "migration"

    def migrate(self, pop: Population, agents: dict[str, AgentState], generation: int,
                rng: random.Random) -> list[tuple[str, str, str]]:
        """Apply moves to `pop` and return (agent, from group, to group)."""
        raise NotImplementedError


class NoMigration(Migration):
    def migrate(self, pop, agents, generation, rng):
        return []


class RandomMigration(Migration):
    """Every `interval` generations, round(rate * N) randomly chosen agents each swap places with a random agent of a
    different group. Swaps keep every group's size (and the agent count) fixed."""

    def __init__(self, rate: float = 0.1, interval: int = 5):
        if not 0 <= rate <= 1 or interval < 1:
            raise ValueError("random migration needs rate in [0, 1] and interval >= 1")
        self.rate, self.interval = rate, interval

    def migrate(self, pop, agents, generation, rng):
        if generation % self.interval != 0 or len(pop.groups) < 2 or self.rate == 0:
            return []
        n = len(pop.agents)
        k = max(1, round(self.rate * n))
        moves = []
        for a in rng.sample(sorted(pop.agents), min(k, n)):
            ga = pop.group_of(a)
            others = [b for b in sorted(pop.agents) if pop.group_of(b) != ga]
            if not others:
                continue
            b = rng.choice(others)
            gb = pop.group_of(b)
            pop.move(a, gb)
            pop.move(b, ga)
            moves += [(a, ga, gb), (b, gb, ga)]
        return moves


class BestToNeighbor(Migration):
    """Every `interval` generations each group's best agent swaps with the worst agent of the next group on a ring."""

    def __init__(self, interval: int = 5):
        self.interval = interval

    def migrate(self, pop, agents, generation, rng):
        if generation % self.interval != 0 or len(pop.groups) < 2:
            return []
        gs = pop.groups
        bests = {g: max(pop.members[g], key=lambda a: (agents[a].score, a)) for g in gs if pop.members[g]}
        moves = []
        for i, g in enumerate(gs):
            nxt = gs[(i + 1) % len(gs)]
            a = bests.get(g)
            if a is None or not pop.members[nxt]:
                continue
            b = min((x for x in pop.members[nxt] if x not in bests.values()), key=lambda x: (agents[x].score, x),
                    default=None)
            if b is None:
                continue
            pop.move(a, nxt)
            pop.move(b, g)
            moves += [(a, g, nxt), (b, nxt, g)]
        return moves


REGISTRY = {"none": NoMigration, "random": RandomMigration, "best_to_neighbor": BestToNeighbor}
