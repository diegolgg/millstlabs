"""Routing: who sends their artifact to whom this generation. Respects the topology."""

from __future__ import annotations

import random

from .base import AgentState, Policy
from .population import Population, Topology


class Routing(Policy):
    kind = "routing"

    def route(self, agents: dict[str, AgentState], pop: Population, topo: Topology,
              rng: random.Random) -> list[tuple[str, str]]:
        raise NotImplementedError


class NoRouting(Routing):
    def route(self, agents, pop, topo, rng):
        return []


class BroadcastGroup(Routing):
    """Every agent sends to every reachable agent of its own group."""

    def route(self, agents, pop, topo, rng):
        out = []
        for s in pop.agents:
            for r in pop.members[pop.group_of(s)]:
                if topo.reachable(s, r, pop):
                    out.append((s, r))
        return out


class BestToAll(Routing):
    """The best agent of each group (by incumbent score) sends to every other reachable agent."""

    def __init__(self, scope: str = "group"):
        self.scope = scope

    def route(self, agents, pop, topo, rng):
        out = []
        pools = [pop.members[g] for g in pop.groups] if self.scope == "group" else [pop.agents]
        for members in pools:
            if not members:
                continue
            best = max(members, key=lambda a: (agents[a].score, a))
            out += [(best, r) for r in pop.agents if topo.reachable(best, r, pop) and agents[r].score < agents[best].score]
        return out


class RandomK(Routing):
    """Each agent sends to k agents drawn uniformly from those it can reach."""

    def __init__(self, k: int = 1):
        self.k = k

    def route(self, agents, pop, topo, rng):
        out = []
        for s in pop.agents:
            reach = [r for r in pop.agents if topo.reachable(s, r, pop)]
            for r in rng.sample(reach, min(self.k, len(reach))):
                out.append((s, r))
        return out


REGISTRY = {"none": NoRouting, "broadcast_group": BroadcastGroup, "best_to_all": BestToAll, "random_k": RandomK}
