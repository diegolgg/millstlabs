"""Groups, membership and topology (who can reach whom)."""

from __future__ import annotations

from typing import Any

from .base import Policy


class Population:
    def __init__(self, groups: int, agents_per_group: int):
        self.members: dict[str, list[str]] = {f"g{g}": [f"g{g}a{a}" for a in range(agents_per_group)]
                                              for g in range(groups)}

    @property
    def groups(self) -> list[str]:
        return list(self.members)

    @property
    def agents(self) -> list[str]:
        return [a for g in self.members for a in self.members[g]]

    def group_of(self, agent: str) -> str:
        for g, ms in self.members.items():
            if agent in ms:
                return g
        raise KeyError(agent)

    def move(self, agent: str, to_group: str) -> None:
        self.members[self.group_of(agent)].remove(agent)
        self.members[to_group].append(agent)

    def state_dict(self) -> dict[str, Any]:
        return {"members": {g: list(m) for g, m in self.members.items()}}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.members = {g: list(m) for g, m in s["members"].items()}


class Topology(Policy):
    kind = "topology"

    def reachable(self, sender: str, receiver: str, pop: Population) -> bool:
        raise NotImplementedError


class Isolated(Topology):
    """Groups do not talk to each other; agents inside a group can."""

    def reachable(self, sender, receiver, pop):
        return sender != receiver and pop.group_of(sender) == pop.group_of(receiver)


class Full(Topology):
    def reachable(self, sender, receiver, pop):
        return sender != receiver


class Ring(Topology):
    """Groups on a ring: an agent reaches its own group and the two neighbouring groups."""

    def reachable(self, sender, receiver, pop):
        if sender == receiver:
            return False
        gs = pop.groups
        i, j = gs.index(pop.group_of(sender)), gs.index(pop.group_of(receiver))
        return min((i - j) % len(gs), (j - i) % len(gs)) <= 1


class Islands(Topology):
    """Derex & Boyd partial connectivity: isolated groups plus periodic migration of agents between them.
    Reachability is within the island; the migration itself is done by `migration: random` with the same
    rate and interval (the runner wires `migration_rate`/`interval` from here when migration is left at `none`)."""

    def __init__(self, migration_rate: float = 0.1, interval: int = 5):
        self.migration_rate, self.interval = migration_rate, interval

    def reachable(self, sender, receiver, pop):
        return sender != receiver and pop.group_of(sender) == pop.group_of(receiver)


REGISTRY = {"isolated": Isolated, "full": Full, "ring": Ring, "islands": Islands}
