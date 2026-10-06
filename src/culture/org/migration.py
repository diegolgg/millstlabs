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


REGISTRY = {"none": NoMigration}
