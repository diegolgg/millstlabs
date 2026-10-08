"""Name -> class registries for every organization policy kind; `make(kind, spec)` builds one from config."""

from __future__ import annotations

from typing import Any

from . import adoption, allocation, credit, delivery, environment, migration, population, routing, selection, verification

REGISTRIES: dict[str, dict[str, Any]] = {
    "topology": population.REGISTRY,
    "routing": routing.REGISTRY,
    "delivery": delivery.REGISTRY,
    "verification": verification.REGISTRY,
    "adoption": adoption.REGISTRY,
    "credit": credit.REGISTRY,
    "allocation": allocation.REGISTRY,
    "selection": selection.REGISTRY,
    "migration": migration.REGISTRY,
    "environment": environment.REGISTRY,
    "teaching_cost": allocation.TEACHING_COST_REGISTRY,
}


def make(kind: str, spec) -> Any:
    reg = REGISTRIES[kind]
    if spec.name not in reg:
        raise ValueError(f"unknown {kind} policy {spec.name!r}; options: {sorted(reg)}")
    return reg[spec.name](**spec.params)


def check(kind: str, spec) -> None:
    from ..run.config import ConfigError

    try:
        make(kind, spec)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"org.{kind}: {e}") from e


def effective_migration(topology, migration):
    """The migration policy a run actually uses: islands(G, migration_rate, interval) carries its own migration rule,
    used when the configured migration is `none` (an explicit migration policy wins)."""
    if isinstance(topology, population.Islands) and isinstance(migration, migration_mod.NoMigration):
        return migration_mod.RandomMigration(topology.migration_rate, topology.interval)
    return migration


migration_mod = migration
