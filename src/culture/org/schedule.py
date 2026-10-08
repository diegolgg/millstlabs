"""Lever schedule (S1, switchback perturbation): organization policies that change at fixed generations inside one run.

Config: `org.schedule: [{at: 31, set: {routing: none}}, {at: 61, set: {routing: broadcast_group}}, ...]`. A lever is
any organization policy field (`routing`, `verification`, `topology`, `selection`, ...) or the boolean
`quarantine_unverified`. From generation `at` on, the lever takes the new value; levers not named keep their configured
value. `at` >= 1 and strictly increasing (generation 0, the warm start, always runs the configured organization).

The values in effect at generation g are a pure function of g and the config, so resume needs no extra state: the
run context re-applies the schedule at the checkpointed generation before it loads the policies' state, and the next
generation applies its own switch exactly as an uninterrupted run would.

When a policy is rebuilt (through `registry.make`) the old policy's state is handed to the new one through
`load_state_dict`, which every policy reads tolerantly. The case that matters is selection: `keep_best_k` and
`shinka_weighted` share the per-group archive format, so a switch keeps the archive (a switch to `keep_best_k` prunes
it to k at the next record). Islands migration: when the topology is `islands` and the migration policy is `none`, the
islands' own rate and interval drive random migration (`registry.effective_migration`), re-derived after every switch
that touches `topology` or `migration`. A switch to islands with `migration_rate: 0` is therefore "migration off" with
the same reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import registry

QUARANTINE = "quarantine_unverified"
LEVERS = tuple(sorted(registry.REGISTRIES)) + (QUARANTINE,)


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    """A policy value (same shape as run.config.PolicySpec; org/ does not import the run layer)."""

    name: str
    params: tuple[tuple[str, Any], ...] = ()

    @staticmethod
    def of(v: Any, where: str) -> "Spec":
        if isinstance(v, str):
            return Spec(v)
        if isinstance(v, dict) and "name" in v:
            params = dict(v.get("params", {}))
            params.update({k: x for k, x in v.items() if k not in ("name", "params")})
            return Spec(v["name"], tuple(sorted(params.items())))
        if hasattr(v, "name") and hasattr(v, "params"):  # run.config.PolicySpec
            return Spec(v.name, tuple(sorted(dict(v.params).items())))
        raise ScheduleError(f"{where}: a policy is a name or a mapping with `name`")

    def make(self, kind: str):
        return registry.make(kind, _SpecView(self.name, dict(self.params)))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **dict(self.params)}


@dataclass
class _SpecView:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Switch:
    at: int
    values: tuple[tuple[str, Any], ...]  # (lever, Spec or bool), sorted by lever


def parse_schedule(entries: list[dict[str, Any]] | None) -> list[Switch]:
    """Validate and normalize `org.schedule`. Raises ScheduleError with the offending entry."""
    out: list[Switch] = []
    last = 0
    for i, e in enumerate(entries or []):
        where = f"entry {i}"
        if not isinstance(e, dict) or set(e) != {"at", "set"}:
            raise ScheduleError(f"{where}: each entry is a mapping with exactly `at` and `set`")
        at = e["at"]
        if isinstance(at, bool) or not isinstance(at, int) or at < 1:
            raise ScheduleError(f"{where}: `at` must be an integer >= 1 (generation 0 is the warm start)")
        if at <= last:
            raise ScheduleError(f"{where}: `at` values must be strictly increasing")
        last = at
        if not isinstance(e["set"], dict) or not e["set"]:
            raise ScheduleError(f"{where}: `set` must be a non-empty mapping of lever -> value")
        vals = []
        for lever, v in sorted(e["set"].items()):
            if lever not in LEVERS:
                raise ScheduleError(f"{where}: unknown lever {lever!r}; options: {list(LEVERS)}")
            if lever == QUARANTINE:
                if not isinstance(v, bool):
                    raise ScheduleError(f"{where}: {QUARANTINE} must be true or false")
                vals.append((lever, v))
                continue
            spec = Spec.of(v, f"{where}.{lever}")
            try:
                spec.make(lever)
            except (TypeError, ValueError) as err:
                raise ScheduleError(f"{where}.{lever}: {err}") from err
            vals.append((lever, spec))
        out.append(Switch(at, tuple(vals)))
    return out


def describe_value(v: Any) -> Any:
    return v.describe() if isinstance(v, Spec) else v


class LeverSchedule:
    """The schedule of one run. `org` is the run's OrgConfig (base values of every lever)."""

    def __init__(self, org):
        self.org = org
        self.switches = parse_schedule(getattr(org, "schedule", None))
        self.levers = sorted({lever for s in self.switches for lever, _ in s.values})
        self.base = {lever: self._base_value(lever) for lever in self.levers}
        self.active = dict(self.base)  # what is currently applied to the policies

    def __bool__(self) -> bool:
        return bool(self.switches)

    def _base_value(self, lever: str) -> Any:
        v = getattr(self.org, lever)
        return bool(v) if lever == QUARANTINE else Spec.of(v, lever)

    def at(self, generation: int) -> dict[str, Any]:
        """Lever values in effect during `generation` (base values, then every switch with at <= generation)."""
        out = dict(self.base)
        for s in self.switches:
            if s.at > generation:
                break
            out.update(dict(s.values))
        return out

    def switch_generations(self) -> list[int]:
        return [s.at for s in self.switches]

    def describe(self, generation: int) -> dict[str, Any]:
        """JSON-able lever values for the generation record."""
        return {k: describe_value(v) for k, v in sorted(self.at(generation).items())}

    def apply(self, pol: dict[str, Any], generation: int) -> dict[str, Any]:
        """Bring the policies in `pol` to the values in effect at `generation`; returns those values. A policy is
        rebuilt only when its value changes, and receives its predecessor's state."""
        want = self.at(generation)
        rewire = False
        for lever, v in sorted(want.items()):
            if self.active.get(lever) == v:
                continue
            self.active[lever] = v
            if lever == QUARANTINE:
                continue
            new = v.make(lever)
            old = pol.get(lever)
            if old is not None:
                new.load_state_dict(old.state_dict())
            pol[lever] = new
            rewire |= lever in ("topology", "migration")
        if rewire:
            mig = self.active.get("migration") or Spec.of(self.org.migration, "migration")
            pol["migration"] = registry.effective_migration(pol["topology"], mig.make("migration"))
        return want

    def quarantine(self, generation: int) -> bool:
        return bool(self.at(generation).get(QUARANTINE, self.org.quarantine_unverified))
