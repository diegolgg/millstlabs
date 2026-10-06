"""Environment schedule: the game rules in force at each generation (Rogers test via variant switches)."""

from __future__ import annotations

from typing import Any

from .base import Policy


class Environment(Policy):
    kind = "environment"

    def params_for(self, generation: int, base: dict[str, Any]) -> dict[str, Any]:
        return dict(base)


class Fixed(Environment):
    pass


REGISTRY = {"fixed": Fixed}
