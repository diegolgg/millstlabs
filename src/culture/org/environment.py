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


class VariantSwitch(Environment):
    """Switch to an HLE-expressible rule variant from `at_generation` on (Rogers test). `variant` is a mapping of
    game fields (colors, ranks, hand_size, max_information_tokens, max_life_tokens) or one of the named presets."""

    PRESETS = {
        "4_suits": {"colors": 4},
        "hand4_clues6": {"hand_size": 4, "max_information_tokens": 6},
    }
    FIELDS = {"colors", "ranks", "hand_size", "max_information_tokens", "max_life_tokens"}

    def __init__(self, at_generation: int = 10, variant: str | dict = "hand4_clues6"):
        v = self.PRESETS[variant] if isinstance(variant, str) else dict(variant)
        if not v or not set(v) <= self.FIELDS:
            raise ValueError(f"variant must set some of {sorted(self.FIELDS)} (or be one of {sorted(self.PRESETS)})")
        self.at, self.variant = at_generation, v
        self.name = variant if isinstance(variant, str) else ",".join(f"{k}={x}" for k, x in sorted(v.items()))

    def params_for(self, generation, base):
        p = dict(base)
        if generation >= self.at:
            p.update(self.variant)
            p["_note"] = f" (rule variant in force: {self.name})"
        return p


REGISTRY = {"fixed": Fixed, "variant_switch": VariantSwitch}
