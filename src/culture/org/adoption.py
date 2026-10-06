"""Adoption: what the receiver does with a verified (or unverified) delivered artifact."""

from __future__ import annotations

from ..artifacts.schema import Verification
from .base import Policy


class Adoption(Policy):
    kind = "adoption"
    llm_calls = 0

    def decide(self, verification: Verification | None) -> str:
        """'adopt', 'merge' or 'reject'."""
        raise NotImplementedError


class ReplaceIfBetter(Adoption):
    """Replace the incumbent when verification passes. With verification `none` there is nothing to compare, so the
    receiver adopts blindly: that is the 'transfer without verification' condition."""

    def decide(self, verification):
        if verification is None:
            return "adopt"
        return "adopt" if verification.passed else "reject"


class Never(Adoption):
    def decide(self, verification):
        return "reject"


class MergeLLM(Adoption):
    """One LLM call merges the payload into the incumbent when verification passes (or blindly without it)."""

    llm_calls = 1

    def decide(self, verification):
        if verification is None or verification.passed:
            return "merge"
        return "reject"


REGISTRY = {"replace_if_better": ReplaceIfBetter, "never": Never, "merge_llm": MergeLLM}
