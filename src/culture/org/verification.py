"""Verification: the receiver's engine-time check of a delivered artifact before (or after) adopting it."""

from __future__ import annotations

from ..artifacts.schema import Verification
from .base import AgentState, Judge, Policy


class VerificationPolicy(Policy):
    kind = "verification"
    n = 0
    adopt_first = False  # critical social learning adopts before checking

    def verify(self, message_id: str, payload: str, receiver: AgentState, judge: Judge, seeds: list[int],
               generation: int, illegal_max: float) -> Verification | None:
        raise NotImplementedError


class NoVerification(VerificationPolicy):
    def verify(self, message_id, payload, receiver, judge, seeds, generation, illegal_max):
        return None


class SelfPlay(VerificationPolicy):
    """Payload and incumbent each play N seeded self-play games on the same verification seeds."""

    def __init__(self, n: int = 200, margin: float = 0.0):
        self.n, self.margin = n, margin

    def _score(self, aid, judge, seeds):
        return judge.selfplay(aid, seeds)

    def verify(self, message_id, payload, receiver, judge, seeds, generation, illegal_max):
        seeds = seeds[: self.n]
        p_score, p_illegal = self._score(payload, judge, seeds)
        i_score = self._score(receiver.incumbent, judge, seeds)[0] if receiver.incumbent else float("-inf")
        passed = p_illegal <= illegal_max and p_score > i_score + self.margin
        return Verification(message_id, receiver.id, generation, type(self).__name__, len(seeds), p_score,
                            i_score, None, {}, p_illegal, passed)


class SelfPlayCrossplay(SelfPlay):
    """Self-play plus cross-play with the incumbent and the anchors: the payload must also beat the incumbent when
    paired with the anchors (a partner-robustness check)."""

    def __init__(self, n: int = 200, margin: float = 0.0):
        super().__init__(n, margin)

    def verify(self, message_id, payload, receiver, judge, seeds, generation, illegal_max):
        v = super().verify(message_id, payload, receiver, judge, seeds, generation, illegal_max)
        seeds = seeds[: self.n]
        v.anchors = judge.anchors(payload, seeds)
        if receiver.incumbent:
            v.crossplay_with_incumbent = judge.crossplay(payload, receiver.incumbent, seeds)
            inc_anchor = judge.anchors(receiver.incumbent, seeds)
            pa = sum(v.anchors.values()) / max(len(v.anchors), 1)
            ia = sum(inc_anchor.values()) / max(len(inc_anchor), 1)
            v.passed = v.passed and pa >= ia
        v.policy = "SelfPlayCrossplay"
        return v


class CriticalSocialLearning(SelfPlay):
    """Enquist et al.: adopt first, re-verify after one revision, revert if below theta.

    `verify` returns a provisional pass (no games); the generation loop records a pending check on the receiver and,
    after its next revision, calls `recheck`: if the adopted lineage's self-play is below `theta` (absolute score) or
    below the pre-adoption incumbent on the same seeds, the agent reverts to its previous artifact."""

    adopt_first = True

    def __init__(self, theta: float = 0.0, n: int = 200):
        super().__init__(n, 0.0)
        self.theta = theta

    def verify(self, message_id, payload, receiver, judge, seeds, generation, illegal_max):
        return Verification(message_id, receiver.id, generation, "CriticalSocialLearning", 0, float("nan"),
                            float("nan"), None, {}, 0.0, True)

    def recheck(self, current: str, previous: str | None, judge: Judge, seeds: list[int], illegal_max: float) -> dict:
        seeds = seeds[: self.n]
        cur, cur_illegal = judge.selfplay(current, seeds)
        prev = judge.selfplay(previous, seeds)[0] if previous else float("-inf")
        keep = cur_illegal <= illegal_max and cur >= self.theta and cur >= prev
        return {"keep": keep, "current": cur, "previous": prev, "n": len(seeds)}


REGISTRY = {"none": NoVerification, "selfplay": SelfPlay, "selfplay_crossplay": SelfPlayCrossplay,
            "critical_social_learning": CriticalSocialLearning}
