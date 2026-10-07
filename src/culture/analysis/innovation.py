"""Innovation base rate (spec idea O; fix round 1, step 12).

epsilon = P(a revision improves on its parent on held-out deals by more than a declared margin), estimated per
revise-context condition with a Beta posterior and a 95% equal-tailed credible interval.

Outcome of one revision: delta = candidate - parent, both self-play means on the generation's held-out evaluation
deals (paired, fix round 1 step 4; disjoint from the feedback deals whose traces the model saw, step 10). It counts as
an innovation iff delta > margin (strictly; delta == margin is not one). "Incumbent" in the definition is the revised
parent, which is the incumbent under keep_best_k.

Context conditions, read from each agent's `revise_context` in the generation record:
- `nothing`: no feedback traces and nothing received;
- `feedback_only`: own feedback traces, nothing received (no delivered message, no corpus entry);
- `feedback_plus_received`: own feedback traces plus received artifacts (delivered messages or corpus entries);
- `received_only`: received artifacts without traces (reported, not one of the three named conditions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scipy import stats as st

from .metrics import load_run

CONTEXTS = ("nothing", "feedback_only", "feedback_plus_received")


def beta_posterior(successes: int, n: int, prior: tuple[float, float] = (1.0, 1.0),
                   level: float = 0.95) -> dict[str, float]:
    """Beta(a0 + s, b0 + n - s) posterior for a Bernoulli rate; equal-tailed credible interval."""
    if not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    a, b = prior[0] + successes, prior[1] + n - successes
    tail = (1 - level) / 2
    return {"n": int(n), "successes": int(successes), "a": float(a), "b": float(b), "mean": a / (a + b),
            "lo": float(st.beta.ppf(tail, a, b)), "hi": float(st.beta.ppf(1 - tail, a, b)), "level": level}


def context_of(rc: dict[str, Any] | None) -> str | None:
    if not rc:
        return None
    received = (rc.get("received", 0) or 0) > 0 or (rc.get("corpus", 0) or 0) > 0
    if rc.get("traces"):
        return "feedback_plus_received" if received else "feedback_only"
    return "received_only" if received else "nothing"


def revision_outcomes(run: dict[str, Any], margin: float = 0.0) -> list[dict[str, Any]]:
    """One row per revision that produced an admissible candidate with a paired parent score."""
    out = []
    for r in run["generations"]:
        for a, v in sorted(r["agents"].items()):
            if not v.get("candidate") or v.get("candidate_score") is None or v.get("parent_score") is None:
                continue
            delta = float(v["candidate_score"]) - float(v["parent_score"])
            out.append({"generation": r["generation"], "agent": a, "candidate": v["candidate"],
                        "context": context_of(v.get("revise_context")), "delta": delta,
                        "improved": delta > margin})
    return out


def innovation_by_context(outcomes: list[dict[str, Any]], prior: tuple[float, float] = (1.0, 1.0),
                          level: float = 0.95) -> dict[str, dict[str, float]]:
    groups: dict[str, list[bool]] = {}
    for o in outcomes:
        if o["context"] is not None:
            groups.setdefault(o["context"], []).append(bool(o["improved"]))
    return {c: beta_posterior(sum(v), len(v), prior, level) for c, v in sorted(groups.items())}


def estimate(run_dirs: list[str | Path], margin: float = 0.0, prior: tuple[float, float] = (1.0, 1.0),
             level: float = 0.95) -> dict[str, Any]:
    outcomes = [o for d in run_dirs for o in revision_outcomes(load_run(d), margin)]
    return {"margin": margin, "prior": list(prior), "level": level, "n_revisions": len(outcomes),
            "by_context": innovation_by_context(outcomes, prior, level)}


# ---------------------------------------------------------------------------------------------- probe helper
CONTEXT_OVERRIDES: dict[str, dict[str, Any]] = {
    "nothing": {"evaluation": {"feedback_games": 0}, "org": {"routing": "none"}, "corpus": {"enabled": False}},
    "feedback_only": {"org": {"routing": "none"}, "corpus": {"enabled": False}},
    "feedback_plus_received": {"org": {"routing": "broadcast_group", "verification": "none"},
                               "corpus": {"enabled": True}},
}


def run_innovation_probe(base, out_dir: str | Path, contexts: tuple[str, ...] = CONTEXTS, margin: float = 0.0,
                         seeds: tuple[int, ...] = (0,)) -> dict[str, Any]:
    """Run the base config once per context condition (and population seed) and estimate epsilon per context."""
    from ..run.config import from_dict
    from ..run.runner import run_config

    dirs = []
    for c in contexts:
        for s in seeds:
            cfg = from_dict(dict(CONTEXT_OVERRIDES[c], condition=f"ctx_{c}", population={"seed": s}), base)
            d = Path(out_dir) / c / f"p{s}"
            run_config(cfg, d)
            dirs.append(d)
    return estimate(dirs, margin)
