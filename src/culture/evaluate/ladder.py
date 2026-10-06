"""Frozen ladder: the current snapshot against every past generation's frozen snapshot and the anchor set."""

from __future__ import annotations

import numpy as np

from ..bots.runner import BotSpec
from .crossplay import crossplay
from .pool import Evaluator


def anchor_scores(ev: Evaluator, spec: BotSpec, anchors: list[str], seeds: list[int]) -> dict[str, float]:
    out = {}
    for name in anchors:
        res = crossplay(ev, spec, BotSpec.anchor(name), seeds)
        out[name] = float(np.mean([r.score for r in res])) if res else float("nan")
    return out


def ladder(ev: Evaluator, current: BotSpec, frozen: list[tuple[str, BotSpec]], anchors: list[str],
           seeds: list[int]) -> dict[str, float]:
    """Mean cross-play score of `current` with each frozen snapshot (label -> score) and each anchor."""
    out = {label: float(np.mean([r.score for r in crossplay(ev, current, spec, seeds)])) for label, spec in frozen}
    out.update({f"anchor:{k}": v for k, v in anchor_scores(ev, current, anchors, seeds).items()})
    return out


def non_cycling(ladder_rows: dict[int, dict[str, float]]) -> float:
    """Fraction of (later, earlier) generation pairs where the later snapshot scores at least as well against the
    earliest frozen opponent set. 1.0 means no cycling on the ladder."""
    gens = sorted(ladder_rows)
    ok = tot = 0
    for i, g in enumerate(gens):
        for h in gens[i + 1:]:
            common = set(ladder_rows[g]) & set(ladder_rows[h])
            if not common:
                continue
            tot += 1
            ok += np.mean([ladder_rows[h][k] for k in common]) >= np.mean([ladder_rows[g][k] for k in common])
    return ok / tot if tot else float("nan")
