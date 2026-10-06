"""Cross-play on shared seeds. Seats alternate by seed index so neither artifact always moves first."""

from __future__ import annotations

import numpy as np

from ..bots.runner import BotSpec, GameResult
from .pool import Evaluator


def pair_jobs(a: BotSpec, b: BotSpec, seeds: list[int]) -> list[tuple[tuple[BotSpec, ...], list[int]]]:
    if a.key == b.key:
        return [((a, a), seeds)]
    return [((a, b), seeds[0::2]), ((b, a), seeds[1::2])]


def crossplay(ev: Evaluator, a: BotSpec, b: BotSpec, seeds: list[int]) -> list[GameResult]:
    """Games of a with b on `seeds`, returned in seed order."""
    out = ev.run(pair_jobs(a, b, seeds))
    if len(out) == 1:
        return out[0]
    merged = {r.seed: r for part in out for r in part}
    return [merged[s] for s in seeds]


def scores(results: list[GameResult]) -> np.ndarray:
    return np.array([r.score for r in results], dtype=float)


def crossplay_matrix(ev: Evaluator, specs: list[BotSpec], seeds: list[int]) -> np.ndarray:
    """Mean score for every ordered pair; symmetric because seats alternate. Diagonal is self-play."""
    jobs, index = [], []
    for i, a in enumerate(specs):
        for j, b in enumerate(specs):
            if j < i:
                continue
            pj = pair_jobs(a, b, seeds)
            index.append((i, j, len(jobs), len(pj)))
            jobs.extend(pj)
    out = ev.run(jobs)
    m = np.zeros((len(specs), len(specs)))
    for i, j, start, n in index:
        vals = [r.score for part in out[start:start + n] for r in part]
        m[i, j] = m[j, i] = float(np.mean(vals)) if vals else float("nan")
    return m
