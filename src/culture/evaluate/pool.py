"""Process pool for game evaluation, with per-generation memoization of (seats, seed) -> GameResult.

Games are deterministic given (seat keys, seed, rules), so memoization is exact: an artifact's self-play on this
generation's seeds is computed once even if verification, evaluation and the ladder all ask for it.
Workers are spawned with PYTHONHASHSEED=0 (set-iteration order is stable across processes) and best-effort rlimits.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from ..bots.runner import BotSpec, GameResult, SandboxLimits, play_games
from ..game.hanabi import HanabiParams

RLIMIT_STATUS: dict[str, str] = {}


def _worker_init(limits: SandboxLimits) -> None:
    import resource

    for name, lim, value in (("cpu", resource.RLIMIT_CPU, limits.cpu_seconds_per_task * 100),
                             ("as", resource.RLIMIT_AS, limits.memory_mb * 1024 * 1024)):
        try:
            soft, hard = resource.getrlimit(lim)
            new_hard = hard if hard != resource.RLIM_INFINITY and hard < value else value
            resource.setrlimit(lim, (min(value, new_hard), new_hard))
            RLIMIT_STATUS[name] = "set"
        except (ValueError, OSError) as e:  # macOS refuses RLIMIT_AS; recorded in the status doc
            RLIMIT_STATUS[name] = f"unavailable: {e}"


def _run_chunk(params_dict: dict, specs: tuple[BotSpec, ...], seeds: list[int], limits: SandboxLimits) -> list[GameResult]:
    return play_games(HanabiParams(**params_dict), list(specs), seeds, limits)


def _worker_rlimit_status() -> dict:
    return dict(RLIMIT_STATUS)


class Evaluator:
    def __init__(self, params: HanabiParams | None = None, limits: SandboxLimits | None = None, workers: int = 0,
                 chunk: int = 16):
        self.params = params or HanabiParams()
        self.limits = limits or SandboxLimits()
        self.workers = workers
        self.chunk = chunk
        self._memo: dict[tuple, GameResult] = {}
        self._pool: ProcessPoolExecutor | None = None
        self.games_played = 0

    # ------------------------------------------------------------------ pool management
    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # Workers may be spawned lazily, so the variable stays set for the life of this process (children only).
            os.environ["PYTHONHASHSEED"] = "0"
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"),
                                             initializer=_worker_init, initargs=(self.limits,))
        return self._pool

    def rlimit_status(self) -> dict:
        if self.workers <= 0:
            return {"mode": "in-process (no rlimits)"}
        return self._get_pool().submit(_worker_rlimit_status).result()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def clear_memo(self) -> None:
        self._memo.clear()

    def set_params(self, params: HanabiParams) -> None:
        if params != self.params:
            self.params = params
            self._memo.clear()

    # ------------------------------------------------------------------ evaluation
    def run(self, jobs: list[tuple[tuple[BotSpec, ...], list[int]]]) -> list[list[GameResult]]:
        """Each job is (seat specs, seeds). Returns results per job, in seed order."""
        missing: dict[tuple[BotSpec, ...], list[int]] = {}
        for specs, seeds in jobs:
            keys = tuple(s.key for s in specs)
            for sd in seeds:
                if (keys, sd) not in self._memo:
                    lst = missing.setdefault(tuple(specs), [])
                    if not lst or lst[-1] != sd:
                        lst.append(sd)
        tasks = []
        for specs, seeds in missing.items():
            seeds = list(dict.fromkeys(seeds))
            for i in range(0, len(seeds), self.chunk):
                tasks.append((specs, seeds[i:i + self.chunk]))
        if tasks:
            if self.workers > 0 and len(tasks) > 1:
                pool = self._get_pool()
                pd = asdict(self.params)
                futs = [pool.submit(_run_chunk, pd, specs, seeds, self.limits) for specs, seeds in tasks]
                outs = [f.result() for f in futs]
            else:
                outs = [play_games(self.params, list(specs), seeds, self.limits) for specs, seeds in tasks]
            for (specs, seeds), res in zip(tasks, outs):
                keys = tuple(s.key for s in specs)
                for r in res:
                    self._memo[(keys, r.seed)] = r
                self.games_played += len(res)
        return [[self._memo[(tuple(s.key for s in specs), sd)] for sd in seeds] for specs, seeds in jobs]
