"""Measure engine and evaluator throughput on this machine; prints JSON (copied into docs/sandbox1-status.md)."""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
from hanabi_learning_environment import pyhanabi

from culture.bots.runner import BotSpec, play_games
from culture.evaluate.pool import Evaluator
from culture.evaluate.selfplay import selfplay
from culture.game.hanabi import HanabiParams


def raw_engine_steps_per_s(n_games: int = 300) -> float:
    """HLE alone: first-legal-move policy through pyhanabi, no observation conversion."""
    steps, t = 0, time.perf_counter()
    for s in range(n_games):
        g = pyhanabi.HanabiGame({"players": 2, "seed": s, "random_start_player": False})
        st = g.new_initial_state()
        while not st.is_terminal():
            if st.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                st.deal_random_card()
                continue
            st.apply_move(st.legal_moves()[-1])
            steps += 1
    return steps / (time.perf_counter() - t)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
    P = HanabiParams()
    out = {"machine": f"{platform.machine()} {platform.platform()}", "cpus": os.cpu_count(), "workers": workers,
           "hle_raw_steps_per_s_1core": round(raw_engine_steps_per_s())}
    for name in ("random", "piers", "canaan_piers"):
        t = time.perf_counter()
        rs = play_games(P, [BotSpec.anchor(name)] * 2, list(range(100)))
        dt = time.perf_counter() - t
        out[f"{name}_1core_games_per_s"] = round(100 / dt, 1)
        out[f"{name}_1core_steps_per_s"] = round(sum(r.turns for r in rs) / dt)
    ev = Evaluator(P, workers=workers)
    ev.run([((BotSpec.anchor("random"),) * 2, [0, 1, 2, 3] * 1)])  # warm the pool
    means = {}
    for name in ("canaan_piers", "canaan_iggi", "canaan_flawed", "piers", "iggi", "flawed", "simple", "random"):
        t = time.perf_counter()
        rs = selfplay(ev, BotSpec.anchor(name), list(range(n)))
        dt = time.perf_counter() - t
        sc = np.array([r.score for r in rs], dtype=float)
        means[name] = {"mean": round(float(sc.mean()), 3), "sd": round(float(sc.std(ddof=1)), 3),
                       "se": round(float(sc.std(ddof=1) / np.sqrt(n)), 3), "zero_frac": round(float((sc == 0).mean()), 3),
                       "pool_games_per_s": round(n / dt, 1)}
    out["selfplay_1000"] = means
    out["rlimits"] = ev.rlimit_status()
    ev.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
