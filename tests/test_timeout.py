"""Fix round 1, step 2: a bare `except` no longer swallows the hard timeout, and the pool has a wall-clock watchdog."""

import time

import pytest

from culture.bots.runner import BotSpec, SandboxLimits, check_source, play_game
from culture.evaluate.pool import Evaluator
from culture.evaluate.selfplay import seat_rates, selfplay
from culture.game.hanabi import HanabiParams

P = HanabiParams()

# The review's exploit: a Python-level loop that catches everything the hard timeout could raise.
SWALLOW = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        while True:
            try:
                x = 1
            except:
                pass
'''
SWALLOW_BASE = SWALLOW.replace("except:", "except BaseException:")
# a loop that swallows via `except Exception` -- must NOT catch the BaseException hard timeout
SWALLOW_EXC = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        while True:
            try:
                x = sum(range(1000))
            except Exception:
                pass
'''
# a C-level loop that ignores SIGALRM for seconds: in-worker the SIGALRM cannot interrupt it, so only the pool
# watchdog can stop it (a Python-level `while True` would be caught by the worker's own SIGALRM).
CLEVEL = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        return sum(range(10 ** 9))
'''


def test_bare_except_is_rejected_at_source():
    assert any("bare `except:`" in p for p in check_source(SWALLOW))
    assert any("except BaseException" in p for p in check_source(SWALLOW_BASE))


def test_except_exception_cannot_swallow_the_hard_timeout():
    # `except Exception` is legal source, so this runs; the BaseException timeout escapes it and is counted.
    lim = SandboxLimits(hard_timeout_s=0.2, max_hard_timeouts_per_game=2)
    t0 = time.perf_counter()
    r = play_game(P, [BotSpec("swallow_exc", SWALLOW_EXC), BotSpec.anchor("piers")], 0, lim)
    dt = time.perf_counter() - t0
    assert r.timeouts[0] >= 1
    assert r.illegal[0] == r.moves[0] and r.errors[0] == "hard timeout"
    assert dt < 2 * 0.2 * lim.max_hard_timeouts_per_game + 1.0  # stopped within a small multiple of the limit
    assert r.illegal[1] == 0


def test_pool_watchdog_stops_a_signal_ignoring_loop_and_survives():
    from culture.bots.anchors import anchor_source

    # chunk=1 with two seeds makes two tasks, so the pool path (workers>0 and >1 task) runs and the watchdog applies.
    ev = Evaluator(P, SandboxLimits(hard_timeout_s=0.2), workers=2, chunk=1, task_wall_budget_s=1.5)
    try:
        t0 = time.perf_counter()
        res = selfplay(ev, BotSpec("clevel", CLEVEL), [0, 1])
        dt = time.perf_counter() - t0
        assert dt < 2 * 1.5 + 3.0  # within about 2x the budget, not the many seconds the loop would take
        assert ev.watchdog_trips >= 1
        rates = seat_rates(res, "clevel")
        assert rates["illegal_rate"] == 1.0 and all("watchdog" in (r.errors[0] or "") for r in res)
        # the pool was rebuilt: a well-behaved bot still evaluates afterwards
        good = selfplay(ev, BotSpec("piers_art", anchor_source("piers")), [0, 1, 2])
        assert all(g.digest != "watchdog" for g in good)
    finally:
        ev.close()


def test_normal_bot_never_trips_the_watchdog():
    from culture.bots.anchors import anchor_source

    ev = Evaluator(P, workers=2, chunk=4, task_wall_budget_s=60.0)
    try:
        res = selfplay(ev, BotSpec("piers_ok", anchor_source("piers")), list(range(8)))
        assert ev.watchdog_trips == 0 and all(r.digest != "watchdog" for r in res)
    finally:
        ev.close()
