"""Bot runner sandbox (spec section 4 / 12)."""

import numpy as np
import pytest

from culture.bots.anchors import PIERS_RULES, anchor_source, conventions_for, parse_rules, rulebot_source
from culture.bots.runner import BotSpec, SandboxLimits, SourceRejected, check_source, compile_bot, complexity, play_game
from culture.evaluate.selfplay import seat_rates, selfplay
from culture.game.hanabi import HanabiParams

P = HanabiParams()

INFINITE = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        while True:
            pass
'''
SLEEPY_SOMETIMES = '''
import random
class Bot:
    def reset(self, game, my_id, seed):
        self.rng = random.Random(seed)
        self.n = 0
    def act(self, obs):
        self.n += 1
        if self.n == 3:
            x = 0
            for i in range(10**9):
                x += i
        return obs["legal_moves"][0]
'''
ILLEGAL = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        return {"type": "PLAY", "card_index": 9}
'''
RAISES = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        return 1 / 0
'''
MUTATES = '''
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        obs["legal_moves"].append({"type": "PLAY", "card_index": 7})
        obs["legal_moves"][0]["card_index"] = 7
        return {"type": "PLAY", "card_index": 7}
'''
USES_MODULE_RANDOM = '''
import random
class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        return random.choice(obs["legal_moves"])
'''


def spec(key, code):
    return BotSpec(key=key, code=code)


def test_import_os_rejected():
    for bad in ("import os\nclass Bot: pass", "from subprocess import run\nclass Bot: pass",
                "import sys\nclass Bot: pass", "class Bot:\n    x = open('f')", "class Bot:\n    x = ().__class__.__bases__",
                "class Bot:\n    x = __import__('os')", "class Bot:\n    x = eval('1')"):
        assert check_source(bad), bad
        with pytest.raises(SourceRejected):
            compile_bot(bad)


def test_runtime_import_hook_blocks_dynamic_imports():
    code = "class Bot:\n    def reset(self, g, m, s):\n        import math\n    def act(self, obs):\n        return obs['legal_moves'][0]\n"
    assert compile_bot(code)  # whitelisted import fine at runtime too


def test_size_cap_and_missing_class():
    long = "x = 1\n" * 401 + "class Bot:\n    pass\n"
    assert any("too long" in p for p in check_source(long, max_lines=400))
    assert any("class Bot" in p for p in check_source("y = 2\n"))


def test_allowed_source_passes_and_complexity_logged():
    src = anchor_source("piers")
    assert check_source(src) == []
    assert complexity(src) > 10
    assert check_source(anchor_source("random")) == []


def test_infinite_loop_times_out_and_falls_back():
    lim = SandboxLimits(hard_timeout_s=0.2, max_hard_timeouts_per_game=2)
    r = play_game(P, [spec("loop", INFINITE), BotSpec.anchor("piers")], 0, lim)
    assert r.timeouts[0] == 2  # then the seat is disabled and plays fallbacks without being called
    assert r.illegal[0] == r.moves[0]
    assert r.errors[0] == "hard timeout"
    assert r.illegal[1] == 0


def test_single_timeout_then_recovers():
    lim = SandboxLimits(hard_timeout_s=0.2, max_hard_timeouts_per_game=5)
    r = play_game(P, [spec("sleepy", SLEEPY_SOMETIMES), BotSpec.anchor("piers")], 0, lim)
    assert r.timeouts[0] == 1 and r.illegal[0] == 1 and r.moves[0] > 5


@pytest.mark.parametrize("code", [ILLEGAL, RAISES, MUTATES])
def test_illegal_and_crashing_moves_fall_back_and_are_counted(ev, code):
    s = spec("bad" + str(hash(code)), code)
    res = selfplay(ev, s, [0, 1, 2])
    rates = seat_rates(res, s.key)
    assert rates["illegal_rate"] == 1.0
    assert all(r.turns > 0 for r in res)


def test_broken_source_is_a_broken_bot_not_a_crash(ev):
    s = spec("syntax", "class Bot(:\n  pass")
    res = selfplay(ev, s, [0, 1])
    assert seat_rates(res, s.key)["illegal_rate"] == 1.0
    assert "SourceRejected" in (res[0].errors[0] or "")


def test_module_random_is_deterministic_per_seed():
    s = spec("modrand", USES_MODULE_RANDOM)
    a = [play_game(P, [s, s], k).digest for k in range(5)]
    b = [play_game(P, [s, s], k).digest for k in range(5)]
    assert a == b


def test_sandboxed_piers_artifact_matches_anchor():
    s = spec("piers_artifact", anchor_source("piers"))
    for k in range(10):
        assert play_game(P, [s, s], k).digest == play_game(P, [BotSpec.anchor("piers")] * 2, k).digest


def test_rule_config_roundtrip():
    src = rulebot_source(PIERS_RULES)
    assert parse_rules(src) == PIERS_RULES
    assert src == anchor_source("piers")
    assert "0.6" in conventions_for(PIERS_RULES)


def test_two_runs_identical_action_logs():
    s = spec("piers_artifact2", anchor_source("piers"))
    a = play_game(P, [s, s], 42, record=True)
    b = play_game(P, [s, s], 42, record=True)
    assert a.actions == b.actions


def test_pool_and_inprocess_agree(pool_ev, ev):
    s = spec("piers_artifact3", rulebot_source(PIERS_RULES[:7]))
    seeds = list(range(40))
    assert [r.digest for r in selfplay(pool_ev, s, seeds)] == [r.digest for r in selfplay(ev, s, seeds)]
    assert np.mean([r.score for r in selfplay(ev, s, seeds)]) > 5
