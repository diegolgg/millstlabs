"""Fix round 1, step 1: sandbox escapes found by the 2026-10-06 review, and nearby ones, must fail inside the sandbox;
legitimate bot idioms must keep working."""

import pytest

from culture.bots.anchors import anchor_source
from culture.bots.runner import BotSpec, SandboxLimits, bot_factory, check_source, compile_bot, play_game
from culture.game.hanabi import HanabiGame, HanabiParams

P = HanabiParams()
LIM = SandboxLimits(hard_timeout_s=1.0)

# The bot body runs `attempt(obs)` once per act call and appends what it got to LEAK (a module global the test reads
# back through the compiled class); any exception is recorded as a string.
TEMPLATE = '''
import random
import typing
import functools
import dataclasses
import collections
LEAK = []

def attempt(obs):
{body}

class Bot:
    def reset(self, game, my_id, seed):
        pass
    def act(self, obs):
        try:
            LEAK.append(("ok", attempt(obs)))
        except Exception as e:
            LEAK.append(("error", type(e).__name__ + ": " + str(e)))
        return obs["legal_moves"][0]
'''


def bot(body: str) -> str:
    return TEMPLATE.format(body="\n".join("    " + line for line in body.strip().splitlines()))


def run_leak(code: str, key: str):
    """Play one game; return (static problems, the LEAK list of the first act call or None if the bot never ran)."""
    problems = check_source(code)
    spec = BotSpec(key, code)
    res = play_game(P, [spec, BotSpec.anchor("piers")], 0, LIM)
    if problems:
        assert res.illegal[0] == res.moves[0] and "SourceRejected" in (res.errors[0] or "")
        return problems, None
    cls = bot_factory(spec, LIM)
    leak = cls.act.__globals__["LEAK"]
    return problems, leak


# ---------------------------------------------------------------------------------- the four demonstrated escapes
DEMONSTRATED = {
    "dunder_by_string": 'return getattr(getattr(getattr(1,"__cl"+"ass__"),"__ba"+"ses__")[0], "__subcl"+"asses__")()',
    "random_os": 'fd = random._os.open("/etc/hosts", 0)\nreturn random._os.read(fd, 50)',
    "typing_sys_builtins": 'return typing.sys.modules["builtins"].open("/etc/hosts").read()',
    "getframe_walk": ('f = typing.sys._getframe()\nwhile f is not None:\n    m = f.f_locals.get("match")\n'
                      '    if m is not None:\n        return str(m.hle_state())\n    f = f.f_back\nreturn None'),
}

# Variants that dodge the static checks and must be stopped at run time.
RUNTIME = {
    "getattr_private_module": 'return getattr(random, "_" + "os")',
    "getattr_dunder_built": 'u = "_" * 2\nreturn getattr(1, u + "class" + u)',
    "module_attr_is_hidden": 'return getattr(typing, "s" + "ys")',
    "forwardref_evaluate": 'fr = typing.ForwardRef("1")\nreturn fr._evaluate({}, None, frozenset())',
    "forwardref_unbound_self": ('class H:\n    def ev(self):\n        return self._evaluate({}, None, frozenset())\n'
                                'return H.ev(typing.ForwardRef("1"))'),
    "forwardref_subclass": ('class F(typing.ForwardRef, _root=True):\n    pass\n'
                            'return F("1")._evaluate({}, None, frozenset())'),
    "frame_by_getattr": 'def g():\n    yield 1\nreturn getattr(g(), "gi_" + "frame")',
    "get_type_hints_removed": "return typing.get_type_hints",
    "singledispatch_removed": "return functools.singledispatch",
    "update_wrapper_custom_lists": ('class W:\n    pass\nu = "_" * 2\n'
                                    'return functools.update_wrapper(W(), random.Random.seed, (), (u + "globals" + u,))'),
    "dataclass_name_injection": ('u = "_" * 2\nX = type("X", (), {u + "annotations" + u: {"a=0)#": int}})\n'
                                 "return dataclasses.dataclass(X)"),
    "setattr_on_library_class": 'setattr(random.Random, "choice", None)\nreturn 1',
    "proxy_is_read_only": "m = random\nm.choice = None\nreturn 1",
    "private_of_proxy": "return random._attrs",
}

# Variants the static checks must reject.
STATIC = {
    "frame_attribute": "def g():\n    yield 1\nreturn g().gi_frame",
    "format_traversal_literal": 'def g():\n    yield 1\nreturn "{0.gi_frame}".format(g())',
    "format_traversal_dynamic": 'def g():\n    yield 1\nreturn ("{0." + "gi_frame}").format(g())',
    "format_unbound": 'return str.format("{0.real}", 1)',
    "match_class_attr": "def g():\n    yield 1\nmatch g():\n    case object(gi_frame=f):\n        return f",
    "mro": "return Exception.mro()",
    "dunder_name_store": "__annotations__ = {}\nreturn 1",
    "monkeypatch_import": "random.Random.choice = None\nreturn 1",
    "private_import": "from typing import _GenericAlias\nreturn 1",
    "submodule_import": "import collections.abc as c\nimport typing.re\nreturn 1",
}


@pytest.mark.parametrize("name", sorted(DEMONSTRATED))
def test_demonstrated_escapes_fail(name):
    problems, leak = run_leak(bot(DEMONSTRATED[name]), f"esc_{name}")
    assert problems, f"{name} passed the static checks"
    assert leak is None


@pytest.mark.parametrize("name", sorted(RUNTIME))
def test_runtime_escapes_fail(name):
    problems, leak = run_leak(bot(RUNTIME[name]), f"rt_{name}")
    assert problems == [], problems  # the point is the run-time layer
    assert leak, "the bot never ran"
    status, value = leak[0]
    assert status == "error", f"{name} returned {value!r}"


@pytest.mark.parametrize("name", sorted(STATIC))
def test_static_rejections(name):
    assert check_source(bot(STATIC[name])), name


def test_token_scan_marks_artifact_invalid():
    for tok in ("sys", "_os", "_getframe", "__subclasses__", "__globals__", "__builtins__"):
        code = f"# uses {tok}\nclass Bot:\n    def reset(self, g, m, s):\n        pass\n    def act(self, obs):\n        return obs['legal_moves'][0]\n"
        assert any("suspicious tokens" in p for p in check_source(code)), tok
    assert not any("suspicious" in p for p in check_source(anchor_source("piers")))


SPY = '''
import typing
import random
SEEN = []
STOLEN = []

class Bot:
    def reset(self, game, my_id, seed):
        self.me = my_id
    def act(self, obs):
        SEEN.append([(c["color"], c["rank"]) for c in obs["my_hand"]])
        for probe in (lambda: getattr(typing, "s" + "ys"), lambda: getattr(random, "_" + "inst")):
            try:
                STOLEN.append(probe())
            except Exception:
                pass
        return obs["legal_moves"][0]
'''


def test_bot_cannot_see_its_own_cards():
    spec = BotSpec("spy", SPY)
    assert check_source(SPY) == []
    res = play_game(P, [spec, BotSpec.anchor("piers")], 3, LIM)
    assert res.errors[0] is None and res.illegal[0] == 0
    cls = bot_factory(spec, LIM)
    seen, stolen = cls.act.__globals__["SEEN"], cls.act.__globals__["STOLEN"]
    assert seen and all(c == (None, None) for hand in seen for c in hand)
    assert stolen == []
    match = HanabiGame(P).new_game(3)  # the engine does know the cards the bot was dealt
    hand0 = match.hle_state().player_hands()[0]
    assert len(hand0) == P.hand_size and all(c.color() >= 0 and c.rank() >= 0 for c in hand0)


LEGIT = '''
"""A docstring may mention __init__."""
import collections
import collections.abc
import dataclasses
import functools
import itertools
import math
import random
from collections import namedtuple
from typing import List, Optional

Card = namedtuple("Card", "color rank")


@dataclasses.dataclass(frozen=True)
class Note:
    slot: int
    why: str = ""


def memo(f):
    @functools.wraps(f)
    def inner(*a):
        return f(*a)
    return inner


class Base:
    def _helper(self):
        return 1


class Bot(Base):
    __slots__ = ("_rng", "_seen", "me")

    def __init__(self):
        self._seen = 0

    def reset(self, game, my_id, seed):
        self._rng = random.Random(seed)
        self.me = my_id

    @memo
    def _pick(self, n: int) -> int:
        return self._rng.randrange(n)

    def act(self, obs):
        self._seen += 1
        c = Card("R", 1)._replace(rank=2)
        assert c._asdict()["rank"] == 2 and isinstance(obs, collections.abc.Mapping)
        assert getattr(self, "_seen", None) == self._seen and hasattr(self, "_rng")
        assert super()._helper() == 1 and Note(1).slot == 1
        label = "{:.2f} {}".format(math.pi, len(list(itertools.islice(itertools.count(), 3))))
        xs: List[Optional[int]] = [None, 1]
        if __name__ == "__main__":
            pass
        return obs["legal_moves"][self._pick(len(obs["legal_moves"])) if label and xs else 0]
'''


def test_legitimate_idioms_still_work():
    assert check_source(LEGIT) == []
    res = play_game(P, [BotSpec("legit", LEGIT)] * 2, 0, LIM)
    assert res.errors == [None, None] and res.illegal == [0, 0]


def test_seats_get_their_own_game_description():
    code = ('class Bot:\n    def reset(self, game, my_id, seed):\n        game["colors"].clear()\n'
            '        self.n = game["num_players"]\n    def act(self, obs):\n        return obs["legal_moves"][0]\n')
    spy = ('SEEN = []\nclass Bot:\n    def reset(self, game, my_id, seed):\n        SEEN.append(list(game["colors"]))\n'
           '    def act(self, obs):\n        return obs["legal_moves"][0]\n')
    play_game(P, [BotSpec("clears", code), BotSpec("sees", spy)], 0, LIM)
    assert bot_factory(BotSpec("sees", spy), LIM).act.__globals__["SEEN"] == [["R", "Y", "G", "W", "B"]]


def test_compile_bot_rejects_and_anchors_pass():
    for name in ("random", "simple", "piers", "iggi", "flawed"):
        assert check_source(anchor_source(name)) == [], name
        compile_bot(anchor_source(name))
