"""Sandboxed execution of LLM-written bot code (spec section 4).

Guards against bugs, not adversaries:
- source check by AST: imports only from the whitelist, no dunder escapes, no I/O builtins, size cap;
- exec with a restricted builtins table and an import hook that only admits the whitelist;
- per-move soft timeout (counted) and hard timeout (SIGALRM, move replaced by the fallback);
- worker processes (see evaluate/pool.py) add CPU and memory rlimits and a fixed PYTHONHASHSEED.
Illegal moves, exceptions and hard timeouts all become the fallback move (discard oldest, else first legal) and count
towards `illegal_rate`.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import random
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..game.hanabi import HanabiGame, HanabiParams, IllegalMove, move_key
from ..game.seeds import bot_seed
from .interface import ALLOWED_IMPORTS

BANNED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "breakpoint",
    "memoryview", "__builtins__", "exit", "quit", "help", "setattr", "delattr", "__loader__", "__spec__",
}
ALLOWED_DUNDERS = {"__init__", "__name__", "__class__", "__eq__", "__hash__", "__lt__", "__repr__", "__str__", "__len__"}
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "getattr", "hasattr",
    "hash", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "type",
    "property", "staticmethod", "classmethod", "super", "Exception", "ValueError", "KeyError", "IndexError",
    "TypeError", "StopIteration", "AttributeError", "ZeroDivisionError", "RuntimeError", "AssertionError",
    "NotImplementedError", "ArithmeticError", "LookupError", "True", "False", "None", "callable", "chr", "ord",
    "format", "id", "bin", "hex", "NotImplemented", "Ellipsis",
]


class SourceRejected(ValueError):
    pass


@dataclass(frozen=True)
class SandboxLimits:
    soft_timeout_ms: float = 50.0
    hard_timeout_s: float = 1.0
    max_lines: int = 400
    max_hard_timeouts_per_game: int = 2  # after this many, the seat plays fallback moves for the rest of the game
    cpu_seconds_per_task: int = 600
    memory_mb: int = 4096


def check_source(code: str, max_lines: int = 400) -> list[str]:
    """Return a list of violations (empty means the source is admissible)."""
    problems: list[str] = []
    n_lines = len(code.splitlines())
    if n_lines > max_lines:
        problems.append(f"too long: {n_lines} lines > {max_lines}")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return problems + [f"syntax error: {e.msg} (line {e.lineno})"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of {a.name!r} not allowed (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from {node.module!r} not allowed (line {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"name {node.id!r} not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
            problems.append(f"attribute {node.attr!r} not allowed (line {node.lineno})")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Bot" for n in tree.body):
        problems.append("no top-level `class Bot`")
    return problems


def complexity(code: str) -> int:
    """Logged complexity measure: 1 + number of branch points (if/for/while/try/boolop/comprehension/lambda)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1
    kinds = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.comprehension, ast.Lambda, ast.IfExp)
    return 1 + sum(isinstance(n, kinds) for n in ast.walk(tree))


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in bot code")
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    b = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES if hasattr(builtins, k)}
    b["__import__"] = _restricted_import
    b["__build_class__"] = builtins.__build_class__
    b["__name__"] = "bot"
    return b


def compile_bot(code: str, max_lines: int = 400, trusted: bool = False) -> type:
    """Check and exec bot source; return its `Bot` class. `trusted` skips the checks (anchors only)."""
    if not trusted:
        problems = check_source(code, max_lines)
        if problems:
            raise SourceRejected("; ".join(problems[:5]))
    namespace: dict[str, Any] = {"__name__": "bot_module", "__builtins__": builtins if trusted else _safe_builtins()}
    exec(compile(code, "<bot.py>", "exec"), namespace)  # noqa: S102 - the point of the sandbox
    cls = namespace.get("Bot")
    if not isinstance(cls, type) or not callable(getattr(cls, "act", None)) or not callable(getattr(cls, "reset", None)):
        raise SourceRejected("`Bot` must be a class with reset() and act()")
    return cls


# ----------------------------------------------------------------------------------------------- bot specs
@dataclass(frozen=True, eq=False)
class BotSpec:
    """What to put in a seat. `code is None` means a named anchor (trusted, not sandboxed)."""

    key: str
    code: str | None = None

    def __eq__(self, other):
        return isinstance(other, BotSpec) and other.key == self.key

    def __hash__(self):
        return hash(self.key)

    @staticmethod
    def anchor(name: str) -> "BotSpec":
        return BotSpec(key=f"anchor:{name}")

    @property
    def is_anchor(self) -> bool:
        return self.code is None


_CLASS_CACHE: dict[str, Any] = {}


def bot_factory(spec: BotSpec, limits: SandboxLimits):
    """Return a zero-arg constructor for the seat's bot (compiled once per process and key)."""
    if spec.key in _CLASS_CACHE:
        return _CLASS_CACHE[spec.key]
    if spec.code is None:
        name = spec.key.split(":", 1)[1]
        if name.startswith("canaan_"):
            from .anchors.canaan_loader import CanaanBot

            kind = name[len("canaan_"):]
            factory: Any = lambda: CanaanBot(kind)  # noqa: E731
        else:
            from .anchors import anchor_source

            factory = compile_bot(anchor_source(name), trusted=True)
    else:
        try:
            factory = compile_bot(spec.code, limits.max_lines)
        except BaseException as e:  # noqa: BLE001 - any failure to load is a broken bot, not a harness crash
            if isinstance(e, KeyboardInterrupt):
                raise
            factory = _BrokenBot.factory(f"{type(e).__name__}: {e}")
    if len(_CLASS_CACHE) > 512:
        _CLASS_CACHE.clear()
    _CLASS_CACHE[spec.key] = factory
    return factory


class _BrokenBot:
    def __init__(self, err: str):
        self.err = err

    @staticmethod
    def factory(err: str):
        return lambda: _BrokenBot(err)

    def reset(self, game, my_id, seed):
        raise RuntimeError(self.err)

    def act(self, obs):
        raise RuntimeError(self.err)


# ----------------------------------------------------------------------------------------------- timeouts
class HardTimeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise HardTimeout()


class _Alarm:
    """setitimer-based hard timeout; a no-op off the main thread."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.active = threading.current_thread() is threading.main_thread() and seconds > 0

    def __enter__(self):
        if self.active:
            self._old = signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        return False


# ----------------------------------------------------------------------------------------------- games
@dataclass
class GameResult:
    seed: int
    score: int
    turns: int
    seats: list[str]
    moves: list[int]
    illegal: list[int]
    timeouts: list[int]
    slow: list[int]
    errors: list[str | None]
    digest: str
    actions: list | None = None
    trace: list | None = None
    deck: list | None = None
    hl_actions: list | None = None


def _key_to_move(k: tuple) -> dict[str, Any]:
    if k[0] in ("PLAY", "DISCARD"):
        return {"type": k[0], "card_index": k[1]}
    if k[0] == "REVEAL_COLOR":
        return {"type": k[0], "target": k[1], "color": k[2]}
    return {"type": k[0], "target": k[1], "rank": k[2]}


class _Seat:
    def __init__(self, spec: BotSpec, factory, limits: SandboxLimits):
        self.spec, self.limits = spec, limits
        self.factory = factory
        self.bot = None
        self.moves = self.illegal = self.timeouts = self.slow = 0
        self.error: str | None = None
        self.disabled = False

    def _guard(self, fn, *args):
        """Run bot code. Returns (ok, value). Trusted anchors run unguarded for speed."""
        if self.spec.is_anchor:
            return True, fn(*args)
        t0 = time.perf_counter()
        try:
            with _Alarm(self.limits.hard_timeout_s):
                out = fn(*args)
        except HardTimeout:
            self.timeouts += 1
            self._note("hard timeout")
            if self.timeouts >= self.limits.max_hard_timeouts_per_game:
                self.disabled = True
            return False, None
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            self._note(f"{type(e).__name__}: {e}")
            return False, None
        if (time.perf_counter() - t0) * 1000 > self.limits.soft_timeout_ms:
            self.slow += 1
        return True, out

    def _note(self, msg: str):
        if self.error is None:
            self.error = msg[:300]

    def reset(self, game_desc, seat, seed):
        ok, bot = self._guard(self.factory)
        if not ok:
            self.disabled = True
            return
        self.bot = bot
        ok, _ = self._guard(bot.reset, game_desc, seat, seed)
        if not ok:
            self.disabled = True

    def act(self, obs, legal_keys: set, fallback: dict) -> tuple[dict, bool]:
        self.moves += 1
        if self.disabled or self.bot is None:
            self.illegal += 1
            return fallback, False
        ok, move = self._guard(self.bot.act, obs)
        if ok:
            try:
                k = move_key(move)
            except (IllegalMove, KeyError, TypeError, ValueError, AttributeError):
                k = None
            if k is not None and k in legal_keys:
                return _key_to_move(k), True
            self._note(f"illegal move {str(move)[:120]}")
        self.illegal += 1
        return fallback, False


def play_game(params: HanabiParams, specs: list[BotSpec], seed: int, limits: SandboxLimits | None = None,
              trace: bool = False, record: bool = False) -> GameResult:
    """One game. `record` keeps the action log, deal order and hanab.live actions; `trace` keeps per-turn observations."""
    limits = limits or SandboxLimits()
    game = HanabiGame(params)
    match = game.new_game(seed)
    desc = params.describe()
    random.seed(seed)  # bots that misuse the module-level RNG still behave deterministically
    seats = [_Seat(s, bot_factory(s, limits), limits) for s in specs]
    for p, seat in enumerate(seats):
        seat.reset(desc, p, bot_seed(seed, p))
    h = hashlib.sha256()
    actions: list | None = [] if record else None
    turns: list | None = [] if trace else None
    while not match.is_terminal():
        p = match.current_player
        obs = match.observation(p)
        legal = obs["legal_moves"]
        legal_keys = {move_key(m) for m in legal}
        fallback = dict(game.fallback_move(legal))
        snapshot = json.dumps(obs) if trace else None
        move, ok = seats[p].act(obs, legal_keys, fallback)
        match.step(move, [_key_to_move(k) for k in legal_keys] if not ok else [move])
        h.update(f"{p}:{json.dumps(move, sort_keys=True)};".encode())
        if record:
            actions.append([p, move])
        if trace:
            turns.append({"obs": json.loads(snapshot), "move": move, "ok": ok})
    return GameResult(
        seed=seed,
        score=match.score(),
        turns=match.turn,
        seats=[s.key for s in specs],
        moves=[s.moves for s in seats],
        illegal=[s.illegal for s in seats],
        timeouts=[s.timeouts for s in seats],
        slow=[s.slow for s in seats],
        errors=[s.error for s in seats],
        digest=h.hexdigest()[:16],
        actions=actions,
        trace=turns,
        deck=[list(c) for c in match.deck] if record else None,
        hl_actions=list(match.actions) if record else None,
    )


def play_games(params: HanabiParams, specs: list[BotSpec], seeds: list[int], limits: SandboxLimits | None = None,
               trace: bool = False, record: bool = False) -> list[GameResult]:
    return [play_game(params, specs, s, limits, trace=trace, record=record) for s in seeds]
