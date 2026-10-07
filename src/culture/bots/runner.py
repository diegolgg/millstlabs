"""Sandboxed execution of LLM-written bot code (spec section 4).

Guards against bugs, not adversaries (the restrictions themselves live in bots/sandbox.py):
- source check by AST plus a token scan; exec with a restricted builtins table, proxy modules and a private-attribute
  guard;
- per-move soft timeout (counted) and hard timeout (SIGALRM, move replaced by the fallback);
- worker processes (see evaluate/pool.py) add CPU and memory rlimits, a fixed PYTHONHASHSEED and a wall-clock watchdog.
Illegal moves, exceptions and hard timeouts all become the fallback move (discard oldest, else first legal) and count
towards `illegal_rate`.
"""

from __future__ import annotations

import builtins
import copy
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
from .sandbox import (SandboxViolation, SourceRejected, check_source, compile_sandboxed, complexity,  # noqa: F401
                      suspicious_tokens)


@dataclass(frozen=True)
class SandboxLimits:
    soft_timeout_ms: float = 50.0
    hard_timeout_s: float = 1.0
    max_lines: int = 400
    max_hard_timeouts_per_game: int = 2  # after this many, the seat plays fallback moves for the rest of the game
    cpu_seconds_per_task: int = 600
    memory_mb: int = 4096


def compile_bot(code: str, max_lines: int = 400, trusted: bool = False) -> type:
    """Check and exec bot source; return its `Bot` class. `trusted` skips the checks (anchors only)."""
    if trusted:
        code_obj = compile(code, "<bot.py>", "exec", dont_inherit=True)
        namespace: dict[str, Any] = {"__name__": "bot_module", "__builtins__": builtins}
    else:
        problems = check_source(code, max_lines)
        if problems:
            raise SourceRejected("; ".join(problems[:5]))
        code_obj, namespace = compile_sandboxed(code)
    exec(code_obj, namespace)  # noqa: S102 - the point of the sandbox
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
    """Hard-timeout signal. A BaseException (not an Exception) so a bot's `except Exception` cannot catch it; bare
    `except:`/`except BaseException` are rejected by the source check, so bot code cannot swallow it either."""


# True between an alarm firing and the next re-arm, so a swallowed HardTimeout is still detected after the bot returns.
_FIRED = [False]


def _on_alarm(signum, frame):
    _FIRED[0] = True
    # re-arm a short repeating timer: if the exception is somehow swallowed, it fires again within 10 ms, bounding the
    # overrun to about the hard limit rather than running unbounded.
    signal.setitimer(signal.ITIMER_REAL, 0.01)
    raise HardTimeout()


class _Alarm:
    """setitimer-based hard timeout; a no-op off the main thread. `fired` reports whether the timer went off (so a
    swallowed timeout is caught after the bot call returns)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.active = threading.current_thread() is threading.main_thread() and seconds > 0

    @property
    def fired(self) -> bool:
        return self.active and _FIRED[0]

    def __enter__(self):
        if self.active:
            _FIRED[0] = False
            self._old = signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
            _FIRED[0] = False
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
            with _Alarm(self.limits.hard_timeout_s) as alarm:
                out = fn(*args)
                swallowed = alarm.fired  # the timer went off but the bot returned anyway (it caught HardTimeout)
        except HardTimeout:
            return self._on_timeout()
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            self._note(f"{type(e).__name__}: {e}")
            return False, None
        if swallowed:
            return self._on_timeout()
        if (time.perf_counter() - t0) * 1000 > self.limits.soft_timeout_ms:
            self.slow += 1
        return True, out

    def _on_timeout(self):
        self.timeouts += 1
        self._note("hard timeout")
        if self.timeouts >= self.limits.max_hard_timeouts_per_game:
            self.disabled = True
        return False, None

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
    random.seed(bot_seed(seed, -1))  # module-level RNG: deterministic for bots that misuse it, and not the deal seed
    seats = [_Seat(s, bot_factory(s, limits), limits) for s in specs]
    for p, seat in enumerate(seats):
        seat.reset(copy.deepcopy(desc), p, bot_seed(seed, p))  # own copy: one seat cannot edit the other's rules
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
