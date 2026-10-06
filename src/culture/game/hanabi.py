"""Hanabi adapter over hanabi-learning-environment (HLE, vendored at 54e7959).

Converts HLE's relative, 0-based view into the bot-facing observation JSON of spec section 3: absolute player ids,
1-based ranks, colors R Y G W B. One `pyhanabi.HanabiGame` per game (`seed = base + game_index`) so a game's deals never
depend on earlier games. Illegal moves are rejected here (HLE would assert). The match also tracks deal order so the game
can be exported as a hanab.live replay.

Deviation from the spec's JSON (recorded in docs/sandbox1-status.md): every card carries
`"possible": {"colors": [...], "ranks": [...]}`, HLE's plausibility sets including negative information. HLE exposes it to
every agent and Canaan's agents use it, so leaving it out would make the adapter lose information the raw engine gives.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from hanabi_learning_environment import pyhanabi

from .interface import IllegalMove

ALL_COLORS = ["R", "Y", "G", "W", "B"]
_MT = pyhanabi.HanabiMoveType
_TYPE_NAME = {_MT.PLAY: "PLAY", _MT.DISCARD: "DISCARD", _MT.REVEAL_COLOR: "REVEAL_COLOR", _MT.REVEAL_RANK: "REVEAL_RANK"}


@dataclass(frozen=True)
class HanabiParams:
    """HLE-expressible rule parameters. Defaults are 2-player No Variant."""

    players: int = 2
    colors: int = 5
    ranks: int = 5
    hand_size: int = 5
    max_information_tokens: int = 8
    max_life_tokens: int = 3

    def hle_config(self, seed: int) -> dict[str, Any]:
        return {
            "players": self.players,
            "colors": self.colors,
            "ranks": self.ranks,
            "hand_size": self.hand_size,
            "max_information_tokens": self.max_information_tokens,
            "max_life_tokens": self.max_life_tokens,
            "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
            "seed": int(seed),
            "random_start_player": False,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "num_players": self.players,
            "hand_size": self.hand_size,
            "colors": ALL_COLORS[: self.colors],
            "ranks": list(range(1, self.ranks + 1)),
            "max_information_tokens": self.max_information_tokens,
            "max_life_tokens": self.max_life_tokens,
        }


def move_key(move: dict[str, Any]) -> tuple:
    """Canonical tuple for a bot-facing move dict (used for legality checks and action logs)."""
    t = move.get("type")
    if t in ("PLAY", "DISCARD"):
        return (t, int(move["card_index"]))
    if t == "REVEAL_COLOR":
        return (t, int(move["target"]), str(move["color"]))
    if t == "REVEAL_RANK":
        return (t, int(move["target"]), int(move["rank"]))
    raise IllegalMove(f"unknown move type {t!r}")


class HanabiMatch:
    """One game. `observation(p)` is the only view a bot ever gets."""

    def __init__(self, params: HanabiParams, seed: int):
        self.params = params
        self.seed = int(seed)
        self.turn = 0
        self._colors = ALL_COLORS[: params.colors]
        self._game = pyhanabi.HanabiGame(params.hle_config(self.seed))
        self._state = self._game.new_initial_state()
        # deal-order bookkeeping for the hanab.live export
        self.deck: list[tuple[int, int]] = []  # (color index, rank index 0-based) in deal order
        self._slot_order: list[list[int]] = [[] for _ in range(params.players)]
        self.actions: list[dict[str, Any]] = []  # hanab.live-style action records
        self._deal()

    # ----------------------------------------------------------------- engine plumbing
    def _deal(self) -> None:
        while self._state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            sizes = [len(h) for h in self._state.player_hands()]
            self._state.deal_random_card()
            hands = self._state.player_hands()
            for p, h in enumerate(hands):
                if len(h) > sizes[p]:
                    card = h[-1]
                    self._slot_order[p].append(len(self.deck))
                    self.deck.append((card.color(), card.rank()))

    @property
    def current_player(self) -> int:
        return self._state.cur_player()

    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def score(self) -> int:
        """HLE score: sum of fireworks, or 0 if all lives were lost."""
        return self._state.score()

    def fireworks_sum(self) -> int:
        return sum(self._state.fireworks())

    def hle_state(self):
        return self._state

    # ----------------------------------------------------------------- observations
    def _card_json(self, card, knowledge, visible: bool) -> dict[str, Any]:
        colors = self._colors
        c = knowledge.color()
        r = knowledge.rank()
        out = {
            "color": colors[card.color()] if visible else None,
            "rank": card.rank() + 1 if visible else None,
            "hints": {"color": colors[c] if c is not None else None, "rank": r + 1 if r is not None else None},
            "possible": {
                "colors": [colors[i] for i in range(len(colors)) if knowledge.color_plausible(i)],
                "ranks": [i + 1 for i in range(self.params.ranks) if knowledge.rank_plausible(i)],
            },
        }
        return out

    def _legal_json(self, obs, me: int) -> list[dict[str, Any]]:
        n = self.params.players
        out = []
        for m in obs.legal_moves():
            t = m.type()
            if t == _MT.PLAY or t == _MT.DISCARD:
                out.append({"type": _TYPE_NAME[t], "card_index": m.card_index()})
            elif t == _MT.REVEAL_COLOR:
                out.append({"type": "REVEAL_COLOR", "target": (me + m.target_offset()) % n, "color": self._colors[m.color()]})
            elif t == _MT.REVEAL_RANK:
                out.append({"type": "REVEAL_RANK", "target": (me + m.target_offset()) % n, "rank": m.rank() + 1})
        return out

    def observation(self, player: int) -> dict[str, Any]:
        n = self.params.players
        obs = self._state.observation(player)
        hands = obs.observed_hands()
        know = obs.card_knowledge()
        out_hands: dict[str, list] = {}
        my_hand: list = []
        for off in range(n):
            p = (player + off) % n
            cards = [self._card_json(c, k, visible=off != 0) for c, k in zip(hands[off], know[off])]
            if off == 0:
                my_hand = cards
            else:
                out_hands[str(p)] = cards
        last = []
        for item in obs.last_moves():
            mv = item.move()
            t = mv.type()
            if t == _MT.DEAL or t == _MT.INVALID:
                continue
            actor = (player + item.player()) % n
            rec: dict[str, Any] = {"player": actor, "type": _TYPE_NAME[t]}
            if t == _MT.PLAY or t == _MT.DISCARD:
                rec.update(card_index=mv.card_index(), color=self._colors[item.color()], rank=item.rank() + 1)
                if t == _MT.PLAY:
                    rec["scored"] = bool(item.scored())
            else:
                rec["target"] = (actor + mv.target_offset()) % n
                if t == _MT.REVEAL_COLOR:
                    rec["color"] = self._colors[mv.color()]
                else:
                    rec["rank"] = mv.rank() + 1
                rec["touched"] = item.card_info_revealed()
            last.append(rec)
        fw = obs.fireworks()
        return {
            "game": self.params.describe(),
            "me": player,
            "current_player": self._state.cur_player(),
            "turn": self.turn,
            "deck_size": obs.deck_size(),
            "info_tokens": obs.information_tokens(),
            "life_tokens": obs.life_tokens(),
            "fireworks": {self._colors[i]: fw[i] for i in range(len(self._colors))},
            "discards": [{"color": self._colors[c.color()], "rank": c.rank() + 1} for c in obs.discard_pile()],
            "hands": out_hands,
            "my_hand": my_hand,
            "last_moves": last,  # most recent first, back to (and including) this player's own last move
            "legal_moves": self._legal_json(obs, player) if player == self._state.cur_player() else [],
        }

    def legal_moves(self, player: int | None = None) -> list[dict[str, Any]]:
        p = self._state.cur_player() if player is None else player
        if p != self._state.cur_player():
            return []
        return self._legal_json(self._state.observation(p), p)

    def is_legal(self, move: dict[str, Any]) -> bool:
        try:
            k = move_key(move)
        except (IllegalMove, KeyError, TypeError, ValueError):
            return False
        return any(move_key(m) == k for m in self.legal_moves())

    # ----------------------------------------------------------------- moves
    def _to_hle(self, move: dict[str, Any]):
        n = self.params.players
        cur = self._state.cur_player()
        t = move["type"]
        if t == "PLAY":
            return pyhanabi.HanabiMove.get_play_move(int(move["card_index"]))
        if t == "DISCARD":
            return pyhanabi.HanabiMove.get_discard_move(int(move["card_index"]))
        if t == "REVEAL_COLOR":
            return pyhanabi.HanabiMove.get_reveal_color_move((int(move["target"]) - cur) % n, self._colors.index(move["color"]))
        if t == "REVEAL_RANK":
            return pyhanabi.HanabiMove.get_reveal_rank_move((int(move["target"]) - cur) % n, int(move["rank"]) - 1)
        raise IllegalMove(f"unknown move type {t!r}")

    def step(self, move: dict[str, Any], legal_moves: list[dict[str, Any]] | None = None) -> None:
        """Apply a move. `legal_moves` may be passed when the caller already holds the current list (saves a rebuild)."""
        if self._state.is_terminal():
            raise IllegalMove("game is over")
        try:
            k = move_key(move)
        except (KeyError, TypeError, ValueError) as e:
            raise IllegalMove(f"malformed move {move!r}") from e
        legal = self.legal_moves() if legal_moves is None else legal_moves
        if not any(move_key(m) == k for m in legal):
            raise IllegalMove(f"illegal move {move!r}")
        cur = self._state.cur_player()
        t = move["type"]
        if t in ("PLAY", "DISCARD"):
            order = self._slot_order[cur].pop(int(move["card_index"]))
            self.actions.append({"type": 0 if t == "PLAY" else 1, "target": order})
        elif t == "REVEAL_COLOR":
            self.actions.append({"type": 2, "target": int(move["target"]), "value": self._colors.index(move["color"])})
        else:
            self.actions.append({"type": 3, "target": int(move["target"]), "value": int(move["rank"])})
        self._state.apply_move(self._to_hle(move))
        self.turn += 1
        self._deal()


class HanabiGame:
    """`Game` implementation for Hanabi."""

    name = "hanabi"

    def __init__(self, params: HanabiParams | None = None):
        self.params = params or HanabiParams()
        self.num_players = self.params.players

    def new_game(self, seed: int) -> HanabiMatch:
        return HanabiMatch(self.params, seed)

    def describe(self) -> dict[str, Any]:
        return self.params.describe()

    @staticmethod
    def fallback_move(legal_moves: list[dict[str, Any]]) -> dict[str, Any]:
        """Spec section 4: discard oldest if legal, else the first legal move."""
        for m in legal_moves:
            if m["type"] == "DISCARD" and m["card_index"] == 0:
                return m
        return legal_moves[0]

    def digest(self) -> str:
        return hashlib.sha256(json.dumps({"game": self.name, **asdict(self.params)}, sort_keys=True).encode()).hexdigest()[:16]
