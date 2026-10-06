"""hanab.live replay export (format 3.0.0) and an independent reloader.

Paste the exported JSON into hanab.live "Watch Specific Replay -> JSON Data" to watch a game.
Format facts (Hanabi-Live/hanabi-live misc/example_game_with_comments.jsonc): `deck` top to bottom, dealt to player 0
until full, then player 1; actions `type` 0 play / 1 discard (target = deal-order card index), 2 color clue / 3 rank
clue (target = player index, value = suit index or rank), 4 end game. "No Variant" suits are Red, Yellow, Green, Blue,
Purple, so HLE's colors map R->0, Y->1, G->2, B->3, W->4 (HLE's White plays the role of Purple).

The reloader is a separate pure-Python rules implementation (not HLE), so a round trip is a genuine second parse.
"""

from __future__ import annotations

from typing import Any

HLE_TO_SUIT = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3}  # HLE color index (R Y G W B) -> No Variant suit index
COPIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}


def export(deck: list, hl_actions: list, players: list[str] | None = None, num_colors: int = 5,
           num_ranks: int = 5) -> dict[str, Any]:
    """Build hanab.live JSON from a recorded game (`GameResult.deck` / `.hl_actions` with record=True).

    Undealt cards are appended in a canonical order; they are never drawn, so the replay is unaffected."""
    if num_colors != 5 or num_ranks != 5:
        raise ValueError("only No Variant (5 colors, 5 ranks) maps onto a hanab.live variant name here")
    players = players or ["Alice", "Bob"]
    dealt = [{"suitIndex": HLE_TO_SUIT[c], "rank": r + 1} for c, r in deck]
    remaining = {(s, r): COPIES[r] for s in range(5) for r in range(1, 6)}
    for card in dealt:
        remaining[(card["suitIndex"], card["rank"])] -= 1
    rest = [{"suitIndex": s, "rank": r} for (s, r), k in sorted(remaining.items()) for _ in range(k)]
    actions = []
    for a in hl_actions:
        if a["type"] == 2:
            actions.append({"type": 2, "target": a["target"], "value": HLE_TO_SUIT[a["value"]]})
        else:
            actions.append(dict(a))
    return {"players": players, "deck": dealt + rest, "actions": actions, "options": {"variant": "No Variant"}}


def replay(game_json: dict[str, Any], hand_size: int = 5, clues: int = 8, strikes: int = 3) -> dict[str, Any]:
    """Re-simulate a hanab.live JSON game with plain rules. Returns final score and state summary.

    Score follows HLE's convention (0 after the third strike), which is also hanab.live's strikeout score."""
    n = len(game_json["players"])
    deck = game_json["deck"]
    hands: list[list[int]] = [[] for _ in range(n)]
    nxt = 0
    for p in range(n):
        for _ in range(hand_size):
            hands[p].append(nxt)
            nxt += 1
    stacks = [0] * 5
    tokens, lives, turn, turns_left = clues, strikes, 0, None
    cur = 0
    for a in game_json["actions"]:
        t = a["type"]
        deck_was_empty = nxt >= len(deck)
        if t == 4:
            break
        if t in (0, 1):
            card_id = a["target"]
            if card_id not in hands[cur]:
                raise ValueError(f"turn {turn}: player {cur} does not hold card {card_id}")
            hands[cur].remove(card_id)
            card = deck[card_id]
            if t == 0:
                if stacks[card["suitIndex"]] + 1 == card["rank"]:
                    stacks[card["suitIndex"]] += 1
                    if card["rank"] == 5 and tokens < clues:
                        tokens += 1
                else:
                    lives -= 1
            else:
                if tokens >= clues:
                    raise ValueError(f"turn {turn}: discard at max clues")
                tokens += 1
            if nxt < len(deck):
                hands[cur].append(nxt)
                nxt += 1
                if nxt == len(deck):
                    turns_left = n  # after the last draw every player, the drawer included, gets one more turn
        else:
            if tokens <= 0:
                raise ValueError(f"turn {turn}: clue with no tokens")
            target = a["target"]
            if target == cur:
                raise ValueError(f"turn {turn}: self clue")
            if t == 2:
                touched = [c for c in hands[target] if deck[c]["suitIndex"] == a["value"]]
            else:
                touched = [c for c in hands[target] if deck[c]["rank"] == a["value"]]
            if not touched:
                raise ValueError(f"turn {turn}: empty clue")
            tokens -= 1
        turn += 1
        if lives <= 0 or sum(stacks) == 25:
            break
        if deck_was_empty:
            turns_left -= 1
            if turns_left == 0:
                break
        cur = (cur + 1) % n
    score = 0 if lives <= 0 else sum(stacks)
    return {"score": score, "stacks": stacks, "lives": lives, "clues": tokens, "turns": turn}
