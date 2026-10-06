"""Rule-list Hanabi bot written against the sandbox contract (observation JSON in, legal move out; whitelisted imports).

This file is itself a valid artifact `bot.py`. The anchors Piers, IGGI and Flawed are this file with the CONFIG block set
to Canaan et al.'s rule lists; it reproduces the vendored agents move for move (test_bots.py). The stub LLM backend
writes perturbed copies of this file by rewriting the CONFIG block only.
"""

import random

# === CONFIG ===
RULES = [["hail_mary", {}], ["play_safe_card", {}], ["play_probably_safe", {"threshold": 0.6, "require_extra_lives": True}], ["tell_anyone_useful_card", {}], ["tell_dispensable", {"min_information_tokens": 3}], ["osawa_discard", {}], ["discard_oldest_first", {}], ["tell_randomly", {}], ["discard_randomly", {}]]
# === END CONFIG ===

COPIES = [3, 2, 2, 2, 1]


class Bot:
    name = "rulebot"

    def reset(self, game, my_id, seed):
        self.game = game
        self.me = my_id
        self.n = game["num_players"]
        self.colors = game["colors"]
        self.max_info = game.get("max_information_tokens", 8)
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers
    def others(self, obs):
        return [(self.me + off) % self.n for off in range(1, self.n)]

    def max_fireworks(self, obs):
        counts = {}
        for d in obs["discards"]:
            key = (d["color"], d["rank"] - 1)
            counts[key] = counts.get(key, 0) + 1
        mx = {c: 5 for c in self.colors}
        for (c, r), k in counts.items():
            if k >= COPIES[r] and mx[c] >= r:
                mx[c] = r
        return mx

    def visible_counts(self, obs):
        seen = {}
        for p in self.others(obs):
            for card in obs["hands"][str(p)]:
                key = (card["color"], card["rank"] - 1)
                seen[key] = seen.get(key, 0) + 1
        for d in obs["discards"]:
            key = (d["color"], d["rank"] - 1)
            seen[key] = seen.get(key, 0) + 1
        for c in obs["fireworks"]:
            for r in range(obs["fireworks"][c]):
                seen[(c, r)] = seen.get((c, r), 0) + 1
        return seen

    def plausible(self, card):
        out = []
        for c in self.colors:
            if c in card["possible"]["colors"]:
                for r in range(5):
                    if (r + 1) in card["possible"]["ranks"]:
                        out.append((c, r))
        return out

    def probability(self, obs, test):
        seen = self.visible_counts(obs)
        probs = []
        for card in obs["my_hand"]:
            total = 0
            good = 0
            for c, r in self.plausible(card):
                left = COPIES[r] - seen.get((c, r), 0)
                total += left
                if test(c, r):
                    good += left
            probs.append(good / total)
        return probs

    @staticmethod
    def argmax(values):
        best = 0
        for i in range(len(values)):
            if values[i] > values[best]:
                best = i
        return best

    # ------------------------------------------------------------------ rules (each returns a move or None)
    def play_safe_card(self, obs):
        fw = obs["fireworks"]
        for i, card in enumerate(obs["my_hand"]):
            if all(fw[c] == r for c, r in self.plausible(card)):
                return {"type": "PLAY", "card_index": i}
        return None

    def play_if_certain(self, obs):
        fw = obs["fireworks"]
        for i, card in enumerate(obs["my_hand"]):
            h = card["hints"]
            if h["color"] is not None and h["rank"] is not None and h["rank"] - 1 == fw[h["color"]]:
                return {"type": "PLAY", "card_index": i}
        return None

    def play_probably_safe(self, obs, threshold=0.95, require_extra_lives=False):
        fw = obs["fireworks"]
        p = self.probability(obs, lambda c, r: fw[c] == r)
        i = self.argmax(p)
        if not require_extra_lives or obs["life_tokens"] > 1:
            if p[i] >= threshold:
                return {"type": "PLAY", "card_index": i}
        return None

    def hail_mary(self, obs):
        if obs["deck_size"] == 0 and obs["life_tokens"] > 1:
            return self.play_probably_safe(obs, 0.0)
        return None

    def tell_anyone_useful_card(self, obs):
        fw = obs["fireworks"]
        if obs["info_tokens"] > 0:
            for p in self.others(obs):
                for card in obs["hands"][str(p)]:
                    playable = card["rank"] - 1 == fw[card["color"]]
                    if playable and card["hints"]["rank"] is None:
                        return {"type": "REVEAL_RANK", "target": p, "rank": card["rank"]}
                    if playable and card["hints"]["color"] is None:
                        return {"type": "REVEAL_COLOR", "target": p, "color": card["color"]}
        return None

    def tell_playable_card_outer(self, obs):
        return self.tell_anyone_useful_card(obs)

    def tell_dispensable(self, obs, min_information_tokens=8):
        fw = obs["fireworks"]
        if obs["info_tokens"] < min_information_tokens and obs["info_tokens"] > 0:
            low = min(fw.values())
            for p in self.others(obs):
                for card in obs["hands"][str(p)]:
                    c, r = card["color"], card["rank"] - 1
                    kc, kr = card["hints"]["color"], card["hints"]["rank"]
                    if kc is None and fw[c] == 5:
                        return {"type": "REVEAL_COLOR", "target": p, "color": c}
                    if kr is None and r < low:
                        return {"type": "REVEAL_RANK", "target": p, "rank": r + 1}
                    if r < fw[c]:
                        if kc is None and kr is not None:
                            return {"type": "REVEAL_COLOR", "target": p, "color": c}
                        if kc is not None and kr is None:
                            return {"type": "REVEAL_RANK", "target": p, "rank": r + 1}
        return None

    def osawa_discard(self, obs):
        if obs["info_tokens"] == self.max_info:
            return None
        fw = obs["fireworks"]
        mx = self.max_fireworks(obs)
        low = min(fw.values())
        for i, card in enumerate(obs["my_hand"]):
            c = card["hints"]["color"]
            r = None if card["hints"]["rank"] is None else card["hints"]["rank"] - 1
            if c is not None and fw[c] == 5:
                return {"type": "DISCARD", "card_index": i}
            if c is not None and r is not None and (r < fw[c] or r >= mx[c]):
                return {"type": "DISCARD", "card_index": i}
            if r is not None and r < low:
                return {"type": "DISCARD", "card_index": i}
        for i, card in enumerate(obs["my_hand"]):
            if not any(r < mx[c] for c, r in self.plausible(card)):
                return {"type": "DISCARD", "card_index": i}
        return None

    def discard_oldest_first(self, obs):
        if obs["info_tokens"] < self.max_info:
            return {"type": "DISCARD", "card_index": 0}
        return None

    def discard_probably_useless(self, obs, threshold=0.75):
        if obs["info_tokens"] < self.max_info:
            fw = obs["fireworks"]
            mx = self.max_fireworks(obs)
            p = self.probability(obs, lambda c, r: r < fw[c] or r >= mx[c])
            i = self.argmax(p)
            if p[i] >= threshold:
                return {"type": "DISCARD", "card_index": i}
        return None

    def tell_randomly(self, obs):
        if obs["info_tokens"] > 0:
            p = (self.me + 1) % self.n
            card = self.rng.choice(obs["hands"][str(p)])
            if self.rng.randint(0, 1) == 0:
                return {"type": "REVEAL_RANK", "target": p, "rank": card["rank"]}
            return {"type": "REVEAL_COLOR", "target": p, "color": card["color"]}
        return None

    def tell_unknown(self, obs):
        if obs["info_tokens"] > 0:
            p = (self.me + 1) % self.n
            for card in obs["hands"][str(p)]:
                if card["hints"]["color"] is None:
                    return {"type": "REVEAL_COLOR", "target": p, "color": card["color"]}
                if card["hints"]["rank"] is None:
                    return {"type": "REVEAL_RANK", "target": p, "rank": card["rank"]}
        return None

    def discard_randomly(self, obs):
        if obs["info_tokens"] < self.max_info:
            return {"type": "DISCARD", "card_index": self.rng.randint(0, len(obs["my_hand"]) - 1)}
        return None

    def legal_random(self, obs):
        return self.rng.choice(obs["legal_moves"])

    # ------------------------------------------------------------------ policy
    def act(self, obs):
        for rule, params in RULES:
            move = getattr(self, rule)(obs, **params)
            if move is not None:
                return move
        return self.legal_random(obs)
