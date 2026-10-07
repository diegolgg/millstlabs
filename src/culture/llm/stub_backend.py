"""Zero-cost backend for debugging and CI: canned artifacts built from the anchor bots with seeded perturbations.

Every response is a pure function of the request (seeded by the request hash), so record/replay and resume are exact.
Artifacts are `rulebot.py` with a perturbed rule list:
- author: start from a seeded choice of Canaan's Piers / IGGI / Flawed rule lists, apply a few mutations;
- revise: read the current rule list out of the prompt, apply one or two mutations (threshold nudges, rule
  insert/delete/swap); answered as a full rewrite or as a SEARCH/REPLACE diff of the CONFIG line;
- merge: crossover of the two rule lists in the prompt; teach: a prose delta; repair: a valid version.
A configurable fraction of artifact responses is deliberately broken (syntax error) to exercise the repair path.
Usage numbers are fake but plausible (about 4 characters per token, the system prefix billed as a prompt-cache write on
first sight and a read afterwards), so the cost ledger, cache accounting and dollar curves are exercised end to end.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from ..bots.anchors import CONFIG_RE, RULE_DOCS, RULE_LISTS, conventions_for, parse_rules, rulebot_source
from .backend import Request, Response, Usage, approx_tokens, cost_usd

PARAM_RULES = {
    "play_probably_safe": {"threshold": (0.0, 1.0), "require_extra_lives": (False, True)},
    "tell_dispensable": {"min_information_tokens": (1, 8)},
    "discard_probably_useless": {"threshold": (0.3, 1.0)},
}
RULE_POOL = sorted(RULE_DOCS)


@dataclass
class StubConfig:
    base_weights: dict[str, float] = field(default_factory=lambda: {"piers": 1.0, "iggi": 1.0, "flawed": 1.0})
    author_mutations: tuple[int, int] = (2, 4)
    revise_mutations: tuple[int, int] = (1, 2)
    diff_rate: float = 0.3
    invalid_rate: float = 0.03
    teach_words: int = 60
    # "hill": revisions mutate the rule list (selection then hill-climbs). "null": revisions return the parent's rule
    # list unchanged with a fresh RNG salt (pure noise, no expected change), for null-population calibration.
    mode: str = "hill"


def _rng_for(req: Request) -> random.Random:
    h = hashlib.sha256(json.dumps(req.key_material("stub"), sort_keys=True).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _fresh_params(name: str, rng: random.Random) -> dict:
    if name == "play_probably_safe":
        return {"threshold": round(rng.uniform(0.3, 0.95), 2), "require_extra_lives": rng.random() < 0.5}
    if name == "tell_dispensable":
        return {"min_information_tokens": rng.randint(1, 8)}
    if name == "discard_probably_useless":
        return {"threshold": round(rng.uniform(0.5, 1.0), 2)}
    return {}


def mutate(rules: list, rng: random.Random) -> tuple[list, str]:
    """One mutation of a rule list; returns (new list, description)."""
    rules = [[n, dict(p)] for n, p in rules]
    ops = ["nudge", "insert", "delete", "swap"]
    tunable = [i for i, (n, _) in enumerate(rules) if n in PARAM_RULES]
    for _ in range(10):
        op = rng.choice(ops)
        if op == "nudge" and tunable:
            i = rng.choice(tunable)
            name, params = rules[i]
            key = rng.choice(sorted(PARAM_RULES[name]))
            lo, hi = PARAM_RULES[name][key]
            if isinstance(lo, bool):
                params[key] = not params.get(key, False)
            elif isinstance(lo, int):
                params[key] = max(lo, min(hi, params.get(key, (lo + hi) // 2) + rng.choice([-1, 1])))
            else:
                params[key] = round(max(lo, min(hi, params.get(key, 0.95) + rng.uniform(-0.2, 0.2))), 2)
            return rules, f"changed {name}.{key} to {params[key]}"
        if op == "insert" and len(rules) < 14:
            name = rng.choice(RULE_POOL)
            pos = rng.randint(0, len(rules))
            rules.insert(pos, [name, _fresh_params(name, rng)])
            return rules, f"added rule {name} at position {pos + 1}"
        if op == "delete" and len(rules) > 2:
            i = rng.randrange(len(rules))
            name = rules.pop(i)[0]
            return rules, f"removed rule {name}"
        if op == "swap" and len(rules) > 1:
            i = rng.randrange(len(rules) - 1)
            rules[i], rules[i + 1] = rules[i + 1], rules[i]
            return rules, f"swapped {rules[i + 1][0]} and {rules[i][0]}"
    return rules, "no change"


def _rules_in(text: str) -> list[list]:
    out = []
    for m in CONFIG_RE.finditer(text):
        r = parse_rules(m.group(0))
        if r is not None:
            out.append(r)
    return out


def _config_line(rules: list) -> str:
    return "RULES = " + repr(rules)


class StubBackend:
    name = "stub"

    def __init__(self, config: StubConfig | None = None):
        self.config = config or StubConfig()
        self._warm: set[str] = set()

    # ------------------------------------------------------------------ provider-side state (prompt-cache warmth)
    def _prefix_hash(self, req: Request) -> str:
        return hashlib.sha256(json.dumps(req.system, sort_keys=True).encode()).hexdigest()[:16]

    def observe(self, req: Request) -> None:
        self._warm.add(self._prefix_hash(req))

    def state_dict(self) -> dict[str, Any]:
        return {"warm": sorted(self._warm)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._warm = set(state.get("warm", []))

    # ------------------------------------------------------------------ responses
    def _artifact_text(self, req: Request, rng: random.Random) -> str:
        cfg = self.config
        user = req.messages[-1]["content"]
        found = _rules_in(user)
        notes = []
        if req.tag == "author" or not found:
            names = sorted(cfg.base_weights)
            base = rng.choices(names, weights=[cfg.base_weights[n] for n in names])[0]
            rules = [[n, dict(p)] for n, p in RULE_LISTS[base]]
            lo, hi = cfg.author_mutations
            notes.append(f"started from a {base}-like rule list")
        elif req.tag == "merge" and len(found) >= 2:
            a, b = found[0], found[1]
            cut_a, cut_b = rng.randint(1, len(a)), rng.randint(0, len(b))
            rules = [list(x) for x in a[:cut_a]] + [list(x) for x in b[cut_b:]]
            lo, hi = 0, 1
            notes.append(f"kept the first {cut_a} rules of A and the last {len(b) - cut_b} of B")
        elif req.tag == "repair":
            rules = found[-1]
            lo, hi = 0, 0
            notes.append("fixed the syntax error")
        else:
            rules = found[0]
            lo, hi = cfg.revise_mutations
        noise = 0
        if cfg.mode == "null" and req.tag != "author" and found:
            rules, lo, hi = found[0], 0, 0  # the parent's (or receiver's own) rule list, untouched
            noise = rng.randint(1, 2**30)
            notes.append("kept the rule list; new random stream")
        for _ in range(rng.randint(lo, hi)):
            rules, what = mutate(rules, rng)
            notes.append(what)
        code = rulebot_source(rules, noise=noise)
        conventions = conventions_for(rules, title="Conventions")
        broken = req.tag != "repair" and rng.random() < cfg.invalid_rate
        if broken:
            code = code.replace("    def act(self, obs):", "    def act(self, obs)", 1)
        if req.tag == "revise" and found and not broken and not noise and rng.random() < cfg.diff_rate:
            diff = f"<<<<<<< SEARCH\n{_config_line(found[0])}\n=======\n{_config_line(rules)}\n>>>>>>> REPLACE"
            return (f"{'; '.join(notes)}.\n<bot_diff>\n{diff}\n</bot_diff>\n"
                    f"<conventions>\n{conventions}</conventions>\n")
        return f"{'; '.join(notes)}.\n<bot>\n```python\n{code}```\n</bot>\n<conventions>\n{conventions}</conventions>\n"

    def _teach_text(self, req: Request, rng: random.Random) -> str:
        found = _rules_in(req.messages[-1]["content"])
        names = [n for n, _ in found[0]] if found else []
        lead = ", then ".join(names[:4]) if names else "my rule order"
        return (f"<delta>\nMy bot tries {lead}. The ordering matters most: playing only cards that are certainly "
                f"playable and spending hints on playable cards keeps lives. Try adopting the first "
                f"{rng.randint(2, 4)} rules as written.\n</delta>\n")

    def complete(self, req: Request) -> Response:
        rng = _rng_for(req)
        if req.tag in ("author", "revise", "merge", "repair"):
            text = self._artifact_text(req, rng)
        elif req.tag == "teach":
            text = self._teach_text(req, rng)
        else:
            text = "<note>\nacknowledged\n</note>\n"
        sys_text = "".join(b.get("text", "") for b in req.system)
        sys_tokens = approx_tokens(sys_text)
        total_in = approx_tokens(req.prompt_text())
        h = self._prefix_hash(req)
        warm = h in self._warm
        self._warm.add(h)
        usage = Usage(
            input_tokens=max(1, total_in - sys_tokens),
            output_tokens=approx_tokens(text) + 200 * (req.effort == "high"),  # stand-in for thinking tokens
            cache_read_input_tokens=sys_tokens if warm else 0,
            cache_creation_input_tokens=0 if warm else sys_tokens,
        )
        return Response(text=text, usage=usage, cost_usd=cost_usd(req.model, usage), model=req.model, backend=self.name)

    def complete_batch(self, reqs: list[Request]) -> list[Response]:
        out = [self.complete(r) for r in reqs]
        for r, q in zip(out, reqs):
            r.batch = True
            r.cost_usd = cost_usd(q.model, r.usage, batch=True)
        return out
