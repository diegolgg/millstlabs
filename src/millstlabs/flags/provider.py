"""Paper prompts and a cached, conservatively budgeted OpenAI vision adapter."""
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from hashlib import sha256

import numpy as np

from .task import image_url

SYSTEM = """You must output only valid JSON. No extra keys, no markdown,
and no text outside the JSON object.
You are one player in a flag identification game.
Choose exactly one country from the allowed countries listed in the user message.
Follow the exact output schema given in the user message."""


class BudgetExhausted(RuntimeError):
    pass


def user_text(countries, memory, bandwidth):
    # Transcribed from Figure 11 of Pavlova & Tanaka, arXiv:2609.19124v1.
    schema = '{"country":"<one allowed country>"}' if bandwidth == 1 else \
        '{"country":"<one allowed country>","reason":"<one sentence>"}'
    return ("All players are identifying the same underlying flag.\n"
            "You always see the same private crop.\n"
            "Transcript memory shows messages you observed from previous interactions with other players.\n"
            f"Allowed countries: {json.dumps(countries)}\n"
            "Transcript memory (oldest -> newest):\n"
            + ("\n".join(f"- {m}" for m in memory) or "[]")
            + "\nOutput JSON exactly:\n" + schema)


def parse_report(text, countries, bandwidth):
    try:
        data = json.loads(text)
        expected = {"country"} if bandwidth == 1 else {"country", "reason"}
        if set(data) != expected or data["country"] not in countries:
            raise ValueError("schema")
        if bandwidth == 3 and not isinstance(data["reason"], str):
            raise ValueError("reason")
        return {"country": data["country"], "reason": data.get("reason", ""), "valid": True}
    except (ValueError, TypeError, KeyError):
        # No repair call or privileged answer substitution; invalid answers score incorrect.
        return {"country": "__invalid__", "reason": "Invalid model response", "valid": False}


class OpenAIProvider:
    """Serial requests. An uncertain failed call retains its full spend reservation.

    The local runner lock protects the database. The ceiling is cumulative across
    restarts, and must be increased explicitly to spend more in the same output.
    """
    def __init__(self, cfg, database, max_usd, transport=None, retry_uncertain=False):
        import httpx
        if not (cfg.model == "gpt-4o" or cfg.model.startswith("gpt-4o-2024-")):
            raise ValueError("This cost guard is priced for GPT-4o; configure and verify pricing before another model")
        self.cfg, self.max_usd = cfg, max_usd
        self.retry_uncertain = retry_uncertain
        self.client = httpx.Client(timeout=60, transport=transport)
        self.db = sqlite3.connect(database)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, cost REAL, response TEXT, state TEXT)")
        self.db.commit()
        self.requests = self.cache_hits = 0
        self.prompt_tokens = self.completion_tokens = 0

    @property
    def spent(self):
        return self.db.execute("SELECT COALESCE(SUM(cost),0) FROM calls").fetchone()[0]

    def complete(self, crop, memory, countries, bandwidth, seed):
        text = user_text(countries, memory, bandwidth)
        cfg = self.cfg
        body = {"model": cfg.model, "temperature": cfg.temperature, "top_p": cfg.top_p,
                "max_completion_tokens": cfg.max_completion_tokens, "seed": seed, "store": False,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": [{"type": "text", "text": text},
                              {"type": "image_url", "image_url": {"url": image_url(crop, cfg.render_scale),
                                                                    "detail": cfg.image_detail}}]}]}
        key = sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        saved = self.db.execute("SELECT response,state FROM calls WHERE key=?", (key,)).fetchone()
        if saved and saved[1] != "complete":
            if not self.retry_uncertain:
                raise RuntimeError("A prior request has an uncertain outcome. Explicit --retry-uncertain permits another call while retaining the old cost reservation. No automatic paid retry.")
            self.db.execute("UPDATE calls SET key=?,state='uncertain_reserved' WHERE key=?",
                            (key+f":uncertain:{time.time_ns()}", key))
            self.db.commit()
            saved = None
        if saved and saved[1] == "complete":
            response = json.loads(saved[0])
            self.cache_hits += 1
        else:
            key_value = os.environ.get("OPENAI_API_KEY")
            if not key_value:
                raise RuntimeError("Set OPENAI_API_KEY in your local terminal. No API request was sent.")
            # UTF-8 bytes upper-bound text tokens; spare 1,024 tokens cover chat framing.
            # Deliberately overestimate image tiles by using original dimensions.
            tiles = math.ceil(crop.shape[1]*cfg.render_scale/512)*math.ceil(crop.shape[0]*cfg.render_scale/512)
            reserve = ((len((SYSTEM+text).encode())+1024+85+170*tiles)*2.5
                       + cfg.max_completion_tokens*10)/1_000_000
            if self.spent+reserve > self.max_usd:
                raise BudgetExhausted(f"Cumulative cost guard ${self.max_usd:.2f} reached; checkpoint saved before new spending")
            self.db.execute("INSERT INTO calls VALUES (?,?,NULL,'pending')", (key, reserve))
            self.db.commit()
            self.requests += 1
            try:
                result = self.client.post("https://api.openai.com/v1/chat/completions", json=body,
                                          headers={"Authorization": "Bearer "+key_value})
                if result.status_code != 200:
                    # Authentication/rate-limit rejection has no generated completion. Other
                    # errors remain reserved because their billing status can be uncertain.
                    if result.status_code in {400, 401, 403, 404, 429}:
                        self.db.execute("DELETE FROM calls WHERE key=?", (key,))
                        self.db.commit()
                    raise RuntimeError(f"OpenAI returned HTTP {result.status_code}; no automatic retry. Check key, balance, model access or rate limit.")
                response = result.json()
                usage = response["usage"]
                cost = (usage["prompt_tokens"]*2.5+usage["completion_tokens"]*10)/1_000_000
                # Charge all input at the uncached rate: this never undercounts cached input.
                self.db.execute("UPDATE calls SET cost=?,response=?,state='complete' WHERE key=?",
                                (cost, json.dumps(response), key))
                self.db.commit()
            except Exception:
                self.db.commit()
                raise
        usage = response["usage"]
        self.prompt_tokens += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]
        parsed = parse_report(response["choices"][0]["message"].get("content") or "", countries, bandwidth)
        return {**parsed, "request_hash": key, "model": response.get("model"),
                "system_fingerprint": response.get("system_fingerprint"), "usage": usage}

    def close(self):
        self.client.close()
        self.db.close()


class MockProvider:
    """Explicit offline wiring test. Uses only supplied crop/memory and the public catalog.

    This is a synthetic classifier, NOT an LLM and NOT scientific evidence.
    """
    def __init__(self, images):
        self.images = images
        self.requests = self.cache_hits = self.prompt_tokens = self.completion_tokens = 0
        self.spent = 0.

    def complete(self, crop, memory, countries, bandwidth, seed):
        colors = {tuple(p) for p in crop.reshape(-1, 3)}
        for line in memory:
            for color in re.findall(r"Verified sensor fact:.*?RGB #([0-9a-f]{6})", line):
                colors.add(tuple(bytes.fromhex(color)))
        candidates = [c for c in countries if colors <= {tuple(p) for p in self.images[c].reshape(-1, 3)}]
        candidates = candidates or countries
        counts = Counter(line.split(" | ")[0].split(": ")[-1] for line in memory if not line.startswith("Verified"))
        rng = np.random.default_rng(seed)
        weights = np.asarray([1+counts[c] for c in candidates], dtype=float)
        choice = str(rng.choice(candidates, p=weights/weights.sum()))
        self.requests += 1
        return {"country": choice, "reason": "Synthetic color-compatibility test", "valid": True,
                "model": "mock-not-an-llm", "system_fingerprint": "mock-v1", "usage": {}}

    def close(self):
        pass
