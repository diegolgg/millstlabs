"""LLM layer interface: Request / Response, the Backend protocol, prices and the cost ledger (spec section 6).

Only the stub backend exists in this build. The Anthropic and OpenAI-compatible backends named in the spec are
deliberately not written yet (no key tonight, and untested SDK code is worse than none); they plug in behind the same
`Backend` protocol and the cache/ledger without touching anything else.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

# $ per million tokens (input, output). Cache reads bill at 0.1x input, cache writes at 1.25x; batch halves everything.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
}
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25
BATCH_MULT = 0.5


@dataclass
class Request:
    system: list[dict[str, Any]]  # frozen prefix first, with cache_control on it
    messages: list[dict[str, Any]]
    model: str
    tag: str  # author | revise | teach | ingest | merge | repair
    seed_tag: str  # run id + agent + generation
    max_tokens: int = 16000
    effort: str = "high"
    meta: dict[str, Any] = field(default_factory=dict)  # ledger keys (run, group, agent, generation); not hashed

    def key_material(self, backend: str) -> dict[str, Any]:
        return {"backend": backend, "model": self.model, "effort": self.effort, "system": self.system,
                "messages": self.messages, "max_tokens": self.max_tokens, "seed_tag": self.seed_tag}

    def prompt_text(self) -> str:
        parts = [b.get("text", "") for b in self.system]
        for m in self.messages:
            c = m["content"]
            parts.append(c if isinstance(c, str) else "".join(x.get("text", "") for x in c))
        return "\n".join(parts)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class Response:
    text: str
    usage: Usage
    cost_usd: float | None
    model: str
    backend: str
    cached: bool = False  # served from the record/replay cache (no new spend)
    request_key: str = ""
    batch: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Response":
        d = dict(d)
        d["usage"] = Usage(**d["usage"])
        return Response(**d)


class Backend(Protocol):
    name: str

    def complete(self, req: Request) -> Response: ...

    def complete_batch(self, reqs: list[Request]) -> list[Response]: ...


def cost_usd(model: str, usage: Usage, batch: bool = False) -> float | None:
    """Dollar cost from the price table; None for an unpriced model (e.g. GPT Luna until its price is confirmed)."""
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    c = (usage.input_tokens * pin + usage.cache_read_input_tokens * pin * CACHE_READ_MULT
         + usage.cache_creation_input_tokens * pin * CACHE_WRITE_MULT + usage.output_tokens * pout) / 1e6
    return c * (BATCH_MULT if batch else 1.0)


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class CostLedger:
    """Append-only JSONL: one row per LLM call with (run, group, agent, generation, tag) and usage.

    `cached` rows were served by the record/replay cache. `spend_usd` is what that call cost when it was first made, so a
    replayed or resumed run reports the same curve as the original; `new_spend_usd` is 0 for cache hits."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.rows: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            with open(self.path) as f:
                self.rows = [json.loads(line) for line in f if line.strip()]

    def record(self, req: Request, resp: Response, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "run": req.meta.get("run"), "group": req.meta.get("group"), "agent": req.meta.get("agent"),
            "generation": req.meta.get("generation"), "tag": req.tag, "model": resp.model, "backend": resp.backend,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": resp.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": resp.usage.cache_creation_input_tokens,
            "spend_usd": resp.cost_usd, "new_spend_usd": 0.0 if resp.cached else resp.cost_usd,
            "cached": resp.cached, "batch": resp.batch, "request_key": resp.request_key,
        }
        if extra:
            row.update(extra)
        self.rows.append(row)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def record_refusal(self, req: Request, reason: str, budget_left: float) -> dict[str, Any]:
        """A call the budget refused: logged with zero usage and the reason, so overspend attempts are auditable.
        `refused` rows carry no spend and are skipped by the usage totals."""
        row = {
            "run": req.meta.get("run"), "group": req.meta.get("group"), "agent": req.meta.get("agent"),
            "generation": req.meta.get("generation"), "tag": req.tag, "model": req.model, "backend": None,
            "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "spend_usd": 0.0, "new_spend_usd": 0.0, "cached": False, "batch": False, "request_key": "",
            "refused": True, "refusal_reason": reason, "budget_left": budget_left,
        }
        self.rows.append(row)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def totals(self, by: str | None = None) -> dict[str, Any]:
        def agg(all_rows):
            rows = [r for r in all_rows if not r.get("refused")]  # refusals are not calls and carry no usage
            return {
                "calls": len(rows),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "cache_read_input_tokens": sum(r["cache_read_input_tokens"] for r in rows),
                "cache_creation_input_tokens": sum(r["cache_creation_input_tokens"] for r in rows),
                "tokens": sum(r["input_tokens"] + r["output_tokens"] + r["cache_read_input_tokens"]
                              + r["cache_creation_input_tokens"] for r in rows),
                "spend_usd": sum(r["spend_usd"] or 0.0 for r in rows),
                "new_spend_usd": sum(r["new_spend_usd"] or 0.0 for r in rows),
            }

        if by is None:
            return agg(self.rows)
        groups: dict[Any, list] = {}
        for r in self.rows:
            groups.setdefault(r[by], []).append(r)
        return {k: agg(v) for k, v in groups.items()}

    def reload(self) -> None:
        self.__init__(self.path)
