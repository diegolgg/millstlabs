"""OpenAI-compatible chat backend for local open-weight servers (spec section 6; fix round 1, step 9).

Primary target: `mlx_lm.server` on this machine, started as

    mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --host 127.0.0.1 --port 8080 --max-tokens 8192 \\
        --chat-template-args '{"enable_thinking": false}'

which was measured bit-reproducible for seeded, temperature-0 requests. Thinking is disabled at server start, so the
request carries no thinking switch. The body is plain OpenAI chat-completions: `model`, `messages`, `max_tokens`,
`temperature`, `seed`, `stream: false`, plus `extra_body` merged in verbatim (the hook for server-specific fields, e.g.
`{"chat_template_kwargs": {"enable_thinking": false}}` or Ollama's `{"options": {"num_ctx": 16384}}`). Ollama's /v1
route works with the same code but was measured non-reproducible for the MLX-runner model, so it is not relied on.

The sampling parameters (seed, temperature, extra_body) are part of the cache key (`key_extra`), so changing the seed
never serves a response sampled under another seed. Usage is taken from the response; local calls cost $0.
Transport is stdlib urllib (no extra dependency; the request body is exactly what is written here).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .backend import Request, Response, Usage
from .spend import usage_cost_usd

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


class BackendError(RuntimeError):
    pass


@dataclass
class OpenAICompatConfig:
    base_url: str = DEFAULT_BASE_URL
    seed: int = 0
    temperature: float = 0.0
    request_timeout_s: float = 1200.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_key: str = "local"  # local servers ignore it; never a real key
    # hosted provider (see run/config.py LLMConfig): the key is read from os.environ[api_key_env] at call time and is
    # never stored on this object, logged, cached or hashed
    api_key_env: str = ""
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    max_usd: float = 0.0
    spend_file: str = ""  # resolved path of the shared spend database (required when api_key_env is set)
    retry_uncertain: bool = False
    max_rate_limit_retries: int = 8
    concurrency: int = 1

    @property
    def hosted(self) -> bool:
        return bool(self.api_key_env)


class OpenAICompatBackend:
    name = "openai_compat"

    def __init__(self, config: OpenAICompatConfig | None = None):
        self.config = config or OpenAICompatConfig()
        self.last_seconds = 0.0

    # ------------------------------------------------------------------ cache key
    def key_extra(self) -> dict[str, Any]:
        """Sampling parameters that change the response: folded into the cache key."""
        c = self.config
        return {"seed": c.seed, "temperature": c.temperature, "extra_body": c.extra_body}

    # ------------------------------------------------------------------ request
    def body(self, req: Request) -> dict[str, Any]:
        system = "\n\n".join(b.get("text", "") for b in req.system)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
             else "".join(x.get("text", "") for x in m["content"])} for m in req.messages]
        body = {"model": req.model, "messages": messages, "max_tokens": req.max_tokens,
                "temperature": self.config.temperature, "seed": self.config.seed, "stream": False}
        body.update(self.config.extra_body)
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(body).encode()
        hreq = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"})
        try:
            with urllib.request.urlopen(hreq, timeout=self.config.request_timeout_s) as r:  # noqa: S310 - local URL
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise BackendError(f"{url}: HTTP {e.code}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BackendError(f"{url}: {e}") from e

    def complete(self, req: Request) -> Response:
        t0 = time.perf_counter()
        out = self._post(self.body(req))
        self.last_seconds = time.perf_counter() - t0
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"malformed response: {str(out)[:500]}") from e
        text = _THINK_RE.sub("", text)  # in case a server ignores the no-thinking switch
        u = out.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        usage = Usage(input_tokens=max(int(u.get("prompt_tokens", 0)) - int(cached), 0),
                      output_tokens=int(u.get("completion_tokens", 0)), cache_read_input_tokens=int(cached))
        if not self.config.hosted:
            return Response(text=text, usage=usage, cost_usd=0.0, model=req.model, backend=self.name)
        return Response(text=text, usage=usage, cost_usd=self.price(usage), model=req.model, backend=self.name,
                        provider_request_id=_opt_str(out.get("id")),
                        system_fingerprint=_opt_str(out.get("system_fingerprint")))

    def price(self, usage: Usage) -> float:
        """Dollar cost of a hosted call from the configured per-million-token prices. Provider-cached prompt tokens are
        charged at the full input rate (never under-counts)."""
        c = self.config
        if c.price_in_per_mtok is None or c.price_out_per_mtok is None:
            raise BackendError("hosted call without llm.price_in_per_mtok / llm.price_out_per_mtok")
        return usage_cost_usd(usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens,
                              usage.output_tokens, c.price_in_per_mtok, c.price_out_per_mtok)

    def complete_batch(self, reqs: list[Request]) -> list[Response]:
        return [self.complete(r) for r in reqs]


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)
