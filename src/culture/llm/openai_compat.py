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

Hosted providers (DeepInfra, Fireworks, OpenRouter, any OpenAI-compatible URL): set `api_key_env` to the NAME of the
environment variable holding the key (read at call time, never stored, logged, cached or hashed), both per-million-token
prices and `max_usd`. Every call then reserves a conservative estimate in the shared spend database (llm/spend.py)
before it is sent and settles the priced usage after; see `_complete_hosted` for the failure semantics. With
`api_key_env` empty the local path above is unchanged: `Bearer local`, cost 0, no spend rows.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .backend import Request, Response, Usage
from .cache import request_key
from .spend import SpendGuard, estimate_usd, usage_cost_usd

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


class _Unbilled(Exception):
    """No completion was generated: HTTP 429, or the connection failed before any request reached the provider."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class _Rejected(Exception):
    """HTTP 400/401/403/404: the provider refused the request outright (no completion, not billed). Never retried."""

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


class _Uncertain(Exception):
    """The request may have been processed and billed (5xx, other 4xx, timeout or reset mid-request)."""


REJECTED_STATUSES = (400, 401, 403, 404)


class OpenAICompatBackend:
    name = "openai_compat"
    backoff_base_s = 1.0  # unbilled-retry backoff: base * 2**attempt, capped (tests shrink these)
    backoff_cap_s = 60.0

    def __init__(self, config: OpenAICompatConfig | None = None):
        self.config = config or OpenAICompatConfig()
        self.last_seconds = 0.0
        self.guard: SpendGuard | None = None
        if self.config.hosted:
            if not self.config.spend_file:
                raise BackendError("a hosted backend (api_key_env set) needs a spend file: set llm.spend_file or run "
                                   "with an output directory (default <run dir>/spend.sqlite)")
            self.guard = SpendGuard(self.config.spend_file, self.config.max_usd)

    # ------------------------------------------------------------------ cache key
    def key_extra(self) -> dict[str, Any]:
        """Sampling parameters that change the response: folded into the cache key."""
        c = self.config
        return {"seed": c.seed, "temperature": c.temperature, "extra_body": c.extra_body}

    def guard_key(self, req: Request) -> str:
        """Spend-guard key: the cache key of the request, qualified by the run (`meta.run`) so runs sharing one spend
        file never block each other's identical requests."""
        key = request_key(self.name, req, self.key_extra())
        run = req.meta.get("run")
        return f"{run}|{key}" if run else key

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
        if self.config.hosted:
            return self._complete_hosted(req)
        t0 = time.perf_counter()
        out = self._post(self.body(req))
        self.last_seconds = time.perf_counter() - t0
        return self._response(req, out)

    def _response(self, req: Request, out: dict[str, Any]) -> Response:
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"malformed response: {str(out)[:500]}") from e
        text = _THINK_RE.sub("", text)  # in case a server ignores the no-thinking switch
        usage = _usage(out)
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

    # ------------------------------------------------------------------ hosted (paid) path
    def _complete_hosted(self, req: Request) -> Response:
        """Reserve, send, settle. 400/401/403/404 release and raise (no retry); 429 and connection failures before any
        response release and retry with backoff (unbilled); anything else marks the reservation uncertain and raises,
        so the money stays reserved until a human looks."""
        c, guard = self.config, self.guard
        assert guard is not None
        key_value = os.environ.get(c.api_key_env, "")
        if not key_value:
            raise BackendError(f"llm.api_key_env names the environment variable {c.api_key_env!r}, which is not set. "
                               "No request was sent.")
        body = self.body(req)
        gkey = self.guard_key(req)
        est = estimate_usd(req.prompt_text(), req.max_tokens, c.price_in_per_mtok, c.price_out_per_mtok)
        retries = 0
        while True:
            guard.reserve(gkey, est, retry_uncertain=c.retry_uncertain)  # BudgetExhausted / UncertainOutcome
            t0 = time.perf_counter()
            try:
                raw = self._send_hosted(body, key_value)
            except _Unbilled as e:
                guard.release(gkey)
                if retries >= c.max_rate_limit_retries:
                    raise BackendError(f"{e} (unbilled; gave up after {retries} retries, "
                                       "llm.max_rate_limit_retries)") from None
                time.sleep(self._backoff(retries, e.retry_after))
                retries += 1
                continue
            except _Rejected as e:
                guard.release(gkey)
                raise BackendError(f"{e} (HTTP {e.status}: rejected, not billed, not retried; check the key, the "
                                   "model id, account access and the request)") from None
            except BaseException as e:
                guard.mark_uncertain(gkey)
                if isinstance(e, _Uncertain):
                    raise BackendError(f"{e} (outcome uncertain: the reservation is kept; set llm.retry_uncertain: "
                                       "true to re-send this request)") from None
                raise
            break
        self.last_seconds = time.perf_counter() - t0
        try:
            out = json.loads(raw.decode())
            u = out["usage"]
            int(u["prompt_tokens"]), int(u["completion_tokens"])
        except BaseException as e:
            guard.mark_uncertain(gkey)
            raise BackendError(f"malformed response body (no usage; outcome uncertain, the reservation is kept): "
                               f"{raw[:500]!r}") from e
        guard.settle(gkey, self.price(_usage(out)))
        return self._response(req, out)  # a 200 with usage but no content is settled, then raises as malformed

    def _send_hosted(self, body: dict[str, Any], key_value: str) -> bytes:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        hreq = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key_value}"})
        try:
            with urllib.request.urlopen(hreq, timeout=self.config.request_timeout_s) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300]
            except Exception:  # noqa: BLE001 - the body is diagnostic only
                detail = b""
            msg = f"{url}: HTTP {e.code}: {detail!r}"
            if e.code == 429:
                raise _Unbilled(msg, _retry_after(e.headers)) from None
            if e.code in REJECTED_STATUSES:
                raise _Rejected(e.code, msg) from None
            raise _Uncertain(msg) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (ConnectionRefusedError, socket.gaierror)):
                raise _Unbilled(f"{url}: connection failed before sending: {e.reason}") from None
            raise _Uncertain(f"{url}: {e.reason}") from None
        except ConnectionRefusedError as e:
            raise _Unbilled(f"{url}: connection refused: {e}") from None
        except (TimeoutError, OSError, http.client.HTTPException) as e:
            raise _Uncertain(f"{url}: {type(e).__name__}: {e}") from None

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        wait = min(self.backoff_cap_s, self.backoff_base_s * 2 ** attempt)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.backoff_cap_s))
        return wait

    def complete_batch(self, reqs: list[Request]) -> list[Response]:
        """In request order. With config.concurrency > 1 (hosted only) the requests run on a thread pool of that size,
        each thread its own urllib request; every call finishes before the first error, if any, is raised."""
        n = self.config.concurrency
        if n <= 1 or len(reqs) <= 1:
            return [self.complete(r) for r in reqs]
        with ThreadPoolExecutor(max_workers=min(n, len(reqs))) as ex:
            futs = [ex.submit(self.complete, r) for r in reqs]
        errors = [f.exception() for f in futs]
        first = next((e for e in errors if e is not None), None)
        if first is not None:
            raise first
        return [f.result() for f in futs]


def _usage(out: dict[str, Any]) -> Usage:
    u = out.get("usage") or {}
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return Usage(input_tokens=max(int(u.get("prompt_tokens", 0)) - int(cached), 0),
                 output_tokens=int(u.get("completion_tokens", 0)), cache_read_input_tokens=int(cached))


def _retry_after(headers: Any) -> float | None:
    try:
        v = headers.get("Retry-After") if headers is not None else None
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)
