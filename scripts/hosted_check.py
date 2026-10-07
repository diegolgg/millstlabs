"""Tiny, capped smoke check of a hosted OpenAI-compatible provider through the real backend and spend guard.

    PYTHONPATH=src .venv/bin/python scripts/hosted_check.py \\
        --base-url https://api.deepinfra.com/v1/openai --model deepseek-ai/DeepSeek-V4.1-Flash \\
        --api-key-env DEEPINFRA_API_KEY --price-in 0.10 --price-out 0.40 --max-usd 0.05 --n 1

Makes `n` chat completions of a ~20-token prompt with max_tokens 32, then repeats the first prompt once, and prints
usage, cost, latency, provider request id, system fingerprint, the guard's totals, and whether the repeat returned
the identical text (is the provider deterministic at temperature 0 with a seed?). There is no cache: every call is
sent. Refuses to run without a positive --max-usd. The spend file is cumulative: calls made by earlier checks against
the same file count toward the cap. The key is read from the named environment variable and is never printed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from culture.llm.backend import Request
from culture.llm.openai_compat import OpenAICompatBackend, OpenAICompatConfig
from culture.llm.spend import estimate_usd

PROMPT = "Reply with one short sentence naming a prime number between {lo} and {hi}."  # about 20 tokens with framing
MAX_TOKENS = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", required=True, help="NAME of the environment variable holding the key")
    ap.add_argument("--price-in", type=float, required=True, help="$ per million input tokens")
    ap.add_argument("--price-out", type=float, required=True, help="$ per million output tokens")
    ap.add_argument("--max-usd", type=float, default=None, help="cumulative cap for the spend file (required, > 0)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--spend-file", default="runs/hosted_check/spend.sqlite")
    ap.add_argument("--timeout", type=float, default=120.0)
    return ap.parse_args(argv)


def _req(model: str, i: int, tag: str) -> Request:
    return Request(system=[], messages=[{"role": "user", "content": PROMPT.format(lo=10 * i + 10, hi=10 * i + 30)}],
                   model=model, tag="hosted_check", seed_tag=f"hosted_check/{i}", max_tokens=MAX_TOKENS,
                   meta={"run": f"hosted_check/{tag}/{time.time_ns()}"})


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.max_usd is None or not a.max_usd > 0:
        print("refusing to run: --max-usd must be given and > 0 (no paid call without an explicit cap)", file=sys.stderr)
        return 2
    if a.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    Path(a.spend_file).parent.mkdir(parents=True, exist_ok=True)
    be = OpenAICompatBackend(OpenAICompatConfig(
        base_url=a.base_url, seed=a.seed, temperature=a.temperature, request_timeout_s=a.timeout,
        api_key_env=a.api_key_env, price_in_per_mtok=a.price_in, price_out_per_mtok=a.price_out, max_usd=a.max_usd,
        spend_file=a.spend_file, max_rate_limit_retries=2))
    est = estimate_usd(_req(a.model, 0, "x").prompt_text(), MAX_TOKENS, a.price_in, a.price_out)
    print(f"provider {a.base_url}  model {a.model}  key from ${a.api_key_env}  cap ${a.max_usd:.4f}  "
          f"spend file {a.spend_file}  estimate per call ${est:.6f}")
    print(f"guard before: {be.guard.totals()}")
    texts = []
    for i in list(range(a.n)) + [0]:
        repeat = len(texts) == a.n
        r = be.complete(_req(a.model, i, "repeat" if repeat else "first"))
        texts.append(r.text)
        print(f"{'repeat of 0' if repeat else f'call {i}'}: in {r.usage.input_tokens}+{r.usage.cache_read_input_tokens}"
              f" cached, out {r.usage.output_tokens}, cost ${r.cost_usd:.8f}, latency {be.last_seconds:.2f}s, "
              f"request id {r.provider_request_id}, fingerprint {r.system_fingerprint}")
        print(f"  text: {r.text!r}")
    print(f"deterministic at temperature {a.temperature} with seed {a.seed}: identical texts = {texts[0] == texts[-1]}")
    print(f"guard after: {be.guard.totals()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
