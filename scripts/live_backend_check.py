"""Live check of the openai_compat backend against a local MLX server (zero cost; never a paid API).

One agent starts from the IGGI anchor (no authoring call) and makes one revise call through the full agent step:
prompt with failure traces, backend call, parse, sandbox admission (smoke game), one repair call if needed. Then the
revise request is rebuilt and sent twice more with the cache off, to check bit-reproducibility. Writes
docs/results/live-mlx-revise.json and the model's bot as live-mlx-revise-candidate.py.

Usage: python scripts/live_backend_check.py [--base-url http://127.0.0.1:8080/v1] [--model ...] [--max-tokens 8192]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.agents import agent as A  # noqa: E402
from culture.llm.backend import Request  # noqa: E402
from culture.evaluate.selfplay import first_error, seat_rates, selfplay  # noqa: E402
from culture.run.config import from_dict  # noqa: E402
from culture.run.context import RunContext  # noqa: E402
from culture.run.generation import run_generation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "live-mlx-revise.json"))
    a = ap.parse_args()
    try:
        with urllib.request.urlopen(a.base_url.rstrip("/") + "/models", timeout=5) as r:  # noqa: S310 - local
            models = [m["id"] for m in json.loads(r.read())["data"]]
    except Exception as e:  # noqa: BLE001
        print(f"server not reachable at {a.base_url}: {e}; live check skipped")
        return 2
    if a.model not in models:
        print(f"model {a.model} not served (have {models}); live check skipped")
        return 2
    cfg = from_dict({"name": "live_check", "population": {"groups": 1, "agents_per_group": 1, "warm_start": "iggi"},
                     "evaluation": {"selfplay_games": 40, "crossplay_games": 0, "anchor_games": 20,
                                    "between_group_games": 0, "ladder_every": 0, "workers": 0},
                     "llm": {"backend": "openai_compat", "model": a.model, "base_url": a.base_url,
                             "max_tokens": a.max_tokens, "cache_mode": "off", "seed": 0, "temperature": 0.0}})
    with tempfile.TemporaryDirectory() as d:
        ctx = RunContext(cfg, Path(d) / "run")
        try:
            run_generation(ctx, 0)  # warm start from the anchor: no LLM call
            t0 = time.perf_counter()
            rec = run_generation(ctx, 1)  # one revise through the full agent step (plus a repair if needed)
            gen_seconds = time.perf_counter() - t0
            rows = [r for r in ctx.ledger.rows if r["generation"] == 1 and not r.get("refused")]
            ag = rec["agents"]["g0a0"]
            cand = {}
            if ag["candidate"]:  # re-score on generation 1's held-out deals (evaluations of rejected candidates are pruned)
                res = selfplay(ctx.evaluator, ctx.spec(ag["candidate"]), ctx.seeds(1))
                cand = {"illegal_rate": seat_rates(res, ag["candidate"])["illegal_rate"],
                        "first_error": first_error(res, ag["candidate"]),
                        "lines": len(ctx.store.get(ag["candidate"]).code.splitlines())}
                code_path = Path(a.out).with_name(Path(a.out).stem + "-candidate.py")  # the model's bot, for inspection
                code_path.write_text(ctx.store.get(ag["candidate"]).code)
            revise = [r for r in rows if r["tag"] == "revise"][0]
            # reproducibility: send the identical first revise request again, cache off
            parent = ctx.store.get(rec["agents"]["g0a0"]["start"])
            user = A.revise_prompt(ctx, ctx.agents["g0a0"], parent, None, "", "", 1)
            req = Request(system=A.system_blocks(ctx), messages=[{"role": "user", "content": user}], model=a.model,
                          tag="revise", seed_tag="repro", max_tokens=a.max_tokens)
            inner = ctx.backend.inner
            r1 = inner.complete(req)
            s1 = inner.last_seconds
            r2 = inner.complete(req)
            out = {
                "date": time.strftime("%Y-%m-%d %H:%M"), "base_url": a.base_url, "model": a.model,
                "sampling": {"seed": 0, "temperature": 0.0, "max_tokens": a.max_tokens},
                "revise_call": {"input_tokens": revise["input_tokens"] + revise["cache_read_input_tokens"],
                                "output_tokens": revise["output_tokens"], "seconds": revise["latency_s"]},
                "calls_in_generation": [{"tag": r["tag"], "input_tokens": r["input_tokens"],
                                         "output_tokens": r["output_tokens"], "seconds": r["latency_s"]} for r in rows],
                "admissible": ag["candidate"] is not None,
                "repaired": ctx.agents["g0a0"].counters.get("repaired", 0),
                "failed_revisions": ctx.agents["g0a0"].counters.get("failed_revisions", 0),
                "candidate_selfplay_mean": ag["candidate_score"], "parent_selfplay_mean": ag["parent_score"],
                "candidate_illegal_rate": cand.get("illegal_rate"),
                "candidate_first_error": cand.get("first_error"),
                "candidate_lines": cand.get("lines"),
                "accepted": rec["accepted"].get("g0a0"),
                "generation_seconds": round(gen_seconds, 1),
                "reproducible": {"identical_text": r1.text == r2.text, "seconds_each": round(s1, 1),
                                 "output_tokens": r1.usage.output_tokens},
            }
        finally:
            ctx.close()
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
