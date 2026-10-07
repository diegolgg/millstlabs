"""A1 transmission-fidelity parameters (files/prereg/A1-transmission-fidelity.md sections 2 and 3; Overnight 2, step 5).

One student with a weak incumbent S0 (the C1 student, self-play about 3) receives one message from a teacher whose
artifact is T = Piers, in one medium: `code` (bot.py only), `prose` (conventions.md only; the student must reimplement)
or `both`. The cover note is content-free and identical across media, and the harness's evidence for T is attached as in
any message. Verification is off and nothing is adopted (adoption `never`): the student's copy S' is whatever its one
revision call writes from S0 with the message in view. A revision that fails (unparseable or inadmissible after the
repair call) leaves the student with S0, so S' = S0 for that replicate (counted and reported).

Replicate r: experiment seed base + r (so the feedback traces in the prompt differ) and LLM sampling seed r, as in C1;
at temperature 0 a sampling seed alone does not change the output (amendment A1-1). Every copy is measured on one
common set of evaluation deals (`eval_experiment_seed`): V(S', S') self-play, V(S', T) cross-play with T, and v_T, v_S0.

Outputs in <out>/: rows.jsonl (one row per (replicate, medium), resumable), summary.json.

    python -m culture.run.transmission --spec configs/a1_params_mlx.yaml --out runs/a1-params
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..agents import agent as A
from ..artifacts.schema import Artifact, TeachingMessage
from ..bots.anchors import anchor_conventions, anchor_source, conventions_for, parse_rules, rulebot_source
from ..evaluate import stats as S
from ..game.seeds import generation_seeds
from ..org.base import AgentState
from .config import from_dict
from .context import RunContext
from .single_student import STUDENT_RULES

MEDIA = ("code", "prose", "both")
COVER = "I am sending you my strategy below. Use whatever helps you."
EULER_GAMMA = 0.5772156649015329
DEFAULTS: dict[str, Any] = {
    "name": "a1", "teacher": "piers", "media": list(MEDIA), "replicates": list(range(30)), "generation": 1,
    "eval_experiment_seed": None, "eval_games": None, "N": 4, "n_boot": 2000, "base": {},
    "org": {"verification": "none", "adoption": "never"},
}


def load_spec(spec: str | Path | dict) -> dict[str, Any]:
    d = spec if isinstance(spec, dict) else yaml.safe_load(Path(spec).read_text())
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy(d))
    for m in out["media"]:
        if m not in MEDIA:
            raise ValueError(f"unknown medium {m!r}")
    return out


def payload_text(medium: str, code: str, conventions: str) -> str:
    parts = []
    if medium in ("prose", "both"):
        parts.append("The sender's conventions.md:\n" + conventions.strip())
    if medium in ("code", "both"):
        parts.append("The sender's bot.py:\n```python\n" + code.rstrip() + "\n```")
    return "\n\n".join(parts)


def replicate_config(spec: dict[str, Any], r: int, cache_dir: Path):
    cfg = from_dict(copy.deepcopy(spec["base"]))
    llm = {"seed": r}
    if cfg.llm.cache_mode != "off" and not cfg.llm.cache_dir:
        llm["cache_dir"] = str(cache_dir)
    return from_dict({"name": spec["name"], "condition": "a1", "experiment_seed": cfg.experiment_seed + r,
                      "population": {"groups": 1, "agents_per_group": 1, "seed": r, "warm_start": "random"},
                      "org": {"routing": "none", **spec["org"]}, "corpus": {"enabled": False}, "llm": llm}, cfg)


def one_copy(ctx: RunContext, spec: dict[str, Any], medium: str, eval_seeds: list[int]) -> dict[str, Any]:
    g = int(spec["generation"])
    s0 = Artifact.make(rulebot_source(STUDENT_RULES), conventions_for(STUDENT_RULES), author="student", group="g0",
                       generation=0, origin="seed")
    ctx.add_artifact(s0)
    t = Artifact.make(anchor_source(spec["teacher"]), anchor_conventions(spec["teacher"]), author="T1",
                      group="teachers", generation=0, origin="seed")
    ctx.add_artifact(t)
    ctx.evaluate_many(g - 1, [(t.id, []), (s0.id, [])])
    student = AgentState("g0a0", "g0", incumbent=s0.id)
    ctx.agents = {"g0a0": student}
    ctx.start_budgets(None)
    n_rows = len(ctx.ledger.rows)
    m = TeachingMessage(id=f"g{g - 1}:T1>g0a0", sender="T1", receiver="g0a0", generation=g - 1, artifact_id=t.id,
                        delta_text=COVER, evidence=ctx.evals[t.id], delivered=True, touched_at=g,
                        sender_group="teachers", receiver_group="g0")
    text = A.ingest_text(m, False) + "\n" + payload_text(medium, t.code, t.conventions)
    user = A.revise_prompt(ctx, student, s0, ctx.evals.get(s0.id), text, "", g)
    art = A.produce(ctx, student, "revise", user, g, s0, [s0.id], [m.id], "revise")
    if art is not None:
        ctx.add_artifact(art)
    copy_id = art.id if art is not None else s0.id
    v_ss, ill = ctx.selfplay(copy_id, eval_seeds)
    v_st = ctx.crossplay(copy_id, t.id, eval_seeds)
    v_t = ctx.selfplay(t.id, eval_seeds)[0]
    v_s0 = ctx.selfplay(s0.id, eval_seeds)[0]
    code = ctx.store.get(copy_id).code
    calls = [{"tag": r["tag"], "request_key": r["request_key"], "input_tokens": r["input_tokens"],
              "cache_read_input_tokens": r["cache_read_input_tokens"], "output_tokens": r["output_tokens"],
              "cached": r["cached"], "latency_s": r.get("latency_s")}
             for r in ctx.ledger.rows[n_rows:] if not r.get("refused")]
    rules = parse_rules(code)
    return {"medium": medium, "admissible": art is not None, "copy": copy_id, "V_SS": v_ss, "V_ST": v_st,
            "v_T": v_t, "v_S0": v_s0, "copy_illegal_rate": ill, "exact_code_copy": code == t.code,
            "same_rule_list_as_teacher": rules is not None and rules == parse_rules(t.code),
            "prompt_sha": hashlib.sha256(user.encode()).hexdigest()[:16], "calls": calls}


def run(spec: str | Path | dict, out: str | Path, progress: bool = False) -> dict[str, Any]:
    spec = load_spec(spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    path = out / "rows.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    done = {(r["replicate"], r["medium"]) for r in rows}
    base = from_dict(copy.deepcopy(spec["base"]))
    e_seed = spec["eval_experiment_seed"] if spec["eval_experiment_seed"] is not None else base.experiment_seed
    n_eval = int(spec["eval_games"] or base.evaluation.selfplay_games)
    eval_seeds = generation_seeds(int(e_seed), int(spec["generation"]), n_eval, "eval").seeds
    for r in spec["replicates"]:  # seed-major, so a partial run stays balanced across media
        todo = [mu for mu in spec["media"] if (r, mu) not in done]
        if not todo:
            continue
        cfg = replicate_config(spec, r, out / "llm_cache")
        for mu in todo:
            ctx = RunContext(cfg, None)
            try:
                t0 = time.perf_counter()
                row = {"replicate": r, **one_copy(ctx, spec, mu, eval_seeds),
                       "wall_seconds": round(time.perf_counter() - t0, 2)}
            finally:
                ctx.close()
            with open(path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
            if progress:
                print(f"[r{r} {mu}] V(S',S')={row['V_SS']:.2f} V(S',T)={row['V_ST']:.2f} admissible={row['admissible']} "
                      f"same_rules={row['same_rule_list_as_teacher']} ({row['wall_seconds']}s)", flush=True)
    summary = summarize(rows, spec)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _boot(x: np.ndarray, f, n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def parameters(v_ss: np.ndarray, v_st: np.ndarray, v_t: float, N: int, n_boot: int) -> dict[str, Any]:
    """alpha = v_T - E[V(S',S')]; beta = sd(V(S',S')) sqrt(6)/pi (Gumbel scale); rho = E[V(S',T)] / v_T;
    delta(N) = -alpha + beta (gamma + ln N). Percentile bootstrap over replicates (pairs resampled together)."""
    k = math.sqrt(6) / math.pi
    c = EULER_GAMMA + math.log(N)
    both = np.column_stack([v_ss, v_st])

    def a(z): return v_t - z[:, 0].mean()
    def b(z): return z[:, 0].std(ddof=1) * k if len(z) > 1 else 0.0
    def rho(z): return z[:, 1].mean() / v_t
    def d(z): return -a(z) + b(z) * c

    out = {}
    for name, f in (("alpha", a), ("beta", b), ("rho", rho), (f"delta_N{N}", d)):
        lo, hi = _boot(both, f, n_boot)
        out[name] = {"point": float(f(both)), "lo": lo, "hi": hi}
    dd = out[f"delta_N{N}"]
    out["predicted_sign_at_N"] = {"N": N, "sign": "+" if dd["point"] > 0 else "-" if dd["point"] < 0 else "0",
                                  "interval_excludes_0": dd["lo"] > 0 or dd["hi"] < 0}
    return out


def summarize(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    N, nb = int(spec.get("N", 4)), int(spec.get("n_boot", 2000))
    out: dict[str, Any] = {"N": N, "media": {}}
    alphas = {}
    for mu in spec["media"]:
        rs = sorted((r for r in rows if r["medium"] == mu), key=lambda r: r["replicate"])
        if not rs:
            continue
        v_t = float(np.mean([r["v_T"] for r in rs]))
        v_ss, v_st = np.array([r["V_SS"] for r in rs]), np.array([r["V_ST"] for r in rs])
        adm = np.array([r["admissible"] for r in rs])
        fresh = [c for r in rs for c in r["calls"] if not c["cached"]]
        res = {"n": len(rs), "v_T": v_t, "v_S0": float(np.mean([r["v_S0"] for r in rs])),
               "V_SS": {"mean": float(v_ss.mean()), "sd": float(v_ss.std(ddof=1)) if len(rs) > 1 else None,
                        "iqm": S.iqm(v_ss), "values": v_ss.tolist()},
               "V_ST": {"mean": float(v_st.mean()), "values": v_st.tolist()},
               "admissible": int(adm.sum()), "failed_revisions": int((~adm).sum()),
               "same_rule_list_as_teacher": int(sum(r["same_rule_list_as_teacher"] for r in rs)),
               "exact_code_copies": int(sum(r["exact_code_copy"] for r in rs)),
               "backend_calls": len(fresh), "backend_seconds": round(sum(c["latency_s"] or 0 for c in fresh), 1),
               "parameters": parameters(v_ss, v_st, v_t, N, nb) if len(rs) > 1 else None}
        if adm.sum() > 1 and (~adm).any():
            res["parameters_admissible_only"] = parameters(v_ss[adm], v_st[adm], v_t, N, nb)
        out["media"][mu] = res
        alphas[mu] = v_ss
    if "code" in alphas and "prose" in alphas:
        # alpha(prose) - alpha(code) = E[V_SS(code)] - E[V_SS(prose)]; independent replicates per medium, but the
        # same replicate seeds, so pairs are resampled together when both media have the same replicates
        rc = {r["replicate"]: r["V_SS"] for r in rows if r["medium"] == "code"}
        rp = {r["replicate"]: r["V_SS"] for r in rows if r["medium"] == "prose"}
        common = sorted(set(rc) & set(rp))
        diff = np.array([rc[r] - rp[r] for r in common])
        lo, hi = _boot(diff[:, None], lambda z: z[:, 0].mean(), nb) if len(diff) > 1 else (None, None)
        point = float(diff.mean()) if len(diff) else None
        out["medium_ordering"] = {"definition": "alpha(prose) - alpha(code), paired by replicate seed",
                                  "point": point, "lo": lo, "hi": hi, "n": len(diff),
                                  "rule": "claim 'code transmits with less loss than prose' if >= 1 point with the "
                                          "bootstrap interval excluding 0",
                                  "met": bool(point is not None and lo is not None and point >= 1.0
                                              and (lo > 0 or hi < 0))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    s = run(a.spec, a.out, progress=not a.quiet)
    print(json.dumps({mu: {k: v for k, v in m.items() if k in ("n", "failed_revisions", "parameters")}
                      for mu, m in s["media"].items()}, indent=1))


if __name__ == "__main__":
    main()
