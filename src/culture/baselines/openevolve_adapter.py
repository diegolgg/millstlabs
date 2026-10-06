"""OpenEvolve as an external baseline on the same Game (spec section 14 step 7, section 15).

- `evaluate(program_path) -> dict` is OpenEvolve's evaluator contract. It scores a bot.py with our sandbox and our
  protocol: self-play on a fixed shared seed set, cross-play with the anchors, and (out of band, via the
  CULTURE_OE_SNAPSHOT file) cross-play with a population snapshot, since every external loop scores one program in
  isolation. `combined_score` is the self-play mean.
- `StubOpenEvolveLLM` plugs our stub backend into OpenEvolve through its `init_client` hook, so the full OpenEvolve
  loop runs with zero API calls; every call goes through our cache and cost ledger (`matched calls` comparisons read
  the same ledger as our own runs). Only the stub is wired tonight.

Environment knobs (read by `evaluate`, which OpenEvolve may run in another process): CULTURE_OE_GAMES (default 50),
CULTURE_OE_SEED (experiment seed, default 0), CULTURE_OE_ANCHORS (comma list, default piers,iggi),
CULTURE_OE_SNAPSHOT (JSON file: [{"id": ..., "code": ...}, ...]).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from culture.bots.runner import BotSpec, SandboxLimits, check_source
from culture.evaluate.crossplay import crossplay
from culture.evaluate.pool import Evaluator
from culture.evaluate.selfplay import seat_rates, selfplay
from culture.game.hanabi import HanabiParams
from culture.game.seeds import generation_seeds

EVALUATOR_PATH = Path(__file__).resolve()


def evaluate(program_path: str) -> dict[str, Any]:
    code = Path(program_path).read_text()
    problems = check_source(code)
    if problems:
        return {"combined_score": 0.0, "valid": 0.0, "error": "; ".join(problems[:3])}
    n = int(os.environ.get("CULTURE_OE_GAMES", "50"))
    seeds = generation_seeds(int(os.environ.get("CULTURE_OE_SEED", "0")), 0, n, "eval").seeds
    ev = Evaluator(HanabiParams(), SandboxLimits(), workers=0)
    spec = BotSpec("oe:" + hashlib.sha256(code.encode()).hexdigest()[:16], code)
    res = selfplay(ev, spec, seeds)
    rates = seat_rates(res, spec.key)
    out: dict[str, Any] = {"combined_score": float(np.mean([r.score for r in res])), "valid": 1.0,
                           "illegal_rate": rates["illegal_rate"]}
    for name in [a for a in os.environ.get("CULTURE_OE_ANCHORS", "piers,iggi").split(",") if a]:
        out[f"anchor_{name}"] = float(np.mean([r.score for r in crossplay(ev, spec, BotSpec.anchor(name), seeds)]))
    snap = os.environ.get("CULTURE_OE_SNAPSHOT")
    if snap and Path(snap).exists():
        partners = json.loads(Path(snap).read_text())
        vals = [np.mean([r.score for r in crossplay(ev, spec, BotSpec(p["id"], p["code"]), seeds)]) for p in partners]
        out["crossplay_snapshot"] = float(np.mean(vals)) if vals else 0.0
    return out


# ------------------------------------------------------------------------------------------- stub LLM for OpenEvolve
_LEDGER_PATH_ENV = "CULTURE_OE_LEDGER"


def _backend():
    from culture.llm.backend import CostLedger
    from culture.llm.cache import CachedBackend
    from culture.llm.stub_backend import StubBackend, StubConfig

    ledger = CostLedger(os.environ.get(_LEDGER_PATH_ENV) or None)
    return CachedBackend(StubBackend(StubConfig(diff_rate=0.0, invalid_rate=0.0)), None, "off", ledger)


try:
    from openevolve.llm.base import LLMInterface
except ImportError:  # OpenEvolve is optional; the evaluator works without it
    LLMInterface = object


class StubOpenEvolveLLM(LLMInterface):
    def __init__(self, model_cfg=None):
        self.backend = _backend()
        self.calls = 0
        self.model = "claude-haiku-4-5"

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message: str, messages: list[dict[str, str]], **kwargs) -> str:
        from culture.llm.backend import Request

        self.calls += 1
        req = Request(system=[{"type": "text", "text": system_message or ""}],
                      messages=[{"role": "user", "content": messages[-1]["content"]}], model=self.model, tag="revise",
                      seed_tag=f"openevolve/{self.calls}",
                      meta={"run": "openevolve", "group": "openevolve", "agent": "openevolve", "generation": self.calls})
        text = self.backend.complete(req).text
        # OpenEvolve's full-rewrite parser takes the first fenced code block
        return text


def init_stub_client(model_cfg) -> StubOpenEvolveLLM:
    return StubOpenEvolveLLM(model_cfg)


def run_openevolve(initial_code: str, iterations: int, out_dir: str | Path, games: int = 30):
    """Run OpenEvolve on Hanabi with the stub LLM. Returns OpenEvolve's EvolutionResult."""
    from openevolve.api import run_evolution
    from openevolve.config import Config, LLMModelConfig

    os.environ["CULTURE_OE_GAMES"] = str(games)
    os.environ.setdefault(_LEDGER_PATH_ENV, str(Path(out_dir) / "ledger.jsonl"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.llm.models = [LLMModelConfig(name="stub", init_client=init_stub_client, weight=1.0)]
    cfg.llm.evaluator_models = cfg.llm.models
    cfg.diff_based_evolution = False
    cfg.max_iterations = iterations
    cfg.evaluator.cascade_evaluation = False
    cfg.evaluator.parallel_evaluations = 1
    cfg.evaluator.timeout = 300
    cfg.database.in_memory = True
    cfg.random_seed = 0
    return run_evolution(initial_code, str(EVALUATOR_PATH), config=cfg, iterations=iterations, output_dir=str(out_dir),
                         cleanup=False)
