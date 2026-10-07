"""Shared warm starts (fix round 1, step 5).

Generation 0 is authored once per population seed into a warm-start set, keyed by a hash of everything that can change
the authored artifacts, and every condition of an experiment forks from it. Conditions never call the backend for
generation 0, so with a real (or non-deterministic) backend there is no race between conditions authoring their own
warm starts in parallel, no duplicated spend, and the pairing (identical generation-0 incumbents) holds by
construction rather than by cache luck.

Layout: `<root>/p<population seed>-<key>/warm_start.json` plus that set's own ledger and LLM cache. The set is built in
a temporary directory and renamed into place, so a crash never leaves a half-built set behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..artifacts.schema import Artifact, artifact_id
from ..llm import prompts
from .config import ExperimentConfig, from_dict

SET_FILE = "warm_start.json"


class WarmStartMismatch(RuntimeError):
    pass


def warm_start_key(cfg: ExperimentConfig) -> str:
    """Hash of every input that can change the generation-0 artifacts. The condition and the organization policies
    are deliberately absent: they do not touch authoring."""
    llm = asdict(cfg.llm)
    llm.pop("cache_mode", None)
    llm.pop("cache_dir", None)
    material = {
        "name": cfg.name,  # the seed-tag prefix
        "experiment_seed": cfg.experiment_seed,
        "population": {k: v for k, v in asdict(cfg.population).items() if k != "warm_start_set"},
        "game": asdict(cfg.game),
        "sandbox": asdict(cfg.sandbox),
        "llm": llm,
        "prompts": prompts.prompt_hashes(),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def needs_authoring(cfg: ExperimentConfig) -> bool:
    return cfg.population.warm_start == "author"


def author_initial(ctx) -> dict[str, Artifact]:
    """The generation-0 artifact of every agent: authored by one LLM call (plus repair), or an anchor."""
    from ..agents import agent as A
    from ..bots.anchors import anchor_conventions, anchor_source

    pcfg = ctx.cfg.population
    out: dict[str, Artifact] = {}
    for aid in sorted(ctx.agents):
        ag = ctx.agents[aid]
        anchor = pcfg.warm_start_overrides.get(aid) or (pcfg.warm_start if pcfg.warm_start != "author" else None)
        art = None
        if anchor is None:
            art = A.produce(ctx, ag, "author", A.author_prompt(ag, 0), 0, None, [], [], "author")
            if art is None:
                anchor = "random"  # authoring failed twice: start from the random bot (counted in failed_revisions)
        if art is None:
            art = Artifact.make(anchor_source(anchor), anchor_conventions(anchor), author=f"anchor:{anchor}",
                                group=ag.group, generation=0, origin="seed")
        out[aid] = art
    return out


def build_warm_start(cfg: ExperimentConfig, root: str | Path) -> Path:
    """Author generation 0 for this population seed once (or reuse an existing set with the same key)."""
    from .context import RunContext
    from .generation import _apply_environment

    key = warm_start_key(cfg)
    final = Path(root) / f"p{cfg.population.seed}-{key}"
    path = final / SET_FILE
    if path.exists():
        load_warm_start(path, cfg)  # verifies the key
        return path
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    wcfg = from_dict({"condition": "_warm_start", "population": {"warm_start_set": ""}}, cfg)
    ctx = RunContext(wcfg, tmp)
    try:
        _apply_environment(ctx, 0)
        ctx.start_budgets(None)
        arts = author_initial(ctx)
        totals = ctx.ledger.totals()
        doc = {"key": key, "population_seed": cfg.population.seed, "name": cfg.name,
               "agents": {aid: art.to_dict() for aid, art in sorted(arts.items())},
               "content": {aid: art.id for aid, art in sorted(arts.items())},
               "ledger_totals": {k: totals[k] for k in ("calls", "tokens", "spend_usd", "new_spend_usd")},
               "counters": {aid: dict(sorted(ag.counters.items())) for aid, ag in sorted(ctx.agents.items())}}
        with open(tmp / SET_FILE, "w") as f:
            json.dump(doc, f, sort_keys=True, indent=1)
    finally:
        ctx.close()
    os.replace(tmp, final)
    return path


def load_warm_start(path: str | Path, cfg: ExperimentConfig) -> dict[str, Any]:
    """Read a warm-start set and check it was built for this configuration (same key, same population seed) and
    that every artifact's content hash matches its id."""
    doc = json.loads(Path(path).read_text())
    key = warm_start_key(cfg)
    if doc["key"] != key or doc["population_seed"] != cfg.population.seed:
        raise WarmStartMismatch(f"{path} was built for warm-start key {doc['key']} (population seed "
                                f"{doc['population_seed']}); this run needs {key} (seed {cfg.population.seed})")
    arts = {aid: Artifact.from_dict(d) for aid, d in doc["agents"].items()}
    for aid, art in arts.items():
        if artifact_id(art.code, art.conventions) != art.id or doc["content"][aid] != art.id:
            raise WarmStartMismatch(f"{path}: artifact for {aid} does not match its content hash")
    doc["artifacts"] = arts
    return doc
