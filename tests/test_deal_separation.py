"""Fix round 1, step 10: three disjoint deal sets per generation (feedback traces shown to the LLM, verification,
held-out evaluation) and a per-artifact generalization gap in the generation record."""

import json
import re

import pytest

from culture.agents import agent as A
from culture.artifacts.schema import Artifact
from culture.bots.anchors import anchor_conventions, anchor_source
from culture.game.seeds import generation_seeds
from culture.run.config import from_dict
from culture.run.context import RunContext
from culture.run.runner import run_config

CFG = {
    "name": "deals",
    "population": {"groups": 1, "agents_per_group": 2, "seed": 3},
    "evaluation": {"selfplay_games": 12, "crossplay_games": 4, "anchor_games": 4, "between_group_games": 0,
                   "ladder_every": 0, "workers": 0, "feedback_games": 10},
    "org": {"verification": {"name": "selfplay", "n": 12}},
    "runner": {"generations": 3},
}


@pytest.mark.parametrize("exp_seed", [0, 11, 12345])
def test_feedback_verification_evaluation_are_disjoint(exp_seed):
    for g in range(0, 60, 7):
        fb = set(generation_seeds(exp_seed, g, 5000, "feedback").seeds)
        ve = set(generation_seeds(exp_seed, g, 5000, "verify").seeds)
        ev = set(generation_seeds(exp_seed, g, 5000, "eval").seeds)
        assert not (fb & ve) and not (fb & ev) and not (ve & ev)
        nxt = set(generation_seeds(exp_seed, g + 1, 5000, "eval").seeds)
        assert not (fb & nxt)  # and the next generation's held-out deals are not this generation's feedback deals


def test_traces_shown_to_the_llm_come_only_from_feedback_deals():
    ctx = RunContext(from_dict(CFG), None)
    try:
        art = Artifact.make(anchor_source("flawed"), anchor_conventions("flawed"), author="t", group="g0",
                            generation=0)
        ctx.add_artifact(art)
        for g in (1, 2):
            shown = {int(x) for x in re.findall(r"Game seed (\d+):", A.failure_text(ctx, art, g))}
            assert shown and shown <= set(ctx.seeds(g, "feedback", 10))
            assert not shown & set(ctx.seeds(g, "eval", 12)) and not shown & set(ctx.seeds(g, "verify", 12))
    finally:
        ctx.close()
    no_fb = RunContext(from_dict(dict(CFG, evaluation=dict(CFG["evaluation"], feedback_games=0))), None)
    try:
        assert A.failure_text(no_fb, art, 1).startswith("(no feedback")
    finally:
        no_fb.close()


def test_generalization_gap_is_recorded(tmp_path):
    run_config(CFG, tmp_path / "r")
    recs = [json.loads(x) for x in (tmp_path / "r" / "generations.jsonl").read_text().splitlines()]
    seen = 0
    for r in recs:
        gen = r["generalization"]
        incs = {v["incumbent"] for v in r["agents"].values()}
        cands = {v["candidate"] for v in r["agents"].values() if v["candidate"]}
        assert incs <= set(gen) and cands <= set(gen)
        for aid, row in gen.items():
            assert row["generalization_gap"] == pytest.approx(row["eval"] - row["feedback"], abs=1e-3)
            seen += 1
    assert seen >= 4
