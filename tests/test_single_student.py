"""Fix round 1, step 13: the single-student driver on the stub (no model calls)."""

import json

import pytest

from culture.analysis import credit as C
from culture.bots.anchors import anchor_source, rulebot_source
from culture.bots.runner import BotSpec
from culture.evaluate.selfplay import selfplay
from culture.evaluate.pool import Evaluator
from culture.game.hanabi import HanabiParams
from culture.game.seeds import generation_seeds
from culture.run.single_student import STUDENT_RULES, run_single_student

SMALL = {"selfplay_games": 30, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
         "ladder_every": 0, "workers": 0}


def spec(**kw):
    d = {"name": "c1_test", "k": 3, "replicates": [0, 1], "delivery": "exhaustive",
         "base": {"experiment_seed": 7, "evaluation": SMALL},
         "strata": {"verified": {"org": {"verification": {"name": "selfplay", "n": 30},
                                         "adoption": "replace_if_better"}},
                    "unverified": {"org": {"verification": "none", "adoption": "replace_if_better"}}}}
    d.update(kw)
    return d


def _score(code_key, code, seeds):
    return float(sum(r.score for r in selfplay(Evaluator(HanabiParams(), workers=0), BotSpec(code_key, code), seeds))
                 / len(seeds))


def test_oracle_recovers_planted_effects_without_revision(tmp_path):
    """Revision off: the only effect of a message is adoption, so the planted effects are exact engine numbers."""
    s = spec(base={"experiment_seed": 7, "evaluation": SMALL, "runner": {"revise": False}})
    summ = run_single_student(s, tmp_path / "planted")
    for r in (0, 1):
        seeds = generation_seeds(7 + r, 1, 30, "eval").seeds
        s0 = _score("s0", rulebot_source(STUDENT_RULES), seeds)
        piers = _score("piers_p", anchor_source("piers"), seeds) - s0
        flawed = _score("flawed_p", anchor_source("flawed"), seeds) - s0
        # verified: Piers adopted (it beats the student), Flawed rejected, the re-sent bot is a duplicate
        ver = summ["strata"]["verified"]["per_replicate"][str(r)]
        assert ver["shapley"] == pytest.approx([piers, 0.0, 0.0], abs=1e-9)
        # unverified: blind adoption in teacher order, so the last adopted payload wins; analytic Shapley for k=3
        unv = summ["strata"]["unverified"]["per_replicate"][str(r)]
        v = {m: (flawed if m & 2 else piers if m & 1 else 0.0) for m in range(8)}
        assert unv["v"] == {str(m): pytest.approx(x, abs=1e-9) for m, x in v.items()}
        assert unv["shapley"] == pytest.approx(C.shapley(v, 3).tolist(), abs=1e-9)
        for stratum in ("verified", "unverified"):
            assert summ["strata"][stratum]["per_replicate"][str(r)]["efficiency_gap"] == pytest.approx(0.0, abs=1e-9)


def test_full_loop_with_stub_revisions_and_resume(tmp_path):
    out = tmp_path / "c1"
    summ = run_single_student(spec(), out)
    rows = [json.loads(x) for x in (out / "replays.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 8  # strata x replicates x subsets
    revise_calls = [c for r in rows for c in r["calls"] if c["tag"] == "revise"]
    assert len(revise_calls) == len(rows) and all(len(c["request_key"]) == 64 for c in revise_calls)
    for stratum, s in summ["strata"].items():
        assert s["complete_replicates"] == [0, 1]
        for rep in s["per_replicate"].values():
            assert sum(rep["shapley"]) == pytest.approx(rep["v"]["7"] - rep["v"]["0"])  # efficiency
        for n in ("tau_itt", "leave_one_out", "equal_split", "ridge_bernoulli", "singles_pairs", "plackett_burman"):
            assert n in s["estimators"]
        assert s["operations"]["admissible_rate"] is not None and s["primary_contrast"]["per_replicate"]
    # rerun: every replay is already on disk, nothing new is played
    n_before = len(rows)
    run_single_student(spec(), out)
    assert len((out / "replays.jsonl").read_text().splitlines()) == n_before
    C.figure(summ).savefig(tmp_path / "fig.png")
    assert "leave_one_out" in C.table(summ)


def test_designed_delivery_patterns(tmp_path):
    from culture.run.single_student import delivery_masks, load_spec

    s7 = load_spec({"k": 7, "delivery": "pb8"})
    assert s7["teachers"][3:] == ["null"] * 4 and len(delivery_masks(s7, 0, 0)) == 8
    sb = load_spec({"k": 3, "delivery": {"name": "bernoulli", "p": 0.5, "runs": 8}})
    assert delivery_masks(sb, 0, 0) == delivery_masks(sb, 0, 0)  # seeded
    assert len(delivery_masks(load_spec({"k": 4, "delivery": "singles_pairs"}), 0, 0)) == 1 + 4 + 6


def test_per_stratum_replicates(tmp_path):
    """`replicates` as a dict runs a different number of seeds per stratum (full C1: R = 29 verified, 8 unverified)."""
    from culture.run.single_student import stratum_replicates

    s = spec(replicates={"verified": [0, 1], "unverified": [0]},
             base={"experiment_seed": 7, "evaluation": SMALL, "runner": {"revise": False}})
    assert stratum_replicates(s, "verified") == [0, 1] and stratum_replicates(s, "unverified") == [0]
    assert stratum_replicates(spec(), "unverified") == [0, 1]
    out = tmp_path / "per_stratum"
    summ = run_single_student(s, out)
    rows = [json.loads(x) for x in (out / "replays.jsonl").read_text().splitlines()]
    assert sorted({(r["stratum"], r["replicate"]) for r in rows}) == [("unverified", 0), ("verified", 0),
                                                                        ("verified", 1)]
    assert summ["strata"]["verified"]["complete_replicates"] == [0, 1]
    assert summ["strata"]["unverified"]["complete_replicates"] == [0]
