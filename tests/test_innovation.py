"""Fix round 1, step 12: innovation base rate with a Beta posterior per revise-context condition."""

import numpy as np
import pytest

from culture.analysis.innovation import (beta_posterior, context_of, innovation_by_context, revision_outcomes,
                                         run_innovation_probe)
from culture.analysis.metrics import load_run
from culture.run.config import from_dict


def test_beta_posterior_known_values():
    p = beta_posterior(3, 10)  # Beta(4, 8)
    assert p["a"] == 4 and p["b"] == 8 and p["mean"] == pytest.approx(4 / 12)
    assert 0.1 < p["lo"] < p["mean"] < p["hi"] < 0.65
    flat = beta_posterior(0, 0)
    assert flat["mean"] == 0.5 and flat["lo"] == pytest.approx(0.025) and flat["hi"] == pytest.approx(0.975)
    with pytest.raises(ValueError):
        beta_posterior(5, 3)


def test_planted_rates_are_recovered_per_context():
    rng = np.random.default_rng(1)
    planted = {"nothing": 0.05, "feedback_only": 0.2, "feedback_plus_received": 0.5}
    outcomes = [{"context": c, "improved": bool(x)} for c, eps in planted.items()
                for x in rng.random(4000) < eps]
    est = innovation_by_context(outcomes)
    for c, eps in planted.items():
        e = est[c]
        assert e["n"] == 4000 and abs(e["mean"] - eps) < 0.02 and e["lo"] <= eps <= e["hi"]
        assert e["hi"] - e["lo"] < 4 * 1.96 * np.sqrt(eps * (1 - eps) / 4000) + 0.01  # width ~ the binomial SE
    small = innovation_by_context(outcomes[:20])
    assert small["nothing"]["hi"] - small["nothing"]["lo"] > est["nothing"]["hi"] - est["nothing"]["lo"]


def test_outcomes_from_records_margin_and_context():
    def agent(cs, ps, rc):
        return {"candidate": "c", "candidate_score": cs, "parent_score": ps, "revise_context": rc}

    run = {"generations": [{"generation": 1, "agents": {
        "a": agent(12.0, 10.0, {"traces": True, "received": 0, "corpus": 0}),
        "b": agent(10.5, 10.0, {"traces": True, "received": 2, "corpus": 0}),
        "c": agent(9.0, 10.0, {"traces": False, "received": 0, "corpus": 0}),
        "d": {"candidate": None, "candidate_score": None, "parent_score": None}}}]}
    out = {o["agent"]: o for o in revision_outcomes(run, margin=0.5)}
    assert set(out) == {"a", "b", "c"}
    assert out["a"]["improved"] and not out["b"]["improved"]  # delta == margin is not an innovation
    assert [out[k]["context"] for k in "abc"] == ["feedback_only", "feedback_plus_received", "nothing"]
    assert context_of({"traces": False, "received": 1}) == "received_only"


def test_probe_runs_all_three_contexts(tmp_path):
    base = from_dict({"name": "inno", "population": {"groups": 1, "agents_per_group": 2},
                      "evaluation": {"selfplay_games": 8, "crossplay_games": 2, "anchor_games": 2,
                                     "between_group_games": 0, "ladder_every": 0, "workers": 0},
                      "runner": {"generations": 3}})
    res = run_innovation_probe(base, tmp_path / "probe")
    assert set(res["by_context"]) >= {"nothing", "feedback_only", "feedback_plus_received"}
    for c, e in res["by_context"].items():
        assert 0 <= e["lo"] <= e["mean"] <= e["hi"] <= 1 and e["n"] >= 1
    rows = revision_outcomes(load_run(tmp_path / "probe" / "nothing" / "p0"))
    assert rows and all(o["context"] == "nothing" for o in rows)
