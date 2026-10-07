"""Overnight 2, step 5: the A1 transmission-fidelity driver on the stub (no model calls)."""

import json
import math

import numpy as np
import pytest

from culture.agents import agent as A
from culture.bots.anchors import anchor_conventions, anchor_source
from culture.run import transmission as T

SMALL = {"selfplay_games": 20, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
         "ladder_every": 0, "workers": 0}


def test_media_put_the_right_content_in_the_prompt_and_rows_resume(tmp_path, monkeypatch):
    prompts = {}
    orig = A.produce

    def spy(ctx, agent, tag_, user, *a, **kw):
        if tag_ == "revise":
            prompts.setdefault(len(prompts), user)
        return orig(ctx, agent, tag_, user, *a, **kw)

    monkeypatch.setattr(A, "produce", spy)
    spec = {"name": "a1_test", "replicates": [0, 1], "base": {"experiment_seed": 3, "evaluation": SMALL}, "n_boot": 50}
    s = T.run(spec, tmp_path / "a1")
    rows = [json.loads(x) for x in (tmp_path / "a1" / "rows.jsonl").read_text().splitlines()]
    assert [(r["replicate"], r["medium"]) for r in rows] == [(0, "code"), (0, "prose"), (0, "both"), (1, "code"),
                                                              (1, "prose"), (1, "both")]
    code_line = "RULES = [['hail_mary'"  # Piers' CONFIG line: only in the teacher's code
    conv_line = anchor_conventions("piers").strip().splitlines()[0]
    p = [prompts[i] for i in range(3)]
    assert code_line in p[0] and conv_line not in p[0]  # code only
    assert code_line not in p[1] and conv_line in p[1]  # prose only
    assert code_line in p[2] and conv_line in p[2]  # both
    assert all(T.COVER in x for x in p)
    for r in rows:  # every copy measured on the same deals
        assert r["v_T"] == rows[0]["v_T"] and r["v_S0"] == rows[0]["v_S0"]
    assert set(s["media"]) == {"code", "prose", "both"} and s["media"]["code"]["n"] == 2
    n = len(rows)
    T.run(spec, tmp_path / "a1")  # resume: nothing new
    assert len((tmp_path / "a1" / "rows.jsonl").read_text().splitlines()) == n


def test_parameters_known_answers():
    v_ss = np.array([10.0, 12.0, 14.0, 16.0])
    v_st = np.array([8.0, 8.0, 8.0, 8.0])
    p = T.parameters(v_ss, v_st, v_t=17.0, N=4, n_boot=100)
    assert p["alpha"]["point"] == pytest.approx(17.0 - 13.0)
    beta = np.std(v_ss, ddof=1) * math.sqrt(6) / math.pi
    assert p["beta"]["point"] == pytest.approx(beta)
    assert p["rho"]["point"] == pytest.approx(8.0 / 17.0)
    assert p["delta_N4"]["point"] == pytest.approx(-4.0 + beta * (0.5772156649015329 + math.log(4)))
    assert p["predicted_sign_at_N"]["sign"] == ("+" if p["delta_N4"]["point"] > 0 else "-")
