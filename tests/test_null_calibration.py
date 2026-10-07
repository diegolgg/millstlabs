"""Fix round 1, step 11: null-population calibration (null stub, null bands, figures)."""

import numpy as np

from culture.analysis import figures
from culture.analysis.metrics import load_run, series
from culture.analysis.null import SERIES_METRICS, calibrate, inside, null_band, run_null_populations
from culture.bots.anchors import PIERS_RULES, parse_rules, rulebot_source
from culture.llm.backend import Request
from culture.llm.stub_backend import StubBackend, StubConfig
from culture.run.config import from_dict

BASE = {
    "name": "null_test",
    "population": {"groups": 2, "agents_per_group": 2},
    "evaluation": {"selfplay_games": 8, "crossplay_games": 4, "anchor_games": 4, "between_group_games": 6,
                   "ladder_every": 0, "workers": 0},
    "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 8}},
    "runner": {"generations": 4},
}


def test_band_has_conformal_coverage_on_exchangeable_draws():
    rng = np.random.default_rng(0)
    b = null_band(rng.normal(size=(399, 1)), 0.95)  # r = 10: guaranteed coverage (400 - 20) / 400 = 0.95
    new = rng.normal(size=4000)
    cov = np.mean((new >= b["lo"][0]) & (new <= b["hi"][0]))
    assert b["coverage"][0] == 0.95 and 0.92 <= cov <= 0.98
    small = null_band(rng.normal(size=(10, 3)), 0.95)  # K < 39: envelope, honest lower coverage
    assert np.allclose(small["coverage"], 9 / 11)


def test_null_stub_revision_keeps_the_rule_list():
    src = rulebot_source(PIERS_RULES)
    req = Request(system=[{"type": "text", "text": "s"}], messages=[{"role": "user", "content": src}],
                  model="claude-haiku-4-5", tag="revise", seed_tag="x")
    for i in range(20):
        req.seed_tag = f"x{i}"
        text = StubBackend(StubConfig(mode="null", invalid_rate=0.0)).complete(req).text
        assert parse_rules(text) == PIERS_RULES and "NOISE = " in text  # same policy, fresh random stream
    hill = [parse_rules(StubBackend(StubConfig(invalid_rate=0.0, diff_rate=0.0)).complete(
        Request(system=req.system, messages=req.messages, model=req.model, tag="revise", seed_tag=f"h{i}")).text)
        for i in range(20)]
    assert any(r != PIERS_RULES for r in hill)  # the default stub does mutate


def test_null_band_contains_the_null_runs_own_metrics(tmp_path):
    base = from_dict(BASE)
    dirs = run_null_populations(base, 4, tmp_path / "null", processes=1)
    cal = calibrate(dirs)
    assert set(cal["metrics"]) == set(SERIES_METRICS) and "retained_innovations" in cal["run_level"]
    assert cal["coverage"] < 0.95 and "envelope" in cal["coverage_note"]  # K=4: honest about coverage
    for d in dirs:
        s = series(load_run(d))
        for m in SERIES_METRICS:
            assert inside(cal, m, s[m].tolist()).all(), (d, m)
    # the null bands draw on the headline figures
    s = series(load_run(dirs[0]))
    figures.between_vs_tokens(s, null=cal).savefig(tmp_path / "between.png")
    figures.score_vs_generation(s, null=cal).savefig(tmp_path / "best.png")
