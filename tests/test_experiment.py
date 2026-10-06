"""Experiment runner: conditions share deals and warm starts (paired), population seeds differ."""

import yaml

from culture.analysis.transfer import curves, load_experiment, paired_deltas, pairing_checks
from culture.run.experiment import configs, load_spec, run_experiment

SPEC = {
    "base": {"name": "exp_test", "experiment_seed": 5, "population": {"groups": 1, "agents_per_group": 2},
             "evaluation": {"selfplay_games": 12, "crossplay_games": 4, "anchor_games": 4, "between_group_games": 0,
                            "ladder_every": 0, "workers": 0},
             "corpus": {"enabled": False}, "runner": {"generations": 3}},
    "conditions": {"solo": {"org": {"routing": "none", "verification": "none"}},
                   "transfer": {"org": {"routing": "best_to_all", "verification": {"name": "selfplay", "n": 12}}}},
    "population_seeds": [0, 1],
}


def test_conditions_are_paired(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(SPEC))
    cs = configs(load_spec(spec_path))
    assert len(cs) == 4 and len({c.digest() for _, _, c in cs}) == 4
    assert {c.experiment_seed for _, _, c in cs} == {5}
    run_experiment(spec_path, tmp_path / "out", processes=1)
    exp = load_experiment(tmp_path / "out")
    chk = pairing_checks(exp)
    assert chk["warm_starts_identical"] and chk["student_identical"] and chk["final_deals_identical"]
    p0 = exp["solo"][0]["generations"][0]["agents"]
    p1 = exp["solo"][1]["generations"][0]["agents"]
    assert {a: v["incumbent"] for a, v in p0.items()} != {a: v["incumbent"] for a, v in p1.items()}
    c = curves(exp, n_boot=50)
    assert c["solo"]["iqm"].shape == (3,) and (c["transfer"]["lo"] <= c["transfer"]["hi"]).all()
    rows = paired_deltas(exp, [("transfer", "solo")])
    assert rows[0]["n_games"] == 24 and rows[0]["lo"] <= rows[0]["mean"] <= rows[0]["hi"]
