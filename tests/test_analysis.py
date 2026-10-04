import json

import pytest

from millstlabs.config import ExperimentConfig
from millstlabs.experiments import analyze, matrix


def test_matrix_budget(tmp_path):
    result = matrix(ExperimentConfig(), tmp_path / "matrix.jsonl")
    assert result["runs"] == 60
    assert result["training_decisions"] == 120_000_000
    assert result["warmstart_decisions"] == 250_000


def test_analysis_uses_paired_training_seeds(tmp_path):
    for method, scores in [("r_adult", [20, 40]), ("r_initial", [10, 20])]:
        for seed, score in enumerate(scores):
            run = tmp_path / f"{method}-{seed}"
            run.mkdir()
            meta = {"profile": "prosocial", "method": method, "seed": seed, "config_digest": "test", "warmstart_sha256": str(seed)}
            (run / "manifest.jsonl").write_text(json.dumps(meta) + "\n")
            evaluation = {"checkpoint": 100, "restricted_mean_lifetime": score,
                          "episodes": [{"restricted_mean_lifetime": score}] * 32}
            (run / "evaluation.jsonl").write_text(json.dumps(evaluation) + "\n")
    result = analyze(tmp_path, tmp_path / "analysis.json")
    contrast = result["paired_contrasts"][0]
    assert contrast["n_training_seeds"] == 2
    assert contrast["mean"] == 15
    assert contrast["standard_error"] == pytest.approx(5)
