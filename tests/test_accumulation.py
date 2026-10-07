"""G1/S1 analysis (analysis/accumulation.py) on synthetic generation records with known answers."""

import json
import math

import numpy as np
import pytest

from culture.analysis import accumulation as acc


def _rec(g, mean, best, between, within, levers=None):
    r = {"generation": g, "population_mean": mean, "population_best": best,
         "between_group": None if between is None else {"offdiag_mean": between, "diag_mean": best},
         "within_group_crossplay": within,
         "cost": {"tokens": 0, "spend_usd": 0.0, "by_tag": {}},
         "agents": {"g0a0": {"score": mean, "anchor_score": mean, "credit": 0.0, "candidate": None,
                             "incumbent": f"art{g}", "candidate_score": None}},
         "groups": {"g0": {"best": best}}, "diversity": {"distinct_incumbents": 1, "code_clusters": 1},
         "hc": {"hc": None}, "teaching": {}, "ladder": None, "volatile": {"wall_seconds": 1.5}}
    if levers is not None:
        r["levers"] = levers
    return r


def make_run(d, condition, seed, means, between=None, within=None, calls_per_gen=2, refused_at=(), levers=None):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"config": {"name": "syn", "condition": condition,
                                                          "population": {"seed": seed}}}))
    with open(d / "generations.jsonl", "w") as f:
        for g, m in enumerate(means):
            b = None if between is None else between[g]
            w = None if within is None else within[g]
            f.write(json.dumps(_rec(g, m, m + 1, b, w, None if levers is None else levers[g])) + "\n")
    with open(d / "ledger.jsonl", "w") as f:
        for g in range(len(means)):
            for _ in range(calls_per_gen):
                f.write(json.dumps({"generation": g, "tag": "revise", "input_tokens": 10, "output_tokens": 5,
                                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}) + "\n")
            if g in refused_at:
                f.write(json.dumps({"generation": g, "tag": "teach", "refused": True, "input_tokens": 0,
                                    "output_tokens": 0}) + "\n")
    with open(d / "artifacts.jsonl", "w") as f:
        for g in range(len(means)):
            f.write(json.dumps({"id": f"art{g}", "generation": g, "parents": [f"art{g - 1}"] if g else []}) + "\n")
    return d


def test_trajectories_calls_and_cross_play_fields(tmp_path):
    means = [5.0, 8.0, 12.0, 17.5, 18.0, 21.0]
    make_run(tmp_path / "iso" / "p0", "isolated", 0, means, between=[1, 2, 3, 4, 5, 6], refused_at=(2,))
    make_run(tmp_path / "full" / "p0", "full", 0, means, within=[6, 5, 4, 3, 2, 1], calls_per_gen=3)
    tr = acc.trajectories(acc.run_dirs_under(tmp_path))
    assert set(tr) == {"isolated", "full"} and set(tr["isolated"]) == {0}
    iso, full = tr["isolated"][0], tr["full"][0]
    assert iso["calls"].tolist() == [2, 4, 6, 8, 10, 12]  # refusals are not calls
    assert iso["refusals"].tolist() == [0, 0, 1, 0, 0, 0]
    assert iso["tokens"][-1] == 6 * 2 * 15
    assert np.isnan(iso["within_group"]).all() and iso["between_group"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.isnan(full["between_group"]).all() and full["within_group"][0] == 6
    # every incumbent beats the archive best, but none has a descendant 10 generations later in a 6-generation run
    assert iso["retained_innovations"].shape == (6,) and iso["retained_innovations"][-1] == 0


def test_retained_innovations_series(tmp_path):
    # a chain art0 -> art1 -> ... each better than the last: art_g is retained once art_{g+10} exists
    make_run(tmp_path / "c" / "p0", "c", 0, [float(g) for g in range(15)])
    r = acc.run_trajectory(tmp_path / "c" / "p0")["retained_innovations"]
    assert r.tolist() == [0] * 10 + [1, 2, 3, 4, 5]


def test_calls_to_threshold_hits_and_censoring(tmp_path):
    make_run(tmp_path / "a" / "p0", "a", 0, [5.0, 8.0, 12.0, 17.5, 18.0, 21.0], calls_per_gen=3)
    t = acc.run_trajectory(tmp_path / "a" / "p0")
    r = acc.calls_to_threshold(t, thresholds=(17.0, 20.0, 30.0))
    assert r["17.0"] == {"generation": 3, "calls": 12.0, "censored": False, "calls_at_end": 18.0, "last_generation": 5}
    assert r["20.0"]["generation"] == 5 and r["20.0"]["calls"] == 18.0
    assert r["30.0"]["censored"] and r["30.0"]["calls"] is None and r["30.0"]["calls_at_end"] == 18.0
    assert acc.calls_to_threshold(tmp_path / "a" / "p0", (17.0,))["17.0"]["generation"] == 3  # from a directory


def test_condition_contrast_is_paired_by_seed(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(5):
        base = 10 + 3 * s + rng.normal(0, 0.1, 4)  # large between-seed spread, removed by pairing
        make_run(tmp_path / "b" / f"p{s}", "b", s, list(base))
        make_run(tmp_path / "a" / f"p{s}", "a", s, list(base + 2.0 + 0.01 * s))
    tr = acc.trajectories(acc.run_dirs_under(tmp_path), retained=False)
    c = acc.condition_contrast(tr, "a", "b")
    assert c["generation"] == 3 and c["n"] == 5 and c["seeds"] == [0, 1, 2, 3, 4]
    assert c["iqm"] == pytest.approx(2.02, abs=1e-6)  # middle three of 2.00 .. 2.04
    assert c["ci"][0] <= 2.02 <= c["ci"][1] and c["ci"][1] - c["ci"][0] < 0.05
    assert acc.condition_contrast(tr, "a", "b", generation=0)["generation"] == 0


def test_null_band_is_the_envelope_for_small_k(tmp_path):
    for s in range(5):
        make_run(tmp_path / "n" / f"p{s}", "null", s, [10.0 + s, 11.0 + s, 9.0 - s])
    nb = acc.null_band(acc.run_dirs_under(tmp_path), "population_mean")
    assert nb["k"] == 5 and nb["lo"].tolist() == [10, 11, 5] and nb["hi"].tolist() == [14, 15, 9]
    assert nb["coverage"][0] == pytest.approx(4 / 6)  # (K - 1)/(K + 1): K = 5 is below the 95% order-statistic band
    assert nb["calls"].tolist() == [2, 4, 6]


def test_switchback_recovers_a_planted_effect_under_drift():
    rng = np.random.default_rng(1)
    period, n = 30, 300
    lever = np.array([1 if (i // period) % 2 == 0 else 0 for i in range(n)])
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.3)  # AR(1): 1/e crossing near lag ln(e^-1)/ln(0.8) = 4.5
    y = 10 + 1.5 * lever + 0.02 * np.arange(n) + noise
    r = acc.switchback_contrasts({"generation": np.arange(1, n + 1)}, lever, y, period=period, burn_in=5)
    assert r["n_switches"] == 9 and len(r["blocks"]) == 10 and all(b["kept"] == 25 for b in r["blocks"])
    assert r["blocks"][0]["first_generation"] == 1 and r["switches"][0] == {
        "at": 31, "to": "off", "on_minus_off": pytest.approx(r["blocks"][0]["mean"] - r["blocks"][1]["mean"])}
    assert r["contrast"] == pytest.approx(1.5, abs=0.25)  # adjacent differences cancel the linear drift
    assert 3 <= r["mixing_time"] <= 8 and not r["mixing_time_censored"]
    assert r["mixing_time_raw"] >= r["mixing_time"]  # the square wave and drift inflate the raw autocorrelation
    M = float(np.max(np.abs(np.diff(y))))
    assert r["M_max_step"] == pytest.approx(M)
    assert r["bias_bound"] == pytest.approx(4 * M * (1 / period) * (1 + r["mixing_time"]))
    assert r["bias_bound_theorem_M"] == pytest.approx(4 * float(np.max(np.abs(y))) / period * (1 + r["mixing_time"]))
    # no effect, no drift: contrast near 0
    r0 = acc.switchback_contrasts(None, lever, 10 + noise, period=period, burn_in=5)
    assert abs(r0["contrast"]) < 0.3


def test_mixing_time_edge_cases():
    assert acc.mixing_time(np.ones(10)) == (0, False)
    assert acc.mixing_time([1.0, 2.0]) == (0, True)
    assert acc.mixing_time(np.arange(40, dtype=float), max_lag=5) == (5, True)  # a trend stays correlated: censored


def test_lever_series_from_logged_levers_and_s1_report(tmp_path):
    on = {"routing": {"name": "broadcast_group"}}
    off = {"routing": {"name": "none"}}
    levers = [on] + [on if (g - 1) // 3 % 2 == 0 else off for g in range(1, 13)]
    means = [10.0] + [12.0 if (g - 1) // 3 % 2 == 0 else 10.0 for g in range(1, 13)]
    for s in range(2):
        make_run(tmp_path / "teaching" / f"p{s}", "teaching", s, means, levers=levers)
    t = acc.run_trajectory(tmp_path / "teaching" / "p0", retained=False)
    assert acc.lever_on_series(t, "routing").tolist() == [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    rep = acc.s1_report(tmp_path, period=3, burn_in=1)
    r = rep["levers"]["teaching"]["outcomes"]["population_mean"]
    assert rep["levers"]["teaching"]["lever"] == "routing"
    assert r["contrast_mean_over_seeds"] == pytest.approx(2.0) and r["per_seed"]["0"]["n_switches"] == 3
    json.dumps(rep, allow_nan=False)


def test_g1_report_and_figure(tmp_path):
    for c, shift in (("isolated", 0.0), ("full", 1.0), ("organized", 2.0)):
        for s in range(3):
            m = [8.0 + shift + 0.5 * g + 0.1 * s for g in range(30)]
            make_run(tmp_path / "g1" / c / f"p{s}", c, s, m,
                     between=None if c == "full" else [5.0 + 0.1 * g for g in range(30)],
                     within=None if c == "isolated" else [6.0] * 30)
            make_run(tmp_path / "null" / c / f"p{s}", c, s, [8.0 + shift + 0.01 * s] * 30)
    rep = acc.g1_report(tmp_path / "g1", tmp_path / "null", generations=29)
    org = rep["conditions"]["organized"]
    assert org["final"]["population_mean"]["iqm"] == pytest.approx(8 + 2 + 14.5 + 0.1)
    assert org["calls_to_threshold"]["17.0"]["seeds_reached"] == 3
    c = [x for x in rep["contrasts"] if x["a"] == "organized" and x["b"] == "isolated" and x["key"] == "population_mean"]
    assert c and c[0]["iqm"] == pytest.approx(2.0)
    assert rep["null_band"]["organized"]["real_median_final_inside"] is False
    json.dumps(rep, allow_nan=False)
    png = tmp_path / "g1.png"
    acc.g1_figure(tmp_path / "g1", tmp_path / "null", png)
    assert png.stat().st_size > 10_000
    assert not math.isnan(rep["wall_clock"]["g1"]["seconds_per_generation_mean"])
