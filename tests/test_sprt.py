"""Fix round 1, step 15: fixed-N verification and Wald's SPRT on synthetic paired differences with known delta."""

import numpy as np
import pytest

from culture.analysis.sprt import decision_rule, fixed_n, operating_characteristics, sprt, wald_asn_h1

SIGMA = 3.5


def streams(delta, n_trials=1500, length=2000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(delta, SIGMA, size=(n_trials, length))


def test_fixed_n_t_test_has_nominal_false_adoption():
    d = streams(0.0, 2000, 100, seed=1)
    rate = np.mean([fixed_n(x, 100, "t-test") for x in d])
    assert 0.035 <= rate <= 0.065
    assert np.mean([fixed_n(x, 100, "mean>0") for x in d]) == pytest.approx(0.5, abs=0.04)  # the current rule


def test_sprt_error_rates_and_sample_size():
    h0 = [sprt(x) for x in streams(0.0, seed=2)]
    h1 = [sprt(x) for x in streams(1.0, seed=3)]
    assert np.mean([a for a, _ in h0]) <= 0.08  # false adoption near alpha = 0.05 (online sigma adds a little)
    assert np.mean([not a for a, _ in h1]) <= 0.08  # missed improvement near beta = 0.05
    asn = np.mean([n for _, n in h1])
    assert 0.6 * wald_asn_h1(SIGMA) <= asn <= 1.4 * wald_asn_h1(SIGMA)
    # at matched error rates, the fixed-N t-test needs far more deals: N = 100 still misses more often than the SPRT
    miss_100 = np.mean([not fixed_n(x, 100, "t-test") for x in streams(1.0, seed=4)])
    assert asn < 100 and miss_100 > np.mean([not a for a, _ in h1])


def test_sprt_known_sigma_and_truncation():
    x = np.full(50, 0.5)  # exactly delta1 / 2: never decisive with known sigma, truncated at the cap
    adopt, n = sprt(x, sigma=SIGMA, cap=50)
    assert n == 50 and adopt is False
    assert sprt(np.full(100, 5.0), sigma=SIGMA)[0] is True


def test_operating_characteristics_counts():
    trials = [{"procedure": "p", "delta": -1.0, "adopt": True, "deals": 10},
              {"procedure": "p", "delta": -1.0, "adopt": False, "deals": 20},
              {"procedure": "p", "delta": 1.1, "adopt": False, "deals": 30},
              {"procedure": "p", "delta": 0.5, "adopt": True, "deals": 40}]
    oc = operating_characteristics(trials)["p"]
    assert oc["false_adoption"] == 0.5 and oc["missed_improvement"] == 1.0 and oc["n_h1"] == 1
    assert oc["deals_mean"] == 25 and oc["deals_near_delta1"] == 30 and oc["missed_near_delta1"] == 1.0


def test_decision_rule_compares_with_the_smallest_equally_accurate_fixed_n():
    def row(fa, miss, deals):
        return {"false_adoption": fa, "missed_near_delta1": miss, "deals_mean": deals, "deals_near_delta1": deals}

    oc = {"SPRT": row(0.003, 0.10, 109), "fixed N=50 (t-test)": row(0.006, 0.50, 50),
          "fixed N=200 (t-test)": row(0.004, 0.12, 200), "fixed N=400 (t-test)": row(0.004, 0.0, 400),
          "fixed N=200 (mean>0)": row(0.06, 0.0, 200)}
    r = decision_rule(oc)
    assert r["smallest_as_accurate"] == "fixed N=400 (t-test)"  # N=50 and N=200 miss more improvements than the SPRT
    assert r["saving_vs_smallest"] == pytest.approx(1 - 109 / 400) and r["passes"]
    oc["fixed N=100 (t-test)"] = row(0.003, 0.09, 100)
    assert decision_rule(oc)["smallest_as_accurate"] == "fixed N=100 (t-test)" and not decision_rule(oc)["passes"]
