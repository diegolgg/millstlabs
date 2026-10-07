"""P1 accumulation model (src/culture/analysis/accumulation_model.py): limiting cases and reproducibility."""

import time

import numpy as np

from culture.analysis.accumulation_model import (G1_ORGANIZATIONS, Fidelity, Innovation, Organization, Verification,
                                                 predicted_order, simulate, summarize)

INIT = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0])
PERFECT = Verification(0.0, 0.0)


def innov(eps_weak=0.6, eps_top=0.2):
    return Innovation(levels=[3.0, 17.0], eps=[eps_weak, eps_top], gains=[[4.0, 8.0, 10.0], [0.6, 0.7, 0.9]])


def no_innov():
    return Innovation(levels=[3.0], eps=[0.0], gains=[[1.0]])


def test_full_sharing_converges_to_initial_best_in_one_generation():
    r = simulate(Organization("full", "full"), INIT, Fidelity(1.0, 0.0, 0.0), PERFECT, no_innov(),
                 generations=3, runs=20, seed=1)
    assert np.allclose(r.mean[:, 1], 12.0) and np.allclose(r.mean[:, 3], 12.0)
    assert np.allclose(r.final, 12.0)
    # the exact-adoption mode is the same limit
    r2 = simulate(Organization("full", "full"), INIT, Fidelity(mode="exact"), PERFECT, no_innov(), generations=1,
                  runs=5, seed=1)
    assert np.allclose(r2.final, 12.0)


def test_no_sharing_and_no_innovation_changes_nothing():
    r = simulate(Organization("isolated", "isolated"), INIT, Fidelity(0.6, 0.1, 0.5), Verification(0.06, 0.0),
                 no_innov(), generations=50, runs=10, seed=2)
    assert np.allclose(r.final, INIT)
    assert np.allclose(r.mean, INIT.mean())


def test_with_sharing_off_the_three_organizations_coincide():
    out = [simulate(o, INIT, Fidelity(0.0, 0.5, 0.5), Verification(0.06, 0.1), innov(), generations=40, runs=50,
                    seed=3) for o in G1_ORGANIZATIONS]
    for r in out[1:]:
        assert np.array_equal(r.mean, out[0].mean)
        assert np.array_equal(r.final, out[0].final)


def test_reproducible_by_seed_and_different_across_seeds():
    args = (Organization("organized", "organized"), INIT, Fidelity(0.56, 0.07, 0.55), Verification(0.06, 0.0),
            innov())
    a = simulate(*args, generations=30, runs=40, seed=7)
    b = simulate(*args, generations=30, runs=40, seed=7)
    c = simulate(*args, generations=30, runs=40, seed=8)
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.best, b.best)
    assert not np.array_equal(a.mean, c.mean)


def test_skill_stays_in_bounds_and_innovation_never_loses():
    r = simulate(Organization("isolated", "isolated"), INIT, Fidelity(), PERFECT, innov(0.9, 0.9), generations=60,
                 runs=30, seed=4)
    assert r.final.max() <= 25.0 and r.final.min() >= 0.0
    assert np.all(np.diff(r.mean, axis=1) >= -1e-12)


def test_interpolation_of_epsilon_and_gain():
    inn = Innovation(levels=[2.0, 12.0], eps=[0.6, 0.2], gains=[[10.0, 10.0], [0.0, 0.0]])
    s = np.array([0.0, 2.0, 7.0, 12.0, 20.0])
    assert np.allclose(inn.epsilon(s), [0.6, 0.6, 0.4, 0.2, 0.2])
    assert np.allclose(inn.gain(s, np.full(5, 0.5)), [10.0, 10.0, 5.0, 0.0, 0.0])
    inn2 = Innovation(levels=[2.0, 12.0], eps=[0.6, 0.2], gains=[[1.0], [1.0]], beyond="linear_to_cap", cap=22.0)
    assert np.allclose(inn2.epsilon(np.array([17.0, 22.0])), [0.1, 0.0])


def test_sharing_beats_isolation_and_thousand_runs_are_fast():
    t = time.perf_counter()
    res = {o.name: simulate(o, np.full(8, 3.0), Fidelity(0.9, 0.2, 0.3), Verification(0.0, 0.05), innov(),
                            generations=100, runs=1000, seed=0) for o in G1_ORGANIZATIONS}
    assert time.perf_counter() - t < 20.0
    po = predicted_order(res)
    assert po["order"][-1] == "isolated"
    s = summarize(res["full"])
    assert len(s["mean_trajectory"]["expected"]) == 101
