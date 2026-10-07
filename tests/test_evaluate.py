"""Evaluators and statistics."""

import numpy as np

from culture.bots.runner import BotSpec
from culture.evaluate import stats
from culture.evaluate.crossplay import crossplay, crossplay_matrix
from culture.evaluate.ladder import anchor_scores, ladder, non_cycling
from culture.evaluate.selfplay import selfplay


def test_paired_delta_against_itself_is_exactly_zero(ev):
    s = BotSpec.anchor("iggi")
    a = [r.score for r in selfplay(ev, s, list(range(30)))]
    b = [r.score for r in selfplay(ev, s, list(range(30)))]
    d = stats.paired_delta(a, b)
    assert d.mean == 0.0 and d.ci == (0.0, 0.0)


def test_bootstrap_ci_width_matches_sigma_arithmetic():
    rng = np.random.default_rng(1)
    sigma, n = 2.0, 400
    x = rng.normal(17, sigma, n)
    lo, hi = stats.bootstrap_ci(x, stat=np.mean, n_boot=4000)
    expected = 2 * stats.ci_halfwidth_from_sigma(sigma, n)
    assert abs((hi - lo) - expected) / expected < 0.12
    assert stats.games_for_halfwidth(2.0, 0.2) == 385


def test_iqm_and_stratified_bootstrap():
    assert stats.iqm([0, 1, 2, 3, 4, 5, 6, 100]) == np.mean([2, 3, 4, 5])
    strata = [np.full(10, 1.0), np.full(10, 3.0)]
    point, (lo, hi), _ = stats.stratified_bootstrap(strata, stat=np.mean, n_boot=200)
    assert point == 2.0 and lo == hi == 2.0  # resampling within strata keeps each stratum's mean fixed


def test_hc_null_rate_nominal():
    n, level = 50, 0.1
    thr = stats.hc_null_threshold(n, level=level, sims=3000, seed=1)
    rng = np.random.default_rng(7)
    rate = np.mean([stats.higher_criticism(stats.z_to_p(rng.normal(size=n)))[0] > thr for _ in range(2000)])
    assert abs(rate - level) < 0.03


def test_hc_detects_sparse_signal():
    n = 200
    thr = stats.hc_null_threshold(n, level=0.05, sims=2000, seed=2)
    rng = np.random.default_rng(3)
    z = rng.normal(size=n)
    z[:10] += 3.5
    assert stats.higher_criticism(stats.z_to_p(z))[0] > thr


def test_crossplay_alternates_seats_and_matrix(ev):
    a, b = BotSpec.anchor("piers"), BotSpec.anchor("iggi")
    res = crossplay(ev, a, b, list(range(10)))
    # canonical seats (fix round 1, step 6): lower key in seat 0 on even seeds, whatever the argument order
    assert [r.seats for r in res[:2]] == [["anchor:iggi", "anchor:piers"], ["anchor:piers", "anchor:iggi"]]
    m = crossplay_matrix(ev, [a, b, BotSpec.anchor("flawed")], list(range(20)))
    assert np.allclose(m, m.T) and m[2, 2] == 0 and m[0, 0] > 14


def test_memo_reuses_games(ev):
    s = BotSpec.anchor("piers")
    selfplay(ev, s, list(range(10)))
    n = ev.games_played
    selfplay(ev, s, list(range(10)))
    assert ev.games_played == n


def test_ladder_and_anchor_scores(ev):
    cur = BotSpec.anchor("piers")
    lad = ladder(ev, cur, [("g0", BotSpec.anchor("random")), ("g1", BotSpec.anchor("iggi"))], ["piers"], list(range(10)))
    assert lad["g0"] == 0 and lad["anchor:piers"] > 10
    assert anchor_scores(ev, cur, ["flawed"], list(range(4)))["flawed"] >= 0
    assert non_cycling({0: {"a": 1.0}, 1: {"a": 2.0}, 2: {"a": 3.0}}) == 1.0
