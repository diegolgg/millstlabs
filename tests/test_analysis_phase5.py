"""Section-10 toolkit on synthetic data with known answers: planted sparse improvement (HC with Efron's empirical
null), planted communities (DCMM via Mixed-SCORE), planted clusters (IF-PCA)."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from culture.analysis.graphs import membership_over_time, mixed_score, nmi, simulate_dcmm
from culture.analysis.hc import empirical_null, hc_test, plateau
from culture.analysis.ifpca import clustering_accuracy, ifpca


# ------------------------------------------------------------------ Higher Criticism with the empirical null
def inflated_null(rng, n, delta0=0.4, sigma0=1.5):
    """Correlated agents (shared teachers, shared seeds) inflate and shift the null of the z-scores."""
    return delta0 + sigma0 * rng.normal(size=n)


def test_empirical_null_estimates_inflated_null():
    rng = np.random.default_rng(0)
    z = inflated_null(rng, 2000)
    z[:60] += 6.0  # sparse improvers in the tail
    for method in ("robust", "mle"):
        d, s = empirical_null(z, method)
        assert abs(d - 0.4) < 0.15 and abs(s - 1.5) < 0.15, method


def test_theoretical_null_false_alarms_empirical_null_does_not():
    rng = np.random.default_rng(1)
    n, reps = 200, 60
    theo = np.mean([hc_test(inflated_null(rng, n), null="theoretical", sims=500).detected for _ in range(reps)])
    emp = np.mean([hc_test(inflated_null(rng, n), null="empirical", sims=500).detected for _ in range(reps)])
    assert theo > 0.8  # N(0,1) reference fires on pure (inflated) noise
    assert emp < 0.15  # nominal level 0.05, with Monte-Carlo slack


def test_planted_sparse_improvement_detected_and_fraction_estimated():
    rng = np.random.default_rng(2)
    n, k = 400, 20  # 5% of agents improved
    hits, fracs = [], []
    for _ in range(20):
        z = inflated_null(rng, n)
        z[:k] += 1.5 * 4.0  # shift of 4 null-sd units
        r = hc_test(z, null="empirical", sims=500)
        hits.append(r.detected)
        fracs.append(r.fraction_improved)
    assert np.mean(hits) > 0.9
    assert 0.0 < np.mean(fracs) <= 0.05 + 0.02  # a lower bound on the 5% that improved, and not wildly below
    assert np.mean(fracs) > 0.015


def test_plateau_detection():
    class R:
        def __init__(self, d):
            self.detected = d

    seq = [R(True), R(True), R(False), R(True), R(False), R(False), R(False), R(False)]
    assert plateau(seq, k=4) == 4 and plateau(seq, k=5) is None


# ------------------------------------------------------------------ DCMM
def _aligned(Pi, pih):
    _, c = linear_sum_assignment(-(Pi.T @ pih))
    return pih[:, c], c


def test_dcmm_recovers_planted_communities_influence_and_leakage():
    n, K = 600, 3
    rng = np.random.default_rng(1)
    theta = rng.uniform(0.4, 1.0, n)
    P = np.array([[1, 0.15, 0.1], [0.15, 1, 0.2], [0.1, 0.2, 1]])
    A, Pi, th = simulate_dcmm(n, K, P, theta, pure_frac=0.7, seed=2, scale=0.6)
    fit = mixed_score(A, K)
    pih, c = _aligned(Pi, fit.pi)
    pure = Pi.max(axis=1) == 1
    assert (pih[pure].argmax(1) == Pi[pure].argmax(1)).mean() > 0.95
    assert np.abs(pih - Pi).sum(axis=1).mean() / 2 < 0.2
    assert np.corrcoef(fit.theta, th)[0, 1] > 0.8  # influence (degree parameter)
    Ph = fit.P[np.ix_(c, c)]
    assert np.allclose(np.diag(Ph), 1) and 0 < fit.leakage() < 0.3
    assert Ph[1, 2] > Ph[0, 2]  # the most-connected pair of lineages has the most leakage


def test_membership_over_time_tracks_a_merge_of_lineages():
    """Two lineages that never exchange, then a period where everyone adopts across: NMI with the original
    groups is high early and drops once ideas cross group boundaries."""
    rng = np.random.default_rng(3)
    nodes = [f"a{i}" for i in range(40)]
    group = np.array([0] * 20 + [1] * 20)
    edges = []
    for g in range(1, 41):
        for _ in range(60):
            i = rng.integers(40)
            same = rng.random() < (0.95 if g <= 20 else 0.5)
            pool = np.where(group == group[i])[0] if same else np.where(group != group[i])[0]
            j = rng.choice(pool[pool != i])
            edges.append({"sender": nodes[i], "receiver": nodes[j], "generation": g})
    track = membership_over_time(edges, nodes, 2, [20, 40], window=20)
    early = nmi(track[0]["pi"].argmax(1), group)
    late = nmi(track[1]["pi"].argmax(1), group)
    assert early > 0.8 and late < early - 0.4


# ------------------------------------------------------------------ IF-PCA
def planted_clusters(seed=0, n=90, p=2000, m=40, shift=2.0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(3), n // 3)
    X = rng.normal(size=(n, p))
    mu = np.zeros((3, m))
    mu[0], mu[1] = shift, -shift
    mu[2, : m // 2], mu[2, m // 2:] = shift, -shift
    X[:, :m] += mu[lab]
    return X, lab, m


def test_ifpca_counts_and_recovers_planted_clusters():
    X, lab, m = planted_clusters()
    r = ifpca(X)
    assert r.K == 3
    assert clustering_accuracy(lab, r.labels) == 1.0
    assert np.all(r.selected < m) and len(r.selected) >= m // 2  # no uninformative feature survives the screen


def test_feature_selection_matters():
    X, lab, _ = planted_clusters(seed=1, p=8000)
    sel = clustering_accuracy(lab, ifpca(X, K=3).labels)
    allf = clustering_accuracy(lab, ifpca(X, K=3, select=False).labels)
    assert sel == 1.0 and allf < sel


def test_dcmm_on_disconnected_isolated_groups():
    """Isolated groups give a disconnected adoption graph; with regularization the fit still recovers the groups,
    gives every node a positive influence and reports near-zero leakage (truth: 0)."""
    rng = np.random.default_rng(4)
    n, K = 30, 3
    group = np.repeat(np.arange(K), n // K)
    A = np.zeros((n, n))
    for _ in range(400):
        i = rng.integers(n)
        pool = np.where((group == group[i]) & (np.arange(n) != i))[0]
        A[i, rng.choice(pool)] += 1
    fit = mixed_score(A + A.T, K)
    assert nmi(fit.hard(), group) == 1.0
    assert fit.theta.min() > 0.3 and fit.leakage() < 0.15
