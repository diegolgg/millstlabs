"""IF-PCA for counting distinct strategies (spec section 10, item 5; Jin & Wang 2016).

Rows are artifacts, columns are features (behaviour on a probe set). Features are screened by a Kolmogorov-Smirnov
departure-from-normality score, the screening threshold is chosen by Higher Criticism over the features' null
p-values, and the retained features go to PCA plus k-means. Selecting first matters: with many uninformative features,
PCA on everything loses the cluster signal (tested on planted clusters)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps
from scipy.cluster.vq import kmeans2


def ks_scores(X: np.ndarray) -> np.ndarray:
    """sqrt(n) * sup |F_j - Phi| for each standardized column (constant columns score 0)."""
    n = X.shape[0]
    sd = X.std(axis=0, ddof=1)
    Z = (X - X.mean(axis=0)) / np.where(sd == 0, 1, sd)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if sd[j] == 0:
            continue
        out[j] = np.sqrt(n) * sps.kstest(Z[:, j], "norm").statistic
    return out


def ks_null(n: int, sims: int = 2000, seed: int = 0) -> np.ndarray:
    """Monte-Carlo null distribution of the standardized-column KS score for n samples (Gaussian column)."""
    rng = np.random.default_rng(seed)
    return np.sort(ks_scores(rng.normal(size=(n, sims))))


@dataclass
class IFPCAResult:
    labels: np.ndarray
    K: int
    selected: np.ndarray  # indices of retained features
    threshold: float
    eigenvalues: np.ndarray


def estimate_k(Xs: np.ndarray, kmax: int = 8) -> int:
    """Number of strategies = 1 + number of eigenvalues of the selected-feature sample covariance above the
    Marchenko-Pastur bulk edge (with a 10% margin)."""
    n, p = Xs.shape
    if p == 0:
        return 1
    Z = (Xs - Xs.mean(axis=0)) / np.where(Xs.std(axis=0) == 0, 1, Xs.std(axis=0))
    ev = np.sort(np.linalg.svd(Z, compute_uv=False) ** 2 / n)[::-1]
    edge = (1 + np.sqrt(p / n)) ** 2 * 1.1
    return int(min(kmax, 1 + np.sum(ev > edge)))


def ifpca(X: np.ndarray, K: int | None = None, alpha0: float = 0.5, seed: int = 0, select: bool = True) -> IFPCAResult:
    X = np.asarray(X, float)
    n, p = X.shape
    if select:
        psi = ks_scores(X)
        null = ks_null(n, seed=seed)
        pv = 1.0 - np.searchsorted(null, psi, side="left") / null.size
        pv = np.clip(pv, 0.5 / null.size, 1.0)
        order = np.argsort(pv)
        ps = pv[order]
        i = np.arange(1, p + 1)
        hc = np.sqrt(p) * (i / p - ps) / np.sqrt(ps * (1 - ps) + 1e-12)
        kmax = max(1, int(alpha0 * p))
        m = int(np.argmax(hc[:kmax])) + 1
        selected = np.sort(order[:m])
        threshold = float(ps[m - 1])
    else:
        selected, threshold = np.arange(p), 1.0
    Xs = X[:, selected]
    Kh = K if K is not None else estimate_k(Xs)
    Z = (Xs - Xs.mean(axis=0)) / np.where(Xs.std(axis=0) == 0, 1, Xs.std(axis=0))
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    ev = s ** 2 / n
    if Kh <= 1:
        return IFPCAResult(np.zeros(n, int), 1, selected, threshold, ev)
    emb = U[:, : Kh - 1] * s[: Kh - 1]
    best, best_inertia = None, np.inf
    for r in range(10):  # k-means restarts
        cent, lab = kmeans2(emb, Kh, seed=seed + r, minit="++")
        inertia = float(((emb - cent[lab]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return IFPCAResult(best, Kh, selected, threshold, ev)


def clustering_accuracy(true, pred) -> float:
    """Best-permutation agreement between two labelings."""
    from scipy.optimize import linear_sum_assignment

    t, p = np.asarray(true), np.asarray(pred)
    ut, up = np.unique(t), np.unique(p)
    C = np.array([[np.sum((t == a) & (p == b)) for b in up] for a in ut])
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / t.size)
