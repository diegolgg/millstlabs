"""Teaching/adoption graph analysis: degree-corrected mixed-membership (DCMM) fit by Mixed-SCORE (Jin, Ke & Luo),
influence, community matrix, membership over time, agreement with the configured groups (spec section 10, item 2).

Model: E[A] = Omega = Theta Pi P Pi' Theta, with theta_i the degree (influence) parameter, pi_i a membership vector over
K idea-lineages and P the community matrix (its off-diagonal is cross-lineage leakage). The leading eigenvectors satisfy
Xi = Theta Pi B, so the SCORE ratios r_i = xi_i[2:] / xi_i[1] are convex combinations of K simplex vertices; barycentric
coordinates give the memberships, and xi_i[1] gives theta_i."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment


@dataclass
class DCMMFit:
    nodes: list[str]
    K: int
    pi: np.ndarray  # n x K memberships (rows sum to 1)
    theta: np.ndarray  # n degree parameters (influence), scaled to max 1
    P: np.ndarray  # K x K community matrix, unit diagonal
    vertices: np.ndarray

    def hard(self) -> np.ndarray:
        return self.pi.argmax(axis=1)

    def leakage(self) -> float:
        off = self.P[~np.eye(self.K, dtype=bool)]
        return float(off.mean()) if off.size else 0.0


def _spa(Y: np.ndarray, K: int) -> list[int]:
    """Successive projection: K rows of Y that span the simplex."""
    R = Y.copy()
    idx = []
    for _ in range(K):
        j = int(np.argmax((R ** 2).sum(axis=1)))
        idx.append(j)
        u = R[j] / (np.linalg.norm(R[j]) + 1e-12)
        R = R - np.outer(R @ u, u)
    return idx


def n_components(A: np.ndarray) -> int:
    from scipy.sparse.csgraph import connected_components

    return int(connected_components((np.asarray(A) > 0).astype(float), directed=False)[0])


def mixed_score(A: np.ndarray, K: int, nodes: list[str] | None = None, seed: int = 0,
                tau: float | None = None) -> DCMMFit:
    """Mixed-SCORE. A disconnected graph (for example isolated groups) breaks the Perron-vector ratios, so when the
    graph has more than one component it is regularized, A + tau * mean_degree / n (Amini et al.), tau = 0.25 by
    default; pass tau = 0 to disable or a value to force it."""
    A = np.asarray(A, float)
    A = (A + A.T) / 2
    n = A.shape[0]
    if tau is None:
        tau = 0.25 if n_components(A) > 1 else 0.0
    if tau > 0:
        A = A + tau * A.sum(axis=1).mean() / n
    nodes = nodes or [str(i) for i in range(n)]
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(-np.abs(vals))[:K]
    lam, Xi = vals[order], vecs[:, order]
    if Xi[:, 0].sum() < 0:
        Xi[:, 0] = -Xi[:, 0]
    x1 = np.where(np.abs(Xi[:, 0]) < 1e-12, 1e-12, Xi[:, 0])
    T = np.log(max(n, 3))
    R = np.clip(Xi[:, 1:] / x1[:, None], -T, T)
    if K == 1:
        pi = np.ones((n, 1))
        V = np.zeros((1, 0))
    else:
        L = min(n, 10 * K)
        centers = R
        if n > L:
            centers, _ = kmeans2(R, L, seed=seed, minit="++")
        Y = np.hstack([np.ones((centers.shape[0], 1)), centers])
        V = centers[_spa(Y, K)]
        M = np.hstack([np.ones((K, 1)), V])  # K x K
        W = np.linalg.solve(M.T, np.hstack([np.ones((n, 1)), R]).T).T  # barycentric coordinates
        W = np.clip(W, 0, None)
        W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
        b1 = 1.0 / np.sqrt(np.abs(lam[0] + np.einsum("kj,j,kj->k", V, lam[1:], V)) + 1e-12)
        pi = W / b1[None, :]
        pi = pi / np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)
    b1 = (1.0 / np.sqrt(np.abs(lam[0] + np.einsum("kj,j,kj->k", V, lam[1:], V)) + 1e-12)) if K > 1 else np.ones(1)
    theta = np.abs(x1) / np.clip(pi @ b1, 1e-12, None)
    theta = theta / theta.max() if theta.max() > 0 else theta
    # community matrix: B = [b1, b1 * v] ; P proportional to B Lambda B'
    B = np.hstack([b1[:, None], b1[:, None] * V]) if K > 1 else np.ones((1, 1))
    P = B @ np.diag(lam) @ B.T
    d = np.sqrt(np.clip(np.abs(np.diag(P)), 1e-12, None))
    P = P / np.outer(d, d)
    return DCMMFit(nodes, K, pi, theta, P, V)


def adjacency(edges: list[dict[str, Any]], nodes: list[str], weight: str | None = None) -> np.ndarray:
    ix = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for e in edges:
        if e["sender"] in ix and e["receiver"] in ix:
            A[ix[e["sender"]], ix[e["receiver"]]] += float(e.get(weight, 1.0)) if weight else 1.0
    return A


def influence(edges: list[dict[str, Any]], nodes: list[str]) -> dict[str, float]:
    """Out-degree of the adoption graph: how many adoptions each agent's artifacts caused."""
    A = adjacency(edges, nodes)
    return dict(zip(nodes, A.sum(axis=1).tolist()))


def align(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Permutation of cur's columns maximizing overlap with prev (same node order), Hungarian matching."""
    cost = -(prev.T @ cur)
    _, cols = linear_sum_assignment(cost)
    return cols


def membership_over_time(edges: list[dict[str, Any]], nodes: list[str], K: int, generations: list[int],
                         window: int | None = None) -> list[dict[str, Any]]:
    """Fit DCMM on the adoption graph accumulated up to each generation in `generations` (or over a sliding window),
    with communities aligned across time."""
    out, prev = [], None
    for g in generations:
        es = [e for e in edges if e["generation"] <= g and (window is None or e["generation"] > g - window)]
        A = adjacency(es, nodes)
        if A.sum() == 0:
            out.append({"generation": g, "pi": None})
            continue
        fit = mixed_score(A + A.T, K, nodes)
        pi = fit.pi
        if prev is not None:
            pi = pi[:, align(prev, pi)]
        prev = pi
        out.append({"generation": g, "pi": pi, "theta": fit.theta, "leakage": fit.leakage()})
    return out


def nmi(a, b) -> float:
    """Normalized mutual information between two hard labelings."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((ua.size, ub.size))
    np.add.at(C, (ia, ib), 1)
    C /= C.sum()
    pa, pb = C.sum(axis=1), C.sum(axis=0)
    nz = C > 0
    mi = float((C[nz] * np.log(C[nz] / np.outer(pa, pb)[nz])).sum())
    ha = -float((pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = -float((pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / np.sqrt(ha * hb) if ha > 0 and hb > 0 else (1.0 if ha == hb == 0 else 0.0)


def simulate_dcmm(n: int, K: int, P: np.ndarray, theta: np.ndarray, pure_frac: float = 0.7, seed: int = 0,
                  scale: float = 1.0):
    """Planted DCMM graph for tests: returns (A, Pi, theta)."""
    rng = np.random.default_rng(seed)
    Pi = np.zeros((n, K))
    n_pure = int(pure_frac * n)
    Pi[np.arange(n_pure), np.arange(n_pure) % K] = 1.0
    Pi[n_pure:] = rng.dirichlet(np.ones(K), size=n - n_pure)
    Omega = scale * np.outer(theta, theta) * (Pi @ P @ Pi.T)
    Omega = np.clip(Omega, 0, 1)
    U = rng.uniform(size=(n, n))
    A = np.triu((U < Omega).astype(float), 1)
    return A + A.T, Pi, theta
