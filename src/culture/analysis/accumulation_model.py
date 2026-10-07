"""Henrich-style accumulation model for P1 (research.tex Experiment P1): a discrete-generation simulator of the skill
distribution of N agents under three organizations, parameterized only by single-student micro-parameters.

One generation, for every agent i with skill s_i (held-out self-play, capped at `cap`):

1. Copy (sharing organizations only). Agent i receives the best artifact among its reachable exemplars b_i:
   full sharing, every other agent; organized, the other members of its group (plus, under migration mode "copy",
   the other group's members with probability `migration_rate` per generation). It copies it:
     - fidelity mode "mixture": with probability p_s the copy scores b_i - L, L ~ Normal(mu_loss, sd_loss); on
       failure the agent keeps its own artifact (nothing to verify);
     - mode "empirical": the copy scores b_i - L with L resampled from the measured losses (failed copies included);
     - mode "exact": the copy is the artifact itself (the harness's replace_if_better adopts the sender's code).
2. Verify. A copy c with delta = c - s_i is adopted with probability 1 - miss if delta > 0 and false_adoption if
   delta <= 0, or with P(adopt | delta) interpolated from an operating-characteristic curve when one is given.
3. Innovate. With probability epsilon(s) the agent improves by a gain drawn from the measured improvement
   distribution at that level; otherwise it is unchanged (a failed revision is rejected by the harness's
   accept-if-not-worse rule, so innovation never loses skill).
4. Migrate (organized only, mode "swap"): every `migration_interval` generations, round(rate * N) agents each swap
   group with a random agent of another group, as `org.migration.RandomMigration` does. Agents keep their skill.

epsilon(level) and the gain distribution are given at measured anchor levels; between anchors epsilon is linear in the
level and the gain is the quantile interpolation of the two empirical distributions (draw u, return
(1 - w) Q_0(u) + w Q_1(u)); beyond the anchors the nearer anchor's values are used ("flat"), or epsilon falls
linearly to zero at the cap ("linear_to_cap"). `Innovation.epsilon_fn` swaps in any other function.

Random numbers come from separate streams per step (copy, verification, innovation, migration), drawn for every agent
whether or not it uses them, so organizations run with the same seed share their innovation draws (common random
numbers): with sharing off, all three organizations give identical paths. Pure numpy; vectorized over Monte Carlo runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

CAP = 25.0


@dataclass
class Fidelity:
    p_s: float = 1.0
    mu_loss: float = 0.0
    sd_loss: float = 0.0
    mode: str = "mixture"  # "mixture" | "empirical" | "exact"
    losses: Sequence[float] | None = None  # for mode "empirical"
    loss_floor: float | None = None  # e.g. 0.0: a copy never exceeds its exemplar (switches off the copy-noise ratchet)

    def __post_init__(self):
        if self.mode not in ("mixture", "empirical", "exact"):
            raise ValueError(f"unknown fidelity mode {self.mode!r}")
        if self.mode == "empirical" and not self.losses:
            raise ValueError("fidelity mode 'empirical' needs losses")


@dataclass
class Verification:
    false_adoption: float = 0.0
    miss: float = 0.0
    oc_delta: Sequence[float] | None = None  # optional P(adopt | delta) curve, used instead of the two rates
    oc_p: Sequence[float] | None = None

    def p_adopt(self, delta: np.ndarray) -> np.ndarray:
        if self.oc_delta is not None:
            return np.interp(delta, np.asarray(self.oc_delta, float), np.asarray(self.oc_p, float))
        return np.where(delta > 0, 1.0 - self.miss, self.false_adoption)


@dataclass
class Innovation:
    levels: Sequence[float]  # anchor levels, increasing
    eps: Sequence[float]  # epsilon at each anchor
    gains: Sequence[Sequence[float]]  # empirical gain-given-improvement sample at each anchor
    beyond: str = "flat"  # "flat" | "linear_to_cap"
    cap: float = CAP
    epsilon_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self):
        self.levels = np.asarray(self.levels, float)
        self.eps = np.asarray(self.eps, float)
        if len(self.levels) != len(self.eps) or len(self.levels) != len(self.gains):
            raise ValueError("levels, eps and gains must have the same length")
        if np.any(np.diff(self.levels) <= 0):
            raise ValueError("anchor levels must increase")
        self._q = [np.sort(np.asarray(g, float)) if len(g) else np.zeros(1) for g in self.gains]

    def epsilon(self, s: np.ndarray) -> np.ndarray:
        if self.epsilon_fn is not None:
            return np.clip(self.epsilon_fn(s), 0.0, 1.0)
        e = np.interp(s, self.levels, self.eps)
        if self.beyond == "linear_to_cap":
            top, etop = self.levels[-1], self.eps[-1]
            frac = np.clip((self.cap - s) / max(self.cap - top, 1e-9), 0.0, 1.0)
            e = np.where(s > top, etop * frac, e)
        return np.clip(e, 0.0, 1.0)

    @staticmethod
    def _quantile(q: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.interp(u * (len(q) - 1), np.arange(len(q)), q)

    def gain(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        if len(self.levels) == 1:
            return self._quantile(self._q[0], u)
        out = np.empty_like(s)
        lo_idx = np.clip(np.searchsorted(self.levels, s, side="right") - 1, 0, len(self.levels) - 2)
        L0, L1 = self.levels[lo_idx], self.levels[lo_idx + 1]
        w = np.clip((s - L0) / (L1 - L0), 0.0, 1.0)
        for j in range(len(self.levels) - 1):
            m = lo_idx == j
            if m.any():
                out[m] = (1 - w[m]) * self._quantile(self._q[j], u[m]) + w[m] * self._quantile(self._q[j + 1], u[m])
        return out


@dataclass
class Organization:
    name: str
    kind: str  # "isolated" | "full" | "organized"
    groups: int = 2
    migration_rate: float = 0.25
    migration_interval: int = 5
    migration_mode: str = "swap"  # "swap" | "copy" | "none"

    def __post_init__(self):
        if self.kind not in ("isolated", "full", "organized"):
            raise ValueError(f"unknown organization kind {self.kind!r}")


G1_ORGANIZATIONS = (
    Organization("isolated", "isolated"),
    Organization("full", "full"),
    Organization("organized", "organized", groups=2, migration_rate=0.25, migration_interval=5),
)


@dataclass
class SimResult:
    org: str
    mean: np.ndarray  # (runs, generations + 1): population mean skill
    best: np.ndarray  # (runs, generations + 1): population best skill
    final: np.ndarray  # (runs, N): skills at the last generation
    adoptions: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (generations,) mean adoptions per run


def _streams(seed: int):
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(5)]


def simulate(org: Organization, init: np.ndarray | Sequence[float], fidelity: Fidelity, verification: Verification,
             innovation: Innovation, generations: int = 100, runs: int = 1000, seed: int = 0,
             cap: float = CAP) -> SimResult:
    """Monte Carlo of `runs` independent populations. `init` is (N,) shared by all runs, or (runs, N)."""
    init = np.asarray(init, float)
    s = np.broadcast_to(init, (runs, init.shape[-1])).copy() if init.ndim == 1 else init.copy()
    R, N = s.shape
    s = np.clip(s, 0.0, cap)
    r_copy, r_ver, r_inn, r_mig, r_init = _streams(seed)

    if org.kind == "organized":
        size = N // org.groups
        group = np.broadcast_to(np.repeat(np.arange(org.groups), size)[:N], (R, N)).copy()
    else:
        group = np.zeros((R, N), int)
    eye = np.eye(N, dtype=bool)
    losses = np.asarray(fidelity.losses, float) if fidelity.losses is not None else None

    mean = np.empty((R, generations + 1))
    best = np.empty((R, generations + 1))
    adoptions = np.zeros(generations)
    mean[:, 0], best[:, 0] = s.mean(1), s.max(1)

    for g in range(1, generations + 1):
        # draws for every agent, every generation, every organization (common random numbers)
        u_succ = r_copy.random((R, N))
        z_loss = r_copy.standard_normal((R, N))
        i_emp = r_copy.integers(0, len(losses) if losses is not None else 1, (R, N))
        u_ver = r_ver.random((R, N))
        u_mx = r_ver.random((R, N))  # migration-as-copy access
        u_eps = r_inn.random((R, N))
        u_gain = r_inn.random((R, N))

        if org.kind != "isolated":
            if org.kind == "full":
                reach = np.broadcast_to(~eye, (R, N, N))
            else:
                same = group[:, :, None] == group[:, None, :]  # (R, N, N): j reachable from i
                reach = same & ~eye
                if org.migration_mode == "copy" and org.migration_rate > 0:
                    reach = reach | ((u_mx < org.migration_rate)[:, :, None] & ~same)
            cand_src = np.where(reach, s[:, None, :], -np.inf).max(2)  # best reachable exemplar
            has = np.isfinite(cand_src)
            if fidelity.mode == "exact":
                loss, copied = np.zeros((R, N)), has
            elif fidelity.mode == "empirical":
                loss, copied = losses[i_emp], has
            else:
                copied = has & (u_succ < fidelity.p_s)
                loss = fidelity.mu_loss + fidelity.sd_loss * z_loss
            if fidelity.loss_floor is not None:
                loss = np.maximum(loss, fidelity.loss_floor)
            cand = cand_src - loss
            cand = np.clip(np.where(copied, cand, s), 0.0, cap)
            delta = cand - s
            adopt = copied & (u_ver < verification.p_adopt(delta))
            s = np.where(adopt, cand, s)
            adoptions[g - 1] = adopt.sum(1).mean()

        eps = innovation.epsilon(s)
        imp = u_eps < eps
        s = np.clip(np.where(imp, s + innovation.gain(s, u_gain), s), 0.0, cap)

        if (org.kind == "organized" and org.migration_mode == "swap" and org.migration_rate > 0
                and org.groups > 1 and g % org.migration_interval == 0):
            k = max(1, round(org.migration_rate * N))
            for _ in range(k):
                a = r_mig.integers(0, N, R)
                ga = group[np.arange(R), a]
                # a random agent of another group
                other = group != ga[:, None]
                pick = r_mig.random((R, N)) * other
                b = pick.argmax(1)
                ok = other[np.arange(R), b]
                gb = group[np.arange(R), b]
                rows = np.arange(R)[ok]
                group[rows, a[ok]] = gb[ok]
                group[rows, b[ok]] = ga[ok]

        mean[:, g], best[:, g] = s.mean(1), s.max(1)

    return SimResult(org.name, mean, best, s, adoptions)


def _at(m: np.ndarray, k: int, a: float) -> dict:
    R = m.shape[0]
    blocks = m[: (R // k) * k].reshape(-1, k).mean(1) if R >= k else m
    return {"expected": float(m.mean()), "mc_se": float(m.std(ddof=1) / np.sqrt(R)),
            "one_seed_interval": [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))],
            f"mean_of_{k}_seeds_interval": [float(np.quantile(blocks, a)), float(np.quantile(blocks, 1 - a))],
            "runs": int(R)}


def summarize(res: SimResult, seeds_per_experiment: int = 5, level: float = 0.95,
              checkpoints: Sequence[int] = ()) -> dict:
    """Trajectories and generation-final summaries (plus the same at `checkpoints`). Intervals: the Monte Carlo
    expectation's standard error; the predictive interval for one population seed; and for the mean of
    `seeds_per_experiment` population seeds (the G1 design), from disjoint blocks of runs."""
    a = (1 - level) / 2
    m, b = res.mean, res.best
    k = seeds_per_experiment
    return {
        "organization": res.org,
        "mean_trajectory": {"expected": m.mean(0).tolist(), "lo": np.quantile(m, a, 0).tolist(),
                            "hi": np.quantile(m, 1 - a, 0).tolist()},
        "best_trajectory": {"expected": b.mean(0).tolist(), "lo": np.quantile(b, a, 0).tolist(),
                            "hi": np.quantile(b, 1 - a, 0).tolist()},
        "final_mean": _at(m[:, -1], k, a),
        "mean_at": {str(g): _at(m[:, g], k, a) for g in checkpoints},
        "final_best": {"expected": float(b[:, -1].mean())},
        "adoptions_per_generation": float(res.adoptions.mean()) if res.adoptions.size else 0.0,
    }


def predicted_order(results: dict[str, SimResult], seeds_per_experiment: int = 5, generation: int = -1) -> dict:
    """Order of organizations by expected mean at `generation` (default: the last), and how often that order holds
    strictly in one paired population seed (common random numbers) and in the mean of `seeds_per_experiment` seeds.
    Ties (for example every population at the cap) count as the order not holding."""
    gi = generation
    names = sorted(results, key=lambda n: -results[n].mean[:, gi].mean())
    fin = np.stack([results[n].mean[:, gi] for n in names])  # (orgs, R)
    holds = np.all(np.diff(fin, axis=0) < 0, axis=0)
    k = seeds_per_experiment
    R = fin.shape[1]
    blk = fin[:, : (R // k) * k].reshape(len(names), -1, k).mean(2)
    holds_k = np.all(np.diff(blk, axis=0) < 0, axis=0)
    signs = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = results[x].mean[:, gi] - results[y].mean[:, gi]
            signs[f"{x} - {y}"] = {"expected": float(d.mean()), "p_positive_one_seed": float((d > 0).mean()),
                                   "p_tie_one_seed": float((np.abs(d) < 1e-9).mean())}
    return {"order": names, "p_order_one_seed": float(holds.mean()), f"p_order_mean_of_{k}_seeds": float(holds_k.mean()),
            "pairwise": signs}
