"""Adapter ground truth (spec section 12): unmodified Canaan agents through raw HLE and through our adapter give
identical trajectories; 1,000-game means reproduce the published numbers; Flawed scores about 0 (no info leak)."""

import multiprocessing as mp

import numpy as np
import pytest

from culture.bots.runner import BotSpec
from culture.evaluate.selfplay import selfplay

from _raw_helpers import raw_and_adapter

N = 1000
SEEDS = list(range(N))


@pytest.mark.parametrize("kind", ["piers", "iggi", "flawed"])
def test_raw_hle_and_adapter_identical_trajectories(kind):
    n = 1000 if kind != "flawed" else 300
    with mp.get_context("spawn").Pool(10) as pool:
        out = pool.map(raw_and_adapter, [(kind, s) for s in range(n)], chunksize=10)
    assert all(same for _, _, same, _ in out), [s for s, _, same, _ in out if not same][:5]
    assert all(a == b for _, a, _, b in out)


@pytest.mark.parametrize("kind,target,tol", [("piers", 16.99, 0.5), ("iggi", 15.99, 0.5)])
def test_published_means_1000_games(pool_ev, kind, target, tol):
    res = selfplay(pool_ev, BotSpec.anchor("canaan_" + kind), SEEDS)
    mean = np.mean([r.score for r in res])
    assert abs(mean - target) <= tol, mean


def test_flawed_scores_about_zero(pool_ev):
    res = selfplay(pool_ev, BotSpec.anchor("canaan_flawed"), SEEDS)
    assert np.mean([r.score for r in res]) < 0.5


@pytest.mark.parametrize("kind", ["piers", "iggi", "flawed"])
def test_native_port_matches_vendored_move_for_move(pool_ev, kind):
    a = selfplay(pool_ev, BotSpec.anchor("canaan_" + kind), SEEDS)
    b = selfplay(pool_ev, BotSpec.anchor(kind), SEEDS)
    assert [r.digest for r in a] == [r.digest for r in b]


def test_mixed_seats_match(pool_ev):
    """Piers with IGGI: vendored pair and native pair agree, so cross-play anchors are faithful too."""
    a = pool_ev.run([((BotSpec.anchor("canaan_piers"), BotSpec.anchor("canaan_iggi")), list(range(200)))])[0]
    b = pool_ev.run([((BotSpec.anchor("piers"), BotSpec.anchor("iggi")), list(range(200)))])[0]
    assert [r.digest for r in a] == [r.digest for r in b]
