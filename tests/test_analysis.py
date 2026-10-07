"""Analysis over run logs: series extraction, trajectory shape, report, figures render."""

import numpy as np

from culture.analysis import figures
from culture.analysis.metrics import adoption_edges, changepoints, load_run, series, slope, trajectory_shape
from culture.analysis.report import run_report
from culture.run.runner import run_config

TINY = {
    "name": "tiny_analysis",
    "population": {"groups": 2, "agents_per_group": 2, "seed": 3},
    "evaluation": {"selfplay_games": 8, "crossplay_games": 4, "anchor_games": 4, "between_group_games": 4,
                   "ladder_every": 2, "ladder_games": 2, "workers": 0},
    "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 8}},
    "runner": {"generations": 4},
}


def test_series_report_and_figures(tmp_path):
    run_config(TINY, tmp_path / "r")
    run = load_run(tmp_path / "r")
    s = series(run)
    assert s["generation"].tolist() == [0, 1, 2, 3]
    assert np.all(np.diff(s["tokens"]) > 0)  # tokens are cumulative
    assert set(s["group_best"]) == {"g0", "g1"} and not np.isnan(s["between_offdiag"]).any()
    assert s["teach_sent"][0] == 4 and s["teach_delivered"][1] == 4
    assert isinstance(adoption_edges(run), list)
    rep = run_report(tmp_path / "r")
    assert "between-group cross-play" in rep
    for f in (figures.between_vs_tokens, figures.score_vs_generation, figures.teaching_rates, figures.tokens_by_tag,
              figures.diversity):
        fig = f(s)
        fig.savefig(tmp_path / f"{f.__name__}.png")
    # crossing plumbing, with a data-derived threshold (a fixed constant broke whenever the stub curve shifted)
    thr = float(np.nanmax(s["between_offdiag"]))
    first = int(s["generation"][np.where(s["between_offdiag"] >= thr)[0][0]])
    assert trajectory_shape(s, threshold=thr)["first_gen_between_above_threshold"] == first
    assert trajectory_shape(s, threshold=thr + 1.0)["first_gen_between_above_threshold"] is None


def test_changepoints_and_slope_known_answers():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40), np.full(40, 2.0)] + rng.normal(0, 0.3, 80)
    assert changepoints(y) == [40]
    assert changepoints(rng.normal(0, 1, 80)) == []
    assert slope(np.arange(10) * 0.5) == np.float64(0.5).item()
