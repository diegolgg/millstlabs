"""Headline figures (matplotlib). Palette: the dataviz reference categorical order (blue, orange, aqua, yellow, ...),
one y-axis per chart, thin 2px lines, recessive grid, legends for two or more series."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REFERENCE_LINES = {"OBL cross-play 23.76": 23.76, "humans 23.37": 23.37, "SmartBot 22.99": 22.99,
                   "o3 with deductions 17.5": 17.5}


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _refs(ax, xmax):
    for label, y in REFERENCE_LINES.items():
        ax.axhline(y, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(xmax, y, " " + label, color=MUTED, fontsize=7, va="center", ha="left")


def between_vs_tokens(s: dict[str, Any], title: str = "Between-group cross-play vs tokens"):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = s["tokens"] / 1e6
    ax.plot(x, s["between_offdiag"], color=SERIES[0], linewidth=2, label="between-group cross-play (group bests)")
    ax.plot(x, s["between_diag"], color=SERIES[1], linewidth=2, label="self-play of group bests")
    ax.plot(x, s["population_mean"], color=SERIES[2], linewidth=1.5, label="population mean self-play")
    _refs(ax, x.max() if x.size else 1)
    ax.set_ylim(0, 25)
    _style(ax, title, "cumulative tokens (millions)", "score (0 to 25)")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    fig.tight_layout()
    return fig


def score_vs_generation(s: dict[str, Any], title: str = "Group best self-play by generation"):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (gname, y) in enumerate(sorted(s["group_best"].items())):
        ax.plot(s["generation"], y, color=SERIES[i % len(SERIES)], linewidth=2, label=gname)
    ax.plot(s["generation"], s["anchor_mean"], color=MUTED, linewidth=1, label="mean score with anchors")
    _style(ax, title, "generation", "self-play score")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=4)
    fig.tight_layout()
    return fig


def teaching_rates(s: dict[str, Any], title: str = "Teaching: adoption and verification pass rates"):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(s["generation"], s["adoption_rate"], color=SERIES[0], linewidth=2, label="adopted / delivered")
    ax.plot(s["generation"], s["verification_pass_rate"], color=SERIES[1], linewidth=2, label="passed / verified")
    ax.set_ylim(0, 1)
    _style(ax, title, "generation", "rate")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    return fig


def tokens_by_tag(s: dict[str, Any], title: str = "Cumulative tokens by call type"):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, (t, y) in enumerate(sorted(s["tokens_by_tag"].items())):
        ax.plot(s["generation"], y / 1e6, color=SERIES[i % len(SERIES)], linewidth=2, label=t)
    _style(ax, title, "generation", "tokens (millions)")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    return fig


def diversity(s: dict[str, Any], title: str = "Diversity: distinct incumbents"):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(s["generation"], s["distinct_incumbents"], color=SERIES[0], linewidth=2, label="distinct incumbents")
    ax.plot(s["generation"], s["code_clusters"], color=SERIES[1], linewidth=2, label="code clusters")
    _style(ax, title, "generation", "count")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    return fig


def conditions_vs_tokens(curves: dict[str, dict[str, np.ndarray]], title: str, ylabel: str = "score"):
    """curves[condition] = {"tokens": x, "iqm": y, "lo": lo, "hi": hi}: IQM line with a bootstrap band."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (name, c) in enumerate(curves.items()):
        col = SERIES[i % len(SERIES)]
        ax.fill_between(c["tokens"] / 1e6, c["lo"], c["hi"], color=col, alpha=0.15, linewidth=0)
        ax.plot(c["tokens"] / 1e6, c["iqm"], color=col, linewidth=2, label=name)
    _style(ax, title, "cumulative tokens (millions)", ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    return fig


def paired_deltas(rows: list[dict[str, Any]], title: str):
    """rows: {"label", "mean", "lo", "hi"}; a dot-and-interval chart with a zero line."""
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.5 * len(rows)))
    for i, r in enumerate(rows):
        ax.plot([r["lo"], r["hi"]], [i, i], color=SERIES[0], linewidth=2)
        ax.plot([r["mean"]], [i], "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(r["hi"], i, f"  {r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]", va="center", fontsize=8, color=TEXT)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=8, color=TEXT)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _style(ax, title, "paired difference in score (same seeds)", "")
    fig.tight_layout()
    return fig
