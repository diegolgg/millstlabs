"""Markdown summary of one run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import adoption_edges, load_run, retained_innovations, series, trajectory_shape


def run_report(run_dir: str | Path) -> str:
    run = load_run(run_dir)
    s = series(run)
    recs = run["generations"]
    if not recs:
        return "# empty run\n"
    last = recs[-1]
    shape = trajectory_shape(s)
    cfg = run["config"]
    edges = adoption_edges(run)
    cross_group = sum(1 for e in edges if e["sender_group"] != e["receiver_group"])
    lines = [
        f"# Run {cfg['name']} / {cfg['condition']} (population seed {cfg['population']['seed']})",
        "",
        f"Backend: {cfg['llm']['backend']} (stub numbers are not model evidence). Config digest "
        f"{run['manifest'].get('config_digest')}. Generations: {len(recs)}.",
        "",
        "| metric | first | last |",
        "|---|---|---|",
        f"| population mean self-play | {s['population_mean'][0]:.2f} | {s['population_mean'][-1]:.2f} |",
        f"| population best self-play | {s['population_best'][0]:.2f} | {s['population_best'][-1]:.2f} |",
        f"| between-group cross-play | {s['between_offdiag'][0]:.2f} | {s['between_offdiag'][-1]:.2f} |",
        f"| distinct incumbents | {s['distinct_incumbents'][0]:.0f} | {s['distinct_incumbents'][-1]:.0f} |",
        f"| cumulative tokens | {s['tokens'][0]:.0f} | {s['tokens'][-1]:.0f} |",
        f"| cumulative spend (USD, nominal for stub) | {s['spend_usd'][0]:.2f} | {s['spend_usd'][-1]:.2f} |",
        "",
        f"Slope of population mean: first third {shape['slope_first_third']:+.4f}/gen, last third "
        f"{shape['slope_last_third']:+.4f}/gen; changepoints at generations {shape['changepoints']}.",
        f"Adoptions: {len(edges)} ({cross_group} across groups). Messages delivered: "
        f"{int(np.nansum(s['teach_delivered']))}; verification pass rate overall "
        f"{np.nansum(s['teach_passed']) / max(np.nansum(s['teach_verified']), 1):.2f}.",
        f"Retained innovations (k=10): {retained_innovations(run)}.",
        f"Credit (cumulative) by agent: " + ", ".join(f"{a} {v['credit']:.2f}" for a, v in sorted(last['agents'].items())),
        "",
    ]
    return "\n".join(lines)
