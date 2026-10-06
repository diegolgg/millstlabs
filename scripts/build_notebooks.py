"""Write and execute the notebooks (stub backend only). Usage: python scripts/build_notebooks.py probe|analysis|transfer"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return nbformat.v4.new_markdown_cell(s)


def code(s):
    return nbformat.v4.new_code_cell(s)


def build(name: str, cells: list, timeout: int = 1800) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "culture", "display_name": "culture (.venv)", "language": "python"}
    NotebookClient(nb, timeout=timeout, kernel_name="culture", resources={"metadata": {"path": str(ROOT)}}).execute()
    path = ROOT / "notebooks" / f"{name}.ipynb"
    nbformat.write(nb, path)
    return path


PROBE = [
    md("# 00 Probe\n\nSpec sections 13-14. One agent per model tier writes a bot (author call) and revises it once, "
       "with no teaching; a second independent agent of the same tier does the same; we cross-play the two.\n\n"
       "**Scope: stub backend only.** No API key was used. Every number below exercises the pipeline (scores of "
       "perturbed anchor bots, fake token counts priced at the tier's list price) and says nothing about a real model. "
       "The same notebook runs unchanged once a real backend exists."),
    code("import json, os, sys\nsys.path.insert(0, 'src')  # repo root is the working directory\n"
         "os.environ.setdefault('PYTHONHASHSEED', '0')\n"
         "from culture.analysis.probe import run_probe\n"
         "OUT = 'runs/probe'\nresults = {m: run_probe('configs/probe.yaml', OUT, m) for m in ['claude-haiku-4-5', 'claude-sonnet-5']}"),
    code("for m, r in results.items():\n"
         "    print(m)\n"
         "    for a, v in r['agents'].items():\n"
         "        print(f\"  {a}: self-play {v['selfplay']:.2f}, illegal {v['illegal_rate']:.3f}, {v['lines']} lines, H-group keywords {v['hgroup_keywords']}\")\n"
         "    print(f\"  cross-play between the two independent agents: {r['crossplay_between_independent_agents']:.2f}\")\n"
         "    print(f\"  conventions bag-of-words cosine between agents: {r['conventions_similarity_between_agents']:.2f}\")\n"
         "    for t, c in r['cost_per_call'].items():\n"
         "        print(f\"  {t}: {c['calls']} calls, {c['tokens_per_call']:.0f} tokens/call, ${c['usd_per_call']:.4f}/call (nominal, stub)\")\n"
         "    print('  gates:', r['gates'])"),
    md("## Gates (decided before running, spec section 14)\n\n"
       "- **Headroom**: generation-one bots below 20 in self-play.\n"
       "- **Cost**: tokens per revise and teach call measured; stage sizes derive from them.\n"
       "- **Null alive**: independent agents' cross-play well below their self-play (here: below 0.8 of mean self-play).\n\n"
       "With the stub, the null gate is expected to **fail**: stub bots are all variants of one rule-list template and "
       "share card-index semantics, so they are mutually compatible. That is a property of the stub, not evidence about "
       "LLM-written conventions."),
    code("print(json.dumps({m: r['gates'] for m, r in results.items()}, indent=1))"),
]

ANALYSIS = [
    md("# Analysis (spec section 13)\n\nReads a run's JSONL logs and cost ledger and draws the headline figures. "
       "**Scope: the logs come from the phase-2 dry run on the stub backend** (3 groups of 4 agents, 200 generations, "
       "in-group teaching with self-play verification). Agents are perturbed copies of anchor rule lists, so curves "
       "show that the pipeline measures what it should, not that anything learns. Token and dollar axes are the stub's "
       "fake usage priced at Haiku 4.5 list prices."),
    code("import json, os, sys\nsys.path.insert(0, 'src')\n%matplotlib inline\n"
         "from culture.analysis.metrics import load_run, series, trajectory_shape, adoption_edges, retained_innovations\n"
         "from culture.analysis import figures\nfrom culture.analysis.report import run_report\n"
         "RUN = os.environ.get('ANALYSIS_RUN', 'runs/dry/ref')\nrun = load_run(RUN)\ns = series(run)\n"
         "print(RUN, len(run['generations']), 'generations')"),
    md("## Headline: between-group cross-play versus cumulative tokens\n\nDashed lines are external reference "
       "points from the literature (OBL cross-play, humans, SmartBot, o3 with deductions). In the stub dry run the "
       "between-group number sits near self-play because every stub bot shares one template."),
    code("figures.between_vs_tokens(s);"),
    code("figures.score_vs_generation(s);"),
    code("figures.teaching_rates(s);"),
    code("figures.tokens_by_tag(s);"),
    code("figures.diversity(s);"),
    md("## Teaching and adoption graph\n\nAdoption edges (sender -> receiver) within and across groups. The DCMM "
       "community fit (phase 5) is added below once built."),
    code("edges = adoption_edges(run)\nwithin = sum(e['sender_group'] == e['receiver_group'] for e in edges)\n"
         "print(f'{len(edges)} adoptions: {within} within groups, {len(edges) - within} across groups')\n"
         "print('retained innovations (k=10):', retained_innovations(run))\n"
         "print('trajectory shape:', trajectory_shape(s, threshold=17.0))"),
    md("## Run report"),
    code("from IPython.display import Markdown\nMarkdown(run_report(RUN))"),
    md("## Resume check and log growth (phase 2)"),
    code("p = 'runs/dry/kill_resume_check.json'\nprint(open(p).read() if os.path.exists(p) else 'not run')"),
]

TRANSFER = [
    md("# 01 Transfer (queue item 1, stage 1)\n\nConditions: **solo** (no routing), **transfer_unverified** (the better "
       "of two agents sends its artifact; the receiver adopts blindly), **transfer_verified** (the receiver adopts only "
       "if the payload beats its incumbent on 100 seeded self-play games). Two agents, 10 generations after the warm "
       "start, 3 population seeds, paired deals. The student is the agent with the lower generation-0 score.\n\n"
       "**Scope: stub backend.** Agents are perturbed copies of anchor rule lists and there is no sabotage in this "
       "experiment, so verification can only cost (false rejections) here; this run validates the experiment pipeline, "
       "pairing and statistics, not the hypothesis."),
    code("import json, os, sys\nsys.path.insert(0, 'src')\n%matplotlib inline\nimport numpy as np\n"
         "from culture.analysis.transfer import load_experiment, pairing_checks, curves, paired_deltas\n"
         "from culture.analysis import figures\nexp = load_experiment('runs/stage1')\npairing_checks(exp)"),
    code("c = curves(exp)\nfigures.conditions_vs_tokens({k: v for k, v in c.items()}, "
         "'Student score vs cumulative population tokens (IQM over 3 population seeds, bootstrap band)', "
         "'student self-play score');"),
    code("for k, v in c.items():\n    print(f\"{k:22s} final IQM {v['iqm'][-1]:5.2f}  per seed {np.round(v['per_seed'][:, -1], 2)}  "
         "tokens {v['tokens'][-1]:,.0f}\")"),
    md("## Paired deltas\n\nGame-level differences of the two conditions' final student artifacts on the identical "
       "final-generation deals (100 games per population seed), bootstrap stratified by population seed. With 3 "
       "population seeds the interval is conditional on these three populations; the per-seed means show the "
       "between-population spread, which is the honest uncertainty at this n."),
    code("rows = paired_deltas(exp, [('transfer_verified', 'solo'), ('transfer_unverified', 'solo'), "
         "('transfer_verified', 'transfer_unverified')])\n"
         "figures.paired_deltas(rows, 'Final student score, paired differences (95% stratified bootstrap)');\n"
         "for r in rows:\n    print(r['label'], round(r['mean'], 2), (round(r['lo'], 2), round(r['hi'], 2)), "
         "'per seed', np.round(r['per_seed_mean'], 2), 'AUC delta per seed', np.round(r['auc_delta_per_seed'], 2), "
         "'token ratio', round(r['tokens_ratio'], 2))"),
    md("## Learning smoke test (spec section 12)\n\nRun through the real generation loop with LLM revision off; the "
       "same checks are asserted in `tests/test_smoke_learning.py`."),
    code("import subprocess\nprint(subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_smoke_learning.py'], "
         "capture_output=True, text=True).stdout[-400:])"),
]

ANALYSIS_P5 = [
    md("## Section-10 toolkit on the dry-run logs (phase 5)\n\nEach tool is validated on synthetic data with a "
       "known answer in `tests/test_analysis_phase5.py`; here they run on the stub dry run, where the expected answers "
       "are mostly structural (isolated groups should come back as separate idea-communities)."),
    md("### Higher Criticism with Efron's empirical null\n\nPer generation: each agent's paired z-score of its "
       "incumbent against the previous generation's on the same deals. With 12 agents the test has little power; the "
       "point is the plumbing and the plateau signal."),
    code("import numpy as np\nfrom culture.analysis.hc import hc_test, plateau\n"
         "res = [hc_test(r['hc']['z'], null='empirical', sims=300) if len(r['hc']['z']) >= 5 else None for r in run['generations']]\n"
         "det = [r.detected for r in res if r is not None]\n"
         "print(f'generations with HC detection: {sum(det)} of {len(det)}; first plateau (5 generations under threshold) starts at generation {plateau(res, 5)}')\n"
         "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(7.5, 3))\n"
         "g = [r['generation'] for r, x in zip(run['generations'], res) if x is not None]\n"
         "ax.plot(g, [x.hc for x in res if x is not None], color=figures.SERIES[0], linewidth=2, label='HC (empirical null)')\n"
         "ax.plot(g, [x.threshold for x in res if x is not None], color=figures.MUTED, linewidth=1, linestyle=(0, (3, 3)), label='null 95% threshold')\n"
         "figures._style(ax, 'Did anyone improve? Higher Criticism per generation', 'generation', 'HC+')\nax.legend(frameon=False, fontsize=8);"),
    md("### DCMM on the adoption graph"),
    code("from culture.analysis.graphs import membership_over_time, mixed_score, adjacency, nmi, influence\n"
         "nodes = sorted(run['generations'][0]['agents'])\ngroups = [run['generations'][-1]['agents'][a]['group'] for a in nodes]\n"
         "edges = adoption_edges(run)\nA = adjacency(edges, nodes)\nfit = mixed_score(A + A.T, K=len(set(groups)), nodes=nodes)\n"
         "print('NMI(recovered communities, configured groups) =', round(nmi(fit.hard(), groups), 3))\n"
         "print('cross-lineage leakage (mean off-diagonal of P) =', round(fit.leakage(), 3))\n"
         "inf = influence(edges, nodes)\nprint('top influence (adoptions caused):', sorted(inf.items(), key=lambda kv: -kv[1])[:4])\n"
         "print('theta (degree parameter):', dict(zip(nodes, np.round(fit.theta, 2))))\n"
         "track = membership_over_time(edges, nodes, len(set(groups)), list(range(20, len(run['generations']), 20)), window=40)\n"
         "print('NMI over time:', [(t['generation'], round(nmi(t['pi'].argmax(1), groups), 2)) for t in track if t['pi'] is not None])"),
    md("### IF-PCA: how many distinct strategies?\n\nBehavioural fingerprints of the distinct incumbents of the last 50 "
       "generations on a fixed probe set of 200 observations (one-hot action category per probe). On binary features "
       "the KS screen keeps every feature on which artifacts disagree, so this reduces to spectral clustering on those."),
    code("from culture.analysis.behavior import probe_set, actions, features, disagreement\nfrom culture.analysis.ifpca import ifpca\n"
         "from culture.artifacts.store import ArtifactStore\nstore = ArtifactStore(RUN + '/artifacts')\n"
         "ids = sorted({a['incumbent'] for r in run['generations'][-50:] for a in r['agents'].values()})\n"
         "probes = probe_set(200)\nacts = [actions(store.get(i).code, probes) for i in ids]\nX = features(acts)\n"
         "r = ifpca(X) if len(ids) >= 6 else None\nD = disagreement(acts)\n"
         "print(f'{len(ids)} distinct incumbents; IF-PCA strategies K = {r.K if r else None}; features kept {len(r.selected) if r else 0} of {X.shape[1]}')\n"
         "print('mean pairwise action disagreement on the probe set:', round(float(D[np.triu_indices(len(ids), 1)].mean()), 3))"),
]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if which == "probe":
        print(build("00_probe", PROBE))
    elif which == "analysis":
        print(build("analysis", ANALYSIS))
    elif which == "analysis_p5":
        print(build("analysis", ANALYSIS + ANALYSIS_P5))
    elif which == "transfer":
        print(build("01_transfer", TRANSFER))
