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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if which == "probe":
        print(build("00_probe", PROBE))
