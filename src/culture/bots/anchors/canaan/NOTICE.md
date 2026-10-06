# Vendored Canaan et al. rule-based Hanabi agents

Source: https://github.com/rocanaan/hanabi-ad-hoc-learning at commit `dde3edbf7c59d5f1edb0aae13660c06774413169`,
directory `Experiments/Rulebased/`. License: Apache-2.0 (copy in `LICENSE`).

Files copied **unmodified**: `ruleset.py`, `rulebased_agent.py`, `piers_agent.py`, `iggi_agent.py`, `flawed_agent.py`.
They import `rl_env` and `pyhanabi` as top-level modules; `culture/bots/anchors/canaan_loader.py` aliases those names to
`hanabi_learning_environment.rl_env` / `.pyhanabi` before importing, and swaps the module-level `random` for a per-bot
`random.Random(seed)` so games are deterministic. Nothing in these files is edited.

Published 2-player self-play means (Canaan et al., AIIDE 2020, Table 1): Piers 16.99, IGGI 15.99, Flawed ~0.
