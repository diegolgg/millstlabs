# State (keep to ten lines)

- Branch restructured 2026-10-08 (notebook layout). Code unchanged except the new `culture.lab` entry point.
- Running: a stub population pilot in an agent worktree (`.claude/worktrees/agent-a6009eb90bd8de5ec/runs/g1s1-stub`), harmless; results land in that worktree's `docs/results/`.
- Blocked on Enrico: pick the first three ideas from `IDEAS.md`; set `budget.weekly_usd` in the registry; choose a hosted provider + model (`configs/hosted_example.yaml`).
- Local model server: `mlx_lm.server` on port 8080 (Qwen3.6-35B-A3B 4-bit), free, ~60–80 s per revise call, one call at a time.
- Hosted backend built and tested against a fake server; never used against a real provider; no key set on this machine.
- Last full test run: 269 passed, 1 skipped (2026-10-07, before the restructure).
