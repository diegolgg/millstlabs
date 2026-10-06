Scope: offline engineering validation only, on the stub LLM backend. Zero API calls were made, no model and no local model ran, and nothing here is evidence about LLM agents or about any organizational mechanism. The only empirical claims are the engine/anchor ground truth (Canaan Piers, IGGI, Flawed reproduced through our adapter) and machine throughput. Every score produced by an "agent" below is a perturbed copy of an anchor rule list written by the stub.

# Sandbox 1 build status (enrico branch)

Spec: `files/sandbox-architecture.md` (left untracked, as found). Package: `src/culture/`. Tests: `tests/`. Setup: `scripts/setup_env.sh`.

## Environment and engine

- **HLE build path: source build, second attempt.** Attempt 1 (`uv pip install ./third_party/hanabi-learning-environment`) failed: CMake 4.x removed compatibility with `cmake_minimum_required(VERSION 2.8.11)`. Attempt 2 with `CMAKE_POLICY_VERSION_MINIMUM=3.5` built and imported cleanly. The open_spiel arm64 fallback was not needed. Submodule pinned at `54e79594f4b6fb40ebb3004289c6db0e34a8b5fb`; build artifacts are deleted afterwards so the submodule stays clean.
- Python 3.11.16 venv via uv. macOS keeps re-flagging `site-packages/*.pth` as `hidden`, and Python skips hidden `.pth` files, so the editable install is invisible. The fix is a `sitecustomize.py` that adds `src/` to the path (written by the setup script; pytest also sets `pythonpath = ["src"]`). Notebooks run on a venv kernel named `culture` (also installed by the setup script).
- Canaan et al. agents vendored **unmodified** from `rocanaan/hanabi-ad-hoc-learning@dde3edb` (`Experiments/Rulebased/`, Apache-2.0) into `src/culture/bots/anchors/canaan/` with `NOTICE.md`.

### Measured on this machine (Apple M4 Pro, 14 cores, macOS 26.5.1; `scripts/bench_engine.py 1000 12`)

| What | Value |
|---|---|
| raw HLE (pyhanabi, no conversion) | 121,136 steps/s/core |
| random bot through adapter + sandbox | 1,169 games/s/core (14,643 steps/s) |
| native Piers port through adapter | 158 games/s/core (11,319 steps/s) |
| vendored Canaan Piers through adapter | 83 games/s/core |
| pool of 12 workers, Piers self-play | 1,429 games/s |
| pool of 12 workers, vendored Canaan Piers | 671 games/s |

1,000-game self-play means (seeds 0 to 999): **Canaan Piers 16.993** (published 16.99, sd 1.97, se 0.06), **IGGI 15.863** (published 15.99, sd 4.73, se 0.15; within the 0.5 tolerance), **Flawed 0.000**, HLE SimpleAgent 0.000 (the original HLE agent also scores 0 in 2-player: it plays any hinted card), random 0.000. The native ports produce byte-identical action logs to the vendored agents on all 1,000 seeds, and the vendored agents give identical trajectories through raw HLE dicts and through our adapter (1,000 seeds for Piers and IGGI, 300 for Flawed).

Worker rlimits: CPU limit set; **memory limit unavailable on macOS** (`setrlimit(RLIMIT_AS)` is refused), so memory is not capped.

## Phase 1: build steps 1 to 6 (done)

What exists, by spec step:

1. **Engine adapter** `game/hanabi.py` (observation JSON, 1-based ranks, absolute targets, legality check before HLE, deal-order tracking), `game/seeds.py` (`seed = base + index`, per-generation shared seed sets independent of condition), anchors (`random`, `simple`, Canaan `piers`/`iggi`/`flawed` native ports in `bots/anchors/rulebot.py`, vendored originals behind `canaan_loader.py`), hanab.live export plus an independent pure-Python reloader `game/hanablive.py` (round trip checked informally on 400 games; the formal test is phase 6), reference raw-HLE driver `game/hle_raw.py`.
2. **Sandbox** `bots/runner.py`: AST import whitelist, banned builtins and dunder access, restricted `__builtins__` with an import hook, 400-line cap, complexity measure, SIGALRM hard timeout with fallback, soft-timeout counter, illegal/timeout accounting, seat disabled after 2 hard timeouts in a game; `evaluate/pool.py` process pool (spawn, `PYTHONHASHSEED=0`, rlimits) with exact per-generation memoization; `evaluate/selfplay.py`, `crossplay.py` (seats alternate by seed), `ladder.py`, `stats.py` (IQM, vectorized IQM bootstrap, stratified bootstrap, paired delta, CI width from sigma, games-for-halfwidth, HC+ and its Monte-Carlo null threshold).
3. **LLM layer** `llm/backend.py` (Request/Response/Usage, price table for Haiku 4.5 and Sonnet 5, batch and cache multipliers, cost ledger per run/group/agent/generation/tag), `llm/cache.py` (record / replay / replay_strict / off, atomic files), `llm/stub_backend.py`, `llm/parsing.py` (full rewrite or SEARCH/REPLACE diff; full conventions or delta), versioned prompts `llm/prompts/{system,author,revise,teach,ingest,merge,repair}.md` with hashes in the manifest.
4. **Artifacts** `artifacts/schema.py`, `store.py` (content-addressed store, harness evidence registry, per-group corpus with evidence- and ownership-checked `deposit`, `retrieve` writing the touch log), `provenance.py` (DAG with ancestors, clade, depth, teachers_of, edges).
5. **Agent step and loop** `agents/agent.py` (budgeted calls, parse, sandbox admission with a smoke game, one repair call, teach, merge, failure traces), `org/` v0 policies, `run/config.py` (dataclasses, YAML, strict validation, digest), `run/context.py`, `run/generation.py` (the only place that orders steps), `run/runner.py` (deployment lock, config freeze, atomic checkpoint with log byte offsets, resume by truncation, wall-clock budget, initial evaluation written once), `run/manifest.py`, CLI `python -m culture.run`.
6. **Probe** `analysis/probe.py`, `configs/probe.yaml`, `notebooks/00_probe.ipynb` (executed; stub only). Stub probe result: gen-1 self-play 6.5 to 16.9 (headroom gate passes), cost per call measured (stub revise about 8,900 tokens, nominal $0.019 at Haiku prices), **null-alive gate fails**, as expected for the stub (see open questions).

Organization policies built in phase 1: topology `isolated`, `full`, `ring`; routing `none`, `broadcast_group`, `best_to_all`, `random_k`; delivery `deterministic`; verification `none`, `selfplay(n)`, `selfplay_crossplay`; adoption `replace_if_better`, `never`, `merge_llm`; credit `none`, `paired_delta`; allocation `uniform`, `proportional`; selection `keep_best_k`; migration `none`; environment `fixed`; teaching cost `free`, `costly(c, credit_share)`. The class files for `bernoulli`, `critical_social_learning`, `softmax_floor` and `islands` were also written while building the modules; their tests and dry runs are phase 4.

Tests: **76 passed** (test_engine 8, test_adapter 9, test_bots 15, test_evaluate 8, test_cache 12, test_artifacts 5, test_org 10, test_runner 8; one run of the suite takes about 80 s). The runner tests include: kill-and-resume equals uninterrupted at 4 kill points (after receive, verify, revise, evaluate), a 1,000-generation dry run with bounded per-generation log size, config-digest refusal, deployment lock, and an end-to-end run in `replay_strict` from the committed fixture `tests/fixtures/llm_cache_smoke/` (regenerate with `scripts/regen_fixture.py` after changing prompts or the stub).

## Phase 2: 200-generation stub dry run, real crash and resume, bounded logs, analysis notebook (done)

Config `configs/dryrun.yaml`: 3 groups of 4 agents, broadcast teaching inside groups, `selfplay(n=40)` verification, paired-delta credit, 40 self-play / 16 cross-play / 16 anchor games per artifact per generation, 6 evaluation workers. `scripts/kill_resume_check.py` ran the config twice side by side. One run went uninterrupted. The other was hit with **SIGKILL to the whole process group about 1.5 s into generation 101** (checkpoint at 100, partial log lines on disk), then restarted with the same command. The resumed run ended **byte-identical** to the uninterrupted one: equal digests for the checkpoint state, all six JSONL logs and the artifact file set. Wall time stripped; result in `docs/results/phase2-dryrun-resume.json`.

**Log sizes are bounded per generation.** The generation record averaged 11.9 KB over the first 10 generations and 12.6 KB over the last 10 (max 13.5 KB; the only growing field is the ladder's frozen-snapshot list, 3 entries per 20 generations). Other logs grow linearly at constant rates: messages about 42 KB, ledger 10 KB, touch 7 KB, provenance 5 KB, artifact index 3 KB per generation. Checkpoint 94 KB. The content-addressed artifact store (about 125 KB/generation) and the LLM cache (about 118 KB/generation) dominate disk, about 50 MB per 200 generations at this population size. For 1,000-plus-generation runs: about 0.3 MB per generation, or about 0.6 GB per 2,000 generations.

Dry-run outcome on the stub (pipeline behaviour, not learning evidence): population mean self-play rose from 9.0 to 18.1 and the best from 17.2 to 18.3, by stub hill climbing on rule-list mutations. Changepoints at generations 5 and 29, then flat (last-third slope −0.003/gen). 7,164 messages were delivered and 9% passed verification; 567 adoptions, all within groups (isolated topology). 7 retained innovations (k = 10). Between-group cross-play of group bests rose from 16.6 to 17.6 and tracked self-play throughout, as expected for stub bots that share one template. 30.4 M stub tokens, $50.28 nominal at Haiku prices for 200 generations × 12 agents. A real run of this shape would cost that order of money, which is why the queue screens at smaller sizes first.

Built: `analysis/metrics.py` (series, slope, binary-segmentation changepoints, trajectory shape, adoption edges, retained innovations), `analysis/figures.py` (between-group cross-play vs tokens with the OBL / human / SmartBot / o3 reference lines, group bests, teaching rates, tokens by call type, diversity, condition comparisons, paired-delta intervals), `analysis/report.py`, `notebooks/analysis.ipynb` (executed on the dry-run logs), `tests/test_analysis.py`.

Tests: **78 passed** (76 + 2).

Fixes made during this phase:
- The generation-0 record now uses the same `teaching` block as later generations.
- Retained innovations are now scored from the generation records; artifacts are logged without their evaluation, so the first version always reported 0.
- `sitecustomize.py` now appends `src/` instead of prepending, so an explicit `PYTHONPATH` (a worktree) wins.

Phase 1 already contained the two resume-identity fixes found while writing the runner tests: evaluation dicts are built in sorted order, so prompts are identical before and after a JSON checkpoint; and artifact logging is decided from the truncated log, not from the store.

## Deviations from the spec

1. **Observation JSON adds `possible`** (`{"colors": [...], "ranks": [...]}`) to every card, own and partner's. It is HLE's plausibility set, which includes negative information. HLE gives it to every agent and Canaan's agents depend on it. Without it the adapter loses information the raw engine provides, and trajectories cannot match. Partner cards also carry `hints`/`possible` (what the partner knows).
2. `last_moves` is most-recent-first (HLE order) and goes back to and includes the observer's own last move. PLAY/DISCARD entries include the revealed `color`/`rank`; PLAY includes `scored`.
3. The hanab.live exporter and reloader live in `game/hanablive.py`, not in `hanabi.py`. HLE's colors map to hanab.live No Variant suits as R, Y, G, B, W -> Red, Yellow, Green, Blue, Purple (W plays Purple).
4. Evaluation anchors are the native rule-list ports (`piers`, `iggi`, `flawed`), which are about twice as fast and can be passed around as artifacts. They are tested move-for-move identical to the vendored originals (`canaan_*`) on 1,000 seeds.
5. The soft timeout (50 ms) is counted (`slow_rate`) but not enforced. Replacing a move on wall time would make games depend on machine load. The hard timeout (1 s) is enforced.
6. **The Anthropic and OpenAI-compatible backends were not written.** The instruction was that the stub is the only backend run tonight, and I judged untested SDK code worse than none. `validate()` refuses any backend other than `stub`. They plug in behind `Backend` with no other change.
7. The default model is `claude-haiku-4-5` (screening tier per section 17). The spec's `Request` comment says Opus, but section 17 rules Opus out for agents.
8. `ingest.md` is a prompt fragment rendered into the revise context, not a separate LLM call. `system.md` (the frozen prefix) and `repair.md` were added.
9. The config digest excludes `runner.generations`, `runner.wall_clock_budget_s`, `evaluation.workers` and `debug`, so a run can be extended or resumed with a different worker count.
10. Seed tags omit the condition, so all conditions of an experiment get byte-identical warm starts (paired). A second condition's warm-start calls are cache hits, and the ledger marks them `cached` with zero new spend.
11. The diversity "code embedding" is token-count cosine (no embedding model offline). The probe's H-group comparison is a keyword list plus bag-of-words cosine; the spec asked for embedding similarity against the rulebook.
12. Higher Criticism uses HC+ (only p > 1/n): with paired seeds a near-deterministic improvement has p about 0, and plain HC* explodes.
13. Paired-delta credit window: before = the student's incumbent at the start of the generation, after = its incumbent at the end, both against the anchor set on that generation's seeds. Messages sent in generation g are received in g+1.
14. Topology `isolated` means groups are isolated from each other; agents inside a group can still exchange. `best_to_all` only sends to agents scoring strictly below the sender.
15. Workers run with `PYTHONHASHSEED=0`, and the CLI re-execs itself with it, so bots that iterate string sets stay deterministic across processes.
16. Extra test file `tests/test_artifacts.py`.

## Blockers

None in phase 1.

## Proposals (not built; the spec does not name them)

- A stub mode that writes **mutually incompatible** conventions (for example, each stub "family" uses a different hint-meaning table) so the null-alive gate and the between-group headline can be exercised before any API spend. Today every stub bot shares one template and cross-plays well.
- Store the full request text next to each cached response (optionally) so a replay mismatch can be diffed directly.
- Port the H-group level 1 to 3 rulebook text into the repo so the probe's conventions diff can use it once an embedding model is allowed.

## Open questions for Enrico

1. The spec file `files/sandbox-architecture.md` is untracked on `enrico`. I left it alone. Commit it?
2. Memory caps for bot workers do not work on macOS. Is CPU plus wall timeouts enough for now, or should bots run under a container on Linux for long runs?
