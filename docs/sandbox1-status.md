Scope: engineering validation, three pilots, and Overnight 2 (2026-10-07): the full C1 Arm 1, a C2 pilot and the A1 parameters, all at $0 (zero paid API calls). Model calls used only a local open-weight model (Qwen3.6-35B-A3B 4-bit on mlx_lm.server, this machine, seeded, temperature 0): the step-9 backend check, the C1 pilot (74 calls), and Overnight 2 (322 calls: C1 at the pre-registered R = 29 verified / 8 unverified, the C2 pilot on 5 seeds, A1 at 30 replicates per medium). That is one model block, so no estimator or claim is accepted; the Gemma replication block, C1 Arm 2, the full C2 and A1's check populations have not run. B1 is engine-only; D1, the null calibrations and every stub run prove pipelines only. Every "agent" score from a stub run is a perturbed copy of an anchor rule list.

# Sandbox 1 build status (enrico branch)

## At a glance

| phase | commit | result |
|---|---|---|
| 1 build steps 1 to 6 | `eb20fab` | HLE from source; Piers 16.993 / IGGI 15.863 / Flawed 0 over 1,000 games through the adapter, identical trajectories to raw HLE; full loop, stub LLM, probe notebook; 76 tests |
| 2 dry run | `060fda2` | 200 generations × 12 agents; SIGKILL mid-generation 101, resume byte-identical; per-generation record about 12 KB; analysis notebook; 78 tests |
| 3 queue item 1 | `ce56131` | solo vs transfer (unverified / verified), 3 seeds, paired; figure and paired deltas; learning smoke test passes all three claims; 82 tests |
| 4 mechanisms | `f867e26` | 11 mechanisms, unit tests with known answers, 20-generation dry-run checks all pass; 94 tests |
| 5 analysis toolkit | `e478c6c` | HC + empirical null, DCMM, IF-PCA recover planted answers; applied to dry-run logs; 103 tests |
| 6 baselines, export | `dfb75cc` | OpenEvolve runs on our Game with the stub LLM; hanab.live export round trip exact on 100 games; 111 tests |
| fix round 1, steps 1 to 12 | `9fb78c6` to `f06372e` | the 8 review defects fixed with regression tests; local MLX backend (live-checked: bit-reproducible); disjoint feedback/verification/evaluation deals; null calibration (K=40); innovation base rate; 183 passed, 1 skipped (opt-in live test) |
| fix round 1, steps 13 to 16 | `e3bd512` to `673178b` | C1 single-student driver with exact Shapley oracle; C1 pilot on stub and MLX (80 replays); B1 SPRT vs fixed N (SPRT passes the 30% rule); D1 pilot on the null stub; **205 passed, 1 skipped** |
| Overnight 2 | `a970163` to `f374fb8` | quarantine policy (C2 stratum); C1 full on MLX: **no confirmatory credit estimator passes** (only exhaustive replay does); C2 pilot: primary mechanical at temperature 0; A1 parameters: medium-ordering rule not met; 322 model calls, $0; **214 passed, 1 skipped** |

How to reproduce: `scripts/setup_env.sh`, then `.venv/bin/python -m pytest -q`. Runs: `python -m culture.run --config configs/<x>.yaml --out runs/<x>`, `scripts/run_experiment.py`, `scripts/kill_resume_check.py`, `scripts/phase4_dryruns.py`, `scripts/bench_engine.py`, `scripts/build_notebooks.py probe|transfer|analysis_p5`. Run outputs live in `runs/` (git-ignored).

Spec: `files/sandbox-architecture.md` (committed by the planning session in `1c823cc`). Package: `src/culture/`. Tests: `tests/`. Setup: `scripts/setup_env.sh`.

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
6. **Probe** `analysis/probe.py`, `configs/probe.yaml`, `notebooks/00_probe.ipynb` (executed; stub only). Stub probe result: gen-1 self-play 6.5 to 16.9 (headroom gate passes), cost per call measured (stub revise about 8,900 tokens, nominal $0.019 at Haiku prices), **null-alive gate fails**, as expected for the stub (see proposals).

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

## Phase 3: queue item 1 on the stub backend, and the learning smoke test (done)

**Experiment** (`configs/stage1_transfer.yaml`, `scripts/run_experiment.py`, `src/culture/run/experiment.py`, `src/culture/analysis/transfer.py`, `notebooks/01_transfer.ipynb`): 2 agents in one group, warm start plus 10 generations, 3 population seeds, experiment seed 11 shared by every run. Conditions: `solo` (no routing); `transfer_unverified` (`best_to_all` routing, verification `none`, so blind adoption); `transfer_verified` (`best_to_all`, `selfplay(n=100)` verification, `replace_if_better`). The **student** is the agent with the lower generation-0 score. Pairing checks pass: warm starts identical across conditions (content hashes), same student in every condition of a seed, and final evaluations on identical deals.

Stub results, for pipeline validation only (agents are perturbed anchor rule lists; no sabotage in this experiment):

| condition | final student IQM over seeds | per seed | cumulative tokens |
|---|---|---|---|
| solo | 6.91 | 4.95, 0.24, 15.53 | 169,297 |
| transfer_unverified | 16.19 | 15.29, 15.20, 18.09 | 196,584 |
| transfer_verified | 13.47 | 14.98, 7.74, 17.68 | 189,813 |

Paired deltas (final student artifacts, game-level differences on the identical final deals, 300 games, bootstrap stratified by population seed): verified minus solo **+6.56** [6.17, 6.96]; unverified minus solo **+9.29** [8.91, 9.68]; verified minus unverified **−2.73** [−2.96, −2.48]. These intervals are conditional on these three populations. The per-seed means (verified minus unverified: −0.31, −7.46, −0.41) show that the between-population spread dominates. One seed (p1) drives the gap, and it comes from a single lucky stub mutation in the unverified run at generation 6 (8.4 → 15.5), not from verification: both conditions adopted the teacher's artifact at generation 1. With 3 seeds the two transfer conditions cannot be separated. Transfer conditions spent 12 to 16% more tokens (teach calls). In a stub world without bad payloads, verification can only cost (false rejections), so this figure is not a test of the hypothesis.

**Learning smoke test** (`tests/test_smoke_learning.py`, through the real generation loop, LLM revision off): oracle teacher g0a0 holds Piers; student g0a2 holds Random; in the sabotage cases a second sender g0a1 holds Flawed, whose message is processed after the oracle's. All three claims hold: verification on, the student adopts Piers and scores 17 ± 1 on 100 games; verification off with sabotage, it adopts Piers then Flawed and drops to 0.0; verification on with sabotage, it adopts Piers and rejects Flawed. Paired-delta credit for the oracle is above 15. Deviation: the spec's "student drops to 0" needs a student that holds something better than 0 first, hence the two-sender design (oracle first, saboteur second) in one generation.

Also added: `tests/test_experiment.py` (conditions share deals and warm starts; population seeds differ; curves and paired deltas compute). Tests: **82 passed** (78 + 3 smoke-learning + 1 experiment).

## Phase 4: remaining section-8 mechanisms (done)

Each mechanism is a small pluggable class selected by config. Each has unit tests on synthetic data with known answers (`tests/test_org_phase4.py`) and a 20-generation stub dry run (3 groups of 4) whose check confirms the mechanism actually acted (`scripts/phase4_dryruns.py` → `docs/results/phase4-dryruns.json`, all 11 checks pass).

| mechanism | class (config name) | known-answer unit test | dry-run check (20 gens) |
|---|---|---|---|
| islands with migration | `Islands` topology (`islands`, migration_rate, interval) + `RandomMigration` (`random`) | swaps conserve agents and every group's size over 40 gens | migrations at gens 5, 10, 15, 20; 24 moves; sizes constant |
| best-to-neighbor migration | `BestToNeighbor` | each group's best moves one step round the ring | 30 moves, sizes constant |
| Bernoulli delivery | `Bernoulli` (`bernoulli`, p) | rate within 0.015 of p over 10,000 draws (phase 1) | delivered 365 of 720 = 0.507 |
| critical social learning | `CriticalSocialLearning` (theta, n) | adopts with 0 games; reverts if below theta or below the pre-adoption artifact | 662 provisional adoptions, 240 rechecks, 116 reverts |
| lineage-decay credit | `LineageDecay` (gamma) | direct teacher full share, grand-teacher gamma × share on a synthetic DAG | 72 credit events to non-direct teachers |
| datamodel-regression credit | `DatamodelRegression` (lam, window) | 5 planted teachers (+3, 0, +1, 0, −2 per message) recovered within 0.15 under Bernoulli(0.5) delivery | 20 refits, 12 teachers in the last fit |
| softmax_floor allocation | `SoftmaxFloor` (T, eps, by) | sums to budget, floor eps/G respected | sums and floor hold every generation; uneven in 18 of 20 |
| nash_relative allocation | `NashRelative` (eps) | max-entropy Nash averaging: rock-paper-scissors gives uniform, a dominant group takes the mass, **a cloned group splits rather than doubles its share** (softmax rewards the clone) | sums and floor hold; dominant group gets 15.6 of 18 |
| shinka_weighted selection | `ShinkaWeighted` (lam, cap) | weights equal sigmoid(lam (s − median)/MAD) / (1 + children); sampling frequencies match within 0.03 | 227 revisions from a non-incumbent parent |
| hgm_clade_ts selection | `HGMCladeTS` (alpha, tau) | clade success/failure counts, Thompson sampling prefers the productive clade, UCB-Air widening threshold at the right step | 31 revisions, 209 agent-steps skipped by widening |
| variant switch | `VariantSwitch` (at_generation, variant) | rules change exactly at the switch; non-HLE fields rejected | hand size 5 → 4 (with 6 clue tokens) at gen 10; anchors and stub bots keep playing (pop. mean 15.9 → 15.5) |

The phase-1 options `merge_llm` adoption and `costly` teaching cost now also have an end-to-end loop test (a Random receiver merges Piers through one merge call).

Design notes and deviations:
- `datamodel_regression`'s parameter is `lam` (`lambda` is a Python keyword). Its regressors are teacher indicators pooled over students and generations, with a ridge fit refit each generation over a `window` of generations. Credit is beta_T × (messages from T delivered in the window). It is intent-to-treat: the receiver's adoption decision is part of the effect.
- `nash_relative` is implemented as **Nash averaging** (Balduzzi et al. 2018), with share = eps/G + (1 − eps) × the max-entropy Nash mixture of the group-vs-group meta-game. The meta-game payoff is the mean sign of the paired per-deal score difference between group-best artifacts. The spec names the rule but not its formula, so this is my reading, chosen because it is the cloning-robust rule that queue item 8 tests. **With eps = 0.2 it starves non-dominant groups** (each got 1.2 of 18 calls and skipped 160 calls in 20 generations). For real runs a larger eps or a temperature is probably needed.
- `hgm_clade_ts` maps HGM's "evaluate instead of expand" branch to skipping the LLM revision (re-evaluation is free here). With alpha = 0.6, counting one step per agent per generation, this is very conservative: about 13% of agent-steps made a revision. Under matched budget that may be the point; under matched generations it starves search. **Open question below.**
- `islands` uses its own `migration_rate`/`interval` through `random` migration when `migration` is left at `none`. Migration swaps agents so group sizes stay fixed.
- Critical social learning keeps the pre-generation incumbent as the revert target when several messages are adopted in one generation, and rechecks after the agent's revision in the same generation.

Tests: **94 passed** (82 + 12 in `test_org_phase4.py`).

## Phase 5: section-10 analysis toolkit (done)

Each tool is in `src/culture/analysis/`, tested on synthetic data with a planted known answer (`tests/test_analysis_phase5.py`), and applied to the phase-2 dry-run logs in `notebooks/analysis.ipynb`.

1. **Higher Criticism with Efron's empirical null** (`hc.py`). The null N(δ0, σ0²) is estimated from the centre of the z-scores; the default is robust (median, IQR/1.349), with an optional truncated-normal MLE. Then HC+ runs on the standardized one-sided p-values, against a Monte-Carlo threshold for the same procedure under the null. Also included: a Meinshausen-Rice lower bound on the fraction of agents that improved, a plateau detector (HC under threshold for k generations), and paired z-scores from games × agents matrices. Known answers:
   - Inflated null (δ0 = 0.4, σ0 = 1.5, as correlated agents produce): the estimate is within 0.15 of the truth by both methods.
   - Pure noise: theoretical-null HC fires on more than 80% of reps; empirical-null HC stays near nominal (below 15% at level 0.05 with Monte-Carlo slack).
   - 5% planted improvers (shift of 4 null sds): detected in more than 90% of reps; fraction estimate between 1.5% and 7% (it is a lower bound).
   - Deviation: the unbounded MLE drifted badly (δ0 = −134 on one draw), so it is now bounded and robust is the default.
2. **DCMM on the teaching graph** (`graphs.py`). Mixed-SCORE: K leading eigenvectors, SCORE ratios, vertex hunting (k-means denoising plus successive projection), barycentric memberships with the b1 correction. Outputs: θ (influence), P (community matrix, unit diagonal, off-diagonal = cross-lineage leakage), membership over time with Hungarian alignment across windows, and NMI against the configured groups. Known answers:
   - Planted 600-node DCMM graph, K = 3, 30% mixed nodes: pure-node accuracy above 0.95 (1.0 observed), mean membership L1/2 error below 0.2 (0.11 observed), θ correlation above 0.8 (0.91 observed), and the most-connected pair of lineages shows the most leakage. Absolute leakage is underestimated (0.08 vs 0.15 planted).
   - Planted lineage merge (two lineages that stop being separate halfway): NMI with the original groups drops from above 0.8 to more than 0.4 lower.
   - Disconnected graphs (isolated groups) break Mixed-SCORE's Perron-vector ratios: θ came out 0 for every node outside one component, and leakage was meaningless. The fit now regularizes (A + τ·d̄/n, τ = 0.25) when the graph has more than one component. Known answer: planted isolated groups give NMI 1.0, every θ above 0.3, leakage below 0.15 (truth 0).
3. **IF-PCA strategy counting** (`ifpca.py`, `behavior.py`). KS departure-from-normality score per feature, a Monte-Carlo null, the HC threshold for selection, then PCA, with K from the Marchenko-Pastur edge and k-means. The behavioural fingerprint is the move each artifact makes on a fixed probe set of 200 observations from anchor games, plus pairwise action disagreement. Known answers:
   - 3 planted clusters differing in 40 of 2,000 features: K = 3, accuracy 1.0, no uninformative feature selected.
   - With 8,000 features, selection beats PCA on all features.
   - Below the detection regime (shift 1.5 on 30 features) it selected 5 features and estimated K = 2, a limit to keep in mind for small populations.

Applied to the phase-2 dry-run logs (`notebooks/analysis.ipynb`, stub, isolated topology):
- HC with the empirical null detects improvement in 23 of 199 generations. With 12 agents per generation power is low, and stub hill climbing mostly yields ties.
- DCMM recovers the configured groups exactly (NMI 1.0, leakage 0.06, no cross-group adoptions to find): idea-communities coincide with groups, the "no transfer" pattern the spec predicts for isolated groups. Top influence: g0a1 caused 77 adoptions.
- IF-PCA on the behavioural fingerprints of the 295 distinct incumbents of the last 50 generations (200 probes, 800 binary features) keeps 54 features and counts **K = 6** strategies. Mean pairwise action disagreement is 0.115. On binary features the KS screen degenerates to "keep features on which artifacts disagree", so treat K as indicative.

Tests: **103 passed** (94 + 9 in `test_analysis_phase5.py`).

## Phase 6: OpenEvolve adapter (stub-tested) and the hanab.live round trip (done)

**OpenEvolve adapter** (`src/culture/baselines/openevolve_adapter.py`, OpenEvolve 0.4.0 from PyPI, optional extra `baselines`):
- `evaluate(program_path)` is OpenEvolve's evaluator contract, implemented with our sandbox and protocol: self-play on a fixed shared seed set, anchor cross-play, and cross-play with a population snapshot fetched out of band from `CULTURE_OE_SNAPSHOT`. Every external loop scores one program in isolation, which is why the snapshot comes in from outside. `combined_score` = self-play mean.
- `StubOpenEvolveLLM` plugs our stub backend into OpenEvolve through its `init_client` hook, so the whole OpenEvolve loop runs with zero API calls, and every call lands in our cost ledger.

Stub test: 6 OpenEvolve iterations from IGGI made 6 stub calls (46k fake tokens), and each child was scored by our evaluator (0.25, 16.3, 16.2, 0.0, ...). The best stayed IGGI (16.3 on 20 games). Tests check the evaluator contract (Piers about 17, a sandbox violation is `valid = 0`, the snapshot is used) and that the OpenEvolve run completes with ledgered stub calls.

Deviations:
- The evaluator is passed to OpenEvolve as a file path, not a Python callable. OpenEvolve serializes callables by source text into a temp module for its worker processes; a module path is the robust equivalent.
- OpenEvolve runs its iterations in worker processes, each with its own stub client and call counter, so two workers can issue an identical (prompt, seed tag) pair and get identical stub answers. Harmless for a stub test; a real backend would make the seed tag include the iteration id.
- ShinkaEvolve and the generational control are not built (spec: day 2).

**hanab.live round trip** (`tests/test_hanablive.py`): 100 games across four bot pairings (Piers/Piers, IGGI/Piers, Flawed/Flawed, Random/Simple), each exported to hanab.live JSON (format 3.0.0, No Variant), written to disk, reloaded and re-simulated by the independent pure-Python rules in `game/hanablive.py`. Every game gives the same final score and the same number of turns. Further checks: deck has 50 cards with correct copy counts; card targets are deal-order indices; clue targets are player indices; a corrupted replay (a clue to oneself) is rejected. I did not paste an export into the hanab.live website (no browser in this run); the format follows `misc/example_game_with_comments.jsonc` from the Hanabi-Live repository.

Tests: **111 passed** (103 + 6 hanablive + 2 baselines), about 2 minutes on this machine.

## Fix round 1 (2026-10-06): review defects and three new pieces

All steps are commits on `enrico` prefixed "Fix round 1, step N:". Each of steps 1 to 13, 15 and 16 has a regression
test that fails on the code before the step and passes after (checked by running the new test file against the
previous commit in a throwaway worktree); step 14 is results only. Suite after step 12: 183 passed, 1 skipped; after
step 16: **205 passed, 1 skipped** (the skip is the opt-in live MLX test, `CULTURE_LIVE=1`).

| step | what changed | guarded by |
|---|---|---|
| 1 sandbox escape | New `bots/sandbox.py`. Static checks reject dunder names and attributes (short allowlist), string/bytes constants containing `__` (docstrings and `"__main__"` excepted), frame/introspection attributes (`gi_frame`, `f_back`, `f_locals`, `tb_frame`, `mro`, ...), `str.format`/`format_map` except on a literal template without attribute fields, stores into imported modules, private names in imports and class patterns. A token scan marks any artifact mentioning `sys`, `_os`, `_getframe`, `__subclasses__`, `__globals__`, `__builtins__` invalid. Imports return read-only proxy modules with only public, non-module attributes (`collections.abc` proxied as a submodule); `typing.get_type_hints`, `functools.singledispatch(method)`, `random.SystemRandom` removed; `functools.wraps/update_wrapper/total_ordering` and `dataclasses.dataclass/make_dataclass` replaced by checked versions. `getattr/hasattr/setattr` reject dunder, frame and format names, and `_x` names unless the attribute belongs to a class the bot defined; `setattr` only writes to bot-owned objects. Every `obj._x` in the source is rewritten to a run-time check of the same rule (so `self._cache` works and `typing.ForwardRef(...)._evaluate` does not). Each seat gets its own copy of the game description. | `tests/test_sandbox.py`: the four demonstrated escapes fail; 14 run-time and 10 static variants fail; a spy bot's view of its own hand is all `None` while the engine holds real cards; a bot using namedtuples, dataclasses, `functools.wraps`, private attributes on itself, `super()._helper()` still runs clean |
| 2 swallowed timeout | `HardTimeout` stays a `BaseException`, so `except Exception` cannot catch it; the source check rejects bare `except:` and `except BaseException`; the alarm sets a fired flag and re-arms a 10 ms repeating timer, and a call that returns after the alarm fired counts as a timeout. `evaluate/pool.py` adds a wall-clock watchdog (`future.result(timeout=...)`): an overrunning chunk returns broken results for its seats, its workers are killed, and the pool is rebuilt. | `tests/test_timeout.py`: the review's looping bot is rejected and counted; an `except Exception` loop is stopped within about 2x the limit and counted; a C-level loop that ignores SIGALRM (`sum(range(10**9))`) is stopped by the watchdog in about 1.5 s and the pool keeps working |
| 3 soft budget cap | The budget is a per-generation pool per group. A call reserves its estimated cost before it is sent (1 call; or prompt estimate plus `max_tokens` tokens; or their dollar cost) and is refused if that exceeds the pool. Refusals go to the ledger with the reason and are excluded from call and usage totals. Teaching cost debits the same pool. | `tests/test_budget.py`: a group of 2 with `per_group_per_generation=1` makes exactly one revise call per generation; the other agent's attempt is refused and logged |
| 4 HGM labels | Each candidate is recorded with its parent and the parent's self-play on the same generation's deals (re-scored if the parent is not an incumbent). Label: success if child minus parent > margin (default 0), failure if < -margin, **no label on a tie**. Clade counts include the node's own outcome. Records carry `parent_score`. | `tests/test_hgm_paired.py`: a behaviourally identical candidate is never labelled on any of four generations (under the old rule its label depended on which generation's deals the parent was scored on); run labels equal the sign of the paired difference. The phase-4 HGM test moved to the paired API |
| 5 warm-start race | `run/warmstart.py` authors generation 0 once per population seed into a set keyed by a hash of everything that changes authoring (name, experiment seed, population, game, sandbox limits, LLM settings, prompt hashes), built in a temp directory and renamed into place. `run_experiment` builds the sets first, then forks every condition from them (`population.warm_start_set`); a forked run verifies key and content hashes and makes no generation-0 backend call. | `tests/test_warmstart.py`: three conditions × two seeds have byte-identical generation-0 incumbents, zero generation-0 author/repair calls in any condition's ledger, and the set's ledger holds one author call per agent |
| 6 cross-play order | Pair jobs are canonical: lower key in seat 0 on even seeds, higher key on odd seeds. | `tests/test_crossplay_symmetry.py`: `crossplay(A,B) == crossplay(B,A)` game for game; the reversed request plays zero new games |
| 7 bot seed | `bot_seed = sha256(salt, game seed, seat) mod 2^31-1`; the module-level RNG is reseeded with the seat -1 hash, not the deal seed. | `tests/test_bot_seed.py`: identical across processes and `PYTHONHASHSEED` values; no constant stride, not the old formula |
| 8 real spend | Ledger totals, generation records and the run report add `real_spend_usd` = new spend excluding rows flagged `replayed_within_run`. | `tests/test_real_spend.py`: replaying a run's own calls under the same run id reports the full curve but 0 real spend |
| 9 local backend | `llm/openai_compat.py`: OpenAI chat-completions over stdlib urllib to `http://127.0.0.1:8080/v1`, model `mlx-community/Qwen3.6-35B-A3B-4bit`; body = model, messages, max_tokens, temperature (0), seed, stream false, plus `extra_body`; timeout 1200 s; usage from the response; cost $0. The backend's seed, temperature and extra_body enter the cache key (stub keys unchanged). The ledger records each call's latency. | `tests/test_openai_compat.py` with a fake HTTP server: request body, usage, extra_body, a different seed misses the cache, clear error when the server is down, one revise through the full agent step |
| 10 deal separation | New `feedback` seed purpose; per generation the feedback, verification and evaluation deals are disjoint (at most 100,000 games per purpose). Failure traces shown to the LLM come only from the parent's lowest-scoring feedback games. Every generation record has `generalization`: per incumbent and candidate, evaluation score, feedback score and `generalization_gap` = evaluation minus feedback. `evaluation.feedback_games` (None = min(selfplay_games, 40); 0 = no traces). | `tests/test_deal_separation.py` |
| 11 null calibration | Stub `mode: null`: a revision returns the parent's rule list with a fresh RNG salt (same policy, new random stream). `analysis/null.py` builds per-generation bands over K null populations by conformal order statistics; with K < 39 it falls back to the [min, max] envelope and says so. `figures.between_vs_tokens` and `score_vs_generation` take `null=`. `scripts/null_calibration.py` → `docs/results/null-calibration.json`. | `tests/test_null_calibration.py`: conformal coverage 0.92-0.98 on synthetic draws; null revisions keep the rule list; the band contains the null runs' own metrics |
| 12 innovation base rate | `analysis/innovation.py`: epsilon per revise context (nothing / feedback_only / feedback_plus_received) with a Beta(1,1) posterior and 95% credible interval; outcomes are paired held-out candidate-minus-parent differences against a declared margin. Records carry `revise_context`. Wired into `notebooks/00_probe.ipynb`. | `tests/test_innovation.py`: planted rates 0.05/0.2/0.5 recovered within 0.02 with covering intervals |

### Results produced in this round

- **Live MLX check (step 9)**, `docs/results/live-mlx-revise.json` and the model's bot `live-mlx-revise-candidate.py`.
  One agent started from IGGI and made one revise call through the full agent step: **8,110 input and 2,399 output
  tokens in 45 to 63 s** (three runs; about 50 output tokens/s with an 8k-token prompt). The code was **admissible**
  (passed the source checks and the smoke game, no repair call), but **48% of its moves raised
  `UnboundLocalError`**, so it scored 0.0 against its parent's 14.9 on the held-out deals and was not accepted. The
  sandbox did not cause this: run as plain Python with no sandbox, the same code raises on 66 of 138 moves. Note: admission only
  rejects a bot whose every smoke-game move fails, so such bots reach evaluation, which then rejects them.
  **Reproducibility**: the identical revise request sent twice gave bit-identical text, and three separate processes
  produced the same token counts and the same bug. Ollama was not used (the coordinator measured it as
  non-reproducible for this model).
- **Null calibration (step 11)**, K = 40 null populations of `configs/null.yaml` (2 groups of 2, 10 generations,
  in-group teaching with self-play verification), stub only, 336 s, guaranteed per-point coverage 0.951. Medians at
  generation 1 → 9: population best 15.6 → 16.5, population mean 14.7 → 15.3, between-group cross-play 14.3 → 14.2,
  HC statistic 0 (95% band up to about 1.1), retained innovations 0 in all 40 runs. The bands are wide (population
  best 5.9 to 17.3) because warm starts differ across population seeds. **This null removes innovation, not
  transfer**: copying a group's best warm start still raises the population mean, so a real run should be read
  against this band for innovation and against a no-teaching null for transfer.
- **Innovation probe (step 12)**, stub, margin 0.5, two population seeds per context: 1 of 30 revisions improved in
  each context (epsilon 0.06, 95% CrI 0.01 to 0.17). The stub ignores context, so equal rates are the expected
  answer here; the contexts were checked to be applied and the per-revision outcomes differ.

### Deviations and notes

- Step 1 goes beyond the review's list. Each extra rule closes a path found while fixing the listed ones: frame
  walking through generator `gi_frame` (no dunder needed), `str.format` attribute traversal, `typing.get_type_hints`
  and `functools.singledispatch` evaluating strings with the real builtins, `functools.update_wrapper` copying
  arbitrary attributes, dataclass field names injected into generated code, and `ForwardRef._evaluate` reached
  through a single-underscore attribute (hence the run-time private-attribute guard). This is still not a
  security boundary against a determined adversary; the watchdog and process isolation are the backstop.
  Residual: a bot can still monkeypatch a shared standard-library class through an alias (`R = random.Random;
  R.choice = f`), which would leak into other bots in the same worker process.
- Step 1 legitimately rejects some LLM idioms: string constants containing `__`, `str.format` on a non-literal,
  `obj._x` on objects the bot did not define, `except:`. The system prompt now states the rules, and a rejected
  artifact gets one repair call with the reason.
- Step 3 changes the budget from equal per-agent shares to a group pool drawn in agent order. Within a generation the
  first agents in sorted order spend first.
- Step 8: on a genuine crash and resume the ledger rows of calls made before the crash are truncated, so those calls
  (paid once) appear only as replays and are not in `real_spend_usd`. `real_spend_usd` is exact for the replay case
  the review named.
- Step 9: stdlib urllib, not the `openai` package (no new dependency; the body is exactly what the code writes). There
  are no Ollama `options`/`think` fields, per the coordinator's change: thinking is disabled when the MLX server
  starts (`--chat-template-args '{"enable_thinking": false}'`). The same backend talks to Ollama's `/v1` route through
  `extra_body`, but it is not relied on for determinism.
- Step 10: when `feedback_games` was 40 for every config, the 1,000-generation test went from about 30 s to 290 s,
  hence the scaled default.
- Fallout handled along the way: the stub fixture `tests/fixtures/llm_cache_smoke` was re-recorded after steps 1, 6, 7
  and 10 (prompt or game changes). `test_analysis`'s crossing check now uses a data-derived threshold (a fixed
  constant broke whenever the stub curve shifted). `test_evaluate`'s seat assertion follows the canonical rule, and the
  phase-4 HGM test follows the paired API.
- `CLAUDE.md` still lists these review defects as open and says the stub is the only backend. I did not edit it (it
  belongs to the planning session).

### Overnight extension: steps 13 to 16

| step | what was built | guarded by |
|---|---|---|
| 13 C1 driver | `culture.run.single_student` (CLI `python -m culture.run.single_student --spec ... --out ...`): one weak student (self-play 3.2) and k teachers carrying fixed artifacts (Piers with plain prose; Flawed with persuasive prose and honest low evidence; the student's own bot re-sent; null teachers beyond 3). One replay = one generation from the same student state under one delivered subset: verify/adopt per stratum, one revise call, evaluate, accept if not worse; outcome = held-out paired self-play improvement. Delivery: exhaustive (oracle), bernoulli(p, runs), pb8 (k <= 7), singles_pairs. Replicate r = experiment seed + r and sampling seed r (in the cache key). Writes `replays.jsonl` (with the revise request hash) and `summary.json`; re-runs skip finished replays. `analysis/credit.py`: exact Shapley, ITT tau (Banzhaf), adopter effect (undefined when never adopted), leave-one-out, equal split, ridge on Bernoulli(1/2) delivery pooled across seeds, singles-plus-pairs, Plackett-Burman; RMSE, Spearman, sign error on the sabotaged teacher, replays per seed; IQM with bootstrap intervals; the primary contrast with a noncentral-t power calculation; figure and table. | `tests/test_credit_estimators.py` (efficiency; hand-computed game; additive and second-order games; PB8 orthogonality; metrics; power), `tests/test_single_student.py` (with revision off the oracle recovers the planted engine effects exactly in both strata; the full stub loop; resume) |
| 14 C1 pilot | `configs/c1_pilot_stub.yaml`, `configs/c1_pilot_mlx.yaml`, `scripts/c1_report.py` → `docs/results/c1-pilot.json`, `c1-pilot-{stub,mlx}.png` and `-table.md` | results only (pipeline tested in step 13) |
| 15 B1 | `analysis/sprt.py` (fixed N with the current `mean > 0` rule and a one-sided t-test; Wald SPRT with online sigma; operating characteristics; the decision rule), `scripts/b1_sprt.py` → `docs/results/b1-sprt.json` and `.png` | `tests/test_sprt.py` |
| 16 D1 pilot | `sabotage` config (epsilon, payload, text) applied after delivery with common random numbers across epsilon; messages carry `sabotaged`; `analysis/breakdown.py`; `scripts/d1_pilot.py`, `configs/d1_pilot.yaml` → `docs/results/d1-pilot.json` and `.png` | `tests/test_sabotage.py` |

#### C1 pilot (step 14): k = 3, exhaustive delivery, 5 seeds, both strata, 300 evaluation deals

**MLX** (Qwen3.6-35B-A3B 4-bit, temperature 0): 80 replays, 74 backend calls, 11 cache hits (a prompt identical
across strata, e.g. nothing delivered), 5 repair calls. About **10.5k tokens per call (2.6k to 2.9k output) and 54 to
56 s per call**; 1.15 hours of replay time (longer on the wall clock, because B1 and D1 shared the machine during the run). **Admissible: 39 of 40 revisions per stratum.**
Accepted: 45% (verified), 68% (unverified). **Adoption when delivered**: Piers 100% in both strata; Flawed 100%
without verification, 0% with it; the re-sent bot is always a duplicate. **Generalization gap** (held-out minus
feedback-deal score of each candidate): +0.03 (verified) and +0.13 (unverified) points, so no sign of overfitting
the shown traces at this size.

Per-seed Shapley credit [Piers, Flawed, re-sent bot] in points, with v(none) and v(all):

| stratum | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| verified | [10.2, -4.6, 1.9] | [5.1, -1.8, -1.7] | [10.4, -3.4, -2.3] | [12.6, -1.6, 3.2] | [5.2, 1.1, -4.6] |
| unverified | [2.1, -13.2, 1.6] | [3.3, -13.6, -2.7] | [4.2, -4.9, 2.4] | [6.6, -8.5, 4.8] | [5.3, -12.0, -8.4] |

v(none) (the student revising with its own feedback only) is 7.2, 11.9, 9.0, 0.0, 12.7 by seed. The ITT tau values are
within about 1 point of Shapley (full numbers in `c1-pilot.json`).

Estimators against the Shapley oracle (MLX, IQM over 5 seeds):

| estimator | replays/seed | RMSE verified | RMSE unverified | Spearman (both) | sign error on Flawed |
|---|---|---|---|---|---|
| tau (exhaustive ITT) | 8 | 0.74 | 0.71 | 1.00 | 0 |
| Plackett-Burman | 8 | 0.74 | 0.71 | 1.00 | 0 |
| ridge, Bernoulli(1/2), pooled | 8 | 2.58 | 3.92 | 1.00 | 0.2 / 0 |
| singles plus pairs | 7 | 2.95 | 2.85 | 1.00 | 0 |
| leave-one-out | 4 | 3.16 | 4.38 | 0.87 / 0.67 | 1.0 / 0 |
| equal split | 1 | 3.99 | 7.44 | 0.87 / 0.00 | 1.0 / 0.4 |

**Primary contrast** RMSE(ridge) - RMSE(leave-one-out): mean -0.79 (verified) and -0.39 (unverified); **between-seed
SD 1.85 and 0.86**; **R for 80% power at a 1-point minimum effect, two-sided alpha 0.05: 29 (verified) and 8
(unverified)**. Taking the larger, the full C1 needs about 29 seeds per stratum, under the prereg's cap of 60; at about
55 s per call that is 8 × 29 × 2 = 464 calls, about 7 hours.

What this pilot does and does not say:
- At k = 3 the Plackett-Burman design is the full factorial, so it equals tau and is not a cheaper estimator here. The
  cheap estimators (leave-one-out, equal split, pooled ridge, singles plus pairs) all miss the prereg's RMSE <= 2.0
  in this pilot. With 5 seeds this is planning information, not a verdict.
- Under verification, Flawed (always rejected) still gets negative Shapley credit in 4 of 5 seeds (-1.6 to -4.6):
  its persuasive text changes the revision. But the pure-text placebo (the re-sent bot) also moves outcomes in
  either direction, by up to 4.6 points with verification and 8.4 without. At n = 5, Flawed's text effect cannot be told apart from prompt-perturbation noise.
  Treat it as a hypothesis for the full run, not a finding.
- **Stub pilot** (same design, zero calls) ran first to prove the pipeline; its numbers are in the same JSON and only
  validate plumbing.

#### B1 (step 15): fixed-N verification versus Wald's SPRT, engine only

Bank: 70 anchor-family bots (random one- or two-rule mutations of Piers and IGGI, plus two graded families: Piers
with the `play_probably_safe` threshold 0.3 to 0.9 and the extra-lives switch, and IGGI with `play_probably_safe(t)`
for t from 0.62 to 0.95), 360 ordered pairs. Each pair's delta and sigma come from 2,000 deals; tests use 10 separate
600-deal blocks per pair. Median sigma 3.6 (the program assumed 3 to 4). 188 pairs have delta <= 0, 90 have delta >= 1,
82 lie in between. 16 minutes of engine time.

| procedure | false adoption (delta <= 0) | missed, 1 <= delta <= 1.25 | deals (all) | deals near delta = 1 |
|---|---|---|---|---|
| SPRT (alpha = beta = 0.05, delta1 = 1) | 0.3% | 10% | 40 | 109 |
| fixed N = 50, t-test | 0.6% | 50% | 50 | 50 |
| fixed N = 100, t-test | 0.8% | 24% | 100 | 100 |
| fixed N = 200, t-test | 0.4% | 12% | 200 | 200 |
| fixed N = 400, t-test | 0.4% | 0% | 400 | 400 |
| fixed N = 200, `mean > 0` (current harness rule) | 6.1% | 0% | 200 | 200 |

**Decision rule (program section B1): passes.** Of the fixed-N t-tests on the grid, only N = 400 is as accurate as
the SPRT, and the SPRT uses **73% fewer deals near delta = 1** (109 vs 400). N = 200 narrowly misses (12% vs 10%
missed), so against an interpolated N of about 230 the saving would be closer to 50%; either way it clears 30%. A side
finding: the harness's current rule (adopt if the paired mean is positive) adopts a non-improvement 6% of the time at
N = 100 to 200, and 8.8% at N = 50.

#### D1 pilot (step 16): breakdown point on the null stub

Four students, 10 generations, in-group broadcast; epsilon in {0, 0.1, 0.2, 0.3, 0.5}; verification on (selfplay
n = 200) or off; 5 seeds; 50 runs, 17 minutes, zero model calls. The null stub's revisions keep the rule list, so skill
moves only through copying and sabotage.

| epsilon | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| Y_G, verification off | 10.2 | 1.8 | 1.7 | 0.0 | 0.0 |
| Y_G, verification on | 15.7 | 15.8 | 15.7 | 15.8 | 15.9 |
| sabotaged messages adopted, off / on | 0 / 0 | 63 / 0 | 116 / 0 | 159 / 0 | 202 / 0 |

eps*(off) = 0.04; eps*(on) is censored (> 0.5); the contrast is at least 0.46 (bootstrap 95% interval [0.43, 0.47],
censored values at 0.5). The per-cell spread over seeds is the null band (min, max, median in `d1-pilot.json`).
Note: even at epsilon = 0 the unverified arm ends lower (10.2 vs 15.7), because blind adoption also takes worse honest
payloads. This pilot proves the pipeline; on the stub the outcome is close to forced, and the model run is what tests
whether verification is sufficient.

#### Deviations and notes (steps 13 to 16)

- C1 placebo ordering: the driver processes the re-sent bot (and null teachers) before the other messages, while the
  student still holds that bot, so it is a duplicate acting only through its text. Processed after an adoption,
  blind adoption would take the old bot back and the "placebo" would carry a large planted effect.
- C1 Arm 2 (k = 7, Plackett-Burman, singles plus pairs, 128-run oracle) is implemented and unit-tested on synthetic
  games but was not run.
- B1: the first two bank versions had no pairs with delta just above 1 (the random mutations change behaviour a lot or
  not at all), so the decision rule could not be evaluated; the graded families fixed that. The decision rule's first
  implementation had its "matched" inequality inverted (it compared with fixed N tests less accurate than the SPRT);
  this was fixed, unit-tested, and re-applied to the saved operating characteristics, which did not change. The JSON
  carries a note.
- D1 uses the Flawed anchor itself as the sabotage payload (not a perturbed Flawed family), with the C1 persuasive
  text.
- Housekeeping: two byte-identical `" 2"` copies of `analysis/innovation.py` and `tests/test_innovation.py` appeared
  in the working tree. The repository is under `~/Desktop`, and file sync makes such copies when a file is deleted
  and recreated quickly. I removed them (they were never committed and never counted in a reported test total). Watch
  for this when moving files in this directory.
- Test count at the end: **205 passed, 1 skipped** (the skip is the opt-in live MLX test, `CULTURE_LIVE=1`).

## Overnight 2 (2026-10-07, 01:02 to 06:04 unattended; analysis finished in the morning)

Scope: one model block only (Qwen3.6-35B-A3B 4-bit on mlx_lm.server, this machine, seeded, temperature 0), one
student, one generation, k = 3 fixed teachers, 300 held-out deals per evaluation. C1 Arm 1 ran in full at the
pre-registered R. C2 is a 5-seed pilot. A1 measured the transmission parameters only; its check populations were not
run. Nothing here is accepted for use, because acceptance needs the Gemma replication block, which has not been run.
Estimators and claims can only be rejected, or pass this block. $0 spent; no paid API calls.

Commits on `enrico`, each prefixed "Overnight 2:":
- `a970163` step 1 setup;
- `ce5a00b` step 2;
- `404a659` step 5 preparation (A1 driver and amendments);
- `5369d86` step 3, C1 analysis;
- `f0c9963` step 4, C2 pilot;
- `4b4f725` step 5, A1 parameters;
- `f374fb8` C1 exploratory population-average errors;
- this section.

### What ran where

| run | backend | design | backend calls | backend seconds | wall clock |
|---|---|---|---|---|---|
| C1 full (step 1) | MLX | k = 3, exhaustive (8 replays per seed per stratum); verified R = 29, unverified R = 8; the pilot's 5 seeds reused | 221 tonight (203 verified incl. 11 repairs; 18 unverified) plus 7 cross-stratum cache hits; 295 for all of C1 with the pilot | 10,795 tonight (about 49 s and 10.8k tokens per call) | 01:07 to 04:31 (3 h 24 min, of which about 22 min machine sleep) |
| C2 pilot (step 4) | MLX | verified_quarantine, the 5 pilot seeds, exhaustive | 11 (10 revise, 1 repair) plus 33 within-run cache hits | 538 | 04:31 to 04:40 |
| A1 parameters (step 5) | MLX | teacher Piers; media code / prose / both; 30 replicates each; one revision; verification off; no adoption | 90 (no repairs, no failed revisions) | 4,787 | 04:40 to 06:04 |
| strict cache-replay check | cache only | the 80 C1 pilot replays replayed with the current code, misses forbidden | 0 | 0 | 49 s |
| C2 on the stub (step 2) | stub | verified and verified_quarantine, 5 seeds | 0 | 0 | 83 s |
| null stub (C1 and C2 bands) | stub, `null` mode | the C1 design, K = 20 seeds, strata verified, unverified, verified_quarantine (480 replays) | 0 | 0 | about 8 min |
| A1 on the stub | stub | the A1 design (90 copies) | 0 | 0 | 7 min |

MLX total tonight: **322 backend calls, 16,119 s of model time (4.5 h)**. Revisions admissible: 97% verified, 98%
unverified, 100% in C2 and A1.

### New code (step 2 and supporting)

- **`org.quarantine_unverified`** (config; default off; left out of the config digest while off, so older configs keep
  their digests).
  - With it on, a received message is shown in the revision prompt only if the receiver's engine verification ran
    and passed. Rejected, duplicate and unverified messages are withheld entirely: prose, evidence and verification
    line.
  - Applies in the generation loop and in the single-student driver. A revision's teaching sources list only the
    messages it was shown.
  - The re-sent own bot is a duplicate, never verified, so quarantine withholds it too.
  - Tests in `tests/test_quarantine.py` (6):
    - with quarantine on, the revise prompt of a student that rejected a message contains none of that message's
      text; with it off, it does;
    - the same holds in the generation loop under sabotage;
    - the config flag and digest behave as described;
    - the C2 analysis gives known answers on a synthetic game.
- **Single-student driver.**
  - The `replicates` setting can give a different seed list per stratum.
  - Strata can be named from a catalogue that includes `verified_quarantine`.
  - Each replay records which senders were withheld.
- **`analysis/quarantine.py`, `scripts/c2_report.py`**: C2's paired quantities, intervals, implied R and the literal
  decision rules.
- **`culture.run.transmission`, `scripts/a1_report.py`**: the A1 driver and its report. Tests:
  `tests/test_transmission.py` (2).
- **`scripts/run_supervised.py`**: reruns a resumable command after a crash, and restarts the MLX server if
  `/v1/models` stops answering.
  - Long runs started from frozen copies of `src/culture` (`runs/c1-full/_code`, `runs/chain/_code`), so editing
    code during the night could not reach the workers they spawn.
  - Model calls ran strictly one at a time (`runs/chain/chain.sh`). Concurrent requests could be batched on the
    server, which would break bit-reproducibility.
- **`scripts/c1_full_report.py`**: the C1 analysis, figure and table.

### C1 verdicts (files/prereg/C1-credit-estimators.md section 8, applied literally)

Rule: accept an estimator if, in both model blocks, all three hold:
- Spearman IQM >= 0.9 with bootstrap lower bound >= 0.8;
- RMSE IQM <= 2.0 points;
- sign error on T2 <= 5%.

An estimator must pass in both strata. The pre-registration was silent on this; the planning session confirmed it
on 2026-10-07, pending Enrico.

Metrics are IQM over seeds, with a 95% bootstrap over seeds within each stratum. Sign error is counted over the
seeds where |oracle| >= 0.5 (26 of 29 verified, 8 of 8 unverified).

| estimator | role | replays/seed | RMSE verified | RMSE unverified | Spearman (lower bound) | sign error on Flawed | verdict |
|---|---|---|---|---|---|---|---|
| (i) leave-one-out | confirmatory | 4 | 2.83 [2.54, 3.23] | 4.50 [3.45, 5.32] | 0.87 (0.84) / 0.50 (0.50) | 88% / 0% | fails (all three rules verified; Spearman and RMSE unverified) |
| (ii) equal split (paired_delta v0) | confirmatory | 1 | 3.49 [2.99, 4.00] | 7.22 [6.40, 7.97] | 0.87 / 0.00 | 100% / 50% | fails |
| (iii) ridge on Bernoulli(1/2) delivery, pooled | confirmatory | 8 | 2.28 [1.98, 2.52] | 3.92 [3.36, 4.28] | 1.00 (0.90) / 1.00 (0.88) | 19% / 0% | fails (RMSE in both strata; sign error verified) |
| exhaustive ITT (tau) | exploratory | 8 | 0.67 [0.53, 0.81] | 0.71 [0.27, 1.28] | 1.00 / 1.00 | 4% (1 of 26) / 0% | passes this block in both strata |
| Plackett-Burman 8-run | exploratory | 8 | identical to tau (at k = 3 it is the full factorial) | | | | passes this block |
| singles plus pairs | exploratory (pre-registered for Arm 2) | 7 | 2.67 [2.11, 3.26] | 2.83 [1.08, 5.11] | 1.00 / 1.00 | 23% / 0% | fails |

- **Overall: no confirmatory estimator passes.** The only estimator that passes is exhaustive replay, which costs as
  much as the oracle at k = 3.
- **The section 8 fallback is not formally triggered.** "Causal credit is not available at this cost" is conditioned
  on R = 60; we ran the power-rule R (29 and 8).
- **More seeds would very likely not change these verdicts.**
  - Leave-one-out and equal split do not pool, so their per-seed RMSE does not shrink with R.
  - Any pooled estimator has a per-seed RMSE floor here of 2.19 (verified) and 3.20 (unverified): the oracle's own
    seed-mean scores that, because the true credit varies between seeds.
  - None of 200 alternative random Bernoulli draws for the ridge passes.
- **Primary contrast**, RMSE(ridge) − RMSE(leave-one-out), paired two-sided t-test:
  - verified: −0.69, 95% interval [−1.11, −0.27], p = 0.0023;
  - unverified: −0.57 [−1.83, 0.69], p = 0.32.
  - With a refitted-ridge bootstrap: [−1.09, −0.32] and [−1.73, 0.19].
  - Ridge is better than leave-one-out, by less than the 1-point minimum effect the design was powered for.
- **Multiplicity** (planning session ruling, pending Enrico):
  - BH at q = 0.1 applies only to the tests that have p-values: the two primary contrasts. The adjusted p-values are
    0.005 (verified, survives) and 0.32 (unverified).
  - The section 8 thresholds are acceptance criteria, not tests.
  - The Flawed-versus-placebo contrast below is exploratory: it is reported with its p-value but is outside the BH
    family.
- **Effects per teacher** (ITT tau, IQM [95% interval]):

  | stratum | Piers | Flawed | re-sent own bot |
  |---|---|---|---|
  | verified | +10.0 [9.1, 10.8], adopted every time | −2.2 [−2.9, −1.4], never adopted, so the adopter effect is undefined | +0.7 [−0.5, 1.6] |
  | unverified | +4.8 [3.2, 6.1] | −10.6 [−13.0, −6.3], adopted every time (adopter effect = ITT) | −0.3 [−4.3, 2.4] |

  Shapley IQMs are within 0.9 of tau.
- **Exploratory: rejected text still hurts.** With verification on, Flawed's effect minus the placebo's is −2.44
  [−3.69, −1.19], p = 0.0004 (29 seeds).
  - So a rejected message's prose changes the student's revision more than a generic text change does. This
    resolves the pilot's open question: it is not prompt-perturbation noise.
  - Piers × Flawed interaction (verified): +3.76 [2.15, 5.37]. Adopting Piers partly undoes the harm of Flawed's text.
- **Null stub (K = 20).**
  - With verification on, leave-one-out and equal split are near-exact when the model adds no text effect (null RMSE
    0.08 and 0.10). Their failure on MLX therefore comes from the model's text effects.
  - Without verification they fail even on the null (5.8 and 6.9). That failure is structural: under blind adoption
    the last adopted bot wins, which leave-one-out cannot see.
- **Exploratory, requested by the planning session for a possible C1b.**
  - A pooled estimator targets the population-average credit E_seed[tau_j], not each seed's tau_j.
  - Against that target (RMSE over teachers, seed bootstrap refitting the ridge):
    - ridge: 0.47 [0.17, 1.10] verified, 1.28 [0.57, 2.36] unverified;
    - seed-averaged leave-one-out: 1.43 [0.81, 2.21] and 2.41 [0.96, 4.68];
    - seed-averaged equal split: 2.98 and 6.41.
  - Not a verdict; it is in `strata.*.exploratory_population_average`.
- **Operations.**
  - Accepted candidates: 44% verified, 70% unverified.
  - Generalization gap (held-out minus feedback-deal score): +0.11 and +0.05.
  - The student's own revision alone (v(none)) ranges from 0 to 13.4 by seed; it was rejected outright on 9 of 29
    seeds.
- **Files.** `docs/results/c1-full.json` (scope line first), `c1-full.png` (estimate against oracle for every
  estimator and stratum, identity line, null-stub envelope), `c1-full-table.md`, and ledger row C1.

### C2 pilot (step 4; files/prereg/C2-quarantine.md)

- **Not judged.** It is a pilot: the decision rules need the full R and two model blocks.
- **The primary quantity is mechanical at temperature 0.**
  - Quarantine withholds every message that did not pass verification, so the revision prompt depends only on the
    delivered messages that passed. Flawed never passes.
  - Under quarantine the 8 delivery subsets collapse to two distinct prompts per seed (nothing delivered, and Piers
    only). On every seed, v(S) equals v(S restricted to Piers), with zero deviation.
  - So tau_flawed(quarantine) = 0 by construction, and the rule "tau_flawed(quarantine) IQM in [−1, 1]" cannot fail.
  - Primary = tau_flawed(quarantine) − tau_flawed(verified) = −tau_flawed(verified): IQM 2.75 [−0.53, 4.36] on 5
    seeds. The implied R for 80% power at the 2-point minimum effect is 16.
  - Projection, not a C2 run: on C1's 29 verified seeds the same quantity is 2.23 [1.35, 2.94].
- **The empirical content is C1's Flawed-versus-placebo contrast** above: a rejected message's text lowers the
  student, and quarantine removes that channel by construction.
- **Secondary** (Piers under quarantine minus under verification): −2.67 [−5.59, 0.85].
  - This is not a loss to the student. Piers' ITT under verification includes its interaction with Flawed's text,
    which quarantine removes.
  - Exploratory check of the student's own outcome: with all three messages delivered, quarantine equals
    verification on 4 of 5 seeds and is +0.79 on the fifth.
  - Averaged over the 8 subsets, quarantine is +1.27 (IQM 1.53 [−0.54, 2.79]).
- **Cross-session reproducibility.**
  - C2 ran under a new run name, so no C1 cache entry could be reused.
  - Its 10 fresh calls (nothing delivered and Piers only, per seed) produced the same candidate artifacts and the
    same scores as C1's calls hours earlier.
- **Null stub band:** primary [−0.12, 0.16], secondary [−0.43, 0.27].
- **Files.** `docs/results/c2-pilot.json` and ledger row C2.

### A1 parameters (step 5; files/prereg/A1-transmission-fidelity.md sections 2, 3 and the section 8 amendments)

Piers scores v_T = 16.7 on the common 300 deals, and the student's incumbent scores 2.5.

| medium | copy loss alpha | Gumbel scale beta | rho = cross-play / v_T | delta(N = 4) | rule-list identical to Piers |
|---|---|---|---|---|---|
| code | 1.66 [0.49, 3.15] | 2.89 [1.13, 4.09] | 0.94 [0.88, 0.98] | +4.02 [1.57, 5.23] | 12 of 30 |
| prose | 1.82 [1.00, 2.77] | 1.95 [0.88, 2.76] | 0.94 [0.90, 0.97] | +2.01 [0.46, 3.00] | 0 of 30 |
| both | 1.19 [0.51, 2.25] | 2.01 [0.65, 3.21] | 0.96 [0.92, 0.98] | +2.75 [0.44, 4.35] | 1 of 30 |

- **Medium-ordering rule** (claim "code transmits with less loss than prose" if alpha(prose) − alpha(code) >= 1
  with the interval excluding 0): 0.16 [−1.51, 1.75]. **Not met; no claim.**
- **Model-support rule: not judged.** The check populations were not run.
- **The predicted sign at N = 4 is positive in every medium.** Caveat: the copy scores are a point mass near v_T
  (copies that behave exactly like Piers) plus a heavy left tail of copies near the incumbent (2 to 5 points).
  - beta comes from the SD, so that tail inflates beta, and with it delta. The left-tail copies are ones selection
    would discard.
  - Read delta as an upper bound until the check populations test it.
- **Copy quality.**
  - Prose copies had a 5% illegal-move rate, against 0% for code copies and 0.6% for both.
  - Some copies score 17.6, above Piers.
- **Files.** `docs/results/a1-params.json` and ledger row A1.

### Amendments (logged in the pre-registration files, with times)

- **A1-1** (2026-10-07 01:34, before any A1 call). At temperature 0 a sampling seed does not change the output, so
  30 sampling seeds on one prompt would give identical copies (beta = 0 by construction).
  - Replicate r instead uses experiment seed 300 + r, so the feedback traces in the prompt differ, as in C1.
  - All copies, Piers and the incumbent are scored on one common set of 300 deals (experiment seed 299).
  - beta is therefore dispersion across prompt contexts, not sampling noise.
  - The planning session judged this sound; **it stays flagged for Enrico's sign-off.**
- **A1-2** (same time). "Verification off" is implemented as verification none plus adoption `never`.
  - The cover note carries no content and is identical across media.
  - A failed revision leaves S' = S0, and parameters are also reported over admissible copies only. There were no
    failures, so the two coincide.
- **A1 scope note.** Student and teacher share one rule-bot template, and Piers' conventions name rule functions that
  exist in the student's code. Copying is easier here than between unrelated codebases.
- **No amendment to C1 or C2 tonight.** No decision rule was changed.

### Deviations and notes

- **C1 reused the pilot.** The full config keeps the pilot's run name, because the name enters the LLM cache key.
  Before the pilot's 80 rows were copied in, a strict cache replay with the current code reproduced all 80
  bit-identically (scores, adoptions, acceptances and request keys).
- **The ridge draw depends on stratum order.** The ridge's Bernoulli subset per seed is seeded with the stratum's
  index in the sorted `spec.json` (unverified 0, verified 1), as in the pilot report. The verdict does not depend on
  the draw (0 of 200 draws pass).
- **Counting fix in the C2 report.** A cache hit on an entry recorded by the same run id is ledgered as not cached
  and carries no latency (the crash-resume convention in `llm/cache.py`). The first version of the report counted
  those as backend calls; the committed version counts them as within-run hits. C1 and A1 counts are unaffected:
  every one of their calls has a latency.
- **Two idle-sleep gaps.** The Mac idle-slept 03:50 to 03:56 and 03:56 to 04:12 EDT after the display turned off,
  while a C1 request was in flight.
  - The request completed after wake. Latencies exclude the sleep (`perf_counter`).
  - Outputs were unaffected; about 22 minutes of wall clock were lost.
  - `caffeinate -i -s` was started at 04:14 for 4.5 h.
  - Unattended runs on this machine need `caffeinate` from the start.
- **One model stream at a time.** Steps 4 and 5 started automatically, in sequence, after C1. A1 began at 04:40 with
  202 minutes remaining, satisfying the "more than 90 minutes" rule.

### Open questions for Enrico

1. **A1 amendments.** Sign off on A1-1 (temperature 0, so replicates vary the feedback deals) and A1-2 (no adoption;
   a failed copy counts as S0).
2. **C1b.**
   - Should pooled estimators be judged against the population-average credit, the quantity they target, rather
     than per-seed credit? The exploratory numbers are above: ridge 0.47 and 1.28.
   - That would be an amendment for a new experiment, C1b, not a change to C1's verdict.
3. **C2.** At temperature 0 the pre-registered primary quantity is mechanical. Options:
   - run the full C2 anyway, which by projection gives 2.23 [1.35, 2.94];
   - replace it with a pre-registered test of the Flawed-versus-placebo contrast;
   - redefine the cost-to-good-teachers quantity on the student's outcome rather than on Piers' ITT;
   - add a temperature > 0 block, where quarantine is not mechanical.
4. **Gemma block.** C1 acceptance needs it. Exhaustive ITT is the only estimator that passes this block.
5. **A1 check populations.** Should beta be estimated robustly before they run, given the bimodal copy
   distribution? For example, from the upper part of the distribution or from a two-component mixture. That would
   be an amendment before the run.
6. **Rulings to confirm.** The planning session's rulings: an estimator must pass in both strata, and the BH family
   is the two primary contrasts.

Test suite at the end: **214 passed, 1 skipped** (the skip is the opt-in live MLX test, `CULTURE_LIVE=1`): the 205 from fix round 1 plus 9 new tests (1 per-stratum replicates, 6 quarantine, 2 transmission).

## Deviations from the spec

1. **Observation JSON adds `possible`** (`{"colors": [...], "ranks": [...]}`) to every card, own and partner's. It is HLE's plausibility set, which includes negative information. HLE gives it to every agent and Canaan's agents depend on it. Without it the adapter loses information the raw engine provides, and trajectories cannot match. Partner cards also carry `hints`/`possible` (what the partner knows).
2. `last_moves` is most-recent-first (HLE order) and goes back to and includes the observer's own last move. PLAY/DISCARD entries include the revealed `color`/`rank`; PLAY includes `scored`.
3. The hanab.live exporter and reloader live in `game/hanablive.py`, not in `hanabi.py`. HLE's colors map to hanab.live No Variant suits as R, Y, G, B, W -> Red, Yellow, Green, Blue, Purple (W plays Purple).
4. Evaluation anchors are the native rule-list ports (`piers`, `iggi`, `flawed`), which are about twice as fast and can be passed around as artifacts. They are tested move-for-move identical to the vendored originals (`canaan_*`) on 1,000 seeds.
5. The soft timeout (50 ms) is counted (`slow_rate`) but not enforced. Replacing a move on wall time would make games depend on machine load. The hard timeout (1 s) is enforced.
6. **The Anthropic backend is not written** (no paid calls; open weights are the default). The OpenAI-compatible backend exists since fix round 1, step 9, targeting the local MLX server; `validate()` accepts `stub` and `openai_compat`.
7. The default model is `claude-haiku-4-5` (screening tier per section 17). The spec's `Request` comment says Opus, but section 17 rules Opus out for agents.
8. `ingest.md` is a prompt fragment rendered into the revise context, not a separate LLM call. `system.md` (the frozen prefix) and `repair.md` were added.
9. The config digest excludes `runner.generations`, `runner.wall_clock_budget_s`, `evaluation.workers` and `debug`, so a run can be extended or resumed with a different worker count.
10. Warm starts are authored once per population seed into a shared set that every condition forks from (fix round 1, step 5); conditions make no generation-0 backend calls. (Before: seed tags omitted the condition and the second condition hit the cache, which races with a real backend.)
11. The diversity "code embedding" is token-count cosine (no embedding model offline). The probe's H-group comparison is a keyword list plus bag-of-words cosine; the spec asked for embedding similarity against the rulebook.
12. Higher Criticism uses HC+ (only p > 1/n): with paired seeds a near-deterministic improvement has p about 0, and plain HC* explodes.
13. Paired-delta credit window: before = the student's incumbent at the start of the generation, after = its incumbent at the end, both against the anchor set on that generation's seeds. Messages sent in generation g are received in g+1.
14. Topology `isolated` means groups are isolated from each other; agents inside a group can still exchange. `best_to_all` only sends to agents scoring strictly below the sender.
15. Workers run with `PYTHONHASHSEED=0`, and the CLI re-execs itself with it, so bots that iterate string sets stay deterministic across processes.
16. Extra test file `tests/test_artifacts.py`.

## Blockers

None. No phase was blocked for more than 30 minutes on one issue. The longest detours were the CMake 4 incompatibility, the hidden `.pth` files and the resume-identity bugs; all are fixed and described above.

## Proposals (not built; the spec does not name them)

- A stub mode that writes **mutually incompatible** conventions (for example, each stub "family" uses a different hint-meaning table) so the null-alive gate and the between-group headline can be exercised before any API spend. Today every stub bot shares one template and cross-plays well.
- Store the full request text next to each cached response (optionally) so a replay mismatch can be diffed directly.
- Port the H-group level 1 to 3 rulebook text into the repo so the probe's conventions diff can use it once an embedding model is allowed.

## Open questions for Enrico

1. The spec file `files/sandbox-architecture.md` is untracked on `enrico`. I left it alone. Commit it?
2. Phase 3 defines the student as the agent with the lower generation-0 score and uses `best_to_all` routing for the transfer conditions (the better agent sends to the worse one). Is that the setting you want for the Jha et al. comparison, or should the teacher be fixed (for example, a stronger tier)? A fixed teacher needs a role-based routing the spec does not name.
3. HGM widening with alpha = 0.6 lets about 1 in 8 agent-steps revise. Should agent-steps count once per group-generation instead (much more expansion), or is the conservative reading what you want for matched-budget comparisons?
4. `nash_relative` is implemented as max-entropy Nash averaging. Is that the rule you meant? If yes, what floor (eps) do you want so non-dominant groups are not starved?
5. Memory caps for bot workers do not work on macOS. Is CPU plus wall timeouts enough for now, or should bots run under a container on Linux for long runs?
