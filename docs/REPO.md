# REPO — how to run anything (session-facing)

## Setup and tests
- `scripts/setup_env.sh` once; then `.venv/bin/python -m pytest -q` (about 5 minutes, 269+ tests). Package: `src/culture/`.
- Engine: Hanabi Learning Environment vendored at `third_party/hanabi-learning-environment` (commit 54e7959), built
  from source with `CMAKE_POLICY_VERSION_MINIMUM=3.5`. Reference bots and their 1,000-game self-play: Piers 16.99,
  IGGI 15.86, Flawed 0.00 (Canaan et al. rule lists, ported in `src/culture/bots/anchors/`).
- Never edit `src/millstlabs` (Diego's package, on `main`).

## The lab entry point (built 2026-10-08; see `src/culture/lab/`)
`python -m culture.lab run configs/E##-slug.yaml --mode smoke|pilot|full [--out runs/E##]`
`python -m culture.lab check` validates registry ↔ IDEAS.md ↔ findings ↔ notes ↔ configs.
A config = label, title, primitive (`student` | `population` | `chain`), backend (+ cap), design (seeds, factors →
full factorial, `paired: true`), primitive params, outcomes, one figure spec, `prediction:` note path. Modes
override: smoke = stub, one seed, first level of each factor; pilot = 3 seeds; full = seeds from the pilot's
power calculation. Every run lands as `runs/E##/<cell>/<seed>/`, `summary.json` (per cell: outcome, bootstrap
interval, null band if a stub-null exists), `figure.png`, `finding_stub.md`. The runner refuses `full` without
the note and refuses hosted without a passed smoke and a cap; it records real spend in the registry.

## The three primitives underneath (unchanged drivers)
- `student`: `src/culture/run/single_student.py`. One learner, k fixed teachers, one generation per replay;
  delivery `exhaustive | bernoulli | pb8 | singles_pairs`; strata `verified | unverified | verified_quarantine`.
  Analysis in `src/culture/analysis/credit.py` (exact Shapley, ITT, leave-one-out, equal split, ridge, singles+pairs).
- `population`: `python -m culture.run --config <yaml> --out runs/<x>`; experiments with conditions and seeds via
  `scripts/run_experiment.py <spec.yaml> <out> [processes]` (`src/culture/run/experiment.py`; spec has `base`,
  `conditions`, `population_seeds`, optional `extends`). Organization dials are `org.*` in `src/culture/run/config.py`;
  `org.schedule` toggles dials at generations (switchback). Warm starts authored once per seed and shared across
  conditions. Analysis: `src/culture/analysis/{metrics,accumulation,null,hc,graphs,ifpca,breakdown}.py`.
- `chain`: one agent revising repeatedly; `scripts/p1_eps_probe.py` is the current implementation (Design B).
- Transmission (copy fidelity): `src/culture/run/transmission.py`, `scripts/a1_report.py`.

## Backends and money
- `llm.backend: stub` (free, deterministic; `stub.mode: null` = revisions keep the rule list, for null bands).
- `llm.backend: openai_compat` to the local MLX server: start with
  `caffeinate -i -s .venv/bin/python -m mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --host 127.0.0.1 --port 8080 --max-tokens 8192 --chat-template-args '{"enable_thinking": false}'`.
  Seeded temperature-0 requests are bit-reproducible (this is what makes exact counterfactual replay possible).
  ~55–85 s and ~11k tokens per revise call. One request at a time; concurrent requests get batched and break
  reproducibility. Ollama is not reproducible for this model; do not use it.
- Hosted: same backend with `api_key_env`, `price_in_per_mtok`, `price_out_per_mtok`, `max_usd` all set
  (`configs/hosted_example.yaml`); `src/culture/llm/spend.py` reserves before send, caps cumulatively across
  processes, retries only unbilled failures, never auto-retries a paid one. Probe a provider with
  `scripts/hosted_check.py --max-usd 1` before anything else. Flash-class open-weight models cost on the order of
  a few tenths of a cent per revise call; verify on the provider's price page.
- Parallelism on hosted = the experiment process pool (condition × seed). Within-run concurrency is wired in
  `complete_batch` but the generation loop does not use it yet.

## Where data lives
- `docs/results/*.json|png`: every measured number so far (credit run `c1-full.json`, copy fidelity `a1-params.json`,
  sequential test `b1-sprt.json`, breakdown stub `d1-pilot.json`, innovation ladder `p1-eps-ladder.json`,
  quarantine pilot `c2-pilot.json`). Findings cite these.
- `runs/` is git-ignored; each run has `generations.jsonl`, `messages.jsonl`, `ledger.jsonl` (every model call,
  cost, latency), `artifacts/` (content-addressed bots + conventions docs), `llm_cache/` (record/replay).
- Conventions documents of every artifact are in each run's `artifacts/` store: the raw material for idea-maps.

## Glossary for the archive (old labels → plain names)
C1 credit estimators vs exact replay · C1b pooled estimators vs population-average credit · C2 quarantine ·
A1 copy fidelity by medium · B1 sequential verification · D1 breakdown point under sabotage · P1 predict
accumulation from micro-parameters · G1 three organizations at matched budget · S1 switchback levers ·
I1 influence-bounded organizations · R1 recombination · H1 horizon at fixed budget.
`archive/tex/research.pdf` holds their formal designs; `archive/prereg/` the pre-registrations and ledger;
`archive/docs/sandbox1-status.md` the build history and every number.

## Subagent prompt template (run and write)
> Repo `<path>`, branch `enrico`, venv `.venv/bin/python`. Read `CLAUDE.md` and `docs/REPO.md`. Run
> `python -m culture.lab run <config> --mode <mode>` for each of: <list>. Do not change code. When each finishes,
> fill its `runs/<label>/finding_stub.md` into `findings/<label>-<slug>.md` using `findings/_template.md`, in plain
> language, numbers in a table with intervals, one figure, "what it changes" in ≤5 lines, status `draft`. Do not
> add a line to `RESULTS.md` (the main session does). Report: per config, cells × seeds run, wall clock, calls,
> spend, the headline number, anything that failed.
Builds go to a worktree agent (`isolation: worktree`) on a cheaper model with tests required; merge from the main
session. Analyses that only create new files run in the main checkout.

## Known gaps
- A bot can monkeypatch a shared stdlib class through an alias (sandbox residual; recorded, not fixed).
- Memory limits for bot workers are unavailable on macOS; CPU and wall-clock timeouts only.
- Desktop file sync duplicates files as "name 2.ext"; delete, never read.
