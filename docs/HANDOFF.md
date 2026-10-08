# Handoff for a new session (updated 2026-10-07, end of the second planning session)

Read in this order, then start at "Next actions". Everything is on branch `enrico`.

1. `CLAUDE.md` (how to work with Enrico; repo facts; rules).
2. `tex/research.pdf` (the research notes: goal, game, population process, causal framework, estimators, rigor
   protocol, experiments as designs). This is the canonical statement of what we are doing.
3. `docs/sandbox1-status.md` ("Overnight 2", "Hosted backend", "G1/S1 preparation (stub)" sections).
4. `files/prereg/LEDGER.md` and `files/prereg/*.md` (pre-registrations, amendments, outcomes; G1 and S1 are DRAFTS
   awaiting sign-off and are not in the ledger).
5. `docs/p1-notes.md` and `docs/results/p1-*.json` (P1 micro-parameters, the draft accumulation model, the local
   epsilon ladder); `docs/results/*.json` only if a number is needed.

## Where we are

- Harness: `src/culture/` (engine adapter, sandbox, evaluators, signed evidence, LLM cache with replay, runner with
  resume, single-student driver with the exact credit oracle, population driver with organization policies, a lever
  schedule for switchback runs, hosted OpenAI-compatible backend behind a sqlite spend guard). **269 tests pass,
  1 skipped.** Local model: Qwen3.6-35B-A3B via MLX, seeded, bit-reproducible, 55 to 85 s per call.
- Run so far (all $0, one model block, nothing accepted): C1 full (no cheap credit estimator meets the per-replicate
  criterion; exact replay does; pooled ridge is fine against population-average credit); B1 (SPRT saves ~73% of
  deals); C2 pilot (quarantine mechanical at temperature 0); A1 parameters (no medium claim; copies bimodal); D1 pilot
  on the null stub only; P1 epsilon ladder probe on the local model (below).
- No paid call has ever been made. No provider key is set on this machine.

## Done in the second session (2026-10-07 evening, handoff actions 1 to 3, $0)

1. **Hosted backend** (`src/culture/llm/spend.py`, `openai_compat.py`, `configs/hosted_example.yaml`,
   `scripts/hosted_check.py`, 33 tests). Reserve before send, cumulative cap across processes and restarts, no
   automatic paid retry (429 and connection errors retried unbilled; 5xx keeps the reservation as "uncertain" and the
   next identical call is refused unless `retry_uncertain`). A hosted config needs `api_key_env`, both prices and
   `max_usd > 0` or it refuses to load. Parallelism is the experiment's process pool (condition x seed); within-run
   concurrency is wired in `complete_batch` but the generation loop does not call it yet. Local MLX path
   byte-identical to before.
2. **G1 and S1 specs and the pilot driver** (`configs/g1.yaml`, `g1_null.yaml`, `g1_pilot.yaml`, `s1.yaml`,
   `org.schedule` lever switching with resume, canonical warm starts shared across layouts,
   `analysis/accumulation.py`, `scripts/g1_pilot.py`). Isolated = 8 groups of 1. The 24-call cap never binds, so
   spend is NOT matched across conditions (about 16 calls per generation with teaching, 8 without): either compare on
   the calls axis or run isolated longer. The stub pilot (`runs/g1s1-stub` in the worktree
   `.claude/worktrees/agent-a6009eb90bd8de5ec`) was still running at handoff: S1 phase nearly done, G1 and null
   phases not started; about 45 s per generation on the stub alone (engine verification dominates).
   Summarize later with `scripts/g1_pilot.py ... --skip-run` (command in the status doc).
3. **P1** (`scripts/p1_params.py`, `analysis/accumulation_model.py`, `scripts/p1_predict.py`,
   `scripts/p1_eps_probe.py`, `docs/p1-notes.md`). Measured: copy success p_s(both) 0.57 [0.39, 0.73] by the 1-point
   rule; eps(weak student, 2.75) 0.69 [0.51, 0.83] with gains of 8.7 +/- 3.3; eps(IGGI level 15.8) 0.19 [0.07, 0.34]
   at margin 0.5; **eps(Piers level 17.0) = 0 of 30 [0.00, 0.11]**; chain from Piers over 40 generations: 1 of 43
   revisions in the 17 to 18 bin improved by > 0.5, and the incumbent's behaviour on common deals never left
   Piers' 16.7. So the local model does not innovate above about 17: the draft model's saturation at the 25-point
   cap (flat-epsilon extrapolation) is not real for this model, and a 20-point threshold is probably unreachable
   here. The hosted model's ladder must be measured separately before predicting G1.
4. **Drafts awaiting sign-off:** `files/prereg/G1-accumulation.md` (six decisions in section 0, including matched
   budget and the primary outcome's horizon), `files/prereg/S1-switchback.md` (five decisions: first-difference
   outcome, counterbalanced schedule, burn-in, lever set, minimum effect 0.05/generation).

## Decisions Enrico must make before anything paid runs

- Provider, model and spend cap for the hosted pilot (proposal from the previous session: DeepInfra or Fireworks,
  DeepSeek V4.1 Flash or GLM-5.3 Flash, $50 week-one cap). Then run `scripts/hosted_check.py` with a tiny cap to
  measure price, latency and determinism before any experiment.
- G1: matched budget rule (calls axis vs longer isolated run); primary outcome and horizon (generation-100 level vs
  calls-to-threshold vs gap at the model's predicted peak vs area between curves); temperature; R1 fold-in (proposed
  no); compatibility threshold.
- S1: outcome scale, schedule, burn-in, lever set, minimum effect.
- P1: copy model (1-point rule vs EM vs exact adoption), whether innovation at 8 to 14 points needs measuring before
  predicting (90 local calls), which G1 outcome P1 predicts.
- Earlier open items still pending: A1 amendments; C1b criterion; C2 re-registration; Gemma block; rulings on
  "both strata" and the BH family.

## Next actions, in order (agent speed)

1. Enrico's decisions above; finalize G1 and S1 pre-registrations and add them to the ledger.
2. `scripts/hosted_check.py` on the chosen provider with a $1 cap; record price, latency, determinism in the status doc.
3. Fill `configs/g1_pilot.yaml`'s llm block from `configs/hosted_example.yaml`; run the G1 null-stub band (K = 10)
   and the 3-seed, 30-generation hosted pilot (~2,200 calls); power calculation; then the full G1.
4. Measure the hosted model's epsilon ladder (`scripts/p1_eps_probe.py` with the hosted backend, capped) and the
   warm-start distribution; pre-register the P1 prediction before G1 unblinds.
5. S1 hosted pilot (3 seeds x 5 levers x 90 generations) after G1's pilot has fixed the base organization.
6. D1 on the hosted model at pilot size, reaction rules as a factor.

## Process rules that must survive the handoff

Pre-register before running; amendments are logged, never silent; paired seeds; three disjoint deal sets; null bands;
second model block before any claim is accepted; BH across the ledger's p-valued tests; novelty claims cite the three
nearest works; no paid call without a cap; `caffeinate` before unattended local runs; one call at a time on MLX;
Desktop file sync creates empty " 2" duplicate directories under `runs/` (harmless, ignore).
