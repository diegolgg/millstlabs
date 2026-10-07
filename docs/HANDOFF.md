# Handoff for a new session (written 2026-10-07, end of the first planning session)

Read in this order, then start at "Next actions". Everything is on branch `enrico`.

1. `CLAUDE.md` (how to work with Enrico; repo facts; rules).
2. `tex/research.pdf` (the research notes: goal, game, population process, causal framework, estimators, rigor
   protocol, experiments as designs). This is the canonical statement of what we are doing.
3. `docs/sandbox1-status.md` (what is built and validated; "Fix round 1" and "Overnight 2" sections).
4. `files/prereg/LEDGER.md` and `files/prereg/*.md` (pre-registrations, amendments, outcomes so far).
5. `docs/results/*.json` only if a number is needed; `reports/Sandbox game selection lit review.md` for why Hanabi.

## Where we are

- Harness: `src/culture/` (engine adapter, sandbox, evaluators, artifact store with signed evidence, LLM cache with
  replay, runner with resume, single-student driver with the exact credit oracle, population driver with the
  organization policies). 214 tests pass. Local model: Qwen3.6-35B-A3B via MLX, seeded, bit-reproducible, ~55 s per call.
- Run so far (all $0): C1 full on one model block (credit estimators vs oracle; verdict: no cheap estimator meets the
  per-replicate criterion; exact replay does; pooled regression is fine against population-average credit); B1 (SPRT
  verification saves ~73% of deals at matched error); C2 pilot (quarantine is mechanical at temperature 0); A1
  parameters (fidelity by medium: no difference; copies are bimodal); D1 pilot on the null stub only.
- Two design lessons: any delivered text perturbs the LLM's revision (no placebo exists; verification of code is not
  verification of ideas); pooled estimators must be judged against the population-average credit.

## The reorientation decided on 2026-10-07 (supersedes earlier "needs funding" statements)

Group-level and long-run experiments do NOT need funding. They need (a) a hosted open-weight backend with parallel
calls and a hard spend cap, because they do not need bit-exact replay (paired seeds + replicates suffice), and
(b) switchback / interrupted designs for long runs. Exact replay on the local MLX model is reserved for the credit
oracle (single-student work). Headline for preseed = accumulation under organization, not robustness.

Headline experiments (designs in `tex/research.tex` section 5; pre-register before running):
- **G1 accumulation under organization.** Three organizations at matched budget (isolated; full sharing; groups with
  teaching + verification + migration); 8 agents, 100 generations, 5 seeds; hosted model; ~12k calls, ~$30. Outcomes
  vs a frozen ladder: best and mean self-play, within/between-group cross-play, retained innovations, and calls to
  reach fixed score thresholds (17 = Piers level, 20), so the result is also an efficiency claim comparable to the
  one-pager's. Fold in R1 (plant two complementary partial conventions; does the population combine them).
- **P1 predicted-vs-observed accumulation.** Predict G1's trajectory from micro-parameters measured in single-student
  runs (innovation rate eps, fidelity mixture: success probability + within-cluster loss, verification operating
  characteristics from B1) using a Henrich-style copy-the-best model with a verification filter; compare to observed.
- **S1 switchback perturbation.** One 300-generation run per seed with a lever toggled every 30 generations
  (teaching, verification, quarantine, shrinkage selection, migration); interrupted-series estimates with the
  mixing-time bias bound (Wager ch. 15). ~$5 per run.
- **I1 influence-bounded organizations (novel framing).** The organization as an estimator of "the best artifact"
  from noisy, partly adversarial messages: per-message influence (C1's oracle) is its influence function, D1's
  breakdown point its breakdown point, and the DP-style claim is that no single message moves the population's law by
  more than eps, composing over generations, with randomized delivery as amplification by subsampling. Test with a
  planted persistent saboteur in G1/S1.
- **D1** (breakdown point with reaction rules) stays as the supporting robustness figure; **C1b, C2'** stay as
  single-student validations of the credit instrument.

## Next actions, in order (agent speed)

1. Hosted backend: `openai_compat` already targets any OpenAI-compatible endpoint; add provider config (DeepInfra or
   Fireworks; model DeepSeek V4.1 Flash or GLM-5.3 Flash, MIT), parallel calls, Diego-style cost guard (reserve before
   send, cumulative cap, no auto paid retry). Proposed week-one cap $50; confirm with Enrico before the first paid call.
2. Pre-register G1 and S1 (one page each, template in research.tex Implementation "Pre-registration"); null-stub
   pilot; 3-seed pilot on the hosted model (~$10); then power-sized run.
3. Measure the micro-parameters P1 needs on the local model if missing (eps per context; fidelity mixture; B1 ROC
   exists), write the predictive model, pre-register the prediction before G1 unblinds.
4. D1 on the hosted model at pilot size, reaction rules as a factor.

## Coordination with Diego's one-pager (2026-10-07)

His Sandbox 1 bullets claim efficiency (fivefold discovery acceleration; saturating games at <= 1/10 the compute).
Ours should be the complementary claims, each with a figure behind it before it is written: causal credit computed by
exact counterfactual replay on a seeded open-weight model; verification of code is not verification of ideas (rejected
text still shifts learners; quarantine removes it); sequential verification at ~1/4 the evaluation cost; and, once G1
runs, organized populations reach a score threshold in fewer calls than isolated ones. Do not write a claim whose
figure does not exist yet.

## Open questions for Enrico

- Hosted provider and model; spend cap confirmation.
- C1b criterion (population-average credit) and C2' re-registration: sign-off.
- A1 mixture summary and the temperature-0 amendment: sign-off.
- Which lever schedule for S1.

## Process rules that must survive the handoff

Pre-register before running; amendments are logged, never silent; paired seeds; three disjoint deal sets; null bands;
second model block before any claim is accepted; BH across the ledger's p-valued tests; novelty claims cite the three
nearest works; no paid call without a cap; `caffeinate` before unattended local runs; one call at a time on MLX.
