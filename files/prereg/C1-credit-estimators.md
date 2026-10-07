# Pre-registration C1: causal credit for teachers, estimators against planted truth

Status: finalized 2026-10-06 (Enrico approved the estimand and the open choices in chat). Nothing below has been run.
Ledger entry: `files/prereg/LEDGER.md` row C1.

## 1. Question

A student receives k messages in one generation from teachers whose effects are unknown. Which estimator of each teacher's causal effect on the student is accurate enough to use inside the organization, at what cost in LLM calls? This is edge item 2 (credit over a teaching DAG with several teachers).

## 2. Setting

- One student agent, incumbent S0 = a weak rule list (self-play about 3 on the shared deals).
- k = 3 teachers per generation, one message each, each message carrying an artifact plus harness-signed evidence.
- Backend: local Qwen3.6-35B-A3B at temperature 0 with a fixed sampling seed per replicate; stub backend first for pipeline checks. No paid calls.
- Two strata, run separately: verification on (`selfplay(n=200)`, `replace_if_better`) and verification off (adopt what is delivered). Credit is defined with the adoption rule inside the effect (intent to treat), because that is what the organization will see.
- Deals: three disjoint seed purposes per generation (feedback traces shown to the LLM; verification; held-out evaluation). Evaluation uses 300 paired deals; per-deal SD for similar bots is about 4, so the SE of a paired mean is about 0.23 points.

## 3. Estimand (decided)

Let W in {0,1}^k record which messages were delivered and Y(W) the student's held-out self-play score after the
generation minus before, both on the same deals. The student's update is a deterministic function of (state, W, seed).

- **Primary, intent-to-treat.** For message j,
  tau_j = E_W[ Y(W with j = 1) − Y(W with j = 0) ],
  averaged over delivery of the other messages. The treatment is delivery, the only thing the organization controls;
  verification, rejection and adoption are inside the effect. Identified by the delivery randomization alone.
- **Alongside, Shapley.** phi_j with value function v(S) = Y(S) over delivered subsets S (Yang et al. 2025 formalism);
  also intent-to-treat; satisfies efficiency, sum_j phi_j = v(all) − v(none). Computed from the same replays.
- **Secondary, effect among adopters.** Delivery is an instrument for adoption; a message cannot be adopted unless
  delivered (one-sided noncompliance), so the local average treatment effect is identified (Wager ch. 10):
  effect among adopters = tau_j / P(adopt j | j delivered). Reported for interpretation; undefined when no seed adopts
  (reported as such). Not used in decision rules, because conditioning on adoption directly would reintroduce selection.
- No interference: one student per replicate, so SUTVA holds and chapter 12 machinery is not needed here.

Worked cases: Piers with verification on, adopted in 90% of seeds, +14 when adopted: tau = 12.6, adopter effect 14.
Flawed with verification on, never adopted: tau = 0, adopter effect undefined. Flawed with verification off: both about −3.

## 4. Planted truth and the oracle (adopting published definitions)

Teachers carry artifacts with known standalone effects:
- T1 = Piers (large positive; about +14 on a random-level student if adopted),
- T2 = Flawed with persuasive prose and honest (low) evidence (negative when verification is off; 0 when on),
- T3 = the student's own incumbent re-sent (0).

The oracle is the Shapley value of each message with the value function v(S) = the student's held-out paired
improvement when exactly the subset S of messages is delivered, computed by exact replay over all 2^k subsets under
identical seeds. This is the formalism of Yang et al. 2025 (arXiv 2511.10687, "one agent message = one player",
continuous replay-grounded v), which we implement ourselves since no code was released. For k = 3 it is 8 LLM calls per
seed per stratum. The average main effect tau_j of section 3 is computed from the same 8 replays and reported alongside
(Shapley satisfies efficiency, sum of credits = v(all) − v(none); the main effect does not).

## 5. Estimators compared (existing tools first)

- (i) leave-one-out ablation, as in SHARP (arXiv 2602.08335) generalized to a continuous outcome: credit_j =
  v(all) − v(all \ {j}); k + 1 replays (4 at k = 3). First order only.
- (ii) `paired_delta` v0 (ours): the student's paired improvement split equally among adopted messages; 1 replay.
- (iii) randomized delivery, Bernoulli(1/2) per message, plus ridge regression of v on delivery indicators pooled over
  seeds (a datamodels-style estimator); uses a random 8-run subset of the oracle's replays per seed, so no extra calls,
  and amortizes across the population.
- (iv) singles-plus-pairs replay, the GCJR pattern of Li et al. 2026 (arXiv 2608.29228) with the binary success
  threshold replaced by the continuous outcome: 1 + k + C(k,2) replays (7 at k = 3, 29 at k = 7); recovers pairwise
  interactions. Tested in Arm 2 (k = 7, four null teachers added) against an exhaustive 128-run oracle on 5 seeds,
  alongside an 8-run Plackett–Burman design and 8 Bernoulli runs.

Shapley-Coop (arXiv 2506.07388) is excluded: its Shapley numbers are LLM-estimated rather than replay-grounded, and its
pricing layer solves a bargaining problem we do not have.

## 6. Design

- Arm 1: k = 3, R population seeds, both strata. Pilot R = 5 to estimate the between-seed SD of the primary contrast; then R from the power rule below, capped at 60.
- Arm 2: k = 7, 5 seeds with the 128-run oracle, 30 seeds for the two 8-run estimators.
- Every estimator sees the same seeds, deals and model. Replication block: repeat Arm 1 at R = 10 on Gemma 4 26B-A4B; an estimator is accepted only if it passes in both blocks.

## 7. Metrics

Per estimator, per seed: RMSE of tau_hat against the oracle over the k teachers; Spearman rank correlation with the oracle; sign error on T2. Aggregated as IQM over seeds with stratified-bootstrap 95% intervals. LLM calls per seed are recorded from the ledger.

## 8. Decision rules, fixed now (thresholds approved 2026-10-06)

Accept an estimator for use in the organization if, in both model blocks:
- Spearman IQM ≥ 0.9 with bootstrap lower bound ≥ 0.8,
- RMSE IQM ≤ 2.0 points,
- sign error on T2 ≤ 5%.
Among accepted estimators prefer the fewest LLM calls per seed. Primary confirmatory contrast: RMSE of (iii) minus RMSE of (i); minimum effect worth detecting 1.0 point; R chosen for 80% power at two-sided alpha 0.05 from the pilot SD. If no estimator passes at R = 60, the conclusion is that causal credit is not available at this cost with this organization, and the one-pager's "credit for students' improvement" is reworded or dropped.

## 9. Secondary, exploratory, labelled as such

Influence of a message as the divergence between the law of Y with and without it across sampling seeds (the privacy-loss object); interaction tau_12 (does Piers plus Flawed interact); comparison of the two strata.

## 10. Multiplicity and null calibration

Three confirmatory statistics per estimator, three estimators in Arm 1 (i, ii, iii): nine tests, BH at q = 0.1 across the program ledger. The same pipeline is run with the null stub (no real improvement) to obtain null distributions of RMSE and Spearman for each estimator; figures carry null bands.

## 11. Compute

Arm 1 oracle: 8 calls × R seeds × 2 strata. At R = 30 and about 1.5 minutes per call single-stream, about 12 hours on the M4 Pro, overnight. Estimators reuse the oracle's runs, so no further calls. Engine time is negligible. Cost $0.

## 12. Outputs

One figure: tau_hat versus oracle tau per estimator (points per seed, identity line, null band). One table: Spearman, RMSE, sign error, calls per seed, per estimator per block, with intervals. A ledger row with the pre-specified rules and the outcome, including negative.

## 13. Nearest prior work (from files/prereg/novelty-notes.md)

Shapley and process-reward credit for multi-LLM agents (Yang et al. 2025, conceptual); counterfactual replay for failure
localization in agent DAGs (Li et al. 2026); Aumann–Shapley attribution at scale (Tang et al. 2026). The combination
tested here (randomized delivery plus regression, checked against exact seeded replay, as a teacher's causal effect on a
paired-seed student) was not found in the 2026-10-06 arXiv check. Claim phrasing must follow the novelty notes.

## 14. Contingency: LLM sampling is not bit-reproducible

Observed 2026-10-06: Qwen3.6-35B-A3B under Ollama on Apple silicon gives different outputs for identical seeded,
temperature-0 requests (divergence after about 700 tokens), with parallelism 1. Being tested: CPU-only inference and
MLX single-stream serving. If no runtime is bit-reproducible, xi is treated as random: v(S) is estimated by r
repeated replays per subset (r chosen from a pilot of the within-subset SD so that the SE of v(S) is below 0.5
points), the oracle becomes the expected Shapley value, and the LLM noise enters the between-replicate variance that
the power calculation already uses. Cost multiplies by r. The paired-deal design is unaffected.

## 15. Amendments after the pilot (2026-10-07; logged, not silently edited)

Pilot: 5 seeds x 2 strata on MLX, 80 replays, 74 model calls, 54 s and about 10.5k tokens per call, 97.5% of revisions
admissible. Results in docs/results/c1-pilot.json.

1. **T3 is a placebo, not a known zero.** The re-sent own incumbent moved outcomes by -2.9 to +1.9 points: any message
   changes the LLM's revision. The oracle still defines the truth for T3; the sign-error check uses T2 only (as written
   in section 8). No decision rule changes.
2. **Finding to carry into D1 and a new lever.** With verification on, Flawed was never adopted (0 of 5) yet its
   ITT effect was -3 to -5 points in 4 of 5 seeds: the message's prose and evidence sit in the revision prompt, so
   verification gates code adoption but not ideas. Proposed new policy `quarantine_unverified` (rejected or
   unverified messages are withheld from the revision prompt). To be pre-registered as C2, not folded into C1.
3. **Estimators at k = 3.** Only the full-replay estimators (tau_itt, Plackett-Burman at k=3 = exhaustive) met
   RMSE <= 2 (0.7). Ridge on Bernoulli deliveries 2.6 (verified) / 3.9 (unverified) with 5 pooled seeds; leave-one-out
   3.2 / 4.4; singles-plus-pairs 3.0 / 2.9. Rank order was right for all but equal-split (Spearman 1.0). The full run
   at R = 29 is the pre-registered test; ridge pools across seeds and may pass with R = 29, leave-one-out cannot pool.
4. **Sample size.** Between-seed SD of the primary contrast 1.85 (verified) and 0.86 (unverified); R = 29 and 8 for
   80% power at 1 point; about 300 model calls, 4.5 to 7 hours on MLX, $0.
5. **Baseline is not weak.** v(none) was 7.2 to 11.9: the student's own single revision from a ~3 bot already reaches
   7 to 12. Effects are measured relative to that, as the estimand says.
