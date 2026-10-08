# Pre-registration G1: accumulation under organization

Status: DRAFT 2026-10-07, written from research.tex Experiment 5.7. Not signed off; not in the ledger until Enrico
approves the decisions in section 0. Nothing has run on a model. The stub pilot (pipeline only) is in
docs/results/g1-stub-pilot.json once it exists.

## 0. Decisions needed before this is final

1. Hosted provider and model for block 1, and the second model for block 2 (candidates: DeepSeek V4.1 Flash,
   GLM-5.3 Flash; both MIT). Spend cap for the pilot and for the full run.
2. Temperature. Proposed 0 with a fixed seed per (run, agent, generation): a provider may still not be deterministic,
   and that is fine, because replicates are population seeds, not replays. Alternative: 0.3, so the model's own
   randomness is part of the replicate variance by design.
3. Matched budget. Proposed: match the number of revise calls per generation (8 per condition); teach calls are the
   organization's overhead, counted on the x-axis of every efficiency figure (calls to threshold) but not deducted from
   the revise budget. Alternative: match total calls, which starves teaching under the current loop order (revise
   spends the pool before teach runs). A third option is teaching without a model call (send the artifact and its
   conventions document as written), which makes the two budgets equal; it changes what "teaching" means.
4. R1 fold-in. Proposed: do NOT plant partial conventions in G1; keep G1 a clean accumulation contrast and run R1
   separately on the same pilot infrastructure. Planting near-Piers parts into the warm start would change the
   accumulation question (the population would start with most of Piers' knowledge).
5. Compatibility rule. Proposed threshold for "between-group cross-play does not fall below the isolated population's
   cross-play": the paired difference's 95% interval is not entirely below -1 point.
6. Primary outcome and horizon (raised by the P1 draft prediction, docs/results/p1-prediction.json). With the
   innovation rate extrapolated flat above the Piers level, the model predicts every organization saturating near the
   25-point cap before generation 100, so a generation-100 level contrast could be a tie even when organization
   clearly speeds accumulation; the predicted organized-minus-isolated gap peaks near generation 20 and fades. Whether
   that saturation is real depends on the innovation rate above 17, which the local probe (scripts/p1_eps_probe.py)
   is measuring. Proposed: keep the level contrast but at the generation where the P1 model predicts the largest gap
   (fixed before unblinding from the measured parameters), and make calls-to-threshold a co-primary outcome.
   Alternative: the area between the organized and isolated mean curves over the 100 generations (a cumulative
   advantage), which is robust to where the gap peaks.

## 1. Question

At a matched budget of revise calls, does a population organized into groups with teaching, verification and migration
reach a higher mean skill, and reach fixed skill thresholds in fewer total calls, than isolated agents and than
unstructured full sharing? Edge item 1 (the group as the unit that competes and reproduces, with teaching), and the
trajectory figure the pre-seed needs.

## 2. Setting

- 8 agents, warm start authored once per population seed and shared by every condition (run/warmstart.py), then 100
  generations. Hosted open-weight model, runs in parallel across conditions and seeds, one shared spend file.
- Three conditions, same deals, same warm starts, same evaluation:
  - `isolated`: no messages (routing none). Eight agents in one group so the metrics are defined on the same code
    path; "between-group" cross-play for this condition is the mean cross-play over all pairs.
  - `full`: one group of 8, broadcast routing, verification selfplay(n=200), replace_if_better, no migration.
  - `organized`: two groups of 4 (islands), broadcast within group, selfplay(n=200) verification, replace_if_better,
    random migration (rate 0.25 every 5 generations), shared corpus per group.
- Three disjoint deal sets per generation (feedback traces; verification; held-out evaluation, 200 deals). Frozen
  ladder every 10 generations against the anchors (Piers 16.99, IGGI 15.86) and every past snapshot.
- Null band from the null stub (revisions keep the rule list) at K = 10 population seeds, same spec.

## 3. Estimands

Let Y_c,s(g) be the population mean held-out self-play at generation g in condition c and population seed s, and
B_c,s(g) the population best. The unit is the population; interference inside a population is the treatment, not a
nuisance (the whole organization is assigned), so no exposure mapping is needed at this level.

- Primary: Delta_1 = E_s[Y_organized,s(100) - Y_isolated,s(100)], paired by seed (same warm starts, same deals).
- Secondary accumulation contrasts: Delta_2 = organized minus full at 100; Delta_3 = full minus isolated at 100.
- Efficiency: for thresholds t in {17 (Piers level), 20}, C_c,s(t) = cumulative model calls (revise, repair and
  teach) at the first generation where B_c,s reaches t on the frozen ladder's evaluation; censored at the run's total
  calls. Reported as the ratio of medians across conditions with a bootstrap interval; censoring is reported, not
  imputed.
- Compatibility: X_c,s(g) = mean between-group cross-play (organized), mean all-pairs cross-play (isolated and
  full). Constraint contrast Delta_X = X_organized(100) - X_isolated(100).
- Retained innovations (artifacts that beat the archive and still have descendants 10 generations later), per
  condition.

## 4. Design and inference

- Paired by population seed across the three conditions. Pilot: 3 seeds at 30 generations on the hosted model
  (about 2,200 calls), to measure the between-seed SD of Delta_1 at generation 30 and the per-call price. Then R by
  the power rule (80%, two-sided alpha 0.05, minimum effect 2 points), capped at 10 seeds.
- Intervals: bootstrap over seeds (stratified), IQM point estimates. The per-seed outcome is a mean over 8 agents,
  so the seed-level SD already carries the within-population dependence.
- Second block: the same spec on a second hosted model with the same seeds; acceptance needs both blocks.
- Every curve is plotted against generations and against cumulative calls and tokens.

## 5. Decision rules, fixed now

- Claim "organization accumulates faster" if Delta_1 >= 2 points with the 95% interval excluding 0, in both blocks,
  and Delta_X's interval is not entirely below -1 point (decision 5).
- Claim "groups add over full sharing" only if Delta_2 >= 1 point with the interval excluding 0 in both blocks;
  otherwise report that full sharing accounts for the gain at this scale.
- Efficiency claim "organized reaches Piers level in fewer calls" if the median ratio C_isolated(17)/C_organized(17)
  >= 1.5 with the interval above 1, in both blocks; otherwise no efficiency claim is written.
- Falsification: if Delta_1's interval includes 0 at R = 10, the one-pager's accumulation sentence is not written;
  if full beats organized, the group-level claim is dropped and the finding reported as such.

## 6. Stopping and spend

- Fixed horizon of 100 generations. A hard cap in the spend file stops every job before the next paid call; a run
  stopped by the cap is reported as censored at the generation reached, with all metrics up to there.
- Pilot cap and full-run cap to be set by Enrico (decision 1). Estimate at 10.8k tokens per call: 3 x 5 x 100 x 8 =
  12,000 revise calls plus about 5% repairs plus teach calls (up to 8 per generation in the sharing conditions);
  about 150 M tokens for block 1. Dollars follow from the provider's price page; the previous session's estimate was
  about $30 per block.

## 7. Multiplicity and null calibration

Confirmatory tests with p-values: Delta_1, Delta_2 (two). BH at q = 0.1 across the ledger. The efficiency and
compatibility rules are acceptance criteria. Null-stub bands on every figure.

## 8. Outputs

Figure 1: mean and best self-play against generation and against cumulative calls, three conditions, seed bands,
null band, Piers and IGGI lines. Figure 2: between-group cross-play against generation. Table: generation-100 means
and contrasts with intervals; calls to 17 and 20 per condition; retained innovations. Ledger row with outcome.

## 9. Nearest prior work

Peer-verified accumulation in LLM populations (POLIS, arXiv 2507.21166); generational cooperation (Vallinder and
Hughes 2024, arXiv 2412.10270); transmission-chain studies (Perez et al. 2024). Group-structured populations of
program-writing agents at matched budget, with a frozen ladder and pre-registered contrasts, were not found in the
2026-10-06 searches; the claim phrasing must follow files/prereg/novelty-notes.md and a fresh search before any
public document.
