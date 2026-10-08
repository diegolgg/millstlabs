# Credit: only full counterfactual replay recovers who helped

**Status:** preliminary (one model; the second-model replication never ran)
**Date:** 2026-10-07 · **Label:** none (predates the notebook) · **Archive:** `archive/prereg/C1-credit-estimators.md`,
`archive/docs/sandbox1-status.md` "C1 verdicts" · **Data:** `docs/results/c1-full.json`, figure `docs/results/c1-full.png`

## Question
A student receives three messages in one generation. Which estimator of each message's causal effect on the
student is accurate enough to run inside the organization, and at what cost in model calls?

## What we expected
(Pre-registered.) An estimator is usable if, per seed, its rank correlation with the truth is ≥ 0.9, its RMSE is
≤ 2 points, and it gets the sign of the harmful message right ≥ 95% of the time. The hope was that randomized
delivery plus regression (no extra calls) would pass.

## Setup
One weak student (self-play ≈ 3). Three fixed teachers: a 17-point bot with plain prose; a 0-point bot with
persuasive prose and honest low evidence; the student's own bot re-sent. Truth = exact replay of all 8 delivery
subsets under the same seed (the model is bit-reproducible at temperature 0), giving each message's effect as the
average over the others' delivery. 29 seeds with verification on, 8 with it off; 295 local calls; $0.

## Result
| estimator | replays per seed | RMSE, verification on | RMSE, off | sign error on the bad message |
|---|---|---|---|---|
| full replay (truth) | 8 | 0.67 [0.53, 0.81] | 0.71 | 4% |
| leave-one-out | 4 | 2.83 [2.54, 3.23] | 4.50 | 88% / 0% |
| equal split among adopted | 1 | 3.49 | 7.22 | 100% / 50% |
| pooled regression on random delivery | 8 shared | 2.28 [1.98, 2.52] | 3.92 | 19% / 0% |

Effects themselves (verification on): the good bot +10.0 [9.1, 10.8], adopted every time; the bad bot −2.2
[−2.9, −1.4], never adopted; the re-sent own bot +0.7 [−0.5, 1.6].

Why the cheap ones fail: the true per-message effect varies across seeds with a spread of about 2.2 points
(verification on) and 3.2 (off). Anything that pools across seeds has that as an RMSE floor. Against the
seed-averaged effect instead, pooled regression is within 0.47 [0.17, 1.10] points, which is what it actually
estimates.

## What it changes
Per-student credit from one generation of logs is not available with three messages per learner. Two ways out:
fewer messages per learner so a single replay suffices, or settle for population-average credit, which pooled
regression gives for free. This is the origin of proposed idea 1 (when does credit become recoverable).

## Caveats
One model. Three messages only. The "placebo" (own bot re-sent) is not a null: any delivered text changes the
revision, so there is no zero-effect message to calibrate against.
