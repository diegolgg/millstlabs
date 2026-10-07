# Hypothesis ledger

Every pre-registered experiment, in order, with its outcome. Negatives stay. BH at q = 0.1 across confirmatory tests.

| id | question | pre-registered | run | confirmatory tests | outcome |
|---|---|---|---|---|---|
| C1 | which teacher-credit estimator recovers planted causal effects at matched cost | 2026-10-06 (final; ITT primary, Shapley alongside, adopter effect secondary); amended 2026-10-07 (T3 is a placebo, not a known zero) | pilot only: 5 seeds x 2 strata on MLX, 74 calls, docs/results/c1-pilot.json | 9 | pilot: only full-replay estimators meet RMSE<=2; Spearman 1.0 for ridge/PB/singles-pairs; R=29 (verified), 8 (unverified); full run pending decision |
| B1 | SPRT vs fixed-N verification on anchor-family pairs | research program section B1 | engine only, 360 pairs, docs/results/b1-sprt.json | 1 | SPRT 0.3% false adoptions, 10% misses near delta=1, 109 deals; matches fixed N=400 with 73% fewer deals: passes the 30% rule |
| D1 | breakdown point of execution-based verification vs sabotage fraction | draft 2026-10-06 | pilot on null stub only (4 students, 10 gens, 5 eps, 2 arms, 5 seeds) | 2 | stub: without verification skill 10.2 -> 1.8 at eps=0.1 (breakdown ~0.04); with verification flat, 0/692 sabotaged adopted; model run pending |
| A1 | copy loss and copy variance by medium; Henrich sign prediction | to draft | not yet | 1 | pending |
| C2 | does quarantining unverified messages from the revision prompt remove bad-idea influence | 2026-10-07 01:00 | not yet | 2 | pending |
