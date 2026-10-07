Scope: C1 Arm 1 only (k = 3), one model block (Qwen3.6-35B-A3B 4-bit on MLX, seeded, temperature 0); the Gemma replication block and Arm 2 (k = 7) were not run, so no estimator can be accepted, only rejected or passed in this block. One student, one generation, 300 held-out deals; R = unverified 8, verified 29 seeds (the pilot's 5 included). Null band from the null stub (K = 20), not from a model.

| stratum | estimator | role | replays/seed | RMSE IQM [95% CI] | Spearman IQM [95% CI] | sign error T2 | null RMSE IQM [min, max] | passes this block |
|---|---|---|---|---|---|---|---|---|
| verified (R=29) | leave_one_out | confirmatory | 4 | 2.83 [2.54, 3.23] | 0.87 [0.84, 0.89] | 0.88 (of 26) | 0.08 [0.01, 0.19] | no |
| verified (R=29) | equal_split | confirmatory | 1 | 3.49 [2.99, 4.00] | 0.87 [0.87, 0.87] | 1.00 (of 26) | 0.10 [0.00, 0.34] | no |
| verified (R=29) | ridge_bernoulli | confirmatory | 8 | 2.28 [1.98, 2.52] | 1.00 [0.90, 1.00] | 0.19 (of 26) | 0.27 [0.08, 0.41] | no |
| verified (R=29) | tau_itt | exploratory | 8 | 0.67 [0.53, 0.81] | 1.00 [1.00, 1.00] | 0.04 (of 26) | 0.01 [0.00, 0.09] | yes |
| verified (R=29) | plackett_burman | exploratory | 8 | 0.67 [0.53, 0.81] | 1.00 [1.00, 1.00] | 0.04 (of 26) | 0.01 [0.00, 0.09] | yes |
| verified (R=29) | singles_pairs | exploratory | 7 | 2.67 [2.11, 3.26] | 1.00 [1.00, 1.00] | 0.23 (of 26) | 0.06 [0.00, 0.37] | no |
| unverified (R=8) | leave_one_out | confirmatory | 4 | 4.50 [3.45, 5.32] | 0.50 [0.50, 0.88] | 0.00 (of 8) | 5.80 [5.46, 5.90] | no |
| unverified (R=8) | equal_split | confirmatory | 1 | 7.22 [6.40, 7.97] | 0.00 [0.00, 0.22] | 0.50 (of 8) | 6.94 [6.87, 7.02] | no |
| unverified (R=8) | ridge_bernoulli | confirmatory | 8 | 3.92 [3.36, 4.28] | 1.00 [0.88, 1.00] | 0.00 (of 8) | 0.77 [0.69, 0.85] | no |
| unverified (R=8) | tau_itt | exploratory | 8 | 0.71 [0.27, 1.28] | 1.00 [1.00, 1.00] | 0.00 (of 8) | 0.01 [0.00, 0.07] | yes |
| unverified (R=8) | plackett_burman | exploratory | 8 | 0.71 [0.27, 1.28] | 1.00 [1.00, 1.00] | 0.00 (of 8) | 0.01 [0.00, 0.07] | yes |
| unverified (R=8) | singles_pairs | exploratory | 7 | 2.83 [1.08, 5.11] | 1.00 [1.00, 1.00] | 0.00 (of 8) | 0.04 [0.00, 0.29] | no |
