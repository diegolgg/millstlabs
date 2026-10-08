# Pre-registration D1: breakdown point of execution-based verification against sabotaged messages

Status: draft 2026-10-06, to run after C1. Thresholds approved in chat (70% collapse criterion; 0.2 difference).

## 1. Question
At what fraction of sabotaged messages does a population's skill collapse, and does execution-based verification
before adoption raise that fraction? Edge item 5 (designing against persuasive-but-wrong ideas).

## 2. Setting
- One group of n students (pilot n = 4, G = 10 generations; main n = 8, G = 20), broadcast routing inside the group,
  local MLX model, seeded. Each delivered message is independently sabotaged with probability epsilon: the payload is
  replaced by a Flawed-family artifact with persuasive prose and honest (low) harness evidence. Sabotage is a Bernoulli
  treatment on messages with known probability, so exposure probabilities are known (Wager Thm 12.1).
- Factor 2: verification on (`selfplay(n=200)`, `replace_if_better`) versus off (adopt what is delivered).
- epsilon in {0, 0.1, 0.2, 0.3, 0.5}. Paired deals across all cells; three disjoint deal sets per generation.

## 3. Estimand
Interference is real inside the group (one student's adoption changes what others receive), so the unit of analysis is
the group-generation and the outcome is the population mean held-out self-play at generation G, Y_G(epsilon, v).
Exposure contrast (Wager eq. 12.2): tau(epsilon, v) = E[Y_G(epsilon, v) − Y_G(0, v)] for v in {on, off}.
Breakdown point epsilon*(v) = the smallest epsilon on the grid with E[Y_G(epsilon, v)] < 0.7 E[Y_G(0, v)], with
linear interpolation between grid points for the point estimate.

## 4. Design and inference
- Population seeds: pilot 5 per cell on the null stub, then on the model; main R from the pilot SD for 80% power on
  the primary contrast at alpha 0.05, capped at 30 per cell.
- Variance: finite-population estimator with the randomization dependency graph of the group (all students share
  teachers, so G is dense); PSD-projected HAC variance (Wager Cor. 12.6). Sharp-null permutation test of no effect of
  epsilon within each verification arm (Wager Thm 11.1), permuting sabotage indicators under the design.
- Primary contrast: epsilon*(on) − epsilon*(off), bootstrap over population seeds, stratified.
- Secondary: the per-message influence function from C1 applied here (influence of a sabotaged message on Y_G);
  adoption rate of sabotaged messages by arm; time to collapse.

## 5. Decision rules, fixed now
Claim "verification raises the breakdown point" only if epsilon*(on) − epsilon*(off) ≥ 0.2 with the bootstrap 95%
interval excluding 0, in both model blocks. Claim "verification is not sufficient" if epsilon*(on) ≤ 0.3.

## 6. Null calibration and multiplicity
Null stub run at every epsilon gives the null band for Y_G. Two confirmatory tests (the contrast; the permutation test
in the verification-off arm); BH across the ledger.

## 7. Compute
Pilot: 4 students × 10 generations × 5 epsilon × 2 arms × 5 seeds = 2,000 model calls, about 3 hours at 5 s per
call on MLX (short bots) and more for long ones; main run scales linearly. $0.

## 8. Nearest prior work
Byzantine-robustness thresholds for consensus and aggregation (Liu et al. 2026, arXiv 2604.17139; Lee et al. 2026,
arXiv 2605.09076); memory-poisoning gates with point attack rates (MAPLE-Guard 2026, arXiv 2608.00426). A breakdown
curve for execution-based verification in a learning population was not found in the 2026-10-06 arXiv check.
