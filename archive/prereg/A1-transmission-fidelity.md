# Pre-registration A1: copy loss, copy variance, and the accumulation prediction

Status: draft 2026-10-06, to run after D1.

## 1. Question
How faithfully does a student reproduce a received strategy, by medium (code only, prose only, both), and does
Henrich's accumulation condition, fitted to the measured parameters, predict whether a small population improves?
Edge item 4 (a formal model that predicts when organization beats scale).

## 2. Setting
One student with a weak incumbent (self-play about 3). Teacher artifact T = Piers, v_T = 16.99 on the shared deals.
Medium mu in {code, prose, both}: `code` sends bot.py only; `prose` sends conventions.md only (the student must
reimplement); `both` sends both. One revision call per replicate; verification off (we want the copy, not the filter).
30 sampling seeds per medium on the MLX model; same deals for all.

## 3. Estimands
- Copy loss alpha(mu) = v_T − E[V(S', S')] and copy dispersion beta(mu) = the Gumbel scale of V(S', S') across seeds
  (sd × sqrt(6)/pi), both with bootstrap intervals.
- Sufficient-statistic ratio rho(mu) = E[V(S', T)] / v_T: cross-play of the student's copy with the original, over
  self-play of the original. rho near 1 means the medium carried the compatibility-relevant content.
- Accumulation prediction: delta(mu, N) = −alpha(mu) + beta(mu) (0.577 + ln N) for N = 4 (the number of candidates the
  selection step ranks in the check population). Sign predicts improvement.

## 4. Check population
For each medium: 4 students, copy-the-best-with-verification-off, 10 generations, 10 population seeds. Observed sign
= sign of (population mean at generation 10 minus generation 1).

## 5. Decision rules, fixed now
- Medium ordering claim "code transmits with less loss than prose" if alpha(prose) − alpha(code) ≥ 1 point with the
  bootstrap interval excluding 0.
- Model support claim if the predicted sign matches the observed sign in at least 8 of 10 population seeds for each
  medium, in both model blocks. If the prediction fails for a medium, report it; the model's additive-noise assumption
  is then the first suspect and is stated as such.

## 6. Compute
90 calls for the parameters plus 3 × 4 × 10 × 10 = 1,200 for the check populations; one night on MLX; $0.

## 7. Nearest prior work
LLM telephone-game and cultural-evolution line (Perez et al. 2024, arXiv 2407.04503 and 2403.08882; Vallinder and
Hughes 2024, arXiv 2412.10270; POLIS 2025, arXiv 2507.21166). Henrich's parameterization and a code-versus-prose
comparison were not found in the 2026-10-06 arXiv check.

## 8. Amendments before the parameter run (2026-10-07 01:34 EDT; logged, not silently edited)

Recorded before any model call for A1. Decision rules (section 5) unchanged.

1. **A1-1, source of replicate variation.** Section 2 says "30 sampling seeds per medium; same deals for all". The
   program samples at temperature 0, where the sampling seed does not change the output, so 30 sampling seeds on one
   prompt would give 30 identical copies and beta = 0 by construction. Replicate r instead uses experiment seed
   300 + r (the feedback traces shown in the prompt differ, as in C1's replicates) and sampling seed r. "Same deals for
   all" is kept for measurement: every copy, the teacher and the student's incumbent are scored on one common set of
   300 evaluation deals (experiment seed 299), and v_T is Piers' self-play on those deals (paired), not the 16.99 of
   the 1,000-game anchor table. Copy dispersion beta is therefore dispersion across prompt contexts at temperature 0,
   which is the noise source the check populations have; it is not sampling noise.
2. **A1-2, no adoption and failed copies.** "Verification off" is implemented as verification none plus adoption
   `never`: the payload is never installed by the harness, so S' is whatever the one revision call writes from the
   student's incumbent S0 with the message in view. The message carries a content-free cover note identical across
   media ("I am sending you my strategy below. Use whatever helps you.") and the harness's evidence for Piers; the
   medium decides whether Piers' bot.py, conventions.md or both are appended to it. A revision that is unparseable or
   inadmissible after the one repair call leaves the student with S0, so S' = S0 for that replicate (the student's
   artifact after its revision step, which is what selection sees in a population). Failures are counted per medium and
   the parameters are also reported over admissible copies only.
3. **Scope note.** The student's bot and Piers are the same rule-bot template with different CONFIG lines, and Piers'
   conventions name rule functions that exist in the student's own code. Transmission is therefore easier here than
   between unrelated codebases; alpha should be read as a lower bound on copy loss for that harder case.
