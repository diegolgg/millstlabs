# Pre-registration C2: does withholding unverified messages from the revision prompt remove the influence of bad ideas?

Status: pre-registered 2026-10-07 01:00, before any run. Motivated by the C1 pilot finding (amendment 2): with
verification on, a sabotaged message that was never adopted still lowered the student by 3 to 5 points because its
prose and evidence were visible to the LLM during revision.

## 1. Question
Verification currently gates adoption of code. Does a `quarantine_unverified` policy, under which a message is shown to
the revision prompt only after it passes verification, remove the sabotaged message's effect while preserving the good
teacher's effect? Edge item 5.

## 2. Setting
Identical to C1 (one student, k = 3 teachers: Piers, Flawed with persuasive prose and honest low evidence, the
student's own incumbent re-sent as a placebo; exhaustive delivery over 8 subsets; MLX, seeded). One new stratum:
`verified_quarantine` = verification on plus quarantine. Compared against C1's `verified` stratum on the same seeds.

## 3. Estimands
Intent-to-treat effects tau_j per message (C1 section 3), under quarantine and under plain verification.
Primary quantity: tau_flawed(quarantine) − tau_flawed(verified), paired by seed.
Secondary: tau_piers(quarantine) − tau_piers(verified), paired by seed (the cost of quarantine on a good teacher).

## 4. Design
Pilot: 5 seeds (the C1 pilot seeds), 8 replays each, about 40 model calls. Then R from the pilot SD for 80% power at
a 2-point minimum effect on the primary quantity, capped at 29 (the C1 verified R), so the same seeds serve both.

## 5. Decision rules, fixed now
Claim "quarantine removes the influence of unverified bad ideas" if the primary quantity's IQM ≥ 2 points with the
stratified-bootstrap 95% interval excluding 0, and tau_flawed(quarantine) has IQM within [−1, 1]. Claim "quarantine is
costless for good teachers" if the secondary quantity's interval contains 0 and its IQM is above −1. Both in two
model blocks before either claim is written.

## 6. Null calibration, multiplicity
Null stub run of the same design gives the band. Two confirmatory tests; BH across the ledger.

## 7. Nearest prior work
Memory-poisoning gates that quarantine writes before promotion (MAPLE-Guard 2026, arXiv 2608.00426) are the nearest
mechanism; they report attack-success rates, not the effect on a learner's revision. Not found: an estimate of the
influence of a rejected message's text on an LLM agent's subsequent revision.
