# Verification as a sequential test: same accuracy as 400 games with 73% fewer

**Status:** solid (engine only; no model involved)
**Date:** 2026-10-06 · **Label:** none · **Data:** `docs/results/b1-sprt.json`, `b1-sprt.png`

## Question
Adopting a received bot means deciding whether it beats the incumbent on paired games. How many games does a
sequential test need compared with a fixed number, at the same error rates?

## What we expected
Wald's sequential test should match a fixed test's error rates with at least 30% fewer games near a 1-point
improvement (its expected sample size scales like 2σ² log(1/α)/δ² and σ ≈ 3.6 here).

## Setup
A bank of 70 reference-family bots (one- and two-rule mutations of the 17- and 16-point bots, plus two graded
families), 360 ordered pairs, true differences from 2,000 games each; tests on ten separate 600-game blocks per
pair.

## Result
| rule | false adoption (δ ≤ 0) | missed, 1 ≤ δ ≤ 1.25 | games near δ = 1 |
|---|---|---|---|
| sequential (α = β = 0.05, δ₁ = 1) | 0.3% | 10% | 109 |
| fixed 200, t-test | 0.4% | 12% | 200 |
| fixed 400, t-test | 0.4% | 0% | 400 |
| fixed 200, adopt if mean > 0 (the harness default) | 6.1% | 0% | 200 |

## What it changes
The sequential test is the default verification rule. The harness's current "adopt if the mean is positive"
rule leaks 6% of non-improvements, which is the leak rate a that enters the contagion threshold (idea 3).
Verification cost is also what makes "many small generations" affordable (idea 11).

## Caveats
Reference-family bots only; model-written bots may have different per-game variance. Engine time only.
