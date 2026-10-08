# Breakdown point of verification, on the null stub only

**Status:** preliminary (stub; proves the pipeline, not the claim)
**Date:** 2026-10-06 · **Label:** none · **Archive:** `archive/prereg/D1-breakdown-point.md` · **Data:** `docs/results/d1-pilot.json`, `d1-pilot.png`

## Question
What fraction of sabotaged messages can a group tolerate before its skill collapses, with and without
execution-based verification before adoption?

## What we expected
Collapse without verification at a small sabotage rate; no collapse with it.

## Setup
Four students, 10 generations, broadcast inside the group; each delivered message replaced with probability ε by
the 0-point bot with persuasive prose and honest evidence; ε ∈ {0, 0.1, 0.2, 0.3, 0.5}; verification on (200 games,
adopt if better) or off; 5 seeds; stub revisions (keep the rule list), so skill moves only by copying and sabotage.

## Result
| ε | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| group mean, no verification | 10.2 | 1.8 | 1.7 | 0.0 | 0.0 |
| group mean, verification | 15.7 | 15.8 | 15.7 | 15.8 | 15.9 |

Breakdown without verification ≈ 0.04; with verification censored above 0.5; 0 of 692 sabotaged messages adopted.
Even at ε = 0 the unverified group ends lower (10.2 vs 15.7): blind adoption also takes worse honest bots.

## What it changes
The pipeline for contagion experiments works (sabotage is common-random-numbers across ε, so cells are paired).
On the stub the outcome is forced; the model run is what matters, and it should be designed as a threshold test
(fan-out × leak) rather than a curve: proposed idea 3.

## Caveats
Stub only. No model, no prose channel.
