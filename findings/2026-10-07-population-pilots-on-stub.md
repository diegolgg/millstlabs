# Population designs on the stub: what they cost and what the pipeline gets wrong

**Status:** preliminary (stub; infrastructure facts)
**Date:** 2026-10-07 · **Label:** none · **Data:** `docs/results/s1-stub-pilot.json` (partial), agent worktree `runs/g1s1-stub`

## Question
Before any paid population run: how long does a generation take, is the budget actually matched across
organizations, and does the switchback analysis behave on a null?

## Setup
Eight agents; three organizations (isolated = 8 groups of 1; full sharing = 1 group of 8; two groups of 4 with
migration); 50 to 90 generations; stub revisions (hill-climbing on rule lists); 10 processes in parallel.

## Result
- Wall clock: about 45 s per generation alone, 89 s with ten runs sharing the machine; engine time dominates
  (verification: 7 messages × 200 games per agent per generation, about 5,500 games per generation).
- Budget is not matched: teaching is a model call per sender, so sharing organizations make about 16 calls per
  generation against 8 for isolated. The cap never binds. Comparisons must be on a calls axis, or isolated must
  run longer, or teaching must not cost a call (proposed idea 5).
- Switchback contrasts on the stub (teaching, verification, quarantine toggled every 30 generations) are all
  about −0.07 points per generation with bias bounds of 0.2 to 0.9: everything below its own bound, as it should
  be for a stub that plateaus by generation 10.
- No budget refusals; lever switches took effect on the right generations; resume across a switch is
  byte-identical.

## What it changes
Population runs are feasible at 8 agents × 100 generations in a night on a hosted model; verification cost,
not model cost, is the engine bottleneck, which argues for the sequential test. The budget mismatch is a design
decision Enrico has to make before any accumulation claim.

## Caveats
Stub only; the organized-vs-isolated numbers on the stub mean nothing.
