# Quarantine removes the prose effect by construction; at temperature 0 there is nothing left to measure

**Status:** preliminary (5-seed pilot)
**Date:** 2026-10-07 · **Label:** none · **Archive:** `archive/prereg/C2-quarantine.md` · **Data:** `docs/results/c2-pilot.json`

## Question
If unverified or rejected messages are hidden from the revision prompt entirely (quarantine), does the harm from
bad prose go away, and does it cost anything for good teachers?

## What we expected
(Pre-registered.) The bad bot's effect under quarantine should sit in [−1, 1]; the good bot's effect under
quarantine within a point of its effect without.

## Setup
Same student and teachers as the credit run; quarantine on; 5 seeds; 11 local calls (most delivery subsets
collapse to the same prompt).

## Result
Under quarantine the eight delivery subsets collapse to two prompts (nothing delivered; the good bot only), and
the model is deterministic, so the bad bot's effect is exactly zero on every seed. The pre-registered primary
quantity is therefore mechanical, not empirical. The student's own outcome with everything delivered: quarantine
equals plain verification on 4 of 5 seeds and is +0.8 on the fifth.

## What it changes
Quarantine works, trivially. The real question is the trade-off it creates: it also hides a good idea whose code
failed verification. That is what proposed idea 6 tests (quarantine vs idea-trial vs evidence-gated reading),
and it needs a planted good-idea-with-a-bug to be meaningful.

## Caveats
Pilot size; the design as pre-registered cannot fail at temperature 0 and should not be re-run as written.
