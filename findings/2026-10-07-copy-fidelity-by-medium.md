# Copying a good bot from code, prose, or both: same loss, different failure modes

**Status:** preliminary (one model; the check populations never ran)
**Date:** 2026-10-07 · **Label:** none · **Archive:** `archive/prereg/A1-transmission-fidelity.md` · **Data:** `docs/results/a1-params.json`

## Question
When a weak student is sent a 17-point bot as code, as prose, or as both, and revises once, how good is the copy?

## What we expected
(Pre-registered.) Code transmits with less loss than prose by at least a point.

## Setup
Student at 2.5; teacher at 16.7 on the common 300 deals; 30 replicates per medium (feedback traces vary, since
the model is deterministic); verification off and no adoption, so the copy is whatever one revision writes with
the message in view; 90 local calls.

## Result
| medium | copy loss (points below teacher) | cross-play with teacher / teacher's self-play | rule list identical to teacher | illegal moves |
|---|---|---|---|---|
| code | 1.66 [0.49, 3.15] | 0.94 | 12 of 30 | 0% |
| prose | 1.82 [1.00, 2.77] | 0.94 | 0 of 30 | 5% |
| both | 1.19 [0.51, 2.25] | 0.96 | 1 of 30 | 0.6% |

Prose minus code: 0.16 [−1.51, 1.75]. No medium claim. The copy distribution is bimodal: a cluster at the
teacher's level (code: exact clones; prose: re-implementations) plus a tail of failed copies near the student's
own level. Some copies score above the teacher.

## What it changes
Fidelity is a success probability plus a within-cluster loss, not a mean and a variance. For accumulation models
the question is whether a copy can exceed its teacher (it did, rarely); that single assumption decides whether
sharing ratchets to the ceiling. Prose copies cross-play as well as code copies on average, but the illegal-move
rate says they differ in edge cases: proposed idea 10.

## Caveats
Student and teacher share one rule-bot template, so copying is easier than between unrelated codebases.
