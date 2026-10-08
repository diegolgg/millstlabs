# The local model cannot improve on the 17-point bot

**Status:** solid for this model (Qwen3.6-35B-A3B 4-bit)
**Date:** 2026-10-07 · **Label:** none · **Data:** `docs/results/p1-eps-ladder.json`, `docs/results/p1-micro-params.json`

## Question
How often does one revision improve a bot, as a function of how good the bot already is?

## What we expected
A falling rate with level; the open question was whether it reaches zero before the game's ceiling (25).

## Setup
Independent single revisions of a fixed incumbent, feedback traces varying across 30 replicates, held-out
paired evaluation on 300 deals; plus two 40-generation chains from the 17-point bot under accept-if-not-worse;
140 local calls. The weak-level number comes from the credit run's "nothing delivered" replays.

## Result
| incumbent level | P(improve by > 0.5) | gain when it improves |
|---|---|---|
| 2.75 (weak student) | 20 of 29 = 0.69 [0.51, 0.83] | 8.7 ± 3.3 |
| 15.8 (the 16-point bot) | 5 of 30 = 0.19 [0.07, 0.34] | 0.6 to 1.8 |
| 17.0 (the 17-point bot) | 0 of 30 [0.00, 0.11] | — |
| 17–18 (chains) | 1 of 58 | 0.86 |

The chains' incumbents never left the 17-point bot's behaviour on common deals; most revisions returned the
code unchanged.

## What it changes
For this model the ceiling is about 17, so accumulation experiments on it can only show the climb from weak to
17, and a 20-point threshold is unreachable. Any hosted model gets the same 30-call ladder before it is used
(proposed idea 8). It also killed the earlier draft prediction that populations saturate at 25: that came from
extrapolating the weak-level rate upward.

## Caveats
One model; one prompt; the chain is 40 generations.
