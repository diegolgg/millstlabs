# P1 draft: micro-parameters and a first prediction of G1 (2026-10-07, $0, zero model calls)

**Terms.** *Skill*: an agent's held-out self-play score (0 to 25). *Copy*: the artifact a student ends up with after
being sent a teacher's artifact. *Fidelity mixture*: a copy either succeeds (lands within 1 point of the teacher) with
probability p_s, with a small loss inside that success cluster, or fails. *Innovation rate epsilon*: the probability
that one revision raises skill by more than a margin (0.5 points unless stated). *Verification*: the paired test that
decides whether a received artifact replaces the incumbent; *false adoption* is adopting something no better, *miss* is
rejecting a real improvement. *Exemplar*: an artifact an agent can see and copy.

## What was measured, and from where

| quantity | value [95% interval] | source (observations) |
|---|---|---|
| p_s, medium both (code + prose) | 0.57 (17/30) [0.39, 0.73] | a1-params.json `V_SS.values` (30 copies of Piers, v_T = 16.7) |
| p_s, code / prose | 0.63 [0.45, 0.78] / 0.40 [0.25, 0.58] | same (30 each) |
| loss in success cluster, both | mean 0.07, SD 0.55 (copies can beat Piers) | same (17) |
| failed copies, both | mean score 14.0 (13 copies; only one below 13) | same |
| EM two-Gaussian fit, both | upper component weight 0.97, mean 15.9, SD 1.0 | same |
| epsilon, weak student (2.75) | 0.69 (20/29) [0.51, 0.83]; gain given gain 8.7 (SD 3.3) | c1-full.json `v['0']`, 29 unique seeds |
| epsilon, Piers level (17.0), Piers' message in prompt | 0.17 (5/29) [0.08, 0.35]; gain 0.71 (SD 0.11) | runs/c1-full/replays.jsonl mask 1 (29) |
| same at margin 1.0 | 0/29 [0.00, 0.12] | same |
| verification, current rule (200 deals, mean > 0) | false adoption 6.1%, miss near delta = 1: 0% | b1-sprt.json (1,880 / 170 tests) |
| verification, SPRT | false adoption 0.3%, miss near delta = 1: 10% | same |

Files: `docs/results/p1-micro-params.json` (rules and provenance), `p1-prediction.json`, `p1-prediction.png`.

## The prediction (draft, not pre-registered)

Eight agents start at 2.75; 1,000 Monte Carlo runs. Order **full > organized > isolated** at every generation.
Generations to a population mean of 17: full 4, organized 5, isolated 10; to 20: 8, 10, 32. The organized-minus-isolated
gap peaks at **6.0 points near generation 21**, then shrinks as isolated agents catch up. **At generation 100 all three
are at the cap** (24.9 to 25.0), so the gap there is 0.1. Fidelity, verification rule and migration barely move this:
the peak gap stays between 5.5 and 6.6 across those rows. What moves it is innovation above the Piers level.

## Assumptions (one per line)

1. Skill is one number; copying and innovation act on it additively.
2. Copy loss measured for one teacher at 16.7 applies at every teacher level.
3. **Most suspect: epsilon and the gain above 17 stay at their Piers-level values (0.17, +0.7) up to the cap.** Nothing
   above 17 was measured, and this alone makes every population saturate.
4. Between 2.75 and 17, epsilon is linear in skill and the gain distribution interpolates between the two measured ones.
5. The Piers-level epsilon (measured with Piers' message in the prompt) stands in for an agent revising alone.
6. Each agent copies only the best exemplar it can reach; several messages in one prompt are not modelled.
7. Verification errors are the engine-bot rates from B1, applied to model-written candidates.
8. Innovation never loses skill (accept-if-not-worse); copy noise can exceed the teacher, so copying ratchets upward.
9. All parameters come from local Qwen; G1 is planned on a hosted model.

## Missing, and what fills it

- Epsilon at Piers level with nothing delivered: 30 revisions of a Piers incumbent, local MLX, about 25 to 30 minutes.
- Epsilon and gain at about 8, 12 and 18 (a level ladder): about 90 calls, 75 minutes.
- Copy fidelity from other teachers (IGGI, a weak bot): about 30 calls, 25 minutes.
- Verification on model-written candidates: rescore A1 and C1 candidates with the B1 code, engine only, minutes.
- Everything on the G1 hosted model, and the G1 warm-start distribution (24 authoring calls).

## Three decisions for you

**1. How a copy is modelled.** *For:* the fidelity term of the model (edge item 4). *What:* choose between the 1-point
binary rule (p_s 0.57), the EM fit (p_s 0.97, loss 0.8) and exact adoption. *Why it matters:* the harness's
`replace_if_better` adopts the sender's code exactly, so A1's loss only applies when teaching sends prose or the
student rewrites. *Details:* in this model the choice changes the predicted peak gap by under 0.6 points. So it matters
for the medium claim, not for G1. My proposal: exact adoption for G1, mixture for a prose-only arm.

**2. Measure innovation at and above Piers before predicting.** *For:* making P1 a prediction rather than an
extrapolation. *What:* the ~120-call ladder above, overnight, $0. *Why:* assumption 3 decides the level at generation
100. With epsilon falling to zero at the cap, the generation-100 gap becomes 1.9. With the margin at 1.0 (no
measured improvement at Piers), it becomes 8.2. *Details:* the model takes any epsilon function
(`Innovation.epsilon_fn`), so the measured curve slots straight in.

**3. Which G1 outcome P1 should predict.** *For:* the P1 and G1 decision rules. *What:* keep the generation-100 mean,
or switch to generations (or calls) to reach 17 and 20, or the gap at generation 30 (the pilot's horizon). *Why:* if
isolated agents also reach the ceiling, a 2-point rule at generation 100 fails even when organization clearly
accelerates accumulation. The model puts the gap below 2 after generation 57. *Details:* this bears on G1's draft
pre-registration, whose primary is at generation 100. Migration as swaps or as cross-group copying changes nothing
here (peak 6.0 vs 6.2), so that question can wait.

## Update after the local epsilon ladder (2026-10-07 22:10)

The probe ran (140 local calls, $0; `docs/results/p1-eps-ladder.json`). At margin 0.5 the innovation rate is 0 of 30 at
the Piers level (95% upper bound 0.11), 5 of 30 at the IGGI level, and 1 of 58 in the 17-to-18 bin of the two
40-generation chains, whose incumbents never left Piers' behaviour on common deals. So the local model does not
innovate above about 17, and the earlier saturation at 25 was an artifact of extrapolating the weak-level rate.

With the measured ladder (`docs/results/p1-prediction-ladder.json`) the prediction splits on one modelling choice:
- Copy model with symmetric within-cluster noise (the default): full and organized still climb to the 25 cap, because
  a copy can come out slightly better than its teacher and verification keeps it, generation after generation. That
  ratchet is a property of the model, not something measured; A1 saw copies above Piers, but rarely.
- Copies never exceed the teacher, or exact adoption as the harness does: full and organized reach about 21 to 22 at
  generation 100, isolated about 17.3, gap about 4 points, still growing at generation 100.
Both readings agree on the order (full >= organized > isolated) and on isolated plateauing near the Piers level.
Decision 1 above (how a copy is modelled) therefore decides the headline prediction and must be settled, with the
hosted model's own ladder measured, before the P1 prediction is pre-registered.
