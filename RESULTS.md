# Results

One line per finding, under the question it answers. Status: *draft* = Enrico has not read it; *preliminary* =
pilot size or one model; *solid* = powered or unambiguous; *overturned* = contradicted later, with the link.
Everything so far is on one local model (Qwen3.6-35B-A3B, seeded, temperature 0) unless it says engine-only or
stub. Details and figures: `findings/`.

## Does organization make a population accumulate faster than isolation?
- Nothing on a model yet. Stub pilots of the population designs run and bound the costs
  (`findings/2026-10-07-population-pilots-on-stub.md`, *preliminary*).

## Can the swarm tell who helped (credit)?
- Only full counterfactual replay recovers planted per-message credit; leave-one-out, equal split and pooled
  regression all miss, and the between-seed spread of true credit (about 2 points) is a floor for anything that
  pools. Pooled regression does recover the seed-averaged credit to within half a point.
  (`findings/2026-10-07-credit-estimators.md`, *preliminary*: one model.)

## Do bad ideas spread, and what stops them?
- A rejected message still moves the learner: its prose in the revision prompt lowers the student by 2.4 points
  even when its code is never adopted. Verification gates code, not ideas.
  (`findings/2026-10-07-rejected-prose-still-hurts.md`, *preliminary*.)
- Quarantining unverified text removes that effect by construction; at temperature 0 there is nothing left to
  measure, so the empirical content is the line above. (`findings/2026-10-07-quarantine-at-temperature-zero.md`.)
- On the stub, without verification the group collapses at a 4% sabotage rate; with verification it is flat up to
  50% and 0 of 692 bad artifacts were adopted. Pipeline proof, not a model result.
  (`findings/2026-10-06-breakdown-on-stub.md`, *preliminary*.)

## What transfers, and at what cost?
- A sequential test for "is the received bot better than mine" matches a fixed 400-game check with 73% fewer games;
  the current adopt-if-mean-positive rule on 200 games wrongly adopts 6% of the time.
  (`findings/2026-10-06-sequential-verification.md`, *solid*, engine-only.)
- Copying a 17-point bot from code, prose, or both loses 1.2 to 1.8 points on average with no difference between
  media; copies are bimodal (exact-behaviour clones plus a tail of failed copies), and prose copies have a 5%
  illegal-move rate. (`findings/2026-10-07-copy-fidelity-by-medium.md`, *preliminary*.)

## Where is the ceiling: the model or the game?
- The local model cannot improve the 17-point reference bot: 0 of 30 single revisions, and two 40-generation chains
  never left its behaviour; at the 16-point bot it improves 1 in 6; at a 3-point bot 2 in 3 with gains of 9.
  (`findings/2026-10-07-local-model-ceiling.md`, *solid* for this model.)
