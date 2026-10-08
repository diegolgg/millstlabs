# Ideas

How this file works. An idea is one line, a prediction, and the cheapest run that could kill it. It starts under
*proposed* (ranked by what we learn per hour and dollar; Claude's ranking, Enrico overrules by editing). When
Enrico agrees it runs, it gets the next label `E##` in the table, a design note in `notes/`, and moves to
*agreed*. When it has run, its finding is in `findings/` and its line is in `RESULTS.md`. Ideas nobody wants
yet go to *parked* with the reason. Each idea says how it relates to Diego's work: orthogonal / overlaps /
builds-on (see `docs/DIEGO.md`).

Vocabulary (dials, measurements, reference bots) is defined in `notes/hanabi-and-swarms.pdf`.

## Labels

| label | idea | config | note | finding | status |
|---|---|---|---|---|---|
| (none yet) | | | | | |

## Agreed

(none yet; Enrico picks from below)

## Proposed

1. **When can the swarm tell who helped?** With 3 messages per learner, no cheap credit estimator recovered
   the planted effects: the between-seed spread of the true effect (about 2 points) is larger than the per-message
   effect we care about, so any estimator that pools across seeds has an error floor at that spread, and ones that
   don't pool are noise. Prediction: with 1 message per learner, one replay (with vs without) recovers credit to
   within 1 point; at 2 it still does; at 5 and 7 it fails even with 30 seeds, and the failure point is where the
   per-message effect divided by the between-seed spread drops below roughly 1. Cheapest run: one student, messages
   per generation in {1, 2, 3, 5}, 10 seeds, local model (exact replay), compare leave-one-out and full replay;
   plot error vs messages. ~400 local calls, free, one night. Why it matters: it says whether "credit agents for
   their students' improvement" can work at all, and under what organization (few messages per learner, verified
   large effects). Diego: orthogonal.

2. **Sweet-spot migration.** Two groups that never talk drift to incompatible conventions (cross-play collapses);
   one big group that shares everything explores one path. Prediction: between-group cross-play at generation 40
   rises monotonically with migration rate; best self-play is hump-shaped, peaking at a low rate (one swap every
   5 to 10 generations), and the peak moves toward more migration as the innovation rate falls. Cheapest run: two
   groups of four, 40 generations, migration in {0, 0.05, 0.25, 1} swaps per 5 generations, 3 seeds, hosted model.
   ~1,000 calls per rate. Diego: orthogonal (his groups don't migrate).

3. **Bad-idea contagion threshold.** A persuasive wrong artifact spreads if each carrier teaches k agents and
   verification lets it through with probability a, with k·a > 1 (a branching process). We measured a: 6% for the
   current adopt-if-mean-positive rule on 200 games, 0.3% for the sequential test, 100% with no check. Prediction:
   broadcast to 7 with verification is subcritical (0.4) and the bad artifact dies in 3 generations; without
   verification (k·a = 7) it is everywhere by generation 3; the sequential test is subcritical even at fan-out 50.
   Cheapest run: one group of 8, 20 generations, 20% of messages sabotaged, fan-out {1, 3, 7} × check {none, fixed
   200, sequential}, 3 seeds; outcome = fraction holding the bad artifact over time. Hosted, ~1,500 calls. Second
   channel to watch: rejected prose still moves learners (finding below), which verification cannot block and
   quarantine can. Diego: orthogonal.

4. **Organization vs more agents at a fixed number of calls.** Prediction: 4 agents with sharing beat 8
   isolated agents at the same total calls, and the gap grows with horizon (bigger at 100 generations than 20).
   Cheapest run: budgets {400, 800} calls, splits {4×100, 8×50, 16×25}, sharing on/off, 3 seeds. Hosted. This is
   the sentence a VC repeats, so it needs a real model (see 8). Diego: overlaps (his "fivefold" claim); worth
   doing on the same axis so the two sandboxes agree.

5. **Teach without a model call.** Teaching currently costs a model call per sender, so sharing conditions spend
   twice what isolated ones do and budgets can't be matched. Prediction: sending the artifact plus its conventions
   document verbatim transfers as well as a model-written note (copy loss was no better with prose than code).
   Cheapest run: two agents, 10 generations, teach modes {model note, verbatim}, 10 seeds; outcome = student
   score after adoption. Small build (a "verbatim" teach mode). Diego: orthogonal.

6. **Which reaction rule beats bad prose without losing good ideas?** Rejected text lowers the learner by
   about 2.4 points. Quarantine (hide unverified text) removes that by construction but also hides a good idea
   whose code happened to fail. Idea-trial (implement the idea in a branch, test it, show the text only if the
   branch wins) should keep those. Prediction: quarantine and idea-trial both remove the harm; idea-trial keeps
   about half of the good-idea-bad-code cases quarantine throws away, at one extra call per message. Cheapest run:
   one student, 3 messages including a good idea with a planted bug, rules {read everything, quarantine,
   evidence-gated, idea-trial}, 20 seeds, local. Small build (idea-trial). Diego: orthogonal.

7. **Recombination.** Two agents start with complementary halves of a good strategy (one has the good discard
   rule, one the good hint rule; each alone scores little). Prediction: with sharing, an artifact that beats both
   parents by a point appears within 10 generations with both in its ancestry; isolated agents never produce it.
   Cheapest run: 4 agents, 20 generations, sharing on/off, 5 seeds. Needs two partial reference bots (half a day).
   Diego: orthogonal.

8. **Model ceiling vs game ceiling.** The local model cannot improve on the 17-point reference bot (0 of 30 tries;
   two 40-generation chains stayed flat). Prediction: a stronger hosted model gets to 19+ in one revision at least
   10% of the time. Cheapest run: 30 single revisions of the reference bot on the hosted model, ~30 calls, a few
   dollars. Decides whether accumulation experiments can show anything above 17. Do this first on any new model.
   Diego: orthogonal.

9. **An idea-map from the conventions documents.** Every artifact carries a plain-language conventions doc. Fit
   a topic model with anchor words (Topic-SCORE) to all docs from a population run; each agent gets topic weights
   per generation; test whether two groups' docs diverge (DELVE). Prediction: isolated groups separate in topic
   space within 20 generations; migrating groups do not; and a "new topic appears" event coincides with a jump in
   best self-play. Zero model calls; runs on logs we have. First figure that shows ideas moving, not scores.
   Diego: builds-on (his corpus notes are the same object).

10. **Prose vs code: compatibility, not score.** Copy loss was the same for prose and code teaching. Prediction:
    prose copies cross-play worse with their teacher (different edge cases) even when self-play matches. Zero
    calls: compute cross-play on the copies we already have. Diego: orthogonal.

11. **Many small generations vs few big ones at the same budget.** Prediction: 200 generations of one revision
    beat 20 generations of ten only when teaching is on. Hosted. Diego: orthogonal (his horizon is fixed).

12. **Toggle a lever mid-run.** One 300-generation run, teaching switched every 30 generations; read the effect
    off the rate of improvement in on vs off blocks. The mechanism exists. Prediction: visible within one block.
    Hosted, ~2,600 calls. Diego: orthogonal.

## Parked

- Credit as a market (agents paid in compute for their students' gains): needs recoverable credit first (idea 1).
- Learned routing (who to listen to, as a bandit): needs many generations and a credit signal.
- A second game: not until Hanabi has three findings good enough for the deck.
- Mean-field theory of the population: Enrico ruled it out as not useful now (2026-10-06).
