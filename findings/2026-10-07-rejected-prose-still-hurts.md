# A rejected message still moves the learner through its prose

**Status:** preliminary (one model; exploratory contrast inside the credit run)
**Date:** 2026-10-07 · **Label:** none · **Data:** `docs/results/c1-full.json` (strata.verified, teachers Flawed vs self)

## Question
If a student verifies a received bot, finds it worse, and does not adopt it, is the student unaffected?

## What we expected
Not pre-registered as a test. The pilot (5 seeds) showed the rejected bad bot still had a negative effect in 4 of
5 seeds, but so did the re-sent own bot in either direction, so we could not tell text effect from prompt noise.

## Setup
Same run as the credit finding. Verification on, so the 0-point bot with persuasive prose is rejected in every
seed. Compare its effect on the student with the effect of the re-sent own bot (also never adopted, generic
text), paired by seed, 29 seeds.

## Result
Bad bot minus own bot: **−2.44 points [−3.69, −1.19], p = 0.0004.** The good bot partly undoes it: the
interaction between the good bot and the bad bot's text is +3.8 [2.2, 5.4].

## What it changes
Verification gates code adoption, not ideas. The revision prompt still shows the rejected message's prose and
evidence, and the model listens. So "verify before adopting" is not enough against persuasive-but-wrong ideas;
you also have to decide what the learner is allowed to read. That is proposed idea 6 (reaction rules) and the
second channel in idea 3 (contagion through prose that no verification can stop).

## Caveats
One model, one prompt format. The size of the effect probably depends on how the prompt presents received
messages.
