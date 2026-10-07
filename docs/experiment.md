# Experimental decisions and boundaries

## Numerical inconsistencies resolved

| Item | Implemented value | Rationale |
| --- | --- | --- |
| Founders / recovery cohort | 8 | Confirmed by Diego; applies to every condition |
| Living population cap | 16 | Confirmed by Diego |
| Social denominator | 16, fixed through deaths/births | Explicit user correction; never normalize by current survivors |
| Standardized evaluation cohort | 8 | Score is mean over the actual eight evaluated bodies |
| Food production | Mean 6/tick | Three independent station rates in {1,2,3} |
| Main training budget | 80M decisions | Four profiles × two methods × five seeds × 2M |
| Including R-initial | 120M decisions | Sixty runs × 2M; excludes warm starts/evaluation |

No ecological parameters were tuned after seeing held-out learned-policy evaluation. The pilot uses development seeds 101–105; test evaluation starts at seed 1,000,000. The backbone is pinned to a downloaded Hugging Face commit and its resolved revision is logged.

## Simulator conventions

Ticks resolve movement, WATCH/alarms, feeding, energy/starvation, predator movement/attack, completed gestations/births and new conceptions, and station replenishment in that order. Births act next tick. Every founder is mature, fertile in reproduction methods, and has no initial birth cooldown.

Reproduction uses `environment.reproduction_mode: gestation` by default, with `gestation_ticks: 16`. Set `reproduction_mode: immediate` to run the original birth process. Both reproduction methods (`r_adult` and `r_initial`) use the selected mode; iteration disables reproduction. The frozen overnight CPU and cooperative pilot presets explicitly retain immediate mode and their original configuration digests.

In gestation mode, eligible mature prey pay 40 energy at conception, then continue ordinary actions and learning. A pregnancy conceived at simulator tick t completes at t + 16. Completed pregnancies are resolved before new conceptions. Birth requires capacity and a free adjacent cell, but does not recheck energy or charge again. Blocked births remain pending; cooldown starts at successful birth. Parent death records a pregnancy loss without refund. Pending pregnancies, private histories and prepared notes are part of simulator checkpoints and forks.

At completion, the trainer writes one private survival note using the frozen backbone with action adapters disabled, fixed greedy decoding and a 64-token limit. The writer receives the latest 128 ticks of the mother's private observations/outcomes and her inherited note; each tick's text is bounded to fit the model context, prioritizing actions, outcomes, food and threats. It runs once even if birth remains blocked. The child receives this persistent observation separately from reset recurrent memory. Text policies reserve up to 64 additional input tokens for advice. Tiny/structured development backends use a bounded observation extract instead of neural generation; their numeric controllers expose pregnancy status but do not interpret prose. Direct simulator clients may inject a `note_writer` callback; without one, notes are empty. Neural note tokens and time are recorded separately. Notes persist through recovery, standardized evaluation and controlled probes.

Immediate mode retains the old ten-value self observation and birth-time energy charge, and writes no survival notes. Gestation adds pregnancy status and ticks remaining to the self observation. Create a warm start for the selected observation layout; weights from the two layouts are not interchangeable.

Movement conflicts select random winners per destination. Chains into occupied stationary cells fail; chains into vacated cells and simultaneous swaps/cycles succeed. Failed attempted moves still incur movement cost. Feeding uses equal-share water filling with per-agent intake/energy caps; unused allocations are redistributed. Multiple reachable stations choose the most stocked with random ties.

Local vision is a 5×5 square with Bresenham line-of-sight blocking. Radius-six detection, broadcasts, and predator target distance use Manhattan distance. Alarms bypass walls within broadcast distance, as a communication signal. Direct sight supersedes an older alarm. The predator may share a prey's cell and can attack at distance zero or one; it makes at most one attack per cooldown. Movement toward a target follows a shortest path around obstacles. These are explicit conventions where the prose did not specify geometry.

Station positions are public; stock observations are cached with timestamps. Replenishment rates are private. Visible companions expose position, energy, and whether they watched on the preceding tick, enabling the low-energy-peer and existing-watcher probes. Text encoding retains the complete local grid, visible threats, station memories, peer count, watcher count, and weakest visible peer. It summarizes remaining peer energy data to fit 256 tokens. The tokenizer enforces the cap and logs observations reaching it.

All maps have connected traversable space, dispersed stations, founders within Manhattan distance five of some station, and at least distance eight from the predator. Unique individual identities are bounded by a configurable allocation guard (`max_individuals=100000`), which raises an error rather than silently replacing agents. There is no maximum biological age or forced ecological time limit.

## Learning and social credit

Only frozen pretrained parameters are shared. Each agent owns rank-four Q/V LoRA in the final four transformer blocks, its projection, 64-unit GRU, action head, trainable WATCH bias, two value heads, optimizer, memory, and buffer. PEFT activates one agent at a time; gradient isolation is tested with an actual local Llama/PEFT model. Inference and training are sequential across agents. Dropout is disabled to keep collected and replayed probabilities comparable.

Warm start behavior-clones exactly 50,000 observation-limited heuristic transitions per seed. One resulting checkpoint initializes every condition paired by seed. It is not a global teacher that continues updating. Profile WATCH bias is applied after warm start; social preference is an objective coefficient, not a mutable policy parameter.

Living agents update every 128 world ticks in each ecological trial. PPO reuses each eligible transition twice, in contiguous recurrent sequences of up to 32. Replay starts each sequence from its recorded collection hidden state, retains action masks/log probabilities, and backpropagates through the sequence. This is truncated recurrent PPO; it does not replay an agent's entire lifetime after every weight update.

Personal GAE terminates at death and bootstraps across live rollout boundaries. Social accounting uses fixed, non-overlapping 256-tick windows aligned to the ecological trial. Public tick gives policies information about window phase. Social values bootstrap at the 128-tick boundary and have zero continuation at the 256-tick boundary.

When an agent dies, its unfinished live transitions wait until its social window closes (or extinction). Subsequent population survival contributes a discounted realized tail, including newborns, without generating actions for the dead. That tail enters the final living action's social target. Its private policy is updated and only then archived. This avoids erasing social consequences at death and never mixes credit across recovered trials.

Birth copies the latest committed parent parameters before any update scheduled at the end of that tick. R-initial uses the profile's shared starting checkpoint instead. Only the action weight tensor and WATCH bias mutate. KL is parent‖child, averaged over fixed calibration histories, reduced by halving perturbations until ≤0.01. Other learned components copy exactly; memory/optimizer/buffers reset. Inherited weights are independent tensors, with no retroactive parent updates.

Extinction first finishes pending terminal updates, then restores the latest eight *distinct deaths in chronological death order*, one copy each, into a new seeded map. Generation and lineage labels persist, but age, energy, memory, optimizer, and personal histories reset. The event is an external restart, never a birth.

## Budget boundaries and practical limits

A tick is indivisible: collection stops before exceeding the decision budget, leaving fewer than `population_cap` decisions unused. Checkpoint thresholds are evaluated at the first completed tick crossing them, and both requested and actual decision counts are logged. This bounded difference must be included in matched-budget reporting.

At final budget, living residual rollouts are trained with value bootstraps; surviving bodies are not declared dead. Incomplete post-death windows are preserved in the final checkpoint and counted as `pending_dead_social_updates`, rather than fabricated or assigned outcomes from a new trial. Those incomplete dead buffers are not trained without their future outcomes. This small end-of-budget censoring is an explicit limitation of this first implementation; inspect its magnitude before inference from a full sweep.

## Evaluation and interpretation

Each checkpoint samples eight living policies (with replacement if necessary), resets bodies/memory, freezes learning, and disables births. Scores average restricted lifetimes across eight bodies per episode, then across maps. Intermediate/final evaluation uses 8/32 unseen maps and a 2,048-tick horizon. Sampling and environmental seeds are shared across conditions; action RNGs are separate from collection RNGs. Censoring at horizon is distinct from death.

Five controlled observation probes replay common histories independently through each policy's own network, reporting all seven action probabilities. They are synthetic sensory interventions, not claims about naturally occurring trajectories. `env.fork()` permits simulator counterfactuals with initially coupled RNG states; diverging branches can consume different numbers of random draws.

Every sixteenth newborn additionally receives a frozen cohort assay on two distinct held-out maps, eight copies, horizon 128, before its first RL update. This bounds costs and provides a comparable early-life quality measure by generation. `newborn_evaluation_every=1` tests every child. The native pre-update survival log has variable exposure time until the next rollout; do not use it alone as proof of accumulation.

Analysis aggregates within trained runs before computing uncertainty across training seeds. Paired R-adult minus R-initial addresses inheritance; R-adult minus iteration addresses the total learning-process effect, including demographic differences. Five seeds give limited precision. The code does not assume any profile or method wins, and does not turn heuristic pilot or tiny-backend results into such a claim.
