# Diego's sandbox on `main`: a living extract

For Enrico and for Claude sessions on `enrico`. Built from everything committed on `main` at the hash at the bottom:
README, all of `docs/`, `docs/results/*.json`, `configs/`, notebook markdown, and `src/millstlabs` docstrings. Read
`main` with `git show main:<path>`; never check it out, never edit `src/millstlabs`.

`main` commits only warm-start checkpoints under `runs/`, not training outputs. "Not on main" means "not committed",
not "never ran". Ask Diego before saying one of his runs failed or did not happen.

## 1. What his sandbox is

1. **Game.** A predator–prey ecology (PettingZoo, 20×20 grid, ~10% walls, 3 food stations, 1 scripted predator). 8 founder prey, cap 16. Each tick every living prey picks north/south/east/west/feed/watch/rest. WATCH means giving up the move or feed that tick to see the predator at radius 6 and auto-broadcast a truthful alarm. A **decision** is one living prey's action in one tick, and every budget is counted in decisions. The score is **restricted mean lifetime**: the mean of min(lifetime, horizon) over 8 reset bodies on held-out maps, with learning and births off.
2. **Agents.** Each prey is frozen, shared SmolLM2-135M-Instruct plus its own **LoRA** (small trainable low-rank matrices on the last 4 layers), a GRU controller and an action head. Each prey is trained on its own data by **PPO** (clipped policy-gradient RL). Reward = own survival + α·(other prey alive)/16, where α is the **social preference**. Profiles: individual 0, prosocial +0.5, competitive −0.5, and vigilance (α 0, WATCH-logit bias +1).
3. **Inheritance.** In **R-adult**, a newborn copies its parent's current weights plus a tiny mutation (KL ≤ 0.01). In **R-initial**, a newborn gets the common starting policy; this is the control for mere replacement. In **iteration** there are no births and agents relearn over repeated lifetimes. All conditions on a seed start from one **warm start**, made by behaviour cloning on demonstrations from a food-seeking heuristic.
4. **Flag Game** (Pavlova & Tanaka, arXiv:2609.19124). 8 frozen GPT-4o vision agents each see a private crop of a flag and talk pairwise to name the country. Diego adds a small per-agent RL controller that decides when to deposit or retrieve pixel-verified colour facts. Built and tested offline only; **no paid call has been made**.
5. **Corpus and discovery tools.** Agents deposit first-hand facts (walls, food stock), each checked against the author's own sensor record (fabrications are rejected), and retrieve peers' facts. Later variants attach grounded or LLM-written notes ("prose") and persistent "tentative tactics". His main evaluation device is the **frozen access ablation**: the same trained weights evaluated with peer access removed.

## 2. Experiments

Status: *specified* (doc or config only) · *validated* (mechanism smoke, too short for any learning claim) · *run*
(results committed) · *launched* (warm start committed or run referenced, results not on main).

### 2.1 Heuristic ecology calibration: run, reported
- **Question.** Does the ecology exercise feeding, predation, births and generations, with gaps between policies?
- **Setup.** Random, food-seeking and vigilant heuristics × reproduction on/off × 5 dev seeds (101–105). 2,048-tick cap, 30 episodes, no learning.
- **Result.** Without reproduction, lifetimes were random 84.1, food 148.9, vigilant 239.3. With reproduction, all 10 fertile food/vigilant episodes went extinct between tick 291 and tick 1,141. They had 6–17 births and reached generation 1–4. All 7 pilot checks passed.
- **My reading (not his).** In all 15 reproduction episodes the population never reached the cap of 16 (`cap_fraction` 0). The "not always at cap" check passes trivially, and density pressure never binds.
- **His reading.** "This establishes that the simulator can exercise the mechanisms, not that learned agents have discovered them."
- **Source.** `docs/validation.md`, `docs/overnight-cpu.md`, `docs/results/calibration.json`.

### 2.2 Integration smokes: run, reported
- **Setup.** A smoke of the three methods on the non-LLM `tiny` backend, and a SmolLM2 integration run (16 demos, 128 decisions, 64 optimizer steps, 32 eval decisions). 24 tests.
- **Mac smoke review.** Three directories share one seed and one trajectory, so they are one replicate. The run had 128 decisions over 16 ticks, one FEED, and no births or deaths. A 4-tick eval had everyone survive, which only shows the horizon is uninformative. A 32-tick notebook eval gave 27.875/32 with 3 predation deaths. Action probabilities differed by ≤ 0.0031 across the 5 behavioural probes.
- **His reading.** "An execution check, too short to assess sustained learning." "Inheritance was never exercised in these learned runs."
- **Source.** `docs/validation.md`, `docs/overnight-cpu.md`, `results/mac-smoke-review.json`, `results/smollm-integration.json`.

### 2.3 Main study from the original spec: specified, never run
- **Question.** Separate four effects: personality, demographic (replacement), inheritance (R-adult beats R-initial), and learning process (reproduction beats iteration). Also test accumulation: are later-generation newborns better before any training?
- **Setup.** 4 profiles × {iteration, R-adult, R-initial} × 5 seeds = 60 runs × 2M decisions = 120M decisions. 50,000-demo warm start per seed. Evals at 250k/500k/1M/2M decisions on 8 maps (32 at the final eval), 2,048-tick horizon. Target hardware: one 24 GB GPU.
- **Cost.** The CPU benchmark ran at 7.78 decisions/s, which means 71 h per run and ~4,300 h for the sweep on CPU. Collection alone is up to 30.7B input tokens, plus up to 917,504 eval decisions per run.
- **Contrasts.** R-adult − R-initial isolates inheritance. R-adult − iteration is the whole learning-process effect, confounded with demography.
- **Status.** "No 2-million-decision training run, 60-run sweep, CUDA benchmark, GPU container build, or Slurm submission has been performed."
- **Spec numbers he corrected.** 8 founders (not 16), social denominator 16 (not 32), food 6/tick (not 12), and 120M decisions (the spec said 60M).
- **Source.** `docs/original-spec.txt`, `docs/experiment.md`, `docs/compute.md`, `configs/main.yaml`.

### 2.4 Granular profile sweep: specified
- **Setup.** 9 profiles: α ∈ {−1, −0.5, −0.25, 0, +0.25, +0.5, +1} plus vigilance bias {0.5, 1}. Not run.
- **Source.** `configs/granular.yaml`.

### 2.5 Warm-start readiness: run, reported (negative)
- **Question.** Does a 4,096-demo warm start survive better than random before any PPO?
- **Setup.** Seed 11, 2 maps, 512 ticks, 8 bodies.
- **Result.** Lifetime **76.25 vs 83.19 for random** (food heuristic 171.0, vigilant 216.9). All 16 bodies died (5 starvation, 11 predation). 81 of 1,220 actions were FEED (6.6%). Probe responses were nearly identical across the 5 probes.
- **His reading.** "This one-seed observation does not establish a statistically reliable deficit, but the stronger warm start has not demonstrated useful survival improvement."
- **Source.** `docs/overnight-cpu.md`, `results/warmstart-readiness.json`.

### 2.6 Overnight CPU profile pilot: launched, results not on main
- **Question.** Exploratory profile × method differences on one seed.
- **Setup.** 4 profiles × 3 methods × seed 11 = 12 runs × 16,384 = 196,608 decisions. All 12 runs share one 4,096-demo warm start. Evals at 0, 8,192 and 16,384 decisions on 2 maps (4 at the final eval), 512 ticks.
- **Reference levels.** Heuristics on the same 4 maps: random 80.2, food 173.4, vigilant 213.6.
- **Cost.** 7.24 decisions/s, so 7.5 h of training; he budgets 9–14 h total.
- **Superseded variants.** `local-pilot` (prosocial × 3 methods) and `overnight-cpu` (a 3-seed sweep).
- **His interpretation order.** "did the warm start learn useful behavior; did continued training improve common-map lifetime; were there births and mature descendants; …" Also: "If births are absent, the inheritance comparison is unexercised."
- **Source.** `docs/overnight-cpu.md`, `configs/overnight-cpu.yaml`; warm start at `runs/overnight-profiles/warm-s11.pt`.

### 2.7 Predator temperature calibration: run, reported
- **Question.** Can predator temperature serve as a measured difficulty dial?
- **Setup.** In the new predator kernel, the predator moves with probability 0.9, and the next cell has probability ∝ exp(−Δdistance/T). T = null restores the original 70/20/10 kernel.
- **Result** (8 dev maps, 512 ticks, food / vigilant heuristic):

  | T | food | vigilant |
  |---|---|---|
  | 0 | 138.0 | 190.9 |
  | 0.25 | 143.4 | 190.9 |
  | 1 | 197.5 | 240.4 |
  | 4 | 296.6 | 303.6 |

  Lower T is harder.
- **His reading.** "Temperature is an uncertainty parameter; difficulty must be measured."
- **Source.** `docs/next-experiment.md`, `results/next-predator-calibration.json`.

### 2.8 Next LLM core matrix, split actor and critic: specified, validated
- **Question.** Does learning happen once the critic is isolated from the LLM actor? The new design has a separate numeric GRU critic, a Huber value loss scaled by 100, and updates every 32 ticks.
- **Why.** "The earlier audit found projection saturation and a much larger critic gradient in one inspected minibatch."
- **Setup.** 4 profiles × 3 methods × seed 11 × 65,536 decisions = 786,432 decisions, from an 8,192-demo warm start. Evals at 0, 16k, 32k and 65k decisions. Frozen final policies are assayed at predator T = 0, 1, 4.
- **Cost.** 4.75 decisions/s (4 threads) and 5.37 (1 thread), so 40–46 h of training alone.
- **Validation.** 512 decisions and 1,536 eval decisions; the LoRA weights changed. The 32-tick horizon is "deliberately too short for a survival claim."
- **Promotion gate (advisory).** Passes when at least 3 nonzero checkpoints have mean lifetime ≥ half the horizon, ≥ 25% survival, a lifetime range ≤ 5% of the horizon, and a gain ≥ 5% of the horizon over the initial policy. In his words it is "not a statistical saturation test."
- **Source.** `docs/next-experiment.md`, `configs/next-core.yaml`, `results/next-llm-benchmarks.json`, `results/next-llm-validation.json`.

### 2.9 Focused cooperative R-adult run: launched, results not on main
- **Question.** Can the split architecture learn at all with 131,072 decisions?
- **Setup.** Prosocial, R-adult, seed 11, 131,072 decisions. Checkpoints at 0, 16k, 32k, 65k, 98k and 131k. 8 held-out maps plus 4 dev maps, 512 ticks. Temperature assays at 0, 1 and 4. He estimates 9–14 h.
- **Predeclared criterion.** At the final checkpoint, ≥ 20% higher restricted mean lifetime than the initial policy on the same 8 maps. It must come with better food intake per living decision and must not be driven by a single map. "The 20% bar is a chosen practical effect size, not a statistical significance threshold."
- **His prior.** "modest learned behavior is plausible, but strong sustained survival is not yet the outcome I would bet on."
- **Limitation he names.** The run has no R-initial or individual-profile control, so it "cannot identify the benefit of inheritance or prosociality."
- **Evidence it ran.** The warm start is committed at `runs/next-llm-cooperative/`. The discovery validation notes that its timings overlapped "a separate cooperative CPU job" on 2026-10-05.
- **Source.** `docs/next-experiment.md#focused-cooperative-r-adult-run`, `configs/next-cooperative.yaml`.

### 2.10 Learned symbol channel, communication vs muted: specified
- **Question.** Does a cheap learned message channel help beyond the automatic alarms?
- **Setup.** With each move, an agent emits one of 3 symbols or stays silent. Messages have radius 6, capacity 4, cost 0.02 energy, and last one observation.
  - Arms: *communication* vs *muted* (packets dropped), with identical action spaces and a byte-identical warm start.
  - Extra eval: the trained communicating policy with messages muted at test time.
  - Individual and prosocial profiles, R-adult, 65,536 decisions each.
- **His reading.** "Neither alone proves teaching or knowledge accumulation." "Do not infer the utility of natural-language collaboration from a failure of these arbitrary symbols."
- **Source.** `docs/next-experiment.md`, `configs/next-communication.yaml`, `configs/next-muted.yaml`.

### 2.11 Shared-critic and non-LLM diagnostics: specified
- **Setup.** `--phase shared` runs the same LLM architecture with a shared actor/critic representation; different warm starts confound it. `--phase numeric` runs a non-LLM structured recurrent policy for 163,840 decisions and is "never the default".
- **Source.** `configs/next-shared.yaml`, `configs/next-numeric.yaml`.

### 2.12 Communal-knowledge discovery with explicit tools: validated, full run not on main
- **Question.** "does access to peers' verified observations help a population discover more of an unfamiliar map, without reducing survival?"
- **Setup.** Prosocial, R-adult, seed 11, 4 arms × 32,768 = 131,072 decisions.
  - Arms: `private_plain`, `shared_plain`, `private_guided`, `shared_guided`. *Guided* means a decaying count-based novelty bonus (0.05 → 0 over the first 75% of training) and predator T = 4 for the first half, then 1.
  - Actions: the joint action is 7 moves × 5 tools (none, put_terrain, put_food, get_terrain, get_food). Each tool use costs 0.05 energy and handles ≤ 4 facts.
  - Corpus: food facts expire after 16 ticks; the corpus resets on every map.
  - An optional 8-arm factorial exists.
- **Metric.** New first-hand verified facts for the whole population per 1,000 allocated decision opportunities (8×512 per map, counting opportunities lost to death). Copying a peer's fact cannot raise it.
- **Predeclared criterion.**
  1. ≥ 20% higher discovery rate than the matched private arm.
  2. Lifetime no lower, and a positive discovery difference on ≥ 3 of 4 maps.
  3. Real peer retrieval, and fewer discoveries when the same shared policy is evaluated with private access.
  4. A non-smoke budget.
- **Validation** (128 decisions per arm, 8-tick eval).
  - All four arms were identical: 49 new facts at eval, lifetime 8.0.
  - In training, the shared arms made 52 distinct peer imports; the private arms made 0.
  - LoRA moved ≤ 0.0008 from the warm start.
  - His label: "Operational validation only."
- **His expectation.** The bottleneck is "learning to change physical behavior appropriately after a retrieval, not implementing storage". Station coordinates are already public, so useful extra information may be scarce.
- **Source.** `docs/discovery.md`, `configs/discovery.yaml`, `results/discovery-validation.json`.

### 2.13 Flag Game with a verified-corpus controller: validated offline only
- **Question.** "can a learned choice of when to deposit and retrieve verified sensory facts improve terminal truth mass, relative to the same vision agents with private memory?" **Terminal truth mass** is the fraction of agents naming the true country at the end. "Agreement alone is not a success metric. A wrong unanimous answer scores zero accuracy."
- **Setup.**
  - Agents: 8 frozen `gpt-4o-2024-08-06` agents, pairwise protocol, 10 rounds, 8-entry transcript, temperature 0.2.
  - Arms: `paper_reference`, `private_frozen`, `shared_frozen`, `private_rl`, `shared_rl`, plus a `shared_rl_private_access` eval.
  - Facts: a fact says "agent i's crop contains RGB #RRGGBB" and is checked against the crop's pixels.
  - Controller reward: own terminal correctness + 0.5 × others' mean correctness − 0.01 per tool use.
- **Sizes.**

  | size | trials (train / eval) | max calls | cap |
  |---|---|---|---|
  | smoke | 1 / 2 (one round) | 336 | $2 |
  | pilot | 8 / 6 | 8,736 | $10 |
  | full | 64 / 60 | 81,984 | none set; ~46 h at 2 s/call |
- **Result.** 0 OpenAI requests made, 60 tests passing, and a 14-job mock smoke completed. "Live GPT access, cost, latency and learning remain to be measured."
- **Limitations he states.**
  - It is a documented reconstruction (`exact_paper_reproduction: false`): the 28-country pool, seeds, model snapshot and κ are unknown.
  - Colour presence cannot separate flags with the same colours.
  - Static crops mean no new discovery happens within a trial.
  - One controller-training seed.
- **Source.** `docs/flag-game.md`, `configs/flag-game.yaml`, `results/flag-game-validation.json`.

### 2.14 Corpus notes, grounded vs prose: prepared and reported, PPO run not on main
- **Question.** He frames it as a causal chain: "an agent observes something, publishes it, a peer reads it, the peer changes its action, and its outcome improves. Note counts establish only the beginning of that chain."
- **Setup.** Prosocial, R-adult, seed 11, 2 styles × 32,768 decisions, 8,192-demo warm start per style.
  - Reading: each decision sees ≤ 1 retrieved note, chosen by a heuristic relevance ranking.
  - Writing: a publication head decides whether to publish (16-tick cooldown, 0.05 energy). *Grounded* notes render the observation; *prose* notes are ≤ 48 tokens written by SmolLM2. The service attaches verified evidence either way.
  - Shaping: +0.1 reward per novel fact delivered to a living peer, −0.01 per write; feed-imitation loss 0.1; entropy 0.003.
- **Smoke result** (128 decisions per style).
  - Notes delivered 15 new peer facts.
  - Neither style passed the food-cue probe: swapping which remote station has food changed action probabilities by ≤ 5e-6.
  - Prose contained unsupported claims ("3 feet high and 3 feet wide", "32 units per day").
- **Full preparation** (warm start plus initial eval, zero PPO).
  - Grounded/prose lifetime 110.25/110.94.
  - Agents fed on 53 of 74 hungry opportunities, and none survived to tick 512.
  - In his words, "neither benefited from peer access on the two evaluation maps."
- **Source.** `docs/corpus-notes.md`, `configs/corpus-notes.yaml`, `results/corpus-notes-validation.json`.

### 2.15 Demonstration label-dependence audit: run, reported (negative)
- **Question.** Do the grounded demonstrations ever make the teacher's action depend on peer notes?
- **Setup.** Replay the heuristic teacher over its 8,192 demonstrations with peer information removed, holding observations and tie-breaking RNG fixed.
- **Result.** 7,194 of 8,192 examples contained peer context, and **0 teacher actions changed**, whether only the current note or all peer memory was removed.
- **His reading.** "simply adding peer text to an imitation dataset does not teach a decision that depends on that text." This led to the balanced reader curriculum in §4.
- **Source.** `docs/corpus-notes.md`, `results/corpus-notes-demo-audit.json`, `src/millstlabs/demo_audit.py`.

### 2.16 Direct LLM actions with persistent tactics and zero demonstrations: validated, launched (the README's "current" experiment)
- **Question.** Can SmolLM2 learn the game from its own token probabilities, with no demonstrations, no imitation loss and no action head? Do persistent text tactics help?
- **Setup.**
  - Actions: the LM's next-token probabilities over the 7 action words, restricted to legal moves.
  - Learning: private LoRA trained by PPO (adapter lr 3e-5, 16-tick rollouts, sequence length 4), a separate numeric critic, mutation off.
  - Publishing: a yes/no token trained by PPO, with a small bonus for delivering a novel outcome pattern to another lineage. When it says yes, the frozen base model writes a tactic of ≤ 48 tokens (32-tick cooldown).
  - Memory: tactics persist across ecological resets with lineage ownership (256 kept). Reads and writes cost zero energy.
  - Size and eval: 2 styles × 8,192 decisions, checkpoints at 0, 2,048 and 8,192, a 256-tick assay, and 3 access conditions (shared / lineage-private / none).
- **Progress measures.** `change_from_pretrained_start`; `tactic_access_benefit` (shared vs none, frozen weights); `peer_access_benefit` (shared vs lineage-private, frozen weights).
- **Validation** (128 decisions per style).
  - 0 food consumed. Actions were almost all north/south (grounded: 51 N, 66 S of 128).
  - On the food-direction probe, the probability of the correct direction went from 0.008 to 0.006.
  - 8 notes were published, and retrieved notes reached 3 of 19 sampled prompts.
  - Everyone survived the 8-tick assay, which he calls "not a meaningful survival success". 83 tests pass.
- **Source.** `docs/autonomous-llm.md`, `configs/autonomous-llm.yaml`, `results/autonomous-llm-validation.json`; warm starts (zero demos) at `runs/autonomous-llm/{grounded,prose}/`.

### 2.17 What main does and does not support in the one-pager
The one-pager's Sandbox 1 bullets are "accelerating SmolLM2 discovery fivefold in predator-prey environment over
equivalent model calls" and "saturating multiple skillful games with ≤ 1/10th the compute of baseline models". On
`main` at this hash:
- **No result measures discovery acceleration.**
  - The only discovery measurement is the 128-decision validation (2.12), where all four arms found the same 49 facts.
  - That experiment's own success bar is +20%, not ×5, and its full run is not on main.
  - Nothing on main compares arms at equal SmolLM2 model calls. Budgets are in decisions; encoder forwards are logged but never matched.
- **Every learning number on main is flat or negative.**
  - Warm start below random: 76.25 vs 83.19.
  - Corpus notes: no benefit from peer access.
  - Demo audit: 0 of 7,194 teacher actions depend on notes.
  - The zero-demo agent eats nothing and fails the direction probe.
  - His README: "they do not establish that reproduction, inherited learning, or prosociality improves learned performance."
- **No game is saturated.**
  - Nothing has passed his own promotion gate, and he calls that gate "not proof of saturation".
  - The Flag Game has made zero GPT-4o calls, so there is no compute comparison against a baseline model.
- **"fine-tuned personalities in SmolLM2 and GPT-4o" is not what main shows.**
  - GPT-4o is frozen: "Its weights are not fine-tuned."
  - The α personalities exist only for SmolLM2.
  - The word "gestation" does not appear on main.
- **Caveat.** The launched runs (2.6, 2.9, 2.16, and the PPO phase of 2.14) may have results on Diego's machine. Ask him which run each bullet comes from before anyone repeats it.

## 3. Heuristics and engineering practices

**Already emulated on this branch:**

| His practice | Our file |
|---|---|
| Cost guard | `src/culture/llm/spend.py` (modelled on his `flags/provider.py`) |
| Shared warm starts | `src/culture/run/warmstart.py` |
| Record/replay cache | `src/culture/llm/cache.py` |
| Config freeze, deployment lock, atomic checkpoints with log byte offsets, wall-clock budget | `src/culture/run/runner.py` |

**Cost guard** (`main:src/millstlabs/flags/provider.py`):
- Before sending, reserve a conservative maximum for each request: all input priced at the uncached rate, plus 1,024 spare tokens, overestimated image tiles and the full completion cap.
- The cap is cumulative and kept in sqlite, so it survives resumes. Each reservation is reconciled from the usage the API returns.
- An ambiguous failure keeps its reservation and blocks retry unless the operator passes `--retry-uncertain`. In his words: "No automatic paid retry."
- A 400/401/403/404/429 releases its reservation, since no completion was generated, and stops the run with a message. Any other non-200 keeps its reservation. Nothing is retried automatically, and the key is never logged.
- The pricing adapter refuses any model it is not priced for. The mock backend reports itself as `mock-not-an-llm`.

**Run modes.**
- `plan`: prints the matrix, decision budgets and call upper bounds before any compute.
- `smoke`: the real model at a tiny budget, in a separate directory.
- `prepare`: warm starts and initial evals with zero RL, run before he commits to an overnight.
- `run`, and `report` (read-only).
- `benchmark`, `calibrate` (dev maps only) and `gate` (advisory).

**Resumable soft hour budget.** `--hours N`; rerunning the same command resumes. The limit is checked between ticks.
"A paused cohort is not a negative result."

**Frozen configs.** Each output directory is bound to a config digest. A changed config needs a new directory,
incompatible resumes are rejected, and a lock stops concurrent writers.

**Pairing.** One byte-identical warm start per seed is shared by every condition. Conditions use the same eval maps,
body resets and action-RNG seeds, with separate RNGs for collection and actions.

**Unit of replication.** "Evaluation episodes from one trained run are not independent training replicates." He
aggregates within a run first, then computes uncertainty across training seeds.

**Budgets.** Matched in decisions, logging realized vs requested. Eval decisions are counted separately. Tokens and
wall time are reported but not equalized.

**Dev/test separation.** Calibration uses dev seeds only. "Do not tune ecology or hyperparameters separately for
whichever method looks weaker on these held-out maps." "reserve a fresh final map bank for confirmation."

**Controls that isolate one mechanism.** R-initial; the frozen access ablation; test-time muting; a frozen, untrained
tool controller as the mechanism control; shuffled reports (proposed).

**Thresholds.** Predeclared practical thresholds (20%), labelled as not significance tests. Every paired map difference
is reported, and no aggregate may depend on a single map.

**Fix the bottleneck before buying compute.** "If feeding is poor or the reader cannot use the controlled food cue, fix
that bottleneck before buying more training time." "Increasing the cooperation weight or adding conversation first would
make the credit problem more complicated without fixing basic navigation."

**Audit information use cheaply.** The demo audit and the cue probes show that information being present in the input
is not information being used. "Sensitivity alone does not demonstrate useful behavior."

**Observation privacy.** He never feeds `infos` or `render()` to a policy. Tool masks never leak unread peer deposits,
and all reads happen before all writes.

**Measure the difficulty dial; do not assume it.** See 2.7.

**Claims he refuses**, in his words:
- "they do not establish that reproduction, inherited learning, or prosociality improves learned performance" (README)
- "This is new-information delivery credit, not causal credit for improving the recipient. It can reward information the recipient ignores."
- "The evidence service verifies the attached measurements, not the author's interpretation."
- "A final corpus benefit can come from demonstrated tool competence; it is not automatically evidence that RL learned to collaborate."
- "Operational success means the runner resumes reproducibly and the intended weights receive gradients. Scientific success requires held-out improvement over warm start and heuristics … Passing the former does not establish the latter."
- "Social reward currently pays for others remaining alive, not for the causal benefit of a particular teaching or helping action."
- "A failed screen could reflect poor language use, weak information opportunities or insufficient optimization. It does not refute the broader swarm thesis."
- "This test changes a bundle of engineering choices; improvement over the old pilot would not identify which change caused it."

## 4. His ideas and open questions (as stated in his docs)

**Proposed next experiments:**
- **Balanced reader curriculum.** Pair identical local observations with different valid remote food reports that require different moves. Vary layout, direction, obstacles and phrasing, and test on unseen layouts and on withheld or mismatched notes. (`corpus-notes.md`)
- **Demonstration competence.** Compare 8,192 vs 32,768 demos. If RL erases that competence, add an arm with a decaying imitation/KL anchor. (`next-experiment.md`)
- **Credit assignment.** GAE λ 0.99 vs 0.95, then separately a 64-tick rollout arm; never change both at once.
- **Easier predator first.** Train at predator T = 4 and evaluate at T = 4 and T = 1. A staged 4 → 1 curriculum needs an explicit weight-transfer protocol.
- **Potential-based shaping.** An arm adding β(γΦ(s′) − Φ(s)) on a bounded energy/navigation potential.
- **Adaptation stability.** A lower LoRA learning rate (1e-4 → 3e-5), or train the action head first with the adapter fixed. Choose between them using the KL diagnostics.
- **Centralized critic.** A training-only critic with permutation-invariant population features, if social value prediction stays poor.
- **Adaptive curriculum.** An adaptive training curriculum with a stage-transfer protocol, only after the promotion gate passes.
- **Language-swarm test.** Short grounded reports (food, threat, intended destination) with timestamps, sender identity, bandwidth cost and receiver verification. Arms: delivered vs muted vs shuffled. Use "a capable instruction-tuned LLM".
- **Navigation-scored blockages.** If imports happen without behavioural benefit, make an observed route blockage consequential and score navigation directly.
- **Zero-demo agent next steps.** Higher sampling entropy, shorter prompts, a larger local model or easier geometry, keeping the held-out predator temperatures fixed.

**Flag Game follow-ups:**
- Import the authors' catalog and trial manifest.
- Add the broadcast and manager protocols.
- Add richer verified facts (spatial relations, hypotheses).
- Replicate with new training seeds and a fresh eval bank.
- The paper's α (prompted social-evidence uptake) is not varied yet.

**Larger designs and compute:**
- The eight-arm discovery factorial (novelty-only vs curriculum-only).
- The granular profile sweep.
- The 60-run GPU main study.
- Compute work: mixed-adapter inference batching and a frozen-prefix cache.

**Named as not implemented:**
- Abstract strategy transfer across maps.
- Learned semantic retrieval.
- Token-level RL on prose.
- Learned deception (alarms are truthful by design).
- Teacher credit, "verification of a learned claim, group migration or cumulative language-mediated knowledge", which "remain untested".

**Open limitations he flags:**
- End-of-budget censoring of dead agents' unfinished social windows.
- Short lives fragment each agent's training data.
- Delayed survival credit: a TD residual 100 steps away gets weight ≈ 0.005 at γ = 0.999, λ = 0.95.
- A null result cannot separate a weak information bottleneck from poor retrieval use or too little training.

## 5. Overlap map with `enrico`

This branch has program-writing agents on Hanabi, exact counterfactual credit on a seeded model, verification as
sequential testing, quarantine of unverified text, the breakdown point under sabotage, organizations (groups, migration,
routing, budget), and accumulation vs isolation. Diego himself lists most of this as untested on `main`: "Neither
mechanism directly implements teacher credit, verification of a learned claim, group migration or cumulative
language-mediated knowledge."

| Diego (`main`) | Relation | Why |
|---|---|---|
| Predator–prey ecology as the testbed | orthogonal | Different game and score (lifetime vs Hanabi points). No partner-compatibility notion, and only heuristics as reference policies. |
| Learning in weights (private LoRA + PPO) | orthogonal | We learn in artifacts (bot code plus a conventions doc) through LLM revision; there are no gradients on this branch. |
| Social preference α, profiles, granular sweep | orthogonal | His incentives live in the reward. We have none at the reward level; our nearest levers are budget and routing. |
| R-adult vs R-initial vs iteration | **overlaps** | Same question as our accumulation vs isolation: does transmitted learning beat restarting? His R-initial plays the role of our no-teaching control. His transmission is vertical weight copying; ours is horizontal artifact teaching across generations. |
| Mutation with KL ≤ 0.01, extinction recovery, predator kernel | orthogonal | Ecology mechanics with no analogue here. |
| Predator temperature dial and promotion gate | orthogonal (same role) | Both ask whether there is headroom. We answer it with the model-ceiling finding: the local model improved the 17-point bot in 0 of 30 tries. |
| Learned symbol channel, communication vs muted, test-time muting | overlaps (design) | Same control logic as our teaching on/off and quarantine; the channel is 3 symbols vs programs and prose. |
| Discovery tools: evidence-checked deposits, shared vs private access | **overlaps** | Same thesis bullet (a verified communal corpus). His check is deterministic provenance, so a lie is impossible. Ours is a statistical test of a performance claim (SPRT), where a persuasive wrong artifact can pass. |
| Frozen access ablation (same weights, access removed) | builds-on | The corpus-level version of our counterfactual replay; we do it per message with exact replay. |
| Notes, grounded vs prose ("verifies the attached measurements, not the author's interpretation") | **overlaps (strongest)** | Same object as our code-vs-prose copy fidelity and "rejected prose still hurts" (−2.44 points). Quarantine closes the channel he leaves open. |
| Delivery credit (+0.1 per novel fact delivered) | builds-on | He says it is "not causal credit for improving the recipient". Our exact counterfactual credit is the causal version. |
| Demonstration label-dependence audit | overlaps (method) | A remove-the-message counterfactual like our replay, with the same lesson: information present is not information used. |
| Zero-demo LLM actions + persistent tactics (shared / lineage-private / none) | **overlaps** | His tactics are persistent text artifacts like our conventions docs, and lineage-private vs shared is our isolation vs sharing. `tactic_access_benefit` is an access ablation at frozen weights. |
| Novelty bonus, fixed predator curriculum | orthogonal | Exploration shaping; nothing similar here. |
| Split critic, shared critic, numeric diagnostic | orthogonal | RL architecture choices. |
| Behavioural and cue probes | orthogonal (kindred method) | He replays synthetic inputs through each policy; our analogues are cross-play and probe gates on fixed seeds. |
| Flag Game: frozen GPT-4o, pixel-verified facts, RL controller | orthogonal game; overlaps on verification and wrong consensus | A wrong unanimous answer is the outcome our breakdown and contagion work studies. His facts cannot be false, only uninformative. His cost guard is already emulated here. |
| One-pager "fivefold discovery over equivalent model calls" | overlaps | `IDEAS.md` #4 (organization vs more agents at a fixed number of calls) is the same axis. Run it so the two sandboxes agree. |

**What Enrico's setup can do that Diego's cannot:**
- **Exact counterfactual replay.** A seeded, temperature-0 local model plus the cache can rerun a generation with one message removed while everything else stays identical, so per-message credit has a planted truth. His PPO samples actions and updates weights, so a counterfactual means retraining, and `env.fork()` diverges once paths differ.
- **Cross-play.** We score two agents' strategies together, which measures how compatible conventions are between agents and between groups. His evaluator samples 8 policies from one population and has no notion of a partner.
- **A known ceiling.** Strong reference bots (Piers 16.99, IGGI 15.86) make headroom and saturation measurable. No learned agent on main has yet beaten his random baseline.
- **Inspectable learning.** Artifacts are programs with provenance, the teaching DAG is explicit, and a diff shows what changed. His knowledge lives in 93k–216k LoRA and controller parameters per agent.
- **Adversarial content.** We can plant sabotage or persuasive-wrong artifacts and measure the breakdown point. His validator makes false facts impossible; only prose interpretation can be wrong.
- **Organization levers.** Groups, migration, routing and budget allocation. His population is one group with one profile.
- **Cheap generations.** We spend one LLM revision per generation. He runs ~5–7 decisions/s on CPU, so a 131k-decision run takes a night.

**What Diego's setup can do that Enrico's cannot:**
- Learning in weights by RL, inherited vertically with controlled mutation.
- Endogenous demography: births, deaths, extinction, lineages and generation depth, under resource competition.
- Incentives in the objective (prosocial or competitive α) and their measured trade-offs.
- Agents that learn *when* to share and when to read (deposit and retrieve are RL actions); our sharing is fixed by protocol.
- Deterministic fact verification from sensor provenance.
- Frontier-model agents in a published protocol (Flag Game), with belief-dynamics observables: consensus, polarization, wrong consensus.
- Embodied real-time control under partial observability, with a PettingZoo and HTTP API for external controllers.

## 6. How to keep this file current

```bash
git rev-parse main                      # compare with main_hash below and diego.main_hash_summarized in experiments/registry.yaml
git log --oneline <main_hash>..main     # what landed since this extract
git diff --stat <main_hash>..main       # which docs / configs / results / src files changed
git show main:<path>                    # read a changed file (never check out main, never edit src/millstlabs)
```

A session that finds new commits on `main` (the CLAUDE.md start-up check) spawns a subagent to do this before
proposing ideas:
1. **Read the delta.** Read every changed doc, config and `docs/results/*.json`, plus docstrings of changed `src/millstlabs` modules. A binary-only commit (e.g. `runs/**/warm-s11.pt`) still says which runs were launched.
2. **Re-extract only the delta into the relevant section.**
   - A new experiment gets a new §2 block with its status.
   - A status change, such as a launched run's results being committed, updates that block and §2.17.
   - A new practice goes in §3, a new idea or open question in §4, and a new mechanism in a §5 row. Then recheck the `Diego:` tag on each idea in `IDEAS.md`.
3. **Update the hash.** Replace the `main_hash:` line and date below with the new `git rev-parse main`, and set `diego.main_hash_summarized` in `experiments/registry.yaml` to the same hash.
4. **Do not rewrite unchanged sections.** Keep numbers sourced to a file on `main`, and mark your own readings as yours.

main_hash: d0fe393ba925197b9057308d15e8f252899df5a3
summarized: 2026-10-08
