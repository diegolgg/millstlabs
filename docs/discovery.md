# LLM corpus deployment: explicit deposit, retrieval, and measured discovery

The experiment asks a narrow question: **does access to peers' verified observations help a population discover more of an unfamiliar map, without reducing survival?** It compares trained private-memory populations with trained shared-corpus populations. The default actor remains the pinned local SmolLM2-135M LLM with private LoRA and a recurrent action head. No hosted model API or GPU is required.

## Run it

From the repository root with the existing `.venv`:

```bash
bash scripts/discovery.sh --mode plan
bash scripts/discovery.sh --mode smoke
bash scripts/discovery.sh --mode run --hours 12
bash scripts/discovery.sh --mode report
```

On macOS, `caffeinate -i bash scripts/discovery.sh --mode run --hours 12` keeps the computer awake. The smoke command runs the **real LLM**, with 128 decisions per arm, in `runs/discovery-smoke`. Training uses a separate `runs/discovery-proof` directory. An optional `--smoke-backend structured` is explicitly a non-LLM plumbing check; give it another `--output` directory.

Repeat the identical run command to resume. The runner freezes configurations, checks a deployment lock, reuses the common warm start, skips completed conditions, and runs conditions serially. The 12-hour invocation budget is soft: an ongoing update or evaluation completes before pausing. Recovery restores corpus contents, evidence, novelty counts, policies, optimizer/recurrent states, RNGs and pending social credit. Do not change a frozen configuration; use a new output directory for another design.

The [notebook](../notebooks/discovery.ipynb) displays the plan and results; training is disabled by default. Start Jupyter with `bash scripts/notebook.sh` and open it. The earlier cooperative run uses its own directory and configuration; this deployment can wait until that job finishes to avoid CPU contention.

## Conditions and compute

All conditions use **prosocial (+0.5), R-adult, seed 11**, eight founders, cap 16 and social denominator 16. Every policy starts from the same byte-identical 8,192-demonstration checkpoint. Demonstrations cover movement and explicit tool use, so this does not test invention of the communication protocol.

| Condition | Retrieval access | Training guidance | Decisions |
| --- | --- | --- | ---: |
| `private_plain` | Own deposits | None | 32,768 |
| `shared_plain` | Population's deposits | None | 32,768 |
| `private_guided` | Own deposits | Decaying novelty bonus and predator schedule | 32,768 |
| `shared_guided` | Population's deposits | Same bonus and schedule | 32,768 |

The total is **131,072 population-wide training decisions**, plus warm start and evaluation. This spreads the previous focused experiment's decision budget across four conditions; it does not give each condition 131,072 decisions. Private means independent trainable policies and private deposited memories **in the same world**. Both controls retain the game's existing WATCH alarms and interactions.

The plan reports up to **327,680 frozen evaluation decisions** across all four conditions, excluding small behavioral probes. Actual evaluations stop when the population dies. Four full conditions can take more than one overnight invocation, especially if survival improves. Earlier, shorter LLM inputs measured roughly 4.7–5.4 training decisions/second; corpus inputs add work, and those figures are not a benchmark of this deployment. Budget roughly a day, potentially longer, and use the resumable hour limit rather than a completion promise. Full corpora, logs and model checkpoints also consume disk space.

For a target-machine measurement with the actual corpus input/action layout:

```bash
HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m millstlabs.cli --config configs/discovery.yaml \
  benchmark --ticks 64 --output runs/discovery-benchmark.json
```

This timing includes PPO but excludes evaluation, demonstrations, probes and checkpoint I/O. Benchmark separately from other training for an interpretable estimate. `--full --output runs/discovery-factorial` expands to eight arms, separating novelty-only and curriculum-only guidance; it doubles training and evaluation budgets.

## What agents can deposit and retrieve

Each decision selects a physical action and one tool: `none`, `put_terrain`, `put_food`, `get_terrain`, or `get_food`. The joint action is `physical + 7 * tool`, giving 35 choices. The LLM action controller chooses whether to use a tool and which fact type to request. A deterministic adapter selects the bounded payload; the model does not generate arbitrary fact text or choose query coordinates in this version.

* **Deposit:** write up to four first-hand observed facts. Terrain records say wall/passable at a coordinate. Food records contain the observed stock and observation tick. A validator checks the author's own registered sensory evidence; fabricated values and another agent's claimed observations are rejected.
* **Retrieve:** read up to four deposited facts of the selected type. Both modes use the same spatial ranking, based on current position and the nearest already-known food station. Terrain retrieval prioritizes walls. Facts currently visible to the reader are excluded. Food observations expire after 16 ticks; their timestamps remain visible, so an old measurement is not presented as current truth.
* **Costs and ordering:** every non-`none` operation costs 0.05 energy, even an empty read. All reads precede all writes in a simultaneous turn. Results appear in the next observation and expire from the explicit context after that observation; the recurrent policy can remember them.
* **Persistence:** deposits survive their author's death on that map. A new map starts an empty corpus, including during extinction recovery and every held-out episode. Static facts never cross map boundaries.

The system records observations as evidence for validation and scoring. **Recording an observation does not publish it, and a deposit does not automatically deliver it.** Another agent must explicitly retrieve it. Evidence and the evaluator's union of observations are never exposed as a searchable policy corpus. Tool availability depends only on the caller's observations and submissions, so masks do not leak unread peer deposits.

The HTTP API exposes the same operations. Create a session with `corpus_mode: "shared"` or `"private"`. A step accepts physical `actions` and optional named `knowledge_actions`:

```python
requests.post(f"{url}/environments/{session_id}/step", json={
    "expected_tick": tick,
    "actions": {agent: 6 for agent in living_agents},
    "knowledge_actions": {living_agents[0]: "put_terrain"},
})
# On a later step, a receiver can select "get_terrain".
```

Physical action 6 means REST. Supply every living agent's physical action; unspecified tools mean `none`. The endpoint also accepts encoded joint actions. `GET /schema` documents the names, and `GET /environments/{id}` returns the resulting agent observations. Route only each agent's own observation to its policy; the operator API also exposes privileged diagnostics. `KnowledgeCorpus.deposit()` and `.retrieve()` provide the corresponding Python interface, with evidence checking.

## How learning is guided

The frozen LLM backbone supplies actor features; private LoRA and the actor GRU/action heads learn through recurrent PPO. An independent critic GRU predicts personal, social and intrinsic returns from the same permitted observations. Its gradients do not update the actor or LLM adapters. Each agent has its own experience and optimizer. Corpus contents are not shared gradients or shared hidden states.

The guided actor advantage is `A_personal + 0.5*A_social + A_intrinsic`. Intrinsic rewards are already weighted at collection time. The novelty reward is `beta / sqrt(1 + prior_visits)` for coarse location, hunger and visible-threat categories. Counts are keyed by lineage and persist across births and recovery; descendants share that lineage's trainer-only counts. Clocks, body IDs and tiny energy changes cannot manufacture novelty. The coefficient decreases linearly from 0.05 to zero over the first 75% of training. A separate value head estimates this reward, avoiding a changing mixture in the personal survival target. All four arms have that head so their architectures and warm checkpoint match.

Guided runs use predator temperature 4 for the first half and 1 for the second half. Both corpus conditions face the same **predeclared schedule**. This is not a saturation-triggered adaptive curriculum: that would change exposure differently between conditions. Prior heuristic calibration found temperature 4 easier than 1; the order need not hold for every learned policy. Actor sampling temperature stays 1. Held-out evaluations always use predator temperature 1 and no intrinsic rewards.

The exploration signal supplies earlier feedback, consistent with [count-based exploration](https://arxiv.org/abs/1606.01868); it does not guarantee useful discovery or preserve the original optimal policy. There is no reward for deposit count, retrieval count or copying notes. The four-arm design identifies the effect of corpus access within each guidance setting; only the optional eight-arm design separates the two guidance components.

## Evaluation and interpretation

Checkpoints 0, 16,384 and 32,768 use four fixed held-out maps, eight reset bodies and a 512-tick horizon, without reproduction or restart. Two separate development maps are also recorded. Final shared policies receive a matched **private-access evaluation with the same frozen weights**, isolating the effect of peer access at test time. Every evaluation starts with an empty corpus.

The primary metric is **new first-hand verified facts across the whole population per 1,000 allocated decision opportunities**, excluding observations available at the initial state. A fact identity is `(type, coordinate)`. Re-observing or refreshing food stock does not create a new identity. Both private and shared populations use the same evaluator-side union; a larger shared library or copying a peer's fact cannot by itself improve the discovery score.

There are 8 × 512 allocated opportunities per map, including opportunities lost after death. This prevents early extinction from inflating the primary rate. Per-actual-decision discovery is secondary. Discovery curves, time to 100 new facts (null if never reached), and time-averaged cumulative discoveries distinguish early discovery from eventual map coverage when final coverage saturates.

Also report food per living decision, total consumption, restricted mean lifetime, end survival, starvation/predation, tool calls, accepted/rejected deposits, distinct peer imports and later first-hand terrain reconfirmations. Processed tokens and elapsed evaluation time expose computational cost; these are not held exactly equal when retrieved context lengths differ.

The exploratory corpus criterion requires all of:

1. At least 20% higher primary discovery rate than the matched private condition.
2. Mean lifetime no lower and positive discovery differences on at least three of four maps.
3. Actual peer retrieval, and fewer discoveries when the same final shared policy is evaluated with private access.
4. A non-smoke training/evaluation budget.

These are practical, predeclared thresholds, not a significance test. Reports include every paired map difference, each arm's change from warm start, and the change in the shared-versus-private advantage since warm start. **A final corpus benefit can come from demonstrated tool competence; it is not automatically evidence that RL learned to collaborate.** One training seed does not establish replication, inheritance benefits, or the general swarm thesis.

My expectation is cautious. The strongest bottleneck is learning to change physical behavior appropriately after a retrieval, not implementing storage. Short lives, delayed cooperative credit and a very small LLM remain limiting. Food-station coordinates are already known in this game, and local perception already reveals nearby walls; useful extra information may be scarce. A null result could reflect an insufficient information bottleneck, poor retrieval use, or inadequate training. It would not refute language-based cooperation. If imports occur without behavioral benefit, the next controlled test should make an observed route blockage consequential and score navigation directly, before paying for a much larger open-ended swarm.

## Saved outputs

`runs/discovery-proof/discovery-report.json` contains completion status, paired comparisons and the exploratory checks. Under each condition's `prosocial-r_adult-s11/`:

| File | Contents |
| --- | --- |
| `knowledge.jsonl` | Cumulative discovery, deposit and retrieval counters within each map/trial |
| `corpus_events.jsonl` | Explicit writes, rejected writes, reads, record provenance and timestamps |
| `corpus-checkpoint-*.json` | Inspectable deposited stores for the current training map |
| `evaluation.jsonl` | Fixed-map discovery curves, performance, tool use and compute |
| `evaluation_private_corpus.jsonl` | Frozen shared-policy access ablation |
| `development_evaluation.jsonl` | Separate development-map measurements |
| `curriculum.jsonl`, `updates.jsonl` | Actual temperature changes, PPO diagnostics and weighted intrinsic rewards |
| `latest.pt`, `checkpoint-*.pt` | Complete resumable experiment state |

See [the validation record](results/discovery-validation.json). Its very short LLM run verifies real deposits, peer retrieval, private isolation, updates and checkpointing. It is intentionally too short to demonstrate learning or improved survival.
