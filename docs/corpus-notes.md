# Predator–prey with optional corpus notes

The hypothesis is a causal chain: an agent observes something, publishes it, a
peer reads it, the peer changes its action, and its outcome improves. Note counts
establish only the beginning of that chain.

This experiment keeps predator–prey, eight founders, cap sixteen, social
normalization by sixteen, cooperative preference 0.5, and adult-policy inheritance.
It compares **grounded notes** with **free-form LLM notes** using the actual local
SmolLM2-135M-Instruct model and private LoRA adapters. No paid API or GPU is needed.
It does not resume the earlier cooperative experiment.

## Choosing actions through the corpus

1. An agent receives its permitted observations and at most one relevant deposited
   note. Retrieval prioritizes fresh food reports when hungry, then ranks by
   distance to the agent and its nearest known food station. This is a relevance
   heuristic, not learned search.
   Observing a fact does **not** automatically publish it.
2. The LLM encodes this context. A recurrent actor uses its language features plus
   direct numerical inputs for energy, local geometry and legal movement.
   **Corpus facts are excluded from this direct numerical actor input**: they
   enter through language. The separate critic can also use retrieved evidence.
3. The actor samples one of seven physical moves. An optional publication head
   decides whether to leave a note when fresh evidence is available and its
   sixteen-tick cooldown permits. There is no `get_food`/`put_terrain` tool menu.
4. The grounded arm renders a compact observation as text. The prose arm generates
   up to 48 new tokens. The service attaches observed facts, time and author.
   Publications become available after all agents have chosen their moves.

The service currently chooses candidate evidence: fresh food first, otherwise
observed walls, at most four facts. The agent learns **whether** to publish, not
which evidence to attach. Prose can still be wrong. The evidence service verifies
the attached measurements, **not the author's interpretation**. This is bounded
language sharing, not unrestricted research or learned semantic retrieval.

## What learns and what is rewarded

Physical moves and publication decisions receive the personal/cooperative PPO
advantages. Publication also receives `0.1` when its note reaches a living peer's
next context with previously unknown evidence, minus `0.01` per write. Each
recipient/fact version is credited once per map, with at most one bonus per
recipient/note. Writing costs `0.05` energy. Longer notes and repeated reads earn
no extra credit. Credit is immediate; useful deliveries many ticks later are not
retroactively credited to the original publication in this version.

This is **new-information delivery credit**, not causal credit for improving
the recipient. It can reward information the recipient ignores. Frozen access
ablations determine whether sharing actually helps. Do not interpret the shaping
reward as a successful solution to teaching credit assignment.

PPO updates private LoRA, the recurrent movement controller, publication head and
separate value function. The writer uses the frozen backbone with action adapters
temporarily disabled and KV caching. **Prose wording is not trained with
token-level RL.** Growing memory and changing context are distinct from parameter
learning.

Both styles receive 8,192 observation-limited demonstration decisions with shared
note context. The teacher uses retrieved walls and food reports to choose routes.
Each style has its own warm baseline because its text differs. Initialization,
architecture and budgets match; demonstration-trained weights need not match.

## Other bottlenecks addressed

- Direct physical inputs address weak sensitivity to hunger and geometry while
  keeping corpus content in the language path.
- A `0.1` imitation loss preserves FEED on visited states where it is legal,
  energy is below 80 and no predator is visible. It never overrides deployed
  actions. Entropy is reduced to `0.003`. This is an explicit behavioral prior;
  any improvement cannot be attributed solely to RL.
- Demonstrations now consume corpus evidence. A synthetic probe holds local
  sensors fixed and swaps which remote food station has stock. It records both
  probability changes and correct preferred directions. Sensitivity alone does
  not demonstrate useful behavior.
- Food notes expire after sixteen ticks, including their prose. The corpus keeps
  at most 256 recent notes per map. Dead authors' notes persist until eviction or
  map reset; coordinate facts never carry into different maps. Abstract strategy
  transfer across maps is not implemented.
- CPU work is bounded by one retrieved note, a 512-token context, publication
  cooldown, cached generation and serial execution. Tokens, generation counts,
  context-cap encounters and elapsed time are logged. There is no non-LLM fallback.
- Atomic checkpoints contain notes, policies, optimizers, buffers and RNG state.
  An initial checkpoint exists before evaluation. Resume truncates uncommitted
  log suffixes. A JSON sidecar exposes committed progress without loading models.

Remaining limitations: the 135M model is a weak reasoner; private lifetimes
fragment training data; survival feedback is delayed; and station coordinates
are already known, limiting the information advantage. Prose can add distraction,
hallucinations and compute without adding knowledge. The first real smoke run
inferred food-supply stability from one stock reading. The prompt was tightened,
but a prompt is not a semantic verifier.

## Experiment and interpretation

Default: **two styles × 32,768 population decisions**, cooperative R-adult, seed 11.
A decision is one living agent's move, not one game or update. Warm starts and
evaluation are additional work. Predator diffusion temperature stays **1.0**,
including evaluation. Novelty reward and changing curriculum are disabled.

At 0, 16,384 and 32,768 decisions, evaluate frozen policies on matched unseen maps,
with shared notes and with peer access withheld. Initial/midpoint use two maps;
final uses four; horizon is 512 ticks. Evaluation has no learning or reproduction
and starts a fresh corpus on each map. Withheld access retains private notes.
This tests the complete access condition, including subsequent trajectory changes,
not one specific note's causal effect.

The maximum evaluation allocation is 131,072 additional agent decisions across
both arms; extinction reduces actual use. There are 16,384 demonstration decisions.
Twelve hours is a **soft invocation budget**, not a promise both arms finish:
warm starts and evaluations finish before pausing. Repeat the command to resume.
Compare wall time and tokens as well as equal physical-decision budgets.

Before scaling, look for food and held-out lifetime gains from each style's own
warm baseline, preserved hungry feeding, positive frozen peer-access ablations,
and appropriate responses to the cue probe. Inspect per-map effects. A prose
advantage should justify its extra compute. One training seed is an exploratory
screen; map repetitions are not independent training runs. A failed screen could
reflect poor language use, weak information opportunities or insufficient
optimization. It does not refute the broader swarm thesis.

The [saved CPU smoke validation](results/corpus-notes-validation.json) completed
both real-LLM arms with 128 training decisions and 256 evaluation decisions each,
in about four minutes combined including their small warm starts. LoRA and
publication weights changed, and new information reached peers. Neither policy
passed the food-cue probe. Prose still included unsupported statements after
prompt tightening. These are execution results, not evidence of learned benefit.

## Terminal commands

The first full preparation completed both styles (8,192 demonstrations each),
but neither benefited from peer access on the two evaluation maps. Grounded/prose
mean lifetime was 110.25/110.94 ticks; both fed on 53/74 hungry opportunities and
had zero survivors at 512 ticks. These are demonstration-trained baselines, not
post-PPO outcomes.

An [audit of the grounded demonstrations](results/corpus-notes-demo-audit.json)
found peer context in 7,194/8,192 examples but **zero changed teacher actions**
when removing peer information. The check holds physical observations and
tie-breaking RNG fixed, including a separate helper that never imports peer wall
memory. This is a label-dependence audit, not a counterfactual survival rollout.
It exposes insufficient supervision for using notes: simply adding peer text to
an imitation dataset does not teach a decision that depends on that text.

Reproduce the audit without loading or training an LLM:

```bash
.venv/bin/python -m millstlabs.demo_audit
```

The next recommended change is a balanced reader curriculum: pair identical
local observations with different valid remote food reports that require
different physical moves; vary locations, direction, obstacles and phrasing;
test on unseen layouts and withheld/mismatched notes. Establish this prerequisite
before expanding the existing PPO budget. This targeted curriculum is a proposed
next experiment, not part of the current runner. Successful imitation would
establish information use, not autonomous discovery. Subsequent RL should measure
new strategies through actual food/survival improvement and retain the frozen
peer-access control; note volume alone is not a discovery objective.

```bash
cd /Users/diego/Downloads/millstlabs
bash scripts/corpus_notes.sh --mode plan

# Short real-LLM check of both styles; use a fresh output for a fresh check.
bash scripts/corpus_notes.sh --mode smoke --output runs/my-notes-smoke
bash scripts/corpus_notes.sh --mode report --output runs/my-notes-smoke

# Overnight on macOS. Repeat this same command to resume.
caffeinate -i bash scripts/corpus_notes.sh --mode run --hours 12
```

Recommended before committing the overnight budget: run
`caffeinate -i bash scripts/corpus_notes.sh --mode prepare --hours 12`, then
`bash scripts/corpus_notes.sh --mode report`. This performs the full demonstrations
and initial frozen evaluations for both styles, with **zero PPO collection**.
The run command reuses these warm starts and resumes their checkpoints. If feeding
is poor or the reader cannot use the controlled food cue, fix that bottleneck
before buying more training time. Preparation itself includes LLM training and
evaluation, so it is substantially longer than the smoke test.

On Linux/Codespaces, use `/workspaces/millstlabs` and omit `caffeinate -i`.
The wrapper uses `.venv/bin/python`; activation is unnecessary. For a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train,api,dev]'
```

First use downloads the pinned model if necessary. Once cached, prefix commands
with `HF_HUB_OFFLINE=1` to use it offline. No API key is used. Changing configuration
requires a new output directory; incompatible resumes are rejected.

Inspect in a second terminal:

```bash
cd /Users/diego/Downloads/millstlabs
bash scripts/corpus_notes.sh --mode report
tail -n 2 runs/corpus-notes/*/prosocial-r_adult-s11/ecology.jsonl
tail -n 2 runs/corpus-notes/*/prosocial-r_adult-s11/knowledge.jsonl
tail -n 3 runs/corpus-notes/prose/prosocial-r_adult-s11/corpus_events.jsonl
tail -n 3 runs/corpus-notes/prose/prosocial-r_adult-s11/note_decisions.jsonl
cat runs/corpus-notes/*/prosocial-r_adult-s11/checkpoint-status.json
```

An arm's directory appears when it starts. Report is read-only and does not infer
process liveness from saved summaries. Logged decisions can exceed the committed
checkpoint after interruption; resume replays that suffix. Ctrl+C stops a run;
repeat the run command to resume. An interrupted, unsaved warm start restarts.

| File | Meaning |
|---|---|
| `ecology.jsonl` | Population, food, deaths and physical action counts |
| `corpus_events.jsonl` | Actual notes, evidence, authors and delivery credit |
| `note_decisions.jsonl` | Chosen moves, context note IDs and publication decisions |
| `knowledge.jsonl` | Cumulative sharing, generation and shaping-reward counts |
| `updates.jsonl` | PPO diagnostics and publication reward |
| `evaluation.jsonl` | Held-out food and survival with shared access |
| `evaluation_private_corpus.jsonl` | Frozen peer-access ablation |
| `note_probe.jsonl` | Controlled cue sensitivity and direction checks |
| `corpus-checkpoint-*.json` | Human-readable corpus snapshots |
| `checkpoint-status.json`, `latest.pt` | Committed progress and resumable state |

For one style, use e.g. `--style grounded --output runs/notes-grounded` when
creating a new deployment. Keep that selection when resuming the directory.

## Optional HTTP interface

Start `.venv/bin/python -m millstlabs.cli serve`. Create an environment with
`{"seed":11,"corpus_mode":"shared","corpus_interface":"notes","note_style":"prose"}`.
`POST /environments/{id}/step` accepts `expected_tick`, one physical action per
living agent, and optional `"notes":{"prey_0":"I observed ..."}`. Observations
include retrieved notes and publication availability. The service attaches the
author's permitted evidence and enforces cooldown and a 480-character limit.
The caller supplies prose; grounded mode renders evidence itself. No knowledge
tool commands are required. The old API remains available for old experiments.
