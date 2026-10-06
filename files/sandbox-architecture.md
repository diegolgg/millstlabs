# Sandbox 1 architecture: organized LLM populations on Hanabi

Status: design spec, 2026-10-05. Written to be handed to a coding agent and built in one day. Lives on the `enrico` branch under `files/`. Companion documents: `lit-map.md`, `../reports/Sandbox game selection lit review.md` (game choice and evaluation rationale), `../research_notes/Sandbox game selection lit review/*.md` (sourced facts). Open choices were resolved in chat on 2026-10-05 and are recorded in section 17.

## 0. What this sandbox is for

One sentence: measure which organizational mechanisms (teaching, verification before adoption, credit for students' improvement, competition between groups for compute, migration) make a population of LLM agents keep improving at a skillful game over a long horizon, at matched token budget, against isolated agents and against the open self-improving-agent harnesses.

The LLM is never in the per-turn game loop. Agents write **artifacts** (a written convention document plus the Python bot that implements it); a deterministic engine scores artifacts for free. LLM spend scales with authoring and teaching events, not with games. This is what makes thousands of generations, paired-seed counterfactuals, and apples-to-apples baselines possible.

Design principles, in priority order:

1. **Hypothesis throughput.** Every mechanism is a pluggable policy; every experiment is a YAML config; a screening run finishes in minutes. Nothing requires a code fork.
2. **Long-running is the variable, not a nuisance.** The lab's hypothesis is that social dynamics that produce collective improvement need long horizons. So the runner is built for thousands of generations with resume, every metric is a trajectory rather than an endpoint, and "same budget, more generations" is itself an experiment (section 16, item 2).
3. **Record and replay every LLM call.** A bug in the organization layer or the analysis is fixed and the run replays from cache without re-spending tokens.
4. **Paired seeds everywhere.** Every condition in a comparison sees the same deals, the same anchors, the same warm start, verified by content hash.
5. **Ground truth before any LLM call.** The engine, adapter and evaluator reproduce published bot scores before an agent writes a line.
6. **Game-agnostic core.** The organization layer talks to a `Game` interface; Hanabi is the first instance and a second game (Liar's Dice was the candidate) is deferred until a Hanabi result exists. Nothing in `org/` imports from `game/hanabi`.
7. **Honest accounting.** Every curve is plotted against cumulative tokens and dollars by call type.

## 1. Fixed versus variable

| Fixed for the whole program | Varies per experiment (config) |
|---|---|
| Game and engine (2-player Hanabi, No Variant; second game deferred) | Population size, number of groups, topology |
| Artifact format (conventions.md + bot.py) and bot interface | Routing (who receives what), delivery (deterministic or randomized) |
| Evaluation protocol (paired seeds, anchors, frozen ladder, cross-play) | Verification policy, adoption rule |
| LLM backend interface and cache | Credit rule, allocation rule, selection rule, migration rule |
| Logging schema, checkpoint format, config digest | Model tier, effort, per-group budget, teaching cost, horizon |
| Metrics and statistics code | Environment change schedule (rule-variant switches) |

## 2. Repository layout

Lives on the `enrico` branch, separate package from Diego's `millstlabs`, same discipline.

```
src/culture/
  game/          interface.py          Game protocol: new_game(seed) -> state; legal_moves; step; score; observation(player) -> dict
                 hanabi.py             HLE adapter (observation JSON, 1-based ranks, absolute targets, legal-move check, hanab.live export)
                 liars_dice.py         (deferred) OpenSpiel adapter (+ exploitability via policy wrapper)
                 seeds.py              seed = base + game_index; shared seed sets per experiment
  bots/          interface.py          Bot protocol (section 4)
                 runner.py             sandboxed execution of LLM-written bot code (subprocess, rlimits, import whitelist, per-move timeout)
                 anchors/              random.py, simple.py (HLE SimpleAgent port), canaan/ (Piers, IGGI, Flawed; vendored, Apache-2.0), obl_l1.py (optional, JaxMARL flax)
  evaluate/      selfplay.py           N seeded games of one artifact with itself -> per-game scores
                 crossplay.py          matrix of artifact pairs on shared seeds
                 ladder.py             current snapshot vs frozen past generations + anchors
                 stats.py              IQM, stratified bootstrap, paired deltas, CI widths from sigma, Higher Criticism
                 exploitability.py     (deferred, second game only)
  artifacts/     schema.py             Artifact, Evaluation, TeachingMessage (dataclasses, JSON)
                 store.py              per-group corpus: deposit, retrieve, touch log
                 provenance.py         DAG of parents + teaching edges; lineage queries
  llm/           backend.py            Backend.complete(request) -> Response; cost ledger
                 anthropic_backend.py  Anthropic SDK; prompt caching on frozen prefix; batch mode
                 openai_compat.py      OpenAI-compatible endpoint (OpenAI models such as GPT Luna, vLLM, Together, Ollama)
                 stub_backend.py       zero-cost backend returning canned artifacts (anchor bots with seeded perturbations) for debugging and CI
                 cache.py              record / replay / replay_strict keyed by request hash
                 prompts/              versioned templates: author.md, revise.md, teach.md, ingest.md, merge.md
  agents/        agent.py              one agent's generation step: ingest -> verify -> revise -> evaluate -> teach
  org/           population.py         groups, membership, budgets
                 routing.py            none | broadcast_group | best_to_all | random_k | program (Colas-style share())
                 delivery.py           deterministic | bernoulli(p)
                 verification.py       none | selfplay(N) | selfplay_crossplay | critical_social_learning(theta)
                 adoption.py           replace_if_better | merge_llm | never
                 credit.py             none | paired_delta | lineage_decay | datamodel_regression | tmc_shapley | eom_payment
                 allocation.py         uniform | proportional | softmax_floor | nash_relative | eom_wealth
                 selection.py          keep_best_k | shinka_weighted | hgm_clade_ts | funsearch_islands
                 migration.py          none | random | best_to_neighbor
                 environment.py        fixed | variant_switch(at_generation, variant)
  run/           config.py             dataclasses, validate(), digest()
                 generation.py         one generation for the whole population (the only place that orders the steps)
                 runner.py             loop, atomic checkpoints, resume, wall-clock budget, deployment lock
                 manifest.py           run manifest: config digest, prompt hashes, anchor hashes, warm-start hashes
  analysis/      metrics.py            per-generation metrics from JSONL
                 graphs.py             teaching/adoption graph: DCMM fit, influence, community membership over time
                 figures.py            headline figures
                 report.py             markdown summary per run
  baselines/     openevolve_adapter.py evaluator shim so OpenEvolve runs on the same Game at matched calls
                 shinka_adapter.py     evaluate.py contract for ShinkaEvolve
                 generational.py       Vallinder-Hughes-style generational prompt transmission (control)
configs/         probe.yaml stage1_transfer.yaml stage2_lineages.yaml stage3_credit.yaml horizon.yaml baselines.yaml
tests/           test_engine.py test_adapter.py test_bots.py test_evaluate.py test_cache.py test_org.py test_smoke_learning.py test_runner.py
notebooks/       00_probe.ipynb 01_transfer.ipynb 02_lineages.ipynb 03_credit.ipynb analysis.ipynb
third_party/     hanabi-learning-environment (git submodule pinned at 54e7959, the commit OpenSpiel ships)
```

## 3. Engine layer

**Hanabi.** Vendor `hanabi-learning-environment` as a submodule pinned at commit `54e7959` and build from source in a Python 3.11 venv (`pip install ./third_party/hanabi-learning-environment`; needs CMake and clang; there is no macOS/arm64 wheel on PyPI). Fallback if the build fights the machine: `pip install open_spiel` (arm64 wheel, built with Hanabi on) and drive `pyspiel.load_game("hanabi")`, writing the observation parser against its string observations.

Facts the adapter relies on (verified in the fact sheet):

- `rl_env.make("Hanabi-Full", num_players=2)` or `HanabiEnv({...,'seed': s, 'random_start_player': False})`. The RNG lives on the game object; **one env per game**, `seed = base + game_index`, otherwise game k depends on earlier games' draw counts.
- `step(move_dict) -> (obs, reward, done, info)`; illegal moves raise `AssertionError`, so the adapter validates against `legal_moves` first.
- Observation dict per player: `fireworks`, `information_tokens`, `life_tokens`, `discard_pile`, `observed_hands`, `card_knowledge`, `legal_moves`, `deck_size`; ranks are 0-based in HLE. Strip `pyhanabi` and `vectorized` before serializing.
- Throughput about 2.1e3 steps/s/core through Python, roughly 30 games/s/core; 1,000 games on 8 cores is seconds of engine time. Wall time is dominated by bot code.

**Observation JSON given to bots** (ranks 1-based, targets absolute, colors `R Y G W B`):

```json
{"game": {"num_players": 2, "hand_size": 5, "colors": ["R","Y","G","W","B"], "ranks": [1,2,3,4,5]},
 "me": 0, "current_player": 0, "turn": 12,
 "deck_size": 28, "info_tokens": 6, "life_tokens": 3,
 "fireworks": {"R": 1, "Y": 0, "G": 2, "W": 0, "B": 0},
 "discards": [{"color": "R", "rank": 1}],
 "hands": {"1": [{"color": "G", "rank": 3, "hints": {"color": null, "rank": 3}}]},
 "my_hand": [{"color": null, "rank": null, "hints": {"color": "R", "rank": null}}],
 "last_moves": [{"player": 1, "type": "REVEAL_RANK", "target": 0, "rank": 1, "touched": [0, 2]}],
 "legal_moves": [{"type": "PLAY", "card_index": 0}, {"type": "DISCARD", "card_index": 0},
                 {"type": "REVEAL_COLOR", "target": 1, "color": "G"}, {"type": "REVEAL_RANK", "target": 1, "rank": 3}]}
```

`last_moves` comes from `pyhanabi.last_moves()` on the raw observation object (exists on `HanabiObservation`, not in the dict; verify during build). Keep `card_index` 0-based as the hand-slot position.

**Replay export.** Write hanab.live replay JSON (format 3.0.0: `players`, `deck` in deal order, `actions` with `type 0..4`, `options.variant: "No Variant"`). hanab.live's `target` for play/discard is the deal-order index, so the exporter tracks slot-to-deal-index. A human watches any game by pasting the JSON into "Watch Specific Replay → JSON Data". This is the interpretability path; it costs one afternoon and is worth it.

**Second game (deferred).** Liar's Dice via OpenSpiel was the candidate cross-check (exact exploitability, CFR+ anchor). Not built until a Hanabi result exists. If revived, bots must be decision rules with a per-decision compute cap and no precomputed policy tables, otherwise an agent that recalls CFR solves the one-die game in generation one.

## 4. Bot interface and execution sandbox

What LLM-written code must implement:

```python
class Bot:
    name: str = "my_bot"
    def reset(self, game: dict, my_id: int, seed: int) -> None: ...
    def act(self, obs: dict) -> dict:  # must return an element of obs["legal_moves"]
        ...
```

Rules enforced by the runner, not by trust:

- Pure Python; imports whitelisted by AST inspection (`math`, `random`, `itertools`, `collections`, `functools`, `dataclasses`, `typing`); no I/O, no network, no subprocess.
- Deterministic given `seed`: the bot's only randomness source is `random.Random(seed)`. Determinism is tested (two runs, identical action logs).
- Runs inside a worker process with `resource` limits (CPU seconds, memory) and a per-move soft timeout (default 50 ms, hard 1 s). Illegal move or timeout → fallback action (discard oldest, else first legal), counted in `illegal_rate`; verification fails if `illegal_rate > 0.01`.
- Size cap (400 lines, configurable) and a logged complexity measure, to resist the codebase bloat CodeClash reports.

The sandbox protects against bugs (infinite loops, accidental file access), not adversaries; these are our own model's outputs.

## 5. Artifacts, messages, corpus, provenance

```python
@dataclass
class Artifact:
    id: str                 # sha256(code + conventions)
    code: str               # bot.py
    conventions: str        # markdown: the rules the code implements, in prose
    author: str             # agent id
    group: str
    generation: int
    parents: list[str]      # artifact ids this was derived from (own previous and/or adopted)
    teaching_sources: list[str]  # message ids ingested when producing this
    evaluation: Evaluation | None
    verification: Verification | None

@dataclass
class Evaluation:
    seed_base: int; n_games: int
    selfplay_scores: list[int]; selfplay_iqm: float; selfplay_ci: tuple[float, float]
    crossplay: dict[str, float]    # artifact id -> mean score on shared seeds
    anchors: dict[str, float]      # anchor name -> mean score paired with this bot
    illegal_rate: float; lines: int; wall_seconds: float

@dataclass
class TeachingMessage:
    id: str; sender: str; receiver: str; generation: int
    artifact_id: str               # the payload
    delta_text: str                # what changed and why, written by the sender
    evidence: Evaluation           # the sender's own evaluation of the payload
    delivered: bool                # set by the delivery policy (randomized delivery)
    touched_at: int | None         # generation at which the receiver actually read it
```

**Corpus** is per group: `deposit(artifact, evidence)` validates that `evidence` is a real evaluation record produced by the harness (hash-checked), so an agent cannot publish claims it did not earn; `retrieve(policy)` returns artifacts and writes the **touch log** `(receiver, message_or_artifact_id, generation, used)`. The touch log is what makes per-teacher credit computable; without it, credit assignment is guesswork. Two corpus interfaces (ours and Diego's `knowledge.py`) should converge on the same verbs so the organization layer is shared later; not a day-one requirement.

**Provenance DAG**: nodes are artifacts, edges are `parents` and `teaching_sources`. Lineage queries (clade of an artifact, teachers of a student, depth) are used by credit and selection policies, and the DAG plus touch log are the data for the graph analyses in section 10.

## 6. LLM layer

```python
@dataclass
class Request:
    system: list[dict]          # frozen prefix first (rules, bot interface, conventions doc), cache_control on it
    messages: list[dict]
    model: str                  # claude-opus-5 default; claude-sonnet-5, claude-haiku-4-5 as tiers
    max_tokens: int = 16000
    effort: str = "high"        # output_config.effort
    tag: str                    # author | revise | teach | ingest | merge (for the cost ledger)
    seed_tag: str               # run id + agent + generation, so identical prompts in different runs are cached separately when desired

class Backend(Protocol):
    def complete(self, req: Request) -> Response: ...      # Response: text, usage, cost_usd, cached: bool
    def complete_batch(self, reqs: list[Request]) -> list[Response]: ...  # Message Batches when a generation is embarrassingly parallel
```

- **Anthropic backend**: `anthropic.Anthropic()` resolves credentials from the environment. Adaptive thinking (omit `thinking` on Opus 5; set `output_config.effort`). `cache_control: {"type": "ephemeral"}` on the frozen system prefix; verify `usage.cache_read_input_tokens > 0` in tests. Generation-parallel authoring goes through `client.messages.batches.create` (50% price; results keyed by `custom_id`, never by position). Include server-side fallbacks on Opus 5 per the SDK guidance.
- **OpenAI-compatible backend**: `base_url` + model name, for vLLM, Together, Fireworks, Ollama. Same `Request`. This is how open-weight models enter later without touching `org/`.
- **Cache**: key = sha256(backend, model, effort, system, messages, max_tokens, seed_tag). Modes `record` (call and store), `replay` (serve hits, call on miss), `replay_strict` (raise on miss; used in CI and when re-analyzing). Stored with usage and cost.
- **Cost ledger**: tokens and dollars per `(run, group, agent, generation, tag)`, cached tokens separately. Every figure reads this ledger for its x-axis.
- **Prompt templates** are files whose hashes go in the run manifest. Prompts describe the task and the interface, not style; the thing under test is the organization, so prompts are frozen across conditions within an experiment. Prompts never describe the credit, selection or allocation rule (agents told the rule game it; a disclosure condition is a queue item).

**Model tiers (decided 2026-10-05).** Spend ramps with confidence: $0 for all debugging and CI (stub backend plus replay cache), under $1 for the probe, a few dollars for stage 1, and only after the probe gates pass does any run cost more. Screening default is Haiku 4.5 ($1/$5 per million tokens, half in batch; roughly $0.012 per revise call, about $0.20 per 12-agent generation, about $15 per 100-generation run). GPT Luna is probed alongside Haiku through the OpenAI-compatible backend and takes the screening slot if it writes comparable bots for less. Sonnet 5 ($2/$10) is reserved for headline runs. Opus is not used for agents.

## 7. The agent step

One agent, one generation, in this order. `generation.py` is the only file that orders these; policies are injected.

1. **Receive**: messages routed to this agent this generation, after the delivery policy has flipped `delivered`.
2. **Ingest and verify**: for each delivered artifact, run the verification policy (engine time only): N seeded self-play games, cross-play against the agent's incumbent, anchors. The adoption policy then replaces, merges (one LLM call with both artifacts and both evaluations), or rejects. Record the decision and the scores on the message (this is the verification log).
3. **Revise**: one LLM call. Context: own current artifact and conventions, own evaluation with failure examples (traces of the three lowest-scoring games, in the observation JSON), adopted material and its evidence, and the group's retrieved corpus view. Output: a full rewrite or a SEARCH/REPLACE diff of `bot.py` plus a conventions delta. Invalid code gets one repair call with the traceback, then counts as a failed revision.
4. **Evaluate**: self-play on the generation's shared seed set, cross-play against group incumbents, anchors. Paired seeds mean every agent in every condition plays the same deals this generation.
5. **Teach** (if the routing policy selects receivers and the budget allows): one LLM call producing `delta_text`; payload is the artifact; `evidence` is attached by the harness, not written by the agent.

Budget: each LLM call debits the agent's share of its group's budget for this generation. When the group's budget is exhausted, remaining agents skip revise/teach (they still evaluate, which is free).

## 8. Organization policies

Each is a small class with a single method and a config block. Expected size 30 to 150 lines each.

| Policy | Options (v0 in bold) | Notes |
|---|---|---|
| Topology | **isolated**, full, islands(G, migration_rate, interval), ring | Derex & Boyd partial connectivity is `islands` |
| Routing | **none**, **broadcast_group**, best_to_all, random_k, program | `program` is a Colas-style `share(step, agent_states) -> exchanges` evolved by the LLM; stage 3+ |
| Delivery | **deterministic**, bernoulli(p) | bernoulli is required for datamodel credit |
| Verification | none, **selfplay(N=200)**, selfplay_crossplay, critical_social_learning(theta) | critical: adopt first, re-verify after one revision, revert if below theta |
| Adoption | **replace_if_better**, merge_llm, never | merge costs one LLM call |
| Credit | none, **paired_delta**, lineage_decay(gamma), datamodel_regression(lambda), tmc_shapley, eom_payment | see section 11 |
| Allocation | **uniform**, proportional, softmax_floor(T, eps), nash_relative, eom_wealth | softmax_floor's `eps/G` floor keeps every group sampleable (same role as the uniform floor in Online TASS) |
| Selection | **keep_best_k**, shinka_weighted(lambda=10), hgm_clade_ts(alpha=0.6), funsearch_islands(reset_period) | formulas in section 15 |
| Migration | **none**, random(rate, interval), best_to_neighbor | |
| Teaching cost | **free**, costly(c) | with `credit_share(alpha)` the teacher receives alpha of credited student improvement as budget |
| Environment | **fixed**, variant_switch(at_generation, variant) | Rogers test; variants must be HLE-expressible (colors, ranks, hand size, tokens) |
| Horizon | generations G and per-generation budget b, with G × b fixed per experiment | the long-running test: many small generations versus few large ones at equal total spend |

## 9. Evaluation suite

Game-agnostic core, computed per generation from frozen snapshots:

- **Score**: per-artifact self-play IQM on the shared seed set; group best and group mean.
- **Cross-play**: within-group and between-group matrices on shared seeds. Between-group cross-play versus self-play is the headline Hanabi quantity (the no-teaching null is the SAD-style collapse).
- **Frozen ladder**: current snapshot versus every past generation's snapshot and the anchor set (random, SimpleAgent, Canaan Piers and IGGI, optional OBL-L1). Non-cycling check: later generations should dominate earlier ones on the ladder.
- **Diversity**: number of distinct artifact clusters by code embedding (cosine threshold, as ShinkaEvolve's novelty check) and behavioral distance on a fixed probe set of 200 observations (action-distribution disagreement). Nash-cluster count later.
- **Retained innovations**: artifacts that beat the archive best at generation g and still have descendants at g+k.
- **Credit and teaching**: credit per teacher, teaching rate, adoption rate, verification pass rate, fraction of adopted artifacts later reverted.
- **Trajectory shape** (the long-running claims): generation at which between-group cross-play first exceeds a threshold; slope of improvement in the last third versus the first third of the run; changepoints in the score and diversity series; survival of innovations across environment switches.
- **Cost**: cumulative tokens and dollars by call type; every curve's x-axis.
- **Second game (deferred)**: exploitability per generation if Liar's Dice is revived.

Statistics: paired seeds across conditions; IQM with stratified bootstrap over population seeds; headline comparisons use at least 10 population seeds, queue screening uses 3; effect-size thresholds are written in the experiment config before the run. Sample sizes from the sigma arithmetic in the report: about 30 games per measurement for low-variance code bots, 200 for screening, 1,000 for anchor-level claims.

## 10. Analysis toolkit from high-dimensional statistics (STAT 236)

These are the tools from Enrico's coursework that answer questions the standard population metrics cannot. The data they need (per-agent z-scores against a frozen reference, the teaching and adoption graph, behavioral feature vectors) all come from the logs above, so none requires new instrumentation. Ranked by usefulness.

1. **Higher Criticism for "did anyone improve?"** Each generation, every agent's score delta against the frozen previous generation on shared seeds is a z-score. HC detects that a sparse subset of agents improved before any individual is significant at Bonferroni, and estimates the fraction that improved. Use Efron's empirical null, because agents sharing teachers and seeds inflate the null variance. A plateau is HC staying under its threshold for k generations; this is the stopping and allocation signal.
2. **Community detection on the teaching graph.** Fit a degree-corrected mixed-membership model (DCMM) to the adoption graph (edge i→j when j adopted an artifact from i, weighted by credit). Per-node degree parameter θ is influence (hub teachers); membership vectors π are idea lineages; the off-diagonal of the community matrix is cross-lineage leakage. Track π over generations to watch an idea propagate, and compare recovered communities with the configured groups: a population whose idea-communities ignore its group boundaries has achieved transfer; one whose communities coincide with groups has not.
3. **Rare-weak phase diagram as the organization-versus-scale template.** Map p = number of candidate artifacts, ϑ = rarity of genuinely better ones (ε = p^−ϑ), r = evaluation signal-to-noise (τ = sqrt(2 r log p)). Accumulation needs the population in the almost-full-recovery region r > ϑ, not merely the detectable region. Scale alone (more candidates) raises ϑ toward the undetectable corner; organization is anything that raises r (more evaluation per candidate, exploiting correlation) or lowers ϑ (teaching concentrates good variants). This gives the formal "when organization beats scale" statement the one-pager promises, with measurable inputs.
4. **Innovated Higher Criticism for correlated agents.** Agents that share a teacher have correlated score noise; build the covariance from the games-by-agents score matrix and transform by its inverse (or project out the top principal components, which capture common opponents and global drift). The detection region strictly expands.
5. **IF-PCA for counting distinct strategies.** Rows are artifacts, columns are behavioral features (action distributions on the probe set); rank features by a KS score, keep those above an HC threshold, then spectral clustering. The theory warns that adding uninformative features destroys the clustering unless features are selected first, so "log more features" is not free.
6. **Erdős–Rényi thresholds for teaching degree.** Average teaching out-degree above 1 gives a giant component (ideas can reach a constant fraction of the population); above log n gives full reachability. Routing policies should keep per-agent degree above these regardless of population size, or transfer cannot happen for graph reasons.
7. **HC feature selection for "what do winners do differently?"** Two-sample z-scores on action n-gram counts between high- and low-scoring artifacts; HC picks the feature set; the signed list is an interpretable summary of what the population learned, and can be fed back to agents as a teaching hint (an experiment, not a default).

Items 1, 2 and 5 go in `analysis/` in the first build as functions over the logs; 3 is a modelling section for the white paper; 4, 6, 7 are queue items.

## 11. Credit for students' improvement

No published metric does this; we define it and say so. v0 and v1 below; the rest are queue items.

- **v0 `paired_delta`.** For a message m from teacher T adopted by student s at generation g: `delta = f(s_after; A, seeds_g) - f(s_before; A, seeds_g)` on identical seeds against the anchor set A. `credit[T] += delta`. If several messages are adopted in one window, split equally and flag the window. Assumes additivity and is biased when the student also self-revised in the window; cheap, low variance.
- **v1 `datamodel_regression`.** Delivery is Bernoulli(p=0.5) and logged. After each generation, regress each student's delta on the inclusion indicators of messages it could have received over a window, with ridge or lasso; `credit[T] = sum of coefficients over T's messages`. Unbiased for multiple teachers; needs many more windows than messages, so pool across students and generations with message-type features. This is the touch-pattern idea from the DP work, and the mechanism nobody in the literature uses.
- **`lineage_decay(gamma)`**: propagate a student's later deltas to its teachers' teachers with decay; pools evidence, over-credits.
- **`tmc_shapley`**: truncated Monte Carlo Shapley over the set of teachers with replayable ingestion; exact in expectation, expensive.
- **`eom_payment`**: Economy-of-Minds bucket brigade, bids and wealth; credits bid calibration, not measured improvement; collusion-exploitable; included as the market baseline.

Credit feeds two things: the allocation policy (an agent's share of group budget) and the teaching incentive (`credit_share`). It is also always reported as a metric regardless of whether it is used.

## 12. Testing

Written before the LLM is connected; the whole pipeline must pass without a single model call.

- **Engine determinism**: same seed and seeded RandomAgent produce byte-identical action logs across runs and processes.
- **Adapter correctness**: Canaan's `piers_agent.py` unmodified, driven through raw HLE and through our adapter on the same seeds, yields identical trajectories; 1,000-game mean 16.99 within 0.5; IGGI 15.99 within 0.5; `Flawed` scores about 0. The Flawed check catches hidden-information leaks in the adapter.
- **Sandbox**: an infinite-loop bot times out and falls back; an import of `os` is rejected; `illegal_rate` is counted.
- **Stats**: paired delta of an artifact against itself is exactly 0; bootstrap CI widths match the sigma arithmetic on synthetic scores; HC on pure-null z-scores stays below threshold at the nominal rate.
- **Cache**: replay reproduces byte-identical responses; `replay_strict` raises on miss; ledger totals equal the sum of per-call usage; cache hits are recorded on repeated frozen prefixes.
- **Organization units**: credit propagates correctly on a synthetic provenance DAG; allocation sums to the budget and respects the floor; migration conserves agent count; routing respects topology; Bernoulli delivery rate matches p over 10,000 draws.
- **Learning smoke test (no LLM)**: an oracle teacher hands Canaan Piers (about 17) to a student holding RandomAgent (about 0). With verification on, the student adopts and scores 17. With verification off and a sabotaged payload (`Flawed`), the student adopts and drops to 0. With verification on and the same sabotage, the student rejects. This proves teaching, verification and adoption work end to end, deterministically.
- **Runner**: config digest refuses a changed config in a used output directory; a run killed mid-generation resumes to the identical state; deployment lock refuses concurrent access; a 1,000-generation dry run with stub policies completes and its logs stay bounded.
- **End-to-end tiny run** in CI with the LLM cache in `replay_strict` from a committed fixture.

## 13. Notebooks

- `00_probe.ipynb`: one agent, no teaching, one generation, for each candidate model tier. Reports: self-play score, illegal rate, lines, tokens and dollars per call, and a diff of the generated conventions against the H-group rulebook (embedding similarity plus a keyword list such as "finesse", "chop", "5 save"). Second independent agent, same model: cross-play between the two. Gates in section 14.
- `01_transfer.ipynb`: stage 1, three conditions.
- `02_lineages.ipynb`, `03_credit.ipynb`: stages 2 and 3.
- `analysis.ipynb`: reads JSONL and the ledger, produces the headline figures: between-group cross-play versus tokens with anchors (OBL cross-play 23.76, humans 23.37, SmartBot 22.99, o3 with deductions 17.5); the teaching-graph communities over time; exploitability versus tokens only if the second game is revived.

## 14. Build order and gates

Single-day build, in this order; each step has its tests passing before the next starts.

1. Engine adapter, seeds, anchors, hanab.live export, engine tests. (about 2 hours)
2. Bot runner sandbox, self-play and cross-play evaluators, stats. (1 hour)
3. LLM backend, cache, ledger, prompt templates. (1 hour)
4. Artifact schema, corpus, provenance, touch log. (1 hour)
5. Agent step, v0 policies, generation loop, config digest and freeze, checkpoint and resume, lock. (2 hours)
6. Probe notebook and probe run. (1 hour)
7. OpenEvolve adapter (lightest external harness, `Callable` evaluator). ShinkaEvolve adapter next day. (1 hour)

Probe gates, decided before running:

- **Headroom**: generation-one bots score below 20 in self-play for the chosen tier. If they land at 22 or above and the conventions diff reads as H-group recall, switch the main runs to an official hanab.live variant (see section 17) and keep No Variant for anchoring.
- **Cost**: tokens per revise and teach call measured; stage sizes derived from it, not guessed.
- **Null alive**: two independent agents' bots cross-play well below their self-play. If they are already compatible, the between-group headline has no range and the experiment design shifts to score and diversity.

## 15. Import versus build

| Source | Verdict | What we take |
|---|---|---|
| OpenEvolve (Apache-2.0, light deps) | Run as external baseline | `evaluate(program_path) -> dict`; its `PopulationStrategy` and `IslandSelector` hooks can host our policies if we ever want to run inside it |
| ShinkaEvolve (Apache-2.0, heavy deps) | Run as external baseline, day 2 | `evaluate.py --program_path --results_dir` writing `metrics.json` with `combined_score`; crib its weighted parent rule: `s_i = sigmoid(lambda (score_i - median) / MAD)`, `h_i = 1/(1+children_i)`, `p_i ∝ s_i h_i`, lambda=10 |
| HGM (SWE-bench-bound) | Reimplement the rule (about 50 lines) | clade counts `nC_s, nC_f` over the subtree; `CMP = nC_s/(nC_s+nC_f)`; parent by Thompson sampling `Beta(tau(1+nC_s), tau(1+nC_f))`; UCB-Air widening `n^alpha >= |T|`, alpha=0.6 |
| DGM | Reimplement parent rule (15 lines) and the staged-evaluation gate | same sigmoid rule with alpha0=0.5; cheap 10-game screen before full evaluation |
| FunSearch | Reimplement islands and worst-half reset (40 lines) | 10 islands, Boltzmann over score clusters, every `reset_period` discard the worse half and reseed from survivors' best |
| AlphaEvolve | Prompt shape only | previous programs with scores, SEARCH/REPLACE diff format, rendered evaluation feedback |
| Economy of Minds | Reimplement as `eom_payment` and `eom_wealth` | fixed bids, winner pays predecessor, outcome to last actor, rent, bankruptcy removal, mutate-the-richest |
| Colas et al. | Reimplement as `routing: program` (stage 3+) | `share(step, agent_states) -> {receiver: [(sender, artifact_id)]}`, 20-item receiver bottleneck, Thompson sampling over programs, improve/diversify rewrite prompts |
| Vallinder & Hughes | Reimplement as the generational control | text strategies, top half survive, next generation sees ancestors' strategies |
| Diego's `millstlabs` | Copy discipline, not code | content-hashed paired warm start, config digest and freeze, atomic checkpoints with log byte offsets, deployment lock, initial evaluation that cannot be back-filled, opportunity-normalized metrics; converge corpus verbs with his `knowledge.py` later |
| Canaan et al. rule-based agents (Apache-2.0) | Vendor | Piers, IGGI, Flawed as Python anchors that consume HLE dicts unchanged |
| JaxMARL + OBL-L1 flax weights | Optional anchor | 20.92 self-play, CPU inference; cross-engine parity check |
| hanabi-engine (Rust) | Replay analysis only | `analyze --explain` on exported hanab.live JSON as a second parser and a human-readable explanation of H-group decisions |

The cross-cutting fact from the harness review: every external loop scores one program in isolation. Our evaluator takes `(candidate, population_snapshot)` so cross-play and ladders are first-class; when an external baseline runs, its evaluator fetches the snapshot out of band.

## 16. Hypothesis queue (default order; reorder after the probe)

Each item is one config diff and one figure. Screening at 3 seeds, promotion to 10 seeds if the effect clears the pre-registered threshold.

1. **Transfer with verification beats solo at matched budget.** Conditions: solo; teacher's artifact delivered without verification; with verification. Headline: student score versus tokens. Replicates the Jha et al. 2026 setting and tests whether verification flips the sign.
2. **Horizon matters.** Same total budget split as 20 generations with large per-generation budgets versus 200 generations with small ones, with and without teaching. If long-running with small steps wins only when teaching is on, that is the lab's central hypothesis in one figure.
3. **Credit by randomized delivery recovers the true teachers.** Bernoulli delivery, datamodel regression; validate against planted ground truth (an oracle teacher whose effect we know). Nobody has done this.
4. **Fidelity as a knob.** Transmit conventions text only, text plus code, or code only. Measure copy loss alpha and copy variance beta from paired-seed student scores and test Henrich's accumulation condition `N* > exp(alpha/beta - 0.577)` on the live population.
5. **Persuasive-but-wrong injection.** Seed a convincing bad convention with real (bad) code; measure spread with and without verification.
6. **Partial connectivity.** isolated versus full versus islands at three migration rates; between-group cross-play, diversity, and recovered idea-communities versus generations.
7. **Teaching as a costly act with credit share.** Do self-interested agents teach only when credited.
8. **Allocation rules.** uniform versus softmax_floor versus nash_relative under a cloning attack (duplicate a group and see who gains).
9. **Selection rules at matched budget.** keep_best_k versus shinka_weighted versus hgm_clade_ts: does judging artifacts by their descendants beat judging by their own score, as HGM found.
10. **Critical social learning** versus verify-always: cheaper verification with the same protection.
11. **Environment change.** Switch to a variant mid-run; populations with verification and migration recover faster than copy-only populations (Rogers).
12. **Learned routing.** Evolve `share()`; content routing versus topology, as Colas found.
13. **Rule disclosure.** Tell agents the credit and selection rules versus not; measure gaming.
14. **External baselines at matched calls.** OpenEvolve, ShinkaEvolve, generational control, versus the full mechanism.
15. **Second-game replication** of items 1, 2, 6 and 14 (deferred; Liar's Dice with exploitability was the candidate).

## 17. Decisions (resolved 2026-10-05)

- **Model tier**: Haiku 4.5 for screening; GPT Luna probed alongside and adopted if comparable and cheaper; Sonnet 5 for headline runs; stub backend and replay cache for all debugging. No agent runs on Opus.
- **Variant fallback**: decided only if the probe shows recall at the ceiling. HLE-expressible options are the hanab.live "4 Suits" variant or hand size 4 with 6 clue tokens; six-suit variants are not expressible in the engine.
- **Teaching cost**: free until queue item 7, where costly teaching is the experiment.
- **Population size**: 3 groups of 4 for debugging and screening (the minimum that yields a between-group cross-play matrix); 4 groups of 5 for headline runs; 64 or more agents once a result exists and swarm scale matters for the demo. Henrich's model says N enters as log N, so size is second-order next to fidelity and evaluation noise; this is one of the contrarian points we expect to show.
- **Second game**: deferred. Hanabi only until a result exists.
- **Corpus interface with Diego's `knowledge.py`**: build ours without constraint; align names and schema later only where it costs nothing.

## 18. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Recall puts agents at the ceiling in generation one | Probe gate; hanab.live variant fallback; headline is cross-play dynamics, not raw score |
| Teaching does not help at matched budget (Jha et al. 2026) | Ablation ladder is the experiment; a clean negative with mechanism is still the paper |
| LLM mutation collapses diversity (Mutation Without Variation, 2026) | Diversity metrics reported alongside score; islands and novelty rejection as policies |
| Results indistinguishable from seed noise | Paired seeds, sigma-based sample sizes, pre-registered effect sizes, 10 seeds for headlines, HC for sparse improvement |
| A bug late in a long run | Record/replay cache; config digest; checkpoint and resume; smoke tests with known answers before every real run; 1,000-generation dry run with stubs |
| HLE build problems on Apple Silicon | pyspiel Hanabi fallback from the arm64 wheel |
| Agents game the organizational mechanism when told the rule (Rahwan-lab 2026) | Do not describe the credit or selection rule in prompts; disclosure is a queue item |

## 19. Coordination with Diego's sandbox

Reviewed against his eight commits of 2026-10-05 (predator-prey with SmolLM2 LoRA agents; corpus notes; demo audit; direct LLM actions and persistent tactics; the Flag Game with GPT-4o agents and a verified colour corpus). None of his results files claims learning or a corpus benefit; the one real finding is negative and useful: a model that cannot do the task cannot learn to use shared notes, and adding peer text to imitation data does not teach a decision that depends on it.

**Split of the question.** Diego tests whether a *learned controller* improves information routing among frozen models, on one-shot information pooling (Flag Game) and ecological survival. We test whether *capable* models improve collectively through artifact exchange with causal credit over long horizons. His delivery credit is, in his own words, not causal credit for improving the recipient; our paired-seed student-improvement credit is the causal version. That is the cleanest complementarity, and the one to say out loud in the one-pager.

**Shared evaluation grammar** (adopt as-is so the two sandboxes read as two tests of one thesis):

- Access ablation with frozen artifacts: shared, lineage-private, none, same seeds. The only accepted evidence of a corpus or teaching benefit in either sandbox.
- Paired seeds and the population run as the unit of uncertainty; per-trial paired bootstrap.
- A scope line as the first field of every results JSON, and the `docs/<experiment>.md` plus `docs/results/<experiment>-validation.json` convention.
- One cost-guarded provider module shared between branches: content-hash cache, reserve-before-send, cumulative dollar cap, no automatic paid retry, model fingerprint recorded per call. Ours adds replay modes and the per-tag ledger; his adds the reservation logic. Merge rather than maintain two.
- Common corpus verbs: `deposit(item, evidence)` with mechanical, ownership-bound validation; `retrieve(policy)` that writes a touch log; prose stored but never certified.

**Do not duplicate.** Free-text tactics with heuristic retrieval (he has it, no positive result yet); delivery-based teaching credit; anything that updates weights; the 135M substrate.

**Borrow.** Frozen-access ablation; demo-audit-style pre-flight checks (does the signal exist in the data, does the policy respond to it at all) before any paid run; the deploy-loop patterns (job list, digests, atomic checkpoint, PPO rollback on failed trial write, `plan` mode with a call upper bound); the resume-equals-uninterrupted test.

**A joint result worth planning for.** "Verification gates adoption" demonstrated in both sandboxes: his verified colour deposits versus prose notes, our verified artifacts versus unverified ones. Same claim, two substrates, one figure each.

## Sources

Fact sheets under `../research_notes/Sandbox game selection lit review/` and the three mechanism extractions in this session: HLE `rl_env.py`, `hanabi_game.cc`; JaxMARL README and NeurIPS 2024 paper; hanab.live `example_game_with_comments.jsonc`; OpenSpiel `policy.py`, `exploitability.py`, `cfr.py`, wheels workflow; ShinkaEvolve `shinka/database/parents.py`, `islands.py`, `wrap_eval.py`; OpenEvolve `openevolve/api.py`, `database.py`, `population.py`; HGM `hgm.py`, `tree.py`; DGM `DGM_outer.py`; Economy of Minds §2 and App. A, C; Colas et al. Methods and App. C; Vallinder & Hughes §2; Henrich 2004 eqs. 2 to 4; Canaan et al. AIIDE 2020 Table 1; Bard et al. 2020 Table 1; OBL Table 1; STAT 236 lecture notes (multiple testing, phase transition, HC, IF-PCA, network models).
