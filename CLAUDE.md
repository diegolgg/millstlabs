# Mill Street Labs, Sandbox 1 (branch `enrico`) — working context

Read this first in every session, then `files/sandbox-architecture.md` (build spec), `docs/sandbox1-status.md`
(what is built and validated), `files/lit-map.md`, and `reports/Sandbox game selection lit review.md`.
Diego's sandbox is on `main` (package `millstlabs`); ours is `src/culture/`. Never edit `src/millstlabs`.

## How to work with Enrico (learned 2026-10-05/06)

- **Structure every proposal as: what it is for (which goal or edge item), what it is at a high level, why it is
  worth studying, then the technical details.** He said this explicitly. Do not lead with mechanism.
- **Define terms before using them.** He does not know Hanabi, and he will not guess what "cross-play",
  "artifact", "touch log" or "corpus" mean. One sentence per term, in a vocabulary block, before the argument.
- **Explain games and settings from scratch**: goal, players, cooperative or competitive, what a turn is, a worked
  example. He asked for this after a too-brief description.
- **Tie everything to the startup's edge, not just to "transfer helps".** He rejected "preseed only needs evidence
  of transfer" because many systems transfer knowledge; the edge is causal credit over a teaching DAG, a formal
  model that predicts when organization beats scale, and robustness to persuasive-but-wrong ideas, done rigorously.
- **Rigor is part of the edge.** Think in estimands, designs, decision rules, power, pre-registration, FDR across
  the hypothesis program. Say which test, which assumptions, and whether the setting fits before naming a theory
  (he caught an over-reach on rare/weak theory). "Vague and imprecise" is the failure mode he named.
- **Iterate ideas fast, but with acceptance rules.** He wants many hypotheses tried, each small, with a stated rule
  for when one idea beats another. Breadth without acceptance rules is as bad as one big experiment.
- **Discussion first, then files.** He interrupts when I jump to deliverables. Align in chat, then write.
- **Plain summaries of Diego's work**, direct and simple, with overlap vs orthogonality and what to emulate.
  Steelman Diego's position before disagreeing; he is usually right about counterfactuals and compute.
- **Candid best case / most likely / worst case** when he asks whether something is worth doing.
- **Cost**: he does not want paid API calls for anything unproven. Debug on stubs, then local open-weight
  models, then paid only for a measured, capped run. No Claude in the agent swarm; open weights are the default.
- **Model for the planning sessions**: Fable for strategy, review and synthesis; a cheaper model for builds.
- **Plan at agent speed, not human speed.** Enrico has had to repeat this. Builds, evaluations and tests are done
  by LLM agents he or I spawn, in parallel, in hours. Never propose week-based timelines or "one experiment at a
  time" sequencing that assumes a human implementer. Propose what to run tonight. Spawn builds on a cheaper
  model (Opus/Sonnet) from this session rather than asking him to hand prompts around.
- **Time matters: preseed is imminent** (his roommate wants to raise soon). Prioritize what produces a defensible,
  rigorous figure fastest; defer anything that only pays off post-funding.
- **Machine:** Apple M4 Pro, 48 GB unified memory, 2 TB disk (about 1.7 TB free). Local open-weight MoE models
  up to ~30 GB quantized are fine; dense 27B+ is too slow for 30-seed designs.
- **Reuse before novelty.** If an existing tool or estimator fits (e.g. a published counterfactual-replay or Shapley
  credit method), adopt it and cite it; judge fit explicitly (right setting, not overkill, not overfit). Propose
  genuinely novel methods only where nothing fits, and say why nothing fits. The goal is a trustworthy result,
  not a novel one.
- **Fill gaps from the textbooks he uploads** (`files/wager__causal-inf.pdf`, `files/sutton-barto__rl.pdf`): causal
  inference and RL are the gaps he named; frame designs in potential outcomes, interference, off-policy
  evaluation, bandits and credit assignment when they fit, with page-cited definitions.
- **If unsure, ask; never assume.** Enrico's rule. This includes course contents and labels: STAT 212 is
  stochastic processes (he CA'd it); do not tag ideas with course numbers or paper names unless certain.
  Say "I'm not sure which course covers X" rather than guess.
- **Novelty claims need a stated basis.** "No published work found in the 2026-10-05 searches (six research
  notes)" is allowed; "never been done" is not. Before a novelty claim goes into any document, run a targeted
  search and cite the three nearest works. Do not inherit novelty claims from external.tex or lit-map.md.
- **Idea triage he has given (2026-10-06):** credit estimators vs planted truth, breakdown point of verification,
  accumulation parameters, SPRT verification, winner's-curse correction, adaptive evaluation allocation are
  interesting; mean-field theory and bandit teacher choice are not useful now; spectral diversity is marginal.
  He wants experiment-first novel ideas and new ways to be rigorous, drawing on high-dim stats, grad-level
  stats, some TCS and pure math as a way of thinking, not as a list of tools to name-drop.
- Use his background: STAT 212 (stochastic processes), 221 (adaptive subsampling, chi-square staleness),
  236 (HC, phase transitions, IF-PCA, DCMM), DP research (influence of one record, inclusion indicators,
  leave-one-out reference), robust statistics, RMT. He wants these as formulations, not decorations.

## Repo facts

- `.venv/bin/python -m pytest -q` (about 2 min). `python -m culture.run --config configs/<x>.yaml --out runs/<x>`.
- Engine: HLE vendored at `third_party/hanabi-learning-environment` (commit 54e7959), built from source
  (`CMAKE_POLICY_VERSION_MINIMUM=3.5`). Anchors: Piers 16.99, IGGI 15.86, Flawed 0 over 1,000 games.
- Backends: stub (debugging, CI) and `openai_compat` targeting the local MLX server (seed in the cache key). No paid
  backend is wired; none should be without an explicit, capped instruction.
- Fix round 1 (2026-10-06/07, commits 9fb78c6..737194b): the eight review defects are fixed with regression tests;
  disjoint feedback/verification/evaluation deals; null calibration (K=40); innovation base rate; C1 single-student
  driver with exact Shapley oracle; C1 pilot on MLX (5 seeds x 2 strata); B1 SPRT study; D1 pilot on the null stub.
  Suite: 205 passed, 1 skipped (opt-in live MLX test). Read docs/sandbox1-status.md "Fix round 1" and
  docs/results/c1-pilot.json, b1-sprt.json, d1-pilot.json before planning the next run.
- Known open sandbox gap: a bot can monkeypatch a shared stdlib class through an alias (recorded, not fixed).
- C1 pilot findings that change design: a rejected (verified-off) Flawed message still lowers the student by 3 to 5
  points because its prose and evidence sit in the revision prompt (verification gates code adoption, not ideas); the
  "self" placebo is not a null (effects of -3 to +2); only full-replay estimators meet RMSE <= 2 at k=3 (ridge 2.6/3.9,
  LOO 3.2/4.4) while rank order is right (Spearman 1.0); R for 80% power at 1 point = 29 (verified), 8 (unverified).
- Local model (2026-10-06): Qwen3.6-35B-A3B. **Serve with MLX** (`.venv/bin/python -m mlx_lm.server --model
  mlx-community/Qwen3.6-35B-A3B-4bit --host 127.0.0.1 --port 8080 --max-tokens 8192 --chat-template-args
  '{"enable_thinking": false}'`): seeded temperature-0 requests are bit-reproducible, about 65 tok/s, runnable code.
  Ollama 0.40 runs this model on its own MLX runner that ignores seed/temperature/num_gpu; do not use it for replay.
- Memory notes live in `~/.claude/projects/-Users-enricoyao-bate-Desktop-prep--startup-millstlabs/memory/`.
