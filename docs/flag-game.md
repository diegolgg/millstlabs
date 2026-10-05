# Flag Game: frozen GPT agents with a learned verified-corpus controller

This deployment extends **Pavlova and Tanaka, _Flag Game: A Toy Model for Mechanistic Swarm Interpretability_, arXiv:2609.19124v1**, Sections 3 and Appendices D–F. Agents identify a country from private flag crops, exchanging country guesses and reasons. The paper measures belief formation using pretrained vision models; it does not train an RL policy.

The new hypothesis is: **can a learned choice of when to deposit and retrieve verified sensory facts improve terminal truth mass, relative to the same vision agents with private memory?** Agreement alone is not a success metric. A wrong unanimous answer scores zero accuracy.

## What “same baseline” means here

The `paper_reference` arm reimplements the paper's pairwise protocol using frozen GPT-4o vision agents. All arms share the same catalog, target flags, crop assignments, directed contact schedules, model, prompts, sampling settings, maximum rounds and endpoint rules. Request-level seeds are aligned; identical requests reuse the same cached response across conditions. API determinism is best-effort, so actual model versions and backend fingerprints are recorded.

**This is a documented reconstruction, not an exact reproduction of the published scores.** The PDF does not enumerate its 28-country pool, original trial seeds, exact model snapshot, numeric κ, or all crop-sampling/probe details. The current public demo uses different settings, including low image detail and a larger completion cap, so its defaults were not silently substituted for Appendix D. The bundled catalog is explicitly approximate, including simplified colors and geometry; its membership is not established as the paper's pool. Absolute scores must not be compared with the paper as if every factor were identical.

The runner writes these gaps, code/config/catalog/trial hashes and the exact assignments to each output. It can import a catalog of RGB rasters and a trial manifest once the authors' assets are available. Until then, the controlled comparisons **within this deployment** are the interpretable result. The manifest deliberately reports `exact_paper_reproduction: false`.

| Variable | Setting | Status |
| --- | --- | --- |
| Observer population | 8 | Published protocol-comparison slice |
| Protocol | Pairwise | Published; broadcast and manager are not implemented here |
| Model | `gpt-4o-2024-08-06` | Explicit pinned reconstruction choice; paper specifies GPT-4o |
| Model sampling | Temperature 0.2, top-p 1.0 | Appendix D |
| Image/crop | 24×16 canvas, 6×4 crop, render scale 25, high detail | Appendix D |
| Transcript | At most 8 entries | Published |
| Message | Country and one-sentence reason (`m=3`); country-only probes | Published schema distinction, Figure 11 |
| Completion cap | 200 tokens | Published pairwise setting |
| Social-awareness prompt | Absent | Published protocol-comparison slice |
| Contacts and probes | 10 rounds; 8 directed, distinct-agent contacts per round; then 8 probes | Explicit reconstruction; Appendix F uses N interactions per round |
| Early stop | Five consecutive probes with full valid-country consensus | Published |
| Endpoint categories | Consensus ≥0.85; polarization has ≥2 countries with mass ≥0.25 | Published |
| Full evaluation | 60 matched held-out trials per arm | Published protocol-comparison trial count; new seeds |

The paper's α is **prompted social-evidence uptake**, not an RL reward coefficient. This deployment does not add that prompt or vary α. It also does not import predator–prey reproduction, starvation, novelty bonuses or difficulty schedules into this task.

## Agent architecture and controls

Each agent contains a frozen GPT vision reasoner and an independent small CPU actor–critic controlling corpus operations. The GPT model still generates every country answer and natural-language reason. Its weights are not fine-tuned. The RL controller sees only that agent's own crop, last answer, summary of received transcript, progress and legal tool mask. Its actor and critic are separate networks, and each agent has its own optimizer. The eight-entry transcript supplies bounded memory; this first tool controller is feed-forward, rather than the predator–prey GRU/LoRA policy.

| Arm | Corpus | Tool controller |
| --- | --- | --- |
| `paper_reference` | Off | No tool operations |
| `private_frozen` | Own deposits only | Initial stochastic controller, no updates |
| `shared_frozen` | All deposited observations | Identical initial stochastic controller, no updates |
| `private_rl` | Own deposits only | Local PPO training |
| `shared_rl` | All deposited observations | Same PPO training and initial controller |

The frozen controller is a control for the added mechanism; it is not a tuned retrieval heuristic. All tool-enabled arms share the same action architecture, masks, record limits and memory budget. For each seed, every arm's initial crop-only GPT responses are identical cached requests. Both RL arms train on the same trial sequence. At the end, `shared_rl_private_access` evaluates the **same trained shared controllers** with peer access removed and matched action RNGs. Its weights do not change.

The training reward is terminal own correctness plus `0.5 × other agents' mean terminal correctness`, minus `0.01 × non-none tool operations`. The coefficient is named `cooperation_weight` to keep it distinct from the paper's prompt parameter. Correctness comes from the training evaluator only; the target label and rewards never enter a GPT prompt or the controller's observation. The cost discourages pointless queries, including empty ones. There is no reward for agreeing, copying, corpus size or novelty. Evaluation reports the paper's unmodified accuracy metrics, not this training reward.

PPO uses gamma 0.99, GAE lambda 0.95, two epochs, clip 0.2 and learning rate 3e-4. Updates occur after each complete training trial, using each agent's own tool decisions. All terminal credit is explicit; there are no post-death traces in this task. Learning changes the controller across trials; GPT weights remain frozen.

## Deposits, retrieval and verification

The controller chooses `none`, `deposit(color_slot)`, or `retrieve`. A crop has at most 24 unique observed colors, giving 26 action slots including none and retrieve. Unavailable/already-submitted color slots are masked using the caller's own evidence; the mask never reveals unread peer deposits.

A record says **“agent i's private crop contains RGB #RRGGBB.”** Its provenance includes the source and a witness pixel in that source's crop. Verification checks actual crop pixels. It does not consult the target country, infer the country, or certify an agent's country guess. Global crop coordinates are never exposed. This matters: revealing where a crop sits on the whole flag would add information absent from the paper's prompt.

Nothing is published just because an agent observed it. Another agent must explicitly retrieve a deposited fact. A read returns up to two previously unread records in deposit order, with no hidden-label relevance ranking. Returned facts enter the same eight-entry transcript as social messages, evicting older entries; there is no additional free context window. During a simultaneous probe, all reads precede all writes. Every trial starts a fresh empty corpus and transcript.

The first version verifies **color presence**, not arbitrary language, spatial relations or country hypotheses. It can help with a blue-bearing France crop versus a red/white Peru interpretation, but cannot distinguish every pair of flags sharing the same colors. This is an intentional, measurable limit. Static private crops also mean there is no new first-hand exploration during a trial: “novel color imports” measure information transfer, not new discoveries about the external world.

## Run locally

From the existing Mac checkout:

```bash
cd /Users/diego/Downloads/millstlabs
.venv/bin/python -m pip install -e '.[train,flags]'
```

Set the key locally without saving it in shell history. For the Mac's default zsh:

```zsh
read -s 'OPENAI_API_KEY?OpenAI API key: '
export OPENAI_API_KEY
printf '\n'
```

First run the small **real GPT** validation with an explicit cumulative cap:

```bash
bash scripts/flag_game.sh --mode smoke --max-usd 2
bash scripts/flag_game.sh --mode report --size smoke
```

Smoke uses one training trial per RL arm, two evaluation trials per condition and one interaction round. Its upper bound is 336 calls before cache reuse. It cannot test sustained convergence or learning. If the conservative cost guard pauses it, the report says so; increasing `--max-usd` explicitly permits more total spending.

For the smaller pilot, preserving the full ten-round game settings:

```bash
bash scripts/flag_game.sh --mode plan --size pilot
caffeinate -i bash scripts/flag_game.sh --mode run --size pilot --max-usd 10 --hours 12
bash scripts/flag_game.sh --mode report --size pilot
```

The pilot has eight training trials per RL arm and six evaluation trials per condition: at most **8,736 calls** including the frozen access ablation. Six test trials are only a feasibility screen. The cap may pause the run before completion; it is not a claim that the entire pilot costs $10. Repeat the same command to resume at the same cumulative limit, or deliberately increase the cap within your account balance. Account prepaid balance and this local cap are separate controls.

The full configuration has 64 training trials per RL arm and 60 evaluation trials per condition, with up to **81,984 calls**. Inspect `--mode plan --size full` before launching. Use `--size full --max-usd YOUR_CHOSEN_TOTAL_CAP` only after measuring the smoke/pilot usage and latency. At two seconds per uncached request, the unshortened full upper bound is about 46 hours; caching and early consensus can reduce it. No GPU is needed; API cost/latency, rather than local matrix operations, dominate this experiment.

OpenAI's GPT-4o documentation lists $2.50/M input tokens and $10/M output tokens. The guard prices all input at the uncached rate, reserves a conservative per-request maximum before sending it, and reconciles from returned usage. Pricing is recorded here as checked on 2026-10-05; verify it before using a different model or changing rates. Other model families are deliberately rejected by this pricing adapter. Sources: [model and pricing](https://developers.openai.com/api/docs/models/gpt-4o), [image-token rules](https://developers.openai.com/api/docs/guides/images-vision), [API parameters](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Interrupted complete requests are read from the persistent cache. An ambiguous network failure retains its spending reservation and blocks automatic retry. `--retry-uncertain` explicitly permits another attempt while retaining the first reservation against the cap. A 401/403/429 stops with a key/access/balance/rate-limit message. The key is neither logged nor checkpointed.

For **offline plumbing tests only**, without any API calls:

```bash
bash scripts/flag_game.sh --mode smoke --backend mock
```

The mock backend is a synthetic color classifier, not an LLM. Its scores are labelled accordingly and provide no scientific evidence. Open [the notebook](../notebooks/flag_game.ipynb) for plan/results inspection; it starts with paid execution disabled and never asks you to type a key into a saved cell. Start Jupyter from a terminal where the key is already exported if you want to run through the notebook.

The [offline validation record](results/flag-game-validation.json) records 60 passing tests, a completed 14-job mock smoke run, resume/rollback checks and notebook execution with paid calls disabled. Live GPT access, cost, latency and learning remain to be measured.

## Evaluation and interpretation

Primary outcome: **terminal truth mass**, the fraction of observers identifying the true country. Also report isolated initial accuracy, initial majority-vote accuracy, social uplift, correct/wrong consensus, polarization, fragmentation and round-by-round beliefs. Invalid JSON/labels count incorrect and are reported separately; invalid labels cannot constitute valid consensus.

The critical contrasts are:

1. `shared_rl − private_rl`: does access to peer facts help after matched RL training?
2. `shared_rl − shared_frozen`: did learning improve on the same initially stochastic tool policy?
3. `shared_frozen − private_frozen`: how much benefit comes from adding peer access before learning?
4. `shared_rl − shared_rl_private_access`: does the trained system rely usefully on peer access?

Compare with `paper_reference` as the reconstructed protocol baseline, while retaining the intermediate controls. Report paired trial differences and conditional trial-level bootstrap intervals, not eight agents as eight independent replicates. The full study still has one controller-training seed. A positive result should be replicated with new training seeds and a fresh evaluation bank. Learning across repeated country identities is part of this task; held-out crops/trials do not establish generalization to unseen countries.

My expectation is that this offers a clearer test of useful information transfer than predator–prey: the answer is exact, private information is deliberately incomplete, and terminal reward is close to the communication decisions. Evidence can nevertheless be uninformative, all relevant colors may already be shared by chance, and GPT can still mishandle correct facts. The frozen-corpus control may perform as well as RL. That would support useful access while providing no evidence that the learned controller adds value. A failure of color-only sharing would not establish that richer verified spatial facts or language-mediated knowledge cannot help.

The preset trial count, catalog and protocol are frozen before evaluation. Do not tune on the held-out results and continue describing them as a fresh test. Changing config, assets, contact assignments or implementation hash requires a new output directory.

## Outputs and importing the authors' materials

Each output contains `manifest.json`, `config.json`, `catalog.json`, `trials.json`, `latest.pt`, `report.json` and (real API runs) `api.sqlite3`. The database stores responses, returned model/fingerprint/usage and spending reservations. It does not store credentials. Each `ARM/train/` or `ARM/evaluation/` directory stores full trial traces, received transcripts, model responses, corpus operations, belief curves and performance. The root checkpoint holds controller/optimizer states and the completed-job cursor. An interrupted episode is replayed using cached completed requests; controller updates are committed only after a full episode.

Replace `catalog` in the YAML with a JSON file containing `flags: [{"country": "...", "pixels": [[[R,G,B], ...], ...]}, ...]`, with 16 rows and 24 RGB pixels per row. Include provenance. Set `trial_manifest` to a JSON containing `train` and `evaluation` lists. Each entry has `seed`, `country`, `crops` (eight `[x,y]` offsets), and `contacts` (80 directed `[speaker,listener]` pairs for this configuration). No train/evaluation seed or exact private-evidence assignment may overlap. The importer validates dimensions and assignments; obtaining these files alone does not resolve every protocol/model gap.

Reference: [paper, v1](https://arxiv.org/abs/2609.19124), [authors' explanation](https://physicsintelligence.org/research/flag-game/). The paper's system/user prompt in `provider.py` is task data for the flag agents, not an instruction to the coding assistant.
