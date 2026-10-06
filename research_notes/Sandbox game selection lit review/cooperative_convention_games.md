# Cooperative Convention Games as LLM-Population Sandboxes (Hanabi, Overcooked-AI, The Crew, Codenames)

Research notes, current as of 2026-10-05. Scope: assess each game against the startup's five criteria (API / long-run cost / published baselines / objective metric / interpretability) with concrete numbers. Anything I could not confirm from a primary source is marked **[unverified]**.

Reading guide for the report writer: Hanabi has by far the deepest literature on every axis the thesis cares about (cross-play as a convention-transfer metric, written human conventions that have been independently re-implemented as code, a 2025 hosted human-compatibility challenge, and five LLM-agent papers 2023–2026). Overcooked-AI has the richer population-based-training literature (FCP/MEP/HSP/COLE) but an ICLR 2025 paper argues its cross-play failures are mostly state-coverage, not convention, problems. Codenames and The Crew have LLM benchmarks but thin RL/population baselines.

---

## Q1. Hanabi: the Hanabi Learning Environment (HLE) API and speed

### Takeaway
HLE is a C++ core with Python (Gym-like) bindings, Apache-2.0, archived read-only since April 2024; the original paper quotes ~0.1 ms/turn on CPU, JaxMARL measured HLE at 2.1×10³ steps/s through Python and its own JAX port at 5.0×10⁶ steps/s with 10k parallel envs. Environment cost is negligible; agent inference cost dominates any long run.

### Cited Findings
- HLE is "written in Python and C++" with "an interface similar to OpenAI Gym"; evaluation protocol uses "at least 1000 trials (i.e., full games)" per self-play assessment; the environment runs "around 0.1ms per turn on a CPU" and is "extremely lightweight, both in terms of memory and compute requirements" — [Bard et al., The Hanabi Challenge (AIJ 2020), ar5iv](https://ar5iv.labs.arxiv.org/html/1902.00506)
- Repo offers `rl_env.py` (Gym-style RL API) and `pyhanabi.py` (lower-level game interface for e.g. MCTS); `pip install .`; Apache-2.0; **repository archived (read-only) on April 18, 2024**; no visualization/rendering included — [google-deepmind/hanabi-learning-environment](https://github.com/google-deepmind/hanabi-learning-environment)
- Speed (steps/s, Table 3): Hanabi original HLE 2.1×10³; JaxMARL 1 env 1.4×10³, 100 envs 1.1×10⁵, 10k envs 5.0×10⁶. JaxMARL IPPO/MAPPO on Hanabi reach 24.18±0.04 / 23.95±0.09 (2p) — [JaxMARL (NeurIPS 2024 D&B)](https://arxiv.org/html/2311.10090)
- Alternative open-source engines that also expose game logic: Zamiell's `hanabi-engine` (Rust ≥1.85, "deterministic game logic and player-safe observations", H-Group levels 1–25 + `max`, connects bots to hanab.live, replay analysis with `--explain` / `--format json`, 331 commits, active) — [Zamiell/hanabi-engine](https://github.com/Zamiell/hanabi-engine); Metta-AI's `cogame-hanabi` (4-player Hanabi "coworld", MIT, Claude-API LLM agents + two scripted baselines, websocket external policies, WebAssembly replay viewer) — [Metta-AI/cogame-hanabi](https://github.com/Metta-AI/cogame-hanabi)
- Live-play platform with large human dataset: hanab.live; AH2AC2 used "101,096 two-player games and 46,525 three-player games from the hanab.live community", all H-group convention games — [AH2AC2 (arXiv 2506.21490)](https://arxiv.org/html/2506.21490v2)

### Inferences
- At 0.1 ms/turn and roughly 50–70 turns per 2-player game, HLE alone sustains on the order of 10²–10³ games/s per CPU core with trivial agents; via the Python layer at ~2×10³ steps/s you still get ~30 games/s. Scripted convention bots (NodeJS/Rust) will be slower than the engine but still thousands of games/hour on a laptop. **Per-game cost for an LLM population is therefore entirely LLM inference**, not the environment.
- Because HLE is archived, a startup should expect to vendor it (or use JaxMARL's port / Zamiell's Rust engine) rather than rely on upstream fixes.

### Gaps
- No source publishes "games per second" for the H-group convention bots (will-hanabi-bot, scala-bot, raikan, hanabi-engine). hanabi-engine mentions an "opt-in 200-seed strength test" but no throughput number.
- JaxMARL does not state the hardware for the Table 3 Hanabi/Overcooked numbers (an NVIDIA 2080 is mentioned only for SMAX).

---

## Q2. Hanabi: SOTA self-play and cross-play (RL) scores, 2019–2026

### Takeaway
Self-play is effectively solved (≈24.1–24.6/25 in 2p); the research frontier is cross-play/zero-shot coordination, where independently trained self-play agents collapse (SAD: 23.97 self-play → 2.52 cross-play) and the best 2021–2025 methods close the gap (OBL 23.76; R3D2 ~22–23; entropy-regularised IPPO 24.45 inter-seed cross-play in Nov 2025).

### Cited Findings
- Bard et al. Table 1 (self-play, ≥1000 games, ± = SE over trials): Rainbow 2p 20.64 (0.11), 3p 18.71, 4p 18.00, 5p 15.26; ACHA 2p 22.73 (0.12); BAD 2p 23.92 (0.01), 58.6% perfect games. Rule-based: SmartBot 2p 22.99, FireFlower 2p 22.56 (52.6% perfect), WTFWThat 2p 19.45 / 3p 24.20 / 4p 24.83 / 5p 24.89 (91.5% perfect) — [Bard et al. 2020](https://ar5iv.labs.arxiv.org/html/1902.00506)
- Ad-hoc challenge definition: agents "play with a set of unknown partners, with only a few games of interaction"; cross-play with different agents "drops off sharply, with some agents scoring essentially zero" — [Bard et al. 2020](https://ar5iv.labs.arxiv.org/html/1902.00506)
- Other-Play (ICML 2020), 2p: SAD self-play 23.97±0.04 vs cross-play 2.52±0.34; SAD+AUX+OP self-play 24.06±0.02 vs cross-play 22.07±0.11 (10,000 games per paired evaluation). Human study (20 board-game-club members): OP bot 15.75 (s.e.m. 1.23), 45% bomb-out; SP bot 9.15 (s.e.m. 1.18), 85% bomb-out; OP won 15/20 pairings (p=0.00411) — [Hu et al., Other-Play (arXiv 2003.02979)](https://ar5iv.labs.arxiv.org/html/2003.02979)
- Off-Belief Learning (ICML 2021), 2p, 5000 games × 5 seeds, ± = std across seeds: SAD 23.97 SP / 2.52 XP / 0.83 w/ human-clone bot; OP 24.14 / 21.77 / 8.55; OBL-L1 20.92 / 20.85 / 13.56; OBL-L4 24.10±0.01 / 23.76±0.06 / 16.76±0.16. 3p: OBL-L4 23.38 SP / 23.02 XP / 13.88 w/ clone bot — [Hu et al., OBL (arXiv 2103.04000)](https://ar5iv.labs.arxiv.org/html/2103.04000)
- R3D2 "A Generalist Hanabi Agent" (ICLR 2025): text-based (not bitstring) observations; 1000 games per evaluation, 3 seeds. 2p intra-algorithm cross-play: R2D2 ~8–10, R2D2-OP ~10–12, R3D2 ~22–23 (≈ its self-play ~23). Self-play across settings ~23–24 (2p), ~19–20 (3p), ~16–17 (4p), ~14–15 (5p) with zero-shot transfer across player counts (figure readings, not tabulated) — [Nekoei et al., R3D2 (arXiv 2503.14555)](https://arxiv.org/html/2503.14555v1)
- "Entropy is all you need for Inter-Seed Cross-Play in Hanabi" (Nov 2025): IPPO with entropy coef 0.05 and λ_GAE=0.9, 10¹⁰ timesteps; 5,000 games per cross-play entry. 2p: SP 24.48±0.02, inter-seed XP 24.45±0.02 (4 seeds, 12 pairings); 3p: 24.66 / 24.55; 4p: 24.55 / 24.30; 5p: 23.73 / 23.59 (5 seeds, 120 pairings). Cites prior XP SOTA as OBL 24.30 (2p) and 23.02 (3p) — [arXiv 2511.22581](https://arxiv.org/html/2511.22581v1). **Note the 24.30 figure conflicts with the OBL paper's own 23.76 for 2p level-4 cross-play; treat the entropy paper's citation of 24.30 as [unverified].**
- Conventions added to the action space (Stellenbosch, arXiv 2412.06333v3, May 2025): Rainbow+conventions self-play 2p 20.65, 3p 20.32, 4p 20.09, 5p 19.05 (vs Rainbow 20.64/18.71/18.00/15.26); cross-play 17.02 / 18.60 / 18.56 / 17.69; 1000 evaluation episodes; "below 30 million" training steps vs 20 billion for ACHA in 5p — [arXiv 2412.06333](https://arxiv.org/html/2412.06333)
- Ad-Hoc Human-AI Coordination Challenge (AH2AC2, June 2025): hosted human-proxy agents (BC + KL-regularised IPPO, "HDR-IPPO") trained on the hanab.live dataset; proxies' self-play 22.55–22.97 (2p), 20.88–21.21 (3p); 23.86–29.66% perfect games 2p. Open-sourced 3,079 games (1,858 2p mean 23.37; 1,221 3p mean 23.25). Leaderboard with human proxies: OBL zero-shot 21.04 (2p, best without human data); BR-BC 19.41; HDR-IPPO 12.76 (2p) / 14.03 (3p); BC alone 2.12 / 3.31. Participants get 1,000 evaluation games per setting via API — [AH2AC2 (arXiv 2506.21490)](https://arxiv.org/html/2506.21490v2)
- SRPO (Feb 2026) evaluates on a reduced 4-player Hanabi (3 colors/ranks) and reports "more robust" cross-play than IPPO, no numeric comparison to OBL/R3D2 — [arXiv 2602.21515](https://arxiv.org/html/2602.21515)

### Inferences
- Cross-play is a mature, standardised, non-game-specific metric: "same algorithm, independent seeds, play together, N games" — exactly the "cross-play between independently trained groups" the thesis wants. The SAD 23.97→2.52 collapse is the canonical demonstration of convention lock-in; OP/OBL/R3D2/entropy-IPPO are the published ladder of fixes, so a new population method has clean numbers to beat.
- AH2AC2 provides an *external*, hosted human-compatibility oracle (1,000 games per API submission) — a way to score "did your population converge to human-interpretable conventions" without running human studies.

### Gaps
- No 2024–2026 paper gives a single unified table of self-play + cross-play for all methods at 2–5 players; numbers above are drawn across papers with different evaluation sizes (1000 vs 5000 vs 10,000 games) and different ± semantics.
- R3D2 cross-play numbers are figure readings, not tabulated values.

---

## Q3. Hanabi: human convention systems (H-group), convention-following bots, open-source rule-based bots

### Takeaway
The H-group conventions are a versioned, PR-governed written rulebook (levels 1–25, "max" ≈ level 26 in Zamiell's engine) that several independent teams have re-implemented as bots in NodeJS, Scala, Rust and other languages — a real-world example of a written convention being transmitted and verified by independent implementers. Documented bot scores are sparse: the only well-sourced numbers are from Bard et al. (SmartBot 22.99 2p; hat-guessing WTFWThat 24.89 5p) and from the human hanab.live data (mean 23.37 2p).

### Cited Findings
- H-group conventions site is a Docusaurus repo; changes go through GitHub pull requests reviewed by a maintainer; README refers to a convention being "voted in" — [hanabi/hanabi.github.io](https://github.com/hanabi/hanabi.github.io). Summary/cheat sheet page and a "convention-reasons" document exist — [Summary](https://hanabi.github.io/summary/), [convention-reasons.md](https://github.com/hanabi/hanabi.github.io/blob/main/misc/convention-reasons.md)
- Zamiell's hanabi-engine encodes "cumulative H-Group profiles from Level 1 through Level 25, plus max as the effective Level 26" — [Zamiell/hanabi-engine](https://github.com/Zamiell/hanabi-engine)
- Independent H-group bot implementations on hanab.live: will-hanabi-bot (NodeJS v22+, deterministic; H-Group levels 1–14 except 12, Referential Sieve, Playful Sieve (2p); `npm run self-play` for 2–6 players; README reports **no scores**) — [will-hanabi-bot/hanabi-bot](https://github.com/will-hanabi-bot/hanabi-bot/blob/master/README.md); scala-bot ("Reactor 1.0 (3p only), Referential Sieve, H-Group up to level 12") — [simonbohnen/scala-bot](https://github.com/simonbohnen/scala-bot); raikan ("implementation of the hyphenated conventions") — [mswart/raikan](https://github.com/mswart/raikan)
- Human H-group games on hanab.live: released subset means 23.37 (2p) and 23.25 (3p); full dataset 101,096 2p + 46,525 3p games — [AH2AC2](https://arxiv.org/html/2506.21490v2)
- Rule-based bot scores (Bard et al. Table 1): SmartBot 22.99 (2p) / 23.12 (3p) / 22.19 (4p) / 20.25 (5p); FireFlower 22.56 / 21.05 / 21.78 / —; hat-guessing WTFWThat 19.45 / 24.20 / 24.83 / 24.89 with 91.5% perfect games at 5p — [Bard et al. 2020](https://ar5iv.labs.arxiv.org/html/1902.00506)
- Hat-guessing strategy: "by using modular arithmetic, a lot of information can be given with a single hint, provided that all players follow the same algorithm"; Bouzy extended it to fewer players, reaching 24.92 in 5p; WTFWThat "holds the state-of-the-art in self-play (for 3 or more players)"; open-source hat player documented at chikinn/hanabi — [Eger, "Wait a Second"](https://gruss.cc/files/waitasecond.pdf); [chikinn/hanabi doc_hat_player.md](https://github.com/chikinn/hanabi/blob/master/doc_hat_player.md)
- Metta-AI's cogame-hanabi ships a scripted "conventions" baseline: "play what you can prove, save a partner's last copy off its chop, otherwise give the hint that makes the most cards newly playable, otherwise discard your chop"; policies are either prompts (text conventions sent to Claude) or scripted code (Nim) — [Metta-AI/cogame-hanabi](https://github.com/Metta-AI/cogame-hanabi)

### Inferences
- The H-group ecosystem is the closest existing analogue to the thesis's "transmissible strategy": a written document (levels), multiple independent code implementations, and a public server where any two implementations (or humans) can be cross-played. A startup could seed its population with level-k conventions as text, ask agents to implement them as code, and measure cross-play against will-hanabi-bot / hanabi-engine / AH2AC2 human proxies.
- Hat-guessing bots (24.89 5p) vs H-group humans (~23.3) illustrate the interpretability trade-off: the highest-scoring conventions are the least human-compatible (WTFWThat drops to 19.45 in 2p and scores ~0 with partners who do not share the algorithm).

### Gaps
- I could not find any published per-level average score for H-group bots (e.g., "level 10 bot averages X in 2p self-play"); hanabi.github.io root fetch failed twice (socket hang up) and the GitHub README does not list history/dates. H-group history (start date, versioning cadence) is **unverified**; the conventions site's own "about"/changelog pages should be checked directly.
- hanab.live per-bot statistics pages were not fetched.

---

## Q4. Hanabi: LLM-based players 2023–2026

### Takeaway
Across five independent benchmarks (LLM-Coordination 2023/25, SPIN-Bench 2025, LLM-Hanabi Oct 2025, MARS Oct 2025, Sparks Jan 2026), frontier reasoning models score ~15–17.5/25 in 2p with heavy engine-provided deduction help, non-reasoning models <10/25, and no LLM approaches the first quartile of human scores; prompting is text-state + chain-of-thought, with one paper (Sparks) doing SFT+GRPO on a 4B model to 12.3/25. No published LLM Hanabi result is competitive with RL (>23) or H-group humans (~23).

### Cited Findings
- LLM-Coordination (arXiv 2310.03903v3, April 2025): Hanabi 2p GPT-4-turbo 13.33±0.88; GPT-4o 8.33±1.20; GPT-3.5-turbo 1.33±0.72; OBL baseline 24.10±0.01. LLMs show "marked difficulty" with partners' beliefs; explicit ToM/verification steps in the prompt substantially raise scores; LLMs are robust to unseen partners (cross-play) unlike self-play RL; code on GitHub — [LLM-Coordination](https://arxiv.org/html/2310.03903v3)
- SPIN-Bench (v5, Oct 2025): 9 LLMs; top model o1 averages 16.4 (2p), 14.8 (3p), 14.8 (4p), 14.2 (5p); human reference from 54,977 BoardGameGeek games (15–25 range across quartiles); "none [of the LLMs] approach even the first quartile of human scores" — [SPIN-Bench (arXiv 2503.12349)](https://arxiv.org/html/2503.12349)
- LLM-Hanabi (Oct 6, 2025): 14 models; CoT + structured JSON rationale/ToM outputs; LLM-as-judge ToM scoring; 30–50 games per model; first-order ToM correlates with score r=0.76 vs second-order r=0.58. The extracted "top game scores" (DeepSeek-R1 30.00±3.45, QwQ-32B 28.27, GPT-4.1 28.56) **exceed 25, so either a non-standard scoring/variant is used or the extraction is wrong — [unverified; check the paper's scoring definition]** — [LLM-Hanabi (arXiv 2510.04980)](https://arxiv.org/html/2510.04980)
- MARS (Oct 17, 2025): self-play RL on Qwen3-4B; on "Mini Hanabi" (2p) baseline 1.200 (MCTS-100) → MARS 2.222; "Simple Hanabi" 0.833 → 2.036; 1000 games per configuration — [MARS (arXiv 2510.15414)](https://arxiv.org/html/2510.15414v1)
- "Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents" (Jan 2026; earlier NeurIPS 2025 version "Are LLMs Generalist Hanabi Agents?"): 17 LLMs (4B to 600B+), 2–5 players, **10 games per configuration (40 per model)**. Sherlock setting (engine supplies deductions): o3 17.5, o4-mini 17.0, Grok-3-mini 16.4, Gemini 2.5 Pro 15.4, DeepSeek R1 14.9; Watson (minimal context): o3 15.9, o4-mini 15.0; non-reasoning GPT-4.1 / Claude Sonnet 3.7 / Grok-3 <10/25. Mycroft (model must track state itself): drops of 1.2 (o3) to 3.7 (o4-mini, Gemini 2.5 Pro). Qwen3-4B: 4.8 base → 5.8 SFT (HanabiLogs, 1,520 logs) → 12.3 after GRPO (HanabiRewards, 560 games, LLM-as-judge move values); 8.3 in Mycroft. References: experienced humans ~18–23, RL self-play >23, R3D2 ≥20 (2–4p). Mixed-model cross-play interpolates between the two models' self-play scores — [arXiv 2601.18077](https://arxiv.org/html/2601.18077); [NeurIPS 2025 page](https://neurips.cc/virtual/2025/137233)
- Independent practitioner report (Feb 2026): off-the-shelf LLM agents 3.5–6.7 points (GPT-4.1-mini 3.467, Gemini-3-Flash-Preview 5.667, Intellect-3 6.733; Grok4-fast best cost/performance); Qwen3-4B SFT+RL 2.9 → 8.4 after 220 RL steps; Qwen3-235B MoE ~12.2 at 500 steps "with no signs of stopping"; "Tiny Hanabi" Qwen3-1.7B 5.5/6 — [nphard.io, Feb 23 2026](https://nphard.io/2026/02/23/hanabi.html) (blog, not peer-reviewed)
- Metta-AI cogame-hanabi: LLM agents via Anthropic Claude API ("one request per turn with fallback"), hints are "the only channel between you — there is no chat"; ranking by mean score across episodes; no scores published in README — [Metta-AI/cogame-hanabi](https://github.com/Metta-AI/cogame-hanabi)

### Inferences
- Current LLMs sit ~6–8 points below RL/human SOTA, which is good news for the sandbox: there is large headroom for a population to improve, and improvement is measurable against fixed external yardsticks (OBL, H-group bots, AH2AC2 proxies).
- Every LLM paper uses tiny evaluation sizes (10–50 games/model) — a tell that per-game LLM cost is the binding constraint. See Q6 for what this does to confidence intervals.
- The biggest lever found so far is not prompting but (a) engine-side deduction ("Sherlock" +1–4 points) and (b) RL fine-tuning (+156%). A population framework where agents write *code* (deduction helpers, conventions) that other agents adopt is thus well matched to where gains actually come from.

### Gaps
- No paper reports dollar or token cost per Hanabi game for LLM agents; Sparks' 10-games-per-config design is the only indirect signal. Per-game prompt sizes are unverified.
- No LLM paper reports cross-play between independently prompted/trained LLM groups as a convention-transfer metric (Sparks reports only mixed-model teams).

---

## Q5. Hanabi: LLMs writing bots as code, evolving conventions, population / cultural-evolution / ZSC framing

### Takeaway
There is no published work (found) where LLMs write Hanabi bots as code or evolve Hanabi conventions in a population; the closest pieces are (i) Metta-AI's cogame-hanabi testbed where "a policy is just a prompt" with conventions as experimental conditions, (ii) Ashery et al. (Science Advances 2025) showing spontaneous conventions in LLM populations in a naming game, and (iii) the RL cross-play/ZSC literature, which already uses cross-play as the operational measure of convention transfer.

### Cited Findings
- Cross-play as ZSC proxy: "ZSC performance is evaluated through cross-play, where agents of the same learning algorithm come from independent training runs"; self-play agents "learn highly specialized conventions that are not transferable to novel partners" — [R3D2 (arXiv 2503.14555)](https://arxiv.org/pdf/2503.14555)
- Human conventions injected into the action space yield "a significant improvement on the performance of existing techniques for self-play and cross-play" (numbers in Q2) — [arXiv 2412.06333](https://arxiv.org/html/2412.06333)
- Emergent social conventions in decentralised LLM populations (naming game, not Hanabi): universal conventions emerge; collective bias emerges even when individuals are unbiased; committed adversarial minorities can flip conventions — [Ashery, Aiello, Baronchelli, Science Advances 11(20) 2025](https://www.science.org/doi/10.1126/sciadv.adu9368); [arXiv 2410.08948](https://arxiv.org/pdf/2410.08948)
- Metta-AI cogame-hanabi: LLM (Claude) or scripted (Nim) policies; conventions and common knowledge manipulable as experimental conditions; WebAssembly replay viewer — [Metta-AI/cogame-hanabi](https://github.com/Metta-AI/cogame-hanabi). A related small repo ("hanabi-ck") exists — [pabloloyola/hanabi-ck](https://github.com/pabloloyola/hanabi-ck) **[contents unverified]**
- LLM-driven code evolution exists outside Hanabi (FunSearch lineage, ShinkaEvolve Sept 2025) — [ShinkaEvolve (arXiv 2509.19349)](https://arxiv.org/pdf/2509.19349); a June 2026 paper on co-evolutionary LLM strategy evolution targets *adversarial* games — [arXiv 2606.10389](https://arxiv.org/pdf/2606.10389) **[abstract only]**
- Overcooked precedent for "LLM reasoning distilled into executable policy code": Co-pi-tree (June 2026) learns "an executable policy tree" from LLM reasoning, +35.4% reward, −77.7% LLM queries, −97.1% latency — [arXiv 2606.08596](https://arxiv.org/abs/2606.08596)

### Inferences
- The thesis's "teach by transmitting code or written conventions, verify before adopting" maps directly onto existing Hanabi measurement tools: cross-play (within-group vs between-group), and AH2AC2 human proxies (human-compatibility). This is an open niche: no Hanabi paper yet reports an LLM population with code transmission.
- Convention bots as code are known to be writable by humans in a few thousand lines (will-hanabi-bot, scala-bot, raikan), so "agent writes a convention bot" is a plausible LLM task; Co-pi-tree shows the distill-to-code loop works in Overcooked.

### Gaps
- No source found on LLMs generating Hanabi bot code, nor on evolutionary/population search over Hanabi conventions. Searches: "LLM generates Hanabi bot code evolutionary conventions", "Hanabi LLM in-context conventions population emergent".
- Metta-AI repo last-update date and any scores are unverified.

---

## Q6. How many Hanabi games to estimate a mean score to ±0.5 points?

### Takeaway
It depends almost entirely on the per-game standard deviation σ, which ranges from ~1.4 (strong, rarely-bombing RL agents) to ~3.5–4 (Rainbow/ACHA-level agents) and plausibly 6–9 for LLM agents with high bomb-out (score-0) rates. For a 95% CI of ±0.5 you need roughly 35 games at σ=1.5, ~190 at σ=3.5, ~385 at σ=5, ~1000 at σ=8; for ±0.5 as one standard error, divide those by ~3.8.

### Cited Findings
- Bard et al.: ≥1000 games, ± = standard error over trials; Rainbow 2p 20.64 (0.11), ACHA 2p 22.73 (0.12), BAD 23.92 (0.01), WTFWThat 5p 24.89 (0.00) — [Bard et al. 2020](https://ar5iv.labs.arxiv.org/html/1902.00506)
- Entropy-IPPO: 24.48±0.02 with 5,000 games per entry — [arXiv 2511.22581](https://arxiv.org/html/2511.22581v1)
- OBL: 5,000 games × 5 seeds; ± is std across seeds (not per-game) — [OBL](https://ar5iv.labs.arxiv.org/html/2103.04000)
- Other-Play human study: 45% (OP) and 85% (SP) bomb-out rates with s.e.m. 1.23 / 1.18 over 40 games each — [Other-Play](https://ar5iv.labs.arxiv.org/html/2003.02979)
- LLM benchmarks use 10 games/config (Sparks), 30–50 games/model (LLM-Hanabi), 1,000 games/setting (AH2AC2 API, R3D2) — [Sparks](https://arxiv.org/html/2601.18077); [LLM-Hanabi](https://arxiv.org/html/2510.04980); [AH2AC2](https://arxiv.org/html/2506.21490v2); [R3D2](https://arxiv.org/html/2503.14555v1)

### Inferences
- Back-solving σ from reported SEs (σ = SE·√n, assuming the stated n): Rainbow 0.11·√1000 ≈ 3.5; ACHA ≈ 3.8; entropy-IPPO 0.02·√5000 ≈ 1.4 (if that ± is a per-game SE, which the paper does not state explicitly). The OP human study s.e.m. 1.23 over 40 games implies σ ≈ 7.8 — a bimodal (bomb-out vs ~20) distribution typical of weak/LLM agents.
- Required n for a 95% CI half-width of 0.5: n ≈ (1.96σ/0.5)² ≈ 15.4σ² → σ=1.4: ~30; σ=3.5: ~190; σ=5: ~385; σ=8: ~985. For "±0.5 = 1 SE": n = 4σ² → 8 / 49 / 100 / 256.
- Practical consequence: Sparks' 10 games/config gives ±~2–5 points per configuration for LLM agents, so model rankings within a few points are not statistically separable; a population experiment that wants ±0.5 per generation per group needs on the order of 200–1,000 games per measurement for LLM-grade agents, which is where per-game LLM cost bites. Measuring against deterministic bots (hanabi-engine, will-hanabi-bot) with fixed seeds reduces variance via paired comparisons (not found in literature; inference).

### Gaps
- No paper directly reports per-game score std for LLM Hanabi agents; the σ≈8 figure is inferred from the 2020 human-AI study, not from an LLM paper.

---

## Q7. Overcooked-AI: API, speed, SOTA/cross-play, LLM agents, population-based training, visualiser

### Takeaway
Overcooked-AI (MIT, Python; `OvercookedGridworld`/`OvercookedEnv`, 5 classic layouts, 400-step episodes, scores ≈ soups×20 in the 100–280 range) has the richest population-based-training literature (FCP, MEP, TrajeDi, HSP, COLE, E3T; ZSC-Eval toolkit) and good visual tools (StateVisualizer, overcooked-demo web UI, JaxMARL JIT renderer), but an ICLR 2025 paper shows its cross-play failures are largely state-coverage artefacts and proposes OvercookedV2; LLM agents (ProAgent, HLA, Collab-Overcooked, Co-pi-tree) need macro-action hierarchies because the environment is real-time at ~1.9×10³ steps/s per Python env.

### Cited Findings
- API/tools: `pip install overcooked-ai` or `uv sync`; classes `OvercookedGridworld` (game logic), `OvercookedEnv`; `StateVisualizer`; browser demo at humancompatibleai.github.io/overcooked-demo; trajectory replay; 5 classical layouts + `layout_generator.py`; pretrained checkpoints and human data in repo; "DRL and BC implementations are now deprecated"; MIT — [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai)
- Speed (steps/s): original Overcooked-AI 1.9×10³; JaxMARL 1 env 3.6×10³, 100 envs 3.0×10⁵, 10k envs 1.7×10⁷; JaxMARL Overcooked rendering is JIT-compiled — [JaxMARL](https://arxiv.org/html/2311.10090)
- Self-play vs cross-play on classic layouts (OvercookedV2, ICLR 2025): Cramped Room SP 259±1 / XP 257±2; Asymmetric Advantages 278 / 278; Coordination Ring 214±8 / 96±72; Forced Coordination 199±1 / 138±70; Counter Circuit 163±14 / 59±57. With state augmentation XP rises to 198 / 193 / 140 on the three hard layouts. Conclusion: "ZSC failures can largely be attributed to poor state-coverage rather than more sophisticated coordination challenges... the Overcooked environment is therefore not suitable as a ZSC benchmark." OvercookedV2 (JAX) adds asymmetric information and stochasticity; on V2 layouts SP→XP gaps of 96–220 points remain for both SP and Other-Play — [OvercookedV2 (arXiv 2503.17821)](https://arxiv.org/html/2503.17821); [ICLR 2025 page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9d521f48cd3a9b218290a274af16905a-Abstract-Conference.html)
- Population methods benchmark: ZSC-Eval (NeurIPS 2024 D&B) implements SP, FCP, MEP, TrajeDi, HSP, COLE, E3T on 9 Overcooked layouts; 30 evaluation partners × 50 episodes × 5 seeds; BR-Prox metric; human study with 145 participants, 400-step (~1 min) rounds, Spearman 0.90–1.00 between ZSC-Eval and human rankings; behaviour-cloned human proxies "cannot mimic real human behaviors" (correlation 0.10–0.90 by layout); no Hanabi — [ZSC-Eval (arXiv 2310.05208)](https://arxiv.org/html/2310.05208). BR-Prox values quoted in search (FCP 0.78, MEP 0.78, HSP 0.80, COLE 0.75, TrajeDi 0.81, E3T 0.66, SP 0.20) are for the Google Research Football scenario, **not Overcooked**; per-layout Overcooked numbers are in the paper's figures only.
- LLM agents: ProAgent (AAAI 2024) with FCP/MEP/COLE-style partners: Cramped Room 197.3±6.1, Asymmetric Advantages 228.7±23, Coordination Ring 175.3±29, Forced Coordination 49.7±33.1, Counter Circuit 126.3±32.3; "five episodes" per pairing; "over 10% average improvement" vs SOTA with human proxies; LLM identity not stated in extracted text — [ProAgent (arXiv 2308.11339)](https://arxiv.org/html/2308.11339); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/29710/31219)
- HLA (AAMAS 2024): Slow Mind GPT-3.5 + Fast Mind Llama2-13B-chat + scripted executor; 60-volunteer study; scores Ring 114.4 / Partition 100.3 / Bottleneck 130.3 / Quick 117.2 vs best baseline 92.5 / 57.7 / 103.8 / 71.2 ("~50% higher"); macro-action latency 1.07 s vs 2.30–4.16 s; atomic 0.08 s — [HLA (arXiv 2312.15224)](https://arxiv.org/html/2312.15224)
- Collab-Overcooked (EMNLP 2025 main): 13 LLMs, 30 open-ended tasks, natural-language communication, process-oriented metrics; LLMs strong at goal interpretation, "significant shortcomings in active collaboration and continuous adaptation"; code at github.com/YusaeMeow/Collab-Overcooked — [arXiv 2502.20073](https://arxiv.org/abs/2502.20073); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.249/)
- Co-pi-tree (June 2026): distils LLM reasoning into an executable policy tree, refined in closed loop; +35.4% average reward, −77.7% LLM queries, −97.1% latency — [arXiv 2606.08596](https://arxiv.org/abs/2606.08596)
- LLM-Coordination Overcooked: GPT-4-turbo 260.0±11.55 on Asymmetric Advantages vs PBT 216.9±1.31 on Cramped Room (different layouts, so not directly comparable) — [LLM-Coordination](https://arxiv.org/html/2310.03903v3)
- Overcooked Generalisation Challenge (UED-based layout generation) exists as an alternative benchmark — [arXiv 2406.17949](https://arxiv.org/html/2406.17949v3) **[abstract only]**

### Inferences
- Overcooked's strengths for the thesis: mature population methods to compare against, a toolkit (ZSC-Eval) with a non-game-specific metric (BR-Prox), and the best visualiser of the four games. Its weaknesses: (1) per-step real-time control makes raw LLM play impractical (HLA needs a scripted executor; ProAgent uses macro actions); (2) OvercookedV2 undermines classic Overcooked as a *convention* benchmark — much of the cross-play gap is state coverage, not signalling; (3) scores are layout-specific (100–280) so "absolute score" is less portable than Hanabi's 0–25.
- Strategy-as-code is natural here (Co-pi-tree policy trees, scripted executors) but what is transmitted is mostly motor-planning heuristics, not communicative conventions.

### Gaps
- No per-layout Overcooked cross-play table for FCP/MEP/HSP/COLE extracted (figures only in ZSC-Eval); numbers for those methods vs human proxies from the original FCP/MEP papers were not fetched.
- ProAgent's underlying LLM and cost not extracted. No 2026 Overcooked LLM population paper found; "Benchmarking Open-Ended Multi-Agent Coordination in Language Agents" (arXiv 2606.08340) and CoCoBench (arXiv 2608.28266) appeared in search but were not read.

---

## Q8. Other cooperative games with LLM literature and an absolute score (Codenames, The Crew, Hanabi variants)

### Takeaway
Codenames has a solid LLM benchmark (9 LLMs, 100 trials per pairing, metric = mean turns to win, with an open framework) and The Crew has an unofficial 8-model study (mission success rate); neither has RL/population baselines or a convention-transfer metric comparable to Hanabi cross-play. Mini/Tiny Hanabi variants are used for cheap RL fine-tuning of small LLMs.

### Cited Findings
- Codenames (Stephenson et al., arXiv 2412.11373v2, Apr 2025): 9 LLMs (o1-preview, o1-mini, o3-mini, GPT-4o, Gemini-1.5, Sonnet-3.5, DeepSeek-R1, DeepSeek-V3, Llama-3.1) vs Word2Vec/GloVe agents; single-team metric = mean turns (lower better) and loss %; 100 trials per pairing. o1-preview self-pair 8.41 turns / 13% loss; Word2Vec self-pair 6.81 turns / 0% loss; GPT-4o+Llama-3.1 11.17 / 17%. Two-team win rates: GPT-4o vs GPT-4o 40/60; Sonnet-3.5 vs Llama-3.1 61/39. Finding: LLMs do not beat traditional agents with identical partners but degrade far less with mismatched partners. Framework: github.com/stepmat/Codenames_GPT (ToG_2025) — [arXiv 2412.11373](https://arxiv.org/html/2412.11373)
- Codenames ad-hoc concept-forming eval (GEM workshop 2025) — [ACL Anthology 2025.gem-1.63](https://aclanthology.org/2025.gem-1.63.pdf) **[not read]**
- The Crew ("LLMs Play The Crew", unofficial, MIT code): 8 OpenAI models (GPT-4o-mini, GPT-4.1-nano, GPT-5, ...), 10 missions × 10 trials × 8 models; random-baseline success 0.5–9% on hard missions; GPT-5 100% on easy/medium, 73% on hard; others at/below random on hard missions; text rules + state + self-notes + structured-output move selection; "smaller models struggle... with reasoning about their own role in the mission" — [LLMs Play The Crew](https://ekkarpinski.github.io/LLMCrew/) (date unstated; GPT-5 implies ≥Aug 2025)
- Hanabi variants for cheap LLM RL: "Mini Hanabi"/"Simple Hanabi" (MARS; scores ~2 out of a small max) — [MARS](https://arxiv.org/html/2510.15414v1); "Tiny Hanabi" (max 6; Qwen3-1.7B 5.5/6) — [nphard.io](https://nphard.io/2026/02/23/hanabi.html); SRPO's 4-player 3-color/3-rank Hanabi — [arXiv 2602.21515](https://arxiv.org/html/2602.21515)
- SPIN-Bench bundles Hanabi with competitive/negotiation games (Diplomacy etc.) as an LLM suite — [SPIN-Bench](https://arxiv.org/html/2503.12349)

### Inferences
- Codenames' metric (turns to clear the board) is objective and absolute, and the game is extremely cheap per game (≈10 LLM calls), but strategies are embedding/word-association heuristics, not multi-step signalling conventions, and there are no RL/population SOTA baselines beyond word-vector bots.
- The Crew's per-mission success rate is objective but the only study is unofficial with 10 trials per cell; no RL baseline, no API beyond the author's repo.

### Gaps
- No peer-reviewed The Crew LLM paper found. No cross-play/population results for Codenames or The Crew.

---

## Q9. For each game: strategy as code/written rules, and visual replay

### Takeaway
Hanabi uniquely has both: a canonical written convention document (H-group) with multiple independent code implementations, and replay viewers (hanab.live, Metta WASM viewer, hanabi-engine `--explain`). Overcooked has the best live visualiser and a 2026 precedent for distilling LLM reasoning into executable policy code. Codenames/The Crew have simple text logs only.

### Cited Findings
- Hanabi written rules → code: H-group levels (text, PR-governed) — [hanabi/hanabi.github.io](https://github.com/hanabi/hanabi.github.io); implementations in NodeJS — [will-hanabi-bot](https://github.com/will-hanabi-bot/hanabi-bot/blob/master/README.md), Scala — [scala-bot](https://github.com/simonbohnen/scala-bot), Rust — [hanabi-engine](https://github.com/Zamiell/hanabi-engine), plus raikan — [mswart/raikan](https://github.com/mswart/raikan); Metta policies as prompts or Nim code — [cogame-hanabi](https://github.com/Metta-AI/cogame-hanabi); conventions as RL macro-actions — [arXiv 2412.06333](https://arxiv.org/html/2412.06333)
- Hanabi replay: hanab.live platform and dataset (format used by AH2AC2) — [hanab.live](https://hanab.live/), [AH2AC2](https://arxiv.org/html/2506.21490v2); hanabi-engine "Hanabi Live replay analysis" with per-turn `--explain` — [hanabi-engine](https://github.com/Zamiell/hanabi-engine); Metta "static WebAssembly replay viewer" — [cogame-hanabi](https://github.com/Metta-AI/cogame-hanabi); HLE itself has no renderer — [HLE repo](https://github.com/google-deepmind/hanabi-learning-environment)
- Overcooked: `StateVisualizer`, overcooked-demo browser UI, trajectory replay — [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai); JIT-compiled renderer — [JaxMARL](https://arxiv.org/html/2311.10090); executable policy trees from LLM reasoning — [Co-pi-tree](https://arxiv.org/abs/2606.08596); scripted executor + LLM planner — [HLA](https://arxiv.org/html/2312.15224)
- Codenames: open framework (stepmat/Codenames_GPT) — [arXiv 2412.11373](https://arxiv.org/html/2412.11373); The Crew: MIT repo, text-based — [LLMCrew](https://ekkarpinski.github.io/LLMCrew/)

### Inferences
- In Hanabi a "teachable strategy" can be shipped at three levels of fidelity: prose (H-group level text), a prompt (Metta), or a bot/policy library (Rust/NodeJS). Verification-before-adoption is operationalisable as: run N self-play games with the candidate bot, then cross-play with your group's current bot.
- Interpretability edge: Hanabi game logs are short (~60 turns, discrete actions) and human-auditable via hanab.live replays; Overcooked logs are 400-step motor trajectories, visually clear but semantically harder to audit for "what convention was used".

### Gaps
- No standard JSON schema for Hanabi replays across HLE / hanab.live / Metta was confirmed; hanab.live's export format not fetched.

---

## Q10. Known problems: cooperative-only, convention lock-in, LLM theory-of-mind weakness, cost

### Takeaway
All four games are pure-cooperative (no in-game competition; competition must be imposed at the population/compute level). Convention lock-in is quantitatively documented in Hanabi (SAD 23.97→2.52 XP) and is also the mechanism that makes it measurable. LLM ToM weakness is consistently reported (Hanabi ≤17.5/25 even with engine deductions; Overcooked LLMs weak at "active collaboration"; The Crew smaller models at random). Cost shows up as tiny evaluation sizes in every LLM paper.

### Cited Findings
- Lock-in: SAD cross-play 2.52±0.34 vs self-play 23.97±0.04; SAD with human-clone bot 0.83 — [OBL](https://ar5iv.labs.arxiv.org/html/2103.04000); hat-guessing WTFWThat 24.89 (5p) but 19.45 (2p) and requires "all players follow the same algorithm" — [Bard et al.](https://ar5iv.labs.arxiv.org/html/1902.00506), [Eger](https://gruss.cc/files/waitasecond.pdf)
- Overcooked's cross-play gap is largely state coverage, not coordination; "not suitable as a ZSC benchmark" (classic layouts) — [OvercookedV2](https://arxiv.org/html/2503.17821)
- LLM ToM: "marked difficulty with scenarios demanding active consideration of partners' beliefs and intentions" — [LLM-Coordination](https://arxiv.org/html/2310.03903v3); "none approach even the first quartile of human scores" — [SPIN-Bench](https://arxiv.org/html/2503.12349); non-reasoning models <10/25; 1.2–3.7-point drop when models must track state themselves — [Sparks](https://arxiv.org/html/2601.18077); first-order ToM r=0.76 with score — [LLM-Hanabi](https://arxiv.org/html/2510.04980); Overcooked LLMs show "significant shortcomings in active collaboration and continuous adaptation" — [Collab-Overcooked](https://arxiv.org/abs/2502.20073); The Crew smaller models "struggle... with reasoning about their own role" — [LLMCrew](https://ekkarpinski.github.io/LLMCrew/)
- Cost signals: 10 games per configuration (Sparks); 30–50 (LLM-Hanabi); 5 episodes per pairing (ProAgent); HLA needed a two-tier LLM + scripted executor to hit 1.07 s macro-action latency — [Sparks](https://arxiv.org/html/2601.18077), [LLM-Hanabi](https://arxiv.org/html/2510.04980), [ProAgent](https://arxiv.org/html/2308.11339), [HLA](https://arxiv.org/html/2312.15224). Practitioner note that Grok4-fast had the best cost/performance among hosted models — [nphard.io](https://nphard.io/2026/02/23/hanabi.html)
- Human-proxy limitations: BC proxies "cannot mimic real human behaviors" (Overcooked, ZSC-Eval) — [ZSC-Eval](https://arxiv.org/html/2310.05208); AH2AC2 addresses this in Hanabi with KL-regularised proxies scoring 22.6–23.0 self-play — [AH2AC2](https://arxiv.org/html/2506.21490v2)

### Inferences
- Lock-in is a feature for the thesis: between-group cross-play collapse is the null hypothesis that "teaching + verification + migration" is supposed to beat, and Hanabi gives the sharpest, best-documented version of that signal.
- Cost mitigation options with published precedent: run most games with code bots the agents wrote (HLE/Rust engine at >10² games/s), use LLM calls only to author/revise bots or for sampled games; or use reduced variants (Tiny/Mini Hanabi) for early generations and full Hanabi for evaluation.

### Gaps
- No source quantifies LLM-Hanabi cost per game in dollars/tokens; no source studies LLM populations with between-group competition on any of these games.
