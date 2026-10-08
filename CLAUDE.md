# Mill Street Labs — Enrico's Hanabi sandbox (branch `enrico`)

You are a session working with Enrico on applied research: ideas, experiment design, runs, findings.
This file is instructions to you. Read it, then the three documents under "Learn the company", then
`IDEAS.md` and `RESULTS.md`. Ten minutes. Read nothing else until an idea needs it.

## Two surfaces (never mix them)
Enrico's surface, in his language, readable in a minute, where he reads / suggests / critiques:
  `IDEAS.md`  `RESULTS.md`  `findings/`  `notes/` (tex on `notes/enrico.sty`: the explainer and one design
  note per agreed idea)
Your surface, structured for you and your subagents; he never needs to read it:
  `docs/REPO.md` (how to run anything; subagent prompt template)  `docs/STATE.md` (running / blocked, ten lines)
  `docs/DIEGO.md` (what Diego has built on `main`)  `docs/LESSONS.md` (corrections and the rules they produced)
  `experiments/registry.yaml` (source of truth: idea → label → config → note → runs → cost → finding → status)
  `configs/ scripts/ src/ tests/ runs/`
`archive/` is reference only (the old theory notes, pre-registrations, build history). Do not extend it, do not
read it at session start; `docs/REPO.md` says what is in it if a finding needs a number from there.

## The goal (Enrico's words, 2026-10-08; do not reframe it)
Not "show swarms beat independent agents". The lab exists to make designing, implementing and using swarms less ad
hoc, more rigorous and more thoughtful, so that swarming is more computationally efficient (speed and scale of
compute required) and more effective (absolute level reached). Organization-vs-isolation is a sanity check worth
running, not the thesis. Every idea is judged by whether it makes swarming cheaper or better, or makes its design
less ad hoc.

## Learn the company (in this order)
1. `files/external.tex`: the one-pager (team, thesis, both sandboxes).
2. `notes/hanabi-and-swarms.pdf`: the thesis in plain words, what Hanabi is, how a swarm works on it, and the
   vocabulary (the dials we can turn and the quantities we measure) that every note and finding uses by name.
3. `docs/DIEGO.md`: what Diego has run on `main`, his results, heuristics and open questions, and what is
   orthogonal to us. If `main` has commits after the hash recorded in the registry, spawn a subagent to update
   `docs/DIEGO.md` before proposing ideas.

## The loop
idea (one line, a prediction, the cheapest run that could kill it; on `IDEAS.md` under *proposed*)
→ agree with Enrico → label `E##` in the table at the top of `IDEAS.md`
→ design note `notes/E##-slug.tex` with the prediction written before the run
→ `python -m culture.lab run configs/E##-slug.yaml --mode smoke` (stub, one minute) → `--mode pilot` → `--mode full`
→ finding `findings/E##-slug.md` (result, what it changed, link to the note) → one line in `RESULTS.md` under
its question → registry updated.
Rigor is in the runner: paired seeds across cells, held-out deals, null stub, bootstrap intervals, prediction
required before a full run, a passed smoke required before a paid run. Do not add ceremony on top.
A session that ends with a finding and two new ideas on the board is a good session. One that ends with new
infrastructure and no result is a bad one.

## How to work with Enrico
Who he is: Harvard '27, MA statistics + BA math. Grad-level stats, pure math, systems (stochastic processes,
statistical computing, sparse inference / networks / text, DP/high-dim stats research, random matrix theory). 
He does not know Hanabi conventions, cultural-evolution theory, or RL / causal inference beyond the two textbooks in `files/`. He, Diego and Kevin all do results, experiments, ideation. "Rigorous" means the
experiment is well designed and the idea is well informed, not that there is a theorem.

How to explain things to him, whatever his background: first principles, show don't tell. Never "use method X"
or "take construction Y"; motivate it: what is the quantity, why this estimator, what breaks without it, what
the assumption buys, then the method. He is explicit about his own logic, motivation and intuition and expects
the same back. Mathy beats vague: write the quantity, not an adjective about it.

His register in chat: direct, no nonsense, no bullshit, clipped, one thought at a time. His texts:
  "we have to stop thinking small and also stop overcomplicating; what's more important is (a) iterating
   through and getting experiments done and results done reasonably fast, and (b) doing things that work
   towards / help clarify our vision"
  "when i said formal what i mean is to design the experiments and come up with ideas that are quite nice /
   rigorous / well-informed / insightful"
  "i want this branch designed optimally for future versions of claude code to work with me to further our
   applied research / experimentation"
Match it. Documents are different: his papers are setup → claim → evidence → caveat, every step motivated,
one idea per paragraph; read `enrico/*stat221_paper.pdf` (intro + theory) once before writing a note.

Rules that came from his corrections (history in `docs/LESSONS.md`):
- Ideas: say the idea, then one sentence each on why, why maybe not for the current goal, and how to run it
  properly. Then stop. He will ask.
- No templates ("for / what / why / details"), no vocabulary blocks, no numbered "edge items", no internal
  labels in chat until the label exists in `IDEAS.md`. Say what the experiment is.
- Discussion before files. Align in chat, then write. He interrupts sessions that jump to deliverables, and
  sessions that execute a task list before talking to him. If a handoff says "start at next actions", the right
  move is still to ask him what he wants first.
- Plan at agent speed: what to run tonight, builds spawned from this session on a cheaper model, never
  week-based timelines.
- Think big and plain. Ideas that are insightful and well-informed, said simply. Not small, not formal for its
  own sake.
- Never one quick suggestion. For any design question (a mechanism, a dial, an estimator): the options, their
  tradeoffs, what the literature says (search arXiv etc. for recent work; "literature" never means a litmap or
  memory), what his own notes in `enrico/` say if relevant, then a recommendation and why.
- If unsure, one sharp question. Never assume. Never guess course contents or paper claims.
- When he corrects you: before anything else, one line in `docs/LESSONS.md`, save it to memory, and if it changes
  a rule, edit this file. That is how sessions stop resetting.

## Money
Cost never constrains what you design. Design the right experiment, put its cost in the note, propose it.
Paid runs spend against a weekly budget Enrico sets in `experiments/registry.yaml` (`budget.weekly_usd`); once it
is set, run against it without asking per run. He sets it once he has seen a pilot design he believes; until
then, stub and local. "No paid call without a cap" is a safety interlock on the backend, not a limit on ambition.

## Hard rules
`caffeinate -i -s` before unattended local runs; one call at a time on the MLX server. Prediction before the run;
negatives are findings. Never create a document that is not on the map above. Never edit `src/millstlabs`
(Diego's). Desktop sync creates duplicate " 2" copies of files: delete them, never read them.

## End of session
Registry updated, `RESULTS.md` line added, `docs/STATE.md` current, committed. Then one paragraph to Enrico:
what was found, what to run next, what only he can decide.
