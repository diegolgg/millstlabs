# Direct LLM action discovery, without task demonstrations

The previous corpus policy used LLM hidden features plus a separately trained seven-action head. This experiment replaces that head with the pretrained model's own next-token probabilities for `north`, `south`, `east`, `west`, `feed`, `watch`, and `rest`. Sampling is constrained to legal actions. Each decision is a real local SmolLM2 forward pass; no API key or external GPU is needed.

The LLM receives public rules, permitted sensors, station offsets computed from those sensors, its recent action outcomes, and a retrieved tentative tactic. There are **zero demonstrations, zero feeding-imitation losses, and no heuristic action teacher**. The model is pretrained, so this tests learning without task examples, not learning from a blank slate. The output is one action token, not a generated explanation or a multi-step deliberation.

## Run and inspect

From the Mac terminal:

```bash
cd /Users/diego/Downloads/millstlabs
bash scripts/autonomous.sh --mode plan
bash scripts/autonomous.sh --mode smoke
caffeinate -i bash scripts/autonomous.sh --mode run --hours 12
```

On Linux, omit `caffeinate -i`. The existing `.venv` and downloaded model are reused. The smoke has its own output directory. Repeat the **same run command** to resume the committed checkpoint; the 12-hour limit is a soft invocation limit, and an evaluation or initialization can overrun it. This starts a new experiment in `runs/autonomous-llm`; previous experiments are separate.

The default compares both requested note-writing styles, `grounded` and `prose`, sequentially, with one cooperative adult-inheritance condition and seed 11. Each gets 8,192 training decisions (16,384 combined). To run only free-form prose, use `--style prose --output runs/autonomous-llm-prose` consistently on run/report commands.

```bash
bash scripts/autonomous.sh --mode report
tail -n 3 runs/autonomous-llm/*/prosocial-r_adult-s11/ecology.jsonl
tail -n 3 runs/autonomous-llm/*/prosocial-r_adult-s11/updates.jsonl
tail -n 3 runs/autonomous-llm/*/prosocial-r_adult-s11/corpus_events.jsonl
cat runs/autonomous-llm/*/prosocial-r_adult-s11/checkpoint-status.json
```

Read an exact sampled LLM input and action:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
paths = list(Path('runs/autonomous-llm').glob('*/prosocial-r_adult-s11/llm_decisions.jsonl'))
if not paths:
    raise SystemExit('No training decisions yet; initialization/evaluation may still be running.')
p = max(paths, key=lambda f: f.stat().st_mtime)
print(p)
row = json.loads(p.read_text().splitlines()[-1])
print(row['prompt'])
print('LLM action:', row['output'], 'probability:', row['probability'])
PY
```

`note_decisions.jsonl` records every action, selected-note IDs and publication decision. `llm_decisions.jsonl` saves full prompts for the first two ticks, publication ticks, and every 64th tick to limit disk use. `corpus-checkpoint-*.json` saves accumulated text with observation provenance. `checkpoint-status.json` reports durable progress; growing logs show recent activity but are not themselves proof a process remains alive.

## What learns

Each individual has private rank-four LoRA adapters in the final four transformer layers. PPO updates these adapters from its sampled action and subsequent personal/cooperative survival outcomes. The base model and tied language output embeddings stay frozen. A separate recurrent numeric critic estimates personal and social returns; critic gradients do not update the language policy. Cooperative preference is 0.5, social reward is normalized by 16, and there are eight founders with a population cap of 16. Children inherit the parent's current adapters. Mutation is disabled in this pilot to isolate learning.

At publication opportunities, the same adapted LLM receives a separate yes/no question. PPO trains that language-token choice from survival advantage plus a small bonus for delivery of a previously unknown outcome pattern to another lineage. This bonus is an information-delivery proxy, **not proof that a note helps**. Automatic retrieval requires no tool-choice menu.

When publication is selected, the frozen base LLM writes a short tentative tactic from the author's actual recent outcomes. The service attaches those outcomes independently. The prose itself is **not** PPO-trained or certified true; a small model can still invent an interpretation. With `--style grounded`, the note contains literal observed outcomes instead of generated prose. Positive energy changes, survival, births, and before/after distances to known station coordinates are observable evidence; the writer never sees hidden food stocks or simulator plans.

Text and evidence survive ecological resets, with lineage ownership. Old map facts and recent personal histories are cleared. Retrieval ranks up to 256 retained notes by a simple match on low energy, legal feeding, and visible danger, then recency. This is bounded memory, not an unlimited archive or learned search engine. Prompt construction drops old optional history, then notes if necessary, before allowing essential sensors or the action request to be truncated.

## Energy and compute

Both reading and writing cost **zero game energy**; publication has no reward penalty. Physical movement, metabolism, watching and reproduction retain their game costs. Writing has a 32-tick per-agent cooldown, a 48-token generation cap, and one retrieved note per decision to bound CPU time. These are compute limits, not energy charges. New HTTP sessions also use zero knowledge energy cost; older frozen experimental configs retain their recorded settings for reproducibility.

The default [configuration](../configs/autonomous-llm.yaml) allocates 8,192 training decisions per style, with assessments at 0, 2,048 and 8,192. It uses CPU, one Torch thread, 1,024-token maximum inputs, and training sequences of four to bound memory. The initialization file is still called `warm-s11.pt` for runner compatibility, but contains **zero demonstration transitions and zero initial gradient updates**.

Assessment has two maps at the start/intermediate checkpoint and four at the end, eight agents, a 256-tick cap, and three access conditions. That is at most 49,152 evaluation decisions per style in addition to training; early extinction reduces actual cost. Expect an overnight pilot that may need resuming over multiple nights, not a guaranteed 12-hour completion. Reading a long prompt every move is substantially more expensive than the earlier short-feature controller.

## What would count as progress

1. **Learning from experience:** higher food consumption, hungry-feeding fraction, and restricted mean lifetime than the initial pretrained policy on the same map seeds. Use `change_from_pretrained_start` in the report. No demonstration baseline is substituted.
2. **Useful accumulated tactics:** shared access beats no retrieved tactics with frozen weights. Use `tactic_access_benefit`.
3. **Useful transfer across lineages:** shared access beats lineage-private access with the same frozen weights and initial corpus. Use `peer_access_benefit`.

Every evaluation map starts from an independent copy of the training corpus. Policies stay frozen, though within-map observations and notes can accumulate; evaluation writes never enter training memory. The no-tactic control blocks all retrieval while retaining the same direct LLM and recent experience. Private means lineage-private, including inherited notes. The comparison isolates access under shared-trained policies; it is not a separately trained private population. A single seed and four final maps support exploratory conclusions only.

The 256-tick assay is a pilot, not evidence of indefinite survival. Synthetic directional cue probes measure sensitivity and basic correctness, not field performance. Counting notes or changing probabilities alone cannot establish learning.

The main risks are weak spatial reasoning in a 135M model, sparse feeding successes, delayed survival credit, and repetitive or incorrect prose. Removing examples makes the test more faithful to autonomous discovery but harder. If performance is flat, inspect actual prompts, feeding opportunities and tactic evidence before simply increasing duration. Subsequent controlled changes could test higher sampling entropy, shorter prompts, a larger local model, or initially easier geometry; keep fixed held-out predator temperatures for comparable measurement.

## Validation on October 5, 2026

[Saved real-model results](results/autonomous-llm-validation.json): both styles completed 128 training decisions, 384 frozen evaluation decisions, and 64 optimizer updates. Each initialized with zero demonstrations/updates, changed all eight private LoRA adapters, and published eight notes. Sampled prompts confirm retrieved notes reached the actual action model. The 83-test suite passed, including physical energy invariance, independent adapters, persistent tactics, evaluation isolation and checkpoint/resume.

Elapsed runner time was 543 seconds for grounded notes and 712 seconds for prose, measured during development rather than an isolated speed benchmark. Neither arm consumed food, and both mainly selected north/south. All agents survived the deliberately short eight-tick assays; that is not a meaningful survival success. The synthetic food-direction probe also failed basic directional competence. These results verify the execution path and expose the starting policy's weakness; they do not establish useful learning. Smoke prose is capped at 24 tokens and can end mid-sentence; the larger run permits 48 tokens, without guaranteeing accurate interpretations.
