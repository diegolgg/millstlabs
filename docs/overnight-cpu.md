# Saved-results review and overnight CPU design

## What has actually been observed

Reviewed on 2026-10-04: the Mac's `runs/local-20261004-145231/trained`, `runs/model-check/trained`, and `runs/notebook/20261004-145905-edb19d/r_adult`. These share seed 11 and the same 128-decision training trajectory. They are **not three independent replicates**. Codespace results have not yet been accessible; this assessment does not include them.

| Observation | Meaning |
| --- | --- |
| 128 decisions over 16 world ticks, 16 warm-start demonstrations | An execution check, too short to assess sustained learning |
| One FEED action, eight energy units consumed, no births or deaths during training | Inheritance was never exercised in these learned runs |
| CLI evaluation: all eight survive four ticks | A ceiling caused by an uninformative horizon |
| Notebook evaluation: mean lifetime 27.875/32 ticks; five of eight survive; three predation deaths | Longer tests reveal failures that the four-tick test hides |
| Final action probabilities vary by at most 0.0031 across five controlled probes | Little differentiation across those tested contexts; not evidence of a learned social strategy |

The separate 30-episode heuristic ecology calibration produces births, maturation and multiple generations. All ten fertile food/vigilant episodes eventually go extinct (291–1,141 ticks). This establishes that the simulator can exercise the mechanisms, not that learned agents have discovered them. Extinction and external restarts must remain visible in the analysis. The reviewed Mac smoke records are exported in [mac-smoke-review.json](results/mac-smoke-review.json).

**New readiness check:** trained a 4,096-demonstration warm start for seed 11 and evaluated it before PPO on the first two 512-tick maps. Mean lifetime was **76.25 ticks versus 83.1875 for random actions on those same maps**. All 16 agents died (five starvation, eleven predation); 81 of 1,220 actions were FEED (6.64%). Controlled probe responses still varied little. This one-seed observation does not establish a statistically reliable deficit, but the stronger warm start has not demonstrated useful survival improvement. See [warmstart-readiness.json](results/warmstart-readiness.json).

**Recommendation after that check:** stage the larger experiment. Run the first three paired conditions (`--seed 11`) to test whether PPO improves behavior and whether reproduction occurs. Review those before adding seeds 22 and 33. The full nine-run overnight plan below is prepared, but a blind expansion should not be presented as validation of a working inheritance mechanism. The Mac already has the verified seed-11 warm start and a paused initial evaluation in `runs/overnight-cpu`; the runner reuses both. Codespaces will generate its own warm starts.

## Fixed pilot design

`configs/overnight-cpu.yaml` uses the actual pinned SmolLM2-135M model, private LoRA adapters and recurrent PPO, entirely on CPU. It holds the prosocial profile fixed (social preference 0.5, no vigilance prior) and compares:

* `iteration`: independent learning without reproduction.
* `r_adult`: reproduction with the parent's current learned policy.
* `r_initial`: reproduction with the common initial policy, controlling for demographic replacement.

The decisive inheritance contrast is `r_adult - r_initial`. The contrast with `iteration` also changes demography and cannot isolate inheritance by itself. This pilot cannot establish that prosociality is better than other personality profiles or that explicit teaching works; those require additional conditions/mechanisms.

| Setting | Value |
| --- | --- |
| Training seeds | 11, 22, 33, paired across methods |
| Training decisions | 16,384 per condition; 147,456 across nine runs |
| Warm start | 4,096 demonstrations per seed, reused across methods; 12,288 total |
| Population | Eight founders, cap 16, social reward denominator 16 |
| PPO and social windows | 128 and 256 world ticks; sequence length 32; two PPO epochs |
| Evaluation checkpoints | Before any learning, 8,192 decisions, final 16,384-decision budget |
| Evaluation | Two maps at initial/intermediate; four final maps; horizon 512 ticks; eight reset bodies; no reproduction |
| Evaluation seeds | 2,000,000–2,000,003, separate from main-study maps |
| Newborn probes | Every eighth birth; one map; horizon 128 |
| Execution | One process at a time, four CPU threads, rotated method order across seeds |

The trainer respects whole population ticks, so realized decisions can fall just below the nominal budget. Initial-to-final changes in `status.json` use only common maps. Final method comparisons use all four maps. Map episodes are repeated measurements within a trained seed, not independent training replicates. Three training seeds provide exploratory effect estimates, not a convincing confirmatory significance test.

The new matched heuristic baseline evaluation gives mean restricted lifetimes **80.156 random, 173.406 food-seeking, 213.625 vigilant** over these four 512-tick maps. These are reference levels, not targets to optimize the test maps against. The individual episodes are saved in [overnight-baselines.json](results/overnight-baselines.json).

## CPU and storage budget

The [longer Mac benchmark](results/local-pilot-cpu-benchmark.json) collected 673 decisions before extinction at tick 106 and performed two PPO epochs: **92.90 seconds, 7.24 decisions/second, 1.46 GiB peak process RSS**. This benchmark used the identical per-run learning settings in `configs/local-pilot.yaml`, which contains one seed rather than three. Its reported sweep estimate is therefore for three runs; multiply by three for the overnight plan.

At that rate, nine runs need **5.65 hours of training alone**. The benchmark's measured inference time was about 0.051 seconds per decision. Scheduled evaluations have a worst-case total of 294,912 decisions, or roughly another 4.2 hours at that inference rate; early deaths reduce this. Warm starts, policy probes, births, model loads and checkpoint writes add work. **Budget roughly 7–11 hours on this Mac, with substantial uncertainty.** A slower CPU can need another night. The ten-hour runner limit pauses safely and does not guarantee the entire matrix finishes in ten hours.

The 1.46 GiB measurement is from eight non-reproducing agents. It is not a memory guarantee for a reproducing population with archived policies, dead-agent credit and longer histories. Start serially, monitor RAM and swapping, and keep several GiB of headroom. Short-run snapshots were approximately 9 MiB each; long-run snapshots grow with history and archived policies. Allow several GB of free disk space for model cache, logs and checkpoints. Do not launch nine concurrent trainers.

## Run from either checkout

From the repository root, after the environment used by the existing notebook has been installed:

```bash
git pull --ff-only
bash scripts/overnight.sh                     # display plan only
bash scripts/overnight.sh --mode benchmark    # measure THIS CPU
bash scripts/overnight.sh --mode run --hours 10 --seed 11
```

On a Mac, prevent idle sleep for the run with:

```bash
caffeinate -i bash scripts/overnight.sh --mode run --hours 10 --seed 11
```

The staged command above runs only the first three paired conditions (roughly 2–4 hours on this Mac, not guaranteed). Review their feeding, birth and maturation records before expanding with `bash scripts/overnight.sh --mode run --hours 10`. That unrestricted command completes all three seeds and skips completed conditions. `configs/local-pilot.yaml` is also available as a separate one-seed experiment; use a different output directory if selecting that config.

Outputs go to `runs/overnight-cpu`. Rerun the **same command and config** to resume. A changed config requires a new `--output` directory. The limit is per invocation and soft: warm starts, optimizer steps and evaluations finish before pausing. Abrupt process termination can lose the work since the last saved checkpoint. The runner refuses concurrent access to the same output directory.

For Jupyter, launch `bash scripts/notebook.sh`, open `notebooks/overnight_cpu.ipynb`, set `RUN_TRAINING = True`, and execute the training cell. The default Run All only shows the plan and existing results. The kernel must remain running. On a physical Mac, keep it plugged in and awake. Codespaces is a separate remote CPU machine; use its measured throughput, and ensure its session remains available for the run.

## What to inspect the next morning

```bash
bash scripts/overnight.sh --mode report
```

* `status.json`: which of nine conditions finished, actual budgets, initial/final lifetime, and change on common maps. A paused cohort is not a negative result.
* `analysis.json`: generated when all nine finish; paired training-seed contrasts, population statistics, newborn maturation and generation summaries.
* `baselines.json`: random and observation-limited heuristic references on the same map distribution.
* Per-condition `evaluation.jsonl`, `probes.jsonl`: survival, feeding/watch rates, predation/starvation and controlled behavioral responses at each checkpoint.
* `ecology.jsonl`, `events.jsonl`, `newborn_evaluation.jsonl`, `extinction.jsonl`, `recovery.jsonl`: births, maturation, inherited policies, sampled newborn quality and external resets. Some files are absent when the corresponding event never occurs.
* `updates.jsonl`, `summary.json`, checkpoints: optimizer diagnostics, compute accounting, unfinished dead-agent social updates, and resumable state.

Interpret results in this order: did the warm start learn useful behavior; did continued training improve common-map lifetime; were there births and mature descendants; did adult inheritance beat initial-policy inheritance consistently across seeds; did inherited newborn quality improve across generations? If births are absent, the inheritance comparison is unexercised. If repeated extinction/restarts dominate, it is a recovery-heavy experiment rather than a persistent population. If all policies die mostly by starvation and barely feed, investigate learning and observations before multiplying compute. Do not tune ecology or hyperparameters separately for whichever method looks weaker on these held-out maps.

Pending dead-agent social windows at the final budget are reported but cannot receive a fully realized future beyond the run. Keep the same truncation rule across methods and inspect its frequency. The 16,384-decision pilot is far smaller than the planned main study; absence of improvement here does not establish that long-running inheritance fails.
