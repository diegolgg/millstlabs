# Local verification

The repository includes deterministic tests for the PettingZoo contract, connected seeded maps, observation privacy/line of sight, feeding allocation, movement conflicts/swaps, birth energy and cap rules, WATCH alarms, post-death social accounting, live bootstraps, mutation KL, private-gradient isolation, resume/recovery, HTTP lifecycle, and all three learning methods.

Final local run: **24 tests passed**. The three-method offline smoke workflow also completed. The actual-model integration run completed 16 warm-start transitions, 128 training decisions, 64 PPO optimizer steps, and 32 held-out evaluation decisions. These deliberately small checks establish executable integration, not convergence.

The neural integration test uses a small randomly initialized Llama through actual Transformers/PEFT APIs, with no network. A separate local benchmark downloaded and executed the actual pinned SmolLM2-135M-Instruct model on CPU, including LoRA gradients and two PPO epochs. This caught and resolved a root-module naming difference between AutoModel and AutoModelForCausalLM when selecting the last four layers.

Development calibration uses 30 episodes: five seeds × three baselines × reproduction on/off, each capped at 2,048 ticks. The saved report is in `docs/results/calibration.json`. Among non-reproduction episodes, checks measured random, food-seeking, and vigilant heuristic restricted lifetimes of 84.075, 148.925, and 239.325 ticks respectively. Those are heuristic pilot observations, not trained-agent findings. The report records newborn maturation/death and generation depth. Any future parameter change invalidates this pilot's config digest and requires rerunning calibration.

The simulator passed the available PettingZoo parallel test. PettingZoo warns at extinction that some possible IDs never appeared: this is expected for the reserved future birth-ID pool. Current FastAPI/Starlette emits an upstream httpx deprecation warning in its test client. Neither warning indicates a failed behavioral test.

No 2-million-decision training run, 60-run sweep, CUDA benchmark, GPU container build, or Slurm submission has been performed. Use the GPU benchmark and review the pilot before allocating the full experiment. See `docs/results/` for bounded verification artifacts and recorded dependency versions; generated checkpoints/caches/runs are intentionally excluded from Git.
