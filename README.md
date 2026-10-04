# Mill Street Labs: population-learning sandbox

A runnable predator–prey game for studying independent learning, social preferences, demographic replacement, and adult-policy inheritance. Includes a PettingZoo environment, recurrent PPO trainer, SmolLM2/LoRA integration, standardized evaluation, a local HTTP API, and compute instrumentation.

This is an experimental platform. The included smoke runs and heuristic pilot verify operation; they do not establish that reproduction, inherited learning, or prosociality improves learned performance.

## Run locally

For an interactive CPU walkthrough, open **[notebooks/local_sandbox.ipynb](notebooks/local_sandbox.ipynb)**. It includes an animated heuristic demo, editable run settings, actual SmolLM2 training, and plots of ecology, evaluation, and behavioral probes:

```bash
bash scripts/notebook.sh
```

The launcher creates `.venv` if needed, installs missing dependencies (CPU PyTorch on fresh Linux installs), and serves Jupyter from the repository root using that environment. No activation or Mac-specific path is needed. First launch includes package/model downloads; the approximately one-minute notebook timing was measured **after installation on the development Mac**, not on a fresh Codespace.

**GitHub Codespaces:** run the command from `/workspaces/millstlabs`. Open port **8888** using **Ports → Open in Browser**, leaving its visibility **Private**. If Jupyter asks for a token, paste the one it prints in that terminal. Open `notebooks/local_sandbox.ipynb` and choose **Run → Run All Cells**. See [GitHub's port-forwarding documentation](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace). Keep the server terminal running; Ctrl+C twice stops it. If an earlier Jupyter server is already running, stop it first, or use `PORT=8889 bash scripts/notebook.sh`.

Choose **Run → Run All Cells**, or execute cells with Shift+Enter. The default is a short CPU run; switch `BACKEND` to `"tiny"` in the settings cell for a faster plumbing check. Outputs go to a new timestamped directory under `runs/notebook/`. The notebook calls the same production trainer as the CLI.

For the larger CPU experiment, open **[notebooks/overnight_cpu.ipynb](notebooks/overnight_cpu.ipynb)** or run:

```bash
bash scripts/overnight.sh --mode benchmark
bash scripts/overnight.sh --mode run --hours 10 --seed 11
```

Start with the first three-method cohort: the longer warm-start check has not yet beaten random survival. Remove `--seed 11` to expand to all three paired seeds after reviewing learning and reproduction. The serial runner reuses a shared warm start per seed and resumes from the same output directory. See [the saved-results review and overnight design](docs/overnight-cpu.md) for timing, measurements and interpretation. The notebook defaults to displaying the plan; training is explicitly enabled in its settings cell.

Python 3.11+; a NVIDIA GPU is recommended for the full 120-million-decision study. The smaller SmolLM2 pilot runs on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,api,dev]'
pytest -q
bash scripts/smoke.sh
```

The smoke script makes one shared warm start, trains all three methods with the explicitly simplified `tiny` backend, evaluates checkpoints, and writes a paired analysis to `runs/smoke/analysis.json`. Use a new output directory for repeated smoke runs. `tiny` is a functional test backend, not SmolLM2 and not a scientific substitute for it.

For the simulator alone, `pip install -e .` is enough:

```bash
millst demo --ticks 512 --render
millst calibrate --ticks 2048 --output runs/calibration.json
```

## Run the specified experiment

The default config uses **8 founders, population cap 16, and social denominator 16**, as confirmed by Diego. There are 40 main runs and 20 R-initial controls: **120 million training decisions**, plus warm starts and evaluation. Three stations replenishing at a mean of two units each produce **6**, not 12, energy units per tick.

First measure the actual deployment hardware:

```bash
millst --config configs/main.yaml benchmark --ticks 128 --output runs/gpu-benchmark.json
millst --config configs/main.yaml calibrate --ticks 2048
millst --config configs/main.yaml matrix --output runs/matrix.jsonl
```

Inspect the pilot and freeze config changes before comparison. The benchmark loads the public, pinned [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) checkpoint on first use. No hosted LLM API key is needed; subsequent cached runs can use `HF_HUB_OFFLINE=1`.

Create one common warm start per seed. Every profile and method with that seed must use the same checkpoint:

```bash
for seed in 11 22 33 44 55; do
  millst warmstart --seed "$seed" --output "checkpoints/warm-s$seed.pt"
done

millst train --profile prosocial --method r_adult --seed 11 \
  --warmstart checkpoints/warm-s11.pt --output runs/main/prosocial-r_adult-s11
```

Run a whole matrix sequentially on one GPU, or submit the supplied job array:

```bash
for index in $(seq 0 59); do
  millst run-index --index "$index" --resume
done
# Alternative on a configured Slurm cluster, from the repository root:
sbatch scripts/slurm_array.sh

millst analyze --root runs/main --output runs/main-analysis.json
```

`configs/granular.yaml` adds finer social preferences and vigilance priors. Generate a separate manifest for this exploratory matrix; it contains more than 60 runs. All configuration fields are defined in `src/millstlabs/config.py` and YAML files override defaults.

## Pause and resume

Every rollout and evaluation checkpoint is saved atomically. Resume restores the simulator, random generators, policies, optimizer states, recurrent memories, unfinished rollouts, and pending post-death credit. Resuming requires the identical config, method, profile, and seed.

```bash
millst train --profile prosocial --method r_adult --seed 11 \
  --warmstart checkpoints/warm-s11.pt --output runs/main/prosocial-r_adult-s11 \
  --resume runs/main/prosocial-r_adult-s11/latest.pt --max-wall-seconds 3600
```

The wall limit is checked between ticks; an in-progress PPO update or evaluation finishes first. Load only trusted checkpoint files: full recovery snapshots use PyTorch's pickle format. A process killed during a rollout resumes from the last completed checkpoint; recovery trims uncommitted log suffixes to the checkpoint's recorded byte offsets.

## Environment API

Use the [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/) directly for highest throughput:

```python
from millstlabs.env import parallel_env

env = parallel_env(reproduction=True, social_preference=0.5)
observations, infos = env.reset(seed=11)
while env.agents:
    actions = {agent: 6 for agent in env.agents}  # REST, replace with your policy
    observations, rewards, terminated, truncated, infos = env.step(actions)
```

Supply one action for each current living agent. Newborn IDs appear in the returned observations and act on the next call. Dead IDs receive one terminal result, then disappear from `env.agents`. Never feed `infos`, `render()`, or the environment object into a policy: these expose privileged diagnostics. Individual observations contain only permitted local perception, remembered supplies, alarms, and self state.

For an external controller:

```bash
millst serve
# Interactive endpoint documentation: http://127.0.0.1:8000/docs
python examples/http_client.py
```

Endpoints: `POST /environments`, `POST /environments/{id}/step`, `GET /environments/{id}`, `GET /environments/{id}/render`, `DELETE /environments/{id}`. Every step includes `expected_tick`, preventing duplicate action submission. The service has 16 in-memory sessions, one worker, and binds to loopback by default. Put it behind authentication if exposing it beyond your machine. The API serves simulation, not training jobs.

```bash
docker build -t millst-sandbox .
docker run --rm -p 127.0.0.1:8000:8000 millst-sandbox
# GPU benchmark image; requires NVIDIA Container Toolkit:
docker build -f Dockerfile.gpu -t millst-train .
docker run --rm --gpus all -v "$PWD/runs:/app/runs" millst-train
```

## What is recorded

| File | Purpose |
| --- | --- |
| `config.yaml`, `manifest.jsonl` | Frozen config/digest, warm-start hash, model revision, software versions, condition and seed |
| `ecology.jsonl`, `events.jsonl` | Population, ages, cap occupancy, food, deaths, births, lineages, attacks, alarms, mutations |
| `updates.jsonl` | Per-agent PPO samples, losses, entropy |
| `evaluation.jsonl` | Held-out lifetime, survival, death causes; population and ages at sampling |
| `probes.jsonl` | Seven-action probabilities for the five controlled behavioral probes |
| `newborn.jsonl`, `newborn_evaluation.jsonl` | Native survival before first update and periodic frozen newborn cohort assays |
| `extinction.jsonl`, `recovery.jsonl` | Trial duration and external restarts, distinct from biological birth |
| `latest.pt`, `checkpoint-*.pt` | Resume snapshots, including unfinished social accounting |
| `summary.json` | Actual decisions, evaluation overhead, wall time, token/gradient counts, CUDA measurements |

See [experimental decisions](docs/experiment.md), [compute planning](docs/compute.md), and [local verification](docs/validation.md). The original supplied spec is preserved in `docs/original-spec.txt`.
