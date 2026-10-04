# Compute and deployment

The intended starting machine is one NVIDIA GPU with 24 GB VRAM, four CPU cores and at least 32 GB RAM. This is a target to benchmark, not a measured VRAM guarantee. Local verification was on CPU; no CUDA hardware was available in the implementation session.

SmolLM2 uses one frozen backbone per process. AutoModel omits the unused language-model output head. Each individual has 93,194 trainable parameters in the tested model: private adapter, projection/GRU, action/value heads and WATCH bias. Pending-death policies remain resident until their finite social windows close, then are archived on CPU and removed from PEFT. Historical ancestors do not accumulate on GPU. R-initial stores the starting policy on CPU.

The baseline processes agents sequentially and batches observations only within a PPO sequence for the active adapter. This favors transparent isolation. It does not yet implement mixed-adapter inference batching or a frozen-prefix cache. Backpropagation traverses the adapted final layers; shared base weights never update. CUDA uses bfloat16 when supported and otherwise float32, with float32 controllers. CPU uses float32. Tensor/token counts and actual throughput are measured rather than inferred from parameter count.

Run:

```bash
millst benchmark --ticks 128 --output runs/gpu-benchmark.json
```

It performs collection plus two PPO epochs, reporting decisions/second, parameter count, token processing, update count, peak allocated CUDA memory, and rough full-run hours. It does not run evaluation, mutation probes or warm start, so its extrapolation is optimistic. Repeat with representative rollout/population settings on the intended hardware before a full budget.

At 256 input tokens, 120 million decisions require up to 30.72 billion input tokens for collection alone; two PPO reuses add up to 61.44 billion more. Padding, bootstrapping and evaluation add work. Actual compact observations were substantially shorter in the local check, but a small LLM still incurs material compute cost per world decision.

Standard checkpoint evaluation alone can add up to **917,504 decisions per run**: 8 bodies × 2,048 ticks × (3 × 8 intermediate maps + 32 final maps). Across 60 runs this is up to 55,050,240 extra decisions. Deaths reduce that bound. Newborn assays add up to 2,048 decisions every 16 births by default; mutations also replay short calibration histories. Warm starts add 250,000 demonstration transitions over five seeds plus their optimization passes.

`processed_tokens`, `encoder_forwards`, `gradient_updates`, `inference_seconds` and wall seconds include actual work performed in the process. `evaluation_decisions` explicitly separates evaluation from the training decision budget; warm-start resource counters are stored in their own checkpoints. `cuda_model_stream_seconds` sums CUDA-event elapsed time for encoder execution and backward/optimizer work; it excludes tokenization, simulator CPU time and small controller-only forward operations, and is not an occupancy/utilization measure. `cuda_peak_bytes` is PyTorch allocated-memory high-water mark, not total GPU reservation.

Use the Python interface for training; HTTP serialization is for integration/testing. Keep one active run per GPU until a benchmark demonstrates safe concurrency. The Slurm example sets `%1` to serialize jobs. All 60 jobs share the five read-only warm starts, but have separate output paths. Do not launch multiple writers for the same run directory.

If memory is constrained, reduce sequence length only in a separate frozen pilot config; this changes recurrent training and must apply to every compared condition. Do not silently switch to the `tiny` backend, share learned adapters, reduce one condition's PPO reuse, or cache representations across adapter updates. These would change the experiment.

The CPU simulator/API and GPU trainer have separate Dockerfiles. GPU Docker and Slurm definitions are supplied for deployment but were not executed locally. Install NVIDIA drivers/Container Toolkit or adapt scheduler resource directives to your cluster.
