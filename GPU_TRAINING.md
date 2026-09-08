# GPU training and CPU comparison

`train_gpu.py` runs batched ecosystems and ARS co-evolution on CUDA. Numerical state, spawning, random fields, policies, fitness, observation statistics, and ARS updates remain on the selected device. Python launches work and handles occasional logs, checkpoints, and optional snapshots. `train.py` remains the existing CPU training entry point.

The implementation is in `lib/gpu/`. It also runs on CPU tensors for numerical validation. CPU reference comparisons, full-graph tracing, checkpoint/resume, and command-line smoke tests can run without a GPU. CUDA execution and performance must be verified on an NVIDIA machine; they were not available on the development machine.

Use Python 3.10 or newer on Linux. Install a CUDA-enabled PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/), choosing a CUDA runtime supported by your driver. Then, from the repository root:

```bash
python -m pip install -r requirements-gpu.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
python -m pytest -q
```

The tests automatically enable CUDA cases when a device is available. These compare CUDA eager, CUDA Graph, and compiled execution, including replay after reset, changing rollout horizons, and numerical agreement with the CPU equations. The compile tests can take longer on their first invocation. Local development used PyTorch 2.2.2; benchmark reports record the exact version installed on the target machine.

Start with this comparison:

```bash
python benchmark_gpu.py \
  --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 \
  --workers 0 --warmup 2 --repeats 5 \
  --execution cuda-graph --output benchmark.json
```

This runs the existing CPU trainer with multiple workers, then the tensor trainer on CUDA. Both use the same grid, policy architecture, perturbation count, number of worlds, horizon, species, reward settings, and fixed policy temperature. The benchmark includes spawning/reset, policy perturbations, every ecological tick, observation-statistics updates, and the ARS update. CPU configuration loading and multiprocessing costs are part of the existing baseline. Logging and checkpoint disk I/O are excluded.

The report contains:

- Every measured iteration time, median and range, and CPU/GPU speedup.
- World-ticks and cell-ticks per second.
- Setup and first-iteration time separately from warmed execution.
- GPU peak allocated/reserved memory, hardware, Python/Torch/CUDA versions, and git revision.
- Complete benchmark arguments and final tensor-backend diagnostics.

Warmup iterations do update the policies. Both implementations start from the same seeded Torch policy initialization, but use different environmental and perturbation random generators. A matching seed provides reproducibility within a backend, not identical CPU/GPU worlds or training trajectories. Numerical parity is tested separately with identical state, actions, perturbations, and injected environmental noise. Repeating benchmarks across several seeds is appropriate for consequential comparisons.

To investigate kernel fusion after the initial CUDA Graph benchmark:

```bash
python benchmark_gpu.py \
  --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 \
  --backends gpu --execution compile-graph \
  --warmup 2 --repeats 5 --output benchmark-compiled.json \
  --trace gpu-trace.json
```

`--trace` records an additional warmed iteration outside the benchmark timings. Open the Chrome-format trace with Perfetto and inspect copies, host synchronizations, and launch gaps. Explicit CUDA synchronization is used at benchmark timing boundaries. Do not interpret asynchronous Python dispatch time as GPU execution time.

| Execution mode | Behavior |
| --- | --- |
| `eager` | Tensor operations on the selected device, launched per tick; useful for diagnostics |
| `cuda-graph` (default) | Replays fixed blocks of ticks with persistent buffers; no compiler dependency |
| `compile` | Compiles the complete tick with `fullgraph=True`; errors on a graph break |
| `compile-graph` | Compiles the tick, then captures/replays blocks with CUDA Graphs |

Graph modes capture one tick and a `--graph-ticks` block (default 32). A changed horizon uses these same captures, including any remainder. Changing the world count reallocates buffers and prepares new captures. Compile/capture failures are raised explicitly; the program never silently switches to CPU or another execution mode. PyTorch describes the launch-overhead mechanism and capture restrictions in its [CUDA Graphs guide](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

Use `--pairs-per-batch 4` if the full candidate batch exceeds VRAM. Both perturbation signs and all worlds are evaluated for each chunk, and the weights are updated only after every chunk completes. The final partial chunk uses fixed-size buffers but excludes its padded slots from fitness and statistics. Smaller chunks reduce memory use and may reduce throughput. Biomass/reserve storage scales with simultaneous ecosystems; complete per-tick trajectories and autograd tapes are not retained.

Train and save policies with:

```bash
python train_gpu.py \
  --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 \
  --generations 10 --iter-per-gen 20 \
  --execution cuda-graph --run-name mareld-gpu
```

By default all decision makers co-evolve. `--species gadoids pelagic_fish` limits trained species while retaining the other policies in every ecosystem; `--no-coevolution` trains targets in round-robin order. Common reward, normalization, mortality, migration, and policy-network options have the same meanings as the CPU runner. Underscore aliases are accepted for the main existing training options. Run either entry point with `--help` for its complete interface.

Training defaults match the ordinary CPU CLI's reward modifiers: entropy coefficient 0.1, argmax penalty 0.3, and temperature annealing from 3 to 1 over 10 generations. The benchmark instead fixes temperature at 1 for both backends. For a run without those modifiers, pass `--entropy-coef 0 --argmax-penalty 0 --temp-start 1 --temp-end 1`; for the equivalent benchmark use the first two flags and `--temperature 1`. CPU `--profile` presets and the interactive Pygame UI are not part of the headless GPU CLI.

`results/mareld-gpu/` contains `trainer.pth` for complete resume, `policy_<species>.pth` files compatible with the existing inference/CPU-training loaders, `gpu_run.json` with configuration metadata, and `training.jsonl` with compact numerical diagnostics. For a non-default network, pass the matching `--policynetwork LAYERS NODES ACTIVATION` to existing inference tools. The CPU batched-policy path now honors sigmoid/ReLU/tanh consistently with individual policies; previously it always used sigmoid.

Resume with the same numerical/configuration options and output directory:

```bash
python train_gpu.py \
  --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 \
  --generations 10 --iter-per-gen 20 \
  --run-name mareld-gpu --resume
```

`--generations` specifies additional generations, including the rest of a partially completed generation. Resume restores weights, normalization, random counters, cached worlds, and the iteration position. Changing biological or training options intentionally changes the continuation; recorded options are metadata rather than automatic overrides. `--init-from results/existing-run` imports existing CPU policies and normalization into a new GPU run without importing an optimizer/random history. Only load checkpoints you trust, since the existing `.pth` format contains Python/NumPy objects.

`--checkpoint-every N` sets the checkpoint interval in generations; the final completed state is also saved. SIGINT/Ctrl+C and SIGTERM request a stop after the current complete update/round so a partially applied ARS update is not saved. Unexpected execution errors leave the previous checkpoint intact. `--log-every N` reduces metrics readbacks; logged timing is the average per update since the preceding log boundary, including startup work at the first boundary.

`--worlds-refresh generation` reuses spawn biomass maps within a generation, while reserves and runtime noise remain pair-specific. The default refreshes worlds every iteration. `--worlds-schedule '1@0,3@10,5@50'` changes the number of worlds at generation boundaries. Random fields are keyed by world, pair, iteration, species/sample, and tick, with shared keys for positive and negative perturbations; changing chunk size does not change their identity.

`--snapshot-every N` optionally saves one final candidate ecosystem as an `.npz` every N generations. It is a perturbed training candidate from the final chunk, not a baseline-policy evaluation or a complete trajectory. Existing inference tools remain the route for baseline-policy visualization. Snapshots and checkpoint/metrics readbacks are explicit output boundaries; none occur inside ecological ticks.

For a quick check on a machine without CUDA:

```bash
python train_gpu.py --project mareld2.yaml --device cpu --execution eager \
  --grid 6x6 --n-deltas 2 --worlds 2 --ticks 5 \
  --generations 1 --iter-per-gen 2 --output /tmp/eco-evo-smoke
python benchmark_gpu.py --project mareld2.yaml --grid 6x6 \
  --n-deltas 2 --worlds 2 --ticks 5 --workers 2 \
  --backends cpu tensor-cpu --warmup 1 --repeats 2 \
  --output /tmp/eco-evo-benchmark.json
```

No GPU speedup is claimed from these CPU-only checks. The meaningful performance result is the synchronized comparison on the target NVIDIA machine.
