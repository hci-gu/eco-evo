# GPU training and CPU comparison

For the optional 10%–300% biomass bounds, warning penalties, and failed-rollout
scoring, see [Population stability training](POPULATION_STABILITY.md).

`train_gpu.py` runs batched ecosystems and ARS co-evolution on CUDA. Numerical state, spawning, random fields, policies, fitness, observation statistics, and ARS updates remain on the selected device. Python launches work and handles occasional logs, checkpoints, optional snapshots, and an opt-in live viewer. `train.py` remains the existing CPU training entry point.

The implementation is in `lib/gpu/`. It also runs on CPU tensors for numerical validation. CPU reference comparisons, full-graph tracing, checkpoint/resume, and command-line smoke tests can run without a GPU. CUDA execution and performance must be verified on an NVIDIA machine; they were not available on the development machine.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.5.9 or newer). On the NVIDIA machine, use Linux with a driver compatible with CUDA 12.8. From the repository root:

```bash
uv sync --locked
uv run --locked python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
uv run --locked python -m pytest -q
```

uv manages Python 3.12 and the locked dependencies. On Linux/Windows, `pyproject.toml` explicitly selects the official PyTorch CUDA 12.8 index for both benchmark backends; macOS uses native PyPI wheels. No manual Torch installation, extra flags, or environment activation is needed. `uv run` also syncs the environment, so the initial `uv sync` is optional. The lockfile pins Torch 2.11.0 (2.2.2 on Intel Macs) and the CUDA runtime packages. See the [project README](README.md) for the other entry points and dependency maintenance.

The tests automatically enable CUDA cases when a device is available. These compare CUDA eager, CUDA Graph, and compiled execution, including replay after reset, changing rollout horizons, and numerical agreement with the CPU equations. The compile tests can take longer on their first invocation. Benchmark reports record the exact version installed on the target machine.

Start with this comparison:

```bash
uv run --locked benchmark_gpu.py \
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

Sub-threshold split suppression is part of both engines. A move action splits the
moving share of a cell across up to four directions, and the extinction sweep
zeroes every cell that ends the tick below
`extinction_threshold_factor * min_split_biomass`, so a diffusing decision maker
would otherwise bleed biomass through the sweep. An outflow whose destination
would still be sub-threshold after receiving it is therefore cancelled and stays
in the source cell. Off-grid directions are left alone, so `--migration on` keeps
its own edge concentration. The tensor engine lacked this step until it was
mirrored in `TensorEcosystem.suppress_splits`; `tests/test_gpu_ecosystem.py`
compares the two engines tick by tick on a fixture where the suppression fires.

To investigate kernel fusion after the initial CUDA Graph benchmark:

```bash
uv run --locked benchmark_gpu.py \
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
uv run --locked train_gpu.py \
  --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 \
  --generations 10 --iter-per-gen 20 \
  --execution cuda-graph --run-name mareld-gpu
```

By default all decision makers co-evolve. `--species gadoids pelagic_fish` limits trained species while retaining the other policies in every ecosystem; `--no-coevolution` trains targets in round-robin order. Common reward, normalization, mortality, migration, and policy-network options have the same meanings as the CPU runner. Underscore aliases are accepted for the main existing training options. Run either entry point with `--help` for its complete interface.

Training defaults match the ordinary CPU CLI's reward modifiers: entropy coefficient 0.1, argmax penalty 0.3, and temperature annealing from 3 to 1 over 10 generations. The benchmark instead fixes temperature at 1 for both backends. For a run without those modifiers, pass `--entropy-coef 0 --argmax-penalty 0 --temp-start 1 --temp-end 1`; for the equivalent benchmark use the first two flags and `--temperature 1`.

### Local per-cell reward

`--local_reward` selects the per-cell source-tracked fitness instead of the
global `log(E_total/E_0)`. Per tick and cell, `A(c,t)` is the cell's energy
(`biomass * energy_content + reserve`) at the start of the tick and `B(c,t+1)`
the end-of-tick energy of exactly the population that started in `c`, tracked
through the move/split into the plus-shaped set `{c, N, E, S, W}`; `reward(c)`
is the ratio. The quotient is scale invariant, so a small cell's good decision
counts as much as a large cell's.

The tensor engine computes the same quantity as `train.py`: a destination cell's
end-of-tick energy is split between its sources in proportion to the biomass
each delivered, which is exact because everything after the movement is
cell-wise multiplicative. Agreement with the CPU trainer is asserted in
`tests/test_gpu_local_reward.py`.

```bash
uv run --locked train_gpu.py --project mareld2.yaml --profile info \
  --local_reward --local_reward_theta 0 --run-name mareld-local
```

`--local_reward_metric log|ratio`, `--local_reward_norm mean|sum|grid`,
`--local_reward_theta`, `--local_reward_clip LO HI` and
`--local_reward_min_energy_factor` have the same meanings and defaults as on the
CPU runner. `theta=0` weights every occupied cell equally; `theta=1` with
`--local_reward_norm mean` collapses back onto the global energy growth rate, so
one flag value provides both A/B baselines. The flag cannot be combined with
`--legacy-reward` or `--population-stability`. With `--migration on` immigrated
biomass has no source cell and is excluded, so the mass-balance identity is
exact only with migration off.

Pick the normalisation deliberately, because two of the three carry a
cell-count gradient (sections 82 and 84). `training.jsonl` logs
`local_occupancy`, the mean number of participating cells per tick, which is
what to watch:

| `--local_reward_norm` | Divides by | Cell-count gradient |
| --- | --- | --- |
| `sum` | nothing | fitness grows with every cell added (rewards spreading thin) |
| `mean` (default) | the weight of the active cells | fitness grows when a cell dies and leaves the denominator |
| `grid` | the constant cell count | none: with `log` a neutral cell contributes exactly 0 |

`grid` is the only combination in which the cell count is orthogonal to the
fitness by construction, and the only one where `sum_t log(B/A)` telescopes onto
`log(end/start)` for a surviving cell line, so a slow bleed is priced the way
the global reward prices it. The default clip `0.2 5.0` is symmetric in log
space (`log(5) == -log(0.2)`); runs made before that change used `0.2 2.0`.

### Training profiles

Both `train.py` and `train_gpu.py` accept `--profile sanity`, `--profile info`,
or `--profile deep`, using the same preset definitions:

| Profile | Generations | Updates/generation | Delta pairs | Ticks/rollout | Worlds | Temperature schedule length |
| --- | --- | --- | --- | --- | --- | --- |
| `sanity` | 10 | 15 | 16 | 100 | 3 | 8 generations |
| `info` | 10 | 20 | 16 | 150 | 3 | 30 generations |
| `deep` | 80 | 20 | 20 | 200 | 3 | 60 generations |

All profiles enable co-evolution, set entropy and argmax coefficients to 0,
set both start/end temperatures to 1, and disable uniform-bias initialization.
With both temperatures at 1, the schedule length has no effect unless you
override one of those temperatures. Other options keep their ordinary defaults.

Explicit flags always win, regardless of position or hyphen/underscore aliases.
For example, `--profile info --ticks 5000 --iter-per-gen 50` retains 5,000 ticks
and 50 updates instead of the preset's 150 and 20. The resolved settings are
printed at startup and saved in `gpu_run.json` and checkpoint options. Resume
still requires supplying the desired profile/options again.

```bash
uv run train_gpu.py --project mareld2.yaml --profile info --grid 16x16 --population-stability --currents on --run-name mareld-info-currents
```

World schedules take precedence over the preset world count. Profiles apply to
training; they do not change visualization evaluation settings. `benchmark_gpu.py`
continues to use its explicit benchmark options.

### Live visualization

Add `--visual` to open the same Pygame viewer used by CPU training:

```bash
uv run --locked train_gpu.py --project mareld2.yaml --profile info --visual --run-name mareld-gpu-visual
```

The viewer shows biomass heatmaps, reward/biomass/energy/action/loss plots, and playback of a fixed inference world. After each completed ARS update it copies the current unperturbed policies and frozen normalization statistics to a separate CPU probe. Probe evaluation preserves the training state and random streams. Reward plots show the actual training rewards; heatmaps show the probe ecosystem. Probe records are appended to `biomass.jsonl`.

Click the **progress** tab (or cycle tabs with Tab) for the survival graph across the entire training run. This is included in `--visual` on both trainers. The x-axis is completed training updates; the y-axis is consecutive inference ticks within the biomass bounds. The tab retains all evaluations, independently of probe playback and the other tabs' rolling buffers.

Progress evaluates the initial policies and every 20 updates by default, using a separate fixed scenario with a 1,000-tick cap, biomass bounds of 0.3–3 times the starting biomass, and temperature 1. Change these with `--eval-every`, `--eval-ticks`, `--biomass-bounds`, `--eval-seed`, and `--eval-temperature` directly on the training command. Probe sliders do not change this comparison scenario. The viewer stays responsive during evaluation.

History is saved to `progress/survival.jsonl` and reloaded with `--resume` in the same run directory; abandoned evaluations beyond the checkpoint are removed. Keep evaluation settings unchanged when resuming, or use a new `--plot-dir`. Histories belong to individual named runs.

Biomass and spawn controls change the probe world. The rollout-length slider changes the probe length; the separate `n_eval_ticks` slider changes training horizons between iterations. Closing the window (or pressing Q/Esc) disables visualization and training continues. Viewer failures also leave training running. Pygame events stay on the main thread while a single worker runs each GPU update, keeping the window responsive during compilation and execution. Visualization adds CPU probe time and a GPU synchronization boundary per update.

The window requires a desktop display on the machine running training.

`results/mareld-gpu/` contains `trainer.pth` for complete resume, `policy_<species>.pth` files compatible with the existing inference/CPU-training loaders, `gpu_run.json` with configuration metadata, and `training.jsonl` with compact numerical diagnostics. For a non-default network, pass the matching `--policynetwork LAYERS NODES ACTIVATION` to existing inference tools. The CPU batched-policy path now honors sigmoid/ReLU/tanh consistently with individual policies; previously it always used sigmoid.

Resume with the same numerical/configuration options and output directory:

```bash
uv run --locked train_gpu.py \
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
uv run --locked train_gpu.py --project mareld2.yaml --device cpu --execution eager \
  --grid 6x6 --n-deltas 2 --worlds 2 --ticks 5 \
  --generations 1 --iter-per-gen 2 --output /tmp/eco-evo-smoke
uv run --locked benchmark_gpu.py --project mareld2.yaml --grid 6x6 \
  --n-deltas 2 --worlds 2 --ticks 5 --workers 2 \
  --backends cpu tensor-cpu --warmup 1 --repeats 2 \
  --output /tmp/eco-evo-benchmark.json
```

No GPU speedup is claimed from these CPU-only checks. The meaningful performance result is the synchronized comparison on the target NVIDIA machine.
