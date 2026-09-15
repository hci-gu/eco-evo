# Eco-evo

Ecosystem simulation and ARS co-evolution training, with the original CPU runner and a batched GPU implementation.

To experiment with 10%–300% population bounds and failure penalties, see
[Population stability training](POPULATION_STABILITY.md).

Use `--currents on` for seeded passive movement of non-decision makers during
training or inference. See [random currents](CURRENTS.md) for tuning.

## Run with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run commands from the repository root. Use uv 0.5.9 or newer.

```bash
uv sync --locked
uv run --locked python -m pytest -q
```

uv manages Python 3.12, the virtual environment, and dependencies from `uv.lock`. The initial `uv sync` is optional: `uv run` also creates and syncs the environment. There is no separate pip installation or environment activation step. `pyproject.toml` replaces the requirements files as the dependency source.

Linux and Windows automatically install Torch from the official **CUDA 12.8** index. An NVIDIA GPU and a compatible driver are required for GPU execution; the CPU runner works with the same Torch installation. The first download includes the CUDA runtime and can be large. macOS installs native Torch from PyPI for CPU validation (Intel Macs use Torch 2.2.2). The GPU trainer requires CUDA; it does not fall back to CPU automatically. The platform-specific index configuration follows [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Benchmark CPU against GPU

On the NVIDIA machine (Linux recommended):

```bash
uv run --locked python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
uv run --locked benchmark_gpu.py \
  --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 \
  --workers 0 --warmup 2 --repeats 5 \
  --execution cuda-graph --output benchmark.json
```

See [GPU_TRAINING.md](GPU_TRAINING.md) for benchmark interpretation, VRAM controls, profiling, training, and resume options.

## Entry points

All existing scripts run through the same environment:

```bash
uv run train_gpu.py --project mareld2.yaml --generations 10 --run-name mareld-gpu
uv run train.py --project mareld2.yaml
uv run inference.py --help
uv run simple_inference.py
uv run manual_play.py
uv run api.py --host 127.0.0.1 --port 8000
uv run fgconfig/fgconfig.py
```

Interactive scripts require a graphical desktop. The configuration editor also requires Tk support in the Python installation; on Linux this may require your distribution's Tk package. Use `--help` on the training and inference runners to see their arguments.

## Simple training progress graph

Both trainers include a **progress** tab in `--visual`. Click it (or cycle tabs with Tab) to view consecutive inference survival ticks against completed training updates, using the same species colors and controls as the other live plots:

```bash
uv run --locked train_gpu.py --project mareld2.yaml --profile info --visual --run-name mareld-gpu
# For CPU training, use train.py with the same flags.
```

The tab keeps the entire named run's history, including previous sessions loaded with `--resume`. New probe rollouts and playback do not reset it, and early evaluations are retained beyond the other plots' rolling window. History and evaluation settings are stored in `<run>/progress/`. The fixed evaluation scenario is independent of the viewer's probe sliders, so measurements remain comparable throughout training.

The defaults are an initial evaluation, then one every 20 updates, with a 1,000-tick cap and bounds of 0.3–3 times each species' starting biomass. Pass `--eval-every`, `--eval-ticks`, `--biomass-bounds`, `--eval-seed`, `--eval-temperature`, or `--plot-dir` directly to either trainer to customize progress. Changing evaluation settings requires a new `--plot-dir`.

The output defaults to `results/<run-name>/progress/` (or `<--output>/progress/` for the GPU trainer). Set `--plot-dir PATH` to choose another folder. It contains:

- `survival.jsonl`: the persistent history, including evaluation number, training step, measurements and initial biomass for each decision maker, plus the actual inference length (`ticks_run`).
- `config.json`: the evaluation settings used for this history.

A baseline is saved as evaluation 1 before training starts, followed by evaluations at multiples of `--eval-every`. One training step means one completed optimizer update: a joint update in co-evolution mode, or a single-species update in round-robin mode. For example, with `--eval-every 20`, evaluations 1, 2 and 3 correspond to training steps 0, 20 and 40. The graph includes only decision-making species with policies. Non-acting groups such as phytoplankton remain in the ecosystem but are excluded from the graph, including when resuming older histories.

For each species, `B0` is its total biomass at the start of the inference rollout. The bounds are inclusive: `0.3 × B0 <= B(t) <= 3.0 × B0` by default. A breach on tick 1 scores 0; a breach on tick 10 scores 9; staying inside the bounds for the full rollout scores `--eval-ticks`. Recovery after a breach does not restart the counter. Nonfinite biomass counts as a breach. Groups with zero starting biomass are excluded from the graph and recorded as `null`.

Inference stops as soon as every decision maker has left the biomass bounds, even if non-acting groups are still alive. For example, `--eval-ticks 5000` is a maximum: if the last decision maker breaches its bounds at tick 230, evaluation stops at tick 230 and training resumes. The console reports the actual number of ticks used.

Evaluation uses the existing **CPU inference simulation for both trainers**, with copied current weights and frozen observation-normalization statistics. It uses the project's `inference_initial_biomass` values and the training grid, mortality and migration settings. Each evaluation reuses the same world and environmental-noise seed (`--eval-seed`, default `20260530`) and a fixed softmax temperature (`--eval-temperature`, default `1.0`). Training randomness and statistics are preserved. This is one repeatable inference scenario; reaching the tick cap means the species stayed in range for that horizon. Longer horizons add time to each training pause.

To resume, use the same command with `--resume`. The graph history continues, with a fresh evaluation of the loaded checkpoint; records beyond the resumed step are discarded. GPU step numbering comes from its complete trainer checkpoint. CPU numbering follows the existing trainer's generation-based resume logic, so keep `--iter-per-gen`, co-evolution mode and target species unchanged when continuing a CPU graph. Use a new `--plot-dir` when changing evaluation settings or starting fresh in an existing run folder.

## Dependencies

Use `uv add PACKAGE` for runtime dependencies or `uv add --dev PACKAGE` for development tools. Commit both `pyproject.toml` and `uv.lock` when dependencies change. To intentionally refresh locked versions, run `uv lock --upgrade`, then validate with `uv run --locked python -m pytest -q`. Benchmark commands use `--locked` so a stale lockfile fails instead of silently changing the environment.
