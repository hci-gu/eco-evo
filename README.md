# Eco-evo

Ecosystem simulation and ARS co-evolution training, with the original CPU runner and a batched GPU implementation.

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

## Dependencies

Use `uv add PACKAGE` for runtime dependencies or `uv add --dev PACKAGE` for development tools. Commit both `pyproject.toml` and `uv.lock` when dependencies change. To intentionally refresh locked versions, run `uv lock --upgrade`, then validate with `uv run --locked python -m pytest -q`. Benchmark commands use `--locked` so a stale lockfile fails instead of silently changing the environment.
