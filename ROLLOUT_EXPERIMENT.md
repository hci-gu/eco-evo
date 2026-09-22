# Rollout-length experiment

Run from the repository root on the NVIDIA machine:

```bash
uv run python tools/rollout_experiment.py --device cuda --rollouts 100 300 1000 --seeds 0 1 2 --generations 25 --output results/rollout-experiment
```

This runs **nine independent jobs, sequentially**, with fresh policies: three
rollout lengths times three training seeds. Each job runs 25 generations of
50 updates (1,250 ARS updates). Use `--seeds 0` for a quicker three-job pilot.
`--dry-run` prints the full argument lists without creating files or training.

The shared settings match the food-blob test: 24x24 torus, randomized blob
segments of 40-160 ticks, constant phytoplankton/benthic biomass, survival-first
reward (floor 0.3), mortality off, migration flag on (torus disables boundary
redistribution), 16 deltas, three worlds refreshed each iteration, profile info,
2x48 tanh networks, learning rate 0.03 and sigma 0.05. No visual viewer is
opened; this avoids adding random visual probes to experiment runtimes.

Progress is evaluated headlessly at initialization, every 50 updates and at
the final update, up to 5,000 ticks, within 0.1-10x initial biomass. Every job
uses evaluation seed 20260530 and temperature 1. The evaluation does not
consume training RNG state. This is a controlled fixed-world comparison, not
a multi-world generalization test. Training seeds and horizons are paired;
identical seeds do not guarantee identical trajectories after policies diverge.

## Outputs

Open `results/rollout-experiment/report.html` after/during the experiment.
It is rebuilt after every job, including failed/interrupted jobs:

- `comparison.png`: one panel per species, shared evaluation axes, all raw
  observations at 50% opacity. For each horizon, lines average matched
  checkpoints across participating seeds, then take the trailing five-point mean.
- Per-job `progress/progress.png`: all species, raw dots at 50% opacity plus
  a trailing five-evaluation mean. The same smoothing is used in the live
  CPU/GPU training viewer. Raw `survival.jsonl` measurements remain unchanged.
- `summary.csv` / `summary.json`: final, best and last-window mean scores,
  last-window cap-hit fraction, updates, wall time and candidate-world-ticks.
- `aggregate.json`: completed runs only, mean and population standard deviation
  across seeds of the last-window score. Failed/partial runs are not ranked.
- `manifest.json`: exact settings, source/config fingerprint, commands and status.
- Each job retains normal checkpoints, `gpu_run.json`, `training.jsonl` and
  `console.log`. Logs are redirected there rather than flooding the runner console.

Smoothing is **trailing**, includes the current point, and uses fewer points
at startup. `--progress-window 5` changes this. Missing biomass is missing data,
not zero survival. A score of 5,000 is capped; it says nothing about later ticks.
Best scores are descriptive, not the primary comparison: isolated peaks can
hide unstable behavior. Equal generation counts mean equal optimizer updates,
**not equal compute**: candidate-world-ticks are `2*deltas*worlds*updates*horizon`.
Wall time includes evaluation, startup and checkpointing.

## Resume and report regeneration

Append `--resume` to the **same command**. Completed jobs are skipped; partial
jobs resume from their last saved trainer checkpoint up to the original total
generation target (not 25 additional generations). Source/config changes are
rejected to avoid silently mixing experiments. A failure before the first
checkpoint requires a new output directory if progress files already exist.
Check `console.log` for failures; the runner continues other jobs and exits
nonzero if any job failed. Ctrl+C stops the experiment; on Windows a terminated
child may resume from the last generation checkpoint rather than its last update.

Regenerate images and the report without retraining:

```bash
uv run python tools/rollout_experiment.py --output results/rollout-experiment --report-only
```

The generic GPU trainer now also supports `--progress` without `--visual`,
saving its own headless progress history and image. CPU training's existing
live progress tab uses the same raw-dot/mean-line presentation.

Small CPU smoke run (not a meaningful training experiment):

```bash
uv run python tools/rollout_experiment.py --device cpu --rollouts 2 3 --seeds 0 --generations 1 --iter-per-gen 1 --eval-ticks 4 --grid 5x6 --n-deltas 2 --worlds 1 --output results/rollout-smoke
```
