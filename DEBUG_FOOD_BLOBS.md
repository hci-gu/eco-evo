# Bouncing food blobs: disposable training experiment

Enable `--debug-food-blobs` in `train.py`, `train_gpu.py`, or `inference.py`.
The active non-decision makers `phytoplankton` and `benthic_community` share a
compact, smooth blob. Its centre moves diagonally and reflects at the map
boundaries. The footprint keeps its shape instead of diffusing into the edges.
Each group's total biomass and reserve are restored to their rollout-start
values after every tick. Predation still feeds consumers, but cannot exhaust
the food source. Ordinary food growth, carrying-capacity limits, extinction
and current drift do not determine the resulting food field in this mode.
Decision makers retain their normal policies, energetics and rewards.

This is an artificial food-following task, not an ecological viability result.
The two food groups overlap to provide one clear target. No target position is
added to policy observations: agents must use their existing local observations.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--debug-food-blobs` | off | Enable the experiment |
| `--food-blob-speed` | `0.1` | Centre speed in cells/tick; `0` is a stationary control |
| `--food-blob-radius` | `0` | Radius in cells; `0` means one sixth of the shorter grid side, at least 1 |
| `--food-blob-seed` | `0` | Trajectory seed combined with the rollout seed |

Both CPU and tensor training use deterministic trajectories, with identical
food for the positive/negative ARS perturbations. Training probes, progress
evaluations, and the live viewer inherit the experiment settings. Food totals
come from the usual training/inference biomass configuration. Viewer biomass
sliders set new totals on the next rollout; food spawn-shape overrides are
superseded by the blob. Existing fixed colour scales make it easy to see motion.
`--currents` and `--migration` do not change the prescribed food trajectory.

Start a fresh experiment (using a new run name):

```bash
uv run python train.py --project mareld2.yaml --run-name food-blobs \
  --grid '30*30' --debug-food-blobs --food-blob-speed 0.1 \
  --n_eval_ticks 300 --generations 20 --iter-per-gen 10 --visual
```

Or use the GPU trainer on a CUDA machine:

```bash
uv run python train_gpu.py --project mareld2.yaml --run-name food-blobs-gpu \
  --grid '30*30' --debug-food-blobs --food-blob-speed 0.1 \
  --ticks 300 --generations 20 --iter-per-gen 10 --visual
```

View a saved model with the same experiment settings:

```bash
uv run python inference.py --project mareld2.yaml --run-name food-blobs \
  --grid '30*30' --debug-food-blobs --food-blob-speed 0.1 \
  --ticks 1000 --seed 17 --visual --rnd-baseline
```

Repeat the debug flags when resuming or running standalone inference. Policies
retain the usual checkpoint format; the run/progress metadata records the
experiment. The random baseline compares biomass/energy curves, while heatmaps
show the trained agents. A useful control is a separate training run with
`--food-blob-speed 0`. Zooplankton now defaults to movement speed 1.0 in the
shared species library (also outside debug mode), up from 0.02, so its maximum
movement can keep up with the default blob. Use a horizon long enough to
see movement and a bounce; the usual 15-tick training horizon is too short for
that comparison on large grids.

In training and inference viewers, click **Colors: fixed [C]** or press `C`
to switch between rollout-start colour limits (the default) and dynamic
per-species limits from the displayed frame. Dynamic colours reveal small
remaining populations but are not comparable across ticks; the colourbar shows
the current limit. This also works during recording and replay, with linear or
log colours. The biomass scale multiplier applies in both modes; switching
colour modes does not change the biomass line plot or simulation.

Internal habitat obstacles clip and renormalise the food footprint. If the
footprint is entirely blocked, food is placed at the nearest accessible cell(s).
The trajectory only reflects at the rectangular map boundary; it does not plan
routes around obstacles. A fully inaccessible map is rejected.

## Removal

The experiment lives in `lib/environments/ecosystem_env/debug_food.py`. Remove
its explicit hooks/config plumbing in the environment, tensor engine/runner,
builders, CLI and visual/progress builders, plus `tests/test_debug_food.py`.
It adds no library parameters, policy inputs, reward terms or mandatory state
to ordinary runs. Leaving the flag off keeps ordinary simulation behaviour.
