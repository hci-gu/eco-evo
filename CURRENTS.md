# Random currents

Enable the same option in `train.py`, `train_gpu.py`, `benchmark_gpu.py`, or
`inference.py`:

```bash
--currents on
```

To combine it with population stability, add these to your training command:

```bash
--population-stability --currents on --run-name mareld-stability-currents
```

With `--visual`, training probes and progress evaluations automatically inherit
the current settings. Standalone inference
and resumed training need the flags supplied again.

| Option | Default | Meaning |
| --- | --- | --- |
| `--currents` | `off` | Enable with `on` |
| `--current-strength` | `0.1` | Maximum fraction transported out of each cell per tick, from 0 to 1 |
| `--current-period` | `20` | Ticks between random target flow vectors |
| `--current-seed` | `0` | Seed mixed with each rollout's seed |

For example, `--currents on --current-strength 0.2 --current-period 10` gives
stronger, more frequently changing flow. Strength is a fraction per simulation
tick, not a physical speed. Underscore aliases are accepted for the tuning flags.

This is a simple bulk flow: one scrolling cloud field shared by a world, sampled
per cell. It transfers biomass and its energy reserve to north/east/south/west
neighbours before population growth. This is an experiment in moving food
availability, not a hydrodynamic model.

## Which groups the field carries

Participation is per functional group, not per kind. Each FG has a
`current_response` in `[0, 1]` -- the share of the flow that carries it, scaling
its biomass flux and the energy that follows it together. The FG editor exposes
it as one field, "External Forces (wind/currents; share carried, 0=off)".

Decision makers can respond too: a swimmer with `current_response > 0` is
advected on top of the move its policy chose, in the same step, so wind and
currents act on fish as well as on plankton.

The default reproduces the behaviour that predated the per-group flag, so a
library written before it is unchanged: **non decision makers drift (1.0),
decision makers do not (0.0)**. In `fg_library.yaml` today, phytoplankton is at
1.0 and benthic_community at 0.0.

With `--local_reward`, the per-cell reward follows a drifting decision maker
correctly: the field's shares are composed into the movement shares, so the
end-of-tick energy is still credited to the cell the population started in.

The transport step conserves each group's biomass and reserve, retains material
at grid boundaries, and prevents transport into inaccessible cells. These closed
current boundaries also apply when swimmer migration is enabled. Existing growth,
predation, habitat masks, and extinction rules still operate normally and can
change total biomass elsewhere in the tick.

Current randomness is independent of the ecology's random stream. Both signs of
each ARS perturbation see identical currents; world batching/chunking does not
change them. Identical current seeds, world keys and ticks give the same flow on
CPU and GPU, but the trainers have different world-key generation schemes. GPU
flow generation stays on the device and works with fixed-shape graph execution.
