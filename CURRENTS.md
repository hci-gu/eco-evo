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

In `train_progress.py`, put the current flags after `--`. Training probes and
progress evaluations automatically inherit the settings. Standalone inference
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

This is a simple bulk flow: one vector shared by all non-decision makers in a
world, interpolated between random targets. It transfers biomass and its energy
reserve to north/east/south/west neighbours before population growth. It does not
apply to decision makers. All non-decision makers participate, including benthic
groups; this is an experiment in moving food availability, not a hydrodynamic
model or a distinction between drifting and attached organisms.

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
