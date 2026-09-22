# Cloud-driven currents

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
| `--current-period` | `20` | Ticks to scroll the cloud field one cell along each of +X and +Y |
| `--current-seed` | `0` | Seed mixed with each rollout's seed |
| `--current-scale` | `12` | Coarse noise lattice spacing in cells, at least 1 |

For example, `--currents on --current-strength 0.2 --current-period 10
--current-scale 12` gives up to 20% biomass transport per tick, with clouds
scrolling 0.1 cells/tick along each axis. Strength and texture scrolling speed
are independent. Strength is a fraction per simulation tick, not a physical
speed. Underscore aliases are accepted for the tuning flags.

The field is smooth, seeded value noise, mixing a coarse layer (75%) and a layer
at half the spatial scale (25%). Sampling coordinates are
`((x - tick / period) / scale, (y - tick / period) / scale)`: the cloud pattern
scrolls continuously towards +XY without regenerating or integer-shifting a
texture. In array coordinates +X is east and +Y is south.

Every cell shares its field value across responding species. For noise value
`cloud` in [0, 1], the outgoing fraction is
`strength * cloud * current_response`, split equally east and south. The rest
stays in place. Biomass and its energy reserve use identical fractions, without
an active swimming cost. This step runs after active movement and before growth,
including in worlds without decision makers.

Set `current_response` in the species definition in `fgconfig/fg_library.yaml`
to select susceptibility: 0 anchors the species, 1 gives full drift, and values
between scale transport proportionally. Missing values default to 1 for backward
compatibility with other non-decision-maker groups. The supplied library sets
`phytoplankton: 1.0` and `benthic_community: 0.0`. Decision makers are excluded
regardless of this parameter; zooplankton is a decision maker in this project.

This is a spatially varying diagonal drift model, not a hydrodynamic solver.
It varies strength, not direction, and does not generate eddies. Persistent
east/south drift can accumulate biomass at downstream closed boundaries when
migration is off. Use `--currents on --migration on` to recirculate boundary
outflow through the same immigration mechanism as active movement.

The transport step conserves each group's biomass and reserve and prevents
transport into inaccessible cells. With `--migration off` (the default), material
stays at closed grid boundaries. With `--migration on`, outgoing biomass and
energy re-enter along the boundary using the same habitat-weighted immigration
and minimum-split concentration rules as swimmers. This redistributes emigrants
across eligible boundary cells; it does not wrap them to the opposite cell.
Internal habitat barriers remain closed. Existing growth, predation, habitat
masks, and extinction rules still operate normally and can change total biomass
elsewhere in the tick.

Current randomness is independent of the ecology's random stream. Both signs of
each ARS perturbation see identical currents; world batching/chunking does not
change them. Identical current seeds, world keys and ticks give the same flow on
CPU and GPU, but the trainers have different world-key generation schemes. GPU
flow generation stays on the device and works with fixed-shape graph execution.

The noise and CPU transport live in `lib/environments/ecosystem_env/movement.py`.
`currents.py` retains configuration, CLI arguments and compatibility exports;
`TensorEcosystem.advect` samples the same field on the device. Coordinates are
cached, while the field is a pure function of coordinates, seed and tick. There
is no evolving random texture state to reset between ARS perturbations.

## Change from the former bulk-current model

Enabling currents now selects the cloud field instead of one world-wide random
vector. `--current-period` now means ticks per cell of scrolling, rather than
ticks between random vector targets. Existing CLI commands and imports still
work, but old current-enabled trajectories are intentionally not reproduced.
The observation layout is unchanged, so old policies still load; evaluate or
retrain them for the changed food transport. Currents remain off by default.
