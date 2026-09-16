# Population stability experiment

Add `--population-stability` to either trainer. The mode is off by default. Use a new run name
to compare with existing training.

```bash
# CPU
uv run --locked train.py --project mareld2.yaml --profile sanity \
  --population-stability --run-name mareld-stability-cpu

# GPU
uv run --locked train_gpu.py --project mareld2.yaml --grid 60x60 \
  --n-deltas 16 --worlds 3 --ticks 150 --generations 10 --iter-per-gen 20 \
  --population-stability --run-name mareld-stability-gpu
```

Each world's reference is each decision maker's total biomass at rollout start.
All initially present decision makers are protected, including those not selected
for policy updates by `--species`. Initially absent groups and non-decision makers
do not trigger population bounds. A breach by any protected species fails the
whole world, so predators also receive a penalty when decision-making prey collapse.

| Option | Default | Meaning |
| --- | --- | --- |
| `--population-min` | `0.10` | Fail below 10% of starting biomass |
| `--population-max` | `3.0` | Fail at or above 300% |
| `--population-warning-min` | `0.20` | Lower warning starts below 20% |
| `--population-warning-max` | `2.5` | Upper warning starts above 250% |

Underscore aliases are accepted. Bounds must satisfy
`0 < min < warning-min <= 1 <= warning-max < max`.

Warning penalties rise linearly from 0 to 1 toward either failure boundary.
The largest warning across protected species is subtracted from every species'
tick reward. The base tick reward is `log((B*energy_content + R + eps)/(E0 + eps))`,
clipped to `[-3, 1]`. Fitness is the mean over the **original** rollout horizon.
The first failed tick and all remaining ticks receive `-5` for every species,
with no further energy reward. Later recovery cannot undo failure. For example,
one neutral tick followed by failure in a four-tick rollout scores `-15/4`.
This preserves a signal for delaying failure; each failed tick scores worse
than any valid tick. The existing entropy/argmax modifiers still apply afterward.

CPU rollouts stop immediately on failure. GPU worlds freeze and stop contributing
observation/action statistics after the failure tick, while fixed GPU graphs still
execute for the requested horizon; do not expect a GPU speedup from this mode.
GPU logs include `population_failure_fraction` (failed worlds) and
`population_valid_fraction` (fraction of horizon before failure). The mode requires
the default integral total-energy reward and rejects legacy/final-value reward flags.

With `--visual`, add `--biomass-bounds 0.1 3.0` to the training command to align
the progress graph’s displayed range with this experiment. The progress plot still measures each species
independently using inclusive bounds; training fails the whole world at the upper
boundary. Inference/probe simulations remain full ecological simulations.

When resuming, supply these flags again: recorded options do not automatically
override command-line settings. Starting a new run keeps comparisons easiest.
