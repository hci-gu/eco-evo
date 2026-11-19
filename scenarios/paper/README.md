# Paper scenarios

Inference runs with trained policies (no training) for the paper's figures.
Each YAML file here is one scenario; `lib/scenarios/spec.py` documents every key.
The runs step the world on the tensor engine (`lib/gpu`): CUDA on a GPU machine,
the CPU for small local checks. Nothing is written back to `mareld2.yaml` or the
library.

| File | Figure | Variants x seeds | Years |
|---|---|---|---|
| `current_fishing.yaml` | today's fishing kept constant | 1 x 5 | 100 |
| `no_fishing.yaml` | no fishing (after a warm-up with today's) | 1 x 5 | 100 |
| `fishing_constant.yaml` | harvest curve, effort k x today's | 9 x 3 | 100 |
| `fishing_rebuild.yaml` | 5 years at r x today's, then k x today's | 21 x 3 | 100 |
| `habitat_trawling.yaml` | bottom trawling b x today's | 6 x 3 | 50 |
| `warming.yaml` | + dT on every layer, Q10 relative to today | 5 x 3 | 50 |
| `eutrophication.yaml` | nitrate reservoir x n | 5 x 3 | 20 |
| `windfarm_poseidon.yaml` | noise, trawling ban, both (placeholder footprint) | 4 x 5 | 50 |

## Running

```bash
# On the GPU machine (all scenarios; finished runs are skipped)
uv run tools/paper/run_scenarios.py scenarios/paper/*.yaml --device cuda

# What would run, and what is done
uv run tools/paper/run_scenarios.py scenarios/paper/*.yaml --dry-run

# Local smoke test: small grid, short, one seed
uv run tools/paper/run_scenarios.py scenarios/paper/*.yaml \
    --grid 12x12 --days 5 --warmup-days 2 --seeds 0 --out results/paper_smoke
```

The defaults are the trained run's world: `run_name: 2026-11-17_1`, a 60 x 60 grid,
mortality, migration and currents on (as `inference.py --migration on --mortality on
--currents on`), and the run's training warm-up (60 days). `--grid`, `--years` /
`--days`, `--seeds`, `--run-name` and `--warmup-days` override them for every
scenario.

Output: `results/paper/<scenario>/<variant>/seed_<n>/` with `timeseries.npz` (daily
biomass, reserve, juveniles, cumulative losses by cause, imports, and catch per
impact), `spatial.npz` (biomass maps every `spatial_every_days`) and `meta.json`.

## Figures

The notebooks in `notebooks/paper/` read those directories. They do not simulate.

```bash
uv sync --group paper
uv run --group paper marimo edit notebooks/paper/02_optimized_fishing.py
# another results root:
PAPER_RESULTS=results/paper_smoke uv run --group paper marimo edit notebooks/paper/
```

Each notebook has a "Save figures" checkbox that writes PDF and PNG files to
`figures/paper/`.

## What the scenarios do and do not change

* **Fishing.** The impact fields are today's EMODnet effort maps, multiplied by a
  piecewise-constant intensity per phase (`schedule`). `fishing` stands for both
  trawl impacts. The library tables are linear up to 100 (bottom) and 10 (pelagic)
  MW-h/km2/yr, far above the maps' maxima (9.6 and 0.43), so even k = 4 is not
  clipped. The catch is the engine's booked impact loss. It is split per impact by
  each impact's expected share at the start of each day, which is exact for a
  group fished by only one impact.
* **Open population.** Recruits are imported, and planktivores, the water column,
  mackerel and krill relax towards fixed densities outside the box. Rebuilding and
  the yield curve are therefore the box's local response, not the stocks'.
* **Habitat degradation.** Bottom trawling is a mortality on the benthos and the
  demersal fish. It does not lower the beds' carrying capacity, so there is no
  habitat-loss mechanism yet.
* **Warming.** This is a step change. A uniform offset on the temperature tables
  would leave every Q10 that is normalised to an annual mean of 1 unchanged
  (ingestion, growth, and metabolism with `t_ref: annual_mean`). So the scenario
  multiplies each table by q10^(dT/10), relative to today's climate
  (`runner.apply_warming`). Endotherms, producers, nutrients and stratification are
  not warmed.
* **Wind farm.** `maps/poseidon_placeholder.npz` holds an arbitrary footprint and
  noise field (`tools/paper/make_windfarm_placeholder.py`); replace it with
  Poseidon's outline. In the library, noise acts only on the porpoises' energy cost
  (`porpoises_impacted_by_windfarm_noise`). The policies do not observe it, because
  they were trained without it. Inside the farm the trawl effort is removed, not
  moved elsewhere.
* **Long horizon.** The policies were trained on one-year episodes without fishing.
  100 years is far outside that, so check that `no_fishing` and `current_fishing`
  stay stable before reading the sweeps.
