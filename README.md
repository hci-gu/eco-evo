# eco-evo-pop

`eco-evo-pop` is a research-oriented ecosystem simulator and training sandbox for multi-species fish population dynamics. The active training path uses a custom PettingZoo environment plus a simple NumPy evolutionary algorithm to train per-species policies that move, forage, and survive on a spatial map of the Baltic Sea.

The project also contains older Stable-Baselines3 and bioMARL experiments. Those files are still useful for reference, but the default path for new development is:

1. `main.py`
2. `lib/runners/petting_zoo.py`
3. `lib/environments/petting_zoo.py`
4. `lib/world/*`
5. `lib/config/*`

## What a new contributor should know first

- The project is not packaged as a library. Work from the repo root and run scripts directly with `python ...`.
- The main training loop is evolutionary, not PPO.
- The most important configuration surface is the frozen `Settings` dataclass in `lib/config/settings.py`.
- Training configs are plain `key=value` text files loaded by `load_settings(...)`.
- Result folders are derived from the config file basename only. If two configs in different folders share the same filename, they will write to the same `results/<basename>/` directory.
- Best agents are saved with `Model.save(...)`, which uses `np.savez(...)`. In practice, files passed as `something.npy` end up on disk as `something.npy.npz`.

## Environment setup

The repository already includes a Conda/Micromamba environment file.

```bash
micromamba create -f env.yml
micromamba activate eco-evo-mixed
```

Main dependencies:

- Python 3.11
- NumPy, SciPy, Numba
- PettingZoo, Gymnasium
- Stable-Baselines3
- PyTorch
- Matplotlib, Seaborn, Pygame
- Flask for the HTTP API

If you prefer Conda instead of Micromamba, `conda env create -f env.yml` should work as well.

## Quick start

### 1. Run a real training config

The easiest way to get started is to run one of the existing bundled configs:

```bash
python main.py --config_folder runs/2026-03-05_signal_v3_unstick
```

That folder currently contains one config file, so this runs one training job.

Outputs will be written under:

```text
results/2026-03-05_signal_bundle_v3_unstick/
```

Look for:

- `agents/`: best saved agents per acting species
- `generations_data.json`: summarized generation metrics
- generated plots from `lib.visualize.plot_generations(...)`

### 2. Run a very fast smoke-test training

For debugging, create a small one-off config file with values like:

```text
world_size=8
num_workers=1
enable_logging=True
enable_plotting=False
generations_per_run=3
num_agents=4
agent_evaluations=2
elitism_selection=2
tournament_selection=2
mutation_rate=0.18
mutation_rate_decay=0.998
mutation_rate_min=0.05
fitness_method=biomass_pct
biomass_fitness_scope=agent
fitness_eval_steps=20
two_stage_eval_enabled=False
relative_baseline_enabled=False
age_groups=3
age_step_interval=20
```

Then run:

```bash
mkdir -p runs/local_smoke
$EDITOR runs/local_smoke/local_smoke.txt
python main.py --config_folder runs/local_smoke
```

Notes:

- `world_size <= 9` uses an all-water debug map in `read_map_from_file(...)`.
- Smaller `num_agents`, `agent_evaluations`, and `generations_per_run` reduce runtime dramatically.
- Disable plotting for faster local iteration.

### 3. Play as an agent manually

The fastest way to understand the environment is to control one species yourself:

```bash
python start_manual_agent.py \
  --species cod \
  --render-mode human
```

Useful options:

- `--species cod` controls the final cod age group by default
- `--control-all-age-groups` controls every age group for a base species
- `--max-steps 50` limits the run
- `--map-seed 123` makes the reset deterministic

Manual play writes CSV and plot outputs under `results/manual_play/`.

### 4. Browse helper scripts

There is a small script launcher:

```bash
python run_script.py --list
```

This exposes diagnostics and evaluation helpers such as:

- `play_as_agent`
- `biomass_stability`
- `energy_sanity_check`
- `fish_biomass_eval`
- `eval_multi_species`

## Default developer workflow

For most contributions, this is the shortest path:

1. Start with an existing config in `runs/`.
2. Run training through `main.py`.
3. Inspect saved agents in `results/<config-name>/agents/`.
4. Use `start_manual_agent.py` to understand behavior qualitatively.
5. Use one of the scripts in `scripts/` for targeted diagnostics.
6. Only move into the SB3 or bioMARL code if you specifically need those experiments.

## Project structure

```text
api.py                         Flask API for uploaded maps and simulation playback
main.py                        Main evolutionary training entry point
run_script.py                  Small launcher for scripts/
start_manual_agent.py          Shortcut to manual interactive play
env.yml                        Conda/Micromamba environment
data.csv                       Historical biomass + fishing data
runs/                          Training config bundles
maps/baltic/                   Default map.png + depth.png
agents/                        Agent-set folders consumable by the API
lib/config/                    Global settings and species definitions
lib/model.py                   Lightweight NumPy policy network
lib/runners/petting_zoo.py     Main training/evaluation runner
lib/environments/petting_zoo.py Main multi-agent environment
lib/world/                     Map loading and world update logic
lib/visualize.py               Pygame + Matplotlib visualization
scripts/                       Diagnostics, evaluation, and exploratory tooling
```

## How training works

### High-level flow

`main.py`:

1. reads every `.txt` file in `--config_folder`
2. loads `Settings` from each file
3. creates a `PettingZooRunner`
4. calls `runner.train()`

`PettingZooRunner.train()` repeatedly:

1. evaluates every candidate in every species population
2. logs generation metrics and plots
3. evolves the population with elitism, tournament selection, SBX crossover, and mutation
4. saves new best agents to `results/.../agents/`

### Populations and species

The active species are defined from base species plus optional age groups:

- Base species: `plankton`, `sprat`, `herring`, `cod`
- Acting base species: `sprat`, `herring`, `cod`
- Plankton is hardcoded and not learned

With `age_groups=3`, the acting population becomes:

- `sprat__a0`, `sprat__a1`, `sprat__a2`
- `herring__a0`, `herring__a1`, `herring__a2`
- `cod__a0`, `cod__a1`, `cod__a2`

Age-group expansion happens in `lib/config/const.py`, and model channel offsets are rebuilt in `lib/model.py`.

### Candidate model

`lib/model.py` defines a very small fully-connected NumPy policy:

- input: flattened 3x3 neighborhood patch
- hidden size: `48`
- output: softmax over 5 actions

Actions are:

- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`
- `EAT`

The policy outputs action probabilities per cell, not a single global action.

### Fitness modes

The runner supports three fitness families:

- `simple`: reward by survival length
- `biomass_pct`: percent biomass change over a short horizon
- `trajectory_shaped`: dense short-horizon reward based on biomass delta, energy delta, crash penalties, and terminal biomass

The current configs in `runs/` are using the biomass-oriented short-horizon setup rather than simple survival.

### Evaluation tricks used during training

`PettingZooRunner.evaluate_population()` includes several important stabilizers:

- paired opponent evaluation: all candidates in a generation see the same opponent draw for a given evaluation slot
- opponent snapshots: other-species populations can be frozen for several generations
- same-base teammate locking: age groups from the same base species can be aligned by index
- two-stage evaluation: short pass for all candidates, long pass for top candidates
- relative baseline scoring: candidate score can be adjusted against a matched random-policy baseline
- early winner short-circuiting for the final candidate in `simple` mode

If training behavior changes, this file is the first place to inspect.

## Environment and world model

### World tensor layout

The simulation world is a dense tensor with channels for:

- terrain one-hot: land, water, out-of-bounds
- biomass per species
- energy per species
- smell per species

Channel offsets are stored in `lib.model.MODEL_OFFSETS`.

There is also a separate `world_data` tensor used for auxiliary per-cell state such as:

- current direction
- plankton cluster markers
- plankton respawn counters
- depth from `depth.png`
- a scratch/bookkeeping channel

### Observation model

Each acting species receives a local 3x3 patch around every interior cell. The runner flattens that patch before feeding it to the model.

Conceptually, each decision is:

1. observe a normalized local 3x3 neighborhood
2. emit 5 action probabilities for that cell
3. apply movement and/or eating at the cell level

### Step order

The main environment is an AEC PettingZoo environment in `lib/environments/petting_zoo.py`.

Within a cycle:

1. agents act one after another in randomized order
2. plankton uses hardcoded spawning/growth logic
3. fish movement, metabolism, eating, and death are applied
4. smell may be updated
5. after the last agent in the cycle:
   - age transitions are applied
   - offspring spawning is applied
   - the agent order is reshuffled

### Termination

The world ends when a base species biomass falls outside the configured alive bounds:

- below `min_percent_alive * initial_biomass`
- above `max_percent_alive * initial_biomass`

This check is implemented in `lib/world/update_world.py::world_is_alive(...)`.

## Map loading and initialization

`lib/world/map.py` loads two images from a map folder:

- `map.png`: terrain classes
- `depth.png`: grayscale depth values

The default folder is `maps/baltic/`.

Important implementation details:

- maps are resized to `world_size x world_size`
- for `world_size <= 9`, the code bypasses the image and generates an all-water debug map
- current training uses `add_species_to_map_even(...)`, not the older noise-based placement
- species biomass is initialized across valid water cells and smell channels are reset to zero

## Biology and update rules

The biological parameters live in `lib/config/species.py`.

Each species/age-group has:

- starting biomass
- biomass bounds
- energy costs
- mortality parameters
- reproduction frequency
- growth rate
- prey list
- age transition metadata

Core update logic lives in `lib/world/update_world.py`:

- `all_movement_delta(...)`: movement plus metabolic and mortality effects
- `apply_movement_delta(...)`: applies biomass and energy movement results
- `matrix_perform_eating(...)`: prey consumption and predator energy gain
- `apply_age_transitions_table(...)`: continuous age progression
- `spawn_offspring_table(...)`: mature age groups create offspring
- `world_is_alive(...)`: global termination check

Plankton behavior is intentionally special-cased in `lib/world/plankton.py` with logistic regrowth plus respawn delays.

Smell diffusion and decay live in `lib/world/smell.py`.

## Configuration system

The config system is intentionally simple:

- one config file = one `key=value` text file
- loading is handled by `load_settings(...)`
- output folder is `results/<config-file-basename>/`

Examples of high-impact settings:

- world and runtime:
  - `world_size`
  - `max_steps`
  - `num_workers`
  - `enable_logging`
  - `enable_plotting`
- evolution:
  - `num_agents`
  - `agent_evaluations`
  - `elitism_selection`
  - `tournament_selection`
  - `mutation_rate`
- fitness:
  - `fitness_method`
  - `biomass_fitness_scope`
  - `fitness_eval_steps`
  - `two_stage_eval_enabled`
  - `relative_baseline_enabled`
- training curriculum:
  - `training_initial_energy_scale`
  - `training_energy_decay_per_cycle`
  - `training_plankton_cell_fraction`
  - `training_non_plankton_cell_fraction`
  - `training_cod_spawn_cells`
  - `training_cod_prey_clear_radius`
- age structure:
  - `age_groups`
  - `age_step_interval`

Config parser caveats:

- no inline comments
- every line must contain exactly one `=`
- unknown keys will fail when `load_settings(...)` tries to access the corresponding `Settings` field

## Evaluation and diagnostics

### Manual inspection

Use `start_manual_agent.py` or `scripts/play_as_agent.py` to play a species and watch biomass/fitness changes over time.

### Batch diagnostics

Useful scripts include:

- `scripts/short_horizon_feasibility.py`: checks whether short-horizon biomass objectives are achievable at all
- `scripts/energy_sanity_check.py`: sanity-checks energy flow and mortality behavior
- `scripts/biomass_stability_test.py`: perturbation/stability diagnostics
- `scripts/test_fitness_breakdown.py`: inspect shaped-fitness component contributions
- `scripts/test_fitness_spread.py`: inspect whether selection signal is collapsing
- `scripts/test_dynamic_eating.py`: quick policy response check
- `scripts/test_agent_outputs.py`: inspect action distributions from random models

### Historical data tools

- `data.csv` contains yearly biomass and fishing values
- `estimate_baseline.py` computes a naive carry-forward baseline
- some older evaluation scripts compare simulations against historical trajectories

Not all historical evaluation scripts are on the main path today, but the data file is still important context.

## HTTP API

`api.py` exposes a small Flask server for:

1. uploading a map (`map.png` + `depth.png`)
2. running a simulation on that uploaded map
3. retrieving sampled biomass snapshots
4. listing available agent sets under `agents/`

Start it with:

```bash
python api.py
```

Then read:

- `API_DOCS.md`

Agent set behavior:

- if an agent-set folder contains one `.npz` model, that model is reused for all acting species
- if it contains multiple files, filenames must let the API infer which acting species each file belongs to

## Legacy and experimental code paths

You will see several older stacks in the repo:

- `lib/runners/rl_runner.py`
- `lib/environments/sb3_wrapper.py`
- `train_and_plot.py`
- `train_multi_species.py`
- `train_global.py`
- `lib/environments/pbm_gym.py`
- `lib/environments/bioMARL/*`

These are useful if you are comparing approaches or reviving earlier work, but they are not the current default contributor path.

## Contributing effectively

If you are making changes, start by deciding which layer you are touching:

- config and experiment behavior:
  - `lib/config/settings.py`
  - config files in `runs/`
- biological rules:
  - `lib/config/species.py`
  - `lib/world/update_world.py`
  - `lib/world/plankton.py`
- observation/action semantics:
  - `lib/model.py`
  - `lib/environments/petting_zoo.py`
- training/evolution behavior:
  - `lib/runners/petting_zoo.py`
- rendering/debugging:
  - `lib/visualize.py`
- API integration:
  - `api.py`

Recommended contributor loop:

1. make a change
2. run a tiny smoke config
3. manually inspect behavior with `start_manual_agent.py`
4. run one or more targeted scripts from `scripts/`
5. only then launch a longer training run

## Common gotchas

- Training result folders collide on config basename, not full path.
- Best-agent files are saved as `.npy.npz`, not plain `.npy`.
- `plankton` is hardcoded and should not be treated like a learned policy.
- Age groups change both species naming and tensor channel layout.
- Tiny worlds behave differently because map loading switches to an all-water debug mode.
- Some scripts in the repo target older code paths. If imports look stale, verify whether the script is part of the current PettingZoo runner workflow before debugging it deeply.

## Suggested reading order

If you want to fully understand the current system, read files in this order:

1. `main.py`
2. `lib/config/settings.py`
3. `lib/config/species.py`
4. `lib/model.py`
5. `lib/environments/petting_zoo.py`
6. `lib/world/map.py`
7. `lib/world/update_world.py`
8. `lib/runners/petting_zoo.py`
9. `lib/visualize.py`
10. `api.py`

That sequence matches the actual runtime path of training and evaluation.
