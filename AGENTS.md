# AGENTS.md - Mareld / Poseidon Nord

Distilled project context for AI agents. Keep it short; the full history lives in
`mareld_resume.txt` (see "Deep context" at the bottom).

## 1. What this project is

Agent-based marine ecosystem simulator for the Mareld / Poseidon Nord offshore case.
Functional groups (FGs) live on a 60x60 grid (1 km/cell) and the
decision-making FGs are driven by small policy networks trained with ARS
(Augmented Random Search), co-evolution on by default.

The tick length is the `--tick-length` CLI flag (1-6 h, default 6;
sections 97, 103 and 106). The tick pipeline itself is
tick-agnostic - every library parameter is *per tick* - so the value
drives the tick <-> real-time conversions and, when it is not 6, an
in-memory conversion of every tick-dependent library parameter as the
FGs are built. Nothing is ever written back: `fg_library.yaml` has one
calibration and the FG editor can no longer change it. `lib/world/tick_time.py` is the single definition point; read it via the run's
`--tick-length` (or `LIBRARY_TICK_HOURS` when reading the library's own
numbers) rather than hardcoding 4 ticks/day anywhere.

`Strategi.pdf` defines the mathematics ("The Tick") and `Method.pdf` the
observation/neighbourhood conventions. `Strategi.pdf` is a **draft of an
initial model to develop further, not a specification to conform to** -
the code matching it is not by itself evidence that a term is right. See
section 114, where the growth term is faithful to the document and still
fabricates mass.

## 2. Language policy (hard rule)

- All code, comments, variable names, config keys, UI strings, prints and docs in **English**.
- **Chat answers and status updates to the user: Swedish.**
- Swedish display names only as an optional UI overlay (`sv_mapping` in `fgconfig.py`).
- `mareld_resume.txt` is **ASCII-only**; that restriction applies to that file only.
  Verify: `python3 -c "print(sum(1 for b in open('mareld_resume.txt','rb').read() if b>127))"` -> must print 0.

## 3. Repository map (essentials)

| Path | Role |
|---|---|
| `train.py` | ARS training entry point. Requires `--project`. Co-evolution default-on. |
| `inference.py` | Runs trained policies from `results/<run-name>/policy_<fg>.pth`. |
| `run_mareld_mvp.py`, `compare_scenarios.py` | MVP scenario run / "noll" vs "projekt" comparison. |
| `mareld2.yaml` | Project manifest (`--project` value). |
| `fgconfig/fgconfig.py` | Tkinter GUI for FGs, impacts, matrices, spawn strategies. |
| `fgconfig/fg_library.yaml` | Central biological library (species/interaction/impact definitions). |
| `lib/environments/ecosystem.py` | `EcosystemEnvironment` - "The Tick" pipeline, heavily optimised. |
| `lib/runners/trainer.py`, `parallel_worker.py` | ARS trainer (CRN, ARS-V2 obs-norm, top-b) + multiprocessing worker. |
| `lib/world/`, `lib/config/config_loader.py` | Grid, `FunctionalGroup`, project/library loading. |
| `lib/world/growth_budget.py` | Derives every FG's `growth_rate` from `r_max`, `natural_mortality` and the checked `predation_mortality` pairs; shared by loader, FG editor and tests (section 129). |
| `lib/world/daylight.py` | Daylight calendar (section 137): sun elevation from latitude and date, per-tick light level, and the per-pair attack-rate multiplier (annual mean exactly 1), and the light-limited producer growth factor (section 138: P-I curve over the mixed layer, 1 on the reference day in April). Off unless `simulation_settings.daylight.enabled` in the manifest; turning it on adds one observation channel (checkpoints invalid). The live viewer shows the month and a day-length frame around each heatmap (section 142). |
| `lib/world/temperature.py` | Water temperature on the daylight calendar (section 143): Q10 multiplier on each DM's `resting_metabolism` (`metabolism_q10`, `metabolism_t_ref` in the library; monthly layer temperatures and `group_layers` in `simulation_settings.temperature`). Needs daylight on; adds no observation channel. |
| `lib/world/tick_time.py` | Tick length: bounds (1-6 h), the `--tick-length` flag, tick<->real-time conversion, label rendering, and the per-parameter rescale rules the LOADER applies at runtime. `fg_library.yaml` is always calibrated at `LIBRARY_TICK_HOURS` (6) and is never rewritten for a run (sections 97, 103, 106). |
| `lib/spawn/`, `lib/viz/`, `tools/` | Spawn strategies, live pygame visualiser, offline plot/calibration tools. |
| `lib/environments/ecosystem_env/source_tracking.py` | `--local_reward`: per-cell source tracking, `reward(c)=B(c,t+1)/A(c,t)`. |
| `lib/diagnostics/viability.py`, `tools/viability.py` | Long-term viability rig - frozen behaviour, no ARS. A diagnostic for the mechanics, not the judge of the world: its verdict (normative corner `--spawn colocated --behaviour greedy`) is about that behaviour, and trained policies can be viable where it fails (`VIABILITY.md` section 6, resume 123). Two factors (behaviour x spawn geometry). The hand-coded arms allocate the eat mass by marginal energy return (water-filling), never evenly - section 92. Criterion in `VIABILITY.md` (sections 90, 91, 92). |
| `tests/` | pytest suite - keep green. |
| `results/<run-name>/` | Checkpoints `{'state_dict': ..., 'obs_stats': {...}}`. |

## 4. Functional groups

- Decision makers (have a policy net): `zooplankton`, `pelagic_fish`, `gadoids`,
  `porpoises`, `seals`, `seabirds`.
- Non decision makers (logistic growth toward carrying capacity):
  `phytoplankton`, `benthic_community`.
- English snake_case IDs are canonical.

## 5. The Tick (per tick, random FG order)

`_calculate_decisions` -> `_apply_impact_mortality` -> `_apply_predation`
-> `_apply_movement` -> `_apply_growth`

- Decisions: batched `torch.bmm` over all DMs; per-DM compact observation layout
  driven by the Observability matrix; `Rest` doubles as a hide action.
- Impacts: linear interpolation in `impact_table`, clipped to endpoints.
- Holling ceiling is 1/h (`energy_balance.intake_ceiling`); `max_intake_rate`
  is the attack rate a. A pair may override it (`<pred>_preys_on_<prey>:
  max_intake_rate`) to set the half-saturation 1/(a h) without moving the
  ceiling - zoo -> phyto uses 20 t/km2 (section 134).
- Daylight (opt-in, section 137): a pair with `dark_ratio` < 1 has
  `a(t) = a * m(t)`, m from the sun's elevation vs `light_threshold_deg`,
  normalised so the library `a` is the annual mean; every DM then also
  observes the light level (last input slot). Only herring -> zoo is set.
- Predation: vectorised, Holling type II with Beddington-DeAngelis crowding
  (`a_eff = a / (1 + a*h*B_prey_visible + w*B_pred)`), hidden (rested) fraction
  protected, energy gain buffered. `w` is the per-FG `interference` parameter
  (`interference: 0` = pure Holling; `gadoids: 1.0`, `pelagic_fish: 0.7` in the
  library, section 86). It is what removes the 2-cell vertical bands.
- Movement: slice-assign, per-action metabolic cost; energy follows biomass.
  With `simulation_settings.temperature` (section 143) the cost of a DM
  with `metabolism_q10` is `resting_metabolism * q10**((T - t_ref)/10)`,
  T from its layer's monthly table on the daylight calendar; herring
  t_ref is 14 C (the library value is a summer cost), gadoids and
  zooplankton `annual_mean`, porpoises none (endotherm). On in
  `mareld2.yaml`; attack rates are not temperature-scaled.
- Growth: NDM logistic; DM `s = R/(B*ME)`, `q = s - u` (`maintenance_level`, default 0.3).
  With the daylight calendar a producer with `light_saturation` grows at
  `growth_rate * P(t)/P_ref` (section 138; 0 at night, needs the site's
  light climate in the manifest). The old `seasonal_amplitude` /
  `seasonal_period` sine is gone.
- M1 can be exposure-weighted (section 139: `m1_visual_share`,
  `m1_tactile_share`, `depth_risk_ratio`): visual part follows the
  visible biomass and the light, tactile part is stronger on hidden
  biomass. Implemented on CPU and GPU but not set in the library - the
  viability gate in section 139.3 did not justify it.
- `prey_includes_reserve` (section 140): a prey DM's energy reserve is
  eaten with it; the static quality is taken at
  `reserve_reference_fill`. Set for `pelagic_fish` (section 140.6):
  lean 4000 MJ/t + reserve up to 7000, starvation from 0.2 of it.
- `functional_response: 2|3` on a predation pair forces the Holling type
  for that pair (section 141); without it a specialist gets III and a
  generalist II, as before. Set on gadoids -> pelagic_fish (141.4),
  where gadoids also eat zooplankton (krill proxy) and each other.
- `growth_rate` is DERIVED (`lib/world/growth_budget.py`, section 129): DM
  `g = (r_max + M1 + sum checked predation_mortality) / ((1-u)*1460)`,
  NDM `r = (r_max + sum M2) / 1460`. Checking/unchecking a predator in the
  FG editor moves the prey's growth by that predator's share; muted FGs are
  absent. With `--mortality off` the loader leaves M1 out of g (pass
  `apply_natural_mortality` to the loader wherever an env is built). Edit
  `r_max`, `natural_mortality` (M1, non-FG losses only) or the pair's
  `predation_mortality`, never `growth_rate` itself.
- Hunger gate `h = min(1, max(0, (1 - s)/(1 - u)))` for every FG: full
  appetite up to the maintenance level, zero at a full reserve (section 133);
  `satiation_scale` no longer exists (section 130).

## 6. Commands

```bash
# Train (mandatory --project). Non-interactive: the [Y/n] prompt blocks otherwise.
python3 train.py --project mareld2.yaml --run-name <name> < /dev/null
python3 train.py --project mareld2.yaml --species pelagic_fish \
    --generations 1 --iter-per-gen 1 --workers 1     # smoke test

# Tests and syntax check
python3 -m pytest tests/ -q
python3 -m py_compile fgconfig/fgconfig.py lib/environments/ecosystem.py \
    lib/runners/trainer.py lib/runners/parallel_worker.py train.py

# Mechanics diagnostic: frozen behaviour, no training. Exit 1 = the frozen
# greedy corner fails; judge viability with trained policies (VIABILITY.md 6).
python3 tools/viability.py                                  # ~4 min, see VIABILITY.md
python3 tools/viability.py --spawn configured               # geometry as drawn
python3 tools/viability.py --ticks 300 --seeds 1 --behaviour eat   # smoke test

# Can each DM close its energy budget over its RATION? Seconds, no rollout.
# "min share" = minimum share of the best prey in the diet; a pair that
# cannot pay on its own is low-quality food, not pure loss (section 92).
python3 tools/probes/budget_gate.py
# Monthly attack-rate multipliers and producer growth factors (137, 138).
python3 tools/probes/daylight_table.py

# GUI, inference, plots
python3 fgconfig/fgconfig.py
python3 inference.py --project mareld2.yaml --run-name <name>
python3 tools/biomass_html.py results/<run-name>      # -> plots.html
```

Useful flags: `--profile sanity|info|deep` (run-size presets; action hacks are off with or without one),
`--visual`, `--rollouts_per_delta`, `--mortality on`,
`--tick-length HOURS` (1-6, default 6; section 106),
`--mortality_multiplier FACTOR` (scales every FG's `natural_mortality`;
`keep = 1 - FACTOR*rate`, needs `--mortality on`, section 88),
`--no-mass-balance` (revert to the pre-116 growth term, which adds biomass
without debiting the energy reserve; the mass-balanced term is ON by
default since section 120 and the library is calibrated for it -
`--mass-balance` is kept as a no-op, sections 114-116, 120),
`--migration on`,
`--policynetwork LAYERS NODES [ACTIVATION]`, `--resume`,
`--local_reward` (per-cell source-tracked reward, sections 79 and 80;
available on `train_gpu.py` as well), `--rnd_baseline [all|solo]`
(random-action `_rnd` curves in the live plot; `all` = every DM random,
`solo` = leave-one-out per DM; on `train.py`, `train_gpu.py --visual` and
`inference.py --visual`, sections 62 and 96). Prefer
`--local_reward_norm grid` with the default `log` metric: `sum` rewards
spreading thin and `mean` rewards killing the worst cells (sections 82, 84).

## 7. Working conventions

- **Biology drives, not action-hacks.** Fix parameters/mechanisms before adding
  shaping terms. `argmax_penalty`, `entropy_coef`, temperature annealing and
  `uniform_bias_init` default to off on `train.py` and `train_gpu.py`, with or
  without `--profile` (section 128); they are explicit opt-in flags only.
- Don't redo performance work (batching, vectorisation, slice-assign, CRN)
  without measuring first.
- Changing the Observability matrix, observable impacts, the FG set or the grid
  layout invalidates existing `.pth` checkpoints (retraining required).
- Commit **only** on explicit user request, with
  `--trailer "Co-authored-by: Junie <junie@jetbrains.com>"`.
- Untracked artifacts outside agent scope unless asked: `make_*.py`, `*.png`,
  `ab_test_*.py`, `resume.md`, `konvergensproblem.txt`, PDFs, `results/`.
- Always validate changes with the pytest suite plus a 1-generation training run.

## 8. Deep context - `mareld_resume.txt`

`mareld_resume.txt` (~5700 lines) is the full project history: every design
decision, calibration, A/B test and open problem, organised in numbered sections
(0 language policy, 1 repo layout, 3 the tick, 5 CLI, 6 project format,
7 GUI, 9-10 ARS/co-evolution, 11 known discrepancies vs the PDFs, 14 git log,
26-28 spawn and multi-world, 35-36 ecological fixes, 41 observability/hide,
46-49 starvation and reward shaping, 63-71 latest diagnostic arcs,
79 local per-cell reward, 90-92 the viability rig).

**Do not read it end-to-end** - it is large and will eat the context window.
Search it for the topic at hand and read only the matching section, e.g.:

```bash
grep -n "^[0-9]\+\(\.[0-9A-Z]\+\)\? " mareld_resume.txt   # section index
```

Keep it updated: when an arc of work finishes, append a new numbered section
there (ASCII only) rather than expanding this file.

`resume.md` is a separate, narrower resume for the `fgconfig/fgconfig.py` GUI.
