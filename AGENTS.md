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
| `lib/world/growth_budget.py` | Derives every FG's `growth_rate` from `r_max`, `natural_mortality` and the checked `predation_mortality` pairs, paid at the ideal-conditions reserve s* (section 160); shared by loader, FG editor and tests (section 129). |
| `lib/world/daylight.py` | Daylight calendar (section 137): sun elevation from latitude and date, per-tick light level, and the per-pair attack-rate multiplier (annual mean exactly 1), and the light-limited producer growth factor (section 138: P-I curve over the mixed layer, 1 on the reference day in April). Off unless `simulation_settings.daylight.enabled` in the manifest; turning it on adds one observation channel (checkpoints invalid). The live viewer shows the month and a day-length frame around each heatmap (section 142), and the inference plot a month / clock-time band under its x axis (section 155). |
| `lib/world/temperature.py` | Water temperature on the daylight calendar (section 143): Q10 multiplier on each DM's `resting_metabolism` (`metabolism_q10`, `metabolism_t_ref` in the library; monthly layer temperatures and `group_layers` in `simulation_settings.temperature`). Needs daylight on; adds no observation channel. `ingestion_q10` (section 181) scales a DM's whole functional response (a and 1/h together, annual mean 1); set on zooplankton (2.8). `growth_q10` (section 183) scales a DM's growth branch (reserve -> biomass, annual mean 1; starvation unscaled); set on zooplankton (3.0, Huntley & Lopez 1992). Zooplankton is calibrated against A17's resident stock in Dec-Mar (the totals are Calanus risen from the Norwegian Trench). |
| `lib/world/nutrients.py` | Nutrient pool of light-limited producers (section 172): growth `r L(t) B N/(N + k_N Y M)` paid out of a per-cell pool, supplied from below the mixed layer (Evans & Parslow) and by a share of the grazing. Off unless `simulation_settings.nutrients.enabled` (on in `mareld2.yaml` since section 174, with seasonal pycnocline exchange; 1.0 m/d Nov-Feb since section 203); the pool rides in the producer's unused `energy_reserve` slot on both engines. No observation channel. With `nutrients.subsurface_layer` (section 180, on in `mareld2.yaml`) the column has two layers, each with a producer (`water_layer: mixed|subsurface`) and pool, chained reservoir (50 m) -> subsurface -> mixed layer, entrainment/detrainment moving water and algae (`layer_tables`, `population_change.apply_layer_exchange`, GPU `layer_exchange`). |
| `lib/world/impact_maps.py` | Impact fields: the training sampler (baseline maps, spawned observable impacts), the shared .npz reader (section 161). |
| `lib/world/habitat.py` | Depth habitat (section 207): the manifest's `simulation_settings.depth_map` (maps/depth.npz, tools/make_depth_map.py; also the spawn ref field `depth`), K per cell from a species' `capacity_depth_table`, and the `spawn: {mode: equilibrium}` start pattern B*(c) = K(c)(1 - (F(c) + M2)/r) from today's baseline impact maps. Both engines (`FunctionalGroup.capacity_map` / `equilibrium_map`). |
| `lib/world/tick_time.py` | Tick length: bounds (1-6 h), the `--tick-length` flag, tick<->real-time conversion, label rendering, and the per-parameter rescale rules the LOADER applies at runtime. `fg_library.yaml` is always calibrated at `LIBRARY_TICK_HOURS` (6) and is never rewritten for a run (sections 97, 103, 106). |
| `lib/spawn/`, `lib/viz/`, `tools/` | Spawn strategies, live pygame visualiser, offline plot/calibration tools. |
| `lib/environments/ecosystem_env/source_tracking.py` | `--local_reward`: per-cell source tracking, `reward(c)=B(c,t+1)/A(c,t)`. |
| `lib/diagnostics/viability.py`, `tools/viability.py` | Long-term viability rig - frozen behaviour, no ARS. A diagnostic for the mechanics, not the judge of the world: its verdict (normative corner `--spawn colocated --behaviour greedy`) is about that behaviour, and trained policies can be viable where it fails (`VIABILITY.md` section 6, resume 123). Two factors (behaviour x spawn geometry). The hand-coded arms allocate the eat mass by marginal energy return (water-filling), never evenly - section 92. Criterion in `VIABILITY.md` (sections 90, 91, 92). |
| `tests/` | pytest suite - keep green. |
| `results/<run-name>/` | Checkpoints `{'state_dict': ..., 'obs_stats': {...}}`. |

## 4. Functional groups

- Decision makers (have a policy net): `zooplankton`, `pelagic_fish`, `gadoids`,
  `porpoises`, `seals`, `seabirds`. `mareld2.yaml` mutes only `seabirds`;
  seals (harbour seals only) are on since section 166 and spawn on
  gadoids, which carry ~90 % of their diet (pair attack rates 0.2,
  Type II, vs the pelagic pool 2e-3, Type III - a Type III `a` is per
  (t/cell)^2; 0.015 made fix60's seals 85 % herring eaters, section
  168; 1e-4 gave fix68's seals no pelagic fish at all, 185.10; at 3e-4
  (185.16) even a perfectly switching seal on fix70's fields took
  < 0.3 %, at 2e-3 4-9 % - judge diets with the engine's VISIBLE prey,
  section 199.9).
- Non decision makers: `phytoplankton` (nutrient-limited since 172),
  `subsurface_phytoplankton` (section 180: the A17 subsurface
  chlorophyll maximum, the same algae in the layer [M(t), 30 m],
  shade-adapted I_k 30, eaten by zooplankton and microzooplankton
  diluted over the layer thickness; zooplankton observes it),
  `benthic_community` (logistic, r 1.0, K per cell from depth 143 ->
  71.5 t/km2 at 100 -> 300 m, box mean 120, start = today's equilibrium
  ~95 t/km2 under trawling; section 207), `microzooplankton` (section 177:
  grows only by grazing the phytoplankton in the population step;
  eaten by zooplankton, which observes it).
- English snake_case IDs are canonical.

## 5. The Tick (per tick, random FG order)

`_calculate_decisions` -> `_apply_impact_mortality` -> `_apply_predation`
-> `_apply_movement` -> `_apply_growth`

- Decisions: batched `torch.bmm` over all DMs; per-DM compact observation layout
  driven by the Observability matrix; `Rest` doubles as a hide action.
- Impacts: linear interpolation in `impact_table`, clipped to endpoints.
  `biomass_factor` is a loss per tick, `energy_factor` multiplies every
  action cost by `1 + sum`. Training: an impact with a `baseline_map`
  (Impact Editor; today's field, e.g. fishing effort per cell) gets that
  map in every world; otherwise an observable impact is spawned per world
  (so groups learn behaviour against it) and any other impact is absent.
  Inference reads only `inference.impact_maps` (no map = no impact): add
  the baseline file there to keep today's fishing, or another file for a
  scenario - one field per impact, never stacked
  (`lib/world/impact_maps.py`). The GPU engine mirrors all of it (impact
  mortality and cost, observable channels in train.py's slots; section
  161). Impacts are outside the growth budget. `mareld2.yaml` fishes
  today's world: `bottom_trawling` (gadoids, F 0.12/yr) and
  `pelagic_trawling` (pelagic fish, F 0.10/yr), the `maps/` effort maps
  as baseline and inference maps, linear library tables calibrated to
  the maps' mean effort. Impact mortality acts on every group with a
  table, non decision makers too (biomass factor only): bottom trawling
  removes benthos at d x SAR (d 0.10, SAR 0.75 per MW-h/km2), ~0.2/yr.
  Porpoise gillnet bycatch (~0.017/yr, met outside the grid) rides on
  the bottom-trawl map too (section 165).
- Holling ceiling is 1/h (`energy_balance.intake_ceiling`); `max_intake_rate`
  is the attack rate a. A pair may override it (`<pred>_preys_on_<prey>:
  max_intake_rate`) to set the half-saturation 1/(a h) (Type III:
  1/sqrt(a h)) without moving the ceiling.
- Daylight (opt-in, section 137): a pair with `dark_ratio` < 1 has
  `a(t) = a * m(t)`, m from the sun's elevation vs `light_threshold_deg`,
  normalised so the library `a` is the annual mean; every DM then also
  observes the light level (last input slot). Set on herring -> zoo
  (0.1, -8 deg) and gadoids -> herring (0.6, -5 deg; section 144).
- Predation: vectorised, Holling type II with Beddington-DeAngelis crowding
  (`a_eff = a / (1 + a*h*B_prey_visible + w*B_pred)`), hidden (rested) fraction
  protected, energy gain buffered. `w` is the per-FG `interference` parameter
  (`interference: 0` = pure Holling; `gadoids: 1.0` in the library,
  section 86, where it removed the 2-cell vertical bands; `pelagic_fish:
  0.05` since section 162 - 0.7 had no herring data behind it and cut
  winter intake by two thirds). Herring -> zooplankton is Type II
  (`functional_response: 2`) with `visibility_floor` 0.5 (section 162).
- Movement: slice-assign, per-action metabolic cost; energy follows biomass.
  `min_split_biomass` is one indivisible unit (porpoises: one animal):
  below it a cell moves whole (argmax), and from it no spatial fraction
  of a split holds less than one unit - sub-unit fractions go to the
  others proportionally (`decisions.merge_subunit_fractions`, on by
  default, `unit_splits=False` on the env reverts; section 159).
  With `simulation_settings.temperature` (section 143) the cost of a DM
  with `metabolism_q10` is `resting_metabolism * q10**((T - t_ref)/10)`,
  T from its layer's monthly table on the daylight calendar; herring
  t_ref is 14 C (the library value is a summer cost), gadoids and
  zooplankton `annual_mean`, porpoises none (endotherm). On in
  `mareld2.yaml`. Intake is temperature-scaled only for a DM with
  `ingestion_q10` (section 181: zooplankton 2.8, the whole response,
  annual mean 1; it moved the trained zooplankton's spring from February
  to March-April as at A17); `growth_q10` (section 183, zooplankton 3.0)
  scales its growth branch the same way.
- Growth: NDM logistic; DM `s = R/(B*ME)`, `q = s - u` (`maintenance_level`, default 0.3).
  A DM with `income_growth_kappa` kappa (section 184) grows out of this
  tick's income instead, `G = min(kappa I - C, B g (s - u) ec)` in
  `movement.apply_energy_costs` (both engines), and never turns its
  reserve into biomass: no food, no growth, the fat is kept for the
  winter (the reserve-funded law kept growing on it after the food had
  gone). Its derived g is paid over `feeding_season_months` (x 12/n).
  Set on herring (kappa 0.75, season 5.5 months; resume 184.8).
  A DM with `regional_carrying_capacity` K (t/km2, section 185) loses
  `r_max/1460 * B_region/K` of its biomass and reserve per tick after
  M1 (B_region = grid-mean density; both engines, with or without
  `--mortality`), kept OUT of the derived g, so it grows at
  `r_max (1 - B/K) - F` (the logistic its r_max prior comes from). The
  loader derives `capacity_loss_rate`. Gadoids: r_max 0.37 (today's
  productivity, 184.18) and K 1.02 (today's effective capacity, not the
  historical ceiling); without it they grew at r_max - F, ate the
  pelagic pool down and crashed (184.13). Pelagic fish: r_max 0.34
  (site mix, today's recruitment) and K 5.0 since 185.20 - stock-scale
  compensation plus net emigration; without it the pool grazed the
  zooplankton down year after year (185.17-19). Seals: K 0.0075
  (27 t, no hunt impact; sections 199, 206) - the grid holds the
  offshore tail of a metapopulation regulated off the grid; without it
  they grew ~6 %/yr on gadoids, halved them and crashed; at K = the
  15 t start they settled at ~0.5 K, because the loss is charged at
  r_max while they realise ~0.06 /yr net. Porpoises have no K; their
  attack rates are x1.5 since section 206 (they fell to ~0.3 of the
  start in 15-20 years at the old rates).
  With the daylight calendar a producer with `light_saturation` grows at
  `growth_rate * P(t)/P_ref` (section 138; 0 at night, needs the site's
  light climate in the manifest). The old `seasonal_amplitude` /
  `seasonal_period` sine is gone. With `simulation_settings.nutrients`
  (section 172) a producer with `nutrient_half_saturation` grows on a
  nutrient pool instead of logistically (K stays a ceiling); alone it
  gives the nitrate cycle but no spring bloom. An NDM's
  `natural_mortality` is its own loss M1 (respiration, sinking, lysis;
  section 173): removed before growth, follows `--mortality`, and
  counted in its derived growth `(r_max + M1 + sum M2)/1460`.
  Phytoplankton M1 0.005 (its sinking loss; r_max 67.882 keeps g 0.0670)
  is on since section 174: pool + M1 give a March bloom; checkpoints
  from before 174 were trained in a flat-phytoplankton world. A
  nutrient-stress loss (`stress_loss_max`, section 175) exists but is
  OFF; since section 177 the summer minimum comes from grazing.
- Grazing NDMs (section 177): an NDM with a checked
  `<ndm>_preys_on_<prey>` pair carrying `max_intake_rate` grazes at the
  start of the population step (Holling II, optional `food_threshold`
  = ERSEM perception term, `grazing_q10` on its temperature layer,
  `growth_efficiency`, `excretion_regeneration` to the prey's pool) and
  counts as a predator in the prey's growth budget. Phytoplankton g is
  therefore GROSS (0.1675/tick): microzooplankton carries the 60 %
  micrograzing as M2 147/yr. A pair with `mixed_layer_reference_m` has
  a(t) = a M_ref/M(t) (grazers feed on concentration); the zooplankton
  pairs use it (half-saturation 50 ug C/l = 5 t/km2 at 10 m on the
  subsurface pair). Zooplankton -> microzooplankton is Type III (prey
  switching, section 178): with Type II the trained zooplankton grazed
  it extinct in March. Zooplankton -> (mixed-layer) phytoplankton is
  Type III too since section 202 (a 0.03, half-saturation 2.9 t/km2):
  Type II grazed the summer mixed layer to ~0.01 ug chl/l.
  A pair on a subsurface prey scales by M_ref/H(t), H = bottom - M(t)
  (section 180). A grazing pair may carry an edible share of a producer
  with a pool, `e = edibility_floor + (1 - floor) N/(N + K)` with K
  `edibility_half_saturation_umol_l` (section 182): set on
  microzooplankton -> subsurface phytoplankton (0.1, 2 umol/l; the
  late-summer Ceratium layer at the pycnocline), which stocks the
  autumn entrainment and raised October-November zooplankton.
- Regenerated N (section 202): `mortality_regeneration` (0-1) returns
  that share of a group's M1 to the nutrient pools in the same tick,
  split mixed / subsurface by where it fed this tick (thickness shares
  if it fed on neither); `egestion_regeneration` on a grazing NDM
  returns that share of its egesta (1 - gge)(1 - excretion) to the
  prey's pool. Zooplankton and microzooplankton 0.5, micro egesta 1.0.
  Both engines add the M1 return after every group's growth. It lifted
  PP 47 -> 57 g C/m2/yr and the May-July zooplankton to A17 levels; the
  production was capped by the new-N supply (~39 g C, f-ratio ~0.9).
  Not in the growth budget (nitrogen, not consumer mass).
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
  where gadoids also eat zooplankton (krill proxy) and each other, and
  on seals -> pelagic_fish (166).
- Pair shares (section 208, both engines, each in (0, 1], inert at 1):
  `predator_share` phi caps the predator's eat share on that prey,
  `eat = min(eat, phi)` with full a and 1/h (only phi of the pooled
  group can exploit the prey; the excess is wasted; growth_budget pays
  phi x the ceiling). A mean field a phi, h / phi made every fish half
  a benthivore and an energy maximiser dropped benthos - do not use
  it. `accessible_fraction` f scales the prey biomass in the pair's
  Holling term and harvest cap (infauna out of reach); observations
  are untouched. Set on gadoids -> benthos (phi 0.53, f 0.2).
- `growth_rate` is DERIVED (`lib/world/growth_budget.py`, section 129): DM
  `g = (r_max + M1 + sum checked predation_mortality) / ((s*-u)*1460)`,
  NDM `r = (r_max + sum M2) / 1460`; an income-funded DM's g is x 12 /
  `feeding_season_months` (section 184); a `regional_carrying_capacity`
  loss is not in g (section 185). s* is the reserve a DM settles on
  under ideal conditions (best checked prey at the Holling ceiling, eating
  every tick; closed form from the library, no policy - section 160); it
  replaced the full reserve 1 - u, which the hunger gate never lets a DM
  reach. Checking/unchecking a predator in the FG editor moves the prey's
  growth; muted FGs are absent. A pair a group-biomass reward cannot
  learn (cannibalism) does not belong in the budget: gadoid cannibalism
  is unchecked (section 160). With `--mortality off` the loader leaves M1 out of g (pass
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
# Plankton against SHARK A17 (section 205): column biomass at A17's
# measured C:chl, surface carbon, nitrate, zooplankton, PP and f-ratio.
python3 tools/probes/a17_score.py run results/<run-name>      # ~2 min
python3 tools/probes/a17_score.py score a.json b.json         # saved runs
# Monthly attack-rate multipliers and producer growth factors (137, 138).
python3 tools/probes/daylight_table.py
# Fishing-effort maps (MW-h/km2/year per cell) from EMODnet -> maps/*.npz;
# the library's trawling tables are calibrated to their mean (section 161).
python3 tools/make_fishing_maps.py
# Mean depth per cell from EMODnet -> maps/depth.npz (section 207).
python3 tools/make_depth_map.py

# GUI, inference, plots
python3 fgconfig/fgconfig.py
python3 inference.py --project mareld2.yaml --run-name <name>
python3 tools/biomass_html.py results/<run-name>      # -> plots.html
```

Useful flags: `--profile sanity|info|deep` (run-size presets; action hacks are off with or without one),
`--visual`, `--rollouts_per_delta`, `--mortality on`,
`--tick-length HOURS` (1-6, default 6; section 106),
`--warmup-days D` (train.py with co-evolution and train_gpu.py: each world
first runs D days with the unperturbed policies, shared by the +/-
candidates, and the reward starts from that state - the plankton settle
from the season-blind start totals; default 0, section 200; start totals
are drawn log-uniformly from the manifest ranges),
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
`inference.py --visual`, sections 62 and 96), `--norm rolling|fixed`
(heatmap normalisation the `--visual` viewer starts in, default
`rolling`; N toggles it; same three entry points, section 197). Prefer
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
