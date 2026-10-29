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
| `lib/environments/ecosystem_env/stages.py` | Juvenile compartments (section 223.32): a DM with `juvenile_stage` carries its juveniles J inside B (reserve pro rata), new tonnes to J in the share (kappa J + rho(t) A)/(kappa J + rho(t) A + A), graduation exp(-gamma) per tick, losses with B, fishing at `fishing_selectivity`, pair `stage_access: [f_J, f_A]`, one observation channel per staged group observed. Both engines (GPU: a `juvenile` [E, G, C] state through `step`, rollout and trainer); tests/test_juvenile_stage.py. |
| `lib/world/impact_maps.py` | Impact fields: the training sampler (baseline maps, spawned observable impacts), the shared .npz reader (section 161). |
| `lib/world/habitat.py` | Depth habitat (section 207): the manifest's `simulation_settings.depth_map` (maps/depth.npz, tools/make_depth_map.py; also the spawn ref field `depth`), K per cell from a species' `capacity_depth_table`, and the `spawn: {mode: equilibrium}` start pattern B*(c) = K(c)(1 - (F(c) + M2)/r) from today's baseline impact maps. Both engines (`FunctionalGroup.capacity_map` / `equilibrium_map`). |
| `lib/world/tick_time.py` | Tick length: bounds (1-6 h), the `--tick-length` flag, tick<->real-time conversion, label rendering, and the per-parameter rescale rules the LOADER applies at runtime. `fg_library.yaml` is always calibrated at `LIBRARY_TICK_HOURS` (6) and is never rewritten for a run (sections 97, 103, 106). |
| `lib/spawn/`, `lib/viz/`, `tools/` | Spawn strategies, live pygame visualiser, offline plot/calibration tools. |
| `lib/runners/training_progress.py` | The viewer's "long run" tab (id `progress`, renamed in 223.21; section 223.12): a frozen copy of the current policies run three years (`PROBE_YEARS`, library ticks) in the inference world, in a separate process every `--eval-every` updates; the tab shows the latest probe (every group's biomass as % of start, daily); `<run>/progress/probe.jsonl` keeps survival ticks, year-end ratios and a viability block per probe, `probe_latest.json` the trajectory. Each probe also scores VIABILITY.md section 1 on whole probe years (`probe_viability`; one seed - tracks training, never a reward; section 223.18); the `viability` tab and the `plots.html` tab of that name plot each group's margin and V per probe (223.20). |
| `lib/environments/ecosystem_env/source_tracking.py` | `--local_reward`: per-cell source tracking, `reward(c)=B(c,t+1)/A(c,t)`. |
| `lib/diagnostics/viability.py`, `tools/viability.py` | Long-term viability rig - frozen behaviour, no ARS. A diagnostic for the mechanics, not the judge of the world: its verdict (normative corner `--spawn colocated --behaviour greedy`) is about that behaviour, and trained policies can be viable where it fails (`VIABILITY.md` section 6, resume 123). Two factors (behaviour x spawn geometry). The hand-coded arms allocate the eat mass by marginal energy return (water-filling), never evenly - section 92. Criterion in `VIABILITY.md` (sections 90, 91, 92). |
| `tests/` | pytest suite - keep green. |
| `results/<run-name>/` | Checkpoints `{'state_dict': ..., 'obs_stats': {...}}`. |

## 4. Functional groups

- Decision makers (have a policy net): `zooplankton`, `planktivorous_fish`,
  `benthivorous_fish`, `predatory_fish`, `whiting`, `porpoises`, `seals`,
  `seabirds`.
  `mareld2.yaml` mutes only `seabirds`. Since section 209 the fish are
  three feeding groups (were `pelagic_fish` + `gadoids`; every value a
  biomass-weighted species value, research_notes/"Fiskgrupper i rutan"):
  planktivores = herring 0.79, blue whiting, Norway pout, sprat (3.3
  t/km2, the HERAS acoustic level); benthivores = haddock, cod,
  flatfish (1.53 t/km2); predators = saithe, whiting, mackerel,
  spurdog, hake (1.20 t/km2) - the demersal levels are the top of the
  long-run NS-IBTS span (2001-25, user's choice; the old 0.69 t/km2
  gadoid start had no source). Since section 223.26 the predators are
  two groups by diet (G1, research_notes/"Rovfiskens uppdelning"
  params_split_223.md): `whiting` = the bed feeder (0.446 t/km2, 1605 t;
  the only predator with an epifauna pair, `bottom` layer) and
  `predatory_fish` = the water column (saithe 0.65, mackerel, spurdog,
  horse mackerel, hake, ...; 0.754 t/km2, 2715 t; no bed pairs) - the
  id kept, its meaning changed; checkpoints before 223.26 are invalid. Ceilings from measured Cmax per
  species (energy convention of section 148). Seals (harbour seals
  only, on since 166) spawn on the demersal groups (benthivores and both
  predator groups) and take ~60 % benthivores, 15 % predators (whiting
  0.44 of it), 25 % planktivores (Type III, a 2e-3 -
  a Type III `a` is per (t/cell)^2; judge diets with the engine's
  VISIBLE prey, section 199.9).
- Non decision makers: `phytoplankton` (nutrient-limited since 172),
  `subsurface_phytoplankton` (section 180: the A17 subsurface
  chlorophyll maximum, the same algae in the layer [M(t), 30 m],
  shade-adapted I_k 30, eaten by zooplankton and microzooplankton
  diluted over the layer thickness; zooplankton observes it),
  `benthic_infauna` (since section 221 the INFAUNA only - polychaetes,
  brittle stars, bivalves, small burrowing crustaceans; was
  `benthic_community` before section 221; r_max
  1.0, derived growth, K per cell from depth 132 -> 66 t/km2 at
  100 -> 300 m, E 2600 MJ/t, start = today's equilibrium ~94 t/km2;
  since section 223.12 a REFUGE with exchange: `exposed_share` 0.2 is
  the accessible part A (in its reserve slot), the rest out of reach,
  `refuge_exchange_rate` 2.5 /yr relaxes A towards 0.2 B, each part
  logistic on its own K; predation, the Holling term and the
  observations see A only - `lib/environments/ecosystem_env/refuge.py`,
  both engines, tests/test_refuge.py; both keys are fields in the FG
  editor's NDM panel, 0 / empty = no refuge),
  `mobile_epifauna` (section 221 split it out of the benthos so the prey
  the fish take can be depleted; since 223.15 (E1) the EXPOSED taxa only:
  crabs, hermit crabs and Munida 2.45, northern shrimp 0.56, small shrimps
  and mysids 0.545, cephalopods and small demersal fish 0.23 t/km2 on the
  mass-balance stocks; r_max 1.14 stock-weighted, six-row depth K 6.8 ->
  4.7, E 3750, start ~13 600 t; research_notes/"Havskrafta och exponerad
  epifauna", Swedish letters), `norway_lobster` (section 223.15: Nephrops,
  4.57 t/km2 from the UWTV strata on the mud share, r_max 0.34, depth K
  ending at 300 m, E 3600, start ~16 500 t; in burrows ~93 % of the time;
  eaten by the benthivores (cod) at M2 0.0225, 1.5 % of their
  consumption, research_notes/"Havskrafta i fiskdieter"; predators ->
  Nephrops unchecked), `microzooplankton` (section 177:
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
  today's world: `bottom_trawling` (benthivores F 0.17, water-column
  predators 0.213, whiting 0.075 /yr) and `pelagic_trawling`
  (planktivores 0.10, water-column predators 0.029, whiting 0.005),
  the `maps/` effort maps
  as baseline and inference maps, linear library tables calibrated to
  the maps' mean effort. Impact mortality acts on every group with a
  table, non decision makers too (biomass factor only): bottom trawling
  kills infauna at d x SAR (d 0.06, SAR 0.75 per MW-h/km2), 0.136/yr,
  and removes the epifauna by catch plus the kill of the uncaught part:
  exposed taxa 0.186/yr (northern shrimp, crabs, small shrimps), Nephrops
  0.087/yr (sections 218, 221, 223.15).
  Porpoise gillnet bycatch (~0.017/yr, met outside the grid) rides on
  the bottom-trawl map too (section 165).
- Holling ceiling is 1/h (`energy_balance.intake_ceiling`); `max_intake_rate`
  is the attack rate a. A pair may override it (`<pred>_preys_on_<prey>:
  max_intake_rate`) to set the half-saturation 1/(a h) (Type III:
  1/sqrt(a h)) without moving the ceiling. Every fish pair carries its
  own a from encounter theory since section 220 (research_notes/
  "Attackrater ur motesteori", Swedish letters; prey refuges and
  schooling left to the prey's hide action). Since section 223.26 every
  a is per tonne of the WHOLE predator group (the engine has no phi
  since 221; the 220 "per capable tonne" values over-credited by 1/phi):
  planktivores -> zoo 0.017; benthivores -> infauna 0.036, -> epifauna
  0.15, -> Nephrops 0.44, -> planktivores 0.0062, -> whiting 0.0026;
  whiting -> epifauna 0.27, -> planktivores 0.021, -> zoo 0.0023, ->
  benthivores 0.21; water column -> planktivores 0.0117, -> zoo 0.0006,
  -> benthivores 0.053, -> whiting 0.0106 (sections 220, 221, 223.15,
  223.26; no predator eats infauna or Nephrops, the water column no
  epifauna). The
  benthic pairs are a_enc x an exposure factor e derived per prey item
  type (buried bodies found only by a cue, cropped arm tips and siphons,
  crab size window, flatfish chemo-tactile width; infauna e 0.165,
  epifauna 0.43 / 0.39; research_notes/"Exponering av bottenbyten",
  Swedish letters). They stay saturated: the field feeding level (0.22-
  0.53 of Cmax) is a time-budget matter, not a. The benthivores' ceiling
  is 26.18 MJ/t/tick at the bottom temperature (223.15), per prey since
  223.28: ceiling_j = 26.18 (E_j/4.184)^0.25 / armour_j (gastric
  evacuation ~E^-0.75; armour epifauna 1.75, Nephrops 2.0), h = E /
  ceiling_j (herring 238, whiting 177, epifauna 258, Nephrops 286,
  infauna 112; research_notes/"Magtomning och energitak", Swedish
  letters); every other predator keeps h = E / its ceiling. The predator
  ceilings are Bjornsson-anchored (x1.30, 223.26): whiting 31.34 (bottom
  layer), water column 32.71. Never set a from model output.
- Daylight (opt-in, section 137): a pair with `dark_ratio` < 1 has
  `a(t) = a * m(t)`, m from the sun's elevation vs `light_threshold_deg`,
  normalised so the library `a` is the annual mean; every DM then also
  observes the light level (last input slot). Since section 220 (derived
  night/day ratios): planktivores -> zoo 0.02 at 0 deg; benthivores ->
  planktivores 0.06 and -> whiting 0.10 at 0 deg (140 m is too dim
  for -5); at -5 deg (223.26): whiting -> planktivores 0.33, -> zoo
  0.78, -> benthivores 0.64; water column -> planktivores 0.39, -> zoo
  0.30, -> benthivores 0.48, -> whiting 0.48; benthic pairs none.
- Predation: vectorised, Holling type II with Beddington-DeAngelis crowding
  (`a_eff = a / (1 + a*h*B_prey_visible + w*B_pred)`), hidden (rested) fraction
  protected, energy gain buffered. `w` is the per-FG `interference` parameter
  (`interference: 0` = pure Holling; the old gadoid 1.0 of section 86
  removed 2-cell vertical bands; benthivores 0 since 223.28 (the 0.45
  had no fish data; research_notes/"Interferens hos bottenfisk"),
  predators 0.1 (no field evidence), planktivores 0.05 since section
  162 - 0.7 had no herring data behind it and cut winter intake by two
  thirds). Planktivores -> zooplankton is Type II
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
  t_ref is 14 C (the library value is a summer cost), the demersal
  groups (water-column predators at_50m, benthivores the `bottom` layer
  since 223.15 and whiting since 223.26,
  SHARK A16/A17 at each cell's depth, annual 7.4 C) and zooplankton
  `annual_mean`, porpoises none
  (endotherm). On in
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
  loader derives `capacity_loss_rate`. K = B*/(1 - F/r) on every fish
  group (section 209): benthivores 3.39 (r 0.31, F 0.17; fragile, cod
  alone has F > r), water-column predators 2.14 (r 0.374, F 0.24;
  1.05-11 with saithe F), whiting 0.533 (r 0.49, F 0.08; 223.26),
  planktivores 4.8
  (r 0.37; stock-scale compensation plus net emigration, 185.20).
  Without K the old gadoids grew at r_max - F, ate the pelagic pool down
  and crashed (184.13), and the pelagic pool grazed the zooplankton
  down year after year (185.17-19). Seals: K 0.0075
  (27 t, no hunt impact; sections 199, 206) - the grid holds the
  offshore tail of a metapopulation regulated off the grid; without it
  they grew ~6 %/yr on the old gadoids, halved them and crashed; at K = the
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
  `reserve_reference_fill`. Set for `planktivorous_fish` (section
  140.6; 209): lean 4000 MJ/t + reserve up to 6250, starvation from 0.2
  of it, `predatory_fish` (water column, lean 4240 + 2600 at fill 0.5 =
  5540), `whiting` (lean 3900 + 2000 at fill 0.45 = 4800; 223.26) and,
  since section 223.9, `benthivorous_fish` (lean 3600 + 1800 at fill
  0.5 = 4500; was whole-body 4500 with each new tonne charged its
  reserve on top). Benthivore pair M2 since 223.9: budget Q/B 4.46,
  the no-pool 17.7 % of the diet redistributed (infauna 0.0396,
  epifauna 0.276, planktivores 0.245, predators 0.128 - all of it on
  whiting since 223.26, cod's fish prey in the pool being whiting; M2
  0.18 since 223.28 = SMS cod -> whiting scaled to the model's
  cod:whiting ratio, was 0.345). Benthivore resting metabolism re-read at the
  bottom layer: 3.33 (223.26).
- `functional_response: 2|3` on a predation pair forces the Holling type
  for that pair (section 141); without it a specialist gets III and a
  generalist II, as before. Since section 209 every fish -> fish pair
  is Type II (measured prey switching in North Sea gadoids is negative,
  Rindorf et al. 2006; the old Type III on gadoids -> herring was a
  stability fix, 141.4); seals -> planktivores stays Type III (166).
- Pair shares (section 208, both engines, each in (0, 1], inert at 1):
  `accessible_fraction` f scales the prey biomass in the pair's Holling
  term and harvest cap (infauna out of reach, Nephrops in burrows);
  observations are untouched. Benthivores -> epifauna 0.62 (223.15), whiting
  -> epifauna 0.29 (sections 221, 223.15, 223.26), water column -> whiting
  0.47 (gape); the infauna pair's 0.2 became the
  infauna's own refuge in 223.12. On a prey with a juvenile compartment
  (planktivores, benthivores; 223.32) a pair may carry `stage_access:
  [f_J, f_A]` instead (members' prey-size windows on the stage lengths):
  water column -> benthivores [0.43, 0.027] (was f 0.075), whiting ->
  benthivores [0.049, 0] (was 0.05), water column -> planktivores [0.32,
  0.11], whiting -> planktivores [0.12, 0.008], benthivores ->
  planktivores [0.49, 0.14], seals -> benthivores [1, 0.34], porpoises ->
  benthivores [1, 0.55]; research_notes/"Arsungarnas andel" (A-ring)
  parameters_B.md. `predator_share` phi
  (`eat = min(eat, phi)`, a hard cap on a learned action) is REMOVED
  from the library since section 221; the mechanism stays in both
  engines, inert at 1 (tests/test_pair_shares.py). It stood in for
  saturated pairs and missing structure (cod piscivory in the
  benthivores; a never-depleting benthos), which density-dependent a
  (220) and the epifauna group (221) now carry. Do not reintroduce it;
  a mean field a phi, h / phi failed too (208.5).
- `growth_rate` is DERIVED (`lib/world/growth_budget.py`, section 129): DM
  `g = (r_max + M1 + sum checked predation_mortality) / ((s*-u)*1460)`,
  NDM `r = (r_max + sum M2) / 1460`; an income-funded DM's g is x 12 /
  `feeding_season_months` (section 184); a `regional_carrying_capacity`
  loss is not in g (section 185). s* is the reserve a DM settles on
  under ideal conditions (checked prey at the Holling ceilings, the best
  pair at the whole eat share since phi is gone (221), eating every
  tick; closed form from the library, no policy - section 160); it
  replaced the full reserve 1 - u, which the hunger gate never lets a DM
  reach. Checking/unchecking a predator in the FG editor moves the prey's
  growth; muted FGs are absent. A pair a group-biomass reward cannot
  learn (cannibalism) does not belong in the budget: within-group
  cannibalism is unchecked (section 160); cross-group fish predation
  (predators on juvenile benthivores) is a learnable pair and is in it. With `--mortality off` the loader leaves M1 out of g (pass
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
python3 train.py --project mareld2.yaml --species planktivorous_fish \
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
`--depleted-patches SHARE` (train_gpu.py, default 0.5: that share of the
training worlds starts with depleted benthic patches, 5-30 % of the cells
at 1-32 % of their biomass - a refuge group loses accessible tonnes - so
the fish meet emptied beds and can learn to leave them; training-world
design, not biology; 0 = off; section 223.12),
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
