# Brief: parameters for the three fish groups (split of gadoids + pelagic_fish)

Shared brief for the parameter reviews (2026-10-24). The model's `gadoids` and `pelagic_fish`
groups are replaced by three feeding groups (`fish_guilds.md`, `gadoid_level.md`):

| Group | Members (long-run 2001-25 biomass share in the group) | Target biomass in the grid |
|---|---|---|
| `planktivorous_fish` | herring ~80-90 % (acoustic), blue whiting, Norway pout, sprat; argentines, silvery pout, pearlside, sandeels < 1 % | ~3.3 t/km2 (12 000 t; acoustic annual mean ~2.5-3 + blue whiting 0.3-0.5) |
| `benthivorous_fish` | haddock 39 %, cod 31 %, long rough dab 13 %, dab 4.5 %, plaice 3.8 %, witch 1.7 %, starry ray 1.5 %, hagfish 1.3 %, grey gurnard, lemon sole, weever, others < 1 % | 1.53 t/km2 (5500 t) |
| `predatory_fish` | saithe 41 %, whiting 37 %, mackerel 8 %, spurdog 5 %, horse mackerel 3.4 %, hake 2.5 %, anglerfish 1.4 %, ling, velvet belly, pollack < 1 % | 1.20 t/km2 (4300 t) |

Shares: NS-IBTS GOV Q1+Q3 2001-25, depth-stratified to the grid (`tables/long_run_species_by_group.csv`).
Levels: the user chose the top of the long-run span (GOV wing, q = 1, x 1.18) for both demersal
groups; planktivores from the acoustic survey (`../Ekolodsdata planktivorer/acoustic_planktivores.md`,
grid correction at the top). Use the LONG-RUN species weights (2001-25), not 2021-25.

## Model conventions (read before choosing values)
- Tick = 6 h (LIBRARY_TICK_HOURS), 1460 ticks/yr; biomass t/cell (1 cell = 1 km2); energy MJ.
- Library: `fgconfig/fg_library.yaml` (read the comments on `gadoids`, `pelagic_fish` and their
  pairs: they document every method and source). History: `mareld_resume.txt` - search, never read
  end-to-end (`grep -n "^1[0-9][0-9] " mareld_resume.txt` for the section index).
- Species fields a decision maker needs: `r_max` (/yr, today's productivity, Euler-Lotka adjusted
  like section 184.18), `natural_mortality` M1 (/tick; non-FG losses only: M0 + predation by
  groups NOT in the model; predation by model groups is M2 via pairs), `energy_content` (MJ/t
  lean wet weight as eaten), `max_energy_reserve` (MJ/t), `maintenance_level`, `resting_metabolism`
  (MJ/t/tick, annual mean), `metabolism_q10` / `metabolism_t_ref`, `movement_speed` (cells/tick),
  `feeding_cost`, `movement_cost` (multipliers of resting metabolism), `starve_rate`,
  `interference` (Beddington-DeAngelis w), `visibility_floor`, `regional_carrying_capacity` K
  (t/km2; K = B*/(1 - F/r), section 185), `income_growth_kappa` / `feeding_season_months` /
  `prey_includes_reserve` (herring, section 140/184), fishing F per gear (`bottom_trawling`,
  `pelagic_trawling`; reports/Fiskeridödlighet i Skagerrak idag.md).
- Pair fields (`<pred>_preys_on_<prey>`): `handling_time` h (ticks per t of prey per t of predator;
  ceiling 1/h t prey/t pred/tick). The user decided (2026-10-24) to set the ceilings from MEASURED
  Cmax per species (maximum daily ration at the area's temperatures, ~6-9 C in the bottom water,
  `simulation_settings.temperature` in mareld2.yaml), biomass-weighted, expressed per prey via
  energy (section 148: the stomach limit is energy; h_prey = h_ref x E_prey / E_ref). The old
  gadoid ceiling was ~1.6x cod Cmax (h 87 vs 135-154; resume 208.2). `max_intake_rate` a (attack
  rate), `assimilation_factor`, `functional_response` 2|3, `predator_share` (eat-share cap,
  section 208), `accessible_fraction` (benthos 0.2), `dark_ratio`/`light_threshold_deg` (light-
  dependent hunting, sections 137, 144), `visibility_floor`, and `predation_mortality` = M2 on the
  PREY (/yr) = consumption of that prey by this predator group / prey biomass, with consumption =
  Q/B x B_pred x diet share by mass (diet by model pool, recent data preferred; diets are checks
  of mechanism-based parameters, never fitted targets, but the budget's M2 needs a consumption
  estimate).
- Model prey pools: `benthic_community` (infauna + epibenthos incl. mysids, amphipods, Crangon,
  Pandalus, Nephrops, crabs), `zooplankton` (copepods + krill; A17-calibrated, mostly copepods),
  `planktivorous_fish`, `benthivorous_fish`, `predatory_fish`. Cross-group predation (e.g. whiting
  and saithe on juvenile haddock/cod; cod on Norway pout and herring) is now a learnable pair.
  Within-group cannibalism stays out of the budget (section 160).
- Existing species-level material: `reports/` (growth rate per FG, F today, M0, density dependence,
  light, winter metabolism), `research_notes/` (Torskfiskarnas reglering, Torskfiskars
  energiinnehåll, Den pelagiska poolens reglering, Torskfiskarnas bytesandelar - stage-resolved prey
  scores and the 2020-25 ICES stomach data by model pool, Torskfiskarnas artsammansättning,
  Restdödlighet M0, Bärförmåga för säl och tumlare). Reuse it and cite it; research only the gaps.

## Rules
- Biology drives: values from measurable mechanisms and literature, biomass-weighted over the
  members; diets only as checks or for the consumption behind M2. Say plainly when a value is a
  judgement.
- Write English, short plain sentences, tags [F] full text read / data downloaded, [A] abstract,
  [S] secondary, [D] computed here.
- Do not edit code, `fg_library.yaml`, `mareld2.yaml`, tests or `mareld_resume.txt`.
- Downloads go in a `data/` subfolder (untrusted; README), scripts in `scripts/`, run Python on
  downloads with `python3 -I`, pass paths as arguments, never execute downloaded code, no file
  > 50 MB.
- Privacy: the user's email address may be sent ONLY to the Unpaywall API (api.unpaywall.org) as
  its `email` parameter. The user said, verbatim: "Det är OK med mig att använda Unpaywall." Never
  send it to any other service, never write it into any file, note, script or log; no registrations.
