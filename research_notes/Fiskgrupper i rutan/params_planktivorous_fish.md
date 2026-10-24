# Parameters for `planktivorous_fish` (replaces `pelagic_fish`)

Review for the fish-group split (`split_brief.md`, 2026-10-24/25). The old group was calibrated on
herring alone. The new members are blue whiting, Norway pout and sprat (argentines, silvery pout and
pearlside < 1 %). Question per field: do the new members move the group value enough to matter, at
their biomass shares? No code, library or manifest was changed.

Tags: [F] full text read / data downloaded, [A] abstract only, [S] secondary, [D] computed here.
Evidence: `planktivores/evidence_pout_bluewhiting.md` (productivity, M, F, diet, light of pout and
blue whiting), `planktivores/evidence_cmax_energy.md` (Cmax, energy density, fat, Q10). Scripts:
`scripts/planktivores_weights.py` (all weighted values; output `tables/planktivores_weights_stdout.txt`),
`scripts/planktivores_ecopath.py` (EcoBase 457 diets and M2 by new group; output
`tables/planktivores_ecopath_stdout.txt`), `scripts/planktivores_temperature.py`, and the
subagents' scripts in `planktivores/scripts/`. Downloads: `planktivores/data/` (README there).

## Bottom line

Weights (annual-mean biomass, section 0): herring 0.79, blue whiting 0.105, Norway pout 0.075,
sprat 0.03. Low-herring / high-herring sensitivity: 0.67 / 0.90 herring.

| Field | Current (herring) | New | Range | Basis |
|---|---|---|---|---|
| Target biomass | 12 000 t (3.33 t/km2) | **keep 12 000 t** | 11 300-13 700 t | herring 2.6 + blue whiting 0.35 + pout 0.25 + sprat 0.1 = 3.30 t/km2 [D] |
| `r_max` (/yr) | 0.34 | **0.37** | 0.30-0.46 | herring 0.35, sprat 0.48, pout 0.46, blue whiting 0.45 (r_now, productivity.md method). Small rise: mackerel (0.25) left the pool, the new members are a bit more productive [D] |
| `regional_carrying_capacity` (t/km2) | 5.0 | **4.8** (keep 5.0 if r stays 0.34) | 4.5-5.0 | K = 3.33/(1 - P/r), P 0.115 (library) or F 0.114 (brief); r 0.37 gives 4.8 either way [D] |
| `natural_mortality` M1 (/yr; /tick) | 0.580; 3.97e-4 | **0.26; 1.8e-4** | 0.11-0.38 | herring M0 0.25 + non-model predators ~0.01; sprat 0.42; pout 0.30 (incl. spawning death); blue whiting 0.20. The drop is mostly NOT the new members: rays, gurnards, mackerel, horse mackerel, hake and spurdog are now model groups, so their predation is M2 on pairs [D] |
| F, `pelagic_trawling` (/yr) | 0.10 | **keep 0.10** | 0.01-0.20 | weighted stock-level F 0.11 (sprat 0.45, blue whiting 0.18, pout 0.03 today). +0.01 is far inside the herring uncertainty. No directed pout or blue whiting fishery in the Skagerrak [D] |
| F, `bottom_trawling` share | 2 % of F | **keep** | - | pout and blue whiting discards in shrimp trawls, 100-250 t/yr, < 0.01 /yr on the group [D] |
| `energy_content` (MJ/t, lean) | 4000 | **keep** | 3800-4100 | lean ED ~4.0 kJ/g fits all four members within 5 % [F/S] |
| `max_energy_reserve` (MJ/t) | 7000 | **6250** | 5400-6650 | max fat: herring 7000 (adult; box juveniles ~6100), sprat 6700, blue whiting 3600, pout 2000 [D]. **Moves the static prey quality from 7500 to ~7100 MJ/t: see the flag below** |
| `maintenance_level`, `starve_rate`, `reserve_reference_fill`, `prey_includes_reserve` | 0.2, 0.025, 0.5, true | **keep** | - | herring sets them; the lean members do not change the winter logic (judgement) |
| `resting_metabolism`, `metabolism_q10`, `metabolism_t_ref`, temperature layer | 8.0, 2.2, 14 C, mean_0_50m | **keep** | Q10 2.2-2.4 | no pout / blue whiting respirometry; proxies give Q10 2.2-2.4. Putting the deep members in the 50 m layer changes the group cost by -3 % per year, -9 % in Jul-Sep [D] |
| `income_growth_kappa`, `feeding_season_months` | 0.75, 5.5 | **keep** | season 5.5-6.0 | pout feeds year-round (income), blue whiting Apr-Sep; weighted season 6.0 months would lower g by 8 % (judgement: keep) |
| `interference`, `visibility_floor`, `movement_speed`, `min_split_biomass` | 0.05, 0.35, 1.0, 0.5 | **keep** | - | no member-specific evidence that changes them |
| Pair zoo: `handling_time` h | 25 | **40** | 32-47 | weighted Cmax 0.42 kJ/g/d (feeding season 0.46) read as energy = 9.3-10 %BW/d on 4.5 kJ/g zooplankton -> h 43 / 39. h 25 is 1.6-1.7x above any measured Cmax [D] |
| Pair zoo: `max_intake_rate` a | species 0.04 | **keep 0.04** (alternative: pair override 0.025) | 0.025-0.04 | a sets winter intake, where herring feed in the linear part of the curve (A17 zooplankton 0.2-0.9 t/km2 Dec-Feb); section 162 found winter intake binding. With 0.04 the half-saturation 1/(a h) falls from 1.0 to 0.63 t/km2; 0.025 would keep 1.0 but cut winter intake 30-37 % (judgement) |
| Pair zoo: `functional_response` | 2 | **keep 2** | - | section 162; no member evidence for Type III |
| Pair zoo: `dark_ratio` / `light_threshold_deg` | 0.1 / -8 | **0.25** / keep -8 | 0.1-0.34 | pout and blue whiting feed as much at night as by day (dark_ratio ~1; Onsrud et al. 2004; Skagerrak blue whiting night-biased). Weighted night/day attack ratio 0.26 (0.18 winter, ~0.3 in Q3). Judgement |
| Pair zoo: `visibility_floor` | 0.5 | **keep** | - | no member-specific evidence |
| Pair zoo: `assimilation_factor` | 0.8 | **keep** | - | - |
| Pair zoo: `predation_mortality` (M2 on zooplankton, /yr) | 5.24 | **1.6** | 1.2-3.1 | Q/B 5.0 x 3.33 t/km2 x zooplankton share 0.91 = 15 t/km2/yr; / 9.72 t/km2 (start 35 000 t) = 1.55; / 6.76 (A17 resident mean) = 2.2 [D]. **Knock-on for the zooplankton budget: see the flag below** |
| Pair benthos (`planktivorous_fish_preys_on_benthic_community`) | unchecked | **stay unchecked** | - | benthos is ~6 % of the group diet (pout 0.3 x 7.5 %, blue whiting 0.12 x 10.5 %, herring 2-5 %): 1 t/km2/yr = M2 0.01 /yr on the benthos [D] |
| Pair phytoplankton, fish larvae | unchecked / no pool | **stay** | - | sprat's 10 % phytoplankton = 0.3 % of the group diet; herring's 5-11 % fish larvae have no pool (count them as zooplankton) |

### Flags for the caller
1. **Prey quality.** Every predator pair on this group uses the static quality
   4000 + 0.5 x reserve. With 6250 it is 7125 MJ/t, not 7500. The section-148 energy rule then
   gives h = 130 x 7125/4500 = 206 for a gadoid-type predator (217 today). The predator reviews
   should use 7125 if the reserve changes.
2. **Zooplankton budget.** The 5.24 was 25 /yr x 21 % (the North Sea Ecopath pelagic pool's share of
   copepod mortality, sandeel included). At the site's 3.3 t/km2 the fish take ~1.6 /yr. The
   zooplankton's M1 19.7 was derived as 25 x (1 - 0.21). If the fish M2 drops to 1.6, either M1 rises
   to ~23 (total copepod mortality kept at 25, derived g unchanged) or g falls by ~6 %. That is the
   zooplankton review's call. Check: 5.24 x 9.72 t/km2 / 3.33 t/km2 / 0.91 would need a group Q/B of
   ~17 /yr, 3x the literature.
3. **Derived growth and kappa.** A new h moves s* (ideal-conditions reserve, section 160) and so the
   loader-derived `growth_rate`. A new M1 and r_max move it too. `income_growth_kappa` 0.75 was set
   where the ideal-conditions budget closes at today's s* (library comment), so it sits downstream of
   the same derivation: re-check that kappa I - C still covers the growth demand B g (s* - u) ec at the
   new ceiling (hand check: 0.75 x 90 - 15 = 52 MJ/t/tick at the ceiling in summer, far above the
   eat cost; the growth demand itself must come from the loader). Do not hand-edit `growth_rate`.
4. **M1 cut is conditional on the predator pairs.** M1 falls 0.58 -> 0.26 because the predation by
   rays, gurnards, mackerel, horse mackerel, hake and spurdog moves onto pairs. At Ecopath rates the
   new fish groups take ~0.72 /yr of this group, biomass-weighted (herring 0.56 at 50/50 stages,
   sprat 1.45, pout 1.34, blue whiting 1.25; predatory_fish ~0.55, benthivorous_fish ~0.17), against
   the old gadoid pair 0.30. Old loss budget M1 + M2 ~0.89 /yr; new ~0.26 + 0.72 + mammals 0.02 ~
   1.0. If the predator reviews land below ~0.6 /yr in total on this group, the budget (and g)
   shrinks: reconcile the three reviews before deriving.
5. **Ecopath parse.** The parse in `../Litteraturbaserad growth rate per FG/ecopath_ns1991/` shifts
   columns for the gadoid predators, which hit Norway pout and blue whiting (pout "77 % benthos",
   "crabs 0.65 /yr"). This review uses EcoBase model 457 (`../benthivores/data/ecobase/model_457.xml`)
   for all five rows. EcoBase 457 also differs from the report's Table 3.3 for juvenile herring
   (B 0.085, Q/B 12.5, EE 0.98 against 0.63, 5.63, 0.67), so juvenile-herring M0 is uncertain
   (0.03-0.43).

---

## 0. Weights and target

Annual-mean biomass in the grid, t/km2 [D]:

| Member | Central | Range | Basis |
|---|---|---|---|
| Herring | 2.60 | 2.1-3.4 | acoustic July grid mean 4.29 (2010-26) x summer-to-annual 0.6 (0.5-0.8) (`../Ekolodsdata planktivorer/acoustic_planktivores.md`, grid correction) |
| Blue whiting | 0.35 | 0.2-0.5 | GOV Q1+Q3 long-run 0.35 (`tables/long_run_species_by_group.csv`); Q3 visitor below 100 m, missed acoustically |
| Norway pout | 0.25 | 0.15-0.35 | resident near the bottom; GOV long-run 0.25, acoustic 2010-19 0.2-0.3 |
| Sprat | 0.10 | 0.05-0.2 | acoustic July grid 0.16, Q1 a tenth of Q3 |
| **Sum** | **3.30** | 3.15-3.8 | = 11 900 t; keep the 12 000 t start |

- Shares: herring 0.788, blue whiting 0.106, pout 0.076, sprat 0.030 (low-herring 0.67 / 0.16 /
  0.11 / 0.06; high-herring 0.90 / 0.05 / 0.04 / 0.01).
- The GOV long-run shares (herring 50 %, blue whiting 25 %, pout 18 %) are a bottom-trawl floor for
  pelagic herring. They are not used as weights.
- Season: blue whiting is there in Jul-Sep only. In Q3 the composition is ~78 % herring, ~15 %
  blue whiting; in winter ~88 % herring, ~11 % pout. Annual-mean weights are used for annual fields;
  the seasonal shares are used only for the temperature and light checks.
- Herring in the box are juvenile-heavy: by biomass 1-2 ringers 89 % in Q1, age 0-1 64 % in Q3
  (`../../Sillens vinterbudget/winter_budget.md`, section 4). Juvenile and adult herring are weighted
  50/50 where Ecopath splits them.

## 1. Herring-calibrated fields

### 1.1 r_max
Same method as `../../Den pelagiska poolens reglering/productivity.md`: FishBase CMSY prior, Ricker d
from ICES Standard Graphs, r_now = r_max (1 - ln d / 3).

| Member | prior r_max | d | r_now | Source |
|---|---|---|---|---|
| Herring (NSAS/WBSS Skagerrak mix) | 0.44 | 1.7 / 2.25 | **0.35** (0.27-0.40) | productivity.md [D] |
| Sprat | 0.48 | 0.8, capped at 1 | **0.48** (0.48-0.56) | productivity.md [D] (capped, its conservative choice) |
| Norway pout | 0.50 (0.33-0.75) | 1.27 (1.02-1.59) | **0.46**; 0.38 with the real T of 2 yr | evidence_pout_bluewhiting.md 1.1-1.4 [F/D] |
| Blue whiting | 0.47 (0.31-0.70) | 1.13 (no real decline) | **0.45** | same [F/D] |

- Weighted: **0.373** (0.30-0.46; low-herring 0.39, high-herring 0.36) [D].
- The library's 0.34 was a site mix with mackerel (r_now 0.25, 19 %). Mackerel moves to
  `predatory_fish`, so herring alone already gives 0.35.
- Norway pout's prior looks low for a fish that spawns once at age 2: the CMSY rule r ~ 3/T gives
  ~1.5, and SP/TSB was ~0.7 in 1984-2000. At pout r 1.0 the group value is 0.415. Pout is short-lived
  and productive, but at 7.5 % of the biomass it moves the group by at most +0.04.
- Realised surplus production today is low for both new members: pout SP/TSB 0.14 (median 0; TSB
  falling 11 %/yr, SSB below Blim), blue whiting 0.19 [D]. That supports a K, not a higher r.
- **New value 0.37.** It is a recomputation, not a new mechanism; 0.34 sits inside the range.

### 1.2 regional_carrying_capacity K
- Brief: K = B*/(1 - F/r). The library used the surplus production P 0.115 in place of F.
- r 0.34, P 0.115: 5.03 (today's 5.0). r 0.34, F 0.10: 4.72. r 0.373, F 0.114: 4.79.
  r 0.373, P 0.115: 4.81 [D].
- **4.8 with r 0.37; keep 5.0 if r stays 0.34.** The difference (-4 %) is below the uncertainty of
  B* (3.15-3.8).

### 1.3 natural_mortality M1
M1 = M0 + predation by groups NOT in the model. Under the new group set the model holds
benthivorous fish (cod, haddock, flatfish, rays, gurnards), predatory fish (saithe, whiting,
mackerel, horse mackerel, spurdog, hake, anglerfish, ling), seals, porpoises and seabirds (muted;
its predation is not put in M1, as before).

North Sea Ecopath 1991, EcoBase 457, mapped to the new groups (`tables/planktivores_ecopath_stdout.txt`) [D]:

| Member | M0 = P/B (1-EE) | M2 by predatory_fish | M2 by benthivorous_fish | M2 by seabirds / seals / porpoises | not in model |
|---|---|---|---|---|---|
| Herring adult | 0.23 | 0.10 (saithe 0.08) | 0.02 | 0.005 / 0.001 / 0.001 | ~0 |
| Herring juvenile | 0.03 (EcoBase) - 0.43 (report Table 3.3) | 0.35 | 0.64 (juvenile cod 0.59) | 0.014 / 0 / 0.004 | 0.024 (squid, baleen whales) |
| Sprat | 0.37 (0.44 report) | 1.33 | 0.12 | 0.039 / 0.001 / 0.054 | 0.05 |
| Norway pout | 0.55 | 1.12 | 0.23 | 0 / 0 / 0.048 | 0.015 (squid) |
| Blue whiting | 0.27 | 1.10 | 0.15 | 0 / 0 / 0.009 | 0.21 (squid) |

Member M1, /yr:
- **Herring 0.26** (0.10-0.35): adult M0 0.23-0.25 for both stages (juvenile M0 is 0.03 or 0.43
  depending on the Ecopath version), plus ~0.01 squid / baleen whales. SMS uses 0.1 (low end).
- **Sprat 0.42** (0.20-0.49): M0 0.37-0.44 + 0.05. SMS M1 0.2.
- **Norway pout 0.30** (0.20-0.55): SMS M1 0.2, plus spawning death not in the predation M
  (Sparholt et al. 2002, biomass-weighted 0.32-0.36); every SMS predator of pout is a model group
  (hake 32 %, saithe 26 %, whiting 23 %, gurnard 9 %; SMS 2016-22 M2 1.89). The "high M" of pout is
  mostly M2, which belongs on the predator pairs.
- **Blue whiting 0.20** (0.10-0.40): the stock's total assessment M is 0.2; Ecopath M0 0.27 plus squid
  0.21 is the high end.

Weighted: **0.26 /yr = 1.8e-4 per tick** (0.11-0.38) [D]. The current 0.580 = pool M0 0.391 (with
sandeel, mackerel, horse mackerel) + 0.189 non-FG predation (sharks, rays, other fish). Most of that
0.189 is now in model groups (rays and gurnards in benthivorous_fish; spurdog, mackerel, horse
mackerel, hake in predatory_fish). So the cut comes from the new group set and from using herring's
own M0, not from the new members. The M2 that moves onto pairs (Ecopath: predatory fish 0.1-1.3,
benthivorous fish 0.02-0.6 per member) is for the predator reviews.

### 1.4 Fishing F
Stock-level convention (catch / mean total biomass, `../../reports/Fiskeridödlighet i Skagerrak idag.md`):
- herring 0.10 (library; the local Skagerrak take is ~0.005);
- sprat 0.45 (spr.27.3a4, 2025; `ices_pelagic_f.md` Q4);
- Norway pout 0.16 in 2016-25, 0.02-0.04 in 2024-25 (zero advice 2025-27, bycatch-only TAC);
- blue whiting 0.18 (0.21 in 2025), 99 % pelagic trawl far outside the Skagerrak [D, evidence note 1.5, 3].
- Local 3.a.20: pout 130-180 t/yr and blue whiting 33-78 t/yr incl. discards in 2022-24, > 98 %
  bottom otter trawl (shrimp-trawl discards). No directed fishery for either [D].
- Weighted **0.11** (0.01-0.20). **Keep 0.10**: the step is far inside the herring range. Gear split:
  blue whiting would go on the pelagic map, pout on both, but at < 0.01 /yr neither changes the 98 / 2
  split. A changed F would change the pelagic table's q calibration (caller's part).

### 1.5 Energy content and fat reserve
From `planktivores/evidence_cmax_energy.md` 1c (model form ED = lean + 0.396 kJ/g per fat %):

| Member | Lean ED (kJ/g) | Max fat % | Max reserve (MJ/t) | Source status |
|---|---|---|---|---|
| Herring | 4.0 | ~18 adult (Jun-Jul); ~15 juvenile 15-20 cm in Q3 | 7000 (adult; library); ~6100 juvenile | [F]/[S] Pedersen & Hislop 2001 via JNCC 633 |
| Sprat | ~4.0 | 15-20 (Oct-Dec) | ~6700 | [F] fat series |
| Norway pout | ~4.0 | ~5 | ~2000 | proxies only (Torry; Trisopterus) |
| Blue whiting | ~3.8 | ~9 (Aug; mostly liver) | ~3600 | [F] fat series |

- `energy_content` **4000, keep**.
- `max_energy_reserve` weighted **6250** (5400-6650); with the box's juvenile herring at 6100 it
  would be ~5550 [D]. The library value is an adult-herring value; the lean gadoid members and the
  juvenile herring pull it down 10-20 %. Recommended: 6250 (keeps herring's adult anchor, adds only
  the member effect). Herring dominance means 7000 is defensible if the section-140 winter
  calibration is not to be touched; then say so in the library comment.
- Prey quality at fill 0.5: 4000 + 0.5 x 6250 = 7125 MJ/t (flag 1).
- `maintenance_level` 0.2 (starvation below ~2 % fat): keep. Pout and blue whiting live at 2-5 %
  fat, but the group's reserve is a mean over a herring-dominated pool.

### 1.6 Temperature: Q10, t_ref, layer
- No respirometry for pout or blue whiting; cod and teleost proxies give Q10 2.2-2.4 [S]. Sprat 2.2
  (Meskendahl 2013) [F]. **Keep Q10 2.2.**
- t_ref 14 C means the resting cost 8 MJ/t/tick is a summer herring value. Keep.
- Layer: pout and blue whiting live at ~6-10 C (50 m layer or deeper). With month-specific shares the
  group multiplier changes by +1 % (Dec-Mar) to -9 % (Jul-Sep; an upper bound, the script puts
  blue whiting at 1.4 t/km2 in Q3 against GOV ~0.7-1.0), -3 % over the year
  (`scripts/planktivores_temperature.py`) [D]. **Keep `mean_0_50m`.**

### 1.7 Income growth and feeding season
- Pout: income feeder, year-round, dips during spawning (Jan, Mar-Apr) and in Aug; fat 4-5 %.
- Blue whiting: feeds Apr-Sep, stores liver fat (2 -> 9 %) and spends it on spawning.
- Sprat: like herring.
- Weighted season ~6.0 months (pout counted as 12) [D]; g would fall 8 %. **Keep kappa 0.75 and
  5.5 months**: the rule (no food, no growth) fits all members, and 0.5 month is within the
  herring uncertainty. Judgement.

### 1.8 Depth and light behaviour
- Herring bite by sight (dark_ratio 0.1). Pout shows no day/night difference in stomach fullness on
  krill (Onsrud et al. 2004, Oslofjord); Skagerrak blue whiting feed mostly at night (evidence note 5).
  Both read as dark_ratio ~1 (the engine treats >= 1 as no light effect).
- A one-pool group has one night/day attack ratio: sum w_i dr_i = 0.79 x 0.1 + 0.03 x 0.1 + 0.076 x
  1 + 0.106 x 1 = **0.26** (0.18 in winter without blue whiting, ~0.3 in Q3) [D].
- **0.25 (range 0.1-0.34), judgement.** Because the daylight multiplier is normalised to an annual mean
  of 1, this flattens the diel and seasonal cycle of a; it does not raise the mean. Keep -8 deg.
- Vertical habitat (pout and blue whiting near the bottom below 100 m) has no counterpart in a
  one-layer cell. It would argue for a different `visibility_floor` towards some predators (cod,
  seals), which is for the predator pairs.

## 2. Cmax and handling time on zooplankton

Full derivation: `planktivores/evidence_cmax_energy.md` sections 1a-1b, 2.

- Herring: Wisconsin Cmax (Rudstam 1988; CA 0.642, CB -0.256, Thornton-Lessem f(T)), 60 g, 0-50 m
  temperatures: 0.165 g/g/d over the year, 0.180 in Apr-mid-Sep. It is a wet-mass ration for
  zooplankton of ~2.6-3.6 kJ/g; read as energy at 2.85 kJ/g: **0.47 kJ/g/d** (season 0.51) [F/D].
- Sprat (10 g, herring function): 0.75 kJ/g/d. Norway pout (25 g) and blue whiting (100 g): no Cmax
  exists; cod proxy (Hansson 1996, fish food 4.18 kJ/g) at 50 m temperatures: 0.18 and 0.135 kJ/g/d [D].
- Weighted in energy space: **0.42 kJ/g/d** (feeding season 0.46) = 9.3-10.1 %BW/d of 4.5 kJ/g
  zooplankton -> **h 43** (season 39). Range over the energy reading: 0.38-0.57 kJ/g/d, h 32-47.
- Current h 25 = 0.72 kJ/g/d = 1.6-1.7x the central Cmax. It equals Cmax only if the wet-mass
  ration is applied to 4.5 kJ/g prey, against the project's energy-limit convention (section 148).
  The gadoid ceiling was 1.6x cod Cmax for the same reason (resume 208.2).
- **New h 40** (between the annual 43 and the feeding-season 39; the ceiling binds in summer, when
  herring eat near it). Ceiling 0.025 t/t/tick = 10 %BW/d = 0.45 kJ/g/d.
- Biggest uncertainty: the Wisconsin herring Cmax is 3-4x the ad-libitum lab rations it is said to
  come from (De Silva & Balbontin 1974: 3.5-12.7 %BW/d); a curve through those gives h ~110-120. Field
  rations of adult NSS herring (Bachiller et al. 2018, ~5-6 %/d at 2-6 C) sit at the Wisconsin value,
  so it is kept as Cmax and the lab curve is a lower bound.
- Energy check by hand: at h 40 the ceiling pays 0.025 x 0.8 x 4500 = 90 MJ/t/tick against an eat
  cost of 8 x 1.9 x Q10 factor = 7.6-15 MJ/t/tick (Mar-Sep). The eat cost closes with a wide margin
  at the ceiling; whether kappa 0.75 still covers the growth demand at the new s* needs the loader
  (flag 3). At the same visible zooplankton as in section 160.4 (summer intake ~400 MJ/t/day
  at h 25, i.e. ~1.3 t/km2 visible), h 40 with a 0.04 gives ~300 MJ/t/day (-25 %) [D].
- Half-saturation: 1/(a h) = 1.0 t/km2 today (a 0.04, h 25). At h 40 it is 0.63 t/km2 with a 0.04,
  or 1.0 t/km2 with a pair override `max_intake_rate` 0.025. **Keep a 0.04.** In winter the herring
  feed in the linear part of the curve (A17 resident zooplankton 0.2-0.9 t/km2 in Dec-Feb, about half
  of it visible), so intake there is ~a x B_visible: a 0.025 would cut winter intake by 30-37 %
  (at 0.3 t/km2 visible: 0.0081 -> 0.0058 t/t/tick) [D]. Section 162 raised the visibility floor and
  cut interference precisely because winter intake was binding, and rejected changes that only cut
  winter intake. The literature half-saturations (15-40 t/km2 column-equivalent,
  `../../Sillens vinterföda/herring_functional_response.md`) were weighed there and lost to the winter
  evidence. So h 40 lowers only the summer ceiling (-37 %); the alternative 0.025 is listed with its
  winter cost (judgement).

## 3. The zooplankton pair

- `functional_response` 2, `visibility_floor` 0.5, `assimilation_factor` 0.8: **keep** (section 162;
  nothing in the new members argues otherwise).
- `dark_ratio` 0.25, `handling_time` 40, `max_intake_rate` kept at 0.04: sections 1.8 and 2.
- `predation_mortality` (M2 on zooplankton), consumption-based [D]:

| Member | Q/B (/yr) | zooplankton share | zooplankton eaten per t (/yr) | Source |
|---|---|---|---|---|
| Herring | 4.7 (juvenile 5.63, adult 4.34; Heath 2005 ration 4.4) | 0.95 (EcoBase 457: 0.95-0.98) | 4.47 | Ecopath, Heath |
| Sprat | 6.0 | 0.80 | 4.80 | Ecopath |
| Norway pout | 4.5 (4-5.05) | 0.70 (0.3-0.8) | 3.15 | evidence note 4; EcoBase 0.76 |
| Blue whiting | 7.5 (6-9) | 0.77 | 5.78 | Bachiller 2018 ~6; Ecopath 9.06 |
| **Group** | **5.0** | **0.91** | **4.51** | [D] |

- Consumption 4.51 x 3.33 = **15.0 t/km2/yr**.
- / 9.72 t/km2 (manifest start 35 000 t / 3600) = **1.55 /yr**; / 6.76 t/km2 (A17 resident 12-month
  mean, `tools/probes/a17_score.py`) = 2.22 /yr. Range over Q/B, diet and the weight sets: 1.2-3.1.
- **New value 1.6** (same denominator as the start biomasses). It matches the literature share:
  15 t/km2/yr is 15-25 % of the site's zooplankton production (zooplankton_consumption.md).
- Other prey pools: benthos ~6 % of the diet (1 t/km2/yr, M2 0.01 /yr on 95 t/km2 benthos) - leave
  the pair unchecked; a checked pair would let the herring-dominated group eat benthos at herring's
  ceiling. Fish larvae (herring 5-11 %) have no pool; they are mesozooplankton-sized and are
  counted in the zooplankton share. Blue whiting eats ~10 % fish (mostly pout and mesopelagics) -
  within-group, out of the budget (section 160).

## 4. Predators of this group in the model (list only)

The other reviews derive these pairs. Ecopath 1991 indication of M2 on the members in brackets.
- `benthivorous_fish`: cod (juvenile and adult; on herring juveniles, pout, blue whiting), haddock
  (pout), rays, gurnards, flatfish (minor).
- `predatory_fish`: saithe (herring, pout, blue whiting), whiting (sprat, pout, juvenile herring),
  hake (blue whiting, pout), mackerel (sprat), horse mackerel (pout, blue whiting, juvenile herring),
  anglerfish, ling, spurdog.
- `seals` (herring ~5 % of the diet; pout in Skagerrak scats).
- `porpoises` (herring and sprat; pout, whiting).
- `seabirds` (muted).
- Within the group: blue whiting on pout and small clupeids (out of the budget).

## 5. Main uncertainties
1. The herring Cmax (Wisconsin vs lab rations; h 35-47 vs ~115).
2. The summer-to-annual factor for herring (0.5-0.8) sets the weights and the target.
3. Juvenile-herring M0 differs 15x between the two Ecopath versions; M1 0.11-0.38.
4. Norway pout r_max (prior 0.5 vs ~1-1.5 by generation time) - small at 7.5 % weight.
5. No measured seasonal energy density for Norway pout (proxies only); Pedersen & Hislop 2001 tables
   not obtained.
6. Pout and blue whiting are not closed populations in the box (larval drift in, adults out; blue
   whiting a summer visitor). Their r and K describe stock-scale rates, not local recruitment.
