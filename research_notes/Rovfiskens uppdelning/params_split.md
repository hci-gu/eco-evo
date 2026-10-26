# Splitting `predatory_fish` into a bed-feeding and a water-column group: derivations

Written 2026-10-26 for resume section 223.6 (fix76: the predators took 82 % of their diet from the
mobile epifauna through a pair that is saturated down to near-extinction of the prey). Biology first:
every value is a measurement, a mass balance of measured rations and diets, or a stated judgement (J).
No model output was used to set anything; no library, manifest, code, AGENTS.md or resume file was
edited (a training run is live). Tags: [F] full text / data read, [A] abstract, [S] secondary,
[D] computed here, J judgement. No new literature was searched: every fact below comes from the
evidence already in the repository, cited by file.

Sources reused (all relative to `research_notes/`):
- `Fiskgrupper i rutan/params_predatory_fish.md` and `predators/` (evidence A physiology, B population,
  C diet and consumption; `inputs/species_values.csv`, `diet_consumption_central.csv`,
  `pred_scores.csv`; `tables/pred_season.csv`, `pred_length_biomass.csv`, `ices_diet_pools*.csv`).
- `Fiskgrupper i rutan/params_top_predators.md` (seal and porpoise diets and rations),
  `params_benthivorous_fish.md`, `benthivores/tables/` (cod's fish prey), `tables/long_series_by_year.csv`.
- `Attackrater ur mötesteori/` (section 220 encounter method; `scripts/pred_encounter_a.py` is executed
  unchanged up to its own evaluation).
- `Infauna och rörlig epifauna/params_split.md` and `tables/pred_benthos_split_stdout.txt` (section 221).

Scripts (`scripts/`, run from this folder with `python3 -I`; outputs in `tables/`):

| Script | What it does | Output |
|---|---|---|
| `split_fields.py "../Fiskgrupper i rutan/predators" tables` | biomass-weighted species fields, diets over the model's prey groups, M2 per pair, gape-based accessible fractions, for two groupings | `split_fields_stdout.txt`, `split_fields_G1.json`, `split_fields_G2.json` |
| `split_encounter.py "../Attackrater ur mötesteori" "../Fiskgrupper i rutan/predators" tables 2000` | section 220 encounter a per group, four averaging variants, MC 5-95 %, night/day ratios; new pair "-> whiting as prey" | `split_encounter_stdout.txt`, `split_encounter.json` |
| `split_pairs_diagnostic.py tables "../Fiskgrupper i rutan"` | seal / porpoise / benthivore pairs on each group, derived growth (s*, g), budget-gate identity, saturation and demand per pair, the step-4 diagnostic, each group's own 2001-25 span | `split_pairs_diagnostic_stdout.txt` |

Reproduction check: with all ten members the scripts return the pooled library values: B 1.20, r 0.417,
F 0.182, K 2.13, ceiling 25.5, Q 4.00; encounter a (capable variant, dispersed herring) 0.0119 on
planktivores (library 0.012), 0.0015 on zooplankton (library 0.002 incl. the Staniland shrimp capture),
0.178 on benthivores (0.18); epifauna per edible tonne 0.326 without scores (library 0.33) and 0.086 with
scores (221's alternative).

---

## 1. Who forages where: the grouping

### 1.1 What the split must do in the engine

The engine has no vertical position for a fish group. The only things a split controls are which pairs
each group has (and their a, h, f), which group a prey sees as its predator, and the group's own fields.
A group with a checked, saturated pair can put its whole eat share on it at the Holling ceiling. So the
operational criterion is: **which members forage on the bed for bed invertebrates** (the mobile epifauna)?
Habitat (where the fish sits) is not the criterion; diet is. J.

### 1.2 Evidence per member

Diet = consumption-weighted shares of total food by mass (evidence C bottom line). B = benthos; the
epifauna share of B from the ICES stomach DB (`pred_benthos_split_stdout.txt`).

| Member (share of pooled biomass) | Where it sits | What it eats (P / Bf / W / Z / B) | Bed invertebrates in the diet | Verdict |
|---|---|---|---|---|
| Whiting (0.37) | demersal, near the bed by day; "pelagic whiting" feed at night (Pedersen 1999 [A], 2000 [A]); resident all year (FishMap [F]; Q1 74 % of the pooled group's survey biomass) | 0.28 / 0.08 / 0.15 / 0.28 / 0.21 (Skagerrak Apr 1991 B 0.25-0.35, box 1991 0.60, Skagerrak 2022 0.15, C/S North Sea 2022 0.21 [F]) | **0.14** (B 0.21 x epifauna 0.65: Pandalus 0.44 of B pre-2000, small shrimps, mysids, crabs; 0.61-0.74 by period) | **bed** |
| Saithe (0.41) | adults "close to the bottom, down at depths of more than 300 m during day time", rising at night (FishMap [F]); semi-pelagic schools on the trench slope Apr-Dec | 0.58 / 0.05 / 0.02 / 0.33 / 0.01 (Mehl 2005 [F]: B 0.3-0.5 %; Daan 1989 [S]: benthos <= 1 %) | **0.01** (one stomach of Crangon in 2024) | water column: sits near the bed by day but takes Norway pout, blue whiting and krill there, not bed invertebrates |
| Mackerel (0.08) | upper 0-50 m in summer, deep and non-feeding in winter (Jansen [F], FishMap [F]) | 0.30 / 0.01 / 0.01 / 0.65 / 0.02 | ~0.01 | water column |
| Spurdog (0.05) | DVM, median 198.5 m by day, 60 m at night (Klöcker et al. 2024 [F]) | 0.42 / 0.22 / 0.22 / 0.03 / 0.06 (Ecopath 1991 after Ellis 1996; no post-2000 by-mass diet) | ~0 (Skagerrak 2024: 2 g of infauna) | water column (J): fish eater that leaves the bed at night; its Bf share is a fish pair, not bed access |
| Horse mackerel (0.034) | near the bed by day, krill eater (Dahl & Kirkegaard 1986 [F]; Aug 2023 box 87 % krill) | 0.25 / 0.07 / 0.03 / 0.60 / 0.05 | 0.05 (small shrimps, mysids; 4-6 stomachs) | water column (J, borderline) |
| Hake (0.025) | demersal by day, pelagic feeding at night (Modica 2011 [F]) | 0.77 / 0.01 / 0.21 / 0 / 0.005 (Skagerrak Aug 2024, n 50 [F]: herring 65 %, whiting) | **0.005** | water column by diet; demersal by habitat (J) |
| Anglerfish (0.014) | bed ambush, sedentary | 0.35 / 0.30 / 0.25 / 0 / 0.02 | 0.01 | water column by diet (fish eater); J |
| Ling, velvet belly, pollack (0.016) | demersal / deep / saithe-like | proxies: anglerfish, horse mackerel, saithe (J) | small | water column (J) |

Findings:
1. **Bed-invertebrate feeding is whiting.** Whiting carries 92 % of the pooled group's benthos consumption
   (whiting 0.326 of 0.355 t/t-group/yr [D]) and 89 % of the pooled epifauna encounter a (layer variant,
   `split_encounter_stdout.txt`; juvenile whiting alone 1.6 km2/t/tick per edible tonne, adults 0.40).
   Hake, saithe, anglerfish and spurdog sit on or near the bed by day but eat fish or krill there.
2. **Saithe's feeding depth does not give it bed access.** Adult saithe are near the bottom by day on the
   trench slope, but their prey there (Norway pout 5-10 m above the bed, blue whiting in a near-bottom layer
   on the slope, krill in the deep scattering layer) are the model's planktivores and zooplankton. Benthos
   is 0.3-1 % of every saithe diet found.
3. **Juvenile vs adult whiting.** Juveniles (< 20 cm, 25 % of whiting biomass) eat krill and hyperbenthos;
   adults >= 25 cm are "almost exclusively piscivorous" in the southern North Sea (Lauerburg 2018 [F]) but
   the box and Skagerrak adults still take Pandalus (pre-2000 Skagerrak B 0.25-0.35 [F]). Both stages forage
   on the bed (scores 0.40 / 0.18). A functional group has no stage structure, so whiting goes whole; the
   epifauna term is 57 % juvenile, 43 % adult in the group a [D].
4. **The habitat grouping (hake, spurdog, anglerfish with whiting) is not supported by diet for hake**
   (Bf 0.01, B 0.005), and only partly for spurdog and anglerfish, whose bed-related prey is settled
   benthivorous fish, which saithe also eats (SMS 2023: saithe is the largest predator on haddock 1+, M2
   0.19 of 0.40 [F, evidence B]). The benthivore criterion therefore does not separate the members; the
   bed-invertebrate criterion does, and it isolates whiting.
5. **Two-way split: the evidence supports it, with the bed group = whiting.** A third group (hake + spurdog
   + anglerfish + ling, 0.10 of the pooled biomass, 0.12 t/km2) would be thin and has no pair that whiting
   or saithe lack. Mackerel + horse mackerel (0.11, zooplankton 0.6-0.65) are close to planktivores, but
   moving them would change the planktivore group; not recommended here.

**Recommended grouping G1 (diet criterion):** bed = whiting; water column = saithe, mackerel, spurdog,
horse mackerel, hake, anglerfish, ling, velvet belly, pollack.
**Variant G2 (habitat criterion), computed for comparison:** bed = whiting, hake, spurdog, anglerfish,
ling, velvet belly; water column = saithe, mackerel, horse mackerel, pollack. G2 gives hake, spurdog and
anglerfish bed access they do not use (+27 % bed-access biomass, demand at the ceiling 5.3 instead of
4.5 t/km2/yr), turns their predation on whiting into an unbooked within-group transfer, and leaves the
water-column group without the members that eat whiting (cross-pair M2 on the bed group 0.057 instead of
0.157).

Names (J): the bed group could be `bed_feeding_predators` (display "Whiting"); keeping the id
`predatory_fish` for the water-column group saves renames but changes its meaning. "Demersal / pelagic"
names would mislead (saithe, hake and anglerfish are demersal by habitat).

---

## 2. Group fields (G1)

Same construction as section 209 (`weighted_fields.py`): biomass-weighted species values from
`inputs/species_values.csv` with the long-run (2001-25) GOV shares (`pred_season.csv`), corner ranges.
B* = 1.20 t/km2 x the members' share (the top of the long-run span, user's choice; `gadoid_level.md`
section 6). Per-group check of that convention (`split_pairs_diagnostic_stdout.txt`): whiting 2001-25
mean 0.378 t/km2 (median 0.384, 0.13-0.79; 2021-25 0.54), so B* 0.446 = x1.18; the rest mean 0.642
(median 0.53, 0.14-2.08; 2021-25 1.15), B* 0.754 = x1.17. Both are the same x1.18 as the pooled group.

| Field | bed (whiting) | water column | Basis |
|---|---|---|---|
| member shares | whiting 1.00 | saithe 0.651, mackerel 0.128, spurdog 0.080, horse mackerel 0.055, hake 0.040, anglerfish 0.022, ling 0.011, velvet belly 0.008, pollack 0.007 | `pred_season.csv` pooled Q1+Q3 [D] |
| B* (t/km2; t on 3600 km2) | **0.446** (1605 t) | **0.754** (2715 t) | 1.20 x 0.3715 / 0.6285 |
| `r_max` (/yr) | **0.49** (0.44-0.49) | **0.374** (0.32-0.42) | Euler-Lotka r_now per species (evidence B): whiting 0.49; saithe 0.44, mackerel 0.25, spurdog 0.07, horse mackerel 0.33, hake 0.26, anglerfish 0.47, ling 0.55, pollack 0.63, velvet belly 0.10 |
| `natural_mortality` M1 (/yr; /tick) | **0.25** (0.20-0.32); 1.712e-4 | **0.261** (0.18-0.30); 1.791e-4 | whiting SMS residual 0.20 + grey seal / birds 0.053 (cannibalism 0.094 and saithe 0.085 excluded: within group / a pair now); saithe 0.29 (assessment M), mackerel 0.15, spurdog 0.10, horse mackerel 0.30, hake 0.45, anglerfish 0.25 (evidence B) |
| F bottom / pelagic (/yr) | 0.075 / 0.005 (total 0.08, 0.035-0.15) | 0.213 / 0.029 (total 0.242, 0.12-0.29) | catch / total biomass, ICES (evidence B); saithe 0.30 carries the water-column F |
| `bottom_trawling` biomass_factor at 100 | **0.001704** (0.00075-0.0032) | **0.004836** (0.0024-0.0059) | q = F_bot / 1460 / 3.0142 (today's mean effort, section 161) |
| `pelagic_trawling` biomass_factor at 10 | **0.001372** | **0.007983** | q = F_pel / 1460 / 0.02496 |
| `regional_carrying_capacity` K = B*/(1 - F/r) | **0.533** (0.48-0.68) | **2.14** (1.05-11.0) | F/r 0.16 vs 0.65: the water-column K is fragile; saithe F 0.11-0.38 drives it |
| Cmax (kJ/kg/d) -> energy ceiling (MJ/t/tick) | 104 (80-150) -> **26.0** (20-37.5) | 100.7 (77-156) -> **25.2** (19.3-38.9) | ICES gastric evacuation with cod-calibrated fill at 8.4 C (evidence A); mackerel 150 at 13 C |
| `resting_metabolism` (MJ/t/tick) | **4.2** (3.3-5.0) | **3.31** (2.8-5.0) | SMR at the at_50m annual mean: whiting 4.2 (59 mg O2/kg/h [S]); saithe 3.5, mackerel 3.4, spurdog 2.1, anglerfish 1.2 |
| `metabolism_q10` / `metabolism_t_ref`, layer | **2.3** / annual_mean, at_50m | **2.3** (2.27) / annual_mean, at_50m (J: saithe near the bed by day; mackerel's summer mixed layer not represented) | cod 2.2-2.6, ICES GER 2.25, mackerel 2.0, spurdog 2.5 |
| `feeding_cost` (x RM) | **2.0** (2.03) | **2.15** | routine 1.5 x SMR + SDA 0.17 of half the ceiling: 1.5 + 0.17 x 0.5 x ceiling / SMR [D] |
| `movement_cost` (x RM) | 2.0 | 2.0 | cruising at Uopt 2-2.5 x SMR (evidence A Q3) |
| `movement_speed` | 1.0 | 1.0 | Uopt whiting 1.0 BL/s = 5.8 cells/tick at 27 cm; saithe 15; not binding (evidence A Q4) J |
| `energy_content` (lean, MJ/t) | **3900** | **4240** | whiting 3.9 kJ/g; saithe 4.0, mackerel 4.7, spurdog 6.0, hake 3.7 |
| `max_energy_reserve` (MJ/t) | **2000** (1500-2500) | **2600** (1950-3250) | gadoid liver 2000; mackerel 6000, horse mackerel 3800 |
| `prey_includes_reserve` / `reserve_reference_fill` | true / **0.45** (prey quality 4800 MJ/t) | true / **0.50** (prey quality 5530) | weighted mean reserve / max (whiting 900/2000) |
| `maintenance_level` u | 0.5 (engine convention; literature 0.1-0.2) | 0.5 (literature 0.0-0.2) | resume 187.4, as section 209 |
| `starve_rate` (/tick) | **0.011** (0.007-0.018) | **0.0078** (0.005-0.013) | 1.5 x SMR / lean energy / 0.15 (the section 209 construction at u 0.15; the pooled library kept 0.0088 as J) |
| `interference` w | **0.1** (0-0.1) | **0.1** | no field evidence; Rindorf et al. 2006 [F]: predator density not significant for whiting |
| `visibility_floor` (prey side) | **0.5** (0.4-0.6) J | **0.7** (0.6-0.8) J | the pooled 0.6 was a judgement straddling "whiting on the bed hides better, midwater schools of saithe and mackerel badly" |
| `income_growth_kappa` | not set | not set (mackerel is 13 %) | as section 209 |
| Q/B (/yr); consumption Q (t/km2/yr) | 4.2 (3.3-5.6); 1.87 | 2.83 (2.6-3.6); 2.13 | SMS 2023 per age on box sizes, Lauerburg 2018 (whiting); evidence C 3.4 |
| derived `growth_rate` (diagnostic, the loader computes it) | ~0.0041 | ~0.00175 | closed form s* (section 160) with the checked M2 below; pooled library 0.00228 |

Residence (evidence B): whiting all year; the water-column group is ~0.8 present (saithe adults Apr-Dec,
mackerel Jun-Nov, horse mackerel Aug-Dec, hake Apr-Sep). Annual-mean B x annual Q/B is consistent (as for
the pooled group); the closed pool charges the visitors stock-level r and F all year, now concentrated in
one group.

---

## 3. Diets and pairs (G1)

### 3.1 Diet over the model's prey groups

`split_fields.py`: member diets from `diet_consumption_central.csv` (proxies for ling, velvet belly,
pollack J), consumption-weighted by Q/B x B. B is split into epifauna / infauna by the member's epifauna
share (whiting 0.65, hake 0.99, horse mackerel 1.0, saithe 1.0, spurdog 0, mackerel 0.5, anglerfish /
ling 0.5 J). W (fish of the pooled group) is split by the prey species (J, from `ices_diet_pools_taxa.csv`
and Ecopath 1991): saithe, mackerel, pollack: whiting; whiting: whiting 0.95, hake 0.04, horse mackerel
0.01; spurdog: whiting 0.5, mackerel 0.33, horse mackerel 0.17; horse mackerel: horse mackerel 0.88,
whiting 0.12; hake: whiting 0.9; anglerfish, ling: whiting 0.4, hake 0.4, horse mackerel 0.2.

| Diet share of total food | planktivores | zooplankton | epifauna | infauna | benthivores | own group | other group | other |
|---|---|---|---|---|---|---|---|---|
| bed (whiting) | 0.280 | 0.280 | 0.137 | 0.073 | 0.080 | 0.142 | 0.007 | 0 |
| water column | 0.523 | 0.342 | 0.012 | 0.005 | 0.061 | 0.012 | 0.033 | 0.013 |
| pooled (check) | 0.409 | 0.313 | 0.070 | 0.037 | 0.070 | 0.094 | - | 0.007 |

The two groups add back to the pooled flows exactly (epifauna 0.282, planktivores 1.638 t/km2/yr).

### 3.2 Pairs

M2 = Q x diet share / prey B (prey B: planktivores 3.3, zooplankton 9.72, epifauna 7.88 engine
equilibrium mean, infauna 93.6, benthivores 1.53, the other predator group its B*). h = E_prey / the
predator group's energy ceiling (section 148). a = section 220 encounter rate averaged over the group's
stages, "layer" variant: per tonne of the whole group, members counted when their capability score
>= 0.1 (they forage in the prey's layer and can take it), no graded score (J; variants in 3.3).
Ranges of M2: the group's corner Q/B x the dominant member's diet-share range (saithe for the water-column pairs, whiting for the bed pairs) [D]. Night/day = the encounter model's a_night / a_day (220.3 decision 5: derived ratios); thresholds -5 deg kept
on predator pairs (not re-derived, as 220). Type II everywhere (Rindorf et al. 2006: negative switching).
Assimilation 0.8 throughout (0.7 on infauna).

| Pair | diet share | M2 on prey (/yr) | h | a central (MC 5-95 %) | accessible_fraction | x = a h f B today -> share of ceiling | dark_ratio / threshold | checked? |
|---|---|---|---|---|---|---|---|---|
| bed -> planktivorous_fish | 0.280 | **0.159** (0.07-0.57) | **274.0** | **0.021** (0.0068-0.046) | 1 | 19 -> 0.95 | **0.33** / -5 (stomach-based 0.70) | yes; pair visibility_floor 0.25 kept |
| bed -> zooplankton | 0.280 | **0.054** (0.003-0.21) | **173.1** | **0.0018** (0.0009-0.0068) | 1 | 3.0 -> 0.75 (winter 0.13) | **0.78** / -5 | yes |
| bed -> mobile_epifauna | 0.137 | **0.032** (0.007-0.083) | **144.2** | **0.69** per edible tonne (0.26-2.9) | **0.13** (0.06-0.25; 221) | 98 -> 0.99 | none | yes |
| bed -> benthivorous_fish | 0.080 | **0.098** (0.04-0.34) | **173.1** | **0.21** (0.065-0.78) | **0.05** (0.03-0.15; whiting gape ~12 cm + 0-group, J) | 2.9 -> 0.74 | **0.64** / -5 | yes |
| bed -> benthic_infauna | 0.073 | 0.0015 | (179 if checked) | not derived | - | - | - | no (as 221: negligible for the prey; a saturated pair would pay the ceiling on infauna) |
| bed -> water column | 0.007 | 0.019 | (213) | - | - | - | - | no (whiting's W is cannibalism; hake and horse mackerel are traces) |
| bed -> bed | 0.142 | (0.60, unbooked) | - | - | - | - | - | no (section 160) |
| wc -> planktivorous_fish | 0.523 | **0.338** (0.19-0.55) | **283.1** | **0.0117** (0.0046-0.033) | 1 | 11 -> 0.92 (0.79 at 35 % visible) | **0.39** / -5 (stomach 0.57) | yes; floor 0.25 kept |
| wc -> zooplankton | 0.342 | **0.075** (0.03-0.17) | **178.9** | **0.0006** (0.0004-0.0022) | 1 | 1.05 -> 0.51 (winter 0.05) | **0.30** / -5 (stomach 0.86; mackerel feeds by day) | yes |
| wc -> benthivorous_fish | 0.061 | **0.084** (0.05-0.23) | **178.9** | **0.053** (0.018-0.17) | **0.24** (gape: saithe 19 cm, spurdog 27, anglerfish 45, + 0-group) | 3.5 -> 0.78 | **0.48** / -5 | yes |
| wc -> bed (whiting) | 0.033 | **0.157** (0.07-0.28) | **190.8** (E 4800) | **0.0106** (0.004-0.037) | **0.47** (whiting below the members' gape) | 0.43 -> 0.30 | **0.48** / -5 | yes (new learnable cross-group pair) |
| wc -> mobile_epifauna | 0.012 | 0.0033 | (149) | (0.049; horse mackerel 82 %) | - | - | - | **no: the point of the split** |
| wc -> infauna, wc -> wc | 0.005, 0.012 | 0.0001, (0.035) | - | - | - | - | - | no |
| benthivorous_fish -> bed | (cod: whiting is its whole W) | **0.230** (0.13-0.59) | **177.1** (4800/27.1) | 0.026 (220, kept) | 1 | - | 0.10 / 0 (220) | yes |
| benthivorous_fish -> water column | 0 | 0 | - | - | - | - | - | no |
| seals -> bed | whiting 0.44 of the seals' predator share (0.3-0.6, J) | **0.0132** (0.009-0.018) | **22.6** | 0.05 (J, kept) | 1 | - | none | yes; visibility_floor 0.85 (J: on the bed, as benthivores) |
| seals -> water column | 0.56 | **0.0099** (0.007-0.012) | **26.0** | 0.05 | 1 | - | none | yes; floor 0.75 (kept) |
| porpoises -> bed | whiting 0.64 (0.52-1.0, J) | **0.0187** (0.015-0.029) | **28.6** | 0.20 (kept; Ransijn 2021 whiting selectivity) | 1 | - | none | yes; floor 0.75 (J: seabed masking) |
| porpoises -> water column | 0.36 | **0.0062** (0-0.008) | **28.6** | 0.20 (J) | 1 | - | none | yes; floor 0.90 (kept) |

Notes:
- **wc -> bed (new pair).** Saithe 0.066, hake 0.044, spurdog 0.030, anglerfish + ling 0.017 /yr of the
  0.157. Independent check (not used): SMS 2023 saithe on whiting 1+ 0.085 /yr [F]. Geometry: the
  benthivore-pair near-bed form with a whiting layer H 10 m (20-5 m, J; whiting within ~5-30 m of the bed by
  day, Pedersen 1999 [A]); capability scores = the benth_fish scores (proxy, J).
- **Benthivores -> bed.** The pooled 0.085 /yr on 1.20 t/km2 (0.10 t/km2/yr, large cod and grey gurnard)
  is all whiting (`benthivores/tables/cod_fish_prey_pools.csv`: cod's W items are whiting), so on 0.446 t/km2
  it is 0.23 /yr. It rests on the long-run cod share 31 % (8.5 % today); SMS 2023 cod on whiting 1+ is 0.05.
- **Whiting's total M** in the model: M1 0.25 + booked M2 0.42 = 0.67 /yr, against SMS whiting 1+ total M
  0.685 [F] (but with a different split: SMS porpoise 0.19 vs 0.019 here, because the model's porpoise stock
  is 12 t against a survey ~84 t; cod 0.05 vs 0.23).
- **Whiting cannibalism** (0.142 of the diet) is from two post-2000 samples (Skagerrak 2022, 9 whiting;
  C/S North Sea 2022), 0 in 1991; read as M2 it would be 0.60 /yr on whiting, against SMS 0.094. The share is
  probably too high; it books nothing, but every other whiting pair's diet share is of total food, so a
  lower W would raise them by up to 1/(1 - 0.14) = 1.17.
- **Seal and porpoise a** stay the section 209 judgements per tonne of prey (0.05, 0.20); seals' h uses the
  energy ceiling 212.5 MJ/t/tick on each group's whole-body energy.

### 3.3 Attack-rate variants (structural uncertainty, central values)

| Pair | layer (recommended) | noscore (221 convention) | scored (score folded in) | capable (220 convention) |
|---|---|---|---|---|
| bed -> planktivores | 0.021 | 0.040 | 0.0092 | 0.030 |
| bed -> zooplankton | 0.0018 | 0.0019 | 0.0011 | 0.0032 |
| bed -> epifauna | 0.69 | 0.69 | 0.21 | 0.89 |
| bed -> benthivores | 0.21 | 0.54 | 0.11 | 0.31 |
| wc -> planktivores | 0.0117 | 0.0117 | 0.0046 | 0.0070 |
| wc -> zooplankton | 0.0006 | 0.0006 | 0.0005 | 0.0009 |
| wc -> benthivores | 0.053 | 0.147 | 0.017 | 0.066 |
| wc -> whiting | 0.0106 | 0.029 | 0.0033 | 0.013 |

Who sets each a (layer): bed pairs are all whiting (planktivores and benthivores adults only: juvenile
scores 0.04-0.05); wc -> planktivores mackerel 0.49, saithe 0.33, horse mackerel 0.12 (mackerel's large
N per tonne and speed, at score 0.18: the membership threshold is a judgement); wc -> zooplankton mackerel
0.49, saithe 0.28; wc -> benthivores and -> whiting saithe 0.77. The schooled-herring central (220) would
give bed 0.0093 and wc 0.0058 on planktivores with night/day ~1.3.

Note on today's library: the pooled pairs' a 0.012 / 0.002 / 0.18 are per tonne of the capable share
(section 220 convention), but since 221 removed predator_share they act per tonne of the whole group;
the epifauna 0.33 is per tonne of the whole group (noscore). The split is a chance to put every predator
pair on one convention.

---

## 4. Diagnostic (not a fitting step; `split_pairs_diagnostic_stdout.txt`)

Engine identities: intake per tonne at eat share 1 and full hunger a B / (1 + a h B), ceiling 1/h; demand at
the ceiling = B_pred x 1460 / h.

**Bed group on the epifauna.**
- Still saturated: x = a h f B = 98 (0.99 of the ceiling); half-saturation in whole-pool units
  1/(a h f) = **0.078 t/km2** (pooled today 0.16; MC low a 0.26 -> 0.21) against a 7.6 start. Concentrating
  the encounter on whiting raises the per-tonne a; nothing in the split gives the epifauna a density refuge.
- Demand: at the ceiling with the whole eat share **4.5 t/km2/yr** (pooled 11.9); at the diet share
  (field ration) **0.256**.
- Supply: maximum surplus r_gross K/4 = 0.879 x 11.9-12.2 / 4 = **2.6-2.7 t/km2/yr**; benthivores (1.11) and
  the trawl (1.04) leave **0.53**. At the ceiling, an eat share of 0.137 (its diet share) already takes 0.62
  (1.2 x the room), 0.30 takes 1.35 (2.6 x), 0.82 (fix76's diet) takes 3.7 (7 x).
- Two caveats on these bounds: the ceiling demand assumes hunger 1; at the derived s* 0.695 the hunger gate is (1 - s)/(1 - u) = 0.61, so a fed whiting group takes at most ~2.8 t/km2/yr (5 x the room). The room 0.53 is the loose bound (maximum surplus at K/2 minus today's other removals); at today's B 7.63 the consistent room is the pooled predators' budget 0.29 (surplus 2.51 - 1.11 - 1.04 - ... = the 2.44 the budget books).
- G2 (with hake, spurdog, anglerfish): half-saturation 0.090, demand at the ceiling 5.3.
- Reading: the split removes bed access from 63 % of the predator biomass (saithe and mackerel can no
  longer fall back on the bed) and cuts the worst case by 2.6x, but a whiting policy that feeds on the bed
  can still exceed the epifauna's production several times. The limit remains the learned eat share.

**Water-column group.**
- Planktivores: a 0.0117 is saturating at today's density: 0.92 of the ceiling with all planktivores
  visible, 0.79 at 35 % visible; half-saturation 0.30 t/km2. Demand at the ceiling **3.9 t/km2/yr**, at
  eat share 1 and today's density 3.6, at the diet share 1.11; against the predators' budget 0.50 x 3.3 =
  **1.65** (all predators 2.18) and the planktivores' gross production at B* ~3.4.
- Zooplankton: nearly linear (half-saturation 9.3 t/km2; 0.51 of the ceiling at the 9.72 start, 0.05 at a
  winter 0.5 t/km2): demand at eat share 1 at 9.72 is 3.15, at the diet share 0.73, budget 1.26.
- Can it pay? The budget-gate identity needs a pair delivering cost / C = 0.35 of the ceiling at eat share 1
  (whiting 0.41). The water-column group's best pair (planktivores) gives 0.79-0.92: feasible without bed
  pairs. Benthivores give 0.78 (accessible 0.24 of the pool), whiting 0.30. The bed group also gets 0.95 on
  planktivores, so neither group depends on the epifauna to close its budget.
- Reading: the planktivores become the water-column group's ceiling pair; a policy that puts its whole eat
  share there could take more than the planktivores produce. The pooled group had the same pair (a 0.012,
  x ~11) and in fix76 used it lightly (realised M2 0.19 -> 0.07 vs 0.5); with the bed closed to saithe and
  mackerel, the planktivores and the whiting cross pair are what to read after training.

---

## 5. Conflicts with engine conventions, uncertainties, judgements, downstream changes

1. **The split does not create a density refuge on the epifauna.** It changes who has access, not the
   saturation; 223.6's other lever (an accessible epifauna pool that depletes on its own) is untouched.
2. **Derived growth moves.** Whiting's g rises to ~0.0041 /tick (gross 1.16 /yr: M2 0.42 from benthivores
   0.23 and the water-column group 0.16 on a small pool); the water-column group's falls to ~0.00175. The
   budget pays these at the Holling ceilings (160): the bed group reaches them on epifauna and planktivores,
   the water-column group on planktivores.
3. **Cannibalism and within-group transfers.** Whiting W 0.14 (unbooked) is probably overstated (SMS 0.094
   /yr vs 0.60 implied). Spurdog/hake/anglerfish on mackerel/horse mackerel stay within the water-column
   group (unbooked, 0.035 /yr).
4. **Saithe** carries the water-column group: F 0.11-0.38 (K 1.05-11), no post-2000 North Sea diet, survey
   leverage (Q3 visitor). Its Bf share 0.05 (0.02-0.15) sets most of the water-column benthivore pair.
5. **Benthivore predation on whiting** (0.23 /yr) rests on the long-run cod share.
6. **Mackerel** is undersampled by the GOV 5-10x (section 209); it would raise the water-column group's
   zooplankton and planktivore demand if counted.
7. **Attack-rate convention** (3.3): variants differ 2-4x; the layer membership threshold 0.1 is J
   (mackerel at score 0.18 sets half the water-column planktivore a). Saturated pairs (bed -> epifauna,
   bed -> planktivores) are insensitive; the zooplankton and whiting cross pairs are not.
8. **Visibility floors, interference, speeds, seal/porpoise splits and a** are judgements carried over.
9. **Residence:** visitors concentrated in the water-column group (~0.8 present); whiting resident.
10. **The epifauna's own budget changes slightly:** predators' M2 on it 0.032 instead of 0.036 (the
    water-column 0.0033 is left as unresolved predation), so its derived growth and equilibrium share move by
    ~0.4 %.

Downstream changes if implemented (not done):
- FG set: one more decision maker; every checkpoint invalid; `mareld2.yaml` decision_makers (start totals
  1605 t and 2715 t, ranges x0.5-2), `group_layers` (both at_50m), spawn templates (bed:
  FollowDemersalFish; water column: Followfish or FollowDemersalFish, J), seal spawn "on both demersal groups"
  (benthivores + whiting?) to be decided.
- `fg_library.yaml`: species definition (member_species, sources), ~2 x 20 pair keys per new group (the
  matrix grows by a row and a column), observes rows (bed: epifauna, benthivores, planktivores, zooplankton,
  water-column group; water column: planktivores, zooplankton, benthivores, bed group; seals, porpoises,
  planktivores and benthivores observe both), impact tables (bottom and pelagic trawling per group; rotor,
  windfarm noise copied), the planktivores' and zooplankton's M2 now split over two pairs (sums unchanged:
  0.497 and 0.129), benthivores' growth budget (M2 by the water-column group 0.084 and the bed group 0.098
  replace 0.18), the epifauna's (0.032 replaces 0.036).
- Code and tests: `fgconfig` sv_mapping, `lib/viz/i18n.py` names and viewer codes, `api.py` species names,
  `tools/probes/frozen_years.py` codes, `compare_scenarios.py`, `manual_play`, budget_gate; tests that pin
  numbers or the FG set: `tests/test_half_saturation.py` (the eight locked rates and light settings),
  `tests/test_pair_shares.py`, GPU spawn FG count, regional K list, growth-sync and hand-set lists,
  temperature fixtures, habitat start totals; AGENTS.md sections 4-5; a new resume section.
