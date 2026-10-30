# Splitting mackerel out of `predatory_fish` (resume 228): derivations

Written 2026-10-30. A new decision maker `mackerel` (mackerel + horse mackerel) out of the water-column
group of 223.26; `predatory_fish` keeps its id for the rest (saithe 0.80, spurdog, hake, anglerfish,
ling, velvet belly, pollack). Biology first: every value is a measurement, a mass balance of measured
rations and diets, or a stated judgement (J). No model output set a value. Tags as the other notes:
[F] full text / data read, [A] abstract, [S] secondary, [D] computed here, J judgement.

## 0. Decisions (user, 2026-10-30, interactive)

| Question | Choice |
|---|---|
| stage access of the group on the planktivores | annual mean over the quarters (Q3 alone ~0.12 declined) |
| regional K | none (as the water column since 223.37) |
| temperature layer | `mean_0_50m` (its prey's layer; annual 9.31 C) |
| fat cycle | reserve-funded now; the income-funded law (184) is the open next step |
| level | the field density of research_notes/"Makrillens täthet" (box annual mean mackerel 0.80, horse mackerel 0.05 t/km2), not the GOV share (GOV q for mackerel ~0.1) |
| rest water column -> mackerel pair | booked (at the field density its M2 is 0.021, 4.7 % of the group's gross; at the GOV level it was 0.131, 21 %) |

Taken as 223.26 (J, stated): the Bjornsson x1.30 anchor on both ceilings (the mackerel GER Cmax uses the
same cod stomach-fill calibration; summer fattening needs >= ~180 kJ/kg/d); seals and porpoises keep
today's per-tonne M2 on both groups; every DM that observes `predatory_fish` observes `mackerel`, and
`mackerel` observes what the water column observes; the stock F of each member (the closed-pool
convention of 209: visitors are charged stock-level r and F) - for F replaced by the local F for
mackerel in 228.1 (section 7).

## 1. Scripts (run from this folder unless stated; outputs in `tables/`)

| Script | What it does | Output |
|---|---|---|
| `split_encounter_mack.py "../Attackrater ur mötesteori" "../Fiskgrupper i rutan/predators" tables 1000 [densities]` | section 220 encounter a (layer form) for G1_wc (reproduction: 0.01171 / 0.000604 / 0.0532 / 0.01064, exact), the mackerel group and the rest; the new pair rest -> mackerel (encounter on pelagic fish prey over its eaters, prey-size window f); with densities the members' weights are the field ratio | `split_encounter_mack{,_field}_stdout.txt`, `.json`, `_rest_on_mackerel{,_field}.json` |
| `split_fields_mack.py "../Fiskgrupper i rutan/predators" tables [densities]` | biomass-weighted fields at each group's layer; reproduction gate on today's water column (all PASS); pair M2 conservation gate; field-density M2 | `split_fields_mack{,_field}_stdout.txt`, `.json` |
| `../Makrillens bytesstorlek/scripts/mackerel_stage_access.py ... [a_shares]` | [f_J, f_A] of the mackerel group and the rest on the planktivores' stages | `../Makrillens bytesstorlek/tables/mackerel_stage_access{,_field}.*` |
| `rest_on_mackerel_vs_density.py data/fg_library_mack.yaml [B ...]` | the cross pair's booked M2, unpaid share and return vs the rest's other pairs as a function of mackerel density | `rest_on_mackerel_vs_density{,_field}_stdout.txt` |
| from the repository root: `build_split_mack.py fgconfig/fg_library.yaml mareld2.yaml "research_notes/Rovfiskens uppdelning/tables" "research_notes/Rovfiskens uppdelning/data" _field` | text edits of the live library and manifest into `data/fg_library_mack.yaml`, `data/mareld2_mack.yaml` (comments kept); derived growth; verification | `data/build_split_mack.json` |

`[densities]` = `"mackerel=0.80,horse mackerel=0.05"`; `[a_shares]` = `"mackerel=0.9642,horse mackerel=0.0358"`
(the members' shares of the group's planktivore a at the field densities). The applied library is the
`_field` build; the data/ copies and `build_split_mack.json` are that build only (the GOV build, validated
in a worktree first, wrote the same names and is recorded in resume 228 as history). The data/ copies
equal the live files at build time only.

## 2. Reproduction gates

- Today's water column from the nine members at at_50m: r 0.3737 (lib 0.374), M1 0.2615 (0.261), F 0.2128 +
  0.0291, SMR 3.3096 (3.31), ceiling x1.30 32.713 (32.71), lean 4244 (4240), reserve 2598 (2600), fill 0.496
  (0.5), feeding cost 2.340 (2.34), starve 0.0078 (0.0078): PASS.
- Encounter a, G1_wc: exactly the library pair values.
- Stage access: the pooled pair reruns to [0.319, 0.109] (library [0.32, 0.11]).
- Pair M2 conservation at the GOV level: mackerel + rest = today's pair on planktivores 0.257, zooplankton
  0.0756, benthivores 0.0833, whiting 0.157: PASS. At the field level the mackerel group's own M2 rise by
  design (section 4).

## 3. Group fields

| Field | `mackerel` (field level) | `predatory_fish` (rest) | was (water column) |
|---|---|---|---|
| members (weights) | mackerel 0.941, horse mackerel 0.059 | saithe 0.797, spurdog 0.097, hake 0.048, anglerfish 0.027, ling 0.014, velvet belly 0.009, pollack 0.008 | nine members |
| B* t/km2 (t) | 0.85 (3060) | 0.617 (2220) | 0.754 (2715) |
| r_max | 0.255 (0.20-0.35) | 0.396 (0.34-0.43) | 0.374 |
| M1 /yr | 0.159 (0.11-0.21) | 0.276 (0.19-0.31) | 0.261 |
| F bottom / pelagic | 0.0123 / 0.0024 (mackerel's LOCAL F, 228.1) | 0.260 / 0 | 0.213 / 0.029 |
| layer | mean_0_50m (9.31 C) | at_50m (8.37 C) | at_50m |
| Cmax kJ/kg/d -> ceiling | 265 field-anchored -> 66.22 MJ/t/tick (228.2; was 159.8 x1.30 -> 51.94) | 91.7 x1.30 -> 29.79 | 32.71 |
| rest cost = resting_metabolism (Q10) | 4.59, the swimming floor (2.0) (228.2; was SMR 3.63) | 3.29 (2.3) | 3.31 (2.3) |
| feeding / movement cost x RM | 2.24 / 1.03 (228.2; was 2.72 / 2.0) | 2.27 / 2.0 | 2.34 / 2.0 |
| lean + reserve at fill (prey quality) | 4670 + 5870 at 0.55 (7899) | 4180 + 2000 at 0.45 (5080) | 4240 + 2600 at 0.5 (5540) |
| starve_rate | 0.0067 (routine; 228.2) | 0.0079 | 0.0078 |
| regional K | none | none | none |
| visibility floor, interference | 0.7, 0.1 (J, as the water column) | 0.7, 0.1 | same |

## 4. Pairs (per tonne of the WHOLE group; h = prey quality / ceiling)

| Pair | M2 /yr | a | h | f / stage access | night/day at -5 deg |
|---|---|---|---|---|---|
| mackerel -> planktivores | 0.0263 | 0.0435 | 107.6 | [0.078, 0.0037] | 0.36 |
| mackerel -> zooplankton | 0.1299 | 0.0023 | 68.0 | - | 0.42 (stomachs, 228.2) |
| rest -> planktivores | 0.247 | 0.00559 | 239.2 | [0.69, 0.27] | 0.30 |
| rest -> zooplankton | 0.0521 | 0.000241 | 151.1 | - | 0.51 |
| rest -> benthivores | 0.0833 | 0.0651 | 151.1 | [0.43, 0.027] (unchanged) | 0.48 |
| rest -> whiting | 0.157 | 0.013 | 161.1 | f 0.47 (unchanged) | 0.48 |
| rest -> mackerel (new, booked) | 0.021 | 0.00088 | 265.1 | f 0.29 | 0.92 |
| seals -> mackerel | 0.0098 | 0.05 | 37.2 | floor 0.75 | - |
| porpoises -> mackerel | 0.0063 | 0.2 | 28.6 | floor 0.9 | - |

- M2 sources: planktivores by the SMS partial M2 per member (223.37: saithe 0.209, hake 0.037; horse mackerel
  0.008, mackerel 0.002 at the GOV biomass, scaled to the members' field biomass: x8.3 and x1.2);
  zooplankton by the members' consumption Q/B x B x diet share at the pooled pair's calibration (k 1.009);
  rest -> mackerel = the eaters' diet flow 0.018 t/km2/yr (spurdog W 0.22 x 0.50, hake W 0.21 x 0.10,
  anglerfish, ling, velvet belly) / B.
- Mackerel and horse mackerel score below 0.1 on fish of the benthivore pool and on whiting: their layer a
  there is 0, so those pairs stay with the rest (M2 unchanged).
- Unchecked (as 223.26): rest and mackerel -> epifauna, infauna, Nephrops; cannibalism; mackerel -> the
  other fish groups; benthivores and whiting -> mackerel.
- The rest's effective access to herring is neutral: 0.00559 x 0.43 = 0.0024 vs today 0.0117 x 0.19 = 0.0022.

## 5. Budget consequences (derived, `data/build_split_mack.json`)

- growth_rate: mackerel 0.000894 (s* 0.845; best prey zooplankton), predatory_fish 0.0014966 -> 0.0016461
  (s* 0.786; r, M1 and the ceiling of the rest), planktivores 0.0031435 -> 0.0031889 (M2 0.624 -> 0.641),
  zooplankton 0.11854 -> 0.11883 (M2 1.730 -> 1.836), porpoises 0.00046279 -> 0.00045236 (best prey now
  mackerel, quality 7899). Every other group unchanged.
- budget_gate: every pair and predator ok ("any diet"); mackerel ration 4.6 %/day.

## 6. The cross pair at the field density (`tables/rest_on_mackerel_vs_density_field_stdout.txt`)

At 0.85 t/km2 the booked M2 is 0.021 (4.7 % of the group's gross 0.451 /yr); the rest's return per unit eat
share on mackerel is 1.6 MJ/t/tick against herring 19.5, benthivores 16.9, whiting 9.1, zooplankton 7.8 at
grid-mean prey. A cell needs ~5.2 t/km2 of mackerel to match the rest's zooplankton pair, ~28 to match
herring: the pair is booked but will rarely be eaten; what is left unpaid is small at this density.

## 6b. Physiology corrections (228.2, user)

The first build read the species_values.csv rows for mackerel and horse mackerel as at_50m (8.37 C)
values. Evidence A says otherwise, and two of them are not GER values at all:
1. **Rest cost.** Mackerel has no swim bladder and cannot rest. The rest cost is its routine at 1 BL/s,
   6.07 MJ/t/6h at 400 g and 13 C (Bachiller et al. 2018, Q10 1.98), i.e. 4.72 at 9.31 C. Its true SMR
   (3.37 at 13 C, chub mackerel at U = 0) is never reached. Horse mackerel has a swim bladder: SMR 3.4 at
   13 C -> 2.54. Group 4.59 (was 3.63 = the 13 C SMR read as at 8.37 C and scaled up). Routine for
   feeding: mackerel = its floor (1.02 x RM for the group); moving one cell per tick (~0.05 m/s) costs
   nothing above the floor (J): movement 1.03 x RM (horse mackerel keeps 2 x SMR). starve_rate uses the
   routine: 0.0067.
2. **Night/day on zooplankton.** The encounter model has visual particulate feeding only (0.11).
   Mackerel ram-filters copepods down to ~1e-7 uE m-2 s-1 (Macy et al. 1998 [F]); its night stomachs hold
   0.4 (0.3-0.5) of the day's (Dahl & Kirkegaard 1986 [F]); horse mackerel 0.8. Weighted 0.42. The fish
   pair keeps the encounter value (0.36; visual hunting, the 223.26 convention; stomachs give 0.15).
3. **Cmax.** The 150 kJ/kg/d in species_values.csv was a judgement within 120-300, below the
   fattening budget (>= 180) it was meant to satisfy, and the build then applied the layer temperature
   (x1.08) and the Bjornsson GER re-anchoring (x1.30) on top. Now field-anchored: the Norwegian Sea
   feeding-season ration 0.08 g/g/d (~290 kJ/kg/d at ~10 C, J; Bachiller et al. 2018 [F]) is the
   lowest ceiling consistent with it; horse mackerel 152 at 13 C (Cape horse mackerel ration [A]); to
   9.31 C with the mackerel GER term 0.080/C; no Bjornsson factor. Cmax 265, ceiling 66.22 (was 51.94).
   The GER construction (109-118 at 12-13 C, x1.30 = 142-153) fails both field checks for mackerel.

Consequences: mackerel growth_rate 0.000894 -> 0.000825 (s* 0.874, best pair now herring at the
ceiling); h on its pairs 107.6 / 68.0; budget_gate ration 5.9 %/day (Bachiller's mean 8 %). The
upstream species_values.csv rows are left as they are (the 209 pooled-group history); this script
carries the corrected values.

## 7. Open

- **F (decided, user 228.1).** Mackerel carries its LOCAL 3.a.20 F: 343 t/yr / 25 945 t = 0.0132,
  of which the midwater-trawl share 0.025 on the pelagic map, the rest (gillnet, hooks, bottom trawl) on
  the bottom map; horse mackerel keeps 0.003 + 0.035 (stock = local, ~0.03 in 2020-24). Group F 0.0123 +
  0.0024 = 0.0147 (was 0.205 with the stock F, which would have landed ~620 t/yr from the box alone
  against 555 t/yr for all of 3.a). An exception to the 209 closed-pool convention: with no K the group
  now nets r - F ~0.24 /yr before predation and M1.
- Seasonal presence (presence fraction 0.45; Dec-Apr 0.05-0.15 of July) has no mechanism; the group is
  resident at the annual mean. The inference start is the annual mean (3060 t), not a winter state.
- The income-funded fat cycle (184) and the summer layer temperature.
- Mackerel's own planktivore and zooplankton M2 are diet bookings; their realisation needs a trained run.
- FG editor round trip not done (as 223.26). All checkpoints are invalid (one more DM).
