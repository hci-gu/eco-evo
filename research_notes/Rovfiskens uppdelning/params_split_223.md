# Splitting `predatory_fish` on the 223.15 library: derivations

Written 2026-10-27 (resume 223.25 context). This re-derives `params_split.md` (2026-10-26, section-222
library), which is left untouched. Biology first: every value is a measurement, a mass balance of measured
rations and diets, or a stated judgement (J). No model output was used to set a value; section 6 is a
labelled diagnostic. Nothing in the library, manifest, code, tests, AGENTS.md or the resume was edited.
Tags: [F] full text / data read, [A] abstract, [S] secondary, [D] computed here, J judgement. No new
literature was searched: every fact comes from evidence already in `research_notes/`.

Scripts (`scripts/`, run from this folder with `CUDA_VISIBLE_DEVICES="" python3 -I`; outputs in `tables/`):

| Script | What it does | Output |
|---|---|---|
| `split_fields_223.py "../Fiskgrupper i rutan/predators" tables` | species fields at each group's temperature layer, diets over the 223.15 prey groups, M2 per pair; reproduction gate on the current pooled pairs; M2 scaled so the groups add back to the library exactly | `split_fields_223_stdout.txt`, `split_fields_223_G1.json`, `_G2.json` |
| `split_encounter_223.py "../Attackrater ur mötesteori" "../Fiskgrupper i rutan/predators" tables 2000` | section 220 encounter a per group, four averaging conventions, MC 5-95 %, night/day; the epifauna pair x the 223.15 exposure e | `split_encounter_223_stdout.txt`, `split_encounter_223.json` |
| `split_pairs_diagnostic_223.py tables "../Fiskgrupper i rutan" "../Attackrater ur mötesteori"` | closed-form check, benthivore-pair conventions, predation on each group, derived growth, pair tables, diagnostic, benthivore access, M2 conservation, each group's own 2001-25 span | `split_pairs_diagnostic_223_stdout.txt` |

Reproduction gates (all pass):
- Fields, the ten members pooled: B 1.20, r 0.417, F 0.1616 + 0.0201, K 2.13, M1 0.257, ceiling 25.47, SMR
  3.64, Q/B 3.34, lean 4116, ME 2376, fill 0.48 (library 1.20 / 0.42 / 0.162 + 0.020 / 2.12 / 0.257 / 25.5 /
  3.64 / 3.34 / 4100 / 2400 / 0.48).
- Pair M2 from the member diets: planktivores 0.4965, zooplankton 0.1289, exposed epifauna 0.0745 (on B*
  3.78), benthivores 0.1823 vs library 0.50 / 0.13 / 0.076 / 0.18 (ratios 0.98-1.01). The split M2 are
  scaled by these ratios so bed + water column = library exactly (J; it keeps every prey's growth budget).
- Encounter a, pooled: capable 0.0119 planktivores (library 0.012), 0.0015 zooplankton (0.002 incl. the
  Staniland capture factor), 0.178 benthivores (0.18); noscore x e on epifauna 0.127 (library 0.13).
- Closed-form s*, g on the pooled group: g 0.00244824 vs stored 0.00244822.

---

## 1. Grouping: G1 still holds on the new prey structure

What changed since the 222 version and whether it moves the argument:

1. **E1 (223.15): mobile_epifauna is the exposed taxa only; Nephrops is `norway_lobster`.** Whiting's bed
   invertebrates are northern shrimp, small shrimps, mysids and small crabs (ICES stomach DB, 221 note):
   all exposed taxa. `Havskräfta i fiskdieter` (`species_nephrops_summary.csv`): whiting <= 0.55 % of its
   diet in the North Sea 1991 project (Q3-Q4 only), 0 in every Skagerrak, box and Kattegat sample (1991,
   2020-22), 0.2 % in the W Irish Sea, 0.05 % occurrence in DAPSTOM IVa (one Silver Pit outlier, 1979);
   saithe and hake none recorded. Spurdog, anglerfish and ling are occasional Nephrops predators elsewhere
   (J), but carry 6 % of the pooled biomass and benthos shares of 0.02-0.06: < 0.002 t/km2/yr. **No member
   eats Nephrops in quantity: both groups -> norway_lobster unchecked**, as the pooled pair is today.
   E1 removes the 60 % of the old pool that whiting never took, so the bed pair now lands on whiting's prey.
2. **Infauna refuge (223.12-223.13).** The predators have no infauna pair (3.9 % of the pooled diet; 7.3 %
   of whiting's; M2 0.0015): unchecked in both groups, as 221. The refuge does not touch the grouping.
3. **Benthivore diet (223.10).** The benthivores' predator share 0.023 (M2 0.128) is cod and grey gurnard
   on fish of the predator pool; cod's identified fish prey in that pool is whiting (Skagerrak 3.7 of 3.7 g
   per 100 g fish prey, box 10.3 of 10.3, Kattegat 10.1 of 11.3, NNS 10.8 of 11.6;
   `Fiskgrupper i rutan/benthivores/tables/cod_fish_prey_pools.csv` [F]). This puts the whole cross pair on
   whiting in either grouping; it does not separate G1 from G2.
4. **Exposure e (223.15)** was derived on a whiting-dominated composition (predator A: Pandalus 0.40,
   small shrimps 0.18, mysids 0.13, crabs/hermits/Munida 0.29; whiting's crab window S 0.03, Pandalus S
   0.4): it is whiting's e. The note itself flags that "the bed-access gap of non-bed members (~2.7x) is NOT
   in e": the split is what carries that gap.

**Recommendation unchanged: G1.** Bed group = whiting (0.446 t/km2, 1605 t); water-column group = saithe,
mackerel, spurdog, horse mackerel, hake, anglerfish, ling, velvet belly, pollack (0.754 t/km2, 2715 t).
Whiting carries 91 % of the pooled epifauna flow (0.256 of 0.282 t/km2/yr) and 89 % of the pooled
epifauna encounter a; saithe, hake and anglerfish sit near the bed by day but eat fish and krill there
(benthos 0.3-1 %). G2 (habitat: + hake, spurdog, anglerfish, ling, velvet belly) is kept as a variant in
every table; it gives bed access to 27 % more biomass that does not use it, and turns their whiting
predation into an unbooked within-group transfer (wc -> bed M2 0.057 instead of 0.157).

Names (J): the bed group needs a display name whose viewer code does not collide with the benthivores'
`bf` (codes come from the display name, `pygame_viz._unique_abbrevs`; test_viz_abbrev pins `bf` and
`prf`). Suggested: id `bed_predatory_fish` (or `whiting`), display "Whiting" (sv "Vitling", it
"Merlano"); keep id `predatory_fish` / "Predatory Fish" for the water-column group (code `prf` kept; its
meaning changes, which every reader of old results must know).

---

## 2. Group fields (G1)

Construction as section 209 (`weighted_fields.py`): biomass-weighted species values from
`predators/inputs/species_values.csv` with the long-run (2001-25) GOV shares, corner ranges. B* = 1.20
t/km2 x the members' share (top of the long-run span, user's choice). Each group's own span: whiting mean
0.378 (0.13-0.79; 2021-25 0.54) -> B* 0.446 = x1.18; the rest mean 0.642 (0.14-2.08; 2021-25 1.15) -> 0.754
= x1.17 (same x1.18 as the pooled group).

### 2.1 Table (changes vs the 222 version in bold)

| Field | bed (whiting) | water column | Basis |
|---|---|---|---|
| member shares | whiting 1.00 | saithe 0.651, mackerel 0.128, spurdog 0.080, horse mackerel 0.055, hake 0.040, anglerfish 0.022, ling 0.011, velvet belly 0.008, pollack 0.007 | `pred_season.csv` pooled Q1+Q3 |
| B* (t/km2); start total (range x0.5-2) | 0.446; 1605 t (800-3210) | 0.754; 2715 t (1360-5430) | 1.20 x 0.3715 / 0.6285 |
| `r_max` (/yr) | 0.49 (0.44-0.49) | 0.374 (0.32-0.42) | Euler-Lotka r_now per species (209) |
| `natural_mortality` M1 | 0.25 /yr = 1.7123e-4 /tick (0.20-0.32) | 0.261 /yr = 1.7909e-4 /tick (0.18-0.30) | whiting SMS residual 0.20 + grey seal/birds 0.05; members' assessment M (209) |
| F bottom / pelagic (/yr) | 0.075 / 0.005 | 0.213 / 0.029 | catch / biomass per species (209); sums to today's tables (0.162 / 0.020) |
| `bottom_trawling` biomass_factor at 100 | 0.001704 (0.00075-0.0032) | 0.004836 (0.0024-0.0059) | F_bot / 1460 / 3.0142 x 100 |
| `pelagic_trawling` biomass_factor at 10 | 0.001372 | 0.007983 | F_pel / 1460 / 0.02496 x 10 |
| `regional_carrying_capacity` K = B*/(1 - F/r) | 0.533 (0.48-0.68); F/r 0.16 | 2.14 (1.05-11.0); F/r 0.65, saithe F 0.11-0.38 drives it | 185 convention, current trawl tables |
| temperature layer (`group_layers`) | **`bottom`** (7.40 C annual; J) | at_50m (8.37 C; J) | 2.2 |
| Cmax (kJ/kg/d) -> energy ceiling (MJ/t/tick) | **96.4 (74-139) -> 24.11 (18.6-34.8)**; Bjornsson-anchored 125.4 -> 31.3 | 100.7 (77-156) -> 25.16 (19.3-38.9); Bjornsson-anchored 130.9 -> 32.7 | 2.3 |
| `resting_metabolism` (MJ/t/tick) | **3.88 (3.0-4.6)** = 4.2 re-read at 7.40 C; 4.2 if not re-read (as the benthivores' 3.6 in 223.15) | 3.31 (2.8-5.0) | SMR at the layer's annual mean, Q10 2.3 |
| `metabolism_q10` / `metabolism_t_ref` | 2.3 / annual_mean | 2.3 (2.27) / annual_mean | cod 2.2-2.6, GER 2.25, mackerel 2.0, spurdog 2.5 |
| `feeding_cost` / `movement_cost` / `movement_speed` | 2.0 (2.03) / 2.0 / 1.0 | 2.15 / 2.0 / 1.0 | routine 1.5 x SMR + SDA 0.17 x half the ceiling / SMR; cruising 2 x SMR |
| `energy_content` (lean) / `max_energy_reserve` | 3900 / 2000 | 4240 / 2600 | lean + reserve convention (140; the benthivores joined it in 223.10) |
| `prey_includes_reserve` / `reserve_reference_fill` | true / 0.45 -> static prey quality **4800** | true / 0.50 -> **5540** (5532) | weighted mean reserve / max; 0.3715 x 4800 + 0.6285 x 5540 = 5265 vs pooled 5252 |
| `maintenance_level` u | 0.5 (engine convention) | 0.5 | 187.4 |
| `starve_rate` (/tick) | **0.0099** at RM 3.88 (0.011 at 4.2) | 0.0078 | 1.5 x SMR / lean / 0.15 (209 construction) |
| `interference` w | 0.1 | 0.1 | Rindorf et al. 2006: predator density not significant |
| `visibility_floor` (as prey) | 0.5 (0.4-0.6) J | 0.7 (0.6-0.8) J | pooled 0.6 straddled "whiting hides on the bed, midwater schools badly" |
| `income_growth_kappa` | not set | not set | as 209 |
| Q/B; Q (t/km2/yr) | 4.2 (3.3-5.6); 1.87 | 2.83 (2.6-3.6); 2.13 | SMS 2023, Lauerburg 2018; evidence C |
| derived `growth_rate` (diagnostic; the loader computes it) | **~0.00492** (s* 0.677; gross 1.27 /yr) | ~0.00175 (s* 0.755; gross 0.65 /yr) | section 160 closed form with the booked M2 of 4.3 (pooled 0.00245) |

### 2.2 Temperature layer (J)

The bed group's defining activity is foraging on the bed; whiting sit within ~5-30 m of the bed by day
(Pedersen 1999 [A]) and are resident all year (FishMap [F]); 24 % of the grid is shallower than 100 m,
where the `bottom` layer is 7.8 C. 223.15 moved the benthivores to `bottom` on the same argument. Recommend
`bottom` (J). The seasonal shape differs more than the mean: at_50m peaks at 11.7 C in October, the bottom
at 8.5 C in December, so a whiting on at_50m pays ~20 % more maintenance in Sep-Nov. The water-column group
stays at_50m (J): saithe near the bed below 300 m by day (~7 C) and mackerel in the summer mixed layer (~13 C)
average to it; neither extreme is represented.

### 2.3 Ceiling: temperature and the cod anchor (open judgement, both numbers given)

The predator Cmax (evidence A, `calc_ger_cmax.py`) is a gastric-evacuation steady state with the stomach
fill calibrated so that a 500 g cod at 7 C reproduces the FB4 cod Cmax (10.7 g/d). Two 223.15 changes bear
on it:
- **Temperature.** The GER temperature term is e^0.078 T (Andersen's whiting form, used for hake and saithe
  too). At the bottom layer's 7.40 C instead of 8.37 C: x0.927, whiting 104 -> 96.4 kJ/kg/d, ceiling
  26.0 -> **24.11 MJ/t/tick** [D]. The water-column group is unchanged at 25.16.
- **Anchor.** 223.15 replaced FB4 by Bjornsson et al. 2001 ad-lib intake for cod (prey-normalised: x1.31 at
  447 g, x1.29 at 874 g, 7 C; `Exponering av bottenbyten/tables/C_cmax_bottom_stdout.txt`). Since every
  predator Cmax was calibrated on the FB4 cod point, the same re-anchoring multiplies all of them by ~1.30:
  bed 31.3, water column 32.7, and the pooled library ceiling 25.5 -> ~33. The calibrated fill would rise
  from 2.5 to ~4.2 % BW, inside the 5 % upper bound evidence A gave. Not applied centrally (J): the
  predators' ceiling is a GER construction, not a cod Cmax, and no whiting or saithe ad-lib series exists;
  but consistency with 223.15 argues for it. Effect: h on every predator pair /1.30, derived g bed
  0.00492 -> 0.00350, water column 0.00175 -> 0.00144, demand at the ceiling x1.30 (section 6).

### 2.4 Resting metabolism (J)

`metabolism_t_ref: annual_mean` makes the loader read `resting_metabolism` as the cost at the layer's
annual mean. Whiting's 4.2 is the SMR attributed to 8.4 C (Steinhausen et al. 2005 via evidence A; the
source temperature was not recovered). On `bottom` the consistent value is 4.2 x 2.3^(-0.097) = **3.88**.
223.15 left the benthivores' 3.6 un-re-read (3.32 if re-read; "not decided"): the user should decide both
together. With 4.2 kept: cost/C 0.44, g 0.0054.

---

## 3. Diets and pairs (G1)

### 3.1 Diet over the current prey groups

`split_fields_223.py`: member diets from `diet_consumption_central.csv` (proxies for ling, velvet belly,
pollack J), consumption-weighted by Q/B x B; benthos split into exposed epifauna / infauna by the member's
epifauna share (whiting 0.65, hake 0.99, horse mackerel 1.0, saithe 1.0, spurdog 0, mackerel 0.5,
anglerfish/ling 0.5 J), Nephrops share 0 (section 1); own-pool fish split by prey species as in the 222
version (J).

| Diet share of total food | planktivores | zooplankton | exposed epifauna | Nephrops | infauna | benthivores | own group | other group | other |
|---|---|---|---|---|---|---|---|---|---|
| bed (whiting) | 0.280 | 0.280 | 0.137 | 0 | 0.073 | 0.080 | 0.142 | 0.007 | 0 |
| water column | 0.523 | 0.342 | 0.012 | 0 | 0.005 | 0.061 | 0.012 | 0.033 | 0.013 |
| pooled (check) | 0.409 | 0.313 | 0.070 | 0 | 0.037 | 0.070 | 0.094 | - | 0.007 |

Unchanged from the 222 version except that the epifauna share now sits on a 3.78 t/km2 pool (it was 7.88
incl. Nephrops): the pooled flow 0.282 t/km2/yr is the same, so its M2 doubles (0.036 -> 0.076, as 223.15
already booked).

### 3.2 Pairs

M2 = Q x diet share / prey B, scaled to the library sums (3.1 gate). h = E_prey as eaten / the group's
ceiling (section 148). a = section 220 encounter rate, **per tonne of the whole group** (section 5),
"layer" variant: every member that forages in the prey's layer and can take it at all (capability score
>= 0.1) at its own rate; gape on the prey side in `accessible_fraction`, bed exposure in e, the time
budget in the eat share. Night/day = the encounter model's a_night / a_day (220.3 decision 5); thresholds
-5 deg kept on predator pairs, 0 on benthivore pairs (220). Every fish pair Type II (Rindorf et al. 2006).
Assimilation 0.8. "x -> share" = a h f B at today's prey with all visible (low visible in brackets);
diagnostic only.

| Pair | M2 (/yr) | h | a central (MC 5-95 %) | f | dark_ratio / thr | x -> share of ceiling | checked |
|---|---|---|---|---|---|---|---|
| bed -> planktivorous_fish | **0.160** | **295.5** | 0.021 (0.0069-0.048) | 1 | 0.33 / -5 | 20.6 -> 0.95 (0.88) | yes; pair visibility_floor 0.25 kept |
| bed -> zooplankton | 0.054 | **186.6** | 0.0018 (0.0008-0.0066); 0.0023 with the library's Staniland capture factor (x1.33, J) | 1 | 0.78 / -5 | 3.2 -> 0.76 (0.14) | yes |
| bed -> mobile_epifauna (exposed) | **0.069** | **155.5** | **0.27** (0.095-1.21) = a_enc 0.69 per edible tonne x e 0.39 | **0.29** (0.2-0.45; 223.15) | none | 46 -> 0.98 | yes |
| bed -> norway_lobster | 0 | - | - | - | - | - | no (traces) |
| bed -> benthic_infauna | 0.0015 | - | - | - | - | - | no (221 rule) |
| bed -> benthivorous_fish | 0.097 | **186.6** | 0.21 (0.064-0.81) | 0.05 (0.03-0.15; whiting gape ~12 cm + 0-group, J) | 0.64 / -5 | 3.0 -> 0.75 (0.62) | yes |
| bed -> water column | 0.019 | - | - | - | - | - | no (traces of hake / horse mackerel) |
| wc -> planktivorous_fish | **0.340** | 283.1 | 0.0117 (0.0044-0.033) | 1 | 0.39 / -5 | 10.9 -> 0.92 (0.79) | yes; floor 0.25 kept |
| wc -> zooplankton | 0.076 | 178.8 | 0.0006 (0.0004-0.0021) | 1 | 0.30 / -5 | 1.05 -> 0.51 (0.05) | yes |
| wc -> benthivorous_fish | 0.083 | 178.8 | 0.053 (0.018-0.17) | 0.24 (gape: saithe 19, spurdog 27, anglerfish 45 cm + 0-group) | 0.48 / -5 | 3.5 -> 0.78 (0.66) | yes |
| wc -> bed (new) | 0.157 (0.07-0.28) | **190.8** (4800/25.16) | 0.0106 (0.0041-0.035) | 0.47 (whiting below the members' gape) | 0.48 / -5 | 0.43 -> 0.30 (0.18) | yes: learnable cross-group pair |
| wc -> mobile_epifauna | 0.007 | (149) | (0.019; horse mackerel 82 %) | - | - | - | **no: the point of the split** |
| wc -> infauna / Nephrops / wc | 0.0001 / 0 / (0.035) | - | - | - | - | - | no |
| benthivorous_fish -> bed | **0.345** (0.307-0.345) | **183.3** (4800/26.18) | **0.0026** per whole benthivore tonne (library-convention 0.026 per capable tonne; layer 0.0086) | 1 | 0.10 / 0 | 0.21 -> 0.17 | yes |
| benthivorous_fish -> water column | 0 | - | - | - | - | - | no (cod's W is whiting) |
| seals -> bed | 0.0130 (0.009-0.018) | 22.6 | 0.05 (J) | 1 | none | - | yes; visibility_floor 0.85 (J, on the bed) |
| seals -> water column | 0.0098 (0.007-0.012) | 26.0 | 0.05 (J) | 1 | none | - | yes; floor 0.75 (kept) |
| porpoises -> bed | 0.0190 (0.015-0.030) | 28.6 | 0.20 (J) | 1 | none | - | yes; floor 0.75 (J) |
| porpoises -> water column | 0.0063 (0-0.008) | 28.6 | 0.20 (J) | 1 | none | - | yes; floor 0.90 (kept) |
| within each group | (bed 0.60, wc 0.035) | - | - | - | - | - | no (section 160) |

Ranges of M2: the group's corner Q/B x the dominant member's diet-share range, as the 222 version (bed ->
planktivores 0.07-0.57, -> zooplankton 0.003-0.21, -> epifauna 0.015-0.17 (the 222 range on the 3.78 pool), -> benthivores 0.04-0.34; wc ->
planktivores 0.19-0.55, -> zooplankton 0.03-0.17, -> benthivores 0.05-0.23).

Notes:
- **Benthivores -> bed (0.345 /yr).** 223.10's pair (Q/B 4.46 x 1.53 x 0.023 = 0.154 t/km2/yr) lands
  wholly on whiting (section 1.3; 0.89-1.0 by area). On 0.446 t/km2 that is 0.345 /yr (222 version 0.23 at
  Q/B 3.48). Whiting's booked M1 + M2 = 0.25 + 0.345 + 0.157 + 0.013 + 0.019 = **0.78 /yr**, against SMS
  2023 whiting 1+ total M 0.685 [F] in which cod takes only 0.05. Open judgement 1.
- **The benthivore pair's a.** The library's 0.026 is per tonne inside phi 0.10 (220: the large cod,
  starry ray and gurnard that can take whiting; the score is gape only, vertical overlap is in a). With no
  prey-side f to carry that gape, the whole-group value is a x phi = 0.0026 (section 5). At that a the
  pair is unsaturated (x 0.21 at 0.446 t/km2) and the benthivores would need ~15 % of their eat share on
  whiting to realise the booked 0.154 t/km2/yr; with the library's 0.026 it is saturated (x 2.1).
- **Planktivores -> predators** (unchecked, dormant M2 0.0799) and **seabirds -> predators** (muted,
  0.0078): split by biomass (bed 0.37 / wc 0.63) if ever checked (J); no change today.
- **Whiting cannibalism** 0.142 of the diet (two thin post-2000 samples) is unbooked; if it is overstated,
  every other bed pair's diet share rises by up to 1/(1 - 0.142) = 1.17.

### 3.3 Attack-rate variants (central; structural uncertainty)

| Pair | layer (recommended) | noscore | scored | capable (220) |
|---|---|---|---|---|
| bed -> planktivores | 0.0211 | 0.0403 | 0.0092 | 0.0296 |
| bed -> zooplankton | 0.0018 | 0.0019 | 0.0011 | 0.0032 |
| bed -> exposed epifauna (x e) | 0.268 | 0.268 | 0.081 | 0.347 |
| bed -> benthivores | 0.209 | 0.536 | 0.110 | 0.315 |
| wc -> planktivores | 0.0117 | 0.0117 | 0.0046 | 0.0070 |
| wc -> zooplankton | 0.00060 | 0.00061 | 0.00048 | 0.00087 |
| wc -> benthivores | 0.053 | 0.147 | 0.017 | 0.066 |
| wc -> whiting | 0.0106 | 0.029 | 0.0033 | 0.0132 |

Who sets each a (layer): bed pairs are all whiting (planktivores and benthivores adults only); wc ->
planktivores mackerel 0.49, saithe 0.33, horse mackerel 0.12; wc -> zooplankton mackerel 0.49, saithe 0.28;
wc -> benthivores and -> whiting saithe 0.77.

---

## 4. Cross pairs and the other predators of each group

| Predator of | bed (whiting) | water column |
|---|---|---|
| water-column group | 0.157 (new booked pair; was an unbooked within-group transfer); saithe 0.066, hake 0.044, spurdog 0.030, anglerfish + ling 0.017; check (not used): SMS 2023 saithe on whiting 1+ 0.085 | (within group 0.035, unbooked) |
| bed group | (cannibalism, unbooked) | 0.019 unchecked (traces) |
| benthivores | 0.345 (all of 223.10's 0.128 x 1.20) | 0 (unchecked) |
| seals | 0.0130 (whiting 0.44 of the seals' predator share, 0.3-0.6, J) | 0.0098 |
| porpoises | 0.0190 (whiting 0.64, 0.52-1.0, J) | 0.0063 |
| M1 | 0.25 | 0.261 |
| booked total | 0.784 | 0.277 |

M2 conservation on every prey (`split_pairs_diagnostic_223_stdout.txt` section 7):

| Prey | library (pooled) | bed + wc | difference and why |
|---|---|---|---|
| planktivorous_fish | 0.50 | 0.160 + 0.340 = 0.500 | 0 (scaled) |
| zooplankton | 0.13 | 0.054 + 0.076 = 0.130 | 0 |
| mobile_epifauna | 0.076 | 0.069 + 0 = 0.069 | -0.007: wc -> epifauna is unchecked (the split's purpose); its flow (0.026 t/km2/yr: horse mackerel, mackerel, saithe traces) becomes unresolved predation. The epifauna's derived g falls 0.0012027 -> 0.0011979 (-0.4 %), its equilibrium share 0.543 -> 0.545, so the K table should be refitted (x0.996) to keep the 13 600 t start |
| benthivorous_fish | 0.18 | 0.097 + 0.083 = 0.180 | 0 |
| norway_lobster, infauna | 0 (unchecked) | 0 | 0 |
| predators as prey (t/km2/yr) | seals 0.0132, porpoises 0.0132, benthivores 0.154 | seals 0.0058 + 0.0074, porpoises 0.0084 + 0.0048, benthivores 0.154 + 0 | 0; plus the NEW booked wc -> bed flow 0.070 (and a new booked M2 in the bed group's growth budget) |

So the planktivores', zooplankton's and benthivores' derived growth rates are unchanged; only the
epifauna moves (-0.4 %) and the two new groups get their own g.

---

## 5. Attack-rate averaging convention (old open judgement 2)

**What the engine implies.** `predation.apply_predation`: demand = B_pred x eat x hunger x a (f B) /
(1 + a h f B + w B_pred), with `eat` the policy's share of the group's biomass in the cell allocated to
that prey. There is no phi any more (221-222) and the engine does not know which members are allocated: an
allocated tonne is a cross-section of the group. So **a is the clearance per tonne of the whole group's
biomass that forages on that prey**, the biomass-weighted mean over all members (members that cannot
forage there contribute 0). The 220 "capable" convention (sum W s a / sum W s, per tonne inside phi) was
right only while min(eat, phi) capped the allocation; since 221 it over-credits by 1/phi at every eat share
above phi.

**Which per-whole-group form.** Count each factor once:
- Predator pairs: the 209 capability scores are "access x capability" (A x C). C (gape, stage) is carried
  by `accessible_fraction` on fish prey and by e on bed prey; A (time in the prey's layer) is the eat
  share. Folding the score in ("scored") counts both twice; "noscore" lets members that never reach the
  layer (mackerel on the bed, score 0.01) count fully. **Layer** (members with score >= 0.1 at their own
  rate, per tonne of the whole group) is the practical form (J: the 0.1 threshold; mackerel at 0.18 sets
  half the wc planktivore a). For bed prey this equals 223.15's "noscore x e" for whiting.
  Caveat: layer is cleanly consistent only where f or e carries the gape (bed -> epifauna, both ->
  benthivores, wc -> whiting). On the f = 1 pairs (both -> planktivores, both -> zooplankton) partial gape
  scores are dropped (adult whiting 0.40 on planktivores counts as 1), so layer is the upper form and
  scored the lower (factor 2-2.5). It matters little on the planktivore pairs (x at today's B: layer 20.6 /
  10.9, scored 9.0 / 4.3; saturated either way) and most on the zooplankton pairs (bed x 3.2 -> 2.1, wc
  1.05 -> 0.83): the zooplankton a is where the choice is live.
- Benthivore fish pairs: the 220 benthivore scores are gape only (predator-side: which cod can swallow the
  prey), vertical overlap is in a, and no f carries the gape, so the score folds in: **scored** = capable
  x phi.

**Current library pairs this would change** (pooled group, if the split is not applied; central values):

| Pair | library (convention) | per whole group | factor |
|---|---|---|---|
| predatory_fish -> planktivorous_fish | 0.012 (capable) | 0.0152 (layer) | x1.27 |
| predatory_fish -> zooplankton | 0.002 (capable, incl. Staniland x1.33) | 0.0010 (layer; 0.0014 with Staniland) | x0.52-0.69 |
| predatory_fish -> benthivorous_fish | 0.18 (capable) | 0.111 (layer) | x0.62 |
| predatory_fish -> mobile_epifauna | 0.13 (noscore x e) | 0.112 (layer x e) | x0.86 |
| benthivorous_fish -> planktivorous_fish | 0.019 (capable, phi 0.32) | 0.0062 (scored) | x0.32 |
| benthivorous_fish -> predatory_fish | 0.026 (capable, phi 0.10) | 0.0026 (scored) | x0.10 |

Unchanged: benthivores -> infauna / epifauna / Nephrops (a_enc per whole group x e, f), planktivores ->
zooplankton (per foraging fish with the foraging overlap; herring 94 %, overlap 1.0: effectively the
whole group), seals and porpoises (judgements per tonne of prey group). The split makes the conventions
converge where it matters: for whiting alone layer = noscore on every pair.

---

## 6. Diagnostic (labelled; not used to set anything)

`split_pairs_diagnostic_223_stdout.txt`. Engine identities: intake per tonne at eat 1, hunger 1 =
a fB / (1 + a h fB); ceiling 1/h; demand at the ceiling = B_pred x 1460 / h.

**Shares of the ceiling at today's densities (all visible / low visible).** Bed: epifauna 0.98 / 0.98,
planktivores 0.95 / 0.88, zooplankton 0.76 / 0.14, benthivores 0.75 / 0.62. Water column: planktivores
0.92 / 0.79, benthivores 0.78 / 0.66, zooplankton 0.51 / 0.05, whiting 0.30 / 0.18.

**Ceiling demand vs the prey's surplus.**
- Bed -> exposed epifauna: still saturated (x 46; half-saturation 0.083 t/km2 of the pool vs 3.78). E1
  surplus at B* 3.03 t/km2/yr = benthivores 2.04 + trawl 0.70 + predators' budget 0.287 (the room; 0.53 at
  K/2). Bed demand at the ceiling with the whole eat share 4.18 t/km2/yr (pooled group 11.9), at the diet
  share 0.256; eat share 0.137 / 0.3 / 0.6 at the ceiling -> 0.57 / 1.26 / 2.51 = 2.0 / 4.4 / 8.7 x the room.
  A fed whiting group (hunger 0.65 at s* 0.677) takes at most 2.7.
- Wc -> planktivores: demand at the ceiling 3.89, at eat share 1 today 3.56, at the diet share 1.11; the
  pooled predators' budget 1.65, all checked predation 2.52, gross production at B* 3.76 t/km2/yr.
- Both -> benthivores: ceiling demand bed 3.49 + wc 6.16 t/km2/yr vs the benthivores' gross production
  ceiling (r + M1 + booked M2) 0.82 /yr x 1.53 = 1.25 t/km2/yr.

**Budget gate (cost / C vs the best pair).** Bed 0.41 (0.44 with RM 4.2; 0.31 Bjornsson) vs epifauna 0.98,
planktivores 0.95. Water column 0.35 vs planktivores 0.92, benthivores 0.78: the water-column group can pay
its way without any bed pair.

**Access to the benthivores (B 1.53), sum over predator groups.**

| Variant | sum a f B_pred (km2/tick) | linear-limit M2 per unit eat (/yr) | M2 per unit eat share, all / 55 % visible (/yr) | demand at eat 1 (t/km2/yr) |
|---|---|---|---|---|
| pooled, library as is (what fix77 ran) | 0.0324 | 47 | 5.7 / 5.2 | 8.7 |
| pooled, layer convention | 0.0200 | 29 | 5.3 / 4.6 | 8.1 |
| split G1, layer convention | 0.0143 | 21 | 4.8 / 4.1 | 7.4 |
| split G1, capable convention | 0.0190 | 28 | 5.1 / 4.5 | 7.9 |
| split G2, layer convention | 0.0187 | 27 | 4.7 / 3.9 | 7.1 |

Reading: the split cuts the linear access a f B to 0.44 of what fix77 ran (0.71 of the pooled group on the
same convention), but every predator -> benthivore pair stays saturated (x 3-3.5), so the take per unit eat
share falls only ~15 % (5.7 -> 4.8 /yr). fix77's realised predator M2 on benthivores was 0.48-0.65 /yr
(resume 223.25, model output, context only) against a booked 0.18: that is an effective eat x hunger of
~0.1 on the benthivore pair, well inside what either structure allows. The split removes the bed food
(epifauna) from 63 % of the predator biomass and makes the planktivores the water-column group's best pair,
but it does not by itself bound the benthivores' predation: the policy's eat share remains the limiter.

---

## 7. Downstream checklist (if implemented; grep 2026-10-27, read-only)

Core code under `lib/` names no group id: group lists, observation layout and DM count come from the yaml
(`config_loader.py` 717-750 builds observes / menu / impact by key prefix; `state.py` 286-298). The work is
yaml data, tests, viewer name tables and a few probes.

- **FG set and checkpoints.** One more decision maker; every DM that observes the new group gains an input
  slot, so every `policy_*.pth` / `obs_stats` is invalid (train from scratch).
- **`fgconfig/fg_library.yaml`.**
  - Species block (670-761) for each group: member_species, display_name, fields of section 2, sources.
  - Pair keys: 19 `predatory_fish_preys_on_*` / `*_preys_on_predatory_fish` keys today, 7 checked (pred ->
    planktivores 1141, -> epifauna 1308, -> benthivores 1375, -> zooplankton 1437; seals -> pred 1186,
    benthivores -> pred 1357, porpoises -> pred 1539); each new group needs its row and column (the
    matrix grows by one row and one column), plus the new wc -> bed pair. Unchecked pairs that carry M2:
    planktivores -> pred 1461 (0.0799), seabirds -> pred 1575 (0.0078).
  - Observability: `<row>_observes_<col>: {observes: bool}` keys (17 today; predatory_fish observes
    epifauna, benthivores, self, planktivores, porpoises, seals, zooplankton; observed by benthivores 1957,
    planktivores 1992, porpoises 2010, seabirds 2028, seals 2046, zooplankton 2064). Proposed: bed observes
    epifauna, planktivores, zooplankton, benthivores, wc, seals, porpoises, self; wc observes planktivores,
    zooplankton, benthivores, bed, seals, porpoises, self (no epifauna); benthivores, seals, porpoises,
    planktivores and zooplankton observe both. The FG editor forces observes on when preys_on is set
    (fgconfig.py 3543-3563).
  - Impact keys: bottom_trawling 1769 (0.00368 -> 0.001704 / 0.004836), pelagic_trawling 1790 (0.00549 ->
    0.001372 / 0.007983), rotor 1802 and windfarm_noise 1808 copied.
  - Seals `spawn.refs` (908-915: benthivores + predatory_fish): decide whether seals follow whiting (J: both
    new groups, as "both demersal groups" in 209).
  - Stored `growth_rate`: both new groups; mobile_epifauna (0.0011979) and its K table refit; planktivores,
    zooplankton, benthivores unchanged if the M2 sums are kept exactly.
- **`mareld2.yaml`.** `group_layers` (121: bed `bottom`, wc `at_50m`; the loader raises if a Q10 group has
  no entry, test_temperature 148-155); decision_makers (132-135: 4300 t -> 1605 / 2715 t, ranges x0.5-2);
  `FollowDemersalFish` refs (262; J: bed group + benthivores; the wc group could follow `Followfish`).
- **Viewer / i18n.** `lib/viz/i18n.py` FG_NAMES sv (48 "Rovfisk") and it (63) for both ids
  (test_viz_i18n 52-56 `test_names_cover_the_library`); species names (whiting "vitling" exists); viewer
  codes from display names must stay unique in every language (test_viz_abbrev 21, 76; `bf` and `prf`
  pinned at 28, 33); colours are golden-ratio hues by position, so a new DM shifts later groups' hues.
  `fgconfig.py` sv_mapping comes from i18n (line 96), no edit there.
- **Probes and scripts.** `tools/probes/frozen_years.py` 34-38 `AB` codes (a missing id raises KeyError at
  110/120/130; add e.g. 'wht' and keep 'pred'); `api.py` 61-77 `INTERNAL_TO_API_SPECIES` (keep
  `predatoryFish`, add one; 74-75 show the alias pattern used for benthic_community); `compare_scenarios.py`
  30. budget_gate, a17_score, viability, biomass_html, inference, train: data-driven.
- **Tests that pin numbers or sets.**
  - test_half_saturation.py 70-89 (rate lock: pred -> epifauna 0.13, benthivores -> pred 0.026, pred ->
    planktivores 0.012, -> zooplankton 0.002, -> benthivores 0.18) and 92-116 (light settings; pred ->
    epifauna has no pair settings).
  - test_gpu_spawn.py 59 (`(3, 12, 72)` -> 13).
  - test_regional_capacity.py 84 (`CAPPED` list).
  - test_growth_budget.py 155 (HAND_SET), 163-170 (r_max > 0), 173-180 (stored = derived), 214.
  - test_temperature.py 43-47 fixture, 148-155.
  - test_fishing_maps.py 68 (predatory_fish total F 0.182).
  - test_gpu_impacts.py 240-249 (only the three fish groups hit by pelagic trawl; whiting's pelagic F
    0.005 keeps it > 0).
  - test_habitat.py 103-117 (no depth table on fish groups; benthic start totals incl. the epifauna refit).
  - test_viz_member_species.py 132, test_viz_group_fold.py 17-18 (DMS), test_viz_i18n.py 22,
    test_viz_abbrev.py 14-17 (FG lists).
  - Gates over every DM of the live library (both new groups must pass): test_energy_balance_gate.py
    255-291 / 371-374, test_reserve_food.py 93, test_holling_response.py 142,
    test_visibility_floor_pairs.py 138, test_viability.py 513 / 585.
- **Docs.** AGENTS.md sections 4-5 (DM list 67-70, the reserve text 285, pair list), a resume section.
- Stale already (not part of this change): `lib/diagnostics/viability.py` 50, 853-914 and pygame_viz.py
  1015, 2509, 4814 still say `gadoids` / `pelagic_fish`.

---

## 8. Open judgements, ranked by how much each could change the outcome

1. **Benthivores -> whiting M2 0.345 /yr** (223.10 Q/B 4.46, the long-run cod share 31 % vs 8.5 % today,
   cod's W = whiting). It sets whiting's booked M 0.78 vs SMS 0.685 (SMS cod on whiting 0.05) and the bed
   group's g: 0.00492 at 0.345, 0.00423 at 0.23 (the 222 value), 0.00329 at the SMS 0.05. Coupled to
   judgement 2: at the whole-group a 0.0026 the pair is unsaturated and the booked flow needs ~15 % of the
   benthivores' eat share.
2. **Attack-rate convention** (section 5). Layer for predator pairs, scored for benthivore fish pairs
   changes six current library pairs by x0.10-x1.27; benthivores -> predators 0.026 -> 0.0026 is the
   largest. Applying it is a library-wide decision, independent of the split. On the f = 1 pairs
   (planktivores, zooplankton) layer is an upper form and scored a lower one (x2-2.5); live on the
   zooplankton pairs only.
3. **Ceiling anchor FB4 vs Bjornsson (x1.30)** for every GER-derived predator ceiling (2.3): h /1.30 on
   every predator pair, demand at the ceiling x1.30, bed g 0.00492 -> 0.00350. Pairs with the same
   argument in 223.15 went to Bjornsson.
4. **The split does not remove the saturation** (bed -> epifauna x 46, fish pairs x 3-3.5): the learned eat
   share stays the only limit on the bed group's epifauna take (up to 8.7 x the room at eat share 0.6) and
   on both groups' benthivore take. What it changes is who has access.
5. **Bed temperature layer and RM** (bottom vs at_50m; RM 3.88 vs 4.2; decide together with the
   benthivores' 3.6 vs 3.32): g 0.0049 vs 0.0054, ceiling 24.1 vs 26.0.
6. **Saithe** carries the water-column group: F 0.11-0.38 (K 1.05-11), no post-2000 North Sea diet, Q3
   survey leverage; its benthivore share 0.05 (0.02-0.15) sets most of wc -> benthivores.
7. **Whiting cannibalism share 0.142** (two thin samples; implies 0.60 /yr vs SMS 0.094): lower W raises
   every other bed pair's share by up to x1.17.
8. **Grouping G1 vs G2** (diet vs habitat): G2 adds bed access for hake, spurdog, anglerfish (27 % more bed
   biomass) and moves 0.10 of the wc -> whiting predation inside the bed group.
9. **Mackerel** undersampled by the GOV 5-10x; seasonal visitors concentrated in the wc group (~0.8
   present).
10. **Seal and porpoise splits by member and their a, visibility floors 0.5 / 0.7, interference 0.1, the
    epifauna f 0.29 and e 0.39 for whiting alone, the 0.1 layer threshold**: judgements carried over.
