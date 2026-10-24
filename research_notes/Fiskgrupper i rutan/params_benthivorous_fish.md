# Parameters for `benthivorous_fish`

Group: haddock 38.6 %, cod 31.4, long rough dab (LRD) 12.6, dab 4.5, plaice 3.8, witch 1.7, starry ray
1.5, hagfish 1.3, grey gurnard 0.9, lemon sole 0.9, greater weever 0.8, others 2.5 (long-run 2001-25 GOV
biomass shares, `tables/long_run_species_by_group.csv`). Target B* 1.53 t/km2 (5500 t). Every value
below is a biomass-weighted species value. Nothing in the code, library, manifest or resume was edited.

Evidence and computation (all in `benthivores/`):
- `evidence_benthivores_A_bioenergetics.md`: Cmax, SMR, Q10, activity costs, starvation, energy density,
  movement.
- `evidence_benthivores_B_population_fishing.md`: r_max, M0, non-model predation, F per species and gear.
- `evidence_benthivores_C_diets_consumption.md`: diets by model pool, Q/B, cod fish prey, haddock krill,
  gape table.
- `scripts/length_structure.py`: long-run length structure from NS-IBTS (stage shares, biomass-weighted
  W^b for size-dependent rates). `tables/stage_shares_longrun.csv`, `size_factors_longrun.csv`.
- `scripts/group_params.py`: the weighting, h, phi, M2, K and the starvation check;
  `tables/group_params_stdout.txt`, `tables/group_params.csv`. Run with `python3 -I`.

Tags: [F] full text / data, [A] abstract, [S] secondary, [D] computed here. **J** = judgement (no
measurement; my choice, stated). Temperatures: the model applies the manifest's `at_50m` series to this
group (annual mean 8.4 C, 6.6-11.7). The 100-300 m bottom water is colder and steadier (~6-7 C);
where it matters the value at 6.5 C is given too.

## 1. Bottom line: species fields

| Field | Value | Range | Basis |
|---|---|---|---|
| `r_max` (/yr) | **0.31** | 0.22-0.41 | Present-day r per species (CMSY prior - ln(d)/T, section 184.18 method): haddock 0.24, cod 0.35 (reused), plaice 0.40 (R/SSB fall is density dependence, d = 1), dab 0.55, witch 0.28, lemon sole 0.50, gurnard 0.65, starry ray 0.15, LRD 0.30 **J** [F/D, B] |
| `natural_mortality` M1 (/yr) | **0.29** | 0.19-0.46 | M0 0.20 (SMS convention, all bony fish; 0.13-0.17 for hagfish, ray, witch) + predation by groups NOT in the model 0.09 (squid, halibut, megrim, grey seal; 0.04-0.16) [F/D, B] |
| `natural_mortality` (per tick) | **0.000196** | 0.000130-0.000315 | 1 - exp(-M1/1460) |
| `energy_content` (MJ/t), whole body | **4500** | 3900-4800 | roundfish 4.6-4.75 kJ/g (gadoid note), plaice 4.6, LRD 3.2, witch 5.6, lemon sole 5.8, weever 5.3, gurnard 4.4 [F/S, A]. Use this if the library keeps whole-body content with the reserve on top (option a of section 187.5) |
| `energy_content` (MJ/t), lean | 3600 | 3400-3850 | only with the lean/reserve split (option b of 187.5): gadoids 3.8, flatfish 3.6, LRD 2.6 **J** |
| `max_energy_reserve` (MJ/t) | **1800** | 1500-2100 | cod/haddock liver max 2000 (gadoid note), plaice 1700-2000, LRD 850-1100 [A]; with the split, reference fill (4500-3600)/1800 = 0.48 |
| `maintenance_level` u | **0.5** | (lit. 0.1-0.2) | engine convention kept: u is also the hunger knee and growth zero; 0.1 starves the group (187.4) |
| `resting_metabolism` (MJ/t/tick, annual mean) | **3.6** | 2.4-4.3 | SMR 3.0 (haddock 3.0, cod 3.7 **J**: Schurmann & Steffensen's 4.1 lowered for the long-run cod size of ~1.2 kg, test-fish mass not stated in A; 4.1 moves RM by < 0.1, plaice 3.1, LRD 1.6, witch 1.5) x 1.2 routine at rest (haddock routine/SMR 1.42) [F/D, A]. The old 6.0 is 1.5-2x measured SMR |
| `metabolism_q10` / `metabolism_t_ref` | **2.25** / `annual_mean` | 2.0-2.5 | haddock 2.5, cod 2.2 (Schurmann & Steffensen 1997), plaice 2.0 (Fonds 1985) [F, A] |
| `feeding_cost` (x RM) | **2.0** | 1.5-2.5 | plaice fed/unfed respiration 2.4 (1.7-3.0) incl. SDA; feeding + SDA 15 % of a maximal ration; haddock SDA peak 1.3-1.5 x RMR [F, A]; **J** for the pooled value |
| `movement_cost` (x RM) | **1.6** | 1.3-2.3 | haddock cruising 1.48 x SMR, cod 1.32, plaice active/rest 2.4-2.8 [F, A]; weighted 1.8 x SMR = 1.5 x RM; **J** |
| `movement_speed` (cells/tick) | **0.2** | 0.1-0.25 | haddock, cod 0.25 (as gadoids), gurnard 0.2, flatfish and rays sedentary 0.1 **J** [A] |
| `starve_rate` | **0.010** | 0.008-0.013 | T_total (section 46 convention, 95 % loss from s = u, rest only, RM 3.6, ME 1800) = 181 d (0.008: 218 d; 0.013: ~145 d); cod fast 84 d+ without deaths, plaice 180-200 d at 6-7 C to K 0.75 [F, A; D] |
| `interference` w (km2/t) | **0.45** | 0.3-1.0 | **J**: the gadoid w 1.0 is a spatial-crowding calibration at 0.69 t/km2 (section 86); keeping w x B constant at 1.53 gives 0.69/1.53. No fish measurement exists (Täthetsberoende note) |
| `visibility_floor` | **0.55** | 0.4-0.7 | **J**: gadoids 0.6; flatfish and rays bury and are cryptic (0.4); weighted 0.54. The seal (0.85) and porpoise (0.75) pair floors still override |
| `min_split_biomass` | 0.5 | | unchanged (numerical) |
| `income_growth_kappa`, `prey_includes_reserve` | not set | | reserve-funded group; set `prey_includes_reserve` only with the lean split (187.5 b) |
| F `bottom_trawling` (/yr) | **0.17** | 0.14-0.24 | catch / total biomass: cod 0.39, haddock 0.06, dab 0.18, witch 0.12, LRD 0.10 **J**, lemon sole 0.09, plaice 0.04, others ~0.03 [F/D, B]; seines and gillnets on this map |
| F `pelagic_trawling` (/yr) | **0.0004** | 0-0.001 | 0.2 % of the F: industrial bycatch of haddock [D, B]. Can be left out |
| `regional_carrying_capacity` K (t/km2) | **3.4** | 2.3 - none | K = 1.53 / (1 - 0.17/0.31). Fragile: F/r = 0.55, and in the high-F low-r corner F > r (no stationary K). With the 2021-25 weights (F 0.09, r 0.26) K = 2.3. Cod alone has F 0.39 > r 0.35 |

Cmax (section 2) is not a library field; it sets the pair h below.

## 2. Cmax and the ceilings (section 148 convention)

Per species, annual mean over the 50 m series, at the long-run biomass-weighted size (mean of W^b over
the length mix, `tables/size_factors_longrun.csv`) [D from A]:

| Species | Cmax kJ/kg/d (range) | Basis |
|---|---|---|
| Cod | 83 (71-95) | Hansson et al. 1996 (FB4), W^-0.195 over the long-run cod mix (W_eff ~1.2 kg), Kitchell f mean 0.80 [F] |
| Haddock | 144 (111-177) | cod function x 1.3 at the haddock mix (W_eff ~275 g). **J**: juvenile haddock eat 1.4-2x cod at equal size in every lab comparison; no adult data [A] |
| Plaice | 119 (84-142) | Fonds et al. 1985 table and fit routes at 150 g, scaled to 190 g [F] |
| Dab | 100 (68-115) | 0.7 x plaice (DEB ratio); Bagge 1995 2.2 %BW/d at 9.5 C agrees [A] |
| LRD | 86 (50-116) | **J**: 0.6 x plaice, a cold-water slow species (field SMR 2-3x lower than plaice) |
| Witch | 68 (40-92) | **J**: at or below dab |
| Others | gurnard 135, lemon sole 126, ray 50, hagfish 20, weever 90 | **J** analogues |
| **Group** | **108 (82-133)** | biomass-weighted; at 6.5 C ~92 |

- Energy ceiling = 108/4 = **27.1 MJ/t/tick** (20.5-33.2), against the old gadoid 34.6 (138 kJ/kg/d).
  The old ceiling was 1.3x this group's Cmax and 1.7x cod's.
- Haddock carries the largest uncertainty: at x1.0 the group Cmax is ~96 (ceiling 24).
- Cmax Q10 ~2 between 6 and 12 C (cod function 1.85, plaice table ~3, juvenile haddock lower) [D, A]:
  a 1.4-1.8x swing April to October. An `ingestion_q10` ~2 (annual mean 1) is defensible; without it
  the h below are the annual mean.
- The stomach limit is closer to an energy limit: gastric evacuation of wet mass ~E^-0.875 (Temming &
  Herrmann 2003 type model) [A], so the energy convention of section 148 holds.
- Check: the realised Q/B 3.5 /yr (C) is 26 % of the ceiling Q/B on benthos, in line with wild haddock
  feeding at 15-22 % of maximum capacity (Hall 1987) [A].

## 3. Bottom line: pairs

h = E_prey / 27.1 (ticks per t of prey per t of predator). E_prey is the energy as eaten; the planktivore
and predator values are placeholders until those groups' energy contents are fixed (h scales linearly).

| Field | `-> benthic_community` | `-> planktivorous_fish` | `-> predatory_fish` | `-> zooplankton` | `-> benthivorous_fish` |
|---|---|---|---|---|---|
| `preys_on` | **true** | **true** | **true** | **false** (drop) | **false** |
| `handling_time` h | **111** (90-146), E 3000 | **277** (226-366), E 7500 | **175** (143-232), E 4750 | (166 at E 4500) | (out) |
| `max_intake_rate` a | 0.04 | 0.04 | 0.04 | | |
| `assimilation_factor` | 0.7 | 0.8 | 0.8 | (0.8) | |
| `functional_response` | 2 (generalist default) | 3 | 3 | | |
| `predator_share` phi | **0.90** (0.65-1.0) | **0.31** (0.17-0.43) | **0.10** (0.04-0.16) | (0.03) | |
| `accessible_fraction` | **0.2** (0.08-0.45) | not set (1) | not set (1) | | |
| `dark_ratio` / threshold | unset (1) | **0.6** (0.4-1.0) / -5 deg | unset (1) | | |
| `visibility_floor` (pair) | - | 0.25 (kept) | not set | | |
| `predation_mortality` M2 on prey (/yr) | **0.040** (0.029-0.055) | **0.14** (0.07-0.32) | **0.085** (0.05-0.22) | (0.014; 0.002-0.04) | (0.15 t/t/yr, out) |

Basis per pair:
- **a 0.04 stays** on every pair: search width and capture probability have never been measured for
  these fish (0.007-5.8 in the accessibility note, section 208.2); a sets only the approach to a ceiling
  that the benthos pair reaches anyway.
- **Benthos phi 0.90**: morphology/habitat scores (`../Torskfiskarnas bytesandelar`): haddock 1.0 at all
  stages, cod 1.0 / 0.72 / 0.64 for < 30 / 30-50 / > 50 cm weighted by the LONG-RUN cod stages
  (9 / 32 / 59 %; 2021-25 had 13 / 39 / 48) = 0.70, flatfish, rays, gurnard, weever, hagfish 1.0
  [D]. Diet check: benthos 72 % of the mass diet (62-73 % in all variants; plaice 99.5 %, haddock 83 %,
  cod 49 % Skagerrak 2023-25) [F/D, C]: under the cap. Assimilation 0.7 and accessible fraction 0.2 are
  the gadoid values (208.2); no new evidence. No light effect: benthic feeding is chemo-tactile and
  rises at night if anything (light note).
- **Planktivorous fish phi 0.31** (re-scored for the new pool, which now holds Norway pout): gape k x L
  against the near-bottom prey lengths (C, section 4): cod 40 cm swallows 96 % of the Norway pout
  biomass and 45 % of herring, cod 50 cm 100 / 89 %; haddock 30 cm 24 % / 1 %; LRD < 10 % of Norway
  pout. Scores (access x capability) **J**: cod 0.3 / 0.7 / 0.9 by stage, haddock < 40 cm 0.1, >= 40 cm
  0.25, starry ray 0.4, gurnard 0.15, LRD 0.05, other flatfish 0. Cod carries 0.25 of the 0.31. Diet
  check: 8.4 % of the mass diet (6.5-14.7 %); cod Skagerrak 22.5 %, of which Norway pout ~36 % of fish
  mass, herring + sprat 2-7 % [F/D, C]. Type III and dark_ratio 0.6 as the old gadoid -> herring pair
  (sections 141, 144). Most of this M2 falls on Norway pout, a small part of a herring-dominated pool;
  an `accessible_fraction` for the near-bottom part of the pool is an open option, not proposed.
- **Predatory fish phi 0.10**: mechanistically real: cod > 50 cm takes whiting (58 % of near-bottom
  whiting biomass within a 50 cm cod's gape, 93 % for 70 cm), whiting 5 % of the Skagerrak cod diet;
  grey gurnard and starry ray take juvenile whiting (Floeter 2005 [F]). Scores **J**: cod 0 / 0.15 /
  0.4, ray and gurnard 0.3. Saithe (the larger half of the prey group) is mostly out of reach.
- **Zooplankton: drop.** (1) The model pool is A17-calibrated, mostly Calanus; a copepod-sized phi is
  0.03 (cod < 30 cm 0.12, pelagic 0-group only). (2) Haddock, the only member scored as a krill feeder,
  had that score from Barents Sea diel ascents; 485 winter stomachs in 2025 hold 0.4 % zooplankton, and
  no post-2000 summer krill record exists for the area [F, C]. (3) The hyperbenthic crustaceans that
  the demersal fish do eat (mysids, amphipods, Crangon, Pandalus) are already in `benthic_community`.
  (4) Flatfish ~0. The keep variant: phi 0.03-0.05, h 166, M2 0.014 /yr (2.5 % of the diet; 0.6 % with
  the measured winter haddock diet). Dropping it moves < 3 % of the demand to benthos.
- **Within group** (cod on haddock and flatfish, rays on hagfish): 4.3 % of the diet, 0.15 t per t per
  year. Stays out (section 160).

M2 = Q/B x B_pred x diet share / B_prey, with Q/B 3.48 /yr (2.5-4.7; 3.0-3.5 at 6-9 C; haddock 3.8,
cod 3.2, LRD 3.4, dab 3.5, plaice 3.4, lemon sole 4.3, ray 1.7; Ecopath NS 1981/1991, Greenstreet 1997)
and the consumption-weighted diet (B 72 %, P 8.4, W 1.9, Z 2.5, within 4.3, no pool 10.9) [D, C]. Prey
biomasses: benthos 95, zooplankton 9.7 (35 000 t / 3600), planktivores 3.3, predators 1.2 t/km2. The
ranges combine the Q/B range and the diet variants. The 10.9 % with no pool (sandeel, gobies,
cephalopods, unidentified fish, carrion) must be taken from the pools at the ceiling.

## 4. What should NOT carry over from `gadoids`

1. **h 87 / 130 / 159 / 217** (h_ref 130 = 138 kJ/kg/d): 1.3x this group's Cmax. New h_ref (E 4500) is 166.
2. **natural_mortality 0.000284** (0.41 /yr): its 0.214 non-FG predation held gurnards, LRD, rays and
   monkfish, now inside model groups. New 0.29 /yr.
3. **regional_carrying_capacity 1.02** (from 0.69 t/km2, r 0.37, F 0.12). New 3.4 at r 0.31, F 0.17.
4. **r_max 0.37** (gadoid pool incl. saithe and whiting, 2021-25 weights). New 0.31.
5. **predator_share 0.53 / 0.56** on benthos (a gadoid mix with saithe and whiting). New 0.90.
6. **gadoids_preys_on_zooplankton** (M2 0.089, Ecopath 1991 krill share). Drop; at most 0.014.
7. **predation_mortality 0.30 on pelagic fish** (Ecopath 1991, herring). New 0.14 on the planktivore pool,
   mostly Norway pout; herring + sprat are 2-7 % of cod's fish prey.
8. **resting_metabolism 6.0**: 1.5-2x measured SMR; at the field ration (Q/B 3.5 on benthos = 5.0 MJ/t
   per tick assimilated) a 6.0 cost cannot be paid. New 3.6.
9. **energy_content 5500** (autumn maximum of gadoids). New 4500 whole body (flatfish and LRD are leaner).
10. **interference 1.0**: a crowding calibration at 0.69 t/km2. New 0.45 at 1.53.
11. **Pairs where this group is the prey must be re-targeted** (not this note's fields; flags only):
    - `seals_preys_on_gadoids` 0.12 (309 t/yr over 2500 t). Seals eat flatfish (46 % of energy) and
      haddock: on 5500 t the same ~325 t/yr ration gives ~0.035 /yr (0.03-0.045) if 50-75 % of it is
      benthivores **J**.
    - `porpoises_preys_on_gadoids` 0.0557 (Ecopath 1991). SMS 2023 gives porpoises 0.135 /yr on cod of
      ages 2+ (0.19 for 1+), i.e. ~0.04 on the group from cod alone [D, B].
    - predatory_fish -> benthivorous_fish (new pair): SMS 2023 saithe + whiting on haddock 0.25 /yr
      (monkfish 0.013), i.e. ~0.10 on the group [D, B].
    - the gadoid cannibalism pair (0.2168): within-group, stays out.
12. **M1 fallback**: if `predatory_fish_preys_on_benthivorous_fish` is not built and checked, its ~0.10
    /yr sits in neither M1 nor the budget; then add it to M1 (0.29 -> 0.39 /yr, 0.000267 per tick).
13. **Benthos K (section 207)** was derived with the gadoid M2 0.0048: the equilibrium share
    1 - (F + M2)/r goes 0.77 -> 0.73 with M2 0.040, so the per-cell K or the equilibrium start (~95
    t/km2) shifts by ~5 %. Re-derive it with the new M2.

## 5. Main uncertainties

- **The cod share sets F, K and the fish pairs.** Cod is 31 % of the long-run group but 8.5 % in
  2021-25. With 2021-25 weights F falls 0.17 -> 0.09, r 0.31 -> 0.26, K 3.4 -> 2.3, and the fish phi and
  M2 on planktivores roughly halve. F/r = 0.55 makes K sensitive: F 0.20 or r 0.27 alone gives K 4.1-4.3.
- **Haddock Cmax** has no adult measurement (x1.0-1.6 on the cod function; group ceiling 24-30).
- **LRD (12.6 %)** is judgement in r, F, Cmax, SMR and energy density; no assessment, no discard data, no
  ration, closed-access diet (Ntiba & Harding 1993).
- Starvation targets are times to poor condition or fasting without deaths, not measured deaths.
- The phi scores for the fish pairs and the visibility floor are judgements on morphology and gape.
- M1's non-model predation (0.09) rests on the 1991 North Sea Ecopath balance of the old gadoid pool.

Status of the evidence files: A (haddock Cmax, flatfish SMR and starvation), B (all r, M, F), C (diets
from the public ICES stomach database 1991-2025 filtered to Skagerrak / NNS / Norwegian Trench, plus
literature). Closed texts not read: Fonds 1992, Duthie 1982, Hall 1987, Jones 1974, Pedersen & Hislop
2001, Ntiba & Harding 1993, Skjaeraasen & Bergstad 2000, Schückel 2010.
