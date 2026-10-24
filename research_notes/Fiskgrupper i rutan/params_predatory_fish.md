# Parameters for `predatory_fish` (saithe, whiting, mackerel, spurdog, horse mackerel, hake, anglerfish, ...)

Brief: `split_brief.md`. Target 1.20 t/km2 (4300 t). Weights: long-run (2001-25) depth-stratified NS-IBTS GOV
shares, pooled Q1+Q3, normalised over the ten members: saithe 0.409, whiting 0.372, mackerel 0.080,
spurdog 0.050, horse mackerel 0.035, hake 0.025, anglerfish 0.014, ling 0.007, velvet belly 0.005,
pollack 0.004 (`predators/tables/pred_season.csv`). Ranges of weighted means are corner bounds (every
species at its bound), not confidence intervals. Tags: [F] full text / data, [A] abstract, [S] secondary,
[D] computed here, [J] judgement. Nothing in the library, manifest or code was edited.

## Bottom line

### Species fields (per tick = 6 h, 1460 ticks/yr)

| Field | Value | Range | Basis |
|---|---|---|---|
| `r_max` | **0.42 /yr** | 0.36-0.44 | Euler-Lotka r_now = r_max (1 - ln d / 3) per species [D]: saithe 0.44, whiting 0.49, hake 0.26 (resume 184.18), mackerel 0.25, horse mackerel 0.33 (`Den pelagiska poolens reglering/productivity.md`), spurdog 0.07 (FishBase prior, life table, realised production agree), anglerfish 0.47, ling 0.55, pollack 0.63 (evidence B) |
| `natural_mortality` M1 | **0.257 /yr = 1.76e-4 /tick** | 0.19-0.31 /yr | M0 + predation by non-model groups, per species (evidence B): saithe 0.29 (2026 assessment M; SMS has no saithe predators), whiting 0.25 (SMS residual 0.20 + grey seal/birds 0.053), mackerel 0.15, spurdog 0.10, horse mackerel 0.30, hake 0.45, anglerfish 0.25. Excludes within-group predation (see notes) |
| `energy_content` (lean, as eaten) | **4100 MJ/t** | 3800-4400 | lean kJ/g: saithe 4.0, whiting 3.9, hake 3.7 (`Torskfiskars energiinnehåll`), mackerel 4.7, spurdog 6.0, horse mackerel 4.2, anglerfish 3.3 [J] (evidence A Q5) |
| `max_energy_reserve` | **2400 MJ/t** | 1800-3000 | gadoid liver 2000 (cod proxy, wild autumn max to well fed); mackerel fat 2 -> 31 %, ~6000; horse mackerel ~3800; spurdog liver ~2000 [J]; anglerfish 1500 [J]. Mackerel is 20 % of the weighted reserve |
| `prey_includes_reserve` | **true** | - | the liver/fat is eaten with the fish. It matters for the level, not the season: seals and porpoises get 5260 instead of 4100 MJ/t (+28 %). The gadoid part swings only +-13 % over the year; mackerel ~2x, but it is 8 % of the group |
| `reserve_reference_fill` | **0.48** | 0.40-0.55 | weighted mean reserve 1140 MJ/t / 2400 [D]; whole body as eaten 5260 MJ/t (gadoids ~4750, mackerel ~8000) |
| `maintenance_level` u | **0.15** | 0.05-0.25 | gadoid note: wild spring minimum ~0.2 of a 2000 MJ/t liver, danger 0.07, death 0.02-0.035 -> u 0.1; mackerel's spawning low (2-8 % fat) ~0.0-0.2 of its reserve. Lower u moves the derived growth_rate (s* - u) and the hunger gate |
| `starve_rate` | **0.0088 /tick** | 0.006-0.015 | [D] at s = 0 the loss starve_rate x u equals the routine burn (1.5 x SMR) of lean tissue: 1.5 x 3.64 / 4116 / 0.15. Depends on u |
| `resting_metabolism` | **3.64 MJ/t/tick** | 2.97-4.99 | measured SMR at the layer annual mean 8.4 C (evidence A Q2): saithe 3.5, whiting 4.2 (59 mg O2/kg/h [S]), mackerel 3.4 (true SMR; its swimming floor 6.1 gives 3.87 for the group), spurdog 2.1, hake/horse mackerel proxies, anglerfish 1.2 |
| `metabolism_q10` / `metabolism_t_ref` | **2.3 / annual_mean** | 2.0-2.6 | cod SMR 2.2-2.6 [A], ICES GER exponent 2.25 [F], mackerel 2.0 [F], spurdog 2.5-2.75 [D]; layer `at_50m` (annual mean 8.4 C), as for the gadoids |
| `feeding_cost` | **2.1 x RM** | 1.75-2.7 | routine 1.5 x SMR (spurdog 1.52 [A]; saithe activity multiplier 2.0-2.5 at Uopt [A]) + SDA 0.17 of intake (cod, mackerel [F]) at the typical intake ~0.5 x ceiling = 0.17 x 12.7 / 3.64 = 0.6 RM [D] |
| `movement_cost` | **2.0 x RM** | 1.5-2.5 | cruising at Uopt 2-2.5 x SMR (saithe, mackerel; evidence A Q3) |
| `movement_speed` | **1.0** | 0.5-1.0 | physiology allows 6-15 cells/tick cruising for every member but anglerfish (1.4 %): saithe Uopt 1.4 BL/s, mackerel schools 0.44-0.72 m/s net (evidence A Q4). Residence is behaviour, not a speed limit |
| `interference` w | **0.1** (evidence) / 1.0 (numerical choice) | 0-2 | no field evidence of interference for saithe, whiting or mackerel (Rindorf et al. 2006: predator density "none ... significantly affected"); gadoid 1.0 was set against spatial bands (section 86), the interference report allows 0.5-3 without data. Pick 0.1 unless bands return |
| `visibility_floor` (species, prey side) | **0.6** | 0.4-0.8 | [J] no source: how visible a resting fish is to seals and porpoises. Kept from the gadoids; schools of saithe/mackerel in midwater do not hide well, whiting on the bed does |
| `income_growth_kappa` | **not set** | - | only mackerel (8 %) is a fat-cycle fish; the rest store in the liver; the reserve-funded law fits the pooled group |
| `regional_carrying_capacity` K | **2.13 t/km2** | 1.57-2.95 | K = 1.20 / (1 - F/r) = 1.20 / (1 - 0.182/0.417) [D]. Saithe F (0.11-0.38) and r drive the range. Per species F/r is 0.68 (saithe), 0.86 (mackerel), 1.04 (hake): the group F/r 0.44 hides that hake and mackerel are fished at or above r_now |
| F `bottom_trawling` | **0.162 /yr**; table row at 100: **biomass_factor 0.00367** | 0.08-0.19 /yr | catch / total biomass, ICES 2026 (evidence B): saithe 0.30, whiting 0.075, hake 0.27, anglerfish ~0.08 [J], spurdog 0.0076, ling 0.17; gillnet/longline/seine kept on the bottom map (convention). q = 0.162 / 1460 / 3.0142 = 3.673e-5 per unit |
| F `pelagic_trawling` | **0.020 /yr**; table row at 10: **biomass_factor 0.00553** | 0.01-0.03 /yr | mackerel 0.213, horse mackerel 0.035 (catch/TSB 0.038, not 0.053), whiting 0.005. q = 0.020 / 1460 / 0.02496 = 5.528e-4 per unit |
| `min_split_biomass`, `extinction_threshold_factor` | 0.5 / 0.5 | - | not reviewed (kept) |

Total F 0.18 /yr, 89 % on the bottom map (was 0.12, 98/2 for the gadoids).

### Ceiling (Cmax), section 148 energy form

| Item | Value | Range | Basis |
|---|---|---|---|
| Cmax, weighted | **102 kJ/kg/d = 2.44 % BW/d of 4.18 kJ/g prey** | 78-154 | no lab Cmax exists for any member [D, evidence A Q1]: the ICES gastric-evacuation model (Table 3-7, saithe = whiting parameters; mackerel own) with the stomach fill calibrated so cod reproduces the FB4 cod Cmax (Hansson et al. 1996), at 8.4 C: saithe 97, whiting 104, hake 114; mackerel 150 at 13 C (summer fattening needs >= 180; GER gives 118), horse mackerel 120, spurdog 50, anglerfish 45 |
| Energy ceiling | **25.5 MJ/t/tick** | 19.5-38.4 | 102 / 4 (1 MJ/t/tick = 4 kJ/kg/d). Old gadoid ceiling 34.6 (138 kJ/kg/d), i.e. the new one is 0.74 of it |
| Check | field ration 3.34 /yr = 0.91 % BW/d = 50 kJ/kg/d = 0.49 of Cmax | | SMS 2023 and Ecopath 1991 rations sit at 30-60 % of this ceiling |
| `handling_time` per prey | **h = E_prey / 25.5** | E_prey / 38.4 to / 19.5 | energy as eaten; see the pair table (planktivores 7125, benthivores 4500 from the sibling reviews). If E changes, recompute h |

### Pairs

Consumption Q = Q/B x B = 3.34 x 1.20 = 4.00 t/km2/yr (Q/B 2.84-4.32: saithe 3.0, whiting 4.2, mackerel 2.2,
spurdog 2.0, horse mackerel 3.6, hake 3.5, anglerfish 1.8; SMS 2023 and Ecopath 1991, evidence C). Group diet,
consumption-weighted, shares of total food (evidence C): planktivorous fish 0.41, zooplankton 0.31, benthos
0.11, within-group 0.09, benthivorous fish 0.07, other 0.01. M2 on the prey = Q x share / B_prey.

| Field | `-> planktivorous_fish` | `-> zooplankton` | `-> benthic_community` | `-> benthivorous_fish` |
|---|---|---|---|---|
| `preys_on` | true | true | true | **true (new learnable pair)** |
| E as eaten (MJ/t) | 7125 (4000 + 0.5 x 6250, `params_planktivorous_fish.md`) | 4500 | 3000 | 4500 (whole body, `params_benthivorous_fish.md`) |
| `handling_time` h | **280** (186-365) | **177** (117-230) | **118** (78-154) | **177** (117-230) |
| `max_intake_rate` a | **0.04 (inherited, unmeasured)** | 0.04 | 0.04 | 0.04 |
| half-saturation 1/(a h) at a 0.04 (diagnostic) | 0.089 t/km2 | 0.14 t/km2 | 0.21 t/km2 accessible = 1.06 total | 0.14 t/km2 accessible = 0.94 total |
| a lower bound: 1/(h B_acc), half-saturation at today's mean density | 0.0011 | 0.00058 | 0.00045 | 0.025 |
| `assimilation_factor` | 0.8 | 0.8 (0.75-0.8) | 0.7 (0.65-0.75) | 0.8 |
| `functional_response` | **2** | 2 | default (2) | 2 |
| `predator_share` phi | **0.53** (0.45-0.59) | **0.48** (0.39-0.53) | **0.145** (0.12-0.21) | **0.29** (0.24-0.36) |
| `accessible_fraction` | - | - | **0.2** (0.08-0.45; reused) | **0.15** (0.08-0.3) |
| `dark_ratio` / `light_threshold_deg` | **0.6 / -5** (0.4-1.0) | none (1.0) | none (1.0) | none (1.0; 0.6-1.0) |
| `visibility_floor` (pair) | 0.25 (kept) | default (prey's) | default | default (prey's) |
| `predation_mortality` M2 on the prey | **0.50 /yr** (0.31-0.86) | **0.128 /yr** at 9.7 t/km2 (0.05-0.24); **0.23** at the trained annual mean ~5.5 | **0.0046 /yr** (0.002-0.008) | **0.18 /yr** (0.09-0.51; SMS haddock-only floor 0.10) |
| M2 if the diet is renormalised without within-group food | 0.55 | 0.14 | 0.005 | 0.21 |

`predatory_fish_preys_on_predatory_fish` stays unchecked (within-group, section 160).

## Notes per item

**Attack rate a.** It cannot be derived from the literature (as for the gadoid benthos pair,
`../Bentos åtkomlighet för torskfiskar/benthos_accessibility.md`). At the inherited 0.04 every pair is
saturated at today's densities (half-saturation 0.089-0.14 t/km2 for fish and zooplankton), so intake has
no prey-density response. The lower bound puts the half-saturation at today's grid-mean prey density, which
would reproduce the field ration (0.5 of Cmax) only if the whole biomass ate every tick on all its prey. It
is confounded: the time budget (routine 1.5 x SMR, half the time not feeding) explains the 0.5 as well, and
the Holling term sees visible prey only. At the bound, with visible herring ~0.3-0.4 of 3.3 t/km2,
a h B_vis ~0.35 (a 0.0011, h 280) and the pair yields ~5 MJ/t/tick against a feeding cost of 2.1 x 3.64 = 7.7: it cannot close
its own budget. So a lies between "saturated everywhere" (0.04) and "budget fails" (the bound). Settle it in a
probe, not from the literature.

**Functional response.** Biology says Type II: whiting and cod show negative prey switching on fish
(Rindorf, Gislason & Lewy 2006, 36 000 stomachs [F]); only mackerel filter feeding on copepods is Type III-like
(Macy et al. 1998 [F]), 11 % of the zooplankton weight. The gadoid -> herring pair is Type III for a model
reason (section 141: herring went extinct within two years under Type II), not a measured one. With Type II
and a 0.04 (half-saturation 0.089 t/km2) the planktivores have no low-density refuge. Keeping Type II hands the refuge question to the
engine: the pair visibility floor, the planktivores' K, or a lower a. Do not pick III here for stability
without saying so.

**predator_share.** Same stage x species method as `../Torskfiskarnas bytesandelar/gadoid_prey_shares.md`.
Saithe, whiting and hake scores are reused; mackerel, horse mackerel, spurdog, anglerfish, velvet belly and
every `benthivorous_fish` score are judgements from mouth, gape and habitat (`predators/inputs/pred_scores.csv`).
Stage shares are the long-run GOV length biomass: whiting < 20 cm 0.25, saithe >= 60 cm 0.37, hake < 20 cm 0.02.
Zooplankton = 0.5 krill-sized + 0.5 copepod-sized scores (the pool is A17-calibrated, mostly copepods).
The four caps sum to 1.45; joint "planktivores or zooplankton" 0.65 (0.58-0.71); all four 0.67.
Cap test against the renormalised diet (mass shares P 0.45, Z 0.35, B 0.12, Bf 0.08): all below phi by mass.
As eat shares at the energy-equalised ceilings (eat share proportional to mass x E): P 0.58, Z 0.29, B 0.065,
Bf 0.065. **P is near binding** (0.58 vs 0.53, inside the range). Q1 vs Q3 caps (`tables/pred_phi.csv`): benthos
0.20 / 0.12, planktivores 0.41 / 0.58, zooplankton 0.38 / 0.51, benthivores 0.35 / 0.27. One static cap cannot
carry the season.

**accessible_fraction on benthivores.** Share of benthivore biomass a predator can swallow: GOV long run
< 15 cm 0.05, < 20 cm 0.15, < 25 cm 0.26 (`tables/benthivore_prey_sizes_stdout.txt`). Gape limits (prey length
<= k x predator length): whiting ~12 cm, saithe ~19 cm, hake ~28 cm, spurdog 25-30 cm, anglerfish > 40 cm.
Consumption-weighted 0.12, plus the 0-group the GOV does not catch (~0.02-0.05) -> 0.15 [D/J].

**Light.** Consumption-weighted dark_ratio 0.61 on planktivores (saithe 0.6, whiting 0.7, mackerel 0.15,
horse mackerel 0.3, hake 0.5, spurdog 0.85), 0.92 on zooplankton, 0.66 on benthivores (evidence C section 4).
The planktivore pair takes the gadoid value 0.6, -5 deg. Zooplankton stays at 1.0: whiting, saithe and hake
krill feeding peaks at dusk and night (Onsrud et al. 2004: krill 82 % of whiting stomachs day and night), which
the engine cannot express above 1. Benthivores: 1.0 (Strand et al. 2020: night efficient on 0-group gadoids).

**M2 on the prey.** Prey biomasses: planktivores 3.3, zooplankton 35000 t / 3600 km2 = 9.7 (start value; the
trained annual mean was ~5.5 in resume 185.19, which nearly doubles the M2), benthos 95, benthivores 1.53.
- **Planktivores 0.50 /yr is the largest flow handed over.** It rests on saithe diets from the Norwegian coast
  (Mehl 2005, Oct-Nov 1998-2003) and Ecopath 1991, where Norway pout and blue whiting dominate the fish
  part. The planktivore pool here is 80-90 % herring, so the herring part is uncertain. No post-2000 North
  Sea saithe diet exists. It adds to the planktivores' derived growth_rate (budget check for that review).
- Zooplankton 0.13-0.23 depending on the reference biomass; Z share 0.15-0.45 is defensible (saithe krill
  2-92 % between years).
- Benthivores 0.18 /yr (predator side: Q x 0.07 / 1.53): whiting 1+ and saithe on juvenile haddock/cod,
  spurdog and anglerfish on flatfish and gadoids. The benthivore review's ~0.10 (`params_benthivorous_fish.md`
  item 11) is the prey-side SMS 2023 flow of saithe + whiting on haddock only (0.25 /yr on haddock); it omits
  cod, flatfish and the elasmobranchs, so it is the floor of this range. Both reviews should book one value:
  0.18 if the diet share 0.07 is trusted (whiting 0.08, 0.04-0.21, thin in-area data), 0.10 if only the
  assessed flow is.
- Norway pout check: the ICES uploads 2022-24 carry Norway pout under two AphiaIDs, 126444 "Trisopterus
  esmarkii" and 400596 "Trisopterus esmarki" (73 records, 62 of them in cod stomachs; 7 in the seven members the script reads: whiting, hake and anglerfish 2 each, mackerel 1).
  `predators/scripts/predator_diet_pools.py` classifies both as planktivorous fish (prefix match on the WoRMS
  leaf "Trisopterus esmark"; verified on both cached lineages), so the diets and M2 above are not
  under-counted and were not changed.

**Within-group predation (not booked).** Saithe and whiting cannibalism take ~0.18 /yr of whiting 1+
(SMS 2023: saithe 0.085, cannibalism 0.094), ~0.07 /yr on the group. In a pooled group it is a transfer:
the net group loss is about (1 - gross efficiency) x 0.07, which neither M1 nor a pair carries.
M1 with it would be 0.32 /yr.

**Manifest (caller's edit).** `metabolism_t_ref: annual_mean` and the Cmax at 8.4 C assume the group sits on the
`at_50m` temperature layer. `mareld2.yaml` `simulation_settings.temperature.group_layers` maps `gadoids: at_50m`;
`predatory_fish` needs its own entry, or the Q10 term is not applied as intended.

**Model-group predation on this group (downstream, not mine).** SMS puts harbour porpoise M2 on whiting 1+
at 0.19 /yr and cod 0.05: this exists only if `porpoises_preys_on_predatory_fish` and
`benthivorous_fish_preys_on_predatory_fish` are checked and calibrated. The seal pair's 0.12 was fitted to the
old 0.69 t/km2 gadoid pool and needs recalibration. Saithe's M1 0.29 is the assessment's total M; seal and
porpoise pairs add on top of it (small).

## Seasonal visitors in a closed pool

GOV long run (`tables/pred_season.csv`): Q3 kg/h is 2.8 x Q1. Q3 share of Q1+Q3: saithe 0.90, mackerel 0.95,
horse mackerel 0.98, spurdog 0.90, hake 0.84, whiting 0.47. Composition: Q1 74 % whiting, 15 % saithe; Q3 50 %
saithe, 24 % whiting, 10 % mackerel. Literature (evidence B): adult saithe on the trench slope Apr-Dec
(immatures all year), mackerel Jun-Nov, horse mackerel Aug/Sep-Dec; ~0.87 of the year present, weighted.

How a fixed group should treat it:
- The target 1.20 is already presence-averaged (Q1 and Q3 pooled), the weights are annual-mean, and every
  Q/B includes the non-feeding quarters (mackerel Q1 C/B ~0.05). So annual-mean biomass x annual Q/B is
  internally consistent for M2. Do not use feeding-season rations on the annual-mean share (it would count
  the residence twice).
- The model's migration returns biomass at the edge, so the pool is closed. Visitors then produce, die and
  are fished inside the grid at stock-level r and F, the convention already used for herring F
  (`reports/Fiskeridödlighet i Skagerrak idag.md`). Their real spawning and nursery production happens elsewhere.
- Temperature (Q10 on metabolism) and the daylight dark_ratio are the only seasonal levers left. The Q3
  surge of saithe on herring/krill and the Q1 whiting diet (more benthos, benthivores) cannot be expressed by
  static caps.
- Mackerel is undersampled by GOV 5-10x (IESSNS ~2 t/km2 in summer vs GOV 0.2-0.5;
  `../Den pelagiska poolens reglering/stock_composition.md`). Its true share of the group, and so of the
  predation on zooplankton, is higher than 8 %. Whether that mass belongs in this group or is left out of a
  bottom-trawl-based target is a decision for the caller.

## Old gadoid parameters that should NOT carry over

| Old value | Why not |
|---|---|
| h 87 / 130 / 159 / 217 (ceiling 34.6 MJ/t/tick) | 1.4-1.6 x the Cmax of these species (resume 208.2); new ceiling 25.5 |
| `resting_metabolism` 6.0 with costs 1.75 / 2.0 | 6.0 is ~1.6 x SMR, i.e. already routine; with the multipliers it counts activity twice. SMR is 3.6 |
| `energy_content` 5500, `max_energy_reserve` 2500, no reserve split | autumn-maximum whole body and a lab fatty-diet liver; a full fish held 8000 MJ/t |
| `maintenance_level` 0.5 | puts starvation at mean wild condition; 0.1-0.2 from liver data |
| `predator_share` 0.53 on benthos | haddock-driven (benthivores); this group is 0.145 |
| `r_max` 0.37, K 1.02, M1 0.41 /yr | five-gadoid pool with haddock and cod at 0.69 t/km2; now 0.42, 2.13, 0.26 |
| F 0.12 split 98 / 2 | now 0.18 split 89 / 11 (saithe 0.30, mackerel on the pelagic map) |
| `movement_speed` 0.25 | below the physiology of every pelagic member |
| `interference` 1.0 | set against spatial bands, not from feeding data |
| Type III on herring | a stability fix (section 141); measured switching is negative (Type II) |
| seal pair M2 0.12, `gadoids_preys_on_zooplankton` 0.089, herring pair 0.30 | fitted to the old pool and the Ecopath 1991 mix |

## Main uncertainties
1. a (see above): the largest model-side unknown.
2. Saithe: diet (no post-2000 North Sea data), F (0.11-0.38; seasonal visitor, local share unknown), and its
   survey weight (one-haul leverage, Q1 15 % vs Q3 50 %).
3. Cmax: a construction from gastric evacuation plus a cod calibration; no saithe or whiting satiation series.
4. Mackerel: GOV undersampling and its Cmax (110-300 kJ/kg/d).
5. Zooplankton reference biomass (9.7 vs ~5.5) and the krill share of the pool.
6. Prey energy as eaten for the planktivore and benthivore pools sets h; it is the other reviews' call.

## Files
- Evidence: `predators/evidence_A_physiology.md` (Cmax, SMR, Q10, costs, speed, energy density, assimilation),
  `predators/evidence_B_population.md` (r, F by gear, M split, residence),
  `predators/evidence_C_diet_consumption.md` (diets by the new pools, Q/B, light, functional response,
  interference).
- Scripts (run with `python3 -I`, paths as arguments): `predators/scripts/pred_season_length.py` (Q1/Q3 and
  length biomass; guild B gives the benthivore prey sizes), `pred_prey_shares.py` (phi), `weighted_fields.py`
  (weighted fields, h, a bound, M2, K, trawl tables); evidence scripts are listed in each evidence file.
- Inputs: `predators/inputs/species_values.csv`, `pred_scores.csv`, `diet_consumption_central.csv`.
- Tables: `predators/tables/` (`*_stdout.txt` hold the printed numbers). Downloads: `predators/data/`
  (untrusted, README).
