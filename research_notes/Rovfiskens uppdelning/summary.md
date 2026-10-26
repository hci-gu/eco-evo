# Splitting `predatory_fish`: summary

Written 2026-10-26 for resume 223.6. Derivations, sources and every number: `params_split.md`; scripts in
`scripts/`, printed output in `tables/`. Nothing in the library, manifest, code, AGENTS.md or resume was
changed (fix76 is live). No model output was used to set a value; section 4 is a diagnostic only.
J = judgement.

## 1. Main findings

1. **Split by diet, not habitat: bed group = whiting.** Whiting carries 92 % of the pooled group's
   benthos consumption and 89 % of its epifauna encounter rate (juveniles 1.6, adults 0.40 km2/t/tick per
   edible tonne). Saithe sits near the bottom by day on the trench slope (FishMap) but eats Norway pout,
   blue whiting and krill there (benthos 0.3-1 % of every diet); hake is demersal by day but eats herring
   and whiting (Skagerrak 2024, n 50: benthos 0.5 %); spurdog (DVM 200 m by day, 60 m at night) and
   anglerfish eat fish. The habitat grouping (whiting + hake + spurdog + anglerfish) gives bed access to
   members that do not use it and makes their whiting predation an unbooked within-group transfer.
   **Recommended (G1):** bed = whiting (0.446 t/km2, 1605 t); water column = saithe, mackerel, spurdog,
   horse mackerel, hake, anglerfish, ling, velvet belly, pollack (0.754 t/km2, 2715 t). Two-way holds; a
   third group (hake + spurdog + anglerfish + ling, 0.12 t/km2) would be thin and adds no pair.
2. **The split does not remove the saturation.** Bed -> epifauna keeps a = 0.69 per edible tonne
   (whiting alone has a higher per-tonne encounter than the pooled group), half-saturation 1/(a h f) =
   0.078 t/km2 of the whole pool (pooled 0.16) against a 7.6 start. What changes: bed access drops from
   1.20 to 0.446 t/km2 of predators; demand at the ceiling with the whole eat share falls from 11.9 to
   4.5 t/km2/yr; saithe and mackerel can no longer fall back on the bed.
3. **The water-column group can pay its way without bed pairs.** Its planktivore pair (dispersed-herring
   encounter a 0.0117) gives 0.79-0.92 of the ceiling at today's density; the budget gate needs 0.35.
   The risk moves to the planktivores: demand at the ceiling 3.9 t/km2/yr vs the predators' budget 1.65.
4. **New learnable cross pair water column -> whiting** (saithe, hake, spurdog, anglerfish): M2 0.157
   /yr; SMS 2023 saithe on whiting 1+ is 0.085 alone (independent check). Benthivores' whole predator
   pair lands on whiting (cod's fish prey in that pool is whiting): 0.23 /yr. Whiting's total booked M1 +
   M2 = 0.67 /yr against SMS whiting 1+ total M 0.685.

## 2. Recommended parameters (G1; ranges and sources in params_split.md section 2)

| Field | bed group (whiting) | water-column group |
|---|---|---|
| id (J) | `bed_feeding_predators` (display "Whiting") | `predatory_fish` kept, or a new id |
| B* (t/km2) / start total | 0.446 / 1605 t | 0.754 / 2715 t |
| own 2001-25 span (GOV wing) | mean 0.378, 0.13-0.79 (B* = x1.18) | mean 0.642, 0.14-2.08 (x1.17) |
| `r_max` | 0.49 (0.44-0.49) | 0.374 (0.32-0.42) |
| `natural_mortality` | 0.25 /yr = 1.712e-4 /tick | 0.261 /yr = 1.791e-4 /tick |
| F bottom / pelagic | 0.075 / 0.005 | 0.213 / 0.029 |
| `bottom_trawling` at 100 / `pelagic_trawling` at 10 | 0.001704 / 0.001372 | 0.004836 / 0.007983 |
| `regional_carrying_capacity` | 0.533 (0.48-0.68) | 2.14 (1.05-11; F/r 0.65, saithe F drives it) |
| energy ceiling (MJ/t/tick) | 26.0 (Cmax 104 kJ/kg/d) | 25.2 (Cmax 100.7) |
| `resting_metabolism` | 4.2 | 3.31 |
| `metabolism_q10` / t_ref / layer | 2.3 / annual_mean / at_50m | 2.3 / annual_mean / at_50m (J) |
| `feeding_cost` / `movement_cost` / `movement_speed` | 2.0 / 2.0 / 1.0 | 2.15 / 2.0 / 1.0 |
| `energy_content` / `max_energy_reserve` | 3900 / 2000 | 4240 / 2600 |
| `prey_includes_reserve` / `reserve_reference_fill` (prey quality) | true / 0.45 (4800 MJ/t) | true / 0.50 (5530 MJ/t) |
| `maintenance_level` | 0.5 (engine convention; literature 0.1-0.2) | 0.5 |
| `starve_rate` | 0.011 (section 209 construction at u 0.15) | 0.0078 |
| `interference` | 0.1 | 0.1 |
| `visibility_floor` | 0.5 (J) | 0.7 (J) |
| Q/B; derived growth_rate (diagnostic) | 4.2; ~0.0041 | 2.83; ~0.00175 (pooled 0.00228) |

## 3. Pairs (G1)

h = E_prey / the group's ceiling; a = section 220 encounter rate per tonne of the whole group, members
counted where they forage in the prey's layer and can take it (score >= 0.1, J); every fish pair Type II;
thresholds -5 deg kept; assimilation 0.8 (0.7 infauna).

| Pair | M2 on prey (/yr) | h | a (MC 5-95 %) | accessible_fraction | dark_ratio |
|---|---|---|---|---|---|
| bed -> planktivorous_fish | 0.159 | 274.0 | 0.021 (0.0068-0.046) | - | 0.33 |
| bed -> zooplankton | 0.054 | 173.1 | 0.0018 (0.0009-0.0068) | - | 0.78 |
| bed -> mobile_epifauna | 0.032 | 144.2 | 0.69 (0.26-2.9) per edible tonne | 0.13 | none |
| bed -> benthivorous_fish | 0.098 | 173.1 | 0.21 (0.065-0.78) | 0.05 (J, gape) | 0.64 |
| wc -> planktivorous_fish | 0.338 | 283.1 | 0.0117 (0.0046-0.033) | - | 0.39 |
| wc -> zooplankton | 0.075 | 178.9 | 0.0006 (0.0004-0.0022) | - | 0.30 |
| wc -> benthivorous_fish | 0.084 | 178.9 | 0.053 (0.018-0.17) | 0.24 (gape) | 0.48 |
| wc -> bed (new) | 0.157 | 190.8 | 0.0106 (0.004-0.037) | 0.47 (gape) | 0.48 |
| benthivorous_fish -> bed | 0.230 | 177.1 | 0.026 (kept) | - | 0.10 / 0 deg |
| seals -> bed / -> wc | 0.0132 / 0.0099 | 22.6 / 26.0 | 0.05 / 0.05 (J) | - | none |
| porpoises -> bed / -> wc | 0.0187 / 0.0062 | 28.6 / 28.6 | 0.20 / 0.20 (J) | - | none |

Unchecked: wc -> epifauna (1.2 % of its diet; the point of the split), both groups -> infauna, bed -> wc
(0.7 %), within-group (section 160), benthivores -> wc. Sums are conserved: planktivores 0.497,
zooplankton 0.129, epifauna 0.032 (was 0.036; the water-column 0.003 becomes unresolved predation).

## 4. Diagnostic (step 4; not used to set anything)

| Quantity | value |
|---|---|
| epifauna maximum surplus r_gross K/4 | 0.879 x 11.9-12.2 / 4 = 2.6-2.7 t/km2/yr |
| taken by benthivores + trawl at equilibrium | 1.11 + 1.04; room left 0.53 |
| bed group on epifauna: share of ceiling at today's pool | 0.99 (x = 98); half-saturation 0.078 t/km2 of the pool |
| bed demand at the ceiling, whole eat share / diet share | 4.5 / 0.26 t/km2/yr (pooled 11.9) |
| bed at the ceiling with eat share 0.137 / 0.30 / 0.82 | 0.62 / 1.35 / 3.7 t/km2/yr = 1.2 / 2.6 / 7 x the room |
| caveats on the two bounds above | the ceiling demand assumes hunger 1; at the derived s* 0.695 the hunger gate is (1 - s)/(1 - u) = 0.61, so a fed whiting group takes at most ~2.8 t/km2/yr (still 5 x the room). The room 0.53 is the loose bound (maximum surplus at K/2); at today's B 7.63 the consistent room is the pooled predators' budget 0.29 |
| G2 bed group (with hake, spurdog, anglerfish) | half-saturation 0.090; demand at the ceiling 5.3 |
| wc on planktivores: share of ceiling (all / 35 % visible) | 0.92 / 0.79; half-saturation 0.30 t/km2 |
| wc demand on planktivores: ceiling / eat share 1 today / diet share | 3.9 / 3.6 / 1.1 t/km2/yr vs predators' budget 1.65 (all predators 2.18; planktivore gross production ~3.4) |
| wc on zooplankton | 0.51 of ceiling at 9.72 t/km2, 0.05 at 0.5; eat share 1 at 9.72 3.2 vs budget 1.26 |
| budget gate (cost / C) vs best pair today | bed 0.41 vs 0.99 (epifauna) and 0.95 (planktivores); wc 0.35 vs 0.92 (planktivores) |

Reading: the split halves-and-more the worst case on the epifauna and removes the fallback for saithe and
mackerel, but a bed group that feeds on the bed can still eat the epifauna several times faster than it
produces; the policy's eat share remains the only limit there. The planktivores and the new whiting
cross pair are the next things to read after any retraining. The split's value is removing bed access from 63 % of the predator biomass and giving whiting an unsaturated cross-pair predator (half-saturation 1.04 t/km2 vs B 0.446); it leaves 223.6's other lever, an accessible epifauna pool that depletes on its own, as the only mechanism that would stop a bed-feeding policy from overshooting.

## 5. Open judgements and uncertainties

1. Grouping G1 vs G2 (hake, spurdog, anglerfish): diet says G1; habitat says G2.
2. Attack-rate averaging convention (layer / noscore / scored / capable differ 2-4x; table in
   params_split.md 3.3); the layer threshold 0.1 lets mackerel (score 0.18) set half the water-column
   planktivore a. Today's library mixes conventions (220 per capable tonne, 221 per tonne of the group).
3. Whiting cannibalism share 0.14 (two thin post-2000 samples) implies 0.60 /yr, SMS says 0.094.
4. Benthivores -> whiting 0.23 /yr rests on the long-run cod share (31 % vs 8.5 % today).
5. Saithe: F 0.11-0.38 (water-column K 1.05-11), no post-2000 North Sea diet, Q3 survey leverage.
6. Mackerel undersampled by the GOV 5-10x; seasonal visitors now concentrated in the water-column group.
7. Seal / porpoise splits by member (whiting 0.44 / 0.64 of their predator share) and their a are J.
8. Visibility floors 0.5 / 0.7, interference 0.1, speeds 1.0, layers at_50m: J.
9. Downstream (not done): FG set (+1 decision maker, checkpoints invalid), manifest (decision_makers,
   group_layers, spawn templates, seal spawn targets), library rows/columns, observes, impact tables,
   the planktivores', zooplankton's, benthivores' and epifauna's growth budgets, viewer codes / i18n,
   api.py, frozen_years.py, fgconfig sv_mapping, tests that pin numbers (test_half_saturation's locked
   rates, test_pair_shares, GPU spawn FG count, K list, growth-sync lists, habitat start totals),
   AGENTS.md sections 4-5, a resume section. Full list: params_split.md section 5.
