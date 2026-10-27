# Splitting `predatory_fish` on the 223.15 library: summary

Written 2026-10-27. Re-derivation of `summary.md` / `params_split.md` (2026-10-26, section-222 library), which are left untouched. Derivations, sources and every number: `params_split_223.md`; scripts `scripts/*_223.py`, output `tables/*_223*`. Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed. No model output set a value; section 4 is a diagnostic. J = judgement.

## 1. Main findings
1. **G1 holds.** Bed group = whiting (0.446 t/km2, 1605 t); water-column group = saithe, mackerel, spurdog, horse mackerel, hake, anglerfish, ling, velvet belly, pollack (0.754 t/km2, 2715 t). E1 helps the argument: the exposed epifauna is whiting's prey (91 % of the pooled flow), and no member eats Nephrops beyond traces (whiting <= 0.55 %, saithe and hake none), so both groups -> norway_lobster stay unchecked. The infauna refuge does not touch the predators (no infauna pair).
2. **What 223 changes in the fields.** The bed group goes on the `bottom` layer (J, as the benthivores): ceiling 26.0 -> 24.1 MJ/t/tick (GER e^0.078 T at 7.40 C), RM 4.2 -> 3.88 if re-read at 7.40 C. The lean + reserve convention was already used (prey quality 4800 / 5540). K unchanged (0.533 / 2.14).
3. **What 223 changes in the pairs.** Epifauna M2 doubles on the smaller E1 pool (bed 0.069; was 0.032) and the bed pair's a carries e: 0.69 x 0.39 = 0.27, f 0.29. Benthivores -> whiting rises 0.23 -> 0.345 /yr (223.10's 0.128 on 1.20 lands wholly on whiting: cod's fish prey in the pool is whiting), h 183.3 at the 26.18 ceiling. Whiting's booked M becomes 0.78 /yr (SMS 0.685) and its derived g 0.0049 /tick (pooled 0.0024).
4. **The attack-rate convention is settled by the engine.** `demand = B_pred x eat x hunger x a fB/(...)` with no phi: a is per tonne of the whole group. Recommended: "layer" for predator pairs (gape in f, bed exposure in e, time in the eat share; on the f = 1 pairs an upper form), "scored" (capable x phi) for the benthivore fish pairs. Six current library pairs would change (x0.10-x1.27); the largest is benthivores -> predators 0.026 -> 0.0026.
5. **The split does not remove the saturation.** Bed -> epifauna x 46, every fish -> benthivore pair x 3-3.5. It cuts who has access, not how fast a saturated pair eats.

## 2. Recommended fields (G1; ranges in params_split_223.md 2.1)
| Field | bed (whiting) | water column |
|---|---|---|
| B* / start | 0.446 t/km2 / 1605 t | 0.754 / 2715 t |
| r_max; M1 /yr | 0.49; 0.25 | 0.374; 0.261 |
| F bottom / pelagic; bf100 / pf10 | 0.075 / 0.005; 0.001704 / 0.001372 | 0.213 / 0.029; 0.004836 / 0.007983 |
| regional K | 0.533 | 2.14 (1.05-11) |
| layer; ceiling MJ/t/tick | bottom (J); 24.11 (Bjornsson-anchored 31.3) | at_50m; 25.16 (32.7) |
| RM; Q10; feeding / movement cost; speed | 3.88 (4.2 if not re-read); 2.3; 2.0 / 2.0; 1.0 | 3.31; 2.3; 2.15 / 2.0; 1.0 |
| lean / max reserve / ref fill (prey quality) | 3900 / 2000 / 0.45 (4800) | 4240 / 2600 / 0.50 (5540) |
| u; starve_rate; interference; visibility_floor | 0.5; 0.0099; 0.1; 0.5 J | 0.5; 0.0078; 0.1; 0.7 J |
| Q/B; derived g (diagnostic) | 4.2; ~0.0049 (s* 0.677) | 2.83; ~0.0018 (s* 0.755) |

## 3. Pairs (G1; a per tonne of the whole group, layer; Type II; assimilation 0.8)
| Pair | M2 /yr | h | a | f | dark / thr |
|---|---|---|---|---|---|
| bed -> planktivores | 0.160 | 295.5 | 0.021 | 1 | 0.33 / -5 |
| bed -> zooplankton | 0.054 | 186.6 | 0.0018 (0.0023 with Staniland, J) | 1 | 0.78 / -5 |
| bed -> mobile_epifauna | 0.069 | 155.5 | 0.27 (0.69 x e 0.39) | 0.29 | none |
| bed -> benthivores | 0.097 | 186.6 | 0.21 | 0.05 | 0.64 / -5 |
| wc -> planktivores | 0.340 | 283.1 | 0.0117 | 1 | 0.39 / -5 |
| wc -> zooplankton | 0.076 | 178.8 | 0.0006 | 1 | 0.30 / -5 |
| wc -> benthivores | 0.083 | 178.8 | 0.053 | 0.24 | 0.48 / -5 |
| wc -> bed (new) | 0.157 | 190.8 | 0.0106 | 0.47 | 0.48 / -5 |
| benthivores -> bed | 0.345 | 183.3 | 0.0026 whole-group (library convention 0.026) | 1 | 0.10 / 0 |
| seals -> bed / wc | 0.0130 / 0.0098 | 22.6 / 26.0 | 0.05 / 0.05 J | 1 | none |
| porpoises -> bed / wc | 0.0190 / 0.0063 | 28.6 | 0.20 / 0.20 J | 1 | none |

Unchecked: wc -> epifauna (0.007; the point of the split), both -> Nephrops and infauna, bed -> wc, within each group (160), benthivores -> wc. Sums conserved on planktivores 0.50, zooplankton 0.13, benthivores 0.18 and on the predators as prey; the epifauna loses 0.007 (unresolved predation: its g -0.4 %, K table x0.996); the wc -> bed flow 0.070 t/km2/yr becomes booked.

## 4. Diagnostic (not used to set anything)
| Quantity | value |
|---|---|
| share of the ceiling today, bed | epifauna 0.98, planktivores 0.95 (0.88 at 35 % visible), zooplankton 0.76, benthivores 0.75 |
| share of the ceiling today, wc | planktivores 0.92 (0.79), benthivores 0.78, zooplankton 0.51, whiting 0.30 |
| bed on epifauna: demand at the ceiling vs room | 4.18 t/km2/yr (pooled 11.9) vs the predators' room 0.287 at B* (0.53 at K/2); eat share 0.3 -> 4.4 x the room |
| wc on planktivores | demand at the ceiling 3.89 vs pooled budget 1.65, gross production 3.76 |
| budget gate cost/C vs best pair | bed 0.41 vs 0.98; wc 0.35 vs 0.92 (pays without bed pairs) |
| benthivore access sum a f B_pred (km2/tick) | fix77 library 0.0324; pooled layer 0.0200; split layer 0.0143 (x0.44 of fix77) |
| M2 on benthivores per unit eat share (/yr) | fix77 library 5.7; pooled layer 5.3; split layer 4.8 (saturated: -15 % only) |

fix77's realised predator M2 on benthivores 0.48-0.65 /yr (223.25, context only) corresponds to an effective eat x hunger of ~0.1 on that pair in either structure: the split removes the bed food from 63 % of the predator biomass and makes the planktivores the water-column group's best pair, but the policy's eat share remains the limit on benthivore predation.

## 5. Open judgements for the user (ranked)
1. Benthivores -> whiting 0.345 /yr (cod share 31 % long-run, Q/B 4.46): whiting M 0.78 vs SMS 0.685; bed g 0.0049 (0.0042 at the old 0.23, 0.0033 at SMS's 0.05).
2. Apply the whole-group attack-rate convention library-wide (six pairs, x0.10-x1.27; layer vs scored live on the zooplankton pairs)?
3. Ceiling anchor FB4 vs Bjornsson (x1.30 on every predator ceiling; bed g 0.0049 -> 0.0035).
4. Saturation stays: no structural bound on the bed group's epifauna take or either group's benthivore take.
5. Bed layer `bottom` and RM re-read (3.88 vs 4.2), together with the benthivores' 3.6 vs 3.32.
6. Saithe F (wc K 1.05-11). 7. Whiting cannibalism 0.142. 8. G1 vs G2. 9. Mackerel undersampling. 10. Mammal splits, floors, interference, layer threshold 0.1.

## 6. User decisions (2026-10-27)
G1; benthivores -> whiting 0.345 /yr; whole-group attack-rate convention library-wide (all six pairs); Bjornsson anchor (x1.30 on both predator ceilings); bed group on `bottom` with RM re-read (3.88), benthivores' RM re-read in the same step (3.6 -> 3.32); water-column K 2.14 (central saithe F). Build the split.

Downstream (not done): FG set +1 DM (all checkpoints invalid), library rows/columns/observes/impacts, manifest layers / DMs / spawn refs, epifauna g and K refit, i18n names and unique viewer codes, frozen_years AB codes, api.py, compare_scenarios, ~15 tests that pin numbers or FG lists; full list with line numbers in params_split_223.md section 7.
