# Evidence: benthivorous_fish, the planktivore term against the bed terms (2026-11-06)

Research only. Nothing in the library, manifest, code, tests or resume was changed.

**Tags.** [F] stated in a source or measured in a data set. [D] derived here. [J] judgement. [S] secondary or snippet.

**Files.**
- Script: `scripts/pout_vs_bed.py`; its stdout is `tables/pout_vs_bed_stdout.txt`. Section letters A-G below refer to that output.
- Gate run: `tables/budget_gate_stdout.txt`.
- Library: sha256 prefix 4ca160a1e9be.
- Model output is not used anywhere. The gate is library arithmetic, not a rollout.

## 1. The gap in lambda units (section A) [D]

lambda_j = a_j B_vis,j per tick, with B and B_vis as the gate prints them.

| Pair | lambda | Gate share |
|---|---|---|
| infauna | 0.674 | 0.629 |
| epifauna | 0.351 | 0.328 |
| Nephrops | 0.040 | 0.038 |
| planktivores | 0.0064 | 0.006 |
| whiting | 0.0001 | 0.000 |

- The bed total is lambda_bed 1.066. The shared denominator is X_bed = sum h lambda = 177.5, so the bed fills a 6 h ceiling in about 2 minutes of foraging.
- Planktivore shares and the lambda_plk they need:

| Planktivore share | Basis | lambda_plk needed | Times the current term |
|---|---|---|---|
| 0.075 | booked | 0.086 | x13.5 |
| 0.102 | in-area evidence C | 0.121 | x19 |
| 0.123 | ICES deep | 0.150 | x23 |

## 2. The pout term rebuilt per species and per member (section B)

The encounter code (`Attackrater ur mötesteori/scripts/benth_encounter.py`) runs unchanged at Kd 0.25. The rebuild changes its structure in four ways:
- Members are cod >= 20 cm (group share 0.310) at capability 1, and haddock >= 20 cm (0.353) at c_h 0.65 (0.3-1) [D/J, `Bottenfiskens sillbyte`].
- Each species has its own band share F (day / night) [J; herring 0.16 / 0.02 from the reconciliation].
- Each species has its own window, per member [D, `window_by_species_stdout.txt`]. For pout: cod 0.61, haddock 0.35. For herring: cod 0.30, haddock 0.075.
- Light, R and v are those of the code.

| Species | B t/km2 | annual g x F | lambda (per whole benthivore tonne) | of which cod / haddock |
|---|---|---|---|---|
| herring | 2.62 | 0.039 | 0.0041 | 0.0034 / 0.0007 |
| Norway pout | 0.25 | 0.184 | 0.0048 | 0.0032 / 0.0016 |
| blue whiting | 0.35 | 0.078 | 0.0002 | |
| sprat | 0.10 | 0.024 | 0.0003 | |
| **sum** | | | **0.0093** | engine 0.0064 |

- **Species resolution factor: x1.46 [D].** The library value a 0.0074 x f_eff 0.259 multiplies two pool means. One is the pool's overlap sum s_j F_j. The other is the pool-biomass window sum s_j W_j. But F and W are positively correlated: pout is high on both, herring low on both. Haddock's pout window (0.35) is also far wider than its herring window (0.075).
  - `Bottenfiskens sillbyte` found x1.09-1.18 for the window alone. The rest comes from resolving the members.
  - This is bookkeeping, not new biology.
- **Pout is about half of the rebuilt planktivore term.** The field needs 83-99 % pout. With 90 % pout, the field shares require the pout term to be x16 (booked), x23 (0.102) or x28 (0.123) larger at pout 0.25 t/km2.

### Factor check against literature (section C; ratio of the pout term to its central value)

| Factor | Central | Range tested | Effect | Evidence |
|---|---|---|---|---|
| Pout band share, day / night | 0.7 / 0.4 | 0.4/0.1 - 0.9/0.7 | x0.54 - x1.33 | Albert 1994 [A/S]: pout "generally 5-10 m above the sea bed, but somewhat closer by day", deeper with age. Aglen 1996 [S]: pout trawl and acoustics agree best at the 4 m headline height. Kaartvedt et al. 1996 [F, via encounter note]: in the benthic boundary zone by day. **No evidence for a larger share.** |
| Capture success p | 0.3 | 0.1-0.6 | x0.33 - x2.0 | Juanes, Buckel & Scharf 2002 pp. 271-272 [F]: "Usually about half of piscivore attacks on average are successful"; success falls with the prey/predator length ratio (pout/cod ~0.3) -> p 0.5 is x1.67. No gadoid-specific value found [gap]. |
| Reaction distance R_sat | 2 L | 1.5-3 L | x0.65 - x2.0 | Lindholm et al. 1999 [F, via encounter note]: >= 1.5 L. Fall & Fiksen 2020: R 2 m. |
| Non-visual R_nv | 0.6 L | 0.3-1 L | x0.85 - x1.35 | No gadoid measurement [J] |
| Light profile | central | low K - high K | x0.33 - x1.62 | Røstad et al. 2016, Aksnes et al. 2009 [F, via encounter note] |
| Search speed v | 0.4 BL/s | 0.2-0.8 | x0.66 - x1.82 | Winger 2004 [F, via encounter note] |
| Band H | 10 m | 5-20 m | x0.5 - x2.0 | Stensholt & Stensholt 2004, Aglen et al. 1997 [F, via encounter note] |
| Haddock capability c_h | 0.65 | 0.3-1 | x0.82 - x1.18 | SMS haddock pout share ~0.5 of cod's at equal length [D, `Bottenfiskens sillbyte`] |
| Cod window ratio | 0.30 | 0.40 | x1.23 | Herring-specific p95 band |

- MC (independent log-uniform): 5/50/95 % = x0.08 / 0.54 / 3.17.
- The all-high corner is x55. It is not a plausible combination.
- **No single factor is wrong by a large multiple.** Each factor's own range spans <= x2 upwards.

### Vertical geometry and schools (structural variants)

- **Geometry pushes the term down, not up [D/J].**
  - Suppose cod search within 2 m of the bed (19-34 % of tag depths within 2 m of the daily maximum; Stensholt & Stensholt 2004 [F, via encounter note]) and pout sit 3-10 m up.
  - With R 0.4-1.4 m, at most 0-6 % of the pout layer is then within reach, against the band model's 0.7.
  - The in-area stomachs (pout 19-54 % of cod diet) show that cod do reach the pout layer. So the band model is a necessary assumption, not a generous one.
- **Schools lower a gut-limited cod's encounter [D].**
  - A 15 cm pout weighs ~25 g [J]. A 2 kg cod's ration is 0.0089 t/t/tick x 2 kg = 18 g per 6 h, i.e. 0.7 pout per tick.
  - So one school encounter yields at most ~1 prey.
  - Meanwhile the rate of meeting schools falls by (R + r_s)^2 / (n R^2) = 0.04-0.5.
  - This is consistent with the existing convention: schooling is carried by the prey's hide action and visibility floor (`Attackrater ur mötesteori/addendum_validation_behaviour.md`).

## 3. Member-resolved diet vs the pooled gate (section D) [D]

- Each member is saturated on its own gut. Members are consumption-weighted: haddock 0.42, cod 0.29, flatfish and others 0.29.
- Member bed a_eff comes from the encounter code's per-member a_enc x mode e:
  - infauna e: gadoid 0.216, flatfish 0.131 x width 0.80;
  - epifauna e: per member;
  - both scaled to the library a.
- Result: member-resolved planktivore share 0.0100 vs pooled 0.0093, **x1.08**. Pooling the group costs little here.
- Cod alone: 0.025 against field 0.18-0.54 (box-length weighted).

## 4. Bed side: absolute magnitude (section E)

**Exposure-note MC band** (its own 5-95 %; infauna 0.016-0.092, epifauna 0.05-0.53): lambda_bed 0.457-3.00. **The bed can carry at most x2.3 downwards** inside its stated uncertainty.

**Lab anchors** (independent of diets):
- Source: Persson et al. 2012, MEPS 456:245 [F, via `Exponering av bottenbyten` and `Mättnad hos bottenbyten`]. Type II a for ~18 cm cod on sand.
- Measured a: Crangon (buries) 1.8, Palaemon 21, Neomysis 57 cm2/s, i.e. 0.071 / 0.83 / 2.2 km2/t/tick at 55 g.
- Scaled to ~40 cm (per-tonne 2-D a ~ L^-1) [D/J]: 0.032 / 0.37 / 1.0.
- Library epifauna: 0.15 per accessible tonne (0.093 per total tonne).
- The library sits between the burying-shrimp anchor and the free-swimming ones. That is a factor of ~3-5 above Crangon, below Palaemon.
- Crabs (0.31 of A) have no lab anchor.
- Habitat structure cut the lab a 5-9x in Persson 2012 [F]. The box beds are mostly soft mud and sand, so this factor does not obviously apply.
- Infauna: no measured a exists (as `Mättnad hos bottenbyten` Q1 found). Holmes 1979's plaice rate is a lower bound only.
- **No independent datum puts the bed a 10x lower** [D].

**Field intake data do not constrain the bed a from above** [D, from `Mättnad hos bottenbyten`]:
- Flat total intake over 5-250x prey contrasts (Smith et al. 2013 [F]; Johnson et al. 2015 [F]) needs X_bed >~ 3-10.
- X_bed 177 (now), 76 (bed at MC 5 %) and 15 (a hypothetical x12 cut) all pass.

**The earlier "pass" no longer supports the bed a's** [D]:
- `Exponering av bottenbyten` T2 scored the a_eff diet as infauna 0.625 / epifauna 0.274 / planktivores 0.059 against field 0.54 / 0.31 / 0.12.
- That planktivore term was the pre-223.26 one: 0.019 per capable tonne with the whole pool visible.
- Since then membership (x0.377) and the window (x0.26) have cut it ~10x. Both are correct structure.
- So T2 now reads planktivores 0.006 and no longer tests the bed a's. It is not evidence that they are high, either.

**Double-count audit (e, refuge exposed share, accessible_fraction)** [D, cited lines]:

| Item | Finding | Effect on bed lambda if corrected |
|---|---|---|
| Infauna phi 0.2 vs e | phi is the biomass within reach (0-2 cm + Amphiura arm tips; `Refugium och utbyte i bentos/summary.md` table 2, "Benthivores' share of A 1.0 (the reach is already in phi)"). e's factors are states of animals already in that layer: siphon out 0.15, polychaete part at surface 0.25, Amphiura pre-retraction 0.5, tip yield 0.8 (`exposure_e_items.csv`). These are not a second depth cut. | none |
| Capture p | e is divided by a_enc's own p (`exposure_e_items.csv` column "e_relative_to_a_enc_p": t 0.216 -> e 0.43 at p 0.5) | none |
| Amphiura tip mass | Flagged in `Exponering av bottenbyten/summary.md`: tip 1-5 % of body vs 6 % inside phi | x0.9 on phi at most (bed DOWN 1-3 %) |
| Small shrimps | Explicitly "(time out x E x p_rel)/f" (`exposure_e_items.csv`) | none |
| Brachyurans E 0.9, Munida E 0.75 | Exposure counted again on top of f 0.6 / 0.5 (both from the refuge note) | bed UP <= ~10 % (wrong direction for the gap) |
| Nephrops | f 0.02 = emergence 0.07 x size 0.29; e 1.0 | none |
| Across pairs | Only infauna uses the refuge exposed share, only epifauna and Nephrops use accessible_fraction | none |

**No double count lowers the bed by more than a few %.**

## 5. Norway pout biomass (section F) — the user's call

| Convention | Pout t/km2 | Pool | Pout share of pool | Booked benthivore take as pout M2 /yr | + whiting (0.07 booked, 61 % pout) |
|---|---|---|---|---|---|
| raw GOV q = 1 / acoustic (library) | 0.25 | 3.33 | 0.075 | 1.60-1.91 | 0.57 |
| Walker q, Q1 | 0.58 | 3.66 | 0.159 | 0.69-0.82 | 0.25 |
| stock TSB 2024-25 / 1-2e5 km2 | 0.45 (0.45-0.9) | 3.53 | 0.13 | 0.89-1.06 | 0.32 |
| **Walker q, Q1/Q3 mean** | **1.64** | 4.71 | 0.35 | 0.24-0.29 | 0.09 |
| stock TSB 2016-24 / 1-2e5 km2 | 2.0 (1.4-2.8) | 5.08 | 0.39 | 0.20-0.24 | 0.07 |
| Walker q, Q3 (0-group pulse) | 2.70 | 5.78 | 0.47 | 0.15-0.18 | 0.05 |

Sources:
- Walker et al. 2017 q-at-length via `Torskfiskarnas artsammansättning/evidence_B_catchability.md` [S]: 0.066 (<= 11 cm), 0.19 (>= 12 cm).
- The q-corrected kg/h: `Årsungarnas andel/tables/stage_length_biomass.csv` [D] (Q1 83.5, Q3 390.9 kg/h x 0.0069).
- TSB: WGNSSK 2026 via `evidence_pout_bluewhiting.md` [F]. The habitat area is [J].
- SMS 2016-22 [D, same note]: pout M2 1.89 in total, of which benthivore-group predators 0.29, and M1 0.2.

What the table shows:
- **Mortality check [D]:** the booked benthivore take matches SMS's benthivore M2 (0.29) at pout **1.4-1.7 t/km2**. Caveat: SMS is stock-wide, and the box's cod:pout ratio need not equal the stock's.
- At 0.25 t/km2 the benthivores alone would put the stock's whole M2 on pout.
- **The acoustic 0.2-0.3 is not independent of the dead zone.** Pout sits 4-10 m above the bed (Albert; Aglen), partly in or just above the dead zone, and a dead-zone correction did not improve the fit (Aglen 1996 [S]).
- **Mixed convention.** Herring in the pool is acoustic, so it is absolute. Pout is GOV at q = 1, a relative index that under-catches (q 0.07-0.19). The pool's pout:herring ratio is therefore biased low by roughly 1/q, independently of any a.
- **Knock-ons if pout B changes:**
  - Planktivore start biomass and range.
  - Every predator's M2 on planktivores, re-derived as flow / B_pool. Benthivores: 0.483 t/km2/yr / B_pool, giving 0.103 at 4.71.
  - The planktivores' derived g, which pays the booked M2 sum, and their regional K 4.8.
  - The pool J/B (pout J share 0.73), and therefore every stage_access f_eff on the pool.
  - The benthivores stay at GOV q = 1. Walker's cod q is ~0.7-1.4 by age, so the predator:prey ratio is distorted by less than x1.5 [S/J].

## 6. Scenarios (section G) [D]

Gate formula: shares of a x B_vis. The bed is unchanged unless stated. Pout B enters linearly through the species-resolved term, and stage_access is kept, so J/B rises with pout.

| Scenario | infauna | epifauna | Nephrops | planktivores | ratio to booked | gate | sat plk | Engine a equivalent (stage_access [0.52, 0.10] kept) |
|---|---|---|---|---|---|---|---|---|
| current library | 0.629 | 0.328 | 0.038 | 0.006 | 0.08 | OFF | 0.60 | 0.0074 |
| species-resolved, pout 0.25 | 0.627 | 0.327 | 0.038 | 0.009 | 0.12 | OFF | 0.69 | 0.0107 |
| + pout 0.58 | 0.623 | 0.325 | 0.037 | 0.014 | 0.19 | OFF | 0.79 | 0.0155 |
| **+ pout 1.64** | **0.612** | **0.319** | **0.037** | **0.032** | **0.43** | **ok** | 0.89 | **0.0248** |
| + pout 2.70 | 0.601 | 0.313 | 0.036 | 0.049 | 0.65 | ok | 0.93 | 0.0298 |
| pout 1.64, p 0.5 | 0.599 | 0.312 | 0.036 | 0.052 | 0.69 | ok | 0.93 | |
| pout 1.64, bed at MC 5 % | 0.556 | 0.290 | 0.082 | 0.072 | 0.96 | ok | 0.89 | |

- Field: booked planktivore share 0.075; in-area 0.08-0.17.
- **Saturation:** infauna and epifauna stay at 0.99 (0.97 at the bed's MC 5 %), Nephrops at 0.92. The planktivore pair moves towards saturation (0.60 -> 0.89-0.93). No pair leaves saturation.
- **Growth budget:** the derived g uses the ceilings, not a (242.2), so no a change moves any g. Only the pout re-booking (M2 = flow / B) does.

## 7. Prey ranking per gut time (Charnov zero-one rule) [D, library values]

(assim - sda) E / h, in MJ/t/tick:

| Prey | Value |
|---|---|
| planktivores (E as eaten 4000 + 0.5 x 6250 = 7125) | **20.9** |
| infauna | 13.9 |
| epifauna | 10.2 |
| Nephrops | 8.8 |

- **Planktivores rank first.** A rate maximiser always attacks them, so a corrected planktivore lambda will be realised by a trained policy.
- Epifauna and Nephrops rank below infauna. They would be dropped while the infauna pair's x = h lambda exceeds 2.7 and 1.7 respectively; it is 75 now.
- So with saturated infauna, the booked epifauna share 0.318 is not the expected learned diet. This matches fix86's 0.999 infauna (242.3, a diagnostic, not used here).
- It is a separate open question, about either the bed lambda or the per-prey ceilings (armour 1.75 / 2.0). It is not the planktivore gap.

## 8. Not obtained

- Albert 1994 and Aglen 1996 full texts (abstract and snippet only).
- Any gadoid capture success per attack on fish.
- Any field or lab encounter rate of gadoids on infauna.
- A box-scale pout acoustic estimate with a dead-zone correction.
- Walker et al. 2017's supplementary q table.
