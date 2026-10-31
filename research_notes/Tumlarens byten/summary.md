# Porpoise prey pairs: what the attack rates rest on, and which are off (2026-10-31)

Research only. Nothing in the library, manifest, code or tests was changed. Tags: [F] field or stated in a source, [D] derived here, [J] judgement. Quotes and tables in `evidence.md`; scripts in `scripts/` (`then_now_densities.py`, `wc_window.py`, `clearance_test.py`), their stdout in `tables/`; downloads listed in `data/README.md` (NS-IBTS 1991-96 extract, ICES SAG herring, `data/lit/`). Written by the research subagent, saved by the main agent.

## Verdict

1. **No porpoise attack rate was ever derived; each one is the end of a chain of fits.**
   - Section 151 fitted a_herring 0.015 and a_gadoid 0.2 (ratio 13) on frozen policies to an Ecopath diet. The resume says: "a fit to diet data, not a mechanism".
   - Section 206 multiplied both by 1.5 because porpoises declined in frozen runs (model output).
   - Section 209 rescaled the gadoid value by "a x B invariance", giving benthivores 0.12 and predators 0.20. Planktivores 0.0225 was kept.
   - Sections 223.26 and 228 copied 0.2 onto whiting and onto the water-column rest.
   - There is no encounter geometry with a wrong factor in it, as there was for whiting. The fault is the section 151 ratio itself, carried forward unchanged.
   - Section 209 contradicts itself [D]. It derived per-tonne selectivities of 0.50 (benthivores) and 0.86 (predators) relative to planktivores, then set a values 5.3x and 8.9x the planktivore a.

2. **Water column -> predatory_fish is the largest error, and it comes from prey size.**
   - Porpoises swallow fish whole: mostly < 30 cm, limit ~40 cm (Santos & Pierce 2003 [F]).
   - Since section 228 the rest is saithe 0.80, spurdog, hake, anglerfish, ling, velvet belly and pollack. In the box, 4.9 % of saithe biomass is < 40 cm, spurdog 0.6 %, hake 30 % (long-run stratified GOV) [D].
   - Counting only taxa porpoises are known to eat gives f = 0.054 (saithe only 0.039). The range is 0.017 (< 30 cm) to 0.19 (unstratified 2001-20 box rectangles). The pair has no size factor today (f = 1). Concurrent search therefore gives it 0.31 of the diet against 0.073 booked.
   - **Coastal vs offshore [J, a headline].** Every stomach set comes from coastal bycatch or strandings.
     - The %W/%N ratio in Börjesson 2003 puts the eaten saithe at ~18 cm [D, rough].
     - Every saithe record with a size or age is juvenile (northern Norway). In Skagerrak/Kattegat stomachs saithe is not among the top six prey (Aarefjord via Bjørge 2003).
     - GOV holds only 0.1-0.4 % of Skagerrak saithe biomass below 20 cm.
     - So the stomach saithe are coastal 0- and 1-group fish, which the offshore 100-300 m box does not hold at that size.

3. **Benthivores and whiting are 3-15x too high relative to planktivores against every field diet.**
   - Clearance-ratio test (method of the whiting audit, anchor = planktivores). The field diets are the booked M2, Börjesson 2003 and Aarefjord 1995. They are tested both at today's box densities and at 1991-96 densities from NS-IBTS (GOV ratio x today's level; herring by GOV or by a WBSS/NSAS SSB mix).
   - Model ratio over field ratio:
     - benthivores 2.2-14.5 (booked 6.5);
     - whiting 1.7-11 (booked 5.7).
   - For planktivores, benthivores and whiting the 1990s prey field was within 0.7-1.8x of today's box densities, so the result does not depend on the decade.
   - One mechanism fits within the spread of the stomach data: **one porpoise clearance rate for all fish of this size, with differences only through the size windows** (equal clearance [J], windows [D]). Tested against the field [F]:
     - benthivores 0.4-2.7x the field ratios: passes;
     - whiting 0.2-1.25x: the field sits ~2x higher, but inside the Gadidae-split and Börjesson-Aarefjord spread.
   - Field-implied a, keeping a_plk 0.0225 (a test, not a third proposal): benthivores 0.024 (0.008-0.054), whiting 0.054 (0.018-0.12).

4. **The planktivore pair is the anchor; its scale is a fit and is not re-derived.**
   - It is the anchor because it is 0.65 of the diet, its size access is certainly 1, and its density is HERAS-measured.
   - Raising it ~4-5x instead of lowering the bed pairs would put the herring pair alone at 0.90 of the ceiling, ~12.6 %BM/day. The field ration is 5.3 %/day. Already at 0.0225 the herring pair pays 153 MJ/t/tick against an eat cost of 90.
   - So the field ration argues for lowering the demersal pairs, not raising planktivores. This is a scale check, not a derivation.

5. **Ransijn et al. 2021 was misread in section 209.**
   - Their a_i = alpha_i q_i^m (Eq. 7-9) absorbs GOV catchability to the power 1.5, and abundance was rescaled to a maximum of 100. They describe a_i as preference "in relation to an index of abundance and not to absolute estimates of biomass".
   - So "whiting 1.2x, cod 0.6x herring per unit availability" is not a per-tonne measurement in either direction. That includes reading it as support for the library's 8.5.
   - Not used as a test.

## Parameters: current vs proposed

| Pair / key | Current | Proposed (central, range) | Tag | Basis |
|---|---|---|---|---|
| porpoises -> predatory_fish `accessible_fraction` | none (1) | **0.054** (0.017-0.19; saithe only 0.039) | D (window) + J (hake counted; no Kattegat-Skagerrak stomach lists hake) | `tables/wc_window_stdout.txt`; < 40 cm on the box long-run length structure; sharks and anglerfish 0 |
| porpoises -> predatory_fish `max_intake_rate` | 0.2 | **0.0225** under reading (i); 0.2 can stay under reading (ii) | J (equal clearance) | See open point 1. Reading (ii) passes: model/field 0.56-1.22 |
| porpoises -> predatory_fish `predation_mortality` | 0.0077 | **0.0021** (0.0007-0.0077) under (i) | J | Re-booked wc share 0.02. The freed 0.053 goes pro rata (J) to plk 0.0134, ben 0.0067, whiting 0.0200 (Q/B 19.5, 12 t). Alternative: all of it to whiting (+0.0077 -> 0.027) |
| porpoises -> benthivorous_fish `max_intake_rate` | 0.12 | **0.0225** (0.008-0.054); effectively an upper bound | J (equal clearance) + D (stage access [1, 0.55] kept) + F (test) | Field 0.4-2.7x passes; field geomean 0.024. Depth flag: the box bed at 100-300 m is reached "occasionally" (Skagerrak maxima 132-152 m, Teilmann 2007, van Beest 2018; Fontaine 2007) -> x0.3-1 [J], not applied |
| porpoises -> whiting `max_intake_rate` | 0.2 | **0.0225** (0.0225-0.054; 0.045 a J option at the field centre) | J + D (stage access [1, 0.95] kept) + F | Field ~2x the null; same depth flag |
| porpoises -> planktivorous_fish | a 0.0225, M2 0.0127, [1,1], floor 0.95 | a unchanged (anchor; **flagged as a 151/206 fit**); M2 0.0134 if re-booked | F (diet) / fit (a) | — |
| visibility floors 0.95 / 0.75 / 0.75 / 0.9, h 28.6, assimilation | as is | unchanged | J | The floor difference is at most 1.27x, small against 5-15x |

No observation change: `accessible_fraction` leaves observations alone and M2 only moves derived growth slightly, so checkpoints stay valid.

**The gate passes only with the re-booking.** With the proposal and the current booked 0.073, the wc row is OFF (ratio 0.10). Against the re-booked 0.02 it is 0.35, with both shares < 0.05.

## Concurrent-search diet shares, all prey visible, today's box densities (`tables/clearance_test_stdout.txt`)

| | plk | ben | whiting | wc |
|---|---|---|---|---|
| M0 library now | 0.189 | 0.285 | 0.215 | 0.311 |
| M1 library a, wc f 0.054 | 0.268 | 0.404 | 0.305 | 0.024 |
| **Proposed** (a 0.0225 on all, windows) | **0.704** | **0.199** | **0.090** | **0.007** |
| M3 = proposed with whiting 0.045 | 0.646 | 0.183 | 0.165 | 0.006 |
| V1 booked | 0.648 | 0.150 | 0.130 | 0.073 |
| re-booked (wc 0.02) | 0.685 | 0.158 | 0.137 | 0.020 |
| V2 Börjesson 2003 (Gadidae split 0.5) | 0.753 | 0.097 | 0.096 | 0.054 |
| V3 Aarefjord 1995 | 0.591 | 0.221 | 0.188 | 0 |

The box diet predicted from field clearance x today's B:
- Börjesson: 0.60-0.70 / 0.09-0.17 / 0.10-0.19 / 0.07-0.095.
- Aarefjord: 0.43-0.53 / 0.24-0.29 / 0.23-0.28 / 0.

So the booked plk/ben/whiting split is consistent with the field; only the wc share depends on reading (i) or (ii).

Check (not a criterion). Single-pair return at the grid mean, proposal: herring 153, benthivores 50, whiting 30 (50 at 0.045) MJ/t/tick, against an eat cost of 90. Break-even visible density: benthivores 5.4, whiting 2.9 t/km2. With the engine's exclusive per-pair eat allocation, herring becomes the only pair that pays, and whiting will leave the trained diet. Trained runs already eat 63-96 % planktivores.

## Open points (user's call)

1. **Water-column reading.**
   - (i) The stomach saithe are coastal ~18 cm juveniles: a 0.0225, M2 ~0.002 (recommended).
   - (ii) Porpoises eat GOV-sampled small saithe at a 7-16x per-tonne preference over herring: a 0.2 with f 0.054 passes, M2 0.0077 stands.
   - What decides it is the size of the saithe in Börjesson's full text. It could not be reached (Wiley/RG 403, closed on Unpaywall).
2. **Depth factor** for the two bed pairs in a 100-300 m box: no number exists [J 0.3-1].
3. **Planktivore a scale** (0.0225) is a model fit. The field ration suggests it is not too low; there is no biological derivation.
4. **Structure.** With the engine's per-pair exclusive allocation, the field's 25-40 % demersal share cannot emerge when those pairs pay below cost. A shared-handling multi-prey response (section 232) is the structural option.
5. **Seals -> predatory_fish** has the same window defect: harbour seals take 7-35 cm prey from a group that is ~95 % >= 40 cm. Flagged, not solved.
6. **Library comment on porpoises -> whiting** contains a literal `{NOTE}` placeholder.
7. **Data age.** The only local diets are from 1985-96. No recent or offshore Skagerrak porpoise stomachs exist. The Gadidae spp. split (cod vs whiting) moves the benthivore and whiting ratios by ~1.3x each.
