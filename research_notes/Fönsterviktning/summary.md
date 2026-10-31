# Length weighting of every stage_access window ("aw") and the normalisation fix: one library-wide pass (2026-10-31)

Research only. The library, manifest, code and tests were not changed. No model output set any value. Tags: [F] field value or stated in a source, [D] derived here from data, [J] judgement. Written by the research subagent, saved by the main agent.

**Files:**
- `scripts/aw_stage_access.py` - run `python3 -I scripts/aw_stage_access.py <research_notes dir> <scratch dir>`. It imports `Årsungarnas andel/scripts/stage_access.py` unchanged, and the section 220 encounter (`Rovfiskens uppdelning/scripts/split_encounter_223.py` and `Attackrater ur mötesteori/scripts/pred_encounter_a.py`) as exec'd heads.
- `scripts/blue_whiting_lengths.py` - box GOV lengths of blue whiting, same hauls, polygon and strata as `length_structure.py`.
- Output: `tables/aw_stage_access_stdout.txt` (every step, by quarter and per member), `tables/blue_whiting_lengths.csv`, `tables/blue_whiting_lengths_stdout.txt`.

**Reproduction check:** the library method reproduces all 12 current values to rounding, including mackerel's field run [0.078, 0.0037] and section 232's [0.157, 0.0102].

## Verdict

**1. aw applies on 6 of the 12 pairs. On these, f_eff falls to x0.61-0.69 (water column -> benthivores x0.89, whiting -> benthivores x1.52, see below).**
- Pairs: water column -> benthivores, planktivores and whiting; mackerel -> planktivores; whiting -> planktivores and benthivores.
- `pred_encounter_a.py` keeps the detection radius R fixed in metres, speed v proportional to L, and N per tonne proportional to L^-3.
- a(L) per 5-cm class computed from the encounter code itself. Fitted exponents are -2.1 to -2.3, and -2.85 for anglerfish (an ambusher) [D]. So L^-2 holds here.
- Two pairs move differently:
  - Water column -> benthivores lands at x0.89, because the normalisation pushes the other way (point 3).
  - Whiting -> benthivores lands at x1.52, because the normalisation creates an f_A where there was none.

**2. aw does not reach the cod pairs (benthivores -> planktivores, benthivores -> whiting).**
- `benth_encounter.py` section 3 has R_sat = 2 L and v = 0.4 L/s. Swept volume per fish then goes as L^3, and per tonne as L^(3-3.14).
- Fitted exponent: cod -0.36, haddock -0.46 [D]. So aw is close to biomass weighting there.
- Effect: x0.92 on f_J, x0.87 on f_A.
- If L^-2 is wrongly assumed, benthivores -> planktivores would fall to [0.33, 0.042] and benthivores -> whiting to [0.18, 0.011].

**3. The normalisation fix changes sign on the benthivore-prey pairs. The Vattenpelarens audit's 0.016 / 0.0070 for water column -> benthivores f_A is superseded.**
- About 21 % of the benthivore tonnes are long rough dab 12.6, dab 4.5, starry ray 1.5, gurnard 0.9 and lemon sole 0.9 (BEN_W weights).
- These species have no stage rows. The library skipped them, and the audit counted them as outside every window.
- Their measured box lengths (`Fiskgrupper i rutan/benthivores/tables/length_biomass_longrun.csv`; long rough dab p50 18.5 cm, dab 20.5) put them inside the windows of spurdog (0.34 x 70 cm = 24 cm), hake and anglerfish.
- Water column -> benthivores f_A: 0.027 -> 0.057 (normalised), -> 0.038 (aw). Spurdog alone carries 0.024 of the 0.038.
- Counting these species as outside reproduces the audit's 0.0065, which is the bridge between the two results.
- The same species lift seals -> benthivores f_A 0.34 -> 0.50 and porpoises -> benthivores 0.55 -> 0.63.
- Blue whiting (21-32 cm in Q3) now sits in every planktivore f_A denominator. It is all adult under the box J/B definition.

**4. Mammals: aw does not apply.** Their a is a judgement or flow value, and the window is a fixed prey-length range. So there is no predator-length axis. Only the normalisation moves them.

## Table

- Steps, applied one at a time:
  - **thr**: drop predator lengths the a does not count (capability score < 0.1). This is the 232 fix.
  - **norm**: weight species by stage biomass and normalise over all tonnes of the prey group.
  - **aw**: weight the window by biomass(L) x a(L).
  - **ENC**: encounter species weights with the current overlap table (benthivores -> planktivores only).
- Range: a(L) x L^+-0.5. On benthivore-prey pairs the f_A range also spans "unkeyed species counted 0 inside" up to "full lengths" (a convention bound, not a biological one).
- f_eff = J/B f_J + (1 - J/B) f_A, at the box J/B: planktivores 0.38, benthivores 0.15, whiting 0.165.

| Pair | Library [f_J, f_A] | aw-corrected (rounded) | Range f_J; f_A | f_eff before -> after | Ratio | Steps that bite | Tag |
|---|---|---|---|---|---|---|---|
| water column -> benthivores | [0.43, 0.027] | **[0.30, 0.038]** | 0.27-0.33; 0.0065-0.042 | 0.087 -> 0.078 (0.051-0.086) | 0.89 | norm (f_A x2.1, flatfish), aw (x0.70 / x0.67) | D |
| water column -> planktivores | [0.69, 0.27] | **[0.53, 0.14]** | 0.49-0.57; 0.118-0.156 | 0.430 -> 0.286 (0.259-0.314) | 0.67 | norm small (f_A 0.27 -> 0.24), aw x0.76 / x0.57 | D |
| water column -> whiting | [0.57, 0.15] | **[0.41, 0.080]** | 0.37-0.45; 0.068-0.092 | 0.219 -> 0.134 (0.118-0.151) | 0.61 | aw only | D |
| mackerel -> planktivores | [0.078, 0.0037] | **[0.052, 0.0035]** | 0.047-0.058; 0.0029-0.0041 | 0.032 -> 0.022 (0.020-0.025) | 0.69 | norm (+), aw x0.64 | D |
| whiting -> planktivores | [0.16, 0.010] (232) | **[0.10, 0.0073]** | 0.091-0.113; 0.0062-0.0085 | 0.067 -> 0.043 (0.038-0.048) | 0.64 | thr already in, norm (+), aw x0.62 | D |
| whiting -> benthivores | [0.049, 0] | **[0.038, 0.0064]** | 0.034-0.043; 0-0.0074 | 0.0073 -> 0.011 (0.006-0.013) | 1.52 | thr (f_J +35 %, same fix as 232), norm (f_A from 0), aw x0.58 | D, J |
| benthivores -> planktivores | [0.49, 0.14] | **[0.52, 0.10]** | 0.46-0.59; 0.080-0.127 | 0.273 -> 0.263 (0.225-0.302) | 0.96 | norm (f_A 0.136 -> 0.117), ENC (f_J 0.50 -> 0.57), aw small | D |
| benthivores -> whiting | [0.38, 0.042] | **[0.34, 0.036]** | 0.28-0.40; 0.026-0.048 | 0.098 -> 0.086 (0.069-0.105) | 0.88 | aw small | D |
| seals -> benthivores | [1.0, 0.34] | **[1.0, 0.50]** | -; 0.23-0.50 | 0.439 -> 0.576 (0.34-0.58) | 1.31 | norm only; aw n/a | D |
| seals -> whiting | [1.0, 0.84] | [1.0, 0.84], unchanged | - | 0.866 -> 0.865 | 1.00 | none (one species) | D |
| porpoises -> benthivores | [1.0, 0.55] | **[1.0, 0.63]** | -; 0.35-0.63 | 0.618 -> 0.687 (0.45-0.69) | 1.11 | norm only; aw n/a | D |
| porpoises -> whiting | [1.0, 0.95] | [1.0, 0.95], unchanged | - | 0.958 -> 0.962 | 1.00 | none | D |

**Sensitivities on the cod pairs** (central assumes a membership-based a: cod >= 20 cm at 1):

| | benthivores -> planktivores | benthivores -> whiting |
|---|---|---|
| Window weighted by the library a's phi (gape scores) | [0.61, 0.126], f_eff 0.311 | [0.54, 0.063], f_eff 0.141 |
| Haddock as a member, non-gape capability c_h 0.3 / 0.5 / 1 (a-share 0.25 / 0.36 / 0.53) | [0.45, 0.079] / [0.42, 0.069] / [0.38, 0.054], f_eff 0.22 / 0.20 / 0.18 | - |
| L^-2 assumed | [0.33, 0.042] | [0.18, 0.011] |

**Implied juvenile share of the take at the box J/B** (before -> after, against the references in `parameters_B.md` section 4):

| Pair | Before -> after | Reference |
|---|---|---|
| water column -> benthivores | 0.74 -> 0.58 | SMS saithe on haddock 0.77 |
| water column -> planktivores | 0.61 -> 0.71 | SMS saithe on herring 0.15 (nursery caveat) |
| mackerel -> planktivores | 0.93 -> 0.90 | SMS 1.00 |
| whiting -> planktivores | 0.91 -> 0.89 | - |
| whiting -> benthivores | 1.00 -> 0.51 | SMS whiting on haddock 1.00; not like-for-like, because the new f_A is adult flatfish |
| benthivores -> planktivores | 0.68 -> 0.76 | SMS cod on herring 0.19 |
| seals -> benthivores | 0.34 -> 0.26 | grey seal 0.27-0.30; Kappa et al. 2026 harbour seals 0.74-0.86 |

## Method and assumptions

- **Kept from the original derivation:**
  - Window ratios (saithe 0.30, spurdog 0.34, hake 0.50, anglerfish 0.57, mackerel 0.33, horse mackerel 0.31, whiting 0.39, cod 0.30), velvet belly 11 cm, seals 7-35 cm, porpoises 0-40 cm.
  - Prey stage tables, box predator length mixes, diet-identity sets, member a-shares, and the 4-quarter mean.
- **Water column -> planktivores member shares:** the members' contributions to the current a 0.00559 (saithe 0.848, hake 0.061, spurdog 0.061, velvet belly 0.030). The library value used the older renormalised set. The difference in the library method is 0.003.
- **a(L) per length class:** the encounter's own stage function at central factors, with N per tonne = 1e6 / w(L) from the length-weight table.
  - Layer membership m(L) is 1 if the capability score >= 0.1. This reproduces 232's whiting >= 20 cm automatically.
  - Check: saithe a(43 cm) / a(70 cm) = 2.80, against 2.55 in the per-stage table.
- **Benthivores -> planktivores:**
  - ENC uses the CURRENT overlaps (Bottenfiskens F_CUR), not the revised herring overlap.
  - Blue whiting comes in through the data, on f_A only.
  - Bottenfiskens' [0.45, 0.14] applied the x0.86 blue-whiting dilution to f_J too. That is wrong: blue whiting carries no J under the box stage definition.
- **Blue whiting lengths:** box Q3 p5-p90 21.5-31.5 cm, 195 of 1371 hauls, raw GOV [D]. The Q3 shape is used for Q2 and Q4 [J]. The Q1 pool weight is 0.
- **Unkeyed benthivores:** raw GOV long-run shape pooled over Q1+Q3, the same in every quarter [D/J]. They are "all adult" by the engine's own J/B bookkeeping, not by biology: 12-15 cm long rough dab are age 1-2.

## Open points

1. **Cod pairs, phi on the a side.** The a of benthivores -> whiting carries phi 0.10 (cod 0 / 0.15 / 0.4 by stage, gape-based per `params_benthivorous_fish.md`). That is the same gape double count Bottenfiskens found on the planktivore pair. The central values presuppose that the a moves to membership. That a-side change is outside this task.
2. **Hake's diet identity is "all", so its window reaches flatfish.** Its only in-area diet has 0.4 % benthivores. The original method is kept [J]. This drives part of water column -> benthivores f_A.
3. **Whiting -> benthivores x1.5 is the net of two opposite moves.** aw cuts f_J; adult long rough dab and dab below 15 cm create an f_A. About half of the new f_eff rests on that bookkeeping.
4. **Seals -> benthivores** moves further from Kappa's juvenile share (0.26 against 0.74-0.86). That share is a validation target, not an input.
5. **ENC on the other planktivore pairs was not computed.** In the water-column encounter form, pout's a per unit biomass is about herring's, so encounter weights are about pool weights; 232 found the same for whiting. It was applied to benthivores -> planktivores only.
6. **What the range leaves out.** It covers a(L) and the unkeyed-species convention only. Window ratios and member shares were fixed and not re-ranged.
7. **Checks.**
   - Seals and porpoises -> planktivores rightly carry no stage_access (window shares 0.994-1.0, blue whiting included).
   - Mackerel -> benthivores is not a checked pair.
   - Seabirds -> planktivores is muted and has no window.
   - No other stage_access pair exists in the library.
