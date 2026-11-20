# The encounter-realisation factor per group: evidence trail (2026-11-20)

Main-agent fork. No literature search: every field number is held in an earlier note and cited there.
Arithmetic: `scripts/anchors.py` -> `tables/anchors_stdout.txt`. Library copies: `scripts/make_copy.py`
(`tables/make_copy_{a,b}.txt`); frozen sizing with `"Mättnad hos rovfisk"/model/scripts/fill_decomp_all.py`
-> `tables/fill_decomp_copy_{a,b}_s{0,1}.{txt,json}`; the policy-free gate on copy a, `scripts/gate_on_copy.py`
-> `tables/budget_gate_copy_a.txt`. The live library was not edited.

Tags: [F] full text (in the cited note), [A] abstract, [S] secondary / earlier note, [D] arithmetic here,
[J] judgement.

## E0. Rule

Factor = the a implied by a field whole-product measurement / the chain's a under the anchor's own
conditions (prey density, light, temperature, fish length, bed class). The chain is today's library
(sections 220-280). No model output, FIELD_QB or frozen-model potential sets anything. The x0.25-0.35
yardstick of "Mötesteorins överskott" is compared only at the end. That yardstick came from model X_eff
algebra, and three model-side "needed factors" disagree with each other (resume 294.4 vs 294.6: whiting 0.15
on X_eff, 0.114, ~0.03 per cell), because cell heterogeneity breaks the S = X/(1 + X) inversion.

## E1. Planktivores -> zooplankton (clearance-based; library 0.0122)

- **Today's chain** is the 243.3 rerun (`"Kolmulens biomassa"/zoo/planktivores_a_pout164_foraging_stdout.txt`
  [S]): size-free R (k_v 115 herring, 100 sprat), scan 0.5, foraging-fish overlap O 1, capture 0.6/0.7, light
  at 15 m, annual time mean.
  - Group 0.0122, by day 0.0236. Herring 0.928 of the group a, herring day a 0.040 per t herring.
  - With R proportional to length the group a would be 0.0096 (x0.78).
- **V1, Arrhenius & Hansson 1999** (Baltic YOY herring 4-8 cm, Jul-Nov, 0-30 m, daylight initial slope 46 L/g/h
  from the weight-specific consumption vs Z) [F via "Attackrater ur mötesteori" V1]. Size-matched chain on
  small copepods:
  - Size-free R (the library's convention): 437 / 864 / 2258 L/g/h at 8 / 6 / 4 cm -> **factor 0.053
    (0.020-0.105)**.
  - R proportional to L (20 cm anchor): 70-90 L/g/h -> **0.59 (0.51-0.66)**.
  - The datum cannot separate the two. It constrains R(L) and the factor jointly, and the note that chose
    size-free R read V1 as favouring R ~ L. Using V1 both to pick the R scaling and to set a factor would be
    circular [J].
  - Transfer to the box adults [J]:
    - size-free: the YOY factor carries over;
    - R ~ L: the chain is x0.78 at the box size structure, so the factor on the library a is 0.40-0.51.
  - Temperature [D on J]: the field values are at summer temperatures, and the engine multiplies the library a
    by q(T). The planktivore q is 1.07-1.36 from Jul to Nov, which takes the factor x0.75-0.85 lower: **R ~ L
    reading ~0.36 (0.30-0.43); size-free ~0.04 (0.015-0.09)**.
  - Source data: stomach evacuation vs measured plankton, Baltic. This does not share data with the box's Q/B.
- **V2, Kulke 2018** (Baltic field sprat 12 cm; half-saturation k = 291 L^-1.5 from a field functional response;
  lab bite maximum 1-2 /s): 0.5-1.0 m3/h per fish [F via the same note].
  - Chain at 12 cm on small prey: R 5.5 cm, v 0.6 BL/s, p 0.7, scan 0.5 -> 0.86 m3/h. **Factor 0.58-1.17;
    x summer q 0.45-1.0.**
  - Not independent: the chain's scan 0.5 was chosen against 1 with this datum (planktivores_encounter V2).
  - Sprat is 0.05 of the group a.
  - Kulke's own lab finding points to the missing step: at low prey density the video bite rate exceeds tank
    depletion by up to 4-7x, and "sprat cannot perform this behavior continuously" [F via the sprat note].

## E2. Krill pairs (stomach-based; D unmeasured)

- **Form, as in "Krillparen mot vintermagar"** [S]: a_field = R / (D (1 - h_eff R)) against the chain's a_eff
  = library a x bed-class depth access x m(t) x q(t). The krill ration R is from stomachs and GER at each site.
  - D is not measured at any site.
  - **Self-consistent reading:** D_lib = the library's own krill target in the class and month
    (5 t/km2 x quarterly season x depth weight). The factor is then on the product a x D, so it absorbs any
    error in the library's krill density or occupancy ("Krillens fläckighet": acoustic p_eff 0.77, stomach
    p 0.06-0.42).
  - Brackets: D 3 / 8 / 20 t/km2.
  - The factor scales as 1/D: x1.7 at D 3, x0.6 at D 8 and x0.25 at D 20, against D_lib 5.

**Whiting** (library 0.041, h 159.5, Jan a_eff 0.031 on 100-150 m beds) [D on F]:

| site, month | R %BW/d | factor at D_lib (range over R) |
|---|---|---|
| NE+NW North Sea Q1 1981, 25-29 cm | 0.45 (0.34-0.52, GER upper) | 0.0088 (0.0063-0.0105) |
| Skagerrak Q1 1981, 21 cm | 0.14 (0.11-0.16) | 0.0024 (0.0019-0.0028) |
| Skagerrak Apr 1991 | 0.70 (0.30 of ceiling) | 0.030 (0.023-0.038) |
| Oslofjord Nov (fjord) | 0.41 (0.24-0.60) | 0.0057 (0.0032-0.0091) |
| Oslofjord Mar (fjord) | 0.94 (0.61-1.41) | 0.028 (0.015-0.063) |
| Oslofjord Jun (fjord) | 0.29 (0.19-0.43) | 0.0095 (0.006-0.015) |

Geometric mean **0.0099** (shelf rows 0.0086); centrals 0.0024-0.030.

**Planktivores** (library 0.0214, h 57.8, Jan a_eff 0.0115): group Nov-Mar 0.24 %BW/d -> 0.011
(0.0044-0.022); pout North Sea 1980 0.0090; pout Oslofjord Mar 0.037; herring Oslofjord 0.0067. Geometric mean
**0.012** (0.007-0.037). These rows are winter only.
- The library chain is compared as it stands. Pout's chain speed is ~2.6x LOW ("Mötesteorins överskott"); the
  corrected chain is x1.72 on this pair, so its factor would be x0.58 of these values.

**Reconciliation with "Aptit eller fångst i krillskiktet"** (f 0.03-0.26 for whiting, central 0.15):
- That f multiplies the frozen-model yardstick X_eff 4.47, the S -> X inversion of the mean S. The per-cell
  whiting krill X in the frozen world is 14-33 ("Mättnad hos rovfisk"/model/tables/potential_season_*), and
  the chain's own Jan 100-150 m X = a_eff h_eff D = 0.031 x 155 x 5 = 24 [D].
- On the cell X the same field rations give f x 4.47 / 24 = **0.006-0.05**, which is consistent with E2's
  0.0024-0.030. The two notes do not contradict; the earlier f was on the wrong X [D].

**Data shared with the ration [J]:**
- The ICES stomach DB rows (1980, 1981, 1991) are among the stomachs behind the SMS whiting ration and the
  booked in-area krill diet share. The krill factor therefore shares information with whiting's FIELD_QB and
  with the booked krill M2.
- The Oslofjord rows do not.

## E3. Whiting on fish prey: Rindorf & Gislason 2005 (stomach series + trawl density)

- **Source** [F via "Attackrater ur mötesteori"/addendum_validation_behaviour.md]: North Sea, September 1988-89,
  five sites with beds of 44-137 m (four of them < 100 m), 20-35 cm whiting. The fit is Type II,
  P = K T / (q rho + T).
  - Clupeids: K 0.10-0.17 of Wmax, q rho 59-95 per haul.
  - Gadoids: K 0.063-0.12, q rho 965-2907.
  - Wmax 2.7-6.1 %BW/d (mean 3.3).
- **Half-saturation in t/km2 of vulnerable prey** = q rho w / (A c), with A 0.09 km2 (0.05-0.18) [J: gear not
  obtained], c clupeids 0.2 (0.05-1) [J], gadoids 0.5 (0.3-1) [J], and w 15 / 5 g.
- **The field a is the low-density slope** = plateau / half-saturation = K Wmax / half [D].
  - The earlier note converted the half-saturation through the ENGINE's h (a = 1/(half x h)), which mixes the
    field's low plateau with the model's ceiling. The slope is the field measurement of a. The plateau is a
    separate, per-prey ceiling, and is not converted here.
  - Library a is per t/km2 of accessible prey (stage_access); the vulnerable size classes are comparable
    (accessible share 0.079 vs vulnerable 0.11, [J]).
- **Whiting -> planktivores** (library 0.05913, x Sep q 1.02):
  - Field slope 0.0178 km2/t/tick.
  - Against Rindorf's bed mix in the library's bed-class table (0.8 < 100 m at x2.01, 0.2 at 100-150 m at x0.95):
    **factor 0.16**. Against the equal-cell a: 0.30.
  - Corner span 0.011-8.7, i.e. the gear and catchability judgements are not resolved.
- **Whiting -> benthivores** (library 0.2455): field slope 0.0040 -> **factor 0.012** (equal-cell 0.016), corners
  0.0013-0.27.
  - Weak: Rindorf's "gadoids" include juvenile whiting (cannibalism, unchecked in the engine) and pout (in the
    planktivore pool).
- **The other form.** Keep a (the half-saturation reading at today's h gives x1.28 on whiting -> planktivores)
  and put the field plateau (0.10-0.17 of Wmax) into a per-prey ceiling. Only the low-density slope separates
  the two forms, and its corners span three decades.
- **Data shared with the ration:** Rindorf's own 1988-89 sampling, not the ICES DB years [J]. It does not share
  data with the box Q/B.

## E4. Water column (open)

- No whole-product field measurement for saithe, hake or spurdog on any prey was found in the held notes.
- Fall & Fiksen 2020 [F, held in "Intag under födosök"/data/fall2020]: Barents Sea cod on capelin (PROXY
  predator and prey). Their encounter-plus-digestion models predict "high, but reasonable" gut fullness against
  observed stomachs, and cod are gut-limited "down to very low prey densities". This is qualitative and gives
  no ratio, so no factor.
- **Transfers [J], not applied:** whiting's fish slope (0.16-0.30) to the water column's fish pairs; the krill
  factor (~0.01) to water column -> krill (a 0.0107, night/day 0.026).

## E5. Mackerel (open)

- Mackerel -> zooplankton is the measured Macy lab clearance, not an encounter chain. It is excluded and would
  in any case be a lab, not field, whole product.
- Mackerel -> krill and -> planktivores have no field whole-product anchor in the held notes.
- **Transfer [J], not applied:** the planktivore krill factor to mackerel -> krill. Krill is 0.71 of mackerel's
  potential.

## E6. Benthivores

Excluded (resume 294.4: X 70-110 on saturated beds; a factor on a cannot reach them).

## E7. Shared or prey-specific?

| predator | krill (stomach, a x D) | fish (Rindorf slope) | zooplankton (clearance) |
|---|---|---|---|
| whiting | 0.0099 (0.0024-0.030) | 0.16 (0.30 equal-cell; corners 0.011-8.7) | - |
| planktivores | 0.012 (0.007-0.037) | - | 0.36 (R ~ L) / 0.04 (size-free) |

- The central factors differ by prey type by more than 10x within each predator.
- This contradicts the reading of "Mötesteorins överskott" Q2 (resume 294.4): "shared within each predator,
  krill residual <= ~1.5-2x". That reading rested on the diet gate, which compares SHARES at the booked level
  and is blind to a common factor by construction, and on the per-prey test, which is model-side.
- Two caveats keep the contrast from being a clean prey-specific verdict:
  1. The krill factor is on a x D. An over-stated library krill density, or occupancy, sits in it.
  2. The krill rows are mostly winter and Q1. Field whiting krill intake is highest in Q1 in the North Sea, but
     the planktivores' booked annual krill flow is ~3x their Nov-Mar field ration.

## E8. Frozen sizing on library copies (fix105 policies, seeds 0/1, one year; the placement was trained sated, so
the fill cycle is not judged, 294.6)

**Copy a** (factors applied): whiting -> krill x0.0099, -> planktivores x0.164, -> benthivores x0.012;
planktivores -> krill x0.0124, -> zooplankton x0.36. Every other pair is unchanged.

| group | uncapped (live) | realised (live) | fill (live) |
|---|---|---|---|
| planktivores | 0.269 / 0.250 (0.540 / 0.532) | 0.237 / 0.230 (0.261 / 0.262) | 0.719 / 0.702 (0.908) |
| whiting | 0.446 / 0.476 (0.735 / 0.751) | 0.373 / 0.376 (0.389 / 0.390) | 0.870 / 0.941 (0.983) |
| water column, mackerel, benthivores | unchanged within noise | | WC 0.99, mackerel 0.93, benthivores 0.98 |

- **Planktivores' monthly fill (s0):**

  | | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | copy a | 0.32 | 0.14 | 0.16 | 0.47 | 0.84 | 0.99 | 0.99 | 0.98 | 0.96 | 0.91 | 0.77 | 0.55 |
  | field | 0.50 | 0.36 | 0.26 | 0.20 | 0.32 | 0.81 | 0.94 | 0.96 | 0.88 | 0.88 | 0.76 | 0.68 |

  - Nov-Mar potential = realised (food-limited). Krill is 0.03 of the potential.
  - The phase is close and the winter low is deeper and earlier. Not a fit criterion.
- **Whiting:**
  - The potential is now 0.66 epifauna, an unanchored pair, and 0.22 krill.
  - Monthly fill (s0) has its minimum in Oct-Nov (0.75) and its maximum in Apr-Jun (0.95): the opposite of the
    field, the sated-placement artefact of 294.6.
  - Without epifauna the potential is 0.23, below the need of ~0.39.
- **Copy b** (planktivores -> zooplankton x0.042): planktivores uncapped 0.097, fill 0.27 / 0.25 (starvation onset
  0.079). The size-free reading of V1 does not leave a viable planktivore in the frozen world. This is a
  consequence, not used to choose.
- **Budget gate on copy a** (policy-free, start densities) puts four new pairs OFF in the diet gate:
  - planktivores -> krill: encounter share 0.018 vs booked 0.398;
  - whiting -> krill: 0.059 vs 0.557;
  - whiting -> epifauna: 0.674 vs 0.171;
  - whiting -> zooplankton: 0.071 vs 0.023.

  Prey side: planktivores -> krill implied M2 0.096 vs booked 2.07; whiting -> epifauna 0.30 vs 0.076.
  - Applying only the anchored factors makes the diets inconsistent with the booked field diets.
  - Either the unanchored pairs (whiting -> epifauna above all) carry an excess of the same order [J], or the
    krill anchor overstates the cut through D and its winter season.
