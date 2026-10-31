# Whiting on planktivores: layer convention and box-specific booked M2 (step 2, 2026-10-31)

Research only. Nothing in the library, manifest, code or tests was changed. Tags: [F] field value or stated in a source; [D] derived here; [J] judgement. "Step 1" means `summary.md` / `evidence.md` in this folder. Written by the research subagent, saved by the main agent.

Note added by the main agent: step 2 corrects step 1's reading of "small whiting are pelagic" (it refers to 0-group fish of 1.5-5 cm). Step 1's epifauna value 0.008 (applied in section 232) assumed juveniles at 30 m; with both stages in the 10 m bed band step 1's own table gives 0.0116.

## Verdict

**Q1, convention: change whiting -> planktivorous_fish to the overlap ("band") convention.** That means the prey concentration the whiting meets at its own height above the bed, with the reactive distance set by the light at the bed.

The field puts box whiting at the bed and does not put them inside the herring layer:
- **Danish HERAS 2017-19.** Whiting were caught only in day bottom hauls. Night surface hauls in the box rectangles took 0 whiting alongside 400-1400 kg of herring.
- **No diel catch signal** for whiting > 20 cm (Adlerstein & Ehrich 2000; Rindorf 2003).
- **"Small whiting are pelagic"** refers only to 0-group fish of 1.5-5 cm (Bromley et al. 1997). Step 1 used it for 10-30 cm fish, which was wrong.
- **Norway pout share the whiting's layer by day:** only in day bottom hauls, 5-10 m above the bed.
- **Herring come down to the bed by day, but mainly over beds < 80 m.** Over the box's 100-300 m beds their share in the bottom 10 m is only 0.04-0.10.

What the overlap convention gives [D] (`tables/step2_whiting_overlap_stdout.txt`):
- **Whole-group a:** 0.0116 (MC 5-95 % 0.0044-0.030), against 0.021 now.
- **Night/day ratio:** 0.10 at a 0 deg threshold, against 0.33 at -5 deg now.
- **Stage access:** [0.19, 0.010]. Effective clearance a x f_eff = 0.00091, against 0.00139 now (x0.66).
- **Step 1's "a_plk ~0.005" is superseded.** It treated H_whiting/H_herring as a straight multiplier.
  - By day, the concentration per metre at the whiting's height almost equals the layer value: herring x1.05, pout x1.05.
  - The real cuts are the dim light at the bed (R^2 0.44 of saturation by day) and the night collapse of herring and sprat (x0.08 / x0.04).
  - Pout rise x1.75.
- **Member mix of the planktivore take.** Under the overlap convention it is pout 0.64 / herring 0.30 / sprat 0.04.
  - The layer convention gives herring 0.60 / pout 0.28.
  - Deep-water stomachs are pout-dominated: SMS North Sea pout 0.61 / sprat 0.26 / herring 0.12; Skagerrak April 1991, in the box rectangles, pout only.
  - **The member mix is the test that tells the two conventions apart, and it favours the overlap convention.**
- **The fish share of the diet does not tell them apart.** Both conventions give a planktivore share of 0.08-0.12 (0.13-0.19 at window ratio 0.45). That matches the in-area evidence (0.08-0.18) and neither convention reproduces the North Sea 0.44.

**Q2, booked M2: lower whiting -> planktivorous_fish `predation_mortality` from 0.203 to about 0.07 /yr (0.03-0.11)** [D].
- **Half of the 0.203 (0.102) is a sprat term.** It puts M2 3.03 /yr on a box sprat stock of 0.099 t/km2: the box whiting would eat three box sprat stocks a year. This comes from the 223.37 method, which gives the box whiting the North Sea whiting's consumption per whiting tonne of each species (box/North Sea scaling factor 8.9 for sprat). The other halves are pout 0.054 and herring 0.048.
- **Three transfers that respect box prey availability** give 0.064-0.079 (0.050-0.105 over the Q/B range).
- **In-area stomachs agree:**
  - Skagerrak April 1991: planktivore share 0.069, about 0.12 annualised.
  - 2022 Q1/Q3: about 544 Skagerrak whiting held one herring.
  - Oslofjord: krill ~82 %, fish rare.
  - Jutland edge, 1981 Q1 counts: 0.10-0.28.
- **The GOV survey does not under-catch the box's small herring.** Length corrections move the planktivore biomass <= 13 cm by x0.54-1.13. Explaining the gap would need x8.9.
- **Effect on the planktivores' derived growth:** the budget numerator falls 1.271 -> 1.138, so g falls by about 10 % (x0.895; x0.87-0.93).
- **Effect on the K mask:** K's formula is unchanged (it is derived from F only), but the surplus the mask has to cover shrinks.

## Parameters: current vs proposed

| Key | Current | Proposed (central, range) | Tag | Basis |
|---|---|---|---|---|
| whiting -> planktivorous_fish `max_intake_rate` | 0.021 (layer) | **0.0116** (0.0044-0.030) | D | Section 220 adult whiting factors as in 223.26 (7330 fish/t, v 0.8 BL/s, p 0.2, scan 0.5, adult share 0.753). Prey met = B F_j/H_W in a 10 m band; R^2 = R_sat^2[q + (1-q) E_b/(E_b+Ke)] with R_sat 0.5 m, R_nv 0.2 m, Ke 0.02 lx. Light at the bed is computed per cell depth (`maps/depth.npz`) with three Kd profiles. Members' share of a: herring 0.59, pout 0.25, blue whiting 0.14, sprat 0.01 |
| band shares F_j (10 m), day / night | (in-layer) | herring 0.15 (0.04-0.34) / 0.02; pout 0.7 / 0.4; blue whiting 0.3 / 0.15; sprat 0.1 / 0.01 | J (+D, F) | Same values as the benthivore pair (Attackrater benthivores_encounter.md 2.3). Herring check, HERAS by bed depth: 0.66 / 0.10 / 0.04 / 0.06 over < 80 / 80-110 / 110-130 / 130-150 m beds, unmeasured >= 150 m (47 % of cells). Equal-cell weighting ~0.10, weighting by herring backscatter ~0.31; 0.15 is a judgement between them. Pool a moves about x0.7-1.6 across that choice, inside the MC |
| `dark_ratio` / `light_threshold_deg` | 0.33 / -5 | **0.10 (0.05-0.21) / 0** | D (mechanism), J (threshold) | Bed light plus herring and sprat leaving the band at sunset. The upper end 0.21 assumes 30 % of whiting follow the clupeids up at night; the Dana night surface hauls bound that toward 0, so following is the upper bound, not the central value. Threshold 0 as on the benthivore pair (bed light, prey switch at sunset). Rindorf 2003: "No diel pattern in the occurrence of the larger prey, herring and sprat" (p. 270; long digestion), so stomachs cannot test the ratio |
| `stage_access` | [0.16, 0.010] | **[0.19, 0.010]** (band to [0.29, 0.030] at window ratio 0.45) | D, J (0.39 vs 0.45) | Window over whiting >= 20 cm, members weighted by their overlap-convention encounter. Effective 0.079 at J/B 0.38 |
| `predation_mortality` | 0.203 | **0.07** (0.03-0.11) | D (transfers), F (stomach test) | `tables/step2_sms_m2_species_stdout.txt`. (b) linear clearance 0.065-0.071. (c) fixed ration with diet re-weighted by box/North Sea availability 0.064-0.079 (0.050-0.105 at Q/B 3.3-5.6). Check: in-area share 0.07-0.14 x Q 1.87 t/km2/yr / 3.3 = 0.04-0.08 |
| whiting -> planktivores flow (t/km2/yr) | 0.670 (implies diet share 0.36, 0.27-0.46) | **0.23** (0.10-0.36) | D | Q/B 4.2 (3.3-5.6) [F, params_split_223] x B 0.446 x planktivore share 0.11-0.14 |
| planktivorous_fish derived g | 0.0031889 | **x0.895** (x0.87-0.93), direction only | D | g proportional to r_max 0.37 + M1 0.26 + checked sum M2 0.641 (seabirds muted) = 1.271 -> 1.138 |
| planktivorous_fish `regional_carrying_capacity` | 4.8 | unchanged here (direction below) | J | K = B/(1 - F/r) uses F only |

Diet gate diagnostic (`budget_gate`, 1/3-3 rule) with the booked 0.07: the booked planktivore share of whiting's four pairs falls 0.42 -> 0.20. The encounter-implied share is 0.08 (overlap) or 0.12 (layer), so the ratio is 0.41 / 0.61. Today it is 0.20 / 0.29, which is OFF. This is a check, not a criterion.

## Evidence (load-bearing; downloads in `data/step2_lit/`, manifest `data/step2_README.md`)

### Where whiting forage
- **Danish HERAS cruise reports, R/V Dana, June-July 2017/2018/2019 [F].**
  - Whiting totals 1008 / 1643 / 462 kg, of which 1008.0 / 1642.7 / 460.7 kg came from day bottom hauls (EXPO).
  - Night surface hauls (FOTO, headline at the surface) took 0.30 kg in 18 hauls, 0.58 kg in one 42 m Kattegat haul, and 1.58 kg mostly in Kattegat <= 47 m.
  - In the box rectangles (45G0, 46G0, 45G1), whiting came only from day bottom hauls at 84-90 m (16 / 29 / 3 kg), together with herring and pout. The night surface hauls there had 0 whiting and 400-1400 kg herring.
  - Norway pout appeared only in day bottom hauls, e.g. 929 kg at 137 m.
  - Caveats: Table 3 columns are offset by one station in 2018 (and on p. 1 of 2017); they were matched by haul sums. There are no night bottom or midwater hauls, so this bounds only the upper ~20-30 m.
- **Onsrud et al. 2004, ICES JMS 61:1278 (Oslofjord, 120 m) [F].**
  - Fig. 5: by day ~85 % of the whiting in 75-100 m; at night ~80 % in the upper 50 m.
  - "At night, herring, whiting, and sprat were most abundant in the upper 50 m ... Norway pout ... no catches in the upper 25 m" (p. 1284).
  - Diet: krill "up to 99% ... on average ~82%"; stomachs with fish 0.02-0.13.
  - This is a fjord population feeding on krill; it is the only documented case of strong diel migration.
- **Rindorf 2003, MEPS 249:265 [F].**
  - Herring and sprat show no diel pattern in the stomachs (p. 270). Step 1 read "free-swimming prey by day" as applying to clupeids; that result covers krill, shrimp and juvenile whiting.
  - "in most cases, no evidence of vertical migration was found" (p. 274).
  - No bottom foraging at the 137 m site.
- **Adlerstein & Ehrich, ICES CM 2000/K:01 (GOV, 58 N, November) [F]:** "catch rates except those of large whiting vary significantly with time of day". Whiting > 20 cm: p 0.23-0.3.
- **Bromley, Watson & Hislop 1997, ICES JMS 54:846 [F]:** the pelagic whiting were 0-group, 1.5-5.1 cm. "Few fish were caught near the bottom."
- **Pedersen 1999/2000 [A]:** demersal whiting ate pout and herring and fed by day; pelagic whiting ate sprat and sandeel at night. The layer definitions and the site (possibly the southern North Sea) were not accessible.

### Herring at the bed
`tables/step2_heras_bottom_share_stdout.txt` [D]: herring backscatter share in the bottom 10 m by day, by bed depth, as given in the table row above. The step 1 / benthivore 0.34 was carried by logs over beds < 80 m, which hold 3-10x the backscatter per log.

### In-area stomachs
- **Public ICES database:**
  - Skagerrak April 1991, box rectangles, 106 whiting: planktivores 0.069 of identified mass (pout only), krill 0.54.
  - 2022 Q1/Q3, Skagerrak, 252 + 292 whiting: fish prey mostly cannibalism; one 10 cm herring (4.5 g) in a 32 cm whiting at 44G0. Data-quality flags: 95 g of prey in 539 stomachs, one fish uploaded twice (FishID 8455960), and Kattegat Q1 2022 has no prey rows. Do not lean on this sample.
  - 1981, Jutland edge (43F6-43F8), counts only (`tables/step2_ices1981_counts_stdout.txt`) [D]: Q1, 95 whiting with 3 clupeid items, planktivore mass share 0.10-0.28; Q4 0.18-0.35.
- **ICES CRR 219 [F]:** Area 8 (Skagerrak + Kattegat) holds 1047 whiting stomachs from 1991, but no Area 8 diet is tabulated anywhere. The Swedish Q1-Q3 1991 samples are not in the public database (ICES CM 1992/G:12; Dorman et al. 1992).
- **Depth contrast, not in-area:**
  - Dahl & Kirkegaard 1986, Jutland west coast at 11-48 m [F]: herring 29-75 % of whiting diet, 0-group herring of ~9.5 cm.
  - Kattegat 1991 [F]: planktivores 0.47.
  - Shallow clupeid nurseries give a high fish share; deep water gives krill and pout.
- **Bergstad 1991, Norwegian Deep [S, Gordon 2003]:** "exclusively of crustaceans and fish". No numbers; the full text could not be obtained.

### GOV catchability
- **Walker et al. 2017, ICES JMS 74:1448, supplement [F].**
  - Table 3 herring multiplier: 0.11 (9.1 cm), 0.41 (12.6 cm), 0.30 (19.5 cm), 0.22-0.28 for adults.
  - Trap: the CSV's `Efficiency` column is flat at 0.244; the length pattern is only in the `Absolute` column and in Table 3.
  - The sprat curve is herring's curve mapped onto sprat lengths, not a measurement.
- **HAWG 2023 [F].**
  - North Sea herring (NSAS) IBTS-Q3 catchability at age, relative to age 2: age 0 2.49, age 1 1.11.
  - Skagerrak-Kattegat herring (WBSS) Q1: age 1 2.47 relative to age 2.
  - HERAS 3a age 0 was reported only in 1992-94, as 0.2-4.5 % of the stock's biomass.
- **Fraser et al. 2007 [F]:** pelagic species are excluded ("N/A").
- Result (`tables/step2_gov_catchability_stdout.txt`): planktivore biomass <= 13 cm moves x0.54-1.13 by quarter. The box's whiting-sized herring are scarce in fact, not under-sampled.

## The booked M2, the derived growth term and the regional K

- **Growth term.** `growth_budget.py` charges the planktivores' g with every checked booked M2. g = (r_max + M1 + sum M2)/((s* - u) 1460) x 12/5.5. Lowering whiting's 0.203 to 0.07 lowers g by ~10 %. The layer/overlap choice for the a does not enter g (s* uses the planktivores' own ceiling pairs).
- **K mask.** The library derives K = 3.33/(1 - 0.115/0.37) = 4.8 from fishing only. Its loss at the start density is r_max B/K = 0.26 /yr.
- **What K currently stands in for.** AGENTS calls it a temporary mask over booked M2 that trained predators do not realise.
  - Context only, from section 232 / Bestandsreglering model runs, not used as inputs: realised whiting M2 ~0.006 and realised benthivore M2 ~0.0004, against booked 0.203 and 0.145. The unrealised booked M2 is therefore ~0.34 /yr.
  - At the proposed 0.07 it falls to ~0.21 /yr.
- **Direction.** The phantom surplus the mask absorbs shrinks by ~0.13 /yr. K's formula stays. If K were ever re-derived as "the mask", it would move toward the F-only value; the remaining mask is mostly the benthivore pair.
- **A box-consistent pair.** A lower booked M2 and a lower a (overlap) are consistent with each other: the trained whiting would then be asked to realise ~0.07, not 0.203.

## Should the convention apply to every layer-separated pair?

Yes, as one rule [J]:
- **The rule.** a = clearance x prey concentration at the predator's own height (F_j/H_pred), with R set by the light there. The eat share is time allocation among prey, not a vertical position.
- **The library is currently mixed.** The benthivores' fish pairs already use the band form; the whiting (bar 232's epifauna pair), water-column and mackerel pairs use the layer form.
- **When it matters.** Divergence is material only where the predator's band and the prey's layer do not nest. Examples: a bed predator facing midwater prey at night, or a thin prey layer under a thicker predator band.

Affected pairs, not re-derived here:
1. **whiting -> zooplankton (priority).** This is step 1's clearance anchor. The pool is the 0-25 m A17 community; krill (0.12 of it) sit in a 20-40 m layer at ~70-100 m by day. A bed whiting meets almost no copepods, and krill only over shallower cells. The pair is likely cut by several-fold, and the whole whiting diet picture turns on it (`tables/step2_diet_check_stdout.txt`, sensitivity at zooplankton a x0.5 / x0.25).
2. **whiting -> benthivorous_fish.** Settled juveniles in a 2 m layer: about x0.2 (0.1-0.5) on geometry (step 1 open point 3).
3. **predatory_fish (water column) -> benthivorous_fish and -> whiting.** Saithe and hake meeting bed juveniles and the whiting band: likely down.
4. **predatory_fish -> planktivorous_fish and -> zooplankton.** Midwater predators largely inside the prey layers: check, small change expected.
5. **mackerel -> planktivores and -> zooplankton.** Already partly in overlap form (krill x0.5): check.

Already consistent: the benthivores' fish and bed pairs, planktivores -> zooplankton (foraging overlap 1.0), whiting -> mobile_epifauna (232). Seal and porpoise a values are judgements and untouched.

## Open points

1. **Herring band share weighting.** Equal-cell weighting gives ~0.10, backscatter weighting ~0.31; 0.15 is a judgement. HERAS does not see the bed at >= 150 m (47 % of cells).
2. **Pout catchability.** Walker's pout gradient (x1.8) against Fraser's step (x2.9) in the box stage table moves pout's Q3 share <= 13 cm from 0.296 to 0.162. Pout carries 0.25 of the overlap a and most of the f_J rise, so [0.19, 0.010] inherits this choice.
3. **Blue whiting.** The x2.4 rests on a judged band share 0.3 / 0.15 and a 0.05 window share. It is a small term (0.14 of a), but unmeasured.
4. **Missing in-area stomach data.** Swedish 1991 Q1-Q3 and the Area 8 stomachs (1047) are not in the public database; SLU Aqua may hold them. Bergstad 1991 (Norwegian Deep) and the Pedersen full texts were not accessed. These are the only sources that could move the 0.07 by more than its range.
5. **The overlap convention changes the night/day ratio more than the annual a.** Stomachs cannot test it (no diel signal for large prey).
6. **Fish share is not closed.** Neither convention brings whiting's planktivore share near the North Sea 0.44. In the box that share is plausibly ~0.1-0.15, and both conventions and the proposed M2 agree on it.
