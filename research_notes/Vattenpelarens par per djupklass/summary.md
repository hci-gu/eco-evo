# Water-column pairs by bed class: what carries the krill excess? (2026-11-10)

Research note, written by a research subagent and saved by the main agent. Nothing in the library, code,
tests, manifest, resume or AGENTS.md was changed, and no model output set any value.

Files:
- `scripts/wc_terms.py` -> `tables/wc_terms_stdout.txt`, `.json`: the terms of every pair, reproduced from
  the derivation scripts (krill classes exact 0.06774 / 0.03144 / 0.005791 / 0.000136; planktivores 0.00595;
  benthivores 0.00569 against the library's 0.0055; whiting 0.01301).
- `scripts/gate_options.py` -> `tables/gate_options_stdout.txt`, `.json`: the gate by bed class, the factor
  decomposition and the options.
- `scripts/member_size.py` -> `tables/member_size_stdout.txt`: the size check.
- `tables/budget_gate_stdout.txt`: `tools/probes/budget_gate.py` as run on 2026-11-10.
- `evidence.md`: the trail, items E1-E18.
- `data/README.md`: sources and licences (Fujii 2016, CC-BY).

Tags: [F] source, [D] derived here, [J] judgement.

## Verdict

1. **On the a side, two factors carry the excess, and both conflict between pairs: vision and capture.**
   - Within the 150-250 m class, where 0.78 of box saithe live, saithe < 60 cm clear krill at 0.97x the
     planktivore rate per t/km2. That 0.97 is the product of three terms: capture p x2.5; vision R^2 x5.0;
     overlap x0.093.
   - The booked diet and the visible biomasses require about 0.21, so the excess there is x4.6. At equal
     cells it is x12, at the saithe spread x7.5.
   - The overlap term favours the fish in every class (x0.75 to x0.005). The excess therefore comes from
     vision and p.
2. **The vision conflict is the root inconsistency.** The same saithe in the same light is given two
   different reactive-distance rules.
   - Krill pair: a light-driven gadoid function (Ke 0.02 lx). At 0.055 lx (the 150-250 m krill layer) it
     gives R_eff 0.60 m; in the shallow classes 0.71-0.74 m.
   - Fish pairs: R is fixed. Pout and blue whiting are at the "dark" 0.2 m by day and night, herring at
     0.5 m by day. The section 220 premise was "bed ... below ~0.5 lx even at noon".
   - Section 253 applied the measured, brighter box light to the krill pairs only. Under it, beds < 150 m
     get 0.5-2.4 lx by day.
3. **The gate cannot tell which vision rule is right, because the two bracket the booking.** Krill share of
   the encounter, against booked 0.361:

   | variant | equal cells | saithe spread | 150-250 m only |
   |---|---|---|---|
   | library | 0.87 | 0.81 | 0.73 |
   | krill pair's vision rule applied to the fish (V_one) | 0.69 | 0.70 | 0.69 |
   | fish pairs' vision rule applied to krill (V_fv) | 0.13-0.30 | 0.10-0.24 | 0.08-0.20 |

   - The quantity that decides it is a gadoid's reactive distance on krill and on 10-20 cm fish at
     0.05-0.5 lx. It is not measured anywhere in the notes.
   - So neither the krill a nor the fish a's can be called "the wrong one" from the gate. What is wrong is
     that they use two vision rules for one predator.
4. **Field test over 150-250 m beds: the field share lies between the booking and the encounter, and below
   the library value.**
   - Saithe krill mass share by season and depth: all seasons 0.19 (0.08-0.27); NE North Sea Q3 0.55
     (0.36-0.78); 2026 Q1 at true depth 0.0003 (5 stations).
   - Group value in Q3 [D]: about 0.47 (0.32-0.69). The library's 0.73 sits at the top of the Q3 range.
   - Within a length class, the stomachs show no drop in krill feeding from 100-150 to 150-250 m (odds ratio
     0.98, 0.55-1.75). The library class a falls x5.4 between those classes and x12 from < 100 m.
5. **Size: the field's krill share falls with saithe length. The library pair cannot represent this, by
   construction.**
   - Field: CRR 164 winter euphausiid mass 48 % at 30-40 cm, 76 % at 40-50, 21 % at 50-70, 9 % at 70-100,
     0.1 % at >= 100 cm; CRR 164 summer 28 / 37 / 46 / 34 / 13 %; CRR 219 Q2 by age 49 % at 42 cm down to
     7 % at 95 cm; FO falls with length in every bed class; Fujii 2016 SIZE rho 0.155 (p < 0.01), DEPTH
     0.099; Bergstad 1991 abstract "small saithe ... and the euphausids as prey".
   - Library: the krill share at 150-250 m is 0.78 for saithe < 60 cm and 0.79 for >= 60 cm. Both stages
     see krill at R_sat about 0.86 m, the fish R is fixed, and a per tonne scales as N_t v_rel for every
     prey, so the ratio is size-independent, against a x3-5 fall in the field across the box's length range.
   - Under V_one the share does fall, 0.75 to 0.57, only from the 2 L_pred cap on fish R (the fish R0 of
     5.8 m capped to 0.86 / 1.40 m, while krill R0 stays at 0.87 m).
   - The depth pattern rides on size: 25-35 cm saithe over 102 m beds, >= 55 cm over about 210 m beds.
6. **Ranking of the inconsistency.**
   1. The vision rule differs between pairs. It is the largest term (R^2 x5-7.7), and which side is right is
      unmeasured.
   2. Capture p: krill 0.5 [J] against fish 0.2 (Turesson & Bronmark 2004, a lab value; the Juanes 2002
      review gives piscivores about 0.5). That is x2.5, of which the review supports about x1.3.
   3. Structure: no size dependence of the krill : fish ratio; a krill class contrast that the stomachs do
      not show; a policy that the contrast draws onto shallow cells (fix94: 0.60 of the water column over
      < 100 m).
   4. The booking: the Q3 field at 150-250 m (0.47) is above the booked 0.36, the all-season value (0.19)
      below. It is not the main carrier.
   5. The visible fish biomass (window share 0.376): no case was found against it.

## Task 1. Terms of every water-column pair as the library has them

Units: a in km2 per whole-group tonne per 6-h tick, per t/km2.

| term | krill (0.026; 0.068 / 0.031 / 0.0058 / 0.00014) | planktivores (0.0060) | benthivores (0.0055) | whiting (0.013) | mackerel (0.00088) |
|---|---|---|---|---|---|
| source | 245/253/262: krill_pairs_wc.py field + krill_geometry.py | 220 pred_encounter_a.py, 223.26/228 split, 243.3 shares | 220 in-layer 0.0651 x 243 member overlaps | 220/223.26 whiting_prey | 228 split_encounter_mack |
| prey density | 5 t/km2 x depth shape: class B 2.08 / 5.02 / 6.11 / 7.12 | 4.72 | 1.53 | 0.45 | 0.85 |
| visible share | 1 (no window, no refuge) | [0.65, 0.18] -> 0.376 (B_vis 1.776); visibility_floor 0.25 | [0.42, 0.12] -> 0.166 (0.254) | [0.41, 0.08] -> 0.134 (0.060) | 0.14 (0.119) |
| R | R^2 = R_nv^2 (1 - g) + R_sat^2 g; R_sat = min(R0 0.869 m, 2 L_pred); R_nv 0.1 m | herring 0.5 m day / 0.2 m night; pout and blue whiting 0.2 m day and night | 0.25 day / 0.2 night; elasmobranchs 0.3 | as benthivores | herring geometry |
| light / vision | g = E/(E + 0.02 lx), measured light per depth and hour | fixed (220, central-profile era) | fixed | fixed | fixed |
| capture p | 0.5 [J] | 0.2 (lab) | 0.2 | 0.2 | 0.2 |
| scan | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| speeds | saithe 1.0 BL/s, x0.6 at night; u 0.04 m/s | same; u 0.2 / 0.1 m/s | same; u 0.1 | same; u 0.1 | eaters' speeds |
| overlap | band: saithe exponential from the bed (15 m) x krill layer 1 / 13 / 36 / 161 m above the median bed; O_eff (saithe < 60) 22.3 / 9.7 / 2.8 / 0.16 e-3 /m | in-layer 1/H (70 / 40 / 15 / 50 m); pool O_eff 30.0e-3 /m | in-layer 1/2 m x time near the bed (saithe 0.05 ... anglerfish 1) | in-layer 1/10 m, no overlap factor | in-layer |
| day / night | astronomical day, 0 deg; dark_ratio 0.02 | D_DAY 0.5; 0.41 at -5 deg | 0.48 | 0.48 | 0.92 |
| depth table | yes (the class a's) | none | none | none | none |
| members | saithe < 60 cm 0.504 (0.76-0.79 of the class a), >= 60 cm 0.293, spurdog non-visual, hake, velvet belly (> 150 m only), pollack; anglerfish and ling 0 | capability score >= 0.1; saithe 0.80 of the a | saithe 0.45, spurdog 0.18, hake 0.21, anglerfish 0.12 | saithe 0.77 | eaters only (not saithe) |

Places where one physical quantity has two conventions:
1. Reactive distance against light: krill the light-driven function under the measured light; fish fixed
   dark values. At the < 150 m beds (0.5-2.4 lx by day) the fish R is about 1/3.5 of the krill R, for a
   prey 6x longer.
2. R scaling with prey and predator size: krill R is set by prey image size and the 2 L cap; fish R is
   constant, independent of predator length. This is why the library ratio is size-independent.
3. Vertical overlap: krill band; fish in-layer; benthivores in-layer x a judged time-near-bed factor.
   Whiting has no such factor; the band would lower that pair (band O 48.7e-3 against 100e-3 /m in-layer),
   making it more OFF, not less.
4. Capture p: 0.5 against 0.2.
5. Day/night: krill astronomical day at 0 deg, night R at the 0.1 m floor; fish fixed 0.5 / 0.5, night R
   0.2 m, dark_ratio 0.41-0.48 at -5 deg.
6. Spatial resolution: krill has a depth table; no fish pair has one, so the krill pair alone pulls the
   predator to particular beds.
7. Prey-size window: fish prey carry windows; krill carry none, so large saithe have no lower prey :
   predator size bound.

Mackerel pair: its geometry gives 0.00117 here against the library's 0.00088 (old pool shares and the 244
window). Not re-derived; 0.1 % of the encounter.

## Task 2. Decomposition (saithe < 60 cm, krill a / planktivore a per t/km2)

| class | p | O | R^2 (krill R_eff) | saithe ratio | group ratio (member mix x0.82-0.88) |
|---|---|---|---|---|---|
| < 100 m | 2.50 | 0.746 | 7.15 (0.71 m) | 13.3 | 11.3 |
| 100-150 m | 2.50 | 0.325 | 7.74 (0.74 m) | 6.3 | 5.2 |
| 150-250 m | 2.50 | 0.093 | 5.02 (0.60 m) | 1.17 | 0.97 |
| > 250 m | 2.50 | 0.005 | 1.98 (0.37 m) | 0.026 | 0.023 |

Required ratio = (booked krill / booked planktivores) / (krill B seen / B_vis of the planktivores):

| spread | a ratio | B ratio | required | excess | needed factor on all fish a's (gate) |
|---|---|---|---|---|---|
| equal cells | 4.49 | 1.90 | 0.381 | x11.8 | x12.2 |
| saithe spread | 1.90 | 2.65 | 0.274 | x6.9 | x7.5 |
| 150-250 m | 0.97 | 3.44 | 0.211 | x4.6 | x4.7 |

Where the krill encounter comes from, by class: equal cells 0.36 / 0.51 / 0.13 / 0.00; saithe spread
0.11 / 0.37 / 0.52 / 0.00. The 252.6 statement that "~90 % comes from < 150 m beds" no longer holds after
section 262, which raised the 150-250 m class a x5.

Reading: the x12 at equal cells comes from the shallow classes (vision x7-7.7 with overlap only
x0.3-0.75), which give 87 % of the krill encounter; p x2.5 everywhere; and a required ratio below 1,
because visible krill outweigh visible planktivores x1.9-3.4. Where saithe actually are (150-250 m), the
excess is x4.6.

## Task 3. Field test by bed depth and size

Krill share over 150-250 m beds:

| source | value |
|---|---|
| model: library encounter, group | 0.73 |
| model: saithe members alone | 0.78-0.79 |
| booked, group | 0.361 |
| booked, saithe | 0.40 |
| field, saithe all seasons, N of 57 N | 0.19 (0.08-0.27) |
| field, saithe, NE North Sea Q3 | 0.55 (0.36-0.78) |
| field, NE database annual at box presence | 0.45 |
| field, saithe, 2026 Q1 true depth | 0.0003 (5 stations) |
| field, group Q3 [D] (biomass x Q/B weights) | 0.47 (0.32-0.69) |

The field lies between the booking and the encounter: in Q3 about equidistant in odds from both;
all-season nearer the booking. Within a length class there is no drop in krill feeding with bed depth
(odds ratio 150-250 m 0.98; > 250 m 2.06, 1.02-4.15), which contradicts the class table's falls.

Size: the krill share is clearly size-dependent (CRR 164, CRR 219, the NE frequency of occurrence, Fujii
2016, Bergstad 1991). Bed depth grows with length, so "small saithe shallow eating krill, large deep eating
fish" fits the field. The library's one-a pair gives the same ratio for both saithe stages (0.78 / 0.79).
The group-mean booking already integrates sizes, so size is not why the group gate is off; it matters for
the placement incentive (the depth table pulls the whole group shallow) and for any test by length.

## Task 4. Options (computed effect; none chosen to hit the booking)

Krill encounter share at equal cells / saithe spread / 150-250 m (booked 0.361); the last three columns are
the other pairs at the saithe spread as ratio to booked.

| variant | basis | equal / saithe / 150-250 m | planktivores | benthivores | whiting |
|---|---|---|---|---|---|
| V_lib | library | 0.873 / 0.809 / 0.728 | 0.32 | 0.27 | 0.08 |
| V_p | field p: fish 0.45 [F review], krill 0.6 [J]; library-wide on every fish-prey pair | 0.788 / 0.696 / 0.591 | 0.52 | 0.44 | 0.13 |
| V_one | the krill pair's vision rule and band overlap on all water-column pairs (252.6 method, 262 layer) | 0.687 / 0.705 / 0.693 | 0.40 | 1.08 | 0.25 |
| V_one_p | V_one with field p | 0.540 / 0.562 / 0.548 | 0.59 | 1.62 | 0.37 |
| V_fv lo/hi | the fish pairs' vision rule on krill: R 0.1 m, or 0.17 m (lake-trout RD cap scaled to 25 mm [J]) | 0.133-0.304 / 0.101-0.243 / 0.080-0.198 | 1.28-1.51 | 1.08-1.28 | 0.33-0.39 |
| V_dark (diagnostic) | band overlap with the fish pairs' dark R | 0.955 / 0.934 / 0.901 | 0.08 | 0.33 | 0.03 |

- Light and vision decide the sign. The field at 150-250 m lies between V_one_p and V_fv_hi. V_one sends
  benthivores to 1.1-1.6x booked, against a field saithe share of about 5 %; V_fv puts planktivores at
  1.3-1.5x booked.
- Aligning the overlap alone makes the gate worse (V_dark). Aligning vision alone gives about x0.5 in odds.
- p is library-wide: it would have to be re-derived for every fish-prey pair of every predator, not for the
  water column alone.
- Structure (engine work, no values derived here): (1) size-resolved encounter, either R ~ L_pred (the 220
  RSCALE variant) or a krill window per predator length, which needs a source for the lower prey : predator
  bound; (2) the depth table's class contrast is not supported by the stomachs, but a flatter table is not
  derivable, as the night term rests on the unmeasured dark detection distance; (3) placement: section
  263.3 found the reward gives little signal to leave the shallow cells.
- Booking: the Q3 field at 150-250 m (0.47) would mean about x1.3 on the krill booking; the all-season
  value (0.19) points lower. Section 262's 0.40 stands.

## Measured / derived / judged

- Measured [F]: the stomach numbers above (CRR 164, CRR 219, ICES database, Fujii 2016); saithe bed depth by
  length (GOV); the box light down to about 65 m; Juanes 2002 (review level).
- Derived [D]: every a, factor ratio and gate share above; the group field value 0.47; the size independence
  of the library ratio.
- Judged [J]: krill p 0.5 (0.6 field variant), Ke_gad 0.02 lx, R_nv on krill 0.1 m; the fish pairs' R values
  (0.2 / 0.25 / 0.5 m); the vertical positions; the V_fv scaling; K_deep below 65 m. Both vision rules are
  judgements on unmeasured reactive distances.

## Open, flagged

- Gate against realised diet (263.2: 0.81 against 0.60 at the GOV spread). The gate uses annual-mean a's.
  The krill dark_ratio is 0.02, while the fish pairs' are 0.41-0.48, so per-tick shares swing between day
  and night, and the time mean of shares is not the share of time-mean encounters. A candidate cause,
  untested.
- Prey side, budget_gate today: the krill pool as a whole is fine (implied M2 1.12 of booked). The shortfall
  falls on planktivores (0.037 vs 0.174), benthivores (0.015 vs 0.083) and whiting (0.028 vs 0.157); their
  derived g pays booked M2 that the water column never realises.
- Mackerel pair geometry: 0.00117 here against the library's 0.00088; not re-derived.
- Literature: OpenAlex refused every query (shared rate limit). Bergstad 1991 is still not open access. The
  only new full text is Fujii 2016 (CC-BY).
