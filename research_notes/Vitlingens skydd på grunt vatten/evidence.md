# Evidence: whiting's protection over shallow beds (2026-11-20)

Tags: [F] full text read, [A] abstract or search summary only, [S] secondary, [D] derived here, [J] judgement.
All numbers are quoted from the files in `tables/`; each item names its script.

## Model facts (context only, not used to set anything)

- M1. Per-predator class a's on whiting (live library 280, a x class multiplier; research_notes/"Vitlingens placering"
  tables/pair_depth_mult_live_vs_pre279.txt): WC -> whiting 0.1713 / 0.1104 / 0.0326 / 0.0281, benthivores -> whiting
  0.0718 / 0.0466 / 0.0162 / 0.0148 (< 100 / 100-150 / 150-250 / >= 250 m). Shallow : 150-250 m 5.25 and 4.43.
- M2. The class tables use ONE light profile for every bed: the convention (clear-sky lux x 0.5 cloud, Secchi top
  20 m, SMHI K 0.109 / 0.091 to 65 m) and K_deep below 65 m (research_notes/"Djupljus i synparen"). The section 280.5
  sensitivity changed only K below 65 m by class (0.16 over < 100 m beds); WC -> whiting shallow : 150-250 m went
  5.25 -> 5.43 (`sensitivity.json`), because a < 100 m bed has at most 35 m below 65 m.
- M3. Rule A: R^2 = R_nv^2 (1 - g) + R_sat^2 g, g = E/(E + Ke), Ke 0.02 lx (bracket 2e-4 to 0.1); R_sat =
  min(R0, 2 L_pred), R0^2 exp(c R0) = |C0| A k_v^2 with c 0.25 /m, |C0| 0.7, k_v 100, A = 0.15 L_prey^2
  (research_notes/"Synregel i vattenpelaren"). Fish prey R_nv 0.3 m.
- M4. Capture p per encounter = q p_s / (1 - r (1 - p_s)), q 0.8 (pre-strike escape, bright-lab value), kept
  light-independent by convention (research_notes/"Fangstsannolikhet for fiskbyten").
- M5. Frozen fix104 (section 284): whiting at GOV class shares under the trained water column die at 1.6-1.7 /yr;
  with the water column also at GOV saithe shares 0.70-0.84 /yr (booked 0.49).

## Mechanism 2: near-bed light over shallow beds

- D1 [D] Glider coverage (`glider_shallow_beds.py`): MPE-PAR daytime casts over < 100 m beds 717, in months 1, 2, 5,
  8, 9, 11, 12 only (none Mar, Apr, Jun, Jul, Oct), all on the Swedish side, 57.77-58.27 N, 11.05-11.33 E.
- D2 [D] K_PAR 20-80 m over < 100 m beds 0.149-0.178 (median by 15 m slab), against 0.094-0.121 over 150-250 m beds
  (research_notes/"Ljus under 65 m" tables/voto_k_stdout.txt).
- D3 [D] Same mission, same days: daytime median PAR at 50 m over < 100 m beds is 0.007-0.54 (median 0.18) of that
  over 150-250 m beds; at 65 m 0.000-0.43 (median 0.073). Month-matched: e.g. May 50 m 2.5e-3 vs 0.108 umol, December
  65 m 3.0e-4 vs 4.0e-3 umol. The water over the shallow beds is darker at the same depth, not only deeper-reaching.
- D4 [D] Near-bed (bed - 5 m) daytime PAR over < 100 m beds (median bed 76-92 m): monthly medians 4.4e-5 to 1.5e-3
  umol m-2 s-1 (0.002-0.08 lx); 0-87 % of casts below the sensor threshold (6.6e-5 umol, 3.3e-3 lx). Over
  100-150 m beds 42-100 % below threshold; over 150-250 m and >= 250 m beds 98-100 %.
- D5 [D] Rule-A light term (`shallow_light_factor.py`; f = r2 + (1 - r2) g, daily mean with the month's day
  fraction): mean daytime g at the bed over < 100 m beds at Ke 0.02 lx: glider 0.14-0.74 by month (the convention
  0.62-0.97). F = H(glider)/H(convention) on matched months (validation: this simplified term gives 0.91 for K 0.16 vs 0.09 below 65 m on < 100 m beds; the chain gives 0.93): 0.55 (Ke 0.02, r2 0.06; below-threshold casts at the
  threshold), 0.51 (at 0); Ke 0.002: 0.64-0.82; Ke 0.1: 0.42-0.43; r2 0.035-0.09: 0.47-0.58. If the 2.5x glider vs
  convention level gap of "Ljus under 65 m" section 3 is a glider artefact (all glider E x 2.5): 0.50-0.91, central
  0.68. 100-150 m beds: 0.48 (0.26-0.80; level variant 0.31-0.98). The 150-250 m class is ~1 (dark) in both.
  Placement check: casts west of 11.2 E 0.58 (0.43-0.85); west of 11.15 E over beds >= 70 m 0.50 (0.39-0.81); beds
  >= 70 m 0.47 - the darkness is not confined to the innermost coastal casts. lx per umol is taken at 50 (deep clear
  water); no OCR casts exist over shallow beds (inside the level-variant bracket).
- D6 [D] Danish side (39 % of the shallow whiting biomass in GOV, 58 of 210 shallow hauls; `kd490_sides.py`): MODIS
  surface Kd490 Feb-Oct mean 0.148 over Danish < 100 m beds, 0.235 over Swedish < 100 m beds, 0.148 over 150-250 m
  beds. The Swedish shallow water is the Baltic-outflow coastal band; the Danish shallow surface water is no more
  turbid than the deep box. Satellite K is the top optical depth only; the Danish near-bed column is unmeasured. Estimate [D/J]: the Swedish
  near-bed light x4-6 (the surface-layer difference, 0.235 vs 0.148 over ~15-20 m) gives F 0.78 (0.59-0.97); box
  weighting 0.61 / 0.39 gives F < 100 m = 0.64 (0.49-0.88).
- D7 [D] Beam attenuation: for whiting prey 15-25 cm and a 1.0-1.6 m cap, R0 falls below the cap only at c > ~1.7-3.7
  /m. Near-bed glider backscatter is ~7.7x mid-water (voto_k, mixed sensors); scaling a ~0.15 /m particulate part of
  the library's c 0.25 gives ~1.3 /m [J]. Contrast loss is at most ~10 % on the shallow class; left at x1.0.
- Step inside the field range: with F 0.64 / 0.48 the < 100 : 100-150 m ratio of both pairs steepens from 1.55 to
  ~1.95 (light + escape) - the classes holding 97 % of field whiting; only cod's 100-150 m peak offsets it.
- Already in the model? No. The light convention has no bed-class or coastal-water dependence above 65 m (M2).
- Double count: none with rule A (this is its input light). It overlaps with the 2.5x level gap (handled as the
  level variant) and would also move every other shallow-bed visual pair and the equal-cell a's (not examined).

## Mechanism 1: light-dependent escape and vigilance

- E1 [F] Frey JR, Burford BP, et al. 2026. Prey group cohesion mediates the outcome of predator-prey interactions
  under artificial lighting. Funct Ecol, doi:10.1111/1365-2435.70389 (open access, eScholarship). Largemouth bass on
  groups of juvenile rainbow trout at 20, 0.5 and 0.005 lx: at 20 lx bass attacked most and "attacks were least
  likely to be successful"; predation given attack highest at 0.5 lx; predation FREQUENCY did not differ with light
  (Chisq 2.81, p 0.25). The effect ran through prey group spacing (cohesion), which in the model is the schooling
  factor (whiting schooled share 0.1 / 0.05) - mostly a schooling effect, small for whiting.
- E2 [A] Petersen JH, Gadomski DM 1994. Light-mediated predation by northern squawfish on juvenile chinook salmon.
  J Fish Biol 45:227-242. In a raceway about five times more salmon were captured in 4 h of relative darkness
  ("0-03 lx" in the abstract, likely 0-0.3) than at 160 lx; capture inversely related to light.
- E3 [A] Meager JJ, Domenici P, Shingles A, Utne-Palm AC 2006. Escape responses in juvenile Atlantic cod: the effects
  of turbidity and predator speed. J Exp Biol 209:4174. Putative escape success against a fast model predator 73 % in
  clear water, 21 % in highly turbid water (less responsiveness, mistimed escapes); intermediate turbidity raised
  escape from a slow attack. Visual information decides gadoid escape success.
- E4 [S] Howick & O'Brien 1983 (via "Fangstsannolikhet" E13 and E1): "At high light, bluegills can spot modest-sized
  bass well before being spotted"; bass reactive distance saturates at 1.5-5.6 lx. Mazur & Beauchamp 2003 [S via
  sibling note]: lake char predation low in darkness, peak at 0.50-0.75 lx, then an intermediate rate.
- E5 [A] Strand et al. 2020 JEMBE: large cod and sculpin most effective on newly settled 0-group cod and saithe at
  dusk/night/dawn (via "Torskfiskars ljusberoende vid jakt").
- E6 [S] Field piscivory peaks at crepuscular light (Cerri 1983, Linehan et al. 2001, cited in E1).
- Reading [J]. The prey's visual advantage appears in BRIGHT water (tens of lx and above). Measured near-bed light
  over shallow box beds by day is 0.002-0.08 lx (D4), i.e. the crepuscular range where predation success peaks in
  E1, E4, E5. Under the convention (0.5-3 lx at an 80 m bed) it is also at or below the bass and char peaks. So the
  light-dependent part of escape that rule A lacks is small at the bed: central x0.85 on the < 100 m class
  (0.6-1.0), x0.9 on 100-150 m. No gadoid experiment measures capture per encounter across light on fish prey.
- Double count: q 0.8 already carries bright-lab pre-strike escape on every class; only the light-dependent excess is
  claimed. Schooling cohesion (E1) is in the schooling factor. Gate: such a term also lifts the pair's night/day
  (rule 0.172). Field: no diel pattern for larger fish prey in whiting stomachs (Rindorf 2003, digestion-limited),
  fish prey eaten at dawn and dusk (Rindorf & Gislason 2005), cod most effective at dusk/night (E5) - consistent with
  a flatter light response, not contradicting it.

## Mechanism 3: diel behaviour and vertical position

- G1 [D] The box has 435 day and 1 night GOV hauls (Q1 + Q3 2000-2025); no box-level diel test exists.
- G2 [D] North of 56 N, whiting 10-30 cm night/day catch ratio within rectangle x year x quarter (`gov_day_night.py`):
  Q1+Q3 2000-25: < 100 m 1.50 (0.93-2.51, 82 strata), 100-150 m 0.70 (0.47-1.05), 150-250 m 0.94 (0.58-1.26, 13
  strata); Q2+Q4 1991-97: 0.77 (0.59-1.01), 0.59 (0.41-0.85), 0.60 (0.36-1.03). No class-specific pattern; nothing
  that would put shallow whiting nearer the bed or tighter by night or hide them by day.
- G3 [S] Adlerstein & Ehrich 2000: no diel change in GOV catch rates of whiting > 20 cm at 58 N; Rindorf 2003: "in
  most cases, no evidence of vertical migration"; Oslofjord whiting pelagic, following krill up at night (Onsrud et
  al. 2004) - via "Vitlingens predatorer" and "Torskfiskars ljusberoende vid jakt".
- Model: whiting as prey 0.7 day / 0.5 night in the bottom 10 m, all classes. Factor x1.0 (0.8-1.2).

## Mechanism 4: predator density by bed class (box GOV, Q1 + Q3 2000-2025; `gov_predators_by_class.py`)

- P1 [D] Shapes (density / box equal-cell mean, < 100 / 100-150 / 150-250 / >= 250 m): saithe >= 35 cm 0.02 / 0.26 /
  2.63 / 0.29; cod >= 35 cm 0.77 / 1.86 / 0.69 / 0.26; hake >= 30 cm 2.60 / 0.38 / 0.84 / 0.34 (Q3 3.08 shallow, Q1
  0.12); spurdog 0.67 / 2.77 / 0.03; ling 0.24 / 0.66 / 2.26; anglerfish 0.25 / 0.32 / 1.31 / 2.78; whiting >= 30 cm
  1.47 / 1.41 / 0.77 / 0; prey whiting 10-30 cm 3.62 / 0.68 / 0.06 / 0.00 (0.77 of its biomass over < 100 m beds).
- P2 [D] Water column member-weighted (saithe 0.80; spurdog, hake 0.06; anglerfish, ling 0.04 [J]): 0.23 / 0.43 /
  2.30 / 0.36. Placement factor on the shallow : 150-250 m ratio: WC 0.10 (0.06-0.20), cod 1.12 (0.68-1.85), both
  pairs (a x group density) 0.36 (0.23-0.58): 4.80 -> 1.72 (1.10-2.80) (`combine.py`). With the central light and
  escape factors 0.93 (0.60-1.52, placement corners only); every factor at its corners 0.26-2.95.
- P3 [D] Field profile of the combined hazard, both pairs, relative to 150-250 m: field placement 1.72 / 2.56 / 1.00 /
  0.19 - the most dangerous class is 100-150 m (cod's peak), not < 100 m.
- P4 [S/J] Mammals point the other way: harbour seals haul out on the coasts and porpoises use the shallower coastal
  band; their booked M2 on whiting is 0.034 of 0.49 (research_notes/"Vitlingens predatorer"). Hake, the one water
  column member over shallow beds in summer, eats whiting (13 % in the Skagerrak 2024 stomachs, same note).
- Already in the model? The pair's encounter is per predator; placement is the policy's. The trained WC stands
  shallow (0.28-0.61 of its biomass on < 100 m vs GOV saithe 0.04; section 278), so the model lacks this field
  segregation; section 284's "WC at GOV" arms (M5) are its model-side measure.

## Mechanism 5: other

- O1 [D] Saithe stomachs vs prey density (`saithe_clearance_index.py`): juvenile-gadoid FO in adult saithe is flat
  across classes (0.14-0.21) while GOV juvenile-gadoid density falls 15-20x from < 100 to 150-250 m; a Poisson
  clearance index gives < 100 : 150-250 m 0.04-0.05 (0.02-0.13). NOT diagnostic of the light gradient: for a
  predator near its ration (multi-prey saturation) the diet share depends on relative a B, and a light multiplier
  common to all fish prey cancels; stomachs are 1980-87 rectangle-spread records against 2000-25 GOV. Reported only
  as "the field gives no sign that saithe over shallow beds eat juvenile gadoids faster per unit prey".
- O2 [D] Cod -> whiting stomach power: N of 57 N, 56 cod >= 30 cm stomachs over 150-250 m beds with 0 whiting
  records; 427 over < 100 m with 8; 330 over 100-150 m with 11. Too thin to test a 3-5x gradient.
- O3 Already covered elsewhere: no shelter refuge (Cyanea only 0-group, ~3 % of biomass; "Vitlingens golv");
  schooling factor ~1 ("Stimskydd i motes-a"); no avoidance of cod between 2-4 km cells ("Vitlingens uteblivna
  predation"). Whiting habitat choice by substrate (sand > gravel; Atkinson et al. 2004 via "Vitlingens golv") gives
  no light-related protection.
