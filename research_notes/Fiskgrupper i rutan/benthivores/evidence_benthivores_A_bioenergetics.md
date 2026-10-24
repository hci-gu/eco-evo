# Evidence A - Bioenergetics of the benthivorous_fish members

Purpose: per-species Cmax, standard metabolism, activity costs, starvation, energy density and
movement for the new `benthivorous_fish` group (haddock 38.6 %, cod 31.4, long rough dab 12.6,
dab 4.5, plaice 3.8, witch 1.7, starry ray 1.5, hagfish 1.3, grey gurnard 0.9, lemon sole 0.9,
greater weever 0.8). The caller does the biomass weighting. No model file was edited.

Tags: [F] full text read or data downloaded, [A] abstract only, [S] secondary (citing source named),
[D] computed here (`scripts/bioenergetics_A.py`, output `tables/bioenergetics_A_cmax.csv`,
`tables/bioenergetics_A_smr.csv`, `tables/bioenergetics_A_stdout.txt`). "Judgement" means no
species measurement was found and the value is my choice.

Conventions used throughout:
- Cmax is given in kJ of gross prey energy per kg fish per day, and in % body weight per day of a
  reference prey of 4.184 kJ/g wet (the prey energy density the FB4 cod function assumes). This
  makes the species comparable and lets the caller divide by any prey energy content.
- SMR is in mg O2/kg/h. MJ per tonne per 6 h tick = mg O2/kg/h x 6 x 13.6 / 1000 = mg O2/kg/h x
  0.0816. 13.6 J/mg O2 is the oxycalorific coefficient Fonds et al. (1985) used; I assume it.
- "Annual mean" = SMR at the reference temperature x the mean of the 12 monthly Q10 factors over the
  50 m series (7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3 C). It is 0.4-1.3 %
  above the value at the 8.4 C mean for Q10 2.0-3.0 [D]. "Bottom" = constant 6.5 C (100-300 m
  water, ~6-7 C; midpoint is my choice).

## Bottom line

Masses: gadoids 300 g, flatfish 150 g (other masses in the per-species sections and the csv).

| Species (share) | Cmax 7 C | Cmax 8.4 C | SMR, MJ/t/tick: annual mean 50 m (bottom 6.5 C) | Q10 SMR | SDA / feeding factor | Activity (x SMR) | Starvation | Energy density kJ/g wet: lean / whole / reserve | Speed |
|---|---|---|---|---|---|---|---|---|---|
| Haddock (38.6) | 128 kJ/kg/d, 3.07 %BW/d (central = cod function x1.3, judgement; range x1.0-2.0 = 99-197) | 142 kJ/kg/d, 3.41 % (range 110-219) | central 3.0 (2.5), judgement; range 2.2-3.8: direct Norin value 3.65 (3.04), ratio-to-cod route 2.2-3.1 (1.9-2.7); 1000 g direct 2.87 | 2.5 (2.2-3.1) | SDA 2.3-5.2 % of meal energy, peak 1.3-1.5 x RMR (juveniles, pellets; likely low) | routine 1.42, cruising at Uopt 1.48, max 3.4 | no haddock data; cod 84 d fast at 2-10 C with linear loss, no deaths reported | 3.3-3.8 / ~4.2-4.75 / liver up to ~2000 MJ/t (cod proxy, prior note) | Uopt 1.14 BL/s; local ~0.25 cell/tick (cod analogue) |
| Cod (31.4) | 99 kJ/kg/d, 2.36 % | 110 kJ/kg/d, 2.62 % | 4.12 (3.53), Schurmann & Steffensen | 2.2 | not re-searched | routine 1.18, cruising 1.32, max 2.0 (68 g, 12 C) | 84 d+ (prior note) | 3.8 / 4.75 / 2000 MJ/t (prior note) | 0.2-0.6 m/s; 0.25 cell/tick (prior report) |
| Long rough dab (12.6) | no data; use dab (judgement) | no data | ~1.0-1.3 at ~3 C winter (field, fasting); ~1.5-1.9 at 8.4 C if Q10 2 (judgement-heavy) | none; use 2.0 | none; use plaice | none | stores 31 % of body energy in summer and uses ~24 % for winter metabolism (field) | ~2.8-3.6 whole (mass judgement) / reserve ~850-1100 MJ/t | sedentary (judgement) |
| Dab (4.5) | 78 kJ/kg/d, 1.86 % (plaice x DEB ratio 0.71) | 91 kJ/kg/d, 2.18 % | none measured; plaice level 3.3 (2.9), range to 4.2 (judgement) | ~1.6-2.0 | none; use plaice | none | none; use plaice | none; use plaice | sedentary (judgement) |
| Plaice (3.8) | 109 kJ/kg/d, 2.61 % (lab fit 76; DEB 70) | 135 kJ/kg/d, 3.22 % (fit 92; DEB 80) | 3.27 (2.85); 400 g 2.63; fasting adults in winter ~1.45 | 2.0 (juveniles 2.4) | feeding + SDA = 15 % (11-19) of max ration energy; fed max / unfed min respiration ~2.4 (1.7-3.0) | active / rest 2.4-2.8; "factor 3" | ~180-200 d at 6-7 C, ~120 d at 10 C (to K 0.75, 150 g, [D]); adults fast Dec-Mar | ~3.0-4.1 (spent, K 0.75-0.9) / 4.7 at K 1.05; adult females 5.8 Dec, 4.1 Mar / ~1700-2000 MJ/t | seasonal migration 150-400 km; local small (judgement) |
| Witch (1.7) | no data; at or below dab (judgement) | - | 1.48 (1.26) at 150 g; 400 g 1.10 (5 g fish scaled; Q10 assumed) | assumed 2.2 | none | none | none | 5.6 whole (29-34 cm, Oct) | sedentary (judgement) |
| Lemon sole (0.9) | use plaice (judgement) | - | use plaice (judgement) | - | - | - | - | 5.8 whole (28-29 cm, Oct) | sedentary (judgement) |
| Grey gurnard (0.9) | judgement: cod/haddock-like roundfish forager | - | judgement: haddock-like | - | - | - | - | 4.4 whole (Plimmer 1921, [S]) | active bottom swimmer (judgement) |
| Starry ray (1.5) | judgement: low, plaice or below | - | judgement: plaice level or below | - | - | - | - | no data | sedentary (judgement) |
| Hagfish (1.3) | judgement: very low | - | not an annual mean: 0.8-2.3 at 10-11 C (9.9-28 mg O2/kg/h, other hagfish species, [S]) | - | - | - | judgement: months | no data | sedentary (judgement) |
| Greater weever (0.8) | judgement: flatfish-like ambush predator | - | judgement: plaice level | - | - | - | - | 5.3 whole (18-23 cm, Oct) | sedentary (judgement) |

Key reading of the table:
1. Haddock and cod together are 70 % of the group. No haddock Cmax function exists; both rest on
   the cod function of Hansson et al. 1996 (300 g at 8.4 C: 110 kJ/kg/d). Every direct lab
   comparison found haddock juveniles eating 1.4-2 times cod at equal size, so I put haddock at
   x1.3 (142 kJ/kg/d), range x1.0-2.0. That multiplier is a judgement; adult data are missing.
2. Plaice Cmax per kg is similar to the cod value at the same mass class; dab sits at about
   0.7 of plaice. Witch, long rough dab and lemon sole have no ration data.
3. Measured SMR as an annual mean: cod 4.1, plaice (150 g) 3.3, haddock (300 g) 2.2-3.8 with a
   central 3.0 MJ/t/tick (two routes, see 1b). Deep, slow flatfish (witch; long rough dab in the
   field) are 2-3 times lower. The current gadoid library value 6.0 MJ/t/tick is 1.5-2 times the
   measured SMR of cod and haddock; that gap would be routine activity, not SMR.
4. A Cmax Q10 of about 2 is supported for cod and plaice between 6 and 12 C; haddock juveniles
   show a weaker rise. Over the 50 m series this is a factor of 1.4 (cod function) to 1.8 (plaice)
   between April and October.

---

## 1. Haddock (Melanogrammus aeglefinus)

### 1a. Cmax
- No haddock Cmax function was found. FB4 has none [F, `data/fb4/Parameters_official.csv`]. A
  search for haddock Wisconsin-type or DEB parameters found none; the Add-my-Pet site could not be
  reached from here.
- Juvenile lab Cmax, Peck et al. 2003 (MEPS 251:233) [F]: "The observed maximum food consumption rate
  (Cmax) ranged from 4.9 to 13.0% d-1 and was highest for both species at 12°C". Haddock ~3.8 g:
  9.6 ± 0.7 % d-1 at 8 C and 10.5 ± 1.2 % d-1 at 12 C (Table 2). Cod of 7.3 g: 6.3 and 7.7. Fed a
  pellet diet before trials. Q10 of haddock Cmax 8 -> 12 C = (10.5/9.6)^(10/4) = 1.25 [D].
- Juvenile haddock and cod 40-50 g, Perez-Casanova 2009 (MUN PhD thesis; chapter 2 = Aquaculture
  294:228-235) [F]: "The FC of both species decreased from ~2.0 to 0.6% of body mass between 11 and
  2 °C" (single satiation meal of dry pellets, 5.28-5.35 kcal/g as fed, i.e. ~22 kJ/g). Over 8 weeks
  daily food consumption "decreased dramatically (by ~80%) in both cod and haddock when temperature
  was decreased from 11 to 2 °C and was significantly higher in haddock at both 11 and 2 °C (by
  approx. 2-fold)"; cod ate ~0.8 % BM/d. Dry pellet is about 5 times the energy of wet prey per gram,
  so ~1.6 % BM/d of pellet at 11 C is ~350 kJ/kg/d for a 50 g fish [D, approximate].
- Field: haddock "feed at between 15 and 22% of their maximum capacity" (Hall 1987, J Fish Biol
  31:479; the paper also covers dab) [A, prior note]. Evacuation-based field rations of 40 cm
  haddock (~600 g) 1.6-2.2 %BW/d at 6 C (Jones 1974 model via Ursin et al. 1985) [S, prior note].
  North Sea 1991 annual mean 1.30 %BW/d vs cod 1.44 (Greenstreet et al. 1997) [S, prior note].
  Intake scales as W^0.77-0.80 in haddock (Jones 1978) [A, prior note].
- Central value (judgement): haddock Cmax = 1.3 x the cod function of Hansson et al. 1996,
  range x1.0-2.0. Cod function: Cmax (g prey/g fish/d) = 0.099 W^-0.195 f(T), Kitchell f with
  CQ 2.41, CTO 13.7, CTM 21; prey 4184 J/g.
  - For a multiplier above 1: every direct comparison found puts haddock above cod. Peck 2003:
    x1.4-1.5 at about equal size (juveniles 4-7 g). Perez-Casanova 2009: x~2 daily intake at 50 g;
    their ~350 kJ/kg/d at 11 C is ~1.9 times the cod function at 50 g and 11.7 C [D].
  - For keeping it modest: these are fast-growing hatchery juveniles on pellets. Jones (1974)
    found one evacuation model fits haddock, cod and whiting. Field rations are similar (1.30 vs
    1.44 %BW/d), though field rations are not Cmax.
  - Hall's 15-22 % implies a much higher Cmax (5-7 times field rations). I could not read how he
    measured it, so it only supports the upper end.
  - x1.0 (the cod function itself) is the lower end; it sits below all the haddock lab evidence.

Cmax from the cod function [D] (kJ/kg/d; %BW/d at 4.184 kJ/g in brackets). Multiply by 1.3 for
the haddock central value (300 g: 128 at 7 C, 142 at 8.4 C):

| T (C) | 100 g | 300 g | 1000 g |
|---|---|---|---|
| 6.0 | 112.5 (2.69) | 90.8 (2.17) | 71.8 (1.72) |
| 7.0 | 122.2 (2.92) | 98.6 (2.36) | 78.0 (1.86) |
| 8.4 | 135.7 (3.24) | 109.5 (2.62) | 86.6 (2.07) |
| 10.0 | 150.1 (3.59) | 121.1 (2.90) | 95.8 (2.29) |
| 11.7 | 162.4 (3.88) | 131.1 (3.13) | 103.6 (2.48) |

Check: 500 g at 7 C gives 2.13 %BW/d, as in the prior note [D].

### 1b. SMR
- Norin et al. 2019 (PeerJ 7:e7784, PMC6777481) [F], 12 C-acclimated fish at 12 C, haddock 63 g:
  "SMR (mg O2 h-1 kg-1) ... 84.3 ± 4.2" (76.9 scaled to 100 g with exponent 0.80); RMR 119.4;
  MMR 284.9. Cod 68 g at 12 C: SMR 156.7, RMR 185.0, MMR 315.0. "The SMR of 12 °C-acclimated cod at
  12 °C was higher than the SMR of 12 °C-acclimated haddock at 12 °C by 85" (the decimals were lost
  in my text extraction; 156.7/84.3 = 1.86 [D]).
- Perez-Casanova 2009 [F], juvenile 40-50 g, routine MO2 (Table 2.3): haddock 107.0 / 103.6 at
  11 C and 37.7 at 2 C (HP/LP diets); cod 136.6-143.9 (11 C), 93.7-96.1 (6 C), 57.0-60.1 (2 C).
  "the Q10 value (3.07) for haddock oxygen consumption between 11 and 2 °C was similar to that
  measured in the Atlantic cod".
- Cod Q10 2.2 from Schurmann & Steffensen 1997 (35.5/57.0/78.2 at 5/10/15 C) [prior note].
- [D] Haddock 300 g: 84.3 x (300/63)^-0.2 = 61.7 mg O2/kg/h at 12 C (mass exponent 0.8 as Norin).

| Q10 | SMR 7 C | SMR 8.4 C | MJ/t/tick at 8.4 C | annual mean 50 m | bottom 6.5 C |
|---|---|---|---|---|---|
| 2.2 | 41.6 | 46.5 | 3.79 | 3.81 | 3.26 |
| 2.5 (central, judgement) | 39.0 | 44.4 | 3.62 | 3.65 | 3.04 |
| 3.07 | 35.2 | 41.2 | 3.36 | 3.41 | 2.72 |
| 2.5, 1000 g | 30.7 | 34.9 | 2.85 | 2.87 | 2.39 |

Second route, haddock as a ratio of cod [D]. Norin's own cod SMR (156.7 at 12 C) is about twice
the Schurmann & Steffensen cod used in the cod row (57.0 at 10 C), so their small fish or protocol
give high SMR for both species. Applying the measured haddock/cod ratio to the Schurmann cod
(50.2 mg O2/kg/h at 8.4 C) removes that: Norin ratio 0.54 -> 27.0 mg/kg/h = 2.22 MJ/t/tick annual
mean (1.90 at 6.5 C); Perez-Casanova routine ratio 0.64-0.75 (2 and 11 C) -> 32-38 mg/kg/h =
2.64-3.09 MJ/t/tick (2.26-2.65 at 6.5 C). The Schurmann fish size is not in the prior note, so the
mass match is unchecked.

Choice (judgement): central 3.0 MJ/t/tick annual mean (2.5 at 6.5 C), between the direct route
(3.65) and the ratio route (2.2-3.1); range 2.2-3.8. I lean below the direct value because the same
study's cod is so far above other cod data.

### 1c. SDA and activity
- Perez-Casanova 2009 [F]: "maximum post-feeding oxygen consumption (30-50% above RMR)"; "SDA
  comprised between 3.3-5.2% of dietary energy content"; in the results "SDA as a percentage of the
  ingested energy (SDAENER) ranging from 2.3-3.7%". SDA lasted 55-85 h. These are small single
  pellet meals at ~2 % BM; I expect a larger SDA share on a full natural ration (judgement).
- Norin 2019 [F], haddock at 12 C: Uopt 21.4 cm/s = 1.14 BL/s, Ucrit 56.5 cm/s = 3.02 BL/s, COTmin
  (net, at Uopt) 52.3 mg O2/km/kg. [D] Cost at Uopt = SMR + COTnet x speed = 84.3 + 52.3 x 0.770 =
  124.6 mg/kg/h = 1.48 x SMR. RMR/SMR = 1.42; MMR/SMR = 3.4. Cod: 1.32, 1.18 and 2.0.
- Model reading (judgement): moving at cruising speed costs ~1.5 x SMR; feeding = SDA (peak 1.3-1.5)
  plus search swimming, ~1.75-2 x SMR. The current gadoid values (feeding 1.75, moving 2.0) sit at
  or slightly above this.

### 1d. Starvation and reserves
- No haddock fasting experiment was found. Cod: 84 d at 2, 6 and 10 C with linear weight loss
  (Dutil et al. 2000) [prior note]. The liver is the gadoid reserve; cod liver energy per tonne of
  fish runs from ~40-70 MJ/t (starved to death) to ~2000 MJ/t (well fed) [prior note, section 3].
- Juvenile haddock HSI 9.3-12.6 % on fatty feeds (Perez-Casanova 2009 citing others) [F]; not a
  wild value. Wild haddock HSI was not obtained.
- Energy density: haddock raw fillet 3.29 kJ/g; lean body ~3.8; whole ~4.75 [prior note]. Pedersen &
  Hislop 2001 has haddock values by quarter and length, but I could not open it.

### 1e. Movement
- Norin 2019 Uopt 1.14 BL/s [F]. A 40 cm haddock cruising continuously at 1 BL/s would cover
  ~8.6 km per 6 h, but net displacement of resident gadoids is much smaller.
- No haddock tagging with daily displacement was found. The prior project report
  (`reports/Nettoförflyttning per funktionsgrupp.md`) gives cod: >55 % of days stationary (< 2 km/d)
  and recommends 0.25 cell/tick local movement. I use that for haddock (judgement). Haddock tolerate
  ~13 C upper range vs ~11 C for cod (Norin 2019 citing Brander 1995) [F].

## 2. Cod (Gadus morhua) - reference only

Covered by the prior notes; Cmax as in 1a (same function), SMR 57.0 mg O2/kg/h at 10 C, Q10 2.2
-> 4.12 MJ/t/tick annual mean, 3.53 at 6.5 C [D]. Norin 2019 gives much higher SMR for small cod
(156.7 mg/kg/h at 12 C, 68 g) [F]; the difference is size and protocol. DEB maintenance (Freitas
et al. 2010, [pM] 24.5 J/cm3/d at 10 C, TA 7400 K) is 41.7 mg O2/kg/h-equivalent at 8.4 C [D].

## 3. Plaice (Pleuronectes platessa)

Main source: Fonds, Cronie, Vethaak & van der Puyl 1985, ICES CM 1985/G:54 [F]. This is the ICES
version of the same 2-22 C laboratory work published as Fonds et al. 1992 (Neth J Sea Res 29:127;
not accessible: Unpaywall found no open copy). Young plaice, 1-326 g, fed excess fresh mussel meat
(20.4 J-physiological or 23.3 J-max per mg ash-free dry; I convert to gross with 23.3/20.4).
The 1992 abstract adds that the weight exponent of maximum intake falls from ~0.9 at 2-6 C to
~0.7 at 18-22 C [A, marineinfo]; the 1985 tables use 0.78 throughout.

### 3a. Cmax
- Table 8 (J-physiological per W^0.78 per day): 2 C 78, 6 C 247, 10 C 453, 14 C 528, 18 C 762,
  22 C 792. Table 9 fit: "DFCmax 79 [e^] 0.133 [T]" W^0.78 (n = 4). The fit sits 20-35 % below
  the table at 6-10 C, so I give both.
- Feeding peaks near 18 C: "Both Plaice and Flounder reach a maximum in the daily foodintake and
  growth at about 18 °C."

Cmax [D] (kJ gross/kg/d; %BW/d at 4.184 kJ/g):

| T (C) | 50 g table | 150 g table | 150 g fit | 150 g DEB | 400 g table |
|---|---|---|---|---|---|
| 6.0 | 119.3 (2.85) | 93.7 (2.24) | - | - | 75.5 (1.80) |
| 7.0 | 138.8 (3.32) | 109.0 (2.61) | 76.0 (1.82) | 70.4 (1.68) | 87.9 (2.10) |
| 8.4 | 171.7 (4.10) | 134.8 (3.22) | 91.6 (2.19) | 79.7 (1.91) | 108.6 (2.60) |
| 10.0 | 218.8 (5.23) | 171.8 (4.11) | - | - | 138.5 (3.31) |
| 11.7 | 233.5 (5.58) | 183.4 (4.38) | - | - | 147.8 (3.53) |

"DEB" = Freitas et al. 2010 (Phil Trans R Soc B 365:3553, PMC2981968) [F, fetch extraction: the
per-species values came through a summarising fetch tool, not raw text] female plaice {p_Am} 599
J/cm2/d at 15 C, TA 7000 K, divided by an assimilation efficiency 0.8, with structural volume ~ body
mass [D]. The 0.8 is from Teal et al. 2010 (ICES CM 2010/L:06) [F]: "It is assumed that 20% of the
ingested energy is lost from the body (Brett & Groves 1979)"; their table at 10 C gives "{Pxm} ...
Maximum surface area-specific ingestion rate 487" and {PAm} 390 (487 x 0.8), "[Pm] J cm-3 d-1
Volume-specific maintenance costs 19.4", "Ta K Arrhenius temperature 7000" - the same set as
Freitas, scaled to 10 C.

### 3b. SMR
- Fonds 1985 Table 7, DOCmin (lowest daytime O2 of unfed fish, "probably comes close to the standard
  respiratory rate"), mg O2 per day per W^0.78: 2 C 1.82, 6 C 2.67, 10 C 2.80, 14 C 4.53, 18 C 4.72,
  22 C 5.68. [D] Exponential fit over 2-14 C: 1.606 e^(0.0696 T), Q10 2.01.
- Lever 1974 (NIOZ report 1974-3, Dutch) [F], one-year-old plaice standardised to 10 g: rest 26.7
  ug-atom O/fish/h at 5.5 C and 46.7 at 10.5 C; "Log Rrust = 1.23 + 0.038 T". [D] Conversion: the
  table unit "ugrat O" is read as microgram-atoms of oxygen; 1 ug-atom O = 16 ug O, so 26.7 x 16 =
  427 ug O2/h per 10 g fish = 42.7 mg O2/kg/h (5.5 C) and 46.7 x 16 / 10 = 74.7 (10.5 C); Q10
  10^0.38 = 2.40.
- Field cross-check: Rijnsdorp et al. 1983 (ICES CM 1983/G:31) [F]: winter metabolism of fasting
  North Sea females "log winter metabolism = 0.503 + 0.74 log weight" (kJ over the Dec-Mar spawning
  season). [D] 487 g -> 310 kJ; over ~110 d = 5.8 kJ/kg/d = 1.45 MJ/t/tick at ~5-7 C, about 0.7 of
  the lab SMR (fasting fish depress metabolism; judgement).
- Duthie 1982 (J Exp Biol 97:359) [A]: flatfish standard oxygen consumption is "significantly lower
  than that of roundfish", and maximum consumption and scope are "low in comparison to salmonids".
  The full text (values for three flatfish) could not be retrieved.

SMR [D] (mg O2/kg/h; MJ/t/tick):

| Source, mass | Q10 | 7 C | 8.4 C | MJ/t/tick 8.4 C | annual mean 50 m | bottom 6.5 C |
|---|---|---|---|---|---|---|
| Fonds DOCmin fit, 50 g | 2.01 | 46.0 | 50.7 | 4.14 | 4.16 | 3.63 |
| Fonds DOCmin fit, 150 g | 2.01 | 36.1 | 39.9 | 3.25 | 3.27 | 2.85 |
| Fonds DOCmin fit, 400 g | 2.01 | 29.1 | 32.1 | 2.62 | 2.63 | 2.30 |
| Lever 1974, 10 g | 2.40 | 50.1 | 56.7 | 4.62 | 4.66 | 3.92 |
| DEB [pM] (Freitas 2010) | 2.32 | 46.6 | 52.4 | 4.27 | 4.31 | 3.64 |

### 3c. SDA and activity
- Fonds 1985 Table 8: unfed minimum respiration = 11 % of the maximum ration energy; maximum
  respiration of fully fed fish = 25 %; their difference ("score max-min") = 15 % mean (11-19 %)
  [F]. This covers SDA plus feeding activity at a full ration. DOCmax/DOCmin = 1.73 (6 C), 2.91
  (10 C), 2.07 (14 C), 2.85 (18 C) (Table 7). "Both species can increase their standard metabolic
  rate by a factor 3."
- Jobling & Davies 1980 (J Fish Biol 16:629) [A, via search]: post-feeding oxygen uptake peaks at
  "approximately twice the resting rate" and returns to rest after 24-72 h; SDA rises with ration and
  protein content.
- Lever 1974 [F]: active/rest O2 2.8 at 5.5 C and 2.4 at 10.5 C (10 g fish).
- Model reading (judgement): feeding at full ration ~2-2.5 x SMR; moving is mostly selective tidal
  stream transport (see 3e), so its cost multiplier can be lower than for haddock.

### 3d. Starvation and reserves
- Plaice store energy in the whole body; lipid rises from 3 to 22 % of dry weight from the poorest
  (K 0.7-0.8) to the best (K 1.2-1.3) condition class (Fonds 1985 Table 1) [F].
- [D] Gross whole-body energy from Fonds Table 1 (J-max per mg dry x dry fraction): K 0.75 3.0,
  K 0.85 3.45, K 0.95 4.0, K 1.05 4.7, K 1.15 5.1, K 1.25 6.1 kJ/g wet.
- Adult females [F, Rijnsdorp 1983]: "Energy values per unit fresh weight decreases from 5.81
  kjoules in December to 4.10 kjoules in March"; a 487 g female loses 43 % of her energy over the
  spawning season, 37-56 % of it as eggs; Irish Sea (Dawson & Grimm 1980, via Rijnsdorp) 53 %.
  "From December till March, the time of spawning, it does not feed. (TODD, 1911)."
- Spitz et al. 2010 (Bay of Biscay, October; via PANGAEA 945870) [F data]: plaice 27-34 cm 5.8 kJ/g.
- Fasting rate: Fonds Table 8, weight loss of unfed fish in J-physiological per W^0.78 per day:
  6 C 37, 10 C 63, 14 C 87. [D] A 150 g plaice at K 1.05 (606 kJ) falling to K 0.75 (107 g, 278 kJ)
  loses 328 kJ: ~200 d at 6 C, ~176 d at 7 C, ~118 d at 10 C, ~86 d at 14 C. K 0.75 is the poorest
  class Fonds analysed, not a measured death point (judgement).
- Model reading (judgement): lean body ~3.5-4.0 kJ/g (spent fish 4.1, poor young fish 3.0-3.5);
  reserve up to ~1700-2000 MJ/t (5.8 - 4.1 = 1.7 kJ/g in adult females; 6.1 - 4.0 in young fish).
  Protein use starts below roughly K 0.9 (judgement; lipid share is lowest there).

### 3e. Movement
- Hunter et al. 2004 (J Anim Ecol 73:377) [F]: summer feeding groups; "Northern fish often travelled
  in excess of 250 km to spawn, without the benefit of tidal transport."
- Bolle et al. 2005 (ICES J Mar Sci 62:236) [F]: "The estimated group displacement six months after
  spawning varied between 80 and 120 nautical miles" (150-220 km); ~220 nmi for Southern Bight fish
  in storage-tag data.
- [D] 150-400 km over a 1-3 month migration = 2-13 km/d, i.e. 0.5-3 cells per 6 h tick while
  migrating; summer residence is local (judgement: ~0.1 cell/tick).

## 4. Dab (Limanda limanda)

- Bagge, Nielsen & Steffensen 1995 (ICES CM 1995/J:6) [F]: dab 95-175 g fed boiled mussel to
  satiety daily at 80-90 % O2 saturation and 9.2-9.8 C, 94 days: mean 2.23 %BW/d (0-7.5); lower
  oxygen cut it to 0.49 %. At 10.5 C on raw cod flesh: 0.64-3.97 %BW/d. Gastric emptying "at 7.5°C
  ... about 49 hours (48.9) and at 10.5 C. about 30 hours (30.4)".
- Gwyther & Grove 1981 (via Bagge) [S]: ln(GET) = 1.46 + 0.68 ln M + 0.39 ln W - 0.035 T (weak
  temperature term; Q10 of evacuation 1.42 [D]); "stomach fulness is a major factor controlling
  appetite in dab".
- Hall 1987 [A]: dab fed mostly at night; daily consumption ~40 % lower than haddock.
- Pandian 1970 (Mar Biol 5:1) [A, via search]: small dab stopped feeding at 3 C and fed little at
  8 C; larger females ate most at 8 C.
- DEB (Freitas 2010) [F, fetch extraction]: dab {p_Am} 317 vs plaice 599 J/cm2/d at 15 C, TA 4000
  vs 7000 K, [pM] 29.8 for both. [D] The dab/plaice ratio of {p_Am} is 0.74 at 6 C, 0.68 at 8.4 C, 0.60 at 11.7 C.
- Cmax [D] = plaice (Fonds table) x DEB ratio, 150 g: 6 C 69, 7 C 78 (1.86 %), 8.4 C 91 (2.18 %),
  10 C 109, 11.7 C 110 kJ/kg/d; 50 g at 8.4 C 116, 400 g 73. Check: plaice 150 g at 9.5 C = 159
  kJ/kg/d x 0.65 = 103; Bagge's 2.23 %BW/d of boiled mussel equals ~100 kJ/kg/d if the mussel was
  ~4.5 kJ/g (my assumption; not stated). The two agree.
- SMR: no dab measurement was obtained (Duthie 1982 has one; inaccessible). DEB maintenance at
  8.4 C is 1.27 x plaice (same [pM], lower TA) [D]. Use plaice (judgement), range up to x1.3.
- Starvation, energy density, movement: no dab data found; use plaice for the first two and
  "sedentary" for movement (judgement).

## 5. Long rough dab (Hippoglossoides platessoides; "American plaice" in the NW Atlantic)

- MacKinnon 1972 (J Fish Res Bd Can 29:1749, doi 10.1139/f72-276) [A, Crossref abstract]: "a mature
  35 cm female stored 92 kcal during the summer period and used this energy for metabolism (72 kcal)
  and gonad maturation (20 kcal) in winter months ... in a fish with a total energy content of
  approximately 300 kcal".
- MacKinnon 1973 (J Fish Res Bd Can 30:1717) [A]: "Metabolic expenditures constitute the largest
  fraction (62%) of population energy intake and about 80% of this amount is consumed during
  summer." MacKinnon 1973 (J Exp Mar Biol Ecol 11, metabolism vs growth of American plaice) has
  measured respiration but no abstract or text was obtainable.
- [D] With 350-450 g for a 35 cm female (my assumption): whole-body energy 2.8-3.6 kJ/g; summer
  reserve 850-1100 MJ/t; winter metabolism 72 kcal over ~165 d (my assumption, Nov-Apr) = 4.1-5.2
  kJ/kg/d = 1.0-1.3 MJ/t/tick at St Margaret's Bay winter bottom temperatures (~3 C, judgement).
  With Q10 2 this is ~1.5-1.9 MJ/t/tick at 8.4 C. These are field, fasting-season values.
- Speare et al. 1997 (J Appl Ichthyol 13:171) measured routine O2 of American plaice at 2-14 C; only
  the qualitative result was found [S, via search]: it differed from the other two flatfish mainly
  at 14 C and "did not follow a parabolic response".
- Cmax: none found. Use dab (judgement). Movement: none found; sedentary (judgement).

## 6. Witch (Glyptocephalus cynoglossus)

- Moser, Ross & Sulak 1996 (MEPS 144:57) [F]: shipboard respiration of witch from the Hatteras
  middle slope at 6.0 C, unfed, mean 5.33 g: "Mean respiration rates for wolf eelpouts and witch
  flounder in oxygen saturation >10% were 0.051 and 0.067 mg O2 g-0.7 h-1, respectively."
- [D] Scaled with the paper's exponent 0.7: 150 g 14.9 mg O2/kg/h at 6 C, 400 g 11.1. With an
  assumed Q10 2.2: annual mean 1.48 (150 g) and 1.10 (400 g) MJ/t/tick; 1.26 and 0.94 at 6.5 C.
  The fish were tiny and the scaling is long; treat as "low, about half of plaice" (judgement).
- Energy density 5.6 kJ/g, 29-34 cm, October, Bay of Biscay (Spitz et al. 2010 via PANGAEA) [F data].
- Diet ~70 % polychaetes [S, NOAA guide via search]. No ration or tagging data found.

## 7. Lemon sole (Microstomus kitt)

- Energy density 5.8 kJ/g, 28-29 cm, October (Spitz et al. 2010 via PANGAEA) [F data].
- Nothing else found. Use plaice for Cmax and SMR (judgement).

## 8. One-line members (judgement or analogue)

- Grey gurnard: 4.4 kJ/g whole body (Plimmer 1921, estimated, via JNCC Report 633) [S]. Active
  benthic roundfish; use haddock/cod Cmax and SMR (judgement). Spitz 2010 gives red gurnard 8.2 kJ/g
  [F data], which looks high; not used.
- Starry ray: no data. Skates are sluggish; use plaice-level SMR or below and a low Cmax (judgement).
- Hagfish: no Myxine glutinosa SMR found. Other hagfish [S, search summary citing Cox et al. 2011
  and Forster 1990]: 19.7 mL O2/kg/h at 10 C (Pacific hagfish) and 6.93 mL O2/kg/h at 11 C (New
  Zealand hagfish) = 28 and 9.9 mg O2/kg/h [D, 1.43 mg/mL] = 2.3 and 0.8 MJ/t/tick. Very low
  metabolism and long fasts between scavenging meals (judgement: months).
- Greater weever: 5.3 kJ/g, 18-23 cm, October (Spitz 2010 via PANGAEA) [F data]. Buried ambush
  predator; flatfish-like costs (judgement).

## 9. Is the stomach limit an energy limit? (Temming & Herrmann 2003; Andersen 1999)

Temming & Herrmann 2003 (Fish Res 63:21) fed North Sea cod seven fish prey and brown shrimp; "for
fish prey, much of the variation in the prey-specific evacuation parameters was explained by
differences in the prey's energy density" [A, via search]. Andersen 1999 (J Fish Biol 54:287) found
for whiting that "evacuation time increased by a factor 1.6-1.7 with prey energy density increasing
from 3.4 to 5.6 kJ g-1" [A, via search]; a 1.65-fold rise in energy per gram is offset by a 1.6-1.7
fold slower wet-mass evacuation, so energy leaves the stomach at nearly the same rate. Andersen's
generalised gadoid model, as used by Velasco & Olaso 2000 (ICES CM 2000/Q:11) [F, citing Andersen
pers. comm.], is "dS/dt = -0.00132 L^1.44 e^0.078T E^-0.875 S^0.5 (g/h)" with E the prey energy
density: wet evacuation scales as E^-0.875, so energy evacuation scales as E^0.125, almost
independent of prey quality [D]. Its temperature term gives Q10 2.18 [D]. This supports a ceiling
set in energy (h_prey = h_ref x E_prey / E_ref) for gadoids. For flatfish no such test was found.

## 10. Temperature dependence of feeding (item 6)

| Source | Ratio Cmax 12 C / 6 C | Ratio 11.7 / 6.6 C | Note |
|---|---|---|---|
| Cod, Hansson 1996 (FB4) [D] | 1.46 | 1.37 | Q10 ~1.9 over 6-12 C |
| Plaice, Fonds 1985 table [D] | 1.98 | 1.79 | steep 6 -> 10 C, flat 10 -> 14 C |
| Plaice, Fonds 1985 fit [D] | 2.22 | 1.97 | Q10 3.8 |
| Plaice, DEB TA 7000 K [D] | 1.69 | - | |
| Dab, DEB TA 4000 K [D] | 1.35 | - | |
| Haddock juveniles, Peck 2003 [D] | 8 -> 12 C: 1.09 | - | Q10 1.25; 3.8 g fish |
| Haddock juveniles, Perez-Casanova 2009 [F] | 6 and 11 C equal (~2 %) | - | intake falls 60-80 % at 2 C |
| Gadoid evacuation, Andersen model [D] | 1.60 | - | Q10 2.18 |

Reading (judgement): between 6.6 and 11.7 C an ingestion Q10 of ~2 is supported by cod and plaice;
haddock juveniles show little change above 6 C but a strong drop below. Over the 50 m series the
cod function's annual mean equals 0.992 of its value at 8.4 C [D], so a constant ceiling at the
annual mean would carry the right annual intake capacity but miss a ~1.4-fold summer-autumn high
(October 11.7 C) vs spring low (March-April 6.6 C). Whether that seasonality matters is the caller's
call; bottom-living fish at 100-300 m (6-7 C, stable) see much less of it than the 50 m series.

## Gaps and things not obtained

- No haddock Cmax function; no adult haddock ration experiment read (Jones 1974, Jones & Hislop
  1978, Hall 1987 closed). The haddock central value is the cod function (judgement).
- Fonds et al. 1992, Duthie 1982, Priede & Holliday 1980, Jobling & Davies 1980, Jobling 1980
  (plaice starvation), Dawson & Grimm 1980, Pandian 1970, Pedersen & Hislop 2001, MacKinnon 1973
  (JEMBE), Speare et al. 1997, Tytler 1969: closed or blocked (Cloudflare on the JEB archive);
  Unpaywall found no open copies.
- No measured SMR for dab, lemon sole, long rough dab (lab), starry ray, grey gurnard, weever or
  Myxine glutinosa. No ration data for long rough dab, witch, lemon sole, gurnard, ray, hagfish,
  weever. No tagging speeds for dab, long rough dab, witch, lemon sole, haddock.
- Add-my-Pet DEB entries (which exist for most members) could not be reached from this machine;
  the DEB numbers used are from Freitas et al. 2010 (plaice, dab, cod only).
- The flatfish lab data (Fonds, Lever) are for young fish (1-330 g); 400 g values extrapolate.
- The 13.6 J/mg O2 coefficient is my assumption (Fonds 1985 used it). The DEB assimilation
  efficiency 0.8 is Teal et al. 2010's (Brett & Groves 1979).
- The script's csv keeps the cod function under the label "cod/haddock-central"; the haddock
  central value is that column x1.3 (section 1a).
