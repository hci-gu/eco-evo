# Spawning in the box - evidence trail (2026-11-18)

Tags: [F] full text read, [F-s] / [Ff] full text read through a fetch summariser (paraphrase),
[F via thesis] a paper's reprint read in Rijnsdorp's 1992 thesis, [A] abstract only, [S] secondary,
[D] derived here (script + stdout named), [J] judgement, snip = search snippet.
summary.md could not be written by the agent (the harness refused report files); its text is in
the agent's final message. Literature per member species is in the four subagent files
`lit_cod_haddock.md`, `lit_whiting_watercolumn.md`, `lit_flatfish.md`, `lit_pelagics.md` (their
tags apply); this file holds the DATRAS analysis, the model-side arithmetic and the synthesis
steps, and cites the lit files for the rest.

## 1. DATRAS NS-IBTS in and around the box [D]

Scripts: `scripts/datras_fetch.py` (download), `scripts/datras_maturity.py` ->
`tables/datras_maturity_stdout.txt`, `scripts/datras_stage_detail.py` ->
`tables/datras_stage_detail_stdout.txt`. Data: `data/README.md`.

### 1.1 Coverage

- Valid GOV hauls: region 2692, box (44G0, 44G1, 45G0, 45G1) 989: Q1 460 (Jan 223, Feb 237;
  1991-2025), Q2 68 (Apr 23, May 45; 1991-95 only), Q3 433 (Aug 191, Sep 242; 1991-2025), Q4 28
  (Oct-Nov 1991-96, no CA records).
- Almost all staged fish in the region are Swedish (SE) records.
- **Depth bias.** Box hauls: median 67 m (21-262 m); 65 % of hauls are < 100 m, while 23 % of the
  box bed area is < 100 m (`maps/depth.npz`). Beds > 260 m are not sampled. The mature share is
  therefore also given area-weighted over two strata (< 100 m: 0.232, >= 100 m: 0.768).
- **March-July is not sampled at all after 1995.** The survey windows are Jan-Feb and Aug-Sep. The
  1991-95 Q2 hauls (Apr-May) are few and staged by the old 4-stage scale.

### 1.2 Maturity scales and the staging artefact

- Scales: old IBTS 4-stage (1 immature, 2 maturing, 3 spawning, 4 spent; codes 5-6 in SE
  2008-11, 254 fish, counted mature, stage unknown), M6 (61-66) and SMSF (A-F, SE Q3 2013).
- **Outside Q1 the staging under-counts mature fish.** Q3 and Q2 ogives sit far to the right of
  the Q1 ogives. Examples of the length at 50 % mature:

  | species | Q1 | Q3 | Q2 |
  |---|---|---|---|
  | haddock | 28 cm | 38 cm | none of 499 fish at 28-36 cm |
  | whiting | 22 cm | 32 cm | |
  | Norway pout | 14 cm | never reached | |
  | sprat | | 1 % mature in Q3 | |

  Resting post-spawning adults are evidently recorded as immature (stage 1) outside the
  maturation season. [J] So every mature share below uses the **Q1 ogive** (the IBTS convention:
  ogives come from Q1) applied to each quarter's box length distribution. The direct Q3 staging
  is quoted only where the species matures in summer (hake).

### 1.3 Mature share of each species' biomass in the box [D]

Box HL numbers-at-length per hour (all valid box hauls, years pooled) x W(L) (log-log fit to the
region's CA weights) x the region Q1 ogive (2-cm classes, years pooled). Columns: Q1 unweighted
(by haul), Q1 area-weighted, Q3 area-weighted (Q1 ogive), and the Q1 period split.

| species | Q1 (hauls) | Q1 area-wt | Q3 area-wt | Q1 1991-04 / 2005-14 / 2015-25 | Q1 L50 | staged Q1 n |
|---|---|---|---|---|---|---|
| haddock | 0.29 | 0.38 | 0.44 | 0.21 / 0.31 / 0.36 | 28 cm | 5736 |
| cod | 0.55 | 0.52 | 0.54 | 0.51 / 0.54 / 0.60 | 48 cm | 11485 |
| plaice | 0.64 | 0.66 | 0.68 | 0.65 / 0.62 / 0.62 | 22 cm | 13041 |
| witch | 0.43 | 0.43 | 0.45 | 0.42 / 0.43 / 0.43 | 32 cm | 1190 |
| dab | 0.62 | 0.60 | - | (44 staged fish, outside the box) | ~20 cm | 44 |
| long rough dab | - | - | - | **no maturity records** (202 CA rows, none staged) | - | 0 |
| whiting | 0.38 | 0.43 | 0.39 | 0.33 / 0.42 / 0.41 | 22 cm | 4520 |
| saithe | 0.57 | 0.59 | 0.52 | 0.39 / 0.66 / 0.65 | 50 cm | 751 |
| hake | 0.16 | 0.18 | 0.70 (Q3 staging: 0.74) | | 42 cm (Q3) | 258 / 1018 |
| herring | 0.06 | 0.09 | 0.09 | 0.06 / 0.09 / 0.03 | 26 cm | 23808 |
| Norway pout | 0.16 | 0.20 | 0.59 | 0.13 / 0.20 / 0.31 | 14 cm | 2919 |
| sprat | 0.34 | 0.36 | 0.50 | 0.47 / 0.42 / 0.18 | 12 cm | 7661 |
| mackerel | - | - | - | 0.23 kg/h in Q1, no Q3 staging in the box | | 101 |

Notes:
- Q1 mostly measures fish in their maturing year. Pout's Q3 0.59 is next winter's first spawners
  grown into the ogive [J]. For the Feb-Apr spawning the Q1 value is the relevant one.
- Saithe: only 2.9 kg/h in Q1 against 45.5 kg/h in Q3. Saithe biomass is in the box mostly in
  summer and autumn.
- The deep stratum carries the larger mature shares (haddock 0.42 vs 0.20; whiting 0.52 vs 0.35;
  herring 0.20 vs 0.06; saithe 0.59 vs 0.28).

### 1.4 Direct evidence of spawning in the box: running and spent fish [D]

Numbers in the box, all years:

| species | Jan | Feb | Apr | May | Aug | Sep |
|---|---|---|---|---|---|---|
| cod | 88 spawning, 1 spent | 86 spawning, 14 spent | 3 spawning, 1 spent | 4 spawning, 23 spent | 20 spent | 4 spent |
| plaice | 140 spawning, 8 spent | 227 spawning, 16 spent | - | - | 2 spawning, 10 spent | 1 spent |
| haddock | 2 spawning | 10 spawning, 1 spent | (Q2: 109 staged, none spawning) | | 11 spent | 1 spent |
| whiting | 9 spawning | 5 spawning | | | 4 spawning, 1 spent | |
| saithe | | 4 spawning, 2 spent | | 14 spent | 12 spent | 6 spent |
| witch | | 5 spawning, 4 spent | | | 1 spawning, 4 spent | 1 spent |
| hake | | | | | **40 spawning, 4 spent** | |
| herring | 59 spawning, 10 spent | 13 spawning, 15 spent | | 24 spent | 16 spawning, 6 spent | 6 spawning, 1 spent |
| Norway pout | 3 spawning | 1 spawning | | | | |
| sprat, dab, mackerel, long rough dab | none | none | | | | |

Share of MATURE fish in the spawning stage, by month (region; `datras_stage_detail_stdout.txt`):

| species | Jan | Feb | Apr | May | Aug |
|---|---|---|---|---|---|
| cod | 0.15 | 0.14 | 0.08 | 0.04 (spent 0.11) | |
| plaice | 0.16 | 0.14 | | | |
| haddock | 0.02 | 0.03 | | | |
| whiting | 0.02 | 0.02 | | 0.02 | 0.03 |
| saithe | 0.01 | 0.04 | | spent 0.88 | |
| Norway pout | 0.01 | 0.01 | | | |
| hake | | | | | 0.19 (Jun 1 of 3) |

Readings [J]:
- **Cod and plaice spawn in and around the box in Jan-Feb.** About 15 % of the mature fish are
  running at the survey.
- Cod is still running in April-May, with spent fish dominating by May.
- **Hake spawns in the box in August.**
- Haddock, whiting, saithe and pout are mostly pre-spawning (developing) at the survey. Their
  peaks fall in the unsampled March-May window, so the survey neither confirms nor excludes
  spawning in the box. Saithe's 88 % spent in May (1991-95, region) shows the adults stay or
  return after spawning.
- Herring "spawning" fish in Jan (2009 and 2013, 21-24 fish, 26-28 cm) and Aug-Sep are a few
  per cent of the mature fish. They cannot be assigned to a component from these data.

### 1.5 Weight at the same length, mature minus immature [D]

E2 in `datras_maturity_stdout.txt`: residuals of the region log-log W(L) fit. Within each 2-cm
class holding both maturity classes, take mature minus immature, then average over classes
weighted by the mature n.

| species | Q1 F | Q1 M | Q3 F | Q3 M | Q1 - Q3, F / M |
|---|---|---|---|---|---|
| cod | +4.5 % | +2.2 % | +1.6 % | +1.0 % | +2.9 / +1.2 |
| haddock | +4.0 % | +1.6 % | +0.8 % | +1.0 % | +3.2 / +0.6 |
| whiting | +5.1 % | +2.3 % | +1.2 % | -2.8 % | +3.9 / +5.1 |
| plaice | +4.8 % | -1.3 % | +1.2 % | +2.0 % | +3.6 / -3.3 |
| witch | +2.5 % | +8.2 % | +6.4 % | +6.4 % | -3.9 / +1.8 |
| saithe | -3.9 % | -2.5 % | +0.1 % | +0.9 % | -4.0 / -3.4 |
| Norway pout | +5.9 % | +0.7 % | +2.6 % | -0.6 % | +3.3 / +1.3 |
| herring | +3.2 % | +4.9 % | +4.0 % | +5.5 % | -0.8 / -0.6 |

Reading [J]:
- In Jan-Feb, maturing females of cod, haddock, whiting, plaice and pout weigh 4-6 % more than
  immature fish of the same length. That is roughly the ovary mass at that date, before the
  final hydration.
- The Q3 difference is ~1 %, so the Q1-Q3 excess is 3-4 % of body weight.
- Males: 1-2 %; plaice males are lighter in Q1.
- Saithe's mature fish are lighter in Q1, consistent with a feeding stop around spawning, but the
  sample is small (64-85 mature).
- These are sampling-date values, not peak GSI. Cod, plaice and whiting peak later in Feb-Mar,
  haddock in Mar-Apr, so the peak ovary is larger. Use them as a floor check on the literature
  GSI.

## 2. Literature per member species (subagent files) [tags as in the files]

Each file has a headline table, evidence per question, full citations with DOIs, and a not-found
list. The values carried into section 4 (`tables/species_inputs.csv`) are:

| species | carried value | file | key sources |
|---|---|---|---|
| cod | G 430 (170-1000) MJ/t mature/yr, sex mean (F ~615, M ~250); feeding ~0-0.25 of normal for 50-70 d per fish | lit_cod_haddock.md | Yoneda & Wright 2004 [F] (452-950 eggs/g); Hislop & Bell 1987 [A] (23.19 kJ/g dry egg); Fordham & Trippel 1999 via Dutil et al. 2006 [S] ("cease feeding for as much as 70 days"); Kjesbu et al. 1991 [A]; Michalsen et al. 2008 [F-s] (Lofoten fullness 0.24 of the Barents Sea) |
| haddock | G 270 (100-500); "practically cease to take food" through spawning, 33-37 d per female | lit_cod_haddock.md | Wright et al. 2011 [F]; Filina et al. 2009 [F]; Homans & Vladykov 1954 [A]; Hislop et al. 1978 [A] (captive fish fed through spawning: conflicting) |
| whiting | G ~800 (500-1050); no feeding data (cod proxy) | lit_whiting_watercolumn.md | FishMap / Hislop & Hall 1974 (1700 eggs/g, 14 wk batches) [F]; Hislop 1975 [A]; Cooper 1983 [F] (gutted carcass -16 % F and M; condition min Apr-Jun, max Aug-Sep) |
| saithe | G ~400 (260-690); leaves to spawn at the shelf edge Feb-Mar | lit_whiting_watercolumn.md | Storozhuk et al. 1974 [F]; Skjaeraasen et al. 2017 [F] (captive growth ceased during spawning) |
| plaice | G ~750 (650-850); ~50 % of gonad energy from reserves; ripe F feeding incidence 0.06, M 0.29 of immature; F -24 % weight / -44 % energy | lit_flatfish.md | Rijnsdorp & Ibelings 1989, Rijnsdorp 1989, 1990 [F via thesis] |
| dab | G 430-850; females keep feeding (food pays the gonad) | lit_flatfish.md | Saborowski & Buchholz 1996 [F] |
| long rough dab | G proxy 400-800 [J]; 31 % feeding Dec-Jan vs > 85 % Apr-Aug [A]; spawns Mar-Apr (Clyde) [A] | lit_flatfish.md | Ntiba & Harding 1993 [A]; Bagenal 1957 [A] |
| witch | G 150-350; spawns May-Aug inside the feeding season | lit_flatfish.md | Burnett et al. 1992 [F] |
| hake | 40 spawning fish in the box in August [D]; Skagerrak otherwise a nursery [F] | lit_whiting_watercolumn.md | Werner 2015 [F] |
| spurdog | pups Sep-Dec at Hvaler [F]; G ~75-250 per t of both sexes [D/J] | lit_whiting_watercolumn.md | |
| herring | box herring ~91-94 % immature; NSAS juveniles, WBSS adults spawn outside; local Skagerrak spring spawners on the coast | lit_pelagics.md | Bekkevold et al. 2023 [F]; Slotte 1999 [A], Slotte & Fiksen 2000 [F]; Warren et al. 2026 [F] |
| Norway pout | builds gonads in the box Oct-Feb, "large scale spawning migration out into the North Sea"; few spawners in the Skagerrak; spawning mortality Q1-Q2 | lit_pelagics.md | Raitt 1968 [F]; Lambert et al. 2009 [Ff]; Sparholt et al. 2002 [A] |
| mackerel | income breeder, "feed intensively during the spawning season"; Skagerrak spawning small, Norwegian side | lit_pelagics.md | Jansen et al. 2021 [F]; Iversen 1977 [F]; Jansen & Gislason 2013 [F] |
| sprat | probably spawns in or near the box May-Jul [snip]; no energy data | lit_pelagics.md | |

Disagreements between files and this analysis, resolved here:
- **Whiting mature share.** lit_whiting_watercolumn.md gives 0.65 (5-cm bins, box fish, a crude
  W ~ L^3). This file's region and box-only Q1 ogives both give 0.39-0.43
  (`tables/ogive_box_vs_region_stdout.txt`), 0.43 area-weighted. The carried value is 0.43, and
  the range is 0.38-0.65.
- **Flatfish mature shares.** lit_flatfish.md uses a box defined by shoot position and sex-specific
  ogives, giving plaice F 0.47-0.53 and M 0.75-0.78. Here plaice is 0.64-0.66, sexes pooled.
  The difference does not matter at the group level (plaice 0.038 of the group).

## 3. Production booking: is the annual P net of the spawning loss? [D]

`scripts/sms_spawning_dip.py` -> `tables/sms_spawning_dip_stdout.txt` (SMS 2023 key run, the
file `research_notes/Fiskgrupper i rutan/benthivores/data/sms2023/summary.out` used for P in
`Oppen population/scripts/closure_249.py`).

**The quarterly increments telescope.** For each cohort, the four quarterly ln-increments sum
exactly to the Q1(a) -> Q1(a+1) increment (cod age 3: 0.690 and 0.690; herring age 4: 0.155 and
0.155). P (closure_249) is the biomass-weighted sum of the same quarterly increments around the
year, so it is the net annual change in weight at the same seasonal phase. Whatever is shed or
lost between two Q1s is already netted out of P.

**The SMS gadoid weights do not resolve spawning at all.** Cod, whiting and saithe weights rise
in every quarter from Q1 to Q2. Haddock and saithe ages 3-4 show a small Q4 -> Q1 dip (-0.02 to
-0.08). Herring shows the autumn spawning loss Q3 -> Q4 (-0.13 to -0.21) and a winter Q4 -> Q1
loss; mackerel loses weight Q3 -> Q1. In the DATRAS weights, maturing females in Jan-Feb weigh
+4-6 % more than immature fish of the same length (section 1.5), which is the gonad mass the SMS
gadoid weights do not show quarter by quarter.

Consequence [J]:
1. **Tonnes.** A spawning term that removes tonnes (gonad wet mass shed) while g still pays P
   double counts the loss, because P is already net of it. If tonnes are shed, g must pay gross
   P + shed mass.
   - Shed mass per t of mature fish per year [D]: cod / haddock / whiting females lose ~15-30 % of
     body weight and males ~10-15 %. Cod is captive F 29 %, M 14 % (snip); plaice F 24 %, M 11 %
     [F via thesis]. Part of that is non-gonad tissue.
   - Times the mature share in the box (~0.4 of benthivores), that is ~0.05-0.08 t per t of
     group, i.e. ~7-11 % of P = 0.74.
2. **Energy.** A drain from the reserve that does not remove tonnes leaves P untouched. It is,
   however, an energy cost the derived g and the booked ration do not contain. The field ration
   (Q/B) behind the booking includes it.
   - Size [D]: ~180 MJ/t of benthivores per year, against the field closure ration ~4.94 t/t/yr x
     ~2200 MJ/t = ~10 900 MJ/t/yr ("Ransonen i energi"). That is ~2 % of the annual ration.
   - So the gonad energy cannot by itself explain a +28 % ration in the scratch test; the size
     of that drain is not known here.

## 4. Group values [D]

`scripts/group_energy.py tables/species_inputs.csv` -> `tables/group_energy_stdout.txt`.
Per member: share x in_box x mature x G, with the reserve-funded share and the feeding cut.

- `in_box` is the share of the species' mature box fish whose gonad build AND spawning-period
  fast happen in the box [J from sections 1-2]: cod 1, haddock 0.7, whiting 0.8, flatfish 1,
  saithe 0, hake 1, spurdog 0.5, herring 0, pout 0.5, sprat 1, mackerel 0.
- The cut is fraction x days x the fasting deficit (field metabolism ~20 MJ/t/d for gadoids,
  i.e. benthivore field metabolism 8891 MJ/t/yr = 24 MJ/t/d less activity; 15 for flatfish).

| group | mature share spawning in the box | gametes MJ/t group/yr (lo / mid / hi) | in ME | reserve-funded gonad, ME | feeding cut, ME | reserve drawdown, ME |
|---|---|---|---|---|---|---|
| benthivorous_fish | 0.40 | 97 / 181 / 322 | 0.054 / 0.10 / 0.18 | 0.014 / 0.049 / 0.13 | 0.17 (0.07-0.27 [D]: cut 0.75-1.0 x 35-70 d at 15-24 MJ/t/d, i.e. 0.39-1.65 x the mid) | 0.22 mid (0.08-0.40 with every bound together) |
| whiting | 0.34 | 172 / 275 / 361 | 0.086 / 0.14 / 0.18 | 0.034 / 0.083 / 0.14 | 0.07 (0-0.14, proxy) | 0.10 / 0.15 / 0.21 |
| predatory_fish | 0.06 | 10 / 19 / 30 | 0.005 / 0.009 / 0.015 | <= 0.007 | 0 | <= 0.007 |
| planktivorous_fish | 0.04 | 21 / 34 / 51 | 0.004 / 0.007 / 0.010 | <= 0.006 | 0 | <= 0.006 |
| mackerel | 0 | 0 | 0 | 0 | 0 | 0 |

**Confirmed vs probable.** Running fish in the box confirm only cod (0.163 of the group) and
plaice (0.025): 0.19 confirmed. Haddock (0.103 at in_box 0.7, 0.147 at 1), long rough dab
(0.076), dab (0.027) and witch (0.007) are probable or unresolved. That gives 0.19 confirmed,
0.40 with the probable members.

**Energy per tonne of the species in the box** (G x mature, MJ/t/yr) [D]:

| species | MJ/t | note |
|---|---|---|
| haddock | 103 | |
| cod | 224 | |
| long rough dab | 360 | proxy |
| dab | 384 | |
| plaice | 495 | |
| witch | 108 | |
| whiting | 344 | |
| saithe | 236 | spent outside |
| hake | 315 | |
| herring | 99 | outside |
| Norway pout | 160 | |
| sprat | 288 | proxy |
| mackerel | ~1600 | not shed in the box |

Composition of the benthivore value:
- Cod: 0.163 of the group biomass spawns in the box; 70 MJ/t gametes; 173 MJ/t cut.
- Haddock: 0.103; 28 + 74.
- Long rough dab: 0.076; 46 + 34 (low confidence).
- Plaice: 18 + 25. Dab: 17 + 0. Witch: 2 + 0.
- Starry ray, hagfish and the other members: no values (0.075 of the group).

## 5. Plausibility against start_fill (the test; never an input) [D/J]

**What the benthivore test series is.**
- `start_fill` (0.76 Jan-Feb -> 0.33 May-Jul -> 0.81 Sep-Nov) is the product of two pieces
  (`Svältgräns och startreserv/model/derive_u.py`):
  - **Lipid share.** NIFES haddock liver lipid % (Skagerrak + North Sea, 2015-19) [F]. The
    Skagerrak sub-sample (autumn and winter fish only) has a mean length of 45 cm, 0.92 mature by
    the Q1 ogive. The lengths of the May-Jul fish (North Sea) were not seen, so the series is
    "mostly adult" at best, not established as mature.
  - **HSI.** A [J] HSI shape from NE Arctic immature/maturing fish: 6 % -> 3.5 %.
- Decomposition of the 0.43 ME drop [D]:
  - the measured lipid-share step (58 -> 43.5 %) alone gives 0.19 ME;
  - the [J] HSI step alone gives 0.32 ME;
  - in log terms the HSI step carries ~65 % of the drop.
- Only 3 seasonal bins are data (Jan-Feb, May-Jul, Sep-Nov). The Mar, Apr and Aug values are
  interpolation, so timing can be judged at seasonal resolution only.

**Size, per tonne of mature haddock (one test).**
- Reserve-funded gametes: 135 (20-400) MJ.
- Spawning fast: 0.9 x 40 d x 20 MJ/t/d = 720 (370-1440) MJ, of which only the LIVER part counts
  against this liver series. Fasting gadoids draw on muscle as well: whiting gutted carcass -16 %
  at spawning [F]; cod liver and muscle depleted with spawning [A]; plaice lipid 55-75 % of
  spawning metabolism [F via thesis]. Liver share of the fast 0.5 (0.3-0.7) [J].
- Implied liver drawdown ~495 MJ = **0.28 ME (0.07-0.78)**, against 0.43 ME in the series
  (0.19 of it measured, 0.24 the [J] HSI step).
- Verdict [J]: **same order of magnitude, inside the range; the test cannot be sharper.** With
  the fast charged 100 % to the liver it would be 0.48 (0.22-1.0).

**Group level.**
- The model applies the series to the whole group.
- If only the mature fish draw down, the group target is 0.43 x 0.40 = 0.17. If the whole group
  draws down, it is 0.43. **The target is the bracket 0.17-0.43.** Nothing found decides it: the
  NE Arctic immature HSI (Tronbøl) is no lower than autumn in March, but has no data after April.
- Implied for the group, with the liver share of the fast 0.5 (0.3-0.7): gametes from the
  reserve 0.05 + fast 0.08 = **0.13 ME (0.03-0.32)**.

**Timing (consistent at seasonal resolution only).** Day lengths at 58.15 N,
`tables/day_length_stdout.txt`.

| phase | months | day length |
|---|---|---|
| gonad build | Oct/Nov -> Feb | 11.0 h - -> 8 h + |
| field fill flat | Sep-Feb (0.81 -> 0.76) | |
| benthivore spawning | cod Jan-Apr (peak Feb-Mar); haddock Mar-May; plaice late Dec-Mar | 6.3 -> 17.4 h +, centre ~11-13 h + |
| Jan-Feb -> May-Jul bins | 0.76 -> 0.33 | |

- **Gonad build.** It is not evidence for food funding: the flat Sep-Feb is part of the test.
  Independent evidence that gonads can be food-funded:
  - Yoneda & Wright 2005 [A]: maturing captive females grew as immatures did;
  - Tronbøl 2022 (fish.md) [F]: NE Arctic maturing female haddock hold HSI 5-7 % through April,
    while spent females are at 0.88. The liver is spent at the end of spawning, not during the
    build.
  The liver share of the gonad is unmeasured for gadoids; plaice ~50 % [F via thesis].
- **The trough does NOT come from spawning.**
  - The fast ends with spawning (cod Apr, haddock May). Fish then "feed voraciously for a few
    weeks" (haddock [A]), and somatic condition improves within ~1 month (cod [F]).
  - Yet the liver needs 3-6 months:
    - cod HSI is lowest in May, highest in autumn [A];
    - haddock liver lipid is low May-Jul, back Sep-Nov [F];
    - plaice condition is lowest Mar-Apr, maximum Oct-Nov, despite immediate feeding [F via
      thesis];
    - whiting condition is minimum Apr-Jun, maximum Aug-Sep [F].
  - A reserve_room fish refills within weeks. Candidates [J], untested:
    - (a) soma before liver after spawning;
    - (b) a food-limited ration (field feeding level 0.22-0.53 of Cmax, AGENTS.md section 5).

**Material growth during a fast** [J]. An intake cut also stops material growth (new tonnes
only from this tick's food). SMS P already includes the field fast, so g would have to pay P in
the remaining feeding time:
- 0.4 mature x ~50 d / 365 = ~5 % of the group's feeding days;
- so g x ~1.06 (an estimate, not computed in the engine).

**Planktivores.** Spawning in the box is 0.002-0.006 ME, against a field swing of 0.76 ME
(0.96 in Aug -> 0.20 in Apr). The planktivore cycle is overwintering and herring's set point
(section 271), not spawning in the box.

**Water column and mackerel.** ~0 and 0. Saithe leave to spawn and are scarce in the box in
winter. Mackerel are income breeders that feed hardest while spawning.

## 6. Not found (all files)

- **No liver share of gonad energy, as a number, for any gadoid.** Plaice ~50 % [F via thesis];
  dab <= 22-47 % [D].
- **Intake measurements:**
  - no captive intake measurement during spawning for any species in the area;
  - cod: Fordham & Trippel 1999 was not obtained (read via Dutil 2006 [S] and an abstract);
  - haddock: conflicting (Homans & Vladykov [A] vs Hislop et al. 1978 [A]);
  - whiting, saithe and long rough dab: none.
- **Gonad values:**
  - North Sea female cod GSI;
  - testis energy density for any gadoid;
  - haddock egg dry mass;
  - whiting monthly GSI;
  - long rough dab GSI, fecundity and L50;
  - witch and dab ovary energy;
  - sprat, blue whiting and horse mackerel gonad energy;
  - starry ray egg-case energy;
  - hagfish per-tonne output.
- **Long rough dab** is 0.126 of the benthivores and has no maturity records in DATRAS in the
  region.
- **Unsampled survey window.** Haddock, whiting, saithe and pout peak inside March-July, which
  the survey has not sampled since 1995. Egg surveys for haddock and whiting in the eastern
  Skagerrak were not found.
- **Immature fish.** No liver series for immature haddock or cod in the Skagerrak, so the test
  of whether immature fish share the spring drawdown is open.
- **Sex ratio** in the box was not computed (the CA records are length-stratified).
