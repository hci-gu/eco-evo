# Spawning in the box: which fish, how much, when, and what happens to feeding (2026-11-18)

Written by the research subagent, saved by the main agent. Research only; no library, engine or test files were touched.

Files:
- `evidence.md`: the full evidence trail.
- `lit_cod_haddock.md`, `lit_whiting_watercolumn.md`, `lit_flatfish.md`, `lit_pelagics.md`: notes from four literature subagents, with full citations and DOIs.
- `scripts/` and `tables/`: the DATRAS NS-IBTS analysis (1991-2025, the four box rectangles plus the surrounding Skagerrak/Kattegat) and the group arithmetic.
- `data/README.md` and `papers/README.md`: sources and licences. Only the CC BY PDFs may be committed (Bekkevold 2023, Jansen & Gislason 2013, Mollet 2023).

Tags:
- [F] full text read.
- [F-s]/[Ff] full text read through a fetch summariser (paraphrase).
- [F via thesis] a paper's reprint read in Rijnsdorp's 1992 PhD thesis.
- [A] abstract only.
- [S] secondary (a claim about one paper read in another).
- [D] derived here.
- [J] judgement.
- snip: search snippet only.

Day lengths are at 58.15 N; "+" means days are lengthening, "-" shortening (`tables/day_length_stdout.txt`):

| Date | Day length |
|---|---|
| Jan 1 | 6.3 h + |
| Feb 1 | 8.0 h + |
| Mar 1 | 10.2 h + |
| Apr 1 | 12.9 h + |
| May 1 | 15.4 h + |
| Jun 1 | 17.4 h + |
| Jun 21 | 17.9 h (maximum) |
| Jul 1 | 17.8 h - |
| Aug 1 | 16.1 h - |
| Sep 1 | 13.6 h - |
| Oct 1 | 11.0 h - |
| Nov 1 | 8.5 h - |
| Dec 21 | 6.1 h (minimum) |

## 1. Headline

1. **Only the benthivores, and probably whiting, spawn enough in the box to matter.**
   - Running fish in the box hauls confirm cod (Jan-Feb, and Apr-May) and plaice (Jan-Feb). These two are 0.19 of the benthivore biomass.
   - Adding the probable members (haddock, long rough dab, dab, witch), mature fish that spawn in or next to the box are about 0.40 of the group.
   - In whiting, about 0.34 of the biomass is mature and probably spawns here.
   - The other groups are negligible: planktivores ~0.04, water column ~0.06, mackerel 0.
     - Herring in the box are 91-94 % immature.
     - Norway pout builds gonads here and then leaves.
     - Saithe leaves to spawn.
     - Mackerel are income breeders.
2. **The fast probably matters more for the reserve than the gonad energy. The evidence for the fast is the weakest link.**
   - Gametes are 100-1000 MJ per tonne of mature fish per year, and perhaps half of that comes from the reserve.
   - A spawning fast costs 400-1400 MJ per tonne of mature fish in maintenance. On the mid values the fast is about three quarters of the benthivore drawdown.
   - The fast rests mainly on [A] and [S] sources:
     - Cod eat ~0-0.25 of normal for 50-70 days (Fordham & Trippel 1999, read via Dutil 2006 [S] and the abstract). The Lofoten spawning-ground stomach fullness is 0.24 of the Barents Sea value [F-s].
     - Haddock "practically cease to take food" (Homans & Vladykov 1954 [A]). Captive haddock fed through spawning (Hislop 1978 [A]), which conflicts.
     - Ripe plaice females are almost never found with food in their stomachs at stations where immature fish do feed [F via thesis].
   - This would explain the scratch test: a pure drain is eaten back because model fish keep eating, and wild spawners largely do not.
   - Dab, witch and mackerel feed through spawning.
3. **Gonads CAN be food-funded; the liver's share is unmeasured for gadoids.**
   - Captive maturing cod grew as fast as immature cod through vitellogenesis [A].
   - NE Arctic maturing female haddock keep HSI at 5-7 % through April, while spent females are at 0.88 [F]. The liver is spent at the end of spawning, not during the build-up.
   - In plaice about 50 % of gonad energy comes from reserves [F via thesis].
4. **Size test against `start_fill`: the same order of magnitude, and no sharper than that** (section 5).
5. **The May-Jul trough is not a spawning effect.**
   - Spawners resume feeding at once and somatic condition recovers within ~1 month, yet the liver takes 3-6 months to refill.
   - Some post-spawning mechanism is needed. Not resolved here.
6. **The somatic production P is net of the spawning loss.**
   - P is the cohort-followed sum of quarterly ln-weight increments. Over a year this telescopes exactly to the Q1 -> Q1 increment.
   - Removing tonnes at spawning while g pays P would therefore double count.

## 2. Per member species

Column definitions:
- **"Mature":** the share of the species' biomass in the box that is mature. It uses the Q1 ogive (the region's Jan-Feb staging, 2-cm classes, 1991-2025) applied to the box length composition x W(L), area-weighted over haul depth (< 100 m = 0.23 of the box bed, >= 100 m = 0.77).
  - Staging outside Q1 records resting adults as immature, so the Q1 ogive is used throughout (evidence.md 1.2).
  - Box-only and region ogives agree within 0.04 (`tables/ogive_box_vs_region_stdout.txt`).
- **"G":** gamete energy, MJ per t of MATURE fish per year, sex-averaged: central value (range).
- **"MJ/t box":** G x mature, i.e. MJ per t of the species in the box.

### Benthivores

| Species (share) | Spawns in box? | Mature | G | MJ/t box | Timing (day length) | Feeding during spawning (fraction x duration per mature fish) | Confidence |
|---|---|---|---|---|---|---|---|
| haddock 0.386 | Builds gonads in the box; spawning there unresolved. 12 running fish in Jan-Feb; the Mar-May peak is not sampled [D] | 0.38 (0.29-0.44) | 270 (100-500) | 103 | Build-up from late Oct (10 h -). Spawning Mar-May (10.2 h + to 17.4 h +), peak Apr (12.9-15.4 h +). Liver back Sep-Nov (13.6 h - to 7.4 h -) | "practically cease to take food" [A], ~0-0.1 x 33-37 d per female; conflicts with Hislop 1978 [A] | in box medium; G medium; feeding low-medium |
| cod 0.314 | Yes, in or right next to the box. 174 running fish in Jan-Feb, 7 in Apr-May, spent fish dominate by May [D]; eggs in the Bohuslän fjords [F-s] | 0.52 (0.52-0.56) | 430 (170-1000) | 224 | Build-up Oct-Nov (11.0 h - to 8.5 h -). Spawning Jan-Apr (6.3 h + to 14.1 h +), peak Feb-Mar (8.0-12.9 h +). Liver back after 4-6 months, Sep-Oct (13.6 h - to 9.9 h -) | ~0-0.25 x 50-70 d [S/A]; Lofoten fullness 0.24 [F-s] | in box high; G and feeding medium |
| long rough dab 0.126 | Probably: eggs in the Bohuslän fjords just east (snip). No maturity records in DATRAS | 0.6 (0.4-0.8) [J] | ~600 (400-800), proxy [J] | ~360 | Spawning Mar-Apr (10.2 h + to 15.4 h +; Clyde [A]); "slow recovery" | 31 % feeding in Dec-Jan vs > 85 % in Apr-Aug [A]; seasonal or spawning-related is not resolved | low |
| dab 0.045 | Probably [D] | 0.60 (only 44 staged fish) | 640 (430-850) | 384 | Build-up Nov-Mar. Spawning Jan-Sep, broad peak Feb-Apr (8.0 h + to 15.4 h +) | Females keep feeding; winter food pays for the gonad [F] | medium |
| plaice 0.038 | Yes: 367 running fish in Jan-Feb [D] | 0.66 | 750 (650-850); F 1190-1590, M <= 150 | 495 | Build-up from Oct (11 h -). Spawning late Dec-Mar (6.1 h to 12.9 h +); females ~5 weeks, late Jan to early Mar (7.3-10.2 h +). Condition minimum Mar-Apr, maximum Oct-Nov | Ripe females feed at 0.06 and males at 0.29 of the immature feeding incidence, for 60-100 d; ~50 % of gonad energy from reserves [F via thesis] | high |
| witch 0.017 | Weak: 14 running or spent fish | 0.43 | 250 (150-350) | 108 | Spawning May-Aug (15.4 h + to 13.6 h -) | Spawns inside the feeding season [F, timing] | low |
| starry ray, hagfish, others 0.075 | Starry ray lays egg cases all year (snip); hagfish 11-38 eggs, no season (snip) | - | not found | - | - | not found | - |

### Whiting

| Species (share) | Spawns in box? | Mature | G | MJ/t box | Timing (day length) | Feeding during spawning | Confidence |
|---|---|---|---|---|---|---|---|
| whiting 1.0 | Probably, early in the season: 14 running fish in Jan-Feb, 4 in Aug [D]; the spawning-ground model ends at 10.04 E [D] | 0.43 (0.38-0.65) | 800 (500-1050); F 900-1800, M 100-300 | 344 | Build-up Nov-Feb (8.5 h - to 8 h +). Spawning Feb-Jun, peak Apr (8.0 h + to 17.9 h; peak 12.9-15.4 h +). Condition minimum Apr-Jun, maximum Aug-Sep (16.1 h - to 12.4 h -) | Not found; cod proxy only. Gutted carcass -16 % in both sexes [F] | in box medium; G low-medium; feeding low |

### Water column

| Species (share) | Spawns in box? | Mature | G | MJ/t box | Timing (day length) | Feeding during spawning | Confidence |
|---|---|---|---|---|---|---|---|
| saithe 0.797 | No: builds gonads in or near the box, then leaves for the shelf edge. Q1 biomass in the box is ~0.06 of Q3 [D] | 0.59 (Q1) | 400 (260-690) | 236 (spent outside) | Build-up from late Oct (10 h -). Spawning Feb-Mar, outside the box (8.0-12.9 h +) | Captive fed fish: growth ceased during spawning [F]; intake not measured | medium |
| spurdog 0.097 | Probably: pups Sep-Dec, 70 km north [F] | ~0.5 [J] | ~150 (75-250) per t of both sexes | 75 | Pupping Sep-Dec (13.6 h - to 6.1 h) | not found | low |
| hake 0.048 | Yes, a summer minority: 40 running fish in Aug [D]; the area is otherwise a nursery [F] | 0.70 (Q3) | proxy 450 [J] | 315 | Jul-Aug (17.8 h - to 13.7 h -) | not found | low-medium |
| anglerfish, ling, velvet belly, pollack 0.058 | not found | - | - | - | - | - | - |

### Planktivores

| Species (share) | Spawns in box? | Mature | G | MJ/t box | Timing (day length) | Feeding during spawning | Confidence |
|---|---|---|---|---|---|---|---|
| herring 0.554 | No. North Sea autumn spawners here are juveniles; western Baltic spring-spawner (WBSS) adults spawn outside; local Skagerrak spring spawners are on the coast (Bekkevold 2023 [F]) | 0.09 | 1100 (900-1300) | 99 (outside) | WBSS Mar-May, outside (10.2-17.4 h +). North Sea autumn spawners Aug-Oct, outside (16.1-9.9 h -) | Spring spawners do not feed during maturation (Slotte, other stocks [F/A]) | medium |
| Norway pout 0.35 | Builds gonads in the box, then mostly leaves ("large scale spawning migration out into the North Sea" [F]); most die after spawning [A] | 0.20 (Q1) | ~800 (500-1200), females only | 160 | Build-up Oct-Feb (11 h - to 9 h +). Spawning mid-Feb to Apr (9.1-14.1 h +) | Feeds all winter [S]; "low feeding intensity" was called inconclusive by the author [F] | where: medium; energy: low |
| sprat 0.021 | Probably, in or near the box (snip) | 0.36 | proxy 800 [J] | 288 | May-Jul (15.4 h + to 16.1 h -) | not found | low |
| blue whiting 0.075 | No evidence; adults in the box Aug-Sep only | - | - | - | - | - | low |

### Mackerel

| Species (share) | Spawns in box? | Mature | G | MJ/t box | Timing (day length) | Feeding during spawning | Confidence |
|---|---|---|---|---|---|---|---|
| mackerel 0.941 | Small share at most; Skagerrak spawning is "relatively small" and on the Norwegian side [F] | ~0.8 of Aug-Sep biomass is >= 28 cm | 2000 (1300-2700), females only | ~1600 (not shed in the box) | Late May-Jul, peak late Jun to early Jul (17.4 h + to 17.9 h to 17.8 h -) | None: they "feed intensively during the spawning season" (income breeder) [F] | medium-high |
| horse mackerel 0.059 | No box evidence | - | - | - | - | Income breeder (snip) | low |

DATRAS facts [D] (evidence.md 1.4-1.5):
- Share of mature fish that were running in Jan-Feb: cod 0.14-0.15, plaice 0.14-0.16, haddock 0.02-0.03, whiting 0.02, pout 0.01.
- By May (1991-95), 0.88 of mature saithe were spent.
- Maturing females in Jan-Feb weigh 4-6 % more than immature fish of the same length (cod, haddock, whiting, plaice, pout); in Aug-Sep the difference is ~1 %. This is the ovary mass before final hydration, so it is a floor check on GSI.

## 3. Per group, biomass-weighted [D]

Source: `scripts/group_energy.py tables/species_inputs.csv` -> `tables/group_energy_stdout.txt`.

Definitions:
- **Spawning share** = sum share x in_box x mature.
- **in_box** is the share of the mature fish whose gonad build-up AND spawning fast both happen in the box [J]: cod 1, haddock 0.7, whiting 0.8, flatfish 1, saithe 0, hake 1, spurdog 0.5, herring 0, pout 0.5, sprat 1, mackerel 0.
- **Feeding cut** = fraction x days x a fasting deficit of 20 MJ/t/d for gadoids (field metabolism of 24 MJ/t/d, less activity) and 15 MJ/t/d for flatfish. The cut is shown for the whole reserve (liver plus carcass) and, in brackets, for its liver part at a liver share of 0.5 (0.3-0.7) [J].

| Group (ME) | Spawning share of group | Gametes, MJ/t group/yr | As fraction of ME | Gametes from reserve (ME) | Feeding cut (ME) | Total drawdown (ME), whole / liver | Timing (day length) |
|---|---|---|---|---|---|---|---|
| benthivorous_fish (1800) | 0.40 (0.19 confirmed) | 181 (97-322) | 0.10 (0.05-0.18) | 0.05 (0.01-0.13) | 0.17 (0.07-0.27); liver 0.08 | 0.22 (0.08-0.40) / 0.13 (0.03-0.32) | Build-up 11 h - to 8 h +. Fast and spawning 8 h + to 17 h +, centre 11-13 h +. Recovery to ~12 h - |
| whiting (2000) | 0.34 | 275 (172-361) | 0.14 (0.09-0.18) | 0.08 (0.03-0.14) | 0.07 (0-0.14, cod proxy) | 0.15 (0.10-0.21) | Spawning 8 h + to 17.9 h, peak 13-15 h + |
| predatory_fish (2000) | 0.06 | 19 (10-30) | 0.009 | <= 0.007 | 0 | <= 0.007 | Hake 17.8-13.7 h -; spurdog 13.6 h - to 6.1 h |
| planktivorous_fish (4990) | 0.04 | 34 (21-51) | 0.007 | <= 0.006 | 0 | <= 0.006 | Pout build-up 11 h - to 9 h +; spawning outside the box |
| mackerel (6980) | 0 | 0 | 0 | 0 | 0 | 0 | - |

Benthivore composition, MJ/t of group per year (gametes + cut):
- cod 70 + 173
- haddock 28 + 74
- long rough dab 46 + 34 (low confidence)
- plaice 18 + 25
- dab 17 + 0
- witch 2 + 0

**Reading [J].** Only the benthivores and whiting carry a spawning term worth modelling. In both it is mainly an intake cut for the mature fraction around that fraction's own spawning window. The gonad energy is secondary.

## 4. P and double counting [D/J]

- **P is net of spawning.** P (closure_249) is the biomass-weighted sum of cohort-followed quarterly ln-weight increments (SMS 2023, 2013-22). For each cohort the four increments sum exactly to the Q1 -> Q1 increment (`tables/sms_spawning_dip_stdout.txt`). So the annual P is net of anything shed or lost within the year.
- **SMS gadoid weights do not show spawning.** Cod, whiting, haddock and saithe weights rise from Q1 to Q2. Herring (Q3 -> Q4: -0.13 to -0.21) and mackerel (Q3 -> Q1) do show their losses.
- **Removing tonnes at spawning double counts while g pays P.** If gonad tonnes are shed, g must pay gross P plus the shed mass.
  - Shed mass: ~15-30 % of body weight in females and ~10-15 % in males, part of it not gonad (cod captive F 29 %, M 14 % snip; plaice F 24 %, M 11 % [F via thesis]).
  - Times the mature share, that is ~0.05-0.08 t per t of benthivores, about 7-11 % of P = 0.74.
- **An energy-only drain from the reserve leaves tonnes untouched.** It is an energy cost that is not in the derived g or the booked ration: ~180 MJ/t/yr for the benthivores, ~2 % of the field annual ration (~10 900 MJ/t/yr).
- **An intake cut also stops material growth** (new tonnes come only from the current tick's food). SMS P already includes the field fast, so g would have to pay P in the remaining feeding time. That is ~0.4 mature x ~50 d / 365, about 5 % of the group's feeding days, so g x ~1.06 [J; not computed in the engine].

## 5. Plausibility against start_fill (a test, not a fit) [D/J]

**What the test series is.** The benthivore `start_fill` (0.76 -> 0.33 -> 0.81) is the product of two inputs:
- **NIFES haddock liver lipid %** (Skagerrak + North Sea) [F]. The Skagerrak autumn and winter fish average 45 cm, 0.92 mature by the Q1 ogive. The lengths of the May-Jul North Sea fish were not seen, so "mostly adult" is not established.
- **A [J] HSI shape from NE Arctic immature/maturing fish** (6 % -> 3.5 %).

Of the 0.43 ME drop, the measured lipid step alone gives 0.19 ME and the [J] HSI step alone 0.32 ME (in log terms the HSI step is ~65 % of the drop). Only three seasonal bins are data (Jan-Feb, May-Jul, Sep-Nov); Mar, Apr and Aug are interpolated. Whiting's series is the same haddock proxy.

| Check | Implied | Field | Verdict |
|---|---|---|---|
| Size per t of mature haddock (the one real test) | Gametes from reserve 135 (20-400) MJ, plus the fast's liver part 0.5 (0.3-0.7) x 720 (370-1440) MJ = 0.28 ME (0.07-0.78). Charging the whole fast to the liver gives 0.48 | 0.43 ME (0.19 measured + 0.24 [J] HSI) | Same order of magnitude and inside the range. No sharper test is possible |
| Group level | 0.13 ME liver (0.03-0.32); 0.22 whole reserve | 0.17 if only the mature 0.40 draw down, 0.43 if the whole group does | The target is a 0.17-0.43 bracket. Nothing found decides whether immature fish draw down: NE Arctic immature HSI in March is no lower than in autumn, and there are no data after April |
| Timing of the fall | Fasts Feb-May (8 h + to 17 h +) | 0.76 (Jan-Feb) to 0.33 (May-Jul) | Consistent at seasonal resolution |
| Trough duration | Fast ends with spawning (Apr-May); feeding then "voracious" [A]; somatic condition back in ~1 month [F] | 0.33 held through May-Jul (15-18 h, + then -), refill by Sep-Nov | **Not explained by spawning.** Every gadoid and flatfish with data needs 3-6 months to refill the liver: cod HSI minimum in May, maximum in autumn [A]; haddock lipid low May-Jul [F]; plaice and whiting condition minimum Mar-Jun, maximum Aug-Nov [F] |
| Whiting | 0.15 ME (0.10-0.21) | 0.39 (haddock proxy) | Weak test |
| Planktivores | <= 0.006 ME | 0.76 swing (0.96 in Aug to 0.20 in Apr) | The planktivore cycle is not spawning in the box |

Consequences for a model term [J]. These are not recommendations; the user decides.
1. **Intake cut.** The evidence supports cutting intake to ~0-0.25 of normal for the MATURE fraction (~0.4 of benthivores, ~0.35-0.45 of whiting) around spawning.
   - The group window is 8 h + to 17 h + (cod early, haddock late), and each fish fasts ~35-70 d.
   - The gonad cost (0.05-0.18 ME of the group per year) comes on top.
   - The fast evidence is mostly [A]/[S] and the haddock evidence conflicts.
2. **Recovery.** A pure drain with the reserve_room appetite refills within weeks, while the field refill takes until Sep-Nov. Two candidates, neither tested here:
   - (a) soma before liver after spawning;
   - (b) a food-limited ration (field feeding level 0.22-0.53 of Cmax).
3. **No term needed elsewhere.** Planktivores, the water column and mackerel need no spawning term in the box. Pout's spawning death is a mortality and emigration matter, not a reserve one.

## 6. Open gaps

- **Liver share of gonad energy.** Not measured for any gadoid. Plaice ~50 % [F via thesis].
- **Intake during spawning.**
  - No direct measurement for North Sea or Skagerrak fish.
  - Fordham & Trippel 1999 (cod) was not obtained: read via the abstract and Dutil 2006 [S] only.
  - Haddock evidence conflicts.
  - Nothing for whiting, saithe or long rough dab.
- **Liver share of the fasting loss** (liver vs muscle). Taken as 0.5 (0.3-0.7) [J].
- **Long rough dab** (0.126 of the benthivores). No maturity data in the region; GSI, fecundity and L50 not found; its winter feeding drop may be seasonal rather than tied to spawning.
- **Survey window.** NS-IBTS has not sampled the box in Mar-Jul since 1995, so spawning of haddock, whiting and pout in the box is neither confirmed nor excluded. No eastern Skagerrak egg surveys for haddock or whiting were found.
- **Immature fish.** No Skagerrak liver or condition series for immature gadoids, and no lengths for the NIFES May-Jul fish. This decides whether the group target is 0.17 or 0.43.
- **Gonad energy values not found:**
  - testis energy for any gadoid;
  - North Sea female cod GSI;
  - haddock egg dry mass;
  - whiting monthly GSI;
  - witch and dab ovary energy;
  - sprat, horse mackerel and blue whiting gonad energy;
  - starry ray and hagfish output per tonne.
- **Survey sampling.**
  - Box hauls are shallow-biased: 65 % are < 100 m against 23 % of the area, and none are deeper than ~260 m.
  - The mature shares are area-weighted over only two depth strata.
  - The sex ratio was not computed.
- **The scratch test's +28 % ration** is far more than the gonad energy (~2 % of the annual ration). The size of the drain used in that test is not known here (main agent: it was 2 MJ/t/tick for 90 days, 712 MJ/t of the group, ~4x the benthivores' annual gamete energy of 181 MJ/t).

Notes from the subagent:
- Whiting's mature share differs between sources. `lit_whiting_watercolumn.md` gives 0.65; this analysis gives 0.43, from box-only and region ogives. The range is recorded in `evidence.md` section 2.
- Mature shares are computed differently in this analysis and in `lit_flatfish.md`, which uses a shoot-position box definition and sex-specific ogives. The difference does not matter at group level.
