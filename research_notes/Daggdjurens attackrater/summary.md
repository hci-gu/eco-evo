# Attack rates of the mammal pairs from encounter theory - summary (2026-11-13)

Research subagent; the main agent saved this text (the subagent's Write of this file was refused).
Nothing in the code, library, manifest, tests, AGENTS.md or the resume was changed; no training or
rollout; no value here was set from, or checked against, model output. The fix99/fix100 starvation
numbers are quoted only to say what a derived a would mean, never to choose one.

Grades: [F] measured (field or lab), [A] abstract, snippet or search summary only, [S] secondary (review
or an earlier project note that cites the primary source), [D] derived here, [J] judgement.

Files:
- `scripts/mammal_a.py` -> `tables/mammal_a_stdout.txt`, `tables/mammal_a.json`, plus
  `tables/derived_a.csv` (headline table and per-class depth tables). Reads only `maps/depth.npz`.

```
cd scripts && python3 -I mammal_a.py <repo> ../tables/mammal_a.json > ../tables/mammal_a_stdout.txt   # ~10 s
```

No PDF was saved: the two new open-access sources (Wisniewska et al. 2018; Rojano-Donate et al. 2024)
were read from PMC, and the rest were already on disk in sibling notes.

## 1. Question and effect size

No porpoise or seal attack rate was ever derived. Section 151 fitted the porpoise values to an Ecopath
diet, section 206 multiplied them by 1.5 because porpoises declined (a fit to model output), and the
seal values are judgements rescaled to "the same demersal energy return" (sections 166, 209). This note
derives every mammal pair the way the fish pairs were derived, then asks two questions:
- Is the pair encounter-limited or handling-limited at the box's densities?
- What does that imply for the ration?

## 2. Model conventions used (checked in the code)

- Response: `interactions.multi_prey_rate`. lambda_j = a_j B_j (Type III: a_j B_j^2). The demand is
  B_pred T p_j lambda_j / (1 + sum_k p_k h_k lambda_k + w B_pred), where T = the policy's summed eat share.
  The intake ceiling is 1/h (`energy_balance.intake_ceiling`).
- Porpoise h is a mass ceiling: 28.6 on every pair, so 1/h = 0.035 t/t/tick = 14 % BM/d. The species
  block's `max_intake_rate: 0.035` is the same number; every porpoise pair sets its own a. Seal h =
  E_prey / 212.5 MJ/t/tick, so 1/h = 0.033-0.047 t/t/tick = 13-19 % BM/d.
- Units of a (from `Attackrater ur motesteori/scripts/pred_encounter_a.py`): km2 swept per tonne of the
  WHOLE predator group per 6 h tick of foraging, against prey in t/km2 (cell = 1 km2):

      a = cross-section x speed x T x N_t x p x (time share at the prey's depth) / H / 1e6,

  with T = 21 600 s, N_t = animals per tonne and H the prey layer thickness. For a prey slab thinner than
  the detection disc, the cross-section over H becomes a strip width 2 rho (`area_in_slab`).
- **Two conventions decide what goes into a [D]:**
  - The policy's eat share IS the behavioural time budget (forage vs rest/travel), so the field shares
    stay OUT of a. These are: AR 81 % (Stalder 2020), 62 % of diving time feeding (Rojano-Donate 2024),
    and the seals' ARS shares.
  - What the fish pairs never needed goes IN: the dive cycle (surface time, descent and ascent) and the
    reach of the bed from the surface.
  - Reach is a per-cell condition. The engine's tool for it is a per-pair `depth_access_table` (rows
    {depth_m, factor}, values >= 0, renormalised to an equal-cell mean of 1, `habitat.pair_depth_map`;
    exact zeros are accepted), not a scalar. Each pair is therefore given as a box mean plus a class
    table.
- Day and night: the engine has no light term on the mammal pairs, and `dark_ratio` >= 1 means "no light
  dependence" (`daylight.attack_multiplier`). A pair that is stronger at night cannot be expressed. The
  library a is the annual mean, taken here as 0.5 day + 0.5 night.

## 3. Inputs

### Porpoise

| quantity | low / central / high | grade, source (read) |
|---|---|---|
| mass per animal | 40 / 45 / 62 kg | 45 is the manifest SCANS conversion [S]. 40 is the "representative" animal of Rojano-Donate et al. 2024, doi 10.1126/sciadv.adj7132 [F, PMC full text]. ~62 is implied by Rojano-Donate et al. 2018, doi 10.1242/jeb.185827 [S, library comment]. |
| speed through water while searching | 0.8 / 1.2 / 1.6 m/s | Closing speed 1.4 m/s in one echogram (Wisniewska et al. 2016, doi 10.1016/j.cub.2016.03.069) [F, accepted manuscript read in full, `Tumlarens byten/data/lit`]. Captive dive speeds 0.76-0.91 m/s (Otani 2001, snippet) [A]. Otani et al. 2000 (doi 10.1111/j.1748-7692.2000.tb00973.x) was not obtained. |
| on-axis detection range, single planktivore | 19 / 35 / 60 m | Villadsgaard et al. 2007 sonar equation at SL 178/191/205, doi 10.1242/jeb.02618 [D in `Observabilitetsparen/mammals.md`; PDF read there]. |
| on-axis detection range, 25-35 cm gadoid | 23 / 50 / 90 m | Same source. |
| school detection range | 86 / 200 / 300 m | Same source. |
| beam | -3 dB 13 x 11 deg, DI 24 dB | Koblitz et al. 2012, doi 10.1121/1.3683254 [A, abstract and table via search]. The beam broadens at short range (Wisniewska et al. 2015, doi 10.7554/eLife.05651) [A; page did not load]. |
| half-angle of the searched cone | 6 / 20 / 45 deg | [J]. 6 deg is the beam alone (two-way, at about -4.5 dB). Above that is head scanning, for which no measurement was found. |
| bed clutter on detection of fish within metres of the bed | x 0.25 / 0.5 / 1 | [J]. Gadoids stand above the sonar's dead zone ("Sal och tumlare golv" D5). |
| buried flatfish | x 0.1 / 0.3 / 0.6 | [J]. They lie inside the dead zone (D5); porpoises grub for sandeels (Schneider von Deimling et al. 2023, doi 10.1038/s43247-023-01102-y) [F/D via floors note]. |
| capture p per encounter | 0.5 / 0.85 / 0.97 | Buzz success 0.91-0.99 per animal; Table S1 sensitivity 0.58-0.99 (Wisniewska 2016) [F], multiplied by a pre-buzz escape q of 0.8-1 [J]. Pound-net animals: Hoekendijk et al. 2018, doi 10.1111/mms.12446 [F via floors note]. |
| submerged share of the dive cycle | 0.55 / 0.69 / 0.80 | Mean dive 61 s, mean surface time 27 s, 65 % of time diving (Rojano-Donate 2024) [F]. |
| vertical transit; longest routine dive; wanted bottom time | 1.0-1.6 m/s; 150 / 220 / 330 s; 40 / 70 / 100 s | 95 % of dives < 127 s, max 270 s; bottom-feeding dives 74-78 s in 11 m water (Rojano-Donate 2024) [F]. Skagerrak maximum dives 132-152 m: Teilmann et al. 2007 (J Cetacean Res Manage 9:201, no DOI) and van Beest et al. 2018 (Mar Biol 165:95), both via "Tumlarens byten" [S]. Transit speed [J]. Together these give a bed reach of ~143 m (central). |
| share of foraging dives to the bed (where reachable) | day 0.1 / 0.4 / 0.6; night 0 / 0.1 / 0.3 | Feeding = 37 % pelagic + 25 % bottom (Rojano-Donate 2024) [F]. "Primarily pelagic dives at night" (Wisniewska 2016) [F]. |
| buzz rates | day 48 /h, night 142 /h (median 100 /h, ~2400 /d; 75 % at night); 0-550 /h | Rojano-Donate 2024 [F]; Wisniewska et al. 2016 [F]; Wisniewska et al. 2018, doi 10.1098/rspb.2017.2314 [F, PMC full text: 0-550 /h, buzz-positive minutes ~50 %, most after dusk]. |
| prey | 3-10 cm, ~2 g (17 MJ/d ingested over ~2000 prey) | Wisniewska 2016; Rojano-Donate 2024 [F]. Shallow Danish waters (mean 13-23 m). Box stomachs: herring ~26 cm (Börjesson et al. 2003 via "Tumlarens byten") [S]. |
| stomach and passage | 1.9 kg; 140 min | Wisniewska 2016, citing older sources [S]. |
| field ration | 7-10 % BM/d | Rojano-Donate 2024 (17 MJ/d at 4.2 kJ/g for 40 kg) [F]. Kastelein captive 4-9.5 % (no DOI; via the library comment) [S]. |

### Seal (harbour seal)

| quantity | low / central / high | grade, source |
|---|---|---|
| mass | 45 / 55 / 80 kg | Library unit (population mean) [S]. |
| speed at the bed | 0.5 / 1.0 / 1.4 m/s | 1.4 m/s typical (Vance et al. 2021, Sci Rep 11:6514, doi 10.1038/s41598-021-85376-2) [F, PDF read]. ~0.5 m/s near the vibrissal threshold (Niesterok et al. 2017, doi 10.1242/jeb.158055) [S via `seal_senses.md`]. |
| whisker swath for still prey | 0.2 / 0.5 / 1.0 m | [J]. Breathing currents are detected at cm range (Niesterok 2017) [S]. |
| trail age still followed; follow success | 15 / 30 / 35 s; 0.4 / 0.7 / 0.9 | Age: Dehnhardt et al. 2001, doi 10.1126/science.1060514; Wieskotten et al. 2010, doi 10.1242/jeb.041699 [S]. Follow success: glide phases degrade tracking (doi 10.1242/jeb.047134) [S], value [J]. |
| swimming speed of an active near-bed gadoid; active share | 0.1 / 0.3 / 0.6 m/s; benthivores 0.2 / 0.4 / 0.7, whiting / saithe / pout 0.4 / 0.7 / 0.9 | [J]. Cod spend ~70 % in a near-inactive class on the bed (Hobson et al. 2007, doi 10.3354/meps347101) [S]. |
| capture p | active fish 0.35 / 0.6 / 0.85; buried flatfish 0.6 / 0.8 / 0.9; fish per school contact 0.6 / 1.2 / 3 | Bowen et al. 2002, doi 10.3354/meps244235 [F, PDF read]: per attempt, flounder 79.9 %, buried sand lance 63.7 %, schooling sand lance 37.5 %; 3.2 attempts per school contact. |
| dive cycle | dive 270 / 360 / 480 s; surface / dive 0.18 / 0.25 / 0.35; transit 0.8-1.4 m/s; wanted bottom 100-200 s | Mean dives 3.3-4.4 min at 10-37 m: Bjørge et al. 1995 (no DOI; via `diel_foraging.md`) [A]; Bowen 2002; Vance 2021 [F]. Time diving ~80 % (Bowen table 2) [F]. Longer dives at depth [J]. Together these give a bed reach of ~198 m. |
| path per prey capture attempt | 68 m (median), ~800 attempts / d | Vance 2021 [F]; Wadden Sea, prey density unknown. |
| field ration | 4-6 % BM/d | Härkönen & Heide-Jørgensen 1991, 5.1 % (no DOI; via "Parametervärden för sälar") [S]. |

### Prey (box)

- Densities are the manifest start densities: planktivores 4.72, benthivores 1.53, whiting 0.446, water
  column 0.617 t/km2.
- Stage access and annual-mean juvenile shares are the library's (0.15 benthivores, 0.17 whiting).
- Planktivore pool shares: herring + sprat 0.575, pout 0.35, blue whiting 0.075 (243.3).
- Layers are those of the fish notes:
  - herring 10-100 m by day (H 70) and dispersed in the upper 30-60 m at night (Onsrud et al. 2004,
    doi 10.1016/j.icesjms.2004.09.005; Falk 1968, no DOI; both via `pred_encounter_a.py`) [S];
  - day-schooled share 0.9, schools of ~1.25 t (Misund 1995, via the same script; no DOI) [S];
  - pout within metres of the bed;
  - blue whiting below 150 m;
  - whiting 10-45 m above the bed (Onsrud 2004) [S].
- Bed depths come from `maps/depth.npz`: classes < 100 / 100-125 / 125-150 / 150-200 / 200-250 / >= 250 m
  take 0.23 / 0.16 / 0.14 / 0.19 / 0.13 / 0.15 of the cells.
- Stalder et al. 2020 (AR 81 %) is via `Observabilitetsparen/mammals.md`; no DOI here [S].

## 4. Derivation per pair [D]

Each pair is computed per cell and averaged over equal cells. The Monte Carlo draws 4000 sets: triangular
on the log scale for positive quantities, linear for shares.

**Porpoises -> planktivores.** A night term and a day term, each through the dive cycle:
- Night: dispersed herring and sprat in the 40 m night layer. Cross-section pi (R sin Theta)^2 = 452 m2,
  x 0.69 submerged x p 0.85 x the pelagic share of dives. Pout is taken on bed dives only.
- Day: 10 % of the herring dispersed in a 70 m layer, 90 % in schools (section 5). Pout again on bed
  dives only.
- Parts of the box mean (central): night herring 2.08, day herring 0.083, day pout 0.070, night pout
  0.017. The pair a is 1.12.
- Night/day 13.7 (4-21). It cannot be expressed in the engine; field buzz rates are ~3x higher at night,
  but buzzes count captures, not encounters.

**Porpoises -> benthivores, whiting, water column.**
- Strip width 2 rho at the bed, rho = R_gadoid x clutter x sin Theta.
- For benthivores: gadoid 0.64 + flatfish 0.36 x dead-zone factor (D6 member mix).
- Whiting's layer sits ~20 m above the bed, so its reach is one class deeper; it has no clutter, but its
  layer is thicker.
- Time share = (share of foraging dives that go to the bed) x (bottom time / dive cycle):
  - 0.25 at 80 m;
  - 0.06 at 130 m;
  - 0 beyond ~143 m (central).

**Seals -> benthivores, whiting, water column.**
- The bed search is a 2-D strip. Swath = whisker swath + active share x follow success x (2/pi) x trail
  length (Buffon mean width of a randomly oriented trail). Trail length = prey speed x trail age.
  - Benthivore gadoids: 2.1 m (central).
  - Active whiting and saithe: 3.3 m.
- Flatfish are found only within the whisker swath (0.5 m) at p 0.8.
- Bed time: ~0.19 at 150 m; 0 beyond ~198 m.
- Whiting: x 0.3 for its height above the seal's band [J].
- Water column (saithe juveniles near the bed): x 0.625 for school protection [J; Bowen's schooling /
  single-fish success ratio].

**Seals -> planktivores.**
- Near the bed: pout (active, trail swath) plus herring.
  - Herring's share in the bottom 10 m is 0.5 over beds < 150 m and 0.05 deeper.
  - By day that herring is schooled, with k 1.2 per contact. School factor S 0.001: no meal from schools
    on encounter terms.
- Pelagic night search where the bed is out of reach, at half weight [J].
- Read as Type II. The library pair is Type III; see open point 5.

## 5. The school question

- **Fish convention (Stimskydd):** a school met yields k prey, k ~1 (one meal). Schooling is then a pure
  factor S on a, capped at 1. It is only that while k and N do not depend on local density and the
  predator does not stay with the school.
- **Porpoises:** no study was found of porpoises at herring schools. No tag study pairs buzzes with
  echosounder schools, and Wisniewska 2016 does not report whether its 3-10 cm prey were schooled [gap].
  The DTAG prey are single small fish taken one buzz at a time, at success > 0.9.
- **Seals:** the only measurement is Bowen 2002 [F].
  - Seals split a sand-lance school and pursue the fish that leave it (3.2 attempts at 0.375 per contact).
  - They then return to the same school (52 schools): about 3.5 fish per 10 min while with a school.
  - So for seals k per contact is ~1, as for fish, but the seal stays: it is "school following", which
    the Stimskydd note had set aside for shelf gadoids.
- **What follows [D]:**
  1. Counted per encounter, schools are near-complete protection against both mammals.
     - Porpoise S 0.005 (0.0004-0.02) at k 5 (1-38). Even a full stomach per school met (k 38) gives
       only S 0.04.
     - Seal S 0.001.
     - A school 200 m across is met far more often than a single fish, but holds ~25 000 fish, so one
       meal per school met dilutes per-tonne risk by ~10^2-10^3.
  2. If the mammal stays with a school (Bowen's seals, and porpoises plausibly), the response is not a
     factor on a. Intake while with the school is set by pursuit and handling, and the school is a
     handling-limited patch. The engine cannot hold "with a school" over ticks; the closest form is the
     dispersed-fish a, which treats the school's edge fish as met one by one.
  3. The derived porpoise pair does not depend on this choice. 93 % of its a is the night term, when
     herring are dispersed (sigma night 0; Stimskydd E22, E25, E26).
     - Day herring as computed: 0.083, i.e. 0.04 of the annual mean.
     - With no dispersed herring by day at all: the a falls only from 1.12 to 1.08.
     - **Only if herring did not disperse within porpoise reach at night** would the pair fall to the
       day terms (~0.085). a h B would still be ~11: still handling-limited at the box mean.

## 6. Escape ability: where it enters [D]

| prey quality | enters through | mammal pairs |
|---|---|---|
| cruising speed | v_rel (Gerritsen-Strickler) in the encounter; < 5 % here because v_pred >> u_prey | negligible for both mammals |
| cruising speed (seal) | trail LENGTH: a faster fish leaves a longer trail and is found MORE often | seals: active gadoids 2.1 m swath vs flatfish 0.5 m |
| target strength / swimbladder | detection range R (sonar) | mackerel (no swimbladder) 19-31 m vs gadoids 44-64 m |
| burst speed, C-start, evasive turns | capture p only, including q (escape before the strike) and r (re-strike after a miss) | porpoise 0.85 (prey 3-10 cm still match the 1.4 m/s closing speed for seconds: Wisniewska 2016 fig. 3); seal 0.6, schooling 0.375, buried 0.8 |
| sustained speed (school or long chase) | p (via giving up), and the pair on/off flag | mackerel: not eaten by either mammal (pairs off); no p is derived |
| hiding / burying | detection (dead zone, whisker swath) and the visibility floor | flatfish x 0.3 vs porpoises, no refuge vs seals |

- Nothing about escape is in h. Handling for both mammals is a digestive mass or energy ceiling, not
  per-item pursuit time:
  - Porpoise: buzz to buzz >= 6.5 s at 550/h; 50 g herring at that rate are 27 kg/h against a ceiling of
    0.26 kg/h.
  - Seal: Bowen pursuit plus handling 2-33 s per fish.
- Burst escape therefore lowers a (through p) and never the ceiling. At the box densities it matters only
  where a pair is near the knee: the seal pairs and porpoises -> benthivores.

## 7. Results

### 7.1 Derived a against today (box mean over equal cells; km2/t/tick)

| pair | today | derived central | MC median (5-95 %) | x today | east third | B_vis t/km2 | a h B today -> derived | knee 1/(a h), t/km2: today -> derived |
|---|---|---|---|---|---|---|---|---|
| porpoises -> planktivores | 0.0225 | **1.12** | 0.94 (0.20-3.0) | 50 | 1.12 | 4.72 | 3.0 -> 152 (26-412) | 1.55 -> 0.031 |
| porpoises -> benthivores | 0.12 | **0.135** | 0.11 (0.027-0.35) | 1.1 | 0.16 | 1.05 | 3.6 -> 4.0 (0.8-11) | 0.29 -> 0.26 |
| porpoises -> whiting | 0.20 | **0.45** | 0.33 (0.068-0.98) | 2.2 | 0.56 | 0.43 | 2.4 -> 5.5 (0.8-12) | 0.17 -> 0.078 |
| porpoises -> water column | 0.20 | **0.18** | 0.14 (0.036-0.46) | 0.9 | 0.21 | 0.033 | 0.19 -> 0.17 | 0.17 -> 0.19 |
| seals -> benthivores | 0.10 | **0.071** | 0.045 (0.020-0.099) | 0.7 | 0.090 | 0.88 | 1.9 -> 1.3 (0.4-1.8) | 0.47 -> 0.67 |
| seals -> whiting | 0.05 | **0.047** | 0.027 (0.009-0.074) | 0.9 | 0.060 | 0.39 | 0.44 -> 0.41 | 0.89 -> 0.94 |
| seals -> water column | 0.05 | **0.098** | 0.053 (0.021-0.13) | 2.0 | 0.13 | 0.018 | 0.02 -> 0.04 | 0.84 -> 0.43 |
| seals -> planktivores | 0.002 (Type III) | **0.086 (Type II)** | 0.047 (0.020-0.11) | - | 0.10 | 4.72 | 1.4 -> 12 (2.9-16) | 4.0 (III) -> 0.38 |

- "Central" sets every factor at its central value. The MC median is lower because several factors
  (cone angle, clutter, k, active share) are skewed toward their low end.
- The east third (columns 40-59) is where the seals and their haul-outs are.
- B_vis = start density x stage access. a h B > 1 means handling-limited at the box mean.

### 7.2 Reach per bed-depth class: `depth_access_table` values (class a / equal-cell mean)

| pair | < 100 | 100-125 | 125-150 | 150-200 | 200-250 | >= 250 m |
|---|---|---|---|---|---|---|
| porpoises -> planktivores | 1.01 | 0.97 | 0.94 | 1.02 | 1.02 | 1.02 |
| porpoises -> benthivores, -> water column | 2.86 | 1.74 | 0.41 | 0 | 0 | 0 |
| porpoises -> whiting | 2.38 | 1.80 | 1.02 | 0.08 | 0 | 0 |
| seals -> benthivores, -> whiting, -> water column | 2.01 | 1.67 | 1.23 | 0.50 | 0 | 0 |
| seals -> planktivores | 1.88 | 1.56 | 1.15 | 0.35 | 0.32 | 0.32 |

- The zeros say a porpoise cannot reach the bed beyond ~145 m, nor a seal beyond ~200 m, in a routine
  dive. The 0.35-0.5 cells come from the class mean of the dive-cycle bottom share.
- These tables carry the "depth flag" that "Tumlarens byten" left as [J 0.3-1].

### 7.3 Saturation, ration and diet

**1. Porpoises.**
- At the box mean every porpoise pair except the water column is handling-limited, today and derived:
  a h B 2.4-3.6 today, 4-152 derived.
- Ration at T = 1 (whole tick foraging, hunger 1, every prey attacked, start densities):
  - today 12.6 % BM/d, derived 13.9 (p5 13.5);
  - the ceiling is 14.0 and the field 7-10 % BM/d.
- So at the grid mean the derived a changes nothing: the ceiling, the hunger gate and the policy's T set
  the ration. The derived a does NOT say the porpoises have too little food at the box's mean density.

**2. The knee is where the derived a matters.**
- Porpoises -> planktivores moves from half-saturation at 1.55 t/km2 to 0.031 (0.008-0.19 over the MC
  range). Per cell, single prey, T = 1:

  | planktivores in the cell, t/km2 | 0.03 | 0.1 | 0.3 | 1.0 |
  |---|---|---|---|---|
  | today, % BM/d | 0.26 | 0.85 | 2.3 | 5.5 |
  | derived central, % BM/d | 6.9 | 10.7 | 12.7 | 13.6 |
  | derived p5, % BM/d | 2.0 | 5.0 | 8.8 | 11.9 |

- Break-even on planktivores is 6.2 % BM/d (eat cost 90 MJ/t/tick).
- The fix99 starvation cells held 0.03-0.3 t of planktivores per 1 km2 cell ("Tumlarna i fix99"). With
  today's a they pay 0.26-2.3 % BM/d, far below break-even; with the derived central a, 6.9-12.7 %.
- **The section 151/206 calibration put the porpoise knee about 50x too high.** That is the mechanism by
  which a porpoise "chases planktivores, eats little and starves" in a cell that, by encounter theory,
  still feeds it. The x1.5 of section 206 is not the right order: the derived factor is x50 (x9-135).

**3. Porpoise diet at the box mean.**
- Derived: planktivores 0.94, benthivores 0.03, whiting 0.03, water column 0.00.
- Today: 0.33 / 0.39 / 0.26 / 0.02.
- Booked: 0.65 / 0.15 / 0.13 / 0.02. Börjesson 2003 ~0.75 planktivores; Aarefjord 1995 0.59 (coastal
  stomachs).
- So the derived a corrects today's demersal-heavy diet but overshoots: the bed pairs are 4-5x under
  their booking at the box mean. Over beds < 100 m their a are x2.4-2.9 the mean, which gives a demersal
  share of ~0.14 there.
- The demersal shortfall comes from reach: in a 44-450 m box the porpoise reaches the bed only over
  ~45 % of the cells, and there spends 6-25 % of its foraging time at it. The booking was transferred
  from coastal stomach sets (13-40 m water). This is the "Tumlarens byten" coastal-vs-offshore point, now
  with a number. It is a booking question, not a reason to raise the bed a.

**4. Seals.**
- Benthivores, whiting and the water column come out within x0.7-2 of today. These pairs are
  encounter-limited or near the knee (a h B 0.04-1.3): the seal's vibrissal swath is metres, not tens of
  metres, and bed time is short at 100-200 m.
- Ration at T = 1 at the start densities: today 13.0 % BM/d, derived 12.9, against a ceiling of 18.9 and
  a field ration of 4-6 %. Like the porpoise, the seal is saturated at the box mean; the hunger gate and
  T set the field ration.
- The large change is seals -> planktivores:
  - 0.086 as Type II, against 0.002 Type III today.
  - At equal intake at the box density, Type III a = 0.086 / 4.72 = 0.018 [D], x9 today.
  - It is driven by Norway pout: 0.35 of the pool, near the bed, active, 1.65 t/km2, more than the
    seals' accessible benthivores.
  - It puts the derived seal diet at 0.83 planktivores against a booking of 0.25. Open point 4.

**5. Cross-checks of the cross-sections (order of magnitude only; prey density at the tag sites is
unmeasured) [D].**
- Porpoise: 100 buzzes/h at ~1.2 m/s and 0.69 submerged is one buzz per ~30 m of path. With the 452 m2
  cross-section that needs ~2-5 t/km2 of 2 g fish in 13-23 m water: plausible for Danish coastal waters.
  The cross-section is therefore not grossly overstated.
- Seal: Bowen's flounders were met once per ~95 s of search (~95 m). With a 0.5 m swath that implies
  ~0.02 flounder/m2, ~4 t/km2 at 200 g: plausible. Vance's 68 m per attempt would need ~1-2 t/km2 of 5 g
  fish under a 2 m swath, also plausible.

## 8. Proposal (not applied; user decision)

| pair | a | depth_access_table | type | notes |
|---|---|---|---|---|
| porpoises -> planktivores | 1.1 (0.2-3.0) | flat (none needed) | II | night-dominated (13.7x); the engine cannot raise night a |
| porpoises -> benthivores | 0.135 (0.03-0.35) | 2.86 / 1.74 / 0.41 / 0 / 0 / 0 | II | today's value is right on the box mean by coincidence; the table is the change |
| porpoises -> whiting | 0.45 (0.07-0.98) | 2.38 / 1.80 / 1.02 / 0.08 / 0 / 0 | II | |
| porpoises -> water column | 0.18 (0.04-0.46) | as benthivores | II | with stage_access 0.054 |
| seals -> benthivores | 0.07 (0.02-0.10) | 2.01 / 1.67 / 1.23 / 0.50 / 0 / 0 | II | |
| seals -> whiting | 0.047 (0.009-0.074) | as benthivores | II | the 0.3 height factor is [J] |
| seals -> water column | 0.10 (0.02-0.13) | as benthivores | II | with stage_access 0.029 |
| seals -> planktivores | 0.086 (0.02-0.11) Type II, or 0.018 Type III at equal box intake | 1.88 / 1.56 / 1.15 / 0.35 / 0.32 / 0.32 | II or III | see open point 4 |

- Class edges are < 100 / 100-125 / 125-150 / 150-200 / 200-250 / >= 250 m. The loader wants rows
  {depth_m, factor}, step rows at the class edges as in the krill pairs; zeros are allowed (values >= 0,
  at least one > 0).
- The mammal pairs have no observation channel change, so checkpoints keep their shapes. The payoffs
  change, so retrain.
- Judge with budget_gate's diet gate and its prey-side M2 before training. The porpoise bed pairs will
  likely be OFF on the predator side against the coastal booking (0.03 vs 0.15).

## 9. Open points and gaps

1. **Night herring within porpoise reach.** This carries 93 % of porpoises -> planktivores. It rests on
   the fish notes' layer [S: Onsrud 2004, Falk 1968], not on a box measurement. If herring stayed below
   ~60 m at night the pair would be ~0.085, still handling-limited at the box mean (a h B ~11), with its
   knee at ~0.4 t/km2.
2. **The searched cone of the porpoise sonar** (6-45 deg) is the widest single factor (x50 on the cross-
   section between the bounds). There is no measurement of head scanning during search; only the beam
   (Koblitz 2012) [A].
3. **No porpoise data on herring schools** (k per school, school following, schooled share of prey). Nor
   any offshore or deep-water porpoise tag data: every DTAG animal fed in 13-23 m water, and the tagged
   Skagerrak animals of van Beest 2018 reached 151 m but have no buzz data [S].
4. **Seals and Norway pout.** The derived seal diet is 0.83 planktivores, almost all pout, against the
   booked 0.25. Two readings:
   - The Skagerrak seal "gadids" in otolith diets (50 % in Kappa et al. 2026 [S]) include Norway pout,
     which would make this a booking question.
   - Or seals over 100-200 m beds do not meet pout at the rate a near-bed active fish implies.

   Not resolved: the species split of the Skagerrak seal gadids was not obtained.
5. **Type III on seals -> planktivores.** The pinniped multispecies literature supports Type III (Smout
   2014; Middlemas 2006; via "Parametervärden för sälar" [S]), but no parameter was obtained. The encounter
   derivation gives a Type II a. The Type III value at equal box intake (0.018) is a convention, not a
   derivation.
6. **Seal dive cycle at 100-200 m** is extrapolated from 10-37 m dives (Bowen, Vance, Bjørge); reach ~200
   m [J]. There are no Skagerrak offshore seal dive data. Seal haul-out absence (E1) is not in a; it
   belongs to an outside exchange.
7. **The capture p of porpoises on 15-26 cm box prey** is transferred from 3-10 cm prey (success > 0.9)
   in shallow water, and the Wisniewska animals had just left a pound net (Hoekendijk 2018). The range
   0.5-0.97 covers this, not prey size per se.
8. **The engine averages a over day and night before saturating.** For porpoises the night a is ~14x
   the day a, so in sparse cells the mean a overstates the day and understates the night. No engine form
   exists for dark_ratio > 1.
9. **Mass per animal** (40-62 kg) scales every porpoise a by 1/m; the manifest uses 45, the library unit
   50.
10. **Sources not obtained:** Otani et al. 2000 (swim speed), Koblitz 2012 full text, Wisniewska 2015
    eLife (page did not load), Smout 2014 parameters, Börjesson 2003 full text, any seal-borne camera or
    accelerometer capture success on gadoids.

## Verdict

- **Porpoises -> planktivores is encounter-unlimited by a wide margin.** Derived a 1.1 (0.2-3.0) is ~50x
  the fitted 0.0225. The x1.5 of section 206 compensated, in the wrong order of magnitude, for a knee set
  ~50x too high: half-saturation 0.03 t/km2 instead of 1.55.
- **The porpoises' food at the box mean was never the problem.** Both values saturate there (ration at
  T = 1 is 13-14 % BM/d against a field 7-10 %). What changes is that a porpoise on a nearly empty cell
  (0.03-0.3 t/km2) is fed at 7-13 % BM/d instead of 0.3-2.3 %.
- **The porpoise bed pairs keep their box-mean level** (0.135, 0.45, 0.18 against 0.12, 0.2, 0.2) but
  gain a reach table. That table zeroes them over beds deeper than ~145 m, and their share of the diet
  then falls far below the coastal booking.
- **The seal bed pairs are confirmed within x0.7-2.** They are near the knee: vibrissal swaths of
  0.5-3 m, short bed time at depth.
- **Seals -> planktivores is the other large change (x40 as Type II).** It is dominated by Norway pout
  and needs the diet question settled first.
- **Schooling, counted per encounter, protects herring almost completely from both mammals** (S 0.001-
  0.04). The porpoise pair does not depend on it because porpoises feed at night, when herring are
  dispersed.

## 10. Frozen-policy probe (main agent, 2026-11-13)

Frozen fix100 policies (update 830), 30 x 30 probe grid, porpoises from 21 t, one year, only
porpoises -> planktivores a changed in a library copy (`research_notes/Tumlarna i fix99/scripts/porp_blind.py`
with BLIND=0; tables/probe_fix100_*.txt):

| arm | B after 1 yr / start | starvation + sweep, t: ticks 1-100 / 101-700 / 701-1460 |
|---|---|---|
| base, a 0.0225 | 0.34 | 8.0 / 5.6 / 0.8 |
| a 0.20 (MC p5) | 0.38 | 7.7 / 4.4 / 1.5 |
| a 1.12 (derived central) | 0.38 | 7.2 / 5.7 / 0.8 |
| planktivores blind to porpoises (272.4) | 0.60 | 2.4 / 3.6 / 3.8 |
| blind and a 1.12 | 0.74 | 1.5 / 3.5 / 2.1 |

- The derived a moves the year from 0.34 to 0.38: the knee is not what starves the porpoises under these
  policies. The start fall (8 of 21 t in the first 25 days) is untouched; the blind arm cuts it to 2.4 t.
- In the first 20 ticks 43 % of the porpoise biomass moves each tick and the fill falls to 0.55-0.58
  (lowF 0.16-0.28); a moving share eats nothing in the tick. The loss is time spent moving, not intake
  per foraging tick.
- Caveat: the policies were trained with a 0.0225; a policy trained on the derived a may move less. The
  probe tests payoff at fixed behaviour only.
- Blind and a 1.12 together (tables/probe_fix100_a1.12_blind.txt): 0.74 after one year, 19.9 t of 21 at
  tick 40 (base 14.3), move share 0.19 in the first window (base 0.36-0.43). The two act together: a alone
  +0.04, blind alone +0.26, both +0.40. "Blind" zeroes the planktivores' porpoise channel in the own cell
  and all four neighbours, a stronger cut than the design's neighbour weight (own cell kept, neighbours
  w 0.10).
