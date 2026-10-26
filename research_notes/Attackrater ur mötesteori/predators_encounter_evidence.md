# Evidence for `predators_encounter.md` (predatory_fish attack rates)

Transcribed by the main agent from six literature sub-agents (2026-10-25). Tags as reported:
[F] full text read, [A] abstract only, [S] secondary (cited in another paper, search summary),
[D] arithmetic, J judgement. Items the sub-agents flagged as unverified are marked **UNVERIFIED**.
Downloads (untrusted data) are in `data/vision/`, `data/schools_capture/`, `data/fish_layers/`,
`data/plankton_benthic_prey/`, `data/mackerel/`, `data/minor_members/`, `data/validation/`.
"EN" = an existing note in this repository.

## E1. Detection distance R (data/vision)

| Predator | Prey | Light / water | R | Source | Tag |
|---|---|---|---|---|---|
| Lake trout 330-456 mm | salmonids 55-139 mm | 0.17 lx | "<25 cm" | Vogel & Beauchamp 1999 CJFAS 56:1293, doi:10.1139/f99-071 | [A] |
| same | same | >= 17.8 lx | "about 100 cm"; RDmax 1.012 m | same; equation via Eckmann et al. 2016 | [A]/[S] |
| equation | | | RD = 0.25490 I^0.4747 (I lx) for I <= 17.8, else 1.012 m; "derived from experimental data on adult lake trout foraging on salmonid prey (5.5-13.8 cm TL)", clear water < 0.3 NTU | Eckmann et al. 2016 Ecol Freshw Fish doi:10.1111/eff.12321 citing Beauchamp et al. 1999 | [F] of the citing paper; original **UNVERIFIED** |
| Coastal cutthroat trout | fish | > 18 lx | 187.1 cm | Hansen, Beauchamp & Schoen 2013 TAFS 142:854, doi:10.1080/00028487.2013.785978 | [A] |
| Chinook, yearling | juvenile trout ~50 mm | > 24.9 lx | 122.1 cm; declines with turbidity > 1.65 NTU | same | [A] |
| Chinook | same | low light | levels off at "~50 cm", "potentially owing to a response to non-visual stimulus or an unexplained artifact" | Rohan 2021 PhD, Univ. Washington, ch. 1 | [F] |
| Lake char vs trout | fish | saturating | char asymptote "65% higher"; predation "rose sharply at 0.50-0.75 lx" | Mazur & Beauchamp 2003 EBF 67:397, doi:10.1023/A:1025807711512 | [A] |
| Pollock-like (model) | 15-cm juvenile pollock | saturating / midnight near bed | 2.96 m max; noon mixed layer 2.80-2.95; nepheloid 2.20-2.40; midnight near bottom < 0.1-0.6 m | Rohan 2021 ch. 4 | [F, model] |
| same | Calanus 3.5 mm | saturating / midnight | 11.6 cm max; < 0.2-4 cm at midnight near bottom | same | [F, model] |
| Piscivore 0.5 m (model) | myctophid 6 cm | optimal / dark | rmax 1.5 m (capped at ~1 BL by assumption); non-visual floor rmin 0.1 m; c = 0.3 m-1; clearance 1/2 pi r^2 v "assuming that only half of the visual area is effectively scanned (Varpe and Fiksen 2010)"; pc 0.7; v 0.25 m/s | Langbehn et al. 2019 MEPS 623:161, doi:10.3354/meps13024, Suppl. S1 | [F, model] |
| Juvenile cod age 1 | mysid | 0.01-64 umol m-2 s-1 | no cm values; R "greatest" at 0.01, less at 1.5-6.5 | Meager et al. 2010 MFBP 43:99, doi:10.1080/10236241003798910 | [A] |
| Juvenile cod | mysid | turbidity | chemical cues "from significantly longer distances than when only visual cues were available" | Meager et al. 2005 CJFAS 62:1978, doi:10.1139/f05-104 | [A] |

- Attenuation: Kd(500) 0.036-0.13 m-1 in Norwegian fjord basins (Aksnes et al. 2009 MEPS 387:39 [F]);
  no beam attenuation for 50-200 m in the Skagerrak found. Vision note (EN `../Torskfiskars ljusberoende
  vid jakt/gadoid_vision_thresholds.md`): effective Kd 0.15-0.2 m-1 at 30-60 m.
- Fiksen et al. 2002, Varpe & Fiksen 2010, Langbehn & Varpe 2017 cap R at about one predator body length
  (via Rohan 2021 [F]).
- Search speeds: Dolly Varden 0.295 / 0.235 / 0.040 m/s day / twilight / night (Beauchamp et al. 1999 via
  Eckmann [S]).
- Not accessed: Mazur & Beauchamp 2003, Vogel & Beauchamp 1999 full texts, Meager 2005/2010 full texts,
  Aksnes & Giske 1993, Aksnes & Utne 1997, lateral-line primaries (Montgomery, Coombs). No measured
  R of any member on any prey exists in what was found.

## E2. Capture success and schools (data/schools_capture)

| Source | Predator / prey | Value | Tag |
|---|---|---|---|
| Turesson & Bronmark 2004 J Fish Biol 65:363, doi:10.1111/j.0022-1112.2004.00455.x | perch / roach | 35/199 = 0.176 per attack; "34/168 = 0.20 for attacks towards single prey ... schools of two or more prey ... 1/31 = 0.03"; per sequence "35/47 (74%)" | [F] |
| same | pike / roach | 0.151; "CSsit-and-wait = 8/22 = 0.36, CSpursuit = 17/144 = 0.12" | [F] |
| same | pikeperch / roach | 0.085; "prey were consumed during the night in 13 of the 17 trials" | [F] |
| Herbert-Read et al. 2016 Proc R Soc B 283:20161671 | sailfish / sardine schools | "Only 24% of these attacks resulted in a successful capture" | [F] |
| Neill & Cullen 1974; Landeau & Terborgh 1986; Parrish 1993; Major 1978 | various | success per encounter falls with shoal size (no numbers read) | [A]/[S] |

Field (gadoids on clupeids):
- Pitcher et al. 1996 ICES JMS 53:449 [A]: "Individual cod and haddock predators intimately accompanied the
  herring schools, their attacks causing frequent modifications to, but not dispersal of, school structure."
- Nottestad et al. 2002 ICES JMS 59:393 [F]: predatory events every 27 min per tracked school inshore
  (Pitcher 1996); "Gadoids may accompany the schools and perform frequent shoal attacks but probably do not
  kill as many herring as the whales."
- Nottestad 1998 ICES CM 1998/J:20 [F]: "85% of the [saithe] individuals had recently been eating one or more
  herring"; only "relatively small sized (4.5 - 18 tonnes) and dense (2.6 - 5.2 ind*m-3) schools in shallow
  water (50 - 110 m) were pursued" (killer whales and saithe).
- Andersen et al. 2017 MEPS 583:195, doi:10.3354/meps12319 [F]: "87.1% of the sprat were ingested at dusk
  (39.7%) or dawn (47.4%) ... 12.4% were ingested during night and one sprat during daytime"; cod feeding
  level f = 0.40.

School structure:
- Misund et al. 1995 ICES JMS 52:11 [F] (Norwegian Trench west slope, summer, 29-cm herring): density
  "averaged 4.25 herring m-3" (0.3-22.2); mean area ~370 m2; schools "averaged about 25 000 individuals,
  but varied from about 100 ... to about 420 000".
- Fernandes et al. 2001 ICES CM 2001/Q:08 [F]: day school length 21.2-27.2 m, height 3.5-5.2 m; schools break
  up 18:16-20:56 and re-form 04:17-05:49.
- Beare et al. 2002 ICES JMS 59:469 [F]: 0.40-0.84 herring schools per nmi of track.
- Petitgas et al. 2001 ICES JMS 58:1150 [F]: 0.28-0.59 schools/km2 and 5.7-10 t per school [D, upper bound].
- Krause & Ruxton 2002 via Pertzelan et al. 2022 [S]: "a mere twofold increase in the maximum detection
  distance for a 100-fold increase in the number of fish".
- Not found: attacks per hour per predator, prey caught per school encounter, school detection distance.

## E3. Vertical layers (data/fish_layers)

| Species | Day | Night | Source | Tag |
|---|---|---|---|---|
| Herring (wintering, Norwegian Channel/Skagerrak) | deep | "upper limit of the nightly herring traces ... between 0 and 100 m" | Falk 1968 ICES CM 1968/H:17 | [F] poor OCR |
| Herring (shelf) | near bottom "irrespective of bottom depth" | ascend | Torgersen et al. 1997 Sarsia 82:87 | [F] |
| Herring/whiting/sprat (Oslofjord 120 m) | in upper/middle krill layer (krill > 70 m, 20-40 m thick) | pelagic fish "concentrated in the upper ~30-40 m" | Onsrud et al. 2004 ICES JMS 61:1278 | [F] |
| Herring in the box (HERAS July 2010-20) | backscatter mostly 10-100 m | - | EN `../Ekolodsdata planktivorer/tables/nasc_depth_profile_box.csv` | [D] |
| Norway pout | "generally 5-10 m above the sea bed, but somewhat closer by day" | same | Albert 1994 ICES JMS 51:45 | [A] |
| Norway pout | turbid water: "compact SSL ... about 100 m above bottom" | - | Kaartvedt et al. 1996 MEPS 136:51 | [F] |
| Blue whiting | shelf break: "remained near the bottom both day and night" | off shelf: into upper waters | Torgersen 1997 | [F] |
| Saithe (farms) | "distributed through ~100 m of the water column"; 25-50 m summer, 60-90 m winter | 75 % show diel pattern | Skilbrei & Ottera 2016 ICES JMS 73:1186 | [F] |
| Saithe (Iceland DST) | median ~140 m Mar-Apr, < 50 m summer-autumn; most activity by day | reduced activity | Armannsson & Jonsson 2012 ICES JMS 69:1372 | [F] |
| Saithe, large | "stack" ~100 m below krill swarms at 100-200 m | same | Kaartvedt et al. 2005 MEPS 299:1 | [F] |
| Whiting | demersal sub-population feeds by day | pelagic sub-population at night | Pedersen 2000 [A]; Rindorf 2003 MEPS 249:265 [F] | |
| Mackerel (summer) | "upper 40 m ... during both day and night" | same | Diaz et al. 2013 ICES CM 2013/N:05 | [F] |
| Spurdog | median 198.5 m | median 60.0 m | Klocker et al. 2024 Mov Ecol 12:62 | [F] |
| Hake | near bottom | up into the water column | JFB 2012 81:728 | [S] |

No offshore North Sea / Trench saithe tag study, no Skagerrak sprat/juvenile herring diel study, no
species-resolved IBTS day/night ratio (Adlerstein & Ehrich 2002 [A]: day/night catch rates "typically within a
factor of 2").

## E4. Prey: krill, copepods, bed prey, juvenile gadoids (data/plankton_benthic_prey)

- Krill North Sea mean 0.78 g WW/m2 (Thysanoessa, CPR) x 1.5 for M. norvegica = 1.17, max 2.4 (Mackinson &
  Daskalov 2007 via EN `../Sillens vinterföda/winter_food_and_feeding.md`) -> 8-25 % of the 9.72 t/km2 pool,
  central 12 % [D]. Fladen: "more than 90%" of zooplankton biomass in Mar-Apr (Williams & Lindley 1980,
  second-hand). No Skagerrak krill g/m2 found.
- Krill day layer: upper boundary 70-73 m at noon; night 10-30 m; "restricted to waters deeper than 70 m
  during the day" (Onsrud & Kaartvedt 1998 MEPS 171:209 [F]); 20-40 m thick by day (Onsrud 2004 [F]).
- M. norvegica DW 35-80 mg (Bamstedt & Karlson 1998 MEPS 172:149 [F]) -> 156-356 mg WW [D]; swimming mode
  ~4 cm/s in situ [S]; single krill "around 3 cm s-1" (Hardy & Bainbridge 1954 via Thomasson 2003 [F]).
- Swarm 200 ind/m3, ~11 t (Kaartvedt et al. 2005 [F]; Norwegian Sea).
- Calanus CV-CVI 117-140 ug C [F] -> 1.4-2.3 mg WW [D] (Heath et al., strathprints 18579; **citation title
  / journal conflict UNVERIFIED**).
- Pandalus: grid mean 0.56 t/km2, f 0.7 (EN `../Bottensamhällets epifauna/summary_standing_stock.md`);
  3.55 g per shrimp (NAFO SCR 21/001 [F/D]); leaves the bottom at night (ASMFC [F], Gulf of Maine).
- Crabs, hermit crabs, Munida, small shrimps, mysids 1.27 t/km2 (0.5-4.7), f 0.6 (same EN); mysids and
  hyperbenthos 0.2 (0.05-1.0) t/km2 (EN evidence_B).
- Suprabenthos "76-97 % in the 0-50 cm layer by day" (Cunha 1997 MEPS 157:119, via EN) [A].
- 0-group gadoids: cod 4-6 cm demersal by day, pelagic at night; cod > 8 cm fully demersal (Bromley & Kell
  1999 Acta Adriatica 40:7 [A]).

## E5. Mackerel and minor members (data/mackerel, data/minor_members)

- Macy, Sutherland & Durbin 1998 MEPS 172:89 [F]: fish "26 cm FL and 184 g"; mouth area "A = 0.0132
  FL^1.895 ... 5.02 cm2" (MacKay 1979); routine 28.7 cm/s (1.25 BL/s), feeding 41.5 cm/s (1.80 BL/s);
  clearance (VSC) 18.2-20.2 L/min particulate at 0.7-7.7 large copepods/L, 6.3-7.5 at 50-58/L (filter),
  2.6 L/min filter on small + intermediate copepods; Ivlev I = 1970[1 - e^(-0.0623(C - 0.39))] copepods
  fish-1 (5 min)-1; feeding starts at "4.4 and 14.7 copepods l-1" (rising as light falls), stops at 0.4/L;
  light threshold ~1e-7 uE m-2 s-1 (**exponent OCR UNVERIFIED**). Particulate R implied by the clearance
  = 1.7-1.8 cm [D].
- Hamann et al. 2023 Front Mar Sci 10:1253083 [F]: mouth opening 333 +- 163 mm2 (jaw possibly not fully open),
  raker gap 0.765 mm; swim 0.5 m/s.
- Nottestad 2016 abstract [F]: schools "about 3.8 body lengths sec-1" (over ground or through water unclear).
- Dahl & Kirkegaard 1986 (EN evidence C): "mackerel are feeding only during the day".
- Bachiller et al. 2016 PLoS ONE e0149238 [F]: "Zooplankton abundance did not have any significant effect on
  either SFD or FI".
- Horse mackerel (Mapukata 2002 MSc, Cape; T. t. capensis juveniles) [F]: "obligate particulate feeders";
  clearance 3.2-10.2 (max 23.9) L/min at 175 mm; swim 1.30-1.80 BL/s feeding, 0.39-0.49 resting.
- Hake (Carpentieri & Colloca 2005 Fish Bull 103:411 [F]): "fast-moving pelagic prey that are ambushed in the
  water column"; juveniles near bottom 100-200 m, up at night. No strike/reactive distance found.
- Anglerfish: "Prey detection responses had a range of c. 5 m" (Laurenson et al. 2004 J Fish Biol 65:947 [A]);
  goosefish did not lunge at small live fish in tanks (Stehlik et al. 2021 [F]); "feeding is sporadic",
  34.9 % with prey (Johnson et al. 2008 [F]). No strike distance in cm.
- Spurdog: 1.36-2.49 km/h net, "equally active during both day and night" (Sulikowski et al. 2010 MEPS
  418:249 [F]); "sharks follow their prey", more fast starts by day (Klocker 2024 [F]).

## E6. Validation sources (data/validation)

| Source | What it measures | Key numbers | Tag |
|---|---|---|---|
| Rindorf, Gislason & Lewy 2006 MEPS 325:243 | switching exponent b only (diet ratio vs IBTS catch ratio) | b 0.145-0.350; CI 0.081-0.390 | [F] |
| Kempf, Floeter & Temming 2008 MEPS 367:295 | relative stomach content vs relative prey abundance | steep drop "Below an ra of 0.1"; plateau from ~0.1-0.2; "This early saturation effect may has caused the observation of negative prey switching" | [F] |
| WKCAPELIN 2023 ICES Sci Rep 5:62 | NE Arctic cod on capelin, stock-scale Type III | "levels out at around 500 000 tonnes"; max ~0.15 kg per kg cod per month | [F] |
| Fall et al. 2021 Ecography 44:802 | cod consumption vs local capelin NASC (2 km) | median 0.73 %BM/d; "main increase ... occurred below median density" | [F] |
| Lauerburg et al. 2018 MEPS 594:213 | whiting stomach content vs forage index | "mean stomach content mass was on average 60% less" in 2000-07 (index 285 vs 362) | [F] |
| Engelhard et al. 2008 MEPS 369:213 | whiting per-capita sandeel intake vs local density, log-log | slope 0.08 +- 0.03 vs day sandeel | [F] |
| Rindorf 2002 J Fish Biol 61:579 | feeding probability vs fullness | "saturation is affecting search activity rather than prey or meal size selection" | [F] abstract |
| Temming et al. 2004 J Fish Biol 64:1351 | whiting abundance vs sandeel in stomachs (aggregation) | r2 0.48, night 0.61; **catch table garbled, not used** | [F] |
| Lopez-Lopez et al. 2015 MEPS 541:165 | hake fullness vs blue whiting abundance | threshold "100 to 1000 ind. km-2, suggesting a Holling type III response" | [F] |
| Macy et al. 1998 | mackerel clearance | see E5 | [F] |
| Rindorf & Gislason 2005 JEMBE 324:1 | whiting intake vs local prey density | **not read (403)**; [S]: mostly Type II, plateaus below digestive capacity | [S] |
