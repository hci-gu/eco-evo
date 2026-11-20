# Q2: Foraging modes, time shares and per-mode satiation in whiting, saithe and hake (2026-11-20)

Research subagent, literature only. Nothing outside this folder was edited, no model was run, and no value was
fitted to or chosen from model output.

## Tags, files and conventions

Tags:
- [F] full text read here.
- [A] abstract only.
- [S] secondary: another research note, a search snippet, or a summarising fetch.
- [D] derived here, with the arithmetic shown.
- [J] judgement.

Files:
- Downloads in `data/q2_<source>/`, with `SOURCE.txt` and the extracted `paper.txt`.
  - `q2_skilbrei2016`, `q2_modica2011`, `q2_mergardt1997`, `q2_depontual2013`, `q2_albert1995`.
  - Metadata only: `q2_unpaywall`, `q2_s2`, `q2_europepmc`. The OpenAlex daily budget was used up all session.
  - `q2_rindorf2002` holds only the DTU Orbit landing page. The postprint returned 403 to both curl and the fetch tool.
- Read in place in sibling notes, not copied:
  - research_notes/"Mättnad hos rovfisk"/data: `armannsson2012/paper.txt`, `homrum2013/paper.txt`.
  - research_notes/"Intag under födosök"/data/`rindorf_gislason2005/paper.txt` (DTU postprint).
  - research_notes/"Vitlingens fiskbyten"/data/step2_lit/vertical: `onsrud2004.txt`, `adlerstein_ehrich_CM2000K01.txt`, `bromley1997_icesjms54.txt`.
  - research_notes/"Vitlingens fiskbyten"/data/lit: `pedersen_19426.txt`, `pedersen_19427.txt` (abstracts).
- Rindorf 2003 full text: `data/rindorf2003_meps249/paper.txt`.
- Scripts:
  - `scripts/q2_daylength.py` -> `tables/q2_daylength_stdout.txt` (astronomy only).
  - `scripts/q2_make_csv.py` -> `tables/q2_raw.csv`.
  - `scripts/q2_oa_search.py`: the OpenAlex helper. It was rate limited all session, so no results.

Conventions:
- h = height above the bed; z = depth.
- A "window" is the clock period in which a prey type is ingested. It is NOT a time share spent in a mode. No
  source in this note measures the share of time an individual whiting, saithe or hake spends in a foraging mode.
- Sources marked "held" in the brief are cited, not re-derived.

## Day length used for windows [D]

`tables/q2_daylength_stdout.txt`. Day = sun centre above -0.833 deg. Night = sun below -6 deg (civil). Twilight = the
difference, summed over both ends.

The whiting sites use the clock times printed in Rindorf & Gislason 2005 Table 1 (sunrise, sunset and civil
dawn/dusk per location). The other rows are computed. The box latitude is the manifest's `latitude_deg` 58.15.

| case | day h | day/24 | night/24 | twilight h (both ends) |
|---|---|---|---|---|
| Rindorf sites 1-5, early Sept (R&G Table 1) | 13.37-13.83 | 0.557-0.576 | 0.369-0.392 | 1.23-1.32 |
| Rindorf Loc. 5 (137 m) | 13.45 | 0.560 | 0.385 | 1.30 |
| Mergardt & Temming, 30 May-1 Jun, 57.53 N | 17.49 | 0.729 | 0.183 | 2.11 |
| Modica et al., 27 Jun, 41 N | 15.12 | 0.630 | 0.324 | 1.12 |
| Box 58.15 N: 21 Mar / 21 Jun / 5 Sep / 21 Dec | 12.20 / 18.23 / 13.76 / 6.42 | 0.51 / 0.76 / 0.57 / 0.27 | 0.44 / 0.13 / 0.37 / 0.66 | 1.3 / 2.6 / 1.4 / 1.7 |

---

# Part 1. Per source

## Whiting (Merlangius merlangus)

### W1. Rindorf 2003, MEPS 249:265-276, doi:10.3354/meps249265 [F] (held; re-read for numbers)

Design (p. 266):
- "whiting were caught by bottom trawling at 5 locations in the North Sea in early September 1988 (Locations 1
  and 2) and 1989 (Locations 3, 4 and 5) ... depth (44, 46, 53, 73 and 137 m at Locations 1 to 5) and temperature
  (16.7, 13.7, 10.6, 8.2 and 8.6 C ...). One haul was taken every 4 h in a total of between 48 and 72 h".
- "stomachs from 7738" whiting, 20-35 cm.

Modes by prey, from the abstract:
- "the occurrence of bottom dwelling prey increased significantly during the night at 4 of 5 locations".
- "free swimming prey and prey migrating towards the demersal layer during the day were eaten mainly in the
  daylight hours".

Krill (p. 270):
- "Krill seemed to be ingested primarily around dawn and in the early afternoon until some time after dusk".
- Discussion: "The intake around midday was limited, perhaps indicating that krill is more vulnerable during the
  period of ascent and descent. Krill appeared to be a patchily distributed prey as some periods were
  categorised by complete lack of krill in the stomach contents."

Benthic prey and dab (p. 270):
- "Dab was found primarily at night at all locations ... A second peak was found in the late morning hours,
  ranging from 07:00 h at Location 3 to 11:00 h at Locations 1 and 2."
- Shrimp at Location 2 "appeared to be eaten in the daylight hours and into the early evening". These are
  Pandalus, which are on the bed by day.

The deepest site, the most box-like (p. 273):
- "The lack of 'other benthic invertebrates' was mirrored by the virtual absence of both crabs and flatfish,
  suggesting perhaps that whiting at this location did not forage on the bottom ... The lack of benthic foraging
  may have been linked to the greater depth of this location."

Predator vertical migration (p. 274):
- "no common diel pattern [in catch rates] could be detected for any of the size groups (p > 0.26)".
- "in most cases, no evidence of vertical migration was found".

The method gives no time shares (p. 274):
- "the method cannot be used to estimate the quantitative importance of each time of day in the total ingestion of
  a given prey. Only the times of day when a large proportion of the predators have fed on a given prey can be
  revealed."

Coupling of the two feeding periods (p. 275):
- "if feeding is effectively divided into 2 separate periods, an increase in the abundance of 'daytime' prey will
  not affect the relative density of 'nighttime' prey experienced by the predator. Any effect would have to be
  mediated through satiation of the predator."

Table 4: share of deviance explained by the common diel term bt and the location-specific term abloc,t, given as bt + abloc,t. Crab 0.31 + 0.05; dab 0.17 + 0.21; shrimp 0.11 + 0.29; krill 0.093 + 0.219; herring 0; sprat 0.

Also via this paper [S]: Pedersen's site was 90 m and 6 C, where ~85 % of the stomach content of 20-35 cm whiting
by weight was fish (p. 274).

**Reading:**
- [D] At the four shallow sites the benthic mode is a night mode. In early September the civil night lasts
  0.369-0.392 of 24 h (R&G Table 1 clock times).
- [F] At 137 m the benthic mode was absent.
- [J] The krill mode sits at the light transitions plus the afternoon.

### W2. Rindorf & Gislason 2005, J Exp Mar Biol Ecol, doi:10.1016/j.jembe.2005.04.013 [F]

DTU postprint, read in "Intag under födosök". Same five-site September data.

Abstract:
- "Though a saturation level was reached for all types of fish prey, none of the levels corresponded to the maximum
  digestive capacity of the predator. This was not caused by ingestion of other prey as the amount of other food
  and fish prey ingested were not negatively correlated. ... fish was ingested almost exclusively during dawn and
  dusk and the lack of negative correlation between the intake of fish and other prey may thus be a result of the
  limited time in which fish prey was vulnerable to predation."

Results:
- Maximum daily intake "4.3 to 6.2% body weight ... at the warmest location and around 3.1% body weight at the
  coldest".
- Other food: "the predators maintained an intake of other food between 8 to 22%, 10 to 20% and 8 to 17% of maximum
  daily ration for predators of length 20 to 24.9 cm, 25 to 29.9 cm and 30 to 34.9 cm".
- "The fish intake in per cent of maximum varied between locations by a factor 36, 29 and 17 ... whereas the
  intake of other food varied by no more than a factor 3. There was no significant correlation between the intake
  of fish and other prey ... (correlations = -0.30 to 0.73, P>0 = 0.16 to 0.81)."
- "the intake of the individual prey groups in no case exceeded 20% of maximum daily ration".
- "The multispecies functional response provided very poor fits to the consumption of both larger predator groups".
  - The form is Murdoch's shared-handling Type II, n_i = a_i N_i / (1 + sum a_m t_m N_m).
  - It was fitted on FISH prey only: herring, sprat, and a gadoid group of whiting, haddock and pout.
    "One observation corresponded to the intake of one prey species at one location."
  - The failures were "A high gadoid intake at location 3, where herring density was so high, that few other prey
    should have been consumed according to the model, and a high herring intake compared to that of sprat at
    location 1".
  - The authors explain them by patchiness of schooling prey: "predators which overlapped with the prey became
    saturated while predators in other areas fed on other prey".
  - So this is a test of shared handling among prey within the fish mode, not across modes.
- Diel timing: "All prey species but Norway pout showed two diel peaks in ingestion, one in the morning and one in
  the afternoon ... corresponded closely to the period around dawn and sunset ... The ingestion around midday was
  low ... Norway pout showed only one peak around 15.00 hours ... Ingestion during darkness was low for all
  species but whiting at location 5."

Discussion:
- "Consumption of clupeids by whiting was saturated around a total intake of 10 and 17% of maximum daily ration for
  predators of size 25 to 29.9 cm and 30 to 34.9 cm, respectively, whereas gadoid consumption saturated around 6%".
- No specialists: "no indication was found in the stomach contents that individual predators foraged consistently
  on one prey type".
- "the length of the twilight period may have eliminated the direct choice between fish and other prey".

**Reading [D]:**
- Civil twilight spans 1.23-1.32 h per day at the five sites (Table 1).
- Hauls were 4 h apart, so neither peak can be resolved narrower than one 4-h bin.
- The fish window is therefore bounded by twilight only (0.05 of 24 h) and two 4-h bins (8 h, 0.33).
- The width inside these bounds is resolution-limited: no data.

### W3. Mergardt & Temming 1997, ICES J Mar Sci 54:226-242, doi:10.1006/jmsc.1996.0190 [F]

Bronze OA, `data/q2_mergardt1997`.

Design:
- "Whiting were trawled from a 10 nmi x 10 nmi box at 57.53N, 0.54W during 24 h fishing in May/June 1992. A total
  of 23 hauls ... GOV-trawl ... 30 whiting (size class 25-29.9 cm)".
- Sandeels were the only prey.

Results:
- Abstract: "a single feeding peak with a maximum feeding between 2200 and 2400 h and minimal food intake between
  0800 and 1000 h ... whiting appear to feed during the night hours".
- Catches: "Whiting catch rates were less variable throughout the day, but the two hauls performed at darkness gave
  significantly lower catch rates than the day hauls."

Secondary citations in the discussion [S]:
- Patterson 1985 and Blaxter & Parrish 1958: "during night-time whiting tend to disperse in the upper water layers,
  while they concentrate near the bottom during the day".
- Sandeels "bury in the substratum by night". Dusk and dawn were 2130 h and 0230 h GMT.
- Hall et al. 1995 found a whiting peak at 0300-1000 h instead.

**Reading [J]:** a fish mode (sandeel) at night in late spring. It fits neither the September dawn/dusk fish
windows nor a fixed daylight-only fish mode. Timing follows the prey's availability.

### W4. Pedersen 1999, J Fish Biol 55:1096, doi:10.1111/j.1095-8649.1999.tb00744.x; Pedersen 2000, J Fish Biol 57:402, doi:10.1006/jfbi.2000.1316 [A] (held)

Pedersen 1999, diet:
- "Pelagic North Sea whiting ... preyed upon (pelagic) sprat and sandeel, while demersal whiting preyed upon
  (demersal) Norway pout and herring ... Diet overlaps within layers were low".

Pedersen 2000, feeding time and ration:
- "Pelagic North Sea whiting ... fed at night, while demersal whiting fed by day."
- Specific daily ration "4.38 to 7.84% in 1992 and from 3.99 to 10.31% in 1993" with in situ evacuation, but
  "0.41 to 1.66%" (1992) and "0.78 to 1.75%" (1993) with Andersen's model.

Via Rindorf 2003, p. 274 [S]:
- "catch rates of whiting in the pelagic trawls decreased during daylight (Pedersen 1999)".
- The site was 90 m deep and 6 C.

**Reading [J]:** at one ~90 m site, two layers held whiting with different prey and opposite feeding times. Neither
source gives the share of the population or of time in each layer.

### W5. Onsrud, Kaartvedt, Rostad & Klevjer 2004, ICES J Mar Sci 61:1278 [F] (held; read in "Vitlingens fiskbyten")

Oslofjord, 120 m bed.

Layers by day:
- "The krill layer was restricted to waters deeper than ca. 70 m during the day, mainly concentrated in a 20-40-m
  thick layer".
- "During daytime, herring and whiting were foraging in the upper and middle part of the krill assemblage".
- Bottom-trawl catches of whiting by day were 0-1 fish per haul (Table 2).

At night:
- "herring, whiting, and sprat were most abundant in the upper 50 m".

Feeding:
- "For whiting, krill constituted up to 99% of the stomach content by weight, on average ~82% during both day and
  night ... No systematic trends were apparent comparing proportion of stomachs containing krill or state of
  digestion during the day and night".
- "We suggest that whiting was foraging on krill during both day and night in the Oslofjord."

**Heights [D]:** day krill layer at z 70-110 m over a 120 m bed puts whiting at h ~10-50 m; at night z 0-50 m puts
them at h ~70-120 m.

**Reading [J]:** one pelagic krill mode held round the clock, with depth tracking the krill. There is no benthic
mode at this site.

Onsrud et al. 2005 (held [S]): swimming 14-16 cm/s by day and 10-11 cm/s by night, read as a switch from visual to
non-visual feeding.

### W6. Adlerstein & Ehrich 2000, ICES CM 2000/K:01 [F] (read in "Vitlingens fiskbyten")

- GOV experiment, northern North Sea, November 1997.
- "catch rates except those of large whiting vary significantly with time of day".
- "catch rates of dab, grey gurnard and small whiting are higher at night". Whiting > 20 cm showed no effect.

**Reading [J]:** adult whiting's availability to a bed trawl (headline ~4-5 m) does not change between day and night.
Any night rise of adult whiting in that area is small.

### W7. Box-area pelagic hauls (Danish HERAS cruise reports 2017-2019) [S]

Via research_notes/"Vitlingens fiskbyten"/data/step2_README.md: "whiting only in day bottom hauls; 0 whiting in
night surface hauls in the box rectangles" (June-July).

### W8. Not used

- Bromley, Watson & Hislop 1997 (ICES J Mar Sci 54:846) is on pelagic 0-group gadoids, 1-5 cm, so out of size range.
- Albert 1995 (ICES J Mar Sci 52:873, `data/q2_albert1995`) [F] covers haddock, Norway pout and poor-cod on a coastal bank off SW Norway (75 and 140 m), with no whiting, saithe or hake. It is kept only as a gadoid example of prey-specific diel windows: haddock took polychaetes and isopods "mainly in the early morning and from noon until a few hours after sunset".
- No tag study (DST or acoustic telemetry) on adult whiting was found. A web search and the sibling note
  "Mättnad hos rovfisk" (lit_foraging_limits.md, gap list) both came up empty. This is a gap.

## Saithe (Pollachius virens)

### S1. Armannsson & Jonsson 2012, ICES J Mar Sci 69:1372-1381, doi:10.1093/icesjms/fss121 [F]

Full text in research_notes/"Mättnad hos rovfisk"/data/armannsson2012/paper.txt. It was not available to the
earlier note.

Design:
- 51 DSTs recovered, "268 018 hourly depth recordings". Fish "ranged from 42 to 83 cm (mean length 54 cm)", Iceland.
- 19 tags were analysed at 1-min resolution.
- Depth only: no bed depth and no height above the bed.

Seasonal depth (p. 1375):
- "The median depth was largest during March-April (140 m) but shallowest during summer and autumn (<50 m)".

Diel activity (p. 1375):
- "The average hourly depth change was highest during daylight hours, and nocturnal activities were much reduced
  in winter ... more active in the early hours of daylight compared with later in the day. The activity is lowest
  in July".
- Trawler catches: the "proportion [of saithe in trawler catches] was highest during the hours of darkness but
  diminished during the hours of daylight".
  - The authors give three explanations: reduced night activity (catchability), gear avoidance by day ("during
    daylight the fish can see the gear at a greater distance"), and denser aggregation at night (Neilson et al.
    2003).
  - "Nearer the bed at night" is my inference, not theirs [J].

Excursions (p. 1376):
- "Time spent moving rapidly (moving at >3 m min-1) up or down varied between individuals and ranged from 0.2% to
  33% (average 15%) (Figure 4)".
- Within the free vertical range on "the majority of days (63%) ... On 22% and 2% of the days, individuals
  travelled above or below the FVR limit, respectively, and on 13% of days the saithe crossed both".
- "DVRs calculated with hourly intervals underestimated the DVR by 70%".

Discussion (p. 1377):
- "adult saithe around Iceland prey mostly on pelagic fish and krill ... Reduced activity during the hours of
  darkness could be attributed to the fact that saithe are visual feeders that forage in the daylight hours".

**Reading:**
- [F] An active day mode and a passive night mode.
- [D/J] Vertically fast movement takes ~15 % of time (0.2-33 %), a lower bound since a run needs > 3 m/min for
  >= 1 min. These are candidate pelagic excursions, but the tags record no prey, and no day/night split of the runs
  is given.
- There is no split into benthic and pelagic modes, and no height above the bed.

### S2. Skilbrei & Ottera 2016, ICES J Mar Sci 73:1186-1195, doi:10.1093/icesjms/fsv261 [F]

`data/q2_skilbrei2016`. Acoustic transmitters on saithe at 15 salmon farms in a Norwegian fjord system (beds to
700 m).

Fish:
- Releases of 30 and 32 tagged fish, 38-49 cm and 42-59 cm (Table 1). 12 of the 32 had shallow-limit tags and were
  excluded. Mostly age 3.

Depth:
- "Typical residence depth was from 25 to 50 m in summer and 60 to 90 m during winter ... The fish were usually
  distributed through 100 m of the water column."

Diel patterns (p. 1190):
- "The most common behaviour was the lack of a diel rhythm ... (38.2% of all observations; Figure 7). The two
  alternatives, to move deeper during day or night, were both frequent throughout the year (17.4 vs. 23.6%, and
  4.5% switched between them within the monthly intervals)."
- Abstract: "Fish also moved away from fish farms in daytime (16% of all observations), probably to feed
  elsewhere."
- "Seventy-five percent of the individuals displayed diel movement pattern parts of the time (30 of 40)."

Activity (p. 1189):
- "Saithe activity increased rapidly and reached its maximum around time of sunrise ... active during most of the
  day until sunset ... less active during night. ... in December, mean SD was 9.1 m from 7 to 10 a.m. (95% CI
  7.9-10.3 m), compared with 3.8 m during the period from 18 p.m. to 4 a.m. (3.3-4.4 m)".

Discussion (p. 1191):
- "Several studies have demonstrated that gadoids are active during day and are more passive and usually more
  closely bound to the bottom during the night".
- The "DVM I behaviour may be typical of fish that spend the night in relatively shallow water, and are actively
  searching for food in deeper water during daytime".

Transfer to the box [J]: these are farm-associated fjord fish. Treat the shares as a bracket on individual
variability, not as box values.

### S3. Homrum et al. 2013, ICES J Mar Sci 70:782-792, doi:10.1093/icesjms/fst048 [F]

Conventional tags only, no depth data.
- "Saithe are regarded as a demersal fish, but they do exhibit pelagic behaviour as well ... The pelagic behaviour
  is also reflected in the diet, which is not directly associated with the seabed. The main prey items include
  pelagic crustaceans such as copepods, euphausiids".
- "The pelagic behaviour is evidenced by the bycatch of saithe in the blue whiting fishery".
- "a pronounced proportion of the tags were reported by purse seiners in July-September" (Norwegian releases).

**Reading [J]:** a seasonal pelagic phase in summer, with no time shares.

### S4. Sibling notes (held) [S]

From research_notes/"Sejens hojd over botten" and "Sejens profil over djupa bottnar":

- Profile by day: exponential from the bed, Hs 15 m (in-area 9-31 m), bottom-2-m share 0.125.
- HERAS (June-July), in the area:
  - adults in none of 75 day hauls fished more than 60 m above the bed;
  - 0 of 63 day surface hauls over beds >= 150 m held adults;
  - 11 of 82 night surface hauls held 70-106 cm adults.
- Stensholt et al. 2001 (Oct-Nov): "For depth less than 200m the saithe stay close to seabed"; 60-75 % within
  0.1(bed+10) m; day/night difference "not clear".
- Neilson et al. 2003 (Scotian Shelf, pre-spawning): Hs 15 / 9 m day / night at one site, so nearer the bed at
  night.
- Kaartvedt et al. 2005: saithe in midwater beneath krill swarms over deep water, eating fish.
- Saithe feed mainly by day, and haddock in saithe food peaks in Q3.

Sarno et al. 1994, J Fish Biol 45:811 [S, search snippet of abstract]: "Pollack swam more slowly than saithe during
the day, but at the same speed at night" (two 35-38 cm saithe tracked 170 h at a reef).

## European hake (Merluccius merluccius)

### H1. de Pontual, Jolivet, Bertignac & Fablet 2012, J Fish Biol 81:728-734, doi:10.1111/j.1095-8649.2012.03345.x [A]

Europe PMC abstract: "A pioneering experiment of archival tagging of European hake ... provided evidence of a diel
vertical migration pattern ... Frequent vertical movements across the thermocline were observed and fish
experienced rapid temperature changes >7 C. These tagging-recapture data also suggested a homing behaviour to
feeding grounds."

### H2. de Pontual, Jolivet, Garren & Bertignac 2013, ICES J Mar Sci 70:1416, doi:10.1093/icesjms/fst102 [F]

Archimer, `data/q2_depontual2013`.
- Table 1: the DST fish were 115 hake of 30-39.5 cm (2006) and 144 of 28.9-39.2 cm (2007). Nine were recovered,
  with at most 81 days at liberty. These are 29-40 cm fish from the Bay of Biscay, mostly immature [J]. No hake
  over 40 cm carried a DST.
- Discussion: "data storage tag (DST) tagging showed that European hake are much more eurythermal than previously
  thought as they can experience high temperature gradients and SSTs as high as 20 C during diel vertical movements
  to forage (de Pontual et al., 2012)".

**Reading [J]:**
- The 29-40 cm hake moved from deeper water across the thermocline into water at the SST, "to forage" in the
  authors' wording.
- The phase (night up) is assumed by analogy with the juveniles in H3-H5. The obtained text does not state it [J].
- No time shares or depth ranges are in the obtained text.
- No tag or diel data exist for adult hake (> 40 cm) in this note.

### H3. Modica, Bozzano, Velasco, Albertelli & Olaso 2011, MEPS 440:177-189, doi:10.3354/meps09341 [F]

`data/q2_modica2011`.

Design (Table 1):
- Juveniles, 9.8-11.1 cm mean TL. 24 h bottom-trawl series at 105-108 m, NW Mediterranean, 26-29 June 2000.
- "Sunrise and sunset were at 06:20 and 21:29 h". The catch times are "hh:mm, GMT".
  - Computed for 41-42 N on 27 June, sunrise is ~04:20 GMT and sunset ~19:30 GMT [D].
  - So the printed sunrise and sunset are most likely local summer time (GMT+2), on a different clock from the
    catch times [J].

Diel feeding (abstract):
- "Predatory activity ... was initiated in the early night and continued throughout the night, reaching maximum
  activity in the early morning. The feeding activity progressively decreased after this time, reaching a minimum
  before sunset, when a rest phase was observed."
- "the macrozooplankton (euphausiids) was consumed at night and the hyperbenthos (mysids) during the day".
- "early juveniles (9.3 +- 1.7 cm ...) primarily fed on these resources in these periods. In contrast, advanced
  juveniles (12.0 +- 3.4 cm TL) preyed largely on decapod crustaceans and fishes and fed mostly during the day".

Rest phase, by sample (Table 1, Table 2):
- At 20:25 GMT, 34.5 % empty stomachs and stomach content 0.81 % BWW, against 3.49 % at 05:37.
- Contents at 17:30, 19:20 and 18:03 were 1.30, 1.49 and 3.35 % BWW.

Prey by time:
- "at night, this group was mostly constituted of euphausiids (69.8%) ... in the afternoon mysids (95.4%) were
  dominant".
- "during the night, almost half of the analysed individuals profit from euphasiids".
- Mode logic (p. 185): euphausiids and mysids "do not co-occur and hence the predator in reality has the choice of
  eating the prey that is there or not eating at all".

Daily ration: 4.08-5.09 % BWW (square-root model), 1.5 % with the exponential model.

**Reading [D], low confidence:**
- The low samples are 17:30 (29 June), 19:20 and 20:25 GMT, i.e. around sunset (~19:30 GMT).
- The 18:03 GMT sample from 26 June is high (3.35 % BWW), so the rest window is not cleanly bounded. It spans
  1-3 h, 0.04-0.125 of 24 h, and the authors' "before sunset" depends on which clock is meant.
- The pelagic euphausiid mode coincides with the night: 0.32 of 24 h at 41 N on 27 June.
- No day-mode share is derived.
- [J] These are juveniles in the Mediterranean; the transfer to adult Skagerrak hake is weak.

### H4. Bozzano, Sarda & Rios 2005, Fish Res 73:29-36, doi:10.1016/j.fishres.2005.01.006 [S]

Closed access. Abstract wording via a search snippet: "The 95.5% of the pelagic captures were obtained during the
night, while the maximum catches of the bottom net were obtained at midday, indicating that juvenile hake may
display nocturnal vertical migration. Very small individuals (<5cm TL) were obtained only with the pelagic net".
Also, "Feeding represented one of the possible constraint factors controlling the vertical migration".

### H5. Carpentieri, Colloca & Ardizzone 2008, JMBA 88:1493-1501, doi:10.1017/S0025315408001859 [S]

Abstract via a summarising fetch and search snippet. Juveniles < 16 cm, central Mediterranean, four 24 h surveys.
- Main prey: the euphausiid Nyctiphanes couchii.
- "The fullness index was generally higher during sunrise when migratory activity ceased and juvenile hake return
  to the bottom after feeding."

### H6. In-area hake position [S]

From research_notes/"Sejens hojd over botten": HERAS June-July, "hake in 32/43 day bed hauls". No bed-referenced
night data for hake in the area.

Prior-note proxy [S, A]: Cape hakes (Pillar et al. 1997, ICES J Mar Sci 54:485) show little diel feeding
periodicity, with up to 1/3 of fed fish having fed recently at any hour.

### H7. Not found

- No Norwegian, Skagerrak or North Sea hake tag study with depth was found. Searches were the web and the earlier
  notes; OpenAlex was rate limited all session.
- No adult European hake diel stomach series from the North Atlantic was obtained. Velasco's 2007 Cantabrian thesis
  (eprints.ucm.es) timed out on download.

## Satiation sources

### G1. Rindorf 2002, J Fish Biol 61:579-593, doi:10.1111/j.1095-8649.2002.tb00897.x [A/S]

Full text not obtained: the DTU Orbit postprint returned 403, and CORE is out of scope.

Content via search snippets of the abstract:
- The probability of a whiting stomach containing fresh food was depressed when partially digested food was
  already present.
- Meal size did not change with fullness, read as satiation acting on search activity, not on meal or prey-size
  selection.
- Same September 5-site data.

Fall & Fiksen 2020, Fish Fish (doi:10.1111/faf.12415) [F, in "Intag under födosök"], full sentence: "A trade-off
with predation risk was a proposed explanation for a reduced feeding probability with increasing stomach fullness
observed for whiting (20-36 cm) in the North Sea (Rindorf, 2002). The reduced feeding probability was not related to
absolute gut limitation, suggesting that satiation effects on gadoid feeding may be attributable to several
different" [mechanisms].

The brake is therefore read as behavioural (state-dependent risk taking, search), not as a full gut. Whether the
risk explanation is Rindorf's own or Fall & Fiksen's paraphrase cannot be checked without the full text.

**Gap:** whether Rindorf 2002 tested the effect per prey type (does old fish depress fresh krill?) is unknown
without the full text.

### G2. Other evidence on the brake

- Dempster et al. 2011, via research_notes/"Mättnad hos rovfisk" [S]: wild saithe beside salmon farms hold 2.35x
  the stomach content of saithe away from farms, and half the empty share (0.16 vs 0.31). The field gut is not
  usually full.
- Skilbrei & Ottera 2016 [F]: farm saithe "had fuller stomachs than saithe captured further away from farms, but
  also found considerable amounts of natural prey like pelagic crustaceans". Natural feeding continues beside the
  subsidy.

---

# Part 2. Synthesis per species

Two things are measured nowhere: the time share an individual spends in a mode, and intake per mode. The rows
below are the field windows and positions that exist. "no data" is written where nothing was found.

## Whiting, 20-35 cm

Three prey-defined modes appear in stomach timing:
1. **Bed search for benthic prey**: crabs, "other benthic invertebrates", dab, Crangon. Night at 4 of 5 sites of
   44-73 m [F]. Absent at the 137 m site [F]. At ~90 m, 85 % of the diet is fish [S].
2. **Pelagic krill.**
   - North Sea, September: dawn plus early afternoon to after dusk; low at midday [F].
   - Oslofjord, 120 m: day and night, 82 % of the diet, the fish moving with the krill layer [F].
3. **Free-swimming fish.** Timing depends on prey and season:
   - dawn and dusk peaks in September, clupeids and gadoids [F];
   - single night peak on sandeel in May-June [F];
   - demersal whiting feed by day, pelagic whiting at night [A].

Vertical position of the predator, by day:
- Near the bed: HERAS day bottom hauls only [S]; Blaxter & Parrish and Patterson via Mergardt [S].
- In the krill layer, h 10-50 m, at a fjord site [F/D].

Vertical position at night:
- Partly up into the upper 50 m in a fjord [F].
- Pelagic trawl catches higher at night [S].
- GOV bed-trawl catches in the two dark hauls lower than by day [F] (Mergardt & Temming, May-June).
- Against this:
  - no time-of-day effect on GOV catch rates of > 20 cm whiting in November [F];
  - no diel catch pattern in Rindorf's bottom trawls [F];
  - 0 whiting in the box's night surface hauls [S].
- [J] The night rise is site- and season-dependent and was not found in the box's summer surface hauls.

Schooling vs dispersed: Onsrud 2004 [F] has "fish schools" above or in the upper krill layer by day that "dispersed
at dusk". Herring and whiting are not separated in the acoustics. No adult whiting school observation was found.

## Saithe, 38-83 cm

Two activity states are documented:
1. **Day active / foraging.**
   - Activity peaks at sunrise and stays high to sunset [F, two tag studies].
   - Profile exp(-h/15 m) from the bed by day in the area (held, [S]).
   - Rapid vertical runs take ~15 % of time (0.2-33 %) [F]. This is the only tag-based number for "excursions",
     with no day/night split.
2. **Night rest.**
   - Hourly depth change much reduced, most strongly in winter: SD 3.8 vs 9.1 m in December [F].
   - Nearer the bed at night: Hs 9 vs 15 m at one Scotian Shelf site [S].
   - A larger trawl catch share at night [F]. Its reading as "nearer the bed" is [J]; the authors give activity,
     gear avoidance and aggregation.
   - But a few large adults reach the surface at night (11 of 82 night surface hauls over beds >= 150 m) [S].

Schooling vs dispersed [F/S]:
- Saithe "become more densely aggregated at night" (Neilson et al. 2003, cited by Armannsson & Jonsson and by
  Skilbrei & Ottera).
- Skilbrei & Ottera: "saithe are less aggregated and more active during the day. Reefs may make up the core area of
  the home range, where fish aggregate at night, and from which they disperse over a larger area during the day".
- [J] The pattern is aggregated at rest by night and dispersed while foraging by day.

Prey mode:
- Not separable by tags.
- The diet is pelagic: krill and fish [F/S].
- Midwater beneath krill swarms [S].
- No benthic-prey mode is reported.

Farm fish (fjord), behaviour categories per fish-month [F]:

| category | share of fish-months |
|---|---|
| no DVM | 0.38 |
| deeper by day | 0.17 |
| deeper by night | 0.24 |
| switching | 0.045 |
| away from the farm by day | 0.16 |

## Hake

- **29-40 cm (mostly immature), Bay of Biscay DSTs** [A, F-secondary]:
  - a diel vertical migration across the thermocline into SST water, "to forage";
  - the phase (night up) is assumed [J].
- **Day position in the area:** HERAS, 32 of 43 day bed hauls held hake [S]. No shares or ranges.
- **Adult hake > 40 cm:** no tag or diel data found.
- **Juveniles (< 16 cm, Mediterranean):**
  - pelagic at night, with 95.5 % of pelagic catches at night [S], feeding on euphausiids [F];
  - near the bed by day, with bottom catches peaking at midday [S], feeding on mysids, decapods and fish [F];
  - a 1-3 h low-feeding phase around sunset [F/D, clock ambiguous];
  - fullness peaks at sunrise as they return to the bed [S].

## Tidal patterns

None found for any of the three species in the sources read.

## f_mode table (field windows and positions; usable as f_mode(z) inputs only with the stated tags)

h = height above the bed. "share of 24 h" is a clock window converted with q2_daylength.py, not a measured share of
time in the mode.

| species | mode | day/night | share of 24 h | z or h range | site / season / size | source | tag |
|---|---|---|---|---|---|---|---|
| whiting | benthic search (bed prey) | night | 0.369-0.392 window (Sept) | at the bed (prey bed-bound; h ~0-2 m) | 44-73 m, Sept, 20-35 cm | Rindorf 2003; R&G 2005 Table 1 | F (timing), D (share), J (h) |
| whiting | benthic search | any | 0 (absent) | - | 137 m, Sept | Rindorf 2003 | F |
| whiting | krill (pelagic) | dawn + early afternoon to after dusk | no data (window not quantified; midday low) | no data (North Sea) | 44-137 m, Sept | Rindorf 2003 | F |
| whiting | krill (pelagic) | day | no data (round the clock) | h ~10-50 m (krill layer z 70-110 m at 120 m bed) | Oslofjord, Mar/Jun/Nov | Onsrud 2004 | F, D |
| whiting | krill (pelagic) | night | no data | z 0-50 m (h ~70-120 m) | Oslofjord | Onsrud 2004 | F, D |
| whiting | fish (clupeids, gadoids) | dawn and dusk peaks, low at midday and in darkness | bounds 0.05 (twilight 1.23-1.32 h) to 0.33 (two 4-h bins); width no data | no data | 44-137 m, Sept, 20-35 cm | Rindorf & Gislason 2005 | F, D |
| whiting | fish (sandeel) | night, max 22-24 h, min 08-10 h | no data | sandeel in water column by day, buried at night | 57.5 N, May-Jun, 25-30 cm | Mergardt & Temming 1997 | F |
| whiting | demersal subpopulation (pout, herring) | day feeding | no data | near the bed | ~90 m North Sea, 1992-93 | Pedersen 1999, 2000 | A |
| whiting | pelagic subpopulation (sprat, sandeel) | night feeding | no data | pelagic | same | Pedersen 1999, 2000 | A |
| whiting | position, adults > 20 cm | day vs night | no diel catch difference in GOV (headline ~4-5 m) | bed layer both | N North Sea, Nov | Adlerstein & Ehrich 2000 | F |
| saithe | active (foraging) | day | day length 0.27 (Dec) - 0.76 (Jun) at 58.15 N | exp(-h/15 m), Hs 9-31 m (held) | box area, Jun-Jul | sibling notes; Armannsson 2012; Skilbrei 2016 | S, F, D |
| saithe | rapid vertical runs (> 3 m/min) | no day/night split | 0.15 (0.002-0.33) of time | DVR rises with depth | Iceland, 42-83 cm | Armannsson & Jonsson 2012 | F |
| saithe | rest (aggregated) | night | night length (0.13-0.66 by season, 58.15 N) | nearer the bed: Hs 9 m vs 15 m day | Scotian Shelf, Oct | Neilson 2003 via sibling | S |
| saithe | night surface excursion | night | 11/82 surface hauls held adults (presence, not time) | z 0-~30 m over beds >= 150 m | box area, Jun-Jul | HERAS via sibling | S |
| saithe | farm fish: no DVM / deeper by day / deeper by night / away by day | - | 0.38 / 0.17 / 0.24 / 0.16 of fish-months | z 25-50 m summer, 60-90 m winter (fjord) | Norway fjord, 38-59 cm | Skilbrei & Ottera 2016 | F |
| hake | 29-40 cm vertical migration "to forage" | phase assumed night | no data | across the thermocline to SST water (up to 20 C) | Bay of Biscay | de Pontual 2012, 2013 | A, F, J (phase) |
| hake | day on the bed (all sizes caught) | day | no data | bed layer | box area, Jun-Jul | HERAS via sibling | S |
| hake | adult (> 40 cm), any mode | - | no data | no data | - | - | - |
| hake | juvenile pelagic euphausiid feeding | night | 0.32 window (27 Jun, 41 N) | pelagic (95.5 % of pelagic catches at night) | NW Med, 9-12 cm | Modica 2011; Bozzano 2005 | F, S, D |
| hake | juvenile near-bed feeding (mysids by day; decapods and fish for 12 cm) | day | no data | bed / hyperbenthos | NW Med, 106 m | Modica 2011 | F |
| hake | juvenile low-feeding phase | around sunset | 0.04-0.125 (1-3 h; clocks ambiguous) | on the bed (bottom trawl catches) | NW Med | Modica 2011 | F, D (low confidence) |

---

# Part 3. One gut brake or separate saturation per mode or prey?

## What each source tested

| source | what was tested | result | tag |
|---|---|---|---|
| Rindorf 2002 | P(fresh food) against the amount of older food (all prey pooled as far as known); meal size against fullness | fullness depresses the probability of a new meal (search), not meal size; not absolute gut limitation; a predation-risk trade-off proposed (per Fall & Fiksen) | A, S |
| Rindorf & Gislason 2005 | single-prey functional responses (herring, sprat, whiting, clupeid and gadoid groups); correlation of fish and other-food intake across 5 sites; Murdoch shared-handling multispecies Type II fitted among FISH prey only | each fish prey saturates at 6-17 % of max ration, far below the gut maximum; other food 8-22 %; fish vs other food not negatively correlated (r -0.30 to 0.73, n.s., n = 5 sites); multispecies fish response "very poor fits", attributed to prey patchiness | F |
| Rindorf 2003 | timing of each prey type | bed prey at night, krill at the transitions, fish by day; "Any effect would have to be mediated through satiation" | F |
| Modica 2011 | timing of each prey type, juvenile hake | euphausiids at night, mysids by day, "do not co-occur"; a low-feeding phase around sunset | F |
| Dempster 2011 (via sibling) | stomach content with and without an easy subsidy | x2.35 with a farm, so the natural gut is not at its ceiling | S |

## Reading [J]

1. **What is established for whiting.**
   - Each fish prey, and fish as a whole, saturates at 6-17 % of the maximum ration, far below the gut maximum
     (Rindorf & Gislason 2005 [F]).
   - The only shared-handling test was among fish prey within one mode, and it fitted badly. The authors blame
     patchiness: predators caught in a haul did not all overlap the prey school.
2. **What does not discriminate.**
   - Fish and other-food intakes are uncorrelated over five sites. But all intakes were far below the maximum
     ration, and far from the ceiling neither a shared handling budget nor a shared fullness brake would produce
     crowding.
   - This non-correlation therefore cannot tell "separate per-mode limits" from "one shared brake".
3. **The authors' mechanism for the low plateaus is time-limited vulnerability.**
   - Fish are vulnerable at twilight, bed prey at night, krill at the transitions. Each mode's intake is limited by
     its window and encounter rate.
   - Rindorf 2003 adds that day and night prey can then interact only through satiation.
4. **There is one shared, soft gut-state brake on search** (Rindorf 2002 [A/S]).
   - Fullness from earlier meals lowers the probability of a new meal, not meal size. It is "not related to absolute
     gut limitation" and is read as a risk or search effect.
   - As far as the abstract shows, the older food is pooled over prey. Whether it acts per prey is not known.
5. **Lean of the evidence.**
   - Per-mode intake limits set by time window x encounter, plus one shared soft brake on search through fullness.
   - Not one shared Holling handling budget across modes: the only fit of that form (within fish) failed, and the
     plateaus sit far below the gut maximum.
   - Each leg rests on one study: one study, five sites, one month for the plateaus, and one abstract for the brake.
6. **No data on:** the size of the brake per unit fullness; whether krill in the stomach brakes fish search; and any
   saithe or hake test. Dempster's farm saithe show only that the natural gut is far from full.

---

# Part 4. Gaps

1. No measured time share per foraging mode for any of the three species. Tags give activity (saithe) or depth
   (saithe, small hake), never prey. Stomachs give windows, never time budgets.
2. No whiting tag study at all.
3. No hake tag study north of Biscay, and no adult hake diel stomach series in the North Atlantic. Velasco 2007
   (Cantabrian, OA thesis) is the next lead.
4. No height above the bed from any DST for any of the three species. Only the held HERAS, Stensholt and Neilson
   profiles for saithe exist.
5. The Rindorf 2002 full text, for the size of the fullness effect and whether it was tested per prey type: DTU
   Orbit postprint https://orbit.dtu.dk/files/134977866/Postprint.pdf returned 403 from here (curl, urllib and the
   fetch tool; retried once with the client form that fetched the 2005 postprint).
6. Paywalled or not accessible: Bergstad 1991 (Sarsia 75), Pedersen 1999/2000 full texts, Bozzano 2005, Sarno et
   al. 1994 (both), Hislop et al. 1991, Temming et al. 2004 (J Fish Biol 64:1351, whiting and haddock on sandeel,
   diel periodicity; abstract fragments only).
7. No tidal foraging pattern was found for any of the three species.
8. Schooling vs dispersed is documented only qualitatively: saithe aggregate at night and disperse by day; fish
   schools disperse at dusk in the Oslofjord. There are no shares.
