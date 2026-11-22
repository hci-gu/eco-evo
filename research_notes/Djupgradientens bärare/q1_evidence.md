# Q1: where adult krill sit by day relative to the bed over 150-250 m beds (2026-11-22)

Literature subagent. Field positions only; no parameter is set or recommended. Rows: `tables/q1_krill_bed_rows.csv`.
Tags: [F] full text read, [A] abstract only, [S] secondary, [D] derived here from the source's numbers, [J] judgement.
"Region" = Skagerrak box, Kattegat, Gullmarsfjord, Oslofjord, Norwegian Trench. Everything else is an analogue.

## Verdict (field positions)

1. **150-175 m beds (region): lower half of the column; bottom edge ~10-35 m above the bed, centre ~15-55 m up (low
   confidence).** In-box July day tows caught krill 9-35 m above 157-166 m beds (131-154, 138-148 m), none at 92-110 m
   over 167 m the same day [D, cited; one day]; Bunnefjorden (154 m, hypoxic) 70-130 m in March (centre ~54 m up),
   almost to the seabed in August [F]; Clyde (140-160 m trench, analogue) 60-100 m in summer (centre ~70 m up),
   to the net limit 15 m above the bed in winter [F]. So not uniformly 50-90 m above, but not on the bed either.
2. **200-250 m beds (region): main layer ~50-120 m above the bed, occasional near-bed presence.** Oslofjord 200 m bed:
   true-day catches 80-158 m, mostly 80-130 m (centre ~70-120 m up), none at 150 m in Nov [F/D]; HERAS 0/4 day tows on
   ~204 m beds [D, cited]; Skagen 250 m bed, Aug: 90-150 m (100-160 m up) [F]; Masfjorden ~195 m, Jan morning: 130-150 m
   (45-65 m up) [F/D, analogue]. GOV (day, presence only, non-closing): 3/16 hauls at 200-250 m and 9/34 at 150-200 m
   in Q3, 0/16 at 200-250 m in Q1 [D].
3. **Scaling:** in the region a roughly constant light-set day centre (~80-150 m depth) fits better than a constant height
   above the bed: gap ~0 at 100 m beds, 10-35 m at 150-170 m, 50-120 m at 200 m, 100-160 m at 250 m, > 300 m at 643 m [J
   on D]. Bottom-referenced cases are analogues where the light-set depth is >= the bed (Scotian basins 20-30 m above
   190-250 m beds; St Lawrence < 200 m beds; Barents bank 160-170 m, T. inermis) [F]. Turbidity shifts the layer ~100 m
   (Kaartvedt 1996) [F]; fjord rows (darker water) likely overstate the gap for the open Skagerrak [J]; Thysanoessa sits
   shallower than M. norvegica [F].
4. **Season: no consistent pattern.** Clyde winter layer broader and nearer the bed [F]; Oslofjord 200 m no clear shift
   May/Aug/Nov [F/D]; Bunnefjorden near the seabed in Aug, O2-limited in Oct [F]; GOV near-bed presence on 200-250 m
   beds higher in Q3 than Q1, reverse on 100-150 m [D]. No in-box winter day profiles exist.
5. **Night:** most rise to the upper 10-50 m (Oslofjord ~20 m; Clyde 85-95 % above 60 m; St Lawrence 60-80 % above
   40 m); a deep remnant stays (Kattegat summer: a large proportion below 80 m; often ~1/3 elsewhere) [F/S].
6. **Confidence low-moderate;** region rows at 150-250 m are ~10 tows, one Oslofjord station and one hypoxic basin; the
   function's checkpoints had no 150-250 m bed. Field vs function [D]: 157-166 m HERAS 131-154 m vs 81-116 m (Bunnefjorden
   70-130 m brackets it); ~200 m Oslofjord 80-130 m vs ~86-124 m; 250 m Degnbol 90-150 m vs ~95-145 m. Missing: in-box day 120/200 kHz profiles over
   150-250 m beds with the bottom 1-5 m resolved, by season; winter closing-net near-bed tows; species splits.

---

## A. Region (Skagerrak, Kattegat, Oslofjord, Norwegian Trench)

### A1. ICES acoustic-trawl database, RV Dana HERAS and DK IESSNS hauls, 2021-25 [D, cited, not re-derived]
- Source: `research_notes/"Krill över grunda bottnar"/distribution/evidence.md` section 1.1, and
  `research_notes/"Sejens profil over djupa bottnar"/tables/heras_deep_windows_stdout.txt`. ICES AT database, CC BY 4.0.
- Site: Skagerrak, late June to mid-July, daylight (sun 41-54 deg on the key hauls).
- Rows, as bed / net window / catch:
  - 166 m (in box), 131-154 m, 183 kg (366 kg/h). Window bottom 12 m above the bed.
  - 167 m (in box, 18 km away, 4.5 h earlier), 92-110 m, 0 kg.
  - 157 m, 138-148 m, 54 kg.
  - 213 m, 207-213 m (on the bed), 0 kg.
  - 102 m, 95-102 m, 733 kg (the < 150 m reference).
- Aggregate rows (they overlap the hauls above and are not counted twice): footrope 0-3 m above beds with a median of
  204 m, euphausiids in 0 of 4; footrope 30-60 m above beds with a median of 180 m, 1 of 3, at 1 kg/h.
- Night (net top <= 10 m) over 100-249 m beds: krill in 2 of 7 surface tows.
- Caveats:
  - These are net windows, not layer edges.
  - 16 mm codend, family-level identification only.
  - The trawls targeted fish.
  - One survey window, at the longest days.
  - A layer 1-3 m thick on the sediment would escape the pelagic tows.

### A2. NS-IBTS GOV, Swedish hauls, Euphausiidae presence by bed class [D]
- Data: ICES DATRAS compact CSVs in `research_notes/"Lek i rutan"/data/datras/`. Script
  `scripts/q1_gov_krill_by_bed.py`, output `tables/q1_gov_krill_by_bed_stdout.txt`.
- The denominator is the country-year-quarter strata that recorded euphausiids at least once: Swedish 77AR/26D4,
  2001-2025, 63 records. AphiaID 110671, SpecVal 4/6/7 (no lengths).
- Box 57.3-59.3 N, 8-12.2 E, all hauls in daylight (DayNight "D"); Swedish Q1 hauls fall in January-February and Q3
  hauls in August-September (`hh.csv` Month). Hauls with euphausiids / hauls, by bed class:

  | bed class | Q1 | Q3 |
  |---|---|---|
  | 0-50 m | 1/49 | 0/112 |
  | 50-100 m | 10/105 | 3/104 |
  | 100-150 m | 16/45 | 4/35 |
  | 150-200 m | 1/4 | 9/34 |
  | 200-250 m | 0/16 | 3/16 |
  | 250-400 m | 0/9 | 0/6 |

- Positive hauls over 150-250 m beds:
  - Q3 (12 hauls, 2002-2024): 156, 156, 159, 159, 169, 172, 175, 191, 192, 202, 212 and 226 m.
  - Q1: 154 m (2019).
  - 7 of the 13 lie at 58.0-58.4 N, 9.8-10.9 E, on the southern slope of the Norwegian Trench in the box.
- Caveats:
  - The GOV does not close. Krill can enter while the net is shot or hauled through a mid-water layer, so presence
    is an upper bound on near-bed occurrence.
  - The 20 mm codend liner retains krill poorly.
  - Recording of invertebrates is not standardised, and the records are presence only, with no numbers.
  - The headline is about 4-5 m above the bed.

### A3. Degnbol & Munch-Petersen 1985, ICES CM 1985/H:46 [F]
- Source: `Krill i Skagerrak/data/degnbol1985_icescm_H46/paper.txt`.
- Site: about 250 m bed NNE of Skagen, 7-14 August 1984. Gear: EK400 at 38/120 kHz, and an IKMT that does not close.
- Day:
  - "between 75 and 100 m a diffuse layer, below this, between 90 and 150 m a denser layer, still diffuse but with
    some larger spots ... and finally below 140 m a layer of single echoes" (p. 3).
  - "the maximum by day for 120 kHz just below 100 m (the middle layer) and the maximum by day for 38 kHz below 150 m
    (bottom layer)" (p. 4).
  - "Krill is forming a layer around 100 m depth at day".
- Derived [D]: centre about 105-120 m, so about 130-145 m above the bed. Bottom edge about 100 m above the bed.
- Night: layers rise after sunset; 2 h after sunset "two scattered layers - below the surface and between 50 and 150 m
  depth"; large krill scatter around the day depth.
- Caveats:
  - The 38 kHz bottom layer was single fish echoes (blue whiting).
  - TVG at 120 kHz "expired" below about 150 m, so krill nearer the bed would be under-read at 120 kHz.

### A4. Kaartvedt, Larsen, Hjelmseth & Onsrud 2002, MEPS 228:193-204, doi:10.3354/meps228193 [F]
- Source: `Krill i Skagerrak/data/kaartvedt2002_meps228/`. The p. 199 page image was read to correct the garbled text
  of Table 3.
- Sites:
  - Stn D: 200 m, Drøbak sound, 59°38'N 10°38'E; 30-31 May, 21-22 August and 12-13 November 1996.
  - Stn S: 120 m, inner Oslofjord.
- Gear: 3-ft IKMT at 4 kn with a CTD for the tow profile (not stated to be a closing net); EK500 120 kHz.
- Quotes:
  - "The distribution of scattering layers ascribed to Meganyctiphanes norvegica remained below 70 m at day, ascending
    at night (Figs. 3 & 4)."
  - "At Stn D, M. norvegica migrated all the way to the surface at night in August and November."
  - "M. norvegica could not be located acoustically in May when fish echoes were prominent (not shown), but then
    appeared to remain below 100 m at day, migrating up to about 20 m at night (Table 3)" (pp. 196-197).
- Table 3, Stn D, day tows (krill per nautical mile or litres). The tows at Aug 06:55, 07:55 and 20:30 and Nov 15:40
  are within about 1 h of sunrise or sunset, judged from the Fig. 4 arrows [D]:

  | month | tows |
  |---|---|
  | May | 80-70 m: none; 97-95 m: none; 127-124 m: 70; 158-152 m: 40; 108-100 m: 40 |
  | Aug | 87-71 m: none; 90-80 m: 100; 113-105 m: 160; 180-170 m (20:30): 20 |
  | Nov | 80 m (08:45): none; 100 m: 1.0 l; 107 m: 0.9 l; 150 m (12:05): none; 75 m at 15:40 (dusk): 250 |

- Derived heights above the 200 m bed, true-day tows [D]:
  - May: 42-100 m; no tow below 158 m.
  - August: about 87-120 m.
  - November: about 93-100 m, with none at 50 m above the bed.
- Fig. 4 [D, ±10 m, twilight hours only]:
  - November afternoon: layer at about 85-100 m.
  - In both seasons a strong scatter fills about 130 m down to the bed. The authors do not attribute it, and fish
    traces lie on the bed.
- Night: August 3-44 m, 30-370 per nm. A pre-dawn tow (05:35) took 90 per nm at 140-150 m, so part of the population
  was deep at night. November: 5-45 m, 0.6-6.0 l.
- Stn S (120 m): "stayed below ~70 m during the day" in all four seasons (Onsrud & Kaartvedt 1998, MEPS 171:209,
  doi:10.3354/meps171209 [F], `Krill i Skagerrak/data/onsrud1998_meps171/paper.txt` line 18). The ascent ended at
  about 20 m.
- Caveats:
  - Catches are not quantitative.
  - The IKMT is open on the way down and up.
  - The Oslofjord is turbid, so the layer is likely shallower than in the open Skagerrak.

### A5. Brun 2007, MSc thesis, University of Oslo, Bunnefjorden [F]
- Record: NVA 019c944a3f66-4b061835-e40c-46b5-9f85-c4f0c0138237. Quotes in `data/q1_brun2007_thesis/quotes.txt`.
- Site: a 150 m deep station in Bunnefjorden, November 2005 to January 2006. Hypoxic bottom water, O2 1.0-1.5 ml/l.
- Gear: bottom-mounted upward EK60 120 kHz, and trawls (non-quantitative).
- Quotes:
  - "the krill layer was restricted to waters deeper than ~75 m in November and ~85 m in December".
  - "During daytime catches, krill were caught from 60 m to 120 m".
  - December: "High numbers were caught from 112-120 m".
  - January: "the net tows suggest concentrations deeper than 100 m".
  - Night: "upward migration was mainly arrested at 20-30 m for krill".
  - "there is not an established community of bottom-associated planktivorous fish in this fjord branch".
- Derived [D]: the December catch maximum was 30-38 m above the bed.
- Caveats: hypoxia, an enclosed basin behind a 19 m sill, and a turbid fjord.

### A6. Moksnes 2021, MSc thesis, University of Oslo, Bunnefjorden [F]
- Record: NVA 019c944917c4-b05ed568-1a86-4bb5-a74c-6e28a1c513f1. Quotes in `data/q1_moksnes2021_thesis/quotes.txt`.
- Site: an upward WBAT echosounder at 154 m, 59°47.6'N 10°43.1'E, in March, August and October 2020.
- March: "the deepest was at noon where it spanned from ~70 m down to ~130 m". Individuals sink "from ~90-100 m down
  to the bottom", but "We did not observe a second aggregation of krill at the seabed". Night: up to about 14 m.
- August: "Noon distribution was at around 60 m and almost down to the seabed". The morning range was 45 m to the
  seabed. Two night layers, at about 10 m and 30-60 m.
- October: "The denser SSL avoided waters below ~90, but a faint layer had a range down to even ~120 m. Bottom depths
  were avoided altogether", tied to O2 of 0.67-0.9 mg/l.
- Derived [D]: the bottom edge was about 24 m above the bed in March and about 0-10 m in August.
- Caveat: the same hypoxic basin as A5. An upward sounder misses its own near-field (a few metres) but does see the
  layer down to the bed.

### A7. Tarling, Cottier & Everson 2014, Mar Biol Res 10:537-553, doi:10.1080/17451000.2013.831180 [F]
- Source: accepted manuscript, `Krill i Skagerrak/data/tarling2014_gullmar_bilayers/`.
- Site: Gullmarsfjord, maximum depth about 120 m, sill 43 m, 8-11 September 2003.
- Quotes:
  - "A deep scattering layer between 80 m and 100 m was apparent during the daytime"; "The peak layer of backscatter
    during the daytime was at 90 m"; at night two peaks, at 15 m and 45 m.
  - "euphausiid scattering layer did not go within 30 m of the seabed during its deepest phase during the day" (ms
    lines 281-291, 526-527).
- Background: Liljebladh & Thomasson 2001: 70-90 m by day, August-October 1997 [S].
- Bed of about 120 m: outside the 150-250 m class, shown for scale.

### A8. Kattegat: Lass et al. 2001, MEPS 214:177, doi:10.3354/meps214177 [F]; Tarling et al. 1999 [F, held]; Buchholz et al. 1995 [F, held]
- Site: Alkor Deep, 134 m; Læsø Deep, 143 m.
- Lass: "During the summer cruise in the Kattegat, almost the entire population occurred below 80 m during daytime. A
  fraction of the population was found in the upper layers at 23:00 and 02:00 h, but a large proportion still remained
  below 80 m". In winter, "krill remained below 80 m during the daytime (12:00 h)".
- Buchholz 1995: August 1989 "sharp maximum between 90-105 m" (`Krill över grunda bottnar/distribution/evidence.md`
  section 2).
- Caveat: beds below 150 m, so reference only. The net stopped 15 m above the bed.

### A9. NSEC 2025, IMR Toktrapport 2026-2 [F]
- Source: `Krill i Skagerrak/data/nsec2025/report.txt`.
- Site: Torungen-Hirtshals station 429, 643 m bed, May 2025. Multinet 180 µm, towed from 5 m above the bottom.
- Quotes:
  - "pelagic decapods, euphausiids, and large copepods ... were located below 150 m in the Norwegian Trench".
  - "These taxa may require a habitat where the water column exceeds 150-200 m".
- Caveat: one station, time of day not resolved.

### A10. Indirect: saithe stomachs by bed class [D, cited]
- Source: `Sejens profil over djupa bottnar/summary.md` section 2.
- Adult saithe search within about 15 m of the bed by day.
- Krill frequency of occurrence was 0.79 over 150-250 m beds against 0.72 over 100-150 m beds (Q3, ICES stomachs
  1980-87). Over beds > 250 m it was 0.73-0.79.
- The capture bed is not the feeding bed, and night feeding cannot be excluded. Used here only to note that saithe on
  deep beds do meet krill.

## B. Analogues: Norwegian shelf and fjords, Scotland, Barents Sea

### B1. Melle et al. 1993, ICES CM 1993/L:44 [F]
- Source: `Krill i Skagerrak/data/icescm1993_L44/paper.txt` lines 260-263, 381-385.
- Site: Norwegian shelf 62-70 N, June.
- Quotes:
  - "In areas of the shelf deeper than ca 150 m there was an approximately 50 m thick SSL associated with the bottom.
    The depth of this layer fluctuated with the bottom topography. This layer tended to disappear in shallower regions".
  - The intermediate layer off the shelf, "usually found between 90 and 200 m, was dominated by krill (Meganyctiphanes
    norvegica)".
- Caveat: the bottom SSL was not identified to taxon.

### B2. Kaartvedt, Melle, Knutsen & Skjoldal 1996, MEPS 136:51-58, doi:10.3354/meps136051 [F]
- Quotes in `data/q1_kaartvedt1996_meps136/quotes.txt`.
- Site: outer shelf off northern Norway, beds of about 295-300 m, April 1993.
- Quotes:
  - "Beneath clear waters on the outer Norwegian shelf (about 300 m depth), mesopelagic fish ... were located at
    approximately 150 to 200 m by day. Krill (mainly Thysanoessa inermis) was primarily found below the mesopelagic
    fish and above planktivorous demersal fish".
  - "A more diffuse layer of weaker targets was observed below, extending almost down to the bottom on the outer shelf".
  - Inside a turbid front, "Mesopelagic fish ascended by about 100 m accompanied by a rise of krill".
  - "Outside a patch of fish, the krill layer extended almost down to the sea floor. With increased numbers of fish
    rising from the benthic boundary zone, the distribution of krill became shallower, and a more compact SSL was
    formed about 100 m above bottom".
- Table 1: T. inermis 6.0 kg/nm at 140-170 m (13:00); T. inermis 6.5 and M. norvegica 2.9 kg/nm at 220-170 m over a
  295 m bed (20:25).
- Caveats: April at about 69 N, and the benthic boundary zone was not sampled.

### B3. Baliño & Aksnes 1993, MEPS 102:35-50, doi:10.3354/meps102035 [F; D for the bed]
- Quotes in `data/q1_balino1993_meps102/quotes.txt`.
- Site: Masfjorden, 22-26 January 1988. Stn D at the head of the fjord; the bed of about 195 m was read off the Fig. 9
  echogram [D].
- Quotes:
  - "At Stn D, a thick and dense layer was observed at the depth interval 50 to 150 m (Fig. 9). On 25 January, after
    sunrise, this layer migrated towards the bottom (130 to 150 m)".
  - At night, euphausiids were spread through the water column, "being more concentrated between 40 and 140 m".
- Stn B, 490 m bed: Layer I at 60-140 m by day; Layer II centred at about 150 m.
- Caveats:
  - The echogram covers only the morning (to about 09:30), so the noon position is not shown.
  - A dark fjord in mid-winter.

### B4. Zhou, Zhu & Tande 2005, MEPS 300:159-178, doi:10.3354/meps300159 [F]
- Source: `Krill över grunda bottnar/distribution/data/zhou2005_meps300/p.txt`.
- Site: Ullsfjorden (channel > 200 m) and Sørfjorden (two basins > 150 m), northern Norway, autumn and April.
- Quotes:
  - "The migration depth of Thysanoessa raschii was either to the bottom or to a depth of approximately 100 to 120 m in
    both the fall and spring cruises".
  - By day "they formed a relatively thin very dense layer of >100 to 1000 ind. m-3, and remained immediately above the
    bottom or at depths >100 to 120 m".
  - M. norvegica in Ullsfjorden "did not migrate to deep water, but remained within the top 50 m of water column during
    the day".
- Caveat: which bed goes with "bottom" and which with "100-120 m" is not given station by station.

### B5. Skaret et al. 2020, MEPS 638:13-23, doi:10.3354/meps13254 [F]
- Source: `Krillens fläckighet/data/skaret2020/paper.txt`.
- Site: Great Bank, northern Barents Sea, 77.4 N, "160-170 m deep", 8-9 October 2016. All identified krill were
  T. inermis.
- Quotes:
  - "During daytime, pelagic fish remained from a few to 50 m above the bottom while krill were found close to the
    bottom (Fig. 3b). During night-time, both krill and pelagic fish dispersed throughout the water column".
  - "Light penetrates down to the bottom during daytime and krill are blocked from descending further".
- Caveats: clear Arctic water, and a smaller and shallower-living species.

### B6. Clyde Sea: Lass et al. 2001 [F]; Tarling 2003, MEPS 260:173, doi:10.3354/meps260173 [F]
- Site: a 140-160 m trench (Tarling 2003; Lass: Arran Deep "150 m deep"). MOCNESS sampled to within 15 m of the bottom.
- Summer, July 1996:
  - "most of the population (52%) occurred between 80 and 100 m during daytime (14:00 h)".
  - Tarling 2003: "over 90% of the individuals of both sexes rested between 60 and 100 m during the day"; ADCP maximum
    echo at 60-90 m by day.
- Winter, February-March 1997:
  - "the daytime distribution was evenly spread between 60 and 150 m".
  - "85 to 95% of individuals occurred above 60 m during the night".
- Derived [D, bed taken as 150 m]: in summer the 80-100 m mode is 50-70 m above the bed (60-100 m: 50-90 m); in winter
  the layer reaches the net limit about 15 m above the bed.

## C. Far analogues: NW Atlantic and NE Pacific

### C1. Cochrane, Sameoto & Belliveau 1994, MEPS 107:55-66, doi:10.3354/meps107055 [F]
- Source: `Krillens fläckighet/data/cochrane1994_meps107/paper.txt`.
- Site: LaHave Basin NW slope, ADCP at 192 m, 23 October-12 December 1990.
- Quotes:
  - "M. norvegica in daytime were broadly distributed at about 150 m depth. At night they were uniformly distributed in
    the upper 50 m".
  - "virtually the entire layer remaining above the 7 m gap between the lower most recorded bin and the physical bottom".
  - "Sampling with the BIONESS 1 m above bottom ... plus video observations ... in June 1990, showed few euphausiids
    close to or on the bottom in the day and even fewer at night".
  - "the maximum of the daytime euphausiid layer (220 m) lies below the RDI deployment depth (192 m) but maximizes at a
    comparable height (20 to 30 m) above bottom. This indicates the daytime layer depth is more strongly influenced by
    the proximity of the bottom rather than by ambient light level" (p. 64).
- Caveat: single frequency; the krill attribution rests on the migration pattern.

### C2. Herman, Cochrane & Sameoto 1993, MEPS 94:165-173, doi:10.3354/meps094165 [F]
- Quotes in `data/q1_meps94_165/quotes.txt`.
- Site: Emerald and La Have Basins, beds of about 270 m, 25 September-5 October 1990.
- Quotes:
  - "M. norvegica were located below 160 m depth during daytime and migrated to the upper layers from 0 to 50 m at
    night. Thysanoessa inermis adults were found within the depth range of 70 to 100 m during daytime and were aggregated
    in narrow layers of 10 to 20 m thickness".
  - "Approximately 20% of the water column population of Thysanoessa spp. were found to co-habit the same depth region
    as M. norvegica, i.e. below 160 m".

### C3. Sameoto & Cochrane 1996, DFO Res Doc 96/119 [F]
- Source: `Krillens fläckighet/data/dfo_209440/doc.txt`.
- Quotes:
  - "Thysanosessa spp. are found at about 150 m during daylight hours, whereas M. norvegica prefers depths greater than
    200 m during the day".
  - April 1995: "High daytime levels of 153 kHz backscattering were detected in the region of Emerald Basin extending
    from a depth of 200m to near bottom".
  - "During the day M. norvegica forms layers that are often times found 10 to 20 m above layers of pelagic fish such a
    silver hake".

### C4. Sourisseau, Simard & Saucier 2008, CJFAS 65:574-587, doi:10.1139/F07-179 [F]
- Source: `Krillens fläckighet/data/sourisseau2008_cjfas65/doc.txt`.
- Site: St Lawrence Estuary; "The ship followed the 160 m isobath". September 2003.
- Quotes:
  - "The highest krill concentration estimated from the Sv,120 occurred during daytime, between 100 and 150 m".
  - "In daytime, krill was concentrated in a dense SL below ~90 m"; after midnight krill "reform a deep SL between 90
    and 130 m".
  - At night "60% to 80% of the euphausiid population was above 40 m".
- Caveats: strong tides and a cold intermediate layer, a different water column from the Skagerrak.

### C5. Simard, de Ladurantaye & Therriault 1986, MEPS 32:203-215, doi:10.3354/meps032203 [F]
- Site: St Lawrence Estuary, July 1982.
- Quotes:
  - "Euphausiids were found at depths varying between 50 and 175 m, often with large concentrations just above the
    bottom. The vertical distribution was characterized by 2 principal modes, at 75 and 150 m" (T. raschii and
    M. norvegica).
  - "On the north shelf, significant biomasses were found close to the bottom in areas shallower than 200 m".
- Caveat: upwelling and tidal trapping at the channel edge.

### C6. Greene, Wiebe, Burczynski & Youngbluth 1988, Science 241:359, doi:10.1126/science.241.4863.359 [A]
- Abstract: "High-density demersal layers of krill have been detected in the submarine canyons off Georges Bank ...
  two to three orders of magnitude greater than the highest densities observed in water-column scattering layers".
- No bed depth or season in the abstract.

### C7. Simard & Mackas 1989, CJFAS 46:1238, doi:10.1139/f89-160 [S/analogue]
- Via `Krill över grunda bottnar/distribution/evidence.md` line 288: day layer "deeper at the shelf break (~150-200 m)
  than on the shelf (~100-150 m) and occasionally extended to the bottom".
- Species: E. pacifica and T. spinifera.

### C8. Aarflot et al. 2019, Limnol Oceanogr, doi:10.1002/lno.11079 [F, analogue for scaling only]
- Site: Barents Sea, MOCNESS, beds of 100-500 m. Krill were explicitly excluded.
- Large zooplankton (> 2 mm) "resided primarily close to the bottom, with a strong effect of bottom depth on WMDL
  (bL = 0.75)". Distributions were deeper in clearer water.

## Not obtained
- Klevjer & Kaartvedt 2011, L&O 56:765, doi:10.4319/lo.2011.56.3.0765: bottom-mounted echosounder at 150 m in the
  Oslofjord. Abstract only [A]; no depth data in it; the Wiley PDF is behind a challenge page.
- Klevjer & Kaartvedt 2006, ICES JMS 63:1726 [A]: an OUP challenge page.
- Plourde et al. 2014, ICES JMS 71:272 [A]: daytime weighted mean depth (WMD) varies with surface salinity; no bed
  reference in the abstract.
- Youngbluth et al. 1989 (submersible, Georges Bank canyons): not found via OpenAlex.
- Cochrane et al. 1991 and Sameoto et al. 1993 (Scotian basins): not fetched.
