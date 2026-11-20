# Q2. Do gadoids track krill patches? Evidence trail (2026-11-20)

Literature and stomach-record research only. No model run, no code or library change, no commit. No value was
chosen from model output.

**Tags**
- [F] full text read
- [A] abstract only
- [S] secondary: a citing paper, a fetch-tool summary of a page, or a sibling note
- [D] derived here
- [J] judgement

**Files**
- Downloads: `data/<source>/`, each with a `SOURCE.txt`. They are untrusted and were read only with pdftotext or
  `python3 -I`.
- Scripts: `scripts/q2_*.py`.
- Stdout: `tables/q2_*_stdout.txt`.
- Rows: `tables/q2_raw.csv`.

**Held sources re-read, not re-fetched**
- Rindorf 2003 [F]: research_notes/"Krillintag och jaktsätt"/data/rindorf2003_meps249.
- Rindorf & Gislason 2005 [F]: research_notes/"Intag under födosök"/data/rindorf_gislason2005.
- Kaartvedt et al. 2005 [F]: research_notes/"Krillens flykt och energibudget"/data/kaartvedt2005_meps.
- Onsrud et al. 2004 [F]: research_notes/"Vitlingens fiskbyten"/data/step2_lit/vertical/onsrud2004.txt.
- Buckley et al. 2016 [F]: research_notes/"Krillintag och jaktsätt"/data/q1_buckley2016_dsr2.
- The ICES stomach records: research_notes/"Krill över grunda bottnar"/stomachs/tables/records_linked.csv.gz, the
  table built there from the ICES stomach download.

**Proxy rule**
- Whiting, saithe and hake are targets (proxy = no).
- Cod, haddock, Norway pout, walleye pollock and blue whiting are gadoid proxies (proxy = yes).
- Capelin, as a prey, is a proxy for krill.

---

## Item 1. Aggregative response and spatial correlation, fish vs krill

### 1.1 Mackas, Kieser, Saunders, Yelland, Brown & Moore 1997, CJFAS 54:2080-2096, doi:10.1139/f97-113 [A]

Pacific hake and euphausiids, Vancouver Island shelf break, echo integration. The text was not obtained: the
publisher returned 403 and Unpaywall lists it as closed. The abstract came from OpenAlex
(`data/mackas1997_meta/openalex.json`).

- "Both euphausiids and hake share similar horizontal distributions, although separated vertically by tens of meters
  during daylight hours."
- "Bathymetric and water property patterns provide good coarse-scale prediction of aggregation location. However,
  details of patch location are better explained by flow-field indices such as cross-shore position of the shelf
  break current, intensity of cross-isobath flow and upwelling at the depth of the euphausiid scattering layer".
- Reading [J]:
  - This is co-occurrence in a shared shelf-break habitat, explained by flow and bathymetry. It is not shown to be
    hake tracking krill patches.
  - The abstract gives no correlation coefficient and no scale.
  - Day vertical separation is "tens of meters".

### 1.2 Evans, Gauthier, Robinson, English, Stanley, Wright & Nichol 2026, Ecol Appl 36:e70141, doi:10.1002/eap.70141 [S]

Read on PMC (PMC12796558) through the fetch tool. The quotes are as returned by its summariser; the article was not
saved here, and the fetch covered only the first 100k of 159k characters.

- Model: hake and euphausiid acoustic biomass 2007-2018, "(3 × 3 km resolution)".
- Hotspot overlap:
  - "Co-occurrence between hotspots of hake and euphausiids was lowest in 2007 and 2015 (2% and 9% of total hotspot
    area, respectively)".
  - It was highest "in 2013 and 2017 (20% and 39% of total hotspot area, respectively)".
- "euphausiids and hake showed spatial mismatch in their biomass hotspot distributions", "with low direct overlap
  (apart from 2017)".
- Overlap sits "just deeper than the 200-m contour", "with hake distributed adjacent to euphausiids over slightly
  deeper water".
- No hake-krill correlation coefficient is reported.
- Reading [J]: at 3 km support, hake hotspots overlap 2-39 % of the combined hotspot area. That is adjacency, not
  coincidence.

### 1.3 Phillips, Malick, Gauthier, Haltuch, Hunsicker, Parker-Stetter & Thomas 2023, Fish Oceanogr 32:267-279, doi:10.1111/fog.12628 [S]

The NOAA repository PDF returned an HTML page (`data/phillips2023_noaa49629/blocked_page.html`). These are the fetch
tool's summary of the landing page.

- Age 2+ hake and krill, US-Canada acoustic-trawl survey 2007-2019.
- Predicted co-occurrence averaged 41.0 % at 100-m temperature 1 C below average and 17.0 % at 1 C above.
- Hake vertical distribution "was not correlated with krill depth" [S].
- The grid size was not obtained.

### 1.4 Rose & Leggett 1990, Ecology 71:33-43, doi:10.2307/1940245 [A]

Proxy: cod and capelin.

- Abstract (OpenAlex, `data/openalex_abstracts/10.2307_1940245.json`):
  - "When capelin did not occupy thermal refuges, cod and capelin densities were coherent across a wide range of
    scales (2-30 km)."
  - "At large scales (>4-10 km) densities were in phase (positively correlated), but at scales less than aggregation
    dimensions (3.5 km), densities were out of phase (negatively correlated)."
  - "At the smallest scales measured (2-3 km), densities exhibited maximum negative correlation in both years."
  - "When actually feeding, cod achieved coherent, in-phase distributions with capelin at scales as small as 3.5 m."
- The sign at about 1 km therefore depends on predator activity. This matches the sibling note research_notes/
  "Mättnad hos rovfisk"/lit_foraging_limits.md.

### 1.5 Rindorf & Gislason 2005, JEMBE, doi:10.1016/j.jembe.2005.04.013 [F]

Proxy for krill. This is whiting against trawl-caught FISH prey; krill was not a prey in this test.

- "The natural log to the catch of prey was negatively correlated to that of the predator for all combinations of
  prey and predator size group but two (Fig. 9). However, only the correlation between herring and whiting of length
  25 to 29.9 cm (correlation -0.92, P=0.03) was significant." (postprint lines 414-418, Results "Aggregative response")
- Support: 5 North Sea locations in early September, hauls every 4 h for 48-72 h. Each correlation is across
  locations (n = 5), so it is the 10-100 km scale, not within a location.
- Correction to the sibling note research_notes/"Intag under födosök"/evidence.md E6: its "no aggregative response"
  is a between-location statement for fish prey. It is not krill and not km scale [J].

### 1.6 Kaartvedt, Røstad, Fiksen, Melle, Torgersen, Breien & Klevjer 2005, MEPS 299:1-5, doi:10.3354/meps299001 [F]

- Where: "Norwegian Sea (~70° N, 4° E)", 24-25 May 2004. Not the Oslofjord (lead checked).
- Fish under swarms (p. 3, layout extraction checked):
  - "Underneath most swarms was a 'stack' of fish that could extend almost 100 m below the krill (Figs. 2 & 3)."
  - "Some individuals were also located inside, or close to the swarms".
  - "these fish commonly (but not exclusively) were large piscivores rather than smaller planktivores".
- Fig. 2 caption: "The ship moved at ~4 knots, and 5 swarms are displayed from this 30 min recording period."
  - [D] That is about 3.7 km of track with 5 swarms, one per ~0.7 km in a section the authors picked as rich.
- Swarm geometry and density:
  - Oblate spheroids "50 m vertical and 120 m horizontal".
  - "~200 ind. m-3", "~11 t per swarm".
- Diet: the 60-110 cm saithe "had fish and the cephalopod Gonatus sp. in their stomachs. Apparently, the saithe
  preyed on planktivores attracted by the krill, rather than on the krill itself."
- Reading [J]:
  - This is the only direct observation of gadoids associated with individual krill swarms at the 100-m scale.
  - "Most swarms" had fish, but no count is given.
  - The fish were large saithe that ate fish, and blue whiting (the likely krill eaters) deeper. It is association
    for patrolling, not krill eating by the saithe.
  - Oceanic site, 100-200 m swarm depth.

### 1.7 Skaret, Johansen, Johnsen, Fall, Fiksen, Englund, Fauchald, Gjøsæter, Macaulay & Johannesen 2020, MEPS 638:13-23, doi:10.3354/meps13254 [F]

Proxy: cod. Northern Barents Sea bank, 160-170 m.

- "broadband acoustics mounted on a submersible probe allowed us to track individual cod, revealing that they
  remained mostly within 10 m of the bottom throughout the diel cycle."
- "During daylight, krill pushed towards the bottom, sharing habitat with cod".
- "krill appear to hide from capelin among the cod near bottom".
- Diet: of 239 cod stomachs, "14% were empty. Of the stomachs containing food, 79% contained pelagic fish and/or
  unidentified fish".
- Discussion: "The cod is known to be a potential predator on krill, but prefer pelagic fish, and paradoxically krill may
  therefore gain safety by hiding among the cod."
- Reading [J]: vertical and horizontal co-occurrence with krill does not imply krill eating. Diet choice can
  decouple the two.

### 1.8 Ressler, De Robertis & Kotwicki 2014, MEPS 503:111-122, doi:10.3354/meps10736 [F]

Proxy: walleye pollock and Thysanoessa, eastern Bering Sea.

- Support: "a mesoscale station spacing of 37 km (20 nmi)" (37 x 37 km blocks).
- Pollock biomass was a weak covariate of euphausiid biomass: dropping it cost 0.9 % of deviance; temperature cost
  4.5-6.9 %.
- "Despite negative interannual and spatial relationships between pollock and euphausiid biomass, the effect of
  pollock on euphausiid density was weak".
- The paper notes that the sign can flip with scale (citing Rose & Leggett).
- Ressler et al. 2012 (DSR2 65-70:184, doi:10.1016/j.dsr2.2012.02.015; not obtained [S via search]): standing stocks
  "inversely correlated over time and across the continental shelf".
- Reading [J]: no positive aggregation at 37 km. It says nothing about the km scale.

### 1.9 Swartzman, Brodeur, Napp, Walsh, Hewitt & Demer 1999, ICES JMS 56:545-560, doi:10.1006/jmsc.1999.0480 [A]

Proxy: age-0 pollock, a much smaller fish. The OUP PDF was blocked; the abstract came from OpenAlex.

- "When the plankton biomass was low, fish tended to remain close to existing plankton patches, while at high
  plankton biomass there was no consistent small-scale proximity relationship."
- "the interval-based fish densities tended to increase with increasing plankton density up to some plankton density
  threshold, above which there was no clear association".
- Reading [J]: the aggregative response is conditional on scarcity. No distances are in the abstract.

### 1.10 Buckley et al. 2016, DSR2, doi:10.1016/j.dsr2.2015.10.009, manuscript [F]

Proxy: pollock. Temporal, not spatial.

- "Correlation between the increase in estimated EBS euphausiid biomass and the PF values of euphausiids was
  significant (r = 0.89, P = 0.02) for the largest length category of pollock in this short time series".
- Support: shelf-wide annual means, 37 km grid of stations, n of about 6 years. It is not a patch-scale measure.

### 1.11 ICES stomach records, North Sea and Skagerrak, 1980-91 pooled records [D]

Script `q2_haul_krill_stats2.py`.

**Within-rectangle co-occurrence**
- Stomach positions are mostly ICES rectangle centres (`pos` = rect for 137 of 141 whiting krill records), so no km
  distance can be computed (`q2_records_checks.py`).
- Hauls in the same rectangle (30' x 1°, about 55 x 55 km) within 3 days, in cruises with any krill:

  | species | P(second haul has krill \| first has) | haul marginal | ordered pairs |
  |---|---|---|---|
  | whiting, all sizes | 0.34 | 0.39 | 29 |
  | whiting >= 20 cm | 0.20 | 0.31 | 20 |
  | saithe, all sizes | 0.85 | 0.59 | 47 |
  | saithe 35-70 cm | 0.79 | 0.57 | 38 |

- The pairs are ordered, so the number of independent pairs is half of what is shown.
- Reading [J]:
  - For whiting, the conditional and the marginal cannot be told apart at this n.
  - For saithe the conditional is higher, by about 0.2-0.3.
- An earlier distance table (`q2_haul_pairs.py`) is superseded: with rectangle positions its "0-10 km" bin is the
  same-rectangle bin.

---

## Item 2. How fast: residence and tracking time scales

No tag or acoustic study follows a gadoid relative to a krill patch over hours or days. What exists:

1. **Rindorf 2003 [F], a fixed-station krill time series.** Methods, p. 267: krill "disappeared completely after 24,
   31 and 42 h at Locations 1, 3 and 4, respectively".
   - The Table 2 footnote names only Locations 1 and 3 (24 and 31 h).
   - Krill was modelled at Locations 1-4 only (Table 1), so this is "3 of the 4 modelled sites", not "3 of 5". That
     Location 5 (137 m) was left out for too little krill is an inference [J]: the paper's inclusion rule was more
     than 5 stomachs at all times of day at 2 or more locations.
   - p. 268: "negative trends for krill" in occurrence over the sampling period at 3 locations.
   - Support: one bottom haul every 4 h for 48-72 h. That the repeated hauls covered the same few km is an
     inference [J]; the area per haul is not given.
   - Reading [J]: at a fixed spot, krill in whiting stomachs persisted 1-2 days, then vanished. The krill or the fish
     moved, and the data cannot say which. It sets an upper bound of about 1-2 days on patch residence at haul scale.
2. **Onsrud, Kaartvedt & Breien 2005, CJFAS 62:1822, doi:10.1139/f05-090 [F]** (`data/onsrud2005`), Oslofjord, Norway
   pout and whiting.
   - "Swimming speeds of deep-living fish ... were highest during the day (speeds centred around 14-16 cm s-1) and
     decreased somewhat by night (modes around 10-11 cm s-1)."
   - "The majority of tracks were short (10-20 pings)". At ~2 pings/s that is seconds, so no residence time.
   - [D] At 10-16 cm/s a fish crosses 1 km in 1.7-2.8 h, an upper bound of 9-14 km/d of straight travel.
   - The paper reports bimodal swimming directions, read as tidal influence, and "straight (portions of) swimming
     paths". No tortuosity or area-restricted search relative to krill.
3. **Christiansen et al. 2024, ICES JMS 81:1764, doi:10.1093/icesjms/fsae123** [S: sibling note research_notes/
   "Krillens flykt och energibudget" E9, read in full there], herring schools over krill in the Oslofjord.
   - Voids "typically lasted 5-15 min. for the larger fish schools".
   - In ">75% of the daytime encounters with large fish schools" krill density fell by > 55 %. Small schools (< 5 m)
     caused no change, and the night change was about 0.
   - This is the krill's response time to a passing school, not the fish's tracking time.
4. **Rose & Leggett 1990 [A]:** feeding cod were in phase with capelin "at scales as small as 3.5 m". This is a
   snapshot; no time scale.
5. **Pednekar et al. 2024, Commun Biol 7:1386, doi:10.1038/s42003-024-06952-6** [A, plus S from the sibling note
   research_notes/"Intag under födosök"].
   - Cod and capelin shoals over thousands of km2 form rapidly once a critical density is reached.
   - "~2.5 million cod ate ~10.5 million capelin within a few hours" [S].
   - Proxy prey (capelin), proxy predator (cod).
6. **Kaartvedt et al. 2005 [F]:** the swarms held at 100-200 m "both during the day and the light summer night", with
   fish stacks under them over the two days of observation. No individual was tracked over time.

---

## Item 3. Stomach statistics that separate patch occupancy from diet choice

### 3.1 Rindorf 2003, MEPS 249:265-276, doi:10.3354/meps249265 [F]

Five North Sea sites, whiting 20-35 cm.

- Table 2, krill column:
  - corrected number of stomachs 3497, of which 709 contained krill [D: FO 0.20];
  - reduced-model R2 0.861 with 31 parameters (location, time of day, length, location x time, location x length,
    trend);
  - Pearson scale (dispersion) 1.57.
- Truncation: for krill "only the first part of the sampling period was used ... at these locations" (Locations 1,
  3 and 4, p. 267). The krill-free periods are excluded from these fits.
- So the 1.57, the R2 and the 709/3497 describe the krill-present part of each series only. They cannot be read as
  "little residual patchiness" [J].
- Table 4: the common diel term explains 0.093 and the location-specific diel term 0.219 of the deviance for krill.
- Quotes:
  - p. 273: "Krill appeared to be a patchily distributed prey as some periods were categorised by complete lack of
    krill in the stomach contents."
  - Krill was eaten "primarily around dawn and in the early afternoon until some time after dusk".
- The catch-diet test (Results, "Correlation between the proportion of stomachs ...") addressed vertical subpopulations, not krill patches. 3 of 56 correlations between
  prey occurrence and ln(catch) were significant, "less than may occur from a type II error".

### 3.2 Bogstad, Pennington & Vølstad 1991, ICES C.M. 1991/O:15 [F]

OCR scan, `data/vliz_274641`. Proxy: cod and capelin, Barents Sea winter survey. This is the CM forerunner of Bogstad,
Pennington & Vølstad 1995, Fish Res 23:37, doi:10.1016/0165-7836(94)00341-s, which was not obtained.

- Table 1 gives the intra-haul correlation rho of capelin weight in cod stomachs, by 5-cm length group (40-69 cm):

  | year | capelin abundance | rho | example |
  |---|---|---|---|
  | 1985 | medium | 0.40, 0.62, 0.50, 0.58, 0.70, 0.72 | 45-49 cm: 23 stations, 196 stomachs, between-tow variance 1227, within 768 |
  | 1987 | very low | 0.16, 0.08 | |
  | 1989 | low | 0.26, 0.62, 0.59, 0.75, 0.43, 0.64 | |

- Section 3.1 (printed page 8): "The low precision is due to large intra-haul correlation for stomach contents ... little is gained in
  precision by collecting many stomachs from each trawl haul."
- Section 2.2: "Due to the patchy distribution of prey species, intra-haul correlation likely exists."
- Reading [J]: for a schooling prey, fish caught in the same haul share their capelin content strongly (rho about
  0.4-0.75). Where the prey is present, haul identity is the main factor.
- The same paper (Introduction) notes "the effective sample size could well be smaller than number of sample stations".

### 3.3 ICES stomach records, North Sea + Skagerrak + Kattegat [D]

Scripts `q2_haul_krill_stats2.py` and `q2_haul_krill_stats.py`; stdout in `tables/`.

**Data and caveats**
- 1980-91 records pool the stomachs of one length class per haul, and krill presence is coded per record (0/1).
  Within-haul FO per fish is therefore not available for those years.
- Statistics are restricted to cruises (ship x year x month) with at least one krill record, to remove region and
  season absence.
- Krill per stomach is computed per haul. A haul is dropped only if a krill-positive record in it lacks a count:
  285 of 389 saithe krill records lack counts, so the saithe concentration rests on counted hauls only.
- The ICC is the one-way ANOVA ICC over hauls of cruise-centred record presence (hauls with >= 2 length-class records).
  It measures how strongly length classes in one haul agree. That is a haul-level occupancy signal, with two caveats:
  - it also carries any between-haul gradient left inside a cruise;
  - record presence rises with the number of stomachs per record, so large catches alone produce some within-haul
    agreement.
- Per-fish FO inside krill-positive hauls is bounded from pooled records:
  - lower bound = positive records / stomachs, since each positive record has at least one fish with krill;
  - upper bound = stomachs in positive records / stomachs.

**Results**

| species, sizes | hauls | share of hauls with krill | P(2nd class has krill \| 1st has), vs cruise-marginal expectation | ICC | per-fish FO in positive hauls, bounds | top 10 % of hauls hold, of summed per-stomach means | krill per stomach in positive hauls: median; p90 |
|---|---|---|---|---|---|---|---|
| whiting >= 20 cm | 152 | 0.31 | 0.67 vs 0.33 | 0.51 | 0.10-0.72 (47 hauls, 770 stomachs) | 0.90 | 1.7; 28 |
| whiting, all sizes | 181 | 0.39 | 0.62 vs 0.35 | 0.39 | 0.09-0.65 | 0.85 | 1.4; 14 |
| saithe 35-70 cm | 198 | 0.57 | 0.87 vs 0.73 | 0.43 | 0.07-0.88 (113 hauls) | 0.86 (124 counted hauls) | 10; 124 (39 counted positive hauls) |
| Norway pout (proxy) | 54 | 0.76 | 0.85 vs 0.76 | 0.14 | 0.07-0.87 | 0.65 | 0.61; 4.0 |
| cod >= 20 cm (proxy) | 80 | 0.26 | 0.63 vs 0.32 | 0.46 | 0.09-0.64 (21 hauls) | 0.91 (8 hauls) | 0.33; 3.7 |

The concentration column is computed after the fix to the haul-drop flag; an earlier run dropped low-krill hauls and
overstated it.

**Per-fish anchors for FO inside krill places**
- Rindorf 2003, all hauls at Locations 1-4 in their krill periods: 0.20 [F].
- ICES single stomachs, 4 krill-positive whiting hauls: 0.12-0.33 [D].
- Onsrud 2004, Oslofjord krill layer: whiting 0.35-0.80 [F].
- Each sits inside the pooled bounds, which are too wide to choose between them.

- [J] Whiting and saithe show the occupancy pattern:
  - one third (whiting) to three fifths (saithe) of hauls have krill;
  - where one length class has krill, another class in the same haul has it twice as often (whiting) as the cruise
    rate predicts;
  - 10 % of hauls hold 0.85-0.90 of the summed per-stomach krill means.
- [J] Norway pout is different: krill is in three quarters of hauls and in most classes, with weak haul clustering
  (ICC 0.14). That is a near-continuous krill feeder, consistent with Onsrud 2004.

**Single-stomach records (Number = 1), mostly 2020-23 GOV, Q1 + Q3** (`q2_haul_krill_stats.py`, `q2_records_checks.py`)
- Whiting, 923 fed:
  - only 13 with krill;
  - 2 of 518 fed whiting in the Skagerrak 2022 (38 hauls, median 24 cm);
  - 0 of 793 in the central and southern North Sea 2022-23, which is not krill habitat.
- In the 4 krill-positive whiting hauls with >= 3 fed fish (23 fish), FO inside the haul was 0.12-0.33.
- Saithe singles: 137 fed in 107 hauls. Only 5 hauls have >= 3 fed fish. ICC 0.43 rests on those 5 hauls; too thin.
- Reading [J]: the recent whiting data are near krill-free, so they cannot test within-haul FO.

### 3.4 Onsrud et al. 2004, ICES JMS 61:1278, doi:10.1016/j.icesjms.2004.09.005 [F]

The rival reading. Oslofjord, 120 m bed.

**Design**
- "A total of 66 pelagic and 24 bottom hauls were taken."
- "Effective trawling time ... was generally between 20 and 30 min", at 2 knots: [D] 1.2-1.9 km per tow.

**Results (Table 3, pooled per survey x day/night; no per-haul FO)**
- FO of krill in whiting: Nov 42.3 / 35.2 %, Mar 75.5 / 79.7 %, Jun 42.9 / 34.6 % (day / night).
- Krill per stomach: 1.4-5.9 [S: sibling note E10, read in full there].
- "For whiting, krill constituted up to 99% of the stomach content by weight, on average ~82% during both day and
  night".
- "No systematic trends were apparent comparing proportion of stomachs containing krill or state of digestion during
  the day and night".
- Fish "were distributed together with the krill" by day.

**Reading [J]**
- In a fjord where the krill layer is continuous and the fish sit in it, whiting FO per fish is 35-80 %.
- The North Sea per-fish FO inside krill-positive hauls is only bounded (0.10-0.72), so the two cannot be compared
  point for point.
- What can be said:
  - inside a krill layer, occupancy is near total but FO stays below 1, with half-full stomachs (fullness 2.2-3.3 of
    5, sibling note);
  - in the North Sea, only 0.31-0.39 of hauls hold any krill in whiting.
- So both limits are present:
  - absence of krill at the haul scale in the open North Sea (occupancy);
  - a sub-maximal rate where krill is continuous (inside the layer).

### 3.5 Tanasichuk 1999, Fish Oceanogr 8:150-156, doi:10.1046/j.1365-2419.1999.00100.x [A]

Hake, SW Vancouver Island.

- "The biomass of euphausiids available to hake after 1993 averaged 27% of that for 1991 ... and was 12% of the 1991
  value in 1996."
- Hake "persisted in selecting krill longer than 17 mm and consistently preferred T. spinifera".
- No ration or patch statistic is in the abstract; the sibling note [S] holds that the share in the diet stayed ~89 %.
- Reading [J]: the krill share in the diet is insensitive to a 4-8x fall in krill availability. This is consistent with
  hake finding krill patches whatever the mean density, but it is only a diet share.

---

## Synthesis [J]

**1. Co-occurrence at 3-37 km is weak, adjacent or negative; no study shows a gadoid aggregative response to krill.**
- Hake and krill share shelf-break habitat (Mackas) but overlap only 2-39 % of the combined hotspot area at 3 km
  support (Evans). Hake sit beside the krill, deeper.
- Pollock: no positive relation at 37 km (Ressler).
- Only the Norwegian Sea saithe stacks under swarms are a patch-scale association, and those saithe ate fish.
- Cod and capelin show the scale sign switch: positive above 4-10 km, negative at 2-3.5 km, in phase only while
  feeding.
- No coefficient exists for whiting or saithe against krill at any scale.

**2. Speed is not measured.**
- Bounds:
  - krill vanished from whiting stomachs at fixed stations after 24-42 h;
  - fish swim 10-16 cm/s, so at most 9-14 km/d straight;
  - krill voids around fish schools last 5-15 min.
- Nothing measures fish residence in a krill patch.

**3. The stomach statistics support patch occupancy for whiting, saithe and cod at haul scale, and near-continuous
access for Norway pout.**
- Krill is in 0.26-0.57 of hauls (cruises with any krill).
- It is strongly shared within hauls: second-class odds about 2x the cruise rate for whiting and cod, ICC 0.43-0.51.
- It is concentrated: the top 10 % of hauls hold 0.85-0.91 of the summed per-stomach means.
- Per-fish FO inside positive hauls is only bounded, at 0.1-0.7.
- This agrees with the capelin-in-cod intra-haul rho of 0.4-0.75 (Barents Sea) and Rindorf's "complete lack" periods.
- What it cannot separate: haul-level presence is shared by fish in one haul whether they met a patch, the patch met
  them, or local light and depth made krill available. "Meeting" here includes vertical access, which Skaret and
  Onsrud show matters as much as horizontal finding.

**Evidence against "finding is the limit".**
- Onsrud 2004: in a continuous fjord krill layer, FO is still only 35-80 %, with half-full stomachs.
- Kaartvedt 2005: saithe at swarms did not eat krill.
- Skaret 2020: cod among krill ate fish.
- Each says diet choice or a limit inside the layer also acts.

---

## Not obtained

1. Mackas et al. 1997 full text (CJFAS, 403): the scale of the hake-euphausiid overlap and the vertical separation in
   numbers.
2. Pennington & Vølstad 1994, Biometrics 50:725, doi:10.2307/2532786 (JSTOR), and Bogstad et al. 1995, Fish Res 23:37
   (closed). The 1991 CM gives the cod-capelin rho instead.
3. Ressler et al. 2012 DSR2 (closed). Swartzman et al. 1999 full text (OUP block page): proximity distances.
4. Evans et al. 2026 and Phillips et al. 2023: not saved locally, read only through a summariser. The grid size for
   Phillips is missing.
5. Benoit-Bird Patch Dynamics (Bering Sea 2008-09): the dataset is restricted and no paper on pollock vs euphausiid
   patch scale was found in one search.
6. Tanasichuk 1999/2002 ration tables (closed).
7. Any tag or telemetry study of whiting, saithe or hake relative to krill patches. None found.
8. Any per-haul FO of krill in whiting or saithe with krill density measured in the same haul. None found.
9. Rindorf 2002 (J Fish Biol), which describes the haul design: still 403 at DTU Orbit (sibling note). Haul duration
   and tow length for the five sites are unknown.
