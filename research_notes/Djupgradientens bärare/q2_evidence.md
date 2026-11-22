# Q2. Do saithe and hake enter the krill layer, and when? (2026-11-22)

Literature subagent. Nothing outside this folder was changed; nothing committed. No parameter values set or
recommended. Tags: [F] full text read, [A] abstract only, [S] secondary / analogue species or site / via another
note's download, [D] derived here, [J] judgement.

## Verdict (20 lines)

1. No source measures saithe's or hake's time in the krill layer: no tag gives height above bed, and no diel
   stomach series of saithe with krill was found (Norway, Faroe, Iceland, North Sea, NW Atlantic "pollock").
2. ICES stomach DB [D]: saithe krill FO by sun phase is day 0.52 (n 463) / night 0.47 (38) / twilight 0.39 (70)
   records. This CANNOT resolve timing: krill stays 24-42 h in the stomach, no night krill row has a digestion
   stage, and phase is confounded with quarter. It is not evidence of round-the-clock feeding.
3. Gadoids take krill near the bed where krill come down [F/S, cod analogue]: Barents cod on a 175-240 m bank,
   84 % euphausiids by weight, krill FO flat over 24 h (midnight sun; Dolgov 1992); cod rose to 10-15 m above a
   125-150 m bed when krill rose; krill "close to the bottom" by day among cod within 10 m of a 160-170 m bed
   (Skaret 2020). Krill day layers within 50 m of the bed (Widder 1992 [A]), at the seabed (Hirai & Jones 2011
   [A]), "almost down to the sea floor" over ~295 m (Kaartvedt 1996 [F]).
4. Fish enter the krill layer: Norway pout by day under dim light over 295 m (Kaartvedt 1996 [F]); "fish invaded
   the krill layer at night" (Oslofjord; Onsrud & Kaartvedt 1998 [A]). Saithe: one dawn haul at 100-145 m over
   ~300 m water (0.5 kg/nmi; Kaartvedt 1996 Table 1) - no rate.
5. Hake: Pacific hake and krill "separated vertically by tens of meters during daylight hours" (Mackas 1997 [A]);
   euphausiid feeding leans evening/night (Livingston 1983 [S]; Rexstad & Pikitch 1986 weak [F]). Box hake eat
   almost no krill (FO 0.024, n 126 day records, 2024 uploads) [D].
6. [J] Two field mechanisms fit a flat-or-rising krill FO over deeper beds: (a) krill reaching the bed or its
   lowest ~50 m by day or at twilight; (b) gadoid excursions into the layer (night, dim light). Neither is
   quantified for saithe in the Skagerrak.

---

## D1. Saithe and hake krill occurrence by sun phase, ICES stomach DB [D]

- **Script:** `scripts/saithe_krill_by_hour.py`. Stdout: `tables/saithe_krill_by_hour_stdout.txt`.
- **Inputs, read only:** research_notes/"Krill över grunda bottnar"/stomachs/tables/records_linked.csv.gz, and
  "Födonivå och mättnad i fält"/data/A_ices_stomach/all (HaulInformation Time, PreyInformation DigestionStage).
- **Sun phase:** sun elevation from lat, lon, date and the haul Time, read as UTC (the ICES convention; not
  verified per upload). Day is > 0 deg, twilight -12..0 deg, night < -12 deg.
- **Fed:** a record with any prey row (`anyprey`). The `fed` column is a count of stomachs.
- **Filler times:** haul Time "1200" occurs in 469 of 1278 saithe records and is excluded as suspected filler.
  The tables below are with it excluded.
- **Coverage:** NE and NW North Sea, Q1 and Q3, 1981-87 plus recent uploads. One Skagerrak record.
  Records are often pooled stomachs (median 3 fish per record).

Saithe, fed records, Time != 1200:

| phase | records | krill FO (records) | hauls | krill FO (hauls) |
|---|---|---|---|---|
| day | 463 | 0.52 | 173 | 0.52 |
| twilight | 70 | 0.39 | 32 | 0.47 |
| night | 38 | 0.47 | 21 | 0.43 |

35-70 cm only:

| phase | records | krill FO |
|---|---|---|
| day | 385 | 0.51 |
| twilight | 61 | 0.36 |
| night | 28 | 0.46 |

3-h UTC bins, 35-70 cm:

| UTC bin | 00 | 03 | 06 | 09 | 12 | 15 | 18 | 21 |
|---|---|---|---|---|---|---|---|---|
| krill FO | 0.47 | 0.43 | 0.29 | 0.59 | 0.58 | 0.40 | 0.62 | 0.52 |
| n | 15 | 14 | 95 | 115 | 81 | 88 | 45 | 21 |

The 06-09 UTC dip is carried by Q1 morning hauls: all fed records in that bin are Q1 63 records at FO 0.19 and
Q3 50 at FO 0.46 (Q3 09-12: 0.64). Q1 twilight FO is 0.08 (n 36), Q3 twilight 0.72 (n 25). Season and area
effects are as large as any clock effect (stdout: "records by 3-h UTC bin x quarter").

Quarter x phase, krill FO:

| quarter | day | night | twilight |
|---|---|---|---|
| Q1 | 0.52 (162) | 0.33 (12) | 0.08 (36) |
| Q3 | 0.53 (256) | 0.45 (20) | 0.72 (25) |
| Q4 | 0.38 (29) | 0.83 (6) | 0.50 (6) |

Bed class (rectangle median) x phase, krill FO:

| bed | day | night | twilight |
|---|---|---|---|
| 100-150 m | 0.50 (270) | 0.54 (24) | 0.19 (36) |
| 150-250 m | 0.51 (112) | 0.38 (13) | 0.74 (19) |
| > 250 m | 0.89 (28) | none | 0.40 (5) |

Selection note: the 100-150 -> 150-250 m day FO here (0.50 -> 0.51) is lower than "Sejens profil over djupa
bottnar" (0.65 -> 0.76, all quarters) because this selection includes the NW North Sea, drops the 469 records with
Time "1200", and classes by `bed_rectmed`; it is not a contradiction of that note.

Weights and digestion stage:
- **Krill weight share** exists by day only: 129 records, mean per-record share 0.28, pooled 0.05. The 25 twilight
  records with weights have no krill. No night weights exist.
- **Digestion stage:** of 1509 euphausiid prey rows in fed saithe records, 66 carry a DigestionStage, all stage 1
  and all by day. No twilight or night krill row is staged.

Hake, fed records:
- Krill FO is 0.024 by day (n 126) and 0 in twilight (n 9). There is no night sample.
- Krill weight share pooled is 0.003 (n 109).
- All 135 fed records (Time != 1200) are 2024 uploads: Skagerrak 50, Kattegat 28, NW North Sea 36, C+S North Sea
  13, NE North Sea 8; Q1 34, Q3 101.
- 35-70 cm hake: FO 0.06 (n 36).

Caveats:
- Krill evacuation takes 24-42 h. Rindorf 2003 [F, via "Krillintag och jaktsätt"] found krill "disappeared
  completely after 24, 31 and 42 h". FO by phase therefore integrates over about a day and cannot show the
  feeding window.
- Night and twilight samples are small.
- The capture bed is not the feeding bed.
- UTC is assumed.
- Reading [J]: the data do not contradict feeding at any phase, and they do not show one.

## E1. Dolgov, Yaragina, Ajiad, Mehl & Bogstad 1992, "Daily rations of cod from the Barents Sea"

- **Reference:** 5th PINRO-IMR Symposium, Murmansk. hdl:11250/107818.
- **Tag:** [F] for cod, [S] as an analogue for saithe.
- **Text read:** research_notes/"Krillintag och jaktsätt"/data/q1_brage_cod1992/03_layout.txt (that note's
  download).
- **Design:** 24-h stations with a bottom trawl every ~4 h and 30 cod stomachs per haul.
- **Stations** (Table 1):
  - Rybachy Bank, 70°20'N, 27-28 June 1988, bed 175-240 m, 1.4-1.6 C, 192 stomachs.
  - Murman shallow, 5-7 Aug 1989, bed 125-150 m, 420 stomachs.

Rybachy Bank (unpaginated scan, Results section 2b):
- "Euphausiids formed the bulk (84.1% in terms of weight) of the cod diet".
- "An insignificant reduction in the share of euphausiids (in terms of weight) in the food bolus was recorded during
  the period from 8 p.m. to 4 a.m. ... The frequency of occurrence of euphausiids did not show actual variations
  ... The maximum feeding activity was observed in the period from 8 p.m. to 4 a.m."

Murman shallow (unpaginated scan, Results section 2a):
- "Thysanoessa inermis and Th. raschii 2.0-2.5 cm long as well as Meganyctiphaenes norvegica 3.0-3.5 cm long made
  the bulk (48.4%) of the food bolus".
- "Cod were registered as large individual schools and their behaviour was determined by migrations of
  euphausiids. The euphausiid aggregations migrated to 0-40 m layer during 24 hours which was accompanied by the cod
  migration to the layer 10-15 m from the bottom to result in the catch reduction. The 2nd drop in the yield was
  observed at 8 a.m. when euphausiids moved to the layer 20-30 m above the bottom."

Discussion:
- "the diurnal feeding rhythms can be observed only under conditions of a stable food availability".
- Citing Yaragina 1985: maximum feeding "from 0 to 4 a.m. and from 4 to 8 p.m. ... when cod fed on euphausiids".

Caveats:
- "0-40 m layer" is ambiguous: it could be above the bottom or from the surface.
- At 70°N in late June the sun does not set, so the Rybachy station has no dark phase [J].
- These are cod and Thysanoessa-dominated krill on a bank, not saithe on M. norvegica.
- Time zone not stated.

## E2. Skaret, Johansen, Johnsen, Fall, Fiksen, Englund, Fauchald, Gjøsæter, Macaulay & Johannesen 2020

- **Reference:** "Diel vertical movements determine spatial interactions between cod, pelagic fish and krill on an
  Arctic shelf bank". Mar Ecol Prog Ser 638:13-23. doi:10.3354/meps13254.
- **Tag:** [F] for cod, [S] as an analogue for saithe.
- **Quotes:** `data/q2_skaret2020/skaret2020_quotes.txt`. The PDF was deleted (© Inter-Research).
- **Site:** northern Barents Sea bank, bed 160-170 m, ~1 C, 8-9 Oct 2016. Krill mostly Thysanoessa inermis.
  Broadband probe at ~120 m above the bed; 239 cod stomachs.

Quotes:
- p.13: "[cod] remained mostly within 10 m of the bottom throughout the diel cycle. In the morning, cod lifted slightly
  from the seabed indicating feeding activity, which corresponded with more fresh prey in cod stomachs in the
  morning. During daylight, krill pushed towards the bottom, sharing habitat with cod".
- p.18: "During daytime, pelagic fish remained from a few to 50 m above the bottom while krill were found close to
  the bottom".
- p.20: "Of the stomachs containing food, 79% contained pelagic fish and/or unidentified fish, 65% contained
  identified pelagic fish". The probability of fresh pelagic fish declined with hours since dawn (beta -0.18,
  p = 0.0029).
- p.21: "The cod is known to be a potential predator on krill, but prefer pelagic fish, and paradoxically krill may
  therefore gain safety by hiding among the cod."

Caveats:
- Light reaches the bed on this bank (K 0.064 /m). The authors say the shallow bank blocks the krill's descent.
- No krill share of the cod diet is given in the main text (it is in Supplement Fig. S1, not read).

## E3. Kaartvedt, Melle, Knutsen & Skjoldal 1996

- **Reference:** "Vertical distribution of fish and krill beneath water of varying optical properties". Mar Ecol
  Prog Ser 136:51-58. doi:10.3354/meps136051.
- **Tag:** [F]. The text was read from the Q1 agent's pdftotext extraction (`data/q1_kaartvedt1996_meps136/paper.txt`, since
  replaced by its `quotes.txt`).
- **Site:** outer Norwegian shelf, bed ~295-300 m, April 1993.

Quotes:
- p.51 abstract: "Krill (mainly Thysanoessa inermis) was primarily found below the mesopelagic fish and above
  planktivorous demersal fish (Norway pout Trisopterus esmarkii)."
- p.51 abstract: "Demersal fish left the benthic boundary zone, with ascending Norway pout foraging in the lower part
  of the krill layer."
- Results: "ascending 'demersal' fish were observed in the lower 50 to [number illegible in the extraction] m of the water column during the
  subsequent ~50 km of acoustical recordings in daylight. Their distribution partly extended into the lower
  component of the krill layer".
- Results: "Outside a patch of fish, the krill layer extended almost down to the sea floor."
- Table 2: 9 Norway pout from 170-220 m over a 295 m bed at 20:25 (GMT+1). "Thysanoessa inermis was the only
  identifiable prey item".
- Table 1: Stn 292, Akra trawl, 22 April, 04:10 (GMT+1), fishing depth 145-100 m, "upper layer". Catch includes
  "Pollachius virens (0.5)" kg per nautical mile, with Maurolicus 2.1 and herring 0.6.

Caveats:
- Pout, not saithe. The ascent is tied to a front with low light penetration, an "antipredation window".
- The saithe record is one dawn haul at 100-145 m over ~300 m water. It is not bed-referenced and gives no rate.

## E4. Onsrud & Kaartvedt 1998

- **Reference:** "Diel vertical migration of the krill Meganyctiphanes norvegica in relation to physical
  environment, food and predators". Mar Ecol Prog Ser 171:209-219. doi:10.3354/meps171209.
- **Tag:** [A] (OpenAlex abstract; the int-res page returned 429).
- **Site:** Oslofjord.

Quote (abstract): "Acoustic scattering layers of Meganyctiphanes norvegica in the Oslofjord, Norway, typically stayed
below ~70 m [OCR '-70 m', corrected] during the day. They remained below prevailing fish schools, though varying numbers of deep-living fish
occurred among the krill. ... upward migration was mainly arrested at ~10-30 m [OCR '-10-30 m', corrected] and the krill were scattered throughout
the water column at night. ... Fish schools dispersed at sunset, and fish invaded the krill layer at night."

Caveats: the fish species are not named in the abstract. The companion Onsrud et al. 2004 (whiting in the krill
layer, 120 m bed) is covered in "Krillintag och jaktsätt".

## E5. Mackas, Kieser, Saunders, Yelland, Brown & Moore 1997

- **Reference:** "Aggregation of euphausiids and Pacific hake (Merluccius productus) along the outer continental
  shelf off Vancouver Island". Can J Fish Aquat Sci 54(9):2080-2096 (OpenAlex).
  doi:10.1139/f97-113.
- **Tag:** [A] (OpenAlex).

Quote: "Both euphausiids and hake share similar horizontal distributions, although separated vertically by tens of
meters during daylight hours."

Caveat: the direction (hake above or below the krill) is not stated in the abstract.

## E6. Rexstad & Pikitch 1986, Fish Bull 84:947-956

- **Tag:** [F] (text from the "Krillintag och jaktsätt" download).
- **Material:** Pacific hake, Washington-Oregon, August 1983.

Quotes:
- Results (Fig. 3 text): "There is a weak indication that these fish exhibit a pattern of feeding more heavily at night than during
  the day ... No tows were made between the hours of 2000 and 0700 thus direct evidence of nocturnal feeding was not
  available."
- Discussion, citing Alton & Nelson 1970 and Livingston 1983: "Pacific hake as nocturnal predators that migrate
  vertically to feed near the surface during hours of darkness and dive to deeper water during daylight hours."
- Discussion (p.954): "feeding to satiation during evening hours may have been impossible; consequently, feeding occurred
  whenever euphausiids were encountered."

## E7. Livingston 1983, Fish Bull 81:629-636

- **Tag:** [S], via "Krillintag och jaktsätt"/q1_evidence.md S5.
- **Material:** Pacific hake diel series, 14-15 July 1967, 251 stomachs with food.

Quote: "Stomach content weight per fish weight is highest at 1800 h, with a value of 2.5%".

## E8. Widder, Greene & Youngbluth 1992

- **Reference:** "Bioluminescence of sound-scattering layers in the Gulf of Maine". J Plankton Res 14:1607-1624.
  doi:10.1093/plankt/14.11.1607.
- **Tag:** [A].

Quote: "Meganyctiphanes norvegica formed a SSL within 50 m of the bottom during the day and migrated to the uppermost
30 m of the water column at night".

Caveat: the bed depth is not in the abstract. These are submersible dives in Gulf of Maine basins [J].

## E9. Hirai & Jones 2011

- **Reference:** "The temporal and spatial distribution of krill (Meganyctiphanes norvegica) at the deep seabed of
  the Faroe-Shetland Channel, UK". Mar Biol Res 8:48-60. doi:10.1080/17451000.2011.594891.
- **Tag:** [A].

Quotes:
- "near the seabed (<5 m above the seafloor) ... most abundant near the seabed at 400-600 m depth".
- "densities declined at night owing to daily vertical migrations".
- "higher krill densities at 06:00-09:00 and 18:00-21:00 at a depth of 400-480 m".
- "being consumed by benthic and epibenthic predators".

Caveats: the bed is deeper than the box. The predators are not named in the abstract.

## E10. Carruthers, Neilson, Waters & Perley 2005

- **Reference:** "Long-term changes in the feeding of Pollachius virens on the Scotian Shelf: responses to a dynamic
  ecosystem". J Fish Biol 66(2):327-347. doi:10.1111/j.0022-1112.2005.00594.x.
- **Tag:** [A] (OpenAlex abstract; authors and pages from OpenAlex).

Quote: "During the early period, euphausiids were present in 65% of the pollock stomachs that contained food and only
9% of these stomachs in the recent period" (1958-67 against 1996-2002).

Caveat: no diel or vertical information.

## E11. Already covered elsewhere (not redone)

- Armannsson & Jonsson 2012 (DSTs, depth only):
  - high daytime activity;
  - rapid vertical runs 0.15 of the time;
  - daily vertical range ~1:1 with bed depth.
- Kaartvedt et al. 2005: 60-110 cm saithe under krill swarms ate fish.
- Stensholt et al. 2001: saithe near the bed over beds < 200 m.
- HERAS hauls: adult saithe within 30 m of 150-250 m beds by day; some at the surface at night over deeper beds.
- Onsrud et al. 2004: whiting in the krill layer.
- Modica 2011: juvenile hake eat euphausiids pelagically at night.

See "Krillintag och jaktsätt" and "Sejens profil over djupa bottnar".

## Not obtained / searched without result

- **Not obtainable:**
  - Bergstad 1991 (Sarsia 75) and Du Buit 1991 (Fish Res 12:307): paywalled, as in three earlier notes.
  - Rosen & Holst 2013 (Fish Res 148:64; doi:10.1016/j.fishres.2013.08.002): no open copy. Underwood et al. 2014
    (PLoS ONE, DeepVision, Svalbard) has no saithe.
  - Kaartvedt 2023 L&O (zenodo 8322116): 403. Plymouth "Notes on Euphausiids" (Loch Fyne; per the search snippet,
    "stomachs of Gadus virens well filled with specimens of M. norvegica"): 401, not quoted as evidence.
- **Not relevant here:**
  - Homrum et al. 2013 (doi:10.1093/icesjms/fst048): conventional-tag migration only, no diel depth (abstract).
  - Otterå & Skilbrei 2014 (doi:10.1093/icesjms/fsu096): farm-site residency, no vertical-krill data (abstract).
- **Searched, nothing found:**
  - a North Sea or Thünen saithe DST study with diel depth;
  - a saithe diel stomach series with krill;
  - any height-above-bed time series for saithe or European hake.
