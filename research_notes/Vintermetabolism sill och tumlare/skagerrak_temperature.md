# Seasonal sea water temperature profile, eastern Skagerrak off Bohuslan (Mareld site, ~58.15 N, 10.5-11.2 E)

Method note (applies to all sections): the core numbers below are NOT copied from a report. They were computed
for these notes from the raw SMHI/SHARK monitoring data (national marine monitoring CTD + bottle casts),
downloaded through the public SHARK API on 2026-10-06. Reproducible request:

```
POST https://shark.smhi.se/api/sample/download   (Content-Type: application/json)
{"params":{"tableView":"sharkdata_physicalchemical_columns","delimiters":"point-tab","lineEnd":"unix",
           "encoding":"utf_8","headerLang":"internal_key"},
 "query":{"fromYear":1991,"toYear":2025,"dataTypes":["Physical and Chemical"],
          "parameters":["Temperature CTD","Temperature bottle"],
          "stationName":["Å13","Å14","Å15","Å16","Å17","A17","A16","A13","SLÄGGÖ","ANHOLT E"]}}
```
(18,347 sample rows returned; API docs at https://shark.smhi.se/api/docs, station/parameter lists at
https://shark.smhi.se/api/options.) Only stations Å16, Å17, SLÄGGÖ and ANHOLT E came back under those names.

Processing: CTD temperature, bottle temperature as fallback; values outside -2..30 C dropped. QC: 41 CTD and 76 bottle
temperature values carry flag B (bad) and 14 flag E/R/S; excluding B/S changes no site-table cell at 0-50 m
(max difference 0.00 C), so they were left in. Per cruise visit, temperature averaged inside depth windows 0-2.5 m ("surface"), 4-6, 8.5-11.5 (10 m),
18-22 (20 m), 27-33 (30 m), 45-55 (50 m), 70-80, 95-105 m. Visits averaged to year-month first, then year-months
averaged across years (so years with two cruises in a month do not get double weight). Sampling is ~monthly
snapshots (one or two casts per month, date within the month varies) - a "monthly mean" here is the mean of
monthly snapshots, not a time-integrated monthly mean. Reference period 1991-2020 matches SMHI's "normal"
period ([SMHI RO 81, 2025](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)).

Station geography (medians from the SHARK records):
| Station | Lat N | Lon E | Water depth | Data years in SHARK pull | Visits |
|---|---|---|---|---|---|
| Å16 (Å-transect, Skagerrak) | 58.267 | 10.725 | ~202 m | 1992-2024 (series in SHARK from 1962) | 358 |
| Å17 (Å-transect, Skagerrak) | 58.275 | 10.513 | ~344 m | 1992-2024 (series in SHARK from 1964) | 295 |
| Släggö (Gullmarsfjord mouth) | 58.260 | 11.436 | ~73 m | 2018-2024 only | 221 |
| Anholt E (Kattegat, comparison) | 56.667 | 12.117 | ~60 m | 1991-2024 | 696 |

Å16 and Å17 lie ~13 km north of 58.15 N and span the Mareld longitude band (10.5-10.7 E); they are the
directly representative open-water stations. Släggö is coastal/fjord-mouth and Anholt E is Kattegat (warmer
summer, colder winter surface, more continental).

---

## 1. Monthly mean temperatures (surface and 10-50 m) at stations near the site

### Takeaway
Recommended table for the model: the mean of Å16 and Å17, 1991-2020 (below). Surface ranges from 3.4 C (Feb)
to 17.7 C (Aug); 50 m ranges only from 6.6 C (Mar-Apr) to 11.7 C (Oct). Annual means: 10.3 C (surface),
9.8 (10 m), 9.2 (20 m), 8.8 (30 m), 8.4 (50 m). Confidence: high for the shape and month of extremes,
moderate (+/- ~0.5 C) for any single cell given 17-27 years of monthly snapshots per cell and interannual
SD of 1-2.7 C.

### Cited Findings

**Recommended site table - mean of Å16 and Å17, 1991-2020 (C)** — computed from [SHARK / SMHI national monitoring data](https://shark.smhi.se/api/docs)

| Month | 0-2 m | 5 m | 10 m | 20 m | 30 m | 50 m | 75 m | 100 m | 0-50 m mean* |
|---|---|---|---|---|---|---|---|---|---|
| Jan | 5.0 | 5.1 | 5.4 | 6.3 | 7.0 | 7.7 | 7.8 | 7.9 | 6.3 |
| Feb | 3.4 | 3.5 | 4.3 | 5.6 | 6.5 | 7.2 | 7.2 | 7.2 | 5.4 |
| Mar | 3.9 | 4.0 | 4.3 | 5.3 | 6.0 | 6.6 | 6.7 | 6.7 | 5.2 |
| Apr | 6.3 | 6.1 | 5.6 | 5.8 | 6.2 | 6.6 | 6.7 | 6.8 | 6.1 |
| May | 10.8 | 9.8 | 8.2 | 6.7 | 6.5 | 6.8 | 6.8 | 6.8 | 7.8 |
| Jun | 14.6 | 14.1 | 11.7 | 8.4 | 7.6 | 7.2 | 7.1 | 7.0 | 9.9 |
| Jul | 16.9 | 16.5 | 15.1 | 11.2 | 9.2 | 8.0 | 7.5 | 7.4 | 12.1 |
| Aug | 17.7 | 17.6 | 17.0 | 13.3 | 10.6 | 8.5 | 7.8 | 7.5 | 13.4 |
| Sep | 15.3 | 15.4 | 15.4 | 14.9 | 13.0 | 10.2 | 8.5 | 7.9 | 13.8 |
| Oct | 12.7 | 12.7 | 13.0 | 13.4 | 13.4 | 11.7 | 9.3 | 8.3 | 12.8 |
| Nov | 9.9 | 10.0 | 10.1 | 10.8 | 11.2 | 10.6 | 9.8 | 9.0 | 10.5 |
| Dec | 7.2 | 7.5 | 8.0 | 8.7 | 9.0 | 9.3 | 9.3 | 9.1 | 8.4 |
| Annual | 10.3 | 10.2 | 9.8 | 9.2 | 8.8 | 8.4 | 7.9 | 7.6 | - |

*0-50 m mean = unweighted mean of the 0-2, 10, 20, 30, 50 m levels (not depth-weighted).

**Same site mean, recent period 2011-2024 (C)** — computed from [SHARK](https://shark.smhi.se/api/docs)

| Month | 0-2 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| Jan | 5.6 | 5.9 | 6.6 | 7.2 | 7.9 | 8.1 |
| Feb | 3.8 | 4.5 | 5.6 | 6.5 | 7.1 | 7.4 |
| Mar | 3.8 | 4.0 | 5.1 | 6.0 | 6.8 | 7.0 |
| Apr | 6.4 | 5.8 | 6.0 | 6.5 | 6.9 | 7.1 |
| May | 11.1 | 8.9 | 7.4 | 7.0 | 7.1 | 7.0 |
| Jun | 14.7 | 12.8 | 8.8 | 8.1 | 7.4 | 7.2 |
| Jul | 16.8 | 14.4 | 11.3 | 9.2 | 8.2 | 7.6 |
| Aug | 17.8 | 16.9 | 13.8 | 11.6 | 9.0 | 7.8 |
| Sep | 15.7 | 15.6 | 14.7 | 12.8 | 10.0 | 8.1 |
| Oct | 12.8 | 13.1 | 13.5 | 13.6 | 12.5 | 8.7 |
| Nov | 10.3 | 10.4 | 11.1 | 11.4 | 10.6 | 8.9 |
| Dec | 7.1 | 8.1 | 9.0 | 9.5 | 9.7 | 9.1 |
| Annual | 10.5 | 10.0 | 9.4 | 9.1 | 8.6 | 7.8 |

**Per-station 1991-2020 values (C), surface / 20 m / 50 m** — computed from [SHARK](https://shark.smhi.se/api/docs)

| Month | Å17 0-2 | Å17 20 | Å17 50 | Å16 0-2 | Å16 20 | Å16 50 | Anholt E 0-2 | Anholt E 20 | Anholt E 50 |
|---|---|---|---|---|---|---|---|---|---|
| Jan | 5.1 | 6.4 | 7.9 | 4.9 | 6.2 | 7.5 | 3.0 | 5.4 | 7.3 |
| Feb | 3.5 | 5.6 | 7.4 | 3.4 | 5.5 | 7.0 | 2.2 | 4.9 | 6.3 |
| Mar | 4.0 | 5.5 | 6.9 | 3.7 | 5.2 | 6.3 | 3.0 | 4.6 | 5.6 |
| Apr | 6.4 | 5.8 | 6.7 | 6.3 | 5.7 | 6.4 | 6.6 | 5.1 | 5.5 |
| May | 10.8 | 6.6 | 6.8 | 10.7 | 6.9 | 6.7 | 11.3 | 5.8 | 5.8 |
| Jun | 14.6 | 8.2 | 7.1 | 14.6 | 8.6 | 7.4 | 15.4 | 8.4 | 6.1 |
| Jul | 17.2 | 10.7 | 7.6 | 16.7 | 11.7 | 8.3 | 18.3 | 12.6 | 6.6 |
| Aug | 17.8 | 12.5 | 8.2 | 17.7 | 14.1 | 8.8 | 18.8 | 14.8 | 7.8 |
| Sep | 15.4 | 14.8 | 9.8 | 15.2 | 15.1 | 10.6 | 16.1 | 15.4 | 9.9 |
| Oct | 12.8 | 13.4 | 12.0 | 12.6 | 13.4 | 11.5 | 11.8 | 13.2 | 12.2 |
| Nov | 10.1 | 10.8 | 10.4 | 9.7 | 10.8 | 10.8 | 8.6 | 10.8 | 11.3 |
| Dec | 7.4 | 8.8 | 9.3 | 7.0 | 8.6 | 9.3 | 5.7 | 8.3 | 9.8 |
| Annual | 10.4 | 9.1 | 8.3 | 10.2 | 9.3 | 8.4 | 10.1 | 9.1 | 7.9 |

Years contributing per month-cell (1991-2020): Å17 17-22, Å16 18-27, Anholt E 24-30.
Å17 10 m / 30 m: Jan 5.4/7.1, Feb 4.3/6.8, Mar 4.4/6.2, Apr 5.6/6.2, May 8.1/6.5, Jun 11.9/7.3, Jul 14.9/8.6,
Aug 17.0/9.9, Sep 15.5/12.3, Oct 13.0/13.3, Nov 10.3/11.2, Dec 8.3/9.0.
Å16 10 m / 30 m: Jan 5.3/6.9, Feb 4.3/6.3, Mar 4.2/5.8, Apr 5.6/6.2, May 8.2/6.6, Jun 11.5/7.8, Jul 15.3/9.8,
Aug 17.0/11.3, Sep 15.3/13.8, Oct 13.0/13.5, Nov 9.9/11.1, Dec 7.6/9.1.

Interannual SD of the monthly values (Å17, 1991-2020): surface 1.1-2.5 C (largest Jan and May); 20 m 0.9-2.7 C
(largest Aug); 50 m 0.5-2.6 C (largest Sep-Oct, when the thermocline reaches 50 m in some years but not
others); 100 m 0.4-0.9 C — computed from [SHARK](https://shark.smhi.se/api/docs).

**Släggö (Gullmarsfjord mouth), 2018-2024 only (7 years; 2011-2024 column of the computation)** —
computed from [SHARK](https://shark.smhi.se/api/docs): surface Jan 3.8, Feb 3.5, Mar 3.3, Apr 5.9, May 10.7,
Jun 16.6, Jul 18.4, Aug 19.0, Sep 16.8, Oct 12.3, Nov 9.9, Dec 5.1; 30 m Jan 7.5, Feb 5.9, Mar 5.7, Apr 6.0,
May 6.9, Jun 8.5, Jul 12.6, Aug 14.7, Sep 15.5, Oct 13.9, Nov 12.2, Dec 10.1; 50 m Mar 6.1 (min) to Oct 12.1 (max).
Coastal: ~1 C warmer surface in summer and ~1 C colder in Dec-Mar than Å16/Å17, and a deeper warm
summer layer (30 m reaches 15 C in Sep).

**Published text values (single-year or descriptive, for cross-check):**
- 2024: "In the Skagerrak, the Kattegat and the Sound, along the Swedish coast, the surface water, 0-10 m, was
  coldest in January when the temperature dropped to 2-2.5°C. At the coastal station Släggö it was even colder,
  with a temperature measurement of 1.7°C in January... the maximum was observed in August with surface
  temperatures reaching 18-19°C... in December temperatures had dropped to 6-7°C which is normal." —
  [SMHI Report Oceanography 81 (2025), sec. 3.1.1](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- Along the Å-transect below 100 m "the temperature was around 7-8°C, and the salinity around 35 psu throughout
  the year"; Kattegat below 40 m 5-13 C — [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- SMHI annual reports plot station Å15 (Skagerrak) and Anholt E surface (0-10 m) monthly values against the
  1991-2020 mean +/- 1 SD (Figure 5 of RO 81) — the climatology itself is shown only graphically —
  [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- Earlier-year SMHI report: at Å17 "the surface temperature reached a minimum of 0.5°C in February" (cold year)
  and maximum in July — [SMHI RO 73 (2022), search snippet](https://www.diva-portal.org/smash/get/diva2:1733898/FULLTEXT01.pdf) (year not verified from full text).
- Gullmar fjord surface: lowest in February-March (2-4 C), highest in August-September (15-17 C); in one
  sampling year from 2.5 C in winter to a 20.3 C summer peak — [Frontiers in Microbiology 2022, Synechococcus in Gullmar Fjord](https://pmc.ncbi.nlm.nih.gov/articles/PMC9125215/) (via search summary; not checked against full text).

### Inferences
- Å16 and Å17 differ by < 0.5 C in almost every cell, so averaging them is safe; the Mareld band
  (10.5-11.2 E, slightly south) should sit between Å16/Å17 and the coastal Släggö values, closer to Å16/Å17.
- For a single-layer metabolism multiplier, the depth that matters depends on the organism: herring/sprat
  and porpoise in winter are often below the surface layer, so the 20-50 m columns (winter minimum ~5.3-6.6 C,
  never below ~5 C) are more relevant than the surface (3.4 C). The 0-50 m mean column is a reasonable
  water-column compromise (min 5.2 C in Mar, max 13.8 C in Sep).
- The 2011-2024 table is ~0.2 C warmer on annual mean at all depths than 1991-2020; differences are within
  interannual noise for individual months.

### Gaps
- Stations Å13, Å14, Å15 are not in SHARK: a bounding-box query (58.15-58.40 N, 10.40-11.35 E, 1960-2025)
  returned only Å16 and Å17 as regular stations (plus 2018+ coastal stations Byttelocket, BroA, and scattered
  1960s-80s one-off casts). Their data, if they exist, must be requested from SMHI directly.
- Släggö: a bounding-box query (58.20-58.32 N, 11.35-11.50 E, 1960-2025) also returned Släggö only from 2018
  (plus Vällarna 2018-2024, Pricken 2022-2023). The longer Släggö series that SMHI uses for its 1991-2020
  "normal" is not in public SHARK; its climatology here rests on 7 years only, low confidence.
- No direct Kristineberg/Gullmar long-term temperature table was found as text.
- Data are monthly snapshots; no daily-resolved climatology at depth for the exact coordinates.

---

## 2. Coldest and warmest months; annual range at surface vs 30-50 m

### Takeaway
Surface is coldest in February (3.4 C; March nearly as cold, 3.9 C) and warmest in August (17.7 C). The
extremes lag with depth: 20-30 m minimum in March and maximum in September-October; 50 m minimum in
March-April (6.6 C) and maximum in October (11.7 C). The annual range shrinks from ~14.3 C at the surface to
~9.6 C at 20 m, ~7.4 C at 30 m, ~5.1 C at 50 m, ~3.1 C at 75 m and ~2.4 C at 100 m. Confidence: high.

### Cited Findings
- Month of minimum / maximum, Å16+Å17 mean 1991-2020: surface Feb / Aug; 5 m Feb / Aug; 10 m Feb / Aug;
  20 m Mar / Sep; 30 m Mar / Oct; 50 m Apr (Mar 6.6 vs Apr 6.6, tied) / Oct; 75 m Mar / Nov; 100 m Mar / Dec —
  computed from [SHARK](https://shark.smhi.se/api/docs). In 2011-2024 the 5-50 m minima are all in March.
- Annual range (max-min of monthly means) 1991-2020: Å17 surface 14.3, 10 m 12.7, 20 m 9.3, 30 m 7.1, 50 m 5.3,
  100 m 2.3 C; Å16 surface 14.3, 20 m 9.9, 30 m 8.0, 50 m 5.1, 100 m 2.8 C; Anholt E (Kattegat) surface 16.7,
  20 m 10.7, 30 m 7.9, 50 m 6.7 C — computed from [SHARK](https://shark.smhi.se/api/docs).
- Winter temperature inversion: Dec-Mar the water gets warmer with depth (Feb: 3.4 C at surface vs 7.2 C at
  50 m), because the cold, low-salinity Baltic outflow lies on top of warmer, saltier Skagerrak/Atlantic water —
  computed from [SHARK](https://shark.smhi.se/api/docs); consistent with SMHI's description that in the
  Baltic Proper "below the halocline the temperature was higher than at the surface" in winter and that the
  Å-transect upper layer is governed by the Baltic current — [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf).
- 2024 published: Skagerrak/Kattegat surface (0-10 m) coldest in January (2-2.5 C), maximum in August (18-19 C) —
  [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf).
- Gullmar fjord surface: minimum Feb-Mar (2-4 C), maximum Aug-Sep (15-17 C) — [Frontiers 2022 / PMC9125215](https://pmc.ncbi.nlm.nih.gov/articles/PMC9125215/).

### Inferences
- The "coldest month" depends on depth: for surface-feeding organisms it is February; for anything living
  at 20-50 m it is March (to April at 50 m). A model with one temperature per month for the 0-50 m column has
  its minimum in March (5.2 C) and its maximum in September (13.8 C), i.e. about one month later than the surface.
- Coldest single-year surface values are well below the climatology (0.5 C at Å17 in a cold February; 1.7 C at
  Släggö in Jan 2024), so a monthly climatology underestimates cold-stress extremes.

### Gaps
- Exact day-of-year of min/max cannot be resolved from monthly snapshots.

---

## 3. Seasonal stratification: thermocline onset/breakdown and mixed-layer depth

### Takeaway
Thermal stratification develops in late April-May, is strongest June-August (thermocline at ~10-15 m in
June-July, deepening to ~15-20 m in August and ~35 m in September), and is broken down by October-November;
Dec-Mar the upper layer is thermally inverted and mixed to ~10-20 m above a salinity-controlled (Baltic
current) halocline, not to the bottom. Confidence: moderate (coarse standard-depth casts; the Baltic-current
halocline complicates any mixed-layer definition).

### Cited Findings
Monthly medians from 645 Å16/Å17 casts, 1991-2024 (computed from [SHARK](https://shark.smhi.se/api/docs);
thermocline = depth of maximum downward temperature decrease; T-MLD = first depth where |T - T(surface)| > 0.5 C;
rho-MLD = first depth where density exceeds the surface value by 0.125 kg/m3 using a simple linear equation of
state; vertical resolution only ~5-10 m):

| Month | T-MLD (m) | rho-MLD (m) | Thermocline depth (m) | Max dT/dz (C/m) | T(surface) - T(50 m) (C) |
|---|---|---|---|---|---|
| Jan | 20 | 15 | none (inverted) | 0.01 | -2.0 |
| Feb | 15 | 12.5 | none (inverted) | 0.02 | -3.2 |
| Mar | 15 | 10 | none (inverted) | 0.01 | -2.3 |
| Apr | 15 | 10 | ~7.5 (weak, forming) | 0.14 | -0.3 |
| May | 10 | 7.5 | ~7.5 | 0.43 | +3.8 |
| Jun | 10 | 10 | ~12.5 | 0.74 | +7.3 |
| Jul | 10 | 10 | ~12.5 | 0.60 | +9.2 |
| Aug | 15 | 10 | ~17.5 | 0.53 | +9.2 |
| Sep | 30 | 12.5 | ~35 | 0.33 | +6.0 |
| Oct | 20 | 15 | (eroding, ~60) | 0.16 | +0.5 |
| Nov | 20 | 20 | none | 0.09 | -0.6 |
| Dec | 15 | 10 | none (inverted) | 0.03 | -2.1 |

- SMHI 2024 (Å-transect): "in the beginning of the year, the surface water was generally well-mixed down to
  20-30 meters and as the surface water heated, a thermocline was developed. In June, the thermocline was as its
  shallowest, 10 meters, and coincided with a shallow halocline. As summer continued, a larger water layer heated
  and the depth of the thermocline increased. After the temperature peak in August the surface water started to
  cool down. For the rest of the year there was a layer of warmer water below the surface layer remaining from
  the summer." — [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- SMHI summary (all Swedish areas, 2024): winter water columns "well mixed down to 20-60 m depending on region";
  "Seasonal warming began in April-May, producing a summer thermocline at 10-20 m." — [SMHI RO 81, Summary](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- Nutrients in 2024 summer "very low to depleted above the thermocline mostly located between 15-20 meters" —
  [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- Gullmar fjord: summer pycnocline around 18 m, shallower toward the fjord mouth; in autumn at 10 m along the
  fjord with stronger stratification; summer range 7 C deep to 18 C surface; fjord deep-water exchange once a year
  in winter, mostly January-March — [Frontiers 2022 / PMC9125215](https://pmc.ncbi.nlm.nih.gov/articles/PMC9125215/)
  (via search summary).

### Inferences
- For the model: "stratified" months = May-September (surface-50 m difference > 3.5 C); transition April and
  October; winter (Nov-Mar) has a shallow (~10-20 m) low-salinity surface layer that is COLDER than the water
  below. A mixed-layer depth table for e.g. a light/P-I model: ~10-20 m in winter (haline cap), ~7.5-10 m in
  May-July, ~10-15 m in August, ~15-30 m in September-October, ~20 m in November. Note this is much shallower
  than open North Sea winter mixed layers, because of the Baltic current.
- The deep (>100 m) Skagerrak water is a stable 7-8 C refuge all year.

### Gaps
- Casts are at standard depths (0, 2, 5, 10, 15, 20, 30 m...), so MLD/thermocline values have ~5 m resolution.
- A published, peer-reviewed MLD climatology specific to the Å-transect was not found.

---

## 4. Gridded products for a monthly climatology at the exact coordinates

### Takeaway
Three usable routes: (a) SHARK raw station data (what was used here; free, no login); (b) Copernicus Marine
Baltic Sea physics reanalysis BALTICSEA_MULTIYEAR_PHY_003_011 (domain starts at 9.04 E, so it covers the
Mareld site at ~1-2 km resolution, 56 depth layers); (c) Copernicus NW-Shelf reanalysis
NWSHELF_MULTIYEAR_PHY_004_009 (7 km, 24 levels, 1993-2026, includes mixed-layer thickness). Both Copernicus
products need a free account and the `copernicusmarine` Python toolbox. Confidence: high on identity, moderate on
exact dataset sub-IDs (not verified).

### Cited Findings
- BALTICSEA_MULTIYEAR_PHY_003_011: lateral boundaries 53 N - 65.9 N, 9.04 E - 30.21 E; "the whole Baltic Sea area,
  inclusive the Transition Area to the North Sea"; resolution 0.01666 deg lat x 0.027777 deg lon (~1.8 km);
  56 depth layers — [Copernicus QUID CMEMS-BAL-QUID-003-011](https://documentation.marine.copernicus.eu/QUID/CMEMS-BAL-QUID-003-011.pdf), [product page](https://data.marine.copernicus.eu/product/BALTICSEA_MULTIYEAR_PHY_003_011/description)
- NWSHELF_MULTIYEAR_PHY_004_009: NEMO + NEMOVAR 3DVar assimilating SST and T/S profiles, 7 km, tides, ERA5
  forcing, Baltic boundary from BALTICSEA_REANALYSIS_PHY_003_011; 1 Jan 1993 - 1 Aug 2026; hourly/daily/monthly;
  24 depth levels; variables include temperature, salinity, currents, SSH, mixed layer depth, bottom temperature;
  updated biannually — [Copernicus product page](https://data.marine.copernicus.eu/product/NWSHELF_MULTIYEAR_PHY_004_009/description)
- NWS domain 40.07-65 N, 19.89 W - 13 E, 0.111 x 0.067 deg; free; NetCDF-4; DOI 10.48670/moi-00059 —
  [Copernicus product page](https://data.marine.copernicus.eu/product/NWSHELF_MULTIYEAR_PHY_004_009/description)
- SHARK API (shark.smhi.se/api): datatypes include "Physical and Chemical" (from 1902) and "Profile" (CTD
  profiles, from 2018); parameters "Temperature CTD", "Temperature bottle", "Salinity CTD"; POST
  /api/sample/download returns tab-separated text; filters by station, year, month, depth, bounding box —
  [SHARK API docs](https://shark.smhi.se/api/docs), [options](https://shark.smhi.se/api/options)
- SMHI uses 1991-2020 mean +/- 1 SD as "normal" in its monitoring reports — [SMHI RO 81](https://www.smhi.se/download/18.5a9957c419abf7cf27f476c/1766137855550/RO_81%20The%20Swedish%20National%20Marine%20Monitoring%20Programme%202024%20Hydrography,%20Nutrients,%20Phytoplankton.pdf)
- ICES Report on Ocean Climate (IROC) gives seasonal SST anomalies relative to 1991-2020 for ICES areas including
  Skagerrak/Kattegat — [ICES IROC 2021 (ResearchGate)](https://www.researchgate.net/publication/379841301_ICES_REPORT_ON_OCEAN_CLIMATE_2021) (not read in full).

### Inferences
- For a quick monthly table, the SHARK-derived Å16/Å17 table above is adequate and observational. A reanalysis
  (BAL 003_011, monthly means, 1993-2023, at 58.15 N, 10.5-11.2 E) would give a spatially exact, time-continuous
  monthly climatology and consistent MLD; worth doing if the multiplier turns out sensitive to +/-0.5 C.
- Typical copernicusmarine call (dataset IDs must be looked up in the product page; not verified here):
  `copernicusmarine subset --dataset-id <monthly BAL MY phy dataset> --variable thetao -x 10.5 -X 11.2 -y 58.1 -Y 58.2 -z 0 -Z 60`.

### Gaps
- Exact monthly-dataset IDs inside the two Copernicus products (e.g. "cmems_mod_bal_phy_my_P1M-m") were not
  verified; check the product page "Data access" tab.
- ICES oceanographic database (ocean.ices.dk) and SMHI "Havsytans temperatur" products were not queried.

---

## 5. Recent warming trend (degrees per decade) and choice of reference period

### Takeaway
At the Å-transect the trend is ~+0.25-0.3 C/decade over 1964-2024 (Å16+Å17 combined) and
~+0.2-0.3 C/decade since the mid-1990s (statistically significant at 10-50 m at Å16 over
1995-2023, not at the surface); the Kattegat (Anholt E) warms faster, ~+0.45-0.5 C/decade over 1992-2024.
2011-2024 is ~0.2 C warmer on annual mean than 1991-2020. Use 1991-2020 (SMHI/WMO normal) as the reference, or
2011-2024 to represent "present day"; the difference (~0.2 C) is small compared with the 1-2.7 C interannual SD.
Confidence: moderate.

### Cited Findings
Linear trend of annual-mean anomalies (years with >= 8 sampled months; anomalies vs 1991-2020 monthly means;
+/- = 95% CI) — computed from [SHARK](https://shark.smhi.se/api/docs):

| Station | Depth | Years (n) | Trend C/decade |
|---|---|---|---|
| Å16 | 0-2 m | 1995-2023 (28) | +0.26 +/- 0.27 |
| Å16 | 10 m | 1995-2023 (28) | +0.30 +/- 0.23 |
| Å16 | 30 m | 1995-2023 (28) | +0.33 +/- 0.20 |
| Å16 | 50 m | 1995-2023 (28) | +0.34 +/- 0.20 |
| Å17 | 0-2 m | 2000-2024 (25) | +0.11 +/- 0.35 |
| Å17 | 30 m | 2000-2024 (25) | +0.25 +/- 0.26 |
| Å17 | 50 m | 2000-2024 (25) | +0.22 +/- 0.20 |
| Anholt E | 0-2 m | 1992-2024 (33) | +0.47 +/- 0.21 |
| Anholt E | 10 m | 1992-2024 (33) | +0.52 +/- 0.15 |
| Anholt E | 30 m | 1992-2024 (33) | +0.45 +/- 0.15 |
| Anholt E | 50 m | 1993-2024 (32) | +0.44 +/- 0.14 |

- Skagerrak surface-layer temperature in 2000-2009 was 0.7-1.3 C higher than the 1961-1990 reference; ~0.8 C
  higher at 200 m (0.6-1.1 C by station) — ["Climatic variability in the Skagerrak and coastal waters of Norway", ICES J. Mar. Sci. 69(5):758, 2012 (author names not verified in fetch)](https://academic.oup.com/icesjms/article/69/5/758/646533)
- Annual means site (Å16+Å17): 1991-2020 vs 2011-2024 = 10.3 vs 10.5 C (surface), 9.8 vs 10.0 (10 m),
  8.8 vs 9.1 (30 m), 8.4 vs 8.6 (50 m) — computed from [SHARK](https://shark.smhi.se/api/docs)

### Inferences
- The published step (+0.7-1.3 C from 1961-1990 to 2000-2009) implies that pre-1990 climatologies
  (e.g. older Gullmar or Torungen tables) are ~1 C too cold for present conditions; avoid them.
- Projected forward, +0.25 C/decade means ~+0.3 C between the 1991-2020 centre (2005) and 2020, i.e. the
  2011-2024 table; for a model of today, the 2011-2024 table is the slightly better match.

- Long series at Å16+Å17 (bounding-box query, years with >= 6 sampled months, anomalies vs 1991-2020 monthly
  means) — computed from [SHARK](https://shark.smhi.se/api/docs). Decadal mean anomaly (C), surface / 10 m /
  30 m / 50 m: 1960s -1.15/-0.91/-1.08/-0.90; 1970s -0.85/-0.63/-1.24/-0.95; 1980s -1.42/-1.27/-0.48/-0.50;
  1990s -0.56/-0.42/-0.82/-0.69; 2000s +0.27/+0.14/-0.01/0.00; 2010s -0.07/-0.07/+0.14/+0.14;
  2020-2024 +0.42/+0.52/+0.35/+0.32 (5-10 years per decade; 1960s-80s years are few and unevenly sampled).
  Linear trend 1964-2024 (n = 50-51 years): +0.32 (surface), +0.26 (10 m), +0.29 (30 m), +0.24 (50 m) C/decade.

### Gaps
- A 2025 Continental Shelf Research paper "Warming temperatures and prevalence of marine heatwaves in a
  temperate fjord" ([ScienceDirect S0278434325002031](https://www.sciencedirect.com/science/article/pii/S0278434325002031))
  may concern the Gullmar fjord; access was blocked (HTTP 403), so neither its site nor numbers are verified.
- No Kristineberg long-term (century) series with a per-decade trend was retrieved as text.
- Trend confidence intervals ignore autocorrelation; true uncertainty is somewhat larger.
