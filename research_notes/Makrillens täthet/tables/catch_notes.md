# Mackerel and horse mackerel catches in ICES 3.a (Skagerrak 3.a.20, Kattegat 3.a.21)

Tags: [F] full data/text read, [A] abstract only, [S] secondary, [D] computed here, J judgement.
All downloads 2026-10-30; sources and URLs in `../data/catch/README.md`. Scripts in
`../scripts/catch_*.py` (run with `python3 -I`).

## 0. Headline

- Commercial mackerel catch in all of 3.a is small: about 500-550 t/yr (ICES official landings,
  all countries: mean 2015-2024 573.5 t, 2016-2024 554.5 t, 2020-2024 497.7 t) [D from F].
  WGWIDE "WG catch" for 3.a: 480 t in 2023 (WGWIDE 2024, Table 8.3.3) and 486 t in 2025
  (WGWIDE 2026, Table 8.3.3) [F].
- Norway is NOT the largest 3.a mackerel fisher. 2015-2024 means (ICES official): Sweden
  235.0 t, Denmark 223.4 t, Norway 113.5 t [D from F]. Norway's 3.a catch is not purse seine:
  it is trolling/handline ("Dorg/harp/snik") from vessels under 11 m, plus drift and set gillnets
  [D from F, Norwegian landing notes]. Purse seine: 0-13 t/yr.
- The Norwegian Skagerrak catch is spread over May-September (2012-2025 pooled: Q2 46.7 %,
  Q3 46.7 %, Q4 6.2 %), not an autumn peak. A late-summer peak existed only in 2010-2011
  (1236 t and 674 t; 87.6 % in August-September by month of last catch date), when 11-21 m
  vessels trolled; it ended in 2012 [D from F].
- Roughly 40 % of the 3.a mackerel catch is in the Kattegat (3.a.21), almost all Danish
  (2016-2024 mean DK 168 t, SE 43 t) [D from F].
- Horse mackerel in 3.a: sporadic, 0-800 t/yr, dominated by single pelagic-trawl or purse-seine
  hauls in Aug-Nov; almost nothing in the Kattegat (3.a.21 mean 3.4 t/yr 2016-2024) [D from F].
- Areas: 3.a.20 = 34 041 km2, 3.a.21 = 20 536 km2 (sea area, land cut out) [D], equal to the
  `Area_km2` attribute of the ICES shapefile [F].

## 1. Sources and how they were read

1. ICES Official Nominal Catches 2006-2024 (`ICESCatchDataset2006-2024.csv`, version log
   15/10/2026 "Added catch data 2024") [F]. Annual landings (TLW = tonnes live weight) by
   species x area x country, flag-state basis; no discards. Rows for MAC, HOM, JAX in
   27.3.a, 27.3.a.20, 27.3.a.21, 27.3.a_NK. DK, SE and DE report at subdivision level from
   2016; before 2016 everything is 27.3.a_NK. NO, NL and FO report 27.3.a_NK (Norway: 3.a.20
   only in 2016).
2. Fiskeridirektoratet open landing-note data (`fangstdata_<year>.csv.zip`, 2010-2026) [F].
   Kept only `Dokumenttype` = "Sluttseddeldokument" (the documentation warns that recent
   quantities can lie twice, on both the final note and the landing note) and only Norwegian
   vessels (`Fartoynasjonalitet` NOR; foreign vessels landing in Norway: DNK 2.0 t, SWE 0.1 t of
   MAC over 2010-2026). 3.a = `Omradegruppering` 27_3_A_NK, which in these files coincides
   with Norwegian main area (`Hovedomrade`) 09 "Skagerrak". Month = month of `Siste
   fangstdato` (last catch date of the trip); no record lacked it for 3.a. Weight = `Rundvekt`.
   Data stamp 30.10.2026, so 2026 is partial.
   Cross-check [D]: Norwegian sales-note MAC in 3.a vs ICES official NO 27.3.a: 2016 158.63 vs
   158.63; 2019 107.72 vs 107.71; 2023 94.86 (NOR vessels) vs 95.1; 2024 107.26 vs 107.26.
   HOM 2016 628.12 vs 628.12, 2018 71.84 vs 71.84. The two sources are the same data.
3. Statistics Denmark FISK2 "Landings by Danish vessels" (annual, by fishing region incl.
   "The Skagerrak" and "The Kattegat", species group "4. Atlantic Mackerel") [F]. Matches the
   ICES DK subdivision rows (2024: Skagerrak 28.9 t vs ICES 29; Kattegat 183.9 vs 184) and
   adds 2025: Skagerrak 88.1 t, Kattegat 153.5 t.
4. WGWIDE 2024, 2025, 2026 reports (mackerel chapter 8, western horse mackerel chapter 7,
   North Sea horse mackerel chapter 6) [F].
5. HaV JO 55 SM 2501 (Swedish definitive 2024 statistics) [F]: Table 1, mackerel
   "Skagerrak och Kattegatt" 188 t live weight in 2024 (ICES SE 27.3.a 187.7 t). Annual only.
6. ICES areas shapefile `ICES_Areas_20160601_cut_dense_3857` [F] for the areas.

## 2. Annual catches (tonnes)

### Mackerel, ICES official landings, by country and subdivision, 2016-2024 [F]

| Area / country | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | mean 16-24 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.a.20 SE | 336.7 | 243.5 | 172.0 | 183.3 | 153.9 | 137.4 | 144.8 | 117.2 | 145.1 | 181.5 |
| 3.a.20 DK | 39.2 | 153.0 | 33.0 | 23.0 | 52.0 | 55.0 | 78.0 | 32.0 | 29.0 | 54.9 |
| 3.a NO (all NK except 2016) | 158.6 | 124.4 | 99.2 | 107.7 | 89.5 | 74.2 | 91.8 | 95.1 | 107.3 | 105.3 |
| 3.a.20 DE | 0.6 | 0.2 | 2.0 | 0 | 0 | 0 | 2.0 | 0 | 0 | 0.5 |
| 3.a NK NL+FO | 3.2 | 4.4 | 2.2 | 0 | 0 | 0 | 0 | 0 | 0 | 1.1 |
| 3.a.21 DK | 158.0 | 140.0 | 193.0 | 118.0 | 184.0 | 133.0 | 206.0 | 197.0 | 184.0 | 168.1 |
| 3.a.21 SE | 70.2 | 63.4 | 32.7 | 40.4 | 44.0 | 32.3 | 27.3 | 34.0 | 42.6 | 43.0 |
| 3.a total | 766.5 | 729.0 | 534.1 | 472.4 | 523.4 | 432.0 | 549.9 | 475.3 | 508.0 | 554.5 |

Derived split (`catch_3a_subdiv_estimate.csv`) [D], assigning all 27.3.a_NK (Norway, NL, FO)
to 3.a.20 (J: Norway borders only the Skagerrak and its 3.a catch is landing-note main area 09
"Skagerrak"; NL/FO amounts are 0-4 t): mackerel 3.a.20 mean 2016-2024 = 343.4 t/yr,
3.a.21 = 211.1 t/yr; share in the Kattegat 0.28-0.49 by year, rising over the period
(0.30 in 2016, 0.45 in 2024).

Per km2 [D]: 3.a.20 343.4 t / 34 041 km2 = 0.0101 t/km2/yr; 3.a.21 211.1 t / 20 536 km2 =
0.0103 t/km2/yr.

Longer series [F]: 3.a total (ICES) 2006-2015, all as 27.3.a_NK: NO 1059, 1329, 625.4, 1406,
1236.25, 673.96, 172.78, 140.03, 135.13, 187.36 t (2006-2015), i.e. Norway was the largest
3.a mackerel fisher until 2011 and dropped to 100-190 t from 2012. SE 216.9-406.9 t and DK
43-336 t over 2006-2024 (full rows in `catch_3a_by_country_year.csv`).

2025 (ICES official not yet published) [F]: NO 66.95 t (sales notes), DK Skagerrak 88.1 t and
Kattegat 153.5 t (DST FISK2); WGWIDE WG catch 3.a = 486 t (all countries). Sweden 2025 by
difference: 486 - 67.0 - 241.6 = about 177 t [D; indicative only, the WG figure may include
discards and other countries]. Norway 2026 to 30 Oct: 71.15 t [F].

Means per source and window: `catch_3a_means.csv`. Subdivision rows have no 2015-2024 mean
(no subdivision reporting before 2016; a 2015 value of zero would be structural).

### Horse mackerel (HOM; JAX negligible, <0.1 t/yr) [F]

| Area / country | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| 3.a.20 DK | 16.3 | 18.0 | 286.0 | 166.0 | 9.0 | 7.0 | 186.0 | 4.0 | 4.0 |
| 3.a.20 SE | 116.8 | 21.3 | 9.9 | 355.6 | 59.4 | 3.9 | 1.4 | 0.9 | 0.7 |
| 3.a NO | 628.1 | 285.6 | 71.8 | 0.5 | 0 | 0 | 0.1 | 19.3 | 1.3 |
| 3.a NK FO+NL | 42.5 | 34.0 | 0.6 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3.a.21 DK+SE | 3.2 | 11.1 | 5.6 | 1.2 | 4.2 | 1.8 | 3.0 | 0.1 | 0.2 |
| 3.a total | 807.0 | 370.0 | 373.9 | 523.3 | 72.6 | 12.6 | 190.5 | 24.2 | 6.2 |

Means [D]: 3.a total 245.1 t (2015-2024), 264.5 t (2016-2024), 61.2 t (2020-2024); 3.a.21
3.4 t (2016-2024). Norway had 4088.21 t in 2014 (one November purse-seine catch, sales notes)
[F]. Highly episodic: J, a mean is a poor descriptor; the median year is tens of tonnes.

WGWIDE [F]: since the 2024 benchmark, horse mackerel caught in 3.a (and 4.a) in all quarters
belongs to the western stock (WGWIDE 2026 section 7.11: "catches in Divisions 3.a and 4.a are
now considered to belong to western horse mackerel, independently of the quarter when they are
caught (ICES, 2024a)"; same in chapter 6 text). Western stock 3.a catch: 2023 25 t, 2024 25 t,
2025 55 t (Table 7.2.1.1 of WGWIDE 2024, 2025, 2026). Note: the WG 2024 figure (25 t) does not
match ICES official 2024 (6.2 t); not resolved.

## 3. Seasonality

### Norwegian Skagerrak (main area 09 = 27.3.a), mackerel, share of catch by month of last
catch date [D from F] (`catch_no_3a_month_profile.csv`)

| Window | t/yr | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010-2011 | 955 | 0.002 | 0.030 | 0.034 | 0.028 | 0.484 | 0.392 | 0.028 | 0.000 | 0.000 | 0.065 | 0.904 | 0.030 |
| 2012-2025 | 118 | 0.049 | 0.206 | 0.213 | 0.203 | 0.165 | 0.099 | 0.049 | 0.007 | 0.004 | 0.467 | 0.467 | 0.062 |
| 2015-2024 | 113 | 0.045 | 0.181 | 0.230 | 0.187 | 0.157 | 0.120 | 0.068 | 0.003 | 0.005 | 0.455 | 0.464 | 0.076 |
| 2020-2025 | 87 | 0.037 | 0.137 | 0.244 | 0.206 | 0.190 | 0.151 | 0.022 | 0.006 | 0.003 | 0.419 | 0.547 | 0.031 |

(Jan-Mar and Dec each <= 0.3 %.) In 2010-2011 the peak was in August (48 %) and September
(39 %); the yearly view
(`catch_3a_by_month_or_quarter.csv`) shows 2010: Aug 669 t, Sep 461 t; 2011: Aug 255 t,
Sep 288 t, Oct 51 t. Since 2012 no year repeats the 2010-2011 August-September
concentration. Largest month by year: May (2012, 2013, 2018), June (2016, 2017, 2019, 2020,
2022, 2023), July (2014, 2024, 2026 partial), August (2021, 2025), October only in 2015
(49.7 t, 27 % of 186 t). The 2020-2025 profile sits later (June-September, Q3 54.7 %,
September 15.1 %) than 2012-2019; October is 2-7 % in every window.

Gear, Norwegian 3.a mackerel [D from F] (`catch_no_3a_by_gear_year.csv`): trolling/handline
("Dorg/harp/snik") 1184.6 t in 2010, 626.3 t in 2011, 44-112 t/yr 2012-2025; drift gillnet
("Drivgarn") 6-74 t/yr, falling from ~50 t (2012-2016) to 6-11 t (2023-2025); set gillnet
("Settegarn") 4-20 t/yr; purse seine ("Snurpenot/ringnot") 0-12 t/yr; pelagic trawl
("Flytetral") 20.2 t (2017) and 18.3 t (2023) only. Vessels (from the `length_group` column of the `no_mac_hom_<year>.csv` extracts): in 2010
most catch came from
11-14.99 m (564 t) and 15-20.99 m (483 t) vessels; from 2012 mostly from vessels under 11 m
(58-136 t/yr). J: the Norwegian Skagerrak mackerel fishery is a small-boat coastal summer
fishery, so its monthly profile follows when mackerel are close to the Norwegian coast, not
an offshore fishery.

### Horse mackerel seasonality [F]
- Norwegian 3.a HOM by catch month (sales notes, NOR vessels): every non-trivial catch is in
  Aug-Nov: 2010 Oct 57.5 t (purse seine); 2013 Oct 10.0 t (trawl); 2014 Nov 4088 t (purse
  seine); 2015 Oct 53.9 t (trawl); 2016 Aug 235.7 t, Sep 31.4 t, Nov 359.6 t (trawl); 2017 Aug
  184.7 t, Sep 100.3 t (trawl); 2018 Nov 71.8 t (purse seine); 2023 Sep 19.3 t (trawl).
  Foreign vessels landing in Norway (excluded): DNK trawl Nov 2018 242.7 t, SWE trawl Oct
  2020 56.7 t, FRO trawl Aug 2016 12.4 t.
- WGWIDE western horse mackerel, division 3.a, quarterly (all countries, Table 7.2.1.1):
  2023 Q1 0, Q2 0, Q3 22, Q4 3 (total 25); 2024 1, 0, 13, 11 (25); 2025 0, 1, 11, 43 (55).
  i.e. a Q3-Q4 fishery.

### Mackerel seasonality for the other countries
- EU quarterly data for 3.a.20 are in the caller's local STECF FDI Table A files (not analysed
  here). The FDI files there cover 3.a.20 only; the Kattegat (3.a.21), ~40 % of the 3.a
  mackerel catch and mostly Danish, has no seasonal data in this folder. FDI Table A can be
  pulled for sub_region 27.3.A.21 from the same source used for 3.a.20.
- WGWIDE has no 3.a-specific quarterly mackerel table; Table 8.2.3.1 is stock-wide
  (2025: Q1 24 %, Q2 6 %, Q3 52 %, Q4 18 %) [F] and does not apply to 3.a.

## 4. Areas [D, F]

Computed with `catch_ices_area_km2.py` from `ICES_Areas_20160601_cut_dense_3857` (ICES GIS,
land cut out, so sea area): vertices back to WGS84, then shoelace on the cylindrical
equal-area projection of the WGS84 authalic sphere.
- 27.3.a.20 (Skagerrak): 34 041 km2 [D]; shapefile `Area_km2` = 3.40413688141e+004 [F].
- 27.3.a.21 (Kattegat): 20 536 km2 [D]; shapefile `Area_km2` = 2.05359845460e+004 [F].
- Sum 3.a = 54 577 km2 [D].

## 5. What was not found

- No monthly or quarterly mackerel catch for 3.a as a whole or for the Kattegat (3.a.21) for
  any country. Danish (DST FISK2) and Swedish (HaV JO 55 SM) official statistics are annual;
  the Swedish monthly reports (JO 50 SM) are by coast of landing, not by ICES area (not
  downloaded).
- WGWIDE 2025 (2024 data) Table 8.3.3 has no 3.a row (it jumps from 2.b to 3.b), so the WG
  catch for 3.a in 2024 was not found; ICES official 2024 = 508.0 t.
- ICES official catches for 2025 are not yet published (dataset ends 2024).
- No recreational catch estimate for mackerel in the Skagerrak/Kattegat (Norwegian or Swedish)
  was found. J: commercial 3.a catch is only ~500 t/yr, so recreational removals by rod and
  line along the Norwegian and Swedish Skagerrak coasts could be of a comparable order; this
  matters if catch is used to infer density. Possible sources: HaV JO 57 SM "Fritidsfisket i
  Sverige"; Norwegian IMR recreational surveys (not retrieved).
- No discard figures for Norway; ICES official data are landings only.
