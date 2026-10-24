# Acoustic biomass of planktivorous fish in the Mareld box

Purpose: find the acoustic level of herring, sprat and the other planktivores in the Mareld box
(57.7-58.4 N, 9.9-11.3 E; parts of ICES rectangles 44G0, 44G1, 45G0, 45G1; 3600 km2). It is
compared with the NS-IBTS GOV bottom-trawl floor of 1.3 t/km2 (`../Fiskgrupper i rutan/fish_guilds.md`)
and with the model's pelagic_fish start of 3.3 t/km2. No code or config was changed.

Tags: [F] full text read, or data downloaded and processed; [A] abstract or snippet only;
[S] secondary source; [D] computed here.


> **Correction (main agent, 2026-10-24): the box above is too large.** The bounds in the brief
> (57.7-58.4 N, 9.9-11.3 E, ~6400 km2) were the depth-map download bbox, not the model grid. The grid
> is SWEREF 99 TM E 211-271 km, N 6414-6474 km (3600 km2; corners 57.774 N 10.138 E, 57.810 N
> 11.146 E, 58.312 N 10.066 E, 58.346 N 11.086 E). `scripts/estimate_box.py --grid` selects the
> NASC logs inside it (hauls for allocation unchanged); outputs `tables/*_grid.csv`, summary
> `tables/acoustic_box_summary_grid.csv`. 37-117 logs (nmi) per year in the grid. [D]
>
> | July, grid (2010-26, 17 years) | Mean | Median | Geomean | Min-max |
> |---|---|---|---|---|
> | Herring | 4.3 | 3.2 | 3.4 | 1.0-10.2 |
> | Sprat | 0.16 | 0.10 | 0.11 | 0-0.7 |
> | Planktivores | 4.7 | 3.3 | 3.7 | 1.2-11.2 |
> | Planktivores 2021-25 | 4.6 | 3.6 | 3.8 | 1.3-8.9 |
>
> The margin outside the grid (the dense strip off Skagen and the Swedish coast) held most of the
> sprat and the 2015, 2020 and 2024 peaks (2020: 30.4 -> 3.3; 2024: 11.2 -> 1.3, closer to the
> official stratum-31 0.47). The grid level matches the official stratum-31 mean for 2016-25
> (3.9 t/km2). Trend +2 %/yr (se 3.5 %), not significant. With the section-4 summer-to-annual
> factor 0.5-0.8 (central 0.6) the annual mean is ~2-3 t/km2 (central ~2.5). **The model's 3.3
> t/km2 sits between the July level and the annual mean: of the right order, not too low.** The
> sections below still describe the oversized box.

## Bottom line

HERAS summer survey (RV Dana, about 25 June - 10 July), box mean, t/km2. The years are 2010-2026
(17 cruises). The values come from our re-estimate of the ICES acoustic database data, which
reproduces the official stratum 31 estimates (section 2).

| Species | Acoustic, central (geomean of years) | Median of years | Range of years | Years | GOV Q1+Q3 2021-25 | Note |
|---|---|---|---|---|---|---|
| Herring (NSAS + WBSS) | **6.0** | 6.4 | 2.0-22.6 | 2010-26 | 0.90 (Q1 0.59, Q3 1.19) | 2021-25 mean 6.5 (4.2-9.5) |
| Sprat | **0.35** | 0.32 | 0.0-6.3 | 2010-26 | 0.19 (Q1 0.03, Q3 0.36) | about 0 in 2016-23; 1.7-2.7 in 2024-26 |
| Norway pout | **0.3** | 0.3 | 0.2-1.4 | 2016-20 | 0.20 | after 2020 Dana no longer reports the non-clupeid backscatter |
| Pearlside (Maurolicus) | < 0.1 (max 0.2) | < 0.01 | 0-0.2 | 2016-20 | 0.0001 | only the upper 150 m was integrated |
| **Planktivores (sum)** | **6.9** | 6.9 | 2.3-30 | 2010-26 | 1.3 | herring 87 %, sprat 5-10 %, pout 4 % |
| Blue whiting (borderline) | 0.15 | 0.1 | 0-1.3 | 2016-20 | 0.52 (Q3 1.03) | missed acoustically: deep, below 150 m, near the bottom |
| Mackerel (borderline) | not estimable | - | - | - | 0.15 (Q3 0.29) | no swimbladder; IESSNS surface trawl in July gives about 1.9 (0.2-4.4) |

**Recommendation for the planktivore group (herring + sprat + Norway pout + mesopelagics):**
- **Summer (July) level: 7 t/km2, range 2.5-16.** The range is the 10-90 % spread between years.
- **Annual mean: about 5 t/km2, range 2.5-10.** This assumes winter is about 0.5-0.7 of summer
  (section 4). That factor is the main unverified assumption. GOV probably catches herring better
  in winter, so its Q1/Q3 ratio of 0.5 is more likely an upper bound than a central value. That
  points to the lower half: about 0.6 x summer, or 4-5 t/km2.
- **The model's start month decides which value applies.** A July start takes 7 t/km2. A winter
  or spring start takes 4-5 t/km2.
- Composition: herring 85-90 % (mostly juvenile NSAS 1-2 ringers plus WBSS feeders), sprat 5-10 %
  (very variable), Norway pout about 5 %, mesopelagics about 1 %.
- Keep mackerel (a summer visitor, about 2 t/km2 in July) and blue whiting (0.5-2) out of the sum.
  If they are counted, summer total planktivory rises to about 9-10 t/km2.
- **The model's 3.3 t/km2 is about half the acoustic summer level, and at the low end of the
  annual-mean range.** GOV 1.3 is a floor: acoustics are 3-7 times higher in every year.

## 1. Data and method

**Source [F].** The ICES acoustic-trawl database (acoustic.ices.dk) gives free CSV downloads with no
login. We used:
- per-log NASC (m2/nmi2) by species category and depth channel;
- trawl hauls with catch by species and individual lengths and weights.

**Coverage.** Only Denmark's RV Dana covers the Skagerrak in HERAS. There is no Swedish HERAS
submission: Sweden's RV Argos ran this area in 1984-1990 only [F, PGHERS 1996 via
`evidence/heras_methods_allocation_ts.txt`]. Norway's cruises (58J3) stay west of 7 E.
- Dana data in the database cover 2010-2026.
- 107-199 nmi of transect fall inside the box each year, in 5-9 transect segments.
- 6-16 trawl hauls fall within about 25 km of the box each year.
- Biotic data are missing for 2010-2015 and 2026. Those years borrow the pooled 2016-2020 hauls
  (2010-15) and the 2025 hauls (2026).

**Categories [F].** HER and SPR are pure herring and pure sprat. CLU is a clupeid mix (herring, sprat,
sardine, anchovy). MIX is clupeids plus other fish (HERAS 2023 report p.4-5, `evidence/heras_methods_allocation_ts.txt`).
- 2010-2020 has only MIX.
- 2021-24 has HER, CLU and some MIX.
- 2025-26 has HER, SPR and ANE.

**Estimator [D].** `scripts/estimate_box.py` is a simplified StoX-style echo-integration estimate.
- The box mean NASC is weighted by the distance of each log.
- The mixed categories are split by trawl composition: the haul-equal mean of each species' share
  of backscatter, n_i sigma_i / sum n_j sigma_j.
- The HERAS TS relations are used [F]: clupeids 20logL - 71.2; gadoids, Norway pout, horse mackerel
  and pearlside 20logL - 67.5; mackerel 20logL - 84.9.
- Mackerel is left out of the MIX split. With its very low TS, a few mackerel in the catch would
  turn into 20-150 t/km2 of mackerel (`tables/acoustic_box_by_year_withmac.csv`).
- The bootstrap resamples transect segments and hauls (`tables/acoustic_box_by_year.csv`, columns p05/p95).

Run `bash scripts/run_all.sh` to reproduce everything.

## 2. Validation against the official HERAS estimates

- The international StoX estimates by stratum were read from the HERAS reports [F]
  (`evidence/heras_per_stratum_3a_2015_2025.txt`).
- Stratum 31 (eastern Skagerrak, 6195 km2) covers about 60-65 % of the box. The rest of the box is
  in 41 (Norwegian Trench), 42 (off Jutland) and a sliver of 21 [F/D, strata map in `evidence/heras_strata_areas.txt`].
- We ran our estimator on the stratum 31 polygon (`scripts/stratum31_check.py`,
  `tables/stratum31_validation.csv`) [D]:

| Year | Ours, stratum 31 (t/km2) | Official, stratum 31 | Ratio | Box (ours) | Box / official 31 |
|---|---|---|---|---|---|
| 2016 | 3.6 | 3.9 | 0.93 | 5.9 | 1.5 |
| 2017 | 1.7 | 1.9 | 0.89 | 2.6 | 1.4 |
| 2018 | 2.7 | 4.3 | 0.63 | 2.3 | 0.5 |
| 2019 | 8.5 | 8.7 | 0.98 | 12.5 | 1.4 |
| 2021 | 3.1 | 2.8 | 1.09 | 4.2 | 1.5 |
| 2022 | 4.7 | 4.8 | 0.97 | 6.8 | 1.4 |
| 2023 | 4.1 | 3.2 (3.7 with the area cut) | 1.29 (1.1) | 5.8 | 1.8 |
| 2024 | 3.4 | 0.47 | 7.2 | 9.5 | 20 |
| 2025 | 5.3 | 4.7 | 1.14 | 6.4 | 1.4 |

- **In 8 of 9 years our method lands within 0.6-1.3 of the official stratum 31 value (median 0.97).**
- 2024 is the exception. The official stratum 31 value is 2.9 kt (39 million fish). In 2024, 80 % of
  the stratum 31 NASC is CLU (55 of 69 m2/nmi2) [D]. Stratum 31 NASC is the lowest of the series,
  but our trawl split gives most of the CLU to herring. The official run reports neither herring nor
  sprat for that backscatter. So the disagreement is in how the CLU category was split, not in the
  acoustics. The year is flagged.
- The official stratum 31 mean for 2016-25 is 3.9 t/km2 (median 3.9) [F/D]. Strata 41 and 42 average
  3.6 and 4.4.
- **Geometry caveat.** The lat/lon bounds used here (57.7-58.4 N, 9.9-11.3 E) enclose about 6400
  km2, not the 3600 km2 of the 60 x 60 km grid. Densities are transect means, so the bounding box is
  only a sampling window. But the box's sprat and part of its herring excess over stratum 31 come
  from the south-east margin: the 44G0/44G1 strip near Skagen and the Swedish coast, and the Kattegat
  sliver. If the real grid leaves those margins out, box sprat falls toward the official 0 and
  herring toward the stratum 31 level of 4-5 t/km2.
- **The box is denser than stratum 31 (median ratio 1.4).** The box holds the dense rectangles
  44G0 and 44G1 (Skagen and the Swedish side). It also lacks the Oslofjord mouth (58.5-59.1 N), where
  densities are low. Box NASC is a median 1.6 times the Dana-covered IIIa mean
  (`tables/nasc_box_vs_IIIa.csv`) [D].
- Danish national estimates for the old stratum D, which holds about 90 % of the box, were 1.5-14.2
  t/km2 (2004-2015, mean 7.1) [F, `evidence/dana_cruise_reports_2004_2019.txt`]. That agrees with
  our 2010-15 values of 2.0-13.0.
- Argos found 13 t/km2 of herring in the eastern Skagerrak in August 1989 (110 kt over 8345 km2)
  [F, `evidence/argos_1989_skagerrak_kattegat.txt`].
- Sensitivities [D]: using pelagic (FOT) hauls only raises herring by 0-60 %. Using hauls inside
  the box only changes it by -15 to +55 %. The central estimate uses all hauls within 25 km.

## 3. The other planktivores and the borderline species

- **Sprat.** In July, sprat in IIIa sits in the Kattegat (stratum 21, 1.5-45 kt). Officially it
  is 0 in strata 31 and 41 in 7 of 9 years [F]. In the box, Dana's own allocation gives under
  0.5 t/km2 in most years. It gives 1.7-2.7 t/km2 in 2024-26 (SPR backscatter in 44G0/44G1) and 6.3
  in 2020 [D]. GOV Q3 gives 0.36 and Q1 0.03 [D, guild note]. A central value of 0.3-0.5 t/km2 is a
  fair choice, with large swings between years. Most of the box sprat lies in the south-east margin
  of the bounding box, which may fall outside the real grid (section 2, geometry caveat). In October the northern Kattegat edge just south of
  the box (44G0/44G1, SD21 parts) held 0-14 t/km2 of sprat in 2001-2007 [F, German Solea survey,
  `evidence/seasonality_herring_sprat_3a.txt`]. Sprat may be more abundant there in autumn than in July.
- **Norway pout.** Acoustic 0.2-1.4 t/km2 in 2016-20 (median 0.3), from the MIX split [D]. GOV gives
  0.20 [D]. Argos 1989 gave 0.06 [F]. A rough stock-level density of 0.3-0.5 t/km2 is derived from
  the North Sea + IIIa SSB of 78 kt in 2023 [F/D, `evidence/other_planktivores_skagerrak.txt`]. Pout
  lives near the bottom, where GOV samples it well, so **about 0.2-0.3 t/km2** is consistent across methods.
- **Mesopelagics (Maurolicus, Benthosema).** Dana integrated only to about 150 m before 2021, and
  trawl catches of pearlside are small. The acoustic share is at most 0.2 t/km2 (2018) and usually
  under 0.01 [D]. Gjosaeter (1986) gave a tentative 0.02-1.6 Mt of Maurolicus for the Norwegian Deep
  and Skagerrak, with no area [A]. No Trench g/m2 value was found. In the box, mesopelagics are
  probably under 0.5 t/km2, but this is an open gap.
- **Mackerel.** The acoustic method cannot estimate it. In the DK IESSNS surface trawl in early July
  2018-2026 (`scripts/iessns_swept.py`), the hauls in and next to the box give 0.2-4.4 t/km2, median
  1.9 [D]. This assumes catchability 1 and a 65 m horizontal opening [assumption]. The IESSNS survey
  mean is about 1.7 t/km2 (2022), and the Skagerrak stratum mean was 1.0-2.4 in 2018-22 [F]. Argos 1989
  gave 1.9 [F]. GOV gives 0.15. Mackerel is a summer visitor that leaves in autumn.
- **Blue whiting.** Acoustic 0-1.3 t/km2, median 0.1 [D]. This misses fish below 150 m. GOV Q3 gives
  1.03 and Q1 almost 0 [D]. Argos 1989 gave 2.1, and about 3.2 in the 60-300 m strata [F]. A
  realistic summer value is 0.5-2 t/km2, in the deep Trench part of the box.
- The same surface hauls also show herring at 0-33 t/km2 in the top 30 m [D]. This shows how patchy
  herring is.

## 4. Seasonality and vertical coverage

**Seasonality.**
- HERAS samples late June to early July. GOV samples Q1 (February) and Q3 (August-September).
- WBSS adults leave the western Baltic in Q2 to feed in the Kattegat and Skagerrak. Older fish go on
  to the eastern North Sea. "Towards the end of summer, herring aggregates in the eastern Skagerrak
  and Kattegat" before it winters in the southern Kattegat, the Sound and the western Baltic
  [F, her.27.20-24 stock annex 2026 p.6].
- NSAS juveniles (0-2 ringers) live in IIIa all year [F, NSAS stock annex; HERAS 2023 p.8].
- In stratum 31, WBSS made up 7-67 % of the July herring biomass (median about 24 %) [F/D]. That part
  leaves for the winter.
- GOV herring in Q1 is half of Q3 (0.59 vs 1.19 t/km2) [D]. Sprat in Q1 is a tenth of Q3.
- **So July is near the annual high.** An annual mean of 0.5-0.8 times July is reasonable. We use
  about 0.7, which gives 5 t/km2. No published summer-to-winter fraction for the box was found (gap).

**Vertical coverage.**
- Dana's transducer runs at 4-6 m, and integration starts at 7-10 m. Herring in the top 7-10 m are
  missed. HERAS 2023 flagged near-surface herring and possible underestimation [F].
- The bottom backstep is 1 m, so the dead zone is about 1-2 m at the box's depths [F/D].
- Before 2021, Dana excluded layers below 150 m [F, Dana reports]. In the box, the 100-150 m layer
  carried only 1-5 % of the NASC in 2010-20 (`tables/nasc_depth_profile_box.csv`) [D]. The data from
  2021 on, integrated to 500 m, give similar herring levels. The 150 m cut-off therefore matters little
  for herring, but it does matter for blue whiting and mesopelagics.
- GOV fishes the bottom 4-5 m with a small effective catchability for pelagic schools.
- **Do not sum the two.** Acoustics already cover 7-10 m down to 1-2 m above the bottom. GOV overlaps
  that range for most of its opening. The truly unseen near-bottom layer is only a part of the GOV
  window. A generous add-on is about 0.4 x GOV herring, about 0.4 t/km2, which is below 10 % of the
  acoustic level.
- **Take the acoustic estimate as central and GOV as the floor.** The acoustic box herring is higher
  than GOV herring in every year from 2010 to 2025 (GOV 0.2-2.0 t/km2, using 0.00704 t/km2 per kg/h).
  The log-scale correlation between the two series is only 0.23 (`tables/gov_vs_acoustic_herring.csv`) [D].

## 5. Trend

- Box herring rose by about 5 % per year over 2010-26 (log-linear slope 0.051 +- 0.034 SE), which is
  not significant [D].
- Period means: 2010-14 4.7, 2015-19 7.3, 2020-26 9.9 (median 6.8). 2020 is an outlier at 22.6.
- Official stratum 31 shows no trend over 2016-25 [F/D].

## 6. Uncertainties

- Box sampling: a bootstrap over transect segments and hauls gives 5-95 % intervals of typically x0.4
  to x2.5 per year (wider in 2020 and 2024). This is the main error in a single year.
- Species split: 2010-15 borrow the 2016-20 hauls. The 2021+ categories leave non-clupeids out, so
  pout and blue whiting are absent after 2020.
- The 2024 discrepancy with the official value is unresolved.
- Absolute level: TS -71.2 is the HERAS convention. A 2 dB error moves biomass by 1.6 times. HAWG
  treats HERAS as a relative index.
- Season: we have no direct winter acoustics for the box.
- Gaps: the HERAS 2020 official stratum values (the PDF is over 50 MB); a Trench mesopelagic density;
  Totland, Fassler and Simmonds & MacLennan, which were not obtained.

## Files

- `scripts/`: `download.sh`, `list_submissions.py`, `acoustic_box.py`, `biotic_region.py`,
  `estimate_box.py`, `stratum31_check.py`, `iiia_check.py`, `nasc_rectangles.py`,
  `nasc_depth_profile.py`, `iessns_swept.py`, `gov_vs_acoustic.py`, `summarise.py`, `run_all.sh`.
- `tables/`: `acoustic_box_by_year*.csv` (central plus sensitivities), `acoustic_box_species_by_year.csv`,
  `acoustic_box_summary.csv`, `allocation_hauls_summary.csv`, `stratum31_validation.csv`,
  `iiia_summary.csv` / `iiia_rectangles.csv` (a rough IIIa run that uses approximate water fractions),
  `nasc_by_rectangle.csv`, `nasc_box_vs_IIIa.csv`, `nasc_depth_profile_box.csv`,
  `iessns_dk_skagerrak_swept.csv`, `gov_vs_acoustic_herring.csv`, `ices_acoustic_submissions.tsv`.
- `evidence/`: quotes with page numbers from WGIPS/HERAS reports, Dana cruise reports, stock annexes,
  Argos 1989, and the blind-zone literature. `00_index_sources.txt` lists the sources.
- `data/`: the downloads (untrusted; see `data/README.md`) and `data/derived/`.
