# Mackerel and horse mackerel density in the Skagerrak / Division 3.a (2026-10-30)

Question: an annual-mean biomass density (t/km2, wet weight, all sizes) of Atlantic mackerel
(*Scomber scombrus*) and horse mackerel (*Trachurus trachurus*) in ICES 3.a and the Skagerrak,
as an independent check on the model's 0.138 t/km2 (mackerel 0.70 / horse mackerel 0.30). That
value came from NS-IBTS GOV bottom-trawl swept area in the box.

Tags: [F] full text / data read, [A] abstract only, [S] secondary, [D] computed here, J judgement.
No model output enters any number here. Nothing outside this folder was changed. The box is the
project grid, 57.77-58.35 N, 10.07-11.15 E (SWEREF 99 TM E 211-271 km, N 6414-6474 km).

## 1. Answer

**Mackerel**

| | Annual mean t/km2 | Range (envelope) | Summer (Jun-Aug) t/km2 |
|---|---|---|---|
| Model box (3600 km2, Trench slope) | **0.8** | 0.33-1.7 (profile spread alone 0.65-1.0) | **1.8** (0.9-3.0) |
| Skagerrak, 3.a.20 (34 041 km2) | **0.75** | 0.3-1.5 | 1.7 (0.8-2.7) |
| Kattegat, 3.a.21 (20 536 km2) | ~0.15 (J) | 0.02-0.8 | ~0.3 (J) |
| Division 3.a (54 577 km2) | **0.5** | 0.2-1.2 | ~1.2 |

- **Seasonal presence** [D/J, section 2.5]:
  - At the July level in June-August.
  - About 0.8 of July in September and 0.5 in October.
  - May about 0.5, November about 0.2.
  - December-April 0.05-0.15 of July.
  - In total about 5.4 July-equivalent months, a presence fraction of 0.45 (0.36-0.56 over the three profiles).
- **Box tonnage** [D]: 0.8 t/km2 is about 2 900 t as an annual mean, and about 6 500 t in July.
- **Against the current model value**: mackerel is now 0.138 x 0.70 = 0.097 t/km2. The central estimate is about 8 times higher.
- **GOV catchability.** Against the July survey, GOV's implied catchability for mackerel is q ~ 0.09-0.16 [D]. GOV Q3 is 0.16 t/km2 long-run and 0.29 in 2021-25, against 1.8 in July.
  - This is the single number that justifies replacing the GOV value.
  - GOV samples the bottom 4-5 m. Mackerel feed above the 20-30 m summer thermocline.

**Horse mackerel**

| | Annual mean t/km2 | Range | Autumn peak (Sep-Oct) t/km2 |
|---|---|---|---|
| Box / Skagerrak | **0.05** | 0.02-0.16 | 0.17 (0.07-0.4) |
| Division 3.a | 0.03 | 0.01-0.1 | - |

- **Seasonal presence** [D/J]:
  - Present August-November, with the peak in October.
  - 0.5 of peak in August, 0.8 in November, 0.3 in December.
  - Absent February-June.
  - Presence fraction 0.31 (0.24-0.40).
- **GOV is roughly right here.** The GOV-based model value (0.138 x 0.30 = 0.041) sits inside the range.
  - Horse mackerel keep near the bottom by day.
  - The catch and survey routes agree within a factor of 2-3.

**Combined** (mackerel + horse mackerel, box): ~0.86 t/km2 annual mean (0.35-1.85). Mackerel is
0.94 of it and horse mackerel 0.06, against 0.138 now.

## 2. Routes

### 2.1 IESSNS July swept area (primary route for mackerel)

**The survey and its assumptions**
- Denmark (F/V Ceton) has run the IESSNS in the North Sea every July since 2018.
  - Gear: the Multpelt 832 surface trawl, "fishing depth is from surface to about 30 – 35 m".
  - The stratum runs from the 50 m isobath to 59 N in the Skagerrak: "The eastern limit in the
    Skagerrak was set to 11 E".
  - Sources: [F, `data/iessns/iessns_report_2024.txt` App. 1; `dk_iessns_2025_ceton.txt` p. 2].
  - The box is inside the stratum, except its far eastern edge.
- Wingspread follows "WS = 0.3959 * Door spread + 20.094". It "ranged from 66.7 to 72.5 m" in 2025,
  63.5-70.0 m in 2024 and 66-70 m in 2018 [F, DK cruise reports]. I use 68 m.
- Catchability. The survey states: "The swept area method assumes that potential distribution of
  mackerel outside the survey area – both vertically and horizontally – is a constant percentage
  of the total biomass"; "a constant fraction of the mackerel inside the horizontal trawl opening
  are caught" [F, IESSNS 2024 p. 35]. In practice that is q = 1 in the surface layer, and nothing
  below the headline is counted.
- Thermocline. DK 2025: "A pronounced thermocline in the upper 20 to 30 m was found for all stations
  except for one" [F].
  - The survey's own concern is that south of 60 N, without a thermocline, mackerel "may extend
    beyond the layer fished" [F].
  - J: in the stratified Skagerrak July this error is small, and it would make the survey
    underestimate.

**Haul-level density near the box** [D, `scripts/iessns_box.py` -> `tables/iessns_box_stdout.txt`,
`tables/iessns_box_hauls.csv`]
- Input: DK IESSNS hauls 2018-2026 within 57-59 N, 8-12 E.
- The hauls come from ICES acoustic-trawl database biotic files, extracted by another note
  (`../Ekolodsdata planktivorer/tables/iessns_dk_skagerrak_swept.csv`; provenance in that note's
  `data/README.md`).
- Densities are rescaled from a 65 m to a 68 m opening.

| Haul set | n | Median | Geomean | Range |
|---|---|---|---|---|
| Inside the box (83-207 m; 2019-23, 2026) | 6 | **2.01** | 2.45 | 1.69-4.19 |
| Box + edge (<= 15 km) | 7 | 1.97 | 1.69 | 0.18-4.19 |
| Bottom 100-320 m (box depths) | 7 | 1.83 | 1.59 | 0.24-4.19 |
| Trench >= 400 m (west of the box) | 9 | 0.49 | 0.42 | 0.01-2.99 |
| All Skagerrak hauls | 24 | 1.78 | 1.12 | 0.01-6.91 |

- **Deep water is lower.**
  - The deep Trench hauls and the two deepest hauls next to the box (313 m: 0.24; 480 m: 0.18) are low.
  - Depth weighting with `maps/depth.npz` shares: < 250 m 85.5 % at 2.0, 250-300 m 5.9 % at 1.0 (J),
    >= 300 m 8.7 % at 0.35.
  - That gives a box July value of **1.80** [D].
- Horse mackerel in these hauls: max 0.003 t/km2, i.e. absent in early July.

**North Sea stratum (stratum 13; 278 525-285 781 km2)**
- Ceton mean catch, kg/km2 [F]: 2018 1743, 2019 1009, 2020 1318, 2021 2429, 2022 1689, 2023 2362,
  2024 2004, 2025 2060.
  - Sources: IESSNS 2022/2024/2025 App. 1; DK 2025: "average mackerel density was 2060 kg/km2".
  - Mean 1.83 t/km2 over 2018-25, 2.11 over 2021-25 [D].
- StoX stratum biomass, t [F]: 387 613, 223 112, 257 079, 560 198, 472 627, 648 919 (2018-23, IESSNS 2024
  Table A1-2); 556 655 (2024); 586 020 (2025 bootstrap). That is 0.8-2.3 t/km2 [D].
- **2026 (preliminary).**
  - IESSNS total including the North Sea is a bootstrap mean of 3.3 Mt, against 1.8 Mt for the
    Nordic Seas only (Tables 10a, 10b).
  - So the North Sea held about 1.5 Mt, or 5.2 t/km2 [D].
  - The survey says ages 1-3 "had a significantly higher abundance in the North Sea in 2026" [F].
  - The in-box haul in 2026 was only 1.83 t/km2.
  - Kept out of the central value: preliminary, and possibly a recruitment pulse.

**Catchability check against the assessment** [D from F]
- July IESSNS total (Nordic + North Sea) divided by the assessment TSB on 1 January (ICES SG mac.27.nea,
  2026 assessment):
  - 2022: 7.84 / 6.72 Mt = 1.17
  - 2024: 3.06 / 4.07 Mt = 0.75
  - 2025: 2.29 / 3.73 Mt = 0.61
- Inputs: IESSNS 2022 "7.37 million tonnes", 2024 "2.5", 2025 "1.7". The TSB values are from
  `../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`.
- So the survey neither clearly over- nor under-reads the stock. It misses the stock west of the British
  Isles south of 60 N, so a ratio below 1 is expected at q = 1.
- Scaling to the assessment would multiply the density by 0.86-1.63.
- J: q = 1 is kept as the central value. The spread sits inside the July range of 0.9-3.0.

**Older acoustic surveys of the same area** (mackerel is a weak acoustic target, so these give the
order of magnitude only)
- **RV Dana, 31 Jul - 9 Sep 1987** [F tables, D densities; `tables/lit_degnbol1988_density_stdout.txt`]:
  - Skagerrak strata B+C+D: 1.78 t/km2 (A+B+C+D: 1.61).
  - Northern Kattegat: 0.01.
  - Norwegian Trench south slope: 1.81.
- **RV Eldjarn, same weeks** (echogram scrutiny + handlines): 7.1 t/km2 at 57-58 N and 10.7 at 58-59 N.
  - Dana gave ~40 % of Eldjarn where the two overlapped.
  - In 3.a, 92 % of the mackerel were age 1 (Degnbol et al. 1988, ICES CM 1988/H:40).
- **RV Argos, 1-4 Aug 1989**, eastern Skagerrak, 20-300 m [F, `../Ekolodsdata planktivorer/evidence/
  argos_1989_skagerrak_kattegat.txt`]:
  - Mackerel 16 219 t over 8 345 km2 = 1.9 t/km2, at TS 20logL-77.2.
  - The current HERAS mackerel TS (-84.9) would raise this about 6-fold for the same backscatter [D].
- **HERAS today** does not estimate mackerel. The project's own attempt turned a few mackerel in the
  catch into 2-150 t/km2 (`../Ekolodsdata planktivorer/acoustic_planktivores.md`) [S, project note].
  Route 5 is confirmed closed.

**Summer value adopted** [D/J]
- Box: **1.8 t/km2** (0.9-3.0). Basis: depth-weighted in-box hauls; NS stratum mean 1.83;
  Dana 1987 1.6-1.8; Argos 1989 1.9.
- Skagerrak: 1.7 (0.8-2.7). Basis: all-haul median 1.78, pulled towards the low geomean 1.1 by
  the weak deep Trench.

### 2.2 Catch-based (C / F): a floor only
The annual-mean biomass equals catch / F under Baranov (C = F x mean B), whatever the residence
pattern.

**Catches**
- ICES official landings, 2016-24 means (`tables/catch_notes.md`) [F; D means]:
  - 3.a.20: **343** t/yr. Sweden 182, Denmark 55, Norway 105.
  - 3.a.21: **211** t/yr. Denmark 168, Sweden 43.
  - 3.a total: 555 t/yr.
- Norway's catch is reported as 3.a "not known". I put it in 3.a.20 (J): Norway's landing notes code
  it as main area 09 "Skagerrak".
- The WGWIDE WG catch for 3.a was 480 t in 2023 and 486 t in 2025.
- Norway was the main fisher until 2011, with 1 059-1 406 t/yr in 2006-10. Today it is mostly trolling
  and handlines from boats under 11 m.
- **No directed pelagic fishery.** EU 3.a.20 catch by gear, 2016-24: gillnet 43 %, handline/trolling
  32 %, bottom otter trawl 15 %, pelagic trawl 4 % (`tables/fdi_eu_3a20_mac_hom_stdout.txt`) [D].

**The floor**
- Stock F is 0.215 (2025 catch / TSB) to 0.285 (Fbar 4-8, 2025) [F/D, `../Fiskeridödlighet i Skagerrak
  idag/ices_pelagic_f.md`].
- With that F: B >= 1 200-1 600 t in 3.a.20, i.e. >= 0.035-0.047 t/km2. The same per km2 holds for 3.a
  [D, `tables/annual_mean_stdout.txt`].

**Why it is only a floor**
- The survey-based 3.a.20 annual mean is ~26 000 t. That implies a local F of 0.013 /yr, about 1/20 of
  the stock F [D].
- J: this is plausible. The local catch is coastal hook-and-net fishing, and Norwegian purse-seine and
  trawl catches there are 0-20 t/yr. Fish present in 3.a are not exposed to the stock F while there.
- So the catch route does not contradict the survey route. It only rules out values below ~0.04 t/km2,
  and it shows that GOV-level biomass (0.1) sits near that floor.
- Recreational catch is not included. It may be comparable to the commercial catch (J).

### 2.3 Ecosystem models: consistency checks, not independent measurements
All of these are assessment-derived: whole stock x MSVPA/SMS residence fraction, for ICES IV
(570 000 km2), not 3.a [F, `tables/lit_notes.md`].

**Mackerel**
- **Mackinson & Daskalov 2007** (North Sea 1991): 1.72 t/km2 = (57 kt North Sea component + 923.4 kt
  Western component) / 570 000 km2.
  - The 923.4 kt is the quarter-mean of the Western stock inside ICES IV. MSVPA residence for adults is
    10/5/50/70 % in Q1-Q4.
  - So 1.72 is already an annual mean.
- **WGSAM 2015 key run**: 1.72 -> 0.75 ("New estimation of biomass from SMS 2014").
- **Christensen 1995** (1981): 0.46.
- **SMS 2023 key run**, whole NEA stock in ICES IV, 2022 [D from F files]:
  - Quarterly 0.38 / 0.52 / 1.98 / 2.35 t/km2, annual 1.31.
  - Stock share in the North Sea: 4.9 / 5.8 / 21.2 / 27.7 % (Q1-Q4).
  - Assumption-driven: an old residence table, scaled down after 2000.
- **Kattegat EwE** (base 1982; Belgrano & Tomczak 2025, SIME 2025:3): no mackerel or horse mackerel
  group [F].
- No quantitative Skagerrak or Norwegian Trench model exists.

**Horse mackerel**
- Mackinson & Daskalov 0.579 (SGMSNS 330 kt); WGSAM 2015 0.75; SMS 2023, Western stock in IV, 2022:
  0.014.

**Reading** [D/J]
- The North Sea models' annual mackerel means (0.75-2.0) are of the same order as the box value.
- Their annual/Q3 ratio (SMS 2022: 1.31/1.98 = 0.66) is higher than the 0.45 used here.
- That fits the geography. The northern North Sea holds the overwintering shoals (Q4-Q1), while the
  Skagerrak has few mackerel in winter:
  - GOV Q1/Q3 is 0.05.
  - Jansen et al. 2012: "Relatively few mackerel were caught outside the study area, e.g. in
    Kattegat/Skagerrak" in Q1 IBTS [F].

### 2.4 Stock-scale apportionment (plausibility bound)
- **Recent survey share.** In July 2022-25 the IESSNS North Sea stratum (including the Skagerrak to
  11 E) held 0.47-0.65 Mt [D]:
  - That is 7-16 % of the 1 January TSB of 3.7-6.7 Mt (2022 7 %, 2023 12.5 %, 2024 13.7 %, 2025 15.7 %).
  - Spread over 285 781 km2 it is 1.7-2.3 t/km2, the same as the box.
- **Historical figures:**
  - 1987: "about 50 %" of the Western stock (2.5 Mt) was in the North Sea in Q3 (Degnbol et al. 1988) [F].
  - 1970: the North Sea stock "in the Skagerak and northeastern North Sea" was 632 kt (Hamre 1971) [F].
- Nothing points to a July density above ~3 t/km2 over a large area for today's stock, except the
  preliminary 2026 North Sea value (5.2).

### 2.5 Seasonal presence (sets the annual mean)
Evidence, in decreasing weight:

1. **GOV box Q1 vs Q3, 2001-25** (`../Fiskgrupper i rutan/predators/tables/pred_season.csv`) [D,
   project table]:
   - Mackerel: 1.16 vs 22.43 kg/h, Q1/Q3 = 0.052.
   - Horse mackerel: 0.20 vs 9.94, Q1/Q3 = 0.021.
   - J: GOV q may differ by season (deep, near-bottom fish in winter), which would favour Q1. So 0.05 is
     an upper bound on relative winter density.
2. **Norwegian 3.a mackerel catch by month, 2015-24** (sales notes; `tables/catch_no_3a_month_profile.csv`)
   [F; D shares]:
   - Apr 4.5 %, May 18, Jun 23, Jul 19, Aug 16, Sep 12, Oct 7, Nov-Mar 1.3 %.
   - 2020-25 is later in the year: Q3 55 %.
   - No autumn purse-seine peak since 2012. In 2010-11, 88 % came in Aug-Sep, from trolling vessels.
   - J: this is hook catch, weighted by summer effort. The May-June shares overstate relative density
     and the Sep-Oct shares understate it.
3. **EU 3.a.20 catch by quarter, 2016-24** (STECF FDI, landings + discards, 222 t/yr) [D]:
   - Q1 1.2 %, Q2 48 %, Q3 43 %, Q4 7 %.
   - About 45 % of rows are confidential. The totals match ICES official within ~10 %.
4. **Literature** [F]:
   - Revheim 1969 (via the project's evidence_B_population.md): Norwegian Skagerrak trolling "starts in
     June and lasts to August - September"; "In autumn the mackerel start to withdraw from the coastal
     waters and congregate in larger shoals in the Norwegian Channel and its slope"; the purse-seine
     peak was in October.
   - North Sea spawning runs "mid-May to late June" or "around June/July" (CSWG 2023).
   - Skagerrak fish are mostly ages 1-2 (IESSNS; Degnbol 1988).
   - FishMap: in winter, mackerel are found along "the edge of the Norwegian Deeps".
   - Jansen & Gislason 2011: "occasional winter catches" in the Skagerrak and Kattegat [A].

Profiles, as density relative to July [J] (`scripts/annual_mean.py`):

| Profile | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Presence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| narrow | .03 | .03 | .03 | .1 | .4 | .9 | 1 | .9 | .6 | .25 | .08 | .04 | 0.36 |
| **central** | .05 | .05 | .05 | .15 | .5 | 1 | 1 | 1 | .8 | .5 | .2 | .08 | **0.45** |
| wide | .1 | .1 | .1 | .25 | .7 | 1 | 1 | 1 | 1 | .8 | .5 | .2 | 0.56 |

- December-March brackets GOV's 0.05.
- September stays high for two reasons: the Aug-Sep 1987 Dana survey found July-level densities, and
  GOV Q3 (Aug-Sep) is the yearly maximum.
- October-November follow the autumn withdrawal to the Trench, which runs through the box's slope.

### 2.6 Horse mackerel: the evidence
- **Absent in early summer.**
  - IESSNS early July 2018-26: max 0.003 t/km2 [D].
  - HERAS July acoustic allocation in the box: 0.0001-0.03 t/km2 (`../Ekolodsdata planktivorer/tables/
    acoustic_box_species_by_year.csv`) [S].
- **GOV box.** Q3 (Aug-Sep) 0.070 t/km2, Q1 0.0014. Pooled x1.18 = 0.042, which is the model value [D].
  - J: horse mackerel stay near the bottom by day, so GOV q is about 0.3-1. The peak is then 0.07-0.23.
- **Argos, Aug 1989, acoustic.** 7 442 t / 8 345 km2 = 0.89 t/km2 at TS -71.2. At the HERAS physoclist
  TS (-67.5) it is 0.38 [D].
  - J: high side. 1989 was in the large-Western-stock era after the 1982 year class.
- **Catch, 3.a.20** [F/D]:
  - 210 t/yr in 2016-24, but 55 t/yr in 2020-24. Very sporadic: Norway took 4 088 t in one November
    catch in 2014.
  - Timing: EU Q3 19 %, Q4 80 %. Norway Aug-Nov only. WGWIDE 3.a Q3/Q4: 2023 22/3, 2024 13/11,
    2025 11/43 t [F].
  - With the Western stock F of 0.038 (u 2025) to 0.065 (Fbar 2025), C/F gives an annual mean of
    0.025-0.16 t/km2 [D]. That straddles the GOV-based value.
- **Models.** 1991 North Sea 0.58-0.75; SMS 2022, Western stock in IV, 0.014 [F/D].
- **Season sources:**
  - Catch quarters (above).
  - FishMap: "In autumn, horse mackerel leave the North Sea to return in the spring"; the Western stock
    returns "late autumn (October-November)" [F].
  - WKMSNS 2002: adults are 0/0/5/40 % of the Western stock in the North Sea in Q1-Q4 [F].
  - Since 2024, Skagerrak fish are assigned to the Western stock in all quarters (WGWIDE) [F].
- **Adopted** [D/J]:
  - Peak 0.17 (0.07-0.4) x presence 0.31 = **0.05 t/km2** (0.02-0.16).
  - The Kattegat catch is 3.4 t/yr, taken as ~0, so 3.a = 0.03.

## 3. Final table

| Route | Species | Area / season | t/km2 | Tag | Agrees? |
|---|---|---|---|---|---|
| IESSNS DK surface trawl, hauls in box 2019-26 | mackerel | box, early July | median 2.0 (1.7-4.2); depth-weighted 1.8 | [D] from [F] data | central |
| IESSNS North Sea stratum (Ceton mean / StoX) 2018-25 | mackerel | NS + Skagerrak to 11 E, July | 1.83 (1.0-2.4) / 0.8-2.3 | [F] / [D] | yes |
| IESSNS 2026 preliminary, North Sea | mackerel | July | ~5.2 | [D] from [F] | high, preliminary |
| Dana acoustic 1987 | mackerel | Skagerrak, Aug-Sep | 1.6-1.8 (Kattegat 0.01) | [F]/[D] | yes |
| Eldjarn acoustic 1987 | mackerel | 57-59 N NE North Sea | 7.1-10.7 | [F]/[D] | high side |
| Argos acoustic 1989 | mackerel | E Skagerrak, Aug | 1.9 (x6 at today's TS) | [F]/[D] | order |
| GOV NS-IBTS Q3 (box) | mackerel | Aug-Sep | 0.16 (2001-25) / 0.29 (2021-25) | [D] | no: q ~ 0.1 |
| Catch / stock F | mackerel | 3.a.20 annual | >= 0.035-0.047 | [D] | floor only (local F 0.013) |
| North Sea EwE 1991 / WGSAM 2015 / SMS 2022 | mackerel | ICES IV annual | 1.72 / 0.75 / 1.31 | [F]/[D] | same order |
| Stock share in July (IESSNS NS / TSB) | mackerel | NS stratum | 7-16 % of TSB -> 1.7-2.3 | [D] | yes |
| **Adopted, mackerel** | | **box annual** | **0.8 (0.33-1.7)**; summer 1.8 | [D/J] | |
| | | Skagerrak / 3.a annual | 0.75 (0.3-1.5) / 0.5 (0.2-1.2) | [D/J] | |
| IESSNS / HERAS July | horse mackerel | Skagerrak, July | ~0-0.03 | [D]/[S] | absent in July |
| GOV NS-IBTS Q3 (box) | horse mackerel | Aug-Sep | 0.070 | [D] | yes (q 0.3-1) |
| Argos acoustic 1989 (TS -67.5) | horse mackerel | E Skagerrak, Aug | 0.38 | [D] | high side |
| Catch / stock F | horse mackerel | 3.a.20 annual | 0.025-0.16 | [D] | yes |
| North Sea models | horse mackerel | ICES IV annual | 0.014-0.75 | [F]/[D] | wide |
| **Adopted, horse mackerel** | | **box / Skagerrak annual** | **0.05 (0.02-0.16)**; peak 0.17 | [D/J] | |

## 4. Caveats
1. **The presence profile is judgement.** No source gives a monthly occupancy curve for 3.a, and the
   catch-by-month profile is effort-weighted. At a fixed July density the three profiles move the box
   annual mean between 0.65 and 1.0.
2. **Surface-trawl q = 1** ignores fish below ~30 m (an underestimate) and herding or avoidance (either
   direction). J: probably modest in the stratified Skagerrak July.
3. **Few box hauls.** Only 6 IESSNS hauls fall inside the box, all at 83-207 m. The deep cells are
   handled with nearby deep hauls (0.18-0.49) in the depth weighting.
4. **Year effects.** North Sea July density varied 2.4-fold over 2018-25. The preliminary 2026 data
   suggest a strong influx of 1-3-year-olds; it is not in the central value.
5. **Composition.** Skagerrak summer mackerel are mostly ages 1-2. This affects diet and gape
   (`../Makrillens bytesstorlek/summary.md`), not t/km2.
6. **Kattegat.** The only survey value is Dana 1987 (0.01). Catch per km2 is the same as in the
   Skagerrak (0.010 t/km2/yr) because both fisheries are coastal. The x0.2 factor is judgement; only the
   Skagerrak value matters for the box.
7. **Horse mackerel variability.** Catches and influxes vary strongly between years (C 6-807 t/yr) and
   are tied to the winter Atlantic inflow (Iversen et al. 2002 [A]). A single annual mean describes them
   badly.

## 5. Files
- `scripts/iessns_box.py` -> `tables/iessns_box_stdout.txt`, `tables/iessns_box_hauls.csv`
- `scripts/annual_mean.py` -> `tables/annual_mean_stdout.txt` (depth weighting uses `maps/depth.npz`, read only)
- `scripts/fdi_eu_3a20_quarters.py` -> `tables/fdi_eu_3a20_mac_hom.csv`, `tables/fdi_eu_3a20_mac_hom_stdout.txt`
  (input: `../Fiskgrupper i rutan/planktivores/data/fdi/`)
- Catch statistics: `tables/catch_notes.md`, `tables/catch_*.csv`, `scripts/catch_*.py`, `data/catch/` (README inside)
- Literature and models: `tables/lit_notes.md`, `tables/lit_*_stdout.txt`, `scripts/lit_*.py`, `data/lit/` (README inside)
- IESSNS reports: `data/iessns/` (README inside). All `data/` content is untrusted and never executed.
