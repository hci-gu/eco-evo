# Evidence B - GOV catchability, ling/tusk indicators, Skagerrak landings

Model box: Norwegian Trench slope, Skagerrak (57.81-58.31 N, 10.07-11.15 E; 44-529 m,
mostly 100-300 m mud; NW part 300-530 m). ICES rectangles 44G0, 44G1, 45G0, 45G1.
Tags: [F] full text read / data downloaded, [A] abstract or snippet only, [S] secondary,
[D] computed by us from downloaded data. Downloads are in `data/catchability/<source>/`.
Compiled 2026-10-24.

## A. GOV trawl catchability (q)

### A1. Fraser, Greenstreet & Piet 2007, ICES J Mar Sci 64: 1800-1819 [F, read via web render]
URL: https://academic.oup.com/icesjms/article/64/9/1800/617755 (doi 10.1093/icesjms/fsm145).
OUP blocks scripted download (Cloudflare), so there is no local copy; the full HTML text, Table 4
and the Appendix were read through a web renderer. Re-check the coefficients against the PDF before
using them in production.

- Scope: q for 1-cm length classes in the **Q3 NS-IBTS GOV, North Sea (Subarea IV), 1998-2004**,
  calibrated against stock-assessment (VPA) numbers for assessed species: "estimates of total
  numbers of fish-at-age in the North Sea were obtained from the stock assessments". Corrected
  demersal biomass was 3.8-7.5 Mt, against 837 000 t from raw densities in 2004.
- **Swept-area caveat (important):** for haddock and whiting the swept area is the **whole gear
  (door spread)**, because fish "are herded into the path of the net by the sediment cloud stirred
  up by the otter doors and sweeps". A net-only swept area "would have resulted in catchability
  coefficients >1 for haddock and whiting". All other species use the **net (wing) swept area**.
  So haddock/whiting q are not on the same basis as cod/ling/hake q. A CPUE per wing-swept area or
  per hour needs a door:wing factor for those two before the corrections can be compared.
- Beyond the data range: "a constant q equal to the q obtained at the extreme of the data range was
  assumed". Discussion: "the largest fish of many species, including haddock, whiting, and plaice,
  were increasingly poorly sampled".

Appendix relationships as printed (l = total length, cm):

| Species | Method | Relationship q(l) |
|---|---|---|
| Cod | Assessed | l<=25: 0.15963; 26-96: 0.20935 - 0.00434 l + 0.00009 l^2; l>=97: 0.66822 |
| Haddock* | Assessed | l<=11: 0.0116121; 12-45: -0.370453 + 0.04227 l - 0.000685 l^2; l>=46: 0.123268 |
| Whiting* | Assessed | l<=21: 0.616878; 22-37: -0.028159 l + 1.208229; l>=38: 0.138165 |
| Saithe | Other | "Average q-at-length of haddock, whiting and cod" |
| Norway pout | Assessed | l<=11: 0.066204; l>=12: 0.191855 |
| Hake | Catch ratio | All: 0.0909 (Table 4: 0.08 at 25-29.9 cm, 0.10 at 30-34.9 cm) |
| Ling | Odds ratio | All: 0.104976 ("Where only two data points were available, the average q was used (e.g. ling") |
| Poor cod | Other (via Norway pout) | l<=5: 0.038387; 6-14: -0.002834 l + 0.05256; l>=15: 0.01004 |
| Four-bearded rockling | Catch ratio | l<=10: 0.03298; 11-24: -0.34995 + 0.05352 l - 0.00152 l^2; l>=25: 0.03647 |
| Bib | Other | "Average q-at-length of Norway pout and poor cod" |
| Tusk ("Torsk", *Brosme brosme*), pollack, greater forkbeard, blue ling, three- and five-bearded rockling, rabbit fish | Average | "Roundfish average" |
| Blue whiting, silvery pout | Pelagic | N/A (excluded) |
| Northern rockling | - | "Not caught in GOV" |

\* whole-gear (door) swept-area basis.

q evaluated at selected lengths [D]:

| cm | 10 | 15 | 20 | 25 | 30 | 40 | 50 | 60 | 70 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cod | .160 | .160 | .160 | .160 | .160 | .180 | .217 | .273 | .347 | .548 | .668 |
| Haddock* | .012 | .109 | .201 | .258 | .281 | .224 | .123 | .123 | .123 | .123 | .123 |
| Whiting* | .617 | .617 | .617 | .504 | .363 | .138 | .138 | .138 | .138 | .138 | .138 |
| Saithe (mean of 3) | .263 | .295 | .326 | .307 | .268 | .181 | .160 | .178 | .203 | .270 | .310 |
| Norway pout | .066 | .192 | .192 | ... | | | | | | | |
| Poor cod | .024 | .010 | .010 | ... | | | | | | | |
| 4-b rockling | .033 | .111 | .112 | .036 | .036 | ... | | | | | |
| Hake (all) | .091 | | | | | | | | | | |
| Ling (all) | .105 | | | | | | | | | | |

Continuity check [D]: each constant equals its polynomial evaluated at the cutoff length
(whiting at 38 cm 0.1382 vs 0.138165; haddock at 46 cm 0.1245 vs 0.123268, at 11 cm 0.0116; poor cod
at 15 cm 0.01005), so those pieces join. Cod does not: the printed polynomial gives 0.635 at 97 cm
against the constant 0.668, and 0.157 at 25 cm against 0.1596. The likely cause is the quadratic
coefficient printed to one significant figure (0.00009; about 0.000094 reproduces 0.668). So the
printed cod polynomial is biased low by up to about 5 % at 60-96 cm.
Saithe averages two door-basis q (haddock, whiting) with one net-basis q (cod), so it inherits a
mixed swept-area basis.

- **Ling is covered** (q 0.105 for all lengths, from only two odds-ratio points). **Tusk is listed
  but has no own estimate**: it gets the "Roundfish average". The paper defines that only as the
  mean q-at-length of the roundfish group; **we found no numeric value for it in the text we read**,
  and we have not back-computed one.

### A2. Walker, Maxwell, Le Quesne & Jennings 2017, ICES J Mar Sci 74: 1448-1457 [A]
"Estimating efficiency of survey and commercial trawl gears from comparisons of catch-ratios",
doi 10.1093/icesjms/fsw250, https://academic.oup.com/icesjms/article/74/5/1448/2908651.
Abstract: "Gear efficiency (and associated uncertainty) at length was ultimately estimated for 75
species, seven species-groups and for up to six types of trawl gear per species or species-group
... All estimates ... are made available in a supplementary data file." The per-species values sit
only in `fsw250_supp.zip`, which is behind a signed, time-limited URL and **was not obtained**. The
Table 3 numbers seen in the text (cod age 0-5: 0.11, 0.73, 1.38, 1.41, 1.22, 0.65; Norway pout age 0-4:
0.2, 0.36, 0.53, 0.56, 0.4) are **assessment-scaling multipliers, not q**; do not use them as q.

### A3. NS-IBTS manual, SISP 10 Rev. 11 (ICES 2020) [F]
https://ices-library.figshare.com/articles/_/19051361 (local: `data/catchability/ibts_manual/manual.pdf`)
- Sec. 2.5.3: "The maximum fishing depth for standard stations in the entire survey area is 250 m."
- Sec. 2.3.3: "since 1985, Scotland have used groundgear 'B' (with larger rubber discs) on all
  stations north of 57°30'N due to the rough seabed". Everyone else uses groundgear A.
- Sec. 2.3.2: "Because the acceptable limits were not achieved by some nations, particularly when
  using the long sweeps at deeper stations ... Norway and Sweden will continue to vary sweep length
  with bottom depth." Annex: Q1 sweeps are 60 m below 70 m depth and 110 m deeper; Q3 uses 60 m
  everywhere. Since sweep length drives herding, Q1 and Q3 q may differ for herded species.
- The manual gives no species-level catchability.

### A4. GOV coverage of the model box
Taken together with Nordic Council 2024 (B3: "A large part of Skagerrak, particularly its deep-sea
areas, is not covered by the international bottom trawl survey (IBTS)") [F], the 250 m limit means
the NW trench part of the box (300-530 m) has **no standard GOV stations**. The GOV species shares
describe only the <=250 m part.

### A5. Not pursued
Harley & Myers 2001 (hierarchical Bayesian length-specific q, NW Atlantic surveys): not fetched.
Its q is fitted per stock for another gear and region, not this species list.
Cefas "North Sea GOV gear trials 2007-08" (escape under the groundgear;
https://www.data.gov.uk/dataset/north-sea-gov-gear-trials-2007-08-fisheries-science-partnership1):
metadata only, no results on the page.

## B. Ling and tusk in Skagerrak / Norwegian Deep

### B1. ICES advice 2025 and WGDEEP 2026 [F]
- Ling lin.27.346-91214 (https://doi.org/10.17895/ices.advice.27202680): category 3. The index is
  the "standardized CPUE series from the Norwegian longline fleet operating in subareas 4 and 6 and
  Division 3.a". It is **stock-wide, with no Skagerrak split**. Index (kg/1000 hooks): 2017 163.8,
  2018 142.0, 2019 143.9, 2020 170.9, 2021 not representative, 2022 156.3, 2023 108.1, 2024 81.65,
  roughly a halving since 2020. LF=M/Lmean 1.16 (2022), 1.04 (2023, 2024); Lmean 83 cm vs LF=M 87 cm.
  ICES landings, Subarea 3: 2019 205, 2020 179, 2021 200, 2022 212, 2023 216, 2024 197 t, out of
  13-21 kt for the whole stock (about 1-1.5 %). Gear split 2024 (stock): longline 38 %, bottom trawl 40 %, gillnet 11 %.
- Tusk usk.27.3a45b6a7-912b (https://doi.org/10.17895/ices.advice.27202932): category 3, Norwegian
  longline reference-fleet CPUE ("considered robust, with consistent trends among the main areas ...
  Subarea 4 and divisions 5.b and 6.a"). Index: 2019 160.0, 2020 117.8, 2021 136.3, 2022 111.3,
  2023 105.6, 2024 84.7. LF=M/Lmean 0.96 (2022-2024). Official landings, Div. 3: 2019 8, 2020 13,
  2021 14, 2022 17, 2023 8, 2024 7 t, out of 4-5 kt for the stock (<0.4 %).
- WGDEEP 2026 (https://doi.org/10.17895/ices.pub.32051352): "Now, there are less landings in Subarea 3
  compared to landings in Subarea 4". For tusk, a SPiCT assessment was "not possible ... to recommend
  or approve" (2021 benchmark). For ling, SPiCT and a DATRAS survey index are exploratory only, and
  "model's outputs lacked robustness, particularly due to gaps in survey coverage".
- **Neither stock has an absolute biomass, F or harvest-rate estimate**: only CPUE trends and LBI.

### B2. Norwegian shrimp survey, Skagerrak + Norwegian Deep (IMR, NAFO SCR Docs) [F]/[D]
Source: Søvik & Thangstad, NAFO SCR Doc. 21/001 (2021 survey), with 20/001, 19/059, 18/068 and
15/058 for consistency: https://www.nafo.int/Portals/0/PDFs/sc/2021/scr21-001.pdf. The IMR toktrapport
on hi.no was not located; these NAFO documents are the survey's own annual reports.
- Gear and design: "A Campelen 1800/35 bottom trawl with rockhopper gear", strapping since 2008, cod
  end 20 mm with a 10 mm liner. Q1 (Jan-Feb) since 2006, 111 fixed stations at 111-540 m, strata
  100-200/200-300/300-500/>500 m, area 14 091 nm2 over 3.a + 4.a East. Trawling "around the clock".
  Door spread 52-55 m (2017-2021). "Swept area is estimated by applying a wingspread of 11.7 m".
- Fish index = "mean catch in kg per trawled nm" over all hauls, Divs. 3.a + 4.a East combined.
  **Not the model box, and no q correction.** The 2016 survey "was invalidated" and is excluded.
  Norway pout is not reported.

Mean 2017-2021 (kg/nm) and shares [D]. The density assumes q=1 over the 11.7 m wing swept area
(0.02167 km2 per nm), so it is an uncorrected lower bound.

| Species | 2017 | 2018 | 2019 | 2020 | 2021 | mean | % of demersal gadoids | t/km2 (q=1) |
|---|---|---|---|---|---|---|---|---|
| Saithe | 9.71 | 12.87 | 5.77 | 1.88 | 5.13 | 7.07 | 41.6 | 0.326 |
| Whiting | 2.60 | 4.56 | 5.20 | 2.62 | 4.62 | 3.92 | 23.1 | 0.181 |
| Haddock | 1.86 | 1.51 | 0.97 | 1.15 | 3.94 | 1.89 | 11.1 | 0.087 |
| Cod | 2.00 | 2.05 | 2.58 | 0.58 | 1.00 | 1.64 | 9.7 | 0.076 |
| Hake | 1.24 | 1.66 | 0.91 | 1.00 | 1.16 | 1.19 | 7.0 | 0.055 |
| Ling | 0.90 | 0.99 | 1.09 | 0.41 | 0.27 | 0.73 | 4.3 | 0.034 |
| Greater forkbeard | 0.18 | 0.22 | 0.20 | 0.07 | 0.11 | 0.16 | 0.9 | 0.007 |
| Pollack | 0.10 | 0.15 | 0.22 | 0.19 | 0.09 | 0.15 | 0.9 | 0.007 |
| Tusk ("Cusk") | 0 | 0.14 | 0.38 | 0.02 | 0.02 | 0.11 | 0.7 | 0.005 |
| Blue ling | 0.01 | 0.03 | 0.02 | 0.25 | 0.08 | 0.08 | 0.5 | 0.004 |
| 4-b rockling | 0.04 | 0.05 | 0.09 | 0.05 | 0.04 | 0.05 | 0.3 | 0.002 |
| Blue whiting | 31.18 | 6.38 | 19.68 | 13.04 | 59.02 | 25.86 | (60 % if included) | 1.19 |
| Rabbit fish | 5.99 | 5.03 | 5.40 | 4.35 | 4.01 | 4.96 | (non-gadoid) | 0.229 |

"Some shallow trawl stations yield large catches of saithe, while roundnose grenadier is caught
mainly in the deep parts of Skagerrak" (scr15-058). Saithe and blue whiting are patchy. Reports
after 2021 were not found on NAFO, but WGDEEP 2026 uses shrimp-survey length data for "2020,
2022-2026", so the survey continues.

### B3. Nordic Council of Ministers 2024, Nord 2024:032 [F]
"Fish stocks in Skagerrak with management shortcomings" (DTU Aqua, IMR, SLU authors);
https://pub.norden.org/nord2024-032/ (local PDF). This report combined IBTS Q1 with the Norwegian
shrimp survey "in a statistical model and estimated standardized catch rates for the period
2013-2022". Table 1, trend in landings / abundance indicator: **ling up/up (strong), tusk
weak-up/strong-down**, pollack down/down, greater forkbeard up/weak-up. "also tusk has been in
decline in recent years with stable landing amounts". It publishes no numeric indices.

### B4. Historical ling [A]/[F]
Poulsen, Cooper, Holm & MacKenzie 2007, Fish. Res. 87: 196-207 [A] (https://marineinfo.org/id/publication/119616):
ling in Skagerrak + NE North Sea in 1872 was about 100-200 million fish aged 4+. ICES CM 2005/W:14 [F]
(https://ices-library.figshare.com/articles/25350670): "Ling abundances in the Skagerrak are currently
lower than cod abundances". This is historical context only and has no modern density.

### B5. Swedish SLU data [not found]
No SLU coastal or longline ling/tusk series was found. Fiskbarometern pages were not reached.

### B6. ICES official nominal catches, 2006-2024 (version 2026-10-15) [F]/[D]
https://www.ices.dk/data/Documents/CatchStats/OfficialNominalCatches.zip (local: `ices_nominal/`).
27.3.a = 27.3.a.20 + 27.3.a.21 + 27.3.a_NK; this was checked on the data (e.g. COD 2024: 2633+31+151 = 2815).
**Norway reports 3.a only as `27.3.a_NK`** (NO rows in .20 are 0), and Kattegat (.21) is tiny for
these species. So 27.3.a total is the better Skagerrak figure. Tonnes live weight; "c" flags on 2023-24 values ignored.

| Code | Species | .20 2022-24 mean | .20 2019-23 mean | 27.3.a 2022-24 mean | 27.3.a 2019-23 mean | NO in _NK, 2022-24 |
|---|---|---|---|---|---|---|
| COD | cod | 2366 | 2311 | 2536 | 2489 | 127 |
| HAD | haddock | 4104 | 1939 | 4566 | 2076 | 368 |
| POK | saithe | 1291 | 1269 | 1500 | 1491 | 208 |
| HKE | hake | 975 | 741 | 1043 | 799 | 43 |
| NOP | Norway pout | 569 | 2076 | 569 | 2080 | 0 |
| WHG | whiting | 283 | 244 | 315 | 300 | n.c. |
| POL | pollack | 167 | 180 | 214 | 240 | 46 |
| LIN | ling | 163 | 155 | 203 | 199 | 38 |
| WHB | blue whiting | 7 | 59 | 48 | 69 | n.c. |
| USK | tusk | 1.2 | 1.3 | 10.5 | 12.0 | 9 |
| GFB | greater forkbeard | 1.3 | 1.4 | 3.2 | 2.4 | n.c. |
| RNG | roundnose grenadier | 0.3 | 0.4 | 1.1 | 1.0 | n.c. |

n.c. = not computed by country (_NK totals 2022-24: WHG 5.7, WHB 40.8, GFB 1.8, RNG 0.8 t).
Yearly 27.3.a ling 2019-2024: 204/177/195/208/210/191 t. ICES advice "Subarea 3": 205/179/200/212/216/197.
Tusk nominal 27.3.a: 9/14/13/17/7/7 vs advice Div. 3: 8/13/14/17/8/7. The two sources agree.
Ling in 27.3.a.20 by country (2022-24): DK 144, SE 18 t. Haddock rose strongly in 2022-24 and Norway
pout fell to 13 t in 2024; use 5-year means with care. These are landings for the whole Skagerrak,
not the box.

## C. Exploitation rate and density for ling/tusk

### C1. ICES [F]
No biomass-based exploitation rate exists for either stock (category 3; see B1). The only pressure
indicators are LF=M/Lmean (ling 1.04, tusk 0.96 in 2024). Ling: "Fishing pressure on the stock is
above FMSY proxy" (2023 advice, [S] via search snippet).

### C2. North Sea Ecopath 1991, Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142 [F]/[D]
https://www.cefas.co.uk/publications/techrep/tech142.pdf (local: `ecopath_ns/`). Model area is ICES
area IV, about 570 000 km2, **Skagerrak excluded**.
- FG 24 "Other gadoids (large)": "The group consists of pollack Pollachius pollachius, tusk Brosme
  brosme, ling Molva molva, greater forkbeard Phycis blennoides." Biomass "estimated using the method
  of Sparholt (1990) with cod, whiting, haddock and saithe as reference species. Combined biomass of
  FG 24 was estimated as 8.6 thousand tonnes".
- Table 3.4: B = 0.0486 t/km2 Ecopath-estimated (input 0.015), P/B 1.27, Q/B 3.2 (input 2.18), EE 0.95.
- Table 3.5 (1991): landings 18 867, discards 1 828. No unit is printed; the values are consistent with tonnes.
- [D] Landings / input biomass = 18 867/8 550 = **2.2 /yr**, more than P/B 1.27. That is why Ecopath
  raised B. Landings / balanced biomass (27 700 t) = **0.68 /yr** (0.75 with discards). Both are
  North Sea 1991 values for the four-species group, not Skagerrak today.
- FG 21 Hake: B 0.014 t/km2, P/B 0.82 (same table).

### C3. Density [D]
The only Skagerrak-region densities we can derive are the uncorrected shrimp-survey values in B2:
ling about 0.034 t/km2 and tusk about 0.005 t/km2 (3.a + 4.a East, q=1). No catchability estimate
for the Campelen survey trawl was found, so these cannot be corrected; the GOV q of A1 does not
transfer to this gear. **No published ling or tusk biomass density for Skagerrak was found.**

## Source status

| Source | Status |
|---|---|
| Fraser et al. 2007 (OUP) | Read via web render (text, Table 4, Appendix); curl blocked by Cloudflare, no local copy |
| Walker et al. 2017 (OUP) | Abstract and text only; supplementary zip (all q by length) not obtained |
| NS-IBTS manual SISP 10 Rev 11 | Downloaded (figshare), read |
| ICES advice 2025 ling, tusk | Downloaded (figshare), read |
| ICES stock annexes ling, tusk (2026) | Downloaded, not mined |
| WGDEEP 2026 report (77 MB) | Downloaded, ling/tusk sections grepped |
| ICES CM 2005/W:14 (Poulsen & Cooper) | Downloaded, read |
| Poulsen et al. 2007 Fish. Res. | Abstract only |
| NAFO SCR 15/058, 18/068, 19/059, 20/001, 21/001 (shrimp survey) | Downloaded, tables extracted |
| Shrimp survey 2022+ / IMR toktrapport on hi.no | Not found |
| Nordic Council Nord 2024:032 | PDF downloaded, read (HTML pages blocked) |
| ICES OfficialNominalCatches.zip (2006-2024) | Downloaded, processed |
| Mackinson & Daskalov 2007 (Cefas TR 142) | Downloaded, relevant tables read |
| SLU / Fiskbarometern ling, tusk | Not found |
| Harley & Myers 2001 | Not pursued |
| Cefas GOV gear trials 2007-08 | Metadata only |
