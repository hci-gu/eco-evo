# Current fishing mortality of the gadoid stocks in the Skagerrak (cod, haddock, whiting, saithe, hake) and a biomass-weighted F for the model's "gadoids" group

Scope note: this note uses about 45 tool calls, but only 4 were WebSearch and 2 WebFetch. The rest were direct
shell downloads (curl) of primary ICES files (SAG API, ICES library/figshare PDFs, DATRAS, ICES catch statistics,
WGSAM GitHub) and local processing. That departs from the requested search-heavy method, but it reads the numbers
from the primary tables rather than from snippets:
- ICES Stock Assessment Graphs (SAG) REST API: full summary tables and reference points for every stock below, for
  assessment year 2026, the latest published (terminal data year 2025; advice published 30 June 2026, and
  9 October 2026 for cod). The task asked for "2024 or 2025 advice". The 2026 sheets replace those, so I used 2026
  throughout. (Fetched full text)
- The 2026 ICES advice sheets as PDFs (cod.27.46a7d20, had.27.46a20, whg.27.47d, whg.27.3a, pok.27.3a46,
  hke.27.3a46-8abd, cod.27.21), plus the 2024 pollack sheet (pol.27.3a4). All downloaded from the ICES library
  (figshare) and read as text. (Fetched full text)
- The WKBCOD 2023 benchmark report (substock structure). Downloaded in full and searched. (Fetched full text)
- The ICES Official Nominal Catches 2006-2023 CSV (landings by species and area, including 27.3.a.20). (Fetched full text)
- DATRAS NS-IBTS CPUE-per-length (Q1 and Q3, 2021-2025), age-length-weight records (2024) and haul records (2024),
  used to compute Skagerrak biomass shares and a swept-area cross-check. (Fetched full text; the processing is my own)
- The WGSAM SMS North Sea key run 2023 output files (`summary_table_raw.out`, `summary.out`, `SMS.dat`) on GitHub.
  (Fetched full text)
- WebSearch snippets: Skagerrak surface area (SMHI). One WebFetch failed (a Thuenen explainer PDF: too many redirects).

Every number marked "computed" is my own arithmetic on those files. Nothing in the "Pooled F" section is an ICES
product.

Model box check: 57.81-58.31 N, 10.07-11.15 E lies in ICES statistical rectangles 44G0, 44G1, 45G0, 45G1. Rows
44 = 57.5-58.0 N and 45 = 58.0-58.5 N; G0 = 10-11 E and G1 = 11-12 E. I verified this against the 2024 NS-IBTS haul
positions: hauls tagged 44G0 lie at 57.62-57.86 N, 10.06-10.99 E; 45G0 at 58.08-58.40 N, 10.44-11.00 E; 45G1 at
58.06-58.50 N, 11.01-11.19 E. All are in ICES Subdivision 3.a.20 (Skagerrak). Haul depths in 45G0 run from 104 to
237 m, so the box reaches into the Norwegian Trench slope. (Computed from DATRAS HH 2024)

Key definitions used below:
- Fbar is the mean F over the fully selected ages named in the assessment (cod 2-4, haddock 2-4, whiting 2-5,
  saithe 4-7, hake 1-7).
- Under the Baranov catch equation, C = sum over ages of F_a * Nbar_a * w_a. So catch divided by the mean annual
  biomass of all ages, C/Bbar, is to a close approximation (catch weights differ somewhat from stock weights) the
  biomass-weighted instantaneous F over the whole pool. That is the
  quantity the model needs: a loss rate F applied continuously to the pooled biomass, F/1460 per 6 h tick.
- C/TSB, with TSB the 1 January total-stock biomass, approximates C/Bbar. In the SMS 2022 output, Bbar/TSB is
  0.95-1.04 for the four species (computed). The common conversion -ln(1 - C/TSB) ignores growth within the year,
  so it overstates F. I use it only as an upper bound.

## Q1. Cod: Northern Shelf cod (cod.27.46a7d20), substocks, and the Skagerrak

### Takeaway
Northern Shelf cod is assessed as three substocks (Northwestern, Southern, Viking). ICES assigns Skagerrak (3.a.20)
cod catches to the Viking substock. Latest Viking Fbar(2-4) is 0.426 in 2025, against FMSY 0.198 and FPA 0.409:
twice FMSY, and rising since 2021 (0.32). Southern and Northwestern Fbar are about 0.49-0.51. Skagerrak landings
were 1850-2800 t in 2023-2025, about 9-11% of all Northern Shelf cod landings. The biomass-weighted F on all ages is
about 0.35-0.40/yr (computed). For cod this is close to Fbar, because young cod in the North Sea and Skagerrak are
fished (and discarded) from age 1-2.

### Cited Findings
- 2026 advice summary, verbatim: "Northern shelf cod consists of three reproductively isolated substocks
  (Northwestern, Southern, and Viking), which mix and are caught together. ICES advises that when the maximum
  sustainable yield (MSY) approach and precautionary considerations are applied, there should be zero catch in 2027
  for all substocks." And: "Fishing pressure on all three substocks is above FMSY and FPA. For the Southern and
  Northwestern substocks, spawning stock biomass is below MSY Btrigger and Blim; the Viking substock is above Blim
  and below MSY Btrigger." — [ICES Advice 2026 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018) (Fetched full text)
- 2026 advice: "A review-level benchmark took place in 2025-2026 ... The revised multistock SAM model following this
  benchmark led to a rescaling of fishing mortality, SSB, and recruitment levels. Reference points were hence
  re-estimated for each substock in 2026". Also: "landings of the Viking substock (22-30% in recent years) are lower
  than landings taken from the corresponding Viking area (40-57% since 2000)". — [ICES Advice 2026 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018) (Fetched full text)
- Viking substock, Table 9d / SAG (assessment key 26001), year: recruitment (thousands), SSB (t), model catch (t),
  Fbar 2-4:
  - 2021: 61 654, 12 223, 12 923, 0.324
  - 2022: 96 750, 15 978, 14 461, 0.325
  - 2023: 63 355, 19 269, 16 527, 0.364
  - 2024: 19 371, 19 417, 17 403, 0.418
  - 2025: 47 718, 18 216, 13 122, 0.426 (95% CI 0.315-0.574)
  - 2026 SSB: 16 594
  - Earlier peak: F 0.629 in 2018.
  - Reference points: FMSY 0.198, FPA 0.409, Blim 14 027 t, BPA = MSY Btrigger 19 491 t.
  — [SAG summary 26001](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26001), [ICES Advice 2026](https://doi.org/10.17895/ices.advice.30932018) (Fetched full text)
- Southern substock (SAG 26002): Fbar 0.393 (2022), 0.458, 0.517, 0.511 (2025). SSB 8741 t in 2025.
  FMSY 0.243, FPA 0.395, Blim 14 433 t. Northwestern (SAG 26003): Fbar 0.384 (2022), 0.437, 0.493, 0.490 (2025).
  SSB 26 779 t in 2025. FMSY 0.189, FPA 0.442, Blim 23 422 t. No Flim is given for any substock.
  — [SAG 26002](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26002), [SAG 26003](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26003) (Fetched full text)
- All substocks combined, Table 7: "Catch distribution by fleet in 2025": catch 24 764 t, landings 19 922 t,
  discards 4 842 t (discards include BMS). Demersal trawls and seines of 100 mm and more take 79% of landings.
  — [ICES Advice 2026](https://doi.org/10.17895/ices.advice.30932018) (Fetched full text)
- Table 8, Subdivision 20 (Skagerrak), "ICES estimate of total landings": 2019 3364, 2020 2241, 2021 1973,
  2022 1916, 2023 2803, 2024 2777, 2025 1850 t. Danish fleets take most of it (2025 official: Denmark 1458,
  Sweden 246, Norway 122, Germany 22 t). The combined-area ICES estimate of total landings is 24 943 t (2023),
  26 451 t (2024) and 19 921 t (2025). — [ICES Advice 2026](https://doi.org/10.17895/ices.advice.30932018) (Fetched full text)
- WKBCOD 2023: "all the landings coming from areas IIIa and IIIaN were assumed as Viking substock landings". The
  forecast assumed "the TAC for COD/03AN ... and COD/4N-S62 ... would only be fishing on the Viking stock". A working
  document in the report states that "the Viking stock mainly inhabits the northern parts of the North Sea
  (27.4.a-b) and Skagerrak" and that "Skagerrak (27.3.a.20) is a nursing area for the Southern stock". Genetic
  analysis showed "a dominance of Viking juveniles in the Skagerrak". — [WKBCOD 2023 report (Archimer copy)](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf) (Fetched full text)
- Kattegat cod (cod.27.21), context only: "ICES advises that when the precautionary approach is applied, there should
  be zero catch in 2027 and 2028". F and SSB are relative only (SAG 22381: relative F 1.21 in 2024, 0.91 in 2025;
  catch 82 t in 2025). — [ICES Advice 2026 cod.27.21](https://doi.org/10.17895/ices.advice.30932012) (Fetched full text)

### Inferences
- Skagerrak share (computed): 2803/24 943 = 11.2% (2023), 2777/26 451 = 10.5% (2024), 1850/19 921 = 9.3% (2025).
- Exploitation on total biomass (computed). The Viking catch/SSB in 2025 is 13 122/18 216 = 0.72. SAG gives no TSB
  for cod, so I take TSB/SSB from the SMS key run for North Sea cod: 1.91 (2013-2022 mean) to 2.08 (2020-2022).
  That gives C/TSB = 0.72/1.91 = 0.377, or 0.72/2.08 = 0.346; the upper bound -ln(1 - 0.377) = 0.47.
- Second route (computed): the SMS ratio (C/Bbar)/Fbar for cod is 0.95 (2013-2022) or 0.87 (2020-2022). Applied to
  Viking Fbar 0.426 it gives 0.37-0.40.
- Central cod value: 0.39/yr. Range: 0.35 (3-year mean Fbar 0.403 x 0.87) to 0.50 (the -ln bound, or the Southern
  substock's Fbar 0.51 if the Skagerrak juveniles are Southern).
- Caveat: the 2026 benchmark rescaled the substock F and SSB. The SMS ratios come from the pre-benchmark North Sea
  stock definition, so they only transfer the shape of the selectivity, not absolute levels.

## Q2. Haddock (had.27.46a20)

### Takeaway
Haddock is the dominant gadoid in the Skagerrak by survey biomass, and very lightly fished. Fbar(2-4) was 0.06-0.08
in 2022-2025 (0.079 in 2025), against FMSY = FPA = 0.167. SSB is about 750 kt, four times MSY Btrigger. The
biomass-weighted F is about 0.06/yr (computed). Skagerrak landings rose from about 500 t (2020) to about 7000 t
(2025), 12-13% of the stock's landings in 2024-2025.

### Cited Findings
- 2026 advice, verbatim: "Fishing pressure on the stock is below FMSY, and spawning-stock size is above MSY
  Btrigger, BPA, and Blim." Advice: "total catches in 2027 should be no more than 87 366 tonnes."
  — [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) (Fetched full text)
- Reference points (Table 4): MSY Btrigger 192 109 t; FMSY 0.167 ("Capped at FPA"); Blim 138 250 t; BPA 192 109 t;
  FPA 0.167. No Flim. — [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) (Fetched full text)
- Table 9 / SAG (key 22417), year: SSB (t), catch (t; landings + discards + IBC), Fbar 2-4:
  - 2018: 162 839, 39 687, 0.241
  - 2019: 133 189, 36 320, 0.228
  - 2020: 175 991, 38 928, 0.186
  - 2021: 312 079, 45 299, 0.130
  - 2022: 590 846, 57 384, 0.080
  - 2023: 790 487, 67 076, 0.064
  - 2024: 770 388, 66 868, 0.060
  - 2025: 751 318, 69 839 (landings 53 344, discards 15 292, IBC 1201), 0.079 (CI 0.061-0.102)
  - 2026 SSB: 699 810
  - Forecast Fsq for 2026: 0.068.
  — [SAG summary 22417](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22417), [ICES Advice 2026](https://doi.org/10.17895/ices.advice.30932057) (Fetched full text)
- Table 8, Subdivision 20 official landings by country, 2023 / 2024 / 2025 (t):
  - Germany: 184 / 263 / 368
  - Denmark: 2961 / 4800 / 5014
  - Netherlands: 4 / 12 / C (confidential)
  - Norway: 209 / 679 / 1216
  - Sweden: 347 / 592 / 559
  - The TAC for 3.a.20 was 3589 t (2023), 6233 t (2024) and 5892 t (2025).
  — [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) (Fetched full text)

### Inferences
- Skagerrak landings (computed): 3705 t (2023), 6346 t (2024), at least 7157 t (2025, excluding the confidential
  Dutch figure). That is 8.4%, 12.4% and about 13% of the ICES landings.
- Exploitation on total biomass (computed). C/SSB in 2025 = 69 839/751 318 = 0.093. SMS North Sea haddock TSB/SSB
  is 1.34 (2022, an old-fish-dominated stock) to 1.63 (2013-2022 mean), so C/TSB = 0.057-0.069. The SMS
  (C/Bbar)/Fbar ratio for haddock is 0.82 (2013-2022) or 0.64 (2020-2022), giving 0.079 x 0.64-0.82 = 0.051-0.065.
- Central haddock value: 0.06/yr. Range: 0.045-0.075.

## Q3. Whiting: North Sea (whg.27.47d) and Skagerrak-Kattegat (whg.27.3a)

### Takeaway
Whiting in 3.a is a separate category 3 stock. It has no absolute F, only a length-based proxy that is slightly above
the FMSY proxy (LF=M/Lmean = 1.07 in 2025). Catches are about 1.1 kt, of which 65% are discards. The neighbouring
North Sea stock has Fbar(2-5) 0.093 (2025) against FMSY 0.35, and SSB about 340-395 kt. For the model I use the
North Sea assessment as the quantitative anchor: biomass-weighted F about 0.08/yr (computed). The 3.a indicator
suggests the local rate could be higher, but it cannot be converted to a number.

### Cited Findings
- whg.27.3a 2026, verbatim: "Fishing pressure on the stock is above FMSY proxy, and the stock size index is above
  Itrigger." Table 1 gives: "Mean catch length (Lmean = L2025) 22.6 cm; MSY proxy length (LF = M) 24.2 cm; Fishing
  pressure proxy (LF = M/Lmean) 1.07"; "Index A (2024, 2025) 1.42; Index B (2021, 2022, 2023) 0.99"; "Discard rate
  (2025) 65%"; catch advice for 2027 and 2028 "546 tonnes". Table 3 notes "FPA Not defined".
  — [ICES Advice 2026 whg.27.3a](https://doi.org/10.17895/ices.advice.30932453) (Fetched full text)
- whg.27.3a: "Whiting in Division 3.a are mainly caught as bycatch in demersal fisheries and are mostly discarded
  ... ICES-estimated catches in 2025 were more than double the TAC." And: "There is an unknown degree of mixing with
  whiting in Subarea 4 and the western Baltic." — [ICES Advice 2026 whg.27.3a](https://doi.org/10.17895/ices.advice.30932453) (Fetched full text)
- whg.27.3a Table 5 / summary, year: landings incl. IBC, discards, catch (t), and F proxy where available:
  - 2021: 199, 1014, 1212
  - 2022: 276, 433, 709, proxy 0.98
  - 2023: 338, 388, 726, proxy 1.06
  - 2024: 352, 502, 854, proxy 1.01
  - 2025: 395, 724, 1119, proxy 1.07
  - The biomass index is 1.69 in 2025, up from about 1.0 in 2021-2023.
  — [ICES Advice 2026 whg.27.3a](https://doi.org/10.17895/ices.advice.30932453) (Fetched full text)
- whg.27.47d 2026, verbatim: "Fishing pressure on the stock is below FMSY, and spawning-stock size is above MSY
  Btrigger, BPA, and Blim." FMSY = FPA = 0.35; Blim 144 175 t; BPA = MSY Btrigger 201 845 t; "Fages 2-5 (2026)
  0.093". — [ICES Advice 2026 whg.27.47d](https://doi.org/10.17895/ices.advice.30932456) (Fetched full text)
- whg.27.47d SAG (key 22488), year: SSB (t), catch (t), Fbar 2-5:
  - 2020: 202 405, 34 016, 0.179
  - 2021: 301 245, 35 027, 0.139
  - 2022: 364 075, 27 988, 0.095
  - 2023: 395 137, 29 367, 0.085
  - 2024: 372 758, 36 559, 0.078
  - 2025: 341 967, 33 331 (landings 14 766, discards 16 544, IBC 2021), 0.093
  — [SAG summary 22488](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22488) (Fetched full text)
- Official nominal landings of whiting in 27.3.a.20 were 304 t in 2023, against 314 t in all of 27.3.a (2023).
  — [ICES Official Nominal Catches 2006-2023](https://www.ices.dk/data/Documents/CatchStats/OfficialNominalCatches.zip) (Fetched full text)

### Inferences
- About 97% of 3.a whiting landings come from the Skagerrak, 304/314 in 2023 (computed). Discards are not split by
  subdivision.
- Exploitation on total biomass, North Sea stock (computed). C/SSB in 2025 = 33 331/341 967 = 0.0975. SMS whiting
  TSB/SSB is 1.25 (2013-2022), so C/TSB = 0.078. The SMS (C/Bbar)/Fbar ratio is 0.90-0.96, giving
  0.093 x 0.90 = 0.084.
- Central whiting value: 0.08/yr.
- The LF=M/Lmean proxy above 1 means the length structure in 3.a is consistent with F somewhat above the F = M
  level, which would be higher than the North Sea Fbar. The advice gives no M, so I cannot turn this into a
  number. My range for whiting is 0.035 to 0.15. The low end is the local swept-area estimate in the pooled section;
  the high end is my judgement, not an ICES number.

## Q4. Saithe (pok.27.3a46)

### Takeaway
Saithe is fished above FMSY and FPA: Fbar(4-7) 0.414 in 2025, against FMSY 0.279 and FPA 0.290. SSB (116 722 t) is
below Blim after the 2026 revision of maturity and reference points. Fbar has been 0.35-0.41 for a decade. Because
ages 1-3 are hardly fished, the biomass-weighted F is lower, about 0.29-0.32/yr (computed). The Skagerrak is a small
part of the fishery: about 1.0-1.3 kt of landings per year in 2021-2023.

### Cited Findings
- 2026 advice, verbatim: "Fishing pressure on the stock is above FMSY and FPA, and spawning-stock size is below MSY
  Btrigger, BPA, and Blim." Advice: "catches in 2027 should be no more than 48 320 tonnes."
  — [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) (Fetched full text)
- On the revision: "the data as well as the statistical model used to estimate maturity-at-age were revised, which
  led to a revision of the historical spawning-stock biomass (SSB). These changes contributed to a downwards revision
  of SSB in 2025 and 2026 to below Blim and a lower target fishing mortality." On the juveniles: "Because of the
  coastal distribution of the juveniles, it has not been possible to establish survey estimates for fish at young
  ages." And: "Discards in 2025 were 8% of the total catch by weight." — [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) (Fetched full text)
- Reference points (Table 4): MSY Btrigger 191 280 t; FMSY 0.279; Blim 137 653 t; BPA 191 280 t; FPA 0.290;
  MAP range 0.191-0.290. No Flim. — [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) (Fetched full text)
- SAG (key 22504; recruitment at age 3), year: SSB (t), catch (t), Fbar 4-7:
  - 2016: 176 977, 77 672, 0.356
  - 2019: 176 534, 96 634, 0.381
  - 2021: 151 904, 50 951, 0.363
  - 2022: 133 980, 53 808, 0.351
  - 2023: 133 598, 61 041, 0.369
  - 2024: 104 613, 60 303, 0.385
  - 2025: 116 722, 73 382 (landings 67 683, discards 5698), 0.414 (CI 0.336-0.51)
  — [SAG summary 22504](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22504) (Fetched full text)
- Official nominal landings of saithe in 27.3.a.20: 1153 t (2021), 1303 t (2022), 1020 t (2023). The advice sheets
  do not tabulate 3.a separately; Subarea 4 and Division 3.a are reported together.
  — [ICES Official Nominal Catches 2006-2023](https://www.ices.dk/data/Documents/CatchStats/OfficialNominalCatches.zip) (Fetched full text)

### Inferences
- Skagerrak share (computed): about 1.0-1.3 kt of a 51-61 kt stock catch, roughly 2%.
- Ages covered: the single-species saithe assessment starts at age 3, and no survey covers juveniles. SMS keeps
  saithe ages 0-2 as prey with F = 0 (`SMS.dat`: first catch age 3; `summary.out` has N at ages 0-2), so the SMS
  ratios below cover ages 0+, but those young-age numbers are back-calculated, not assessed. The IBTS saithe CPUE
  used for weighting covers whatever sizes the GOV catches.
- Exploitation on total biomass (computed). C/SSB in 2025 = 73 382/116 722 = 0.629. SMS saithe TSB/SSB is 1.99
  (2013-2022), giving C/TSB = 0.316 (upper bound -ln form 0.38). The 2026 maturity revision lowered SSB relative to
  the maturity SMS used, so this route probably overstates the rate. The SMS (C/Bbar)/Fbar ratio is 0.69-0.71,
  giving 0.414 x 0.69 = 0.286.
- Central saithe value: 0.30/yr. Range: 0.11 (local swept-area estimate, below) to 0.38.

## Q5. Northern hake (hke.27.3a46-8abd), pollack and ling

### Takeaway
Hake is the only stock here where ICES publishes TSB (all ages, both sexes). C/TSB was 0.267 in 2025, against
Fbar(1-7) 0.236, FMSY 0.24 and FPA 0.54. ICES says F is at FMSY. Subarea 3 (Skagerrak-Kattegat) contributes about
0.9-1.3 kt, 2% of the stock catch, and hake is a minor part of the Skagerrak gadoid biomass (about 1%). Pollack (3a4)
is a category 5 stock with no F estimate, and Skagerrak landings are about 150-200 t/yr. Ling: about 160 t/yr in
3.a.20. Neither affects the pooled F.

### Cited Findings
- Hake 2026, verbatim: "Fishing pressure on the stock is at FMSY, and spawning-stock size is above MSY Btrigger,
  BPA, and Blim." Reference points: FMSY 0.24, FPA 0.54, Blim 61 563 t, BPA 78 405 t. Fbar ages 1-7.
  — [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108) (Fetched full text)
- Hake Table 10, TSB footnote: "Total stock size, both sexes, including immature fish". SSB is female-only. Year:
  SSB, TSB, landings, discards (t), Fbar:
  - 2023: 142 247, 241 950, 59 380, 2990, 0.22
  - 2024: 121 402, 211 327, 49 572, 4164, 0.21
  - 2025: 109 006, 193 946, 48 055, 3781, 0.24
  - 2026: SSB 98 611, TSB 181 040
  — [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108), [SAG 22627](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22627) (Fetched full text)
- Hake Table 8, Subarea 3, ICES estimated landings / discards (t): 892 / 43 (2023), 1212 / 102 (2024), 879 / 96 (2025).
  — [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108) (Fetched full text)
- Pollack 3a4, verbatim: "ICES cannot assess the stock and exploitation status relative to MSY or precautionary
  approach (PA) reference points because information to define reference points is not available." Category 5;
  "catches in the years 2025-2027 should be no more than 1 462 tonnes". — [ICES Advice 2024 pol.27.3a4](https://doi.org/10.17895/ices.advice.25019474) (Fetched full text)
- Official landings in 27.3.a.20, 2021 / 2022 / 2023 (t): pollack 150 / 142 / 199; ling 150 / 164 / 162;
  hake 727 / 976 / 828. — [ICES Official Nominal Catches 2006-2023](https://www.ices.dk/data/Documents/CatchStats/OfficialNominalCatches.zip) (Fetched full text)

### Inferences
- Hake (computed): C/TSB 2025 = 51 836/193 946 = 0.267. C/Bbar, with Bbar = (193 946 + 181 040)/2, is 0.276.
  Here C/B exceeds Fbar because F_1-7 averages in lightly fished age-1 fish while older fish carry the biomass.
- Central hake value: 0.27/yr. Range: 0.24-0.31.
- Hake catches in the Skagerrak come from the Norwegian Trench slope. The GOV survey barely samples it (Q1 CPUE
  about 0.5 kg/h).

## Q6. SMS key run 2023 as a cross-check on F by species

### Takeaway
Yes. `summary_table_raw.out` gives SSB, TSB, yield and mean F by species and year up to 2022, and `summary.out`
gives quarterly N, Nbar and weights by age, so C/Bbar can be computed over all ages including the 0-group. First age with catch data: cod 1, whiting 0,
haddock 0, saithe 3; younger ages have F = 0 in SMS. The SMS
mean-F ages are the same as the single-species ones (cod 2-4, whiting 2-5, haddock 2-4, saithe 4-7; `SMS.dat`,
option avg.F.ages). The 2022 SMS values match the trend of the single-species assessments. The SMS-derived ratio
(C/Bbar)/Fbar, the "selectivity conversion", is about 0.9 for cod and whiting, 0.65-0.8 for haddock and about 0.7
for saithe.

### Cited Findings
- WGSAM North Sea key run 2023 files. Computed from them, year: SSB, TSB (t, 1 January), yield, mean F, C/Bbar:
  - Cod 2022: 48 920, 104 276, 21 128, 0.250, 0.213. Cod 2020: F 0.353, C/Bbar 0.309.
  - Whiting 2022: 457 742, 522 300, 27 891, 0.047, 0.056.
  - Haddock 2022: 352 007, 472 228, 56 890, 0.122, 0.127.
  - Saithe 2022: 162 978, 327 912, 53 806, 0.247, 0.158.
  — [SMS summary_table_raw.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/summary_table_raw.out), [SMS summary.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/summary.out), [SMS.dat](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/SMS.dat) (Fetched full text)
- Computed ratios, mean (range):

  | Species | Period | (C/Bbar)/meanF | TSB/SSB |
  |---|---|---|---|
  | Cod | 2013-2022 | 0.95 (0.85-1.18) | 1.91 |
  | Cod | 2020-2022 | 0.87 | 2.08 |
  | Whiting | 2013-2022 | 0.90 (0.80-1.18) | 1.25 |
  | Haddock | 2013-2022 | 0.82 (0.41-1.71) | 1.63 (1.12-2.65) |
  | Haddock | 2020-2022 | 0.64 | 2.02 |
  | Saithe | 2013-2022 | 0.69 (0.58-0.80) | 1.99 |

### Inferences
- The SMS stock areas differ from the current assessments. SMS cod is the old North Sea stock, not the 2023
  Northern Shelf definition, and SMS haddock has no 6.a. So absolute SMS F and SSB values do not match SAG; for
  example, haddock SSB in 2022 is 352 kt in SMS and 591 kt in SAM. I use SMS only for within-stock ratios.
- The haddock ratio swings with year-class strength. When a big year class is in the water (2020: TSB/SSB 2.65),
  the biomass-weighted F falls well below Fbar.

## Pooled F for the model's gadoid group

### Weighting: Skagerrak gadoid biomass shares from NS-IBTS (computed)
Method:
- DATRAS NS-IBTS CPUE per length (numbers per hour), Q1 and Q3 of 2021-2025.
- Converted to kg/h with length-weight relations fitted to the 2024 NS-IBTS age-length-weight records:

  | Species | a | b | n |
  |---|---|---|---|
  | Cod | 0.00675 | 3.105 | 2296 |
  | Haddock | 0.00771 | 3.058 | 10 259 |
  | Whiting | 0.00937 | 2.952 | 8935 |
  | Saithe | 0.00601 | 3.091 | 1500 |
  | Hake | 0.00598 | 3.032 | 836 |

  (W in g, L in cm.)
- Averaged over valid hauls in two areas. "Skagerrak" is east of the Hanstholm-Lindesnes line and north of
  57.75 N east of 10.6 E (Kattegat excluded): 239 hauls. "Box" is rectangles 44G0-45G1: 120 hauls.
- The GOV trawl under-samples large cod, pelagic saithe and deep hake, and its catchability differs by species.
  These are relative survey shares, not absolute biomass.

Mean CPUE in kg/h (share of the five species in brackets):

| Area, years, quarters | Cod | Haddock | Whiting | Saithe | Hake | Total |
|---|---|---|---|---|---|---|
| Skagerrak 2021-25 Q1+Q3 (W1) | 23.8 (5.8%) | 223.4 (54.1%) | 110.0 (26.7%) | 50.0 (12.1%) | 4.4 (1.1%) | 413 |
| Skagerrak 2023-25 Q1+Q3 (W2) | 19.8 (4.2%) | 287.0 (60.9%) | 115.6 (24.5%) | 42.4 (9.0%) | 5.3 (1.1%) | 471 |
| Box 44G0-45G1 2021-25 Q1+Q3 (W3) | 19.0 (5.1%) | 133.6 (35.6%) | 145.3 (38.7%) | 72.1 (19.2%) | 4.7 (1.2%) | 376 |
| Skagerrak 2021-25 Q1 only (W4) | 19.3 (8.5%) | 121.9 (53.4%) | 83.1 (36.4%) | 2.8 (1.2%) | 0.4 (0.2%) | 228 |

Pollack and ling add 0.1-0.2% each and are left out. Saithe is almost absent in Q1 and patchy in Q3; it is a
schooling, semi-pelagic species, and one Q3 2022 haul set gave 214 kg/h. That makes the saithe weight the most
uncertain input.

### Per-stock biomass-weighted F (computed, from Q1-Q5)

| Stock (anchor) | Fbar, latest (2025) | Fbar 2023-25 mean | Biomass-weighted F, central | Range | FMSY |
|---|---|---|---|---|---|
| Cod, Viking substock | 0.426 (2-4) | 0.403 | 0.39 | 0.35-0.50 | 0.198 |
| Haddock | 0.079 (2-4) | 0.068 | 0.06 | 0.045-0.075 | 0.167 |
| Whiting, North Sea (3.a: LBI only) | 0.093 (2-5) | 0.085 | 0.08 | 0.035-0.15 | 0.35 |
| Saithe | 0.414 (4-7) | 0.389 | 0.30 | 0.11-0.38 | 0.279 |
| Hake | 0.236 (1-7) | 0.223 | 0.27 | 0.24-0.31 | 0.24 |

### Pooled result (computed)
F_pool = sum of w_i * F_i over the five stocks, with w_i normalised over them.

Worked example with the central F values and W1 weights:

0.058 x 0.39 + 0.543 x 0.06 + 0.267 x 0.08 + 0.121 x 0.30 + 0.011 x 0.27
= 0.0226 + 0.0326 + 0.0214 + 0.0364 + 0.0029
= 0.116 /yr

| Weights | Fbar-based (2025 Fbar) | Fbar-based (2023-25 mean) | Biomass-weighted, central | Low (all low F) | High (all high F) |
|---|---|---|---|---|---|
| W1 Skagerrak 2021-25 | 0.145 | 0.132 | 0.116 | 0.070 | 0.159 |
| W2 Skagerrak 2023-25 | 0.129 | 0.117 | 0.103 | 0.063 | 0.142 |
| W3 Box 44G0-45G1 | 0.168 | 0.155 | 0.133 | 0.072 | 0.187 |
| W4 Skagerrak Q1 only | 0.118 | 0.107 | 0.099 | 0.068 | 0.143 |

Recommended for the model:
- Central F = 0.12/yr.
- Plausible range 0.07-0.19/yr: low F values with any weights, up to high F values with the box weights.
- Per 6 h tick (computed): 0.12/1460 = 8.2e-5. The range is 4.8e-5 to 1.3e-4.
- Combined with the model's M1 of 0.41, total non-predation removal becomes about 0.53/yr. F is about 23% of it.

### Independent local cross-check: Skagerrak catch / swept-area biomass (computed, rough)
Inputs:
- Swept area per GOV trawl-hour in the 2024 Skagerrak hauls (DATRAS HH): median wingspread 20.6 m x median
  6852 m/h = 0.141 km2/h. So 1 kg/h corresponds to 7.08 kg/km2 at catchability 1.
- Median door spread 65-92 m, roughly 4 x the wingspread.
- Skagerrak surface area: about 32 300 km2 ([SMHI Oceanografi 38](https://www.smhi.se/download/18.18f5a56618fc9f08e832a49d/1717801532204/Oceanografi_38%20Skagerrak%20-%20the%20gateway%20to%20the%20North%20Sea..pdf); Snippet).

Biomass, from 2023-25 Q1+Q3 CPUE, wing-spread scaling (multiply by 228.7 t per kg/h):

| Species | Biomass (t) |
|---|---|
| Cod | 4.5 k |
| Haddock | 65.6 k |
| Whiting | 26.4 k |
| Saithe | 9.7 k |
| Hake | 1.2 k |
| Total | 107.5 k (3.3 t/km2) |

Catches, 2023-25 means. Discards are raised with each stock's overall discard fraction, because area-specific
discards are not published.

| Species | Basis | Catch (t) |
|---|---|---|
| Cod | 2477 t landed / (1 - 0.196) | 3081 |
| Haddock | 5736 t landed / (1 - 0.236) | 7508 |
| Whiting | whole 3.a catch | 900 |
| Saithe | 2021-23 mean landings 1159 t / 0.92 | 1260 |
| Hake | Subarea 3 catch | 1075 |
| Total | | 13 824 (about 0.43 t/km2/yr) |

Result: u = 13 824/107 500 = 0.13/yr. That agrees with the pooled central value of 0.12.

By species, the same arithmetic gives haddock 0.11, whiting 0.034 and saithe 0.13. Cod (0.68) and hake (0.89) are
implausibly high. Both are under-sampled by the GOV (large cod, deep hake), and their Skagerrak catches come largely
from the Norwegian Trench.

With door-spread scaling (about 4 x lower biomass), the total is 0.83 t/km2 and u would be about 0.5. That is
incompatible with the stock assessments, which suggests the effective catchability is closer to wing-spread scaling.

### Sensitivity: Fbar-based vs exploitation-rate-based pooled F
- Plugging Fbar straight in gives 0.12-0.17/yr; the biomass-weighted (C/Bbar) route gives 0.10-0.13/yr. Fbar
  overstates the pool rate by about 15-25%.
- The bias is concentrated in saithe, whose Fbar ages 4-7 are fully recruited while ages 1-3 are barely fished
  (ratio about 0.7), and in haddock (0.65-0.8). Cod and whiting are fished and discarded from age 1-2, so their
  ratio is about 0.9.
- The weighting matters as much as the method. The box weights (W3: more whiting and saithe, less haddock) raise the
  pooled value by about 0.02-0.03 compared with the Skagerrak-wide weights.
- Saithe contributes 23-43% of the pooled F under the Q1+Q3 weights, from only 9-19% of the biomass. The saithe
  share is therefore the single biggest lever. With Q1-only weights, which almost exclude the seasonal saithe, the
  pooled value drops to 0.10.

### Caveat for the model
The rate (0.12/yr) is what should go into the model, not a tonnage. A real flag remains, though:
- The Skagerrak-wide catch of these gadoids is about 0.43 t/km2/yr.
- The model's gadoid biomass is about 0.7 t/km2. Fishing it at the Skagerrak-average catch per km2 would imply
  u = 0.6.
- Wing-spread-scaled IBTS gives about 3.3 t/km2 for the Skagerrak, and 2.7 t/km2 (376 kg/h x 7.08) for the box
  rectangles.

So either the model's starting biomass is low compared with the survey, or much of the Skagerrak catch comes from the
deep Norwegian Trench rather than from the Mareld box. Both are probably partly true. I could not resolve which with
the data fetched here.

## Source status

| Source | What was used | Status |
|---|---|---|
| [ICES SAG API, assessment year 2026 (keys 26001-26003, 26010, 22417, 22488, 22563, 22504, 22627, 22381)](https://sag.ices.dk/SAG_API/api/StockList?year=2026) | Fbar, SSB, catch, TSB (hake), reference points | Fetched full text |
| [ICES Advice 2026 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018) | Substocks, reference points, Table 7, Table 8 (3.a.20 landings), Table 9d | Fetched full text (PDF) |
| [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) | Status, reference points, Tables 7-9, 3.a.20 landings | Fetched full text (PDF) |
| [ICES Advice 2026 whg.27.47d](https://doi.org/10.17895/ices.advice.30932456) | Status, reference points, Fsq | Fetched full text (PDF) |
| [ICES Advice 2026 whg.27.3a](https://doi.org/10.17895/ices.advice.30932453) | rfb rule, LBI proxy, catches, discard rate | Fetched full text (PDF) |
| [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) | Status, reference points, maturity revision | Fetched full text (PDF) |
| [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108) | TSB, Subarea 3 catch, reference points | Fetched full text (PDF) |
| [ICES Advice 2026 cod.27.21](https://doi.org/10.17895/ices.advice.30932012) | Kattegat context | Fetched full text (PDF) |
| [ICES Advice 2024 pol.27.3a4](https://doi.org/10.17895/ices.advice.25019474) | Category 5, no F | Fetched full text (PDF) |
| [WKBCOD 2023 report](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf) (doi 10.17895/ices.pub.22591423) | 3.a.20 cod assigned to Viking; Skagerrak nursery | Fetched full text (PDF) |
| [ICES Official Nominal Catches 2006-2023](https://www.ices.dk/data/Documents/CatchStats/OfficialNominalCatches.zip) | Landings by species in 27.3.a.20 | Fetched full text (CSV) |
| [DATRAS web services](https://datras.ices.dk/WebServices/DATRASWebService.asmx) (getCPUELength, getCAdata, getHHdata; NS-IBTS) | Biomass shares, L-W fits, swept area | Fetched full text; processing my own |
| [WGSAM SMS key run 2023](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023) | SSB, TSB, yield, mean F, quarterly Nbar | Fetched full text |
| [SMHI Oceanografi 38](https://www.smhi.se/download/18.18f5a56618fc9f08e832a49d/1717801532204/Oceanografi_38%20Skagerrak%20-%20the%20gateway%20to%20the%20North%20Sea..pdf) | Skagerrak area 32 300 km2 | Snippet (search result only) |
| Thuenen explainer on northern shelf cod (2024) | Substock areas | Not fetched (redirect loop) |
| WGNSSK 2026 report (stock numbers at age, TSB for cod, haddock, whiting, saithe) | Would give direct TSB | Not fetched; TSB taken from SMS ratios instead |

Not verified:
- The exact substock map (Figure 2 of the cod advice is an image).
- Area-specific discards for 3.a.20.
- Official 3.a.20 landings for 2024-2025 for saithe, whiting and hake. The nominal-catch CSV ends in 2023.
- Absolute TSB for cod, haddock, whiting and saithe in 2025. These are scaled from SMS ratios, not read from an
  assessment table.
