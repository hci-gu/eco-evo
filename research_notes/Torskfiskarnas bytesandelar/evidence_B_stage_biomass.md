# Evidence B - stage biomass shares from stock assessments (independent check of the GOV shares)

Compiled 2026-10-24. Tags: [F] full text read / data downloaded, [A] abstract or snippet,
[S] secondary, [D] computed here. Scripts: `scripts/stock_*.{py,sh}`. Tables: `tables/stock_*.csv`.
Downloads: `data/stock/` (listed in `data/README.md`). All shares are percent of the species' biomass.

## 1. Summary

The stock-level age structure is a check on GOV undersampling of small fish. It does not replace
the box's own mix: the box is a deep Skagerrak slope (44-529 m), and the GOV covers only its
<= 250 m part.

Stock-level share of biomass in the youngest ages, 2021-2025 (the survey years) [D].
In brackets: 2019-2024, which holds the strong 2019-2020 haddock and whiting year classes.

| Stock (run) | Q1 age 1 | Q1 age 2 | Q3 age 0 | Q3 age 1 | Q3 age 2 | Q3 age 0, share of numbers |
|---|---|---|---|---|---|---|
| Haddock had.27.46a20 (SAM, WGNSSK 2026) | 9.5 (15.7) | 23.3 (26.2) | 1.6 (3.8) | 11.3 (18.6) | 22.4 (24.1) | 32 (50) |
| Whiting whg.27.47d (SAM, 2026 run) | 23.0 (25.1) | 22.3 (23.5) | 10.9 (15.0) | 21.2 (23.4) | 19.8 (19.7) | 61 (68) |
| Cod, Viking substock (incl. Skagerrak) | 7.2 (10.1) | 20.9 (21.8) | 1.7 (2.5) * | 11.5 (15.9) | 24.4 (24.6) | 68 (72) * |
| Saithe pok.27.3a46 (starts at age 3) | age 3: 23.8 (20.8) | - | none in the model | - | - | - |
| Hake hke.27.3a46-8abd (SS3, SAG only) | - | - | 0.4-1.5 ** | - | - | - |

\* Cod has no age 0 in the model. Back-projected from age 1 next year with M0 = 1.5/yr (range 1.0-2.5
gives 1.4-2.5 % for 2021-2025). \*\* Crude: SAG recruitment (age 0) x weight at 8-12 cm / TSB.

The 0-group is a small share of biomass even when it dominates numbers. Haddock age 0 is 32 % of the
fish in Q3 but 1.6 % of the biomass. Whiting: 61 % of numbers, 11 % of biomass. Cod: 68 % and 1.7 %.

Recommended stage shares for the box (percent of species biomass; central value in brackets) [D].
"Pelagic 0-group" is the Jun-Jul pelagic stage (3-7 cm), given as percent of the Q3 stock biomass.
In Aug-Sep most 0-group fish have settled at 8-15 cm and sit in the "< 20 cm" class.

| Species | Quarter | Pelagic 0-group (< 7-10 cm) | Small juveniles (< 20 cm) | 20-30 cm | >= 30 cm |
|---|---|---|---|---|---|
| Haddock | Q1 | 0 | 5-12 (8) | 15-26 (20) | 62-79 (72) |
| Haddock | Q3 | 0.1-0.6 in Jun-Jul; 0-0.5 in Aug-Sep | 2-10 (4) | 8-17 (12) | 75-90 (84) |
| Whiting | Q1 | 0 | 10-27 (17) | 31-44 (36) | 39-58 (47) |
| Whiting | Q3 | 1-3 in Jun-Jul; 1.5-2.5 in Aug-Sep | 16-29 (21) | 33-50 (38) | 21-50 (41) |
| Cod | Q1 | 0 | 1-5 (2.5) | 5-7 (6) | 88-95 (91.5) |
| Cod | Q3 | 0.2-1.5 in Jun-Jul; ~1 in Aug-Sep | 2-3 (2.5) | 8-24 (15) | 73-90 (82.5) |
| Saithe | Q1 and Q3 | 0 | 0-0.2 (0) | 0-0.6 (0.3) | > 99 |
| Hake | Q1 | 0 | 1-4 (3) | 6-24 (15) | 72-93 (82) |
| Hake | Q3 | ~0 | 1-2 (1.5) | 5-10 (7) | 88-94 (91) |

What drives the ranges:
- Haddock: year-class strength (2021-2025 vs 2019-2024 doubles the Q1 < 20 cm share). The upper Q3
  bound is the Fraser-corrected box survey. The stock check does not confirm it; the numbers
  ratio favours the raw end (section 4.1).
- Whiting: Fraser vs raw survey, and geography. In Q1 the box holds fewer small whiting than the stock.
  In Q3 it holds more. The weight of age-0 whiting is uncertain by a factor of 2.
- Cod: the assumed M0 for the pelagic and age-0 estimate. In Q3 the box holds many more 20-30 cm
  (age 1) cod than the Viking substock as a whole.
- Saithe: ages 0-2 live in inshore nurseries, so small saithe are absent from the box in both sources.
- Hake: box survey only, plus a crude stock-level age-0 share.

Pool-level sums with the box species mix (haddock 46.6, saithe 29.1, whiting 17.0, cod 5.7,
hake 0.85 %) and the central values [D]: < 20 cm is 6.8 % of gadoid biomass in Q1 (range 4-11) and
5.6 % in Q3 (4-10). 20-30 cm is 16 % in Q1 and 13 % in Q3. The Jun-Jul pelagic 0-group is about
0.5 % (0.2-0.9).

## 2. Method

1. Numbers-at-age N(a) on 1 January, F, M and stock weights were taken from the assessments
   (section 6 lists the sources). Thousands x kg = tonnes.
2. Survival to survey time: N(a, t) = N(a) exp(-(F + M) t), with t = 0.10 (Q1, Jan-Feb) and t = 0.62
   (Q3, mid-August). Age 0 is absent in Q1.
3. In SAM, age 0 on 1 January is a notional number back-projected with the full M0. So
   N0(Q3) = N0 exp(-Z0 0.62) is consistent with the model. Cod starts at age 1, so
   N0(Q3, y) = N1(y + 1) exp(M0 x 0.38), with M0 = 1.5 (1.0-2.5). Viking M at age 1 is about 1.3.
4. Weights: Q1 uses the stock weight-at-age (1 January). Q3 uses two variants. (a) "survey": the
   mean weight-at-age of GOV fish in the Skagerrak region, from the age-length key (step 5). (b)
   "sw_interp": stock weights interpolated 0.6 yr forward, sw(a)^0.4 sw(a+1)^0.6. The age-0 weight
   in Q3 always comes from the survey with the Fraser-corrected length mix. The two variants change
   the age shares by at most 3 points (`stock_age_biomass_share.csv`).
5. Age-length key: DATRAS CA records in the Skagerrak rectangles (43F8-46G1 list in the script),
   NS-IBTS Q1/Q3 2021-2025, give P(age | length). They were applied to the region's haul-mean
   CPUE-at-length. The result is n(L, age), raw and with Fraser q (`stock_alk_nla.csv`). Coverage is
   good: 330-760 aged fish per age 0-2 and quarter for haddock, whiting and cod. Saithe has only
   23-40 at ages 1-2. Hake has no ages.
6. Stock biomass by length class: B(a) spread over length with the region's biomass-weighted
   P(L | age). So the "stock length structure" uses Skagerrak growth. Skagerrak haddock grow faster
   than the North Sea mean. In Q3, age 1 weighs 220 g vs 133 g, and age 2 493 g vs 301 g (CA means).
7. Pelagic 0-group: N0 in mid-June (t = 0.45) x weight at 5 cm (box length-weight fits). Early
   mortality is front-loaded, so the model N0 in June is a lower bound. The table also gives a case
   with 3 x M0 over Jun-Aug (`stock_pelagic_0group.csv`).

Caveats:
- Fraser q was calibrated against assessment (VPA) numbers. So agreement between the Fraser-corrected
  survey and the assessment is partly built in. The informative test is raw survey vs stock.
- The absolute N0 depends on the assumed M0 (N0 = N1 exp(Z0)). If the Aug-Dec mortality were twice
  M0, N0(Q3) would rise x1.5 for haddock and x1.8 for whiting. Their Q3 age-0 shares would then be
  2.4 % and 18 %.
- The P(L | age) from GOV counts carries the gear's within-age selectivity. Raw and Fraser keys differ
  by < 1 point in every class except haddock age 0.
- Whiting age-0 weight: 11 g in the Skagerrak region (median 10.5 cm). The North Sea-wide CA mean is
  22 g, but that mean is length-stratified, so it is biased upward. With 22 g the stock Q3 age-0 share
  would be about 20 % instead of 11 %.

## 3. Per-stock results

### 3.1 Haddock had.27.46a20 [F][D]
Source: SAM run `NShaddock_WGNSSK2026_Run1` on stockassessment.org (ages 0-8+, 1972-2026). The 2024
run (`NShaddock_WGNSSK2024_Run2`) was used as a sensitivity.
- Recruitment (age 0) was 7.8-8.7 billion in 2019-2020. It was 1.0-1.5 billion in 2023-2025 (SAG 22417).
- Q1: age 1 is 9.5 % of the biomass (15.7 % for 2019-2024), age 2 is 23 %.
- Q3: age 0 is 1.6 % (3.8 %), age 1 is 11 % (19 %), age 2 is 22 %.
- Length-at-age in the Skagerrak region [D]. Q1 age 1: median 18.5 cm (p10-p90 15.5-22.5).
  Q1 age 2: 26.5 cm (20.5-35.5). Q3 age 0: 13.5 cm (11.5-15.5) raw, 11.5 cm with Fraser q.
  Q3 age 1: 26.5 cm (20.5-32.5). Q3 age 2: 33.5 cm.
- Stock biomass by length class, 2021-2025. Q1: < 15 cm 0.3 %, 15-20 cm 5.0 %, 20-30 cm 16 %.
  Q3: < 15 cm 1.3 %, 15-20 cm 0.5 %, 20-30 cm 8 %.
- The 2024 run gives larger young ages (Q3 age 0 2.5 %, Q1 age 1 16 %). The 2026 run revised the
  2023-2024 year classes down.

### 3.2 Whiting whg.27.47d [F][D]; whg.27.3a [F]
Source: SAM run `NSwhiting_2026n` (ages 0-8+). Its R, SSB and TSB match SAG key 22488 (the 2026
advice) exactly. The final 2025 run `NSwhiting_2025` (ages 0-6+) was used as a sensitivity. Results
differ by <= 1.5 points.
- Q1: age 1 is 23 % of the biomass, age 2 is 22 %.
- Q3: age 0 is 11 % (15 % for 2019-2024), age 1 is 21 %, age 2 is 20 %.
- Length-at-age in the Skagerrak region. Q1 age 1: 15.5 cm (13.5-18.5). Q1 age 2: 21.5 cm.
  Q3 age 0: 10.5 cm (8.5-13.5), 11 g. Q3 age 1: 20.5 cm (14.5-23.5).
- Stock biomass by length class, 2021-2025. Q1: < 20 cm 25 %, 20-30 cm 36 %. Q3: < 10 cm 1.7 %,
  < 20 cm 17 %, 20-30 cm 36 %.
- whg.27.3a has no analytical assessment. SURBAR showed "internal consistency was virtually absent".
  SPiCT had "very high" uncertainty. Advice uses the rfb rule since 2026 [F, WGNSSK 2026 sec. 21].
  Stock identity is unclear. The section notes "locally spawned whiting eggs in Kattegat". The Danish
  sprat (industrial) fleet "has landed 40-80% of whiting in the area in earlier years", falling to
  16-18 % in 2024-2025. That fleet catches small whiting. Length at first capture for the rfb rule is
  9.0 cm. No age or length composition table is published in the section, so it gives no numbers
  for this check.

### 3.3 Cod: Viking substock of cod.27.46a7d20 [F][D]
- The Northern Shelf cod is assessed as three substocks (Northwestern, Viking, Southern) in a
  multistock SAM. The substock map in WGNSSK 2026 section 4 puts **3.a.20 (Skagerrak) in the Viking
  substock**, together with the eastern part of 4.a. Age 0 is not modelled. Age-0 cod from the Q3+Q4
  surveys are kept "as separate recruitment indices (one for each substock) and forward shifted to
  1st January to represent age 1 cod the following year".
- Numbers-at-age per substock (Table 4.9), F (4.8), estimated stock weights (4.11) and M (4.13) are
  published for 1983-2026. They were parsed from the PDF text.
- Viking, 2021-2025. Q1: age 1 is 7.2 % of the biomass, age 2 21 %, age 3 23 %. Q3: age 0 1.7 %
  (M0 range 1.4-2.5), age 1 11.5 %, age 2 24 %. Stock weight at age 1 is 0.05-0.06 kg. At age 7+ it is
  about 10 kg, so large cod dominate the substock biomass (> 70 cm: 35 %).
- Length-at-age in the Skagerrak region. Q1 age 1: 17.5 cm (13.5-23.5). Q1 age 2: 34.5 cm.
  Q3 age 0: 9.5 cm (7.5-11.5), 8 g. Q3 age 1: 26.5 cm (21.5-31.5).
- Northwestern and Southern substocks give similar young-age shares (Q1 age 1: 12-15 %).
- The box also holds local coastal (fjord) cod. The assessment does not separate them.

### 3.4 Saithe pok.27.3a46 [F][D]
- The SAM assessment starts at age 3 (WGNSSK 2026 Tables 14.3.11, 14.4.2, 14.4.3; M 0.38 at age 3 to
  0.18 at age 10+). Age 3 is 24 % of the biomass in Q1 and 27 % in Q3 (2021-2025). The stock weight at
  age 3 is 0.44-0.77 kg, about 38-45 cm in the Skagerrak key.
- Ages 0-2 are not in the model. WGNSSK 2026: "fish begin migrating out of the inshore nursery areas
  at age 3, but do not fully recruit to the more oceanic population (and fishery) until after age 5"
  [F]. FAO: "During their first 2-3 years of age, saithe remain in shallow coastal waters" [S, quoted
  from the brief, not re-checked].
- The stock length structure has 0 % below 25 cm and 0-0.5 % at 25-30 cm. The box survey agrees.
  In-box young saithe are near zero.

### 3.5 Hake hke.27.3a46-8abd [F, SAG only][D]
- SS3 assessment. N-at-age was not fetched (hake is 0.85 % of the box). SAG 22627: recruitment (age 0)
  averaged 459 million in 2021-2025. TSB averaged 248 kt.
- Age-0 biomass in Q3 = R exp(-0.4 x 0.62) x weight at 8-12 cm (3-10 g): 0.4-1.5 % of TSB.
- The box survey (Fraser q is constant for hake, so it does not change shares) gives < 20 cm 3.5 %
  in Q1 and 1.5 % in Q3, and 20-30 cm 24 % in Q1 and 6 % in Q3.

## 4. Comparison with the box survey

Percent of species biomass, 2021-2025 (`stock_vs_box_comparison.csv`). "Stock" uses the raw-key
length mapping. The Fraser key differs by < 1.2 points.

| Species | Q | Class | Stock 2021-25 | Stock 2019-24 | Box raw | Box Fraser |
|---|---|---|---|---|---|---|
| Haddock | 1 | < 20 cm | 5.3 | 8.6 | 7.7 | 11.6 |
| Haddock | 1 | 20-30 cm | 16.1 | 20.0 | 26.1 | 26.2 |
| Haddock | 3 | < 20 cm | 1.8 | 4.0 | 3.1 | 10.3 |
| Haddock | 3 | 20-30 cm | 8.2 | 12.4 | 16.5 | 12.3 |
| Whiting | 1 | < 20 cm | 25.0 | 27.2 | 16.4 | 10.4 |
| Whiting | 1 | 20-30 cm | 35.9 | 37.3 | 44.1 | 31.1 |
| Whiting | 3 | < 20 cm | 17.0 | 21.7 | 28.8 | 20.9 |
| Whiting | 3 | 20-30 cm | 36.4 | 37.7 | 50.0 | 39.7 |
| Cod (Viking) | 1 | < 20 cm | 3.3 | 4.6 | 0.8 | 1.1 |
| Cod (Viking) | 3 | < 20 cm | 1.8 | 2.7 | 2.2 | 3.2 |
| Cod (Viking) | 3 | 20-30 cm | 7.9 | 11.0 | 17.2 | 24.2 |
| Saithe | 1, 3 | < 30 cm | 0-0.5 | 0-0.5 | 0.1-0.8 | 0.1-0.4 |

### 4.1 Haddock: the stock check does not confirm the large Fraser up-correction of small fish
- The raw box shares of small haddock are already at or above the stock-level shares, in both quarters.
- Numbers are the clearest test. In Q3, age 0 is 53 % of the region's GOV haddock with the raw key and
  86 % with the Fraser key. At stock level it is 32 % (2021-2025) or 50 % (2019-2024). That is 0.5-1.0
  age-0 fish per older fish in the stock, 1.1 with the raw key and 5.9 with the Fraser key. So the
  Fraser key needs the box to hold 6-13 x the stock-average 0-group per older fish. The raw key needs
  only 1.1-2.4 x.
- Biomass gives a weaker test. Age-0 haddock in Q3 are 11.5-15.5 cm, where Fraser q runs from 0.012
  to 0.11. In the box table the Fraser correction lifts the Q3 10-15 cm class from 2.3 % to 8.8 %
  (x 3.8). Under Fraser q, a box with the stock-average age structure (age 0 1.6-3.8 %) would show
  0.4-1.0 % in raw catches. It shows 2.3 %, which implies a 2-6 x local enrichment. That is possible
  for a Skagerrak nursery.
- So the stock check cannot fully separate catchability from nursery enrichment. The numbers ratio
  favours the raw end: Fraser's q for small haddock looks too low here.
- One possible cause: Fraser fitted q against VPA numbers of 1998-2004. Those numbers used other M0
  assumptions, and a higher M0 gives a larger N0 and a smaller q. This is a hypothesis and was not
  checked.
- Recommendation: use the raw-to-stock range for haddock < 20 cm in Q3 (2-4 %). Treat the Fraser value
  (10 %) as an upper bound.

### 4.2 Whiting: Fraser agrees with the stock in Q3; the box is poor in small whiting in Q1
- Q3: stock 17-22 % < 20 cm, Fraser 21 %, raw 29 %. Whiting q is flat (0.62) up to 21 cm and low
  (0.14) for large fish. So raw counts over-weight small fish, and the Fraser correction is the right
  direction. The agreement is partly built in (Fraser was calibrated on VPA).
- Q1: stock 25-27 %, raw 16 %, Fraser 10 %. Both survey versions are below the stock. This is
  geography, not catchability: the box is deep, and age-1 whiting in winter lie mostly elsewhere.

### 4.3 Cod: the box has fewer age-1 cod in Q1 and more in Q3 than the Viking substock
- Q1 < 20 cm: stock 3.3-4.6 %, box 0.8-1.1 %. Q3 20-30 cm (age 1): stock 8-11 %, box 17-24 %. Fraser
  cod q is flat up to 25 cm, so it barely moves the small classes. The difference is local: the
  Skagerrak holds age-1 cod in summer. Large cod (> 70 cm, 35 % of the Viking biomass) are rarer in
  the box (14-20 % in the raw survey).

### 4.4 Saithe and hake
- Saithe agree: below 30 cm the stock (ages 3+ only) and the box are both <= 0.8 %.
- Hake: the stock-level age-0 share (0.4-1.5 %) is the same order as the box < 20 cm share in Q3 (1.5 %).

## 5. Pelagic 0-group (Jun-Jul, 3-7 cm) [D]

`stock_pelagic_0group.csv`, 2021-2025, as percent of the Q3 stock biomass. The first value uses the
model M0. The value in brackets uses 3 x M0 over Jun-Aug.
- Haddock: 0.11 % (0.16 %). For 2019-2024: 0.27 % (0.39 %).
- Whiting: 1.2 % (1.9 %). For 2019-2024: 1.6 % (2.7 %).
- Cod (Viking): 0.26 % (0.44 %). The M0 range 1.0-2.5 gives 0.20-1.05 %.
- Saithe: 0 in the box. Hake: not computed (spawns offshore in summer; tiny share of the box).
Whiting stays pelagic longest and has the largest year classes. This makes it the only gadoid whose
pelagic stage is more than 1 % of its biomass.

## 6. Source status

| Source | What was used | Status |
|---|---|---|
| stockassessment.org, run `NShaddock_WGNSSK2026_Run1` (public run directory; site listing via the advertised guest login) | tab2 F, tab3 N, sw/cw/nm .dat | [F] downloaded, parsed |
| stockassessment.org `NShaddock_WGNSSK2024_Run2` | same, sensitivity | [F] |
| stockassessment.org `NSwhiting_2026n` (= SAG 22488), `NSwhiting_2025` (final 2025) | same | [F] |
| stockassessment.org `NS_saithe_2024_benchmark_final`, `nscod20_ass02` | downloaded; not used (superseded by WGNSSK 2026 tables / substock model) | [F] |
| ICES WGNSSK 2026, sec. 4 cod (figshare file 69947211; doi 10.17895/ices.pub.32676345.v2) | substock map, Tables 4.7c, 4.8, 4.9, 4.11, 4.13; recruitment-index text | [F] |
| WGNSSK 2026, sec. 14 saithe 3a46 (file 69947238) | Tables 14.3.11, 14.4.2, 14.4.3, 14.6.1; M; nursery text | [F] |
| WGNSSK 2026, sec. 21 whiting 3a (file 69947250) | stock status, fisheries, rfb text | [F] |
| ICES SAG API, SummaryTable keys 22417, 22488, 19484, 22504, 22627, 26001-26003, 26010, 22563, 22381 | R, SSB, TSB, recruitment age | [F] |
| DATRAS CA / CPUEL / HH extracts (sibling folder `Torskfiskarnas artsammansättning`) | age-length keys, weights | [F][D] |
| Fraser, Greenstreet & Piet 2007 q functions | via the sibling `evidence_B_catchability.md` | [S] |
| FAO saithe fact sheet quote | from the brief | [S], not re-checked |
| WGSAM SMS key run (quarterly N and weight by age) | figshare 27919473, 31224250 and stock annex 18623579 located | not downloaded (time); would give quarterly age-0 weights and M0 |
| ICES TAF assessment repositories | GitHub search found only WKNSMSE repositories | not public |
| WGBIE hake SS3 numbers-at-age | not pursued (hake 0.85 % of the box) | - |

Reproduce (from the repository root, with B = `research_notes/Torskfiskarnas bytesandelar`):
`scripts/stock_fetch_sam.sh`, `scripts/stock_fetch_sag.sh`, pdftotext of the three PDFs, then
`python3 -I $B/scripts/stock_parse.py $B/data/stock $B/tables`,
`python3 -I $B/scripts/stock_age_length.py "research_notes/Torskfiskarnas artsammansättning" $B/tables`,
`python3 -I $B/scripts/stock_age_shares.py $B/tables $B/data/stock "research_notes/Torskfiskarnas artsammansättning"`,
`python3 -I $B/scripts/stock_compare.py $B/tables`. Run them from a directory outside `data/`.
