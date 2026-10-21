# Stock trends and total fishing mortality of the gadoids in the Skagerrak / North Sea (ICES 3.a, 4)

Scope note:
- **ICES assessment numbers.** Full summary tables (1972/1995-2026) were re-downloaded from the ICES SAG REST API for
  the 2026 assessments. The keys are the ones the sibling note `research_notes/Fiskeridödlighet i Skagerrak idag/ices_gadoid_f.md`
  used: 22417 haddock, 22488 North Sea whiting, 22563 whiting 3.a, 22504 saithe, 22627 northern hake,
  26001-26003 Northern Shelf cod substocks (Viking, Southern, Northwestern) and 22381 Kattegat cod. (Fetched full text, JSON.)
- **Survey numbers.** A new NS-IBTS computation for 2005-2025, Q1 and Q3, from the DATRAS web service
  (`getCPUELength`). Numbers per hour at length were converted to kg/h with the length-weight fits in
  `ices_gadoid_f.md`. Two areas: "Skagerrak" (east of the Hanstholm-Lindesnes line, Kattegat south of 57.75 N excluded,
  20-30 hauls per quarter) and "Box" (ICES rectangles 44G0, 44G1, 45G0 and 45G1, which hold the model grid; 8-16 hauls
  per quarter). (Computed. These are survey indices, not absolute biomass. The GOV trawl under-samples large cod, deep
  hake and pelagic saithe.)
- **Coastal cod.** The literature on the coastal cod comes from a literature sub-search. Each item carries its fetch
  status.
- **Labels.** "Computed" marks my arithmetic. "Inference" marks my reading of it.

## Takeaway

- **The pool is not stationary. It sits near the top of a recruitment pulse.**
  - From about 2008 to 2020 the biomass-weighted pool was close to flat. The SSB-based pooled trend was +2 %/yr in
    2010-2020. The Skagerrak IBTS gadoid index was -1.7 %/yr in 2005-2015 and moved between 130 and 310 kg/h.
  - Two very strong haddock year classes (2019 and 2020), followed by strong whiting year classes, roughly doubled the
    pool between 2020 and 2023. Haddock SSB rose from 176 kt to 790 kt, North Sea whiting SSB from 202 kt to 395 kt, and
    the Skagerrak IBTS index from about 210 to 440-520 kg/h.
  - Since 2023 the pool has drifted down slowly: SSB-based -3.4 %/yr over 2023-2026. Haddock recruitment is back to
    normal (1.0-2.0 billion age-0, against 7.8-8.7 billion in 2019-2020), so the pulse is ageing out.
- **Species differ.**
  - Haddock: lightly fished (F 0.06-0.08) and at a historic high, now declining.
  - Whiting: high and roughly flat.
  - Saithe: declining for 15 years, SSB from 281 kt (2008) to 105-117 kt (2024-2026), now below Blim after a maturity
    revision.
  - Northern hake: declining since 2015 (TSB from 514 kt to 181 kt, about -10 %/yr).
  - Cod: depleted. Fbar is 2x FMSY on every substock and SSB is below MSY Btrigger on all three. In the Skagerrak IBTS,
    cod fell from 60-110 kg/h (2012-2016) to 12-34 kg/h (2019-2025).
  - Kattegat cod is at a historic low, with zero-catch advice.
- **Nothing in the data looks like the pool growing at about 35 %/yr for four years.** The one rise of that size
  (2020-2023) was a step change carried by two year classes born at low SSB. It was not compounding growth of standing
  biomass.

## Cited Findings: ICES assessments (2026, terminal year 2025)

Values are year: SSB (t) / recruitment / Fbar.

**Haddock had.27.46a20** (SAG 22417). Recruitment is at age 0, in thousands.
- 2010: 141 211 / 0.78 bn / 0.238
- 2015: 128 860 / 1.46 bn / 0.350
- 2018: 162 839 / 2.75 bn / 0.241
- 2019: 133 189 / **8.74 bn** / 0.228
- 2020: 175 991 / **7.81 bn** / 0.186
- 2021: 312 079 / 2.43 bn / 0.130
- 2022: 590 846 / 5.76 bn / 0.080
- 2023: 790 487 / 1.33 bn / 0.064
- 2024: 770 388 / 1.53 bn / 0.060
- 2025: 751 318 / 1.03 bn / 0.079
- 2026: SSB 699 810
- Earlier strong year class: 1999 at 24.8 bn.
- Status 2026: "Fishing pressure on the stock is below FMSY, and spawning-stock size is above MSY Btrigger, BPA, and
  Blim." [ICES Advice 2026 had.27.46a20](https://doi.org/10.17895/ices.advice.30932057); [SAG 22417](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22417) (Fetched full text)

**Whiting whg.27.47d** (SAG 22488).
- 2010: 136 549 / 7.7 bn / 0.251
- 2015: 131 108 / 10.4 bn / 0.270
- 2019: 150 385 / 18.3 bn / 0.241
- 2020: 202 405 / 17.7 bn / 0.179
- 2021: 301 245 / 9.9 bn / 0.139
- 2022: 364 075 / 11.5 bn / 0.095
- 2023: 395 137 / 10.0 bn / 0.085
- 2024: 372 758 / 14.5 bn / 0.078
- 2025: 341 967 / 9.7 bn / 0.093
- 2026: SSB 385 516
- SSB was 120-150 kt from 2007 to 2019, below the 1990s level (approximately 160-245 kt).
- [SAG 22488](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22488) (Fetched full text)

**Whiting whg.27.3a** (SAG 22563), relative biomass index only.
- 2001-2004: 1.3-1.6
- 2009-2013: 0.54-0.63
- 2015-2023: 0.92-1.08
- 2024: 1.16
- 2025: 1.69
- [SAG 22563](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22563) (Fetched full text)

**Saithe pok.27.3a46** (SAG 22504). Recruitment is at age 3.
- 2008: 281 331 / 0.442 F
- 2010: 236 310 / 0.399
- 2015: 188 946 / 0.348
- 2020: 170 576 / 0.384
- 2022: 133 980 / 0.351
- 2024: 104 613 / 0.385
- 2025: 116 722 / 0.414
- 2026: SSB 115 676
- Recruitment at age 3 was 46-82 million in 2018-2021, against 115-280 million in 1995-2017.
- Status: "Fishing pressure on the stock is above FMSY and FPA, and spawning-stock size is below MSY Btrigger, BPA, and
  Blim." [ICES Advice 2026 pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291); [SAG 22504](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22504) (Fetched full text)

**Northern hake hke.27.3a46-8abd** (SAG 22627). This is the only stock in the set with TSB. Values are TSB / SSB / F.
- 2005: 106 kt / 42 kt / 0.56
- 2010: 421 / 201 / 0.185
- 2015: **514** / 293 / 0.186
- 2020: 339 / 190 / 0.205
- 2025: 194 / 109 / 0.236
- 2026: 181 / 99
- The large rise in 2006-2015 followed a cut in F from 0.5 to 0.17-0.19 together with strong 2006-2008 year classes.
- Status 2026: "Fishing pressure on the stock is at FMSY, and spawning-stock size is above MSY Btrigger".
  [ICES Advice 2026 hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108); [SAG 22627](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22627) (Fetched full text)

**Northern Shelf cod cod.27.46a7d20, 2026 multistock SAM after the 2025-26 benchmark.** Recruitment is at age 1, in
thousands. The benchmark rescaled the whole series, so these numbers are NOT continuous with the pre-2023
cod.27.47d20 series.
- **Viking** (SAG 26001; ICES assigns Skagerrak catches to it):
  - 2010: 15 415 / 139 k / 0.511
  - 2017: 33 228 / 125 k / 0.549
  - 2020: 14 326 / 158 k / 0.397
  - 2023: 19 269 / 63 k / 0.364
  - 2024: 19 417 / **19 k** / 0.418
  - 2025: 18 216 / 48 k / 0.426
  - 2026: SSB 16 594
  - FMSY is 0.198 and Blim 14 027 t.
- **Southern** (SAG 26002): SSB fell from 26 945 (2009) to 3878 (2020), then 8741 (2025). Fbar is 0.51.
- **Northwestern** (SAG 26003): SSB peaked at 49 873 (2017), then fell to 26 779 (2025) and 21 367 (2026).
- 2026 status: "Fishing pressure on all three substocks is above FMSY and FPA. For the Southern and Northwestern
  substocks, spawning stock biomass is below MSY Btrigger and Blim; the Viking substock is above Blim and below MSY
  Btrigger." Advice: zero catch in 2027.
- In the 2026 series, Viking recruitment for 2021-2026 (19-97 k) is below its 1995-2017 range (46-418 k).
- [ICES Advice 2026 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018); [SAG 26001](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26001), [26002](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26002), [26003](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26003) (Fetched full text)

**Kattegat cod cod.27.21** (SAG 22381). Relative SSB:
- 1997: 3.82
- 2005: 1.15
- 2009: 0.20
- 2015: 1.62 (a short recovery)
- 2020: 0.099
- 2025: 0.155
- 2026: 0.186

Relative F is about 0.9-1.45 throughout, and catch was 82 t in 2025. The advice is zero catch in 2027-2028.
[ICES Advice 2026 cod.27.21](https://doi.org/10.17895/ices.advice.30932012); [SAG 22381](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=22381) (Fetched full text)

## Computed: biomass-weighted pooled trend from the assessments

Method:
- Each stock's SSB (TSB for hake) is indexed to the start year.
- The indices are weighted with the sibling note's Skagerrak survey shares W1 (haddock 0.543, whiting 0.267, saithe
  0.121, cod 0.058, hake 0.011), with the Viking substock for cod.
- The trend is the log change per year of the weighted index.
- SSB lags TSB by a year or two after a recruitment pulse, so the SSB trends understate the speed of the 2020-2022 rise.

| Period | Pooled trend (/yr) | Haddock | Whiting NS | Saithe | Hake (TSB) | Cod Viking |
|---|---|---|---|---|---|---|
| 2010-2020 | **+0.020** | +0.022 | +0.039 | -0.033 | -0.022 | -0.007 |
| 2010-2025 | +0.087 | +0.111 | +0.061 | -0.047 | -0.052 | +0.011 |
| 2015-2025 | +0.138 | +0.176 | +0.096 | -0.048 | -0.097 | -0.027 |
| 2020-2025 | +0.215 | +0.290 | +0.105 | -0.076 | -0.112 | +0.048 |
| 2023-2026 | **-0.034** | -0.041 | -0.008 | -0.048 | -0.097 | -0.050 |

## Computed: NS-IBTS gadoid biomass index in the Skagerrak and in the model's rectangles, 2005-2025

Mean of Q1 and Q3, in kg/h (wing-spread scaling in `ices_gadoid_f.md`: 1 kg/h ≈ 7.1 kg/km2 at catchability 1).

| Year | Skagerrak total | cod | haddock | whiting | saithe | hake | Box 44G0-45G1 total |
|---|---|---|---|---|---|---|---|
| 2005 | 213 | 42 | 44 | 79 | 44 | 4 | 188 |
| 2007 | 202 | 87 | 37 | 39 | 36 | 4 | 197 |
| 2009 | 163 | 34 | 35 | 78 | 13 | 4 | 116 |
| 2011 | 129 | 21 | 43 | 32 | 30 | 3 | 103 |
| 2012 | 268 | 109 | 86 | 44 | 26 | 2 | 196 |
| 2014 | 160 | 81 | 21 | 45 | 9 | 4 | 146 |
| 2016 | 315 | 104 | 15 | 159 | 15 | 22 | 335 |
| 2018 | 152 | 32 | 5 | 85 | 22 | 8 | 131 |
| 2019 | 163 | 9 | 9 | 128 | 10 | 6 | 128 |
| 2020 | 278 | 34 | 70 | 147 | 20 | 7 | 233 |
| 2021 | 211 | 33 | 74 | 85 | 15 | 3 | 149 |
| 2022 | 437 | 27 | 182 | 118 | 107 | 3 | 506 |
| 2023 | 456 | 28 | 336 | 75 | 14 | 4 | 336 |
| 2024 | 520 | 19 | 342 | 104 | 50 | 6 | 376 |
| 2025 | 434 | 13 | 183 | 168 | 64 | 7 | 506 |

Log-linear trends of the total:

| Period | Skagerrak | Box |
|---|---|---|
| 2005-2015 | -0.017 /yr | +0.002 /yr |
| 2005-2025 | +0.040 /yr | +0.041 /yr |
| 2015-2025 | +0.094 /yr | +0.068 /yr |

By species in the Skagerrak, 2015-2025: cod -0.168, haddock +0.41, whiting +0.03, saithe +0.10, hake -0.085.

Year-to-year noise is large: 8-30 hauls a year, and saithe comes in single big hauls, as in 2022. Treat each year as
roughly ±30 %.

### Inferences (stock trends)

1. **"Roughly stationary" holds for 2005-2020, not for 2021-2025.**
   - In the decade before the haddock pulse, the local survey index and the assessments agree on near-zero net change
     of the pooled biomass, to within ±2 %/yr.
   - That is the period in which today's F regime (pooled F about 0.12, `ices_gadoid_f.md`) was roughly in place:
     haddock F 0.2-0.35, whiting 0.2-0.27, saithe 0.35-0.4.
   - Pooled F was actually higher than today in 2010-2020, because haddock and whiting F were 2-4x their 2023-2025 values.
     Inference: the pool was stationary at a somewhat higher F than the model's 0.12.
2. **The 2020-2023 doubling was recruitment-driven.**
   - The haddock 2019 and 2020 year classes were 4-6x the 2010-2018 median. They were spawned by an SSB of 133-176 kt,
     a low SSB for this stock.
   - So the rise came from favourable recruitment conditions, not from a large spawning stock compounding its biomass.
   - The pool now loses biomass slowly as these fish age while F stays low. That is the opposite of what the model does
     from the same starting point.
3. **The model's start (0.69 t/km2) corresponds to which survey level?**
   - If the 2500 t was set from a pre-2022 view, it matches the 2005-2021 plateau (about 130-310 kg/h in the Skagerrak).
   - If it was set from today, it should be about 2x that plateau.
   - Either way, the past 20 years give no example of the pool growing 2.7-fold in four years from a steady start.
   - The IBTS-to-absolute conversion (`ices_gadoid_f.md`) is unresolved by about 4x, so these indices give trends, not
     levels.
4. **The species least like "growing" are the piscivores that eat herring**: cod, saithe and hake. All three decline.
   The pool's net growth comes from haddock (a benthivore, about 0 % herring in its diet in SMS; see
   `predator_prey_cycles.md`) and whiting. A pool that grows by eating herring therefore misrepresents which members are
   growing.

## Coastal cod (literature)

The sources below come from a literature sub-search; fetch status is given for each. Numbers are verbatim from the sources unless marked.

### Takeaway
- **The Skagerrak coastal cod is badly depleted on both the Norwegian and the Swedish side.**
- **Norwegian side, Skagerrak coast:**
  - Commercial landings have fallen 91 % in the west and 98 % in the east.
  - Gillnet CPUE has fallen about 75-80 % since 2006.
  - Total mortality Z has risen from about 1.0 in the 1980s to about 1.5 /yr now (annual survival about 22 %).
- **Swedish side:** large cod has practically vanished from the Bohuslän fjords.
- **Kattegat:** landings went from 20 000 t (1977) to 22 t (2023).
- **Main cause given by the authors: fishing, and on the Norwegian coast mostly recreational fishing.**
  - Recreational fishing takes about 72 % of the harvest in a tag-return study.
  - Seal and cormorant predation are judged minor for cod, though one 2016 working group pointed to cormorants.
  - Warming limits juvenile growth in the coastal Skagerrak.
  - Recruitment failure matters in the east, but not in the west.
  - Eelgrass habitat is not a current cause: cover is the highest since the 1930s.
- **Relevance to the model (Inference):**
  - These are inshore populations (fjord and coastal ecotypes, plus North Sea-origin fish settling inshore), largely inside the baseline.
  - The model box is offshore, at 180-450 m on the Norwegian Trench slope.
  - The coastal Z of 1.0-1.75 should not be carried offshore.
  - Two things do carry over: the cod component of the pool is depleted, and it is not growing.

### Cited Findings
- Stock B, Knutsen H, Nedreaas K, Otterå H, Fall J, Denechaud C, Moland E, Jorde PE, Kleiven AR 2025. "Oppdatert status om torsk langs Vestlandet- og Skagerrakkysten: potensial for bestandsvurdering". Rapport fra havforskningen 2025-42, Institute of Marine Research. [hi.no](https://www.hi.no/hi/nettrapporter/rapport-fra-havforskningen-en-2025-42) (Fetched full text)
  - Landings and CPUE:
    - "Commercial catches of 2000-4000t were sustained from 1914-2000 but have since declined to 250t. Declines by area: Vestlandet 88%, Skagerrak West 91%, Skagerrak East 98%."
    - CPUE: "declines of ca. 75-80% in Skagerrak".
  - Mortality:
    - "Total mortality estimates from Skagerrak are substantially higher and have increased from ca. Z = 1.0 in the 1980s to ca. Z = 1.5 in recent years (annual survival 22% yr-1)."
    - Gillnet-survey Z in 2017-24: 1.32 in Skagerrak West, 1.75 in Skagerrak East.
  - Causes:
    - "fishing mortality is likely the most important cause of population declines".
    - Recruitment is "unlikely the primary driver of decline in Skagerrak West"; in the east "poor recruitment may be more important".
    - "Growth conditions are not limiting cod."
- Aglen A, Nedreaas K, Moland E, Knutsen H, Kleiven AR, Johannessen T, et al. 2016. "Kunnskapsstatus kysttorsk i sør (Svenskegrensa - Stadt)". Fisken og Havet 4/2016, IMR. [PDF (Fiskeridirektoratet)](https://www.fiskeridir.no/Yrkesfiske/Dokumenter/reguleringsmotet/Juni-2016/_/attachment/download/e2a8fa35-618a-4ef9-b212-f26d78406b3f:eb3f9713689ecf9280463c3f539255d2c1b0a094/sak-3-2016-oekosystembasert-forvaltning-vedlegg-6.pdf) (Fetched full text)
  - Flødevigen beach seine: "Fra 1999 til 2010 var fangstene av 0-gruppe ... og 1-gruppe torsk i strandnot redusert med hhv. 43 og 85 % i forhold til langtidsgjennomsnittet (1919-2010)."
  - Older cod: "Forekomsten av eldre torsk har siden 2001 vært historisk lav, rundt 10% av langtidsgjennomsnittet."
  - Catch split (preliminary): "yrkesfiskere 10 %, turistfiskere 15 % og norske fritidsfiskere 75 %".
  - Seals: "selens torskekonsum ... ikke kvantifiseres blant de største dødelighetsfaktorene".
  - It names increased harvest "sammen med økende predasjon fra skarv" as main factors.
- Kleiven AR, Fernández-Chacón A, Nordahl JH, Moland E, Espeland SH, Knutsen H, Olsen EM 2016. "Harvest pressure on coastal Atlantic cod (*Gadus morhua*) from recreational fishing relative to commercial fishing assessed from tag-recovery data". PLoS ONE 11(3): e0149595. [doi](https://doi.org/10.1371/journal.pone.0149595) (Fetched full text)
  - "We estimate the overall annual survival to be 33% (SE 1.5). Recreational rod and line fishing were responsible for 33.7% (SE 2.4) of total mortality, followed by commercial fisheries (15.1% SE 0.8) and recreational fixed gear (6.8% SE 0.4). Natural mortality was 44.4% (SE 2.5) of total mortality."
  - "72% of harvested coastal Atlantic cod was caught by recreational fishers".
  - IMR 2025-42 converts this to Z 1.11, F 0.62 (recreational 0.45, commercial 0.17) and M 0.49. That conversion is IMR's, not Kleiven's.
- Fernández-Chacón A, Moland E, Espeland SH, Kleiven AR, Olsen EM 2017. "Causes of mortality in depleted populations of Atlantic cod estimated from multi-event modelling of mark-recapture and recovery data". CJFAS 74:116-126. [doi](https://doi.org/10.1139/cjfas-2015-0313) (Fetched abstract)
  - "annual survival remained low and stable (~0.3)".
  - Survival would be "above 0.5 if only one or both of the dominant gear types were removed".
- Fernández-Chacón A, Moland E, Espeland SH, Olsen EM 2015. "Demographic effects of full vs. partial protection from harvesting: inference from an empirical before-after control-impact study on Atlantic cod". J. Appl. Ecol. 52:1206-1215. [doi](https://doi.org/10.1111/1365-2664.12477) (Fetched abstract)
  - Inside the MPA, the share of deaths due to fishing fell "from 0.59 ... to 0.32".
- Knutsen H, Espeland SH, Moland E 2022. "Evaluering av tiltak for vern av kysttorsk i sør innført juni 2019". Rapport fra havforskningen 2022-48. [hi.no](https://www.hi.no/hi/nettrapporter/rapport-fra-havforskningen-2022-48) (Fetched; tool summary)
  - "Vi ser i våre data ingen tegn til at bestandssituasjonen for torsk er endret".
- Langangen Ø, Kaartvedt S, Hylland K, Durant JM 2026. "The collapse of Atlantic cod in the Oslofjord". Frontiers in Marine Science. [doi](https://doi.org/10.3389/fmars.2026.1746235) (Fetched abstract)
  - "cod biomass has been reduced by more than 80% from a peak in 2004-2005 to 2018. After 2018, the biomass was further reduced below 10% of the previous levels."
- SLU Aqua 2025, Fiskbarometern, Torsk. [fiskbarometern.se](https://fiskbarometern.se/rapport/2025/species/Torsk) (Fetched full text)
  - "Större torsk är numera så ovanlig i Bohusläns fjordar och skärgårdar att den inte längre fångas i de återkommande provfisketrålningarna."
  - Kattegat: "År 1977 fångades 20 000 ton torsk i Kattegatt ... År 2023 uppgick de totala landningarna till endast 22 ton."
  - On predators (citing ICES WGSAM): "För torskar som är tre till fyra år gamla bedöms nästan all predationsdödlighet komma från säl."
- Svedäng H, Bardon G 2003. "Spatial and temporal aspects of the decline in cod (*Gadus morhua* L.) abundance in the Kattegat and eastern Skagerrak". ICES J. Mar. Sci. 60:32-37. [doi](https://doi.org/10.1006/jmsc.2002.1330) (Fetched abstract)
  - Offshore cod shows "a persistent decline in offshore abundance since the beginning of the 1980s ... culminating in extremely low densities in the late 1990s".
- Cardinale M, Svedäng H 2004. "Modelling recruitment and abundance of Atlantic cod, *Gadus morhua*, in the eastern Skagerrak-Kattegat (North Sea): evidence of severe depletion due to a prolonged period of high fishing pressure". Fish. Res. 69:263-282. [doi](https://doi.org/10.1016/j.fishres.2004.04.001) (Title only)
- Svedäng H, Svenson A 2006. "Cod *Gadus morhua* L. populations as behavioural units: inference from time series on juvenile cod abundance in the Skagerrak". J. Fish Biol. 69 (Suppl. C):151-164. [doi](https://doi.org/10.1111/j.1095-8649.2006.01272.x) (Fetched abstract)
  - "In spite of a reduction in fishing pressure ... and high cod recruitment ... in 2001 and 2003, no recovery could be evidenced."
- Cardinale M, Svedäng H, Bartolino V, Maiorano L, Casini M, Linderholm HW 2012. "Spatial and temporal depletion of haddock and pollack during the last century in the Kattegat-Skagerrak". J. Appl. Ichthyol. 28:200-208. [doi](https://doi.org/10.1111/j.1439-0426.2012.01937.x) (Fetched abstract)
  - Haddock and pollack in Kattegat-Skagerrak were depleted by industrial fishing "several decades before the unprecedented warming trend".
- Rogers LA, Stige LC, Olsen EM, Knutsen H, Chan K-S, Stenseth NC 2011. "Climate and population density drive changes in cod body size throughout a century on the Norwegian coast". PNAS 108:1961-1966. [doi](https://doi.org/10.1073/pnas.1010314108) (Fetched abstract)
  - "Warm summer temperatures in the coastal Skagerrak have limited juvenile growth".
- Dehnhard N, Langset M, Aglen A, Lorentsen S-H, Anker-Nilssen T 2021. "Fish consumption by great cormorants in Norwegian coastal waters - a human-wildlife conflict for wrasses, but not gadids". ICES J. Mar. Sci. 78:1074-1089. [doi](https://doi.org/10.1093/icesjms/fsab004) (Fetched abstract)
  - "Estimated total fish consumption of cod and saithe by great cormorants was <1.7% of estimated fish stocks and <9% of that of human catches and therefore considered minor."
- Sørlie M, Nilssen KT, Bjørge A, Freitas C 2020. "Diet composition and biomass consumption of harbour seals in Telemark and Aust-Agder, Norwegian Skagerrak". Mar. Biol. Res. 16:299-310. [doi](https://doi.org/10.1080/17451000.2020.1751205) (Fetched abstract)
  - "Cod constituted 0.7% (combined index Qi) of the overall diet and 2.3% in terms of biomass ... Annual cod consumption was an estimated 7.1 tons, representing 5% of annual cod landings."

### Inferences (coastal cod)
- The coastal decline is a fishing story: high recreational F inside the baseline and long-term high commercial F in the 1980s-1990s, with regional warming and, in the east, poor recruitment. Predation is not the main driver in these studies.
- This is the opposite of a recovering stock. A model whose cod component grew 35 %/yr would contradict the data.
- But the pooled model group is mostly haddock and whiting offshore (`ices_gadoid_f.md` W1-W3 weights). The coastal cod sets no number for the pool. It says the cod share of the pool should decline or stay flat, not grow.


---

# Q2. Total fishing mortality on the gadoid pool: is F = 0.12 too low as an all-gear, all-area value?

Builds on `research_notes/Fiskeridödlighet i Skagerrak idag/`:
- `ices_gadoid_f.md`: pooled F 0.12, range 0.07-0.19, and a swept-area cross-check u ≈ 0.13.
- `local_fishery.md`: local catch/biomass of 0.26, range 0.2-0.4, from 2010-2020 45G0 landings.
- `catch_by_gear.md`: 98 % of gadoid F is taken by bottom gear.

None of that is redone here. The only additions are:
1. the history of the pooled F;
2. recreational and unassessed removals;
3. the conceptual question of fishing outside the grid;
4. a reconciliation of the 0.12 with the 0.26.

## Takeaway
- **As a present-day, all-gear, all-area rate on today's species mix, 0.12/yr is about right.**
  - Recreational removals add about 2-10 % to it.
  - Discards are already in the ICES catches it is built on.
  - A plausible upper bound is about 0.15-0.19.
- **0.12 is low in two other senses:**
  1. **Composition.** The pooled F depends on the species mix, which sets the weights.
     - Today's haddock-dominated mix gives 0.11.
     - The 2005-2019 survey mix (about 31 % cod, 37 % whiting, 17 % haddock) carries **0.20** at today's species F.
     - The Ecopath-1991 mix that the library's budget and diet come from carries about 0.17.
  2. **History.** In the decade when the pool was stationary (2010-2019), its biomass-weighted F was **0.23-0.30**, two to
     three times the present value.
- **Raising F alone cannot hold the pool.**
  - Zero growth at ideal food needs F ≈ r_max = 0.5. With the surplus under-realised seal and porpoise predation leave
    (resume 184.13), it needs about 0.55.
  - No documented all-gear value for the offshore pool reaches that. Cod alone carries 0.35-0.5, but it is 6-30 % of the pool.
  - The coastal Z of 1.0-1.75 is an inshore cod value (see Q1, coastal cod) and does not apply.
- **Fishing outside the grid is already inside a stock-level F.** An F estimated for a mixing stock is a per-capita rate
  averaged over the stock's whole range. Applying it to a local sample of that stock counts the fishing elsewhere once.
  Adding the stock F on top of a local F would count it twice.
  - The exception is net emigration of grid fish to areas fished harder. That is a migration term, not an F, and nothing
    in the sources quantifies it.

## Cited Findings
- **Recreational catch, Northern Shelf cod.** "Recreational catches are estimated to account for 2.2-6.0% of total
  removals between 2010 and 2024, but values are provisional and not included in the assessment because of the unknown
  age structure in recreational catches and high uncertainty."
  — [ICES Advice 2025 cod.27.46a7d20](https://doi.org/10.17895/ices.advice.27202566) (Fetched full text, via sub-search)
- **Radford Z, Hyder K, Zarauz L, et al. 2018.** "The impact of marine recreational fishing on key fish stocks in European
  waters". PLoS ONE 13(9): e0201666. For cod-347d: "3899t ±1528t ... MRF accounts for only 10% of total removals". Only
  17 % of that estimate was reconstructed from data.
  [doi](https://doi.org/10.1371/journal.pone.0201666) (Fetched full text, via sub-search)
- **Hyder K, Weltersbach MS, Armstrong M, et al. 2018.** "Recreational sea fishing in Europe in a global context". Fish
  and Fisheries 19:225-243. The often-quoted 27 % is for northern sea bass and **western Baltic** cod, not North Sea cod.
  [doi](https://doi.org/10.1111/faf.12251) (Fetched abstract, via sub-search)
- **Kattegat cod.** "Recreational catches are not included in the assessment; the number of these could be similar to
  that of commercial catches." — [ICES Advice 2024 cod.27.21](https://doi.org/10.17895/ices.advice.25019213) (Fetched full text, via sub-search)
- **Swedish recreational cod, sea and coast.** Kept plus released, all Swedish waters, 2023: 311 t ± 305 t. There is no
  Skagerrak/Kattegat split. — [SCB/HaV, Fritidsfiske 2023, JO 57 SM 2401](https://scb.se/contentassets/16f20f2be4e746c9bb9ac9c256b7711d/jo1104_2023a01_sm_jo57sm2401.pdf) (Fetched full text, via sub-search)
- **Norwegian Skagerrak coast.**
  - "Recreational fishing mortality along the Skagerrak coast has previously been estimated as 3x commercial fishing
    mortality."
  - The Kleiven et al. 2016 tagging-based F, as converted by IMR, "applies to this area and not further out where more
    commercial fishing takes place (between baseline and 12 nm)".
  — IMR Rapport fra havforskningen 2025-42 (see Q1) (Fetched full text, via sub-search)
- **Viking cod F differs between assessment years.**
  - 2025 advice (pre-benchmark): F(2-4) 0.28-0.34 in 2021-2024, FMSY 0.187.
  - 2026 SAG (post-benchmark): 0.32-0.42 for the same years.
  - The benchmark rescaled F, so the pooled cod contribution has an assessment-version uncertainty of about ±20 %.
  — [ICES Advice 2025](https://doi.org/10.17895/ices.advice.27202566); [SAG 26001](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=26001)
- **Discards.** The ICES catch figures behind the 0.12 include discards:
  - haddock 2025: 15 292 t of 69 839;
  - North Sea whiting: 16 544 t of 33 331;
  - 3.a whiting: 65 % discards;
  - cod: 4842 t of 24 764.
  — `ices_gadoid_f.md`, from the 2026 advice sheets (Fetched full text)

## Computed: pooled biomass-weighted F under different species mixes and periods

Method:
- Same as `ices_gadoid_f.md`: Fbar from the 2026 SAG tables, times the SMS selectivity conversion (C/Bbar)/Fbar.
- The conversions are cod 0.95, haddock 0.82, whiting 0.90, saithe 0.69, hake 1.13 (C/TSB over Fbar).
- Cod is the Viking substock. North Sea whiting stands in for 3.a.

| Weights (cod / had / whg / pok / hke) | 2010-2019 mean F | 2023-2025 mean F |
|---|---|---|
| W1, Skagerrak IBTS 2021-25 (0.058 / 0.543 / 0.267 / 0.121 / 0.011) | 0.229 | **0.108** |
| Skagerrak IBTS 2005-2019 (0.311 / 0.169 / 0.368 / 0.123 / 0.029) | **0.294** | 0.197 |
| Box 44G0-45G1 IBTS 2005-2019 (0.329 / 0.053 / 0.458 / 0.125 / 0.034) | 0.299 | 0.206 |
| Ecopath NS 1991 pool shares, "other gadoids" dropped (0.136 / 0.227 / 0.330 / 0.295 / 0.008), present F | | about 0.17 |

The weights for the 2005-2019 rows are new survey shares from the NS-IBTS computation in Q1. The Ecopath shares come from
`Restdödlighet M0 hos torskfiskar/model_context.md`: 29 % whiting, 26 % saithe, 20 % haddock, 12 % cod, 12 % other,
0.7 % hake.

Per-species F means, 2010-2019 vs 2023-2025:

| Species | 2010-2019 | 2023-2025 |
|---|---|---|
| Haddock | 0.24 | 0.07 |
| Whiting | 0.23 | 0.09 |
| Saithe | 0.38 | 0.39 |
| Cod (Viking) | 0.47 | 0.40 |
| Hake | 0.20 | 0.22 |

Recreational and unassessed additions, applied to the cod share only (computed):
- Cod is 6-33 % of the pool, depending on the weights.
- A 2-10 % recreational addition on cod F of 0.40 adds 0.0005-0.013 to the pool F.
- Whiting, haddock and saithe recreational catches offshore are not quantified in any source found. Tourist catches
  south of 62 N are mostly saithe and mackerel (Vølstad et al. 2011, ICES JMS 68:1785-1791; snippet only, via sub-search).
- Inference: recreational fishing moves the pooled offshore F by at most about 0.01-0.02.

## Reconciling 0.12 (pooled stock F) with 0.26 (local catch/biomass)
- The 0.26 in `local_fishery.md` divides 2010-2020 landings in 45G0 by the model's own standing biomass (0.7 t/km2).
  - In 2010-2020 the pooled stock F was itself 0.23-0.30 (table above).
  - So the two routes agree for that period, as long as the model's 0.7 t/km2 is close to the true local density.
- The sibling note's swept-area estimate (u ≈ 0.13, 2023-25 catches over IBTS biomass of 3.3 t/km2) is the present-day
  analogue, and it matches 0.12.
- Inference: the 0.26 is the 2010-2020 F, not evidence of extra fishing on top of the stock F.

The open caveat is unchanged from `ices_gadoid_f.md`: the IBTS wing-spread biomass is about 3.3 t/km2, against the
model's 0.69. If the true density is 4x the model's, the local tonnage catch is spread over 4x more fish and u is 4x lower.

## Inferences for the model (Q2)
1. **Keep F at 0.12-0.20.**
   - Use 0.12 if the pool is meant to be today's haddock-dominated mix.
   - Use about 0.17-0.20 if it is meant to be the long-term mix (Ecopath-1991 or 2005-2019 survey), which is what its
     diet and r_max weighting imply.
   - 0.20 is the better-defended value for a mix that eats herring as its main prey. Haddock, the dominant species
     today, eats little herring (`predator_prey_cycles.md`).
2. **Adding the "fishing outside the grid" has no basis.** It is already in a stock F.
3. **F cannot be the main fix.** Even F 0.30, the 2010-2019 value under the historical mix, leaves r_max - F ≈ 0.2/yr at
   ideal food.
4. The model realises only 0.10-0.115 of its nominal 0.12 (resume 184.13).
   - Probably the gadoids are not distributed in proportion to the effort map (Bratten zeros, E-W contrast). That is my
     guess and was not checked in code.
   - A target realised F needs a nominal value about 5-15 % higher.
