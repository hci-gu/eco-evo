# ICES residual (M1) and predation (M2) natural mortality for North Sea / Skagerrak gadoids (cod, haddock, whiting, saithe; hake)

Method note: the WGSAM SMS key-run input and output files are public on GitHub
(ices-eg/wg_WGSAM). The numbers below marked "computed from key-run files" were read directly
from `natmor1.in` (M1 input, quarterly), `natmorm1m2.out` (M1+M2 output, quarterly) and
`partial_m2.out` (M2 split by predator, quarterly) of NorthSeaKeyRun_2023 and summed over the
four quarters to annual rates (annual instantaneous M = sum of quarterly rates; this is how the
SMS stock annex itself tabulates "sum of quarterly M1+M2"). Averaging over years and the
predator ranking are my own arithmetic on those files, not a published table. The 0-group
lives only in quarters 3-4, so its annual figures cover half a year.

Main repository: https://github.com/ices-eg/wg_WGSAM
- 2023 key run folder: https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023
- M1 input file: https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmor1.in
- M1+M2 output: https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmorm1m2.out
- Partial M2 by predator: https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/partial_m2.out
- SMS stock annex 2023 (PDF): https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf
- Key run HTML documentation: https://ices-eg.github.io/wg_WGSAM/NS_2023_key_run.html

As of the repository listing on 2026-10-12, the latest North Sea key run on GitHub is
2023 (folders exist for 2011, 2014, 2017, 2020, 2023; no 2026 North Sea folder; there is a
Baltic 2025 run). — [wg_WGSAM GitHub](https://github.com/ices-eg/wg_WGSAM)

## Q1. What M1 does the North Sea SMS (WGSAM key runs 2011-2023) assume per age for cod, haddock, whiting, saithe?

### Takeaway
M1 in every North Sea SMS key run is a fixed convention, not an estimate: 0.05 per quarter = 0.2 per year
for all ages of cod, haddock, whiting and saithe (0-group: 0.1 for its half year). The only exception is
the 2023 run, where cod ages 3+ get 0.353/yr from 2011 onwards - an ad hoc migration loss (fish leaving
to West of Scotland), not biological mortality. Classic MSVPA (1997 key run, 2002 workshop) also used
M1 = 0.2 for all ages of cod, haddock and whiting.

### Cited Findings
- SMS stock annex 2023, section 3.4.4 (verbatim): "M1 (residual natural mortality) by quarter is set to 0.05 for the species cod, whiting, haddock, saithe, the two sandeel stocks, Norway pout, sprat and 0.0375 for mackerel, and 0.025 for herring, plaice and sole. M1 for non-prey species is the annual natural mortality (M) used in the single-species assessment divided into 4 quarters. For cod M1 is adjusted for age 3+ to mimic migration from the North Sea to the area West of Scotland (see sec 3.1.3)." — [SMS Stock Annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)
- SMS stock annex 2023, section 3.1.3 (cod): "The 2021 cod benchmark introduced an ad hoc adjustment on natural mortality of ages 3+ from 2011 to mimic an assumed 15% migration to the West of Scotland area, which was not included in the assessment area. M1 used for SMS is adjusted using the same approach", with a formula of the form M1_adj = M1 + (term in ln(1 - alpha)), "where alpha (15%) is the migration rate as a proportion". (The formula did not extract cleanly from the PDF.) — [SMS Stock Annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)
- Key run 2023 `natmor1.in`, annual sums computed from key-run files (ages 0..10; whiting plus group at 6):
  - Cod 1974-2010: 0.1 (age 0), 0.2 at ages 1-10. Cod 2011-2022: 0.1, 0.2, 0.2, then **0.353** at ages 3-10 (quarterly 0.0882 instead of 0.05).
  - Haddock 1974-2022: 0.1 (age 0), 0.2 at ages 1-10.
  - Whiting 1974-2022: 0.1 (age 0), 0.2 at ages 1-6+.
  - Saithe 1974-2022: 0.1 (age 0), 0.2 at ages 1-10.
  - For comparison: mackerel 0.15, herring 0.1, sandeels/Norway pout/sprat 0.2, plaice age-dependent 0.495 (age 1) falling to 0.246 (age 10) (Peterson-Wroblewski values copied from the plaice assessment). — [natmor1.in, NS key run 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmor1.in)
- Earlier key runs, `natmor1.in` annual sums computed from key-run files: 2011 key run = 0.2 at all ages for COD, WHG, HAD, POK in 1963-2003 and 0.1 (age 0) / 0.2 (ages 1+) in 2004-2010; 2014, 2017 and 2020 key runs = 0.05 per quarter in all quarters and ages, i.e. 0.2/yr (the 0-group enters the model in quarter 3, so its effective M1 is 0.1 - my reading of the SMS convention). No year-to-year variation for any of the four gadoids in 2011-2020. — [NorthSeaKeyRun_2011](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2011), [2014](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2014), [2017](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2017), [2020](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2020)
- Classic MSVPA (ICES Workshop on MSVPA in the North Sea, ICES CM 2002/D:04): Tables 4.1.4 and 4.1.5 list M1 "all ages" = **0.2** for cod, haddock, whiting, Norway pout, sandeel, sprat and 0.1 for herring. "Residual mortalities were taken from the last MAWG (ICES 1997a) and copied forwards in time." — [ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF)
- The same report on the MSVPA "Other predators" group (cetaceans and piscivorous fish): "The intention was to estimate a part of the residual natural mortality (M1) given as input; however, these values were not lowered in the key run made." The 2002 workshop then dropped that group as "very ad hoc". — [ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF)

### Inferences
- The SMS M1 = 0.2/yr for gadoids is inherited from the MSVPA tradition and never estimated from data. The SMS fit takes it as fixed input.
- The 2023 cod M1 of 0.353 (ages 3+, 2011-2022) contains about 0.153/yr of emigration on top of the conventional 0.2. Do not read it as evidence of higher biological residual mortality. The quarterly increment of 0.0382 (0.0882 - 0.05) fits -ln(1 - 0.15/4) = 0.03822 to four decimals: the 15% annual migration is spread evenly over four quarters and applied as an instantaneous rate each quarter. This is my reconstruction; the annex formula did not extract cleanly. A corollary is that four such quarters remove 1 - 0.9625^4 = 14.2% per year, not exactly 15%.
- The plaice case shows that SMS M1 is not always "non-predation" mortality. The annex says single-species plaice M was used as M1 "ignor[ing] the fact that the natural mortality from grey seals is assumed already to be included".

### Gaps
- The prompt says classic MSVPA used higher M1 at ages 0-1. I did not find that in the 2002 MSVPA workshop tables, which give one "all ages" M1 of 0.2 for gadoids. The MAWG 1997 source (ICES 1997a) was not consulted.
- I found no SMS key-run document that justifies the 0.2 value biologically (no citation to disease, senescence or other causes).

## Q2. What M2 by age does the latest WGSAM key run (2023) estimate, and which predators dominate?

### Takeaway
In the 2023 key run, M2 on gadoid adults is small for cod and haddock and moderate for whiting. Mean
2018-2022: cod age 3 = 0.24, age 4 = 0.04, ages 5+ = 0; haddock age 3 = 0.15, age 4 = 0.10, ages 5-8 =
0.02-0.07; whiting ages 3-5 = 0.27-0.31. Juvenile M2 is high (0-group 1.0-1.9/half-year; age 1 = 0.86-1.02).
For 1-3 year-old cod and 2-6 year-old whiting, the main predators are harbour porpoise and grey seal. For
0-group cod and whiting it is grey gurnard, and for haddock ages 0-2 it is saithe plus whiting. Saithe gets
no M2 at all: it is a predator only. Hake is an external predator with no M estimated.

### Cited Findings
- SMS 2023 species and predators (`species_names.in`): external "other predators" = Fulmar, Guillemot, Herring gull, Kittiwake, Great black-backed gull, Gannet, Puffin, Razorbill, starry ray (A. radiata), grey gurnards, western and North Sea horse mackerel, grey seal, harbour porpoise, hake. Modelled VPA species = cod, whiting, haddock, saithe, mackerel, herring, N and S sandeel, Norway pout, sprat, plaice, sole. — [species_names.in, NS key run 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/species_names.in)
- Hake "was included in the 2014 key run as an 'external predator'" (its North Sea abundance comes from the northern hake SS3 assessment). It is not a prey species, so SMS gives it no M1/M2. — [SMS Stock Annex 2023, sec. 3.2.4](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)
- Annual M2 by age (M_total from `natmorm1m2.out` minus M1 from `natmor1.in`), computed from key-run files. Ages 0, 1, 2, 3, 4, 5, 6, 7, 8:
  - **Cod** 1974-1990: 1.537, 0.806, 0.499, 0.064, 0.006, 0, 0, 0, 0
  - **Cod** 1991-2010: 2.102, 0.862, 0.646, 0.148, 0.023, 0, 0, 0, 0
  - **Cod** 2018-2022: 1.892, 1.017, 0.752, 0.240, 0.043, 0.000, 0, 0, 0 (total M 2018-2022: 1.992, 1.217, 0.952, 0.593, 0.396, 0.353 ...)
  - **Haddock** 1974-1990: 1.104, 1.078, 0.409, 0.227, 0.131, 0.064, 0.036, 0.023, 0.010
  - **Haddock** 2018-2022: 1.024, 0.921, 0.267, 0.146, 0.098, 0.069, 0.051, 0.043, 0.021 (total M: 1.124, 1.121, 0.467, 0.346, 0.298, 0.269, 0.251, 0.243, 0.221)
  - **Whiting** 1974-1990: 1.046, 1.033, 0.497, 0.350, 0.301, 0.269, 0.082 (6+)
  - **Whiting** 2018-2022: 1.544, 0.859, 0.470, 0.314, 0.290, 0.267, 0.157 (6+) (total M: 1.644, 1.059, 0.670, 0.514, 0.490, 0.467, 0.357)
  - **Saithe**: M2 = 0 at all ages in all years (M = M1 = 0.2). — [natmorm1m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmorm1m2.out), [natmor1.in](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmor1.in)
- Published cross-check: SMS stock annex 2023, Table 5-4 "Cod: Natural mortality (sum of quarterly M1+M2)" lists 2018: 2.146, 1.300, 1.028, 0.594, 0.404, 0.353 ... and 2019: 1.841, 1.218, 0.971, 0.571, 0.410, 0.353 ... (ages 0-5+). This matches the file-derived values. — [SMS Stock Annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)
- Predator split of M2, mean 2018-2022, computed from `partial_m2.out` (annual M2 contributed by each predator; share of that age's M2):
  - Cod age 0: grey gurnard 0.844 (45%), whiting 0.386 (20%), harbour porpoise 0.223 (12%), fulmar 0.146 (8%), cod 0.094 (5%), grey seal 0.085 (4%)
  - Cod age 1: harbour porpoise 0.470 (46%), grey seal 0.225 (22%), gannet 0.136 (13%), cod 0.123 (12%), whiting 0.063 (6%)
  - Cod age 2: harbour porpoise 0.470 (62%), grey seal 0.225 (30%), cod 0.058 (8%)
  - Cod age 3: grey seal 0.225 (93%), cod 0.016 (7%); cod age 4: grey seal 0.040 (94%), cod 0.003
  - Whiting age 0: grey gurnard 0.888 (58%), whiting 0.329 (21%), porpoise 0.098, saithe 0.089, cod 0.043
  - Whiting age 1: whiting 0.254 (30%), porpoise 0.206 (24%), saithe 0.166 (19%), guillemot 0.103 (12%), cod 0.063, grey gurnard 0.044
  - Whiting age 2: porpoise 0.206 (44%), saithe 0.093 (20%), whiting 0.086 (18%), cod 0.063 (13%), grey seal 0.022
  - Whiting ages 3-5: porpoise 0.196-0.206 (66-74%), cod 0.034-0.041, saithe 0.015-0.046, grey seal 0.022
  - Haddock age 0: whiting 0.530 (52%), saithe 0.282 (28%), cod 0.096 (9%), guillemot 0.055, grey gurnard 0.030
  - Haddock age 1: saithe 0.468 (51%), whiting 0.263 (29%), cod 0.140 (15%), grey seal 0.050
  - Haddock age 2: saithe 0.130 (49%), cod 0.086 (32%), grey seal 0.050 (19%)
  - Haddock ages 3-6: grey seal 0.031-0.050, cod 0.020-0.052, saithe 0-0.044 — [partial_m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/partial_m2.out), [species_names.in](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/species_names.in)
- Historical MSVPA M2 (2002 workshop), ages 0-6. Cod 1963-2000 average: 2.151, 0.622, 0.205, 0.139, 0.033, 0.029, (0.058). Haddock: 2.064, 1.226, 0.108, 0.052, 0.043, 0.090, 0.036. Whiting: 1.894, 0.952, 0.229, 0.151, 0.174, 0.226, 0.167. Same for 1995-2000: cod 3.710, 0.513, 0.152, 0.225, 0.068, 0.067, (0.138); haddock 2.240, 1.263, 0.072, 0.040, 0.030, 0.185, 0.055; whiting 2.697, 1.127, 0.170, 0.160, 0.232, 0.412, 0.252. "Predation mortality for the 0-group refers exclusively to the second half of the year". MSVPA predators then: cod, haddock, mackerel, saithe, whiting (no seals, birds or porpoise). — [ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF)
- The 2023 key run revised the bird and grey seal series. Plaice became a prey of grey seals. — [SMS Stock Annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf); [WGSAM 2023 outputs (figshare)](https://ices-library.figshare.com/articles/report/Working_Group_on_Multispecies_Assessment_Methods_WGSAM_outputs_from_2023_meeting_/25020968)

### Inferences
- Biomass-weighted adult gadoid M (cod 4+, haddock 3+, saithe 3+, whiting 3+) in the 2023 key run is roughly 0.2 (M1 convention) + 0-0.1 (M2) for cod, haddock and saithe, and 0.2 + ~0.3 for whiting. Whiting adult M2 is mostly harbour porpoise. Because saithe and haddock dominate North Sea gadoid adult biomass, a pooled adult-dominated gadoid group would have SMS total M of about 0.25-0.3/yr, of which roughly 0.2 is the assumed residual M1. This is my rough weighting; I did not compute a biomass-weighted average.
- Harbour porpoise and grey seal predation (the modelled-predator share most relevant to a model with porpoises and seals as their own FGs) is about 0.2-0.7/yr on cod ages 1-3, about 0.2-0.23 on whiting ages 1-5, and about 0.03-0.05 on haddock 2+. Grey gurnard, birds and fish predators act almost only on ages 0-1 (except saithe on haddock age 1-2 and cod/saithe on whiting).

### Gaps
- I did not compute M2 for hake as prey: SMS does not model hake as prey.
- Starry ray and horse mackerel contributions were small for gadoid prey and do not appear in the top lists above. I did not tabulate them separately.

## Q3. What M do the single-species assessments (WGNSSK) currently use?

### Takeaway
Northern shelf cod (benchmark WKBCOD 2023, three substocks in multi-stock SAM) uses smoothed SMS key-run M
(2020 key run at the benchmark), the same for all three substocks. North Sea haddock and whiting use SMS
key-run M as recommended by WGSAM. North Sea saithe moved at WKBGAD 2024 from a fixed M = 0.2 at all ages
to a static Lorenzen M scaled so that M = 0.2 at age 9. Earlier West of Scotland cod used Lorenzen M = 3W^-0.29.

### Cited Findings
- WKBCOD 2023: "Since 2009, variable natural mortality rates at age estimates produced using the stochastic multi-species model SMS (Lewy and Vinther, 2004) have been used in the assessment for North Sea cod ... In the assessment for North Sea cod, the raw estimates of natural mortality from the latest SMS key-run are smoothed to reduce the effects of interannual variability whilst maintaining overall trends." Decision: "It was decided to include natural mortality based on the SMS runs from the North Sea." Section 21.14.5: "Estimates of natural mortality from the latest SMS key-run are used for all three substocks" (figure from the 2020 key run). — [WKBCOD 2023, ICES Sci. Rep. 5:37](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf)
- WKBCOD 2023: the 2021 North Sea cod benchmark (WKNSEA 2021) "introduce[d] an ad hoc adjustment on the natural mortality of ages 3+ from 2011 to account for migration to the West of Scotland area ... seen as an interim solution". Including 6.a in the Northwestern substock "now resolves this discrepancy". — [WKBCOD 2023](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf)
- WKBCOD 2023: West of Scotland cod (cod.27.6a, WKROUND 2012) used Lorenzen M_a = 3 W_a^-0.29, time-varying since 2019/2020. Compared with SMS, "the Lorenzen method estimat[es] lower M-at-age at younger ages and slightly higher M-at-age for older ages". Per the figure text, Lorenzen is lower than SMS for ages 1 and 2 and higher from age 3 on. — [WKBCOD 2023](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf)
- WKBCOD 2023 notes that other area-6a stocks, e.g. "Northern shelf haddock and West of Scotland herring", use the latest SMS M estimates. — [WKBCOD 2023](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf)
- WGSAM 2023 "recommends the use of natural mortality estimates from the North Sea SMS key-run for use in single species stock assessment models of North Sea haddock, herring, Norway pout, sandeel ..., sprat, and whiting". Cod is excluded because the key run "continues to assume a single North Sea cod stock in contrast to the recent split into three components" (search-result summary of the figshare record; full PDF not read). — [WGSAM 2023 outputs](https://ices-library.figshare.com/articles/report/Working_Group_on_Multispecies_Assessment_Methods_WGSAM_outputs_from_2023_meeting_/25020968)
- WGNSSK summary: ecosystem effects enter the cod, haddock and whiting assessments "in the form of varying natural mortalities estimated by ... WGSAM" (search snippet only). — [WGNSSK (DTU Orbit record)](https://orbit.dtu.dk/en/publications/working-group-on-the-assessment-of-demersal-stocks-in-the-north-s/)
- In late 2024 ICES reissued the 2025 northern shelf cod advice (15 511 t instead of 19 321 t) after a forecast-model error. The cut was attributed mainly to the corrected model predicting higher natural mortality (search-result summary). — [ICES explainer](https://www.ices.dk/news-and-events/news-archive/news/Pages/ExplainerNScod24reissue.aspx)

### Inferences
- For haddock and whiting, the single-species M-at-age is (a smoothed version of) the SMS total M = 0.2 + M2. The adult values are therefore close to the Q2 table: haddock ~0.22-0.35 at ages 3-8, whiting ~0.36-0.51 at ages 3-6+.

### Gaps
- I did not obtain the exact smoothed M-at-age matrices used in the latest WGNSSK runs for cod, haddock and whiting (they sit in the stock annexes / WGNSSK input files, which I did not open). The values above are the raw SMS 2023 output, not the smoothed assessment input.
- The current haddock and whiting benchmark years and reports were not verified. No WKBWHITING or recent North Sea haddock benchmark document was found.

## Q4. What M is used for saithe (North Sea / West of Scotland / Skagerrak) and Kattegat cod?

### Takeaway
Saithe (pok.27.3a46): until 2024, M = 0.2 at all ages. Since WKBGAD 2024 it is a static Lorenzen M scaled to
0.2 at age 9, so higher at younger ages. Likelihood profiles favoured a fixed M of about 0.3 slightly over 0.2.
Kattegat cod (cod.27.21): the 2024 stock annex says "A constant natural mortality of 0.2 is assumed for all
ages and years". Unexplained losses (including any extra natural mortality from seals) are absorbed by
SAM catch multipliers, so the assessment F is in effect Z - 0.2.

### Cited Findings
- WKBGAD 2024, North Sea saithe (pok.27.3a46), section 4.2.4: "a static Lorenzen estimate was preferred ... it was agreed to use a multiplicative scalar, as M is a rate ... For the terminal age predation can be expected to be lowest compared to younger ages and 0.2 as residual mortality may be a reasonable assumption for the terminal age. Age nine M = 0.2 was used for scaling in North Sea saithe, as it constituted the last true age group." Profile likelihood "suggested that (1) larger fixed M around 0.3 was performing slightly better than M = 0.2, and (2) the scaling used (M = 0.2 at age nine) was not far off the optimum". The figure caption calls the previous assessment "(fixed M = 0.2)". — [WKBGAD 2024](https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_selected_haddock_and_saithe_stocks_WKBGAD_/25002470)
- WKBGAD 2024 review panel report, in the North Sea saithe (pok.27.3a46) part, which also covers saithe weight-at-age ("The previous North Sea saithe assessment used catch weights..."). It is a second, independent statement for the same stock: it considered fixed M = 0.2 and static scaled Lorenzen. "Profile likelihood plots of M showed a better fit with an M of approximately 0.3. The panel recommended the static scaled Lorenzen M ... given that it was more biologically realistic, there was weak evidence of a time-varying trend and there was a small improvement to model performance." — [WKBGAD 2024](https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_selected_haddock_and_saithe_stocks_WKBGAD_/25002470)
- WKBGAD 2024, Rockall haddock (outside the North Sea, for context): it moved from M = 0.2 at all ages to Lorenzen scaled to 0.2 at age 8, averaged over time. It noted that the SMS-type multispecies estimates for that area include "...anglerfish and grey gurnard but not seals". — [WKBGAD 2024](https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_selected_haddock_and_saithe_stocks_WKBGAD_/25002470)
- Saithe M = 0.2 has a long history: a 1977 ICES paper says the saithe working group assumed M = 0.2 for all saithe stocks (search-result snippet only). — [ICES CM 1977/F:39](https://www.ices.dk/sites/pub/CM Doccuments/1977/F/1977_F39.pdf)
- Kattegat cod: WKBALT 2017 says "data on seal diet and population abundance suggested a higher natural mortality than previously assumed in the stock assessment of this stock. The data were however considered insufficient to directly include them in the stock assessment" (abstract/snippet only). — [WKBALT 2017](https://archimer.ifremer.fr/doc/00415/52643)
- Kattegat cod stock annex 2024 (cod27.21_SA2024), section 2.2.2: "A constant natural mortality of 0.2 is assumed for all ages and years." Section 3.2: assessment model SAM. "triannual catch multipliers are estimated in SAM model for the years 2003 onwards ... The multiplier represents unallocated removals that are considered to be related to fisheries, unaccounted natural mortality and migration. Thus the mortality estimates resulting from the assessment represent total mortality (which includes migration) minus the assumed natural mortality, i.e. Z-0.2." Biology section: "The increasing harbour seal population since 2000s has likely increased natural mortality of cod, though the available data on seal diet composition is still too limited to quantify the resulting predation mortality (WKBALT 2017)." It names adult cod as the main fish predator on juvenile cod, "As other predator species such as whiting, pollack and haddock have declined". — [Stock Annex: Cod in Subdivision 21 (Kattegat), 2024](https://ices-library.figshare.com/articles/report/Stock_Annex_Cod_in_Subdivision_21_Kattegat_/18622304)
- DTU project MERTOR: "at present, the natural mortality due to seal predation is assumed to be constant" in the cod assessments it addresses (snippet). — [DTU Orbit MERTOR](https://orbit.dtu.dk/en/projects/growth-and-natural-mortality-in-cod-effects-of-seals-oxygen-deple/)

### Inferences
- The SMS annex says M1 for non-prey species is copied from the single-species M. A key run after 2024 would therefore presumably import the scaled Lorenzen M for saithe, raising saithe M1 at young ages above 0.2. This is an inference; no such key run exists yet.
- With M_a = c * 3 W_a^-0.29 scaled so that M_9 = 0.2, saithe M at age a is 0.2 * (W_a / W_9)^-0.29. The weights here are my own rough assumption, not taken from a source: about 1 kg at age 3, 2.5 kg at age 5 and 7 kg at age 9. With those, M comes out at about 0.35 at age 3 and 0.27 at age 5, falling to 0.2 at age 9. This is my calculation from the formula; I did not read the actual values.

### Gaps
- I did not find the numeric M-at-age table adopted for North Sea saithe at WKBGAD 2024. It is in WD 4.3 "pok.27.3a46 - stock weights-at-age and natural mortality", which I did not open.
- I did not check whether the 2026 WGBFAS run of Kattegat cod (report published 2026-07) changed the M = 0.2 in the 2024 stock annex.

## Q5. Explicit statements on 'other' (disease, senescence, unexplained) vs predation in adult M

### Takeaway
No ICES source found splits adult gadoid M into disease, senescence or unexplained parts. The residual 0.2 is
a convention, treated in practice as a floor and as "what is left at the oldest ages". It is known to contain
unmodelled predation: the MSVPA "other predators" were meant to take part of it, and plaice M1 includes seal
predation. In SMS, adult cod 5+, haddock 8+ and saithe have M = M1 = 0.2, with zero M2.

### Cited Findings
- WKBGAD 2024: "For the terminal age predation can be expected to be lowest compared to younger ages and 0.2 as residual mortality may be a reasonable assumption for the terminal age." — [WKBGAD 2024](https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_selected_haddock_and_saithe_stocks_WKBGAD_/25002470)
- ICES CM 2002/D:04: the "Other predators" group "represents various unspecified cetaceans and piscivorous fishes ... The intention was to estimate a part of the residual natural mortality (M1) given as input; however, these values were not lowered". — [ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF)
- SMS stock annex 2023 on plaice: "The M values from WGNSSK were used for both M and M1 in SMS. This ignores the fact that the natural mortality from grey seals is assumed already to be included." — [SMS Stock Annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)
- SMS 2023 output: cod ages 5-10 total M = M1 exactly (0.2 before 2011, 0.353 from 2011 with migration). Saithe all ages M = 0.2 with zero M2 (computed from key-run files). — [natmorm1m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmorm1m2.out)
- WGSAM review criteria list "sensitivity to key parameters: consumption rates, residual mortality (M1, M0)". This shows WGSAM treats M1 as an uncertain input to be sensitivity-tested (search-result snippet). — [WGSAM Key Run Review Criteria](https://ices-eg.github.io/wg_WGSAM/ReviewCriteria.html)

### Inferences
- For a pooled adult gadoid group, 0.2/yr is the ICES default for "everything except modelled predators and fishing", not a measured biological residual. It includes predation by unmodelled predators (e.g. cetaceans other than porpoise, sharks/rays not in SMS, and for plaice even seals). SMS also models grey seals and harbour porpoise, but as external predators. The reason 0.2 is an upper-end value for an M1 meant to cover only disease and senescence is that it also absorbs predators SMS lacks: cetaceans other than porpoise, elasmobranchs other than starry ray, harbour seal (SMS has grey seal only), and so on. Benchmark profile likelihoods (WKBGAD 2024 saithe: fixed M ~0.3 fit slightly better than 0.2) point the other way, towards total adult M somewhat above 0.2. The data cannot separate the two.
- Per tick: 0.2/yr equals about 0.000137 per 6 h tick (0.2/1460), if the project's 1460 ticks/yr convention is used. This is plain arithmetic, not a source value.

### Gaps
- No ICES North Sea document found with a quantitative breakdown of adult M1 into disease, parasites, senescence or spawning stress.
- Not consulted: Sparholt 1990, Pope 1991, Gislason & Helgason, Lorenzen 1996, Gislason et al. 2010, Lewy & Vinther 2004 (only cited second-hand via WKBCOD/WKBGAD). Their exact numbers are not reported here.
