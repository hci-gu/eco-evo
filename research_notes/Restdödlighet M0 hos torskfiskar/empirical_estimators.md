# Empirical estimators of natural mortality M for North Sea / Skagerrak gadoids (cod, haddock, whiting, saithe, hake), size dependence, and a biomass-weighted adult M vs M0 = 0.392/yr

Conventions in these notes:
- "Read" = value seen in the cited source. "Secondary" = coefficient/value taken from a software implementation or database rather than the original paper. "Snippet/abstract only" = seen only in a search-engine summary or abstract, full text not read.
- All rates are instantaneous, per year.
- All arithmetic is in the Inferences sections, with formulas and inputs shown. Calculations were run in Python; numbers are rounded to 3 decimals.
- The key conceptual point for the simulator: the life-history estimators below (Then, Hoenig, Hamel & Cope, Pauly, Gislason, Charnov, Lorenzen) all estimate TOTAL natural mortality M (predation + everything else). An Ecopath "other mortality" M0 is the NON-PREDATION residual (comparable to M1 in multispecies models, M = M1 + M2). The two must not be compared as if they were the same quantity.

## Q1. What do the standard empirical estimators give for each species (inputs and calculations)?

### Takeaway
With global FishBase maximum ages (cod/saithe 25 yr, haddock/whiting/hake 20 yr), the maximum-age estimators give a total adult M of 0.17-0.32/yr. Growth-based estimators with North Sea growth parameters give 0.23-0.26 for cod/saithe, 0.35-0.39 for haddock and 0.44-0.50 for whiting. If the much shorter observed lifespans of the fished North Sea stocks (about 12-15 yr) are used instead, the maximum-age estimators rise to 0.36-0.50. Every estimator is a TOTAL-M estimate, not a non-predation residual.

### Cited Findings
Estimator equations:
- Then et al. (2015, ICES JMS 72:82-92) recommend the maximum-age estimator M = 4.899 tmax^-0.916 (prediction error 0.32). Where tmax is missing they recommend the growth-based M = 4.118 K^0.73 Linf^-0.33 (prediction error 0.6, Linf in cm per the corrigendum). The basis was cross-validation over more than 200 species. — [USGS publication record (abstract only)](https://www.usgs.gov/publications/evaluating-predictive-performance-empirical-estimators-natural-mortality-rate-using); [VIMS M database page](https://www.vims.edu/research/departments/fisheries/programs2/mort_db/index.php)
- Implemented coefficients (secondary: the R package fishmethods, `M.empirical`, not the original papers):
  - Pauly (1980) length form: M = 10^(-0.0066 - 0.279 log10 Linf + 0.6543 log10 K + 0.4634 log10 T).
  - Hoenig (1983) fish: M = exp(1.46 - 1.01 ln tmax).
  - Hoenig (1983) joint: M = 4.22 / tmax^0.982.
  - Lorenzen (1996): M = 3.0 Wwet^-0.288 (W in g).
  - Gislason et al. (2010): M = exp(0.55 - 1.61 ln L + 1.44 ln Linf + ln K).
  - Then (2015) tmax: 4.899 tmax^-0.916.
  - Then (2015) growth: 4.118 K^0.73 Linf^-0.33.
  - Charnov et al. (2013): M = (L/Linf)^-1.5 K.
  - Source: [fishmethods M.empirical.R source (CRAN mirror)](https://raw.githubusercontent.com/cran/fishmethods/master/R/M.empirical.R)
- Gislason et al. (2010), from the abstract: M was significantly related to L, Linf and K but not to temperature (R^2 = 0.62, n = 168). The recommended equation is ln(M) = 0.55 - 1.61 ln(L) + 1.44 ln(Linf) + ln(K). — [WUR record (abstract only)](https://research.wur.nl/en/publications/size-growth-temperature-and-the-natural-mortality-of-marine-fish)
- Lorenzen (1996, J Fish Biol 49:627-642): M is a power function of body weight, M_W = M_u W^b. The exponent is about -0.29 for juvenile and adult fish in natural ecosystems, and M at unit weight is higher in natural ecosystems than in aquaculture. The coefficient 3.00 was not seen in the paper itself; it comes from the fishmethods source above. — [DFID research output record (abstract only)](https://www.gov.uk/dfid-research-outputs/the-relationship-between-body-weight-and-natural-mortality-in-juvenile-and-adult-fish-a-comparison-of-natural-ecosystems-and-aquaculture)
- Hamel & Cope (2022, Fish Res 256:106477; full text read):
  - The point estimate is M = 5.40/Amax, the median of a lognormal prior with log-space SD 0.31. The authors derived 0.31 from the 0.44 residual SD by assuming half the variance is error in the M estimates.
  - The coefficient 5.40 is larger than the 5.109 that Then et al. found on untransformed data and the 4.37 of Hamel (2015).
  - The new prior replaces the Hamel (2015) prior (median 4.37/Amax, SD 0.51).
  - Source: [Hamel & Cope 2022 (SEDAR76-RD15 PDF)](https://sedarweb.org/documents/sedar76rd15_development-and-considerations-for-application-of-a-longevity-based-prior-for-the-natural-mortality-rate-pdf)

Inputs (life history):
- FishBase maximum reported ages (global, not North Sea-specific): cod 25 yr, haddock 20, whiting 20, saithe 25, European hake 20.
- FishBase lengths at maturity (Lm): cod 65.4 cm, haddock 34.9, whiting 28.2, saithe 39.1, hake 42.1.
- FishBase resilience-entry K ranges: haddock 0.12-0.30, whiting 0.07-0.74, saithe 0.07-0.17, hake 0.1.
- Sources: [FishBase cod](https://www.fishbase.se/summary/Gadus-morhua.html), [haddock](https://www.fishbase.se/summary/Melanogrammus-aeglefinus.html), [whiting](https://www.fishbase.se/summary/Merlangius-merlangus.html), [saithe](https://www.fishbase.se/summary/Pollachius-virens.html), [hake](https://www.fishbase.se/summary/Merluccius-merluccius.html)
- FishBase cod growth rows for North Sea / Skagerrak localities:
  - Linf ranges 105-132 cm and K 0.134-0.333.
  - For example Linf 116 cm, K 0.208 (North Sea, 7d, Skagerrak; listed M 0.18), Linf 119, K 0.269 (northern North Sea) and Linf 132, K 0.200 (England & Wales North Sea; M 0.20).
  - The temperatures attached to the rows are mostly 8-9 C.
  - The extraction tool flagged the column mapping of the unlabeled values (t0/M/temp) as uncertain.
  - Source: [FishBase cod growth list](https://www.fishbase.se/popdyn/PopGrowthList.php?ID=69&GenusName=Gadus&SpeciesName=morhua)
- Classic North Sea cod growth fits (1970s market samples; snippet only): southern North Sea Linf 110.8 cm, K 0.333; northern North Sea Linf 118.7 cm, K 0.269. — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/0077757974900258)
- mizer North Sea parameter set (secondary; Blanchard et al.), given as w_inf (g) and k_vb:
  - cod 40044 g, 0.216.
  - haddock 3485 g, 0.271.
  - whiting 1192 g, 0.323.
  - saithe 16856 g, 0.175.
  - Source: [mizer NS_species_params.csv](https://raw.githubusercontent.com/sizespectrum/mizer/master/inst/extdata/NS_species_params.csv)
- North Sea haddock growth is temperature-sensitive: warming lowers Linf and raises K (snippet only). — [Baudron et al. 2011, J Fish Biol](https://onlinelibrary.wiley.com/doi/10.1111/j.1095-8649.2011.02940.x)
- Single-species practice:
  - The North Sea SMS multispecies model treats saithe as a predator only, with no modelled predation on it. Its natural mortality is the single-species M.
  - Cod, haddock and whiting take their M from SMS.
  - The SMS age ranges are cod 0-10+ (old stock definition), saithe 3-10+ and whiting 0-6+.
  - Source: [ICES WGSAM 2024 report (2023 key run)](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf)

### Inferences
Inputs used (my choices):

| Species | tmax (yr) | Linf (cm) | K (1/yr) | Source / note |
|---|---|---|---|---|
| Cod | 25 | 120 | 0.20 | Central value of the FishBase North Sea rows (105-132 cm, K 0.13-0.33). The mizer w_inf of 40 kg would imply about 159 cm. |
| Haddock | 20 | 70.4 | 0.271 | Linf back-derived from mizer w_inf with W = 0.01 L^3 (the mizer default a, b), so it is approximate. A sensitivity case uses Linf 60. |
| Whiting | 20 | 49.2 | 0.323 | Same conversion from mizer. |
| Saithe | 25 | 119 | 0.175 | Same conversion from mizer. |
| Hake | 20 | n/a | n/a | No North Sea growth parameters found, so only tmax-based estimates are given. |

Temperature for Pauly (1980) is my assumption: T = 9 C, matching the 8-9 C on the FishBase North Sea cod rows.

Results, total M per year, with the global tmax values:

| Estimator | Cod | Haddock | Whiting | Saithe | Hake |
|---|---|---|---|---|---|
| Then 2015 tmax: 4.899 tmax^-0.916 | 0.257 | 0.315 | 0.315 | 0.257 | 0.315 |
| Hamel & Cope 2022: 5.40/tmax | 0.216 | 0.270 | 0.270 | 0.216 | 0.270 |
| Hoenig 1983 fish: exp(1.46 - 1.01 ln tmax) | 0.167 | 0.209 | 0.209 | 0.167 | 0.209 |
| Hoenig 1983 joint: 4.22/tmax^0.982 | 0.179 | 0.223 | 0.223 | 0.179 | 0.223 |
| Then 2015 growth: 4.118 K^0.73 Linf^-0.33 | 0.262 | 0.390 (0.411 with Linf 60) | 0.499 | 0.238 | - |
| Pauly 1980 (T = 9 C) | 0.250 | 0.354 | 0.439 | 0.230 | - |

Charnov 2013, Gislason 2010 and Lorenzen 1996 depend on body size and are covered in Q2.

Sensitivity to tmax. The global maxima are probably from slow-growing, cold or lightly fished populations, and North Sea assessments use plus-groups at age 10+ (6+ for whiting). With North Sea-like observed lifespans:

| tmax | Then | Hamel & Cope | Hoenig |
|---|---|---|---|
| 15 yr | 0.410 | 0.360 | 0.279 |
| 12 yr | 0.503 | 0.450 | 0.350 |
| 10 yr (whiting) | 0.594 | 0.540 | 0.421 |

The bias runs both ways and I do not pick a side:
- A fishing-truncated tmax inflates M, because the estimators assume an unfished maximum age.
- A global tmax borrowed from slow populations deflates M for the faster-growing North Sea stocks.
- Plausible bounds for the maximum-age estimators are therefore about 0.17-0.32 (global tmax) to 0.36-0.50 (tmax 12-15).

The growth-based estimators (0.23-0.26 for cod and saithe, 0.35-0.50 for haddock and whiting) sit inside that span and rank the species in the expected order: small, fast-growing whiting highest.

For comparison, the single-species and multispecies assessment conventions (Q2) use 0.2 for adult cod and saithe, which sits at the low end of the estimator range.

### Gaps
- I did not read the Then et al. (2015) or Gislason et al. (2010) full texts. The equations come from abstracts plus the fishmethods implementation, and the Lorenzen (1996) coefficient 3.00 only from fishmethods.
- I found no North Sea-specific maximum ages from age-reading databases. The FishBase maxima are global.
- No North Sea Linf/K was found for hake. Northern hake growth (Linf, K) was not retrieved, and FishBase gives only a resilience-entry K = 0.1.
- I did not find current ICES-assessment Linf/K for haddock, whiting or saithe; the mizer w_inf values were converted with a generic W = 0.01 L^3.
- Brodziak et al. (2011) and Maunder et al. (2023) were not read, so nothing from them is cited.

## Q2. Size dependence: M at adult sizes vs juveniles, and how much of adult M is predation vs other causes

### Takeaway
Within a population, M falls roughly as the inverse of length (Lorenzen 2022, "GLIM"). The North Sea multispecies model puts juvenile (age 0-1) M at about 0.8-2.3/yr. Adult M in the same model is:
- cod age 4+: 0.20, almost all residual M1 with predation M2 about 0;
- haddock age 5+: 0.20-0.26 in 1991 and 0.28-0.38 in 2022;
- whiting age 3-6: 0.42-0.53, about half of it predation.

The non-predation part for these species is not estimated anywhere I found. SMS simply fixes M1 = 0.20/yr (0.05 per quarter) for cod, haddock, whiting and saithe.

### Cited Findings
Lorenzen 2022 on size dependence:
- Lorenzen (2022, Fish Res 255:106454; full text read) proposes "generalized length-inverse mortality" (GLIM): M(L) = M_Lr (L/Lr)^c with c about -1 within populations.
- Possible departures: senescence at old ages, and density dependence in juveniles below about 20% of Linf.
- Community-level or among-population scaling is shallower, with c about -0.81 to -1.02.
- The steep Charnov (c = -1.5) and Gislason (-1.61) exponents reflect an among-population pattern in the Gislason et al. (2010) data, not within-population scaling.
- Source: [Lorenzen 2022 (SEDAR76-RD13 PDF)](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf)
- Lorenzen (2022) describes the common practice of fixing the scaling exponent and setting the intercept so that mean Lorenzen M in the mature population equals the constant M from an empirical estimator (Hoenig 1983, Then 2015). On this reading, tmax-based M is an average adult M. — [Lorenzen 2022](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf)
- Lorenzen (2022) on M1 vs M2:
  - In multispecies models, M1 "subsumes all causes of mortality not explicitly accounted for by predation terms (including any predators not explicitly modeled)" but is generally thought to be non-predation mortality.
  - It is "typically fixed a priori", usually at about the single-species constant M.
  - Compared to M2, "the M1 term has received scant scientific attention" even though it can substantially influence total mortality patterns.
  - Lorenzen also argues M1 should not be size- and age-independent.
  - Source: [Lorenzen 2022](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf)
- Lorenzen (2022), Fig. 6 shows North Sea cod: MSVPA total M(L) for 1981 and 1991 (from Pope et al. 2021) against the length-inverse Lorenzen curve. The predation models are roughly length-inverse for fish above about 15 cm and similar to or lower than the Lorenzen curve. Lorenzen suggests M1 may be higher and more size-dependent than the predation models assume. — [Lorenzen 2022](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf)
- Lorenzen, Camp & Garlock (2022, Fish Res 252:106327; abstract/snippet only) find the mortality-length scaling highly consistent at about -1 across empirical studies. — [SEDAR76-RD14 record](https://sedarweb.org/documents/sedar76rd14-natural-mortality-and-body-size-in-fish-populations-pdf)

North Sea SMS (WGSAM 2023 key run, report 2024; full text read):
- "M1 (residual natural mortality) by quarter is set to 0.05 for the species cod, whiting, haddock, saithe, the two sandeel stocks, Norway pout, sprat", i.e. 0.20/yr.
- For cod age 3+ since 2011, M1 is raised via M1adj = M1 - ln(1 - alpha) with alpha = 0.15 to mimic emigration to West of Scotland. This is a non-biological adjustment.
- Source: [WGSAM 2024 report, residual natural mortality section and cod section](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf)

SMS total natural mortality M = M1 + M2 (Table 5-14 of the same report; read values):

| Stock | 1991 | 2022 |
|---|---|---|
| Cod | age 0: 1.597; age 1: 0.986; age 2: 0.749; age 3: 0.283; age 4: 0.201; ages 5-10: 0.200 | age 0: 1.832; age 1: 1.293; age 2: 0.986; age 3: 0.656; age 4: 0.418; ages 5+: 0.353 (inflated by the migration adjustment) |
| Haddock, ages 0-10 | 0.987, 0.815, 0.435, 0.307, 0.285, 0.258, 0.227, 0.219, 0.206, 0.201, 0.200 | 1.277, 1.362, 0.528, 0.385, 0.299, 0.291, 0.283, 0.283, 0.283, 0.378, 0.200 |
| Whiting, ages 0-6 | 1.080, 1.017, 0.583, 0.501, 0.471, 0.471, 0.450 | 1.472, 0.964, 0.692, 0.528, 0.493, 0.423, 0.323 |

Source: [WGSAM 2024 report](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf)

- WGSAM 2024 advises that single-species assessments of haddock and whiting (among others) use the SMS M values. Cod was not in that 2024 list because of the new three-substock definition. — [DTU Orbit record of WGSAM 2024](https://orbit.dtu.dk/en/publications/10b7717f-f011-4d54-8d88-a815b2f06379)

### Inferences
Predation vs non-predation share in SMS (1991, the Ecopath reference year): M2 = M - M1, with M1 = 0.20.

| Stock and ages | M2 | Predation share of adult M |
|---|---|---|
| Cod age 4+ | 0.000-0.001 | About 0% (age 3: M2 0.083, about 29%) |
| Haddock ages 3-9 | 0.107 down to 0.001 | Roughly 35% at age 3, falling to about 0% at age 9 |
| Whiting ages 3-6 | 0.25-0.30 | About 55-60% |

In this framework, then, the non-predation part of adult gadoid M is 0.20/yr by assumption, and it equals essentially all of adult cod and saithe M. The 0.20 itself is a convention, not an estimate.

Size-based estimators at adult sizes, with L from W = 0.01 L^3 (an approximate length-weight relation, my assumption):

| Species, weight (length) | Gislason 2010 | Charnov 2013 | Lorenzen 1996 (3.0 W^-0.288) |
|---|---|---|---|
| Cod, 3 kg (67 cm) | 0.393 | 0.480 | 0.299 |
| Cod, 10 kg (100 cm) | 0.206 | 0.263 | 0.211 |
| Haddock, 0.5 kg (37 cm) | 0.646 (0.51 with Linf 60) | 0.716 | 0.501 |
| Haddock, 1 kg (46 cm) | 0.446 (0.36 with Linf 60) | 0.506 | 0.410 |
| Whiting, 0.2 kg (27 cm) | 0.752 | 0.788 | 0.652 |
| Whiting, 0.4 kg (34 cm) | 0.518 | 0.557 | 0.534 |
| Saithe, 2 kg (58 cm) | 0.422 | 0.508 | 0.336 |
| Saithe, 5 kg (79 cm) | 0.258 | 0.321 | 0.258 |
| Hake, 1 kg / 3 kg (Lorenzen only) | - | - | 0.410 / 0.299 |

Juvenile comparison at 20 cm (about 80 g):
- Gislason: cod 2.75, haddock 1.73, whiting 1.23, saithe 2.38.
- Charnov: 2.94, 1.79, 1.25, 2.54.
- Lorenzen 1996: 0.85 for any species.

Juvenile-to-adult ratio: by Gislason or Charnov a 20 cm cod has roughly 7-13 times the M of a 3-10 kg cod. By Lorenzen 1996 the ratio is about 3-4 times. The SMS ratio for age 1 vs age 5+ cod is about 5 times (0.986 vs 0.200).

Caveat from Lorenzen (2022): the Gislason and Charnov exponents (-1.61, -1.5) are among-population slopes, so these adult-size values (0.4-0.75 for haddock and whiting) probably overstate within-stock adult M. They are also total-M estimates. The size-based estimators nonetheless put adult total M at 0.2-0.4 for large cod and saithe and 0.4-0.75 for haddock and whiting at typical adult sizes, which is above SMS for haddock and comparable to SMS for whiting.

### Gaps
- I found no empirical decomposition of adult gadoid M into disease, parasites, starvation, senescence or spawning stress for North Sea stocks. Gislason et al. (2010) and Lorenzen (2022) give total M only, and SMS fixes M1.
- The Brodziak et al. (2011) review was not read.
- Jakobsen & Ajiad (1999) on spawning mortality in NEA cod was seen only as a citation in a search snippet.

## Q3. Direct estimates of adult natural mortality (tagging, unfished or lightly fished periods)

### Takeaway
Direct adult-M estimates for North Sea gadoids are essentially absent. Assessments use 0.2 by convention (cod, saithe, NEA cod). The few direct or semi-direct estimates I found:
- Skagerrak coastal cod tagging gives an implied M of about 0.5, for mostly young fish around 40 cm with heavy caveats.
- Norwegian fjord cod has Z of only 0.4-0.5.
- In the southern Gulf of St. Lawrence, adult cod M rose to about 0.8 (55-57% annual natural deaths), attributed mainly to grey seal predation.

### Cited Findings
- Kleiven et al. (2016, PLoS ONE 11:e0149595; full text read), Norwegian Skagerrak coast:
  - 9729 cod tagged in 2005-2013, mean 40 cm (range 15-93 cm).
  - Overall annual survival 33% (SE 1.5).
  - "Natural mortality was 44.4% (SE 2.5) of total mortality", estimated as the "dead by other causes" component of a multi-event mark-recovery model.
  - Source: [Kleiven et al. 2016](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0149595)
- Norwegian coastal cod release-recapture (Julliard et al. 2001; snippet only): natural mortality was high at 6-12 months, then fell rapidly and was no longer age-dependent. The adult value was not seen. — [Ecological Applications abstract](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/1051-0761(2001)011[0540:NMAFMI]2.0.CO;2)
- Norwegian fjord cod (snippet only): total mortality Z = 0.4-0.5/yr for age 2+, attributed mostly to low F (F <= 0.3). — [ResearchGate record](https://www.researchgate.net/publication/222045226_Sampling_and_a_mortality_model_of_a_Norwegian_fjord_cod_Gadus_morhua_L_population)
- Norwegian coastal cod long-term tagging (ICES JMS 2025; snippet only): Z about 0.4-0.9 at ages 3-6. M was assumed, not estimated. — [ICES JMS fsaf137](https://academic.oup.com/icesjms/article/82/8/fsaf137/8237823)
- NEA cod:
  - A PINRO working paper (Tretyak, ICES CM 2000/N:09; snippet only, document not opened) states that M "=0.20 for almost all age groups" was the assessment convention. The likely URL is [VLIZ copy](https://www.vliz.be/imisdocs/publications/ocrd/275899.pdf), unverified.
  - A later paper (snippet only; the PMC page was blocked) cites earlier work finding NEA cod M rose from about 0.15 to 0.25 as age at maturation fell from 8+ to 6. — [PMC3352523](https://pmc.ncbi.nlm.nih.gov/articles/PMC3352523/)
- An ICES working document (ICES CM 2001/O:03; snippet only) uses M = 0.2 for cod as an assumed input in a tagging analysis. — [ICES CM 2001/O:03](https://www.ices.dk/sites/pub/CM%20Doccuments/2001/O/O0301.pdf)
- Southern Gulf of St. Lawrence:
  - Swain et al. (MEPS 442:149-167; abstract via snippet) found that adult M is the demographic rate most limiting productivity of cod, white hake and winter skate, with indirect evidence consistent with grey seal predation. — [MEPS abstract](https://int-res.com/abstracts/meps/v442/p149-167)
  - Swain, Benoit & Hammill (AFS 2014; abstract) attributed the non-recovery to "dramatic increases in the natural mortality of adult cod", with only the grey seal hypothesis receiving strong support. — [AFS 2014 abstract](https://afs.confex.com/afs/2014/webprogram/Paper16609.html)
  - A 2016 model found grey seals could account for the excess M if adult cod were 30% of the average grey seal diet. — [AFS 2016 abstract](https://afs.confex.com/afs/2016/webprogram/Paper24555.html)
  - The 2019 DFO assessment, as reported by CBC: about 18% of adults would be expected to die of natural causes per year, versus an estimated 55-57%. — [CBC 2019](https://www.cbc.ca/lite/story/1.5255266)
- Eastern Georges Bank cod (title/abstract only): changes in natural mortality are documented. — [ADS abstract](https://ui.adsabs.harvard.edu/abs/2016JOUC...15..879W/abstract)

### Inferences
- Kleiven 2016 converted to an instantaneous rate under constant competing hazards (my calculation, not the authors'): Z = -ln(0.33) = 1.109/yr, so M = 0.444 x 1.109 = about 0.49/yr. Caveats:
  - "Dead by other causes" in a tag-recovery model absorbs tag loss, tag-induced mortality, permanent emigration and unreported catch, so this is an upper bound on biological M.
  - The fish were coastal and mostly young (mean 40 cm, below cod Lm of 65 cm).
  - It is nevertheless the only Skagerrak-specific field estimate found, and it is not low.
- Gulf of St. Lawrence converted from annual proportions to instantaneous rates (my calculation): 18% gives -ln(0.82) = 0.20; 55-57% gives -ln(0.45 to 0.43) = 0.80-0.84. So "normal" adult cod M = 0.2, and seal-driven M is about 4 times that. Two lessons follow:
  - Adult M in gadoids can be dominated by predation (by marine mammals) when seal populations are large.
  - The 0.2 convention describes a low-predation baseline.
- For the North Sea, SMS includes grey seals as predators, yet it gives cod age 4+ M2 of about 0 in 1991 and the 2022 values are explained by the migration term. The model therefore assigns essentially no predation to adult North Sea cod.

### Gaps
- I found no tagging-based or unfished-period (for example WWII) estimate of adult natural mortality for North Sea / Skagerrak cod, haddock, whiting or saithe, and none for Faroe saithe or cod.
- I did not retrieve the historical Newfoundland mark-recapture baselines (Myers et al. 1997) or a Faroe estimate.
- The original source of NEA cod "M 0.15-0.25" could not be verified.

## Q4. Biomass-weighted adult M for a pooled gadoid group vs M0 = 0.392/yr

### Takeaway
The adult-biomass (SSB) weighted TOTAL natural mortality of a pooled cod + haddock + whiting + saithe group is:
- about 0.25-0.31 with the maximum-age estimators at global tmax;
- about 0.33-0.40 with growth-based estimators;
- about 0.30-0.33 in the North Sea multispecies model (SMS);
- about 0.38-0.45 with maximum-age estimators at fished-stock tmax of 12-15 yr.

M0 = 0.392 is therefore about equal to, or above, the whole natural mortality these estimators assign to adult gadoids. It is about twice the non-predation residual M1 = 0.20 that SMS uses for all four species. If M0 is meant as a non-predation residual, it leaves little or no room for predation mortality on adults in the simulator unless total M is allowed to exceed the empirical range.

### Cited Findings
ICES Stock Assessment Graphs database, 2025 assessments (read via the SAG API). SSB in tonnes:
- Whiting (whg.27.47d, Subarea 4 and 7.d): 2018 156,387; 2022 355,582; 2023 385,309; 2024 358,464; 2025 409,187. — [SAG summary 19484](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19484)
- Haddock (had.27.46a20, Subarea 4, 6.a, SD 20): 2018 153,747; 2022 569,003; 2023 768,243; 2024 771,472; 2025 884,516. F in 2024 was 0.06. — [SAG summary 19508](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19508)
- Saithe (pok.27.3a46, 3.a, 4, 6): 2018 182,816; 2022 138,612; 2024 119,541; 2025 135,337. — [SAG summary 19578](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19578)
- Cod, Northern Shelf (cod.27.46a7d20), three substocks, 2024 SSB:
  - Northwestern 44,308 — [SAG summary 19661](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19661)
  - Southern 8,887 — [SAG summary 19662](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19662)
  - Viking 13,395 — [SAG summary 19663](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19663)
  - The substock series start in 1983.
- Northern hake (hke.27.3a46-8abd): SSB 2024 127,008 t, TSB 222,721 t. The stock spans 3.a, 4, 6, 7 and 8abd (mostly outside the North Sea). — [SAG summary 19777](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19777)
- For 1991 (the Ecopath reference year), the same 2025 SAG series give SSB of about cod 80 kt (sum of substocks), haddock 68 kt, whiting 227 kt and saithe 130 kt. These were read from the same API responses and printed by my script. — [SAG 19484](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19484), [19508](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19508), [19578](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19578), [19661](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19661), [19662](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19662), [19663](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19663) (cod is the sum of the three substock series)
- SMS M1 = 0.05/quarter (0.20/yr) for cod, whiting, haddock and saithe. — [WGSAM 2024 report](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf)

### Inferences
Weights are SSB shares, which suits an adult-dominated group. Hake is excluded because no North Sea share of the northern stock is available.

| Weighting | Cod | Haddock | Whiting | Saithe |
|---|---|---|---|---|
| 1991 (Ecopath year) | 15.9% | 13.4% | 44.9% | 25.8% |
| Mean 2005-2024 (kt: 71 / 254 / 188 / 192) | 10.1% | 36.0% | 26.7% | 27.2% |
| 2024 | 5.1% | 58.6% | 27.2% | 9.1% |

The 2024 split is dominated by the exceptional 2019-2020 haddock year classes and is not representative.

Per-species adult M used for each method:
- tmax and growth estimators: the Q1 values.
- Size-based estimators: the mean of the two adult weights in Q2.
- SMS 2022 adult M: cod 0.20 (migration term removed), haddock ages 3-9 mean 0.315, whiting ages 3-6 mean 0.442, saithe 0.20.
- SMS 1991: cod 0.20, haddock 0.243, whiting 0.473, saithe 0.20.

Weighted adult total M per year:

| Method | 1991 weights | 2005-24 mean | 2024 weights | M0 / weighted M (1991) |
|---|---|---|---|---|
| Hoenig 1983 fish (global tmax) | 0.191 | 0.193 | 0.203 | 2.05 |
| Hamel & Cope 2022 (global tmax) | 0.248 | 0.250 | 0.262 | 1.58 |
| Then 2015 tmax (global tmax) | 0.291 | 0.293 | 0.307 | 1.35 |
| Hamel & Cope, tmax 15/15/12/15 (cod/had/whg/pok) | 0.400 | 0.384 | 0.385 | 0.98 |
| Then tmax, tmax 15/15/12/15 | 0.452 | 0.435 | 0.435 | 0.87 |
| Pauly 1980 (T = 9 C) | 0.344 | 0.332 | 0.361 | 1.14 |
| Then 2015 growth | 0.380 | 0.365 | 0.399 | 1.03 |
| Lorenzen 1996 at adult weights | 0.445 | 0.429 | 0.469 | 0.88 |
| Gislason 2010 at adult lengths (likely overstated, see Q2) | 0.494 | 0.489 | 0.539 | 0.79 |
| Charnov 2013 at adult lengths (likely overstated) | 0.550 | 0.550 | 0.598 | 0.71 |
| SMS total M, 1991 values | 0.329 | - | 0.300 | 1.19 |
| SMS total M, 2022 values | 0.324 | 0.306 | 0.333 | 1.21 |
| SMS residual M1 only (non-predation) | 0.200 | 0.200 | 0.200 | 1.96 |

Interpretation:
1. The like-for-like comparison is M0 against M1. Ecopath "other mortality" M0 = P/B x (1 - EE) is the non-predation residual. The only North Sea number for that is SMS M1 = 0.20/yr, which is an assumed convention, not an estimate. M0 = 0.392 is about 2 times it.
2. M0 against total M: the central life-history estimators and SMS give a pooled adult total M of about 0.25-0.40. M0 = 0.392 alone is already at the top of that band.
   - If the simulator adds explicit predation (M2 from its seal, porpoise, seabird and cannibalism links) on top of M0, the implied total adult M would exceed most empirical estimates.
   - The exceptions are the steep size-based estimators (Gislason, Charnov) and the short-tmax variants. Lorenzen (2022) warns that Gislason and Charnov's among-population slopes overstate within-stock adult M.
3. A residual of about 0.39 can be reconciled with biology in three ways:
   - (a) the pooled group carries a large juvenile share (age 1-2 gadoids have M 0.4-1.3 in SMS);
   - (b) M0 also absorbs non-biological losses (emigration, unreported catch), as the SMS cod migration term does: 0.20 becomes about 0.35;
   - (c) the true tmax of the fished stocks is 12-15 yr, so M is about 0.36-0.45.
   - None of these makes 0.392 a non-predation residual for adults in the sense of SMS M1.
4. Whiting is the one species where 0.392 is close to total adult M (SMS 0.44-0.47). Even there, more than half of whiting M is predation (M2 about 0.25-0.30), so the non-predation part is still about 0.20.

### Gaps
- I did not identify the Ecopath 1991 North Sea model's gadoid group composition, P/B or EE, so I could not check how 0.392 was derived (for example whether it includes juveniles or discards). That is outside this researcher's scope.
- Skagerrak-only biomass shares are not separable from the ICES stock units:
  - cod includes 6.a and 7.d;
  - haddock includes 6.a;
  - whiting includes 7.d;
  - saithe includes 3.a and 6.
- Hake cannot be weighted.
- SSB is used instead of total biomass. SAG did not provide TSB for these four stocks, and including immature fish would raise the weighted total M.
