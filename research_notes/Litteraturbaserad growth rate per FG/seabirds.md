# Seabirds (Skagerrak/Kattegat, North Sea): P/B, r_max / lambda_max and demographic rates for a `growth_rate` (g) parameter

Model context used for conversions (from the assignment): 6 h tick, 1460 ticks/yr; growth term dB = B * g * (s - u) per tick with (s - u) <= ~0.3; residual natural mortality M = 0.117/yr; current g = 0.001/tick -> ceiling gross P/B = 0.001 * 0.3 * 1460 = 0.438/yr. Per-tick values below are per-year / 1460 (rates treated as instantaneous; for lambda use r = ln(lambda) first). Where I computed a number myself (no direct literature value) it is marked **[computed]** and the inputs are cited.

## 1. P/B and Q/B for seabirds in Ecopath models (North Sea, Kattegat/Skagerrak, Baltic, elsewhere) and how they were derived

### Takeaway
Ecopath seabird P/B values found are 0.18-0.4/yr (North Sea 0.28 in the table, 0.4 stated in the text; Iceland 0.177). They are taken from generic marine-mammal/bird sources (Trites et al. 1999) or match an annual mortality rate. None comes from a seabird production budget. Seabird Q/B differs widely between models (38 to 217/yr).

### Cited Findings
- Mackinson & Daskalov (2007), North Sea Ecopath (Cefas Sci. Ser. Tech. Rep. 142), Table 3.1 / balanced model, group 4 "Seabirds": B = 0.003 t/km2, **P/B = 0.28 /yr**, **Q/B = 216.56 /yr**, unassimilated 0.2, P/Q = 0.0013, TL 3.5. Listed references: "ICES, 1996, 2002; Trites et al., 1999" — [Mackinson & Daskalov 2007, Cefas Tech. Rep. 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- The same report's species chapter (section 15.4, Birds) says something different: "Biomass and consumption rate (Q/B) were taken from information used in the MSVPA analyses (ICES, 2002): B = 0.003 t km-2 Q/B = 216. Production rate **P/B = 0.4** was taken from Trites et al. (1999)." So the text says 0.4 and the parameter table says 0.28. The species included are fulmar, gannet, shag, herring gull, great black-backed gull, lesser black-backed gull, kittiwake, terns, guillemot, razorbill, puffin and great skua. The diet is mainly fish (sandeel), some zooplankton, and discards/offal — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- For comparison, the same report derived the seal P/B from r_max: "maximum rate of population growth rate for pinnipeds is believed to be about 12% yr-1 ... The P/B ratio was therefore set at 6%, half of the maximum as used by Trites et al. (1999)" (seals P/B = 0.09 in the table). This is the Trites et al. (1999) convention of P/B = 0.5 * r_max for long-lived groups — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Icelandic waters Ecopath (Ribeiro et al., arXiv 1810.00613, 2019), Table A.2: Seabirds B = 2.831 kt, **P/B = 0.177 /yr**, **Q/B = 38.00 /yr**. The seabird inputs are attributed to Lilliendahl & Solmundsson (1997) — [Ribeiro et al. 2019, "An overview of the marine food web in Icelandic waters using Ecopath with Ecosim"](https://arxiv.org/pdf/1810.00613)
- Western Baltic EwE (Scotti et al. 2022, Front. Mar. Sci.): the seabird P/B was sourced from Tomczak et al. (2009) and Q/B from Mendel et al. (2008). The numbers are only in Supplementary Table S6, which I did not retrieve — [Scotti et al. 2022](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.879998/full)
- Southern Bight of the North Sea Ecopath (v2). Search snippets report "seabirds (discard)" P/B = 0.10 /yr and "seabirds (non-discard)" P/B = 1.12 /yr, plus the figures 60.52 and 48.20, labelled "total mortality". Those two figures look like Q/B values; one snippet attributes Q/B = 60.52 to the empirical formula of Nilsson & Nilsson (1976). **UNVERIFIED**: the page returned HTTP 403 and the snippets are internally inconsistent — [Southern Bight Ecopath v2 (ResearchGate)](https://www.researchgate.net/publication/376517317_Ecopath_model_of_the_Southern_Bight_of_the_North_Sea_version_2)

### Inferences
- With P/B = Z at Ecopath steady state, 0.18-0.28/yr implies a population-wide mortality (juveniles included) of 18-28%/yr. That is consistent with the Leslie-model Z of 0.15-0.20/yr for the main species, computed in section 4 [computed].
- The 0.4/yr in the M&D text comes from the Trites et al. (1999) marine-mammal/bird convention, not from seabird data. 0.28 (table) and 0.177 (Iceland) are the more defensible Ecopath anchors.
- Converted to the model: P/B 0.177 / 0.28 / 0.40 per yr = **1.21e-4 / 1.92e-4 / 2.74e-4 per tick** (gross). With (s-u)max = 0.3 that is **g = 4.0e-4 / 6.4e-4 / 9.1e-4 per tick** [computed]. Ecopath P/B is a steady-state average, though, not a ceiling for a growing population.

### Gaps
- Christensen (1995) North Sea 1981 model: seabird P/B/Q/B not retrieved.
- Harvey et al. (2003) Baltic ([doi](https://dx.doi.org/10.1016/S1054-3139(03)00098-5)) and Tomczak et al. (2012, Ambio; [link](https://link.springer.com/article/10.1007/s13280-012-0324-z)): full text paywalled or redirected. I could not confirm whether they include a seabird group or what its P/B is.
- No Kattegat/Skagerrak-specific Ecopath model with seabird parameters was found.
- Trites et al. (1999), the original source of the 0.4 value, was not retrieved.

## 2. r_max / lambda_max: demographic-invariant estimates and observed maximal colony growth

### Takeaway
The demographic invariant method (Niel & Lebreton 2005), run with UK demographic rates, gives r_max of only 0.06-0.12/yr for the auks, gannet, fulmar and large gulls. Kittiwake, cormorant, eider and terns come out at 0.13-0.22/yr. Observed sustained colony growth (Skomer and Stora Karlsö guillemots about 5%/yr; Bass Rock gannets about 4.4%/yr; UK gannets 1.3-2.2%/yr) lies below these ceilings. Only cormorant colonisation waves, driven by immigration, exceeded 20%/yr.

### Cited Findings
- Niel & Lebreton (2005, Conserv. Biol. 19:826-835) estimate the maximum annual growth rate from adult survival s and age at first breeding a alone, using the invariance r_max * T_opt ≈ 1 (lambda_max^T ≈ e, i.e. ≈ 2.7 per generation) across 13 bird species growing at near-optimal rates — [Niel & Lebreton 2005 (ResearchGate)](https://www.researchgate.net/publication/227700284_Using_Demographic_Invariants_to_Detect_Overharvested_Bird_Populations_from_Incomplete_Data); [NOAA review, Cortés](https://repository.library.noaa.gov/view/noaa/51791/noaa_51791_DS1.pdf)
- O'Brien, Cook & Robinson (2017, J. Environ. Manage. 201:163-171) give the closed form: lambda_max ≈ [(s*a - s + a + 1) + sqrt((s - s*a - a - 1)^2 - 4*s*a^2)] / (2a), and R_max = lambda_max - 1. They follow Dillingham & Fletcher (2008) in using the **highest published** adult survival (kittiwake s = 0.911, Frederiksen et al. 2004; a = 4) as a precaution, because higher s gives a lower lambda_max. They conclude that PBR/R_max-based harvest rules can mislead in seabird impact assessments — [O'Brien et al. 2017 (BTO PDF)](https://www.bto.org/sites/default/files/publications/obrien_cook_robinson_j._env._management_2017.pdf)
- O'Brien et al. (2017), Table 1: a kittiwake Leslie model (adult and immature survival 0.86, juvenile 0.79, a = 4) gives lambda = 1.02 with productivity 0.70, 1.00 with 0.56, and 0.98 with 0.44 chicks/pair (density-independent) — [O'Brien et al. 2017](https://www.bto.org/sites/default/files/publications/obrien_cook_robinson_j._env._management_2017.pdf)
- DIM lambda_max **[computed]** with the formula above, s = adult survival and a = age of recruitment from JNCC Report 552 Table 33 ([Horswill & Robinson 2015](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf)). Per tick = r_max/1460.

| Species | s | a | lambda_max | r_max /yr | r_max /tick |
|---|---|---|---|---|---|
| Common guillemot | 0.939 | 6 (5) | 1.079 (1.089) | 0.076 (0.085) | 5.2e-5 (5.8e-5) |
| Common guillemot, BTO s | 0.946 | 5 | 1.085 | 0.081 | 5.6e-5 |
| Razorbill | 0.895 | 5 (4) | 1.109 (1.127) | 0.103 (0.120) | 7.1e-5 (8.2e-5) |
| Atlantic puffin | 0.906 | 5 (4) | 1.105 (1.122) | 0.100 (0.115) | 6.8e-5 (7.9e-5) |
| Black-legged kittiwake | 0.854 | 4 (3) | 1.144 (1.177) | 0.135 (0.163) | 9.2e-5 (1.1e-4) |
| Kittiwake, max s (D&F 2008 rule) | 0.911 | 4 | 1.120 | 0.113 | 7.7e-5 |
| Northern gannet | 0.919 | 5 (4) | 1.099 (1.115) | 0.094 (0.109) | 6.5e-5 (7.5e-5) |
| Northern fulmar | 0.936 | 9 (8) | 1.061 (1.066) | 0.059 (0.064) | 4.0e-5 (4.4e-5) |
| Herring gull | 0.834 | 5 (4) | 1.128 (1.151) | 0.120 (0.140) | 8.2e-5 (9.6e-5) |
| Great black-backed gull | 0.930 | 5 (4) | 1.094 (1.109) | 0.090 (0.103) | 6.1e-5 (7.1e-5) |
| Lesser black-backed gull | 0.885 | 5 (4) | 1.113 (1.132) | 0.107 (0.124) | 7.3e-5 (8.5e-5) |
| Sandwich tern | 0.898 | 3 (2) | 1.154 (1.202) | 0.143 (0.184) | 9.8e-5 (1.3e-4) |
| Great cormorant | 0.868 | 3 (2) | 1.170 (1.226) | 0.157 (0.204) | 1.1e-4 (1.4e-4) |
| Common eider | 0.886 | 3 (2) | 1.161 (1.212) | 0.149 (0.192) | 1.0e-4 (1.3e-4) |
| Common eider, BTO s | 0.916 | 3 | 1.142 | 0.133 | 9.1e-5 |
| Red-throated diver | 0.840 | 3 (2) | 1.184 (1.246) | 0.169 (0.220) | 1.2e-4 (1.5e-4) |

  (Values in parentheses use a one year earlier recruitment age, as a sensitivity check. BTO s values are from BirdFacts; see section 3.)
- Skomer guillemots "increased at an almost constant rate of 5% per annum in the last 30 years (Meade et al 2013)". Over the same period the Isle of May, Fair Isle and Canna colonies declined. UK guillemots "increased rapidly in all regions of the UK between 1969 and 1985", then the increase slowed — [Horswill & Robinson 2015, JNCC 552, p. 65](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf); the Skomer increase is "explained by intrinsic demographic properties" (Meade et al. 2013) — [ResearchGate](https://www.researchgate.net/publication/260104714_The_population_increase_of_common_guillemots_Uria_aalge_on_Skomer_Island_is_explained_by_intrinsic_demographic_properties)
- Stora Karlsö (Baltic), the largest Baltic guillemot colony: the increase is reported at about 5%/yr (5.1%/yr over the last 11 years before ~2016), the colony has almost tripled since 1980, and it is ~15,700 pairs in 2014 and ~27,500 pairs in 2021. **Verified only via search-result snippets**; the primary paper (Hentati-Sundberg et al.) was not opened — [Stockholm Resilience Centre 2016](https://www.stockholmresilience.org/research/research-news/2016-03-23-less-common-in-the-past.html)
- UK & Ireland gannets: the breeding population grew "by an average of 2% per annum from 1969-1985, 2.2% per annum from 1985-1995, and 1.33% per annum from 1995-2005". The PVA (stochastic, density-independent) gave 1.28%/yr against 1.33% observed, and 0.87%/yr with density dependence — [WWT Consulting 2012, SOSS-04 Gannet PVA (BTO-hosted)](https://www.bto.org/sites/default/files/u28/downloads/Projects/Final_Report_SOSS04_GannetPVA.pdf)
- Bass Rock: 75,259 apparently occupied sites in 2014, +24% since 2009 (≈ 4.4%/yr). It had grown for >100 years until the HPAI outbreak of 2022, after which the colony fell by ~25-30% — [Murray, Wanless & Harris 2015, British Birds](https://britishbirds.co.uk/journal/article/bass-rock-now-worlds-largest-northern-gannet-colony); [Scottish Field](https://www.scottishfield.co.uk/wildlifeandconservation/the-largest-gannet-colony-in-the-world-at-bass-rock-has-shrunk-by-30/) (via search snippet)
- Great cormorant, Baltic/Kattegat region. Denmark: maximal growth 43% in 1972 and an average of 23.8%/yr over 1978-1992 (Bregnballe & Gregersen 1997; Van Eerden & Gregersen 1995). Sweden: maximal 43% in 1992 and ~30%/yr over 1987-1994 (Engström 2001; Lindell 1997). W. Germany averaged 45%/yr (1986-1992), Poland 55%/yr (1988-1992), Estonia 92%/yr (1989-1993). Finland averaged 123%/yr over 1997-2004, but **84% of that growth was immigration** — [Lehikoinen 2006, Ornis Fennica 83:34-46](https://www.ymparisto.fi/sites/default/files/documents/2006_Cormorants%20in%20Finland_OrnFenn_Lehikoinen.pdf)
- Density dependence: JNCC 552 found nine studies on five species in which colony growth slowed as colonies grew. Porter & Coulson (1987) found kittiwake colony growth limited by attractive central sites; herring gulls find it harder to establish territories in dense colonies (Chabrzyk & Coulson 1976) — [Horswill & Robinson 2015](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf)

### Inferences
- Taking ln(1.05) = 0.049/yr for guillemots at Skomer and Stora Karlsö, the best-documented intrinsic growth sits at about 60% of the DIM r_max (0.076-0.085). That is consistent with DIM being a ceiling.
- Colony-level rates above the DIM ceiling (cormorants at 24-123%/yr) involve immigration. They are not intrinsic r for a closed population, so they should not be used for a biomass pool with no immigration term. The exception is if the model area is meant to absorb birds from outside; the Skagerrak offshore area is mostly a foraging and wintering area, not a breeding area.
- For an offshore Skagerrak/Kattegat seabird assemblage dominated by auks, kittiwake, gannet, fulmar and large gulls, a defensible community r_max is **~0.08-0.13/yr (≈ 5.5e-5 to 9e-5 per tick)**. 0.15-0.20/yr is an upper bound that is only plausible if cormorant, eider, terns or divers dominate the biomass.

### Gaps
- Dillingham & Fletcher (2008, Biol. Conserv. 141:1783-1792; 2011) were not opened directly. Their "use max survival" rule is cited via O'Brien et al. (2017).
- The Niel & Lebreton (2005) paper itself was not opened (paywalled); the formula is quoted from O'Brien et al. (2017). Note [computed]: the closed form solves (lambda-1)*T = 1 with T = a + s/(lambda-s). The implicit r_max*T = 1 form (ln(lambda)*T = 1) gives slightly higher values: guillemot (0.939, 6) 1.083 vs 1.079; kittiwake (0.854, 4) 1.158 vs 1.144; gannet (0.919, 5) 1.106 vs 1.099. That is ~5-10% higher r_max, which does not change the recommendation.
- No Skagerrak/Kattegat-specific (Swedish west coast) colony growth series was retrieved for kittiwake, gulls or eider. Swedish west-coast kittiwake colonies (e.g. Hallö/Bohuslän) are known to be small and declining, but I found no citable number in this search.

## 3. Demography: age at first breeding, clutch size, breeding success, survival, chick mass at fledging

### Takeaway
The main offshore species are long-lived with low fecundity. Auks and gannet lay 1 egg and fledge about 0.6-0.7 chicks/pair. Adult survival is 0.85-0.94, first-year survival 0.42-0.80, and recruitment comes at age 4-6 (fulmar 9). Guillemot chicks leave the colony at only ~20-25% of adult mass, so most of their somatic production happens at sea after fledging.

### Cited Findings
- JNCC Report 552, Table 33 (UK national weighted means; juv = 0-1 yr survival; productivity = chicks fledged per pair; age = modal age of recruitment) — [Horswill & Robinson 2015, "Review of Seabird Demographic Rates and Density Dependence", JNCC Report 552](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf):

| Species | Juv (0-1) | Immature | Adult | Productivity | Age recruit |
|---|---|---|---|---|---|
| Common guillemot | 0.560 | 0.792 (1-2), 0.917 (2-3) | 0.939 | 0.672 | 6 |
| Razorbill | * (return rate only) | 0.630 (1-2) | 0.895 | 0.570 | 5 |
| Atlantic puffin | * | 0.709, 0.760, 0.805 (2-5) | 0.906 | 0.617 | 5 |
| Black-legged kittiwake | 0.790 | - | 0.854 | 0.690 | 4 |
| Northern gannet | 0.424 | 0.829, 0.891, 0.895 | 0.919 | 0.700 | 5 |
| Northern fulmar | * (0.26 to recruitment) | - | 0.936 | 0.419 | 9 |
| Herring gull | 0.798 | - | 0.834 | 0.920 | 5 |
| Lesser black-backed gull | 0.820 | - | 0.885 | 0.530 | 5 |
| Great black-backed gull | n/a | - | 0.930 | 1.139 | 5 |
| Great cormorant | 0.540 | - | 0.868 | 1.985 | 3 |
| European shag | 0.513 | 0.737 | 0.858 | 1.303 | 2 |
| Sandwich tern | 0.358 | 0.741 | 0.898 | 0.702 | 3 |
| Common tern | * | 0.441 / 0.850 | 0.883 | 0.764 | 3-4 |
| Arctic tern | n/a | - | 0.837 | 0.380 | 4 |
| Common eider | 0.200 | - | 0.886 | 0.379 | 3 |
| Common scoter | 0.749 | - | 0.783 | 1.838 | 3 |
| Long-tailed duck | n/a | - | 0.730 | 1.900 | 2 |
| Red-throated diver | 0.600 | 0.620 | 0.840 | 0.571 | 3 |
| Great northern diver | * | 0.770 | 0.870 | 0.543 | 6 |

- Guillemot details (JNCC 552 Table 29): productivity in the 1st breeding year 0.620 and from the 2nd year onwards 0.686; regional means North 0.629, East 0.659, West 0.823; national 0.672 (SD 0.147); missed breeding 0.079; natal dispersal 0.580 — [Horswill & Robinson 2015](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf)
- Gannet: mean breeding success 0.698 chicks per apparently occupied nest (SD 0.089) over nine UK colonies, 1961-2009, with the highest at Bass Rock (0.769) and the lowest at Troup Head (0.601) — [WWT Consulting 2012, SOSS-04](https://www.bto.org/sites/default/files/u28/downloads/Projects/Final_Report_SOSS04_GannetPVA.pdf)
- BTO BirdFacts (body mass; clutch; age at first breeding; adult and juvenile survival):
  - Guillemot: 890.9 ± 73.8 g (770-1010 g); 1 egg; 1 brood; fledging (colony departure) 18-25 d; first breeding at 5 yr; adult survival 0.946; juvenile 0.56 — [BTO BirdFacts Guillemot](https://www.bto.org/understanding-birds/birdfacts/guillemot)
  - Razorbill: 612.6 ± 52.8 g; 1 egg (1-2); 14-24 d; 4 yr; adult 0.90; juvenile 0.38 (to age 4) — [BTO BirdFacts Razorbill](https://www.bto.org/understanding-birds/birdfacts/razorbill)
  - Kittiwake: 367.7 ± 37.5 g; 2 eggs (1-3); fledging 33-54 d; 4 yr; adult 0.882; juvenile 0.79 — [BTO BirdFacts Kittiwake](https://www.bto.org/understanding-birds/birdfacts/kittiwake)
  - Gannet: ~3 kg; 1 egg; fledging 84-97 d; 5 yr; adult 0.919 ± 0.002; juvenile 0.30 (to age 4) — [BTO BirdFacts Gannet](https://www.bto.org/understanding-birds/birdfacts/gannet)
  - Herring gull: 970.6 ± 152.3 g; 3 eggs (2-4); 35-40 d; 4 yr; adult ~0.88 (sex-specific values reported); juvenile 0.63 (to age 4) — [BTO BirdFacts Herring Gull](https://www.bto.org/understanding-birds/birdfacts/herring-gull)
  - Eider: 2.13 ± 0.2 kg; 4-6 eggs (1-8); 65-75 d; 3 yr; adult 0.916 ± 0.017; juvenile 0.33 ± 0.06 — [BTO BirdFacts Eider](https://www.bto.org/understanding-birds/birdfacts/eider)
- Guillemot chick departure mass: chicks leave Stora Karlsö at an average age of 19-21 d, weighing **20-25% of adult body mass**. Semi-precocial young leave at ~1/4 of adult mass, accompanied at sea by the male. **Verified via search-result snippet only** (page 403) — [Environmental variability and fledging body mass of Common Guillemot chicks (ResearchGate)](https://www.researchgate.net/publication/257377501_Environmental_variability_and_fledging_body_mass_of_Common_Guillemot_Uria_aalge_chicks); chicks at sea grow about twice as fast as chicks at the colony — [McGill University news, 2017](https://www.mcgill.ca/newsroom/channels/news/why-guillemot-chicks-leap-nest-they-can-fly-266829)
- Cormorant (Finland): adult survival 88% and 1st-year survival 58% (Frederiksen & Bregnballe 2000b), as used by Lehikoinen. Nestling mortality after day 20 was ~1-1.5% — [Lehikoinen 2006](https://www.ymparisto.fi/sites/default/files/documents/2006_Cormorants%20in%20Finland_OrnFenn_Lehikoinen.pdf)

### Inferences
- Single-egg species (guillemot, razorbill, puffin, gannet, fulmar) have a hard ceiling of 1 chick per pair per year, i.e. ≤0.5 fledglings per breeding adult. Observed values are 0.57-0.82 per pair.
- Kittiwake (2 eggs), gulls (3 eggs), cormorant (3-4) and eider (4-6) have higher fecundity ceilings, which matches their higher DIM r_max.

### Gaps
- Fledging-to-adult mass ratios for razorbill (thought to be similar to guillemot's, ~1/4-1/3), kittiwake, gulls and gannet were **not verified** in this search. Gannet fledglings are commonly said to be heavier than adults at fledging (Nelson 2002), and gull and kittiwake chicks fledge near adult mass; neither claim is sourced here.
- No Swedish west-coast demographic estimates (productivity, survival) were retrieved. The UK values are assumed transferable.

## 4. First-principles gross biomass production per unit population biomass, and what it implies for g

### Takeaway
A stage-structured calculation with the JNCC rates gives a gross P/B for stationary seabird populations of about **0.13-0.20/yr** (fulmar 0.09-0.11; cormorant ~0.3-0.39). Adding r_max to the whole-population mortality gives a ceiling gross P/B for a growing population of ~0.22-0.32/yr for the main offshore species. With the model's M = 0.117/yr, the defensible ceiling is gross P/B = r_max + M ≈ **0.19-0.25/yr**, i.e. **g ≈ 4.4e-4 to 5.6e-4 per tick** (≈ 6e-4 at most). The current g = 0.001 (gross 0.438/yr, net r ≈ 0.32/yr, lambda ≈ 1.38/yr) is 2-4x above any seabird r_max in the literature for a closed population.

### Cited Findings
- The input demographic rates are those of JNCC 552 Table 33 ([Horswill & Robinson 2015](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf)) and the guillemot departure mass of ~0.20-0.25 x adult mass ([ResearchGate snippet](https://www.researchgate.net/publication/257377501_Environmental_variability_and_fledging_body_mass_of_Common_Guillemot_Uria_aalge_chicks)). The Leslie post-breeding-census structure and R_max/PBR framework follow [O'Brien et al. 2017](https://www.bto.org/sites/default/files/publications/obrien_cook_robinson_j._env._management_2017.pdf).
- **[computed]** Leslie model with JNCC rates. Productivity p/2 per breeding adult; all individuals breed from the recruitment age. Where juvenile survival was missing, it was solved so that lambda = 1 (razorbill 0.82; GBBG 0.16, implausible; puffin, implausible because JNCC only has return rates — **treat razorbill/puffin/GBBG rows as unreliable**). B = (individuals aged ≥1 + fledglings * (1 + s_juv)/2) * adult mass. Gross P = fledglings/yr * effective mass. The low bound takes effective mass = departure mass (0.25 * adult). The mid bound adds post-fledging growth of survivors (0.25 + 0.75 * sqrt(s_juv)). The high bound takes each fledgling at full adult mass. Adult somatic growth is taken as 0, since seasonal fat cycling is not net production.

| Species | lambda with JNCC rates | Breeders / (age ≥1) | Fledglings per bird aged ≥1 | Z, ages ≥1 | Z, all incl. juv | Gross P/B low / mid / high (/yr) |
|---|---|---|---|---|---|---|
| Common guillemot | 1.034 | 0.60 | 0.20 | 0.079 | 0.145 | 0.044 / 0.142 / 0.175 |
| Black-legged kittiwake | 1.016 | 0.59 | 0.21 | 0.146 | 0.160 | 0.043 / 0.159 / 0.173 |
| Northern gannet | 1.007 | 0.68 | 0.24 | 0.094 | 0.198 | 0.051 / 0.150 / 0.203 |
| Herring gull | 1.007 | 0.47 | 0.22 | 0.166 | 0.176 | 0.045 / 0.167 / 0.181 |
| Lesser black-backed gull | 1.012 | 0.59 | 0.16 | 0.115 | 0.125 | 0.034 / 0.126 / 0.136 |
| Northern fulmar | 1.000 (solved) | 0.59 | 0.12 | 0.064 | 0.113 | 0.028 / 0.089 / 0.113 |
| Razorbill (unreliable) | 1.000 (solved) | 0.62 | 0.18 | 0.143 | 0.151 | 0.038 / 0.140 / 0.151 |
| Sandwich tern | 0.971 | 0.81 | 0.29 | 0.131 | 0.264 | 0.060 / 0.167 / 0.239 |
| Great cormorant | 1.165 | 0.56 | 0.55 | 0.132 | 0.271 | 0.097 / 0.310 / 0.387 |
| Common eider | 0.921 | 0.93 | 0.18 | 0.114 | 0.230 | 0.040 / 0.093 / 0.159 |
| Red-throated diver | 0.941 | 0.77 | 0.22 | 0.191 | 0.237 | 0.047 / 0.155 / 0.187 |

  (The 0.25 departure-mass fraction is verified only for guillemot. For species fledging near adult mass, such as gulls and kittiwake, the "high" column is the relevant one.)
- Ecopath anchors to compare against: 0.177 (Iceland) and 0.28 (North Sea table) — section 1.

### Inferences
- **The mass actually produced inside colonies is small.** For guillemot, chick biomass leaving the colony is only ~0.04-0.05 x population biomass per year. The rest of the "recruitment" production (to ~0.15-0.18/yr) is post-fledging growth at sea. Both count as gross production in the model's single pool. The model has no seasonal breeding, so the whole amount is spread over the year.
- **Stationary check against the model's mortality:** gross P/B at lambda ≈ 1 comes out at 0.13-0.20/yr. For a pool with M = 0.117/yr, the stationary production should equal 0.117/yr. That is close, and implies a mean (s-u) at equilibrium of 0.117/(1460 g). The model's M = 0.117 is close to the ages ≥1 mortality (0.08-0.17); the literature whole-population Z including first-year birds is higher, 0.15-0.20.
- **Ceiling gross P/B for a growing population** [computed], as r_max + Z with literature values: guillemot 0.076-0.085 + 0.145 ≈ **0.22-0.23**; kittiwake 0.11-0.16 + 0.16 ≈ **0.27-0.32**; gannet 0.094-0.109 + 0.198 ≈ **0.29-0.31**; herring gull 0.12-0.14 + 0.18 ≈ **0.30-0.32**; fulmar 0.06 + 0.11 ≈ **0.17**; cormorant 0.16-0.20 + 0.27 ≈ **0.43-0.47**; eider 0.13-0.19 + 0.23 ≈ **0.36-0.42**.
- **What to use in this model:** mortality is fixed at M = 0.117/yr, so the growth ceiling must reproduce r_max *given that M*. The ceiling is gross P/B_max = r_max + 0.117, and g = (r_max + 0.117) / (1460 * 0.3). Using r_max + Z_lit instead would double-count juvenile mortality the model does not have.

| Community r_max (/yr) | Gross P/B_max (/yr) | Gross per tick | g per tick at (s-u)max = 0.3 | Basis |
|---|---|---|---|---|
| 0.076 | 0.193 | 1.32e-4 | **4.4e-4** | guillemot DIM (a=6) |
| 0.10 | 0.217 | 1.49e-4 | **5.0e-4** | auk/gannet/GBBG typical DIM |
| 0.13 | 0.247 | 1.69e-4 | **5.6e-4** | kittiwake/herring gull DIM |
| 0.15 | 0.267 | 1.83e-4 | **6.1e-4** | upper, kittiwake/eider/cormorant mix |
| 0.20 | 0.317 | 2.17e-4 | **7.2e-4** | cormorant/diver/tern with a-1 (extreme) |
| current g = 0.001 | 0.438 | 3.0e-4 | 1.0e-3 | net r = 0.32/yr, lambda = 1.38/yr: not supported for a closed population |

- **Recommended defensible pick** (auk/kittiwake/gannet-dominated offshore Skagerrak assemblage): **g ≈ 5e-4 per tick** (gross P/B_max ≈ 0.22/yr, r_max ≈ 0.10/yr), with a plausible range of 4.4e-4 to 6e-4. This also lies between the Ecopath steady-state values of 0.177 and 0.28. If (s-u) rarely reaches 0.3, the realised ceiling is lower; g should then be computed from the realistic (s-u) at maximal feeding rather than 0.3.
- Caveat: DIM r_max is an upper bound for optimal conditions. The best-documented sustained growth is ~0.05/yr (Skomer and Stora Karlsö guillemots). So even g = 5e-4 lets the pool outgrow real auk populations when food is unlimited. A conservative alternative is r_max ≈ 0.05-0.08, giving **g ≈ 3.8e-4 to 4.4e-4**.

### Gaps
- The calculation ignores egg production of failed breeding attempts, adult moult and fat-cycle production, so the gross P/B is slightly underestimated. Feathers are a few % of body mass per year; not quantified here.
- Departure-mass fractions other than guillemot's are unverified (section 3). The guillemot fraction itself is from a snippet.
- The community composition (biomass shares by species) of seabirds in the Swedish offshore Skagerrak/Kattegat area was not researched. The community r_max weighting is therefore a judgement, not a sourced number.
