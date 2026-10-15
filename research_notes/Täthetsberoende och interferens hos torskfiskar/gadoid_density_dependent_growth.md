# Density-dependent growth, weight-at-age and condition in gadoids: how strong is the empirical brake, and what does it mean for a per-capita intake term?

Scope note: about 60 tool calls (searches, page fetches, PDFs converted with pdftotext, figure pages rendered and read by eye). The table at the end says how each source was read: full text, via a page-summarising fetch (weaker, because a second model filtered its numbers), abstract only, or search snippet only.

**Not verified:**
- the North Sea haddock 1999 year class;
- the Barents Sea haddock 2004-2006 growth reduction (ICES advice snippet only);
- any quantitative whiting weight-at-age versus density relation;
- the NE Arctic cod papers by Marshall et al., Jorgensen and Ottersen named in the brief. These were not reached; Jorgensen 1992 appears only second-hand, via de Cardenas. NE Arctic cod is covered here by Denechaud et al. 2020, Nilssen et al. 1994, de Cardenas 1997 and Mehl 1991.

**Correction to the brief:** Svedang & Hornborg 2014 is in *Nature Communications* 5:4152, not ICES JMS.

"(computed)" marks my own arithmetic on a published relation; those numbers are not in the source.

---

## Q1. Cross-stock syntheses

### Takeaway
Density-dependent growth after recruitment is common across stocks: 56 % of 16 stocks in Lorenzen & Enberg (2002), 26 % or 69 % of ~70 NE Atlantic stocks in Zimmermann et al. (2018) depending on method, and 54 % (33 of 61) for "late growth" in Rindorf et al. (2022). It is usually weaker than density dependence in recruitment. The Lorenzen & Enberg cross-stock law (g = 3.3 B^-1) implies that going from the mean biomass to twice the mean lowers asymptotic length by about 3.3 cm in every stock. That is roughly 3 % of L-infinity for cod-sized fish and 8-10 % for 30-40 cm fish. In that dataset none of the North Sea gadoids (cod, haddock, Norway pout) had significant density dependence; the only significant gadiform entry was Gulf of Maine silver hake. Rindorf et al. give the only gadoid-inclusive magnitude I found: in demersal stocks with significant density dependence, late weight-at-age falls by a median of about 30 % from the minimum to the maximum observed total stock biomass (read off their figure).

### Cited Findings
- Lorenzen & Enberg (2002) use a density-dependent von Bertalanffy model with "L_inf(B) = L_inf(L) - g B, where the competition coefficient g describes the decline in asymptotic length per unit of biomass density". The abstract says: "We analyse 16 fish populations with long-term records of size-at-age and biomass data, and detect significant density-dependent growth in nine." Across the nine significant populations, "Regression analysis demonstrated a highly significant (p < 0.001) relationship of the form g = 3.3 B^-1.0". (Fetched full text) - [Lorenzen & Enberg 2002, Proc R Soc B 269:49-54](https://doi.org/10.1098/rspb.2001.1853)
- Lorenzen & Enberg Table 1, gadiform rows (B in kg/ha; g in cm ha kg^-1 with 95 % CI):
  - (g) North Sea cod: B range 6-17, mean 9.8; L_inf(L) 121.7, L_inf(B) 119.0 (ratio 0.98); g = 0.281 [-0.300, 0.870], not significant.
  - (h) North Sea haddock: B 5-104, mean 20; g = -0.196 [-0.510, 0.039], a negative point estimate, not significant.
  - (p) North Sea Norway pout: B 9-30, mean 16; g = 0.108 [-0.060, 0.280], not significant.
  - (i) Gulf of Maine silver hake (*Merluccius bilinearis*): B 5-69, mean 27; L_inf 42.5 -> 38.4 (ratio 0.90); g = 0.156 [0.070, 0.371], significant.
  
  Following Beverton & Holt (1957), competition for food "is expected to affect the asymptotic size (L_inf or W_inf) of fishes, but not the growth rate (K)". Across all 16 stocks, the ratio of L_inf at mean biomass to L_inf at zero biomass is 0.86-1.10. The authors attribute the negative haddock estimate to "population and time-specific factors" and cite Marshall & Frank (1999) for density-dependent growth in haddock elsewhere. Note: the printed L_inf(L) and L_inf(B) columns for rows (o) and (p) look swapped relative to their ratios; the g values are unaffected. (Fetched full text) - [Lorenzen & Enberg 2002](https://doi.org/10.1098/rspb.2001.1853)
- Zimmermann, Ricard & Heino (2018) fitted Ricker-type models to 70 NE Atlantic and Baltic populations, with g_a = alpha_G,a * exp(beta_G,a * B). B is SSB normalised to its mean, so exp(beta) is the growth multiplier when SSB goes from the mean to twice the mean.
  - "Statistically significant density dependence in recruitment was detected in a large proportion of populations (70%), whereas for density dependence in somatic growth the prevalence of density dependence depended heavily on the method (26% and 69%)": 26 % with weight increments, 69 % with weight-at-age.
  - "97% of populations with statistically significant parameters for growth and recruitment showed a larger impact of density-dependent recruitment on population biomass."
  - Significant *positive* relations were "much more frequent for growth (22% for Gall and from 29% (G2) to 55% (G3)) than for recruitment (4%)", including "various populations of Atlantic cod, Atlantic herring, haddock".
  - Growth density dependence was stronger than recruitment density dependence in western Baltic cod among others, and, with the weight-at-age method, "whiting (Merlangius merlangus) in the North Sea".
  
  (Fetched full text) - [Zimmermann et al. 2018, J Anim Ecol 87:672-681](https://doi.org/10.1111/1365-2656.12800)
- Rindorf et al. (2022), 80 NE Atlantic stocks:
  - "Thirty-seven of the 61 stocks (61%) investigated exhibited a significant correlation between total stock biomass and late growth", split into 33 negative and 4 positive. The probability of a significant negative correlation was 0.54 (95 % CI 0.42-0.66).
  - Quadrant analysis: 45 of 61 stocks showed below-average growth at above-average biomass (demersal: 19 of 27).
  - Early (juvenile) growth was "generally density-independent". Among significant negative cases the early-growth effect "ranged from -5% to -29%".
  - Late growth is weight-at-age in standard-deviation units, averaged over three ages 1-3 years above the youngest reliable age.
  - In 20-year moving windows, demersal correlations ranged from large negative to moderate positive.
  
  (Fetched full text) - [Rindorf et al. 2022, Fish Fish 23:812-828](https://doi.org/10.1111/faf.12650)
- Rindorf et al. (2022) Figure 6, read by eye from the rendered page: the "growth effect" is the relative change in late weight-at-age from minimum to maximum observed total stock biomass. For all 27 demersal stocks the median is about -0.15 (interquartile range about -0.3 to +0.05). For the 13 demersal stocks with significant correlations the median is about -0.30 (IQR about -0.40 to -0.22), with two outliers near -0.75 and -0.8. Benthic (10) and pelagic (10) significant stocks have similar medians, about -0.28 and -0.35. The paper does not list which demersal stocks are in each box. (Figure read; values approximate) - [Rindorf et al. 2022](https://doi.org/10.1111/faf.12650)
- Rochet (2000, ICES JMS 57:1160): exploited populations "exhibit earlier maturity, a faster growth rate, increased fecundity". No gadoid numbers were retrieved. (Snippet) - [Archimer record](https://archimer.ifremer.fr/doc/00000/485/)

### Inferences
- Lorenzen & Enberg's g = 3.3 B^-1 is a statement about the elasticity at the mean. g * B_mean = 3.3 cm in every significant population, so a linear extrapolation from the mean to twice the mean lowers L_inf by 3.3 cm. **(computed)** As a fraction of L_inf: 2.8 % for L_inf = 120 cm (cod), 5.5 % at 60 cm, 8.2 % at 40 cm, 10.3 % at 32 cm (their median-stock model fish). The corresponding drops in asymptotic weight (L^3) are 8 %, 16 %, 23 % and 28 %.
- In the Beverton-Holt derivation, L_inf = H/k, where H is the anabolic coefficient (food intake per unit absorptive surface). **(computed)** A fractional drop in L_inf is therefore the same fractional drop in per-capita intake per unit surface. Lorenzen & Enberg thus imply that per-capita intake falls by about 3-10 % per doubling from mean biomass, with the low end for large fish such as cod.
- Converted to the model's units (1 kg/ha = 0.1 t/km2) **(computed)**: North Sea cod 0.6-1.7 t/km2 (mean ~1.0), North Sea haddock 0.5-10.4 (mean 2.0), Norway pout 0.9-3.0 (mean 1.6), silver hake 0.5-6.9 (mean 2.7). These are stock-wide average densities over the stock area, not local school densities.
- Rindorf's -30 % weight-at-age over the observed biomass range must be divided over the number of doublings in that range, which the paper does not give per stock. For typical assessment ranges (max/min SSB or TSB of 3-10x, i.e. 1.6-3.3 doublings) it gives **(computed)** roughly -9 to -19 % weight-at-age per doubling for the significant demersal stocks, and about half that for the median demersal stock (-15 % over the range). Weight-at-age is proportional to L_inf^3 at fixed K, so the implied per-capita intake change is about one third: **-3 to -6 % per doubling** for significant stocks.

---

## Q2. Cod: Baltic, North Sea, Northeast Arctic, Gulf of St Lawrence / Newfoundland

### Takeaway
The best-controlled estimate is Denechaud et al. (2020) for Northeast Arctic cod. Over a 7-fold range of cod biomass, with temperature, capelin, AMO and fishing in the same model, cod biomass lowers otolith growth by 21.8 %, about -8 % per doubling (computed). The Baltic gives the most extreme and most confounded numbers. Svedang & Hornborg's fit implies L_inf more than halves per doubling of cod numbers. However, Casini et al. find cod abundance non-significant over 1976-2014 once hypoxia, sprat and seals are included. The roughly 50 % drop in Barents Sea weight-at-age in the 1980s came with prey collapse at moderate cod biomass. Gulf of St Lawrence studies rank size-selective fishing above density.

### Cited Findings
- **NE Arctic cod, otolith biochronology 1924-2014 (3,894 fish).** "Cod growth was negatively related to cod population size and positively related to capelin population size". Fishing "had a significant but weak negative direct impact". Cod and capelin biomass enter the model log-transformed. Table 3 gives the predicted change in growth over the observed predictor range:
  - cod biomass 707-4,999 x10^3 t: **-21.78 %**
  - capelin biomass 101-7,806 x10^3 t: +4.67 %
  - Kola temperature 2.85-5.36 C: +8.48 %
  - AMO -0.44 to 0.36: -7.20 %
  - fishing mortality 0.21-1.02: -2.78 %
  
  Cod biomass was "the strongest" effect. (Fetched full text) - [Denechaud et al. 2020, GCB 26:5661](https://doi.org/10.1111/gcb.15298)
- **NE Arctic cod, 1980s.** "Individual growth was dramatically reduced, and the average fish weight decreased by about 50% in most age groups." Cannibalism "increased by a factor of three from 1984 to 1986" and capelin was "seriously depleted" by the end of 1986. Cod biomass was low: about 1 Mt in the early 1980s, just over 1.5 Mt in 1986, against about 6 Mt after WWII. (Fetched abstract) - [Mehl 1991, Polar Res 10:525](https://doi.org/10.3402/polar.v10i2.6763)
- **NE Arctic cod since about 1955.** Specific growth rate is inversely related to numbers at age; a density-or-food plus temperature model fits ages 4-9. (Fetched abstract, paraphrased by the fetch tool) - [Nilssen et al. 1994, ICES Mar Sci Symp 198:449](https://www.cefas.co.uk/data-and-publications/publications/list/publication-abstract/?id=12049)
- **NE Arctic, Iceland and NAFO 2J3KL cod: mean weight-at-age in the stock.** For ages under 7, "the maximum of the series could double the minimum value". "The annual increases seemed to have little relationship with the stock biomass. Nevertheless, the maximums of biomass usually brought about declines in growth rates." Growth relates to temperature and to the capelin/cod biomass ratio. (Fetched full text) - [de Cardenas 1997, NAFO Sci Coun Stud 24:97](https://www.nafo.int/Portals/0/PDFs/Studies/s24/cardenas.pdf)
- **Across cod stocks, temperature as the dominant driver.** The mean weight of a 4-year-old cod "ranges from about 0.6 kg at Labrador to 7.3 kg in the Celtic Sea". A temperature-times-age model "accounts for 92% of the variance for 2-4-year-old cod from 17 stocks". (Snippet of the abstract) - [Brander 1995, ICES J Mar Sci 52:1-10](https://doi.org/10.1016/1054-3139(95)80010-7)
- **Eastern Baltic cod, condition 1976-2014 (Casini et al. 2016).**
  - After the mid-1990s, Fulton's K fell "approximately 20-22%" for 30-59 cm cod, 15 % for 20-29 cm and 11 % for 10-19 cm.
  - 1976-1993: cod abundance was "the only predictor of cod condition (negative effect)", explaining 36.4 % of deviance (F = 9.17, p = 0.008).
  - 1994-2014: sprat biomass (single-predictor deviance 75.5 %) and hypoxic area (65.6 %) were retained, and cod abundance was not.
  - Whole series: cod abundance F = 2.71, **p = 0.11**, deviance contribution 2.33 %, single-predictor deviance 2.03 %. By comparison, hypoxic areas 67.2 % single-predictor (p = 0.0001), seals 63.0 %, sprat 39.4 %. Final model 85.9 %.
  - Replacing abundance with "cod density" (abundance / suitable oxygenated area, which shrank ~30 %) improved the fits. The authors propose hypoxia acts partly "via crowding and density-dependent processes".
  - All predictors were standardised anomalies, so no elasticity can be read off.
  
  (Fetched full text including tables) - [Casini et al. 2016, R Soc Open Sci 3:160416](https://doi.org/10.1098/rsos.160416)
- **Eastern Baltic cod, L_inf by year class 1995-2008 (Svedang & Hornborg 2014).** Figure 6: "L_inf,i = 204.5 * e^(-4.022e-9 * N_sj); multiple r2 = 0.54; P < 0.001". N_sj is the number of fish aged >= 2 years in the assessment, spanning about 220-400 million. L_inf fell from about 100-130 cm for early-1990s year classes to about 40-45 cm (Figure 3). Mean landed weight in the 1-2 kg category fell, and "between 2006 and 2011, mean landing size declined by 30%". The authors attribute the process to increased gear selectivity raising densities of sub-legal fish. (Fetched full text and figures) - [Svedang & Hornborg 2014, Nat Commun 5:4152](https://doi.org/10.1038/ncomms5152)
- **Eastern Baltic cod, feeding level.** Neuenfeldt et al. (2020) find feeding levels of small cod "indicate severe growth limitation", "likely the result of a decrease in benthic prey abundance due to increased hypoxic areas", and "amplified by stunted growth leading to high densities of cod of smaller size". No numbers are given in the text. (Fetched via summariser) - [Neuenfeldt et al. 2020](https://doi.org/10.1093/icesjms/fsz224)
- **Eastern Baltic cod in distress.** Eero et al. (2015) report "low nutritional condition and disappearance of larger individuals" and do not attribute them to density. (Fetched abstract) - [Eero et al. 2015](https://doi.org/10.1093/icesjms/fsv109)
- **Southern Gulf of St Lawrence cod.**
  - Sinclair, Swain & Hanson (2002): "The strongest effect was variation in size-selective mortality, followed by a negative effect of population density and a weak positive effect of occupied temperature". (Fetched abstract) [doi](https://doi.org/10.1139/f02-014)
  - Hanson & Chouinard (1992): the slow-growing 1977-1982 year classes came at low F and high density. The fishery removed "four times as many fish from the upper than lower quartiles", so selective fishing alone can explain the pattern. (Fetched abstract) [doi](https://doi.org/10.1111/j.1095-8649.1992.tb03168.x)
  - Crabtree & Ware (1975, 1949-1973 data): growth rose with temperature and fell with stock size. (Snippet) [pdf](https://www.nafo.int/icnaf/library/docs/1975/res-98.pdf)
- **North Sea cod.**
  - Lorenzen & Enberg found a non-significant g (above).
  - Rijnsdorp et al. (1991) "found no evidence of density dependent growth, of juveniles or adults, in the North Sea for plaice, sole (Solea solea) or cod". This is quoted second-hand in Taylor & Stefansson 1999 (Fetched full text of the citing paper).
  - Baudron et al. (2014): over 40 years "six of eight commercial fish species in the North Sea examined underwent concomitant reductions in asymptotic body size", coinciding with "a 1-2 C increase in water temperature". Press coverage says cod "did not conform" (snippet). (Fetched abstract)
  
  - [Baudron et al. 2014, Glob Change Biol 20:1023-1031](https://doi.org/10.1111/gcb.12514)

### Inferences
- **Denechaud per doubling (computed).** 4,999/707 = 7.07, i.e. 2.82 doublings. -21.78 % / 2.82 = **-7.7 % otolith growth per doubling of cod biomass** (-8.3 % if treated as multiplicative). Otolith increment width tracks length increment. In a von Bertalanffy fish with K fixed, the relative change in length increment at a given size equals the relative change in mass-specific production at that size. That relative change is ΔL_inf / (L_inf - L) = (ΔH/H) / (1 - L/L_inf).
  - So per-capita *production* falls by about 8 % per doubling.
  - The underlying *intake* coefficient H falls by only 8 % x (1 - L/L_inf), i.e. about **3-5 % per doubling** for fish at 40-60 % of L_inf.
  - This is the cleanest number in the set. It has a 91-year record, partials out temperature and prey, and is the strongest single extrinsic effect in that model. It also includes cannibalism as a channel, which the authors name, and the model no longer has cannibalism.
- **Svedang & Hornborg per doubling (computed).** The local elasticity d ln L_inf / d ln N = -4.022e-9 N is -1.0 at N = 250 million and -1.6 at 400 million. From 200 to 400 million, L_inf goes from 91.5 to 40.9 cm, -55 % in length and about -90 % in asymptotic weight. That is an order of magnitude above any other estimate. It is a correlation over 14 year classes during which hypoxia, prey loss and parasites all increased. Treat it as a confounded upper bound, not a calibration target.
- **Baltic, pre-1994 only.** This is the period where Casini et al. find a clean density signal. Cod abundance alone explains 36 % of condition variance. The data are standardised anomalies, so no slope can be computed from the fetched text.
- **Ratio dependence.** De Cardenas and Mehl point the same way. For NE Arctic cod the strongest weight-at-age swings (up to 2x, and the -50 % of the 1980s) follow the capelin/cod ratio more than cod biomass alone. A capelin collapse at moderate cod biomass did far more damage than a 7-fold change in cod biomass did in Denechaud's partial effect.

---

## Q3. Haddock, whiting, saithe

### Takeaway
Haddock has the best-known cases of strong year classes growing slowly: Georges Bank 2003 and 2013, the North Sea 1960s, Iceland after 2003. Yet I found no published per-doubling estimate, and Lorenzen & Enberg give North Sea haddock a negative, non-significant g. The Georges Bank 2003 year class was about 18x the recent average; its weight-at-age fell below the 1986-2000 average while its growth in length did not. Icelandic haddock condition spanned only K 0.71-0.79 over a roughly 6-10x density range. Saithe's growth coefficient correlates with saithe abundance (r = -0.67), but this is confounded with a Norway pout collapse. The documented whiting slowdown is driven by forage fish.

### Cited Findings
- **Eastern Georges Bank haddock, 2003 year class (TRAC 2008).**
  - "The exceptional 2003 year class, estimated at 322.7 million fish at age 1, is the largest". The other recent year classes are "below the 18 million average of the 10 most recent year classes (excluding the 2003 year class)".
  - "Both length and weight at age have generally declined since about 2000. While size at age increased in 2008 for the younger age groups, weights remained below the 1986 to 2000 average, except for age 1."
  - "The size at age for the 2003 year class is smaller than previous year classes, but its rate of growth at length is similar to previous year classes".
  - "DFO survey average weights at length, used to reflect fish condition, exhibit a declining trend since the late 1990s."
  - The numeric weights are only in figures, which could not be extracted.
  
  (Fetched full text) - [TRAC Status Report 2008/02](https://www.bio.gc.ca/info/intercol/trac-cert/documents/reports/TSR_2008_02_E.pdf)
- **Eastern Georges Bank haddock, 2013 year class (TRAC 2019).** The 2014 TRAC gave a preliminary estimate of 1,546 million age-1 fish (snippet). The 2019 report says "There has been a general decline in weights at age since the late 1990s", "As biomass has increased, growth rates and asymptotic length have declined", and "This decline in size at age is exacerbated for the 2013 year class". The 2013 year class is "expected to have very little gains in weight from age 6 to 8". No numeric values were given in the fetched text. (Fetched via summariser) - [TRAC 2019 status report, NOAA repository](https://repository.library.noaa.gov/view/noaa/24763)
- **Icelandic haddock, 1985-2022 (Jonsdottir et al. 2024).**
  - "Density-dependent factors influenced haddock mean weight, which dropped considerable during times of high haddock density." "Haddock condition was found to be influenced by both temperature and haddock density."
  - Ranges: density 0.82-8.07 t/km2 (north) and 1.35-7.46 (south); age-2 mean weight 87-239 (north) and 127-234 (south) (unit unlabelled in the fetch, presumably g); Fulton K 0.71-0.78 (north) and 0.71-0.79 (south).
  - Density had a negative effect on weight (F = 21.25, p < 0.001), K (F = 6.34, p = 0.012) and liver index (F = 4.35, p = 0.037). No slope or percentage appears in the part read.
  
  (Fetched via summariser, first ~100k of 150k characters) - [Jonsdottir et al. 2024, ICES J Mar Sci 81:587-599](https://doi.org/10.1093/icesjms/fsad108)
- **Icelandic haddock (Taylor & Stefansson 1999).** "the abundant 1989 year-class is followed by the slower growing 1990 and 1991 year-classes". "The first abundant year-class does not seem affected by its own abundance. Density dependence may however take the form of reduced growth due to the total size of the population older than the cohort". North Sea 1960s: "large year-classes initially showed low growth ... but after the youngest ages growth increased". Counter-case: 1960s lengths-at-age matched the 1950s "despite the stock biomass being considerably higher" (Jones 1983). Caveat from Bromley 1989: survival of small fish in good year classes can mimic density-dependent growth. (Fetched full text) - [Taylor & Stefansson 1999, J Northw Atl Fish Sci 25:101-114](https://journal.nafo.int/Portals/0/1999/taylor.pdf)
- **Barents Sea (NE Arctic) haddock.** The ICES advice says the 2004-2006 year classes "are estimated to be very strong" and that "density-dependent growth has been observed for this stock and the present growth rate is low". Not verified beyond the snippet, and no magnitude. (Snippet) - [ICES advice NEA haddock (jointfish copy)](https://www.jointfish.com/content/download/259/4924/file/had-arct.pdf)
- **North Sea haddock, 1999 year class.** No source found that quantifies its growth. UNVERIFIED.
- **North Sea saithe, 1987-2012 (Cormon et al. 2016).**
  - The logistic-growth coefficient K_LG (median 0.18 /yr, range 0.10-0.39) "was found to be negatively correlated (with a 1 year time lag) with saithe total abundance (r = -0.67)" and positively with Norway pout biomass (r = 0.41). Temperature was not significant.
  - "Density-dependence and prey availability explained together 46.79% of the model deviance". Density explained 29.67 % and food 17.12 %.
  - The stated slope ("When saithe abundance increased by 659 000 individuals ... K_LG, dropped by 0.01") cannot be reconciled with Figure 5a, whose axis runs 200-500 "(1000 individuals)". The text elsewhere says "sNB < 200 000 t". I do not use the slope.
  - Reading Figure 5a: abundance about 210-270 (axis units) in 1991-1996 with K_LG 0.18-0.26, and about 420-470 in 2002-2005 with K_LG 0.10-0.12. Norway pout biomass fell from about 600-900 kt to about 150-250 kt over the same years (Figure 5b).
  
  (Fetched full text, author version) - [Cormon et al. 2016, MEPS 542:141-151](https://doi.org/10.3354/meps11559)
- **North Sea whiting.**
  - Zimmermann et al. 2018 list North Sea whiting among the few stocks where growth density dependence exceeded recruitment density dependence with the weight-at-age method (Fetched full text).
  - Lauerburg et al. (2018): length-at-age fell in parallel with forage-fish availability between 2000 and 2007 and recovered after 2007. Mean stomach content mass in the low-prey period was "on average 60% less" (Snippet; the abstract was not retrievable from the publisher, which returned HTTP 429).
  - Grubbe & Gislason (1997): "significant decrease in length at age" 1980-1995, most pronounced at older ages (Snippet).
  - No quantitative whiting weight-at-age versus whiting-density relation was found.
  
  - [Zimmermann et al. 2018](https://doi.org/10.1111/1365-2656.12800); [Lauerburg et al. 2018, MEPS 594:213-230](https://doi.org/10.3354/meps12533); [Grubbe & Gislason 1997, DTU Orbit](https://orbit.dtu.dk/en/publications/changes-in-lenght-at-age-of-north-sea-whiting-1980-95/)

### Inferences
- **Haddock** looks the most density-sensitive gadoid in the qualitative record, because single cohorts can be 10-100x their neighbours. Even so, the 18x Georges Bank 2003 cohort (about 4 doublings) slowed weight gain but not growth in length, and no percentage is published. Lower weight-at-length with unchanged length growth is a condition (reserve) effect, the model's own channel.
- **Icelandic haddock condition bound (computed).** The K range of 0.71-0.79 is 11 % over a roughly 6-10x density range (2.6-3.3 doublings), with temperature also varying. Even if all of it were density, that is **at most about 3-4 % condition per doubling**. The age-2 weight range of 87-239 (2.7x) is much larger, but it mixes temperature, area and year-class effects.
- **Saithe (computed, crude).** Abundance roughly doubled from the mid-1990s to 2002-2005 while K_LG fell from about 0.22 to about 0.11, i.e. about -50 % per doubling. But food (Norway pout) fell about 3-4x over the same years, and the authors' own partition gives density only 30 % of deviance. K_LG of a logistic length model is not per-capita production. I would not use this as more than "saithe shows a real but confounded density signal".
- **Whiting** ranks highest for growth density dependence in Zimmermann et al., but the one documented episode is prey-driven: stomach contents fell 60 %. That is an intake loss caused by prey, not by predator density.

---

## Q4. Translation to model terms

### Takeaway
Where density dependence is detected, per-capita intake falls by **3-6 % (at most about 10 %) per doubling of stock biomass**, and per-capita production by **5-15 %** for mid-sized fish (computed below). For North Sea cod and haddock no effect is detectable. The model's g*(s-u) term amplifies intake losses the way the von Bertalanffy anabolism-minus-catabolism term does, so calibrate the intake term to the smaller intake number.

### Cited Findings
(This section uses the numbers already cited in Q1-Q3. No new sources.)

### Inferences
1. **Weight-at-age to intake identity (computed, from the Beverton-Holt / von Bertalanffy derivation used by Lorenzen & Enberg).** dW/dt = H W^(2/3) - k W, so L_inf = H/k and W(t) = W_inf (1 - e^(-K(t-t0)))^3 with K = k/3. If density acts only on H (food intake per unit surface):
   - d ln L_inf = d ln H
   - d ln W(age) = 3 d ln H at every age
   - d ln (production at fixed size) = d ln H / (1 - L/L_inf)
   
   So:
   - A weight-at-age drop of X % means an intake drop of about X/3 %.
   - A length-asymptote drop of X % means an intake drop of X %.
   - The production drop at fixed size is the intake drop times 1/(1 - L/L_inf), i.e. 1.7x at L = 0.4 L_inf and 2.5x at 0.6 L_inf.
2. **Per-doubling estimates on a common scale (computed):**

   | Source | Raw effect | Per doubling, intake (H) | Per doubling, production at fixed size |
   |---|---|---|---|
   | Lorenzen & Enberg cross-stock law, cod-size fish (L_inf 120 cm) | -3.3 cm L_inf, mean to 2x mean | -3 % | -5 to -8 % |
   | Lorenzen & Enberg law, 30-40 cm fish | same | -8 to -10 % | -14 to -25 % |
   | Denechaud NEA cod (partial effect, 7x range) | -21.8 % otolith growth | -3 to -5 % | -8 % |
   | Rindorf significant demersal stocks (median) | about -30 % late weight-at-age over the observed range | -3 to -6 % (assuming 1.6-3.3 doublings) | -6 to -15 % |
   | Rindorf all demersal stocks (median) | about -15 % | -1.5 to -3 % | -3 to -7 % |
   | Iceland haddock condition (upper bound) | K 0.71-0.79 over 6-10x | <= -3 to -4 % (condition) | - |
   | North Sea cod, haddock, Norway pout (Lorenzen & Enberg) | g not significant (haddock negative) | about 0 | about 0 |
   | Baltic cod L_inf (Svedang & Hornborg), confounded | L_inf 91 -> 41 cm, 200 -> 400 M fish | -55 % | not meaningful |

3. **Defensible range.**
   - For "per-capita intake falls by X % when gadoid biomass doubles": X = 3-6 %, central about 4-5 %, upper plausible about 10 %.
   - For "per-capita production falls by X % when gadoid biomass doubles": X = 5-15 %, central about 8-10 %.
   - Zero is a legitimate value for a North Sea / Skagerrak cod-haddock pool, given the North Sea entries.
   
   These are long-run equilibrium responses integrated over years. Year-to-year noise from temperature and prey is of the same size or larger (Denechaud: temperature +8.5 %, AMO -7.2 %, capelin +4.7 % over their ranges; de Cardenas: weight-at-age varies up to 2x between years).
4. **Mapping onto the Beddington-DeAngelis term (computed).** Per-capita intake is I = a B_prey / (D + w B_pred), with D = 1 + a h B_prey. Doubling B_pred at fixed prey and fixed spatial distribution gives a fractional intake drop of r = w B_pred / (D + 2 w B_pred). Solving gives w B_pred / D = r / (1 - 2r):
   - r = 3 % -> w B_pred = 0.032 D
   - r = 5 % -> 0.056 D
   - r = 10 % -> 0.125 D
   
   So the interference term should be a small fraction (about 3-13 %) of the prey-handling denominator D at the typical gadoid cell density. Near the Holling ceiling (large a h B_prey) D is large, and the same w gives a proportionally smaller effect. The library currently has `interference: 1.0` for gadoids. Whether that is too strong or too weak depends on the typical gadoid t/km2 per occupied cell relative to 1 + a h B_prey. That cell density should be read off a run before w is set; this note did not do so.
5. **Stock-wide versus local density.** All empirical estimates use stock-wide biomass, and stocks expand their range as they grow. Jonsdottir et al. 2024 document distributional shifts, and Casini et al. found that abundance per suitable area fitted better than abundance. A cell-level w therefore gives a weaker stock-level brake if the policy disperses. It gives a stronger one in colonies, whose density can be far above the stock-wide 0.6-2 t/km2 of North Sea cod and haddock.
6. **Confounders in the empirical numbers:**
   - Prey and ratio dependence. The model has prey explicitly, so w should be *smaller* than raw correlations suggest.
   - Temperature. The model already applies Q10 to metabolism.
   - Size-selective fishing. Absent in the model.
   - Hypoxia. Baltic only.
   - Cannibalism. Part of Denechaud's density channel; the model has removed it, so interference may carry a modest share of that effect.
   - Survivorship bias in strong year classes (Bromley 1989). This inflates apparent density dependence.

---

## Implications for a density-dependent term on gadoids in the model

- The empirical brake on gadoid growth from their own density is **weak and inconsistent**. Where it is detected, per-capita intake falls by about 3-6 % (at most about 10 %) per doubling of biomass, and per-capita production by about 5-15 %. In North Sea cod and haddock it is often undetectable. A term tuned to stop 35-40 %/yr biomass growth outright through density alone would be much stronger than the literature supports. The literature supports a **modest interference coefficient w**, not a hard cap.
- Calibrate w on the **intake** side. With the BD form, choose w so that w B_typical is about 3-13 % of D = 1 + a h B_prey at typical occupied-cell gadoid densities (central about 5-6 %, i.e. r about 5 %). The model's g*(s - u) term will then amplify that into a production loss of the right order, about 1.5-3x, through the reserve s, just as the anabolism-minus-catabolism structure does in fish. Calibrating w directly to weight-at-age changes (about 3x intake) or to production changes would overshoot.
- Check the result as **condition (reserve s) versus biomass**. A doubling of gadoid biomass at fixed prey should move the equilibrium reserve by a few percent, comparable to the Icelandic haddock Fulton K range of 0.71-0.79 over about 3 doublings, not by tens of percent.
- What should actually stop a 35-40 %/yr gadoid increase in the real system is mainly **prey depletion** (ratio dependence: capelin/cod, Norway pout/saithe, forage fish/whiting), **temperature**, fishing and, historically, cannibalism. Where intra-gadoid density did coincide with large weight losses (Barents Sea 1980s at about -50 %, Baltic 2000s), prey collapse or hypoxia was the dominant documented driver. If the model's gadoids grow unchecked, the first thing to verify is whether gadoid predation is depleting herring and zooplankton prey in the cells where gadoids feed. If prey is not being depleted enough to feed back, that may be a more faithful fix than a larger w.
- Report the brake as **stock-level production versus stock-level biomass** in viability or inference runs, so it can be compared with the 5-15 % per doubling range (see Q4 item 5).
- Do not use the Baltic L_inf law of Svedang & Hornborg (about -55 % L_inf per doubling) as a target. It is confounded by hypoxia, prey loss and parasites, and Casini et al. find cod abundance non-significant over 1976-2014 once these are included.

---

## Source status

| Source | Read as | Used for |
|---|---|---|
| Lorenzen & Enberg 2002 ([doi](https://doi.org/10.1098/rspb.2001.1853)) | Full text | g = 3.3/B law; gadoid rows |
| Zimmermann et al. 2018 ([doi](https://doi.org/10.1111/1365-2656.12800)) | Full text | Prevalence; positive DD; NS whiting |
| Rindorf et al. 2022 ([doi](https://doi.org/10.1111/faf.12650)) | Full text + figures | Late-growth DD 54 %; effect sizes |
| Rochet 2000 ([record](https://archimer.ifremer.fr/doc/00000/485/)) | Snippet | Context only |
| Denechaud et al. 2020 ([doi](https://doi.org/10.1111/gcb.15298)) | Full text | NEA cod -21.78 % -> about -8 %/doubling |
| Mehl 1991 ([doi](https://doi.org/10.3402/polar.v10i2.6763)) | Abstract | -50 % weight; capelin collapse |
| Nilssen et al. 1994 ([record](https://www.cefas.co.uk/data-and-publications/publications/list/publication-abstract/?id=12049)) | Abstract (paraphrased) | NEA cod SGR vs numbers |
| de Cardenas 1997 ([pdf](https://www.nafo.int/Portals/0/PDFs/Studies/s24/cardenas.pdf)) | Full text | 2x weight-at-age swings; capelin/cod ratio |
| Brander 1995 ([doi](https://doi.org/10.1016/1054-3139(95)80010-7)) | Snippet of abstract | Temperature confounder |
| Casini et al. 2016 ([doi](https://doi.org/10.1098/rsos.160416)) | Full text incl. tables | Baltic condition GAMs |
| Svedang & Hornborg 2014 ([doi](https://doi.org/10.1038/ncomms5152)) | Full text + figures | Baltic L_inf law (confounded) |
| Neuenfeldt et al. 2020 ([doi](https://doi.org/10.1093/icesjms/fsz224)) | Summariser; abstract verified | Baltic feeding level |
| Eero et al. 2015 ([doi](https://doi.org/10.1093/icesjms/fsv109)) | Abstract | Baltic context |
| Sinclair, Swain & Hanson 2002 ([doi](https://doi.org/10.1139/f02-014)) | Abstract | Selection > density > temperature |
| Hanson & Chouinard 1992 ([doi](https://doi.org/10.1111/j.1095-8649.1992.tb03168.x)) | Abstract | Fishing-selectivity confounder |
| Crabtree & Ware 1975 ([pdf](https://www.nafo.int/icnaf/library/docs/1975/res-98.pdf)) | Snippet | Early Gulf result |
| Baudron et al. 2014 ([doi](https://doi.org/10.1111/gcb.12514)) | Abstract (+ press snippet) | Temperature and shrinking L_inf |
| Taylor & Stefansson 1999 ([pdf](https://journal.nafo.int/Portals/0/1999/taylor.pdf)) | Full text | Haddock cohort effects; caveats |
| TRAC 2008/02 ([pdf](https://www.bio.gc.ca/info/intercol/trac-cert/documents/reports/TSR_2008_02_E.pdf)) | Full text | Georges Bank 2003 YC |
| TRAC 2019 ([record](https://repository.library.noaa.gov/view/noaa/24763)) | Summariser | Georges Bank 2013 YC |
| Jonsdottir et al. 2024 ([doi](https://doi.org/10.1093/icesjms/fsad108)) | Summariser (partial) | Iceland haddock condition range |
| ICES advice NEA haddock ([pdf](https://www.jointfish.com/content/download/259/4924/file/had-arct.pdf)) | Snippet | 2004-06 YCs (unverified) |
| Cormon et al. 2016 ([doi](https://doi.org/10.3354/meps11559)) | Full text (author version) + figure | Saithe K_LG vs abundance |
| Lauerburg et al. 2018 ([doi](https://doi.org/10.3354/meps12533)) | Snippet | Whiting vs forage fish |
| Grubbe & Gislason 1997 ([record](https://orbit.dtu.dk/en/publications/changes-in-lenght-at-age-of-north-sea-whiting-1980-95/)) | Snippet | Whiting length-at-age trend |
