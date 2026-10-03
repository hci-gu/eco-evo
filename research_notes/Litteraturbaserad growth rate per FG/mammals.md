# Marine mammals (harbour porpoise, harbour seal, grey seal): P/B, r_max and demography, for a gross-production `growth_rate` ceiling

Conventions in these notes: all rates are per year unless marked; per-6h-tick value = per-year / 1460.
lambda = finite annual rate of increase; r = ln(lambda) (instantaneous). For lambda <= 1.13 the two differ by
<= 0.012 (1.12 -> r 0.113; 1.10 -> 0.095; 1.04 -> 0.039).
Verification level is marked on each finding: [V] = read in the primary source (or full text) by me;
[S] = taken from a search-engine summary/snippet attributed to the source, NOT read in the source itself.

## 1. Ecopath P/B and Q/B values used for porpoises / toothed whales / seals

### Takeaway
Ecopath models of the North Sea and Baltic use very low mammal P/B (0.02/yr for toothed whales incl. porpoise;
0.06-0.10/yr for seals). These are equilibrium (P/B ~ Z) values or "half of r_max" / "arbitrary" conventions,
not maximum production capacities, so they are a sanity floor for REALISED P/B, not a ceiling for g*(s-u)_max.

### Cited Findings
- North Sea (Mackinson & Daskalov 2007), Table 3.1 inputs: Toothed whales (harbour porpoise, white-beaked and
  Atlantic white-sided dolphin) B = 0.017 t/km2, **P/B = 0.02/yr**, **Q/B = 17.63/yr**; Seals (grey + harbour)
  B = 0.008 t/km2, **P/B = 0.09/yr**, **Q/B = 26.842/yr**; baleen whales P/B 0.02, Q/B 9.9 [V] —
  [Mackinson, S. & Daskalov, G. 2007. An ecosystem model of the North Sea to support an ecosystem approach to fisheries management: description and parameterisation. Cefas Sci. Ser. Tech. Rep. 142, 196 pp.](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Same report, section 15.2: toothed-whale P/B and Q/B "were estimated for each species following Trites et al.
  (1999) and then weighted means were estimated for the whole group: P/B = 0.02 Q/B = 17.63" [V] — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- **Internal inconsistency in that source:** section 15.3 (Seals) says "The maximum rate of population growth rate
  for pinnipeds is believed to be about 12% yr-1 (Small and DeMaster, 1995). The P/B ratio was therefore set at 6%,
  half of the maximum as used by Trites et al. (1999)", and gives an aggregated seal Q/B = 27.87 (grey seal 26.84
  from ICES 2002; harbour seal 30 from the Trites et al. 1999 formula); the parameter table says P/B 0.09 and
  Q/B 26.842 [V]. Report both 0.06 (text) and 0.09 (table) — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Same report: harbour seal average individual weight used = 63 kg (from Trites et al. 1999); North Sea harbour
  seal abundance 24,000; grey seal biomass 3,000 t; total seal biomass 4,400 t [V] — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Same report: seabirds P/B 0.28 in Table 3.1 but text says "Production rate P/B = 0.4 was taken from Trites et al. (1999)" (another table/text mismatch, for context) [V] — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Baltic Sea (Harvey et al. 2003): seals (mainly ringed + grey seal) B = 0.00045 t/km2, **P/B = 0.10/yr**,
  **Q/B = 12.77/yr**, EE 0.89, BA 0.00004 t/km2/yr; "Seal P/B was arbitrarily set at 0.1 yr-1, a value similar to
  those used for pinnipeds in other systems" [V, article page] — [Harvey, C.J., Cox, S.P., Essington, T.E., Hansson, S. & Kitchell, J.F. 2003. An ecosystem model of food web and fisheries interactions in the Baltic Sea. ICES J. Mar. Sci. 60(5): 939-950.](https://academic.oup.com/icesjms/article/60/5/939/608626)
- Pacific Ocean (Trites, Christensen & Pauly 1997), Ecopath model of FAO area 77: "The production/biomass ratio
  (P/B) was assumed to be 0.1 per year, implying an average longevity of about 10 years" for the whole marine
  mammal group (i.e. P/B set from mortality / longevity) [V] — [Trites, A.W., Christensen, V. & Pauly, D. 1997. Competition between fisheries and marine mammals for prey and primary production in the Pacific Ocean. J. Northw. Atl. Fish. Sci. 22: 173-187.](https://journal.nafo.int/Portals/0/1997-2/Trites.pdf)
- Same paper: mammal daily ration R = 0.1 W^0.8 (W in kg) used for Q/B; rations range from ~1.1% body weight/day
  (50,000 kg baleen whale) to ~4.5%/day (50 kg dolphin) [S] — [Trites et al. 1997](https://journal.nafo.int/Portals/0/1997-2/Trites.pdf); R = 0.1 w^0.8 also quoted in [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf) [V]
- Western Baltic EwE model (Scotti et al. 2022): harbour porpoise P/B taken from Araujo & Bundy (2011), Q/B from
  Andreasen et al. (2017); seal P/B from Harvey et al. (2003) and Mackinson & Daskalov (2007); numeric values are
  in Supplementary Table S6 (not extracted) [V, main text] — [Scotti, M., Opitz, S., MacNeil, L., Kreutle, A., Pusch, C. & Froese, R. 2022. Ecosystem-based fisheries management increases catch and carbon sequestration through recovery of exploited stocks: the western Baltic Sea case study. Front. Mar. Sci. 9: 879998.](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.879998/full)
- Trites & Pauly (1998) is a body-mass paper (mean mass from max length: ln M_mean = a + b ln L_max), not a P/B
  source [S] — [Trites, A.W. & Pauly, D. 1998. Estimating mean body masses of marine mammals from maximum body lengths. Can. J. Zool. 76(5): 886-896.](https://cdnsciencepub.com/doi/10.1139/z97-252)

Per-tick equivalents (per-year/1460): P/B 0.02 -> 1.37e-5; 0.06 -> 4.11e-5; 0.09 -> 6.16e-5; 0.10 -> 6.85e-5.
Q/B 12.77 -> 8.75e-3; 17.63 -> 1.21e-2; 26.84 -> 1.84e-2; 27.87 -> 1.91e-2.

### Inferences
- The Ecopath mammal P/B values are not independent measurements: they are (a) set equal to 1/longevity (Trites et
  al. 1997: 0.1), (b) set to half of r_max (Mackinson & Daskalov 2007 text, citing Trites et al. 1999), or (c) set
  "arbitrarily" (Harvey et al. 2003). Half of r_max was chosen to approximate the productivity at MNPL (~K/2) of
  a logistic population, i.e. a NET surplus-production rate, which is neither the gross P/B (r + Z) nor the
  steady-state P/B (= Z).
- The toothed-whale P/B of 0.02/yr is below the porpoise's own natural mortality (any adult survival estimate
  < 0.98 implies Z > 0.02), so it cannot be a gross production rate for a porpoise population that persists;
  it should not be used to set g.
- For the model, these values bracket realised steady-state P/B in a population near carrying capacity
  (P/B ~ Z ~ 0.06-0.10), which is what the model's growth term should average when the population is near K
  (production balancing the 0.088/yr residual mortality).

### Gaps
- Christensen (1995) North Sea Ecopath mammal P/B/Q/B: not found online.
- Tomczak et al. (2012) Baltic Proper Ecopath seal P/B: not found (paper located: Ecol. Model. 230: 123-147, but
  the seal parameters were not accessible).
- Trites et al. (1999) (the source for the "half of r_max" convention and toothed-whale P/B per species) was not
  read directly; the convention is known only as described in Mackinson & Daskalov (2007).
- No Kattegat/Skagerrak-specific Ecopath mammal parameters found; Scotti et al. (2022) Table S6 (western Baltic)
  and Araujo & Bundy (2011) porpoise P/B values were not extracted.

## 2. Maximum population growth rate (r_max / lambda_max)

### Takeaway
Defensible lambda_max: harbour seal 1.12-1.13 (observed in Kattegat-Skagerrak after the 1988 epizootic, HELCOM
uses 12%); grey seal 1.10 (HELCOM ceiling; Baltic observed 7.5%/yr 1990-2003, ~5%/yr 2003-2021); harbour
porpoise 1.04 (IWC/Wade/HELCOM default) to ~1.10 (life-history estimates 9.4-10%, observed 9.6%/yr in a
recovering Californian stock).

### Cited Findings
Harbour seal
- HELCOM core indicator: "The intrinsic rate of increase in this species is 12% per year (Härkönen et al. 2002)";
  Baltic Proper and Kattegat harbour seals "increasing around 12% per year" (2013); GES assessed against
  "Harbour seals: 12 % intrinsic growth rate"; "Harbour seals mature about one year earlier than grey seals and
  ringed seals, which is why maximum rate of increase in this species is 12-13% per year"; "Harbour seal
  populations outside the Baltic increased by about 12% per year between epizootics in 1988 and 2002" [V] —
  [Härkönen, T., Galatius, A., Bräger, S., Karlsson, O. & Ahola, M. 2013. HELCOM core indicator: Population growth rate, abundance and distribution of marine mammals.](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)
- Kattegat-Skagerrak after the 1988 PDV epizootic: "exponential growth of between 10.2% and 13.6% annually";
  rates > 13% likely a transient effect of an age structure skewed to young females; after 2002 growth was lower,
  5% (Kattegat) and 6.5% (Skagerrak); maximum intrinsic rate "slightly less than 13%"; adult survival "often
  > 95% in harbour seals"; carrying capacity estimate K = 13,965 +/- 772 counted seals; 2023 count 12,507;
  population now declining (-408 ind/yr 2003-2023, SE 242) [V] — [Carroll, D., Ahola, M.P., Carlsson, A.M., Galatius, A., Nilssen, K.T., Härkönen, T. & Harding, K.C. 2025. Declining harbour seal abundance in a previously recovering meta-population. PLoS ONE 20(6): e0326933.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208499/)
- A search summary also gave 1988-2002 growth of 15.2%/yr in the Skagerrak and 9.6%/yr in the Kattegat [S; originating source not identified - candidates in the same results were the HELCOM 2015 seal indicator report and Olsen et al. 2010; treat as unverified] — [HELCOM 2015 core indicator report, Population trends and abundance of seals](https://helcom.fi/wp-content/uploads/2023/03/Population-trends-and-abundance-of-seals_HELCOM-core-indicator-report-2015_web-version.pdf)
- Underlying source for 12-13%: Härkönen, T., Harding, K.C. & Heide-Jørgensen, M.-P. 2002. Rates of increase in
  age-structured populations: a lesson from the European harbour seals. Can. J. Zool. 80: 1498-1510 [S, citation
  only, not read] — cited in [HELCOM 2013](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)

Grey seal
- HELCOM 2013: Baltic grey seals grew ">10 % per year" from early 1990s to mid-2000s, then ~6%; "growth rates
  exceeding 10% (lambda = 1.10) per year are unlikely in healthy grey seal populations"; "Reported values exceeding
  10% should be treated sceptically since they imply unrealistic fecundity and longevity rates"; "population
  growth rate of grey seals can only reach 10% if fertility rates are high (0.95)"; GES uses "Grey seals and
  ringed seals: 10 % intrinsic growth rate" [V] — [HELCOM 2013](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)
- Baltic grey seal: "annual rate of increase of 7.5% since 1990" (lambda = 1.0747, r = 0.072); 2003 population
  >= 19,400 [V] — [Harding, K.C., Härkönen, T., Helander, B. & Karlsson, O. 2007. Status of Baltic grey seals: population assessment and extinction risk. NAMMCO Sci. Publ. 6 (pp. 37-39 cited).](https://septentrio.uit.no/index.php/NAMMCOSP/article/view/2720)
- HELCOM current indicator: Baltic grey seal growth 5.1%/yr 2003-2021 (5.2%/yr 2008-2021); GES threshold
  "3% below the maximum rate of increase", i.e. 7%/yr for grey seals; ~57,000 seals in 2025, ~4.9%/yr over the
  last two decades [S] — [HELCOM indicators: Grey seal abundance](https://indicators.helcom.fi/indicator/grey-seal-abundance/);
  [HELCOM Red List 2024 Halichoerus grypus](https://helcom.fi/wp-content/uploads/2025/11/HELCOM-Red-List-II-Halichoerus-grypus.pdf)
- Note: the current HELCOM threshold (max - 3% = 7%) implies a maximum of 10% for grey seals, consistent with 2013 [S/V].

Harbour porpoise
- HELCOM 2013: "Annual maximum rate of increase for most whales, also harbour porpoise, is about 4% (Woodley and
  Read 1991, Best 1992)"; GES uses "Harbour porpoise: 4 % intrinsic growth rate"; Kattegat, Belt Sea and Baltic
  Proper subpopulation growth rates "are negative" [V] — [HELCOM 2013](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)
- Lockyer (2003) review: population growth rate "probably 9.4%" (range 5-10%) [S, via a library abstract page] —
  [Lockyer, C. 2003. Harbour porpoises (Phocoena phocoena) in the North Atlantic: biological parameters. NAMMCO Sci. Publ. 5: 71-89. doi:10.7557/3.2740](https://porpoise.org/library/harbour-porpoises-phocoena-phocoena-north-atlantic-biological-parameters/)
- Caswell et al. (1998), Monte Carlo life-table analysis (Gulf of Maine): median potential lambda ~1.10, 90% CI
  3-15% [S] — [Caswell, H., Brault, S., Read, A.J. & Smith, T.D. 1998. Harbor porpoise and fisheries: an uncertainty analysis of incidental mortality. Ecol. Appl. 8: 1226-1238.](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/1051-0761(1998)008[1226:HPAFAU]2.0.CO;2)
- Moore & Read (2008), Bayesian analysis of age-at-death data: porpoise growth rate up to 11.6% (upper 90%
  probability interval) [S] — [Moore, J.E. & Read, A.J. 2008. A Bayesian uncertainty analysis of cetacean demography and bycatch mortality using age-at-death data. Ecol. Appl. 18(8).](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/07-0862.1)
- Observed recovery after gillnet bycatch ended (California): Morro Bay, Monterey Bay and San Francisco-Russian
  River stocks grew at 9.6%, 5.8% and 6.1% per year; Morro Bay's 9.6%/yr "is very similar to the maximum possible
  growth rate of 9.4% estimated by Barlow & Hanan (1995) based on plausible life history" [V] —
  [Forney, K.A., Moore, J.E., Barlow, J., Carretta, J.V. & Benson, S.R. 2021. A multidecadal Bayesian trend analysis of harbor porpoise populations off California relative to past fishery bycatch. Mar. Mamm. Sci. doi:10.1111/mms.12764](https://repository.library.noaa.gov/view/noaa/32061/noaa_32061_DS1.pdf)

Defaults
- Wade (1998) PBR defaults: R_max = 0.04 for cetaceans and 0.12 for pinnipeds (lambda_max 1.04 / 1.12) [S] —
  [Wade, P.R. 1998. Calculating limits to the allowable human-caused mortality of cetaceans and pinnipeds. Mar. Mamm. Sci. 14: 1-37.](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1748-7692.1998.tb00688.x);
  [Punt et al. 2020, ICES J. Mar. Sci. 77: 2491](https://academic.oup.com/icesjms/article/77/7-8/2491/5903506)
- Pinniped r_max ~12%/yr attributed to Small & DeMaster (1995) [V, as cited] — [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

Per-tick equivalents of r (per-year/1460): 0.04 -> 2.74e-5; 0.051 -> 3.49e-5; 0.072 -> 4.93e-5;
0.095 -> 6.51e-5; 0.10 -> 6.85e-5; 0.113 -> 7.74e-5; 0.12 -> 8.22e-5; 0.13 -> 8.90e-5.

### Inferences
- Recommended r_max for the calibration: harbour seal 0.12 (lambda) / 0.113 (r); grey seal 0.10 / 0.095;
  porpoise 0.04 (conservative, management default) to 0.10 (life-history and empirical maximum). For porpoise
  the 0.04 default is a deliberately conservative management value (Wade 1998 picked it as a lower bound for
  cetaceans), whereas porpoise-specific evidence (Lockyer 9.4%, Caswell median 10%, Forney 9.6% observed) points
  to ~0.09-0.10; porpoises mature early and calve almost annually, unlike large whales.
- Observed recent local rates are well below r_max (harbour seal Kattegat-Skagerrak now ~0 or negative; Baltic
  grey seal ~5%; porpoise in Kattegat/Belt Sea negative), so a model that hits r_max in the study area would be
  optimistic.

### Gaps
- Härkönen et al. (2002) Can. J. Zool. not read directly (its vital rates behind 12-13% unverified by me).
- Wade (1998), Caswell et al. (1998), Moore & Read (2008) values are from search summaries (abstracts), not the
  full text.
- No published r_max specific to the Belt Sea / Kattegat porpoise population found (a 2024 paper on a negative
  Belt Sea trend exists - researchgate 378847678 - but was not read).

## 3. Demography (age at first reproduction, pregnancy, survival, body masses)

### Takeaway
All three species have one offspring per year at most, first birth at ~4-5 yr (harbour seal, porpoise) or
~5.5 yr (grey seal), pregnancy 0.75-0.95 in healthy populations, adult survival 0.87-0.96 and lower juvenile
survival. Offspring at birth are 8-16% of adult female mass (porpoise ~5 kg vs 55-65 kg; harbour seal 9-11 kg vs
~67 kg; grey seal ~15 kg vs 100-190 kg), so most population production is juvenile somatic growth, not birth mass.

### Cited Findings
Harbour porpoise
- Age at sexual maturity 3-4 yr (both sexes); first parturition "probably 4-5 years"; pregnancy rates "generally
  in the range 0.74-0.986 per year"; calving interval 1.01-1.57 yr; gestation 10-11 months; lactation "probably at
  least 8 months"; birth length 65-75 cm; adult female 153-163 cm, 55-65 kg; adult male 141-149 cm, 46-51 kg;
  maximum longevity 24 yr; survival rate 0.867; growth rate "probably 9.4%" [S] — [Lockyer 2003, NAMMCO Sci. Publ. 5](https://porpoise.org/library/harbour-porpoises-phocoena-phocoena-north-atlantic-biological-parameters/)
- Danish waters: birth 65-75 cm, 4.5-6.7 kg; maturity slightly over 3 yr (~135 cm males, 143 cm females);
  longevity up to 23 yr; peak births June; fewer than 5% live beyond 12 yr; bycaught and stranded animals mostly
  juveniles [V, abstract page] — [Lockyer, C. & Kinze, C. 2003. Status, ecology and life history of harbour porpoise (Phocoena phocoena) in Danish waters. NAMMCO Sci. Publ. 5: 143-175.](https://septentrio.uit.no/index.php/NAMMCOSP/article/view/2745)
- NAMMCO species page: birth mass "6 to 10 kg" (Gaskin 1992, Lockyer 2003); adult 46-65 kg; lactation 8-9 months;
  "Most females produce a calf each year"; maturity ~4-5 yr in the eastern North Atlantic [V] — [NAMMCO: Harbour porpoise](https://nammco.no/harbour-porpoise/)
  (note: conflicts with Lockyer & Kinze's 4.5-6.7 kg birth mass for Danish waters; the latter is regional and preferred)
- German North Sea and Baltic: female maturity (>= 50% with corpora) at 4.95 +/- 0.6 yr; mean age at death 5.70
  +/- 0.27 yr (North Sea) and 3.67 +/- 0.30 yr (Baltic); ~30% of dissected animals had suspected bycatch
  pathology [V] — [Kesselring, T., Viquerat, S., Brehm, R. & Siebert, U. 2017. Coming of age: do female harbour porpoises (Phocoena phocoena) from the North Sea and Baltic Sea have sufficient time to reproduce in a human influenced environment? PLoS ONE 12(10): e0186951.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5650184/)
- Calves grew 66% (in length) during their first year, reaching 84% of adult length [S, snippet; full text 403] —
  [Determination of growth, mass, and body mass index of harbour porpoises, Global Ecol. Conserv. 2023](https://www.sciencedirect.com/science/article/pii/S2351989423000197)

Harbour seal
- Kattegat-Skagerrak / East Atlantic (Härkönen & Heide-Jørgensen 1990): adult survival 0.91; max age 36 yr
  (females), 31 (males); females mature at ~4-5 yr, males 6-7; mean annual pregnancy rate 92% from maturity to
  age 36; Kattegat-Skagerrak adult males ~75 kg, females ~67 kg; mean birth mass 8.7 kg in Kattegat-Skagerrak;
  at 1 yr males ~30 kg [S — values appeared in a search summary attributed to Härkönen, T. & Heide-Jørgensen, M.-P. 1990. Comparative life histories of East Atlantic and other harbour seal populations. Ophelia 32: 211-235; paper not read and the page carrying these numbers was not identified; the HELCOM Red List sheet linked here confirms ONLY max age 36 and maturity 3-6 yr, not these numbers] — [HELCOM Red List: Phoca vitulina vitulina](https://helcom.fi/wp-content/uploads/2019/08/HELCOM-Red-List-Phoca-vitulina-vitulina.pdf)
- HELCOM Red List sheet confirms maximum age 36 yr (Härkönen & Heide-Jørgensen 1990) and female maturity
  "between 3 and 6 years" [V] — [HELCOM Red List: Phoca vitulina vitulina](https://helcom.fi/wp-content/uploads/2019/08/HELCOM-Red-List-Phoca-vitulina-vitulina.pdf)
- Mean pregnancy rates "rarely reach 0.96" in samples of reasonable size (Boulva & McLaren 1979; Bigg 1969;
  Härkönen & Heide-Jørgensen 1990) [V] — [HELCOM 2013](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)
- Pup birth mass 11.2 +/- 0.31 kg; weaned at 24.1 +/- 0.44 days at 24.9 +/- 0.45 kg (NW Atlantic, Sable Island
  harbour seals) [S] — [Bowen, W.D., Ellis, S.L., Iverson, S.J. & Boness, D.J. 2001. Maternal effects on offspring growth rate and weaning mass in harbour seals. Can. J. Zool. 79 (doi:10.1139/z01-075; pages not verified).](https://cdnsciencepub.com/doi/10.1139/z01-075)
- Adult survival in most seal species tops out at 0.95-0.96; pup and subadult survival "always found to be lower
  and more variable" than adult survival [V] — [HELCOM 2013](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)

Grey seal
- Mean age at first birth ~5.5 yr in four independent Atlantic studies (Wiig 1991: 5.35 +/- 0.69; Hammill &
  Gosselin 1995: 5.5 +/- 0.12, annual range 5.03-6.08; Schwartz & Stobo 2000: 5.2; Boyd 1985: ~5.5) [V] — [Harding et al. 2007](https://septentrio.uit.no/index.php/NAMMCOSP/article/view/2720)
- Birth rates of mature females (age 6+): NW Atlantic 0.85-0.88; UK 0.80-0.93; Norway 0.91 (mean fertility 0.907
  weighted); Baltic 0.75 (Bergman 1999); Baltic pregnancy only 20-30% in the PCB era (1970s), ~60% in mid-1990s [V] —
  [Harding et al. 2007, Tables 3-4](https://septentrio.uit.no/index.php/NAMMCOSP/article/view/2720)
- Survival: adults (>4 yr) 0.935 UK (Harwood & Prime 1978); <0.96 Norway (Wiig 1991, likely biased up); 0.88-0.92
  ages 4-9 Canada (Schwartz & Stobo 2000); 0.87 NWA; juvenile 0-4 yr 0.493/yr UK (Harwood & Prime 1978), <0.83
  NWA; first-year survival 0.62-0.90 UK, 0.70-0.76 Baltic; adult survival of many seals 0.87-0.96; Baltic females
  live to 40+ yr [V] — [Harding et al. 2007, Table 5](https://septentrio.uit.no/index.php/NAMMCOSP/article/view/2720)
- Grey seal females first pup at ~5.5 yr, at most one pup per year [V] — [HELCOM 2013](https://helcom.fi/wp-content/uploads/2023/04/HELCOM-CoreIndicator-Population_growth_rate_abundance_and_distribution_of_marine_mammals.pdf)
- Birth mass 14.8-15.8 kg (range 11-20 kg), weaned at 15-18 days, roughly quadrupling birth mass; eastern
  Atlantic adult females 1.6-1.95 m and 100-190 kg [S, encyclopaedic] — [Wikipedia: Grey seal](https://en.wikipedia.org/wiki/Grey_seal)
- Mean population weaning mass 51.5 kg (Sable Island, NW Atlantic); female survival to recruitment rises with
  weaning mass up to that value [S] — [Bowen, W.D. et al. 2015. Offspring size at weaning affects survival to recruitment and reproductive performance of primiparous gray seals. Ecol. Evol. 2015 (PMC4395171; pages not verified).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4395171/)

### Inferences
- Mortality estimates from stranded/bycaught porpoises (Lockyer's 0.867 survival; Kesselring's mean age at death
  5.7 / 3.67 yr) include anthropogenic mortality (bycatch), so they are total Z (~0.14+/yr), not the "residual
  natural mortality" the model uses (0.088/yr). They should not be compared directly with the model's m.
- Porpoise mass at 1 yr: no direct value found. From "84% of adult length" and mass ~ length^3,
  0.84^3 ~ 0.59 of adult mass, i.e. ~30 kg for a 50-55 kg adult (inference, not a measurement).
- Offspring mass at birth relative to the adult female: porpoise 5/60 ~ 8%; harbour seal 9-11/67 ~ 13-16%;
  grey seal 15/150 ~ 10%. With 0.5-0.9 offspring per adult female and females ~half the population, birth mass
  alone is only ~0.02-0.05/yr of population biomass (see section 4).
- Births are strongly pulsed: harbour seal June, grey seal Baltic Feb-Mar / North Sea Nov-Dec, porpoise
  May-August (peak June in Danish waters). The model spreads production uniformly over 1460 ticks.

### Gaps
- Kesselring et al. (2017) pregnancy rate: a search snippet claimed 0.571 for the German North Sea, but my fetch
  of the full text reported that the paper gives no separate pregnancy rate. Contradicted; treat as unverified.
- Juvenile/pup survival for Kattegat-Skagerrak harbour seals (Härkönen et al. 2002; Silva et al. 2021 Ecosphere,
  doi:10.1002/ecs2.3343) not obtained (403).
- Harbour seal pup masses for the study area (only 8.7 kg birth mass, snippet-level) - weaning mass in
  Kattegat-Skagerrak not found; the 24.9 kg value is from Sable Island.
- No natural (non-bycatch) survival estimate for Belt Sea / Kattegat porpoises found.

## 4. Maximum gross P/B from first principles, and the implied g per tick

### Takeaway
Gross P/B at maximum growth = r_max + m. With the model's m = 0.088/yr: harbour seal ~0.21/yr, grey seal ~0.19/yr,
porpoise 0.13-0.19/yr. An independent age-structured calculation (juvenile somatic growth + newborn mass) gives
0.14-0.21/yr for all three species. Hence g ~ 4.3-4.8e-4/tick for seals at (s-u)_max = 0.3 and
~1.8-2.6e-4/tick for porpoises at (s-u)_max = 0.5; the current g = 1e-3 is ~2.1-2.3x too high for seals and
~3.9-5.7x too high for porpoises.

### Cited Findings
- Ecopath convention P/B = Z at steady state, e.g. "P/B ... 0.1 per year, implying an average longevity of about
  10 years" [V] — [Trites et al. 1997](https://journal.nafo.int/Portals/0/1997-2/Trites.pdf)

### Inferences
Inputs (cited in sections 2-3): r_max harbour seal 0.12, grey seal 0.10, porpoise 0.04-0.10; masses, ages at
first birth, pregnancy and survival as in section 3.

Calibration rule (model-consistent). The model has dB/B = g*(s-u) - m per tick (plus predation = 0 for mammals).
At best conditions (s-u at its max) the population should grow at r_max, so
  g * (s-u)_max * 1460 = r_max + m_model,   m_model = 0.088/yr.
If m_model is changed, g must move with it (g scales with r_max + m).

| Group | r_max (lambda-1) | r_max + m (gross P/B ceiling, /yr) | per tick | (s-u)_max | g (per tick) | current g=1e-3 ceiling (/yr) | ratio current/recommended |
|---|---|---|---|---|---|---|---|
| Harbour seal | 0.12 (0.13 upper) | 0.208 (0.218) | 1.42e-4 (1.49e-4) | 0.3 | **4.75e-4** (4.98e-4) | 0.438 | 2.1x |
| Grey seal | 0.10 | 0.188 | 1.29e-4 | 0.3 | **4.29e-4** | 0.438 | 2.3x |
| Porpoise, conservative | 0.04 | 0.128 | 8.77e-5 | 0.5 | **1.75e-4** | 0.730 | 5.7x |
| Porpoise, life-history max | 0.10 | 0.188 | 1.29e-4 | 0.5 | **2.58e-4** | 0.730 | 3.9x |

(Using instantaneous r = ln(lambda) instead: harbour seal 0.113+0.088 = 0.201 -> g 4.6e-4; grey seal
0.095+0.088 = 0.183 -> 4.2e-4; porpoise 0.039/0.095 + 0.088 = 0.127/0.183 -> 1.7e-4/2.5e-4. Using
m = -ln(0.92) = 0.083 instead of 0.088 lowers all by ~0.005/yr.) The single "seals" FG (harbour + grey) would
reasonably take ~4.5e-4.

Independent cross-check: female-based Leslie models (post-breeding census, both sexes given the same schedule,
annual time step), production = newborn mass + somatic growth of survivors (+ half the increment of those that
die), divided by mean annual biomass of the stable age distribution. Mass-at-age schedules (kg) were assembled
from section 3 values and are partly assumed: harbour seal 9, 32, 40, 48, 55, 60, 64, 67, 69, 70...; grey seal
15, 50, 65, 80, 95, 110, 120, 130, 140, 150...; porpoise 5, 28, 38, 45, 50, 54, 57.... Survivals/fecundities are
illustrative choices within the cited ranges:

| Scenario (s0 / s_juv / s_adult; births per mature female, age) | lambda | gross P/B (/yr) | of which newborn | of which somatic | implied biomass-weighted Z |
|---|---|---|---|---|---|
| Harbour seal tuned to lit. max (0.72/0.88/0.95; 0.92 @4+) | 1.113 | 0.201 | 0.045 | 0.156 | 0.094 |
| Harbour seal typical (0.70/0.85/0.91; 0.5 @4, 0.92 @5+) | 1.063 | 0.193 | 0.041 | 0.152 | 0.132 |
| Harbour seal near zero growth (0.60/0.80/0.91; 0.85 @5+) | 1.019 | 0.170 | 0.038 | 0.132 | 0.151 |
| Grey seal tuned to lit. max (0.75/0.90/0.95; 0.5 @5, 0.9 @6+) | 1.100 | 0.171 | 0.032 | 0.139 | 0.077 |
| Grey seal HELCOM-like (0.70/0.86/0.935; 0.5 @5, 0.9 @6+) | 1.069 | 0.168 | 0.033 | 0.136 | 0.101 |
| Porpoise ~10% (0.80/0.88/0.93; 0.95 @4+) | 1.113 | 0.212 | 0.029 | 0.183 | 0.105 |
| Porpoise Lockyer-like (0.80/0.867; 0.4 @4, 0.85 @5+) | 1.042 | 0.191 | 0.023 | 0.167 | 0.150 |
| Porpoise Kesselring-like (0.75/0.85; 0.57 @5+) | 0.968 | 0.142 | 0.016 | 0.126 | 0.175 |
| (Illustrative upper bound, EXCEEDS literature lambda_max) harbour seal 0.85/0.92/0.96 | 1.161 | 0.213 | 0.045 | 0.168 | 0.064 |
| (Illustrative upper bound, EXCEEDS literature lambda_max) grey seal 0.85/0.93/0.96 | 1.143 | 0.189 | 0.035 | 0.154 | 0.056 |
| (Illustrative upper bound, EXCEEDS literature lambda_max) porpoise 0.85/0.90/0.95 | 1.143 | 0.214 | 0.029 | 0.185 | 0.080 |

Conclusions from the cross-check:
- Gross P/B is remarkably insensitive to lambda: 0.14-0.21/yr across all scenarios, because it is dominated
  (75-85%) by juvenile somatic growth; newborn mass is only 0.02-0.045/yr. This independently corroborates the
  r_max + m ceiling of ~0.19-0.21/yr (seals) and supports the upper porpoise value (~0.19-0.21) over the
  conservative 0.13.
- The Leslie runs give biomass-weighted Z of 0.08-0.15/yr in growing populations, i.e. somewhat above the model's
  0.088 (real populations lose many light-weight pups, which costs little biomass). The model's single m = 0.088
  is therefore a reasonable biomass-weighted residual mortality for seals, and on the low side for porpoises.
- Ecopath P/B (0.02-0.10) sit far below these ceilings because they describe a stationary population (P/B = Z)
  or half of r_max; they are a sanity floor for the realised long-run P/B near K, not a value for g*(s-u)_max.
- Caveat (milk-funded growth): much first-year growth (harbour seal ~9-11 -> ~25 kg in ~24 days; grey seal
  ~15 -> ~50 kg in 15-18 days) is a within-pool transfer from the mother's reserves to the pup. The Leslie
  "gross P" counts the pup's gain as tissue production while the mother's loss is a metabolic cost; this matches
  how the model separates growth (g*(s-u)) from maintenance (u), but the realised net biomass gain from
  lactation is smaller than pup mass gain.
- Caveat (seasonality): the model spreads this production evenly over the year; real births are pulsed (section 3).
- Recommended values (per-year gross P/B ceiling -> g/tick): seals (single pool) ~0.20/yr -> g ~4.5e-4 at
  (s-u)_max 0.3 (range 4.3-5.0e-4); porpoises 0.13-0.19/yr -> g ~1.75-2.6e-4 at (s-u)_max 0.5, with ~2.5e-4
  defensible from porpoise-specific evidence and ~1.75e-4 if the conservative Wade/HELCOM 4% default is preferred.

### Gaps
- The mass-at-age schedules are assembled from scattered values (some snippet-level, some from NW Atlantic
  populations) and interpolated; no published Kattegat-Skagerrak mass-at-age curve was read.
- No published direct estimate of gross P/B (r + Z or production per unit biomass) for any of the three species
  was found; the values above are derived.
- Sex differences in mass (male grey seals ~2x female mass; male harbour seals slightly heavier; porpoise
  females heavier) were not modelled; they would change P/B by a few hundredths at most.
