# Benthic prey of gadoids: production, P/B, logistic r and food limitation (Skagerrak / Norwegian Trench slope)

Scope: the model's single logistic `benthic_community` (r 0.82 /yr, K 10 t WW/km2, start 4.44 t WW/km2;
1 t/km2 = 1 g/m2) standing for the benthic prey of cod, haddock, whiting, saithe and Norway pout in a
60x60 km box, 43-533 m deep (mean 168 m), muddy, bottom water ~6-7 C.

Tags: [F] read in full text, [A] abstract only, [S] secondary (source named). "Inference" and
"computed by me" mark my own arithmetic; nothing there is a literature number.

## Short answers

**A. Production, P/B and r of the prey fraction.**
- The "known" Brey-model P/B ~2 /yr for *Amphiura filiformis* is about 4x the only measured
  Skagerrak value: Skold, Loo & Rosenberg (1994), 40 m off the Gullmarsfjord mouth, measured
  **P/B 0.46 /yr** for the whole population and **0.37 /yr** for the dominant adult size group [F].
  They list Galway Bay's 1.97 /yr as a value for the large-adult group, and put the lifespan at >= 20 yr.
- The part of brittle-star production that cropping fish actually take is arm regeneration:
  **0.34 g AFDW m-2 yr-1 = 13.3 % of production** [F]. Biomass is 2.59/0.46 = 5.6 g AFDW m-2, so the
  regeneration flux is **~0.06 /yr** of brittle-star biomass (derived). That is ~3.9 g WW m-2 yr-1 at
  Skold's own WW:AFDW ratio of 11.5, which is already ~2x the model's maximum surplus production
  (rK/4 = 2.05 t WW km-2 yr-1). Independent check: dab in the southern North Sea eat **6 % of the arm
  population per year** (Duineveld & van Noort 1986) [A].
- No measured P/B was found for deep-Skagerrak polychaetes or small crustaceans. The Brey (1999)
  empirical model at 6-7 C and 200 m (computed by me) gives about **1.3-1.8 /yr** for small polychaetes
  and crustaceans (~0.03 kJ per individual), **0.8-1.1** at 0.3 kJ, and **0.55-0.6** for 0.1 kJ
  echinoderms. North Norway (colder) measured 0.75 for polychaetes, 0.45 crustaceans, 0.29 community
  (Nilsen et al. 2006) [F].
- No P/B was found for Nephrops or Pandalus. Natural mortality: Nephrops M 0.3 (males) / 0.2 (females)
  in the Skagerrak-Kattegat assessment (ICES 1993) [F]. Pandalus 3a4a: M 0.75 for all ages historically
  (ICES 1995) [F], best-fit 1.2 constant or 0.9 time-varying (Jorgensen et al. 2014) [F], and the
  WKPRAWN 2022 median vector 1.71 / 1.02 / 0.70 / 0.58 at ages 0.5 / 1.5 / 3.5 / 7.5 [F]. With the
  steady-state identity P/B = Z = M + F (inference), Nephrops P/B >= 0.2-0.3 and Pandalus P/B >= ~0.6-1.0
  (juveniles ~1.7).
- **Logistic r.** Hiddink et al. (2019) [F] give r = **2.41 /yr (longevity 1-3 yr), 1.24 (3-10 yr),
  0.34 (>10 yr)**, fitted as r = 5.31/longevity (CI fits 2.43-11.44). Taxa living < 1 yr showed no
  long-term response, so they have no r. Measured r/(P/B) is roughly 0.7-2 (see A.4). Which r applies
  depends on what `benthic_community` contains:
  - polychaete / small-crustacean / juvenile-shrimp fraction: **r ~1.2-2.4 /yr**;
  - brittle-star dominated (lifespan >= 20 yr): **r ~0.27-0.5 /yr**;
  - the current 0.82 (Hiddink 2017 community) implies a biomass-weighted lifespan of ~6.5 yr.

**B. Food or predation limitation.**
- Strongest evidence: Josefson (1990) [F], 14 stations at 18-300 m in the Skagerrak-Kattegat over
  1972-1988 (one station at 300 m, 58 14.9'N 10 34.3'E). Total biomass rose by a median factor of 1.8,
  the relative increase grew with depth, and biomass tracked land run-off at a 2-year lag. Josefson tests
  the predation-release alternative and rejects it on three grounds; his results are "in accordance
  with hypotheses that soft-sediment infauna in the area is food limited" (the title ends in a "?"). Josefson et al. (1993) [A] and Josefson & Jensen (1992) [A] support this.
- The Kattegat no-take zone shows the counterpart (Skold et al. 2018, 2025) [A]. Brittle stars fell
  45-48 % as benthivorous flatfish increased: fish predation can depress the dominant prey locally.
  The two are compatible: food sets the production ceiling, and predators decide how much of it
  stands as biomass.
- The supply is large. Norwegian Trough organic-carbon accumulation averages 19.4 g C m-2 yr-1
  (Diesing et al. 2021) [F]. Sediment oxidation measured in situ is 2.2-18 mmol C m-2 d-1
  = 9.6-79 g C m-2 yr-1 (Stahl et al. 2004) [A]. More than 90 % of the buried carbon is imported, not
  local (de Haas & van Weering 1997) [A].
- A food-limited macrofauna production ceiling of ~2-20 g C m-2 yr-1 (inference, chain in B.4) is
  **~20-500 g WW m-2 yr-1**. Measured production fits inside it: brittle stars alone 30 g WW m-2 yr-1
  (Skold 1994) [F]; North Norway community 89 [F].
- **Conclusion (inference):** the food ceiling is 10-250x above the model's own maximum production
  (2.05 t WW km-2 yr-1). It cannot bind at K = 10 t/km2 unless K means only the gadoid-accessible slice
  of the benthos. Total macrofauna stocks are ~45-150 g WW m-2 (see the earlier note
  `Litteraturbaserad growth rate per FG/benthos.md`, Q2).

## A. Evidence: production and P/B

### A.1 Brittle stars (Amphiura filiformis, A. chiajei) and arm regeneration
- **Skold, Loo & Rosenberg 1994**, MEPS 103:81-90, mouth of the Gullmarsfjord (58 14.72'N 11 25.80'E),
  40 m, June 1990 - Nov 1991 [F]. https://www.int-res.com/articles/meps/103/m103p081.pdf
  - Abstract: "Disc growth and gonad production accounted for ca 68.9 % (1.8 g AFDW m-2 yr-1) of the
    total annual production in the population. About 13.3 % (0.34 g AFDW m-2 yr-1) of the total production
    was allocated to regeneration of arms, probably a result of cropping by predators. Mean regenerated
    biomass in percent of total biomass for adult A. filiformis was between 12 and 30 % (mean 22 %).
    Annual production/biomass ratio was 0.46 yr-1."
  - Results: total production "2.59 g AFDW m-2 yr-1 and 29.75 g WW m-2 yr-1"; "For the dominating SG 3 ...
    the P/B ratio was 0.37 yr-1." Abundance ~280 ind. m-2 (> 1 mm disc).
  - Discussion: Galway Bay's 1.97 yr-1 "estimated for the large adult size group ... is higher than for the
    comparable size group (SG 3; 0.37 yr-1)". Lifespan ">= 20 yr" (Galway, Oresund) fits this population.
    "Most 'growth' in an adult brittle star in this study was gonad production and regeneration of arms,
    which together contributed 82 % of the total production". "Losses due to predation can be estimated as
    the regenerated biomass, which was about 22 % of total biomass." Predators that crop arms include dab,
    haddock and Nephrops. "Food availability is most probably a key factor for regeneration rate".
  - Derived (computed by me): B = 2.59/0.46 = 5.6 g AFDW m-2 = 65 g WW m-2 (WW:AFDW = 29.75/2.59 = 11.5);
    regeneration flux 0.34/5.6 = 0.06 /yr = 3.9 g WW m-2 yr-1.
- **Duineveld & van Noort 1986**, Neth. J. Sea Res. 20:85-94, Oyster Ground [A].
  https://marineinfo.org/doc/publication/15896
  - "Only 4% of the individuals in a sample had intact arms, whereas 55% of the animals (or 20% of all arms)
    had one or more 'recent' distal regenerations". "Annual consumption of Amphiura arms by dab was
    estimated to be 0.84 g wet weight m-2, which is equivalent to 420 arms or 6% of the arm population."
    Juvenile survival was low (3-10 % over 8-9 months); "grazing by dab will have only little if any effect
    on the population."
- **Josefson & Jensen 1992**, Mar. Biol. 112:615-624, three sites at 55-100 m, 1970-1989 [A].
  https://marineinfo.org/doc/publication/26698
  - Biomass rose at all three sites. "estimated maximum weight-specific growth rates were more than 1.5% d-1
    in the disc diameter interval 3 to 9 mm". The results are "consistent with the hypothesis that recent
    increases in benthic biomass in this area result from increased input of food to the benthos."
- **Skold, Josefson & Loo 2001**, Mar. Biol. 139:519-526, Gullmarsfjord [A].
  https://marineinfo.org/id/publication/260762
  - Gompertz growth; "adult size and sexual maturity (ca. 4 mm disk diameter) are attained around 2 years
    after settlement."
- Bowmer & Keegan 1983 (Mar. Biol. 74:65-71) and O'Connor et al. 1986 (Galway energy budget): only cited in
  Skold 1994 [S]; no numbers retrieved. No production estimate was found for A. chiajei.

### A.2 Polychaetes, small crustaceans: empirical-model and analogue values
- **Nilsen, Pedersen & Nilssen 2006**, MEPS 321:67-77, Sorfjord, North Norway (69 N), Brey (1999) model [F].
  https://www.int-res.com/articles/meps2006/321/m321p067.pdf
  - "Average biomass, production, abundance and P/B were 307 g wet weight (ww) m-2 ..., 89 g ww m-2 yr-1
    ... and 0.29 yr-1". In energy units the P/B is 0.42 /yr.
  - "Estimated average Sorfjord P/B ratios for the statistical groups Polychaeta, Crustacea, Mollusca and
    Echinodermata were 0.75, 0.45, 0.35 and 0.37 yr-1". Arctica 0.18, Ctenodiscus 0.16.
  - "the macrobenthic invertebrate production was much higher than the amount eaten by fish, and predatory
    benthos are probably more important consumers of benthic invertebrates."
  - Caveats (their own): pooled groups underestimate P/B because the model is logarithmic, and Brey v4-04
    is preferred to v2-00. Bottom water there is colder than in the Skagerrak.
  - Their Table 4 (secondary values): Masfjord, W Norway, P/B 2.6; North Sea, Duineveld et al. 1991, P/B 1.9,
    B 76, P 40 g WW m-2 yr-1; Northumberland 0.44.
- **Brey (1999) model v2-00, as printed in Nilsen et al. 2006** [F], my computation (inference):
  log P/B = 8.256 - 2.226 logM - 2432.055/(T+273) + 0.239/D + 0.241 Dsubt + 0.203 Dinf + 0.242 Dmoti
  - 0.287 Dmoll - 0.203 Dpoly - 0.128 Dcrust - 0.457 Dechi + 562.218 logM/(T+273); M in kJ per individual.
  The equation is copied from the PDF text layer; check it against Brey's spreadsheet before relying on it.
  - At T = 6-7 C, D = 200 m, subtidal (1 kJ ~ 43 mg AFDW, assuming ~23 J/mg AFDW):
    - infaunal polychaete: 0.01 kJ 1.6-1.8; 0.03 kJ 1.3-1.4; 0.3 kJ 0.8; 3 kJ 0.5 /yr;
    - motile epifaunal crustacean: 0.03 kJ 1.65-1.8; 0.3 kJ 1.0-1.1 /yr;
    - infaunal echinoderm: 0.1 kJ 0.55-0.6 /yr (against 0.46 measured by Skold 1994).
- **Mackinson & Daskalov 2007** North Sea Ecopath [S via `Litteraturbaserad growth rate per FG/benthos.md`,
  which read it in full]: small infauna (polychaetes) 0.9; small mobile epifauna 1.4-1.9; shrimp 3.0;
  Crangon (Brey model) 0.81-1.07; Nephrops 0.35-0.4. Ecotrophic efficiency (EE) is 0.28 for infaunal
  macrobenthos and 0.91 for small infauna, so predators use only part of the infaunal production.
  https://www.cefas.co.uk/publications/techrep/tech142.pdf
- No measured P/B was found for amphipods, cumaceans or Pandalus juveniles in Skagerrak / North Sea waters.

### A.3 Nephrops and Pandalus (ICES assessments)
- **Nephrops, Skagerrak-Kattegat (FU 3-4)**: "Natural mortality was set to 0.3 for males and 0.2 for
  females." Growth inputs were borrowed from Division IVa (ICES C.M. 1993/Assess:11) [F].
  https://www.vliz.be/imisdocs/publications/ocrd/283709.pdf
- **Nephrops growth, Skagerrak** (Ulmestrand & Eggert 2001, ICES JMS 58:1326-1334) [A, read on the scanned
  PDF page 1]: "The estimates of L-inf (72.9) and K (0.138) for males differ from those currently used in the
  ICES assessment for this area (76 and 0.16, respectively)". https://www.vliz.be/imisdocs/publications/65649.pdf
  Female values are in the body of the scanned PDF and were not read.
- **Pandalus borealis, Skagerrak and Norwegian Deep (pra.27.3a4a)**:
  - ICES 1995: "M has been set at 0.75 for all ages"; "The value used in earlier assessments i.e. 0,75 for
    Div.IIIa, IVaE is not well founded." [F] https://www.vliz.be/imisdocs/publications/ocrd/275381.pdf
  - Jorgensen et al. 2014, NAFO SCR 14/066 [F]: "With the assumption of a constant M across the time series,
    the best statistical model-fit ... is observed for M = 1.2"; with a predation-index M, "M = 0.9".
    Their Table 3: total M 0.07-3.34 (mean 1.44) in the W Gulf of Alaska (Anderson 1991); M from cod
    predation 0.06-2.23 (mean 0.76) in the Barents Sea (Berenboim et al.). Predators in the index: cod,
    roundnose grenadier, velvet belly, blue whiting. https://www.nafo.int/Portals/0/PDFs/sc/2014/scr14-066.pdf
  - WKPRAWN 2022, Table 3.4.4 [F]: M at ages 0.5 / 1.5 / 3.5 / 7.5 is "Median 1.709 1.019 0.699 0.577;
    Lower 1.356 0.809 0.554 0.458; Upper 1.964 1.171 0.803 0.664". Composite life-history M: median 0.61
    (CI 5 % 0.48, 95 % 0.78).
    https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_Pandalus_stocks_WKPRAWN_/19714204
  - No Pandalus P/B was found. With P/B = Z = M + F (Allen's steady-state identity; inference), P/B >= M:
    ~0.6-1.0 for adults and ~1.7 for age 0.5.

### A.4 From P/B to a logistic r
- **Hiddink et al. 2019**, J. Appl. Ecol. 56:1075-1084, author manuscript [F].
  https://repository.library.noaa.gov/view/noaa/57053/noaa_57053_DS1.pdf
  - "Estimates of r ... ranged from 2.41 yr-1 for biota with a T max = 1-3yr, through 1.24 yr-1 for biota
    with a T max = 3-10yr, to r = 0.34 yr-1 for biota with a T max >10yr."
  - Recovery from 0.5K to 0.95K takes "1.2 (0.55 - 2.78 uncertainty interval), ... 2.4 (1.37 - 4.16) and
    8.7 (3.51 - 22.50) years respectively". Fig. 3: "r = 5.31 / longevity, R2 = 0.96"; confidence fits
    "upper: r = 11.44 / longevity, lower: r = 2.43 / longevity".
  - Organisms < 1 yr "increased in abundance shortly after experimental trawling but showed no response"
    in comparative studies, so they have no r estimate. The r values were fitted to abundance (biomass or
    numbers) along trawling gradients.
- **Hiddink et al. 2017**: community biomass r 0.82 /yr (0.42-1.53) [S via the earlier note]. With
  r = 5.31/L this implies L ~ 6.5 yr (computed in the earlier note).
- **Theory (inference)**:
  1. Logistic equilibrium with explicit predation: r(1 - B*/K) = predation rate per unit biomass. Measured
     P/B equals total Z = predation + other deaths (Allen), so P/B >= r(1 - B*/K). At B* = K/2 with
     predation as the dominant loss, r ~ 2 P/B.
  2. Schaefer: maximum surplus production is rK/4 at B = K/2, where surplus/B = r/2. A population held
     near K has a large P/B (tissue turnover) but a net logistic surplus near 0, so the two quantities
     measure different things.
  3. Empirical r/(P/B) (computed by me): 1-3 yr class r 2.41 vs Brey P/B 1.3-1.8 gives ~1.3-1.9; >10 yr
     class r 0.34 vs A. filiformis 0.46 gives ~0.7; Hiddink 2017 community 0.82 vs North Sea community
     P/B 0.8-1.3 gives ~0.6-1.0. A working range is **r ~ 0.7-2 x P/B**.
- **Effective r for the prey fraction (inference)**:
  - polychaetes, amphipods, cumaceans, juvenile shrimp (lifespan 1-3 yr): r 1.8-2.4 (5.31/L, L 2-3 yr);
  - Pandalus (adult M ~0.6-0.7) and medium polychaetes (3-10 yr): r ~1.2;
  - Nephrops and brittle-star bodies (>10 yr): r 0.27-0.5;
  - brittle-star arm tips: renewed at the regeneration rate, not by demography. The flux is ~0.06 /yr of
    whole-animal biomass (Skold 1994), but the turnover of the arm-tip pool itself was not found.

## B. Evidence: food versus predation limitation

### B.1 Long-term change in the Skagerrak-Kattegat (bottom-up)
- **Josefson 1990**, MEPS 66:117-130 [F]. https://www.int-res.com/articles/meps/66/m066p117.pdf
  - 14 stations "below the halocline at water depths from 18 to 300 m", 1972-1988. Stn 10 is at 300 m
    (58 14.9'N 10 34.3'E) and Stns 8, 11 and 12 at 100 m.
  - "Total biomasses showed linear increases by a median factor of 1.8, primarily from 1981 to 1988".
    Biomass without large species also rose, but the factor's digit is illegible in the PDF text layer.
    "Polychaetes contributed more to the increase at the deepest stations, and the absolute increase of
    this group was positively related with water depth." "The greatest increase in relative terms occurred
    at the deeper stations, most of them situated offshore, in the Skagerrak."
  - He rejects predation release on three grounds: "(1) There are no data showing a general decrease in
    stocks of demersal fish or other potential predators" (cod, whiting, plaice and haddock landings did
    not change dramatically); "(2) A part of the increase in polychaete biomass was due to Heteromastus
    filiformis ..., a species which due to its deep burrowing habits is likely to be less affected by
    predation"; "(3) Different predators dominate in different areas, whereas the infaunal biomass change
    was more or less general."
  - "The remaining possibility is increased sedimentation of organic material to the bottom. This
    explanation assumes that the infauna is food limited." Earlier circumstantial evidence: "decreased
    growth at high densities (Josefson 1982); decreased densities of subsurface feeders in areas with low
    content of buried organic material; and decreasing biomass with water depth (Josefson 1985)".
  - Skagerrak infaunal biomass correlated with Swedish run-off, best at a 2-yr lag (p = 0.01). Abstract:
    "reliable data on primary production and sedimentation of organic material are largely lacking".
- **Josefson, Jensen & Aertebjerg 1993**, JEMBE 172:31-45 [A]. https://marineinfo.org/id/publication/26699
  - "the best correlation was obtained when the benthic variables lagged run-off with 1 and 2 years. The
    results are consistent with the hypothesis that sedimentation of pelagic production was particularily
    high in the transition period". Winter DIN and chl a in the top 15 m also rose with run-off.
- **Heip et al. 1992**, ICES JMS 49:13-22, North Sea Benthos Survey [F].
  https://www.vliz.be/imisdocs/publications/ocrd/58011.pdf
  - "Biomass increases consistently in finer sediments and sediments with a higher chlorophyll a content."
    "The correlation between benthic biomass and chlorophyll a in sediments certainly indicates a link
    with surface productivity". "The impact of stratification on food input to the benthos may be one of
    the most important factors explaining large scale patterns".
- **Nilsen et al. 2006** [F]: production far exceeded fish consumption. Citing Clarke (1991), "limited food
  availability is the main cause for the low P/B ratios observed at high latitudes."
- **Stahl et al. 2004** [A]: "no relationship was found between the benthic fluxes and the macrofaunal
  biomass in the chambers".

### B.2 Counter-evidence: top-down effects
- **Skold et al. 2018**, MEPS 586:41-55, Kattegat trawling gradient [A via OpenAlex].
  https://doi.org/10.3354/meps12434
  - "Amphiura chiajei, increased in abundance from low to medium trawling intensities. We interpret this
    ... as a consequence of reduction in predation by benthivorous flatfish and Norway lobster".
- **Skold et al. 2025**, Conserv. Sci. Pract., 426 km2 no-take zone in the S Kattegat, 12 yr [A via OpenAlex].
  https://doi.org/10.1111/csp2.70037
  - Brittle stars "decreased in the closed area (abundance effect size A. filiformis 48%; A. chiajei 45%).
    In parallel, there was an increase in benthivore flatfish ... brittle stars were the staple food. Food
    web effects thus appear to override the decrease in mortality of the dominant macrofauna caused by the
    cessation of trawling".
  - The full text (https://pub.epsilon.slu.se/37348/1/skold-m-et-al-20250612.pdf) could not be fetched,
    so no biomass numbers or depths are available.
- **Skold 1994** [F]: 22 % of adult brittle-star biomass is regenerated tissue, i.e. sublethal predation is
  pervasive. Duineveld & van Noort 1986 [A]: dab take 6 % of arms per year with "little if any effect".
- Reading (inference): the top-down cases are shallow Kattegat (Skold) or flatfish-rich southern North Sea
  (Duineveld) systems where flatfish are the predators. Josefson's offshore deep stations are the closest
  analogue to the model box, and there the bottom-up signal is strongest.

### B.3 Carbon supply to the deep Skagerrak / Norwegian Trench
- **Diesing et al. 2021**, Biogeosciences 18:2139-2160 [F; numbers from a WebFetch extraction of the full
  text, not verbatim]. https://bg.copernicus.org/articles/18/2139/2021/
  - Norwegian Trough organic-carbon accumulation ~4-66 g C m-2 yr-1, mean 19.4, the highest in the North
    Sea; fjords 12-54. ~26 % of the upper-0.1 m stock lies in the Trough.
- **Stahl et al. 2004**, J. Mar. Res. 62:867-907, benthic lander, S and NE Skagerrak [A].
  https://research.chalmers.se/en/publication/251396
  - "Organic carbon oxidation rates (C ox) (2.2-18 mmol C m-2 d-1)"; "burial efficiencies, with annual means
    of ~62% and ~43%"; "POC deposition, from benthic flux and burial measurements, was only 24-78% of the
    sediment trap measured POC deposition, indicating a strong near-bottom lateral transport and
    resuspension". The advected material is of lower quality.
- **de Haas & van Weering 1997**, Mar. Geol. 136:173-187 [A]. https://marineinfo.org/id/publication/31903
  - Organic-carbon accumulation is "0.17 x 10^6 tons yr-1 and 0.83 x 10^6 tons yr-1" in the Norwegian Channel
    and Skagerrak/N Kattegat. "Less than 10% of this is accounted for by local primary production";
    terrigenous sources supply 20 %. Mean OC content is 0.6 % in the Norwegian Channel and 1.8 % in the
    Skagerrak.
- Local production: PP ~150 g C m-2 yr-1 (range 120-200) and export ratio 23-39 % (Richardson et al., late
  summer) [S via `Primarproduktion och C:chl i Skagerrak/pp_and_c_chl.md`].
- Macrofauna share of sediment respiration [S, search-engine summaries, not verified]: 30-40 % in an
  Oresund study whose authors the summary did not name; "equal" to pore-water advection in the German
  Bight (van Beusekom et al. 2021, https://e-docs.geo-leo.de/handle/11858/8755); small in permeable sand.
- Duineveld et al. 1991 macrofauna food demand: 40 / 10 / **3 g C m-2 yr-1** in the southern / central /
  northern North Sea [S via Mackinson & Daskalov 2007, through the earlier note].

### B.4 Production ceiling from the carbon budget (inference, every step an assumption)
1. Carbon reaching the bottom: deposition = burial / burial efficiency = 19.4/0.62 to 19.4/0.43
   = **31-45 g C m-2 yr-1**. Mineralisation is then 12-26 g C, in the lower part of Stahl's measured
   9.6-79 g C. Working range for sediment respiration: 20-40 g C m-2 yr-1.
2. Macrofauna share of respiration 0.3-0.5 [S], so macrofauna respire 6-20 g C m-2 yr-1.
3. Production:respiration. Mackinson P/Q for macrobenthos is 0.15-0.35; with an assumed 30 %
   unassimilated, P/R is ~0.3-1.0. Macrofauna production is then ~2.4-20 g C m-2 yr-1 (central ~5-8).
4. Conversion to wet weight:
   - brittle stars: WW:AFDW 11.5 [F, Skold];
   - polychaetes: AFDW/WW ~16 % (Ricciardi & Bourget, [S via the earlier note]);
   - C/AFDW assumed 0.45 (not retrieved);
   - result: **1 g C ~ 14 g WW (polychaetes) to ~25 g WW (brittle stars with skeleton)**; Ecopath's
     plankton rule is 0.1 g C = 1 g WW [S].
5. Ceiling: ~**20-500 g WW m-2 yr-1**, central ~70-150. For comparison: brittle stars alone at 40 m
   30 [F]; Sorfjord community 89 [F]; North Sea Ecopath macrobenthos ~750 [S]; Duineveld's northern North
   Sea demand of 3 g C gives ~0.5-1 g C of production = ~7-25 g WW.
6. Model: rK/4 = 0.82 x 10 / 4 = **2.05 t WW km-2 yr-1**. At the 4.44 start it is 2.02. The food ceiling
   lies 10-250x above this, so it cannot bind at K = 10 unless K is read as the gadoid-accessible slice.
   On that reading K and r should come from that slice (A.4).

## P/B by taxon

| Taxon | Area | P/B /yr | Method | Tag |
|---|---|---|---|---|
| Amphiura filiformis, population | Skagerrak, Gullmarsfjord mouth, 40 m | 0.46 | cohort / size-group production, 1990-91 | [F] Skold 1994 |
| A. filiformis, adults (SG 3) | same | 0.37 | same | [F] Skold 1994 |
| A. filiformis, arm regeneration flux | same | ~0.06 of total B | 0.34 / 5.6 g AFDW (derived) | [F] data, derivation mine |
| A. filiformis, large adults | Galway Bay | 1.97 | O'Connor et al. 1986 | [S] via Skold 1994 |
| A. filiformis | location not stated (Clemarec & Menesguen 1980) | 2.8 (lifespan assumed 2 yr) | - | [S] via Skold 1994 |
| A. filiformis | North Sea (Ecopath) | 1.97 | Brey model | [S] Mackinson & Daskalov 2007 |
| Echinoderm, 0.1 kJ, infauna | 200 m, 6-7 C | 0.55-0.6 | Brey 1999 v2-00, computed by me | inference |
| Polychaeta | Sorfjord, N Norway, 69 N | 0.75 | Brey 1999 | [F] Nilsen 2006 |
| Polychaete, 0.03 kJ / 0.3 kJ | 200 m, 6-7 C | 1.3-1.4 / 0.8 | Brey 1999, computed by me | inference |
| Small infauna (polychaetes) | North Sea | 0.9 | Ecopath / Brey | [S] Mackinson 2007 |
| Crustacea | Sorfjord | 0.45 | Brey 1999 | [F] Nilsen 2006 |
| Motile crustacean, 0.03 kJ / 0.3 kJ | 200 m, 6-7 C | 1.65-1.8 / 1.0-1.1 | Brey 1999, computed by me | inference |
| Small mobile epifauna | North Sea | 1.4-1.9 | Ecopath | [S] Mackinson 2007 |
| Shrimp / Crangon | North Sea | 3.0 / 0.81-1.07 | Ecopath / Brey | [S] Mackinson 2007 |
| Pandalus borealis (P/B >= M) | Skagerrak + Norw. Deep | M adult 0.58-0.70, juv. 1.71 (median); best-fit 0.9-1.2 | stock assessment M | [F] WKPRAWN 2022, NAFO SCR 14/066 |
| Nephrops norvegicus (P/B >= M) | Skagerrak-Kattegat | M 0.3 males / 0.2 females | assessment input | [F] ICES 1993 |
| Nephrops | North Sea | 0.35-0.4 | Ecopath | [S] Mackinson 2007 |
| Macrobenthic community | Sorfjord | 0.29 (WW), 0.42 (kJ) | Brey 1999 | [F] Nilsen 2006 |
| Macrobenthic community | North Sea, Duineveld 1991 | 1.9 | - | [S] via Nilsen 2006 Table 4 |
| Macrobenthic community | Masfjord, W Norway | 2.6 | Salvanes et al. 1992 | [S] via Nilsen 2006 |
| Community biomass, logistic r | global trawling studies | 0.82 (0.42-1.53) | recovery fit | [S] Hiddink 2017 (earlier note) |
| r by longevity: 1-3 / 3-10 / >10 yr | global gradient studies | 2.41 / 1.24 / 0.34 | trawling-gradient fit | [F] Hiddink 2019 |

## Gaps
- No measured P/B or production for deep (>100 m) Skagerrak polychaetes, amphipods, cumaceans, Calocaris
  or juvenile Pandalus. The Brey values above are model output, not data.
- No production estimate for A. chiajei, the brittle star typical of deep Skagerrak mud
  (Brissopsis-chiajei association).
- No turnover of the arm-tip pool, and no field arm-loss rate for Skagerrak fish.
- Rosenberg's deep-Skagerrak macrofauna papers, Pearson, Josefson & Rosenberg 1985, Josefson 1981/1985
  (deep-community depth gradients) and Graf et al. (benthic-pelagic coupling) were not found or not
  accessible. No biomass by depth at 200-500 m in the Norwegian Trench was retrieved; SMHI and NIVA
  monitoring data would supply it.
- No Skagerrak sediment-trap POC flux numbers (only the 24-78 % ratio). The macrofauna share of sediment
  respiration is unverified for the Skagerrak.
- Pandalus F, Nephrops female growth and the current ICES Nephrops M were not retrieved. The Skold 2025
  full text (biomass time series) was blocked.
- The C:AFDW factor (0.45) is assumed; Brey's conversion spreadsheet was not opened.
