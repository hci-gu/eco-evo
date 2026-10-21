# How much zooplankton does the small pelagic pool eat, and does it control it? (task Q5)

Scope:
- consumption/biomass (Q/B) of the pool's species;
- the share of zooplankton production that planktivorous fish eat in the North Sea and
  Skagerrak;
- the evidence for and against top-down control of zooplankton (*Calanus*) by herring and
  sprat;
- the advective supply of *Calanus* to the Skagerrak;
- a plausibility check of the model's 3-year run against these.

Tags:
- **[L]** literature;
- **[C]** my arithmetic;
- **[I]** inference.

No code or library value was changed.

**Units.** The model's zooplankton tonnes are wet weight (WW). The library's half-saturation
"50 ug C/l = 5 t/km2 at 10 m" implies C:WW = 0.1. The North Sea Ecopath used 12.35 g C =
147 g WW, i.e. C:WW = 0.084. 1 t/km2 = 1 g/m2.

---

## 1. Q/B of the pool's species

### Literature [L]
- **North Sea Ecopath 1991** (Mackinson & Daskalov 2007, Cefas Tech. Rep. 142), from the local
  tables in `../Litteraturbaserad growth rate per FG/ecopath_ns1991/`:

  | Group | B (t/km2) | Q/B (/yr) | Zooplankton share of diet |
  |---|---|---|---|
  | Herring juvenile | 0.63 | 5.63 | 0.84 |
  | Herring adult | 1.966 | 4.34 | 0.93 |
  | Sprat | 0.579 | 6.0 | 0.79 |
  | Mackerel | 1.72 | 1.73 | 0.28 |
  | Horse mackerel | 0.579 | 3.51 | 0.45 |
  | Sandeel | 3.122 | 10.1 | 0.67 |
  | Misc. filter feeders | 0.030 | 10.19 | 0.84 |

  `fish.md` notes alternative values in brackets (sandeel 5.24, sprat 5.28).
  [Cefas Tech. Rep. 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- **Heath 2005, Table 3:** annual mean daily rations (g g-1 d-1 WW), from Greenstreet et al.
  1997.

  | Species | Daily ration | Q/B (/yr) [C] |
  |---|---|---|
  | Herring | 0.0120 | 4.4 |
  | Sprat | 0.0145 | 5.3 |
  | Sandeel | 0.0145 | 5.3 |
  | Mackerel | 0.0048 | 1.75 |
  | Horse mackerel | 0.0164 | 6.0 |

  Herring diet is 11 % omnivorous and 76 % carnivorous zooplankton (Heath's guild classes).
  [doi:10.1016/j.icesjms.2005.01.023](https://doi.org/10.1016/j.icesjms.2005.01.023) (fulltext)
- **Bachiller et al. 2018**, Norwegian Sea bioenergetics model, 2005-2010 (fulltext):
  - NSS herring and NEA mackerel each eat "around 10 times their total biomass" a year
    (results 9.5-12). Blue whiting eats about 6.
  - Herring daily ration is about 6 % BW over a modelled feeding season of 1 April to
    15 September (168 d).
  - The three stocks eat 135 Mt/yr of zooplankton, of which 53-85 Mt are copepods.

  These are large adult fish on an oceanic feeding migration, the upper end for herring.
  [doi:10.1371/journal.pone.0190345](https://doi.org/10.1371/journal.pone.0190345)

### Pool-weighted Q/B [C]

| Weighting | Q/B (/yr) |
|---|---|
| NS Ecopath 1991 biomass weights, sandeel Q/B 10.1 | 52.40 / 8.626 = **6.07** |
| Same, sandeel Q/B 5.24 | **4.32** |
| NS weights without sandeel | 3.79 |
| Herring only (juvenile + adult) | 12.08 / 2.596 = **4.65** |
| Site A: herring 0.8 (4.65), sprat 0.1 (6.0), mackerel 0.1 (1.73) | **4.49** |
| Site A with juvenile herring Q/B (5.63) | **5.28** |

**Zooplankton eaten per unit fish biomass** (Ecopath 1991) [C]:
- pool: 36.8 t of zooplankton per 8.63 t of fish = **4.3 /yr**;
- herring: 10.9 / 2.60 = 4.2 /yr;
- herring + sprat: 13.65 / 3.18 = 4.3 /yr.

**Inference [I].** For a herring-dominated, juvenile-heavy site pool, the literature Q/B is
about **4.5-5.3 /yr**, with about 85-93 % of it zooplankton. Bachiller's ~10 /yr is the ceiling
for adult herring on a summer ocean feeding migration. It is not an annual mean for juveniles
in a shelf sea.

---

## 2. Share of zooplankton production eaten by planktivorous fish

### Literature [L]
- **Heath 2005, Table 4,** North Sea decadal means (g C m-2 y-1):

  | Flow | 1970s | 1990s |
  |---|---|---|
  | Omnivorous zooplankton gross production | 35.7 | 36.6 |
  | Omnivorous zooplankton eaten by fish | 8.43 | 7.05 |
  | Omnivorous zooplankton eaten by carnivorous zooplankton | 13.8 | 9.85 |
  | Carnivorous zooplankton production | 4.15 | 2.95 |
  | Carnivorous zooplankton eaten by fish | 3.89 | 2.58 |
  | Planktivorous fish gross production (whole period) | 3.6-3.8 | |

  "For most of the time-series, the combined consumption of omnivorous zooplankton by fish and
  carnivores accounted for approximately half the omnivorous zooplankton gross production"; the
  fish community's demand "was approximately 40% of gross production, rising to 70% during
  1980/1981".
- **Heath 2007:** fish early life stages roughly double the earlier larval estimate and make up
  28 % of fish zooplankton consumption, raising the fish total by 14 %. "Consumption of
  zooplankton production by fish and other planktivorous predators is now estimated to be
  19-25 gC m-2 year-1." (fulltext)
  [doi:10.1093/icesjms/fsm156](https://doi.org/10.1093/icesjms/fsm156)
- **North Sea Ecopath 1991** (local tables, [C]):
  - Herbivorous/omnivorous zooplankton: B 16 t WW/km2, P/B 9.2, production 147.2 t/km2/yr,
    EE 0.49. Predators of it:
    - carnivorous zooplankton 30.8 t/km2/yr;
    - sandeel 19.1;
    - adult herring 5.4;
    - fish larvae 5.1;
    - sprat 2.75;
    - juvenile herring 2.2;
    - all predators together 67.8 t/km2/yr = 46 % of production.
  - The pelagic pool eats 30.1 t/km2/yr = **20 %** of herbivorous production, plus 6.7 of 13.4
    t/km2/yr carnivorous zooplankton production (50 %). That is 36.8 of 160.6 = **23 %** of
    all zooplankton production.
  - Herring + sprat alone eat 13.65 t/km2/yr = **8.5 %**.
- **Norwegian Sea (Bachiller et al. 2018):** 53-85 Mt of copepods eaten, against *C.
  finmarchicus* production of 200-300 Mt (uncertain), i.e. about **20-40 %** [C].

### Inference [I]
- In the North Sea, all fish together take about **20-25 %** of mesozooplankton production
  (Heath: 19 % of omnivore production plus most carnivorous zooplankton production).
- The small pelagic pool takes about 20-23 % (Ecopath 1991, sandeel included). Herring and
  sprat alone take about 8-9 %.
- Fish plus invertebrate predators take about half. Fish never take a majority of
  production, except in the 1980-81 low-production years (70 %).

---

## 3. Top-down control of zooplankton by herring and sprat: the evidence

### For [L]
- **Fauchald et al. 2011,** PLoS ONE, North Sea 1966-2006, winter CPR zooplankton against IBTS
  clupeids lagged one winter (fulltext):
  - Herring was negatively related to krill: coefficient -0.17 (p < 0.01), detrended -0.20.
  - Sprat was negatively related to *C. helgolandicus*: -0.30 (p < 0.01), detrended -0.33
    (p < 0.001).
  - No herring term was retained for *C. finmarchicus* or *Para/Pseudocalanus*.
  - The authors conclude "herring and sprat were important in regulating zooplankton through
    top-down interactions", with the caveat "could have been due to some external confounding
    factors".

  [PMC3145753](https://pmc.ncbi.nlm.nih.gov/articles/PMC3145753/)
- **Casini et al. 2009,** PNAS 106: 197, Baltic. Above a planktivore abundance threshold of
  about 17 x 10^10 individuals, "abundances of the planktivore sprat ... decouple zooplankton
  dynamics from hydrological circumstances". This is top-down control at very high clupeid
  densities. Central Baltic clupeids are about 15 t/km2 (herring + sprat, Tomczak et al.
  2013 in `fish.md`). (abstract)
  [doi:10.1073/pnas.0806649105](https://doi.org/10.1073/pnas.0806649105)
- **Norwegian Sea.** *Calanus* was low in 1996-1997 after a herring peak and a strong 1995 blue
  whiting cohort (search summary, IMR-related source, not verified). Bachiller et al. 2018 say
  the fish "may have a greater impact on the zooplankton community as foragers" than earlier
  thought, but do not show control.

### Against or weak [L]
- **Lynam et al. 2017,** PNAS, North Sea:
  - "Bottom-up processes, forced by temperature, have dominated change in the abundance of
    planktonic groups since the 1960s."
  - "Predation effects do not appear strong enough to significantly change the trajectories of
    the biomass time series for zooplankton groups."
  - Herring likely depresses *C. finmarchicus* in their model, with no magnitude given.
  - The sprat effect on *C. helgolandicus* is "much smaller than that of temperature".

  [PMC5338359](https://pmc.ncbi.nlm.nih.gov/articles/PMC5338359)
- **Heath 2005:** zooplankton production fluctuations were positively correlated with
  planktivore production, which suggests bottom-up control.
- **Falkenhaug et al. 2022,** Skagerrak (Arendal station 2), 1994-2019:
  - *C. finmarchicus* CV-VI "varied by a factor of 5 between high (~2010) and low (~2001)
    periods", with no trend.
  - One spring generation; adults and CV peak in mid-April to mid-May at about 2 500-3 200
    ind. m-2.
  - The authors analyse no local predation by fish. Predation is mentioned only as a possible
    contributor to the July decline, together with descent to overwintering depth.

  [doi:10.3389/fmars.2022.779335](https://doi.org/10.3389/fmars.2022.779335) (fulltext)

### Inference [I]
- **North Sea and Skagerrak.** The weight of evidence favours bottom-up and advective control
  of zooplankton biomass. Top-down effects of herring and sprat are detectable for specific
  taxa (krill, *C. helgolandicus*) as a modest modulation, not as control.
- **Where real control occurs.** Top-down control that sets zooplankton levels is documented
  where clupeid density is several times higher than the North Sea mean: the Baltic above the
  sprat threshold, at about 15 t/km2 of clupeids.
- **The model's present behaviour.** Herring halve the annual mean zooplankton and cut the
  winter stock 2x over three years. That is stronger top-down control than the
  North Sea/Skagerrak literature supports at 3-7 t/km2.

---

## 4. Advective supply of *Calanus* to the Skagerrak

### Literature [L]
- **Falkenhaug et al. 2022:** the Norwegian Trench (about 700 m, about 40 km from the Arendal
  station) is "a gateway for the advective supply, as well as a habitat for local
  overwintering". The adults in Jan-Feb are an overwintering generation G0.
- **Gao et al. 2021** (individual-based model):
  - High overwintering biomass in the Norwegian Trench.
  - "Inflow from the north constitutes on average 41% of the annual mean biomass in the North
    Sea", which sets the interannual variability.
  - The North Sea population is "not self-sustained".

  Progress in Oceanography 194: 102567,
  [doi:10.1016/j.pocean.2021.102567](https://doi.org/10.1016/j.pocean.2021.102567) (search
  summary and secondary via Falkenhaug)
- **Opdal & Vikebø 2016:** modelled cross-shelf transport of *C. finmarchicus* was stable over
  52 years, and the Norwegian Trench is one of two features funnelling it onto the shelf.
  [PDF](https://bio.uib.no/te/papers/Opdal_2016_Long-term_stability.pdf) (search summary)
- **Skagerrak water exchange** (SMHI Oceanografi 38, citing Rodhe 1987 and Grimås & Svansson
  1985):
  - turnover time of Skagerrak water about 100 days; a few weeks in strong-current areas,
    longer near the coast and below 270 m;
  - inflow at least 15 000 km3/yr; outflow about 15 500 km3/yr.

  [SMHI PDF](https://www.smhi.se/download/18.18f5a56618fc9f08e832a49d/1717801532204/Oceanografi_38%20Skagerrak%20-%20the%20gateway%20to%20the%20North%20Sea..pdf)
  (search summary; Rodhe 1987 not read)

### Inference [I]
- A 100-day turnover means the Skagerrak's water, and the zooplankton in it, is replaced about
  3.5 times a year. In a 60 km box inside the cyclonic circulation it is probably replaced
  faster (weeks).
- Local zooplankton biomass at the site is therefore set largely by import: Atlantic and
  Trench *Calanus*, and the Jutland Current. Local grazing pressure is diluted by that import.
- The model's closed zooplankton pool has no such import. Any herring increase therefore comes
  straight out of a fixed local stock, which exaggerates the food feedback.
- AGENTS.md already treats the site's zooplankton totals as *Calanus* risen from the Trench.
  For the prey, the closed-box assumption has already been abandoned in calibration. The
  dynamics have no import term.

---

## 5. The model's 3-year run against the literature [C]

Inputs from the brief: herring ate 105 / 132 / 150 kt/yr of zooplankton on 3 600 km2. The
pool's annual means were 4.07 / 5.54 / 6.77 t/km2, and zooplankton's 5.88 / 3.59 / 2.94 t/km2.

| Quantity | Year 1 | Year 2 | Year 3 | Literature |
|---|---|---|---|---|
| Consumption (t WW/km2/yr) | 29.2 | 36.7 | 41.7 | site pool 3.33 x 4.3-5.3 = 14-18 today |
| Q/B of pool (zooplankton only) | 7.2 | 6.6 | 6.2 | 4.3-5.3 (NS Ecopath, Heath); 10 Norwegian Sea adults |
| Consumption in carbon (g C/m2/yr, C:WW 0.1) | 2.9 | 3.7 | 4.2 | all North Sea fish: 7.05 + 2.58 = 9.6 (Heath 2005, 1990s) |
| Consumption / mean zooplankton biomass (/yr) | 5.0 | 10.2 | 14.2 | pool 36.8/19.35 = **1.9**; all predators on herbivores 67.8/16 = **4.2** (Ecopath) |
| Share of site zooplankton production* | 30-45 % | 37-56 % | 43-64 % | pool 20-23 %; herring + sprat 8.5 %; all fish 20-25 % |
| Zooplankton annual mean (t WW/km2) | 5.88 | 3.59 | 2.94 | site observations 7.1-10.6 (0-25 m, SHARK) |

\* Site production = observed annual mean biomass 7.1-10.6 g WW m-2 (from
`../Sillens vinterföda/winter_food_and_feeding.md`) x Ecopath P/B 9.2 = 65-98 t/km2/yr. This
uses observed biomass, not the model's depleted biomass, so it is an upper bound on the
production available.

### Inference [I]
1. **The model's Q/B is high but not absurd.** 6.2-7.2 against 4.5-5.3 for a herring-dominated
   North Sea pool, i.e. about 1.2-1.6x. It is below Bachiller's ~10 for ocean-feeding adults.
   The excess is in line with the model's high gross production budget (1.54 /yr).
2. **The loss rate the pool imposes on zooplankton is not plausible after year 1.** It reaches
   10-14 /yr of the zooplankton standing stock in years 2-3. That is 5-7x the Ecopath pool
   rate and 2.5-3.5x the rate from all North Sea predators of herbivorous zooplankton
   together.
3. **At today's density the consumption would fit.** At 3.33 t/km2 and literature Q/B, the pool
   would eat about 14-18 t/km2/yr, 15-27 % of site production, which matches the literature
   share.
4. **The doubled pool does not fit.** At the model's Q/B it eats 43-64 % of the observed
   production. It also faces a zooplankton stock already 1/3 of observed, because the model
   has no import of *Calanus*.
5. **The literature does not support "herring controls zooplankton" as the regulator.** The
   zooplankton decline is an artefact of a closed box plus an unregulated pool. It is not a
   realistic regulating feedback.
   - Two parts of the model shape this: no advective import, and the pool's own growth.
   - A realistic regulation of the pool would act at regional stock level (recruitment, stock
     density). It would leave the zooplankton near its observed, largely advected level.

---

## Source status

| Source | Used for | Read as |
|---|---|---|
| Mackinson & Daskalov 2007, Cefas Tech. Rep. 142 (local tables `ecopath_ns1991/`) | B, Q/B, diet, zooplankton production and consumption | local data (via sibling notes) and computed |
| Heath 2005, ICES JMS 62: 847 ([Strathprints](https://strathprints.strath.ac.uk/18596/)) | Table 3 rations, Table 4 flows, bottom-up statement | fulltext PDF |
| Heath 2007, ICES JMS 64 ([doi:10.1093/icesjms/fsm156](https://doi.org/10.1093/icesjms/fsm156); [Strathprints](https://strathprints.strath.ac.uk/18621/)) | larval consumption, 19-25 g C | fulltext PDF |
| Bachiller et al. 2018, PLoS ONE 13: e0190345 | Norwegian Sea C/B, consumption vs *Calanus* production | fulltext (page fetch) |
| Fauchald et al. 2011, PLoS ONE ([PMC3145753](https://pmc.ncbi.nlm.nih.gov/articles/PMC3145753/)) | top-down coefficients | fulltext (page fetch) |
| Lynam et al. 2017, PNAS 114: 1952 ([PMC5338359](https://pmc.ncbi.nlm.nih.gov/articles/PMC5338359)) | weak predation effects on zooplankton | fulltext (page fetch) |
| Casini et al. 2009, PNAS 106: 197 | Baltic sprat threshold | abstract (search summary) |
| Falkenhaug et al. 2022, Front. Mar. Sci. 9: 779335 | Skagerrak *Calanus* seasonality, Trench gateway | fulltext (page fetch, first 100k chars) |
| Gao et al. 2021, Prog. Oceanogr. 194: 102567 | 41 % inflow share, not self-sustained | search summary + secondary via Falkenhaug |
| Opdal & Vikebø 2016 | stable cross-shelf *Calanus* transport | search summary |
| SMHI Oceanografi 38 (Rodhe 1987; Grimås & Svansson 1985) | Skagerrak turnover ~100 d, inflow 15 000 km3/yr | search summary of the PDF |
| `../Sillens vinterföda/winter_food_and_feeding.md`, `../Zooplankton energibudget/copepod_energetics.md` | site zooplankton biomass 7.1-10.6 g WW m-2, Ecopath zooplankton values | local notes |
| Norwegian Sea *Calanus* low in 1996-97 after herring/blue whiting peak | top-down hint | search summary only, not verified |
