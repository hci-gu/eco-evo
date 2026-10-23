# Regional carrying capacity K for the harbour seal group (Mareld box)

Scope: harbour seal (*Phoca vitulina*) only. The model box is 60 x 60 km
(3,600 km2), SWEREF 99 TM E 211000-271000, N 6414000-6474000, about
57.81-58.31 N, 10.07-11.15 E. Everything here is expressed in the model's
units: tonnes in the box, and t/km2 as a grid-mean density (the engine's
`regional_carrying_capacity`).

Source grades: [F] full text read; [A] abstract only; [S] search snippet
only; [N] not found or could not access. "(Own calculation)" marks my
arithmetic and "(Own judgement)" marks my reasoning. Swedish quotes are
transliterated to ASCII.

Builds on, and does not repeat:
- `../Parametervärden för sälar i Skagerrak/population_and_mortality.md` (counts, r_max, M1, mean mass of 55 kg, hunting).
- `../Parametervärden för sälar i Skagerrak/distribution_and_impacts.md` (telemetry, EIA, haul-out time).
- `../Täthetsberoendets styrka per grupp/mammals_and_seabirds.md` section 4 (Carroll mu, theta-logistic).

Those notes cite Carroll et al. 2025 for the combined fit only. Section 1
below adds the per-region numbers from the supplementary tables.

Files:
- `seals.md` is this note.
- `seal_box_kernel.py` is the script behind the box-share calculation in
  section 3. It is a research script, not model code. It needs an
  EMODnet tile, and its docstring says how to fetch one.

---

## 1. Carroll et al. 2025: K and r by region

Carroll, D., Ahola, M.P., Carlsson, A.M., Galatius, A., Nilssen, K.T.,
Harkonen, T. & Harding, K.C. 2025. Declining harbour seal abundance in a
previously recovering meta-population. PLoS ONE 20(6): e0326933.
[doi:10.1371/journal.pone.0326933](https://doi.org/10.1371/journal.pone.0326933),
[PMC12208499](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208499/).

I read the main text (Europe PMC XML) [F], Supplementary Tables S1-S4
(PDFs) [F] and Figs 1 and 3 [F].

### Data and fit [F, Methods]
- **Data.** August moult aerial surveys from 2003 to 2023, each haul-out
  surveyed 2-3 times a year. The fitted quantity is the annual mean
  count of hauled-out seals.
- **Regions.** The paper uses 18 subregions:
  - Skagerrak = subregions 5-8: Hvaler (Norway, flown in the Swedish
    survey), Koster, Vaderoarna and Lysekil.
  - Norwegian subregions 1-4 (Farsund, Tvedestrand, Kragero, Faerder)
    were counted only in 2016 and 2022. They are excluded from the
    Skagerrak fit.
  - Kattegat = subregions 9-16, starting with Marstrand and Onsala. On
    Fig 1, Marstrand sits on the drawn Skagerrak/Kattegat boundary.
- **Models.** Exponential growth, N_t = N0 e^(rt), and logistic growth,
  N_t = K / (1 + ((K - N0)/N0) e^(-mu t)), both fitted by non-linear
  least squares to Year vs mean count, with t counted from 2003. The
  logistic fitted best (lowest AIC) for the combined area, the Kattegat
  and the Skagerrak.

### S2 Table: parametric fits [F]

| Region | Model | K (counted) | N0 (2003) | mu or r (/yr) | AIC |
|---|---|---|---|---|---|
| Kattegat-Skagerrak | logistic | **13,965 +/- 772** | 5,930 +/- 1,290 | mu 0.28 +/- 0.10 (p 0.016) | 359.2 |
| | exponential | - | 9,285 +/- 867 | r 0.02 +/- 0.01 | 367.8 |
| Kattegat | logistic | **9,509 +/- 510** | 4,242 +/- 726 | mu 0.25 +/- 0.08 | 336.9 |
| | exponential | - | 6,159 +/- 496 | r 0.03 +/- 0.01 | 345.8 |
| **Skagerrak** | logistic | **4,517 +/- 305** | 1,778 +/- 514 | mu 0.33 +/- 0.15 (p 0.045) | 344.4 |
| | exponential | - | 3,013 +/- 361 | r 0.02 +/- 0.01 | 352.0 |
| S.W. Baltic | exponential | (logistic could not be fitted) | 391 +/- 34 | r 0.07 +/- 0.01 | 264.5 |

### S3 Table: GAM-estimated counts [F]

| Region | Maximum | 2023 (CI95) | 2023 / K |
|---|---|---|---|
| Kattegat-Skagerrak | 14,189 (2017) | 12,507 (11,558-13,456) | 0.90 (Own calculation) |
| Kattegat | 9,540 (2017) | 8,640 (8,225-9,054) | 0.91 (Own calculation) |
| **Skagerrak** | 4,767 (2016) | **3,710 (3,309-4,111)** | **0.82** (CI of the 2023 count: 0.73-0.91; K +/- SE: 0.77-0.88) (Own calculation) |

### GAM end-of-series rates of change [F, Results]
- Kattegat-Skagerrak: -408 seals/yr (SE 242, CI -882 to +67).
- Kattegat: -236 seals/yr (SE 111).
- Skagerrak: -228 seals/yr (SE 102, CI -430 to -27).
- Abstract: the combined decline is about 3.3 % of the 2023 count per year.

### Colonies next to the box
- S4 Table [F] gives each subregion's GAM intercept, which is roughly its
  mean count over 2003-2023:
  - Hvaler 275, Koster 1,278, Vaderoarna 1,108, Lysekil 1,227.
  - Marstrand 963, Onsala 1,706.
- Fig 3 [F, my read-off, about +/-100]:

  | Subregion | GAM peak | GAM 2023 | 2023 count point |
  |---|---|---|---|
  | Lysekil | ~1,580 (~2016) | ~1,000 | ~1,320 |
  | Marstrand | still rising | ~1,750 | ~1,780 |
  | Vaderoarna | ~1,450 | ~1,150 | - |
  | Koster | - | ~1,250 | - |

  Together, Lysekil + Marstrand were about 2,830 in 2016 and about 2,750
  in 2023 (Own calculation).
- Carroll calls Lysekil and Onsala the two subregions "with the largest
  absolute rates of decline". Marstrand, the only subregion with a
  positive final rate, "may be influenced by local immigration" [F,
  Discussion].
- (Own judgement) The two colonies next to the box are therefore flat
  together, at about 0.97 of their 2016 sum. Lysekil alone is at about
  0.63 of its peak.
- S1 Table, Norway [F]: the total mean count of subregions 1-4 was 365 in
  2016 and 657.67 in 2022. Faerder alone was 360 in 2022.

### Correction from counted to total seals [F, Discussion]
- Carroll: "previous work has assumed the hauled out fraction ... to be
  between 57% and 65% [5,14]". That multiplies counts by 1.54-1.75, and
  the 2023 population is "close to 20,000". The authors call these
  "coarse estimates only".
- Other values (from `population_and_mortality.md`):
  - HaV 2025 uses 60-80 % observed (factor 1.25-1.67).
  - Olsen et al. 2010 use 1.75.
- Central value used here: **1.6**.
- (Own calculation) Ratios such as N/K do not depend on this factor,
  because counted N and counted K scale together.

### Converting K to total animals (Own calculation)

| Region | K (counted) | K (total, x 1.54-1.75) | Central (x 1.6) |
|---|---|---|---|
| Kattegat-Skagerrak | 13,965 | 21,500-24,400 | 22,300 |
| Skagerrak | 4,517 | 6,960-7,900 | 7,230 |

HaV 2025 estimates 6,000-7,000 animals in the Skagerrak today [F, via the
earlier note], which fits 3,710 counted x 1.6-1.9.

### mu is not r_max
The logistic mu (0.25-0.33/yr) is the fitted slope of a curve that
starts in 2003, right after the 2002 PDV kill. Carroll's Introduction
[F] supplies both reasons it overstates r_max:
- "The highest possible intrinsic annual growth rate for a harbour seal
  population with a stable age structure is slightly less than 13%".
- After 1988 the population grew "between 10.2% and 13.6% annually".
  "Temporary rates of increase above 13% were likely the result of the
  population structure being skewed towards young females following the
  mass mortality event".

So only K carries over to the model. r_max stays 0.11/yr. The
theta-logistic reading of mu (beta_tot about 0.21/yr at x = 0.9) is in
the density note and is not repeated here.

### What causes the levelling [F, Discussion; quotes]
- **Food (leading hypothesis).**
  - "it seems likely that food limitation, leading to lower birth rates,
    and potentially lower juvenile survival, is a contributing factor to
    declining counts".
  - The background is ecosystem change: industrial fishing since the
    1970s, fewer herring and cod, small fish dominating, eutrophication.
  - Seals "have likely shifted their diet in response to these changes,
    potentially limiting population growth".
  - The abstract says the declines may be "the result of well documented
    environmental degradation affecting prey availability and causing
    regionally lowered birth rates".
- **Haul-out space and behaviour.**
  - "Harbour seals compete for resources such as haul-out sites [22,59]
    and potentially for prey [23]".
  - Competition for haul-out areas "is likely to increase with population
    density, potentially leading to seals spending less time hauled out
    during peak moult". Vessel traffic and recreational use (2020) may
    lower counts.
  - Even so, the declines "likely reflect real changes in seal
    abundance", because pup counts and somatic growth fell too.
- **Hunting.**
  - Hunting is "a possible contributing factor".
  - But it is "unlikely ... that hunting alone is responsible ... given
    the negative trends observed in Danish colonies, where hunting
    pressure is negligible".
- **Bycatch.** Poorly quantified. "no reason to believe that bycatch
  levels have increased".
- **Disease.** The paper does not make disease a driver of the recent
  decline. PDV in 1988 and 2002 "killed up to 60% of the population on
  each occasion".
- **Predators and grey seals.** Grey seals "remain uncommon". White
  sharks are absent and killer whales sporadic, so inter-specific
  interactions are "unlikely".
- **Overall.**
  - "A mixture of density dependent and independent control of
    population growth is likely".
  - Logistic K values are "theoretical maximum values", and a logistic
    model "assume[s] a static ecosystem with a fixed carrying capacity".
  - A decline "may indicate an environmental shift rather than the
    population reaching an equilibrium".

---

## 2. Other K and density-dependence estimates

**Silva et al. 2021**, Ecosphere 12: e03343
([doi:10.1002/ecs2.3343](https://doi.org/10.1002/ecs2.3343)).
- Abstract [A]:
  - The metapopulation in Swedish and Danish waters grew from "about 2500
    to 25,000 over the past 40 yr".
  - Trend analysis and somatic growth curves indicate "the population is
    close to carrying capacity".
  - With full reproductive capacity, "hunting of a few hundred animals
    annually is not harmful to the long-term persistence of the
    population", but slightly lower fecundity "makes even limited hunting
    risky".
- K value [S]: a search summary says the PVA used K = 35,000 (total) as a
  "best guess", and that "logistic models suggest that the total carrying
  capacity for both areas can be around 30,000" (with 65 % hauled out). I
  could not verify this. The full text is blocked by Cloudflare at both
  Wiley and the KU repository. If correct, it is about 1.3-1.6 x Carroll's
  K converted to totals. Treat it as an upper bound from an earlier, more
  optimistic analysis.

**Harding et al. 2018**, Front. Ecol. Evol. 6: 59
([doi:10.3389/fevo.2018.00059](https://doi.org/10.3389/fevo.2018.00059)) [F]:
- Areas:
  - The Skagerrak covers about 32,200 km2, of which 8,890 km2 is
    shallower than 50 m.
  - The Kattegat covers 21,800 km2, of which 20,170 km2 is shallower
    than 50 m.
- Why 50 m: "common diving depths for feeding pups and subadult harbor
  seals are typically <50 m (Blanchet et al., 2016)". The 50 m cut is
  about juveniles, not adults.
- Seal density per km2 of ground shallower than 50 m, between the 1988
  and 2002 epizootics:
  - Skagerrak: 0.26 -> 1.18.
  - Kattegat: 0.39 -> 0.54.
  - Limfjord: 0.66 -> 1.66.
- Asymptotic length in the Skagerrak fell by 7 cm. The authors read this
  as "a possible 'smooth route toward carrying capacity'".
- The Limfjord "has fluctuated around a total abundance of 1,500 seals
  ... suggesting that this population has reached the carrying capacity".
- The Skagerrak growth rate fell from 11 %/yr before 2002 to 6.5 % in
  2003-2015.

**OSPAR QSR 2023** seal abundance indicator
([PDF](https://oap.ospar.org/media/filer_public/dd/59/dd59c79f-437e-44d6-b7a0-840fb38ff1f6/p00847_m3_seal_abundance_qsr23.pdf)) [F]:
- Skagerrak AU 22:
  - Count 3,865 in 2020 (series 2003-2020).
  - Long-term +119 % (80 % CI 83-162).
  - Short-term change 2014-2019 -1 % (-13, 13), "inconclusive".
- Kattegat AU 20: 7,529, long-term +99 %.
- Norway (Hvaler - Stad) AU 23: 1,911, with an increase of about 30 %
  from 2011-15 to 2016-20.
- OSPAR gives no K. It states that "it is not possible to know the
  undisturbed state, nor, for some areas, the current carrying capacity".

**Olsen et al. 2010**, NAMMCO Sci. Publ. 8 [F, via
`distribution_and_impacts.md`]: the population "might have reached the
carrying capacity in some areas". They use correction factor 1.75.

**HELCOM 2023** [F, via the earlier note]:
- "levelling off ... as a consequence of density dependence cannot be
  ruled out".
- The lowest estimated Kattegat population is 12,500.
- The limit reference level is 10,000.

**Norway, IMR via Fiskeridirektoratet hearings** [S plus F via the
earlier note]:
- About 900 harbour seals in all of Norwegian Skagerrak.
- The national management target is about 7,000 counted (counted
  2016-2021: 6,857).
- Immigration from Sweden is "sannsynligvis stor".
- No Skagerrak K is given.

**Not found [N]:**
- An ICES WGMME Skagerrak harbour seal K.
- A Danish DCE Skagerrak K.
- A Norwegian Skagerrak-specific population model.

**Historical anchor** [F, Carroll]: 16,500 seals in 1890 from bounty
statistics, "already under considerable hunting pressure". (Own
judgement) Carroll's K converted to totals (about 22,000) is
consistent with this, because the 1890 figure was itself a hunted
population.

**Where the population sits relative to K today**
(Own calculation, from Carroll S2/S3):

| Unit | N2023 / K |
|---|---|
| Skagerrak | 0.82 |
| Kattegat | 0.91 |
| Combined | 0.90 |
| Lysekil colony alone | ~0.63 of its own peak |
| Lysekil + Marstrand | ~0.97 of their 2016 sum |

This document uses **0.82** as the central ratio and 0.90 as the
sensitivity value. The range is 0.73-1.0.

---

## 3. How many seals use the box today (the load-bearing unknown)

### Facts
- **Box depth** (EMODnet mean DTM, given in the task, 1 km cells):
  - Min 42.7 m, max 532.5 m, mean 167.6 m.
  - Area shallower than 50 / 75 / 100 m: 25 / 239 / 832 km2.
  - There is no land and no haul-out in the box.
- My wide-tile check (below) gives: box sea area 3,578 km2, mean 168.0 m,
  26 km2 shallower than 50 m, 825 km2 shallower than 100 m. This matches
  the task's numbers.
- (Own judgement) The earlier guess in `population_and_mortality.md`
  (section 1, inferences) was that 300-1,500 km2 of the box might be
  shallower than 50 m. That premise is false.

### Telemetry and habitat literature
- **Carter et al. 2022**, Front. Mar. Sci. 9: 875869
  ([PDF](https://eprints.gla.ac.uk/274391/1/274391.pdf)) [F]:
  - Distance to haul-out was the "primary driver of distribution for
    harbour seals in all regions".
  - In Table 1, distance explains 6-98.7 % of deviance per region.
    Depth explains only 0.2-1.9 % where it was retained, and it was
    retained in 6 of 10 regions.
  - Density declines with depth in regions 1, 2, 3, 5, 7 and 9. There is
    a strong negative association with areas more than 50 km from the
    haul-out in regions 2 and 4.
  - The per-region response curves are in supplement S9, which I did
    not read [N].
  - Note: in UK regions, water 20-50 km from a haul-out is mostly
    shallower than 100 m. The small depth effect there says little about
    a 150-500 m trench slope (Own judgement).
- **Sharples et al. 2012**, PLoS ONE ([PMC3357409](https://pmc.ncbi.nlm.nih.gov/articles/PMC3357409/)) [F]:
  - Mean trip distance is 11-21 km in rocky, deep-water regions
    (Shetland, Orkney) and 86-100 km on sandbank coasts.
  - Where deep water lies close to the haul-out, seals dive deeper and
    make short trips. "water depth may limit the efficiency of foraging
    thus increasing the profitability of foraging closer to haul-out
    sites where water depth is shallower."
- **Cunningham et al. 2009** [S, via the earlier note]: 50 % of trips
  stayed within 25 km of the haul-out.
- **Riaz et al. 2026** [F, via the earlier note]: Norwegian Skagerrak
  haul-outs "typically occurred within 25 km" of tagging sites. The paper
  does not analyse at-sea locations.
- **Dietz et al. 2013** [S]: maximum range 249 km. Seals range most
  widely in February-March.
- **Bjorge et al. 1995** (Froan, central Norway) [S]:
  - Seals foraged "at or close to the sea bed".
  - Foraging sites ranged from shallow kelp areas 20 km offshore "to
    150-200 m deep basins with muddy sea bed".
  - So adults do use 150-200 m water where it is near.
- **Mareld EIA** (`/home/simon/eco-evo/MKB Mareld.pdf`, Sida 134 av 243,
  section 9.4.1, my own pdftotext extraction) [F]:
  - Seals stay mainly near the coast where there are "storre ytor med
    grunda bottnar dar de fodosoker ned till cirka 40-50 meter djup (SLU
    Artdatabanken, 2021a)".
  - "Nagon omfattande forekomst av sal ar inte sannolik inom det planerade
    vindparkomradet dar vattendjupet uppgar till 182-432 meter samt da det
    inte forekommer nagra narliggande viloplatser pa land."
  - Gullmarsfjorden, Maseskar and Pater Noster lie "cirka 35-40 km" from
    the farm.
  - Seals are therefore not assessed.
- **Diet near the box** (Koster 1977-79 / 1989, % by weight; Lundstrom et
  al. 2010 Table 4) [F, via the earlier note]:
  - Blue whiting 3/7 %, Norway pout 6/11 %, cod 20/23 %. Gadoids total
    about 51/56 %.
  - (Own judgement, general fish knowledge, not sourced here) Blue
    whiting and Norway pout are mainly fish of water deeper than about
    100 m in the Skagerrak. Their presence points to some foraging on
    the Norwegian Trench slope.
- **Survey design** [F, Harding 2018 above]: the 50 m feeding-ground cut
  is for pups and subadults.

### Calculation (Own calculation; script `seal_box_kernel.py`)

**Method**
- **Bathymetry.** Wide EMODnet tile, 57.4-59.2 N and 9.3-12.0 E,
  aggregated to about 1 km cells. Land is excluded. The box is
  selected by transforming each cell to SWEREF 99 TM.
- **Anchors and colony sizes** (2023, from Carroll Fig 3 / S1 Table).
  Coordinates are approximate, from general geography:

  | Colony | Anchor points | Counted 2023 |
  |---|---|---|
  | Lysekil | Smogen/Hallo 58.35 N 11.215 E; Gullmarn mouth 58.255 N 11.38 E; Maseskar 58.09 N 11.33 E | 1,050 |
  | Marstrand | Pater Noster 57.90 N 11.46 E; Marstrand 57.875 N 11.56 E | 1,750 |
  | Vaderoarna | 58.575 N 11.06 E | 1,150 |
  | Koster | 58.88 N 11.02 E | 1,250 |
  | Hvaler | 59.05 N 10.98 E | 400 |
  | Faerder (2022) | 59.03 N 10.45 E | 360 |

- **Kernel.** Each colony's at-sea usage per km2 of sea is proportional
  to exp(-d/L) x w(depth), where d is the straight-line distance to the
  anchor. Usage is normalised over all sea in the tile.
  - With this kernel, 50 % of usage lies within about 1.7 L.
  - L = 15 km therefore puts half of the usage within 25 km of the
    haul-out. That is the most generous reading of Cunningham's "50 % of
    trips within 25 km", since a trip's maximum distance exceeds its
    typical position.
  - L = 10 km matches the short trips on rocky, deep-water coasts
    (Sharples). L = 25-35 km is the long-range sensitivity.
- **Depth weights:**
  - none = Carter taken at face value.
  - mild = exp(-z/100 m).
  - strong = 1 at 50 m or shallower, falling linearly to 0 at 100 m
    (the EIA's 40-50 m).
- **Conversion to tonnes:** counted-seal-equivalents x 1.6 (count to
  total) x 0.8 (share of time at sea outside the moult; Riaz gives about
  17 % hauled out in Aug-Nov) x 55 kg (Leslie mean mass from the earlier
  note). That is 0.0704 t per counted seal-equivalent. The low/high
  factor set (1.54 x 0.75 x 50 kg to 1.75 x 0.9 x 60 kg) spans 0.058-0.095.

**Results: box share of each colony's at-sea usage**

| Depth weight | L (km) | Lysekil | Marstrand | Vaderoarna | Koster | Counted-equivalent | Seals at sea | B_box (t) | t/km2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 10 | 20.3 % | 8.2 % | 4.2 % | 0.2 % | 408 | 522 | 28.7 | 0.0080 |
| none | 15 | 26.2 % | 14.6 % | 8.9 % | 1.4 % | 655 | 838 | 46.1 | 0.0128 |
| none | 25 | 27.8 % | 20.9 % | 14.5 % | 5.4 % | 916 | 1,172 | 64.5 | 0.0179 |
| mild | 10 | 9.9 % | 4.4 % | 1.9 % | 0.1 % | 204 | 261 | **14.4** | 0.0040 |
| mild | 15 | 13.1 % | 7.5 % | 4.2 % | 0.5 % | 324 | 415 | **22.8** | 0.0063 |
| mild | 25 | 14.8 % | 10.5 % | 7.8 % | 2.6 % | 473 | 605 | 33.3 | 0.0092 |
| strong | 10 | 2.0 % | 2.4 % | 0.3 % | 0.0 % | 65 | 83 | 4.6 | 0.0013 |
| strong | 15 | 2.8 % | 3.3 % | 0.7 % | 0.1 % | 96 | 123 | 6.8 | 0.0019 |
| strong | 25 | 3.7 % | 3.9 % | 1.6 % | 0.5 % | 134 | 172 | 9.4 | 0.0026 |

Notes on the table:
- Hvaler and Faerder contribute 0-3 %. At L = 35 km the "none" row
  reaches 71.5 t.
- With the low/high conversion set, mild L = 15 gives 18.7-30.6 t.
- At 80 kg per seal, all tonnages are x 1.45.

**Reading** (Own judgement)
- The box share is dominated by the Lysekil haul-outs (Smogen, Hallo,
  Maseskar) 5-15 km off the NE and E edges, and secondarily by Pater
  Noster and Marstrand.
- **Depth weighting matters more than the distance scale.** Without depth
  weighting, a quarter of the Lysekil seals' usage would fall over a
  trench slope 100-500 m deep. That conflicts with the EIA and with
  Sharples' point that deep water near the haul-out shortens trips.
- With strong weighting, the box holds almost nothing. That conflicts
  with adult foraging at 150-200 m (Bjorge) and with deep-water prey in
  the Koster diet.
- The mild weighting with L = 10-15 km is the defensible middle.
- **B_box_today: central about 20 t** (about 360 seals at sea at 55 kg;
  0.0056 t/km2). Likely range 12-30 t; plausible range 5-45 t.
- **Seasonal caveat.** In June-August (pupping and moult) more animals
  are ashore, so the box holds less. In February-March seals range most
  widely, so it holds more. The figure above is an annual mean.

### Is 15 t still right? (Own judgement)
- The shallow-ground argument behind 15 t is dead: the box has 25 km2
  shallower than 50 m, not hundreds.
- Applied to the box, that argument would give about 1 t: 0.8 seals per
  km2 shallower than 50 m x 25 km2 = about 20 seals.
- But the shallow-ground argument is the wrong model for adults (see the
  kernel above). The kernel route gives a central 20 t (12-30 t).
- So **15 t sits in the lower-central part of the defensible range, not
  at its bottom**. A defensible change is to raise the start to about
  20 t (training range 5-45 t). Keeping 15 t (training range 5-30 t) is
  also defensible.
- I would not lower it.
- **Caveat that cuts the other way (corrected, resume 199.8).** Model
  seals eat about 100 % gadoids. The model's gadoid group has flatfish,
  wrasse and rockling lumped into it (normalisation B, resume 185.10),
  and on that basis real Skagerrak seals eat about 85-95 % "gadoids"
  and 3-10 % pelagic fish. So, as a gadoid predator, 15 t of model seals
  equals about 1.1x as many real seals. An earlier version of this note
  compared with the unlumped 51-56 % gadoids of Koster seals and said
  27-30 t; that was the wrong basis. The diet gap is small and is not a
  strong argument either way on the start biomass.

---

## 4. Two routes to K_box

### Route (a): K_box = B_box_today / (N_today / K_pop)
Ratio 0.82 (Skagerrak), with sensitivity 0.90 (combined). (Own
calculation)

| B_box_today | Ratio 0.82: K_box | t/km2 | Ratio 0.90: K_box | t/km2 |
|---|---|---|---|---|
| 20 t (central) | **24.4 t** | **0.0068** | 22.2 t | 0.0062 |
| 15 t (current start) | 18.3 t | 0.0051 | 16.7 t | 0.0046 |
| 12-30 t (likely range) | 14.6-36.6 t | 0.0041-0.0102 | 13.3-33.3 t | 0.0037-0.0093 |
| 5-45 t (plausible range) | 6.1-55 t | 0.0017-0.0152 | - | - |

### Route (b): K_box = K_pop x box share of the population's at-sea usage
The share here is habitat-based, which keeps route (b) independent of the
kernel in route (a).
- K_pop = Skagerrak K 4,517 counted x 1.6 = 7,230 seals.
- Multiply by 0.8 at sea and by 55 kg. (Own calculation)

| Share basis | Share | K_box (seals at sea) | K_box (t) | t/km2 |
|---|---|---|---|---|
| Ground shallower than 50 m (Harding 2018: box 26 / Skagerrak 8,890 km2) | 0.29 % | 17 | **0.9** | 0.0003 |
| Ground shallower than 100 m (box 825 / 6,950 km2 in the eastern Skagerrak of the tile) | 11.9 % | 686 | **37.7** | 0.0105 |
| Uniform sea area (box 3,600 / Skagerrak 32,200 km2, Harding) | 11.2 % | 646 | **35.6** | 0.0099 |

The 100 m denominator covers only the Skagerrak east of 9.3 E. The
western Norwegian coast and Jammerbugt are missing, so the share is an
upper value.

### Agreement (Own judgement)
- **Without a depth limit, the routes roughly agree.** Route (b) with the
  100 m share or the area share gives 36-38 t, about 1.5 x route (a)'s
  24 t. That is within route (a)'s own range.
- **With the juvenile 50 m feeding-ground logic, they disagree by a factor
  of about 25.** Route (b) then gives about 1 t.
- **The disagreement is the finding.** K_box inherits nearly all of its
  uncertainty from one question: how much adult Bohuslan seals forage on
  the 100-300 m slope 15-40 km offshore. No local telemetry answers it
  yet. The two MARHAB Koster tags from 2025 are anecdotal, and four more
  tags were planned for 2026.
- Route (a) is the better-founded choice because it uses the
  distance-decay structure that every telemetry study supports. The 50 m
  route models pups and subadults, not the population.

---

## 5. What K stands for in this model, and whether to use it (Own judgement)

### What it stands for
- **It belongs with the pelagic pool's K, not the gadoid one.** The
  gadoid K was "today's effective capacity". The pelagic pool's K was
  "stock-scale compensation plus net emigration". The seal K belongs to
  the second kind.
- **The box holds only the offshore tail of a coastal metapopulation.**
  That metapopulation is regulated along the whole coast:
  - It is most likely food-limited (Carroll; Harding 2018 somatic
    growth; Harding 2022 diet shift to poor cod, Norway pout and snake
    blenny).
  - It is limited mostly by prey the model's seals do not eat or that
    live outside the box: flatfish, sandeel, herring in the archipelago,
    small inshore gadoids.
  - It is also regulated at haul-outs (competition for space,
    disturbance, pupping sites).
  - Exchange between colonies redistributes animals. Marstrand gains
    while Lysekil loses, and seals migrate from Sweden to Norway.
- None of these processes exists in the model.

### Food-only regulation is the wrong representation here
- In the model, a seal's only density feedback is its in-box gadoid
  supply, and in-box gadoids carry a fraction of the real diet.
- If food were the only brake, the group would grow until it had eaten
  the gadoids down. That is the 15 -> 24 t then crash pattern already
  observed.
- The model would then express real off-grid regulation as gadoid
  depletion, which is a fabricated predation effect.
- A regional K stands in for regulation that happens off the grid and on
  prey the model does not have. It is not a claim about in-box resources.

### Why the logistic form fits
- Carroll's best-fit model for the counts is a logistic. The pre-2002
  levelling and the post-2016 plateau are what the regional K describes.
- The engine's logistic term, r_max (1 - B/K), uses grid-mean density, so
  it acts at the scale of the group. It does not shape the within-box
  distribution. That is appropriate, because the real regulation is not
  local to the box either.
- The loss term adds to food limitation. If food also binds, the
  equilibrium sits below K, which is the intended behaviour: K is a
  ceiling, and food can hold the group lower.

### Limits
- Carroll warns that the system is not static. The 2016-2023 decline may
  be environmental, so a fixed K describes today and not the future.
- For scenario runs, K should be read as "today's capacity of the coast
  for seals that use the box". If a scenario changes coastal prey, K
  should change too, and the model has no mechanism for that.

### Epizootics
PDV (1988: >50 %, up to 60 %; 2002: about 30-66 % by region) and the 2014
H10N7 avian influenza (about 10 %) are rare pulses outside K. Model them
as a separate pulse scenario if wanted, never by lowering K (HELCOM 2023;
Harkonen et al. 2006; Carroll 2025; via the earlier notes).

---

## 6. Hunting variants

### Today's hunting rates
Sweden and Denmark figures come from `population_and_mortality.md`
section 4 unless marked otherwise.

**Sweden**
- Protective hunting reached 390 seals in 2020 [F, Carroll].
- A licence hunt of about 700 started in 2022, with "less than 50%"
  filled [F, Carroll]. The licence hunt was not renewed in 2024 [F,
  Carroll].
- Bags [S, Naturvardsverket page]:
  - 2023/24: 285 (Vastra Gotaland 178).
  - 2024/25: 31 (Vastra Gotaland 24).
- Quotas: 200 for 2025/26 and 200 for 2026/27, of which Vastra Gotaland
  140 in the first period [F].

**Norway (Skagerrak)**
- 35-49 seals per year in 2015-2022 [F, Carroll].
- 2025 quota proposals: Ostfold 52, Vestfold 50, Telemark 17 [F].

**Denmark**
- About 10 per year in 2018, "since increased" [F, Carroll].

**As rates** (Own calculation)
- Vastra Gotaland (about 7,400 animals):
  - 2.4 %/yr in 2023/24.
  - 0.3 %/yr in 2024/25.
  - 1.9 %/yr at the 2026/27 quota ceiling.
- Norwegian Skagerrak: 35-49 of about 900 is 4-5 %/yr. These are
  Norwegian seals, mainly in Oslofjord and Hvaler, outside the box's
  catchment.
- Box-level biomass rate: **0.005-0.02/yr, central 0.01**, as in the
  earlier note. Hunting is concentrated near gear and skerries, not
  offshore.

### Equilibrium of the model with harvest h
B* = K (1 - h/r_max). With r_max = 0.11 and h = 0.01, B* = 0.909 K.
(Own calculation)

**Variant (a): K from the Carroll ratio plus a hunting impact of 0.01/yr**

| B_box_today | Ratio | K_box | t/km2 | Equilibrium with h = 0.01 | With h = 0.02 |
|---|---|---|---|---|---|
| 20 t | 0.82 | **24.4 t** | **0.0068** | 22.2 t (1.11 x B_today) | 20.0 t |
| 20 t | 0.90 | 22.2 t | 0.0062 | 20.2 t | 18.2 t |
| 15 t | 0.82 | 18.3 t | 0.0051 | 16.6 t | 15.0 t |
| 15 t | 0.90 | 16.7 t | 0.0046 | 15.2 t | 13.6 t |

- With the combined ratio of 0.90, h = 0.01 lands the equilibrium almost
  exactly on today's biomass.
- Do not read the Skagerrak gap (1 - 0.82 = 0.18) as h/r_max. Carroll
  attributes the decline mainly to food and the environment, since
  hunting-free Danish colonies decline too.
- With 0.82 and h = 0.01, the group would settle about 11 % above today.
  With h = 0.02 it settles at today's level.

**Variant (b): K = B_today, no hunting impact**
- K_box = **20 t (0.0056 t/km2)** central, or 15 t (0.0042 t/km2) if the
  start biomass stays at 15 t.
- This is simpler. It folds hunting into "today's regulated state",
  in the same spirit as the gadoid K, and the start biomass is the
  equilibrium.
- Its weakness: a scenario that changes hunting cannot be expressed
  without re-deriving K.

### Which variant to use (Own judgement)
- Variant (a) is the more transparent choice. It keeps hunting as an
  explicit impact, as the project does for fishing.
- Variant (b) is acceptable if no hunting scenario is planned.
- In both, the K loss and M1 = 0.10 are separate terms. K adds a
  density-dependent loss on top of M1 and does not replace it.

---

## Recommendation

| Quantity | Central | Range | t/km2 (central) | Basis |
|---|---|---|---|---|
| B_box_today (start biomass) | **20 t** (about 360 seals at sea) | likely 12-30 t; plausible 5-45 t | 0.0056 | Kernel calculation, mild depth weight, L = 10-15 km (Own calculation). Keeping 15 t is defensible (lower-central); do not lower it. |
| K_pop, Skagerrak | 4,517 counted (+/- 305) = about 7,230 total (6,960-7,900) | - | - | Carroll S2 [F] |
| K_pop, Kattegat-Skagerrak | 13,965 counted (+/- 772) = about 22,300 total (21,500-24,400) | - | - | Carroll S2 [F]. For comparison only (not a range on Carroll's K): Silva 2021 reportedly used 30,000-35,000 total [S, unverified]. |
| N_today / K | 0.82 (Skagerrak) | 0.73-1.0 (0.90 combined; local pair about 0.97) | - | Carroll S3 [F]; Own calculation |
| **K_box, variant (a)** (hunting 0.01/yr as impact) | **24 t** | 15-37 t (likely); 6-55 t (plausible) | **0.0068** (0.0041-0.0102) | Equilibrium about 22 t. If the start stays at 15 t: K = 18 t (0.0051). |
| **K_box, variant (b)** (no hunting) | **20 t** | 12-30 t | **0.0056** | K = B_today. If the start stays at 15 t: K = 15 t (0.0042). |

General rule: K_box = B_start / 0.82 with a 0.01/yr hunting impact, or
K_box = B_start without one. K should follow whatever start biomass is
chosen.

### Main uncertainties, largest first
1. **Box share of adult at-sea usage over the 100-300 m slope (depth
   weighting).** It moves B_box and K_box by a factor of about 10
   (5-45 t). No local telemetry exists. The Swedish MARHAB tags from
   2025-2026 are the source to watch.
2. **Distance scale L.** It changes B_box by about x 1.6 between L = 10
   and 15 km.
3. **Which ratio to use.** Skagerrak 0.82, combined 0.90, Lysekil alone
   0.63, Lysekil + Marstrand 0.97.
4. **Count-to-total factor, time at sea and mean mass.** Together about
   +/-30 %. 80 kg instead of 55 kg would add 45 %.
5. **Haul-out anchors.** Positions and the split of colony counts among
   anchors are approximate. Per-haul-out counts for Hallo, Maseskar and
   Pater Noster were not found [N]. Moving the Smogen/Hallo anchor 5 km
   east (to 11.30 E) changes the mild-weight result by only -2 % to -4 %
   (L = 15: 324 -> 317 counted-equivalents; L = 10: 204 -> 195) (Own
   calculation).
6. **Gadoid-only diet.** Against the lumped target (85-95 % gadoids incl.
   flatfish, wrasse and rockling; resume 185.10) the model seals' gadoid
   predation per tonne is about 1.05-1.2x that of real seals, worth about
   0.015/yr of gadoid M2 at 15 t (resume 199.8). (Corrected: an earlier
   version compared with the unlumped 51-56 % and said "doubles".)

---

## Fetch log
- Carroll 2025:
  - Main text [F] (Europe PMC full-text XML).
  - S1-S4 Tables [F] (journals.plos.org supplementary PDFs s003-s006).
  - Figs 1 and 3 [F].
  - S2 Fig [F, caption only].
- Harding 2018 [F] (Frontiers XML).
- Sharples 2012 [F] (Europe PMC XML).
- Carter 2022 [F] (Glasgow eprints PDF; supplement S9 not read).
- OSPAR QSR 2023 seal indicator [F].
- Mareld EIA [F] (local pdftotext).
- Silva 2021: abstract [A] via OpenAlex. Full text 403 or Cloudflare at
  Wiley and the KU repository. K value [S].
- Bjorge 1995 [S]. Dietz 2013 [S].
- Not found [N]:
  - ICES WGMME and DCE Skagerrak K.
  - Swedish dive-depth telemetry for Bohuslan.
  - Per-haul-out counts.
- EMODnet bathymetry WCS tile (57.4-59.2 N, 9.3-12.0 E) fetched for the
  kernel calculation. It is not stored in the repository.
