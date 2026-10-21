# Q1. Stock composition and density of the pelagic planktivore pool at the Mareld site

Scope: about 45 tool calls. They covered ICES DATRAS NS-IBTS raw haul data (HH and HL, 2018-2024, Q1 and Q3),
WGIPS 2026 (HERAS and IESSNS supplementary documents, PDF to text), the ICES Standard Graphs summary tables, the DTU
"Sandeel fishing grounds" shapefile, EMODnet bathymetry point samples, a DTU Aqua HERAS cruise report, and web
searches. No code or library value was changed.

Labels used throughout:
- **[L]** literature, quoted or paraphrased from the source.
- **[D]** my own computation on primary data (DATRAS, shapefile, survey tables).
- **[C]** my own arithmetic on published numbers.
- **[I]** inference.

Already known and not redone (see the sibling notes):
- Stock-level F, TSB and the HERAS 2024-2025 strata biomasses:
  `../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`.
- The IBTS herring age structure at the site, Q1 against Q3: `../Sillens vinterbudget/winter_budget.md`, section 4.
- North Sea 1991 Ecopath biomasses and P/B: `../Litteraturbaserad growth rate per FG/fish.md`.
- Stock-recruitment and density dependence: `../Täthetsberoendets styrka per grupp/pelagic_fish.md`.

---

## 0. Where the grid actually is

The grid corners in `coordinates.txt` are the outer edges of `vindparker.png`, the 60 x 60 km grid. They put it at
**57.81-58.31 N, 10.07-11.15 E** [D]. This is about half a degree south of the "58.3-58.8 N, 10.5-11.0 E" in the
brief. The rest of this note uses the coordinates file.

- **ICES statistical rectangles.** Rows follow row = floor((lat - 36) x 2) + 1, and the column for 10-11 E is G0.
  The grid therefore lies in **44G0** (57.5-58.0 N) and **45G0** (58.0-58.5 N), with thin slivers of 44G1 and 45G1
  east of 11.0 E [C].
- **ICES subdivision.** All of the grid is in **27.3.a.20, the Skagerrak** [I].
- **HERAS stratum.** DTU Aqua divides HERAS in 3.a into stratum 21 (Kattegat), stratum 31 (inner Skagerrak) and
  strata 41/42 (outer Skagerrak). The grid is probably in **stratum 31**, possibly touching stratum 41 [I]. Source:
  [DTU Aqua, R/V Dana HERAS cruise report 2024](https://fiskeridir.no/English/Coastal-management/Marine-scientific-research/cruises-2024/_/attachment/download/29c5d330-e0d9-4788-a56a-1c2ec8e39de9:84c5f2b81e2044bc21bb6e85a2c2d4185211922e/DANA%20cruise%200424-04.04.2024,%2022.06.2024-08.07.2024.pdf).
- **Depth.** EMODnet DTM point samples [D]:

  | Point | Depth |
  |---|---|
  | 58.06 N 10.60 E | 173 m |
  | 58.15 N 10.75 E | 213 m |
  | 58.30 N 10.10 E | 517 m (Norwegian Trench slope) |
  | 57.85 N 10.30 E | 86 m |
  | 57.85 N 11.10 E | 52 m |
  | 58.30 N 11.10 E | 72 m |

  The repository's `djup.png` has 255 = deepest pixel. If that pixel is the ~517 m point, its quartiles are about
  100 / 140 / 205 m, with about 10 % of cells shallower than about 80 m [D, scale assumed].
  - So the grid is mostly a deep, muddy basin slope. "~100 m" is the shallow end of it.
  - IBTS hauls inside the grid averaged 121 m (Q1) and 145 m (Q3) [D].

---

## 1. Which species are in the pool at the site

### 1.1 Sandeel: absent from the site; it does not belong in the site pool

**Literature**
- Lesser sandeel (*Ammodytes marinus*) is restricted to coarse, well-oxygenated sand.
  - It is absent from sediment with more than 10 % silt/clay (Wright, Jensen & Tuck 2000, J. Sea Res. 44: 243-256,
    [doi:10.1016/S1385-1101(00)00050-2](https://doi.org/10.1016/S1385-1101(00)00050-2); abstract and secondary).
  - Preferred depth is about 30-50 m (habitat models summarised by the
    [Scottish Government sandeel evidence review](https://www.gov.scot/publications/sandeel-consultation-review-scientific-evidence/pages/2/);
    snippet) [L].
- Fishing grounds, taken as a proxy for foraging habitat, cover only about 5 % of the North Sea. Mixing between
  grounds is so small that length distributions differ at 5 km (Jensen, Rindorf, Wright & Mosegaard 2011, ICES J.
  Mar. Sci. 68: 43-51, [doi:10.1093/icesjms/fsq154](https://doi.org/10.1093/icesjms/fsq154); abstract) [L].

**Data**
- The grounds polygons from that study, as updated in 2022
  ([DTU data, doi:10.11583/DTU.21387513](https://doi.org/10.11583/DTU.21387513)), contain 217 grounds covering about
  33 600 km2.
  - **None lies inside the grid** [D].
  - The nearest is a small ground of about 80 km2 at 57.68-57.72 N, 10.01-10.29 E, off Hirtshals-Skagen. It is
    **about 10 km south** of the grid's southern edge.
  - The next is a ground of about 585 km2 at 57.45-57.64 N, 8.99-9.62 E (Jammerbugt), about 35 km from the grid [D].
- NS-IBTS 2018-2024, 203 valid GOV hauls in 57.6-58.6 N, 9.8-11.5 E [D]:
  - Ammodytidae occurred in 3 % of Q1 hauls at about 0 kg/h.
  - *A. tobianus* occurred in 1 % of Q3 hauls (4 % inside the grid) at about 0 kg/h.
  - The GOV under-samples buried sandeel, but the grounds map does not depend on the trawl.

**Inference** [I]
- Adult sandeel do not live in the grid. Pelagic sandeel larvae and 0-group may drift through in spring, and a
  shallow-water *A. tobianus* may occur along the Swedish coast just east of the grid.
- The **site pool weight for sandeel is about 0**.
- The library's r_max of 0.648 carries a 36 % sandeel share at r 1.02 (North Sea 1991 Ecopath weights, report
  section 3.4). It therefore **over-weights the site's productivity**. Without sandeel the same FishBase priors give
  about 0.44 (section 4).

### 1.2 Herring: dominant all year

- **Stock components.** In the Skagerrak the herring are mainly North Sea autumn spawners (NSAS) and western Baltic
  spring spawners (WBSS), plus local Skagerrak spring spawners [L, `ices_pelagic_f.md` Q3].
  - NSAS are mostly 1-2-ringer juveniles from the Skagerrak-Kattegat nursery. They leave for the North Sea at age 1-2.
  - WBSS adults feed here in summer and leave for the Sound in August-September.
- **HERAS stock shares.** WBSS was 32 % of herring biomass in the 2025 HERAS Skagerrak strata and 14 % in 2024. In
  stratum 31 in 2025, NSAS weighed a mean of 41 g and was 8 % mature [L].
- **IBTS at the site.** Q1 biomass is about 90 % 1-2-ringers. The Q1/Q3 index ratio is about 0.33 for ages 1+
  [D, `winter_budget.md`].

### 1.3 Mackerel: a large summer-only component

**Literature**
- The IESSNS North Sea stratum (stratum 13) includes the Skagerrak up to 11 E and north of the 50 m isobath, so it
  covers the grid. It has an area of 285 781 km2. Mean mackerel catch densities in July-August
  ([WGIPS 2026, Suppl. Doc. 4 IESSNS, Appendix 1](https://ices-library.figshare.com/articles/report/Working_Group_on_International_Pelagic_Surveys_WGIPS_/33406126);
  fetched full text) [L]:

  | Year | 2018 | 2019 | 2020 | 2021 | 2023 | 2024 | 2025 |
  |---|---|---|---|---|---|---|---|
  | t/km2 | 1.74 | 1.01 | 1.32 | 2.43 | 2.36 | 2.00 | 2.06 |

  - The 2025 StoX estimate is "586 020 tonnes" for the stratum, i.e. **2.05 t/km2** [C].
  - The text adds that inconsistent cohort signals reflect "interannual variations in the migration of the cohorts in
    and out of the North Sea and may confirm that mackerel in the North Sea is not a separate stock".
- In the DTU HERAS cruise of June-July 2024, mackerel was 44.3 % of the total trawl catch, against 40.2 % herring
  and 8.9 % sprat. One haul was a 7.3 t mackerel catch. Mackerel was "caught throughout the entire survey area with
  the largest catches in the Skagerrak and northern Kattegat" ([DTU Dana 2024 report](https://fiskeridir.no/English/Coastal-management/Marine-scientific-research/cruises-2024/_/attachment/download/29c5d330-e0d9-4788-a56a-1c2ec8e39de9:84c5f2b81e2044bc21bb6e85a2c2d4185211922e/DANA%20cruise%200424-04.04.2024,%2022.06.2024-08.07.2024.pdf);
  fetched full text) [L].

**Data**
- IBTS mackerel in the grid: **0.011 t/km2 in Q1** against **0.22 t/km2 in Q3** by swept area. In the wider box it
  is 0.006 against 0.50 [D].
- Mackerel is essentially absent in January-February.
- The GOV bottom trawl under-catches mackerel by roughly 5-10x relative to IESSNS: 0.2-0.5 against about 2 t/km2 [C].

### 1.4 Sprat: minor offshore, concentrated in the Kattegat and coastal fjords

**Literature**
- HERAS Div. 3.a sprat biomass in July was 3-58 kt over 2006-2025, and 3.3 kt in 2023
  ([WGIPS 2026 Suppl. Doc. 3, Table 5.13](https://ices-library.figshare.com/articles/report/Working_Group_on_International_Pelagic_Surveys_WGIPS_/33406126)).
  - In 2024-2025 "sprat was only found in the Kattegat (stratum 21)". In 2024 its "largest aggregations ... between
    Skagen and Læsø" [L].
- Coastal sprat in Norwegian fjords and Swedish coastal waters is outside both the assessment and the catch tables
  (HAWG 2026; `ices_pelagic_f.md`) [L].

**Data**
- IBTS sprat by swept area: **0.02 t/km2 in Q1 and 0.17 t/km2 in Q3** in the grid; 0.06 and 0.64 in the wider box
  [D].
- Year to year in Q3 (wider box): 0.01 (2022) to 2.04 (2023) t/km2 [D].
- Sprat is about 5-6 % of the pelagic GOV catch in the grid and 6-21 % in the wider box [D].

### 1.5 Horse mackerel: small

- IBTS horse mackerel: 0.001 t/km2 (Q1) and 0.04-0.06 t/km2 (Q3), i.e. 0.1-2 % of the pelagic GOV catch [D].
- Since the 2024 benchmark, 3.a horse mackerel belongs to the western stock, whose TSB is 1.6 Mt over a very large
  area [L, `ices_pelagic_f.md`].

### 1.6 Other planktivores the pool does not hold

**Data**
- In Q3, the wider box has two zooplanktivorous gadoids that are in neither the `pelagic_fish` pool nor `gadoids`
  [D]:
  - Norway pout, 0.34 t/km2 (Q1 0.06);
  - blue whiting, 0.67 t/km2 (Q1 about 0).
- Pearlside (*Maurolicus*) occurs in 35-42 % of hauls, but the GOV does not sample it quantitatively. Anchovy and
  sardine are traces [D].

**Inference** [I]
- The library deliberately leaves Norway pout and blue whiting out of `gadoids` because they eat zooplankton
  (report section 3.5), and they were never added to `pelagic_fish`.
- At the site in summer they may equal 20-50 % of the clupeid biomass by this crude GOV measure. Blue whiting is also
  mainly mesopelagic over the trench.
- They are a **real but unmodelled zooplankton consumer**. I keep them out of the pool weights below, as instructed,
  but flag them for the zooplankton-consumption question.

---

## 2. Density in t/km2

### 2.1 Herring from HERAS (July), per stratum [C]

- **Method.**
  - HERAS biomass per stratum is in WGIPS 2026 Tables 5.14-5.15 (NSAS + WBSS).
  - Stratum areas are not published in the report, so I approximated each one as transect length x transect spacing
    (Table 5.18; systematic parallel design).
  - 1 nmi2 = 3.43 km2.
  - These areas are rough, ±30 %. In both years the Swedish territorial waters of strata 21/31 were partly uncovered,
    by 17 % in stratum 31 in 2025.

| Stratum | Year | Herring (NSAS + WBSS), kt | Track x spacing, nmi2 | Approx. area, km2 | Density, t/km2 |
|---|---|---|---|---|---|
| 31 inner Skagerrak | 2025 | 22.1 + 6.7 = 28.8 | 135 x 10 = 1 350 | 4 630 | **6.2** |
| 31 inner Skagerrak | 2024 | 2.1 + 0.8 = 2.9 | 80 x 10 = 800 | 2 740 | **1.1** |
| 41 outer Skagerrak | 2025 | 30.3 + 23.1 = 53.4 | 170 x 17.5 = 2 975 | 10 200 | 5.2 |
| 41 outer Skagerrak | 2024 | 28.9 + 4.7 = 33.6 | 145 x 18 = 2 610 | 8 950 | 3.8 |
| 42 outer Skagerrak | 2025 | 14.1 + 1.9 = 16.0 | 100 x 17.5 = 1 750 | 6 000 | 2.7 |
| 42 outer Skagerrak | 2024 | 26.4 + 3.6 = 30.0 | 90 x 18 = 1 620 | 5 560 | 5.4 |
| 31+41+42 | 2025 | 98.2 | | 20 830 | **4.7** |
| 31+41+42 | 2024 | 66.5 | | 17 250 | **3.9** |

- **Cross-check against the Skagerrak's area.** The Skagerrak is about 32 000 km2 (SMHI/OSPAR 2007) or 47 000 km2
  (Wikipedia), depending on the boundary. 98.2 kt over it gives **2.1-3.1 t/km2** [C].
- **Reading** [I]: in July, herring density in the Skagerrak is **about 3-5 t/km2**, with single strata at 1-6 t/km2
  between years. Today's July herring alone is therefore about the model's 3.33.

### 2.2 IBTS swept-area densities at the site, 2018-2024 [D]

**Method**
- DATRAS HH/HL, valid GOV hauls.
- Biomass = CatCatchWgt / (towed distance x wing spread).
- These are **minimum** densities. The GOV has a headline of about 5 m and under-samples pelagic schools,
  especially mackerel.

| t/km2 (mean over hauls) | Grid box, Q1 (n 37) | Grid box, Q3 (n 23) | Wider box 57.6-58.6 N 9.8-11.5 E, Q1 (n 119) | Wider box, Q3 (n 84) |
|---|---|---|---|---|
| herring | 0.33 | 2.55 | 0.95 | 1.84 |
| sprat | 0.02 | 0.17 | 0.06 | 0.64 |
| mackerel | 0.01 | 0.22 | 0.006 | 0.50 |
| horse mackerel | 0.001 | 0.06 | 0.001 | 0.04 |
| sandeel | 0 | 0 | ~0 | ~0 |
| *Norway pout (not in pool)* | | | 0.06 | 0.34 |
| *blue whiting (not in pool)* | | | 0 | 0.67 |
| *for scale: cod + haddock + whiting + saithe + hake* | | | 1.33 | 3.13 |

- **Skew.** Distributions are very skewed. Total pelagic per haul in the grid had a mean of 0.37 and median 0.21
  t/km2 in Q1, and a mean of 3.00 and median 0.13 in Q3.
- **Year to year.** Wider-box herring was 0.12-2.0 t/km2 in Q1 and 0.7-3.6 t/km2 in Q3, by year.
- **Herring winter/summer ratio.** It is 0.13 in the grid box and 0.51 in the wider box by biomass, including the
  0-group in Q3. The age-1+ index ratio is 0.33 [`winter_budget.md`].
- **GOV catchability for herring.** July HERAS of about 4-5 t/km2 against August GOV of 1.8-2.6 t/km2 implies an
  effective catchability of about 0.4-0.6 [C, I]. Different months and areas, so only an order of magnitude.

### 2.3 Today's stock-scale densities, for context [C]

| Stock | Biomass today | Area assumed | t/km2 |
|---|---|---|---|
| NSAS herring | TSB 2.96 Mt (2025) | North Sea + 3.a + 7.d, ≈ 650 000 km2 | ≈ 4.5 |
| WBSS herring | TSB 0.27 Mt | | |
| sprat (North Sea + 3.a) | TSB 0.42 Mt | ≈ 630 000 km2 | ≈ 0.7 |
| mackerel, North Sea in summer | 0.59 Mt (IESSNS) | | 2.05 |

Comparison with the North Sea 1991 Ecopath, North Sea-wide: herring 2.60, sprat 0.58, mackerel 1.72, horse mackerel
0.58, sandeel 3.12; total 8.63, or **5.5 without sandeel** [L, fish.md].

---

## 3. Seasonal composition and recommended pool weights at the site [I]

**Construction (all [I], built on [L]/[D] above)**
- **Summer (Jul-Sep):**
  - herring 3.5 t/km2 (HERAS 3-5);
  - mackerel 1.5 (IESSNS 1.0-2.4 North Sea-wide, discounted a little for the inner Skagerrak, where no
    Skagerrak-only value exists);
  - sprat 0.3 (GOV 0.17-0.64 scaled by about 1/0.5 catchability, but HERAS saw no offshore Skagerrak sprat in
    2024-2025);
  - horse mackerel 0.1;
  - sandeel 0.
- **Winter (Jan-Mar):**
  - herring at 0.33 of summer (IBTS age-1+ index ratio; range 0.13-0.5), i.e. 1.2;
  - sprat 0.07;
  - mackerel and horse mackerel about 0.
- **Residence:** mackerel present about 5 months (June-October), herring all year, sprat all year.
- **Annual mean:** herring (6 x 3.5 + 6 x 1.2)/12 = 2.35; mackerel 1.5 x 5/12 = 0.63; sprat 0.2; horse mackerel 0.05.

| Species | Summer (Jul-Sep) t/km2 | Share | Winter (Jan-Mar) t/km2 | Share | Annual mean t/km2 | **Annual biomass weight** |
|---|---|---|---|---|---|---|
| herring (NSAS juveniles + WBSS + local) | 3.5 (1-6) | 65 % | 1.2 (0.5-1.8) | 94 % | 2.35 | **0.73** |
| mackerel | 1.5 (1.0-2.4) | 28 % | ~0 | ~1 % | 0.63 | **0.19** |
| sprat | 0.3 (0.1-0.6) | 5 % | 0.07 | 5 % | 0.20 | **0.06** |
| horse mackerel | 0.1 (0.04-0.2) | 2 % | ~0 | 0 % | 0.05 | **0.02** |
| sandeel | 0 | 0 % | 0 | 0 % | 0 | **0.00** |
| **total** | **5.4** (2.5-9) | | **1.3** (0.6-2) | | **3.2** (1.5-5) | 1.00 |

Uncertainty and caveats [I]:
- Herring and mackerel both vary about 2-5x between years.
- The herring stock split (WBSS 14-32 % of herring biomass in July) also moves between years.
- No Skagerrak-only mackerel biomass exists.
- The winter herring ratio is a demersal-trawl index.
- Weights for an all-year pool should be taken from the annual-mean column. A share-of-biomass in summer
  over-weights mackerel for production that mostly happens elsewhere.

**Consequence for r_max (a pointer for `productivity.md`)** [C]
- Applying these weights to the library's FishBase CMSY priors (herring 0.44, mackerel 0.43, sprat 0.48, horse
  mackerel 0.39):
  r = 0.73 x 0.44 + 0.19 x 0.43 + 0.06 x 0.48 + 0.02 x 0.39 = **0.44 /yr**.
- That is against 0.648 with the North Sea 1991 weights, of which 36 % is sandeel at 1.02.
- Using the North Sea 1991 weights without sandeel also gives 0.44 [C: 2.386 / 5.474]. The drop comes from removing
  sandeel, not from the exact site mix.

---

## 4. Is 3.33 t/km2 plausible for "today"? [I]

**Yes, as an annual mean.** It is plausible within a factor of about 1.5.
- The site annual mean is about 3.2 t/km2, with a range of 1.5-5 (section 3).
- July HERAS Skagerrak herring alone is 3.9-4.7 by stratum area, or 2.1-3.1 by whole-Skagerrak area.

It is **not plausible as a year-round constant**:
- the summer standing stock is about 5 t/km2, with mackerel present;
- the winter stock is about 1-1.5 t/km2, mostly juvenile herring.

A closed pool started at 3.33 in January therefore starts **about 2-3x above** the observed winter stock, and has no
summer immigration of mackerel and WBSS adults. This bears on `migration_scale.md`.

Comparison with the stock scale:
- 3.33 is below today's NSAS mean stock-area density (about 4.5 t/km2 TSB).
- It is below the North Sea 1991 Ecopath pool without sandeel (5.5).
- The Skagerrak is a juvenile nursery plus a summer feeding area, and is not the core habitat of the adult stocks.

---

## 5. Historical levels

**Herring, NSAS** [L, C]
- SSB from ICES SG her.27.3a47d (assessment 2026, key 22468):

  | Year | 1947 | 1950 | 1960 | 1965 | 1970 | 1975-77 | 1990 | 2005 | 2025 |
  |---|---|---|---|---|---|---|---|---|---|
  | SSB (Mt) | 5.0 | 3.7 | 2.5 | 2.0 | 0.45 | ≈ 0.1 | 1.8 | 2.2 | 1.68 |

- Today's SSB is **about 35-45 % of the 1947-1960 level**, and ICES says the stock has been in a "low productivity
  regime since 2002".

**Herring, WBSS** [L, C]
- SSB was about 280 kt in 1995 and 260 kt in 2000. It has been 92-111 kt in 2020-2025, i.e. **about 0.4x** the
  1990s level and below Blim (SG key 21294).

**Skagerrak herring catches** [L, C, I]
- Herring catch in 27.3.a.20 was **109-168 kt/yr in 1991-1995** and 0.35-0.79 kt/yr in 2022-2025 (HAWG 2026 Table
  3.1.1, via `ices_pelagic_f.md`).
- Over 32 000-47 000 km2, the 1990s catch is a removal of **2.3-5.3 t/km2/yr** [C].
- With a stock-level exploitation rate of about 0.3-0.4, that would imply a standing stock of roughly 6-17 t/km2 in
  the 1990s [I]. This is weak: much of that catch was of migrating fish taken in transit, and an in-area exploitation
  rate cannot be derived.
- At minimum, the Skagerrak once supported removals comparable to today's whole standing stock every year.

**Bohuslän herring periods** [L, I]
- Alheit & Hagen 1997 (Fish. Oceanogr. 6: 130-139,
  [doi:10.1046/j.1365-2419.1997.00035.x](https://doi.org/10.1046/j.1365-2419.1997.00035.x); snippet) document nine
  multi-decadal periods of massive coastal herring fisheries. Examples are 1747-1809 and one from 1877 to the
  mid-1890s. **More than 200 000 t** was landed in the 1895-96 season, separated by "interim" periods of 50+ years
  with no fishery. They tie the periods to cold, negative-NAO climate phases.
- These were incursions of large, probably North Sea / Atlanto-Scandian herring, overwintering near the coast. They
  are evidence that **local pelagic biomass in the Skagerrak is set by where large stocks choose to be**, not by local
  production.

**Mackerel** [L]
- The North Sea spawning component was over 2.5 Mt SSB in the early 1960s, collapsed in the 1970s and has been about
  150-230 kt in recent decades. Source: Jansen et al. 2012, PLoS ONE 7: e38758
  ([PMC3380938](https://pmc.ncbi.nlm.nih.gov/articles/PMC3380938); snippet).
- Jansen 2014 (ICES J. Mar. Sci. 71: 299-307,
  [doi:10.1093/icesjms/fst148](https://doi.org/10.1093/icesjms/fst148); snippet) argues it was a "pseudocollapse": a
  shift of the large NE Atlantic stock's spawning out of the North Sea, not a separate local stock lost.
- Summer feeding mackerel in the North Sea today is about 2 t/km2 (IESSNS).

**Sprat (spr.27.3a4)** [L]
- SSB was 308 kt in 1980, collapsed to 28-45 kt in 1985-1990, and has been 112-329 kt since 1995 (SG key 21311).
  There is no long-term trend since the 1990s.

**Synthesis** [I]
- Today's site pool is at a low historical level for its herring part (0.35-0.45x NSAS of the 1950s; 0.4x WBSS of the
  1990s). The mackerel part is about the 1991 level.
- The historical ceiling for the area is several times today's 3.3 t/km2.
- Like the gadoid case, today's level is a depleted, low-productivity state, not a carrying capacity.
- Unlike the gadoids, the site's level is mainly set by the distribution of migratory stocks of 0.3-3 Mt, whose
  ranges are tens to hundreds of times the grid area.

---

## 6. What could not be found

- **HERAS stratum areas** (nmi2) and **per-rectangle herring biomass** for 44G0/45G0. The areas above are
  transect-length x spacing approximations. The HERAS strata map (WGIPS Fig. 5.1) was not checked against the grid;
  the stratum 31 assignment is from the DTU text.
- **An Ecopath model specific to the Kattegat or Skagerrak.** None was found. WGSAM reviews North Sea and Baltic key
  runs only.
- **Skagerrak-only mackerel and horse mackerel biomass.**
- **HERAS 3.a herring biomass time series before 2024**, by stratum.
- A spatial planktivore biomass field by quarter for the North Sea and Kattegat exists in Maar et al. 2014 (Prog.
  Oceanogr. 124: 78-91, [doi:10.1016/j.pocean.2014.03.002](https://doi.org/10.1016/j.pocean.2014.03.002); not
  read). It is a candidate source for `zooplankton_consumption.md`.

---

## Source status

| Source | Used for | How read |
|---|---|---|
| ICES DATRAS NS-IBTS HH/HL web services, 2018-2024 Q1/Q3 ([datras.ices.dk](https://datras.ices.dk/WebServices/DATRASWebService.asmx)) | Species composition and swept-area density at the site | Raw data, own computation |
| [WGIPS 2026 Suppl. Doc. 3 (HERAS)](https://ices-library.figshare.com/articles/report/Working_Group_on_International_Pelagic_Surveys_WGIPS_/33406126) | Herring per stratum, sprat 3.a series, transect lengths | Fetched full text (PDF) |
| [WGIPS 2026 Suppl. Doc. 4 (IESSNS), Appendix 1](https://ices-library.figshare.com/articles/report/Working_Group_on_International_Pelagic_Surveys_WGIPS_/33406126) | North Sea and Skagerrak mackerel density 2018-2025 | Fetched full text (PDF) |
| [DTU Aqua R/V Dana HERAS cruise report 2024](https://fiskeridir.no/English/Coastal-management/Marine-scientific-research/cruises-2024/_/attachment/download/29c5d330-e0d9-4788-a56a-1c2ec8e39de9:84c5f2b81e2044bc21bb6e85a2c2d4185211922e/DANA%20cruise%200424-04.04.2024,%2022.06.2024-08.07.2024.pdf) | Strata 31/41/42 definition, catch composition | Fetched full text (PDF) |
| [ICES SG summary tables](https://sg.ices.dk/StandardGraphsWebServices.asmx), keys 22468, 21294, 21311 | NSAS, WBSS and sprat SSB history | Fetched data |
| [DTU Sandeel fishing grounds, doi:10.11583/DTU.21387513](https://doi.org/10.11583/DTU.21387513) (Jensen et al. 2011) | Sandeel grounds relative to the grid | Shapefile, own computation |
| [EMODnet Bathymetry REST](https://rest.emodnet-bathymetry.eu/) | Depth in the grid | Point samples |
| Wright, Jensen & Tuck 2000, J. Sea Res. 44: 243 | Sandeel sediment limits | Abstract / secondary |
| Jensen et al. 2011, ICES JMS 68: 43, doi:10.1093/icesjms/fsq154 | Sandeel habitat about 5 % of the North Sea, low mixing | Abstract |
| [Scottish Government sandeel evidence review](https://www.gov.scot/publications/sandeel-consultation-review-scientific-evidence/pages/2/) | Sandeel preferred depth 30-50 m | Snippet |
| ICES sandeel stock annex san.sa.3r | Stock area (includes SD 20) | Fetched full text (PDF) |
| [SMHI/OSPAR 2007 eutrophication report](https://www.smhi.se/en/publications/swedish-national-report-on-eutrophication-status-in-the-kattegat-and-the-skagerrak-ospar-assessment-2007-1.6639); [Wikipedia Skagerrak](https://en.wikipedia.org/wiki/Skagerrak) | Skagerrak area 32 000 / 47 000 km2 | Snippet |
| Alheit & Hagen 1997, Fish. Oceanogr. 6: 130 | Bohuslän herring periods, 200 kt in 1895-96 | Snippet |
| Jansen et al. 2012, PLoS ONE ([PMC3380938](https://pmc.ncbi.nlm.nih.gov/articles/PMC3380938)); Jansen 2014, ICES JMS 71: 299 | North Sea mackerel history | Snippet |
| `ices_pelagic_f.md`, `winter_budget.md`, `fish.md`, `pelagic_fish.md`, `reports/Litteraturbaserad growth rate per FG.md` (repo) | Stock F/TSB, HERAS 2024-25, IBTS ages, Ecopath 1991, r_max priors | Read locally |
| Kattegat/Skagerrak Ecopath; HERAS stratum areas and per-rectangle biomass; Skagerrak mackerel biomass | - | Not found |
