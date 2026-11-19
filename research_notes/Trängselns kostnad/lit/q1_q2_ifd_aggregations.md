# Q1-Q2: density-dependent habitat selection, local crowding costs and aggregation densities in demersal fish

Literature notes for "Trängselns kostnad" (the cost of crowding). Model problem: the benthivores
(haddock ~2/3, cod, flatfish) pack at Lloyd's mean crowding of 30-90x the mean, about 50-150 t/km2
in 1 km2 cells, with intake elasticity on own local density of -0.012. Field crowding (NS-IBTS GOV,
Skagerrak 2021-25) is 3.1, and the largest single haul is about 30 t/km2.

Evidence grades:
- [F]: full text read locally, number read in the text or a table.
- [F-w]: publisher full-text HTML read through WebFetch, with the quotes as WebFetch returned them.
  Treat it as one step weaker than [F].
- [A]: abstract only.
- [S]: secondary (a search snippet or another paper citing it).

"(computed)" marks numbers I derived; they are not printed in the source.

Locally saved PDFs and text extractions are in `papers/q12/`, and their licences are in
`papers/q12/LICENCES_q12.md`.

## Key numbers

| Study | Species / area | Spatial scale | Currency | Magnitude | Density range covered | Exploitation vs interference separated? | Grade |
|---|---|---|---|---|---|---|---|
| Auestad 2023 (MSc thesis, UiB), Table 6 and Fig. 21, p. 35 | Haddock, Barents Sea SE, ecosystem survey 7 Aug-3 Oct 2009 (49 stomach stations in SE) | Trawl station (haul) | Stomach fullness TFI = S/L^3 x 10^4, log | ln TFI on ln density: **-0.070 (SE 0.029, p 0.015)** by station x 5 cm group; **-0.068 (SE 0.031, p 0.026)** by station total. This is already an elasticity. Per doubling: **-4.6 to -4.7 %** (computed). R2 0.008-0.009. NW and SW areas: n.s. (-0.067, +0.079; -0.029, -0.044) | Station totals e^5 to e^12 ind/nm2 (Fig. 21 axis, natural log), about 43 to 47 000 ind/km2 (computed), a ~1100x span. The bulk sits at e^7-e^10. The top haul is **~20-30x the median** (computed; the mean is not reported). The fitted line drops TFI by ~38 % across the whole span (computed). | No: prey not in the model | [F] |
| Lindmark, Anderson, Gogina & Casini 2023, ICES JMS 80:1539 | Cod, SE Baltic, Q4 BITS 1993-2019 | Haul and ICES-rectangle median | Le Cren condition | Published version (WebFetch, first 100k of 167k characters): haul-level cod and flounder density "positive but with confidence intervals overlapping 0"; rectangle-level median density "negatively associated with condition" (overlapping 0 per the fetch tool's gloss, not a quote). "We did not detect a negative effect of cod density on cod condition". The preprint said haul-level was negative and "very minor". Covariates were standardised to SD 1; no coefficient is printed in the text. | Not given | No | [F-w] published; [F] preprint |
| Casini et al. 2016, R Soc Open Sci 3:160416 | Cod, central Baltic SD 25-28 | **Stock / SD, annual time series**, not local | Fulton K (40-49 cm) | Cod abundance was the only retained predictor in 1976-93: 36.4 % deviance explained. "cod density" (abundance / area with O2 > 1 ml/l) explained 38.9 %; the authors read this as "crowding, mediated by the shrinkage of suitable habitats". For 1994-2014, sprat and hypoxic area were retained, and crowding via compression is proposed as one pathway. Stock scale; no slope in the text. | Stock-level | No | [F] |
| Thorson et al. 2016, Proc R Soc B 283:20161853 | 92 species in six ecosystems (120 species-ecosystem combinations, l. 347) incl. NS-IBTS | Survey domain | Effective area occupied vs abundance | Mean area elasticity **delta = 0.061 (SE 0.014)**; Gadiformes **0.106 (SE 0.026)**; Pleuronectiformes **0.076 (SE 0.025)**. Implied local-density elasticity on abundance 1 - delta = **0.94 overall, 0.89 gadoids, 0.92 flatfish** (computed). | Each stock's survey time series | n/a | [F] (for-review manuscript, DTU repository) |
| Swain & Wade 1993, CJFAS 50:725 | Cod ages 3-8+, s. Gulf of St Lawrence, 1971-90 | Survey strata / regions | Stock area vs abundance | "Variation in abundance explained 63-94% of the variation in stock area". Density rose slowly in the core and faster at the margins. No elasticity in the abstract. | 1971-90 abundance range | n/a | [A] |
| Swain & Sinclair 1994, CJFAS 51:1046 | Same cod stock | Survey | D90/D95 vs D50 | The 90-95 % area expands with abundance; the 50 % (core) area "did not expand significantly". The core therefore absorbs density increases nearly proportionally. | n/a | n/a | [A] |
| Marshall & Frank 1995, CJFAS 52:1007 | Juvenile haddock age 1-2, SW Scotian Shelf | Survey strata | Local density vs total abundance; length-at-age | Density-dependent habitat selection was found, but it "did not generate correlations between mean length-at-age and total abundance-at-age". The share in high- vs low-growth areas stayed about constant. No magnitude in the abstract. | n/a | n/a | [A] |
| Blanchard et al. 2005, CJFAS 62:2001 | Juvenile cod age 1-2, North Sea, 1977-2002 | Survey | Distribution vs abundance under a temperature-suitability IFD | At low stock size, catches were confined to near-optimal temperature. At high stock size they spread into suboptimal areas. No magnitude in the abstract. | n/a | n/a | [A] |
| Hiddink, Jennings & Kaiser 2005, J Fish Biol 67(B):111 | Haddock, North Sea, 24 sites, Aug-Sep 2004 | Station / site | Liver index (HSI) | **Positive** correlation between local abundance and HSI at a given temperature: "condition ... was better at stations where haddock were most abundant". DDHS explained "only a small part" of year-to-year distribution change. Full text not accessible (Wiley 403; UEA lists it as unavailable). | 24 sites | Stable isotopes showed no trophic shift at high abundance | [A] |
| Jónsdóttir et al. 2024, ICES JMS 81:587 | Haddock, Iceland N and S | **Area means over 30 600 and 47 400 km2** (population scale) | Weight, Fulton K, liver index | Density effect negative on all three (F 21.3, 6.3, 4.4). No beta reported. "No clear decline was noted in stomach fullness during the years of highest haddock density." | Area means 0.82-8.07 t/km2 (N), 1.35-7.46 (S) | No | [F-w] (first 100k characters only) |
| Johannesen et al. 2012, MEPS 458:181 | Cod, Barents Sea 2004-09 | 35 x 35 nmi grid | Feeding success | **Own density not tested.** Feeding success rose with local capelin and polar cod. | n/a | n/a | [F] |
| Fisher & Frank 2004, MEPS 279:201 | Scotian Shelf, 34 stocks | Stock area | Abundance-distribution | Pearson r (not slopes): haddock 4VW 0.65, 4X 0.51; cod 4Vs,W 0.78, 4X 0.20 (n.s.), 4Vn 0.41 (Table 1) | 1970-2001 | n/a | [F] |
| Petitgas 1998, ICES JMS 55:443 | Age-0 hake, Bay of Biscay (a gadiform, but not a target species) | Survey | Aggregation curves | Relative density histogram time-invariant at medium and high abundance; more skewed at low abundance | n/a | n/a | [F] |
| Rueda et al. 2015, Ecosphere 6:175 | Hake, forkbeard, blue whiting, Balearic | Local and regional | Condition | Intraspecific competition "shows a very clear effect", varying with ontogeny. No magnitude (full text 403). | n/a | n/a | [A] |
| Shepherd & Litvak 2004, Fish Fish 5:141 | Review | - | - | Area-abundance correlations can arise without DDHS, and IFD can be tautological without a priori habitat suitability | - | - | [A] |
| Tamdrari et al. 2010 (cod aggregation curves) | - | - | - | **Not accessed** (budget) | - | - | - |

### What Q1 adds up to

1. **Haddock is the only direct local test I found, and it is weak.**
   - Auestad 2023 regresses haul-level stomach fullness on haul-level own density (Barents Sea, 2009).
   - The elasticity is about -0.07 in the one well-sampled area, and n.s. in the two others.
   - R2 is below 0.01. Individual stomachs are the units, which inflates the precision because they are pseudo-replicates within 49 stations.
   - It is a single year in an unpublished MSc thesis.
   - It measures stomach content, a snapshot, not intake rate.
   - It is about 6x the model's -0.012 in magnitude. Over the model's 30-90x crowding (log2 = 4.9-6.5 doublings), it would cut fullness by **21-27 % (computed)**. This is an **extrapolation beyond the data**: the top Barents haul is only ~20-30x the median and ~17 t/km2, while the model's 30-90x the mean and 50-150 t/km2 lie above the fitted range.
2. **Cod at the local scale: no clear crowding cost on condition.**
   - Lindmark 2023 found no clear effect at haul or rectangle scale in the Baltic, with all CIs overlapping 0.
   - Casini 2016 is a stock-scale time series. It says nothing about local crowding beyond "density" defined as stock size over oxygenated area.
3. **Hiddink 2005 (North Sea haddock) points the other way.**
   - Condition (HSI) is *positively* related to local abundance across 24 sites.
   - Fish sit where food is good; this is the classic confound of payoff-vs-density regressions.
   - None of the tests above separates prey depletion from interference.
4. **Basin-model inference is weak for demersal fish.**
   - Effective area grows ~1 % per 10 % abundance in gadoids, 0.6 % on average (Thorson 2016).
   - Local density therefore tracks abundance almost proportionally (elasticity ~0.9).
   - Read through an IFD, this implies the marginal crowding cost between good and poorer habitat is small at survey scales of tens of km. It says nothing about km-scale packing.
   - Swain & Sinclair 1994: the core (D50) does not expand with abundance.

## Q2: aggregation size, density and persistence

| Source | Species, context | Feeding / spawning | Density | Size and persistence | t/km2 (computed; assumed weight) | Grade |
|---|---|---|---|---|---|---|
| Auestad 2023, Fig. 21 | Haddock, Barents SE, Aug-Sep 2009, StoX swept-area by station | Feeding season | Station totals up to e^12 = 1.6e5 ind/nm2, about 47 000 ind/km2. Bulk of stations e^7-e^10 (about 320-6400 ind/km2). | Haul (a few km tow) | 3-4 yr haddock at ~0.35 kg gives **~17 t/km2 at the top haul, 0.1-2 t/km2 for the bulk** | [F] (Fig. 21, p. 35; axis read from the figure) |
| Pednekar, Jain, Godø & Makris 2024, Commun Biol 7:1386 (OAWRS) | Cod (mean 61 cm) converging on a spawning-capelin shoal, Finnmark, 27 Feb 2014 | **Feeding** aggregation, of migrating cod partly en route to spawning | Shoal defined by the critical density **0.06 cod/m2** = 60 000 cod/km2. "Total population of cod above critical population density of 0.06 fish/m2 rises to roughly 2.5 million in the imaged area." | Formed within ~1 h after dawn, propagating as a density wave. Observed over ~4 h (consumption estimate); persistence beyond that not read. Area <= 2.5e6/0.06 m2 = **<= 42 km2** (computed upper bound; the real area is in Fig. 4) | 61 cm cod at ~2.2 kg gives **>= ~130 t/km2 at the shoal boundary**. | [F] |
| Rose & Kulka 1999, CJFAS 56(S1):118 | Northern cod, Bonavista corridor, spring 1990 | **Migrant cod on a cross-shelf route** (spawning status not stated in the abstract), during stock collapse | "approximately 450 000 t was concentrated within 7000 km2 at densities quadruple those of the 1980s"; high-density areas > 0.1 fish/m2 | Held 1990-92 as abundance fell fivefold (hyperaggregation) | **64 t/km2 regional mean** (computed). 0.1 fish/m2 at ~1 kg gives **>= 100 t/km2** in the cores. | [A] |
| Makris et al. 2019 (online 2018), Fish Fish 20:201 (OAWRS) | Cod, Nordic Seas (Lofoten) | **Spawning** | Whole spawning groups imaged | "stretching for tens of kilometres". Group size log-normal, with a stable mean over ~30 yr. | Upper bound only; densities not read | [A] |
| Aglen 1999, ICES JMS 56:345 | Haddock and cod, Barents Sea, 10-day fixed station | Feeding season | - | Large haddock leave the bottom for "a few hours in the daytime". Most fish are in the acoustic dead zone. Trawl catches vary diel and with semi-diel tidal current. | - | [A] |
| Engås et al. 1988 (ICES CM 1988/B:32) | Cod and haddock, Barents Sea | Feeding | - | Haddock aggregations up to ~150 m off the bottom; daytime trawl catches larger | - | [S] |
| Kristensen et al. 2021, Ecol Evol 11:11477 (accepted manuscript, l. 263-266) | Fjord coastal cod, Skagerrak, acoustic telemetry | Resident, feeding | - | 95 % home ranges "varied from 1 to 25 hectares (mean: 7.1 ha, SD: 5.9 ha, median: 5.0 ha)". Most fish were sedentary on 2-3 receivers. | - | [F] |
| Aalvik et al. (search snippet) | Coastal cod, Norwegian Skagerrak, Jun-Aug 2012, n = 32 | Feeding | - | Monthly home range 0.25-5.20 km2 (mean 2.30 km2) | - | [S] |
| Espeland et al. 2007 | Coastal cod, Skagerrak inshore spawning ground | Spawning | - | Average home range 27 ha | - | [S] |
| Neat et al. 2014, J Appl Ecol 51:1564 | Cod around the British Isles, 252 DST tags | Year-round | - | "migration, site fidelity and limited home ranging". No km2 in the abstract. | - | [A] |

### What Q2 adds up to

**Densities of 50-150 t/km2 over a few km2 do occur; in the sources found, only in specific settings and briefly.**
- Field cod reaches ~100+ t/km2 in shoals that form for hours around a dense prey pulse (cod on spawning capelin, Pednekar 2024).
- It also reaches those levels on migration and spawning routes (Rose & Kulka 1999; Lofoten).

**Feeding-season haul data for haddock top out near ~17 t/km2.**
- In the one dataset I could read, the bulk of hauls lies at 0.1-2 t/km2.
- This is consistent with the Skagerrak GOV maximum of ~30 t/km2 given in the brief.
- Barents haddock weights are assumed (~0.35 kg at 3-4 yr). The swept-area densities depend on the StoX effective-width assumptions.

**Fjord coastal cod in the Skagerrak hold 1-25 ha home ranges, mostly sedentary on 2-3 receivers.** The duration was not read. I found no evidence of persistent km-scale packing.

**Gaps.** I found no haddock or flatfish shoal dimensions, nearest-neighbour distances, or plaice feeding-aggregation densities within budget.

## Not accessed / limits

- **Paywalled or blocked:**
  - cdnsciencepub returned 403 (Swain & Wade, Swain & Sinclair, Marshall & Frank, Blanchard, Rose & Kulka): abstracts only.
  - Wiley returned 403 (Hiddink 2005, Rueda 2015, Makris 2019): abstracts only.
  - The physics.mun.ca copy of Rose et al. 2000 timed out.
- **Thorson 2016:** species-level and North Sea ecosystem offsets are only in figures or tables that did not survive text extraction.
- **Casini 2016:** Table 1 for 1994-2014 is ambiguous after XML flattening, so only text statements are used.
- **Not searched:** Tamdrari 2010, Myers & Stokes 1989, and the Norwegian-Russian joint stomach database beyond the thesis.

## Citations

- Auestad, T.S. 2023. Diet variation in Barents Sea haddock (Melanogrammus aeglefinus). MSc thesis, University of Bergen. hdl:11250/3072358
- Lindmark, M., Anderson, S.C., Gogina, M., Casini, M. 2023. ICES J. Mar. Sci. 80:1539-1550. doi:10.1093/icesjms/fsad084. Preprint doi:10.1101/2022.04.19.488709
- Casini, M. et al. 2016. R. Soc. Open Sci. 3:160416. doi:10.1098/rsos.160416
- Thorson, J.T., Rindorf, A., Gao, J., Hanselman, D.H., Winker, H. 2016. Proc. R. Soc. B 283:20161853. doi:10.1098/rspb.2016.1853
- Swain, D.P., Wade, E. 1993. Can. J. Fish. Aquat. Sci. 50:725-733. doi:10.1139/f93-083
- Swain, D.P., Sinclair, A.F. 1994. Can. J. Fish. Aquat. Sci. 51:1046-1054. doi:10.1139/f94-104
- Marshall, C.T., Frank, K.T. 1995. Can. J. Fish. Aquat. Sci. 52:1007-1017. doi:10.1139/f95-099
- Blanchard, J.L. et al. 2005. Can. J. Fish. Aquat. Sci. 62:2001-2009. doi:10.1139/f05-109
- Hiddink, J.G., Jennings, S., Kaiser, M.J. 2005. J. Fish Biol. 67(Suppl B):111-124. doi:10.1111/j.0022-1112.2005.00912.x
- Jónsdóttir, I.G., Björnsson, B., Ragnarsson, S.Á., Elvarsson, B.Þ., Sólmundsson, J. 2024. ICES J. Mar. Sci. 81:587. doi:10.1093/icesjms/fsad108
- Johannesen, E. et al. 2012. Mar. Ecol. Prog. Ser. 458:181-197. doi:10.3354/meps09818
- Fisher, J.A.D., Frank, K.T. 2004. Mar. Ecol. Prog. Ser. 279:201-213. doi:10.3354/meps279201
- Petitgas, P. 1998. ICES J. Mar. Sci. 55:443-453. doi:10.1006/jmsc.1997.0345
- Rueda, L., Massutí, E., Álvarez-Berastegui, D., Hidalgo, M. 2015. Ecosphere 6:175. doi:10.1890/ES15-00087.1
- Shepherd, T.D., Litvak, M.K. 2004. Fish Fish. 5:141-152. doi:10.1111/j.1467-2979.2004.00143.x
- Pednekar, S., Jain, A., Godø, O.R., Makris, N.C. 2024. Commun. Biol. 7:1386. doi:10.1038/s42003-024-06952-6
- Rose, G.A., Kulka, D.W. 1999. Can. J. Fish. Aquat. Sci. 56(S1):118-127. doi:10.1139/f99-207
- Makris, N.C. et al. 2019 (online 2018). Fish Fish. 20:201-213. doi:10.1111/faf.12331
- Aglen, A. 1999. ICES J. Mar. Sci. 56:345-360. doi:10.1006/jmsc.1999.0449
- Kristensen, M.L. et al. 2021. Ecol. Evol. 11:11477-11490. doi:10.1002/ece3.7939
- Neat, F.C. et al. 2014. J. Appl. Ecol. 51:1564-1574. doi:10.1111/1365-2664.12343
