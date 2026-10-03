# Literature P/B (production/biomass) for pelagic planktivores and gadoids - basis for `growth_rate` of `pelagic_fish` and `gadoids`

Unit conventions used throughout: all rates are instantaneous, per year (y-1) unless stated. Per-tick values = per-year / 1460 (6-h tick).
Ecopath convention: P/B = Z = F + M2 + M0 (Allen 1971), where M2 = predation mortality explained in the model and M0 = "other"/unexplained mortality = P/B x (1 - EE).
Everything under "Cited Findings" is stated in the source. Everything I computed (F = C/B, M0, M2, aggregates, per-tick values, r = 2*Fmsy) is under "Inferences" and labelled as computed.

## Q1. P/B (and Q/B) used in Ecopath with Ecosim models (North Sea, Kattegat/Skagerrak, Baltic, other shelves)

### Takeaway
The best documented regional source is the North Sea 1991 EwE model (Mackinson & Daskalov 2007, Cefas Tech Rep 142): adult herring 0.80, sprat 2.28, mackerel 0.60, sandeel 2.28, adult cod 1.19, adult whiting 0.89, adult haddock 1.14, adult saithe 0.95 (juveniles 1.0-2.5) y-1, all set as P/B = Z from ICES/MSVPA, i.e. *exploited-state* values. Central Baltic 1974 models give lower adult clupeid P/B (0.39-0.77) and adult cod ~1.05. No Kattegat- or Skagerrak-specific P/B numbers could be retrieved (the Kattegat EwE exists, ICES 2019/2020, but its parameter table was not accessible).

### Cited Findings

**North Sea 1991 EwE (Mackinson & Daskalov 2007), ICES area IV, ~570,000 km2, base year 1991**
- Method: "Production rate (P/B) in Ecopath is assumed to be equal to total mortality Z (Allen, 1971), which can be estimated as Z = F + M2 + M0 ... For commercial species assessed by the ICES working groups, mortality estimates (Z, F, and M) were compiled from stock assessment reports." For non-assessed species M from Pauly (1980) M = K^0.65 L_inf^-0.279 Tc^0.463 with Tc = 10 C; F = C/B; group P/B = biomass-weighted mean of species P/Bs. Biomasses of cod, haddock, whiting, saithe, Norway pout, herring, sprat, sandeel from MSVPA (ICES 2002b); Q/B from MSVPA rations — [Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142, sections 13.2-13.4](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Table 3.1 (data-derived best estimates) and Table 3.3 (balanced model; values in brackets are Table 3.1 inputs where changed). B in t km-2, P/B and Q/B in y-1 — [Mackinson & Daskalov 2007, Tables 3.1 and 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf):

| Group (stage definition) | B (t km-2) | P/B (y-1) | Q/B (y-1) | EE (balanced) |
|---|---|---|---|---|
| Herring juvenile (age 0-1) | 0.63 | 1.31 | 5.63 | 0.672 |
| Herring adult | 1.966 | 0.80 | 4.34 | 0.691 |
| Sprat | 0.579 | 2.28 | 6.0 (5.28) | 0.806 |
| Mackerel | 1.72 | 0.60 | 1.73 | 0.632 |
| Horse mackerel | 0.579 | 1.2 (1.64) | 3.51 | 0.356 |
| Sandeels | 3.122 | 2.28 | 10.1 (5.24) | 0.785 |
| Norway pout | 1.394 | 2.2 (3.05) | 5.05 | 0.751 |
| Cod juvenile (0-2 y, <40 cm) | 0.079 | 1.79 | 5.96 (4.89) | 0.936 |
| Cod adult (3+, >40 cm) | 0.161 | 1.19 | 3.5 (2.17) | 0.750 |
| Whiting juvenile (0-1 y, <20 cm) | 0.222 | 2.36 | 6.58 | 0.860 |
| Whiting adult (2+) | 0.352 | 0.89 | 5.46 | 0.932 |
| Haddock juvenile (0-1 y, <20 cm) | 0.284 | 2.0 (2.54) | 5.39 (4.16) | 0.453 |
| Haddock adult (2+) | 0.104 | 1.14 | 4.4 (2.35) | 0.972 |
| Saithe juvenile (0-3 y, <40 cm) | 0.281 | 1.0 | 4.94 | 0.315 |
| Saithe adult (4+) | 0.22 (0.191) | 0.95 (0.883) | 3.6 | 0.621 |
| Blue whiting | 0.08 (0.042) | 2.5 | 9.06 | 0.848 |

- Mackerel: "P/B assumed as equal to Z is 0.793 for the North Sea mackerel (ICES 1997) and 0.38 for the Western mackerel (ICES 2002a)"; the two were aggregated to one group, B = 980.4 kt (North Sea component 57 kt from MSVPA, Western component 923.4 kt in the North Sea in 1991), P/B = 0.6 — [Mackinson & Daskalov 2007, section 13.28](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Sandeel "biomass and P/B = Z were based on MSVPA results (ICES 2002b)"; sprat biomass 330 kt — [Mackinson & Daskalov 2007, sections 13.27, 13.30](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Exploitation dominates adult gadoid P/B: for adult saithe and monkfish "they have high fishing mortality and very little other mortality (ie P/B nearly equals F)"; Ecosim no-fishing runs gave "unrealistic" recovery for adult saithe, cod, monkfish, other large gadoids and catfish, so "the best solution was to reduce the proportion of Z (P/B) accounted for by F. This was achieved by increasing the biomass of the groups in Ecopath" — [Mackinson & Daskalov 2007, section 4.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- 1991 catches (landings / discards, tonnes): cod adult 67,431 / 3,366; juvenile cod discards 29,658; whiting adult 84,018 / 19,416, juvenile discards 13,886; haddock adult 50,046 / 4,243, juvenile discards 35,992; saithe adult 66,861 / 0, juvenile discards 31,504; herring adult 487,920 / 1,854, juvenile discards 86,214; sprat 99,579 / 4,218; mackerel 197,163 / 117,941; sandeel 842,574 / 0 — [Mackinson & Daskalov 2007, Table 3.5](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

**Central Baltic (Baltic Proper) EwE, base year 1974**
- Harvey et al. 2003 (ICES SD 25-29+32 for cod/herring, 25-32 sprat; 1974 base, Ecosim 1974-2000): "Values of B, Q/B, and total mortality (=P/B) for sprat, herring, and cod were averaged from quarterly MSVPA estimates for 1974." Juvenile sprat B 2.86, P/B 0.61, Q/B 21.29, EE 0.16; adult sprat B 4.86, P/B 0.64, Q/B 10.13, EE 0.51; juvenile herring B 4.72, P/B 0.45, Q/B 14.71, EE 0.60; adult herring B 6.63, P/B 0.39, Q/B 7.96, EE 0.25; juvenile cod B 1.40, P/B 0.45, Q/B 2.71, EE 0.63; adult cod B 0.73, P/B 1.06, Q/B 2.00, EE 0.94 — [Harvey et al. 2003, ICES J. Mar. Sci. 60:939-950, Table 1](https://academic.oup.com/icesjms/article/60/5/939/769206)
- Tomczak et al. 2013 (Central Baltic SD 25-29 excl. Gulf of Riga, 21 groups, 1974-2006), Table S1 basic input (B t km-2; P/B, Q/B y-1; catch t km-2 y-1): JuvSprat B 1.251, P/B 2.0, Q/B 13.693, EE 0.309, catch 0.072; AdSprat 4.213, 0.77, 6.1, 0.49, catch 0.819; JuvHerring 4.049, 2.2, 5.159, 0.2, catch 0.53; AdHerring 5.619, 0.42, 2.0, 0.619, catch 1.01; JuvCod 0.0947, 1.24, 12.987, 0.058, catch 0; Small cod 0.531, 0.6, 5.968, 0.676, catch 0.204; AdCod 0.49, 1.04, 3.96, 0.733, catch 0.373 — [Tomczak et al. 2013, PLoS ONE 8:e75439, File S1 Table S1](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0075439.s001); [main article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075439)
- Conflict: Harvey 2003 and Tomczak 2013 (same system, same 1974 base) agree for adults (herring 0.39 vs 0.42; cod 1.06 vs 1.04) but disagree 4-5x for juveniles (herring 0.45 vs 2.2; cod 0.45 vs 1.24; sprat 0.61 vs 2.0). Stage definitions probably differ (Tomczak has three cod stanzas); Harvey's juvenile P/B < adult P/B is atypical. Do not average — [Harvey et al. 2003](https://academic.oup.com/icesjms/article/60/5/939/769206); [Tomczak et al. 2013 S1](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0075439.s001)

**Kattegat / Western Baltic**
- A Kattegat EwE model exists (ICES 2019, used by WGINOSE): 39 nodes, 257 links, 29 biota groups including 11 fish species (three split into adults/juveniles, e.g. cod), eight fishing fleets, Ecopath base year 1982, Ecosim calibrated 1982-2008; documented in the ICES WGINOSE report (ICES 2020). The article does not reproduce P/B or Q/B — [ICES J. Mar. Sci. 80(1):218 (2023), "Testing management scenarios for the North Sea ecosystem using qualitative and quantitative models"](https://academic.oup.com/icesjms/article/80/1/218/6965515)
- Western Baltic Sea EwE (Scotti et al. 2022) exists; P/B per group is only in Supplementary Table S6 (not retrieved) — [Scotti et al. 2022, Front. Mar. Sci. 9:879998](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.879998/full)

### Inferences
- Area check (computed): juvenile whiting 0.222 t km-2 x 570,000 km2 = 126,540 t, matching the text's 126.5 kt, so B x 570,000 converts Table 3.1 densities to tonnes. (The text's "10.8 thousand tonnes" for adult saithe is inconsistent with 0.19 t km-2 -> ~108 kt; probably a typo.)
- Mortality decomposition, North Sea 1991, **computed by me** from Tables 3.3 and 3.5 of Mackinson & Daskalov (2007): F = (landings + discards)/(B x 570,000); M0 = P/B x (1 - EE); M2 = P/B - F - M0 (predation incl. cannibalism); M = P/B - F. Per tick = per year / 1460.

| Group | P/B | F | M2 | M0 | M = P/B-F | P/B per tick | M per tick |
|---|---|---|---|---|---|---|---|
| Herring juv | 1.31 | 0.24 | 0.64 | 0.43 | 1.07 | 0.000897 | 0.000733 |
| Herring adult | 0.80 | 0.44 | 0.12 | 0.25 | 0.36 | 0.000548 | 0.000249 |
| Sprat | 2.28 | 0.31 | 1.52 | 0.44 | 1.97 | 0.001562 | 0.001346 |
| Mackerel | 0.60 | 0.32 | 0.06 | 0.22 | 0.28 | 0.000411 | 0.000191 |
| Sandeel | 2.28 | 0.47 | 1.32 | 0.49 | 1.81 | 0.001562 | 0.001237 |
| Horse mackerel | 1.20 | 0.30 | 0.13 | 0.77 | 0.90 | 0.000822 | 0.000618 |
| Cod juv | 1.79 | 0.66 | 1.02 | 0.11 | 1.13 | 0.001226 | 0.000775 |
| Cod adult | 1.19 | 0.77 | 0.12 | 0.30 | 0.42 | 0.000815 | 0.000287 |
| Whiting juv | 2.36 | 0.11 | 1.92 | 0.33 | 2.25 | 0.001616 | 0.001541 |
| Whiting adult | 0.89 | 0.52 | 0.31 | 0.06 | 0.37 | 0.000610 | 0.000256 |
| Haddock juv | 2.00 | 0.22 | 0.68 | 1.09 | 1.78 | 0.001370 | 0.001218 |
| Haddock adult | 1.14 | 0.92 | 0.19 | 0.03 | 0.22 | 0.000781 | 0.000154 |
| Saithe juv | 1.00 | 0.20 | 0.12 | 0.69 | 0.80 | 0.000685 | 0.000550 |
| Saithe adult | 0.95 | 0.53 | 0.06 | 0.36 | 0.42 | 0.000651 | 0.000285 |

- Central Baltic 1974, **computed by me** from Tomczak et al. 2013 Table S1 (F = catch/B): adult sprat F 0.19, M0 0.39, M2 0.18; adult herring F 0.18, M0 0.16, M2 0.08; small cod F 0.38; adult cod F 0.76, M0 0.28, M2 ~0. Biomass-weighted clupeids (both stages, herring+sprat) P/B 1.12, F 0.16 -> M 0.96; cod (all stanzas) P/B 0.85, F 0.52 -> M 0.33. From Harvey 2003, biomass-weighted clupeid P/B 0.50 and cod 0.66.
- Consistent pattern in all three models: juvenile P/B (1.0-2.4) is ~1.5-3x adult P/B (0.4-1.2), and adult gadoid P/B is dominated by F (F/Z 0.55-0.8 in the North Sea 1991).

### Gaps
- Kattegat EwE (ICES 2019 / WGINOSE 2020) parameter table: the figshare report page returned HTTP 403; P/B values for the Kattegat could not be verified. Citation chain: ICES JMS 80:218 (2023) -> ICES 2020 WGINOSE report -> 1982 base year, 29 biota groups.
- Western Baltic EwE (Scotti et al. 2022) Table S6 not retrieved.
- Christensen (1995) North Sea 1981 model and Lynam & Mackinson (2015)/ICES WGSAM North Sea EwE key run (updated 1991 parameters) were not fetched; no Skagerrak-only Ecopath model was found.
- English Channel / Celtic Sea comparison: the Bay of Biscay/Celtic Sea EwE paper (Aquatic Living Resources 2017) returned HTTP 403; no values obtained.

## Q2. Stock-assessment Z, F, M (M1/M2) for North Sea / Skagerrak-Kattegat stocks

### Takeaway
ICES multispecies SMS key runs split M into M1 (residual) and M2 (predation) by species and age for cod, whiting, haddock, saithe, mackerel, herring, sandeels, Norway pout, sprat, but the 2023 key-run values are published only as figures; I could not extract numeric M1/M2. The usable numeric composition comes from the Ecopath tables in Q1 (which are MSVPA-based) plus the Myers 1997 / single-species convention M = 0.2 for adult cod.

### Cited Findings
- The SMS North Sea key run 2023 reports, per species and age, "Residual mortality (M1) at age" (cod, whiting, haddock, saithe, mackerel, herring, N. and S. sandeel, Norway pout, sprat, plaice, sole), "Annual predation mortality (sum of quarterly M2)", and Z partitioned into F, M2 and M1; all as figures — [ICES WGSAM, SMS North Sea key run 2023](https://ices-eg.github.io/wg_WGSAM/NS_2023_key_run.html)
- SMS models cod, haddock, saithe, whiting, hake, plaice, sole, herring, sprat, sandeel, Norway pout, mackerel, horse mackerel, seals, harbour porpoise, grey gurnard, starry ray and 8 bird species as predators/prey; M values from SMS key runs are used in the single-species assessments of North Sea cod, haddock, whiting and herring — [search summary of WGSAM sources; ICES WGSAM key run page](https://ices-eg.github.io/wg_WGSAM/NS_2023_key_run.html)
- Baltic: WGSAM recommended natural mortality from the Baltic SMS key run for Baltic herring and sprat assessments; results are sensitive to "consumption rates, assumptions regarding the residual mortality M1 as well as the size selectivity of cod" and depend largely on the Eastern Baltic cod assessment — [ICES WGSAM 2021, ICES Sci. Rep. 3:115](https://archimer.ifremer.fr/doc/00741/85322/90345.pdf)
- North Sea 1991 Z used as P/B in EwE: mackerel Z = 0.793 (North Sea component, ICES 1997) and 0.38 (Western component, ICES 2002a) — [Mackinson & Daskalov 2007, 13.28](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Cod convention: "the value of M = 0.2 was accepted by the researchers studying each population. This natural mortality corresponds to adult survival of ps = e^-M = 0.8" (20 North Atlantic cod stocks incl. Kattegat, Skagerrak, North Sea) — [Myers, Mertz & Fowlow 1997, Fish. Bull. 95:762-772](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1997/954/myers.pdf); M = 0.2 is "the value of M assumed invariant in most cod stock assessments" — [Björnsson et al. 2022, ICES J. Mar. Sci. 79:1569](https://academic.oup.com/icesjms/article/79/5/1569/6590896)
- Kattegat cod: the long-term decrease and poor state is "likely caused by high total mortality rates and stock-size dependent effects of climate"; ICES advised zero catch for 2024 (and a total stop already in 2002) — [ICES advice cod.27.21, figshare](https://ices-library.figshare.com/articles/report/Cod_Gadus_morhua_in_Subdivision_21_Kattegat_/21820488) (via search summary; numeric Z not retrieved)
- Western Baltic spring-spawning herring (her.27.20-24, Skagerrak, Kattegat, western Baltic): ICES advises zero catch for 2026; stock outside safe biological limits — [ICES advice her.27.20-24](https://ices-library.figshare.com/articles/report/Herring_i_Clupea_harengus_i_in_subdivisions_20_24_spring_spawners_Skagerrak_Kattegat_and_western_Baltic_/27202614) (via search summary)
- Fmsy estimates (ICES F-currency, ensemble) — her.27.20-24 final 0.30; her.27.3a47d 0.38; cod.27.47d20 0.71; cod.27.22-24 0.51; had.27.46a20 0.46; pok.27.3a46 0.38; mac.27.nea 0.39; spr.27.22-32 0.39 — [Sparholt et al. 2021, ICES J. Mar. Sci. 78, Table 1 col. j](https://nwwac.org/wp-content/uploads/2026/02/sparholt-et-al-2021-fsaa175-vol-78.pdf)

### Inferences
- In the North Sea 1991 state (computed in Q1), natural mortality M = M2 + M0 is: adult herring 0.36, sprat 1.97, mackerel 0.28, sandeel 1.81, adult cod 0.42, adult whiting 0.37, adult haddock 0.22, adult saithe 0.42; juveniles 0.8-2.25. These are the parts of P/B that would remain without fishing (before any predator/prey feedback).
- The Ecopath "other mortality" M0, the closest analogue of the Mareld model's residual `natural_mortality` (M1) - plus whatever predation comes from predators Mareld does not model - is 0.25 (adult herring), 0.44 (sprat), 0.22 (mackerel), 0.30 (adult cod), 0.06 (adult whiting), 0.36 (adult saithe). Biomass-weighted M0: herring+sprat 0.32; cod+whiting+saithe 0.32; cod+whiting+haddock+saithe 0.43. All are above the current Mareld M1 values (pelagic 0.146, gadoids 0.102). This is a flag, not a prescription: Ecopath M0 absorbs balancing error.
- Mackerel in the North Sea P/B 0.6 is a blend dominated by the Western component (Z 0.38, ~94% of biomass); the resident North Sea component had Z 0.79. For a Skagerrak pelagic pool dominated by herring/sprat, mackerel should be weighted by its (seasonal) local biomass, not by North Sea biomass.

### Gaps
- Numeric SMS M1 and M2 at age (North Sea key run 2023 and earlier) could not be extracted - values are only in figures on the key run page; I do not report typical values from memory.
- ICES stock-annex M-at-age and recent F/Z for cod.27.21 (Kattegat), her.27.20-24 (WBSS), spr.27.3a4 (sprat Skagerrak-Kattegat-North Sea), cod.27.47d20, whg.27.47d, had.27.46a20, pok.27.3a46 were not retrieved (ICES figshare pages return 403 to the fetch tool).

## Q3. Production in an unfished or lightly fished state

### Takeaway
At steady state P/B = Z (Allen 1971); with F = 0, P/B -> M (= M1 + M2), and the age structure shifts toward older, slower-turnover fish, so unfished P/B is lower than the exploited Ecopath values on two counts. Size-spectrum theory for the North Sea predicts the unfished community turns over about half as fast as the fished one.

### Cited Findings
- P/B = Z (Allen 1971) is the basis of Ecopath P/B; Z = F + M2 + M0 — [Mackinson & Daskalov 2007, 13.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- When F dominates Z ("P/B nearly equals F"), removing fishing in Ecosim produced unrealistic biomass increases, demonstrating that exploited-state P/B overstates unfished production for adult gadoids — [Mackinson & Daskalov 2007, 4.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Jennings & Blanchard (2004), North Sea, size-spectrum prediction of the unexploited community: biomass of fishes 4-16 kg and 16-66 kg was 97.4% and 99.2% lower than without fishing; total fish biomass (64 g-66 kg) 38% lower; "mean turnover time was almost twice as fast" in the fished community, and 70% less primary production was required to sustain it — [Jennings & Blanchard 2004, J. Anim. Ecol. 73:632](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.0021-8790.2004.00839.x) (numbers from the abstract via search result; the Wiley page returned 403, full text not verified)
- Surplus-production r "summarizes natural mortality such as caused by predation by other species, somatic growth such as modulated by available food sources, and recruitment" — [Froese et al. 2018, Marine Policy 93:159-170](https://donnadim.com/wp-content/uploads/2021/12/6_Froese_et_al_2018_rebuilding_plus_suppl.pdf)

### Inferences
- Unfished first-order estimate = M from Q1/Q2 (computed): North Sea 1991 herring+sprat (both herring stages) M 0.80 y-1 (0.000545 per tick); herring+sprat+mackerel 0.61 (0.000420); herring+sprat+mackerel+sandeel 1.08 (0.000739); cod+whiting+saithe (all stages) 0.84 (0.000576); cod+whiting+haddock+saithe 0.96 (0.000657); adult gadoids only 0.375 (0.000257). Central Baltic 1974 clupeids 0.96, cod 0.33.
- Second-order correction: without fishing, the biomass share of juveniles falls (adults accumulate), so the biomass-weighted M moves toward the adult values (herring 0.36, cod 0.42, whiting 0.37, saithe 0.42). Jennings & Blanchard's "turnover almost twice as fast" under fishing points the same way: unfished community P/B roughly half the exploited P/B. Exploited aggregate P/B (herring+sprat 1.17; cod+whiting+saithe 1.26) halved gives ~0.6, between the "M of all stages" and "M of adults" estimates.
- Counter-effect: an unfished system has more piscivores (cod, seals, porpoises), so M2 on herring/sprat and juvenile gadoids would rise. In Mareld this is endogenous (explicit predation), so only M1 plus predation by non-modelled predators belongs in the exogenous budget.

### Gaps
- No Skagerrak/Kattegat-specific unfished production estimate was found. Allen (1971) itself was not fetched (cited via Mackinson & Daskalov).

## Q4. Maximum intrinsic population growth rate r_max (upper bound on net growth)

### Takeaway
Cod r_m from spawner-recruit slopes: Kattegat 0.53, Skagerrak 0.82, North Sea 0.56 y-1 (Myers et al. 1997). CMSY (Schaefer) r for the relevant herring stocks is ~0.5-0.6 y-1, mackerel ~0.4, North Sea cod ~0.6, North Sea haddock ~0.4, North Sea/Skagerrak saithe ~0.7. The current Mareld ceiling for net growth at low density is ~2-3x these values.

### Cited Findings
- Myers, Mertz & Fowlow (1997): r_m from Ricker spawner-recruit slope at origin (alpha~, replacements per spawner) and age at maturity a via the Euler-Lotka equation, assuming M = 0.2 (adult survival 0.8); r_m depends mainly on a, which depends on temperature. Table 1 (r_m y-1, alpha~, a, bottom temp C): Kattegat (south IIIa) 0.53, 3.8, 3, 6.5; Skagerrak (north IIIa) 0.82, 11.2, 3, 6.5; North Sea (IV) 0.56, 9.0, 4, 8.6; S.E. Baltic (22-24) 0.74, 8.4, 3, 7.0; Central Baltic (25-32) 0.53, 3.1, 3, 5.0; Celtic Sea 0.62; Irish Sea 1.03; West of Scotland 0.80; Iceland 0.24; Barents Sea 0.26; N.E. Newfoundland 0.17 — [Myers et al. 1997, Fish. Bull. 95:762-772](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1997/954/myers.pdf)
- Myers, Bowen & Barrowman (1999), >700 spawner-recruit series: maximum annual reproductive rate at low abundance is relatively constant within species and varies little among species; it sets the upper limit to sustainable F — [Myers et al. 1999, Can. J. Fish. Aquat. Sci. 56:2404-2419](https://cdnsciencepub.com/doi/10.1139/f99-201) (abstract only; species alphas not extracted)
- CMSY: r ~ 2 Fmsy ~ 2 M ~ 3 K ~ 3/t_gen ~ 9/t_max (eq. 12); FishBase resilience -> prior r: High 0.6-1.5, Medium 0.2-0.8, Low 0.05-0.5, Very low 0.015-0.1; North Sea herring (her-47d3) resilience "medium" — [Froese et al. 2017, Fish and Fisheries, CMSY paper, Table 2](https://www.fishbase.de/rfroese/CMSY_faf_12190_Rev.pdf)
- Froese et al. (2018) CMSY Fmsy by stock (Sparholt et al. 2021 Table 1, column a): her.27.3a47d 0.26; her.27.20-24 0.31; spr.27.22-32 0.26; mac.27.nea 0.21; cod.27.47d20 0.31; cod.27.22-24 0.26; had.27.46a20 0.19; pok.27.3a46 0.36 — [Sparholt et al. 2021, ICES J. Mar. Sci. 78](https://nwwac.org/wp-content/uploads/2026/02/sparholt-et-al-2021-fsaa175-vol-78.pdf)
- Atlantic cod r bounds: lower 0.095 y-1 (Hutchings & Rangeley 2011), upper 0.3 y-1 (Hutchings 1999) — [ResearchGate figure caption, via search snippet only, not verified](https://www.researchgate.net/figure/Atlantic-cod-Gadus-morhua-a-and-b-population-growth-rate-r-and-c-and-d-the_fig1_253787472)
- Neubauer et al. report reduced resilience of stocks after collapse or prolonged overexploitation — [Froese et al. 2018](https://donnadim.com/wp-content/uploads/2021/12/6_Froese_et_al_2018_rebuilding_plus_suppl.pdf)

### Inferences
- r = 2 x Fmsy holds because CMSY uses a Schaefer model (Bmsy = k/2, MSY = rk/4, Fmsy = r/2). Computed r: herring her.27.3a47d 0.52, WBSS herring her.27.20-24 0.62, Baltic sprat 0.52, NEA mackerel 0.42, North Sea cod 0.62, Western Baltic cod 0.52, North Sea haddock 0.38, saithe 3a46 0.72 (y-1). Per tick: 0.00029-0.00049.
- Myers r_m is in numbers, CMSY r in biomass; both are net rates at low density with all natural mortality (incl. predation) included. Cod: Myers 0.53-0.82 vs CMSY 0.52-0.62 vs Hutchings 0.095-0.3 - a factor ~2-8 spread.
- Current Mareld parameters (computed): at gate saturation (s-u = 0.3) gross P/B = g x 1460 x 0.3 = 1.75 (pelagic) and 1.10 (gadoids); net at low density without predators = 1.75 - 0.146 = 1.6 and 1.10 - 0.102 = 1.0 y-1. That is ~2.5-3x the literature r for herring/sprat (0.5-0.6) and ~1.2-2x for cod (0.5-0.8). Whether this matters depends on how often the gate saturates at low density and how much explicit predation remains then.

### Gaps
- No r_max found for whiting (whg.27.47d absent from Sparholt et al. 2021 Table 1) or for Skagerrak-Kattegat sprat (spr.27.3a4); Myers et al. (1999) clupeid/gadid alpha values not extracted; FishBase per-species r/resilience pages not fetched.

## Q5. Aggregating to a group and mapping to Mareld `growth_rate`

### Takeaway
Biomass-weighted North Sea 1991 aggregates: pelagic (herring + sprat) P/B 1.17, M 0.80; with mackerel 0.97 / 0.61; gadoids (cod + whiting + saithe) P/B 1.26, M 0.84; with haddock 1.38 / 0.96. For an unfished Mareld world the gross-production target is M1 + M2 (modelled predators), roughly 0.4-0.8 y-1 for both groups, and g must be derived from the *mean* gate value, not the 0.3 ceiling.

### Cited Findings
- Ecopath group P/B for multi-species groups "is estimated as a weighted mean (weighted by each species biomass B) of the species P/Bs" — [Mackinson & Daskalov 2007, 13.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Biomasses (t km-2, North Sea 1991): herring 0.63 + 1.966, sprat 0.579, mackerel 1.72, sandeel 3.122, cod 0.079 + 0.161, whiting 0.222 + 0.352, haddock 0.284 + 0.104, saithe 0.281 + 0.22 — [Mackinson & Daskalov 2007, Table 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

### Inferences
- Biomass-weighted aggregates, **computed by me** from Mackinson & Daskalov (2007) Tables 3.3/3.5 (per tick = /1460):

| Aggregate | B (t km-2) | P/B (=Z) | F | M2 | M0 | M = P/B-F | P/B per tick | M per tick |
|---|---|---|---|---|---|---|---|---|
| Herring (juv+ad) + sprat | 3.175 | 1.17 | 0.38 | 0.48 | 0.32 | 0.80 | 0.000802 | 0.000545 |
| + mackerel | 4.895 | 0.97 | 0.36 | 0.33 | 0.29 | 0.61 | 0.000665 | 0.000420 |
| + mackerel + sandeel | 8.017 | 1.48 | 0.40 | 0.71 | 0.37 | 1.08 | 0.001014 | 0.000739 |
| Adult herring + sprat + mackerel | 4.265 | 0.92 | 0.37 | 0.28 | 0.26 | 0.55 | 0.000630 | 0.000374 |
| Cod + whiting + saithe (all stages) | 1.315 | 1.26 | 0.42 | 0.52 | 0.32 | 0.84 | 0.000865 | 0.000576 |
| Cod + whiting + haddock + saithe | 1.703 | 1.38 | 0.42 | 0.53 | 0.43 | 0.96 | 0.000944 | 0.000657 |
| Adult cod + whiting + haddock + saithe | 0.837 | 1.00 | 0.62 | 0.19 | 0.18 | 0.375 | 0.000681 | 0.000257 |
| Central Baltic 1974 clupeids (Tomczak) | 15.13 | 1.12 | 0.16 | - | - | 0.96 | 0.000770 | 0.000660 |
| Central Baltic 1974 cod (Tomczak) | 1.12 | 0.85 | 0.52 | - | - | 0.33 | 0.000581 | 0.000226 |

- Mapping to the model (computed): Mareld growth is dB = B g (s-u) per tick, so realised gross P/B = g x 1460 x E[s-u]. Thus g = (target P/B) / (1460 x E[s-u]). Ceiling case E[s-u] = 0.3: g = P/B / 438. Examples: target 0.80 (herring+sprat M) -> g = 0.0018; 0.61 -> 0.0014; 0.84 (cod+whiting+saithe M) -> 0.0019; 0.375 (adult gadoids M) -> 0.00086. If the mean gate is 0.15, double these. The current g (0.004 pelagic, 0.0025 gadoids) gives ceilings 1.75 and 1.10 y-1. The pelagic ceiling exceeds the *exploited* North Sea 1991 aggregate Z (1.17) by ~50% and is ~2.2x the all-stage unfished M (0.80). The gadoid ceiling is slightly below the exploited Z (1.26) but 1.1-1.3x the all-stage unfished M (0.84-0.96) and ~3x the adult-only M (0.375).
- What belongs in the target (computed/argued): in an unfished Mareld world, steady-state gross P/B = M1 + M2 by Mareld's explicit predators (gadoids, porpoises, seals, seabirds). Predation by groups Mareld lacks (mackerel/horse mackerel on sprat and juveniles, hake, cephalopods, cannibalism within a pool) is not explicit and therefore has to sit in M1 if it is to exist. The Ecopath M0 alone (herring+sprat 0.32, cod+whiting+saithe 0.32) already exceeds current M1 (0.146 / 0.102).
- Community weighting for Skagerrak/Kattegat (inference): herring and sprat dominate; mackerel is a seasonal visitor. "Herring + sprat" (M 0.80; adults-weighted ~0.4-0.6) is the most defensible pelagic basis; adding sandeel raises it (sandeel M 1.81). For gadoids, cod + whiting + saithe (M 0.84, adult-dominated ~0.4) fits; adding haddock raises juvenile weight. Since the Mareld pool has no age structure, the share of juvenile biomass it implicitly represents decides where in the 0.4-0.85 y-1 range the target lies.
- Consistency check against r_max (computed): with target gross P/B ~0.6-0.8 and M1 ~0.15-0.3, the low-density net ceiling would be ~0.3-0.65 y-1, inside the literature r range (herring 0.5-0.6, cod 0.5-0.8 Myers / 0.5-0.6 CMSY). The current parameters give 1.0-1.6 y-1.

### Gaps
- Skagerrak/Kattegat-specific biomass weights (herring vs sprat vs mackerel; cod vs whiting vs saithe) for the Mareld area were not researched here; the aggregates above use North Sea 1991 and Central Baltic 1974 biomasses.
- The realised mean of (s-u) in Mareld runs is a model quantity, not a literature value, and must be measured to convert a target P/B into g.
