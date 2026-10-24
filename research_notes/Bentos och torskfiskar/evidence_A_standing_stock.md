# Evidence A: benthic standing stock in the deep Skagerrak (Mareld box)

Scope: macrofauna and megafauna standing stock that could stand behind `benthic_community` (K 10 t WW/km2,
start 4.44 t WW/km2). Model grid: 57.81-58.31 N, 10.07-11.15 E (`coordinates.txt`). 1 t/km2 = 1 g/m2.

Tags: [F] read in full text (including values read off a figure, marked "fig."), [A] abstract or search snippet
only, [S] secondary citation (via the source named). [D] = primary data downloaded and computed here (SHARK).
Downloaded files were treated as data only.

## Key finding: three long-term monitoring stations lie inside the grid

Josefson (1990) Table 1 [F], with the matching SHARK station names:

| Josefson stn | Position (Table 1) | Decimal | Depth | SHARK name | In grid? |
|---|---|---|---|---|---|
| 10 | 58 14.9'N, 10 34.3'E | 58.248, 10.572 | 300 m | LYSE 8 (58.248, 10.572, 300 m) | yes, inside the Mareld wind-farm polygon |
| 12 | 57 55.9'N, 11 02.2'E | 57.932, 11.037 | 100 m | SK3 (57.932, 11.041, 94-100 m) | yes, SE part |
| 8 | 58 15.2'N, 11 03.5'E | 58.253, 11.058 | 100 m | not in SHARK | yes, E edge |
| 11 | 58 32.5'N, 10 47.5'E | 58.542, 10.792 | 100 m | not in SHARK | no, ~25 km north |

So the box has primary data at 100 m and 300 m. The 300 m station is in the depth band that covers most of the grid.

## Short answers

**Q1. Macrofauna standing stock.** In-grid, the fauna is 5-60 times the model's K, and it depends strongly on depth.
- 300 m (Stn 10 / LYSE 8): 50-85 g WW/m2 over 1973-1988, read off the regression line; single grabs ran 25-140 [F, fig.]. This excludes large species (echinoids, Arctica, Nephrops, sea pens).
  - My estimate of the heart urchins at 300 m is ~15-25 g WW/m2: 2.7 Brissopsis/m2 at LYSE 8 [D] times 5-9 g per animal, the SK3 weight under the kg reading [D]. That would put the total at roughly 65-110 g WW/m2. Animals at 300 m may be smaller than at 100 m.
- 100 m (Stn 8, 11, 12): roughly 60-500 g WW/m2 without large species over 1973-1988 [F, fig.].
- 100 m, later period (SK3, 1996-2011, n = 62 grabs): mean 635 g WW/m2 in total. Of that, 387 g/m2 is not heart urchins [D].
- Trend: biomass rose by a median factor of 1.8 over 1972-1988 at the 14 Skagerrak-Kattegat stations [F]. The absolute polychaete increase grew with depth [F].
- Comparisons: Kattegat (23-65 m) 50-130 g WW/m2 without large species [F, fig.]. North Sea 7 g AFDW/m2, about 45-115 g WW/m2 with the factors in Q4 [F].
- The 71-146 g WW/m2 quoted for Skold et al. 2018 was not found as such. Its Fig. 6c gives 5.0-13.2 g per 0.1 m2, i.e. 50-132 g/m2.

**Q2. Epifauna and megafauna.**
- *Nephrops*, FU 3-4: the 2024 UWTV adjusted mean density is 0.21 burrows/m2, abundance 2975 million [F].
  - At 23-53 g per animal (the ICES mean discard and landing weights) that is 5-11 g WW/m2 over the Nephrops grounds (my calculation).
  - The 2025 abundance is 5063 million, +73 %, i.e. about 8-19 g/m2 if the survey area is the same.
- *Pandalus*, swept-area survey biomass (no catchability correction) [F; t/km2 my calculation]:
  - 2021, Skagerrak strata 11-16 (100-500 m): 4.41 kt over 15,940 km2 = 0.28 t/km2.
  - 2021, 200-300 m stratum 15: 0.48 t/km2.
  - 1997, the series maximum: 0.83 t/km2.
- Brittle stars are caught by the grabs, so they are already in Q1.
  - At SK3, Amphiura spp. average 253 g WW/m2 [D].
  - At 300 m the main brittle star is the small Amphilepis norvegica (148 ind/m2) [D].
- Crabs and other shrimps: no area-based standing stock found.

**Q3. Prey-sized fraction.** The answer differs sharply with depth.
- 100 m (SK3), share of biomass [D]: heart urchins (mostly Brissopsis lyrifera) 39 %, Amphiura spp. 40 %, polychaetes ~10 %, bivalves 7 % (mostly Abra nitida), crustaceans < 1 %.
  - Without the heart urchins, the remaining ~61 % (~390 g WW/m2) is small infauna.
  - Most of that is Amphiura, whose arms and discs are cropped by fish more than the whole animal is eaten.
- 300 m (LYSE 8), share of numbers [D]: small polychaetes 44 %, small bivalves 44 % (Parathyasira, Abra nitida, Nucula, Yoldiella), crustaceans 5 %, ophiuroids 4 %, Brissopsis 0.1 %.
  - By numbers, almost all of it is small-bodied fauna.
  - By biomass, heart urchins may still be about 15-30 % of the total, because each one is large (~15-25 g/m2 by my estimate above).
- No paper was found that gives a quantitative "fish-food" or "available benthos" fraction. Cramer & Daan (1986) state that such a fraction is rarely isolated [F].

**Q4. Conversion factors.**
- AFDW/WW (Ricciardi & Bourget 1998) [F]: polychaetes 16.0 %, bivalves (with shell) 5.8 %, ophiuroids 7.4 %, echinoids 3.5 % (median 2.7 %), decapods 16.5 %, amphipods 16.0 %.
- AFDW/WW, North Sea (Gogina et al. 2022) [F]: polychaetes 14.8 %, bivalves 8.4 %, echinoderms 7.7 %, decapods 11.9 %.
- Species values (Rumohr et al. 1987) [F]: Amphiura filiformis AFDW 13.7-17.0 % WW, A. chiajei 12.2-13.5 % WW, Echinocardium cordatum 2.9-3.1 % WW.

## Evidence list

### Q1 macrofauna

1. **Josefson 1990, MEPS 66:117-130** [F]. https://www.int-res.com/articles/meps/66/m066p117.pdf
   - Methods: "0.1 m2 Smith-McIntyre grab", "sieved on board the ship on a 1 mm screen and preserved in 4 % buffered formalin". "Biomass determinations were made after blotting dry on filter paper." "Molluscs were weighed including shells and tube-dwellers without their tubes".
   - Excluded taxa: echinoids, plus "the Ocean quahog Arctica islandica, the Norway lobster Nephrops norvegicus and pennatulid anthozoans. The category excluding these taxa is hereafter called 'biomass without large species'."
   - Usually 5 replicates per date, sampled in May.
   - Abstract: "Total biomasses showed linear increases by a median factor of 1.8, primarily from 1981 to 1988"; "Polychaetes contributed more to the increase at the deepest stations, and the absolute increase of this group was positively related with water depth."
   - Fig. 2, read off by me (g per 0.1 m2, biomass without large species, regression lines; x10 for g/m2):
     - Stn 10 (300 m): ~5 (1973) to ~8.5 (1988); single grabs 2.5-14.
     - Stn 12 (100 m): ~0-10 (mid-1970s) to ~50 (1988); single grabs 10-80.
     - Stn 8 (100 m): ~10.5 (1983) to ~19 (1988).
     - Stn 11 (100 m): ~6 (1973) to ~20 (1988).
   - Text: "biomass without large species increased with depth (p < 0.05)" (this refers to the factor of change, Table 5). "The greatest increase in relative terms occurred at the deeper stations, most of them situated offshore, in the Skagerrak."
2. **SHARK national monitoring, station SK3** (= Josefson Stn 12), KMRS datasets 1996-2011 [D].
   - Source: https://shark.smhi.se/api/dataset/download/SHARK_Zoobenthos_<year>_KMRS_version_....zip (dataset list at https://shark.smhi.se/api/options).
   - Sampling: 4 grabs/yr (2 in 2011), 0.1 m2 grab, 1000 um mesh, formalin, sampled April-June.
   - Result: mean total 635 g WW/m2. Single-grab range 285-1466, median 599. Mean without echinoids 387 g WW/m2.
   - **Unit caveat (my inference):** in 1996-2010 the "Wet weight" values are evidently stored in kg although labelled g. Totals per year are 0.41-0.90 "g/m2" in 1996-2010 and 966 g/m2 in 2011. Per-animal weights fit kg: 5 Brissopsis = 0.058 "g", i.e. 12 g each, and Amphiura filiformis ~0.05 g each. I multiplied the 1996-2010 values by 1000.
   - The derived "Wet weight/area" field carries the same scale. The SHARK metadata has no unit note.
   - The corrected values agree with Josefson Fig. 2 for the same station in 1988 (400-800 g/m2 without large species).
   - 1983-1995 records have no wet weight.
3. **SHARK station LYSE 8** (= Josefson Stn 10, 300 m), KMRS 1983-1993 [D]. Abundance only, no biomass; composition is under Q3.
4. **Skold et al. 2018, MEPS 586:41-55**, Kattegat, 58 stations at 23-65 m [F]. Accepted manuscript: https://backend.orbit.dtu.dk/ws/portalfiles/portal/190977577/Microsoft_Word_Skold_et_al_Kattegatt_Revised_171124clean.docx.pdf
   - Methods: 0.1 m2 Smith-McIntyre grab, 1.0 mm sieve, 4 % formaldehyde.
   - "All statistical analyses on biomass were tested without large bodied species, i.e. the heart urchins Brissopsis lyrifera and Echinocardium cordatum, and the ocean quahog Arctica islandica".
   - Fig. 6c, "total biomass excluding large bodied species", y axis "g per 0.1 m-2", n = 7 per bar, read by me:
     - 2009: 8.7 (closed) / 10.8 (trawled)
     - 2010: 12.5 / 13.2
     - 2011: 10.0 / 10.1
     - 2014: 5.0 / 10.9
     - i.e. 50-132 g WW/m2.
   - No mean is stated in the text.
5. **Heip et al. 1992, ICES J. Mar. Sci. 49:13-22**, North Sea synoptic survey, spring 1986 [F]. https://www.vliz.be/imisdocs/publications/ocrd/58011.pdf
   - "The average total biomass is 7 g ash-free dry weight (AFDW) m-2 (standard error = 7.6)."
   - The abstract reports a decreasing trend in biomass with latitude, higher biomass in finer sediments, and smaller mean body weight to the north.
   - WW equivalent: 7/0.16 = 44 to 7/0.06 = 117 g WW/m2, depending on the polychaete/bivalve mix (my conversion).
6. **Rosenberg et al. 1987** (J. Exp. Mar. Biol. Ecol. 105:219-251), Oslofjord and northern Skagerrak vs Petersen 1911-14 [S, via Josefson 1990]: "the difference by a factor of 1.8 over 71 yr". No absolute values seen.
7. **Pearson, Josefson & Rosenberg 1985** (J. Exp. Mar. Biol. Ecol. 92:157-206), Kattegat [S, via Josefson 1990 and Skold 2018]: "generally higher abundances and biomass" than Petersen's 1911-14 stations. Full text not obtained.
8. **Kunitzer et al. 1992**: not obtained. Rosenberg et al. 2000 (ECSS) turned out to be the Alkor Deep in the Kattegat (138 m): biomass correlated positively with depth [A]. Not used.

### Q2 epifauna and megafauna

9. **ICES 2025 advice nep.fu.3-4** [F]. https://ndownloader.figshare.com/files/59204465 (record: ices-library.figshare.com, article 27202755)
   - "Stock abundance (2026) 5063 Underwater TV (UWTV) survey 2025; individuals in millions"; "A 73% increase in stock abundance was observed in the 2025 survey."
   - Mean weight in landings 51.1-55.4 g (2017-2024); in discards 21.9-25.1 g. UWTV abundance 2917-5152 million (2017-2024).
   - The survey counts animals with carapace length > 17 mm.
10. **ICES WGNEPS 2025 report** (outputs from the 2024 meeting), Table 6.17, FU 3&4 [F]. https://ndownloader.figshare.com/files/60000956
    - "Adjusted mean density 0.21 burrows/m2"; "Adjusted abundance 2975 million, 5.42 %" CV.
    - Stratified random design with 9 strata since 2017; DK 90 and SWE 92 stations used.
    - My derivations: implied survey area = 2975e6/0.21 = ~14,200 km2, and 0.21 x 23-53 g = 4.8-11 g WW/m2.
    - These assume one animal per burrow. Burrows are on Nephrops mud, not the whole grid.
11. **SLU interim cruise report, Svea, April 2025** [F]: bias-corrected densities of 0.27, 0.32, 0.17 and 0.13 burrows/m2 at four Norwegian Skagerrak stations (58.73-58.83 N, 10.31-10.47 E), just north of the grid. https://www.fiskeridir.no/english/Coastal-management/Marine-scientific-research/cruises-2025/_/attachment/inline/c3f54d98-e12b-45af-a3fe-6803ea573b7a:081513ea7a14dee80595db87fe6cb7da3dd4a789/interim-cruise-report-swedish-nephrops-uwtv-survey-08012025.pdf
    - The 2024 report for the same stations gives 0.026-0.085 [A].
12. **Sovik & Thangstad 2021, NAFO SCR 21/001**, Norwegian shrimp survey, Skagerrak and Norwegian Deep [F]. https://www.nafo.int/Portals/0/PDFs/sc/2021/scr21-001.pdf
    - Table 1, "biomass (Ktons) ... available to the trawl", 2021: total 7.59 kt over 15,748 nm2 (= 54,000 km2, 0.14 t/km2).
    - Skagerrak strata 11-16: 0.70+0.43+0.37+0.69+1.22+1.00 = 4.41 kt over 4648 nm2 (15,940 km2), i.e. 0.28 t/km2. Stratum 15 (200-300 m): 1.22 kt / 739 nm2 = 0.48 t/km2.
    - Table 5, 1997 (series maximum, total index 32,020 t): Skagerrak strata 11-16 sum to 13,270 t = 0.83 t/km2.
    - Text: "In 2021, the biomass was lower than in 2020 (7590 t)". "The recent stock size is at a lower level compared with the stock size in the 80s, 90s and 2000s."
    - Swept area: "wingspread of 11.7 m", "0.019 nm2/hour". No catchability correction, so the values are minimum densities.

### Q3 composition and edible fraction

13. **SK3 (100 m) biomass by group**, 62 grabs, 1996-2011 [D]. g WW/m2 and share:

    | Group | g WW/m2 | Share |
    |---|---|---|
    | Ophiuroids (Amphiura spp.) | 252.6 | 39.8 % |
    | Echinoids (Brissopsis 240.4, Echinocardium 7.7) | 248.1 | 39.0 % |
    | Polychaetes | 63.9 | 10.1 % |
    | Bivalves (Abra nitida 40.2) | 46.9 | 7.4 % |
    | Priapulids, nemerteans and others | 15.3 | 2.4 % |
    | Crustaceans | 4.0 | 0.6 % |
    | Gastropods | 3.2 | 0.5 % |

    - The polychaete total includes about 1 % of other minor taxa. Its main species is Scalibregma inflatum, 43.0 g/m2.
    - Grouping is my own, by genus.
14. **LYSE 8 (300 m) abundance by group**, 51 grabs, 1983-1993, mean 3852 ind/m2 [D].
    - Polychaetes 44 %: Aphelochaeta mcintoshi 17 %, Heteromastus 12 %.
    - Bivalves 44 %: Parathyasira equalis 18 %, Abra nitida 11 %, Nucula tumidula 6 %, Yoldiella 3 %.
    - Crustaceans 5 %: Diastylis lucifera, Eriopisa.
    - Ophiuroids 4 %: Amphilepis norvegica. Sipunculans and priapulids 3 %.
    - Brissopsis lyrifera 2.7 ind/m2. Calocaris 0.4 ind/m2. Nephrops none in grabs.
    - So the community differs completely from 100 m: small deposit-feeders with almost no large echinoderms.
15. **Josefson 1990** [F]: at stations shallower than 100 m most of the total increase "was due to echinoderms". The increase in biomass without large species "was due to amphiuroids and polychaetes".
16. **Cramer & Daan 1986, ICES CM 1986/G:56**, North Sea cod and haddock [F]. https://www.vliz.be/imisdocs/publications/ocrd/274395.pdf
    - "benthos represents a highly heterogeneous aggregate of organisms with very different positions in the food chain. Fish feed only on specific components, which are rarely singled out in benthos studies."
    - "only part of the benthos production is available for such predators". They note that Rachor's production values "refer specifically to macro-endofauna species not consumed by cod and haddock".
    - Cod and haddock consumption is 0.8 (south) to 0.5 (north) g C m-2 yr-1. No fraction is given.
17. **Duineveld & Van Noort 1986**, southern North Sea [A, search snippet]: annual consumption of Amphiura filiformis arms by dab was "0.84 g wet weight m-2 ... 420 arms or 6% of the arm population". This shows that ophiuroids are partly eaten as arm tips, not whole.
18. **Skold 2018 and Josefson 1990** both exclude heart urchins and Arctica from biomass because single large animals dominate grabs. This is the same exclusion a fish-prey index needs.

### Q4 conversion factors

19. **Ricciardi & Bourget 1998, MEPS 163:245-251**, Table 1, mean % (median; 95 % CI) [F]. https://www.vliz.be/imisdocs/publications/ocrd/55005.pdf
    - AFDW/WW:
      - Polychaetes 16.0 (15.6; 15.0-17.0). Errant 17.1, sedentary 15.0.
      - Bivalves 5.8 (5.5; 5.2-6.4).
      - Prosobranchs 7.5.
      - Amphipods 16.0. Decapods 16.5 (14.6-18.4). Cumaceans 7.6.
      - Asteroids 11.2. Ophiuroids 7.4 (6.5; 4.8-10.0). Echinoids 3.5 (2.7; 1.4-5.6). Holothurians 10.9.
      - Nemerteans 20.0.
    - DW/WW: polychaetes 19.8, ophiuroids 47.1, decapods 26.2.
    - Wet weights of molluscs and echinoderms include shells.
    - Preservation lowers AFDW, about one-third in some groups (Table 3). Formalin data such as SHARK and Josefson are therefore close to, or slightly below, fresh WW.
20. **Gogina, Zettler & Zettler 2022, ESSD 14:1-4**, Table 2, North Sea [F]. https://essd.copernicus.org/articles/14/1/2022/essd-14-1-2022.pdf
    - WW to AFDW: polychaetes 0.148, bivalves 0.084, echinoderms 0.077, decapods 0.119, amphipods 0.128, nemerteans 0.158.
    - Formalin material.
21. **Rumohr, Brey & Ankar 1987**, Baltic Mar. Biol. Publ. 9 [F]. https://www.vliz.be/imisdocs/publications/ocrd/231721.pdf
    - AFDW%WW: bivalves 6.9 (excluding Arctica and Astarte), gastropods 8.5, crustaceans 14.5, polychaetes 13.2.
    - Amphiura filiformis "AFDW = 13.7 - 17.0 % WW"; A. chiajei 12.2-13.5 % WW; Echinocardium cordatum 2.9-3.1 % WW.
    - No Brissopsis value was found; the echinoid factor of ~3 % is the closest stand-in.
22. Brey's online conversion handbook (thomas-brey.de/science/virtualhandbook) could not be fetched (certificate error).

## Table of standing stocks

| Area | Depth | g WW/m2 | Original unit / basis | Tag |
|---|---|---|---|---|
| In grid, Stn 10 / LYSE 8, 1973-88 | 300 m | 50-85 on the trend line (single grabs 25-140); total incl. Brissopsis ~65-110 (my estimate) | g/0.1 m2, without large species, formalin, 1 mm | [F] fig. |
| In grid, Stn 12 / SK3, 1973-88 | 100 m | ~100-200 (1973 grabs) to ~500 (1988) on the trend line; single grabs 100-800 | g/0.1 m2, without large species | [F] fig. |
| In grid, SK3, 1996-2011 | 95-100 m | 635 total; 387 without echinoids | g WW/m2 (kg x 1000 corrected, see caveat) | [D] |
| In grid, Stn 8, 1983-88 | 100 m | ~105-190 | g/0.1 m2, without large species | [F] fig. |
| N of grid, Stn 11, 1973-88 | 100 m | ~60-200 | g/0.1 m2, without large species | [F] fig. |
| Kattegat, Skold 2018, 2009-14 | 23-65 m | 50-132 | g/0.1 m2, without heart urchins and Arctica | [F] fig. |
| North Sea, Heip 1992, 1986 | mostly < 100 m | ~45-115 (7 g AFDW) | g AFDW/m2 | [F] |
| FU 3-4 Nephrops grounds, 2024 | Nephrops mud (depth range not read) | ~5-11 | 0.21 burrows/m2 x 23-53 g | [F] + calc |
| FU 3-4 Nephrops, 2025 | same grounds | ~8-19 | 5063 M animals over ~14,200 km2 | [F] + calc |
| Skagerrak Pandalus, 2021 | 100-500 m | 0.28 (0.48 at 200-300 m) | kt over nm2, swept area | [F] + calc |
| Skagerrak Pandalus, 1997 | 100-500 m | 0.83 | t per stratum | [F] + calc |

Relation to the model, stated plainly: K = 10 g WW/m2 is below every in-grid macrofauna value. The lowest is 50-85 g
WW/m2 at 300 m, and that already excludes large species. The start value of 4.44 is an order of magnitude below. Even
the prey-sized part alone (50-85 g at 300 m; ~390 g, ~60 %, at 100 m) is 5-40 times K. Nephrops on its own grounds is
about 0.5-2 times K.

## Gaps

- No biomass at the 300 m station after 1988. SHARK has abundance only to 1993, and SK3 biomass ends in 2011.
- Josefson's 1980s high (factor 1.8) may not hold today. Coastal Skagerrak declines after 2000 were reported by
  Rosenberg & Nilsson 2005 [A]. The SK3 means for 1996-2011 are still high.
- Josefson (1981) "Persistence and structure of two deep macrobenthic communities in the Skagerrak" (100 and 300 m,
  1973-78), Josefson et al. 1993, and Rosenberg et al. 1987 and 1996 were not obtained. They likely hold the best
  absolute numbers for 300 m.
- No Norwegian-side (NIVA, MAREANO) grab biomass for the Norwegian Trench slope was found. MAREANO grab data cover the
  Norwegian and Barents Seas [A].
- No quantitative "fish-food benthos" fraction was found in the literature. The Q3 fractions are composition from
  monitoring data, not gadoid diet selectivity.
- No area-based crab or non-Pandalus shrimp densities were found. The FU 3-4 survey area was derived, not read.
- The SHARK kg/g unit fix for 1996-2010 is my inference. SMHI should confirm it before the numbers are cited
  formally.
