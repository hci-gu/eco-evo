# Energy density of gadoids: lean body, liver reserve and whole body

Purpose: decide whether the pooled `gadoids` group should get the same body/reserve split as herring
(`energy_content` = lean body, `max_energy_reserve` = the liver lipid store, predators eat the reserve
with the fish via `prey_includes_reserve` at a `reserve_reference_fill`), and with which numbers.

Current library values (read, not edited): `energy_content` 5500 MJ/t (read as whole body),
`max_energy_reserve` 2500 MJ/t, `maintenance_level` 0.5, `resting_metabolism` 6.0 MJ/t/tick (6 h tick).

Units: 1 kJ/g wet = 1 MJ/kg = 1000 MJ/t. HSI = liver mass as % of body mass. The denominator differs between
sources, and each value below says which one it uses.

Source tags: **[F]** full text read; **[A]** abstract only; **[2nd]** second-hand via a citing paper;
**[D]** own data analysis; **[L]** laboratory study; **[S]** own synthesis; **[Sn]** search snippet /
database summary only, not verified. "(WebFetch extraction)" means the full article page was read through
a summarising fetch tool rather than as raw text, so the quotes are close to verbatim but not guaranteed.

Two literature subagents worked on this note: one on cod liver and starvation, one on haddock, whiting,
saithe and hake. Their tags are carried through unchanged. I spot-checked the key Dutil et al. 1995 quotes
against the pdftotext output myself.

---

## Sources

| # | Reference | Tag | What it gives |
|---|---|---|---|
| 1 | Dutil, J.-D., Lambert, Y., Chouinard, G.A., Frechet, A. 1995. Fish condition: what should we measure in cod (*Gadus morhua*)? DFO Atl. Fish. Res. Doc. 95/11. https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/183365.pdf | [F][L] (plus field data) | Cod muscle and liver composition; liver 5-30 kJ/g and muscle 1.5-5 kJ/g against water content; fed vs starved-to-death K, HSI and tissue water; condition criteria. K and HSI are on SOMATIC weight. Gulf of St Lawrence. |
| 2 | Dutil, J.-D., Lambert, Y., Chabot, D. 2000. Estimating natural mortality of wild cod from controlled feeding and starvation experiments conducted in the laboratory. ICES CM 2000/R:01. https://www.vliz.be/imisdocs/publications/ocrd/276040.pdf | [F][L] | 84-day fast at 2, 6 and 10 C; linear weight loss; K-loss equations; wild vs lab decline in winter and spring. |
| 3 | Lambert, Y., Dutil, J.-D. 1997b. Condition and energy reserves of Atlantic cod during the collapse of the northern Gulf of St. Lawrence stock. CJFAS 54: 2388-2400. doi:10.1139/f97-145 | [A] | Seasonal HSI: autumn maximum 4.3-5.7 %, May minimum 2.2-3.0 %. |
| 4 | Dutil, J.-D., Lambert, Y. 2000. Natural mortality from poor condition in Atlantic cod. CJFAS 57: 826-836. doi:10.1139/f00-023 | [A] | Qualitative only: K and muscle index separate live from dead fish. |
| 5 | Black, D., Love, R.M. 1986. The sequential mobilisation and restoration of energy reserves in tissues of Atlantic cod during starvation and refeeding. J. Comp. Physiol. B 156: 469-479 | [2nd] via #1, [L] | Starved cod: K 0.59-0.66, HSI 0.8 %, muscle water 88 %, liver water 75 %. |
| 6 | Receveur, A. et al. 2022. Western Baltic cod in distress: decline in energy reserves since 1977. ICES JMS 79: 1187-1201. doi:10.1093/icesjms/fsac042 | [F] (WebFetch extraction) | Western Baltic adult HSI in Q1 3.5-5 % (2000-2010), then falling to about 2 %. HSI on a predicted (length-weight) mass. |
| 7 | Eero, M. et al. 2023. New insights into the recent collapse of Eastern Baltic cod from historical data on stock health. PLoS ONE. doi:10.1371/journal.pone.0286247 | [F] | Eastern Baltic female HSI (total weight): lowest on record about 4.3 % (1950s, and 2017/2019 in Q2). Heavily parasitised livers hold about half the lipid. |
| 8 | Ryberg, M.P. et al. 2020. Physiological condition of Eastern Baltic cod infected with *Contracaecum osculatum*. Conserv. Physiol. 8: coaa093. PMC7507771 | [F] | Liver lipid 0.58 / 0.45 / 0.30 g/g and liver water 36.6 / 44.2 / 54.0 % (uninfected / medium / high). Whole fish EXCLUDING liver 3.67 / 3.49 / 3.32 kJ/g wet. |
| 9 | Rojbek, M.C. et al. 2012. Linking lipid dynamics with the reproductive cycle in Baltic cod. MEPS 471: 215-234. https://www.int-res.com/articles/meps_oa/m471p215.pdf | [F] | Fillet lipid 0.66 %; muscle water 81-83 % before spawning, 84.8 % at spawning, 86.1 % after. HSI lowest at main spawning (values only in a figure). |
| 10 | Yaragina, N.A., Marshall, C.T. 2001. Interannual variations in the liver condition index of cod... Barents Sea. ICES CM 2001/V:31 | [F] | NE Arctic cod liver index about 3-9 % (annual means 1927-2000). |
| 11 | Yaragina, N.A., Marshall, C.T. 2000. ICES JMS 57: 42-55 | [A] | NE Arctic liver index peaks in March-April and is higher in capelin-rich years. |
| 12 | Eliassen, J.-E., Vahl, O. 1982. J. Fish Biol. 20: 707-716 | [A] | Balsfjorden cod accumulate energy October-January and lose it January to spawning (April); gonads explain about 1/3 of the loss. |
| 13 | Lawson, J.W., Magalhaes, A.M., Miller, E.H. 1998. Important prey species of marine vertebrate predators in the northwest Atlantic: proximate composition and energy density. MEPS 164: 13-20. https://www.int-res.com/articles/meps/164/m164p013.pdf | [F] | Whole small cod (20 cm, 131 g; Newfoundland): 78.5 % water, 2.6 % lipid, 16.1 % protein, 4.2 kJ/g. ED independent of size; July 4.16 vs August 4.40 kJ/g. |
| 14 | Ransijn, J., Booth, C., Smout, S. 2019. A calorific map of harbour porpoise prey in the North Sea. JNCC Report 633. https://data.jncc.gov.uk/data/c12c1b45-73ba-4402-a8f5-ec0275a72cf1/JNCC-Report-633-FINAL-WEB.pdf | [F]; the whiting values are [2nd] from Pedersen & Hislop 2001 | Porpoise prey energy table: cod 4.2 kJ/g for all sizes and quarters (Lawson 1998); whiting Q1 and Q3 by length class. |
| 15 | Pedersen, J., Hislop, J.R.G. 2001. Seasonal variations in the energy density of fishes in the North Sea. J. Fish Biol. 59: 380-389 | [A] (abstract cut off on the DTU Orbit page) | Covers saithe, haddock, whiting and Norway pout, NOT cod. ED rises with length, and seasonal differences are "obvious in mature fish". A gadoid ED-on-dry-substance regression relayed by a subagent is NOT verified and is not used ([Sn]). |
| 16 | Hislop, J.R.G., Harris, M.P., Smith, J.G.M. 1991. Variation in the calorific value and total energy content of the lesser sandeel and other fish preyed on by seabirds. J. Zool. 224: 501-517 | [F] (scan read by subagent) | Monthly whiting wet ED by length, northern North Sea 1987; "relatively low (usually < 5 kJ/g wet weight) and varies little from month to month". |
| 17 | Spitz, J., Mourocq, E., Schoen, V., Ridoux, V. 2010. Proximate composition and energy content of forage species from the Bay of Biscay. ICES JMS 67: 909-915. https://archimer.ifremer.fr/doc/00252/36368/34909.pdf | [F] | Whole-body whiting (17-22 cm) 3.9 kJ/g with 0.7 % lipid; hake (22-29 cm) 3.7 kJ/g with 0.7 % lipid. Bay of Biscay, not the North Sea. |
| 18 | Matvaretabellen (Norwegian Food Composition Table). https://www.matvaretabellen.no/api/en/foods.json | [F], food composition table (raw fillet; size and season unknown) | Raw fillet ED: haddock 3.29, whiting 3.33, saithe 3.59 kJ/g (fat 0.0 / 0.6 / 1.3 %). Cod liver raw: 36 % water, 59 % fat, 22.8 kJ/g. |
| 19 | "The Composition of Fish", FAO-hosted copy, https://www.fao.org/4/x5916e/x5916e01.htm (authors not shown on the page; I take it to be the Torry Advisory Note by Murray & Burt) | [F] (WebFetch extraction), food composition table (raw flesh) | Raw flesh energy 280-400 kcal/lb, i.e. 2.6-3.7 kJ/g [S conversion]. Fat 0.1-1.0 %. Cod liver oil 50-75 % of the liver. Seasonal change "noticeable mainly in the liver". |
| 20 | Andreasen, H. et al. 2017. Diet composition and food consumption rate of harbor porpoises in the western Baltic Sea. Mar. Mamm. Sci. https://repository.library.noaa.gov/view/noaa/52475/noaa_52475_DS1.pdf | [F] | Uses quarterly, species- and length-specific ED from Pedersen & Hislop 2001 and Temming & Herrmann 2003. Prey mass needs vary up to 10 % over the year, mainly through herring and sprat. |
| 21 | Hammond, P.S., Wilson, L.J. 2016. Grey seal diet composition and prey consumption. Scottish Marine and Freshwater Science 7(20). https://data.marine.gov.scot/sites/default/files//SMFS%20Vol%207%20No%2020.pdf | [F] | Fish ED from "Murray and Burt 1977", one value per species; grey seal requirement 5497 kcal/day. Seasonal prey ED is listed as a possible refinement. |
| 22 | fb4package 2.1.0 (Fish Bioenergetics 4.0 species database, CRAN). https://cran.r-project.org/src/contrib/fb4package_2.1.0.tar.gz | [D] (I parsed `fish4_parameters.RData`) | Atlantic cod: predator ED 5000 J/g, constant (PREDEDEQ 1), source "Hansson et al. 1996; Mateo 2007". Walleye pollock 5392 J/g. |
| 23 | Parameter table of a North-East Arctic cod state-dependent energy allocation model (ICES JMS table page https://academic.oup.com/view-large/4540833; article not identified) | [2nd] (Holdway & Beamish 1984 via that table; WebFetch extraction) | Somatic ED 4.62 x 10^6 J/kg wet; gonad 6.93 x 10^6 J/kg. |
| 24 | Funk, S. et al. 2023. Tracing growth patterns in cod using bioenergetic modelling. Ecol. Evol. 13: e10751. PMC10667610 | [F] (WebFetch extraction) | Western Baltic cod model: the liver is the internal energy store, with a constant allocation between growth and storage. Its ED values are in appendices that I did not obtain. |
| 25 | Costa, A.M. 2013. Somatic condition, growth and reproduction of hake in the Portuguese coast. Open J. Mar. Sci. 3(1) | [F] (WebFetch extraction) | Hake HSI seasonality, qualitative only: female maximum in October; values in figures only. |
| 26 | Lloret, J. et al. 2008. Lipid (energy) reserves of European hake in the NW Mediterranean. Vie et Milieu 58: 75-85 | [Sn] | The hake liver holds most of the fat; pre-spawning females have less liver lipid. No numbers. |
| 27 | Ramsvatn, S. 2006. MSc thesis, Univ. Oslo (haddock, Oslofjord). https://www.duo.uio.no/handle/10852/11640 | [Sn] | Haddock HSI and liver fat % are closely related; condition dips in winter and at spawning. No numbers. |
| 28 | Tronbol, F. et al. 2022. Tracking oocyte development and the timing of skipped spawning for north-east Arctic haddock. J. Fish Biol. 100: 1464-1474. PMC9321023 | [F] (WebFetch extraction) | Haddock somatic HSI: lower in skipping and spent females. Values only in Fig. 9a. |
| 29 | Juvenile haddock diet trial (Nanton et al.; year and journal not verified) | [Sn][L] | Cultured haddock on 12-24 % lipid feed: HSI 10.0-15.2 %, liver lipid 73.8-79.3 % wet. Not a wild range. |
| 30 | Sveegaard, S. 2011. PhD thesis, Paper V (via project note `Vintermetabolism sill och tumlare/porpoise_winter.md` section 7) | [2nd] | Porpoise prey ED: whiting 4.1-5.0 kJ/g (10-30 cm, Pedersen & Hislop 2001), cod 20 cm 4.2 kJ/g (Lawson 1998). |
| 31 | Project note `Fiskeridödlighet i Skagerrak idag/ices_gadoid_f.md` (IBTS survey biomass shares) | [D] (earlier project analysis) | Today's Skagerrak mix, W1 (2021-25, Q1+Q3): cod 0.058, haddock 0.543, whiting 0.267, saithe 0.121, hake 0.011. |

---

## 1. Whole-body energy density (kJ/g wet), season and size

- **Whiting, North Sea, by quarter and length** [2nd via #14, from Pedersen & Hislop 2001], in kJ/g:

  | Length (cm) | Q1 | Q3 |
  |---|---|---|
  | 1-11.9 | 4.1 | 3.6-3.8 |
  | 12-14.9 | 4.0 | 3.9 |
  | 15-19.9 | 4.3 | 4.7 |
  | 20-24.9 | 4.8 | 5.3 |
  | 25-39.9 | 5.0 | 5.4 |

  - Adults (20-40 cm): the mean of the four cells is 5.1 kJ/g, and Q3 is 8-10 % above Q1 [S].
  - Small fish are lower in Q3 than in Q1.
- **Whiting, northern North Sea 1987, monthly** [F, #16]:
  - February, 13.5-27.5 cm: 4.2-4.4 kJ/g.
  - March, 11.5-20 cm: 3.5-4.6 kJ/g.
  - June, 20-30 cm: 4.7-5.2 kJ/g.
  - August, 7.5-30 cm: 3.0-5.2 kJ/g; 30 cm fish are 5.2 kJ/g in both June and August.
  - The paper's ED-on-water regression: "Whiting CV (kJ/g wet) = -0.339 W + 31.271" (W = % water).
  - The mean of the nine 20-30 cm cells is 4.58 kJ/g [S].
- **Whiting, Bay of Biscay, 17-22 cm** [F, #17]: 3.9 kJ/g (79.5 % water, 16.7 % protein, 0.7 % lipid).
- **Hake, Bay of Biscay, 22-29 cm** [F, #17]: 3.7 kJ/g (80.4 % water, 16 % protein, 0.7 % lipid).
- **Cod**:
  - Newfoundland, about 20 cm [F, #13]: 4.2 ± 0.60 kJ/g. ED was unrelated to size (r² = 0.07). July 4.16 vs August 4.40 kJ/g. Other region; small fish.
  - Model values: 4.62 kJ/g somatic [2nd, #23] and 5.0 kJ/g whole predator [D, #22].
  - No measured whole-body ED of large North Sea cod including the liver was found.
- **Haddock and saithe**: no whole-body wet ED obtained. Pedersen & Hislop 2001 has both, but the numbers were not accessible (#15).
- **Size**: within whiting, ED rises from about 3.6-4.1 kJ/g (< 15 cm) to 5.0-5.4 kJ/g (25-40 cm) [2nd, #14]. The pooled model biomass is mostly adults, so it sits at the upper end. Porpoises eat mainly small gadoids (#20: cod of age groups 1-2), so they sit at the lower end.

## 2. Lean body and muscle

- **Cod muscle** [F, #1]:
  - "proteins accounting for 10 to 15% of fresh weight ... Values ranging up to 18-19%" in other studies.
  - "Lipids make up 1% of muscle fresh weight ... remarkably constant".
  - Muscle ED runs from 1.5 kJ/g at 90 % water to 5 kJ/g at 78 % water (well-fed cod: muscle water 78 %, range 77-80).
  - [S] Two-point line through these endpoints: muscle kJ/g = 1.5 + (90 - water%) x 3.5/12. This gives 4.1 at 81 % water, 3.5 at 83 %, 3.0 at 84.8 % and 2.6 at 86.1 %.
- **Baltic cod fillet** [F, #9]:
  - Lipid 0.66 %.
  - Muscle water 81-83 % before spawning, 84.8 % at spawning and 86.1 % after it. So the muscle itself loses roughly a third of its energy per gram over spawning (4.1 to 2.6 kJ/g by the line above) [S].
- **Whole fish minus liver, Eastern Baltic cod** [F, #8]: 3.67 / 3.49 / 3.32 kJ/g (uninfected / medium / high parasite load); whole-fish water 80.2-81.4 %. This is a stock in poor condition.
- **Raw fillet, food tables** [F, #18, #19]:
  - Matvaretabellen: haddock 3.29, whiting 3.33, saithe 3.59 kJ/g; protein 18.3-19.4 %, water 80-81 %.
  - Torry/FAO table: 2.6-3.7 kJ/g, fat 0.1-1.0 %, protein 14.6-20.3 %.
- **Lean-ish whole bodies** (low lipid, small livers): whiting 3.9 and hake 3.7 kJ/g [F, #17]; small cod 4.2 kJ/g [F, #13].

## 3. Liver: HSI, lipid, energy density, seasonal cycle

**HSI** (basis in brackets):

- **Cod, Gulf of St Lawrence (somatic)** [A, #3]:
  - Autumn maximum 4.3-5.7 %.
  - May minimum 2.2-3.0 %, which the authors call "below those reported for other stocks" (collapse years).
  - [F, #1]: peak "at the end of summer", lowest "in spring and early summer at the end of the spawning season".
- **Cod, well fed in the lab (somatic)** [F][L, #1]: "HSI value of 7.1 % (range 4.2 - 9.9), but HSI values higher than 10% have been observed in cod fed fatty diets".
- **Condition criteria** [F, #1]: "HSI values in the range 2 to 6% ... indicate a good condition ... excellent condition ... HSI values over 6%".
- **Western Baltic cod (predicted mass)** [F (WebFetch extraction), #6]:
  - Q1 3.5-5 % in 2000-2010, then down to about 2 %.
  - Q1 higher than Q4 in almost all years.
  - Spawning late December to March (February-April per #9).
- **Eastern Baltic cod females (total weight)** [F, #7]: the lowest-on-record quarterly means are about 4.3 %.
- **NE Arctic cod** [F, #10; A, #11]: annual means about 3-9 % over 1927-2000, with a pre-spawning peak in March-April.
- **Balsfjorden cod** [A, #12]: energy builds October-January and falls from January to spawning in April.
- **Haddock**:
  - No wild numbers obtained.
  - HSI tracks liver fat % [Sn, #27], and is lower in skipping and spent females [F (WebFetch extraction), #28].
  - Cultured juveniles on fatty feed: 10-15 % [Sn][L, #29].
- **Whiting, saithe**: no numbers obtained.
- **Hake**: female maximum in October; values in figures only [F (WebFetch extraction), #25]. The liver is the main fat store [Sn, #26].

**Liver lipid and liver energy density** (cod):

- [F][L, #1]: "Specific lipid content varied widely from 5 to 70% of fresh weight". Liver ED runs from 5.0 kJ/g at 80 % liver water to 30 kJ/g at 20 % water (R² > 0.96). Well-fed liver water is 22 % (15-32); starved-to-death livers are 75-78 %.
- [S] Two-point line: liver kJ/g = 38.33 - 0.4167 x water%.
  - 22 % water gives 29.2 kJ/g.
  - 36.6 % gives 23.1.
  - 54 % gives 15.8.
  - 60 % gives 13.3.
  - 75-78 % gives 7.1-5.8.
- Eastern Baltic [F, #8]: lipid 0.58 g/g at 36.6 % water (uninfected) and 0.30 g/g at 54 % water (heavily infected).
- Market cod liver [F, #18]: 59 % fat, 22.8 kJ/g. Torry/FAO: liver oil 50-75 % [F (WebFetch extraction), #19].

**Liver energy per tonne of whole fish, cod as proxy** [S]. Liver MJ/t = HSI/100 x liver kJ/g x 1000:

| Condition | HSI (source) | Liver kJ/g (source) | Liver MJ per t fish |
|---|---|---|---|
| Starved to death (lab) | 0.6-1.0 % (#1) | 5.8-7.1 (75-78 % water, #1 + line) | 0.006 x 5.8 x 1000 = 35 to 0.010 x 7.1 x 1000 = 71, i.e. **~40-70** |
| "Alarming" criterion | ~1 % (#1) | 13.3 (60 % water, #1 + line) | **~130** |
| Wild spring minimum | 2.2-3.0 % (#3) | 14-16 (lipid 0.30 g/g: 0.30 x 39.5 + ~0.1 x 23.6 = 14.2; 54 % water: 15.8; #8 + line) | 0.022 x 14 x 1000 = 308 to 0.030 x 16 x 1000 = 480; central 0.026 x 15 x 1000 = **~390** |
| Mean | ~4-5 % (#6 Q1 3.5-5; #7 ~4.3 lowest on record; #10 3-9) | ~23 (36.6 % water: 23.1, #8; food table 22.8, #18) | 0.04 x 23 x 1000 = **~920** to 0.05 x 23 x 1000 = 1150 |
| Wild maximum (autumn) | 4.3-5.7 % (#3) | 29.2 (well-fed 22 % water, #1) | 1256 to **~1665** |
| Well fed, lab | 7.1 % (4.2-9.9) (#1) | 29.2 | **~2070** (up to ~2900 at 9.9 %) |

The HSI values from #1 and #3 are on somatic weight. Per tonne of total fish (with gonads and stomach contents) they are a few % lower.

## 4. Seasonal amplitude of whole-body energy, and the liver share

- [F, #16]: whiting energy "varies little from month to month". [2nd, #14]: adult whiting Q3 is 8-10 % above Q1. Compare herring, 4.4-6.5 kJ/g in Q1 and 10-12 kJ/g in Q3 (same table), roughly twofold.
- [S] Cod proxy, lean 3.85 kJ/g plus the liver from section 3:
  - Spring: 3850 + 390 = 4240 MJ/t.
  - Mean: 3850 + 920 = 4770 MJ/t.
  - Autumn maximum: 3850 + 1665 = 5515 MJ/t.
  - Max/min = 1.30. Gadoids therefore swing about +-13 % around the mean, against about x2 for herring.
- In addition, the lean body itself loses energy around spawning: muscle water rises from 81 % to 86 % [F, #9]. In the model that is biomass loss, not reserve.
- Share of whole-body energy in the liver [S]:
  - Wild autumn maximum: 1665 / 5515 = 30 %.
  - Well fed: 2070 / (3850 + 2070) = 35 %.
  - Mean: 920 / 4770 = 19 %.
  - Spring minimum: 390 / 4240 = 9 %.

## 5. Starvation thresholds and fasting time

- **At death after prolonged starvation** [F][L, #1]: "87 and 88% water in muscle ... 75 and 78% water in liver ... K (0.54 and 0.47 ...) and of course HSI (1.0 and 0.6%, range 0.7-1.4 and 0.3-1.1)". Black & Love 1986: HSI 0.8 %, muscle water 88 % [2nd, #5].
- **Danger criterion** [F, #1]: "survival of cod with condition factor values below 0.70 is jeopardized ... particularly if HSI values are close to 1%, with water contents over 60% in the liver and over 88% in the muscle".
- **Order of mobilisation** [F, #1]: liver lipid goes first. "When no food is available ... muscle water contents >84% mean that energy is being obtained from muscle proteins and not from lipids stored in the liver."
- **Fasting time** [F][L, #2]:
  - Cod 38-58 cm were "deprived of food for a period of 84 days at 2, 6 or 10 C"; "weight loss was linear".
  - K fell by 0.148 / 0.193 / 0.208 over the 84 days. Deaths in that window are not reported.
  - A subagent extrapolated the fitted equations linearly to the lethal mean K of about 0.5, giving about 4.7-7 months [S, mixing total and somatic K; treat as order of magnitude].
- **Wild fish** [F, #2]:
  - Wild cod "fed but did not meet their energy requirements in winter prior to spawning".
  - In April-May the wild decline matched the unfed lab rate.
  - In 1994, 70-90 % of wild cod in April-June had somatic K below 0.75.
- **Eastern Baltic, heavily parasitised** [F, #8]: K 0.68, liver lipid 0.30 g/g, liver water 54 %. These fish are in poor condition but alive.

## 6. Gadoid energy density in porpoise and seal energetics

- **Porpoises, North Sea** [F, #14]:
  - Cod 4.2 kJ/g for all sizes and quarters (Lawson 1998, a Newfoundland value).
  - Whiting by quarter and length (table in section 1).
  - Text: seasonal and length values were available "only for Atlantic herring and whiting".
- **Porpoises, western Baltic** [F, #20]:
  - Quarterly, length-specific values from Pedersen & Hislop 2001.
  - Daily prey mass needs vary up to 10 % over the year, "especially important for herring and sprat".
  - Main prey are cod of age 1-2, herring and gobies.
- **Porpoises, inner Danish waters** [2nd, #30]: whiting 4.1-5.0, cod 4.2 kJ/g.
- **Grey seals, UK** [F, #21]:
  - One fish ED per species, from Murray & Burt 1977, not seasonal.
  - The FAO-hosted Torry table gives raw-flesh values of only 2.6-3.7 kJ/g for these gadoids [F (WebFetch extraction), #19; S conversion]. If those are what the seal study used, its gadoid ED is a fillet value below whole-body values (not verified which Murray & Burt table they used).
- **Harbour seals, Skagerrak**: Kappa et al. 2026 (Mar. Mamm. Sci. 42: e70069) and Sorlie et al. 2020 compute diet energy, but neither full text was accessible.
  - The project's seal note (`Parametervärden för sälar i Skagerrak/diet_and_foraging.md`) back-calculates about 4.8 kJ/g from Harkonen & Heide-Jorgensen's ration (flatfish) [S].
- **Summary**: predator studies use about 4-5 kJ/g for gadoids. Seasonal prey quality matters for clupeids and is small for gadoids.

## 7. Bioenergetic and ecosystem models

- **Fish Bioenergetics 4.0, Atlantic cod** [D, #22]: predator ED 5000 J/g, constant (Hansson et al. 1996; Mateo 2007). Walleye pollock: 5392 J/g.
- **NE Arctic cod energy allocation model** [2nd, #23]: somatic 4.62 kJ/g and gonad 6.93 kJ/g (Holdway & Beamish 1984).
- **Western Baltic cod model** [F (WebFetch extraction), #24]: the liver is an explicit store; ED values were not obtained.
- **DEB (Add-my-Pet)**: not obtained (host certificate error).
- **Ecopath**: works in wet-weight biomass, so it carries no energy density (a property of the method; no number used).

---

## Inference

1. **Whole body.** Pooled adult gadoids average about 4.6-4.9 kJ/g wet over the year [S]:
   - adult whiting 4.6 (#16) to 5.1 (#14);
   - cod model values 4.6-5.0 (#22, #23);
   - small cod 4.2 (#13);
   - hake 3.7 (#17).

   I use 4750 MJ/t as the pooled mean target.
2. **The lean body is about 3.7-4.0 kJ/g:**
   - Ryberg's whole-fish-minus-liver is 3.3-3.7 in a poor-condition stock;
   - low-lipid whole whiting 3.9 and hake 3.7;
   - fillets 3.3-3.6, with skin, gut and skeleton adding little.

   Muscle lipid is about 1 % and nearly constant (#1). So the gadoid reserve really is the liver, as assumed.
3. **The liver reserve (cod proxy) is about 400 MJ/t at the spring minimum, about 900 at the mean and about 1650 at the wild autumn maximum.** It reaches about 2100 MJ/t only in fish fed to satiation and about 2900 on fatty diets. The 2500 in the library is a lab, fatty-diet ceiling.
4. **Gadoid whole-body energy varies about 1.3x over the year, against about 2x for herring.** The maximum is in late summer or autumn and the minimum in spring after spawning.
5. **Starvation (muscle protein use) starts once the liver is nearly empty:**
   - about 130 MJ/t at the danger criterion and 40-70 MJ/t at death;
   - wild fish routinely reach about 400 MJ/t (about 0.2 of a 2000 MJ/t reserve) in spring and survive.
6. **The current 5500 MJ/t** is close to the autumn maximum of the whole body (5515 [S]) and to large Q3 whiting (5.4). It is 16 % above the annual mean (4750) and 43 % above the lean body (3850). As both prey value and build cost it is therefore too high for most of the year. A full current gadoid carries 5500 + 2500 = 8000 MJ/t, a value no gadoid reaches.

## Recommendation [S]

**Lean `energy_content`.** Per-species lean values:

| Species | Lean kJ/g | Basis |
|---|---|---|
| Cod | 3.8 | between Ryberg's 3.3-3.7 (poor stock) and pre-spawning muscle 4.1 |
| Haddock | 3.8 | ASSUMED equal to cod; no whole-body data; fillet 3.29 |
| Whiting | 3.9 | Spitz whole body |
| Saithe | 4.0 | ASSUMED, fillet 0.3 above haddock |
| Hake | 3.7 | Spitz whole body |

Weighted by today's Skagerrak survey mix W1 (#31: cod 0.058, haddock 0.543, whiting 0.267, saithe 0.121, hake 0.011):

0.058 x 3.8 + 0.543 x 3.8 + 0.267 x 3.9 + 0.121 x 4.0 + 0.011 x 3.7 = 0.220 + 2.063 + 1.041 + 0.484 + 0.041 = **3.85 kJ/g, i.e. 3850 MJ/t**.

- The fallback weights (haddock 0.5, whiting 0.2, cod 0.15, saithe 0.1, hake 0.05) give 3.84; the weighting hardly matters.
- Range 3700-4000.
- Use the fed-fish lean value, not the spent-fish one. The post-spawning loss of muscle energy is biomass loss (`starve_rate`), not reserve.

**`max_energy_reserve` (liver store) = 2000 MJ/t of fish**, range 1650-2100. Wild autumn maximum to well-fed lab, cod proxy (section 3).

**`reserve_reference_fill`.** Fill = (4750 - 3850) / 2000 = 0.45. Check: 3850 + 0.45 x 2000 = 4750 MJ/t = the mean whole body. With a mean HSI of 5 % instead of 4 % (liver 1150 MJ/t) the fill would be 1150/2000 = 0.55; the plausible range is 0.45-0.55.

- This presumes `prey_includes_reserve: true` for gadoids, as for herring.
- Alternative: ME 1800 with fill 0.5 gives the same 4750.

**`maintenance_level` u.** The data give these fills of a 2000 MJ/t reserve:

| Condition | Fill |
|---|---|
| Death | 0.02-0.035 |
| Danger criterion | ~0.07 |
| Routine wild spring minimum (survived) | ~0.2 |
| Wild autumn maximum | ~0.8 |

Recommend **u = 0.1** (range 0.05-0.2). With 0.2 the herring convention holds and a little protein use in spring is covered.

The current 0.5 puts the onset of starvation at about 1000-1250 MJ/t of liver energy (HSI about 4-5 % at 23 kJ/g). That is mean wild condition, not starvation.

**Reserve duration check.**

- resting_metabolism is 6.0 MJ/t/tick x 4 = 24 MJ/t/day. Cod SMR at 10 C is 57 mg O2/kg/h x 24 h x 13.6 J/mg = 18.6 MJ/t/day [S; 13.6 J/mg O2 is my assumed oxycalorific coefficient].
- At that cost a 2000 MJ/t liver lasts 2000/24 = 83 days from full, and (1 - 0.1) x 2000 / 24 = 75 days down to u.
- After that the lean body is used.
- This is consistent with cod surviving an 84-day lab fast with linear weight loss (#2) and with months more on protein.

**Model-side consequences (flags for the caller; nothing was edited):**

- Without `prey_includes_reserve` and `reserve_reference_fill` for gadoids, porpoises and seals would get only 3850 MJ/t instead of 5500 (-30 %). With them, at fill 0.45, they get 4750 MJ/t (-14 %), so they need about 16 % more gadoid mass per MJ.
- Porpoises and seals eat mostly small gadoids (< 25 cm, about 4.0-4.5 kJ/g whole body, section 1), so even 4750 MJ/t is generous for what they actually catch.
- Lowering `maintenance_level` changes the derived `growth_rate` (`growth_budget.py` divides by s* - u) and the hunger gate (1 - s)/(1 - u). Rerun the loader and the tests after any change.
- Lowering `energy_content` lowers the cost of building a tonne of gadoid by 30 %.

## Main uncertainties and gaps

- **Haddock (54 % of the pool) and saithe have no wild HSI, liver-lipid or whole-body ED numbers here.** The liver reserve and the u above are the COD cycle applied as a proxy.
  - Haddock is known to store lipid in the liver (#27, #29), but its wild amplitude is unknown.
  - Pedersen & Hislop 2001 (haddock, saithe, whiting by quarter and length) is the key missing full text.
- **No North Sea, Skagerrak or Kattegat seasonal HSI series for any gadoid.** The cod numbers are from the Gulf of St Lawrence, the Baltic and the Barents Sea.
- **Intermediate liver ED values come from a two-point line through Dutil's endpoints**, not from a fitted regression. The HSI bases differ (somatic, total, eviscerated, predicted mass).
- **The lean values for haddock and saithe are assumptions**, built from fillet values.
- **The Skagerrak harbour seal diet-energy papers** (Kappa et al. 2026; Sorlie et al. 2020) were not accessible.
