# Evidence C - Gadoid diet by mass, mapped to the model prey pools (check of phi)

Scope: diet composition by mass of haddock, saithe, whiting, cod and hake, by size class, in the
northern North Sea (NNS), Norwegian Deep, Skagerrak and Kattegat. Each prey category is mapped to a
model pool: `benthic_community` (B), `pelagic_fish` (F), `zooplankton` (Z), other fish in no pool
(O-fish) or other. The numbers only check the morphology-based phi scores. They never set them.

This file extends `../Bentos åtkomlighet för torskfiskar/evidence_C_recent_diets.md` and
`../Torskfiskarnas artsammansättning/evidence_C_diet_morphology.md` section 2. It does not repeat
their tables (Albert 1994, Dorman et al. 1992, Hislop et al. 1991, CRR 219, Mackinson & Daskalov).

**Tags.** [F] full text read or data downloaded and parsed; [A] abstract or search snippet only;
[S] secondary (the encoding source is named; the original was not read).
**Ecosystem state.** Rows from before 2000 are marked **PRE-2000**. They describe another ecosystem
state (other stock sizes, other prey fields) and are context only.

## Main finding (corrects the earlier files)

The earlier files found no post-2000 by-mass diet for gadoids in the Skagerrak, Kattegat or
Norwegian Deep. **The data exist and fill that gap**; the public download was reachable but had not
been tried. The public download of the ICES Stomach Content
database (https://stomachdata.ices.dk, CC BY 4.0) holds individual prey **wet weights** from 2020-2025
for cod, whiting and hake in the Skagerrak and Kattegat, haddock and hake in the NNS (incl. the
Norwegian Trench), and cod in the NNS. I downloaded it by year and computed the shares myself [F].

The download is only the **unrestricted** part of the database. The 2013 collection, most of the
1981 and 1991 Year-of-the-Stomach records, the Dorman et al. 1991 cod set and the 2012-14 hake set
(Huwer et al. 2014) are absent or nearly absent. Saithe has one stomach after 2000.

## Summary table (shares of identified prey mass, %)

Columns: B = benthic_community; F = pelagic_fish (h = herring, s = sprat, c = clupeid not
identified to species); Z = zooplankton (e = euphausiids, cop = copepods); Pas = Pasiphaea and
other pelagic shrimps (judgement case, kept apart); O-fish (u = unidentified fish, which is part of
O-fish); Oth = cephalopods and other. Fed = stomachs with prey. Unidentified remains and
"Crustacea"/"Arthropoda" not identified further are left out of the denominator (they are listed
separately in the species sections). "ICES-DB" = my computation from the ICES download [F].
"Thin" = too few fed stomachs or prey items to be quantitative (the dominant share rests on
fewer than about 10 items). * = stomachs in pooled records that hold prey; the 1981 and 1991
records pool several fish, and empty stomachs inside them cannot be separated.

| Species | Size (cm) | Area, period, season | Source | Fed | B | F (h / s / c) | Z (e / cop) | Pas | O-fish (u) | Oth | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Haddock | 15-19 | NNS, Jan-Feb 2025 | ICES-DB [F] | 68 | 82.0 | 0 | 0 | 0 | 8.4 (8.4) | 0 | |
| Haddock | 20-24 | same | ICES-DB [F] | 42 | 88.2 | 0 | 0 | 0 | 0 | 0 | |
| Haddock | 25-29 | same | ICES-DB [F] | 74 | 82.2 | 0 | 5.7 (5.7 / 0) | 0 | 0.9 (0.9) | 0 | Z = one item |
| Haddock | 30-39 | same | ICES-DB [F] | 200 | 82.3 | 0 | 0 | 0 | 1.4 (0.7) | 10.9 | Oth = 2 octopods |
| Haddock | 40-49 | same | ICES-DB [F] | 91 | 69.6 | 0 | 0 | 0 | 15.4 (15.4) | 12.2 | fish = 10 items; Oth = 1 cephalopod |
| Haddock | 50-59 | same | ICES-DB [F] | 10 | 100 | 0 | 0 | 0 | 0 | 0 | thin |
| Haddock | all | same | ICES-DB [F] | 485 | **79.1** | 0 | 0.4 | 0 | 7.0 (6.7) | 9.2 | 753 stomachs, 45 hauls |
| Haddock | all | Norwegian Trench part (>=150 m, E of 2.5 E), Jan-Feb 2025 | ICES-DB [F] | 84 | 63.4 | 0 | 0.1 | 0 | 0.9 | 32.0 | 30-39 cm: B 95.2 (n 40); Oth = 1 cephalopod |
| Haddock | 20-49 | Norwegian Deep, 1984-87, PRE-2000 | Albert 1994 via earlier file [S] | see earlier file | 54-98 | - | 2-22 euph (summer) | 0-19 | 4-33 fish | - | dry weight |
| Cod | <20 | Skagerrak, 2023-25, Q1+Q3 | ICES-DB [F] | 13 | 71.3 | 0 | 0 | 0 | 0 | 0 | thin; 28.7 unid. crustacea |
| Cod | 20-29 | same | ICES-DB [F] | 13 | 79.5 | 0 | 1.2 | 0 | 17.5 (4.1) | 0 | thin |
| Cod | 30-39 | same | ICES-DB [F] | 13 | 61.3 | 0 | 0 | 0 | 38.7 (0.1) | 0 | thin |
| Cod | 40-49 | same | ICES-DB [F] | 11 | 37.6 | 0 | 0.2 | 0 | 62.1 (7.4) | 0 | thin |
| Cod | 50-69 | same | ICES-DB [F] | 6 | 15.3 | 9.0 (1.6 / 7.4 / 0) | 0 | 0 | 75.7 (12.0) | 0 | thin; Norway pout 45 |
| Cod | all | same | ICES-DB [F] | 56 | **28.8** | 5.7 (1.0 / 4.7 / 0) | 0.1 | 0 | 65.1 (9.1) | 0 | Norway pout 28.8, gadoids 10.6 |
| Cod | <20 | Kattegat, 2024-25, Q1/Q3/Q4 | ICES-DB [F] | 115 | 68.0 | 0 | 0 | 0 | 20.6 (12.3) | 0 | gobies 8.3 |
| Cod | 20-29 | same | ICES-DB [F] | 180 | 49.8 | 12.3 (4.4 / 7.9 / 0) | 0 | 0 | 35.7 (9.2) | 0 | flatfish 8.0 |
| Cod | 30-39 | same | ICES-DB [F] | 53 | 20.4 | 36.5 (3.1 / 33.5 / 0) | 0 | 0 | 39.7 (11.7) | 0.9 | |
| Cod | 40-49 | same | ICES-DB [F] | 12 | 3.2 | 5.8 | 0 | 0 | 90.9 (13.0) | 0 | thin |
| Cod | 50-69 | same | ICES-DB [F] | 8 | 6.5 | 51.7 (46.2 / 5.5 / 0) | 0 | 0 | 41.8 | 0 | thin |
| Cod | all | same | ICES-DB [F] | 370 | **24.0** | 26.1 (13.6 / 12.4 / 0) | 0 | 0 | 48.0 (10.3) | 0.2 | |
| Cod | all | NNS (57.7-59.2 N, 3 W-5.5 E), Jan-Feb 2023 | ICES-DB [F] | 77 | 25.1 | 1.8 (1.2 / 0 / 0.1) | 0 | 0 | 72.5 (0.9) | 0.3 | Norway pout 26.5, gadoids 41.5 |
| Cod | 20-49 | NNS 2023 | ICES-DB [F] | 42 | 24-33 | 0-3 | 0 | 0 | 65-73 | 0-2 | |
| Cod | all | NNS, 1981, Q1/Q3/Q4, PRE-2000 | ICES-DB [F] | 518* | 13.5 | 1.5 | 0.9 | 0 | 83.0 | 1.0 | sandeel 21, NP 17, gadoids 39; 20-29 cm: Z 38 (euph) |
| Cod | all | Skagerrak, Feb 1991, PRE-2000 | Dorman et al. 1992 via earlier file [S] | 294 | 46 | fish 46 (F / O-fish not split) | 0.1 | +7 | (in F column) | - | |
| Whiting | <20 | Skagerrak, Jan-Feb + Aug 2022 | ICES-DB [F] | 9 | 59.5 | 0 | 10.3 | 0 | 30.2 (30.2) | 0 | thin |
| Whiting | 20-29 | same | ICES-DB [F] | 14 | 11.1 | 0 | 0.1 | 0 | 88.8 (6.4) | 0 | thin; whiting eaten |
| Whiting | 30-39 | same | ICES-DB [F] | 8 | 2.3 | 6.6 (6.6 / 0 / 0) | 0 | 0 | 91.1 (4.1) | 0 | thin |
| Whiting | all | same | ICES-DB [F] | 32 | 6.2 | 4.8 (4.8 / 0 / 0) | 0.4 | 0 | 88.7 (6.4) | 0 | **thin**: 69 % = 9 whiting |
| Whiting | all | Kattegat, 2020-22 | ICES-DB [F] | 33 | 6.2 | 0 | 0.6 | 0 | 93.2 (1.1) | 0 | thin (23 g of prey) |
| Whiting | <20 / 20-29 / 30-39 | Skagerrak, Apr 1991, PRE-2000 | ICES-DB [F] | 46* / 87* / 14* | 34.2 / 10.4 / 47.5 | 0 / 6.7 / 28.8 | 47.6 / 60.2 / 7.2 (all euph) | 0 / 5.9 / 14.6 | 18.0 / 14.8 / 1.9 | 0 | Swedish R/V Argos |
| Whiting | all | same | ICES-DB [F] | 147* | **20.4** | 9.8 (6.8 / 3.0 / 0) | **48.9** (48.9 / 0) | 6.7 | 12.9 (0.9) | 0 | |
| Whiting | <20 / 20-29 / 30-39 | Kattegat, Apr 1991, PRE-2000 | ICES-DB [F] | 37* / 71* / 11* | 75.9 / 18.9 / 7.1 | 0 / 55.1 / 59.1 | 4.4 / 12.2 / 0 | 0 | 18.8 / 13.8 / 33.8 | 0 | |
| Whiting | all | same | ICES-DB [F] | 119* | 22.0 | 50.3 (19.2 / 29.9 / 1.1) | 8.5 | 0 | 19.1 | 0 | |
| Saithe | 45 | NNS (58.4 N 1.6 E), Feb 2024 | ICES-DB [F] | 1 | 70.9 | 0 | 0 | 0 | 29.1 | 0 | one fish; not usable |
| Saithe | 30 / 40 / 50 / 70 / 100 | total North Sea, 1981, Q1+Q4, PRE-2000 | Daan 1989 via FishBase [S] | 55 / 98 / 340 / 408 / 203 | 0 / 0 / 0.1 / 0.3 / 0.4-1.3 | fish 42.5 / 23.6 / 78.0 / 89.5 / 98.7 (not split by species) | 47.8 / 76.4 / 21.1 / 9.4 / 0 (euph) | - | - | 9.7 / 0 / 0.8 / 0.8 / 0.9 ceph | n = stomachs |
| Saithe | 30 / 40 / 50 / 70 | total North Sea, 1981, Q2+Q3, PRE-2000 | Daan 1989 via FishBase [S] | 97 / 208 / 403 / 354 | 0 / 0.7 / 0.1 / 0 | fish 71.6 / 62.1 / 54.3 / 65.9 | 28.2 / 37.1 / 45.7 / 33.6 (euph) | - | - | 0.2 / 0 / 0 / 0.5 | amphipods counted B |
| Hake | <20 | Skagerrak, Aug 2024 (+Jan) | ICES-DB [F] | 5 | 3.8 | 91.7 (83.8 / 0 / 8.0) | 0 | 0 | 4.5 | 0 | thin |
| Hake | 20-29 | same | ICES-DB [F] | 15 | 0 | 57.7 (19.7 / 32.5 / 5.4) | 0 | 0 | 42.3 (2.9) | 0 | NP 11, gadoids 28 |
| Hake | 30-39 | same | ICES-DB [F] | 9 | 0.1 | 79.9 (67.3 / 0 / 12.6) | 0 | 0 | 17.6 (2.3) | 2.4 | thin |
| Hake | 40-49 | same | ICES-DB [F] | 6 | 0 | 92.9 (57.2 / 4.1 / 31.7) | 0.8 | 0 | 6.3 | 0 | thin |
| Hake | 50-69 | same | ICES-DB [F] | 6 | 0 | 97.9 (93.7 / 0 / 4.2) | 0 | 0 | 2.1 | 0 | thin |
| Hake | >=70 | same | ICES-DB [F] | 7 | 0.9 | 49.0 (34.9 / 13.2 / 0.9) | 0 | 0 | 49.6 (1.7) | 0.5 | thin; gadoids 48 |
| Hake | all | same | ICES-DB [F] | 48 | **0.2** | **81.5** (65.2 / 6.3 / 10.0) | 0.1 | 0 | 17.8 (1.1) | 0.4 | 42 F items |
| Hake | all | model box only (57.7-58.4 N, 9.9-11.3 E), 2024 | ICES-DB [F] | 33 | 0.3 | 80.6 (71.9 / 6.3 / 2.4) | 0.2 | 0 | 18.5 (0.7) | 0.4 | subset of the row above |
| Hake | all | Kattegat, 2024 | ICES-DB [F] | 28 | 0.6 | 49.7 (40.4 / 5.1 / 4.1) | 1.2 | 0 | 48.4 (2.3) | 0.1 | whiting the main O-fish |
| Hake | all | NNS, Feb + Jul-Aug 2024 | ICES-DB [F] | 28 | 0.2 | 35.9 (35.9 / 0 / 0) | 0 | 0 | 63.2 (47.0) | 0 | thin: F = 1 herring |
| Hake | adults | Norwegian Deep, 1984-87, PRE-2000 | Bergstad 1991 via FishBase [S] | 58 | 0-0.1 B + 2.5-3.8 Pandalus | 0 | 0-1.2 | 0 | 95-98 (blue whiting) | 0 | |

## How the ICES numbers were made

- **Download.** `https://stomachdata.ices.dk/api/download?Country=&Year=<y>&Reporting_organisation=&EcoRegion=`
  for each year with data. Files: `data/diet/ices_stomach/ices_stomach_<year>.zip` (Haul, Predator,
  Prey tables). Prey taxa were classified from the WoRMS lineage of each `AphiaIDPrey`
  (`data/diet/worms/`). The derived table is `data/diet/ices_stomach_derived_diet_by_mass.csv`.
- **Predators.** AphiaID 126436 cod, 126437 haddock, 126438 whiting, 126441 saithe, 126484 hake.
- **Areas** (from the haul shoot position):
  - Sound (Öresund): 55.3-56.15 N, 12.4-13 E. It is kept apart from the Kattegat.
  - Kattegat: 55.5-57.75 N, 10.3-13 E.
  - Skagerrak: north of 57 N and east of the Hanstholm-Lindesnes line, up to 13 E.
  - NNS: north of 57.5 N, 4 W-13 E, outside IIIa.
  - Norwegian Trench subset: NNS hauls at >= 150 m and east of 2.5 E.
  - Model box subset: 57.7-58.4 N, 9.9-11.3 E.
- **Mass.** Prey `Weight` summed per category. Units were normalised: the Norwegian 2025 haddock
  upload and some 2024 rows use mg. A check: 755 g of prey over 485 fed haddock stomachs is
  1.6 g per stomach, about 0.5 % of a 32 cm haddock's weight. That is plausible, and 1000x more
  would not be. Where `FullStomWgt - EmptyStomWgt` exists (German 2023 cod, Norwegian 2024 hake),
  the summed prey mass matches it (median ratio 1.0). All rows used are wet weight.
- **Lengths** are in cm in every upload used. Danish uploads give IndWgt in kg; a 19 cm cod
  weighs 0.061.
- **Dropped:** algae, stones, sand, wood, plastic, nematodes, trematodes. Rows with no taxon
  ("unid", "remains") and "Crustacea"/"Arthropoda"/"Eucarida"/"Malacostraca" are kept out of the
  denominator. They are 0.2-11 % of total mass and are listed per species.
- **Pooling.** The prey mass is summed over all stomachs of a class (pooled diet). There is no
  catch weighting across hauls. Classes dominated by a few large items are flagged "thin".
- **Stomach counts.** The Norwegian 2025 haddock records hold 1-3 fish each (`Number`), so
  "stomachs" counts fish. Many Swedish IBTS records (2022-2023) have no prey rows and a blank
  `StomachEmpty`. For example, 217 Kattegat cod records in 2023 have no prey at all. These were
  probably not analysed, or are restricted, so "Fed" is the honest n, not "records". Danish
  uploads mark empties explicitly.

## Haddock

**Post-2000 by mass: NNS and the western side of the Norwegian Trench, Jan-Feb 2025** [F].
- Sampling: IMR NS-IBTS Q1 (cruise NO135158G22025Q1). 45 hauls with haddock at 57.8-61.7 N,
  0.4 W-4.3 E, 66-211 m deep. There are 475 records (753 fish), 485 of them fed, holding 755 g of prey.
- Benthos is **79 % of identified mass** (82-88 % at 15-39 cm, 70 % at 40-49 cm). The B part splits
  as polychaetes 31 %, ophiuroids 24 % (echinoderms 31 %), pagurids, *Munida*, Caridea, isopods and
  crabs together 16 %.
- **Zooplankton is ~0 in winter**: euphausiids are 0.4 % (2 items). Unidentified fish are 7 %
  (10 items) and cephalopods 9 % (3 items: 2 octopods, 1 other).
- Biomass-weighted over the box haddock length mix (`tables/length_class_pct.csv`, variant
  `A_floor`, quarter `pooled`; `A_sw_floor` is identical for haddock): **B = 0.81**
  (0.79 if the >=50 cm classes take the 40-49 cm value). This is winter only.
- Norwegian Trench subset (84 fed stomachs): B 63 % overall, but 95 % at 30-39 cm (n 40). The
  overall value is pulled down by one cephalopod in the 40-49 cm class.
- Unidentified crustacea: 4.3 % of identified mass (35 items). Most are probably benthic decapods.

PRE-2000 in-area (earlier file, Albert 1994, dry weight): B 0.80-0.98 in winter and 0.54-0.96 in
summer, with euphausiids 2-22 % in summer. The 2025 winter value sits inside the 1984-87 winter
range. **No post-2000 summer haddock data were found**, so the summer euphausiid share cannot be
checked.

Literature, qualitative only:
- Schückel et al. 2010 (northern North Sea, 2007) [A], quoted in the earlier file: "dominated by
  invertebrates in all three boxes, mainly echinoderms and polychaetes".
- ICES FishMap haddock factsheet [A]: "benthic invertebrates become increasingly more important
  as they grow".

## Cod

**Skagerrak 2023-25** [F]. Swedish IBTS Q1 and Q3 2023 plus a few Danish hauls. 36 hauls at
57.4-58.7 N, 8.5-11.2 E, 20-257 m deep (median 92 m). Of 523 records, only **56 stomachs had food**
(633 g).
- B falls with size: 71-80 % below 30 cm, 61 % at 30-39, 38 % at 40-49 and 15 % at 50-69 cm.
- Pooled: **B 29 %, F 6 % (sprat 4.7, herring 1.0), Z 0.1 %, O-fish 65 %** (Norway pout 29,
  gadoids 11, flatfish 7, unidentified 9).
- The B part is crabs (Brachyura, *Pagurus bernhardus*, Polybiidae) and the sea mouse
  *Aphrodita aculeata*.
- **Model box subset:** 27 fed stomachs and 203 g: B 11 %, O-fish 89 % (Norway pout 52, gadoids
  33). Too thin to use alone.

**Kattegat 2023-25** [F]. Swedish BITS/IBTS, Danish BITS; the prey data come from 2024-25 (the 2023
Swedish records have no prey rows). 97 hauls, 6-73 m deep. 370 fed stomachs, 1.15 kg of prey.
- B is 68 % below 20 cm, 50 % at 20-29, 20 % at 30-39 and 3-7 % at 40-69 cm.
- Pooled: **B 24 %, F 26 % (herring 14, sprat 12), O-fish 48 %** (gadoids 16, flatfish 15,
  unidentified 10). Z is 0.
- **Sound** (Öresund, 2021-25; 665 fed, 3.5 kg; mostly 15-40 m): B 42 %, F 17 %, O-fish 41 %. B is 76 %
  below 30 cm and 38-44 % at 30-69 cm.
- The Kattegat and Sound are shallow (median 24-32 m). They are not the box habitat.

**NNS, Jan-Feb 2023** [F]. German and Danish IBTS Q1, 14 hauls, 56-162 m. 77 fed stomachs, 2.0 kg.
- B 25 % (Nephrops 8, *Pandalus borealis* 6, crabs). F 2 %. Z 0. O-fish 73 %: haddock 23,
  Norway pout 26, whiting 8, blue whiting 6.
- The Norwegian Trench subset holds only 12 stomachs from one haul.

PRE-2000 for contrast:
- NNS 1981 (518 fed, ICES-DB [F]): B 13.5 %, O-fish 83 % (sandeel 21, Norway pout 17, gadoids
  39). At 20-29 cm euphausiids were 38 %.
- Skagerrak Feb 1991 (Dorman et al. 1992, earlier file): B 46 %, fish 46 %, krill 0.1 %.
- The 2023 Skagerrak B share (29 % pooled) is lower than 1991 (46 %). But the 2023 pooled mass
  is dominated by a few large cod with Norway pout, and n is 56.

Not obtained: post-2000 Kattegat cod papers by mass. Funk et al. 2021 (western Baltic) [A]:
"the contribution of herring Clupea harengus decreased compared with historic data".

## Whiting

**Post-2000 by mass exists but is thin** [F]. Skagerrak 2022 (Swedish IBTS Q1/Q3): 510 records
but only 32 with prey, holding 95 g. 89 % of that mass is fish, and 69 % is nine whiting
(cannibalism). Kattegat 2020-22: 33 fed stomachs and 23 g of prey. Neither is a diet estimate.
The near-zero fed fraction is not explained by flagged regurgitation: only 6-13 records per upload
are marked regurgitated. Most records probably were not analysed, or their prey rows are
restricted (see the method section). For context only, whiting regurgitate often in trawls. Robb
et al. 1994 (1991 North Sea) [F]: "A further 37% were regurgitated and 16% were empty".

**PRE-2000, April 1991, ICES-DB** [F]. These are the Swedish R/V Argos stomachs of the 1991 ICES
project (IMARES compilation). They are pooled records (51 records, 147 stomachs in records with
prey; positions are rectangle centres), so empties cannot be separated. They are **new in-area by-mass whiting data**; the earlier file
listed them as unreported.
- **Skagerrak** (147 stomachs, 19 rectangles): **Z 49 % (all *Meganyctiphanes norvegica*)**,
  B 20 %, F 10 % (herring 6.8, sprat 3.0), Pasiphaea and *Hymenodora* 6.7 %, O-fish 13 %
  (Norway pout 5). At 20-29 cm: Z 60 %, B 10 %. At 30-39 cm (n 14): B 48 %, F 29 %, Pas 15 %.
- **Kattegat** (119 stomachs): F 50 % (sprat 30, herring 19), B 22 %, O-fish 19 %, Z 8.5 %.
- Model box subset (49 stomachs, 9 rectangles): Z 40 %, B 32 %, Pas 12 %, O-fish 17 %.

Literature:
- Robb et al. 1994 [F]: "a large proportion of the crustaceans eaten in the southern North Sea
  were Crangonids, whereas Euphausiacea were predominunt [sic, OCR] in the northern areas."
- Robb et al. 1994 [F]: "fish and crustacea together accounted for at least 60% of the weight of
  the stomach contents of all size" classes. Whiting 1991, by area only in figures.
- Pedersen 1999, via a search snippet [A]: pelagic whiting ate sprat and sandeel, demersal whiting
  ate Norway pout and herring. Bottom-trawl stomachs therefore under-sample the pelagic part of the
  whiting diet.

## Saithe

**No usable post-2000 by-mass saithe diet was found** for the North Sea or Skagerrak. The ICES
download has one stomach (45 cm, Feb 2024: one Norway pout plus one benthic item).
- IMR's 2021 climate narrative [F] lists prey only: "Adult NS saithe prey on a diversity of
  species, including blue whiting (Micromesistius poutassou), Norway pout (Trisopterus esmarkii),
  sandeel (Ammodytes marinus), herring (Clupea harengus), and krill (Nedreaas, 1987; Mehl,
  unpublished data)."
- Cormon et al. 2014 (ICES JMS 71:1342) [A]: "it is essential to investigate saithe and hake
  respective diets in the North Sea as they are currently lacking." https://doi.org/10.1093/icesjms/fsu120

PRE-2000 by mass (total North Sea 1981; Daan 1989, the ICES 1981 stomach report, encoded by FishBase
[S]):
- Euphausiids fall with size: 48 % at 30 cm, 76 % at 40 cm, 21 % at 50 cm, 9 % at 70 cm and 0 % at
  100 cm (Q1+Q4). In Q2+Q3 they are 28-46 % at 30-70 cm.
- The rest is fish, not split by species in FishBase.
- Benthos is <= 1 % in every class.
- Greenstreet 1996 [S] gives the same picture by quarter: euphausiids 21-57 % at 15-45 cm in Q1/Q4
  and 26-45 % in Q2/Q3; 10-36 % at 75 cm.
- The earlier file has CRR 219 (1991): benthos 0-1.4 %, euphausiids 3-20 %.
- Du Buit 1991 (Fish. Res. 12:307, Scotland) [A, search summary]: crustaceans 30-40 % of food
  weight in the North Sea, and Norway pout "predominant" among the fish.

So for saithe: **B ~0; Z 0.2-0.8 at 30-50 cm, 0.1-0.35 at 70 cm and 0-0.13 at 100 cm (1981); the
rest fish.** The
herring/sprat (F) vs Norway pout/blue whiting (O-fish) split is unknown by mass. Every source names
Norway pout first.

## Hake

**Skagerrak 2024, by mass** [F]. Swedish IBTS Q3 (mostly August), 23 hauls, 30-237 m (median 94 m).
48 fed stomachs and 808 g of prey. 33 of the fed stomachs are inside the model box.
- **F 82 % (herring 65, sprat 6, clupeids 10), O-fish 18 % (whiting and other gadoids 16), B 0.2 %,
  Z 0.1 %.**
- F is 58 % at 20-29 cm (sprat 33) and 80-98 % at 30-69 cm. Gadoids make up about half the diet at
  >= 70 cm (n 7).
- Kattegat 2024 (28 fed): F 50 %, O-fish 48 % (whiting). NNS 2024 (28 fed): F 36 % (one herring),
  O-fish 63 %, of which 47 % unidentified fish.

Corroboration, not by mass:
- Werner 2015 (UiB MSc, 2014; %F and %N), via the earlier file [F]: Norway pout 44 %F and
  crustaceans 13 %F.

PRE-2000:
- Bergstad 1991 (Norwegian Deep, 1984-87), via FishBase [S]: blue whiting 95-98 %, *Pandalus
  borealis* 2.5-3.8 %.
- The Skagerrak 2024 diet (herring-dominated) differs from the Norwegian Deep 1980s diet (blue
  whiting). This may reflect site (shallower IIIa hauls) or season (August).

## Crustacean and prey mapping (as coded)

The classification uses each prey taxon's WoRMS lineage. Rules are applied in this order:
1. Herring (*Clupea harengus*) -> F_herring; sprat (*Sprattus sprattus*) -> F_sprat; other
   Clupeidae -> F_clupeid; *Scomber*, *Trachurus*, *Engraulis* -> F (other pelagic).
2. Other fish (Vertebrata): identified to order or lower -> O-fish. The sub-groups are sandeel
   (Ammodytidae), Norway pout (*Trisopterus esmarkii*), gadoids (Gadiformes, incl. *Trisopterus*
   not identified to species), flatfish, gobies and other. Fish identified only above order
   ("Pisces", Osteichthyes, Teleostei, Actinopterygii, Gnathostomata) -> O-fish unidentified.
3. Cephalopoda -> other.
4. Limacina and Thecosomata -> Z. Oplophoridae and Acanthephyridae (*Hymenodora*), Luciferidae,
   Sergestidae and Pasiphaeidae (*Pasiphaea*) -> **Pas** (kept apart).
5. Euphausiacea -> Z_euph. Copepoda -> Z_cop. Hyperiidea (*Themisto*, *Hyperia*) -> Z_hyp.
   Chaetognatha, Appendicularia (*Oikopleura*), Thaliacea, Ctenophora and *Tomopteris* -> Z_other.
6. Mysida and Lophogastrida -> B. Decapoda (Caridea incl. *Pandalus*, *Crangon*, *Processa*,
   *Pontophilus*, *Philocheras*; *Nephrops*, *Calocaris*, *Callianassa*, *Upogebia*; Brachyura,
   Paguridae, Galatheidae and *Munida*) -> B. "Decapoda" not identified further -> B.
7. Amphipoda other than hyperiids (gammarids, caprellids) -> B. Amphipoda not identified further
   -> B, also reported separately (at most 1.6 % of any row used).
8. Isopoda (e.g. *Cirolana*), Cumacea, Tanaidacea, Ostracoda, Cirripedia, Leptostraca
   (*Nebalia*) and Pycnogonida -> B.
9. Crustacea, Arthropoda, Eucarida and Malacostraca not identified further -> left out of the
   denominator and reported (haddock 4.3 %, cod Skagerrak 0.4 %, cod <20 cm up to 29 %).
10. Annelida (polychaetes incl. *Aphrodita*, Echiura, Sipuncula) except *Tomopteris*, Echinodermata,
    Mollusca except cephalopods and pteropods, Nemertea, Priapulida, Brachiopoda, Anthozoa,
    Porifera and *Branchiostoma* -> B.
11. Hydrozoa, Scyphozoa and unidentified Tunicata -> other. Nematoda, Trematoda and Acanthocephala
    (parasites), algae and other non-food -> dropped.

Effect: mysids and *Pandalus* are B. *Pasiphaea* and *Hymenodora* are Pas, which matters only for
1991 Skagerrak whiting (6.7 %). Pelagic hyperiids are Z, and none appear in the 2000+ gadoid
rows by mass.

## Discrepancies and caveats

- **Z is ~0 in every post-2000 row.** Earlier data show more: whiting in the Skagerrak in April
  1991 had euphausiids at 49 %, and 20-29 cm cod in the NNS in 1981 had 38 %. Possible reasons:
  - Season. Haddock 2025 and cod NNS 2023 are January-February, when euphausiids are deep and
    scarce.
  - Daytime bottom-trawl sampling misses pelagic feeding (Pedersen 1999 [A]).
  - A real change.
  
  This cannot be resolved with these data. Treat Z = 0 as "winter, demersal sample", not as
  "gadoids do not eat krill".
- Most post-2000 IIIa cells are thin (n fed < 20). The ones that can be used quantitatively are:
  - haddock NNS (485 fed, but winter only);
  - hake Skagerrak (48 fed, 42 F items; August);
  - cod Kattegat and Sound (370 and 665 fed; shallow);
  - cod Skagerrak pooled (56 fed).
- The public download is the unrestricted subset only. Pooled shares are not catch-weighted.

## Source status

| Source | What | Status |
|---|---|---|
| ICES Stomach Content database, public download (stomachdata.ices.dk), years 1981-2025 | individual prey wet weights; all ICES-DB rows | **obtained** [F] (`data/diet/ices_stomach/`) |
| WoRMS REST API, AphiaClassificationByAphiaID | prey lineages | **obtained** [F] (`data/diet/worms/`) |
| Robb, Bell, MacMillan & Hislop 1994, ICES CM 1994/G:42 (whiting 1991) | regurgitation, euphausiids in the north | **obtained** [F] https://www.vliz.be/imisdocs/publications/ocrd/275252.pdf |
| IMR 2021, North Sea saithe climate narrative | saithe prey list | **obtained** [F] https://www.hi.no/resources/klimastatus-pa-bestander/20211214_Saithe_NS_narrative.pdf |
| Daan 1989 (ICES 1981 stomach report), Greenstreet 1996: saithe records on FishBase | saithe % by size and quarter | [S] FishBase DietCompoSummary 5634-5645, 6107-6130 (`data/diet/fishbase/detail/`) |
| Bergstad 1991, Sarsia 75:269 (hake, Norwegian Deep) | hake % | [S] FishBase dietcodes 3798, 3799; paper not obtained (doi:10.1080/00364827.1991.10413455, T&F 403) |
| Du Buit 1991, Fish. Res. 12:307-323 (saithe, Scotland) | crustaceans 30-40 %W | abstract-level only [A]; **not obtained** (doi:10.1016/0165-7836(91)90015-8, ScienceDirect 403) |
| Cormon et al. 2016, ICES JMS 73:1370 (hake, saithe, Norway pout in SMS) | no by-mass diets | [A] doi:10.1093/icesjms/fsw050 |
| Cormon et al. 2014, ICES JMS 71:1342 | "diets ... currently lacking" | [A] doi:10.1093/icesjms/fsu120 (publisher page, quote checked) |
| Pedersen 1999 (whiting, pelagic vs demersal) | habitat bias | [A] search snippet; full reference not resolved |
| Huwer et al. 2014, MARE/2012/02 (2012-14 North Sea stomachs incl. 517 hake) | by-mass hake | **not obtained**: WUR "no access"; the data are not in the public ICES download (no hake records 2012-14) |
| ICES 2013 stomach collection, 1991 Year of the Stomach full set, Dorman et al. 1991 cod raw data | by-mass, in area | **not obtained**: restricted or absent from the public download |
| CINEA 2024 (2022-23 stomachs), doi:10.2926/683598 | by number only | see earlier file; partly the same data as the 2022-23 ICES-DB rows |
| Schückel et al. 2010, J Fish Biol 77:98 (haddock NNS 2007) | qualitative | [A] (earlier file) doi:10.1111/j.1095-8649.2010.02657.x |
| Werner 2015, UiB MSc (hake 2014) | %F, %N | [F] in earlier file, hdl.handle.net/1956/10408 |
| Funk et al. 2021, J Fish Biol (western Baltic cod) | herring decline | [A] doi:10.1111/jfb.14615 |
| Kempf, Floeter & Temming (North Sea cod and whiting, 2000s) | - | **not found** in one search; southern North Sea, context only |
| DAPSTOM v4.7 / v6.3 (Cefas) | UK stomach records | **not obtained** (no download attempted; the R-shiny portal was not public) |
| Mackinson & Daskalov 2007 diet matrix | 1991 context | in the earlier file (EcoBase 457); not repeated |
