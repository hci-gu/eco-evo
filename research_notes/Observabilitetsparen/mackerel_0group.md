# Mackerel and horse mackerel on pelagic 0-group gadoids (2026-11-13)

Research note. Nothing in code or library changed. Question: do Atlantic mackerel (and horse mackerel, 0.06
of the model's mackerel group) eat pelagic 0-group cod, haddock, whiting, saithe and Norway pout, how much,
and what follows for the model's recruit import and the mackerel diet.

Grades: [F] measured (stomach data, survey data), [S] secondary (quoted by another source, or a value read
from a source I only saw in part), [J] judgement or inference. "SMS estimate" = output of the ICES SMS
multispecies model (fitted to evacuation-bias-corrected stomach data); it is a model estimate, not a
measurement, and it is not output of this project's model. Scripts were run from the scratchpad; the
inputs are listed in section 6 so every number can be redone.

## 0. Headline

- **Mackerel eat very few 0-group cod, haddock, whiting or saithe.** Every North Sea stomach set puts them
  below ~1.5 % of the diet by weight, and most samples have none:
  - Year of the Stomach 1991 [F]: cod + haddock + whiting 0.08-0.63 % in Q2 and 0.00-0.12 % in Q3;
  - ICES stomach database, June 2022 [F]: whiting 1.3 % of all prey mass, found in 2 of 19 hauls;
  - Dahl & Kirkegaard, July 1985 [F]: 0 % identified gadoids; unidentified fish 11-33 % of weight in the
    smaller mackerel size classes.
- **Prey and timing:** whiting is the species seen most recently (2022 and the DAPSTOM 2004 records), cod in
  1991. The prey are pelagic 0-groups of 3.6-6.5 cm in June-July. Most of them are taken before settlement.
- **Norway pout is the exception.** It is the one gadoid that mackerel eat in quantity: Q3 3-16 % and Q4
  5-12 % of the diet by weight in 1991 [F], mostly age 0. It belongs to the model's planktivore group, not to
  the settled recruits.
- **SMS:** in all three key runs (2017, 2020, 2023) mackerel has M2 on Norway pout only among the gadoids.
  - 2023 key run, mean 2013-22 (SMS estimate): age 0 0.082 /yr (Q3 0.068, Q4 0.014), age 1 0.004 /yr. That
    is ~7 % of Norway pout's age-0 M2 of 1.14.
  - No key run has a mackerel pair on cod, haddock, whiting or saithe (saithe is not an SMS prey at all).
  - North Sea horse mackerel -> whiting age 0 has M2 0.30 /yr (Q3) in the 2020 key run and is absent from
    the 2023 run (section 2.3).
- **Horse mackerel take more gadoids than mackerel, but at 0.06 of the group it hardly matters.**
  - Stomach data: haddock 4-43 % per size class in one coastal haul (July 1985) [F]; whiting + Gadidae 2.5 %
    of prey mass in Q3 2023 [F]; whiting 2.9 % of N. horse mackerel Q3 stomach contents in the SMS stomach
    input (1987/1991) [F].
  - Ecopath 1991: juvenile cod + whiting + saithe + haddock 1.0 % of the diet [S].
  - Weighted by its 0.06 share of the group, horse mackerel adds < 0.1 % to the group's diet [J].
- **Judgement for the model** [J]: the cod, haddock, whiting and saithe that mackerel take are almost all
  taken before settlement. That predation is already inside the recruit import. The diet share is ~0.3 %
  (range 0-1.5 %) of the group's intake, concentrated in June-July. No bed pair is warranted.

## 1. Stomach data

### 1.1 ICES Year of the Stomach 1991 - North Sea total, mackerel by age [F, full text read]

Source: Hislop, J., Bromley, P.J., Daan, N., Gislason, H., Heessen, H.J.L., Robb, A.F., Skagen, D., Sparholt,
H., Temming, A. 1997. Database report of the stomach sampling project 1991. ICES Cooperative Research Report
219. No DOI. Local PDF: `../Sejens diet på unga torskfiskar/data/crr219_ices1997_stomach1991.pdf`; tables
8-D-1..4 (p. 116-120) and 10-D (prey size by area). OCR text, so a few cells are hard to read; the values
below are the readable ones.

% of stomach content weight by mackerel age class (mean length 20-40 cm). "Non-commercial Gadidae" are
gadoids other than the assessed species (e.g. poor cod, rocklings).

| quarter | stomachs | cod | haddock | whiting | Norway pout | unident. Gadidae | non-comm. Gadidae |
|---|---|---|---|---|---|---|---|
| Q1 | 292 | - | - | - | - | 1.21 (age 1) | 1.19 (age 1) |
| Q2 | 2 330 | 0.08-0.60 (age 2: 0.60) | 0.00-0.03 | - | 0.29-0.76 | 0.01-0.40 | 0.00-0.52 |
| Q3 | 2 795 | 0.00-0.09 | - | 0.00-0.06 | 3.41-16.04 (rises with age) | 0.04-0.36 | 0.03-1.14 |
| Q4 | 703 | - | - | - | 5.14-12.08 | - | 0.00-0.02 |

- Cod + haddock + whiting together: Q2 0.08-0.63 %, Q3 0.00-0.12 %, Q1 and Q4 0. The age-0 column
  is filled only in Q4 (n 30); every other value is for mackerel aged 1 to 6+.
- Saithe never appears.
- Prey lengths, Table 10-D [F, OCR]:
  - cod in Q2: 6-7 mm (larvae), 30-35 mm and 50-60 mm (roundfish area 2);
  - cod in Q3: 35-40 mm and 50-60 mm (area 1);
  - Q2 haddock in area 7 (length not readable).
  - These cod are pelagic or settling 0-group.
- Sampling coverage (Table 6-D): no mackerel stomachs in roundfish area 8 ("outside the North Sea") in any
  quarter. The nearest sampled area to the box is area 3 (Q2 103, Q3 259, Q4 75 stomachs).
  - [J, mapping not checked against the report's area map] Area 8 is the Skagerrak/Kattegat and area 3 the
    north-eastern North Sea along the Norwegian Trench.
  - So the 1991 data have **no Skagerrak mackerel stomachs**.
- The report states that unidentified fish were not re-allocated, "the proportional contribution to the
  predators' diet by some or all of the MSVPA species will have been under-estimated to some degree" [F].

### 1.2 Mehl & Westgard 1983 - North Sea 1981-82, 3 674 stomachs [F, full text read]

Source: Mehl, S., Westgard, T. 1983. The diet and consumption of mackerel in the North Sea (a preliminary
report). ICES CM 1983/H:34. https://hdl.handle.net/11250/103794. Local PDF:
`../Fiskgrupper i rutan/predators/data/diet/lit_mackerel/mehl_westgard1983_ICES_CM_1983_H34.pdf`.

- Abstract [F]: "The fish prey items were sandeel, Norway pout, herring, sprat, pearl side, cod, haddock,
  horse mackerel, pipe-fish, dragonet, weever and dab."
- "On a yearly basis fish constituted about 1/3 of the stomach content and about 90% of the fish prey were
  herring, sprat, sand eel and Norway pout" [F].
- Gadoid cells in Tables 5b and 5c (% weight; cells are area x gear x mackerel length; "demersal" = caught by
  bottom trawl) [F]:
  - **Q2:**
    - Gadidae 0-4 cm 0.2 (NE North Sea, pelagic gear, 20-29 cm mackerel, n 10);
    - Gadidae 0-4 cm 0.1 (southern North Sea, demersal gear, 30-39 cm);
    - Norway pout 0-4 cm 3.2 and 5-9 cm 1.9 (NW, demersal, 20-29 cm, n 13).
  - **Q3:**
    - haddock 10-14 cm 2.1 (NW, demersal, 30-39 cm, n 72);
    - Gadidae 5-9 cm 3.3 (same cell);
    - cod 5-9 cm 0.6 and 10-14 cm 1.9 (central, demersal, 30-39 cm, n 198);
    - Gadidae 0-4 cm 0.3 (NW, demersal, 20-29 cm);
    - Norway pout 5-9 cm in most cells, 7.9-22.1 %. The **NE North Sea pelagic** cells have pout at
      13.7-20.7 % (n 81 and 267).
  - Every other cell in Tables 5a (Q1), 5b and 5c has 0 cod, haddock, whiting or saithe. Table 5d (Q4) is
    not in the PDF's text layer and was not checked.
- Reading [J]:
  - The Q3 cod and haddock of 5-14 cm are 0-group near or after settlement. They were eaten by mackerel
    caught **by bottom trawl** in shallow areas (NW and central North Sea, mostly < 100 m).
  - That is the only evidence in the material of mackerel taking gadoids that may already be settled.
  - Weighted over all cells it is well under 1 % of the diet.

### 1.3 Dahl & Kirkegaard 1986 - eastern North Sea (Danish west coast), July 1985 [F, full text read]

Source: Dahl, K., Kirkegaard, E. 1986. Stomach contents of mackerel, horse mackerel and whiting in the
eastern part of the North Sea in July 1985. ICES CM 1986/H:68.
https://www.ices.dk/sites/pub/CM%20Doccuments/1986/H/1986_H68.pdf. Local PDF in
`../Fiskgrupper i rutan/predators/data/diet/lit_mackerel/`.

- 18 Jul - 3 Aug 1985, 11-48 m depth; 336 mackerel and 122 horse mackerel stomachs.
- **Mackerel:** "Herring makes out between 95 to 100% of all identifiable fish species, independent of
  mackerel size groups"; "All the herring identified in the stomachs were O-group fish". No identified gadoid
  in any size class (Table 3); unidentified fish are 11-33 % of the weight in the 20-34 cm classes and 17 %
  in 35-39 cm, so a small gadoid share cannot be ruled out.
- **Horse mackerel** (Table 6): haddock by weight 42.6 % (20-24 cm), 16.0 % (25-29), 4.4 % (30-34),
  26.8 % (35-39), 100 % (40-44, n 2). All of it came from the 11:00 h hauls, i.e. one or two hauls.
  - "the domina-ting fish prey species were in horse mackerel haddock and gobies" for the 20-24 cm class [F].
  - In July in 11-48 m off Jutland these are 0-group haddock near settlement [J].
- "In 1985 very high concentrations of mackerel and horse mackerel were found in the eastern part of the
  North Sea and in Skagerrak" [F]. No Skagerrak stomachs were taken.
- Mackerel fed by day: empty stomachs 36.8 % at 01-06 h vs 7.9 % at 14-16 h [F].

### 1.4 ICES stomach database (stomachdata.ices.dk, CC BY 4.0) [F, raw records analysed]

Local download: `../Torskfiskarnas bytesandelar/data/diet/ices_stomach/`.
- Region box used: 51-62 N, 4 W-13 E.
- "NE North Sea off the Skagerrak entrance" in the table below = east of 7.05 E and north of 57 N. The one
  such haul (57.25 N 7.54 E) is west of the Hanstholm-Lindesnes line, i.e. still North Sea IVa.
- Mackerel (AphiaID 127023) records exist for 2022 and 2025 only.
- Horse mackerel (126822) records exist for 1991 (no prey weights) and 2023.

| predator | when | area | stomachs | prey g | gadoid prey |
|---|---|---|---|---|---|
| mackerel | Jun 2022 (18 hauls, FOT) | central and eastern N Sea | 954 | 1 923 | **whiting 26.7 g = 1.4 %** (16 fish, 39-65 mm) |
| mackerel | 19 Jun 2022 | NE North Sea off the Skagerrak entrance, 57.25 N 7.54 E, 59 m | 100 | 160 | none |
| mackerel | Jan-Feb 2025 | northern N Sea | 157 | 15 (dry wt) | Norway pout 6.2 cm, 0.36 g |
| horse mackerel | Jul-Aug 2023 (GOV) | N Sea | 140 | 95 | whiting 1.75 g (60 mm, in an 18 cm fish), Gadidae 0.61 g (4 fish) = **2.5 %** |
| horse mackerel | Aug 2023 | Skagerrak/Kattegat | 59 | 3.6 | none |

- Both 2022 hauls with whiting were shallow:
  - 14 of the 16 whiting came from one haul (12 Jun, 56.43 N 7.75 E, 28 m, off Thyboron). There whiting was
    37 % of that haul's prey; the mackerel were 28-40 cm.
  - The other 2 were in a haul at 54.82 N 6.07 E (41 m).
- Mackerel diet share of whiting over all 2022 hauls: 26.7 / 2 083 g = **1.3 %**.
- The 2022 Q2 sample is from sandeel grounds. Earlier notes in this repo flag a large unidentified fraction
  (`../Fiskgrupper i rutan/predators/evidence_C_diet_consumption.md`, section 2.1).

### 1.5 DAPSTOM (Cefas) [F, presence records only]

Portal download, 2026-11-13 (no prey mass):
- **mackerel IVa:** 72 stomachs, 55 with food; 1979 May and 2006 Jul; no gadoids.
- **mackerel IVb:** 328 stomachs, 239 with food; 1907-2006, mostly 2004-06; one 32 cm mackerel (5 Jul 2004)
  held three "whiting larvae" of 36, 41 and 43 mm.
- **IIIa:** no records for either species.
- **horse mackerel IVa:** no records.
- **horse mackerel IVb:** 12 stomachs, none with gadoids.
- Frequency of occurrence of gadoids: 1 of 294 fed mackerel stomachs.

### 1.6 Ecopath North Sea 1991 (Mackinson & Daskalov 2007) [S]

Cefas Sci. Ser. Tech. Rep. 142; diet matrix compiled from the 1991 Year of the Stomach. The diet matrix was
read from EcoBase model 457 (`../Fiskgrupper i rutan/benthivores/data/ecobase/model_457.xml`); the report's
text was not read for this question.

| | juvenile cod | juvenile whiting | juvenile haddock | juvenile saithe | Norway pout | other small gadoids |
|---|---|---|---|---|---|---|
| mackerel | 0.0001 | 0.0000 | 0 | 0 | 0.0094 | 0.0114 |
| horse mackerel | 0.0056 | 0.0028 | 0.0009 | 0.0011 | 0.2235 | 0.0340 |

### 1.7 Norwegian Sea studies (context, no North Sea gadoids)

- Bachiller, E., Skaret, G., Nottestad, L., Slotte, A. 2016. Feeding ecology of Northeast Atlantic mackerel,
  Norwegian spring-spawning herring and blue whiting in the Norwegian Sea. PLoS ONE 11: e0149238.
  doi:10.1371/journal.pone.0149238.
  - Full text on disk (`../Fiskgrupper i rutan/predators/data/diet/lit_mackerel/bachiller2016_plosone.pdf`);
    grepped for gadoid prey, not read in full: none found. "Calanoid copepods dominated" mackerel diets [F].
- Langoy, H., Nottestad, L., Skaret, G., Broms, C., Ferno, A. 2012. Overlap in distribution and diets of
  Atlantic mackerel, Norwegian spring-spawning herring and blue whiting in the Norwegian Sea during late
  summer. Marine Biology Research 8: 442-460. doi:10.1080/17451000.2011.642803.
  - DOI and metadata verified via Crossref; full text not read. Offshore Norwegian Sea, outside the gadoid
    nurseries [J].
- Neither is a source for 0-group gadoids.

### 1.8 Jellyfish-associated whiting

- Whiting in Scandinavian waters associate with *Cyanea capillata*: "young gadiid fishes, especially whiting
  ... and young horse-mackerel" [F].
  - Source: Dahl, E. 1961. Sarsia 3: 47-55. doi:10.1080/00364827.1961.10410249. Full text read; PDF saved
    as `papers/10.1080_00364827.1961.10410249.pdf`. Observations off Bergen, July 1960, whiting 3-9 cm.
  - Dahl quotes Malm (1852) that young horse mackerel "attack the jelly-fish" [S].
- Hay, S.J., Hislop, J.R.G., Shanks, A.M. 1990. Neth. J. Sea Res. 25: 113-130.
  doi:10.1016/0077-7579(90)90013-7. Abstract via secondary sources only [S]:
  - 0-group whiting and haddock shelter under scyphomedusae in June-July (ICES 0-group gadoid surveys
    1971-86);
  - no influence of *Cyanea* on survival to age 1 was found.
- Lynam, C.P., Brierley, A.S. 2007. Mar. Biol. 150: 1397-1401. doi:10.1007/s00227-006-0429-7. Abstract only
  [S]: jellyfish abundance may lower 0-group whiting mortality.
- No study was found that measures mackerel or horse mackerel predation on jellyfish-associated whiting.
  [J] The shelter is a pelagic-phase refuge against fish predators in general. The 2022 whiting in mackerel
  stomachs show that it is not complete.

## 2. SMS (ICES WGSAM North Sea key runs)

Sources:
- key-run files at https://github.com/ices-eg/wg_WGSAM (`NorthSeaKeyRun_2017`, `_2020`, `_2023`;
  `partial_m2.out`, `species_names.in`, `summary_stom.out`, `incl_stom.in`, `season_overlap.in`);
- stock annexes `StockAnnex/NorthSea/2017_North_sea.pdf`, `2020_NorthSea.pdf`, `2023_NorthSea.pdf` (full
  text searched).

### 2.1 Predator and prey lists as the key run gives them [F, verbatim]

2023 key run summary sheet (stock annex):
- "Predatory species: Assessed species: Cod, haddock, saithe, whiting, mackerel. Species with given input
  population size: North Sea horse mackerel, western horse mackerel, grey gurnard, starry ray, hake, fulmar,
  gannet, great black backed gull, guillemot, herring gull, kittiwake, puffin, razorbill, grey seal, harbour
  porpoise".
- "Prey species: Cod, haddock, herring, Norway pout, southern North Sea sandeel, northern North Sea sandeel,
  sprat, whiting and plaice".
- "predators and prey (cod, whiting, haddock), prey only (herring, sprat, northern and southern sandeel,
  Norway pout), predator only (saithe, mackerel)".

The annex does not tabulate pairs. The pairs are set by the stomach data that each predator has
("predator/preys size range defined from observations").

### 2.2 Pairs with non-zero M2, derived from `partial_m2.out` [SMS estimate; derived by me]

| key run | predator | prey with M2 > 0 (all years) |
|---|---|---|
| 2017 | mackerel | herring, N. sandeel, S. sandeel, Norway pout, sprat |
| 2017 | N. horse mackerel | herring, N. sandeel, sprat |
| 2017 | W. horse mackerel | N. sandeel, Norway pout |
| 2020 | mackerel | herring, N. sandeel, S. sandeel, Norway pout, sprat |
| 2020 | N. horse mackerel | herring, N. sandeel, sprat, **whiting** |
| 2020 | W. horse mackerel | N. sandeel, Norway pout |
| 2023 | mackerel | herring, N. sandeel, S. sandeel, Norway pout, sprat |
| 2023 | N. horse mackerel | herring, N. sandeel, sprat |
| 2023 | W. horse mackerel | N. sandeel, Norway pout |

Partial M2 on gadoids, annual sum over quarters, mean over the last ten years of each run:

| run (years) | pair | age 0 | age 1 | quarters |
|---|---|---|---|---|
| 2023 (2013-22) | mackerel -> Norway pout | **0.082** | 0.004 | Q3 0.068, Q4 0.014; age 1 Q1-Q2 |
| 2023 (1974-2022) | mackerel -> Norway pout | 0.084 | 0.006 | |
| 2020 (2010-19) | mackerel -> Norway pout | 0.161 | 0.001 | |
| 2017 (2007-16) | mackerel -> Norway pout | 0.140 | 0.001 | |
| 2023 (1974-2022) | W. horse mackerel -> Norway pout | 0.018 | 0 | 0 since 2013 |
| 2020 (2010-19) | **N. horse mackerel -> whiting** | **0.30** | 0 | Q3 only, from the largest horse mackerel size class |
| 2023 | N. horse mackerel -> whiting | 0 | 0 | pair absent |

- No run has mackerel or horse mackerel on cod, haddock or saithe.
- Saithe is not an SMS prey species.
- Context (SMS estimate, 2023, 2013-22 mean), total age-0 M2:
  - **Norway pout 1.14**; by predator: hake 0.32, whiting 0.24, saithe 0.23, gurnards 0.18, **mackerel 0.08**;
  - cod 2.17;
  - whiting 1.77 (gurnards 1.16);
  - haddock 0.97.
- In the 2020 run the N. horse mackerel whiting take was 13 % of whiting's age-0 M2 of 2.27.

### 2.3 The 2020 / 2023 horse mackerel -> whiting discrepancy (unresolved)

- The 2023 stomach input still has whiting in N. horse mackerel stomachs [F, `summary_stom.out`]:
  - Q3 1987 and 1991, prey 5.5-7.5 cm;
  - 0.5-10 % of stomach content by weight per size class;
  - N-weighted Q3 share 2.9 %.
  - SMS's fitted stomach share for these rows is non-zero (stomcon.hat 0.07-0.18).
- `incl_stom.in` includes these samples in both runs, and `season_overlap.in` is 1 for this pair in both.
- Yet `partial_m2.out` and `who_eats_whom_level2.csv` / `who_eats_whom_combined.csv` of the 2023 run have no
  horse mackerel -> whiting flow.
- I did not find the cause. A size-range or suitability change in the 2023 configuration is the likely
  place [J].
- The annex text explains neither the 2020 value nor its removal.

### 2.4 SMS stomach input for mackerel (1981 + 1991, area 1 = North Sea) [F]

`summary_stom.out`; weight shares after SMS's bias correction for evacuation rate; N-haul weighted.

| quarter | other food | Norway pout | S. sandeel | herring | sprat | N. sandeel |
|---|---|---|---|---|---|---|
| Q1 | 1.000 | | | | | |
| Q2 | 0.585 | 0.002 | 0.352 | 0.006 | 0.037 | 0.018 |
| Q3 | 0.808 | 0.021 | 0.026 | 0.066 | 0.047 | 0.032 |
| Q4 | 0.800 | 0.132 | 0.012 | 0.017 | 0.039 | |

- No cod, haddock or whiting rows exist for mackerel, although CRR 219 shows them at < 1 % (section 1.1).
- [J] Either such small shares were dropped when the diet was compiled, or they went to "other food". SMS
  therefore carries no mackerel pair on them whichever way the raw data are read.
- Norway pout prey lengths:
  - Q3: 5.5-13.5 cm (mackerel 27-45 cm);
  - Q4: 5.5-17.5 cm;
  - Q2: 7.5-9 cm (1991 only).
- 2013 mackerel stomachs exist but are not used. The 2020 annex says: "the length of the prey items was not
  recorded, and therefore, the data cannot be used without assigning the prey types to specific length
  groups" [F].

## 3. Timing and depth of the pelagic 0-group

| species | pelagic phase (northern North Sea / Skagerrak) | depth while pelagic | settlement | sources |
|---|---|---|---|---|
| cod | larvae and pelagic juveniles Apr-Jun; 3.9 cm mean on 17-20 June 1991 at 60 N | diel vertical migration, near the surface in the evening, few near the bed | central North Sea July 1994: 4-6 cm demersal by day, pelagic at night; "Cod of over 8 cm in length were wholly demersal". Settled at 49 mm (Stonehaven). Skagerrak coast: settle at 3-5 cm in June (one source July-August) | [S] Bromley et al. 1997; [S] Bromley & Kell 1999 abstract; [S] Bastrikin et al. 2014; [S] web snippets on Skagerrak coastal cod (not read) |
| haddock | pelagic to ~July; 4.1 cm (1.7-7.3) in mid-June 1991 | as cod | 78 mm (one ~2-week pulse, second half of May, at Stonehaven); offshore July; Norwegian Deep 0-group at 80-300 m from late July | `../Sejens diet på unga torskfiskar/evidence_settlement.md` entries 6-13 |
| whiting | pelagic longest; 3.2 cm mean in mid-June 1991; with jellyfish June-August | upper water column, under medusae | 85 mm; "protracted ... from early June to early August" at Stonehaven | [S] Bastrikin et al. 2014 (abstract via secondary); [S] Hay et al. 1990 |
| saithe | spawned Jan-Mar on the northern shelf edge; 4.6 cm pelagic in mid-June 1991 at 60 N; drifts to the Norwegian coast | near surface | coastal nurseries in summer; the Skagerrak coast is "the only part where the occurrence is not regular" | [S] Bromley et al. 1997; [S] old IMR saithe survey text via search snippet |
| Norway pout | spawned Jan-Apr; 4.8 cm in mid-June 1991 | pelagic, then semi-pelagic over 50-250 m | no sharp settlement; age 0 is eaten by mackerel in Q3-Q4 at 5.5-13.5 cm | [S] Bromley et al. 1997; [F] SMS stomach file |

- Bromley, P.J., Watson, T., Hislop, J.R.G. 1997. ICES J. Mar. Sci. 54: 846-853. doi:10.1006/jmsc.1996.0211.
  Read by an earlier agent (PDF in `../Sejens diet på unga torskfiskar/data/settlement/`): 60 N 0 E, 147 m
  water, 17-20 June 1991; "Few fish were caught near the bottom".
- Bastrikin, D.K., Gallego, A., Millar, C.P., Priede, I.G., Jones, E.G. 2014. ICES J. Mar. Sci. 71:
  2101-2113. doi:10.1093/icesjms/fsu029. Coastal Stonehaven, not offshore. Not open access.
- The ICES 0-group gadoid survey (June-July, 1971-86) fished the young gadoid pelagic trawl in three stages:
  near the bed for 0-20 min, then at mid-water or the thermocline, then "close to the surface (5-10 m)".
  Source: Lynam, C.P., Hay, S.J., Brierley, A.S. 2004. Limnol. Oceanogr. 49: 637-643.
  doi:10.4319/lo.2004.49.3.0637. Full text read; PDF saved as `papers/10.4319_lo.2004.49.3.0637.pdf` [F].
- **Overlap with the model's mackerel** (in the box May-October at 0-50 m, peak August) [J]:
  - Cod, haddock and saithe are pelagic in the upper layer while mackerel arrive (May-June) and leave it in
    June-July. Overlap is short and early, when mackerel numbers are still low.
  - Whiting stays pelagic into August, under jellyfish, so it overlaps longest. It is also the gadoid seen in
    recent stomachs.
  - Norway pout age 0 overlaps all summer. Over beds > 100 m it sits deeper than 50 m by day [J].
  - Mackerel feed by day (1.3). Pelagic 0-groups rise toward the surface in the evening (Bromley et al.
    1997). So the overlap is real but brief.
- **Settled recruits at the bed:** the box beds are 44-450 m (median 143 m) and mackerel stay at 0-50 m.
  - Settled cod, haddock and whiting are out of reach over nearly every cell.
  - The only gadoids of near-settled size that mackerel ate (Mehl & Westgard 1983, Q3 cod and haddock
    5-14 cm) came from mackerel caught by bottom trawl in shallow areas.

## 4. What follows for the model [J]

1. **Inside the recruit import.**
   - The model's benthivore and whiting recruits are imported as SMS recruits x box share. Their bed-depth
     shape comes from GOV hauls of settled fish (haddock age 0, cod age 1).
   - The mackerel predation found here is almost all on pelagic 0-groups in May-July, before those fish are
     counted. It is already netted out of the import.
   - Q2 cod in mackerel (1991) and June whiting (2022) are before the SMS recruitment season (SMS recruits
     age 0 in Q3, `rec.season` 3). They are outside SMS's own M2 and inside its recruitment.
   - Q3 whiting predation (July-August) falls after the SMS recruitment date. SMS has no mackerel pair for
     it, so it sits in M1 or is absorbed into the fitted recruitment.
   - At 0.00-0.06 % of the Q3 mackerel diet (1991) it does not move anything.
2. **Post-settlement predation at the bed: none worth a pair.**
   - Mackerel at 0-50 m over 44-450 m beds do not meet settled recruits. The one hint (Mehl & Westgard Q3,
     bottom-trawled mackerel in shallow areas) is < 1 % of the diet and comes from beds the box hardly has.
   - Horse mackerel take haddock near settlement in very shallow coastal water (11-48 m, Dahl & Kirkegaard).
     The box's shallowest cells are 44 m. Horse mackerel are 0.06 of the group and the 2023 Skagerrak horse
     mackerel stomachs held no gadoids.
3. **Diet share.**
   - Cod + haddock + whiting + saithe in the mackerel group's diet: central **~0.3 %**, range 0-1.5 % by
     weight over the season, concentrated in June-July.
   - The range spans the 1991 Q2-Q3 values (0-0.6 %) and the 2022 June value (1.3 %, a shallow eastern haul).
   - The horse mackerel part (Ecopath 1.0 %, stomachs 2.5 % in Q3 2023) adds 0.06 x 0.01-0.03 = 0.001-0.002.
   - Order of magnitude, using library values as inputs (0.85 t/km2 box mean, ration ~2.75 /yr):
     0.003 x 2.3 ~ 0.007 t/km2/yr of 2-3 g fish. That is a few % of the whiting import (0.172 t/km2/yr) if
     all of it were whiting. But it is pre-settlement, so it is already netted out [J].
4. **Norway pout.**
   - Norway pout is the real gadoid prey of mackerel: SMS age-0 M2 0.08 /yr (2023) to 0.16 /yr (2020).
     Stomach shares (1991) are 3-16 % in Q3 and 5-12 % in Q4; Mehl & Westgard found 14-21 % in the NE North
     Sea pelagic Q3 cells.
   - Pout sits in the model's planktivore group. The library's mackerel -> planktivores pair counts herring
     and sprat only, and `../Makrillens bytesstorlek/summary.md` set its gape-weighted pout share at 0.04 of
     the August pool.
   - This is the one place where this question touches the library. I recommend nothing here. The decision
     belongs with that pair's booking.

## 5. Gaps and disputed points

- **No Skagerrak mackerel stomachs with gadoid data.**
  - The 1991 programme sampled none in the area outside the North Sea, and 1981-82 and 1985 have no
    Skagerrak samples either.
  - The one 2022 haul off the Skagerrak entrance (NE North Sea, 100 fish, June) had no gadoids.
  - The 2023 Skagerrak horse mackerel (59) had almost no food.
- **SMS 2020 vs 2023 horse mackerel -> whiting** (0.30 /yr vs absent) is unexplained (section 2.3). Both runs
  use the same 1987/1991 stomach rows.
- **CRR 219 vs SMS for mackerel.** The raw 1991 tables show cod, haddock and whiting at < 1 %, but SMS's
  stomach input for mackerel has no such rows. They are two separate facts: small shares in the raw data, no
  pair in SMS.
- **Under-sampling of small prey.**
  - Fish prey that are digested fast and unidentified ("unidentified fish was not redistributed") are
    under-counted.
  - SMS's length distributions under-represent fish < 5-7 cm (annex 3.3.3: "fish smaller than 5-7 cm are
    hardly caught").
  - Both bias the 0-group share down. A factor 2-3 would still leave it at ~1 % [J].
- **The 2013 North Sea mackerel diet data** were not used by SMS because prey lengths were missing; I did not
  find them published.
- **Settlement timing in the Skagerrak** is supported only by coastal and secondary sources (cod 3-5 cm in
  June vs July-August). No offshore Skagerrak pelagic 0-group survey was read.
- **Not read in full:**
  - Bastrikin et al. 2014, Hay et al. 1990 and Lynam & Brierley 2007 (paywalled; Unpaywall: no OA copy);
  - Langoy et al. 2012;
  - Bromley & Kell 1999 (abstract only).
- **Roundfish-area mapping** in CRR 219 (area 8 = Skagerrak/Kattegat, area 3 = NE North Sea) follows the
  standard numbering. The report's own map was not checked.

## 6. Data and files used

- SMS 2023: `../Sejens diet på unga torskfiskar/data/sms_keyrun2023/summary_stom.out` (identical to the
  GitHub file) and `who_eats_whom_level2.csv`.
- `partial_m2.out`, `species_names.in`, `incl_stom.in`, `season_overlap.in`, `who_eats_whom_combined.csv`
  (2017/2020/2023) and the three stock annexes were downloaded to the scratchpad from
  https://github.com/ices-eg/wg_WGSAM. They are not kept in the repo.
- ICES stomach database: `../Torskfiskarnas bytesandelar/data/diet/ices_stomach/x*/`.
- DAPSTOM: portal endpoint `/umbraco/CefasPlugins/DapstomSurface/Download?searchType=predator&species=MAC|HOM`
  with areas IVa, IVb and IIIa, fetched 2026-11-13. Not kept.
- PDFs saved here: `papers/10.1080_00364827.1961.10410249.pdf` (Dahl 1961) and
  `papers/10.4319_lo.2004.49.3.0637.pdf` (Lynam et al. 2004).
