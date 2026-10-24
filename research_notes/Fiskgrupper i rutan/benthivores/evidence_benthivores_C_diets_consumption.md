# Evidence C - benthivorous_fish: diets by model pool, consumption (Q/B), gape

Scope: diet by mass, mapped to the split model pools, and Q/B for the members of `benthivorous_fish`.
Long-run 2001-25 biomass shares (`../split_brief.md`): haddock 38.6 %, cod 31.4, long rough dab (LRD)
12.6, dab 4.5, plaice 3.8, witch 1.7, starry ray 1.5, hagfish 1.3, grey gurnard 0.9, lemon sole 0.9,
greater weever 0.8, others ~2.5. Diets are CHECKS of mechanism-based parameters and the basis of a
consumption estimate (for M2). They are never fitted targets.

Tags: [F] full text read / data downloaded and parsed; [A] abstract only; [S] secondary (citing source
named); [D] computed here. **PRE-2000** rows describe another ecosystem state and are context only.

Pools: **B** = `benthic_community`; **Z** = `zooplankton` (euphausiids, copepods, pelagic hyperiids);
**P** = `planktivorous_fish` (herring, sprat, Norway pout, blue whiting, argentines); **W** =
`predatory_fish` (whiting, saithe, mackerel, hake, ...); **Bf** = `benthivorous_fish` itself (haddock,
cod, flatfish, rays, gurnards, hagfish; within-group, OUT of the budget); **Oth** = no pool (sandeels,
gobies, dragonets, poor cod/rocklings, sculpins, cephalopods, unidentified fish, carrion). Pelagic
shrimps (*Pasiphaea*) were kept apart; they are 0 in every row here.

Files: scripts in `scripts/`, tables in `tables/`, inputs (judgement values, stated) in `inputs/`,
downloads in `data/` (README lines there).

## 1. Bottom line

Diet shares by mass, % of identified prey, mapped to the pools. "Used" rows are the values carried into
the group consumption (`inputs/diet_species.csv`); where a row is a judgement, it says so.

| Species (biomass share) | B | Z | P | W | Bf | Oth | n, area, period, tag | Q/B /yr low / central / high |
|---|---|---|---|---|---|---|---|---|
| **Haddock (38.6 %)**, measured | 82.6 | 0.4 | 0 | 0 | 0.1 | 16.9 | 485 fish, NNS + W Norwegian Trench, Jan-Feb 2025, ICES-DB [F, D] | |
| Haddock, used (judgement) | 80 | 5 (0.5-11) | 3 | 0 | 1 | 11 | annual Z from Albert 1994 summer 2-22 % (PRE-2000) | 2.35 / **3.8** / 4.75 |
| **Cod (31.4 %)**, box-weighted, used | 49.3 | 0.2 | 22.5 | 5.1 | 11.8 | 11.3 | 66 fed, Skagerrak 2023-25 Q1 + Aug, ICES-DB; unid. gadoids allocated (rule, section 2) [D] | 2.17 / **3.2** / 5.26 |
| Cod, alternative | 28.6 | 0 | 43.1 | 8.6 | 13.7 | 6.0 | 77 fed, NNS Jan-Feb 2023, ICES-DB [D] | |
| **LRD (12.6 %)**, used (judgement) | 85 | 2 | 3 | 2 | 3 | 5 | M&D 1991 diet from Ntiba & Harding 1993 (B 87) [S]; box LRD median 17.5 cm, gape (section 4) | 3.0 / **3.4** / 4.0 |
| LRD, upper bound for fish and Z | 53.6 | 15.4 | 6.4 | 8.3 | 4.1 | 12.2 | 572 fish (61 pooled records) 20-30 cm, central/southern N Sea, 1991, PRE-2000, ICES-DB [D] | |
| LRD, NW North Sea 1991 | 18.0 | 0 | 20.2 | 31.3 | 22.2 | 8.2 | 297 fish (48 records), 209 g, mostly August 0-group gadoids 5-8 cm, PRE-2000 [D] | |
| **Dab (4.5 %)** (judgement) | 95 | 2 | 0 | 0 | 0 | 3 | M&D 1991 B 98.7, Z 1.3 (from De Clerck & Torreele 1988) [S]; no in-area data | 3.36 / **3.5** / 4.2 |
| **Plaice (3.8 %)** | 99.5 | 0 | 0 | 0 | 0 | 0.5 | 185 fed, Skagerrak Jan + Aug 2024, 21-226 m, ICES-DB [D] | 2.8 / **3.42** / 3.42 |
| **Witch (1.7 %)** | 99.6 | 0.2 | 0 | 0 | 0 | 0.2 | 134, Skagerrak 156-490 m, Mar-Oct 1991-96, Bergstad et al. 2003 Table 6 [F] (PRE-2000 sampling) | 2.7 / **3.0** / 4.84 |
| **Starry ray (1.5 %)**, used | 44 | 0 | 10 | 12 | 7 | 27 | 175 adults (mean 55.5 cm), Norwegian Deep + Skagerrak 1984-96, Skjaeraasen & Bergstad 2000 via FishBase [S] | 1.5 / **1.7** / 2.3 |
| Starry ray, ICES | 61.4 | 1.4 | 0 | 2.5 | 27.4 | 7.3 | 39 fed (median 41 cm), Skagerrak Aug 2024, ICES-DB [D]; Bf = one hagfish | |
| **Hagfish (1.3 %)** (judgement) | 20 | 0 | 0 | 0 | 0 | 80 | scavenger; Malangen Ecopath diet 80 % detritus/carrion [F, EcoBase 751] | 2.0 / 4.0 / 6.4 (judgement) |
| **Grey gurnard (0.9 %)** (judgement) | 30 | 2 | 10 | 15 | 12 | 31 | de Gee & Kikkert 1993 via FishBase (fish 61-92 % at 20-35 cm) [S]; Floeter 2005 (gadoids 37 % of weight > 20 cm) [F]; PRE-2000 | 3.0 / **3.2** / 4.67 |
| **Lemon sole (0.9 %)** | 100 | 0 | 0 | 0 | 0 | 0 | Rae 1965 via FishBase (polychaetes, cnidarian polyps) [S]; PRE-2000 | 3.8 / **4.32** / 4.32 |
| Greater weever, others (3.3 %) (judgement) | 75 | 0 | 5 | 0 | 0 | 20 | no source | 3.0 / 3.5 / 4.5 |

**Pooled group** (`scripts/group_diet.py`, `tables/group_diet_consumption.csv`, `tables/qb_pooled.csv`) [D]:
- **Q/B: central 3.5 /yr (low 2.5, high 4.7); 3.0-3.5 /yr is the defensible central range at the
  box's 6-9 C** (section 5).
- **Consumption-weighted diet, central: B 72 %, Z 2.5 %, P 8.4 %, W 1.9 %, Bf 4.3 %, Oth 10.9 %.** Per
  tonne of group biomass per year: B 2.5 t, Z 0.09 t, P 0.29 t, W 0.07 t (within group 0.15 t, no pool
  0.38 t).
- Sensitivity of the P share: 6.5 % (unidentified gadoids not allocated), 14.3 % (cod diet from NNS 2023
  instead of the Skagerrak), 14.7 % (NNS cod and the LRD upper bound together). **The P share rests almost
  entirely on cod.** Z ranges 0.6 % (haddock winter diet all year) to 5.0 % (haddock high summer krill) and
  4.1 % with the LRD upper bound. B stays 62-73 % in every variant.

## 2. Cod: the fish part of the diet by prey species (post-2000, ICES stomach database) [D]

Data: public ICES stomach download (`../../Torskfiskarnas bytesandelar/data/diet/ices_stomach/`), prey
wet weight; areas as in the earlier gadoid file. Skagerrak = Danish IBTS Q1 + Aug 2023 (4 hauls) and one
Swedish August 2023 haul in the box, plus 2024-25 Danish hauls (66 fed stomachs, 20-257 m). NNS =
German and Danish IBTS Q1 2023 (77 fed, 56-162 m). Kattegat 2021-25 (1007 fed, shallow, 6-73 m: not the
box habitat). Scripts `scripts/benth_diet_pools.py`, `scripts/cod_fish_pools.py`; table
`tables/cod_fish_prey_pools.csv`. Fish prey as % of the whole identified diet: Skagerrak 67 %, NNS 75 %,
Kattegat 63 %.

**Reconciliation with `../../Torskfiskarnas bytesandelar/evidence_C_diet_check.md`** (Skagerrak 2023: 42
fed, 614 g, Norway pout 29.9 % of the diet). My Skagerrak 2023 row has 52 fed and 729 g. The 10 extra
stomachs (115 g) are one Danish haul at 57.47 N 8.25 E, just east of the Hanstholm-Lindesnes line in my
area test; the earlier file put it outside the Skagerrak. A first run of my script also sent the
misspelt WoRMS name "*Trisopterus esmarki*" (a separate AphiaID used in the 2023 uploads) to
"unidentified gadoids"; corrected, Norway pout is 30 % of the Skagerrak 2023 diet, as in the earlier
file. (The same pattern, `"Trisopterus esmarkii" in s`, is in `../predators/scripts/predator_diet_pools.py`
and may under-count Norway pout there.)

Shares of the identified FISH mass:

| Area (fish items) | Cod cm | Norway pout | herring | sprat | blue whiting | whiting | haddock | cod | flatfish | sandeel | unid. gadoid (Gu) | unid. fish (Fu) | other no-pool fish |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Skagerrak (12) | < 40 | 4.9 | 0 | 0 | 0 | 12.9 | 0 | 0 | 58.4 | 0 | 0.1 | 3.2 | 20.5 (dragonet) |
| Skagerrak (12) | 40-49 | 0 | 0 | 0 | 0 | 18.7 | 0 | 0 | 0 | 0 | 65.9 (*Trisopterus* sp.) | 10.7 | 3.5 |
| Skagerrak (29) | 50-69 | 46.4 | 1.6 | 7.6 | 0 | 0 | 0 | 12.2 | 6.2 | 0 | 0 | 13.4 | 12.5 (sculpin) |
| **Skagerrak (53)** | **all** | **36.5** | 1.3 | 5.9 | 0 | 3.7 | 0 | 9.5 | 9.2 | 0 | 9.7 | 12.2 | 11.8 |
| NNS (26) | < 40 | 74.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7.9 | 6.6 | 1.5 | 9.2 (other P, other) |
| NNS (13) | 40-49 | 33.8 | 0 | 0 | 0 | 42.2 | 0 | 0 | 16.7 | 0 | 3.8 | 3.6 | 0 |
| NNS (58) | 50-69 | 42.4 | 0 | 0 | 0 | 0 | 49.1 | 0 | 0.8 | 0 | 1.4 | 1.3 | 4.9 |
| NNS (22) | >= 70 | 17.5 | 4.8 | 0 | 25.6 | 20.5 | 20.6 | 0 | 0 | 0 | 3.6 | 0.2 | 7.2 |
| **NNS (119)** | **all** | **35.6** | 1.6 | 0 | 8.4 | 10.8 | **31.6** | 0 | 2.0 | 0.5 | 2.7 | 1.2 | 5.6 |
| **Kattegat (670)** | all | 0 | 13.3 | 16.7 | 0 | 10.1 | 0 | 30.1 | 8.5 | 0.1 | 0.5 | 7.5 | 13.3 (gobies 7.2) |
| NNS 1981, PRE-2000 (242) | all | 19.8 | ~0 | ~0 | 0 | 10.7 | 33.1 | 0 | 4.3 | 25.1 | - | - | - |

Mapped to the model pools, % of fish mass. **Gu rule (a stated judgement):** unidentified gadoids are
split over Norway pout (P), whiting (W) and haddock + cod (Bf) in proportion to the identified gadoid prey
of the same area and cod size class (all sizes of the area if none is identified there).

| Area | P | W | Bf | no pool (sandeel, gobies, dragonets, sculpins, Fu) | Gu | P / W / Bf with Gu allocated |
|---|---|---|---|---|---|---|
| Skagerrak 2023-25 | 43.7 | 3.7 | 18.9 | 24.0 | 9.7 | **50.8 / 4.4 / 20.8** |
| NNS 2023 | 47.3 | 11.6 | 33.7 | 4.8 | 2.7 | **48.5 / 12.0 / 34.8** |
| Kattegat 2021-25 | 30.7 | 11.3 | 38.7 | 18.8 | 0.5 | 30.7 / 11.4 / 39.0 |

So **Norway pout is the main fish prey of cod in and near the box (36 % of fish mass); herring and sprat
together are only 2-7 %.** Within-group fish (haddock, cod, flatfish) is 19-35 %, whiting 4-12 %.

Size of cod that eats each prey (ICES-DB, all areas and years, fish prey with both lengths;
`tables/ices_prey_pred_length_ratio.csv`) [D]:
- Norway pout: 136 items, cod median 62 cm, prey median 10 cm (max 16), prey/cod length median 0.16.
  In the Skagerrak almost only cod >= 50 cm ate Norway pout (one item below 40 cm); in the NNS cod of
  25-80 cm did.
- Herring: 1110 items (mostly Kattegat/Baltic), cod median 46 cm, prey median 15 cm, ratio median 0.31,
  p90 0.42. In the Skagerrak and NNS herring is <= 5 % of fish mass at every cod size. CINEA 2024 [F]: in
  the 2023 North Sea samples herring was found only in 80-90 cm cod (p. 44-45).
- Sprat: cod median 41 cm, prey median 9 cm. Whiting: cod median 50 cm, prey 14 cm. Haddock: cod
  median 70 cm, prey 10 cm (NNS cod 50-80 cm). Flatfish: cod median 46 cm, prey 9 cm. Gobies: cod median
  34 cm, prey 3 cm.
- Sandeel: absent from the Skagerrak sample, 0.5 % in NNS 2023, 25 % in NNS 1981.

Cod whole diet, biomass-weighted over the box cod length mix (`tables/cod_diet_box_weighted.csv`; mix
from `../../Torskfiskarnas bytesandelar/tables/length_class_pct.csv`, A_floor pooled: < 30 cm 18 %, 30-39
27 %, 40-49 22 %, 50-69 24 %, >= 70 9 %; a class with < 5 fed stomachs borrows the neighbouring class):

| Area | B | Z | P | W | Bf | Sand + other no-pool fish + Fu | Gu |
|---|---|---|---|---|---|---|---|
| Skagerrak 2023-25 | 49.3 | 0.2 | 15.8 | 4.4 | 10.1 | 11.3 | 9.0 |
| NNS 2023 | 28.6 | 0 | 41.9 | 8.2 | 12.7 | 5.6 | 2.6 |

Caveats: 53 and 119 fish items; many large items; mostly Q1 (NNS Jan-Feb only; Skagerrak Jan-Feb +
August). Pooled over stomachs, not catch-weighted. Kattegat cod are small and shallow; contrast only.

Literature on cod fish prey after 2000 (all thin for the area):
- CINEA 2024, North Sea stomach study 2022-23, doi:10.2926/683598 [F]: new cod stomachs IIIa 12, IVa 37,
  IVb 144, IVc 5; by number: "sprat (2.90%), herring (1.93%), whiting (1.74%) and haddock (1.16%)" (p. 25);
  "The highest percentage of cannibalism in cod was found in area IIIa (4.3%)" (p. 27).
- Hüssy, Andersen & Pedersen 2016, J Appl Ichthyol 32:928, doi:10.1111/jai.13160 [A]: NE central North
  Sea, Aug-Sep 2009-11; crustaceans fall with cod size "whereas that of fish and predominantly herring
  increased". No shares in the abstract.
- Floeter & Temming 2003, CJFAS, doi:10.1139/f03-007 [A]: "More than 75% of fish found in the stomachs
  of North Sea cod originated from the least preferred quartile of the prey size range".
- No post-2000 Skagerrak or Kattegat cod paper by prey species was found; the SMS 2023 key run uses cod
  stomachs from 1981, 1985-87 and 1991 only [F, `data/lit/sms2023_sizepref/summary_stom.out`].

## 3. Haddock: krill, copepods and fish

**Measured, in area, post-2000** (ICES-DB, IMR NS-IBTS Q1, Jan-Feb 2025; 485 fish in 306 records, 45
hauls, 66-211 m; this re-run reproduces the earlier file's 485 fish and 755 g) [F, D]:
- B 82.6 %. The earlier file's 79.1 % is reproduced if the 32 g of unidentified crustacea stay in the
  denominator (590 g / 747 g); its stated rule says they were excluded (inferred, not confirmed).
- Euphausiids 0.4 % (3 g, 2 items). No copepods.
- Fish: unidentified teleosts 7 % (50 g, 10 items, mostly in 40-49 cm haddock), one flatfish (0.8 g), one
  sculpin *Artediellus atlanticus* (1.2 g). No sandeel, Norway pout or clupeid. Prey fish 4-7 cm
  (prey/predator length 0.11-0.21, n 6).
- Cephalopods 9.6 % (3 items; noisy).

**Summer (Q3) krill or copepods after 2000: no source exists that I or the literature agent could find**
(not by mass, number or frequency):
- Schückel et al. 2010, J Fish Biol 77:98, doi:10.1111/j.1095-8649.2010.02657.x [A] (closed): northern
  North Sea 2007, "dominated by invertebrates in all three boxes, mainly echinoderms and polychaetes";
  "A gradual shift from benthos to fish feeding with increasing M. aeglefinus size was not found." No
  euphausiids mentioned.
- Tam et al. 2016, J Fish Biol 88:2203, doi:10.1111/jfb.12983 [F]: North Sea data are the 1981 and 1991
  ICES stomachs only; "high amounts of fishes in the diet of larger North Sea M. aeglefinus are likely
  to be a consequence of prey availability" (sandeel).
- Huwer et al. 2014 (MARE/2012/02) [A]: North Sea part is grey gurnard, hake and mackerel; no haddock.
  CINEA 2024 [F]: haddock not sampled.
- SMS 2023 key run [F]: haddock stomach input is 1981 and 1991; among modelled fish, haddock eats only
  sandeel and Norway pout.
- PRE-2000 in area: Albert 1994 (Norwegian Deep 1984-87, dry weight) euphausiids 2-22 % in summer, fish
  4-33 % (via the earlier gadoid files [S]).

Judgement for the model: annual haddock Z 5 % (range 0.5-11; winter ~0, summer up to the Albert range),
P ~3 % (part of the unidentified fish), no evidence for herring or sprat in haddock after 2000. The
M&D 1991 haddock diet (sandeel 30 %) cannot be transferred to the box, where sandeel is < 1 % of the
fish community; the 2025 sample (no sandeel, B 79-83 %) is the in-area check.

## 4. Mechanism for the fish pair: prey length vs gape [D]

**Prey lengths near the bottom in the region** (NS-IBTS GOV Q1 + Q3 2001-25, 57.0-59.2 N, 7.5-11.5 E,
50-300 m; simple mean over hauls; `scripts/prey_length_gape.py`, `tables/prey_length_region.csv`).
Biomass-weighted length p10 / p50 / p90 (cm):

| Prey | Q1 | Q3 | Pooled | % of biomass < 15 cm (pooled) |
|---|---|---|---|---|
| Norway pout | 10.5 / 12.5 / 17.5 | 8.5 / 14.5 / 17.5 | 8.5 / 14.5 / 17.5 | 55 |
| Herring | 14.3 / 17.8 / 24.3 | 12.3 / 19.8 / 23.8 | 13.3 / 19.8 / 23.8 | 18 |
| Sprat | 7.3 / 9.8 / 12.8 | 10.3 / 11.8 / 12.8 | 10.3 / 11.3 / 12.8 | 99.5 |
| Blue whiting | 17.5 / 31.5 / 38.5 | 22.5 / 26.5 / 31.5 | 22.5 / 26.5 / 31.5 | 0 |
| Whiting | 16.5 / 22.5 / 32.5 | 13.5 / 23.5 / 29.5 | 15.5 / 23.5 / 31.5 | 9 |
| Haddock | 18.5 / 30.5 / 40.5 | 17.5 / 32.5 / 42.5 | 18.5 / 32.5 / 41.5 | 5 |
| LRD (as prey) | 12.5 / 16.5 / 22.5 | 13.5 / 17.5 / 23.5 | 12.5 / 17.5 / 23.5 | 29 |
| Sandeel (*A. marinus*; the GOV barely samples it) | - | - | 8.5 / 11.5 / 16.5 | 81 |

Predators (biomass p10 / p50 / p90, cm): cod 27.5 / 51.5 / 78.5; haddock 18.5 / 32.5 / 41.5; LRD 12.5 /
17.5 / 23.5; grey gurnard 17.5 / 23.5 / 30.5; starry ray 40.5 / 51.5 / 58.5; dab 14.5 / 20.5 / 27.5;
plaice 21.5 / 27.5 / 34.5.

**Gape rule**: prey length <= k x predator length (`inputs/gape_k.csv`).

| Predator | k low / central / high | Basis |
|---|---|---|
| Cod | 0.44 / 0.47 / 0.50 | Scharf et al. 2000 gape line (`../../Torskfiskarnas bytesandelar/inputs/gape.csv`); ICES-DB cod prey/predator length p99 = 0.46 (9581 fish prey) [D] agrees |
| Haddock | 0.32 / 0.33 / 0.36 | cod line x jaw ratio (earlier file, estimate). Cross-check: SMS largest Norway pout relative to cod's, 0.53 / 0.67 = 0.79 x cod -> k ~0.37 [D]. ICES-DB max observed 0.21 (n 6) |
| LRD | 0.30 / 0.40 / 0.45 | ICES-DB 1991 LRD fish prey (85 items): prey up to 10 cm in the 20-24.9 cm LRD class, ratio 0.40-0.50 by the class bounds (0.44 at the mid-point); 0-group gadoids 7-10 cm in 20-30 cm LRD [D]. An observed maximum is a lower bound of gape. No published LRD gape was found |
| Grey gurnard | 0.30 / 0.33 / 0.42 | cod k x 0.70 (0.64-0.89) [D]: SMS 2023 minimum predator/prey WEIGHT ratios (1981-91 stomachs; `data/lit/sms2023_sizepref/min_max_size_pref.out`) converted to length ratios with the region's length-weight fits (`scripts/sms_minratio_length.py`, `tables/sms_minratio_length.csv`). At 35 cm: gurnard 0.47 (cod prey), 0.58 (whiting), 0.53 (herring), 0.51 (Norway pout), 0.32 (sprat); cod 0.66, 0.65, 0.77, 0.67, 0.48. These are noise-inflated extremes (cod/herring 0.77 is far above the Scharf line), so they are used only as a gurnard:cod RELATIVE measure. Floeter 2005 [F]: "large grey gurnard preferred almost half the size of fish large whiting preferred" |
| Starry ray | 0.25 / 0.33 / 0.46 | ICES-DB 2024 Skagerrak (10 fish items): whiting 9.5 cm in a 32 cm ray (0.30); hagfish 23 cm in a 52 cm ray (0.44; elongate prey) [D] |

**Can this predator stage swallow this prey?** % of the prey's near-bottom biomass (pooled Q1 + Q3) with
length <= k x L, central k (`tables/gape_table.csv` has the low-high ranges):

| Predator, length | max prey cm | Norway pout | herring | sprat | blue whiting | whiting | haddock | sandeel |
|---|---|---|---|---|---|---|---|---|
| Cod 30 cm | 14.1 | 42 | 14 | 98 | 0 | 7 | 3 | 73 |
| Cod 40 cm | 18.8 | 96 | 45 | 100 | 0 | 21 | 11 | 100 |
| Cod 50 cm | 23.5 | 100 | 89 | 100 | 19 | 58 | 21 | 100 |
| Cod 70 cm | 32.9 | 100 | 100 | 100 | 95 | 93 | 55 | 100 |
| Haddock 30 cm | 9.9 | 24 | 1 | 5 | 0 | 2 | 0 | 34 |
| Haddock 40 cm | 13.2 | 37 | 9 | 96 | 0 | 6 | 2 | 66 |
| Haddock 50 cm | 16.5 | 84 | 27 | 100 | 0 | 14 | 8 | 93 |
| LRD 20 cm | 8.0 | 7 | 0 | 2 | 0 | 0 | 0 | 3 |
| LRD 25 cm | 10.0 | 24 | 1 | 5 | 0 | 2 | 0 | 34 |
| LRD 30 cm | 12.0 | 34 | 7 | 70 | 0 | 4 | 1 | 55 |
| LRD 35 cm | 14.0 | 42 | 14 | 98 | 0 | 7 | 3 | 73 |
| Grey gurnard 25 cm | 8.2 | 7 | 0 | 2 | 0 | 0 | 0 | 3 |
| Grey gurnard 30 cm | 9.9 | 24 | 1 | 5 | 0 | 2 | 0 | 34 |
| Starry ray 40 cm | 13.2 | 37 | 9 | 96 | 0 | 6 | 2 | 66 |
| Starry ray 50 cm | 16.5 | 84 | 27 | 100 | 0 | 14 | 8 | 93 |

Integrated over each predator's own near-bottom biomass length distribution (central k; low-high k in
brackets; `tables/gape_weighted_by_predator_length.csv`):

| Predator | Norway pout | herring | sprat | whiting | sandeel |
|---|---|---|---|---|---|
| Cod | 84 (82-87) | 68 (63-72) | 94 (92-96) | 53 (48-58) | 90 |
| Haddock | 25 (23-31) | 5 (4-8) | 37 (32-51) | 3 (3-5) | 38 |
| LRD | 6 (1-10) | 0.2 (0-0.7) | 2 (0.2-6) | 0.4 (0-0.8) | 7 |
| Grey gurnard | 9 (6-22) | 0.5 (0.2-3) | 6 (3-27) | 0.7 (0.4-2.5) | 11 |
| Starry ray | 75 (37-97) | 30 (9-79) | 95 (74-99) | 14 (6-51) | 88 |

Reading:
- **Cod is the only large member whose gape admits most of the herring and Norway pout biomass.**
- Haddock (mostly 25-45 cm) can swallow about a quarter of the Norway pout biomass and almost no herring.
- **LRD as it is in the box (median 17.5 cm) can swallow almost none of the pelagic fish biomass.** Its
  fish prey is 0-group fish (gadoids, sandeels, gobies, 2-10 cm) in summer.
- Grey gurnard likewise eats small (0-group) gadoids: Floeter 2005 [F]: "Approximately 90% of cod (and
  60% of whiting) found in all grey gurnard stomachs were smaller than 10 cm."
- Adult starry rays (box median 51.5 cm) can swallow most Norway pout and sprat.
- The GOV under-samples 0-group fish, so the small-prey end is under-represented in these tables.

## 5. Consumption: Q/B per species and for the pooled group

Sources: EcoBase deposits (downloaded `data/ecobase/model_*.xml`, parsed by `scripts/ecobase_groups.py`
into `tables/ecobase_groups_diets.csv`) [F]: 457 North Sea 1991 (Mackinson & Daskalov 2007, as modified
up to 2009; diets mapped to the pools in `tables/md457_diet_pools.csv` by `scripts/md457_pools.py`), 680
North Sea 1981 (Christensen 1995), 413 North Sea 1974 (Christensen), 789 Eastern Channel + southern North
Sea 2006 (Araignous), 461 West Scotland 2000 (Morissette), 46 Faroe 1997 (Zeller), 63/64 Barents Sea
1990/1995 (Blanchard), 751 Malangen fjord 2017 (Vilas et al. 2020, doi:10.1016/j.jmarsys.2019.103268),
761 Porsangerfjord. M&D 2007 printed Table 3.3 with the data value before balancing in brackets
(`../../Torskfiskarnas artsammansättning/data/catchability/ecopath_ns/tech142.pdf`) [F]. Greenstreet et
al. 1997 daily rations via `../../Bentos och torskfiskar/evidence_C_diets_and_consumption.md` [S].

How M&D got their Q/B (tech142 p. 136-145) [F]: cod and haddock "estimated from mean daily rations by
quarters and quarterly abundance for 1991 from MSVPA (ICES 2002b)"; plaice, dab, lemon sole:
"Consumption rate was taken from Greenstreet (1996)"; gurnards: "Q/B was based on consumption of the grey
gurnard ... reported by the MSVPA working group (ICES 2002b)"; LRD, witch, starry ray + others: "P/B and
Q/B were estimated using empirical formulae in FishBase" (Palomares & Pauly 1998), with "Ambient
temperature ... assumed to be the average temperature of the North Sea = 10°C".

| Species | M&D 1991 balanced (data value before balancing) | NS 1981 (680) | NS 1974 (413) | Other models | Field ration x 365 | Used: low / central / high |
|---|---|---|---|---|---|---|
| Haddock | adult 4.4 (2.35); juv 0-1 5.39 (4.16) print, 14.4 EcoBase | 3.79 (ages pooled) | 2.96; juv 8.0 | W Scotland 4.0; Faroe 3.8; Barents adult 2.3, juv 3.5 | Greenstreet 0.0130/d = 4.75 | 2.35 / **3.8** / 4.75 |
| Cod | adult 3.5 (2.17); juv 0-2 5.96 (4.89) print, 8.0 EcoBase | 3.15 | 2.85; juv 5.0 | W Scotland 3.8; Faroe 3.1; Barents adult 2.0, juv 3.5 | Greenstreet 0.0144/d = 5.26 | 2.17 / **3.2** / 5.26 |
| LRD | 3.4 (4.0, FishBase formula at 10 C) | - | - | Barents "benthic invertebrate feeders" 3.2 (group content not checked) | none found | 3.0 / **3.4** / 4.0 |
| Dab | 3.36 (4.0, Greenstreet 1996) | - | 3.5 | EEC 4.2; FishBase empirical 3.69 (W∞ 756 g, 12 C) [S] | none found | 3.36 / **3.5** / 4.2 |
| Plaice | 3.42 (Greenstreet 1996) | 2.8 | 1.26 | EEC 2.6; W Scotland 3.42; FishBase empirical 2.13-3.42 [S] | none found | 2.8 / **3.42** / 3.42 |
| Witch | 3.0 (FishBase) | - | - | Malangen 4.84 | none found | 2.7 / **3.0** / 4.84 |
| Lemon sole | 4.32 (Greenstreet 1996) | - | - | - | none found | 3.8 / **4.32** / 4.32 |
| Grey gurnard | gurnards 3.2 (MSVPA) | - | 3.0 | EEC 4.54; W Scotland 4.67 | none found | 3.0 / **3.2** / 4.67 |
| Starry ray | starry ray + others 1.7 (FishBase) | rays 2.9 | - | W Scotland rays 1.45; Malangen 2.27 | none found | 1.5 / **1.7** / 2.3 |
| Hagfish | - | - | - | Malangen 6.4 | none found | 2.0 / 4.0 / 6.4 (judgement) |
| Greater weever, others | - | - | - | - | none found | 3.0 / 3.5 / 4.5 (judgement) |

Notes:
- The M&D balancing raised the gadoid Q/B well above the data values (haddock adult 2.35 -> 4.4, cod
  adult 2.17 -> 3.5). Christensen 1995 found MSVPA consumption too low: gross efficiencies
  "unrealistically high ... cod with 35%, whiting 24%, and haddock 36%" (earlier file [F]). So the data
  values are a floor.
- Greenstreet's rations are North Sea means at North Sea temperatures. The box bottom water is ~6-9 C
  and the northern North Sea bottom water is similar (~7-8 C), so the gadoid values need little
  temperature correction (judgement).
- The FishBase-formula values (LRD, witch, starry ray) assume 10 C. The formula's temperature term
  (-1.360 x 1000/K in log10 Q/B) gives x 0.85 at 6 C, x 0.91 at 7.5 C and x 0.96 at 9 C [D]. A Q10 of 2
  gives x 0.76 / 0.84 / 0.93.

**Pooled Q/B** (`inputs/qb_species.csv`, `scripts/qb_pooled.py`, `tables/qb_pooled.csv`) [D]:
- Biomass-weighted: low 2.5, **central 3.5**, high 4.7 /yr. Haddock + cod alone: 3.5 /yr.
- Temperature (judgement): if the central values hold at 8 C, 6-9 C gives 3.0-3.7; if they hold at 10 C,
  2.6-3.35. **A pooled 3.0-3.5 /yr at 6-9 C is the defensible central range.**

## 6. Per-species notes

### 6.1 Long rough dab (12.6 %)
- **ICES-DB 1991** (UK R/V 747S pooled records, 20-30 cm LRD; PRE-2000) [D]:
  - Central/southern North Sea (61 records, 572 fish, 239 g; Jan, Feb, May, Aug): B 53.6 (bivalves 19.5,
    pagurids/crabs 9.7, ophiuroids 7.5, *Pandalus montagui* 4.6), Z 15.4 (all *Hyperia*, 37 g; hyperiids
    in flatfish stomachs are probably taken with jellyfish, kept in Z), sandeel 9.3, whiting 8.3, Norway
    pout 6.4, cod 2.6, other flatfish 1.5, gobies/dragonets/other 2.4.
  - North-west North Sea, 57.5-59 N, mostly 3 W-2 E (48 records, 297 fish, 209 g; Feb, Jun, Aug): whiting
    31.3, Norway pout 19.8, cod ~13, haddock 9.5, sandeel 6.9, B 18.0. The fish are ~150 small items,
    nearly all 0-group whiting, Norway pout, cod and haddock of 5-8 cm in August stomachs: a summer
    0-group feeding event, not year-round piscivory.
  - Fish prey were 2-10 cm (0-group gadoids, sandeels, gobies), mostly in August.
  - These are the larger LRD (20-30 cm). Box LRD biomass median is 17.5 cm, so these fish shares are an
    upper bound for the box.
- Ntiba & Harding 1993, Neth J Sea Res 31:189, doi:10.1016/0077-7579(93)90009-H [A] (closed), PRE-2000:
  "feeds throughout the year with a definite maximum from April till August when over 85% of the fish
  sampled were feeding"; main prey "crustaceans (natant decapods), polychaetes, and echinoderms
  (ophiuroids)"; "also consumes O-group whiting ..., long rough dab, and gobies ... in the east-central
  North Sea nursery grounds during the summer months".
- M&D 1991 (diet from Ntiba & Harding; EcoBase 457) [S, D]: B 86.6, Ofi 6.4, Bf 3.8, sandeel 2.3, P 0.5,
  Z 0. Tech142: "predominantly benthivorous, but unlike dab, fish is found in a greater proportion".
- Context, out of area: Link et al. 2002 (American plaice = LRD, NW Atlantic, 1786 stomachs, mean
  29.9 cm) [F]: "Echinoderms comprised >70% of the diet, with ophiuroids representing approximately 45%";
  not among the piscivores. Dolgov 2003 (Barents Sea, 5659 stomachs) [F]: LRD, thorny skate and Greenland
  halibut "consumed more fisheries waste ... up to 50-70% by stomach content weight". FishBase: Barents
  "nekton 57 %" (Berestovsky 1989) [S].
- No post-2000 in-area LRD diet, no piscivory onset length and no gape measure were found.

### 6.2 Dab (4.5 %)
- No dab records in the public ICES download (any area).
- M&D 1991 (De Clerck & Torreele 1988) [S]: B 98.7 (small mobile epifauna 33, epifauna, infauna), Z 1.3.
  Tech142: "dominantly benthivorous, although larger specimens occasionally feed on small demersal fish".
- Schückel et al. 2012 (NW of Helgoland) via FishBase [S]: "nekton 27 %" (FishBase class; contents not
  checked; southern North Sea).
- Johnson et al. 2015, Proc R Soc B 282:20142336 [F] (Irish Sea): 575 dab stomachs; dab prefer
  burrowing crustaceans, often "as whole chelae only". Kaiser & Ramsay 1997, MEPS 152:307 [F]: dab feed
  opportunistically in trawl tracks; by occurrence only.
- Not obtained (closed): Gibson & Ezzi 1987, Hinz et al. 2005, Knust 1996, De Clerck & Torreele 1988.
- Judgement: B 95, Z 2, Oth 3. Fish and krill are minor.

### 6.3 Plaice (3.8 %)
- ICES-DB 2024 Skagerrak (Swedish + Danish IBTS, Jan + Aug; 185 fed, 14-47 cm, 21-226 m, median 66 m)
  [D]: B 99.5 (ophiuroids 46.9, bivalves 26.0, polychaetes 15.2, other echinoderms 9.8), sandeel 0.4.
  Model-box subset (61 fed): B 100. NNS 2024 (60 fed): B 85.6, cnidarians 14 (one large item). Kattegat
  (150 fed): B 100.
- M&D 1991 [S]: B 98.6.

### 6.4 Witch (1.7 %)
- Bergstad, Wik & Hildre 2003, J Northw Atl Fish Sci 31:165 [F] (data Hildre 2001; Skagerrak 156-490 m,
  1991-96, n 134): Table 6, %W excluding indeterminate: Polychaeta 59.3, Bivalvia 20.4, Decapoda 10.3,
  Actiniaria 5, Amphipoda 1.2, other Crustacea 1.5, Euphausiacea 0.2, Teleostei 0.1. "44% of the
  contents in terms of weight were unidentifiable" (p. 172).
- Mattson 1981 (Raunefjorden) via FishBase [S]: benthic crustaceans ~70 %, fish 1.7 %, euphausiids 0.25 %.
- M&D 1991 [S]: B 99.2. No ICES-DB records.

### 6.5 Lemon sole (0.9 %)
- Rae 1965 via FishBase [S] (unit not stated): polychaetes 35-70 %, cnidarian polyps 12-35 %, molluscs,
  crustaceans; no fish, no zooplankton. FishBase: "Apparently they do not feed in wintertime."
- M&D 1991 [S]: B 99.3. No post-2000 source; no ICES-DB records.

### 6.6 Grey gurnard (0.9 %)
- ICES-DB: 12 fed stomachs, southern North Sea, Nov 2022 (Caridea, unidentified fish); out of area.
- de Gee & Kikkert 1993 (ICES 1991 stomach programme, North Sea Q3) via FishBase [S], by weight: fish
  42.7 % at 12 cm, 61.0 % at 20 cm, 82.3 % at 25 cm, 92.5 % at 35 cm; the rest benthic crustaceans.
- Floeter 2005, PhD thesis, Hamburg [F] (`data/lit/floeter2005_diss/`, p. 59): "the ontogenetic diet
  shift from an invertebrate-dominated to a fish-dominated diet occurred at rather small sizes of
  approximately 20 cm"; "The overall mean fraction of gadoid prey weights in the stomachs of grey gurnard
  larger than 20 cm was 37%, with 52%, 30%, 34% and 30% in the four quarters"; cod plus whiting 18-32 % in
  gurnard > 30 cm.
- Weinert et al. 2010, J Appl Ichthyol 26:75 [A]: Dogger Bank 2006; deep stations dominated by the
  hyperiid *Hyperia galba* (Z).
- M&D 1991 [S]: sandeel 45 %; not transferable to the box.
- Box gurnard biomass median 23.5 cm: near the shift. Judgement row (B 30, Z 2, P 10, W 15, Bf 12, Oth
  31): 0-group whiting -> W, 0-group cod/haddock -> Bf, Norway pout -> P, sandeels/gobies/dragonets/
  unidentified -> Oth. At 0.9 % of the biomass it barely moves the pooled result.

### 6.7 Starry ray (1.5 %)
- ICES-DB 2024 Skagerrak (Swedish IBTS Aug; 39 fed, 24-61 cm, 57-237 m) [D]: B 61.4 (crabs/Polybiidae
  32, Crangonidae 13.4, polychaetes 6.7, amphipods 4.8), Z 1.4 (euphausiids), Bf 27.4 (one hagfish 24 g,
  plus flatfish), dragonets/gobies 5.4, whiting 2.5. Fish rises with size: 50-69 cm Bf 60 %.
- Skjaeraasen & Bergstad 2000, ICES J Mar Sci 57:1249, doi:10.1006/jmsc.2000.0811 [A] (closed): "Upon
  reaching maturity [45 cm], a major diet shift to fish, larger shrimps and squat lobsters (Munida)".
  Via FishBase [S], 175 adults, mean 55.5 cm, % wet weight: fish 55.2 (unidentified 25.9, whiting 8.4,
  witch 5.6, blue whiting 4.3, herring 3.9, hake 3.8, Norway pout 1.7, *Lycodes* 1.4), hagfish 0.9,
  decapods 30.3 (*Munida* 8.6, *Pandalus borealis* 8.1), polychaetes 9.2, Z 0.
- Daan, Johnson, Larsen & Sparholt 1993, ICES CM 1993/G:15 [F] (North Sea 1991, 3201 stomachs): "The food
  of R. radiata consisted for more than 50% of fish from 25 cm onwards"; sandeels "only infrequently".
- Used: the adult diet (box ray median 51.5 cm). The M&D "Starry ray + others" diet (sandeel 49 %) is a
  composite contradicted by Daan et al. 1993; not used.

### 6.8 Hagfish (1.3 %)
- FishBase [S]: "Feeds chiefly on dead and dying fish ... consuming viscera and musculature."
- Malangen fjord Ecopath (EcoBase 751) [F]: diet 80 % detritus/carrion, Q/B 6.4, P/B 0.3.
- Shelton 1978 (JMBA 58:81) not obtained. Judgement: B 20, Oth 80 (carrion is no pool); Q/B 2-6 (low
  metabolism; no measurement).

### 6.9 Greater weever and others (3.3 %)
- No source sought or found; judgement row only.

### 6.10 Haddock and cod: see sections 2 and 3.

## 7. Gaps

- **No public ICES stomach records** for dab, witch, lemon sole, hagfish or weever (any area); LRD only
  1991; grey gurnard only 12 southern stomachs.
- **No post-2000 summer haddock diet** in the NNS, Norwegian Deep or Skagerrak. Haddock Z is a judgement.
- Cod in area: 53-119 fish items, mostly Q1; Gu 3-10 % of fish mass after the name fix.
- Closed or not obtained: Ntiba & Harding 1993, Skjaeraasen & Bergstad 2000, Schückel et al. 2010 and
  2012, Hinz et al. 2005, Gibson & Ezzi 1987, Knust 1996, De Clerck & Torreele 1988, de Gee & Kikkert
  1993, Weinert et al. 2010, Shelton 1978, Rijnsdorp & Vingerhoed 2001, Pihl 1994 (Kattegat cod,
  whiting, plaice, dab, LRD), Greenstreet 1996 (rations by species), Huwer et al. 2014 (1907 grey gurnard
  stomachs 2012-14, not in the public download).
- No measured ration or Cmax for LRD, dab, witch, lemon sole, gurnard, ray or hagfish was found; their
  Q/B are model inputs (FishBase formula or Greenstreet rations).
- No published LRD or grey gurnard gape; the k values are derived from stomach prey sizes [D].

## 8. Source status

| Source | Used for | Status |
|---|---|---|
| ICES stomach database, public download 1981-2025 | all ICES-DB rows | [F] `../../Torskfiskarnas bytesandelar/data/diet/ices_stomach/` |
| WoRMS AphiaClassificationByAphiaID | prey lineages | [F] `data/diet/worms/` (82 new) + earlier caches |
| EcoBase models 457, 680, 413, 789, 461, 46, 63, 64, 751, 761 | Q/B, M&D diets | [F] `data/ecobase/` |
| Mackinson & Daskalov 2007 (Cefas tech. rep. 142) | Q/B methods, Table 3.3 | [F] on disk (Torskfiskarnas artsammansättning) |
| NS-IBTS DATRAS CPUE per length 2001-25 | prey and predator lengths | [F] `../data/extract/`, `../../Torskfiskarnas artsammansättning/data/extract/` |
| SMS 2023 key run (WGSAM GitHub) | gurnard/cod/haddock largest-prey ratios | [F] `data/lit/sms2023_sizepref/` |
| Lewy & Vinther 2004, ICES CM 2004/FF:20 | size preference context (cod eta 5.56, haddock 7.87 in ln weight ratio) | [F] `data/lit/lewy_vinther2004/` |
| Floeter 2005, PhD thesis | grey gurnard diet and prey size | [F] `data/lit/floeter2005_diss/` |
| CINEA 2024, doi:10.2926/683598 | cod 2022-23 by number | [F] `data/lit/cinea2024/` |
| Tam et al. 2016 | haddock | [F] `data/lit/tam2016/` |
| Bergstad, Wik & Hildre 2003 | witch, starry ray | [F] `data/lit/bergstad2003/` |
| Daan et al. 1993, ICES CM 1993/G:15 | starry ray | [F] `data/lit/vliz275169/` |
| Link et al. 2002; Dolgov 2003; Johnson et al. 2015; Kaiser & Ramsay 1997 | LRD, dab context | [F] `data/lit/` |
| FishBase diet and Q/B pages | dab, plaice, LRD, witch, lemon sole, gurnard, ray, hagfish | [S] `data/lit/fishbase_trophic*/` |
| Greenstreet et al. 1997 rations; Christensen 1995 | gadoid Q/B | [S]/[F] via earlier files |
