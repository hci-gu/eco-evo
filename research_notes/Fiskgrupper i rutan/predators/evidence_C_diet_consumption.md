# Evidence C - predatory_fish: diet by pool, consumption (Q/B), light, functional response

Scope: the members of the new `predatory_fish` group (long-run biomass share in the group: saithe 41 %,
whiting 37 %, mackerel 8 %, spurdog 5 %, horse mackerel 3.4 %, hake 2.5 %, anglerfish 1.4 %;
`../split_brief.md`). For each species: diet by mass mapped to the model pools, Q/B, light dependence
of feeding, functional response and interference. Diets are checks and the consumption behind M2,
never fitted targets.

**Tags.** [F] full text read or data downloaded and parsed; [A] abstract only; [S] secondary (source
named); [D] computed here. **PRE-2000** marks rows from another ecosystem state (context only).

**Pools (codes used in the tables).** P = `planktivorous_fish` (herring, sprat, Norway pout, blue
whiting, sandeel, argentines, pearlside, silvery pout); Bf = `benthivorous_fish` (haddock, cod,
flatfish, rays, gurnards, weevers, gobies, dragonets, poor cod, rocklings, eelpouts); W = within-group
`predatory_fish` (whiting, saithe, pollack, mackerel, horse mackerel, hake, spurdog, anglerfish, ling);
Z = `zooplankton` (copepods, euphausiids, hyperiids, Pasiphaea/Hymenodora "by habit", appendicularians,
pelagic decapod larvae); B = `benthic_community` (polychaetes, molluscs, echinoderms, mysids, gammarids,
Crangon, Pandalus, Nephrops, crabs); O = other (cephalopods, cnidarians).
**W is reported as a share of total food and is NOT renormalised away.** M2 uses the share of total
consumption. For the eat-share cap the user may renormalise over P+Bf+Z+B (done once in the bottom line).

## 1. Main findings

1. **The group eats mainly planktivorous fish and krill.** The consumption-weighted group diet is
   **P 0.41, Z 0.31, B 0.11, W 0.09, Bf 0.07, O 0.01** [D]. Without the within-group part:
   P 0.45, Z 0.35, B 0.12, Bf 0.08.
2. **Saithe and whiting carry 85 % of the group's consumption** (saithe 37 %, whiting 47 %) [D].
   Their diets decide the group. Both are uncertain in-area: saithe has no North Sea diet after 2000;
   whiting has one thin post-2000 Skagerrak sample.
3. **The old "Ofish" column splits mostly into P.** Norway pout, sandeel, blue whiting and pearlside
   are P. Haddock and cod (Bf) are 5-20 % of the fish part. Whiting cannibalism (W) is large in post-2000
   samples (Skagerrak 2022, central North Sea 2022) and absent in April 1991.
4. **Group Q/B = 3.35 /yr (range 2.85-4.35)**, biomass-weighted [D]. Spine: SMS key run 2023 (saithe
   2.87, whiting 4.68 stock-wide, mackerel 1.87, horse mackerel 3.60, hake 5.82) and Ecopath North Sea
   1991 (saithe adult 3.6, whiting adult 5.46, mackerel 1.73, horse mackerel 3.51, spurdog 2.0, monkfish
   1.9, hake 2.2).
5. **Light:** fish pairs are moderately light-limited; krill pairs are not. Consumption-weighted
   dark_ratio **~0.6 for the pair on planktivorous fish** (same as the existing gadoid value) and
   **~0.9 (use 1.0) for the pair on zooplankton** [D, judgement]. Mackerel is the only strongly
   visual member, and it is small.
6. **Functional response on fish: Type II.** Whiting and cod show *negative* prey switching
   (Rindorf et al. 2006). Type III is supported only for mackerel filter feeding on copepods
   (feeding threshold and hysteresis, Macy et al. 1998).
7. **No evidence of predator interference** for saithe, whiting or mackerel. Rindorf et al. 2006 found
   no significant predator-density effect.

## 2. Diet by pool, per species

### 2.1 ICES stomach database, re-mapped to the split pools [F]

Source: the public ICES Stomach Content download already on disk
(`../../Torskfiskarnas bytesandelar/data/diet/ices_stomach/`, CC BY 4.0, https://stomachdata.ices.dk).
New script `scripts/predator_diet_pools.py` (WoRMS lineage classifier; 15 missing lineages fetched to
`data/diet/worms/`). Output `tables/ices_diet_pools.csv` (by species, area, period, 10-cm class) and
`tables/ices_diet_pools_taxa.csv` (mass by taxon). Areas as in the earlier file; the Skagerrak line is
clamped at Lindesnes (7.05 E) north of 58 N. Hake Skagerrak 2024 has 50 fed stomachs here vs 48 in the
earlier file; probably an area-polygon difference (not traced).

Unidentified fish were re-allocated (`scripts/size_weighted_diets.py`): gadoids not identified to
species pro-rata over identified gadoids (Norway pout, blue whiting, silvery pout -> P; cod, haddock,
poor cod -> Bf; whiting, saithe, hake -> W); fish not identified below class pro-rata over all
identified fish. Size-weighting uses the box length mix (`../../Torskfiskarnas bytesandelar/tables/
length_class_pct.csv`, variant A_floor, pooled quarters). Values are % of identified prey mass.

| Species | Area, period | Fed | Weighting | P | Bf | W | Z (Pas / euph) | B | O |
|---|---|---|---|---|---|---|---|---|---|
| Whiting | Skagerrak, Apr 1991, PRE-2000 | 54* | box sizes | 20.8 | 4.3 | 0 | 40.1 (9.2 / 30.9) | 34.8 | 0 |
| Whiting | same | 54* | pooled | 16.4 | 5.5 | 0 | 53.6 (6.5 / 47.1) | 24.5 | 0 |
| Whiting | model box, Apr 1991, PRE-2000 | 21* | box sizes | 6.5 | 1.8 | 0 | 31.6 (9.3 / 22.2) | 60.2 | 0 |
| Whiting | Skagerrak, Jan-Feb + Aug 2022 | 38 | box sizes | 16.7 | 20.8 | 45.9 | 1.6 | 15.0 | 0 |
| Whiting | central/southern N Sea 2022 | 518 | box sizes | 33.4 | 5.4 | 39.2 | 0.1 | 21.4 | 0.4 |
| Whiting | Kattegat, Apr 1991, PRE-2000 | 38* | box sizes | 65.4 | 7.7 | 0 | 5.0 | 21.9 | 0 |
| Hake | Skagerrak, Aug 2024 | 50 | box sizes | **77.3** | 0.5 | **21.1** | 0.1 | 0.4 | 0.5 |
| Hake | model box, Aug 2024 | 33 | box sizes | 74.3 | 0 | 24.5 | 0.3 | 0.4 | 0.6 |
| Mackerel | central N Sea (sandeel grounds), Jun 2022 | 900 | pooled | 86.9 + Fu 10 | 0 | 2.9 | 0 (see text) | 0.1 | 0 |
| Mackerel | NNS, Jan-Feb 2025, **dry weight** | 49 | pooled | 2.4 + Fu 7.8 | 0 | 0 | **89.6** (euph 88.9) | 0.2 | 0 |
| Horse mackerel | Skagerrak/box, Aug 2023 | 4-6 | pooled | 0 | 0 | 0 | 87.4 (all euph) | 12.6 | 0 |
| Spurdog | Skagerrak, Aug 2024 | 7 | pooled | thin: 2 g of prey, mostly unidentified fish and Pasiphaea | | | | | |
| Spurdog | central N Sea 1991, PRE-2000 | 7 | pooled | 50.7 | 24.8 | 23.8 | 0 | 0 | 0 |
| Anglerfish | Skagerrak + NNS 2022-24 | 9 | pooled | 0-8 | 0-92 | 0-86 | 0 | 0 | 0 |
| Saithe | NNS Feb 2024 | 1 | - | one fish (Norway pout + Crangon) | | | | | |

\* pooled records (several stomachs each); empties cannot be separated.

Notes on the rows:
- **Whiting 1991 Skagerrak**: P = herring 6.8, Norway pout 5, sprat 3, pearlside 1 (pooled %).
  B is mostly *Pandalus borealis* (B by the brief). Z is *Meganyctiphanes norvegica* plus *Hymenodora*
  (counted Z "by habit"; it is the Pas sub-share). Moving Pas to B changes Z by 6-9 points.
- **Whiting 2022 Skagerrak** is thin: 69 % of the mass is 9 whiting, plus 12 g of cod. The central North
  Sea 2022 sample (518 fed, 125 hauls) confirms that cannibalism is real post-2000: W 39 % size-weighted.
  That sample is shallower and outside the box; it is context.
- **Hake 2024**: P is herring 65, clupeids 10, sprat 6, Norway pout 1 (pooled %); W is whiting.
  Above 70 cm whiting is about half the diet (n 7).
- **Mackerel June 2022** (Danish R/V Dana, 17 hauls, mostly on the sandeel grounds) [F]. Identified mass
  is 87 % sandeel/clupeids. But **unidentified matter is large**: 670 g of rows without a taxon (0.2-5 g
  per fish, no count, "digested remains") and 231 g "Crustacea" next to ~920 g of identified fish. If the
  crustacean rows are zooplankton, Z = 0.20; if the blank rows are also zooplankton, Z = 0.50 [D]. The
  sample is from sandeel grounds in June. It is not a Skagerrak diet.
- **Mackerel Jan-Feb 2025** (Norwegian IBTS Q1, dry weight in mg, 24 hauls) [F]: overwintering fish
  in the NNS, 90 % euphausiids, little food in total (28.6 g dry over 49 fed fish).
- **Saithe 1981 and 1991** records carry counts but no weights. By number (`scripts/saithe_1981_counts.py`,
  `tables/saithe_whiting_1981_1991_by_number.txt`) [F], NNS saithe >= 40 cm, 53 pooled records, Jan-Feb,
  Jun-Jul, Nov: Z 76 % of items (euphausiids); of fish items P 72 % (pearlside 2664, Norway pout 419,
  silvery pout 150, sandeel 88), Bf 10 % (haddock 412, cod 42), gadoids n.i. 14 %, W 0.2 %. By mass the
  small pearlside counts far less. With assumed prey weights (pearlside 1 g, Norway pout 15 g, haddock
  20 g, gadoids n.i. split like the identified ones) the fish part is roughly P 50 %, Bf 50 % [D, rough].
  This is the only in-area species split of the saithe fish diet in the data; PRE-2000.

### 2.2 Ecopath North Sea 1991, diet matrix re-mapped [F, PRE-2000]

Mackinson & Daskalov 2007 (Cefas Sci. Ser. Tech. Rep. 142, https://www.cefas.co.uk/publications/techrep/tech142.pdf,
copy in `data/diet/ecopath/`). Diets read from the EcoBase model 457 XML already on disk
(`../benthivores/data/ecobase/model_457.xml`) with `scripts/ecobase457_predator_diets.py`; output
`tables/ecobase457_predator_diets_stdout.txt`. Each column sums to 1.000. The older text parse
(`ecopath_predator_diets.py`, from `../../Litteraturbaserad growth rate per FG/ecopath_ns1991/diet.json`)
has shifted columns for several predators (column sums 0.4-1.5) and is not used. Diets for the
commercial species come from the 1991 Year of the Stomach: "The diet matrix of commercial species was
compiled using data from the 1991 year of stomachs (Hislop 1997)". The whole North Sea, not the box.

| Ecopath group | P | Bf | W | Z | B | O (ceph, other) | Largest items |
|---|---|---|---|---|---|---|---|
| Saithe (adult, >40 cm) | 0.68 | 0.04 | 0.03 | 0.24 | 0.01 | 0.00 | Norway pout 0.34, herring 0.20, krill 0.13, copepods 0.11 |
| Saithe (juvenile) | 0.37 | 0.02 | 0.00 | 0.57 | 0.03 | 0.01 | krill 0.34, Norway pout 0.24, copepods 0.23 |
| Whiting (adult, >20 cm) | 0.75 | 0.05 | 0.02 | 0.05 | 0.12 | 0.01 | sandeel 0.36, Norway pout 0.21, sprat 0.18 |
| Whiting (juvenile) | 0.46 | 0.05 | 0.00 | 0.17 | 0.29 | 0.02 | sandeel 0.31, Norway pout 0.14 |
| Mackerel | 0.34 | 0.01 | 0.01 | 0.29 | 0.16 | 0.20 | sandeel 0.27, "planktonic microflora" 0.18, krill 0.17, polychaetes 0.12 |
| Horse mackerel | 0.29 | 0.05 | 0.00 | 0.44 (+ fish larvae 0.12) | 0.05 | 0.05 | krill 0.44, Norway pout 0.22 |
| Spurdog | 0.39 | 0.24 | 0.24 | 0.01 | 0.06 | 0.06 | herring 0.19, small gadoids 0.13, whiting 0.12, Norway pout 0.10, mackerel 0.08 |
| Hake | 0.56 | 0.07 | 0.37 | 0 | 0 | 0 | blue whiting 0.33, horse mackerel 0.19, mackerel 0.17 |
| Monkfish | 0.44 | 0.24 | 0.21 | 0 | 0.02 | 0.09 | sandeel 0.17, Norway pout 0.13, herring 0.12, whiting 0.12 |

"Other gadoids (small)" (poor cod, bib, rocklings, silvery pout) is counted Bf; silvery pout belongs to
P, so Bf is slightly high for spurdog and monkfish. For mackerel the "microflora" 0.18 and "polychaetes"
0.12 are probably the 1991 "other food"/unidentified share placed in those cells during balancing; the
mackerel agent recommends moving them to Z (pelagic larvae, appendicularians) [judgement]. Then mackerel
Z = 0.59. Fish larvae (horse mackerel 0.12) are best counted Z (zooplankton-sized, no model pool).

### 2.3 Literature per species

**Saithe.** No North Sea diet by mass after 2000 (earlier file; Cormon et al. 2014: diets "currently
lacking", https://doi.org/10.1093/icesjms/fsu120 [A]).
- Mehl 2005, ICES AFWG WD 7 (http://hdl.handle.net/11250/100697) [F, via the literature agent]:
  Norwegian coast north of 62 N, Oct-Nov 1998-2003, 5865 stomachs, % weight. "Fish was the dominant prey
  category (70-95 %), followed by crustaceans (5-29 %)." "In the smallest size group (20-39 cm), krill was
  the dominating prey item in all sub-areas." "In area C and D, Norway pout was the dominating fish prey,
  followed by blue whiting and haddock. Herring was scarce and no cod was found." Sub-area D (62-63.5 N),
  6-year means of Table 8: **40-59 cm: Z 41 (krill 40.5), P 52 (Norway pout 50.8), Bf 3.6 (haddock),
  B 0.3, O 0.7, unid. fish 1.6; 60-79 cm: Z 16, P 73, Bf 4.2, B 0.5, O 1.4, unid. fish 4.2**. Krill at
  40-59 cm ranged 1.7-92 % between years: a pulse prey.
  Weighted by the box saithe length mix (40-59 cm 71 %, >= 60 cm 29 %) and unid. fish pro-rata:
  **P 0.60, Bf 0.04, W 0, Z 0.34, B 0.004, O 0.01** [D]. Autumn only; coastal Norway, not the Trench.
- Daan 1989 (1981, total North Sea, via FishBase) [S, earlier file]: euphausiids 21-76 % at 30-50 cm,
  9-35 % at 70 cm, ~0 at 100 cm; rest fish; benthos <= 1 %.
- Bergstad 1991, Sarsia 75 (Norwegian Deep incl. Skagerrak, 1984-87) [A]: "Particularly important trophic
  relationships exist ... between small saithe and blue whiting as predators and the euphausids as prey,
  and between saithe and Norway pout." doi:10.1080/00364827.1991.10413455.
- Cormon et al. 2016, ICES JMS 73:1370 [A]: "adult Norway pout (ages 1-3) were mainly predated by saithe,
  which contributed to about half of total predation mortality". doi:10.1093/icesjms/fsw050.
- IMR 2021 narrative [F, earlier file]: prey "blue whiting ..., Norway pout ..., sandeel ..., herring ...,
  and krill".
- Judgement: fish part ~85-95 % P (Norway pout >> blue whiting, herring, sandeel, pearlside), 5-15 % Bf
  (haddock), W ~0-3 %. Krill large below 50 cm.

**Whiting.**
- Hislop et al. 1991, ICES JMS 48:139 (1981, ~9800 stomachs) [A]: "Fish and Crustacea comprised at least
  85% by weight ... the proportion of fish increasing with predator size." doi:10.1093/icesjms/48.2.139.
- Lauerburg et al. 2018, MEPS 594:213 [F]: whiting "(from a size of 25 cm on) is almost exclusively
  piscivorous, having less than 20% of non-fish prey"; "Crustaceans and gobies dominate the diet of
  M. merlangus smaller than 15 cm". doi:10.3354/meps12533.
- Pedersen 1999, J Fish Biol 55:1096 [A]: "Pelagic ... whiting preyed upon (pelagic) sprat and sandeel,
  while demersal whiting preyed upon (demersal) Norway pout and herring." doi:10.1006/jfbi.1999.1112.
  Bottom-trawl stomachs under-sample the pelagic part.
- Onsrud et al. 2004 (Oslofjord, 59.8 N) [F, gadoid light report]: krill ~82 % of whiting stomach mass
  day and night.
- Judgement for the box (deep, krill-rich Trench edge): krill and Pandalus matter more than in the
  North Sea-wide Ecopath diet. Central P 0.28, Bf 0.08, W 0.15, Z 0.28, B 0.21.

**Mackerel** (literature agent, [F] unless marked).
- Mehl & Westgard 1983, ICES CM 1983/H:34 (3674 stomachs, North Sea 1981-82; https://hdl.handle.net/11250/103794),
  % wet weight, copepods / euphausiids / other invertebrates / fish: **NE North Sea annual 36.0 / 49.0 /
  6.6 / 7.3**; Q3 41.0 / 27.5 / 17.6 / 11.2; central North Sea annual fish 27.8; NW 52.6; south 58.4.
  "On a yearly basis fish constituted about 1/3 of the stomach content and about 90% of the fish prey
  were herring, sprat, sand eel and Norway pout." Other invertebrates are mostly appendicularians (Z).
  NE North Sea mapping: Z ~0.90, P ~0.07 [D]. PRE-2000.
- Dahl & Kirkegaard 1986, ICES CM 1986/H:68 (eastern North Sea, July 1985) [F]: "The proportion of fish in
  the stomachs increase with predator size from about 50% in the smallest to 90% in the largest mackerel.
  Herring makes out between 95 to 100% of all identifiable fish species" (0-group herring; a record
  year class). PRE-2000.
- CINEA 2024 EU stomach study [F]: "over 54% of the prey in the stomachs of mackerel were unidentifiable";
  not usable for shares.
- Norwegian Sea (Bachiller et al. 2016, doi:10.1371/journal.pone.0149238): "Calanoid copepods dominated
  the diets"; context only.
- Judgement, Skagerrak summer: Z 0.6-0.8, P 0.2-0.35, others <= 0.02 each. Central Z 0.65, P 0.30.

**Horse mackerel.** Dahl & Kirkegaard 1986 [F]: 25-29 cm herring 66.6, haddock 16.0, unid. fish 13.4;
20-24 cm crustacea 33.4, haddock 42.6, gobies 20.6. ICES FishMap [F]: Skagerrak horse mackerel are mostly
immature 8-16 cm fish, so more zooplanktivorous; the 2023 box sample (4-6 fed) was 87 % krill [F].
Central Z 0.60, P 0.25, Bf 0.07, W 0.03, B 0.05.

**Spurdog.** No post-2000 NE Atlantic diet by mass was found. Holden 1966 [A]: "sandeels were found to be
the commonest prey species" (occurrence). Kloecker et al. 2024, Mov Ecol 12:62 [F]: prey "teleost fish,
crustaceans, squid, and ctenophores"; Norwegian stomachs held pearlside, herring, whiting, Norway pout,
sprat. Ecopath (Ellis et al. 1996 based) is the only by-mass split. Central P 0.42, Bf 0.22, W 0.22,
Z 0.03, B 0.06, O 0.05.

**Hake.** Skagerrak 2024 ICES data above (size-weighted P 0.77, W 0.21). Cormon et al. 2016 [A]: "Saithe
and hake ... have similar diet with Norway pout being an important prey". Bergstad 1991 (Norwegian Deep,
1984-87) [S]: blue whiting 95-98 %. Central P 0.77, W 0.21, Bf 0.01.

**Anglerfish.** Laurenson & Priede 2005, JMBA 85:419 [A]: "the main prey was Norway pout ... and lesser
sandeel ... Diet composition varied seasonally"; no %W in the abstract. IMR 2002 status report [F]:
"Anglerfish are essentially opportunistic feeders, displaying a low degree of prey selectivity";
gadoids and migrating herring off Norway. Central (Ecopath-like): P 0.35, Bf 0.30, W 0.25, O 0.08.

## 3. Consumption, Q/B

### 3.1 SMS North Sea key run 2023 [F data, D computation]

Files from https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023 (`data/diet/sms_consumption/`,
script `scripts/sms_qb.py`); report ICES 2024 WGSAM, ICES Sci. Rep. 6:13, doi:10.17895/ices.pub.25020968.
Ration per individual and quarter = a W^b (Table 3-8). "Food rations were estimated as part of the 2020
key run and has not been updated." No temperature term in SMS; the a-values carry the North Sea
temperature when the stomachs were taken (evacuation exponent 0.081 /C for saithe and whiting, i.e.
about -8 % per degree colder). Q/B = annual consumption / mean biomass, weighted by the key run's own
biomass at age, 2000-2022 (1974-2022; 2013-2022):

| Species | Q/B /yr | Per quarter C/B (Q1-Q4) | Per-age Q/B | Notes |
|---|---|---|---|---|
| Whiting | **4.68** (5.00; 4.57) | 0.84 / 0.85 / 1.66 / 1.23 | age 0 10.5, 1 5.1, 2 4.05, 3 3.42, 4 3.10, 6+ 2.78 | ~1.3 % BW/d |
| Saithe | **2.87** (2.87; 2.88) | 0.40 / 1.16 / 0.61 / 0.72 | 2.5 (age 1) to 3.1 (10+) | ~0.8 % BW/d |
| Mackerel (NEA stock) | **1.87** (1.94; 1.89) | 0.05 / 0.73 / 0.86 / 0.13 | 1.5 (age 1) to 2.3 (10+) | whole stock, whole year; feeding season ~3.2 annualised |
| Horse mackerel (N Sea stock) | **3.60** | 0 / 0.89 / 1.27 / 1.48 | 2.2 (age 1) to 6.8 (6+), Q2-Q4 rate | "In quarter 1, horse mackerel are not feeding." |
| Hake | **5.82** (5.90; 5.62) | 0.74 / 2.11 / 1.26 / 1.47 | 8.6 (age 1) to 3.5 (9+) | text and table disagree (below) |

Caveats [F]: for "mackerel, saithe and horse mackerel ... the estimates may be a result of an
underestimation of zooplankton consumption" (rations rise with size). So SMS may be low for total food
of these three. For hake the report says "consumption in percent body weight for hake was assumed to be
the same as for saithe at a similar weight", yet Table 3-8 gives hake about twice the saithe a; read
literally, hake Q/B would be ~2.9, not 5.8 [D].

Box whiting are 20-40 cm (ages 1-4) more than the stock: length-mix weighting of the per-age values
(<20 cm 15 % at ~6.5, 20-30 cm 35 % at ~4.5, 30-40 cm 40 % at ~3.4, >= 40 cm 9 % at ~2.9) gives
**~4.2** [D, rough age-length mapping].

### 3.2 Ecopath North Sea 1991, Table 3.3 [F]

"Consumption rates (Q/B) of commercial species were estimated from mean daily rations by quarters and
quarterly abundance for 1991 from MSVPA (ICES 2002b)"; others from FishBase empirical formulae
(Palomares & Pauly 1998). Balanced values (data value in brackets):

| Group | B t/km2 | P/B | Q/B | Source of Q/B |
|---|---|---|---|---|
| Saithe adult (>= 4 yr, > 40 cm) | 0.22 (0.19) | 0.95 | **3.6** | MSVPA |
| Saithe juvenile | 0.281 | 1 | **4.94** | MSVPA |
| Whiting adult (>= 2 yr, > 20 cm) | 0.352 | 0.89 | **5.46** | MSVPA |
| Whiting juvenile | 0.222 | 2.36 | **6.58** | MSVPA |
| Mackerel | 1.72 | 0.6 | **1.73** | "based on consumption rates used in MSVPA" |
| Horse mackerel | 0.579 | 1.2 (1.64) | **3.51** | SGMSNS (ICES 2002b) |
| Spurdog (> 40 cm) | 0.0167 (0.013) | 0.6 (0.48) | **2.0** | FishBase; Brett & Blackburn 1978 |
| Hake | 0.014 | 0.82 | **2.2** | FishBase empirical |
| Monkfish | 0.042 (0.015) | 0.7 | **1.9 (1.7)** | FishBase empirical |

The EcoBase 457 XML is a later edit: it holds juvenile whiting Q/B 17.4 and juvenile saithe 8.29; the
report values above are used.

### 3.3 Field rations [F unless marked]

- Whiting: Lauerburg et al. 2018 Table A3 (bioenergetic model): 157 g fish 1.70-1.83 g/d (~1.1 % BW/d),
  411 g 3.68-3.97 g/d (0.9 %), i.e. Q/B ~3.3-3.9 [D]. Temperature coefficient 0.075 /C (Q10 2.1).
  Pedersen 2000 [A]: 0.41-1.75 %/d with Andersen's evacuation model, 4-10 %/d with in-situ rates.
- Saithe: Cormon et al. 2016 Table 2 [A]: ration c_q w^d with c 0.453 / 1.313 / 0.699 / 0.823 and d ~1.01-1.03;
  if kg per quarter (SMS convention), ~3.3 /yr [D, units not stated].
- Mackerel: Mehl & Westgard 1983: "daily ration of 0.6% of an individual's body weight per day", annual
  consumption "approximately 2.25 times the biomass", "on the lower side". Bachiller et al. 2018 (Norwegian
  Sea bioenergetics, doi:10.1371/journal.pone.0190345): up to "0.08 g prey g fish-1 day-1" in summer and
  C/B 9.5-12 /yr; a high-latitude upper bound, not used.
- Hake juveniles (Mediterranean, 13 C; Modica et al. 2011, MEPS 440:177): 1.5 % BWW/d (exponential model).
- Spurdog, anglerfish: no ration found; goosefish evacuation is "nearly ten times slower" than summer
  flounder (Stehlik et al. 2021 [A]), consistent with a low Q/B.

### 3.4 Chosen Q/B (annual, per annual-mean biomass)

| Species | Central | Range | Basis |
|---|---|---|---|
| Saithe | 3.0 | 2.87-3.6 | SMS 2.87; Ecopath adult 3.6; Cormon ~3.3 |
| Whiting | 4.2 | 3.3-5.6 | SMS per age on box sizes; Lauerburg 3.3-3.9; Ecopath 5.63 (adult/juv mix) |
| Mackerel | 2.2 | 1.7-3.2 | SMS 1.87, Ecopath 1.73, Mehl & Westgard 2.25 ("lower side"); 3.2 if presence = feeding season only |
| Horse mackerel | 3.6 | 3.5-4.8 | SMS 3.60, Ecopath 3.51; 4.8 = Q2-Q4 rate |
| Hake | 3.5 | 2.2-5.8 | large box hake (32 % >= 70 cm): SMS per age 3.5-4.5; Ecopath 2.2; SMS text implies ~2.9 |
| Spurdog | 2.0 | 1.5-2.5 | Ecopath (judgement range) |
| Anglerfish | 1.8 | 1.5-2.2 | Ecopath 1.9 (data 1.7) |
| **Group** | **3.35** | **2.85-4.35** | biomass-weighted [D] |

Season: mackerel and horse mackerel feed little in winter (SMS Q1 C/B 0.05 and 0). Mackerel overwinter
along the Norwegian Deep edge ("Mackerel stop feeding almost completely during winter", ICES FishMap [F]);
horse mackerel are in the box roughly Jul-Nov. Their biomass shares are Q1+Q3 long-run averages, so the
annual Q/B values above (which include non-feeding quarters) match that weighting; using the
feeding-season rates on the annual-mean share would double-count residency. The Q/B values carry North
Sea temperatures; the box bottom water is 6-9 C, so in-area rations may be ~10-20 % lower at depth
[judgement, from the -8 %/C evacuation exponent].

## 4. Light dependence of feeding

The existing gadoid report (`/home/simon/eco-evo/reports/Torskfiskars ljusberoende vid jakt.md`)
recommends dark_ratio 0.6, threshold -5 deg for gadoids -> herring, and 1.0 for krill, benthos and
cannibalism. It reads dark_ratio as a diel-overlap parameter fitted to stomach data, not to retinas.
The same reading is used here. New evidence per species:

| Species | Fish pair | Zooplankton pair | Evidence |
|---|---|---|---|
| Saithe | 0.6 (0.5-0.7) | 1.0 | no diel stomach study found; activity peaks at sunrise (Skilbrei & Ottera 2016, gadoid report); krill sits deep by day and shallow at night (Bergstad et al. 2003 [F]: *M. norvegica* "ascends to the surface at night"), reachable both times |
| Whiting | 0.7 (0.5-1.0) | 1.0 | Rindorf 2003 [F]: "free swimming prey ... were eaten mainly in the daylight hours", benthic prey more at night (4 of 5 sites); krill "ingested primarily around dawn and in the early afternoon until some time after dusk". Pedersen 2000 [A]: "Pelagic ... whiting fed at night, while demersal whiting fed by day." Mergardt & Temming 1997 [A]: one peak at 22-24 h. Onsrud 2004: krill 82 % day and night |
| Mackerel | 0.15 (0.1-0.2) | 0.4 (0.3-0.5) | Dahl & Kirkegaard 1986 [F]: empty stomachs 36.8 % at 01-06 h vs 7.9 % at 14-16 h; 25-29 cm contents 0.71 g at night vs 4.3-4.5 g at 14-22 h; "mackerel are feeding only during the day". Macy et al. 1998 [F]: filter feeding continues to ~1e-7 uE m-2 s-1, "This threshold enables mackerel to feed throughout the night near the ocean surface", but the start threshold rose from 4.4 to 14.7 copepods/l as light fell |
| Horse mackerel | 0.3 (0.2-0.4) | 0.8 (0.6-1.0) | Dahl & Kirkegaard [F]: few night samples, "the same tendency as found for mackerel"; Adriatic (Jardas et al. 2004 [A]): "highest feeding intensity ... at night and during early morning hours", euphausiids at night |
| Spurdog | 0.85 (0.7-1.0) | 0.85 | Kloecker et al. 2024 [F]: DVM, median depth 198.5 m by day, 60 m at night; fast starts more by day overall but mixed per individual; feeds "both during the day and night" |
| Hake | 0.5 | 1.0 | Modica et al. 2011 [F]: "macrozooplankton (euphausiids) was consumed at night"; larger juveniles "preyed largely on decapod crustaceans and fishes and fed mostly during the day" |
| Anglerfish | 0.5 (low confidence) | - | Colmenero et al. 2010 [F]: catches fell after dark (trawl activity, not feeding); ambush predator with a lure |

Consumption-weighted over the species [D] (`tables/group_diet_stdout.txt`):
**pair on planktivorous_fish 0.61**, pair on benthivorous_fish 0.66, **pair on zooplankton 0.92**.
Saithe and whiting give 85 % of the P weight and 81 % of the Z weight, so the result is the gadoid
value. Recommendation: **dark_ratio 0.6, light_threshold_deg -5 for predatory_fish -> planktivorous_fish**
(as for the gadoid pair); **1.0 (no key) for predatory_fish -> zooplankton** (0.92 is inside the noise,
and whiting/hake/saithe krill feeding peaks at dusk-night, which the engine cannot express above 1).
Benthivorous fish (juvenile cod/haddock) prey: 0.6-1.0; leave at 1.0 unless one value is wanted for all
fish pairs. Mackerel's strong day bias is real but is 4-11 % of the weights.

## 5. Functional response and prey switching

- **Whiting and cod on fish: negative switching (anti-Type III).** Rindorf, Gislason & Lewy 2006,
  MEPS 325:243 (36 000 stomachs) [F]: model ln(ni/nj) = a + b ln(Ti/Tj); b = 0.15-0.22 with whiting as
  reference prey and 0.28-0.35 with Norway pout. "Prey preference is a decreasing function of the relative
  density of the prey (negative prey switching), in particular for large cod". Larsen & Gislason 1992
  (cited there): for whiting the fit improved down to b = 0.4. Type III is not supported for whiting
  on gadoid and clupeid prey. doi:10.3354/meps325243.
- **Aggregative response.** Whiting aggregate where sandeel are dense (Temming et al. 2004; Engelhard et
  al. 2008 [S, search summaries]). Kempf, Floeter & Temming (ICES poster [S, snippet]): an S-shaped
  response at stock scale can come from spatial overlap. In the model the policies move, so overlap is
  learned; this does not argue for Type III in the pair.
- **Mackerel on copepods: Type III-like.** Macy et al. 1998 [F]: clearance "increased to a maximum of
  19.8 l fish-1 min-1 at a concentration of 3.7 l-1 before declining"; feeding starts at 4.4-14.7
  copepods/l and stops at 0.4/l (hysteresis); "consistent with filter feeding at high concentrations and
  particulate feeding at low concentrations". Mackerel on fish: opportunistic, Type II (tracks 0-group
  herring patches; Dahl & Kirkegaard).
- Saithe on krill, spurdog, hake, anglerfish: no functional-response study found.
- Recommendation [judgement]: **Type II for the fish pairs**; Type II for the zooplankton pair as
  well, since the krill eaters (saithe, whiting) dominate and mackerel is 11 % of the Z weight; Type III
  on zooplankton would need a reason beyond mackerel.

## 6. Predator interference

- Rindorf et al. 2006 [F]: "Although predator density had a positive effect on b in all cases, none of
  the reference fishes were significantly affected by this factor."
- Lauerburg et al. 2018 [F]: whiting "mean stomach content mass was on average 60% less" in the
  low-forage years 2000-2007. This is a prey-density effect, not interference.
- Mackerel: no study of stomach fullness vs local mackerel density found. Kvaavik et al. 2019 [A]: catch
  per station had "both a negative and positive effect on stomach weight". Density dependence is at
  stock scale (growth; Jansen & Burns 2015 [A]). Mackerel-herring diet overlap 0.77 without a prey
  switch (Bachiller et al. 2016).
- See also `/home/simon/eco-evo/reports/Täthetsberoende och interferens hos torskfiskar.md` (gadoid w 1.0
  was set to remove spatial bands, section 86, not from feeding data).
- Judgement: no field basis for interference in this group; a small w (0-0.1) is the
  evidence-based value. A larger w would be a numerical choice and should be named as such.

## 7. Source status (main items)

| Source | Status |
|---|---|
| ICES stomach DB public download (on disk), re-mapped | [F] `tables/ices_diet_pools*.csv` |
| Mackinson & Daskalov 2007, tech142 + EcoBase 457 | [F] `data/diet/ecopath/`, `../benthivores/data/ecobase/` |
| SMS key run 2023 files + WGSAM 2024 report | [F] `data/diet/sms_consumption/` (literature agent) |
| Mehl 2005 (saithe), Rindorf 2003, Rindorf et al. 2006, Lauerburg 2018, Kloecker 2024, Colmenero 2010, Modica 2011, Bergstad 2003 | [F] `data/diet/lit_gadoids_elasmo/` |
| Mehl & Westgard 1983, Dahl & Kirkegaard 1986, Macy et al. 1998, Bachiller 2016/2018, ICES FishMap, CINEA 2024 | [F] `data/diet/lit_mackerel/` |
| Hislop 1991, Pedersen 1999/2000, Holden 1966, Laurenson & Priede 2005, Cormon 2016, Jardas 2004, Kvaavik 2019 | [A] |
| Du Buit 1991, Ellis 1996, Hislop 1997 CRR 219, Rindorf & Gislason 2005, Langoy 2012, Mergardt & Temming full text | not obtained (paywall or blocked) |

## 8. Bottom line

Per species (shares of total food by mass; central, range in brackets; W kept). Inputs:
`inputs/diet_consumption_central.csv`; group numbers from `scripts/group_diet.py`.

| Species (share) | P | Bf | W | Z | B | O | Main data (year, season) | Q/B | dark_ratio fish / zoo |
|---|---|---|---|---|---|---|---|---|---|
| Saithe (41 %) | 0.58 (0.35-0.75) | 0.05 (0.02-0.15) | 0.02 (0-0.03) | 0.33 (0.15-0.6) | 0.01 | 0.01 | Mehl 2005 coast 62-63.5 N, Oct-Nov 1998-2003; Ecopath 1991; ICES 1981 by number (PRE-2000) | 3.0 (2.87-3.6) | 0.6 / 1.0 |
| Whiting (37 %) | 0.28 (0.15-0.75) | 0.08 (0.04-0.21) | 0.15 (0-0.46) | 0.28 (0.02-0.82) | 0.21 (0.07-0.35) | 0 | ICES Skagerrak Apr 1991 (PRE-2000) and 2022 (thin); C/S N Sea 2022; Ecopath 1991; Onsrud 2004 | 4.2 (3.3-5.6) | 0.7 / 1.0 |
| Mackerel (8 %) | 0.30 (0.07-0.87) | 0.01 | 0.01 | 0.65 (0.1-0.9) | 0.02 | 0.01 | Mehl & Westgard 1981-82; Dahl & Kirkegaard Jul 1985; ICES Jun 2022, Jan-Feb 2025 | 2.2 (1.7-3.2) | 0.15 / 0.4 |
| Spurdog (5 %) | 0.42 (0.39-0.51) | 0.22 | 0.22 | 0.03 | 0.06 | 0.05 | Ecopath 1991 (Ellis 1996); ICES 1991 n 7 | 2.0 (1.5-2.5) | 0.85 / 0.85 |
| Horse mackerel (3.4 %) | 0.25 (0-0.65) | 0.07 (0-0.2) | 0.03 (0-0.1) | 0.60 (0.2-0.87) | 0.05 | 0 | Ecopath 1991; Dahl & Kirkegaard 1985; ICES Aug 2023 n 4-6 | 3.6 (3.5-4.8) | 0.3 / 0.8 |
| Hake (2.5 %) | 0.77 (0.52-0.83) | 0.01 | 0.21 (0.13-0.37) | 0 | 0.005 | 0.005 | ICES Skagerrak Aug 2024, n 50 | 3.5 (2.2-5.8) | 0.5 / 1.0 |
| Anglerfish (1.4 %) | 0.35 (0-0.44) | 0.30 (0.24-0.92) | 0.25 (0.21-0.86) | 0 | 0.02 | 0.08 | Ecopath 1991; ICES 2022-24 n 9 | 1.8 (1.5-2.2) | 0.5 / - |

Group `predatory_fish` [D]:

| Quantity | P | Bf | W | Z | B | O |
|---|---|---|---|---|---|---|
| Diet, consumption-weighted (use for M2) | **0.41** | **0.07** | **0.09** | **0.31** | **0.11** | 0.01 |
| Diet, biomass-weighted | 0.43 | 0.07 | 0.09 | 0.32 | 0.09 | 0.01 |
| Diet without W, renormalised (eat-share check) | 0.45 | 0.08 | - | 0.35 | 0.12 | 0.01 |

- **Q/B = 3.35 /yr (2.85-4.35)**, biomass-weighted. Consumption shares: whiting 47 %, saithe 37 %,
  horse mackerel 4 %, mackerel 5 %, spurdog 3 %, hake 3 %, anglerfish 1 %.
- **dark_ratio: 0.6 (threshold -5 deg) on planktivorous_fish; 1.0 on zooplankton; 1.0 (or 0.6) on
  benthivorous_fish.** Consumption-weighted values 0.61 / 0.92 / 0.66.
- **Functional response: Type II** on fish (negative switching measured); Type II on zooplankton.
- **Interference: no evidence**; w 0-0.1.
- Largest uncertainty: the Z share (0.31). It rests on saithe krill (pulse prey, 2-92 % between years)
  and in-area whiting krill from 1991. A Z share of 0.15-0.45 is defensible. Next: whiting W (cannibalism)
  0-0.46 between samples.
- Unidentified fish were allocated pro-rata over identified fish (and unidentified gadoids over
  identified gadoids) in every ICES-derived number. Cephalopods are O; they are <= 0.09 for any species.
