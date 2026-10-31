# Evidence: whiting diet, prey sizes and foraging layer (2026-10-31)

Tags: [F] stated in a source, [D] computed here from data, [J] judgement. Scripts in `scripts/`, stdout
in `tables/`, downloads in `data/` (see `data/README.md`).

## 1. Diet by mass against predator length (North Sea)

**Hislop, Robb, Bell & Armstrong 1991** (ICES J Mar Sci 48:139; the 1981 Year of the Stomach, ~the whole
North Sea), via the ICES FishMap whiting factsheet (`data/lit/fishmap_whiting.txt`, Fig. 4 read from the
rendered page, +/- 3 points) [F]:
- "Immature whiting (< 20 cm) feed on crustaceans such as euphausids, mysids and crangonid shrimps."
- "Whiting >30cm feed almost entirely on fish, including a variety of small species like Norway pout,
  sprat, and sandeel, and the younger age classes of larger species such as herring, cod, and haddock."
- "Whiting feeds more on bottom-dwelling prey at night, whereas pelagic and free swimming prey are
  captured mainly during daylight."
- Fish share of the stomach mass by size class: 10 cm 0.30, 15 cm 0.39, 20 cm 0.60, 25 cm 0.78, 30 cm 0.88,
  40 cm 0.87, 50+ cm 1.0. Euphausiids ~0.1-0.2 and crangonids ~0.1-0.2 below 20 cm, both small above 25 cm.
- The same page (Fig. 5, IBTS Q1 1977-2005): "catches in the Skagerrak/Kattegat are dominated by
  1-year-old whiting" (mode ~17 cm against the North Sea's two modes at 15 and 26 cm).

**Robb, Bell, MacMillan & Hislop 1994** (ICES CM 1994/G:42; 1991 project, ~39 000 stomachs) [F]:
- "Fish und crustacea together accounted for at least 60% by weight of the stornach contents of all size
  classes ... With increase in size, the proportion of fish prey in the diet increased. A step-wise change
  in the size composition ofthe prey associated with the switch from crustacea to fish, occurred around the
  250 mm size group." (summary; OCR spelling kept)
- "a large proportion of the crustaceans eaten in the southern North Sea were Crangonids, whereas
  Euphausiacea were predominunt in the northern areas" (section 3.4).
- "Sandeels und Norway pout were significant components of the diet at all times ... herring occurred
  mainly in the stomachs of larger whiting, und were most noticeable in Quarter 3."

**SMS 2023 key-run stomach file**, whiting as predator, North Sea 1981, 1985-87, 1991, 98 samples
(`tables/sms_whiting_diet_stdout.txt`) [D]. Diet share by mass of the SMS prey species; "other" = every
prey SMS does not model (crustaceans, gobies, dragonets, cephalopods, polychaetes):

| whiting | planktivores (herring / sprat / pout) | benthivores (cod, haddock, plaice, sole) | whiting | sandeel | other |
|---|---|---|---|---|---|
| < 20 cm | 0.081 (0.003 / 0.024 / 0.054) | 0.002 | 0.006 | 0.107 | 0.804 |
| 20-24 | 0.214 (0.012 / 0.100 / 0.101) | 0.002 | 0.017 | 0.200 | 0.568 |
| 25-29 | 0.359 (0.039 / 0.127 / 0.193) | 0.034 | 0.028 | 0.190 | 0.389 |
| 30-34 | 0.514 (0.080 / 0.091 / 0.342) | 0.029 | 0.041 | 0.119 | 0.296 |
| >= 35 | 0.522 (0.087 / 0.067 / 0.368) | 0.075 | 0.039 | 0.079 | 0.284 |

Within the planktivore-pool prey whiting eat Norway pout 0.61, sprat 0.26, herring 0.12 (all sizes) [D],
whereas herring is 0.79 of the pool. Q2 has the lowest planktivore share (0.14), Q4 the highest (0.34).

Weighted to the box whiting length mix by consumption (biomass x L^-0.6; box biomass < 20 cm 0.25,
20-24 0.31, 25-29 0.23, 30-34 0.13, >= 35 0.09 from `../Fiskgrupper i rutan/predators/tables/
pred_length_biomass.csv`): planktivores 0.255, benthivores 0.016, all SMS fish 0.443; Hislop's all-fish
share 0.658 -> invertebrates 0.342 (`tables/clearance_test_stdout.txt`) [D].

## 2. In-area stomachs (ICES stomach database, by mass)

`scripts/whiting_diet_ices.py` -> `tables/whiting_diet_ices_stdout.txt` [D]. Pooled over stomachs, no
catch weighting; the 1981 upload has counts only (no weights) and the 1991 upload holds weights for 256
whiting prey rows, so in-area by-mass data are thin. Shares of identified prey mass:

| area, period | stomachs (fed) | plk | ben | own | zoo | epi | infauna | other fish | top items |
|---|---|---|---|---|---|---|---|---|---|
| Skagerrak, Apr 1991 (Argos) | 106 (75) | 0.069 | 0.002 | 0 | **0.629** | 0.168 | 0.092 | 0.035 | euphausiids 0.54, Pandalus 0.14, pout 0.07, Maurolicus, Lumpenus; no herring |
| Skagerrak 1991, 20-29 cm | 63 (50) | 0.07-0.10 | 0 | 0 | 0.66-0.70 | 0.08 | 0.07-0.13 | 0.03-0.06 | |
| Skagerrak, 2000+ | 539 (518) | 0.047 | 0.13 | 0.69 | 0.003 | 0.052 | 0.018 | 0.06 | 95 g of prey; 9 whiting eaten dominate |
| Kattegat, 1991 | 172 (119) | 0.47 | 0 | 0 | 0.17 | 0.12 | 0.09 | 0.13 | sprat 0.22, herring 0.20, euphausiids 0.17, Crangon 0.06 |

The deep in-area sample (Skagerrak, April) is krill-dominated with pout as the only planktivore; the
shallow Kattegat sample is clupeid-dominated. Robb et al. (euphausiids predominant in the north) agree.

## 3. Foraging layer: pelagic and demersal whiting

**Pedersen 1999** (J Fish Biol 55:1096; abstract, `data/lit/pedersen_19427.txt`) [F]: "Pelagic North Sea
whiting ... preyed upon (pelagic) sprat and sandeel, while demersal whiting preyed upon (demersal) Norway
pout and herring ... The diet of whiting shifted from amphipods and mysids to fish with increasing
predator length, and the length of prey consumed increased significantly with length of whiting."
**Pedersen 2000** (J Fish Biol 57:402; `data/lit/pedersen_19426.txt`) [F]: "Pelagic North Sea whiting ...
fed at night, while demersal whiting fed by day."

**Rindorf 2003** (MEPS 249:265; `../Attackrater ur mötesteori/data/gap_validation/rindorf2005/
rindorf2003.txt`) [F], 7738 whiting 20-35 cm, five North Sea locations:
- "the occurrence of bottom dwelling prey increased significantly during the night at 4 of 5 locations.
  In contrast, free swimming prey and prey migrating towards the demersal layer during the day were eaten
  mainly in the daylight hours." (abstract)
- On the deepest location (5: 137 m, 8.6 C, the most box-like): "The lack of 'other benthic
  invertebrates' was mirrored by the virtual absence of both crabs and flatfish, suggesting perhaps that
  whiting at this location did not forage on the bottom or that the density of benthic prey was low.
  The lack of benthic foraging may have been linked to the greater depth of this location." (p. 273)
- "most authors conclude that small whiting are pelagic and do not migrate vertically in July and
  August (Bailey 1975, Robb 1981, Bromley & Kell 1995)." (p. 273)
- "Invertebrate prey becomes progressively less important as predator size increases"; shrimps were
  "a mixture of Pandalus and Crangon spp.", eaten by day where Pandalus is on the bed (p. 273).
- Mean prey weights (Table 4): shrimp 0.80 g, crab 0.29 g, krill 0.12 g, herring 8.26 g, sprat 9.27 g.

**The library's own whiting layer** [J, earlier note]: whiting "live within ~5-30 m of the bed by day and
partly rise at night" (H_whiting 10 m, 5-20 m; `../Rovfiskens uppdelning/scripts/split_encounter_223.py`,
used for the water column's and the benthivores' pairs on whiting).

## 4. Prey size windows

SMS stomachs, planktivore-pool prey in whiting, mass-weighted (`tables/sms_whiting_diet_stdout.txt`) [D]:

| prey | whiting band | prey length p05 / p50 / p95 / p99 (cm) | prey/predator length p50 / p95 / p99 |
|---|---|---|---|
| herring | 20-24 | 8.5 / 8.5 / 11.0 / 11.0 | 0.38 / 0.49 / 0.49 |
| herring | 25-29 | 8.5 / 8.5 / 12.5 / 13.5 | 0.31 / 0.45 / 0.49 |
| herring | 30-34 | 8.5 / 8.5 / 13.5 / 13.5 | 0.26 / 0.42 / 0.42 |
| sprat | all | 5.5 / 6.0 / 8.5 / 12.5 | 0.22 / 0.34 / 0.48 |
| Norway pout | all | 5.5 / 6.0 / 8.5 / 11.0 | 0.23 / 0.41 / 0.49 |
| all planktivore prey | 20-24 / 25-29 / 30-34 | p95 8.5 / 9.0 / 9.0 | p95 0.38 / 0.33 / 0.28 |

Herring eaten by whiting >= 20 cm: Q1 p50 8.5 cm (p95 12.5), Q3 8.5 (9.0), Q4 9.0 (9.0). Rindorf &
Gislason 2005 [F]: maximum clupeid length in 25-35 cm whiting stomachs 14 cm (modal 12-14), gadoids 11 cm.
`../Årsungarnas andel/evidence_prey_size.md` 2.2 [D]: whiting 20-30 cm p95 0.38 (SMS, all fish prey), ICES-DB
0.46 by number, 0.48 by mass.

Box GOV planktivore juveniles (`tables/planktivore_access_stdout.txt`) [D]: share of the herring J biomass
<= 10 cm 0.000-0.032 by quarter (<= 13 cm 0.02-0.37); sprat J <= 10 cm 0.09-0.83; Norway pout 0.01-0.56.
Herring and sprat have no catchability correction in that table (`../Årsungarnas andel/summary.md`).
The 8.5-cm herring that whiting eat elsewhere hardly exist in the box survey, and the in-area April
stomachs hold no herring at all.
