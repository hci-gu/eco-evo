# Fish biomass in the Mareld box by feeding group

Purpose: size the three proposed fish groups (planktivorous_fish, benthivorous_fish, predatory_fish)
before any model change. Survey: DATRAS NS-IBTS GOV, Q1 + Q3 2021-25, 265 valid hauls in the
Skagerrak + box rectangles, depth-stratified by the grid's area per depth bin (the convention of
`../Torskfiskarnas artsammansättning/gadoid_species_mix.md`). Every fish taxon caught is assigned
(`inputs/guilds.csv`, adult feeding mode; invertebrates dropped). Script `scripts/fish_guilds.py`
(run with `python3 -I`), outputs in `tables/`, stdout in `tables/fish_guilds_stdout.txt`. [D]

## Bottom line (share of all fish biomass caught, pooled Q1/Q3)

| Group | Share | Bootstrap 5-95 % | Q1 / Q3 | Swept area, wing, q = 1 (t/km2) | Main species (share of the group) |
|---|---|---|---|---|---|
| benthivorous_fish | 36 % | 30-43 | 49 / 32 | 2.0 | haddock 72, cod 9, flatfish ~13 (long rough dab 5, plaice 3, dab 3, witch 1), rays, hagfish |
| predatory_fish | 41 % | 32-49 | 25 / 46 | 2.3 | saithe 40, blue whiting 24, whiting 23, mackerel 6, spurdog, anglerfish, hake |
| planktivorous_fish | 23 % | 18-28 | 26 / 22 | 1.3 | herring 69, Norway pout 16, sprat 14; argentines, silvery pout < 1 |

- **Blue whiting is the swing species.** As a planktivore (it eats mostly krill): planktivores 32 %,
  predators 31 %, benthivores 36 % (Q1 26 / 25 / 49, Q3 35 / 33 / 32).
- **Years differ a lot** (pooled, depth-stratified): benthivores 21-53 %, planktivores 9-34 %,
  predators 25-52 %. The 2023-24 haddock peak gives benthivores 49-53 %.
- **Planktivores are a floor.** GOV is a bottom trawl: herring, sprat and mackerel in midwater are
  undersampled, and mesopelagic fish (pearlside 0.0 %, lanternfish) are barely caught. Their real
  share needs an acoustic survey (HERAS, July; Skagerrak is covered). Sandeels: ~0 (7 hauls).
- **Model comparison** (start biomass / 3600 km2): pelagic_fish 3.3 t/km2 vs GOV planktivores 1.3
  (consistent with GOV undersampling); gadoids 0.69 vs GOV benthivores + predators 4.2 (the gap of
  `gadoid_species_mix.md` section 6, now with flatfish and the other predators added).
- Length-weight: 36 taxa fitted to NS-IBTS CA weights; 56 on body-form defaults, together 3.3 %
  of the biomass (long rough dab 2.0 %, the only one that matters).

## Long series 2000-2025 (`scripts/long_series.py`, `tables/long_series_*`) [D]

NS-IBTS Q1 + Q3 2000-2020 downloaded (`data/datras`, 66 MB gz; extracted with the species-mix
extractor into `data/extract`) and joined with 2021-25. Same estimator per year; 25-33 region hauls
per quarter every year from 2001 (2000 has one Q3 haul and is left out). All hauls record all species
(BySpecRecCode 1). Where a year/quarter has no 200-300 m haul, that stratum borrows 100-200 m
(flagged in `long_series_by_year.csv`). Length-weight is the 2021-25 fit, applied to every year.

**Haul-key fix.** Danish and Swedish cruises share the ship code 26D4 and haul numbers. The
species-mix scripts keyed hauls on (year, quarter, ship, haul number) and dropped the collisions:
2013-19 lost most of the Swedish hauls, and 2021-25 lost 7 of 272 region hauls. Keyed on
position as well, 2021-25 pooled gives haddock 47.0, whiting 17.1, saithe 29.3, cod 5.8, hake 0.8 %
(was 46.6 / 17.0 / 29.1 / 5.7 / 0.85) and groups P 24.0 / B 35.7 / R 40.3 % (was 22.7 / 36.3 /
41.0): no material change.

| Period (mean of years) | Planktivores | Benthivores | Predators | Haddock | Whiting | Saithe | Cod | phi benthos / fish / zoo |
|---|---|---|---|---|---|---|---|---|
| 2001-04 | 38 % | 34 % | 28 % | 30 | 31 | 15 | 22 | 0.54 / 0.36 / 0.47 |
| 2005-09 | 20 | 46 | 33 | 18 | 20 | 23 | 37 | 0.50 / 0.44 / 0.45 |
| 2010-14 | 28 | 42 | 30 | 17 | 14 | 26 | 42 | 0.52 / 0.46 / 0.44 |
| 2015-19 | 31 | 25 | 43 | 7 | 35 | 23 | 32 | 0.39 / 0.46 / 0.49 |
| 2020-25 | 26 | 34 | 40 | 42 | 22 | 27 | 8 | 0.54 / 0.38 / 0.51 |
| **2001-25** | **28** | **36** | **35** | 24 | 24 | 23 | 28 | **0.50 / 0.42 / 0.48** |

- **The benthivore species turned over, the group did not.** Cod was 50-60 % of the gadoids in
  2007-16 (100-165 kg/h), whiting dominated 2017-20 (haddock fell to 8-9 kg/h in 2018-19), and
  haddock rose ~40x to 285-352 kg/h in 2023-25 (72 % of the gadoids in 2023). Cod and haddock both
  forage on the bottom, so phi_benthos stays near 0.5 in every period except 2015-19 (0.39, the
  whiting years). Single years: 0.28 (2019) to 0.81 (2023).
- **2021-25 is a haddock peak, not the norm.** Haddock is 24 % of the gadoids over 2001-25 against
  47 % in 2021-25; cod 28 % against 6 %.
- phi (stage-resolved species scores from `../Torskfiskarnas bytesandelar`, raw variant, applied to
  each year's gadoid mix; species-level scores in `tables/species_prey_scores.csv`): the long-run
  means are benthos 0.50, fish 0.42, zooplankton (krill-sized) 0.48, against 0.56 / 0.37 / 0.52 for
  2021-25 pooled. The per-year phi uses the five main species only (minor species < 0.01).
- Groups: benthivores 16-67 % in single years, planktivores 9-51 %, predators 19-56 %. The
  planktivore share stays a GOV floor in every year.

## Blue whiting: decided planktivorous_fish (2026-10-24)

Mechanism, not diet:
- **Morphology**: terminal mouth, lower jaw longer, no barbel (benthos score 0-0.1); 26-34 gill
  rakers, more than cod, haddock and whiting, close to saithe's 35-40 and fit for krill-sized
  prey; slender pelagic body, large eyes, diel vertical migrant (FAO 1990, FishBase;
  `../Torskfiskarnas artsammansättning/evidence_C_diet_morphology.md`).
- **Size in the box**: 86 % of its GOV biomass is 20-30 cm, 13 % 30-40 cm. The model's fish prey
  is the herring pool (median herring length in the region hauls 19 cm). A 20-30 cm blue whiting
  cannot swallow it. The fish it can take (pearlside, lanternfish, 0-group) are < 1 % of the
  planktivores in the box. In the model, its food is zooplankton (the krill proxy).
- **Habitat**: Q3 visitor below 100 m (absent in Q1 and in water < 100 m), krill layer near the
  bottom by day.
- Diet check only: northern North Sea juveniles 100 % pelagic crustaceans (Robb 1981 [S]); Norwegian
  and Barents Seas, 29 cm, 60 % fish by weight (Dolgov et al. 2010 [S]), which is the larger, oceanic
  fish feeding on mesopelagics, not the box's 20-30 cm fish.

Shares with blue whiting in the planktivores (GOV, depth-stratified; re-run of both scripts):
2021-25 pooled P 32 / B 36 / R 31 % (bootstrap P 26-39, R 22-40; Q1 26 / 49 / 25, Q3 35 / 32 / 33);
2001-25 period means P 37 / B 36 / R 27 %. Blue whiting is a Q3 summer visitor (0 in Q1, GOV Q3
~1.0 t/km2; acoustically missed below 150 m): it adds roughly 0.5-1 t/km2 to the July acoustic
planktivore level of the grid (~4.7, `../Ekolodsdata planktivorer/acoustic_planktivores.md`) and
~0.3-0.5 to the annual mean. It is lean and deep-living compared with herring; with herring at
~80-90 % of the group's biomass the group parameters stay herring-like. Mackerel stays in
predatory_fish.
