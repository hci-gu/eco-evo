# Evidence A: NS-IBTS gadiform catches in the Mareld box and the Skagerrak, 2021-2025

All numbers here are **[D]**: computed from DATRAS downloads (`data/README.md`) with
`scripts/extract_datras.py` and `scripts/gadoid_mix.py`; full outputs in `tables/`
(`gadoid_mix_stdout.txt` is the console log). Method as in
`../Fiskeridödlighet i Skagerrak idag/ices_gadoid_f.md` ("Weighting"), extended to every gadiform.

## 1. Haul sets

- Valid (HaulVal V) GOV hauls, NS-IBTS Q1 and Q3, 2021-2025. A species absent from a haul is a zero
  (haul universe from HH, not from the CPUE table).
- **box rects**: ICES rectangles 44G0, 44G1, 45G0, 45G1, 120 hauls (58 Q1, 62 Q3; 118 Swedish, 1 DK,
  1 NO; 5 invalid hauls excluded). Reproduces the earlier note's 120 hauls.
- **box strict**: shoot position inside the grid's bounds 57.81-58.31 N, 10.07-11.15 E: 41 hauls (25 Q1, 16 Q3).
- **Skagerrak**: east of the Hanstholm-Lindesnes line, Kattegat (south of 57.75 N, east of 10.6 E)
  excluded: 232 hauls (earlier note 239; the 59.2 N / 12.5 E outer limits are mine).
- **region** = Skagerrak plus the box rectangles: 265 hauls (used for depth bins).

Hauls per box rectangle and depth bin (box rects, Q1+Q3), with the grid's area per bin from
`maps/depth.npz`:

| Rectangle | <100 m | 100-200 m | 200-300 m | >=300 m |
|---|---|---|---|---|
| 44G0 | 27 | 6 | 0 | 0 |
| 44G1 | 34 | 2 | 0 | 0 |
| 45G0 | 0 | 20 | 7 | 0 |
| 45G1 | 15 | 9 | 0 | 0 |
| **Box rects total** | **76** | **37** | **7** | **0** |
| Box strict | 12 | 22 | 7 | 0 |
| Region | 177 | 66 | 22 | 0 |
| **Grid area share** | **23.2 %** | **49.1 %** | **19.1 %** | **8.7 %** |

- Box-rect haul depth: median 68 m, mean 90 m, max 237 m; the grid's mean depth is 168 m. The
  deepest valid haul in the whole region is 260 m (45F9). **No GOV haul samples the 300-530 m trench
  slope**, which is 8.7 % of the grid.
- 44G1 hauls (34 shallow) lie mostly in the Kattegat mouth / Swedish coast, largely outside the grid.
- Per year and quarter every rectangle has 1-5 hauls (`tables/box_hauls_year_quarter.csv`).

## 2. Gadiform taxa found

Order Gadiformes was taken from WoRMS classification of every AphiaID with CPUE > 0 in the regional
extract (116 taxa; `data/worms/`). Sixteen gadiform species occur; none of Macrouridae, *Raniceps*,
*Ciliata* or *Brosme* has a positive record in the region 2021-2025.

- **Tusk (*Brosme brosme*) is absent from every Skagerrak and box haul 2021-2025.** Its only NS-IBTS
  age-length-weight records in 2021-2025 are 7 Norwegian fish from 48F2, 51F1, 52E9, 52F0
  (northern North Sea).
- Blue ling (*Molva dypterygia*): one fish (56 cm) in one Norwegian haul.
- Threebeard rockling (*Gaidropsarus vulgaris*): one 12 cm fish, outside the region subsets used.
- Recording check (positive hauls by country, region): SE 234 hauls, DK 28, NO 3. Silvery pout (32
  positive SE hauls, 0 DK) and fourbeard rockling (93 SE, 2 DK) look under-recorded by DK, but DK
  fished only 28 region hauls and one in the strict box; the box is 98 % Swedish hauls, so
  no non-recording correction is needed for the box.

## 3. Length-weight relations (W g = a L^b, L cm)

Fitted to NS-IBTS CA individual weights 2021-2025 (all areas, Q1+Q3; class midpoints; two passes of
3-SD log-residual trimming) where n >= 50 **[D]**; otherwise the FishBase Bayesian LWR from the species
summary page **[F]** (fetched 2026-10-24, `data/fishbase/`).

| Species | n used | a | b | Source | FishBase a, b (check) |
|---|---|---|---|---|---|
| Haddock | 64 816 | 0.00545 | 3.148 | CA fit | |
| Whiting | 44 786 | 0.00651 | 3.058 | CA fit | |
| Saithe | 6219 | 0.00649 | 3.079 | CA fit | |
| Cod | 13 566 | 0.00579 | 3.140 | CA fit | |
| Hake | 2764 | 0.00517 | 3.065 | CA fit | |
| Ling | 293 | 0.00371 | 3.088 | CA fit | 0.00380, 3.08 |
| Poor cod | 186 | 0.00384 | 3.305 | CA fit | 0.00741, 3.11 |
| Bib | 107 | 0.00606 | 3.220 | CA fit | 0.00741, 3.10 |
| Norway pout | 11 617 | 0.00553 | 3.093 | CA fit | 0.00603, 3.06 |
| Blue whiting | 4087 | 0.00310 | 3.186 | CA fit | |
| Pollack | (47) | 0.00692 | 3.08 | FishBase | |
| Tusk | (7) | 0.00525 | 3.10 | FishBase | |
| Greater forkbeard, silvery pout, fourbeard rockling, threebeard rockling, blue ling | 0 | 0.00457, 0.00661, 0.00407, 0.00513, 0.00282 | 3.14, 3.08, 3.07, 2.99, 3.05 | FishBase | |

CPUE length classes are 1 cm for every gadiform in the region; weight is taken at the class midpoint
(lower bound + 0.5 cm). Against the earlier note (2024-only fits, five species) the box-rect
five-species total is 398 vs 376 kg/h and the shares agree within 0.6 points.

## 4. Mean CPUE (kg/h), box rects, haul means

| Quarter (n) | Haddock | Whiting | Saithe | Cod | Hake | Ling | Pollack | Poor cod | Rocklings | Forkbeard | Silvery pout | Norway pout | Blue whiting | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 (58) | 114.2 | 95.5 | 1.1 | 21.3 | 0.41 | 0.56 | 0.48 | 0.65 | 0.15 | 0.15 | 0.00 | 6.7 | 0.0 | 241 |
| Q3 (62) | 158.8 | 211.2 | 154.9 | 19.2 | 9.0 | 0.92 | 0.12 | 0.21 | 0.32 | 0.02 | 0.09 | 46.8 | 50.6 | 652 |
| Q1+Q3 (120) | 137.2 | 155.3 | 80.6 | 20.2 | 4.9 | 0.75 | 0.29 | 0.42 | 0.24 | 0.08 | 0.05 | 27.5 | 26.1 | 454 |

Box strict, Q1+Q3 (41): haddock 236, whiting 105, saithe 97, cod 35, hake 6.5, ling 1.75, pollack
0.76, poor cod 0.93, rocklings 0.44, forkbeard 0.12, silvery pout 0.13, Norway pout 48, blue whiting
74; total 606 kg/h. Skagerrak (232): total 504 kg/h. Rectangle-mean-then-mean differs from the haul
mean by < 10 % in total (`tables/mean_kgh.csv`).

## 5. CPUE by depth bin (region hauls, kg/h)

| Species | Q1 <100 (81) | Q1 100-200 (30) | Q1 200-300 (14) | Q3 <100 (96) | Q3 100-200 (36) | Q3 200-300 (8) |
|---|---|---|---|---|---|---|
| Haddock | 88 | 200 | 63 | 246 | 400 | 118 |
| Whiting | 93 | 112 | 17 | 191 | 59 | 0.4 |
| Saithe | 0.7 | 1.6 | 19 | 1.2 | 265 | 459 |
| Cod | 15 | 32 | 16 | 20 | 38 | 18 |
| Hake | 0.1 | 0.3 | 2.3 | 8.9 | 4.4 | 9.7 |
| Ling | 0.07 | 0.38 | 1.78 | 0.24 | 2.2 | 2.3 |
| Fourbeard rockling | 0.05 | 0.85 | 0.73 | 0.09 | 1.1 | 2.2 |
| Silvery pout | 0 | 0 | 0.08 | 0 | 0.21 | 1.06 |
| Forkbeard | 0 | 0.15 | 0.28 | 0 | 0 | 0.15 |
| Norway pout | 3.7 | 4.5 | 34 | 34 | 65 | 25 |
| Blue whiting | 0 | 0.02 | 0.34 | 0 | 204 | 201 |

Whiting is a shallow species, saithe and blue whiting are deep summer species, ling, hake, rocklings,
silvery pout and forkbeard rise with depth. The shallow bias of the box-rect hauls therefore inflates
whiting and deflates saithe/blue whiting and the deep minor species.

## 6. Leverage and sampling error (box rects, Q1+Q3)

- Largest single-haul share of the box-rect sum: saithe 54 %, blue whiting 45 %, forkbeard 48 %,
  pollack 39 %, poor cod 37 %, silvery pout 86 %, hake 18 %, haddock 14 %, whiting 10 %, cod 9 %.
- Bootstrap over hauls (stratified by quarter, 2000 draws), share of the 15 non-zooplanktivore
  species, 5-50-95 %: haddock 24.6 / 34.2 / 45.0; whiting 28.4 / 38.7 / 51.1; saithe 3.2 / 19.7 /
  37.3; cod 3.8 / 5.0 / 6.6; hake 0.7 / 1.2 / 1.9; ling 0.12 / 0.19 / 0.28 %.

## 7. Swept-area density at catchability 1 (wing spread x distance)

Swept area per hour: median 0.140 km2/h (0.115-0.178). Density (t/km2, all 16 gadiforms / without
Norway pout and blue whiting):

| Haul set | Q1 mean | Q1 median | Q3 mean | Q3 median |
|---|---|---|---|---|
| Box rects (core) | 1.70 | 1.02 | 3.98 | 1.66 |
| Box strict (core) | 2.18 | 1.16 | 5.43 | 2.60 |
| Depth-stratified, region hauls (core / all) | 1.72 / 1.80 | 1.18 / 1.34 | 4.45 / 5.80 | 1.98 / 2.88 |

Pooled Q1+Q3 haul mean, box rects: 3.24 t/km2 all gadiforms, 2.88 core. The Q3 means are carried
by a few saithe hauls; the medians are 40-60 % of the means.
