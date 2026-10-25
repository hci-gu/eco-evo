# Evidence A2: Nephrops norvegicus in the Mareld grid

Scope: the Norway lobster share of the mobile epifauna that `benthic_community` should hold
(standing stock in t WW/km2 = g WW/m2, depth profile, r, accessible fraction). Builds on
`../Bentos och torskfiskar/evidence_A_standing_stock.md` (0.21 burrows/m2 FU 3-4 2024, 5-11 g/m2 on the
grounds) and `../Bentos och torskfiskar/evidence_B_production_and_food_limitation.md` (M, Ecopath P/B),
which are not repeated beyond what the arithmetic needs.

Tags: [F] full text read (incl. values read off a figure, "fig."), [A] abstract only, [S] search snippet
or secondary citation, [D] my derivation. Downloaded files were treated as data only.

## Short answer

| Quantity | ~100 m | ~300 m | Grid mean | Basis |
|---|---|---|---|---|
| Burrow density, 2011-24 mean (n/m2 of grid) | 0.16 (0.11 in 2024, 0.19 in 2025) | ~0.03 (0.01-0.05) | 0.15 (0.11 in 2024, 0.18 in 2025) | UWTV strata x stratum means [D] |
| Mean weight per counted animal (g) | 30 (23-45) | 30 (23-45) | | survey counts CL > 17 mm; catch weights [F] [D] |
| **Standing stock (g WW/m2 of grid)** | **4.7 (2.5-8.7)** | **0.9 (0.3-2.3)** | **4.5 (2.4-8.1)** | product [D] |
| r (/yr) | 0.34 (0.25-0.5) | | | Hiddink 2019 class > 10 yr; M 0.2-0.3; Ecopath P/B 0.35-0.4 |
| Accessible fraction f | 0.10 (0.05-0.20) | | | visible animals per burrow at peak emergence 0.086 |

The grounds-only values in the earlier note (5-11 g/m2, 8-19 in 2025) are per m2 of Nephrops ground. Per m2 of
the grid they are about half, because 30 % of the grid lies outside the survey strata and the strata
overlapping the grid have below-average densities (S8 is among the poorest strata in FU 3-4, with S7).

## 1. How much of the grid is Nephrops habitat

### 1a. UWTV survey strata (the best available habitat map)
- **WGNEPS 2025 report** (ICES Scientific Reports 7:107), FU 3&4 section, Figure 6.54 "Nephrops burrow
  density by station in 2024", which draws the nine survey strata (grey) [F, fig.].
  https://ndownloader.figshare.com/files/60000956
  - Table 6.17: "Stratified random, with buffer since 2017"; "since 2017: 9 strata"; "Adjusted mean density
    0.21 burrows/m2"; "Adjusted abundance 2975 million, 5.42 %" (CV). DK 90 and SWE 92 stations used [F].
  - The strata cover the Nephrops grounds (mud fished by Nephrops trawlers); the deep central Skagerrak is
    left white (outside the survey).
- **My overlay [D]**: I rendered Fig. 6.54 at 400 dpi, georeferenced it on the printed 10 E / 11 E and
  58.0 N / 58.5 N grid lines, and classed each pixel as stratum (grey 230) or outside (white). Each
  pixel was projected to SWEREF 99 TM and binned into the model's 60 x 60 cells (`tools/make_fishing_maps.py`
  convention). Files: `data/uwtv_fu34_2024_strata_crop.png`, `scripts/uwtv_share.py`,
  `data/uwtv_survey_share_per_cell.npy`. Georeferencing error is about +-1 km (one cell).
- **Result [D]**: **70 % of the grid lies inside the UWTV strata.** By depth band (depth from `maps/depth.npz`):

  | Depth band (m) | Share of grid | Share of band inside strata |
  |---|---|---|
  | < 100 | 0.23 | 0.75 |
  | 100-150 | 0.30 | 0.83 |
  | 150-200 | 0.19 | 0.91 |
  | 200-250 | 0.13 | 0.74 |
  | 250-300 | 0.06 | 0.17 |
  | > 300 | 0.09 | 0.00 |

  The excluded part is the deep NW quarter of the grid (the trench slope north of ~58.05 N and west of
  ~10.7 E). Three strata overlap the grid (my reading of the stratum borders, +-2 km): **S1** (Danish side,
  west of ~10.71 E; 39 % of the grid), **S8** (Swedish side, north-east; 25 %), **S9** (south-east corner; 5 %).

### 1b. EMODnet seabed substrate (cross-check)
- EMODnet Geology WFS, layers `gtk:seabed_substrate_250k` and `_1m` (Folk 7 classes), downloaded for the grid
  bbox (EPSG:3857) and rasterised at 9 points per cell [D] (`scripts/subs_grid.py`).
- Coverage is incomplete: 45 % of the grid has no polygon at 1:250k (only the Swedish SGU map, "SE-032",
  is present; the Danish and Norwegian side is missing at this scale). Where covered: muddy sand 30 % of
  the grid (mostly < 150 m), mud 19 % (mostly > 150 m), sand 2 %, restricted data 3 %.
- Both mud and muddy sand are Nephrops-capable (the species needs cohesive sediment for burrows; it is
  densest in mud with > 40 % silt-clay; this threshold is general knowledge, not read here). So the
  substrate map does not narrow the UWTV-strata answer and is not used further.
- The Mareld EIA (mkb.txt lines 1361-1363): "bottensubstratet i området huvudsakligen av postglacial lera,
  gyttjelera och lergyttja" [F].

## 2. Density inside the strata

- **WGNEPS 2025 Fig. 6.55**, burrow density by stratum, 2011-2024, mean +- SE [F, fig., +-0.02]
  (`data/wgneps2025_fig6_55_density_by_stratum.png`):

  | Stratum | 2024 | 2011-2024 range | rough mean |
  |---|---|---|---|
  | S1 | 0.19 | 0.16-0.37 | 0.26 |
  | S8 (from 2017) | 0.06 | 0.01-0.17 | 0.09 |
  | S9 (from 2017) | 0.32 | 0.32-0.72 | 0.55 |
  | FU 3-4 overall | 0.21 | | |

- **2025**: ICES advice nep.fu.3-4 (2025): "A 73% increase in stock abundance was observed in the 2025 survey";
  abundance 5063 million [F] (https://ndownloader.figshare.com/files/59204465). No stratum values for 2025
  were seen; I scale all strata by 1.73 [D].
- **Grid-mean density [D]** = sum over strata of (grid share inside stratum x stratum density):
  - 2024: (0.393 x 0.19 + 0.253 x 0.06 + 0.05 x 0.32) = **0.106 burrows/m2**;
  - 2011-24 means: **0.152**; 2025 (x1.73): **0.18**.
  - By depth band (2011-24 means): < 100 m 0.24, 100-150 m 0.16, 150-200 m 0.15, 200-300 m 0.11,
    > 300 m 0.00. 2024 values are ~0.7 of these.
  - At ~300 m: the 250-300 m band is 17 % inside the strata, so ~0.17 x 0.2 = **0.03 burrows/m2**. Nephrops do
    occur deeper (the Norwegian Deep stock, FU 32) but outside the surveyed grounds no density exists; I
    take 0.01-0.05.
- Burrow = animal is the survey's assumption ("adjusted" = corrected for edge effect, detection and
  multiple-entrance burrows). Aguzzi et al. 2021 (section 4) question 1:1 occupancy, which would lower
  biomass; I keep 1:1 as ICES does and put occupancy in the low case only implicitly via the low density.

## 3. Mean weight per animal

- ICES advice 2025 [F]: "Mean weight in projected landings (2026) 53.34 Average 2022-2024; grammes";
  "Mean weight in projected discards (2026) 23.45"; "Projected total discard rate (2026) 29" % by number.
  Catch mean weight = 0.71 x 53.3 + 0.29 x 23.5 = **44.7 g** [D].
- The UWTV counts animals above ~17 mm CL (earlier note, [F]); the population above that size contains
  more small animals than the catch, so the population mean is below 45 g. I take **30 g (23-45)** [D].
  Caveat: the UWTV counts burrows, not animals, so no size distribution of the surveyed population exists;
  "17 mm CL" is the size the ICES abundance is stated to represent (earlier note, [F]), not a measured size
  distribution of the burrow occupants. The 30 g is
  judgement between the discard (23 g) and catch (45 g) mean weights.
- Biomass cross-check [D]: 2975e6 x 30 g / (2975e6 / 0.21 m2 = ~14,200 km2) = 6.3 g/m2 on the grounds; ICES
  landings 2024 6493 t, i.e. ~0.46 t/km2/yr over the grounds, ~7 % of the stock per year, in line with the
  advised harvest rate 5.6-7.9 % by number [F].

## 4. Accessibility to benthivorous fish

- **Aguzzi et al. 2021**, Sci. Rep. 11:5797, doi:10.1038/s41598-021-85240-3 [F, PMC7970996], 3055 UWTV
  transects, Irish FUs, 15-570 m:
  - "The results showed an average level of 1 visible Nephrops individual per 10 burrow systems";
    "This corresponds to a visible occupancy of 0.086 individuals per burrow." Table 3 gives 0.08-0.11 by
    depth for the peak-emergence windows.
  - "full emergence varied from nocturnal toward midday hours with increasing depth of sampling";
    "predominantly dusk and dawn-oriented above 50 m"; "fully diurnal between 340 and 570 m".
  - Door-keeping (animal at the burrow mouth) is 18-41 % of visible animals on the shelf, 32-54 % on the slope.
  - "Even in periods of peak emergence it is possible that not all individuals are visible".
- Nephrops is eaten by cod in the Skagerrak: 3.5 % of cod diet by weight, IYFS Feb 1991 (Dorman et al. 1992,
  in `../Bentos åtkomlighet för torskfiskar/evidence_C_recent_diets.md`) [F via that note].
- **f [D]**: the fully exposed share is ~0.09 at peak emergence and lower outside it; door-keepers add a
  partial exposure. Central **0.10**, low 0.05 (time-averaged below the peak), high 0.20 (door-keepers and
  under-detection of emerged animals). This is half the infauna's 0.2.

## 5. Growth rate r

- M: "Natural mortality was set to 0.3 for males and 0.2 for females" (ICES C.M. 1993/Assess:11) [F via
  evidence_B]. Growth K 0.138-0.16 /yr, L-inf 73-76 mm CL (Ulmestrand & Eggert 2001) [A via evidence_B]:
  slow growth, lifespan > 10 yr.
- Ecopath North Sea (Mackinson & Daskalov 2007): Nephrops P/B 0.35-0.4 [S via evidence_B].
- Harvest rate 2022-24 [D]: landings 6493 t / 53.3 g = 122 M; discards 1254 t / 23.5 g = 53 M, 75 % dead = 40 M;
  (122 + 40)/2975 M = **5.4 %** by number, F ~0.056. Z = M + F ~0.26-0.36, so P/B ~0.3-0.35.
- Hiddink et al. 2019 longevity class > 10 yr: **r 0.34 /yr** [F via evidence_B].
- **r central 0.34, range 0.25-0.5.**

## Gaps
- Stratum borders were read off a printed map; the 2025 stratum densities were not seen.
- No in-grid UWTV station list was read; S8 (the Swedish strip) is among the lowest-density strata in FU 3-4, so the
  grid may be poorer than the FU mean in a way the stratum mean already captures but my S1/S8 border
  placement could shift by ~10 % of the grid.
- Mean weight of the surveyed population (> 17 mm CL) is my judgement between discard and catch weights.
- Occupancy: if Aguzzi's < 1:1 occupancy applied (it is debated), the biomass would be lower, not higher.
