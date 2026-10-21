# Subsurface chlorophyll maximum at SHARK A17 (section 180)

Purpose: the facts behind `subsurface_phytoplankton` and
`nutrients.subsurface_layer` in `mareld2.yaml` (resume section 180).
All values are monthly means over 2010-2023 of SMHI national monitoring
bottle data at A17 (58.28 N 10.51 E, Skagerrak, ~345 m), sample count in
brackets. [D] = derived by us from raw data, downloaded 2026-10-18.

## Request (reproducible)
```
POST https://shark.smhi.se/api/sample/download   (Content-Type: application/json)
{"params":{"tableView":"sharkdata_physicalchemical_columns","delimiters":"point-tab",
           "lineEnd":"unix","encoding":"utf_8","headerLang":"internal_key"},
 "query":{"fromYear":2010,"toYear":2023,"dataTypes":["Physical and Chemical"],
          "stationName":["Å17","A17"]}}
```
Depth bins: the nominal sampling depth +-1 m.

## Chlorophyll-a bottle (ug/l) [D]

| Month | 0 m | 5 m | 10 m | 15 m | 20 m | 30 m | 50 m |
|---|---|---|---|---|---|---|---|
| 1 | 1.39 (10) | 1.29 (12) | 1.08 (12) | 0.63 (12) | 0.45 (12) | 0.29 (12) | 0.24 (12) |
| 2 | 3.68 (13) | 4.02 (13) | 3.95 (13) | 1.68 (13) | 0.85 (13) | 0.34 (12) | 0.25 (13) |
| 3 | 3.13 (14) | 2.94 (14) | 3.78 (14) | 3.38 (13) | 1.53 (14) | 0.66 (14) | 0.39 (14) |
| 4 | 1.39 (14) | 1.32 (13) | 1.61 (14) | 2.44 (13) | 2.33 (13) | 0.86 (13) | 0.23 (13) |
| 5 | 0.92 (12) | 0.85 (12) | 1.07 (12) | 1.54 (12) | 1.30 (12) | 0.93 (11) | 0.41 (12) |
| 6 | 0.68 (11) | 0.69 (11) | 1.03 (11) | 1.29 (11) | 2.06 (11) | 1.35 (11) | 0.35 (11) |
| 7 | 0.38 (13) | 0.47 (12) | 0.53 (13) | 0.74 (13) | 1.16 (12) | 1.59 (13) | 0.33 (12) |
| 8 | 0.63 (12) | 0.63 (12) | 0.67 (12) | 0.88 (12) | 0.95 (11) | 1.01 (12) | 0.46 (12) |
| 9 | 0.78 (13) | 0.78 (13) | 0.82 (13) | 1.02 (13) | 0.90 (13) | 0.56 (13) | 0.36 (12) |
| 10 | 1.89 (14) | 1.86 (12) | 1.81 (13) | 1.06 (12) | 1.08 (13) | 0.78 (12) | 0.45 (13) |
| 11 | 1.56 (14) | 1.55 (14) | 1.50 (14) | 1.26 (14) | 0.89 (14) | 0.45 (13) | 0.36 (13) |
| 12 | 0.88 (14) | 0.81 (14) | 0.62 (14) | 0.55 (14) | 0.40 (14) | 0.30 (14) | 0.26 (14) |

Reading: from April to August chl at 15-30 m exceeds the surface 2-4x
(the maximum deepens from ~15 m in April-May to 20 m in June and 30 m in
July) and in June-July ~90 % of the 0-30 m chlorophyll lies below the
6 m mixed layer. The October maximum is a 0-10 m one.

0-30 m column chlorophyll (trapezoid over 0/5/10/15/20/30 m, mg chl/m2)
[D]: 23 66 73 52 34 39 26 25 25 40 34 17 (Jan..Dec).

## NO2+NO3 (umol/l) [D]

| Month | 0 m | 5 m | 10 m | 15 m | 20 m | 30 m | 50 m | 75 m |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.87 (10) | 5.06 (12) | 5.14 (12) | 5.33 (12) | 5.29 (12) | 5.73 (12) | 6.34 (12) | 6.27 (12) |
| 2 | 4.76 (13) | 4.90 (13) | 4.87 (13) | 5.99 (13) | 6.31 (13) | 6.58 (12) | 7.42 (13) | 8.22 (13) |
| 3 | 2.15 (14) | 2.28 (14) | 2.56 (14) | 4.03 (13) | 5.83 (14) | 6.79 (14) | 7.48 (14) | 8.19 (13) |
| 4 | 0.19 (14) | 0.14 (13) | 0.57 (14) | 1.83 (13) | 4.40 (13) | 6.80 (13) | 7.88 (13) | 8.66 (13) |
| 5 | 0.10 (12) | 0.10 (12) | 0.18 (12) | 0.95 (12) | 2.22 (12) | 4.50 (12) | 7.20 (12) | 7.98 (11) |
| 6 | 0.11 (11) | 0.10 (11) | 0.10 (11) | 0.10 (11) | 0.86 (11) | 3.96 (11) | 7.17 (11) | 8.00 (11) |
| 7 | 0.10 (13) | 0.10 (13) | 0.10 (13) | 0.10 (13) | 0.19 (12) | 2.58 (13) | 6.36 (12) | 7.87 (13) |
| 8 | 0.10 (12) | 0.10 (12) | 0.10 (12) | 0.15 (12) | 0.26 (11) | 1.26 (12) | 4.68 (12) | 8.08 (12) |
| 9 | 0.10 (13) | 0.10 (13) | 0.11 (13) | 0.13 (13) | 0.13 (13) | 0.84 (13) | 4.40 (12) | 7.21 (12) |
| 10 | 0.29 (14) | 0.25 (12) | 0.29 (13) | 0.86 (12) | 0.42 (13) | 0.81 (12) | 2.46 (13) | 5.99 (12) |
| 11 | 1.28 (14) | 1.31 (14) | 1.32 (13) | 1.40 (14) | 1.63 (14) | 2.46 (13) | 4.60 (14) | 7.13 (13) |
| 12 | 3.41 (14) | 3.56 (14) | 3.62 (14) | 3.77 (14) | 3.79 (14) | 3.72 (14) | 4.47 (14) | 5.30 (14) |

The nitracline sits at 15-30 m in summer, under the chlorophyll
maximum; the 50 m row is the manifest's `reservoir_umol_l`.

## NH4 (umol/l) [D]

| Month | 0 m | 5 m | 10 m | 15 m | 20 m | 30 m | 50 m |
|---|---|---|---|---|---|---|---|
| 1 | 0.30 (10) | 0.28 (11) | 0.31 (11) | 0.26 (11) | 0.24 (11) | 0.21 (11) | 0.21 (11) |
| 2 | 0.25 (13) | 0.23 (13) | 0.24 (13) | 0.33 (13) | 0.24 (13) | 0.24 (12) | 0.21 (13) |
| 3 | 0.32 (14) | 0.28 (14) | 0.28 (14) | 0.31 (13) | 0.25 (14) | 0.24 (14) | 0.22 (14) |
| 4 | 0.25 (14) | 0.20 (13) | 0.27 (14) | 0.27 (13) | 0.42 (12) | 0.33 (13) | 0.27 (13) |
| 5 | 0.19 (11) | 0.17 (11) | 0.21 (11) | 0.39 (11) | 0.43 (11) | 0.50 (10) | 0.37 (11) |
| 6 | 0.20 (11) | 0.19 (11) | 0.21 (11) | 0.19 (11) | 0.19 (11) | 0.36 (11) | 0.93 (11) |
| 7 | 0.19 (12) | 0.19 (12) | 0.20 (12) | 0.19 (12) | 0.19 (11) | 0.26 (12) | 0.25 (11) |
| 8 | 0.24 (11) | 0.21 (11) | 0.19 (11) | 0.21 (11) | 0.21 (10) | 0.45 (11) | 0.43 (11) |
| 9 | 0.22 (13) | 0.22 (13) | 0.22 (13) | 0.25 (13) | 0.37 (13) | 0.51 (13) | 0.40 (12) |
| 10 | 0.43 (14) | 0.35 (12) | 0.41 (13) | 0.50 (12) | 0.56 (13) | 0.64 (12) | 0.42 (12) |
| 11 | 0.25 (10) | 0.25 (10) | 0.26 (9) | 0.26 (10) | 0.29 (10) | 0.29 (9) | 0.31 (10) |
| 12 | 0.38 (13) | 0.36 (13) | 0.29 (13) | 0.26 (13) | 0.31 (12) | 0.30 (13) | 0.26 (13) |

## Temperature (C, CTD, bottle where CTD is missing) [D]

| Month | 0 m | 5 m | 10 m | 15 m | 20 m | 30 m | 50 m |
|---|---|---|---|---|---|---|---|
| 1 | 5.0 (11) | 5.2 (12) | 5.4 (12) | 5.9 (12) | 6.4 (12) | 7.3 (12) | 8.2 (12) |
| 2 | 3.1 (13) | 3.2 (13) | 3.7 (13) | 5.0 (13) | 5.7 (13) | 6.7 (13) | 7.4 (13) |
| 3 | 3.9 (14) | 4.0 (14) | 4.2 (14) | 4.8 (14) | 5.6 (14) | 6.3 (14) | 7.0 (14) |
| 4 | 6.3 (14) | 6.1 (14) | 5.7 (14) | 5.6 (14) | 6.0 (14) | 6.4 (14) | 6.9 (14) |
| 5 | 10.9 (12) | 10.0 (12) | 8.8 (12) | 7.4 (12) | 7.0 (12) | 6.7 (12) | 6.9 (12) |
| 6 | 14.8 (10) | 14.4 (11) | 12.7 (11) | 9.4 (11) | 8.4 (11) | 7.8 (11) | 7.2 (11) |
| 7 | 16.9 (11) | 16.2 (13) | 13.7 (13) | 12.5 (13) | 10.7 (13) | 8.5 (13) | 7.7 (13) |
| 8 | 17.9 (12) | 17.5 (12) | 16.8 (12) | 15.0 (12) | 13.1 (12) | 10.7 (12) | 8.4 (12) |
| 9 | 15.7 (13) | 15.6 (13) | 15.5 (13) | 15.4 (13) | 14.4 (13) | 11.9 (13) | 9.5 (13) |
| 10 | 12.7 (14) | 12.7 (14) | 12.8 (14) | 12.9 (14) | 13.3 (14) | 13.1 (14) | 12.1 (14) |
| 11 | 10.4 (14) | 10.5 (14) | 10.5 (14) | 10.7 (14) | 11.2 (14) | 11.4 (14) | 10.5 (14) |
| 12 | 6.8 (14) | 7.4 (14) | 8.1 (14) | 8.3 (14) | 8.5 (14) | 9.0 (14) | 9.5 (14) |

## Secchi depth (m), monthly mean of visits [D]

1: 5.0 2: 7.0 3: 8.7 4: 11.0 5: 10.5 6: 8.3 7: 11.7 8: 7.0 10: 11.0 11: 9.0 12: 11.0

Kd(PAR) below the mixed layer 0.17 /m in the manifest = 1.7 / Secchi
(summer Secchi 8-12 m); the surface PAR Kd 0.29 uses 2.5 / Secchi (Aas,
Hokedal & Sorensen 2014, section 138.3). The two conversions bracket the
blue-shifted light below the surface layer (section 144.2: effective
0.15-0.2). Calibrated within that range [D].

## Primary production target
The 190-290 g C/m2/yr in research_notes/Litteraturbaserad growth rate
per FG/plankton.md section 3 is 14C production in the Kattegat and at the
Gullmar Fjord mouth, not open Skagerrak; it is an upper reference for A17.


## Parameter sources for the subsurface layer (section 181 check)

Subagent literature check, 2026-10-19. Tags: [F] full text/table, [A]
abstract, [S] secondary, [D] derived arithmetic.

**I_k of subsurface-maximum phytoplankton (model 30; surface 111-298).
The model value is BELOW the regional range.**
- Lyngsgaard et al. 2014, MEPS 517:15-33, doi:10.3354/meps11015 [F,
  Table 2]: Aarhus Bight (Kattegat), July 2010, pycnocline/bottom layer
  I_k 63-117 (mean 92.2) umol photons m-2 s-1, surface mean 195.8
  (deep samples at 8-12 m, ~7.5 % of surface PAR).
- Bendtsen & Richardson 2018, Biogeosciences 15:7315-7332,
  doi:10.5194/bg-15-7315-2018 [F, Table 1]: NE North Sea July 2016; E_k
  = Pmax/alpha [D] SCM ~64-86, surface ~116-134 (ratio 0.45-0.5; the
  model's 30 / 111-298 is 0.1-0.27).
- Hickman et al. 2012, MEPS 463:39-57, doi:10.3354/meps09836 [F, read
  from Figs 4-5]: Celtic Sea SCM E_k ~20-100, densest 30-70; surface
  ~100-250. The only source whose range contains 30, at its low end.
- Moore et al. 2006, L&O 51:936-949 [A]: E_k falls from surface to SCM.
- Lund-Hansen 2006, Oceanologia 48:433-446 [F]: SW Kattegat SCM at an
  irradiance of 24 +- 11 uE (10-43) - not an I_k.

**Kd(PAR) (model 0.29 /m mixed layer, 0.17 /m below).**
- Aas, Hokedal & Sorensen 2014, Ocean Sci 10:177-199,
  doi:10.5194/os-10-177-2014 [F]: Skagerrak Secchi 8.3 +- 2.8 m
  (all-year), Kq x Secchi 2.5 +- 0.5 -> Kq ~0.30 /m [D]: supports 0.29.
- Bendtsen & Richardson 2018 [F]: kd 0.14 +- 0.03 /m (177 casts, July,
  56.2-58 N, 6.2-8.3 E, Atlantic-influenced, whole profile): 0.17 is
  ~1 SD above it - plausible. No Skagerrak source splits Kd by layer.

**Exchange across the summer pycnocline (model 0.2 m/d summer ~
4.6e-5 m2/s at dz 20 m; 1.0 m/d winter). Summer is 5-50x the nearest
measurement; winter unverified.**
- Bendtsen & Richardson 2018 [F]: open NE North Sea k_v < 1e-6 m2/s at
  15-30 m, open-water range 1e-6-1e-5, > 1e-4 only at the shelf edge and
  in the bottom boundary layer; open-water nitrate flux < 0.1 mmol N
  m-2 d-1.
- Palmer, Rippeth & Simpson 2008, JGR 113, doi:10.1029/2007JC004531
  [A]: Celtic Sea thermocline ~5e-5 m2/s; Hickman et al. 2012 [F] use
  5e-5 as typical (ventilation ~70 d over 25 m).
- Sharples et al. 2001, L&O 46:486-496 [A]: nitrate flux into the
  thermocline 2.0 (0.8-3.2) mmol m-2 d-1, W English Channel.
- No winter Kz for the Skagerrak/Kattegat found.

Reading: the subsurface layer's light (I_k 30) and nutrient supply
(0.2 m/d) are both on the generous side of the evidence; they were
fitted (section 180, rule eat_pays) to A17 nitrate and column
chlorophyll. A recalibration with I_k ~60-90 and summer exchange
~0.02-0.05 m/d would need another source of the A17 subsurface
production (recycling within the layer: section 179's remineralisation)
- a decision for the user, not made here.
