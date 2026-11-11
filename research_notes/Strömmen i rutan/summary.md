# Near-surface currents in the Mareld box - summary

Question: what moves the plankton in the 60 x 60 km box (SWEREF 99 TM E 211-271 km,
N 6414-6474 km; NW 58.31 N 10.07 E, SE 57.81 N 11.15 E), so the model's passive
plankton current can be set from data. Evidence ids refer to `evidence.md`.

Tags on every claim: **[M]** measured, **[Mod]** numerical model (NorKyst-800, E10,
unless stated), **[S]** secondary source, **[I]** my inference. NorKyst numbers are
for **2023** (360 of 365 days) unless 2022 is named; see "Gaps".

Conventions: u = eastward, v = northward, cm/s; "direction" = direction the water
goes TO, degrees clockwise from north. "Daily" statistics are on daily means of
hourly model output (this removes the ~1 cm/s tide and most of the 14.1-h inertial
oscillation).

## Bottom line

1. The surface water in the box moves at **~10-35 cm/s, mean towards the NE-N**,
   not 0.1-0.3 cm/s towards the E-S. The box mean at 0 m is 15.6 cm/s towards 36 deg
   (u +9.1, v +12.6) [Mod]; the measured mean at 50 m in the box's NE part is
   8-37 cm/s northward month by month (Svansson 1975, E4b) [M]. The model's field is
   50-150 times too slow and has the wrong sign of the north component.
2. Water enters over the **west and south edges** (Jutland Current water) and the
   **SE corner** (Kattegat/Baltic outflow), and leaves over the **north edge** towards
   the Norwegian Coastal Current; the east edge has ~zero net flow [Mod].
3. Surface water stays in the box **~1-3 days** (median first exit 1.6 d at 0 m,
   1.9 d at 25 m; 4.7 d at 25 m in spring) [Mod]. The whole Skagerrak turns over in
   ~100 days, "a few weeks in areas of strong currents" (E1, E3) [M/S].
4. Variability is large and fast: daily-mean std 10-30 cm/s per component,
   decorrelation (e-folding) 1.2-3.7 days, flow against the mean direction on 9-32 %
   of days, local spatial correlation gone within ~20-30 km [Mod]; the measured
   surface transients are ~20 cm/s (E1) and 5-day oscillations dominate at 50 m in
   the box (E4b) [M].
5. With ~2-day residence and copepod generation times of weeks, the box's
   zooplankton is an advected stock; the literature says the same for Calanus in the
   Skagerrak (E7, E8) [M + I].

---

## 1. Mean current: direction, speed, branches

### Measured

- **Inside the box, NE part, 50 m** (current meter "SW of Smogen", 58 14'N 11 03'E,
  bottom 100 m; 3 km from the box's east edge, 12.5 km from its north edge): monthly
  mean northward components 17 (Apr 1971), 19 (Jun 1967), 8 (Jul 1967), 9 (Aug
  1967), 14 (Sep 1967), 29 (Oct 1967), 37 cm/s (Oct 1971); "a summer minimum and a
  highest value in October" (E4b) [M].
- **Section off the Swedish coast** (Rodhe's section 2, starting NE of the box, 9
  crossings 1975-77): a "north-going basic current" with "the high velocity in the
  surface in the outer part of the Baltic current", mean surface contours 20-25 cm/s;
  northward transport per unit width 4-18 m2/s at stations 1-6, -6 m2/s at the
  outermost (E1) [M].
- **Skagerrak as a whole**: a cyclonic circulation "into the Skagerrak along the
  Danish slope of the Trench, to the north outside the Swedish coast and out again
  along the Norwegian coast" (E1) [M]. Its deep part has near-bottom maxima of
  15-20 cm/s (Danish slope), 10-15 cm/s (outside the Swedish coast), 5-10 cm/s
  (Norwegian side); these are deep values, not surface speeds (E1) [M].
- Surface drifters: "speeds up to 1 knot [51 cm/s] on the Danish side and sometimes
  up to 2 knots on the Norwegian side" (May-June 1990, E2) [M]. The Jutland Current:
  "not unusual to measure speeds of between 2 to 3 knots", and it "can ... veer to
  the north east towards the Vaderoarna", a path across the box (E3) [S].
- The Baltic Current "continues northwards along the Swedish coast until it joins the
  Norwegian Coastal current"; ~2500 km3/yr (0.08 Sv); "largest during Summer and
  lowest during Winter" (E3) [S]. Total Skagerrak through-flow ~15 000 km3/yr
  (0.48 Sv) (E3); geostrophic estimates 200 000 m3/s in February to 500 000 m3/s in
  November (E4b, E5) [M/S].
- Tides are negligible for plankton transport: "of the order of 1 cm/s" (E1),
  "about 1-2 cm/s" (E3) [M].

### Modelled (NorKyst-800, 2023; tables/points_annual_2023.csv, box_mean_map_10km_2023.csv)

Annual mean of daily means, 0 m / 25 m:

| point (position in box) | mean 0 m (cm/s, to deg) | mean 25 m | std u / v at 0 m (daily) |
|---|---|---|---|
| NW (7 km E, 7 km S of NW corner; 448 m deep) | 10.4 to 49 | 8.5 to 48 | 14.4 / 13.6 |
| N (north edge, middle; 232 m) | 17.9 to 4 | 14.4 to 5 | 15.1 / 20.4 |
| NE (Swedish side, 101 m; next to the Smogen mooring) | 36.3 to 340 | 28.9 to 340 | 13.7 / 23.7 |
| C (centre; 165 m) | 15.4 to 71 | 12.1 to 79 | 19.1 / 16.4 |
| SW (Danish side; 73 m) | 9.7 to 79 | 8.4 to 97 | 30.9 / 13.1 |
| S (south edge, middle; 118 m) | 15.5 to 59 | 11.3 to 85 | 31.7 / 14.6 |
| SE (12 km from the model's open boundary; 67 m) | 28.7 to 47 | 17.0 to 48 | 20.0 / 15.0 |
| **box mean (6-hourly field)** | **15.6 to 36 (u 9.1, v 12.6)** | **11.6 to 41 (u 7.6, v 8.7)** | |

At 50 m the NE point still has 22.8 cm/s to 340 deg. The measured 50 m monthly
means at the same place (8-37 cm/s northward, 1967/1971) bracket NorKyst's 2023
monthly means there (4-40 cm/s) [M vs Mod; different years, so only the range is
comparable].

Pattern on 10 km cells (0 m annual mean): the eastern third flows N-NNW at
20-33 cm/s (v up to 33 cm/s in the NE cell); the southern half and the west flow
E-ENE at 7-23 cm/s; the SE corner flows NE at ~20-25 cm/s.

Per edge (mean inward normal velocity; % of 6-hourly snapshots with inflow;
tables/box_edges_2023.csv) [Mod]:

| edge | 0 m | 25 m |
|---|---|---|
| west | +14.0 cm/s, in 78 % | +12.9, in 87 % |
| south | +12.9, in 87 % | +5.2, in 75 % |
| east | +0.3, in 52 % | +1.4, in 66 % |
| north | -16.7 (out), in 12 % | -13.9 (out), in 6 % |

2022 (partial: Jan-May, Nov, Dec; tables/box_monthly_2022.csv, points_monthly_2022.csv)
[Mod]: the same pattern. Box-mean 0 m speed by month 22.2, 25.8, 19.5, 9.6, 15.2
(Jan-May), 14.9, 12.3 cm/s (Nov, Dec), always with v > 0 (9-18 cm/s); the NE point
flows NNW at 24-72 cm/s (v) and the Eulerian exchange time is 1.2-1.8 d at 0 m.
The 2023 values for the same months are 22.5, 15.7, 10.6, 13.3, 10.7, 12.5,
10.6 cm/s.

### Which branches pass through the box

- **Jutland Current** (North Sea / Jutland coastal water entering along the Danish
  side): enters over the west and south edges, eastward along the south of the box
  [M: E1, E2, E3 describe it; Mod: the W/S inflow]. Its intrusions are wind-driven and
  intermittent: in 1990 a NW gale blocked it, and it reached section G only after
  southerly winds (E2) [M].
- **Baltic Current / Kattegat outflow**: enters at the SE corner and runs north
  along the Swedish side (the NE point) [M: E3, E4b; Mod]. In NorKyst the SE part of
  the box lies within ~20 km of the model's open boundary (SE corner 3.5 km), so this
  inflow is boundary-forced (E10) [caveat].
- **Combined northward coastal flow -> Norwegian Coastal Current**: leaves over the
  north edge [M: E1, E3; Mod].
- **Atlantic inflow along the trench slope**: deep (near-bottom maxima, E1); it is
  not a surface feature in the box. The box's north part lies over the trench slope,
  where Rodhe's deep circulation flows north "outside the Swedish coast" [M].
- **The cyclonic gyre**: the box is its south-east and east limb [M]. An
  anticyclonic SURFACE tendency in the eastern Skagerrak comes only from
  "unpublished modelling results" (E2); NorKyst 2023 does not show it in the
  annual mean (the NW point flows NE) [Mod vs unpublished Mod]. Johnson's model
  finds "Recirculation often occurs near the Swedish coast" (E5) [Mod].

## 2. Variability

### Measured

- Surface layer: "The magnitude of the transient Ekman velocity is about 20 cm/s and
  the depth of the Ekman layer is some 10-20 m"; along-topography std 17 cm/s at a
  100 m deep station; typical wind (and spin-up) duration ~1 day; at section 1 the
  std of the in-out velocity is 15-20 cm/s in the upper ~20 m (E1).
- Rodhe's section 2 (near the box): transport std 3-14 m2/s against means of
  4-18 m2/s, i.e. std comparable to the mean (E1).
- Reversals: "In the upper layers, the direction of the circulation occasionally
  changes" (E1); NCC water crossed to the Danish coast during NW gales (E2).
- In the box at 50 m: "Oscillations of 5-day type are apparently dominant" in the
  daily means; Baltic-driven oscillations of order 14 days move Kattegat water in and
  out along the Bohuslan coast (E4b).
- Seasonal: at 50 m in the box, a summer minimum and an October maximum (E4b); the
  Skagerrak through-flow is 0.2 Sv in February and 0.5 Sv in November (E4b, E5); the
  Baltic Current is largest in summer (E3).
- HF radar (MET Norway), the only openly available surface-current measurement near
  the box: its footprint ends ~5-15 km NW of the box (0 box cells with >= 30 %
  coverage, E11). At the nearest well-covered cell (58.385 N 10.005 E), 2023, QC-good
  hours (n = 1884): hourly std 18.8 / 20.0 cm/s (u / v), mean only 2.2 cm/s; daily
  means (46 days with >= 12 h): std 11.5 / 9.8 cm/s (tables/hfr_vs_norkyst_2023.csv)
  [M].

### Modelled (NorKyst 2023, daily means; tables/points_*_2023.csv)

- **Std per component**: 9-32 cm/s at 0 m, 7-27 cm/s at 25 m. Largest along the
  E-W axis in the south (SW 30.9, S 31.7 cm/s at 0 m) and N-S on the east side (NE
  23.7). Hourly std is 10-50 % higher than daily (inertial and tidal motion add to
  it).
- **Principal axes** (0 m): E-W (bearing 96-106 deg) in the south and SW, NNW-SSE
  (158 deg) on the Swedish side, N-S at the north edge; close to isotropic at NW and C
  (minor/major 0.8-0.9).
- **Steadiness** (|mean vector| / mean speed): 0.86-0.92 on the Swedish side and in
  the SE, 0.35-0.47 in the south-west (wind-driven, reversing), 0.6-0.7 elsewhere.
- **Reversals**: share of days with flow against the annual-mean direction: 9 % (NE,
  SE), 17-21 % (N, NW, C), 27-32 % (SW, S) at 0 m. The along-axis daily velocity
  changes sign 2-8 times per month (fewest at NE, most in the SW).
- **Decorrelation** of daily anomalies (calendar-month mean removed): e-folding
  1.2-3.7 days (median ~2.1 d), integral time 1.0-3.2 days; lag-1-day
  autocorrelation 0.59-0.80, lag-5-day -0.25 to 0.28. Longest on the Swedish side
  (NE, 2.4-3.7 d), shortest at NW and in the SW v component (~1-1.7 d).
- **Spatial correlation** of daily anomalies: below ~0.4 for most point pairs
  20-30 km apart, except u along the southern band (SW-S-SE, r 0.73-0.85 over
  24-51 km). There is also a coherent box-scale mode: at 25 m, NE-SE u is
  anti-correlated (r -0.65) and v correlated (+0.69) at 43 km, and NE-S u r -0.54
  (the Swedish-side current pulsing as a whole). A zero-mean random field with
  L ~ 20-30 km would miss this; it belongs to the mean / forced part
  (tables/points_spatial_corr_2023.csv).
- **Wind** (regression of daily currents on NorKyst's own 10 m wind at the centre;
  mean wind speed 7.0 m/s): the southern band follows the wind at lag 0 (complex
  correlation with wind stress 0.67-0.71 at SW, S and SE; surface current anomaly
  2-3 % of the wind speed). The regression angle is 20-30 deg counter-clockwise of the
  wind. This is a regression angle on daily means, with the flow steered along the
  E-W isobaths of the Danish slope, not a measured Ekman angle. Centre: 0.42 (1.4 %);
  NW: 0.43 (1.1 %). The Swedish-side current (NE) is only weakly and remotely linked
  (0.28 at 0 m; 0.44-0.55 at 25-50 m at ~140 deg, best at 1-day lag), which fits
  Kattegat/Baltic sea-level control rather than local wind drift
  (tables/wind_response_2023.csv). The classic rules of thumb, "1 % of the velocity of
  the wind" for the surface current and "3 % for drift bottles" (E4b), bracket these
  gains [M].
- **Seasonal** (box mean at 0 m by month; tables/box_monthly_2023.csv): 22.5 (Jan),
  15.7, 10.6, 13.3, 10.7, 10.4 (Jun), 27.6 (Jul), 20.8, 27.0 (Sep), 12.9, 12.5,
  10.6 cm/s (Dec). The range 10.4-27.6 cm/s is about +-45 % about the annual mean.
  At 25 m it runs from 2.9 (Apr) to 22.0 cm/s (Jul), with a spring minimum (Apr-Jun
  2.9-5.8 cm/s) when the stratified surface layer decouples. The model's +-13 % is
  too flat.
- **Model vs measurement**: at the HF radar cell NW of the box, NorKyst 0 m and the
  radar agree in variability (daily std 11.1 / 10.0 vs 11.5 / 9.8 cm/s; mean daily
  speed 13.5 vs 14.1 cm/s) and correlate day to day (r 0.69 for u, 0.76 for v,
  n = 46 days). Both have a weak mean (2-6 cm/s) but in different directions (radar
  to the W, model to the N) [M vs Mod]. At 50 m on the Swedish side, NorKyst's
  monthly northward means span the same range as the 1967/1971 mooring (above). This
  supports using NorKyst's speeds and variability; its mean direction at a single
  point is less certain [I].

## 3. Flushing and residence time

### Measured / literature

- Skagerrak: "The advective residence time for the water down to 400-500 m in the
  Skagerrak is some 100 days" (E1) [M, from mean velocities]; "a few weeks in areas of
  strong currents and somewhat longer in coastal areas or ... below 270 m" (Grimas &
  Svansson 1985, via E3) [S].
- Surface drift cards released across the Skagerrak were "Recovered 11-21 days later"
  (Engstrom 1967, via E3) [M, Lagrangian].
- A flow imbalance of 0.5 Sv "can last for about a day at maximum" before it is
  compensated (E2) [M + dynamics].
- No measured residence time exists for a ~60 km box in the eastern Skagerrak (gap).

### Modelled (NorKyst 2023)

- **Lagrangian first-exit time** (2-D particles on a 2 km lattice, 6-hourly fields,
  released every 5 days; tables/lagrangian_exit_2023.csv): median 1.6 d at 0 m
  (inter-quartile range over release dates 1.25-2.2 d; seasonal medians 1.8 JFM,
  2.3 AMJ, 1.2 JAS, 1.7 OND) and 1.9 d at 25 m (2.5 JFM, 4.7 AMJ, 1.6 JAS, 1.7 OND).
  Even the slowest release lost >= 45-58 % of its particles within 10 days; all
  surface particles left within 30 days. Using daily-mean fields instead changes
  these by < 5 % (tables/lagrangian_exit_2023_dailymean.csv).
- **Eulerian exchange time** A / (inflow integrated over the edges, per m depth):
  median 1.6 d at 0 m (10-90 % of snapshots 1.0-2.7 d), 2.3 d at 25 m (1.3-5.7 d).
  In 2022 (partial), 1.2-1.8 d at 0 m by month.
- Simple crossing time L / |box-mean U| = 60 km / 15.6 cm/s = 4.5 d at 0 m.

Caveats [I]: the particles stay at a fixed depth (no vertical migration or mixing)
and cannot re-enter, so water recirculating near the Swedish coast (E5) counts as
exported. The true renewal time of box water is therefore somewhat longer than the
first-exit time, but of order days, not weeks. This revises the existing note's
guess of "weeks" for a 60 km box (`research_notes/Den pelagiska poolens reglering/
zooplankton_consumption.md`, section 4).

## 4. Advection versus local growth of plankton

- Calanus finmarchicus on the Skagerrak coast: the Norwegian Trench is "a gateway for
  the advective supply, as well as a habitat for local overwintering"; the spring
  cohort is "the offspring of the overwintering stock originating from the Norwegian
  Trench"; "Skagerrak is an advective system and some of the observed CV and adults
  may have been advected from the northern North Sea, or from nearby fjords";
  C. helgolandicus shows local production but "considerable generational 'smearing'
  from both local and various up-stream locations" (E7) [M, interpretation by the
  authors].
- Kiorboe et al.: copepods "have long generation times in comparison to their algal
  food, and advection, predation etc. may, therefore, obscure potential
  copepod-phytoplankton distributional relationships"; advection "may prevent
  accumulation of mesozooplankton with long generation times" (E8) [M,
  interpretation].
- North Sea scale: inflow from the north is ~41 % of the North Sea's annual mean
  Calanus biomass (Gao et al. 2021, model; via the existing note, E7) [Mod, S].
- Nutrients: the Jutland Coastal Water's nitrate reaches the inner Skagerrak only in
  wind-driven intrusions (E2); the persistent nutrient "ridge" over the deep central
  Skagerrak is maintained by the circulation (E2) [M].
- **Inference [I]**: a ~2-day residence is far shorter than a copepod generation
  (weeks) and comparable to a phytoplankton doubling (~1 day). The box's zooplankton
  stock should therefore be close to that of the inflowing water (Jutland water on
  the W/S edges, Kattegat water in the SE), changed by local growth and grazing only
  by what those processes achieve in 1-3 days. Phytoplankton and the nutrient pool
  are partly local (they can change within one transit) and partly imported. No
  box-specific measurement separates the two.

---

## Implications for a 6-h-tick grid model (all [I], from the numbers above)

**Courant number.** Displacement per 6-h tick = U x 21 600 s:

| speed | km per tick | C at 1 km | C at 2 km |
|---|---|---|---|
| model today, 0.1-0.3 cm/s | 0.02-0.06 | 0.02-0.06 | 0.01-0.03 |
| box mean 15.6 cm/s | 3.4 | 3.4 | 1.7 |
| Swedish-side mean 36 cm/s | 7.8 | 7.8 | 3.9 |
| strong event ~80-100 cm/s (mean + 2 std at NE/S; drifters 51-100 cm/s) | 17-22 | 17-22 | 9-11 |

At real speeds, an explicit cell-to-cell advection step needs ~4-8 sub-steps per
tick for the mean flow at 1 km, and ~20 for strong events (or a semi-Lagrangian step).

**What the present field gets wrong.**
- Speed: 50-150 x too slow (0.1-0.3 vs 10-35 cm/s mean, 10-30 cm/s std).
- Direction: the real box mean has a positive (northward) v of 9-13 cm/s, and the
  east third flows N-NNW at 20-35 cm/s. A field that only moves east and south cannot
  represent the main through-flow.
- Season: the domain-mean speed varied x2.7 over 2023 (10-28 cm/s), not +-13 %.
- Boundary closure: re-injecting the outflow evenly round the edge returns the box's
  own plankton to it. In reality water enters over W and S (and the SE corner),
  carrying outside water, and leaves over N; the box water is renewed in ~1-3 days.

**Advection inside the box, or exchange with the outside?** With a 1-3 day
residence, a closed grid with internal advection is the wrong frame: the box is a
window on a through-flow. Two consistent options:

1. **Exchange (relaxation) form**, the same structure as the model's open pools:
   per cell or per box, X <- X_out + (X - X_out) exp(-k dt), with k = 1 / T_res.
   The Lagrangian medians give T_res ~1.6 d at 0 m and ~1.9 d at 25 m, i.e.
   k ~ 0.5-0.6 /d (exp(-k dt) ~ 0.86-0.88 per 6-h tick); in spring at 25 m
   T_res ~4.7 d (k ~ 0.2 /d). Recirculation makes renewal slower than first exit, so
   T_res ~2-5 days (k 0.2-0.5 /d) is the defensible range. This needs X_out(t), the
   plankton and nutrient state of the inflowing water by season. That is a new
   input (e.g. a station climatology of the upstream water), and it is the dominant
   term: with k ~0.3-0.5 /d the box tracks X_out within days.
2. **Open-boundary advection** at real speeds: a monthly mean field
   (tables/box_monthly_2023.csv for the box mean, box_mean_map_10km_2023.csv for the
   spatial pattern) plus a stochastic part. Inflow cells on the W and S edges (and
   the SE corner) take X_out; outflow over N leaves the box; sub-step as above.

**Parameters a mean-plus-variability field would need** (NorKyst 2023, daily
statistics):
- the mean vector per month and depth (box-mean table; for the spatial pattern, the
  10 km map);
- component std of 10-30 cm/s at 0 m (southern band E-W ~31, Swedish side N-S ~24,
  elsewhere 13-20) and 7-27 cm/s at 25 m, with the principal-axis bearings above;
- time correlation as an AR(1) with e-folding ~2 days (per-tick coefficient
  exp(-0.25/2) ~ 0.88; range 0.81-0.93 for 1.2-3.7 d);
- a local spatial correlation length of ~20-30 km (anisotropic, longer E-W in the
  south), plus the coherent Swedish-side mode as part of the forced/mean component;
- optionally, a wind-coupled part in the south (2-3 % of the wind).

A **zero-mean** field with these std, time and length scales would give realistic
patchiness and dispersion (single-particle diffusivity K ~ sigma^2 tau ~
(0.2 m/s)^2 x 2 d ~ 7000 m2/s). But it would miss the 10-15 cm/s mean through-flow
that sets the 1-3 day renewal; without that flow, a closed box would keep its
plankton for weeks instead of days.

**Sampling caveat.** Instantaneous 6-hourly model fields contain the 14.1 h inertial
oscillation (hourly autocorrelation 0.70-0.93 at 12-14 h lag, low at 6 h lag in
places). Drive a 6-h tick with daily-mean statistics (as here), not raw snapshots.

---

## Gaps and honesty notes

- **No continuous measured current record inside the box is openly available.** The
  measured in-box numbers are seven monthly means from one 50 m meter (1965-71, E4b)
  and repeated 1970s sections next to the box (E1). The Swedish HF radars (2014-15)
  that would have covered the box's east side could not be obtained (E12); the
  Norwegian HF radar ends NW of the box (E11).
- The key Swedish process papers on the surface layer and fluxes (Gustafsson &
  Stigebrandt 1996, Rydberg et al. 1996, Gustafsson 1999) and the eddy-model paper
  (Roed & Fossum 2004) were paywalled with no abstract available; none of their
  numbers are used (E13).
- All box statistics (means, std, decorrelation, residence) come from ONE model
  (NorKyst-800) and ONE full year (2023); 2022 is partial (Jan-May, Nov 1-17, Dec)
  and is used only for the month-matched comparison. NorKyst's southern open
  boundary crosses the northern Kattegat 3.5 km from the box's SE corner and ~12 km
  from the SE point; the SE part of the box and the eastern ~15 km of its south edge
  lie within ~20 km (~25 grid cells) of it, so the inflow there is boundary-forced.
  The model agrees with the HF radar on variability (daily r 0.69-0.76), but its mean
  direction at that cell differs; treat mean directions at single points as
  uncertain.
- No measured wind-current regression exists for the box; the wind numbers are
  NorKyst against its own forcing.
- The anticyclonic surface circulation in the eastern Skagerrak (E2) rests on
  unpublished modelling; this work neither confirms nor refutes it.

## Model-side check (main agent, frozen fix96 policies)

`diagnostics/scripts/current_onoff.py`, tables in `diagnostics/tables/`: the viewer's
probe world (30 x 30, seed 20260530, unfished), two years with the run's current
(east/south only, strength 0.1) and with currents off. The current shifts the
summer zooplankton centroid ~2-3 cells to the SE (SE-quadrant share 0.38-0.43 vs
0.23-0.30 off) and moves the other plankton by about a cell or less. Mackerel is the
same in both arms: it sits in the NW, at its W/NW entry edges (SE share 0.00-0.09),
and its spatial correlation with zooplankton is -0.25 to 0.27. The SE mackerel
corner seen in the training viewer came from a less-trained policy.

## Zooplankton's learned eat/rest choice (main agent, frozen fix96 policies)

`diagnostics/scripts/zoo_eat_only.py`, tables `diagnostics/tables/zoo_eat_only_*`: the probe
world (30 x 30, seed 20260530, unfished, the run's currents), two years. Arm "learned" =
the trained zooplankton policy; arm "eat" = the same policy with its rest (hide) mass moved
to its eat actions in proportion to its eat shares (a diagnostic override; every other DM
as trained). Densities are the grid mean per cell (t/km2).

When the learned policy rests (biomass-weighted rest share, year 2), split by tick of day
(dark = the two 6-h ticks around midnight, bright = the two around noon;
`Krillens flykt och energibudget/diagnostics/scripts/hide_pattern.py`):

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bright | 0.07 | 0.02 | 0.24 | 0.55 | 0.71 | 0.76 | 0.80 | 0.81 | 0.77 | 0.61 | 0.45 | 0.29 |
| dark | 0.06 | 0.01 | 0.15 | 0.46 | 0.70 | 0.77 | 0.80 | 0.79 | 0.69 | 0.49 | 0.36 | 0.24 |

The rest is mainly seasonal (high May-Sep, low in Jan-Feb), with a weak diel part
(0.05-0.12 more by day in Mar-Apr and Sep-Nov, none in May-Aug). CORRECTION: a first
version split by a light threshold (night < 0.01) and read "hides by day only, feeds at
night" for Mar-Sep; at a 6-h tick no tick in Mar-Sep has a mean light below 0.01 (the
darkest tick's mean light is 0.1-0.74 then), so those night cells were empty, not 0.
The 6-h tick also means a summer night barely darkens the light observation and the
pairs' light multipliers (darkest tick 0.57-0.74 in May-Jul).

Zooplankton (t/km2), year 2, against A17's resident stock:

| | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|---|---|
| learned | 9.6 | 20.1 | 21.4 | 17.4 | 13.3 | 8.7 | 4.7 | 2.0 | 0.8 |
| eat | 9.2 | 15.1 | 11.0 | 7.2 | 4.9 | 3.1 | 1.5 | 1.0 | 0.6 |
| A17 | 1.6 | 19.3 | 26.0 | 11.4 | 6.9 | 4.6 | 3.7 | 2.9 | 3.5 |

Other effects of forcing it to eat (year 2, learned -> eat): subsurface phytoplankton in
summer roughly halves (23 -> 11-15 t/km2 Apr-Jul), zooplankton grazing on it falls
(383 -> 324 t/km2/yr), planktivores eat less zooplankton (13.2 -> 11.0 t/km2/yr) and more
krill, mean planktivore biomass 0.85 -> 0.79 of start, mackerel's zooplankton intake
1.8 -> 1.2; whiting, the water column and benthivores are unchanged. The fish ate LESS
zooplankton when it could not hide, so its summer decline in the eat arm is mainly its
own energetics and grazing (eating where food does not pay), not fish predation [I].
One seed, one probe world, frozen policies; the eat arm is not an NDM grazer
(it keeps the DM's cost structure and hunger gate).

## Files

- `evidence.md` - sources and quotes (E1-E13).
- `scripts/`:
  - `fetch_norkyst.py` (points + box field), `fetch_wind.py`, `fetch_point.py`
  - `dods_ascii.py` (OPeNDAP ascii parser), `geo.py` (SWEREF 99 TM)
  - `analyse_norkyst.py` (all statistics, particle tracking), `hfr_vs_norkyst.py`
  - Run with `python3 -I`; no credentials or e-mail are used.
- `tables/`:
  - `points_annual|seasonal|monthly|decorrelation|spatial_corr_2023.csv`
  - `box_edges_2023.csv`, `box_monthly_2023.csv`, `box_mean_map_10km_2023.csv`
  - `lagrangian_exit_2023[_dailymean].csv`, `wind_response_2023.csv`
  - `hfr_vs_norkyst_2023.csv`
  - `box_monthly_2022.csv`, `points_monthly_2022.csv` (partial year)
- `data/` (gitignored):
  - NorKyst subsets as parsed `.npz` (`norkyst_2023/`, `norkyst_2022/`,
    `wind_2023.npz`, `point_HFRcell_2023.npz`)
  - grid lat/lon/depth subsets
  - HF radar subsets (`hfr/`)
  - fetch logs
- `diagnostics/` - the main agent's on/off current check (above).
