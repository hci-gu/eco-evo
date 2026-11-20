# Krill patchiness at the 1 km / 6 h grain: evidence trail (q1)

Question: what field statistics describe the patchiness of krill (*Meganyctiphanes norvegica* first, *Thysanoessa*
spp. second) inside one bed-depth class and season? The aim is to convert them to a 1 km x 1 km cell and a 6 h tick.
Bank-edge, shelf-break and bed-depth gradients are reported separately and flagged **gradient, already modelled**.

Date: 2026-11-20. This is literature only: no model runs and no model output were used.

## Conventions

**Tags**
- [F] full text read (pdftotext of the PDF in `data/`, page or table given);
- [S] (written [S, fetch summary] before the merge) full text read only through a summarising fetch of the publisher page, so the quotes are fragments; counted as secondary;
- [A] abstract only (OpenAlex or a search snippet);
- [S] secondary;
- [D] derived by me;
- [J] judgement.

**Files**
- Every sourced number is a row in `tables/q1_raw.csv`.
- Derivations: `scripts/q1_everson_table5.py` with `tables/q1_everson2007_table5.csv` (my transcription of Table 5) and
  `tables/q1_everson_table5_stdout.txt`; `scripts/q1_support_conversion.py` with `tables/q1_support_conversion_stdout.txt`.
- Abstracts: `tables/q1_abstracts_stdout.txt` (via `scripts/q1_abstracts.py`).

**Support** is the sampling unit behind each number. "Areal" means column-integrated (sA, g/m2), which vertical
migration does not change. "Volume" means density at depth (ind/m3, g/m3), which it does.

**Reused** (not redone): Christiansen et al. 2024, Onsrud & Kaartvedt 1998, Cotté & Simard 2005 geometry,
Tarling 1998/1999 and Bøhle & Moksness 1991, as summarised in the sibling notes listed in the brief.

**Sibling outputs in this folder**
- The between-haul spread of the in-box IKMT and the Norwegian Trench trawls is in `tables/haul_spread_stdout.txt`.
- That file comes from `scripts/haul_spread.py`, which I did not write.
- I cite those numbers as [D, sibling] and did not recompute them.

---

## 1. Horizontal patch size and spatial autocorrelation length

### 1a. Residual-type structure, inside one habitat class

**R1. Everson, Tarling & Bergström 2007, ICES J Mar Sci 64:39-48, doi:10.1093/icesjms/fsl010 [F]. Gullmarsfjord (Skagerrak coast).**
- Method:
  - EK60 at 38 and 120 kHz; adult *M. norvegica* separated by a dSv 6-12 dB window;
  - 8 surveys of 8 zig-zag transects on 8 and 10 Sept 2003, four by day and four by night (Table 3).
- Support:
  - one transect, the mean Sv along it converted to ind/m3 (a volume density averaged over the integrated depth range);
  - transect length is not given. Each survey of 8 transects took 60-74 min, so a transect is ~8 min of steaming,
    ~1-2 km [J].
  - Authors: "Bearing in mind the shortness of the transects, the constraint of working in a narrow fjord" (p. 47).
- Quoted [F]:
  - "At the transect level, there are significant differences that could be attributable to either the short length
    of the transects or the spatial distribution of targets around the fjord" (p. 44).
  - "we attribute the variations between transects to be due much more to the spatial distribution of the targets
    within the fjord than to the transect lengths" (p. 44).
  - ANOVA (Table 6): transects p = 0.00127 at 120 kHz with the 6-12 dB filter; surveys not significant (p 0.20-0.27).
- Derived [D] from Table 5, `q1_everson_table5_stdout.txt`:
  - CV between transects per survey:
    - day 0.51-0.84, mean 0.66 over all 8 transects;
    - night 0.20-0.48, mean 0.37.
  - Transect 1 lies at the sill end, where the day layer cannot sit: 0.04-0.09 ind/m3 by day against 0.33-0.79 at
    night. That is a bed-depth gradient. Without it (transects 2-8):
    - day CV 0.34-0.73 (mean 0.54), log-sd 0.50;
    - night CV 0.19-0.48 (mean 0.35), log-sd 0.35;
    - max/mean: day 1.7-2.4, night 1.4-1.7.
  - At 38 kHz the mean CVs agree within 0.02, but single surveys differ by up to 0.11.
  - Check: Table 11 (CV of the survey mean) x sqrt(8) reproduces these between-transect CVs for all 8 surveys at
    both frequencies, so the transcription of Table 5 is consistent with the paper.
- [J] This is the only residual statistic at roughly the cell support in the priority region. It comes from a fjord
  (sill 43-50 m, basin ~120 m, "narrow" transects), not from open Skagerrak water.

**R2. Simard & Lavoie 1999, Can J Fish Aquat Sci 56:1182-1197, doi:10.1139/f99-063 [A]. St Lawrence channel head (proxy).**
- Method: 120/38 kHz echointegration on a systematic grid of 1319 km2, visited 8 times in summers 1994-95;
  geostatistics; *T. raschii* + *M. norvegica*, all 2+.
- Quoted:
  - "Biomass was autocorrelated up to distances of 10-15 km."
  - "An anisotropic structure with radii of ~2 x 5 km, stretched along the channel axis, was discernible at a small
    scale."
- Support: the EDSU, nugget and sill are not in the abstract, nor whether the channel-head trend was removed.
- [J] The 10-15 km range probably includes the channel-head gradient, so it is gradient plus residual. The 2 x 5 km
  structure is the nearest published residual-type scale to a 1 km cell.
- Derived [D] (`q1_support_conversion_stdout.txt`), spherical model with no nugget, correlation between cells 1 km
  apart:
  - 0.31 at a 2 km range;
  - 0.70 at 5 km;
  - 0.85 at 10 km;
  - 0.90 at 15 km.
- Any nugget from sub-cell swarms lowers these values by its share. That share is unknown.

**R3. Plourde, Lehoux, McQuinn & Lesage 2017, DFO CSAS Res Doc 2016/111 [F]. Gulf of St Lawrence, Scotian Shelf-Bay of Fundy, Newfoundland (proxy).**
- Method:
  - GAMs of multifrequency acoustic krill biomass (0.5 km bins) on bathymetry, SST, chlorophyll and sea-level anomaly;
  - residuals tested with Moran's I on a 10 km grid.
- Quoted (Methods, p. 3):
  - "SA was generally significant at lag 1 (I < 0.4), and lag 2 (I < 0.3). The SA value is the result of the
    aggregation of krill and decreased when the resolution of the grid was adjusted to 20 km."
- [J] This is the clearest proxy evidence of residual structure. After habitat, including the bed, is removed, krill
  biomass stays autocorrelated between neighbouring 10 km cells (I up to 0.4). On a 20 km grid it is weaker; the quote does not say it is gone.
  It says nothing about structure below 10 km.

**R4. Swarm-scale (sub-cell) structure in *M. norvegica*.**
- Kaartvedt et al. 2005, MEPS 299:1-5, doi:10.3354/meps299001 [F]. Norwegian Sea, 70 N, May; proxy for an open
  slope:
  - "the swarms were oblate spherical in shape, with 50 m vertical and 120 m horizontal extensions" (p. 2);
  - "~380 000 m3, and numerical densities at ~200 ind. m-3, suggesting a total biomass of ~11 t per swarm" (p. 4);
  - Fig. 2 caption: "The ship moved at ~4 knots, and 5 swarms are displayed from this 30 min recording period."
  - [D] About 1.35 swarms per km of track; about one 120 m swarm per 740 m.
- Cotté & Simard 2005, MEPS 288:199, doi:10.3354/meps288199 [F]. St Lawrence, tidal bank (proxy):
  - "The few hundred metres-wide, dense patches are preserved during flushing by the ebb tide, and may persist for a
    few hours to days (Genin et al. 1994)" (p. 206). The persistence is cited, not measured.
  - Support: 50 m EDSU (about 15 pings), 2 m bins, kriged on a 100 m grid (p. 201).

**R5. Far proxy: Antarctic krill, a true swarmer.**
- Klevjer, Tarling & Fielding 2010, MEPS 409:157, doi:10.3354/meps08602 [F], Table 2 (daytime, Scotia Sea):
  - swarm length, median (IQR): 41-52 m (25-99);
  - inter-swarm distance, median: 96-194 m;
  - detection minimum: 15 m length, 2 m thickness.
- Morin, Okubo & Kawasaki, CCAMLR (full citation not verified; PDF blocked) [S, search snippet]:
  - the semivariance of biomass "does not vary significantly for most distances between points, except below 1 km";
  - the basic swarm is "smaller than 200 m";
  - the spectrum "resembles a white noise at scales of 2-20 km".
  - Weber, El-Sayed & Hampton 1986, as cited there [S]: krill spectra "much flatter with near-zero slopes", against
    -1.66 / -2.04 for phytoplankton and temperature.
- [J] For a swarming krill, most variance sits below 1 km and is spatially uncorrelated beyond it, which acts as a
  nugget at the cell scale. *M. norvegica* in the Skagerrak region is described as a layer (section 3b), so this is
  context only.

### 1b. Gradient-type structure (flagged gradient, already modelled)

- **Cotté & Simard 2005** [F p. 204]: "a patch (> 300 g m-2) elongated against the 100 m bathymetric contour",
  "<100 g m-2" on the shallow bank. Total biomass in the 25 km2 area was 2140-5875 t over 6 visits in 2 days, driven
  by tidal piling (currents >1 m/s). Gradient plus tide; the box has no comparable tide (~1 cm/s, sibling current
  note).
- **Lavoie, Simard & Saucier 2000**, CJFAS 57:1853, doi:10.1139/f00-138 [A]: "this mesoscale aggregation exhibits
  rapid fluctuations in spatial pattern and global abundance"; controlled by "the semidiurnal tidal currents, the
  topography, and the negative phototactism of krill".
- **Maps et al. 2015**, Limnol Oceanogr, doi:10.1002/lno.10145 [A]: "convergence zones where observed krill
  concentrations were systematically higher than average, and within which most of the densest patches were
  observed." Mesoscale circulation sets where dense patches recur.
- **Greene et al. 1988**, Science 241:359 [A]: demersal krill layers in the Georges Bank canyons, "two to three orders
  of magnitude greater than the highest densities observed in water-column scattering layers."
- **Tarling, Cottier & Everson 2014** (Gullmarsfjord bilayers; NORA manuscript copy, held in "Krill i Skagerrak"/data)
  [F]:
  - along the fjord axis at night the lower layer was 2-4 dB stronger than the upper near the sill, and "barely
    distinguishable above background levels (-79 dB)" at the head;
  - a sill-to-head gradient tied to the inflow of Skagerrak water.
- **Melle et al. 1993**, ICES CM 1993/L:44 [F]: on the Norwegian shelf "deeper than ca 150 m there was an
  approximately 50 m thick SSL associated with the bottom ... This layer tended to disappear in shallower regions,
  though occasional patches of high scattering also occured there." The scatterer was not identified. Support: 5 nmi.
- **Cochrane et al. 1994** [F p. 64]: ship transects show "considerable spatial variability in euphausiid scattering
  horizons on scale lengths of tens of kilometers" in the Scotian Shelf basins.
- **Bøhle & Moksness 1991**, ICES CM 1991/L:4 [F]: in April 1991 "the largest quantities of krill were located in an
  area 10-20 nm N of Skagens Rev at bottom depth 130 m and lower". Acoustic euphausiid concentrations were seen there
  in May-June 1988. A recurrent location at the slope: gradient.
- **Sourisseau, Simard & Saucier 2008**, CJFAS 65:574, doi:10.1139/F07-179 [F] (St Lawrence): "dense krill
  concentrations exhibit an important spatial and temporal variability at meso- and smaller scales (<~100 km or <
  months)". Introduction, citing Simard & Lavoie 1999. The rest of the paper is about vertical migration.
- **IMR NSEC trawls in the trench and Skagerrak** [D, sibling `haul_spread_stdout.txt`]:
  - station CV 0.46 in the west and north trench;
  - CV 2.02 in the east and Skagerrak;
  - stations tens of km apart, so the spread mixes gradient and residual.

---

## 2. Persistence and decorrelation time at a fixed place

**P1. Everson et al. 2007, Table 5 [D]. Gullmarsfjord, 7 transects per correlation, so the SE of r is ~0.4.**
- Pattern correlation of log density over transects 2-8, between surveys (120 kHz):
  - 8 Sept, surveys 1-3 (day, day, evening; lags 5.8-11.8 h): r 0.62-0.85;
  - 8 Sept, survey 4 (late night) against surveys 1-3 (lags 3.7-15.5 h): r -0.38 to +0.21;
  - 10 Sept, all 6 pairs of surveys 5-8 (lags 4.1-16.1 h): r 0.78-0.99;
  - survey 4 against surveys 5-8 (lags 32-49 h): r 0.86-0.93;
  - surveys 1-3 against 5-8 (lags 36-64 h): r -0.37 to +0.52.
- The within-day range over all 12 same-day pairs is -0.38 to +0.99.
- Reading [J]: the pattern reorganised once in three days, during the night of 8 Sept, and then held for about two
  days. That is one event, not a decorrelation time.
- Survey means ranged 0.29-0.53 ind/m3 over the 3 days. The authors found no significant difference between surveys.
- Currents: none in this paper. Liljebladh & Thomasson 2001 [A] report Gullmarsfjord velocities ">5 cm s-1" that
  "seemed to have the potential to decrease the peak in the backscatter profile".

**P2. Cochrane, Sameoto & Belliveau 1994, MEPS 107:55-66 [F]. LaHave Basin, Scotian Shelf (proxy, a *M. norvegica* basin).**
- Method: bottom-mounted upward 150 kHz ADCP, 49 days in autumn; hourly profiles, 14-185 m.
- Quoted:
  - "narrow spikes lasting from 1 h or less (i.e. unresolved) to several hours corresponding to the passage of
    spatially localized intensely scattering aggregations of unknown nature. Longer period (> 24 h) trends are also
    evident" (p. 59).
  - "Multi-day fluctuations in total water-column scattering levels ... may be due to advection of large-scale
    spatial concentration inhomogeneities through the region" (p. 64).
  - "a fairly persistent westerly current drift of the order of 20 cm s-1 characterized the seasonally mixed layer
    while much-reduced currents occurred in the lower half of the water column" (p. 64).
- Caveat: single frequency. The paper says it cannot separate fish from zooplankton scatterers; the krill
  attribution rests on the migration pattern.
- [D, read off Fig. 3, 25 h running mean of column-integrated sv, after the non-krill surface layer ends at about
  hour 500]:
  - the level wanders between about 0.7 and 3.7 x 1e-5, mostly 1-2.5;
  - excursions of a factor 2-4 last about 1-3 days.
  - [J] So day-mean krill at a basin point is not constant. A factor ~2 changes over days.
- Support: the beam footprint at 150 m is metres wide, so this is a point time series, areal.
- Note: no autocorrelation function is computed; my reading of the figure is coarse.

**P3. Christiansen et al. 2024, ICES J Mar Sci 81:1764, doi:10.1093/icesjms/fsae123 [S, fetch summary]. Inner Oslofjord (prior note, section 3).**
- Voids after large fish schools "typically lasted 5-15 min"; after trawls "> 1 h", "for hours"; at night voids
  vanish fast.
- Currents "usually <2 cm s-1".
- On whether the layer persists between days, the paper gives nothing explicit [S, fetch summary]. The raw WBAT data (PANGAEA
  973718, CC-BY, ~1.8 TB of raw EK80, about 100 days) would allow an ACF but were not processed (not obtained).
- [J] These are sub-tick transients and average out within a 6 h tick.

**P4. Brierley et al. 2006, Limnol Oceanogr Methods 4:18 [A]. South Georgia (far).**
- Mooring time series showed "step-changes that were not seen during short-term vessel surveys"; krill there are
  "generally passive drifters on currents".

**P5. Between-visit change of a whole aggregation (proxies).**
- Simard & Lavoie 1999 [A]: total biomass over 8 visits to 1319 km2 "varied from 8 ± 2 to 96 ± 8 kt"; [D] about
  6-73 g/m2. "large variations in the global estimates appear to be controlled by horizontal transport".
- Cotté & Simard 2005 [F p. 204]: "The total biomass for the 25 km2 of the estimated area varied between 5875 and
  2140 t over the 6 visits"; "decreased by a factor of 2 during the ebb"; "the peak biomass was 2.5 times lower" two
  days later.
- Sameoto & Cochrane 1996, DFO Atl Fish Res Doc 96/119 [F]: "Orders of magnitude variability occur between two
  consecutive samples (e.g. Emerald Basin between October 1995 and June 1996)". This mixes season and interannual
  change.
- In-box IKMT at fixed stations, repeated through one winter [D, sibling]: 14 of 22 station-winters had at least one
  zero haul next to non-zero hauls; median p2 = mean/max 0.44. This is a weeks-scale temporal spread at a ~2.3 km
  support.

**P6. Conversion arithmetic [D on J inputs], `q1_support_conversion_stdout.txt`.**
- A frozen field advected at U past a cell, with an exponential ACF, has rho(lag) = exp(-lag U / L).
- At U = 5-15 cm/s (box deep currents along the slope, caller):
  - L = 2 km: rho(6 h) 0.20-0.58, rho(1 d) ~0-0.12;
  - L = 5 km: rho(6 h) 0.52-0.81, rho(1 d) 0.07-0.42;
  - L = 10-15 km: rho(6 h) 0.72-0.93, rho(1 d) 0.27-0.75.
- At 2 cm/s (Oslofjord), the same L give rho(1 d) 0.42-0.89.
- [J] The along-slope flow runs along the isobaths, so it carries patches along the same bed class. The model's
  depth-class pool then stays the right mean, and the residual would decorrelate at these rates.
- This arithmetic assumes passive drift. Krill can hold position: in the Kattegat, net advection of ~3 cm/s needed
  to be offset to stay in the Alkor Deep, which "the krill would be capable of" without raising standard metabolism
  (Tarling et al. 1998, via "Krill över grunda bottnar" section 3.3). Active retention lengthens these times. A
  narrow enclosed fjord such as Gullmarsfjord is not a frozen advected field, so Everson's correlations are not used
  to fit L.

---

## 3. Occupancy and distribution shape

### 3a. Discrete swarms

- **Klevjer et al. 2010** [F] (far proxy, Antarctic, swarming):
  - "the proportion of 1 km sections containing no swarms at all ranged from 31% in the 50 to 100 km zone to 51% in
    the 0 to 50 km zone" (p. 163);
  - "only about 1 in 3 of each 1 km reset surveyed during night-time contained swarms at all, whereas > 44% of
    daytime cells contained at least 1 swarm" (p. 161);
  - "> 50% of acoustic biomass occurring in <1% of swarms" (p. 164);
  - median swarms per km 0-2, with a maximum of 18-26 (p. 163);
  - median NASC per km of 0 in the inshore zone, "meaning that over half the resets contained no krill" swarms.
  - Support: a 1 km along-track reset. That is the cell's length but not its area: a 1 km2 cell holds ~1000/w such
    strips.
- **Kaartvedt et al. 2005** [F]: ~11 t per swarm. [D] At ~1.35 swarms per km of track and a 120 m width, a 1 km2 cell
  would hold ~11 swarms, about 120 t/km2 within the swarm field. That is far above the box's ~5 t/km2 mean, so such
  swarm fields must be local [J].
- **Poisson lower bound [D]:** with 1-2 swarms per km of track and 40-120 m swarms, a 1 km2 cell holds ~8-50 swarms,
  so the between-cell CV from counts alone is >= 0.14-0.35. Skewed swarm sizes (">50 % of biomass in <1 % of swarms")
  raise it.
- **McQuinn et al. 2016**, DFO CSAS Res Doc 2016/104 [F text; D figure], NW Gulf of St Lawrence, August 2009-2013,
  daytime 0-80 m "shallow-water swarms" on a kriged surface (transects 10-20 km apart):
  - Fig. 7b, *M. norvegica*, share of surveyed area with core density >= 4 / 8 / 16 / 32 g/m2: 19-53 / 8-25 /
    1.5-16 / 0-7 %;
  - Fig. 7a, *T. raschii*: 22-60 / 15-27 / 2-16 / 0-6 %;
  - Text: "T. raschii forms denser aggregations and is distributed higher in the water column" than
    *M. norvegica*. *M. norvegica* is "distributed vertically over a greater depth range, producing lower volume
    densities (g·m-3), and thus are less concentrated for predators."
  - These are kriged (smoothed) upper-80 m fields, so they overstate cover at a 1 km support.

### 3b. The continuous daytime scattering layer

- **Onsrud & Kaartvedt 1998**, MEPS 171:209 [F] (Oslofjord, 120 m):
  - *M. norvegica* was "the major acoustic target in a diffuse scattering layer (SL)";
  - 116 vertical hauls in the layer gave 1292 *M. norvegica*; 55 casts outside it gave 6;
  - the upper boundary at noon was 70-73 m in Nov, Mar, Jun and Nov (Table 1).
- **Christiansen et al. 2024** [S, fetch summary]:
  - "loose aggregations (∼0.2–0.5 ind. m−3)" by day;
  - the Introduction says *M. norvegica* "does not usually form dense swarms or schools";
  - "The krill assembled in relatively dense patches during the ascent in the afternoon."
- **Degnbol & Munch-Petersen 1985**, ICES CM 1985/H:46 [F] (NNE of Skagen, 250 m station, August):
  - by day, 75-100 m "a diffuse layer, below this, between 90 and 150 m a denser layer, still diffuse but with some
    larger spots intermixed occasionally";
  - below 140 m "a layer of single echoes occasionally getting together in schoollike aggregations".
- **Cochrane et al. 1994** [F]: by day, *M. norvegica* "were broadly distributed at about 150 m depth"; at night
  "uniformly distributed in the upper 50 m".
- **Cotté & Simard 2005** [F p. 204]: the main layer at 100-150 m had "an average density of 4 g m-3"; thin layers
  against the slope reached 50-100 g/m3, up to 500 g/m3.
- **Everson et al. 2007 [D]:** in a fjord-wide survey of transects, every survey's mean was 0.29-0.53 ind/m3, and no
  transect except the sill transect by day fell below ~0.17 ind/m3. That is occupancy close to 1 at ~1-2 km support,
  with day CV ~0.5.

### 3c. In-patch : mean ratios

All of these are [D] on numbers held above.
- Cotté & Simard 2005 [F p. 204]: thin slope layers of 50-100 g/m3 against a main-layer mean of 4 g/m3 give 12-25
  (up to ~125 at 500 g/m3). Volume density, slope piling, so gradient-type.
- Herman et al. 1993 via the Atanassova et al. 2026 review, Table 1 [S] (Scotian Shelf): "Dense krill patches
  50-600 ind./m3" against "Scattered aggregations ~5 ind./m3 mean", a ratio of 10-120. Support not given.
- Everson et al. 2007, max/mean across transects 2-8 (~1-2 km): day 1.7-2.4, night 1.4-1.7.
- [J] Swarm- and slope-scale contrasts are 10-100x; at ~1-2 km transect support in a fjord layer the contrast is ~2x.

### 3d. Net-haul occupancy in and near the box

- Båmstedt & Karlson 1998, MEPS 172:149 [F p. 151], Skagerrak including the box, mostly night MIK hauls at 30-170 m:
  - "During cruise 3, 22 stations (54%) gave krill catches" (April 1991);
  - cruises 1 and 2: 11 stations each (18 and 20 %).
  - Occupancy at haul support, with gear, depth and season mixed.
- In-box night IKMT at ~2.3 km tows [D, sibling]:
  - 14 of 22 station-winters had at least one zero haul;
  - two-state occupancy p2 = mean/max: median 0.44 (quartiles 0.30-0.50); beds >= 150 m 0.49, beds < 150 m 0.35;
  - between-station CV of the in-box means: 1.62.
- Caveat [J]: 5-65 m night hauls with a 5 mm bag and IKMT avoidance inflate zeros, so this is an upper bound on
  patchiness.

---

## 4. Day/night, season and bed depth

**Day vs night.** The fjord evidence agrees that krill are more uniform at night, but not on why.
- Everson et al. 2007 [F]:
  - "the variance from the night observations was significantly lower than that observed by day" (p. 47);
  - Table 10 as printed: 120 kHz F 2.44, p 0.016; 38 kHz F 1.17, p 0.661. The 38 kHz row is identical to the
    38:120 day row, possibly a typesetting duplicate. The text claims lower night variance at both frequencies,
    and Table 11 at 38 kHz agrees with the text;
  - Table 11, CV of the survey mean: day 17.6-29.9 %, night 7.0-17.1 %;
  - the authors' reading: krill "by day tend to aggregate into swarms but after dark likely disperse".
  - [D] Transect CV: day ~0.54, night ~0.35.
- Onsrud & Kaartvedt 1998 [F, abstract]: "the krill were scattered throughout the water column at night".
- Christiansen et al. 2024 [S, fetch summary]: avoidance voids by day, not at night; dense patches at dusk.
- Tarling 1998/1999 (prior notes): about a third of the Kattegat basin stock stays down each night; a 3:1 day:night
  ratio in the basin.
- Klevjer et al. 2010 (far): swarms are present in fewer 1 km resets at night (~33 %) than by day (>44 %).
- [J] The day layer plus predator voids is the patchier state. At night, migration spreads krill over the column and
  the areal patchiness falls.

**Season.**
- No source measures how patchiness itself changes with season for *M. norvegica* in the region.
- Plourde et al. 2017 [F] (proxy): recurrent dense-aggregation areas are "smaller in the summer with a more
  fragmented distribution". For *M. norvegica*, SAK made up 38 % (spring) and 8 % (summer) of cells with any
  predicted dense aggregation.
- Nicol 1984 (prior note, Bay of Fundy): daytime surface swarms in summer, "related to reproductive activity".
  Spawning-season swarming is a possible summer source of extra patchiness [J].

**Bed depth (gradient, already modelled).**
- Everson transect 1 (sill): no day layer.
- Tarling 2014: lower layer only near the sill-inflow end.
- Melle 1993: bottom SSL on beds over ~150 m.
- St Lawrence, Georges Bank: piling on slopes and in canyons.
- In-box IKMT: p2 0.49 on beds >= 150 m against 0.35 on beds < 150 m [D, sibling], so patchier over shallower beds.

---

## 5. Evidence against km-scale patchiness

- **A layer, not swarms, in the region.**
  - Oslofjord: "diffuse scattering layer", net catches confined to it, the same 70-73 m top in four seasons [F].
  - Skagen 250 m: "diffuse ... still diffuse but with some larger spots" [F].
  - Christiansen 2024: "does not usually form dense swarms or schools", 0.2-0.5 ind/m3 [S, fetch summary].
  - Scotian basins: "broadly distributed" by day, "uniformly" at night [F].
- **Fjord-wide occupancy.** Every Gullmarsfjord transect away from the sill held krill. Survey-to-survey means did
  not differ significantly. The night CV between transects was ~0.35 [F/D].
- **Short transients.** Oslofjord voids last minutes to hours and shrink at night [S, fetch summary]. They average out inside a
  6 h tick at a 1 km cell [J].
- **Structure below the cell.** In swarming krill the variance sits below 1 km and looks white from 2 to 20 km
  [S, far proxy]. If that held for *M. norvegica*, km cells would differ by a weakly correlated noise with a Poisson lower bound of CV >= 0.14-0.35 (section 3a, Antarctic
  swarm counts; not an *M. norvegica* estimate),
  not by persistent patches [J/D].
- **Against that:**
  - Plourde's residual Moran's I (<0.4 at 10 km) and Simard & Lavoie's 2 x 5 km structure show residual km-scale
    structure in a krill-rich proxy;
  - Cochrane's factor 2-4 multi-day swings show a day-mean at a point is not stationary;
  - the in-box IKMT zeros, though gear-inflated, show haul-scale absences.

---

## 6. Not obtained

- **McQuinn et al. 2015**, Prog Oceanogr 131:159, doi:10.1016/j.pocean.2014.12.014: closed; no abstract via OpenAlex
  or Semantic Scholar. Its aggregation indices are not read. The DFO 2016/104 and 2016/111 reports were used instead.
- **Simard & Lavoie 1999 full text** (variogram model, nugget, EDSU, detrending): cdnsciencepub 403; abstract only.
- **Lavoie et al. 2000** and **Simard, Marcotte & Naraghi 2003** (ALR 16:137, conditional simulation of the same
  data): abstract only; ALR and Cambridge blocked.
- **Liljebladh & Thomasson 2001**, J Mar Syst 27:301: closed, ScienceDirect 403. **Thomasson 2003 PhD thesis**
  (GUPEA 2077/15871): bot-blocked.
- **Tarling et al. 2001**, DSR I 48:1671 (Ligurian ADCP swarms): closed; no abstract via OpenAlex. Only search-level
  text: daytime aggregations "broadly distributed at about 150 m".
- **Maps et al. 2015 full text**: Wiley 403 twice; abstract only.
- **Morin, Okubo & Kawasaki** (CCAMLR) and **Levin et al.** (CCAMLR): ccamlr.org 403; search snippets only.
- **Weber et al. 1986** full text and **Mackas, Denman & Abbott 1985** (Bull Mar Sci 37:652): not obtained;
  citation only.
- **Røstad & Kaartvedt 2013** (L&O 58:1985, Oslofjord cabled echosounder) and **Klevjer & Kaartvedt 2011** (L&O
  56:765): abstracts only; neither reports layer patchiness.
- **Kaartvedt et al. 2009**, MEPS 395:109 (Masfjorden bottom echosounder): per secondary sources it is about
  mesopelagic fish (glacier lanternfish), not krill. Not used.
- **Onsrud, Kaartvedt & Breien 2005**, CJFAS 62:1822 (doi:10.1139/f05-090) is "In situ swimming speed and swimming
  behaviour of fish feeding on the krill" (Unpaywall title). The folder `data/onsrud2005` is from another agent; I
  did not use it.
- **Christiansen et al. 2024 WBAT raw data** (PANGAEA 973718, ~1.8 TB raw EK80): not processed. It is the one dataset
  in the region that could give a krill-layer ACF at a fixed point over 2-5 d.
- **Anything from the open Skagerrak or the Norwegian Trench:** no acoustic krill survey, variogram or mooring series
  was found. IMR acoustic records of krill layers from the herring and sprat surveys were not found online.
- **Gulf of Maine and Clyde Sea:** no patch statistic found in the time available. Greene 1988 is abstract only.
- **Not pursued** (far proxies, Antarctic swarm structure): Tarling et al. 2009 (DSR I 56:1994) and Lawson et al.
  2008 (DSR II).
- **Sourisseau, Simard & Saucier 2006**, MEPS 314:257 (`data/sourisseau2006_meps314`): a circulation-model study of
  krill aggregation, not field statistics. Not used.
