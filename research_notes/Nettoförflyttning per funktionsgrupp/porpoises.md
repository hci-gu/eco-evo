# Net displacement of harbour porpoises (Phocoena phocoena) at 1 h, 6 h and 24 h

Scope: straight-line displacement per 1 h, 6 h and 24 h, split into directed travel and
area-restricted search (ARS)/foraging, plus swimming speed and path tortuosity, for a 1 km
grid (60 x 60 km offshore Skagerrak, 57.8-58.3 N, 10.1-11.1 E) with a 6 h step (1-6 h
configurable). The model currently allows at most 1 cell per 6 h (0.046 m/s).

Labelling used throughout: **[LIT]** = value stated in a source; **[CODE]** = value read
from the DEPONS source code / parameter file; **[CALC]** = my own calculation from those
values (method stated). Region labels: Skagerrak, Kattegat, Belt Sea, North Sea, Wadden
Sea, plus non-European comparisons (Bay of Fundy, West Greenland, Japan).

---

## 1. Satellite/GPS telemetry movement rates (Skagerrak, Kattegat, Belt Sea, North Sea)

### Takeaway
GPS data at 15 min resolution (3 animals tagged in the Skagerrak, 3 in the Wadden Sea) give a
mean path speed of 0.65 m/s (2.3 km/h) with a mean turning angle of 43 deg. Satellite (Argos)
data from Danish waters do not report a clean km/6 h figure. The usable numbers are: a mean
increase of 22.9 km/day in distance from the tagging site over the first 2-3 days (34
animals, Kattegat/Belt Sea), and daily movement of roughly 14-58 km/day (Bay of Fundy) and
24-42 km/day on average (West Greenland, max 103 km/day). The best per-state numbers are from
Stalder et al. 2020 (57 animals, Danish/North Sea waters, 24 h SSM). Area-restricted days
average about 11 km/day net displacement (mode 3-4 km) and directed days about 57 km/day
(mode about 51 km, tail to about 145 km), both digitised from their figure. Animals are in the
area-restricted state 81% of the time and in the directed state 6% of the time.

### Cited Findings
- **[LIT] North Sea / Skagerrak + Wadden Sea, GPS (van Beest et al. 2018, Mar Biol 165:95).**
  Six porpoises (3 tagged in the Skagerrak, 3 in the Wadden Sea near Romo), tracked 7-12 days
  each. The Fastloc GPS tried to fix a position every 3 min and achieved 3.7-9.6 fixes/h. Tracks
  were regularised to 15 min (tolerance 13-17 min), and the authors also tested 30 and 60 min.
  — [van Beest et al. 2018 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924767/)
- **[LIT]** Mean horizontal speed between successive 15-min GPS locations: **0.65 m/s**
  (min 0.15, max 2.8 m/s). Mean turning angle: **43 deg** (range 0.01-179.8 deg). Speed and
  turning angle differed significantly between individuals.
  — [van Beest et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924767/)
- **[LIT]** Environmental effects, model-averaged on a log10 scale: speed fell with sea-surface
  salinity (beta = -0.070) and rose with water depth (beta = 0.051). Turning angle rose with
  salinity (beta = 5.42). The authors read slow, tortuous movement with long, deep dives as
  foraging, and fast, linear movement with shallow dives as travelling. They give no separate
  numbers for the two states.
  — [van Beest et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924767/)
- **[LIT] Kattegat / Belt Sea, Argos (Teilmann et al. 2008, NERI Technical Report 657).**
  Across 34 porpoises from the Kattegat and Belt Seas, "from the day of deployment and the two
  following days there is a significant increase in distance (22,9 km/day) from the place of
  deployment to the position of the porpoise". From day 3 onward the increase is only
  0.2 km/day. Argos locations were filtered with Argos-Filter v7.03. Note that this measures net
  distance from the release site; the first days may include a post-capture flight from the
  pound-net site.
  — [Teilmann, Sveegaard, Dietz & Petersen 2008, NERI TR 657](https://www.dmu.dk/Pub/FR657.pdf)
- **[LIT] Denmark / North Sea / western Baltic, Argos (Sveegaard et al. 2011, Mar Mamm Sci).**
  64 porpoises tracked 1997-2007 were used to identify nine high-density areas. The abstract
  gives no movement rate. — [marineinfo record](https://marineinfo.org/id/publication/241475);
  [ResearchGate](https://www.researchgate.net/publication/227797014_High-density_areas_for_harbor_porpoises_Phocoena_phocoena_identified_by_satellite_tracking)
- **[LIT] Skane / Oresund-Kattegat-Baltic, Argos (Teilmann, Dietz & Sveegaard 2022, DCE TR 236).**
  111 tagged porpoises (1997-2021), mapped by area, season, month and year. No speeds appear
  in the text I accessed. — [DCE TR 236](https://dce2.au.dk/pub/TR236.pdf)
- **[LIT] North Sea / Skagerrak / Kattegat / Belt Sea / Wadden Sea, Argos state-space model
  (Stalder et al. 2020, MEPS 648:207-219; full text read).** 26,649 Argos locations from 57
  porpoises (1999-2017). Tagging sites: Skagen, Fjellerup Strand, Knebel, Begtrup Vig, Koge
  Bugt, Faeno, Kerteminde, Korsor, Mando, Romo. The first 24 h of each track were removed to
  reduce capture/tagging bias. They fitted a switching SSM (`bsam`, JAGS) with **a 24 h time
  step**, giving 7,762 daily positions: **80.9% area-restricted (AR), 6.0% directed (D),
  13.1% uncertain (U)**. The share of AR per individual was 50.5-99.5%.
  — [Stalder et al. 2020, open-access PDF](https://www.int-res.com/articles/meps_oa/m648p207.pdf)
- **[LIT] Same paper:** the directional persistence gamma was **0.994 +/- 0.0048 for directed**
  and **0.00156 +/- 0.0016 for AR** movement (mean +/- SD). In other words, AR days have no
  day-to-day persistence in direction, and directed days are almost straight. Their Fig. 2b/c
  shows AR turning angles spread uniformly over +/-180 deg and directed angles close to 0.
  AR was more likely at low salinity, high chlorophyll anomaly, low current velocity and steep
  slopes. Males used AR on 95% of daily positions and females on 87%.
  — [Stalder et al. 2020](https://www.int-res.com/articles/meps_oa/m648p207.pdf)
- **[LIT, digitised from figure] Same paper, Fig. 2a, "distance travelled per day" by state.**
  This is net displacement between consecutive 24 h SSM positions. I located the dashed mean
  lines against the x-axis ticks in the 300 dpi PDF render, so read these as +/- 1-2 km:
  - AR: mean about **11 km/day**, mode about 3-4 km/day, almost all below about 30 km/day.
  - Uncertain: mean about **32 km/day**, mode about 28 km/day.
  - Directed: mean about **57 km/day**, mode about 50-52 km/day, tail to about 145 km/day
    (most below 110 km/day).
  — [Stalder et al. 2020, Fig. 2](https://www.int-res.com/articles/meps_oa/m648p207.pdf)
- **[LIT] Same author group, thesis (Stalder 2018, MSc).** The share of time in the transient
  state ranged 0.5-20.0% per animal, and in the resident state 50.5-99.5% (search-result
  abstract). — [MARMAM thread on Stalder et al.](https://devlists.uvic.ca/archives/list/marmam@devlists.uvic.ca/thread/BWVOIUFOUR5TULSSRM5HVDHLZPSS65WH/)
- **[LIT] DEPONS calibration data, Skagen, Argos.** The DEPONS TRACE shows tracks of 25
  porpoises tagged at Skagen (northern Denmark) with "a position every 1-3 days", at most
  150 days per track. The TRACE also states that satellite-tagged porpoises in the inner Danish
  waters "often return to the same area after approximately two days".
  — [Nabe-Nielsen et al. 2018 TRACE, in JASSS 23(2)7 supplement S6](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- **[LIT, second-hand] Bay of Fundy / Gulf of Maine, Argos (Read & Westgate 1997, Mar Biol
  130:315).** Nine porpoises. A later NAMMCO paper cites the daily movement range as
  "13.9 - 58.5 km" (SD = 13.9). — [Nielsen et al., NAMMCO SC20 paper](https://nammco.no/wp-content/uploads/2019/02/sc20-hp-08-nielsen-et-al-offshore-migrations-of-harbour-porpoises.pdf).
  The COSEWIC report says that tagged animals "travelled more than 50 km in a single day".
  — [COSEWIC status report](https://wildlife-species.canada.ca/species-risk-registry/virtual_sara/files/cosewic/sr_harbour_porpoise_e2.pdf)
- **[LIT] West Greenland, Argos (Nielsen et al.; MEPS 597:259-272, 2018).** Two porpoises had
  mean daily travel rates of **24 km** (range 0.6-103 km, SD 21.9) and **42 km**. These animals
  made long oceanic movements, unlike the shelf-bound North Sea animals.
  — [NAMMCO SC20 paper](https://nammco.no/wp-content/uploads/2019/02/sc20-hp-08-nielsen-et-al-offshore-migrations-of-harbour-porpoises.pdf);
  [MEPS 597 abstract](https://int-res.com/abstracts/meps/v597/p259-272)

### Inferences
- [CALC] The GPS mean of 0.65 m/s is a *path* speed at 15-min resolution, not a net
  displacement. Held in a straight line it would be 2.3 km/h, 14 km/6 h and 56 km/day. The
  43 deg mean turn means that most of that path is not net displacement at 6-24 h scales.
- [CALC] The Kattegat/Belt Sea figure of 22.9 km/day over the first days corresponds to about
  5.7 km per 6 h, if spread evenly. This is the only 24 h net-displacement number from Danish
  waters, and it is probably inflated by the post-release departure.
- [CALC] The Argos daily rates (Bay of Fundy 14-58 km/day, Greenland mean 24-42 km/day) are
  roughly 3.5-15 km per 6 h if spread evenly over the day. Daily Argos positions undersample
  the path, so these are net, not path, distances.
- [CALC] Stalder's per-state daily means, spread evenly over the day:
  - Directed: 57 km/day = **about 14 km per 6 h** (0.66 m/s net, close to van Beest's 0.65 m/s
    GPS path speed, which is consistent with almost straight travel). That is about 2.4 km per 1 h.
  - AR: 11 km/day = 2.75 km per 6 h if spread evenly, but AR displacement scales sub-linearly
    with time (random direction each day, gamma about 0), so the 6 h AR displacement is more
    than a quarter of the daily value.
- [CALC] The memory-free DEPONS CRW bound (section 2) gives 14-18 km per 24 h for AR. The
  empirical AR mean of about 11 km/day sits below it, as expected once site fidelity/memory is
  included.

### Gaps
- No paper found that tabulates net displacement per 1 h / 6 h / 24 h by behavioural state for
  the Skagerrak, Kattegat or North Sea. Van Beest et al. 2018 report only pooled speed and turning
  angle. Stalder et al. 2020 give per-state *daily* displacement only in a figure (digitised
  above; no table). No sub-daily per-state displacement exists in the sources reached.
- **Unverified:** a search snippet attributes "5.3 km/h (1.47 m/s)" average speed to Sveegaard
  et al. 2011. It is not in the abstract and is implausible as a continuous mean, so it is not
  used.
- **Unverified:** a snippet reports an average swim speed of about 4.0 km/h for three
  satellite-tracked porpoises in the Kattegat/Great Belt (probably Linnenschmidt et al. 2013,
  "Biosonar, dive and foraging activity of satellite tracked harbor porpoises", Mar Mamm Sci).
  The full text was blocked (HTTP 403).
- Johnston et al. 2005 (Bay of Fundy fine-scale movements) was not retrieved; no numbers.
- Edren et al. 2010 and Teilmann et al. 2013 were not retrieved. I cannot give their km/day.
- The original Read & Westgate 1997 text was not opened; the 13.9-58.5 km/day range is
  second-hand.

---

## 2. DEPONS individual-based model: movement parameters

### Takeaway
DEPONS uses a 30-min time step. Each step a porpoise takes **either** a fine-scale move (a
correlated random walk, CRW, steered by spatial memory) **or** a dispersal move, which is a
fixed-length, almost straight step toward a remembered profitable 2 x 2 km area. There are
**two distinct parameterisations**, and they should not be blended. The 2018 TRACE (model
v1.1; parameters from Nabe-Nielsen et al. 2013/2014, calibrated on a single dead-reckoned
animal) differs from the GitHub v2.0-v3.2 defaults (2019-2025), which add depth and salinity
covariates. In the current defaults the fine-scale CRW gives about 1.0-1.1 km per 30 min
(0.55-0.65 m/s) under Skagerrak-like conditions, and the dispersal step is 1.05-2.0 km per
30 min depending on version.

### Cited Findings
**Model structure (TRACE 2018, Conservation Letters e12563 supplement)**
- [LIT] "In each time step each porpoise agent takes either a dispersal step or a fine-scale
  movement step". The time steps are half-hourly; animals move "in an order that is randomized
  after each half-hour time step".
  — [Nabe-Nielsen et al. 2018 TRACE](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- [LIT] A fine-scale move is the sum of a CRW vector VS = x(k + E), a spatial-memory vector
  VM = sum M[c] i[c], and a noise-deterrence vector VD. The CRW has "a positive correlation
  between the lengths of consecutive steps and a negative correlation between consecutive
  turning angles". — [TRACE](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- [LIT] The fine-scale model was parameterised on log10 step length per 30 min, "as step
  lengths were approximately log-normally distributed".
  — [TRACE](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- [LIT] Calibration caveat: "the CRW behavior was calibrated based on data from a single animal
  equipped with a dead reckoning tag in the inner Danish waters ... unlikely to be
  representative for all animals in all parts of the North Sea". The memory decay rates rS and
  rR were calibrated by pattern-oriented modelling to match satellite-tracked animals in the inner
  Danish waters "with the same home range sizes, mean residence times and displacement
  distances". — [TRACE](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- [LIT] Dispersal: when an animal's energy has fallen for t_todisp = 3 days, it switches from
  fine-scale moves to dispersal. It heads toward the most profitable 2 x 2 km cell group at its
  preferred distance (PSM_dist +/- PSM_tol). "All dispersal steps have the same length ddisp."
  The turning angle grows logistically from near 0 to at most PSM_angle as the animal
  approaches the target (delta = PSM_angle * w / (1 + e^(-z/PSM_log)), with
  z = 3 * d_cum/d_target - 1.5). Dispersal stops at 0.95 * d_target, or when energy improves,
  at land, or in noise. — [TRACE](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- [LIT] Dispersal calibration: ddisp, PSM_angle and PSM_dist were fitted by pattern-oriented
  modelling to home range size, home range length and cumulative distance at day 150 for 25
  Skagen-tagged Argos animals (NE North Sea / Skagerrak).
  — [TRACE, Appendix A](https://www.jasss.org/23/2/7/S6-ODD.pdf)

**Parameterisation A: TRACE 2018 Table 2 (= Nabe-Nielsen et al. 2013b/2014 values)**
- [LIT] a = 0.94 (autocorrelation of log10(d/100)); b = 0.26 (turning-angle autocorrelation);
  m = 0.74; R1 = N(0.42, 0.48) (log10(d/100), d in m per step); R2 = N(0, 38) deg;
  R3 = N(96, 28); dmax_mov = 1.18 (maximum log10(d/100) for fine-scale moves);
  **ddisp = 1.25 km per time step**; PSM_angle = 20 deg; PSM_dist = N(210, 50) km;
  PSM_tol = 5 km; rS = 0.2; rR = 0.1; k = 0.001.
  — [TRACE Table 2](https://www.jasss.org/23/2/7/S6-ODD.pdf)
- [CALC] dmax_mov = 1.18 caps a fine-scale step at 100 * 10^1.18 = 1.51 km per 30 min
  (0.84 m/s). ddisp = 1.25 km per 30 min is 0.69 m/s, or 15 km per 6 h if straight.

**Parameterisation B: DEPONS code defaults, GitHub v2.0 (2019) to master (Nov 2025)**
- [CODE] Step length: presLogMov = a0 * prevLogMov + a1 * depth + a2 * salinity + R1.
  The step is redrawn while it exceeds dmax_mov, and d = 100 m * 10^presLogMov.
  — [Porpoise.java (master, a4c81e9)](https://github.com/jacobnabe/DEPONS/blob/a4c81e9c4751411b56685d2d955c1971d2e73c12/src/dk/au/bios/porpoise/Porpoise.java)
- [CODE] The forward distance is presMov only: "movement length isn't affected by presence of
  food". The memory, food-expectation and deterrence vectors change the *heading*, not the step
  length. — [Porpoise.java](https://github.com/jacobnabe/DEPONS/blob/a4c81e9c4751411b56685d2d955c1971d2e73c12/src/dk/au/bios/porpoise/Porpoise.java)
- [CODE] Turning angle: (b0 * prevAngle + R2) * (b1 * depth + b2 * salinity + b3), redrawn
  while |angle| > 180. — [Porpoise.java](https://github.com/jacobnabe/DEPONS/blob/a4c81e9c4751411b56685d2d955c1971d2e73c12/src/dk/au/bios/porpoise/Porpoise.java)
- [CODE] Defaults that have been unchanged from v2.0 (Jan 2019) to master (Nov 2025):
  a0 = 0.35, a1 = 0.0005 (depth, m), a2 = -0.02 (salinity), **R1 = N(1.25, 0.15)**,
  **dmax_mov = 1.73** (cap 100 * 10^1.73 = 5.4 km per 30 min, about 3 m/s), b0 = -0.024,
  b1 = -0.008, b2 = 0.93, b3 = -14, R2 = N(0, 4).
  — [parameters.xml v2.0 (941a5b2)](https://github.com/jacobnabe/DEPONS/blob/941a5b29818136f26e141c36b5bb117989c5fcf5/DEPONS.rs/parameters.xml);
  [parameters.xml master (a4c81e9)](https://github.com/jacobnabe/DEPONS/blob/a4c81e9c4751411b56685d2d955c1971d2e73c12/DEPONS.rs/parameters.xml)
- [CODE] The dispersal step **ddisp** changed between versions:
  - **1.05 km/30 min** in v2.0-v3.1 (2019-2023), with PSM_angle 20 and PSM_dist N(300, 100)
    — [v3.1 parameters.xml (e89e8d9)](https://github.com/jacobnabe/DEPONS/blob/e89e8d922acde58ca6c2158436364c16c2a5b457/DEPONS.rs/parameters.xml)
  - **1.5** in v3.2, May 2025 ("update default parameters to match kattegat")
    — [v3.2 parameters.xml (6451233)](https://github.com/jacobnabe/DEPONS/blob/64512339fe7e0dab5c7f4b1169721ab598cf1eba/DEPONS.rs/parameters.xml)
  - **2.0** on master, Nov 2025 ("update default values based on calibration"), with
    PSM_angle 40 and PSM_dist N(350, 100)
    — [master parameters.xml (a4c81e9)](https://github.com/jacobnabe/DEPONS/blob/a4c81e9c4751411b56685d2d955c1971d2e73c12/DEPONS.rs/parameters.xml)
  - rS = rR = 0.03 on master. The Kattegat landscape uses a fixed salinity of 34.07.
- [CODE] The current covariate signs (step length down with salinity, turning angle up with
  salinity, step length up with depth) are *sign-consistent* with van Beest et al. 2018
  (speed beta = -0.070 for salinity and +0.051 for depth; turn beta = +5.42 for salinity). I
  have not confirmed that the v2.0 coefficients were fitted to that dataset.
  — [van Beest et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924767/)

### Inferences
- [CALC] I simulated parameterisation B as a pure CRW, with no memory, food or deterrence
  vectors (20,000 tracks, 12 h burn-in, Python re-implementation of the two loops in
  `Porpoise.stdMove`). Depth and salinity are held constant per scenario. The resulting
  30-min step lengths and mean |turn|:

  | Scenario (depth, surface salinity) | 30-min step median / mean | mean speed (p95) | mean abs turn |
  |---|---|---|---|
  | 30 m, 25 psu | 1.50 / 1.61 km | 0.89 m/s (1.53) | 29 deg |
  | 50 m, 30 psu | 1.09 / 1.17 km | 0.65 m/s (1.11) | 45 deg |
  | 100 m, 33 psu | 0.97 / 1.03 km | 0.57 m/s (0.98) | 53 deg |
  | 100 m, 34.07 psu (DEPONS Kattegat) | 0.90 / 0.96 km | 0.53 m/s (0.91) | 56 deg |

  The 50 m / 30 psu case reproduces van Beest's pooled mean speed (0.65 m/s) and mean turning
  angle (43 deg) closely.

- [CALC] Net (straight-line) displacement from the same pure-CRW simulation, in km:

  | Scenario | 1 h median / mean / p90 / p95 | 6 h median / mean / p90 / p95 | 24 h median / mean / p90 / p95 |
  |---|---|---|---|
  | 30 m, 25 psu | 2.9 / 3.1 / 4.4 / 4.9 | 14.9 / 14.6 / 19.3 / 20.6 | 35.8 / 35.8 / 56.4 / 60.6 |
  | 50 m, 30 psu | 2.0 / 2.1 / 3.1 / 3.5 | 8.7 / 8.4 / 12.2 / 13.2 | 17.8 / 18.4 / 30.6 / 33.9 |
  | 100 m, 33 psu | 1.7 / 1.8 / 2.6 / 3.0 | 6.8 / 6.7 / 10.0 / 10.9 | 13.6 / 14.2 / 23.9 / 27.0 |
  | 100 m, 34.07 psu | 1.6 / 1.6 / 2.4 / 2.7 | 6.0 / 5.9 / 9.1 / 9.9 | 12.1 / 12.6 / 21.2 / 23.7 |

  Path length per 6 h is 11-19 km, so 6 h net/path is about 0.5-0.75, and 24 h net/path is
  about 0.25-0.45.

- [CALC] These are **upper bounds for ARS/foraging displacement**. In DEPONS the memory vector
  VM pulls the heading back toward remembered patches without shortening the step. Real
  resident animals therefore have smaller net displacement over 6-24 h than a pure CRW;
  compare the TRACE remark that animals return to the same area after about 2 days.
- [ASSUMPTION] Salinity and depth at the site: the model area at 57.8-58.3 N, 10.1-11.1 E is
  likely 28-33 psu at the surface and about 30-200 m deep, so the 50 m/30 psu and
  100 m/33 psu rows are the relevant ones. I found no source for the site values in this task.
- [CALC] Directed travel (dispersal) in DEPONS: the steps are fixed-length and nearly straight
  early in a dispersal event (the turn is close to 0 until d_cum/d_target is past about 0.5), so
  net displacement is roughly equal to path length:

  | ddisp per 30 min | per 1 h | per 6 h | per 24 h | speed |
  |---|---|---|---|---|
  | 1.05 km (v2.0-v3.1) | 2.1 km | 12.6 km | 50 km | 0.58 m/s |
  | 1.25 km (TRACE 2018) | 2.5 km | 15 km | 60 km | 0.69 m/s |
  | 1.5 km (v3.2) | 3 km | 18 km | 72 km | 0.83 m/s |
  | 2.0 km (master 2025-11) | 4 km | 24 km | 96 km | 1.11 m/s |

  The 24 h values bracket the empirical daily Argos maxima (Bay of Fundy up to 58.5 km/day,
  Greenland up to 103 km/day).

### Gaps
- The Nabe-Nielsen et al. 2013 Oikos appendix, which holds the equations for parameterisation A,
  was not retrievable (the appendix link returned 404). The exact form of the a = 0.94 / R1
  recursion is unknown, so I did not simulate set A.
- I did not run full DEPONS with memory. A realistic ARS displacement (with VM) for 6 h and 24 h
  is not quantified here, only bounded from above.
- The calibration that produced ddisp = 2.0 (Nov 2025) is not documented in the commit message,
  and I found no accompanying paper.

---

## 3. High-resolution DTAG / accelerometer / logger data (fine-scale speed while foraging)

### Takeaway
DTAG work from Danish waters shows near-continuous foraging, day and night. I found no
numeric horizontal speed from those papers. Logger-based swim speeds (Japan, captive) of
roughly 0.8-1.3 m/s are second-hand only. The best fine-scale horizontal speed for the region
remains van Beest et al. 2018's 15-min GPS mean of 0.65 m/s (max 2.8 m/s).

### Cited Findings
- [LIT] **Danish waters, DTAG (Wisniewska et al. 2016, Curr Biol).** Five bycaught porpoises
  were tagged with suction-cup DTAGs for about 24 h each (2012-2014). They "foraged nearly
  continuously, targeting small prey with remarkably high capture success rates": up to about
  200 prey encounters in daytime and 50-550 per hour after dusk, with prey mostly 3-10 cm long
  and >90% capture success.
  — [Aarhus University / DCE news](https://dce.au.dk/en/current/news/news/artikel/harbour-porpoises-are-skilled-hunters-and-eat-almost-constantly);
  [Science News](https://www.sciencenews.org/blog/wild-things/harbor-porpoises-ocean-24-hour-buffet)
- [LIT] The continuous-feeding interpretation was questioned by Hoekendijk et al. (2018, Mar Mamm
  Sci letter), and the authors replied. — [Science News summary](https://www.sciencenews.org/blog/wild-things/harbor-porpoises-ocean-24-hour-buffet)
- [LIT] **Danish North Sea / Skagerrak, GPS + dive data (van Beest et al. 2018).** Dive rate was
  32-53 dives/h, mean dive duration 53 s, mean dive depth 15.5 m. Horizontal speed 0.65 m/s
  mean, turn 43 deg mean (15-min intervals).
  — [van Beest et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924767/)
- [LIT] **Japan, free-ranging, speed logger (Otani et al. 2000, Mar Mamm Sci 16:811).** The
  logger recorded at 1-s intervals with 0.04 m/s resolution. The bibliographic record gives no
  mean value. — [Duke MWBD record](https://seamap.env.duke.edu/mwbd/attributes/Otani2000a)
- [Brief, not verified] The minimum cost of transport at 1.3-1.5 m/s (Otani et al. 2001) is
  taken from the task brief; I did not verify it in this session.

### Inferences
- [CALC] Holding the minimum-COT speed of 1.3-1.5 m/s for 6 h gives a 28-32 km path. Observed
  GPS-derived mean speeds (0.65 m/s) and DEPONS dispersal speeds (0.6-1.1 m/s) are lower, so a
  transit jump set from the minimum-COT speed would overstate typical travel. Use it as an upper
  bound only.
- Because porpoises forage around the clock, a 6 h step will in practice contain mostly ARS
  behaviour. This agrees with Stalder et al.'s about 81% ARS share.

### Gaps
- No horizontal speed (m/s) from the Wisniewska et al. 2016/2018 DTAG datasets was found. The
  DTAG papers report foraging rates, not horizontal displacement, and full texts were not
  accessed.
- A snippet citing captive porpoise swim speeds of 0.76-0.91 m/s (Fisheries Science, Hokkaido)
  and a mean of 1.29 +/- 0.42 m/s (Akamatsu et al. 2002) could not be checked against the
  originals, so neither is used.
- No accelerometer-based travel/foraging speed split was found for the Skagerrak, Kattegat or
  Belt Sea.

---

## 4. Synthesis: transit reach k and local scale for a 1 km grid, 6 h and 1 h steps

### Takeaway
For a 6 h step, set the porpoise transit jump to **k of about 14-15 cells (range 12-18; about
25 as an upper-tail value)**. Two independent lines give this value:
- the empirical directed-state mean of about 57 km/day (Stalder et al. 2020, Danish/North Sea
  Argos SSM), which is 14 km per 6 h;
- the DEPONS dispersal step of 1.05-1.5 km/30 min, which is 12.6-18 km per 6 h.

The local (ARS) scale at 6 h is about 4-7 km net. The upper bound is 7-9 km (median) from a
memory-free DEPONS CRW. The empirical AR daily mean of about 11 km/day implies less.

The current limit of 1 cell per 6 h (0.046 m/s) is roughly 5-15x below both scales. For a 1 h
step the two scales converge at about 2 cells (transit 2-3 cells, ARS about 1.5-2 km).

### Cited Findings
- [LIT] Directed travel step in DEPONS: 1.05 (v2.0-v3.1), 1.25 (TRACE 2018), 1.5 (v3.2) and
  2.0 (master 2025-11) km per 30 min.
  — [TRACE](https://www.jasss.org/23/2/7/S6-ODD.pdf);
  [DEPONS parameters.xml history](https://github.com/jacobnabe/DEPONS/blob/a4c81e9c4751411b56685d2d955c1971d2e73c12/DEPONS.rs/parameters.xml)
- [LIT] Empirical daily movement: 13.9-58.5 km/day (Bay of Fundy, second-hand); mean 24 and
  42 km/day, maximum 103 km/day (West Greenland).
  — [NAMMCO SC20](https://nammco.no/wp-content/uploads/2019/02/sc20-hp-08-nielsen-et-al-offshore-migrations-of-harbour-porpoises.pdf)
- [LIT] 22.9 km/day increase in distance from the tagging site over days 0-2 (Kattegat/Belt Sea,
  n = 34). — [NERI TR 657](https://www.dmu.dk/Pub/FR657.pdf)
- [LIT] Time budget at 24 h SSM resolution, North Sea and Danish waters (57 animals): 80.9% AR,
  6.0% directed, 13.1% uncertain. Directional persistence gamma is 0.994 (directed) versus
  0.0016 (AR). Daily net displacement means (digitised from Fig. 2a): AR about 11 km, uncertain
  about 32 km, directed about 57 km (tail to about 145 km).
  — [Stalder et al. 2020](https://www.int-res.com/articles/meps_oa/m648p207.pdf)
- [LIT] GPS path speed 0.65 m/s mean (0.15-2.8), mean turn 43 deg, 15-min intervals,
  Skagerrak + Wadden Sea. — [van Beest et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924767/)

### Inferences
All entries are [CALC]/judgement built on the findings above.

**Directed travel (transit jump k), net displacement**

| Step | Central | Range | Upper sensitivity | Basis |
|---|---|---|---|---|
| 1 h | 2.4 km (about 2-3 cells) | 2.1-3 km | 4 km | Stalder D mean / 24; DEPONS ddisp 1.05-2.0 km/30 min |
| 6 h | **14-15 km (k about 14-15)** | 12-18 km | about 25 km (Stalder D tail about 100-110 km/day; DEPONS ddisp 2.0 = 24 km); 28-32 km at the min-COT speed | Stalder D mean 57 km/day; DEPONS ddisp; GPS 0.65 m/s held straight = 14 km |
| 24 h | about 57 km (mean of directed days) | mode about 51, bulk about 35-90 km | about 110-145 km | Stalder Fig. 2a; Greenland maximum 103 km |

The 6 h directed value assumes travel speed is constant within the day. Stalder's daily
values are net displacement between daily positions, so true 6 h travel bouts could be
somewhat faster than the daily mean / 4.

**Area-restricted search / foraging, net displacement (upper bound from memory-free CRW,
Skagerrak-like 50-100 m depth, 30-33 psu)**

| Step | Median | Mean | p90 | p95 |
|---|---|---|---|---|
| 1 h | 1.7-2.0 km | 1.8-2.1 km | 2.6-3.1 km | 3.0-3.5 km |
| 6 h | 6.8-8.7 km | 6.7-8.4 km | 10-12 km | 11-13 km |
| 24 h | 13.6-17.8 km | 14-18 km | 24-31 km | 27-34 km |

With spatial memory the realistic values are lower: DEPONS VM pulls the heading back to
remembered patches, and animals return to the same area after about 2 days. The empirical AR
24 h mean is about 11 km (Stalder, mode 3-4 km), against 14-18 km for the memory-free CRW, about
0.6-0.8x of it.

A judgement estimate for the realistic AR net displacement, scaling the CRW bound by about 0.6-0.8:
- 1 h: about 1.3-1.6 km (memory matters little at this scale)
- 6 h: about **4-7 km**
- 24 h: about **11 km mean** (empirical)

**Scaling with step length**
- Travel net displacement scales roughly linearly with time: k(dt) is about 2.5 km/h * dt, so
  2.5 cells at 1 h and 15 at 6 h.
- ARS displacement rises roughly linearly up to about 1 h (net/path about 0.9) and then roughly
  with the square root of time (6 h about 4x the 1 h value, not 6x; 24 h about 2x the 6 h
  value).
- At 1 h, travel (about 2.4 km) and ARS (about 1.5-2 km) differ by less than one cell, so a
  two-scale move adds little. At 6 h the split is clear: local about 4-7 cells versus transit
  about 12-18 cells.

**Implications for the model**
- The current cap of 1 cell per 6 h (0.046 m/s) is about 4-7x below the realistic AR scale,
  7-9x below the memory-free ARS bound (median) and about 14-15x below directed travel.
- A 1-cell local step at 6 h is under-dispersed relative to every dataset found. If the local
  step must stay at 1 cell, the transit jump carries all of the realistic spread.
- With Stalder's time budget (81% AR, 6% directed, 13% uncertain, the last with a daily mean
  of about 32 km, i.e. about 8 km per 6 h), a rough expected 6 h net displacement is
  0.81 * 5.5 + 0.13 * 8 + 0.06 * 14, or about 6.3 km. Treat this as an order-of-magnitude
  check, not a parameter.
- The 60 km domain is crossable in about 4 travel steps (about 1 day) at k = 15. This is
  consistent with porpoise home ranges and dispersal distances that span hundreds of km
  (DEPONS PSM_dist N(210-350, 50-100) km).

### Gaps
- No empirical distribution of 6 h net displacement split by HMM/SSM state exists for the
  Skagerrak in the sources reached. The per-state values above come from Stalder et al.'s daily
  (24 h) SSM values, digitised from a figure, from DEPONS (a calibrated model, not raw data),
  and from simple time conversions. The sub-daily AR estimate (4-7 km per 6 h) is a judgement
  that combines the empirical daily AR mean with the CRW time scaling.
- Seasonal differences (for example faster winter offshore movement) were not quantified.
- How much the DEPONS memory vector shortens 6 h/24 h ARS displacement could be measured by
  running DEPONS itself (code at github.com/jacobnabe/DEPONS); not done here.
