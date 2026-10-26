# Attack rate of `planktivorous_fish -> zooplankton` from encounter theory

Biology-first derivation of the Holling attack rate `a` (library `max_intake_rate`) of the
planktivore group on the zooplankton pool. It uses encounter theory and member biology only; no model
output was read, nothing was tuned, and no code, library, manifest or resume was changed.
Written 2026-10-25.

Tags: [F] full text or data read, [A] abstract only, [S] secondary or search snippet, [D] computed
here, J = judgement (stated). Evidence notes in this folder (factor-level detail and quotes):
`evidence_herring.md`, `evidence_sprat_pout_bluewhiting.md`, `evidence_vertical_overlap.md`.
Computation: `scripts/planktivores_a.py` -> `tables/planktivores_a_stdout.txt` (central, corners,
one-at-a-time, Monte Carlo, validation); size structure `scripts/size_structure.py` ->
`tables/size_structure_stdout.txt`; the forks' scripts and tables are listed in their notes.
Reused, not redone: `../Fiskgrupper i rutan/params_planktivorous_fish.md` (weights, h 40, Q/B),
`../Sillens vinterföda/*.md` (functional-response fits, refuge, winter rations, A17 pool),
`../Sillens vinterbudget/winter_budget.md` (winter herring at the site).

## 1. Bottom line

Engine units: km2 swept per tonne of FORAGING predator per 6 h tick, on the depth-integrated
visible pool. `a` is the annual time mean (the engine normalises m(t) to a time mean of 1),
at the layer's annual-mean temperature. Central = every factor at its central value;
low/high = 5th/95th percentiles of a Monte Carlo over all factors (log-triangular, independent,
1500 draws, seeded). The all-extremes corners (0.0001 / 0.29) are bounds only.

| Member (weight) | N per t | v (BL/s) | R day on Calanus / small copepod | R night | capture p | overlap O day / twilight / night (thickness 25/O, m) | a low / central / high | 1/(a h) central, t/km2 | night/day a |
|---|---|---|---|---|---|---|---|---|---|
| Herring (0.79) | 28 900 (10 400 adults-only) | 1.0 (0.6-1.4) | 14 / 6.3 cm (k_v 115, 60-218) | 0 (no biting below 1e-3 lx); filtering 6.1e3 m3/t/tick | 0.6 (0.3-0.9) | Apr-Sep 0.45 / 0.6 / 0.8 (56 / 42 / 31); Oct-Mar 0.15 / 0.5 / 0.55 (167 / 50 / 45) | 0.0017 / **0.0083** / 0.019 | 3.0 | 0.01 summer, ~0.03 winter (filter only at night) |
| Sprat (0.03) | 115 000 | 0.6 (0.3-1.0) | 12 / 5.5 cm (k_v 100, 60-155) | 0 | 0.7 (0.33-0.86) | 0.45 / 0.6 / 0.8 | 0.0046 / **0.013** / 0.035 | 1.9 | ~0 (stomach data 0.1) |
| Norway pout (0.075) | 66 500 | 0.33 (0.2-0.6) | 12 / 5.5 cm at day; x0.4 at night (J) | as left x 0.63 | 0.7 | 0.03 / 0.06 / 0.10 (830 / 420 / 250) | 0.0002 / **0.0006** / 0.0012 | 39 | 0.95 |
| Blue whiting < 29 cm (0.083) | 12 100 | 0.33 (0.2-0.6) | as pout | as pout | 0.7 | 0 / 0.35 / 0.7 (- / 71 / 36) | 0.0002 / **0.0008** / 0.002 | 31 | no day feeding |
| Blue whiting >= 29 cm (0.022) | 10 700 | 0.33 | as pout | as pout | 0.7 | 0 / 0.03 / 0.07 | 0.0000 / **0.0001** / 0.0002 | 360 | no day feeding |
| **Group** | - | - | - | - | - | - | **0.0018 / 0.0070 / 0.016** (MC median 0.0052) | **3.6** (1.6-14) | **0.06** (0.02-0.24) |

- **Group a: 0.007 km2/t/tick (5-95 %: 0.002-0.016).** With R proportional to fish length
  (the convention the only herring field clearance favours, V1) it is 0.0055. Either way it is
  about one sixth of the library's 0.04. In the Monte Carlo, P(a >= 0.04) = 0.001. Herring carry 93 % of the group a, sprat 6 %, and
  pout and blue whiting together 2 %. Those two forage almost entirely outside the 0-25 m pool.
- **Half-saturation 1/(a h) at h 40: 3.6 t/km2 (1.6-14).** The library value is 0.63. Against today's
  prey, the pool start is 9.72 t/km2, and the monthly resident target is 0.2 in Jan-Feb, 19-26 in
  Apr-May and 3-4 in Sep-Nov (resume 183.6). So the ratio B/(1/(ah)) is about 0.1 in Jan-Feb,
  5-7 in Apr-May and about 1 in Sep-Nov. Herring would feed near saturation only in spring and
  early summer, and in the linear part of the curve from October to March.
- **Seasonal swing (central): about 20x.** Monthly a is 0.0007-0.0011 in Dec-Mar, 0.015-0.016 in
  Apr-May, 0.010-0.013 in Jun-Sep and 0.0015-0.002 in Oct-Nov. The causes, in order: Calanus is not
  in the resident winter pool, so the prey are small and R^2 is about 5x lower; winter herring stay
  deep by day (O 0.15 against 0.45); the days are short. The prey-biomass-weighted annual mean is
  0.0125, against the time mean of 0.007 (section 4).
- **Night/day, as a check on dark_ratio 0.25: the group gives 0.06** (0.02 in summer, 0.07-0.10 in
  winter; MC 0.02-0.24). For herring alone it is 0.01 in summer, because darkness leaves only
  filtering. The earlier biomass-weighted 0.26 (`params_planktivorous_fish.md` 1.8) does not
  survive: it gave pout and blue whiting (ratio ~1) their biomass weight, but they hold only ~2 %
  of the group a on this pool.
- **Light threshold:** the derived a falls to half its daytime value at a sun elevation of -0.6 to
  -2.3 deg (central). It reaches -6 deg only at the high corner. The engine uses -8 deg.
- **Filter vs particulate:** biting dominates at every pool density in the box. Section 2.5 gives
  the crossover numbers.

## 2. Derivation

### 2.1 What `a` is and the unit chain
- Engine (`interactions.holling_a_eff`, `predation.py`): intake = eat x hunger x a B / (1 + a h B +
  w B_pred). B is the visible pool, in t/km2. Kept outside `a`: the foraging fraction (eat
  share), hunger, prey hiding (floor 0.5), interference (w 0.05) and the ceiling (h 40).
- The pool is the resident 0-25 m community at A17. The WP2 hauls are 0-25 m, so the pool is a
  0-25 m integral (resume 183.6; `evidence_vertical_overlap.md` 0; SHARK has no other A17
  strata). The pool's mean concentration in its own layer is therefore **B / 25 m by
  definition**.
- Encounter-weighted overlap: O = sum over depth bins of p_fish(bin) x r_prey(bin), where r is the
  prey concentration relative to B / 25 m. A foraging fish meets the concentration B / H_eff, with
  H_eff = 25 m / O.
- Unit chain [D]:
  `a_i(t) = sum_j s_j(month) x [f pi R_j(t)^2 v p] x (N/t weighting) x 21 600 s x O_i(t) / 25 m / 1e6`.
  Here j is the prey type (Calanus CIV-adult or small copepod), s_j its biomass share, f the share
  of the visual field searched, R the reactive distance, v the search speed and p the capture
  success.
- The prey-biomass weighting is exact: intake = sum_j V_j c_j, so a = sum_j s_j V_j / H_eff.
- Per tonne: with R independent of fish length (see 2.2), sum over individuals of v =
  v_BL x M1 / 100, where M1 = sum n L / sum n W in cm per t (`tables/size_structure_stdout.txt`).
- Light and time: the year is stepped in 10 min with the engine's own sun-elevation formula
  (58.15 N). Periods: day (sun > 0 deg), twilight (0 to -12 deg), night (< -12 deg). These hold
  0.50, 0.17 and 0.33 of the year. The engine's F at -8 deg averages 0.6115 [D].

### 2.2 Reactive distance R
- Form: Aksnes & Utne 1997, small-attenuation limit at saturating light. R = k_v sqrt(A_prey), with
  image area A = 0.75 L W. R^2 is scaled by I/(Ke + I), and R = 0 below the biting threshold. At
  R ~0.1-0.2 m and c 0.3 m-1, beam attenuation changes R by < 6 % [D, evidence_herring 2.1].
- **Herring:** R has never been measured for juveniles or adults.
  - The herring visual models (Varpe & Fiksen 2010, as reported by van Deurs et al. 2015 and
    Langbehn & Varpe 2017 [F]) set R = 1 body length at saturating light on a reference prey.
  - The two reports disagree on that reference prey: 4 mm (A 3e-6 m2) or 1.5 mm.
  - For a 20 cm fish this gives k_v = 115 (4 mm reference) or 218 (1.5 mm reference) [D]. The low
    value, 60, is the Langbehn et al. 2019 Benthosema anchor [F].
  - Central k_v is 115. On Calanus CV/adult (2.5 x 0.8 mm) that gives R = 14 cm, and on 1 x 0.4 mm
    small copepods 6.3 cm. For comparison, larval herring were measured at 0.6-1 BL (Munk &
    Kiørboe 1985 [F]).
- **Members:** no reactive distance exists for sprat, pout or blue whiting. k_v = 100 (60-155) is
  set from three anchors:
  - Langbehn et al. 2019 model parameters: 57-65 [F].
  - Back-calculated from Kulke 2018's lab clearance of sprat: 96-149 [D].
  - Gobiusculus on Calanus: 152 (Utne 1997 [A]).
- **Fish length:** the k_v anchors do not depend on fish size. Langbehn's 6 cm and 50 cm fish give
  the same k_v, so central R is independent of fish length. As a sensitivity [D, from the
  moments], with R proportional to L (anchored at 20 cm) the multipliers are herring x0.81,
  sprat x0.30, pout x0.45, blue whiting x1.5, and the group a becomes 0.0055 (x0.79). The
  size-matched YOY field clearance (V1) favours this variant for herring. The central case keeps
  the size-free form because the box herring are near the 20 cm anchor; the choice moves `a`
  by only about 20 %.
- **Light** [D, script header]:
  - Surface illuminance: 1.3e5 sin(h) lx by day; in twilight, AMS glossary values (450, 3, 0.008 and
    6e-4 lx at 0, -6, -12 and -18 deg); an overcast factor of 0.2 (J).
  - Attenuation: K 0.30 m-1 in 0-20 m (Aas et al. 2014, Secchi 8.3 m [F]) and 0.12 below (Aksnes
    et al. 2009 regression, J).
  - Biting thresholds: 1e-3 lx on Calanus and 1e-2 lx on small prey (Batty et al. 1990 [S]).
  - Ke: no Ke has been measured for herring. Central 1 lx (0.1-10), J between the 0.001 lx
    threshold and the ~10-500 lx saturation of other planktivores.
  - Foraging depth for the clupeids' light: 15 m (8-25), J.
  - Result at 15 m: the clupeid R^2 factor averages 0.97 by day, 0.06 in twilight and 0 at night.
    Ke and the depth each move a by less than +-13 %.
- **Pout and blue whiting** are dim-light and partly tactile foragers (Onsrud et al. 2004 [F]).
  Their R^2 factor is fixed per period, J: day 1.0 (0.5-1), twilight 0.7, night 0.4 (0.1-0.8).
  Pout stomach fullness night/day is 0.9 on krill (Onsrud 2004). These factors barely matter,
  because the overlap removes the two members (2.6).

### 2.3 Search speed v and search mode
- **Herring 1.0 BL/s (0.6-1.4):**
  - Feeding herring in an aquarium swam at 1.36 SL/s (Hamann et al. 2023 [F]).
  - Juveniles swim at 0.34 m/s in light (Batty et al. 1986 [S]).
  - Bioenergetics models use 1 BL/s as the convention (Bachiller et al. 2018 [F]).
  - Deep feeding schools swim at 1.1 BL/s (Nøttestad 2002, via the herring note [F]).
- **Sprat 0.6 BL/s (0.3-1.0):** lab spontaneous swimming is 0.57 BL/s and the feeding maximum
  0.92 BL/s (Meskendahl 2013 [F]).
- **Pout and blue whiting 0.33 BL/s (0.2-0.6):** deep gadoid-sized targets tracked in situ swim at
  about 1/3 BL/s (Kaartvedt et al. 2009 [F]; species not identified, so J).
- **Search mode:** every herring model found treats herring as cruise searchers, and no
  pause-travel description exists for juveniles or adults. Copepods swim at mm-cm/s against fish
  at about 15 cm/s, so the Gerritsen-Strickler correction is a few per cent and is left out.
- **Temperature:** the Wisconsin herring activity term roughly halves v from 9 C to 4 C (Rose 2008,
  via the q10 note [F]). It is not applied here, because `a` is given at the annual-mean
  temperature. Applied, it would deepen the winter minimum.

### 2.4 Capture success p and the searched visual field f
- **Herring p 0.6 (0.3-0.9), J:** no herring data exist. The proxies are:
  - lake herring on copepods, 0.30-0.65 (Link 1996 [A]);
  - stickleback juveniles, 0.92 on Temora and 0.53 on Eurytemora (Viitasalo et al. 1998 [A]);
  - herring on frozen, non-escaping prey, about 1 (Bernreuther 2007 [F]).
- **Members p 0.7 (0.33-0.86):** sprat in Kulke 2018's tank-depletion vs video comparison, 0.70 +-
  0.35 [F]; Langbehn et al. 2019 assume 0.7.
- **f 0.5 (0.25-1), shared by all members:** the half-visual-field convention of Varpe & Fiksen
  2010, Langbehn et al. 2019 and van Deurs et al. 2015 [F]. f = 0.5 is also the value that matches
  Kulke's field sprat clearance (section 3).

### 2.5 Filter feeding (herring)
- Filtering clearance per fish is gape x speed x retention:
  - gape = 0.0047 SL^2 (Hamann et al. 2023 [F]), which is 0.00359 TL^2 under isometric scaling;
  - retention 0.9 for copepods >= 1 mm, judged from the gill-raker gap of 0.30-0.38 mm (Hamann
    2023; Collard et al. 2017 [F]);
  - speed 0.11 m/s filtering in the dark, 0.34 m/s in light (Batty et al. 1986 [S]).
- Result [D]: 29-250 L/h per fish (TL 15-25 cm). For the box herring at dark speed that is
  6.1e3 m3/t/tick, against 1.85e6 m3/t/tick for biting at saturating light on Calanus. Biting
  clears 20-500x more water.
- Filtering pays only where biting is bite-rate limited (0.85-1 bites/s; Bernreuther 2007 [F]).
  The crossover is N* = 12-23 prey/L for 20-25 cm fish in light, and 58-70 /L for 11.5 cm fish
  [D]. That agrees in order of magnitude with the measured switches: 50 % of fish filter at
  18 Calanus/L (Gibson & Ezzi 1990 [S]), and filtering starts above ~50 Artemia nauplii/L (Gibson &
  Ezzi 1985 [S]).
- A17 0-25 m copepod abundance is 2.5-7 /L in winter and 25-31 /L in May-July, mostly small
  copepods (winter_food note [F]). Field diets show size-selective particulate feeding (Dalpadado
  et al. 2000 [F]).
- **Which mode dominates:**
  - In the initial slope (the a regime), biting dominates at every box density.
  - Filtering is a summer patch supplement once biting saturates. That regime belongs to h, not a.
  - Filtering is the only mode at night; it enters as max(biting, filtering) per time step.
  - Sprat do not filter (Kulke 2018 [F]).

### 2.6 Vertical overlap O (`evidence_vertical_overlap.md` 1-4)
- **Prey:**
  - In Skagerrak pump profiles, 0-25 m holds 0.83 (spring) and 0.79 (late summer) of the 0-100 m
    herring-prey copepod biomass (Tönnesson et al. 2006 [F]).
  - The resident copepods do not migrate between day and night (Maar et al. 2006; Tönnesson 2006
    [F]).
  - The concentration below 25 m is 0.1-0.2 of B / 25 m, and in 0-10 m it is 1.5-2x.
  - Winter: no stratified data exist; 0.4-0.7 of the biomass in 0-25 m (J).
- **Herring:**
  - HERAS July logs in the box, 2010-20: 0.42 of the fish echoes are in 0-25 m by day (mean depth
    36 m) and 0.56 at twilight. The top 7-10 m is unseen and there are no night logs [D].
  - North Sea adults lie near the bottom by day and "disperse into surface waters at night"
    (Fernandes et al. 2001 [F]).
  - In winter they are caught near the bottom by day (IBTS Q1) and in the upper 50 m at night
    (Onsrud et al. 2004 [F]).
  - Central O, summer: 0.45 / 0.6 / 0.8 (day / twilight / night). Winter: 0.15 / 0.5 / 0.55. The
    all-fish July O is 0.26-0.40 by day; the shift up to 0.45 for foraging fish is J.
- **Norway pout:** 5-10 m above the bottom by day, with "no catches in the upper 25 m" at night
  (Onsrud 2004 [F]). O is 0.03 / 0.06 / 0.10.
- **Blue whiting** (Skagerrak, Degnbol & Munch-Petersen 1985 [F]):
  - Below 150 m by day and not feeding there.
  - At night, 17-28 cm fish rise close to the surface (O 0.7). Fish of 29-40 cm stay around 100 m
    with the krill (O 0.07).
  - Biomass split 0.79 / 0.21 at 29 cm, from the GOV length frequencies [D].
  - Assumption (J): blue whiting is a Q3 visitor, and its "while present" a is applied at its
    annual weight.
- **Krill and hyperbenthos** are not in the WP2 pool. Most of what pout and blue whiting eat lies
  outside the pool (flag 5 in section 4).

### 2.7 Individuals per tonne
- Source: NS-IBTS GOV CPUE per length class, 2021-25, Skagerrak plus box hauls. Each class's
  midpoint weight comes from the CA length-weight fits, and the box mean is depth-stratified by the
  grid's area per depth bin, with Q1 and Q3 pooled [D, the fish_guilds convention].
- Results (N per t; biomass-weighted mean length):
  - herring 28 900 (20.5 cm);
  - sprat 115 000 (11.2 cm);
  - pout 66 500 (15.0 cm);
  - blue whiting 10 700 (26.3 cm).
- The GOV under-samples large pelagic herring. The adults-only bound (L >= 20 cm, 55 % of the GOV
  herring biomass) gives N/t 10 400 and lowers the group a x0.56. Juveniles dominate the box (Q1:
  1-2 ringers are 89 % of the biomass; `winter_budget.md` 4), so the central case keeps the full
  distribution.

### 2.8 Weights and the group value
- Annual-mean biomass weights: herring 0.79, blue whiting 0.105, pout 0.075, sprat 0.03 (params
  note 0). The group a(t) is the weighted mean of the members' a(t), and the library value is its
  time mean.
- Effect of each factor alone, as a multiplier on a at its low / high value:

| Factor | low | high |
|---|---|---|
| k_v | x0.28 | x3.5 |
| overlap O | x0.54 | x2.2 |
| f | x0.50 | x2.0 |
| capture p | x0.50 | x1.5 |
| v | x0.60 | x1.4 |
| size | x0.56 | x1.1 |
| Ke | x0.87 | x1.08 |
| foraging depth | x0.89 | x1.08 |

### 2.9 What 0.04 would need
0.04 is 5.7x the central value. Any one of these would get there:
- k_v at the van Deurs reference (218, x3.5) together with the full visual field (f 1, x2): 0.049;
- O of about 1 day and night all year, with the Calanus-rich summer prey kept in winter;
- three of the large factors (k_v, O, f) at their high ends at once.

None of these is the central reading of the evidence.

## 3. Validation (not used to set a)

| # | Datum | Value | Compared with the derivation | What it tests |
|---|---|---|---|---|
| V1 | Arrhenius & Hansson 1999, field, YOY Baltic herring 4-8 cm, 0-30 m layer (via `../Sillens vinterföda/herring_functional_response.md`) [F] | initial slope 46 L per g wet fish per daylight hour (2.8e5 m3/t per 6 h); as area 0.011 km2/t/tick at O 1, 0.005 at O 0.45 | **size-matched** theory (central f, p, v; small prey) [D]: with R size-free (6.3 cm) 437-2258 L/g/h, i.e. 9-49x the field value; with R proportional to fish length (20 cm anchor; R 1.3-2.5 cm) 70-90 L/g/h, i.e. 1.5-2.0x | the absolute daytime clearance, and the size scaling of R. The field value is realised intake (non-foraging time and hunger included), so it is a lower bound on a while foraging. A 1.5-2x excess is easily absorbed; 9-49x is not. **It favours R shrinking with fish length**, which gives a group a of 0.0055 instead of 0.007. Kulke's 8.7 cm sprat (k_v 96-149 size-free) points the other way for sprat, which carries only 6 % of the group a |
| V2 | Kulke 2018, field sprat, k = 291 L^-1.5 [F] | 2.6-5.2e5 m3/t/tick -> a 0.010-0.021 at O 1, 0.005-0.009 at O 0.45 | derived sprat clearance 1.6e6 (f 0.5) on 50/50 prey; derived sprat a 0.013 | per-tonne clearance of a clupeid in the field; supports f 0.5 against f 1 |
| V3 | Bernreuther 2007, lab, 10-13 cm herring, frozen copepods [F] | 39 L/h per fish, 1-5 % of the theory value; maximum snatch rate 0.85 /s | - | tank arena, so a lower bound on clearance. The snatch rate tests h (the bite-rate ceiling), not a |
| V4 | Gibson & Ezzi 1985, 1990, 1992 [S/A] | switch to filtering at 18 Calanus/L and 50-114 Artemia/L | crossover 12-70 /L from gape x speed / bite rate [D] | the filtering model and the mode choice (2.5) |
| V5 | Bachiller et al. 2018 [F]; Heath 2005; Ecopath Q/B 4.3-5.6 (params note) | adult summer ration ~6 %BW/d, 0.6-1 of Cmax; annual-mean feeding level ~0.1-0.2 | potential intake at the central a with 0.75 B visible, eat share 1, no interference: 9 %BW/d in Apr-May, 6-8 in Jun-Sep, 1.5 in Oct-Nov, 0-0.5 in Dec-Mar; annual 14.8 /yr, so foraging 0.30 of the time eats Q/B 4.5 /yr [D] | that the summer ceiling, not encounter, sets summer intake (yes: B/(1/(ah)) is 2-7 Apr-Aug), and that the annual consumption needs only part-time foraging (yes) |
| V6 | Lohmeyer & Hempel 1977; winter rations of herring (winter_food note) [F] | 0.4-1.0 %BW/d in Jan-Feb | potential 0.0-0.2 %BW/d on the resident pool in Dec-Feb at the central a [D] | **fails by 5-20x.** Winter herring eat prey outside the resident 0-25 m pool (krill and deep Calanus by day and night in the Oslofjord, Onsrud 2004 [F]), or the winter overlap is higher than assumed. A pool-definition issue (flag 3), not a reason to raise a |
| V7 | Bernreuther 2013; Onsrud 2004; Bernreuther 2007 ms 1 [F] | herring night stomach content 0.17-0.24 of the day peak (Bornholm); fullness 0.68-0.70 (Oslofjord, krill) | derived night/day a 0.01 (herring, summer), 0.06 (group) | the dark ratio. Standing contents include the evening meal and overstate night intake. Together they bound the herring night/day intake at ~0.01-0.2 |
| V8 | Dalpadado et al. 2000 [F] | particulate, size-selective on Calanus CIV+; stomach weight rises with zooplankton biomass across regions | - | the particulate mode and the absence of saturation at field densities |
| V9 | Albert 1995; Onsrud 2004 (pout); Bachiller 2018 (blue whiting) [F] | pout 1.1 %BW/d (~0.3 of Cmax), copepods for small pout; blue whiting ~0.6-0.7 of Cmax, mainly krill | pout and blue whiting a on the pool ~0.0005-0.0008 | that their food is largely outside the pool. Their real rations do not test a on this pool |
| V10 | Model curves, literature (herring_functional_response.md Q2) | column-equivalent half-saturation 15-40 t/km2 (Arrhenius & Hansson field; NEMURO calibrated K) | derived 1/(a h) 3.6 annual, 1.6-2.5 in Apr-Sep, 12-35 in Oct-Mar | the order of the half-saturation. The winter value matches the literature, the summer value is lower |

## 4. Judgement, largest uncertainty, and conflicts with the engine's conventions

**Largest uncertainty: the reactive distance of juvenile and adult herring (k_v).** It is unmeasured.
The 1 BL model convention is reported with two reference prey, and R^2 alone spans x0.28-x3.5 in
`a`. The second is where the foraging herring are relative to the 0-25 m prey (O, x0.54-x2.2).
The third is the share of the visual field searched (f, x0.5-x2).

The judgements (J) are:
- foraging-fish O above the all-fish echo shares;
- Ke 1 lx and a foraging depth of 15 m;
- the gadoid light factors;
- capture success for herring;
- winter prey profiles;
- the blue-whiting presence rule;
- the overcast factor.

None of them moves `a` by more than about 2x alone, and all are stated in section 2.

Where the derivation conflicts with the engine's conventions:
1. **Magnitude.** The central a of 0.007 is about 6x below the library's 0.04, and the half-saturation
   is 3.6 against 0.63 t/km2. Section 162 kept 0.04 because winter intake was binding. Encounter
   theory says the high winter intake cannot come from encounters with the resident pool (V6).
2. **Seasonality cannot be represented.** The derived a swings about 20x over the year, from prey
   size, the fish's winter depth and day length. The engine's m(t) carries only the day-length part.
   With rho 0.25 its monthly mean ranges 0.74-1.33. A single library value either overstates winter
   encounters or understates summer ones:
   - the time mean is 0.007;
   - the prey-biomass-weighted mean is 0.0125, which reproduces annual linear-regime consumption;
   - the summer months alone are 0.010-0.016.

   Three readings of the library value, depending on what the caller wants to preserve (with
   rho 0.25 and -8 deg kept):

   | Library a | What it preserves |
   |---|---|
   | 0.007 | the annual time mean of the derived a(t) (the engine's normalisation) |
   | 0.0125 | annual consumption in the linear regime (a weighted by the prey biomass) |
   | 0.0137 | the derived Apr-Sep daytime group a, 0.0193 = a x (0.25 + 0.75 F)/0.709 at F of about 1 |

   This note gives the time mean as the central value; the choice is the caller's.
3. **The pool definition: the brief and the calibration disagree.** The task brief describes the
   pool as copepod-dominated "with some krill/hyperbenthos". The calibration that fixes its
   biomass, A17 WP2 0-25 m carbon (resume 183.6), excludes krill and hyperbenthos, and from Dec to
   Mar it also excludes Calanus. This derivation follows the calibration. That choice is why pout
   and blue whiting are ~2 % of the group a and why the group night/day ratio is low. On 20 mm
   krill, R is about 0.77 m against about 0.1 m on copepods, which is ~60x in R^2
   (`evidence_sprat_pout_bluewhiting.md` 2.4). If the pool is meant to hold krill, the members'
   a, the group a and the dark ratio all rise, and the pool's biomass target would have to include
   krill too. That decision is the caller's. The winter food of herring, and nearly all the food of pout and blue
   whiting, is not in the 0-25 m resident pool: krill, hyperbenthos, and deep or overwintering
   Calanus. Raising a to cover it would credit the pool with prey it does not hold (V6, V9).
4. **dark_ratio and light_threshold.**
   - Derived night/day: 0.06 for the group (0.02-0.24), 0.01 for herring in summer. The engine
     uses 0.25.
   - Derived half-point: -0.6 to -2.3 deg. The engine uses -8 deg.
   - The library's 0.25 rested on biomass-weighting the members' ~1 ratios, but those members
     contribute ~2 % of the a on this pool.
   - The field stomach data (V7) allow herring a night/day intake up to ~0.2, from moonlit or
     twilight biting near the surface. That biting is not in the threshold model.
5. **Double count with hiding.** The resident copepods do not migrate (Maar 2006; Tönnesson 2006). The
   daytime "refuge" from herring is the fish's own depth plus the dilution below 25 m, which is in O.
   The engine applies zooplankton hiding (visibility floor 0.5) on top of that. Prey-side hiding on
   this pair therefore partly double-counts what O already contains.
6. **a while foraging versus all fish.** O is built from all-fish echo shares, with a judgement
   upward shift for foraging fish. If every fish the policy sends to "eat" forages inside the pool
   layer (O of about 1), a doubles. This interpretation belongs to the engine's eat share. Literature
   cannot settle it.

## 5. References

Full citations; the evidence notes give the reading status and quotes.
- Aas E, Høkedal J, Sørensen K 2014. Secchi depth in the Oslofjord-Skagerrak area: theory, experiments and relationships to other quantities. Ocean Sci 10:177-199. [F]
- Aksnes DL, Dupont N, Staby A, Fiksen Ø, Kaartvedt S, Aure J 2009. Coastal water darkening and implications for mesopelagic regime shifts in Norwegian fjords. MEPS 387:39-49. [F]
- Aksnes DL, Utne ACW 1997. A revised model of visual range in fish. Sarsia 82:137-147. doi 10.1080/00364827.1997.10413647 [A]
- Albert OT 1995. Diel changes in food and feeding of small gadoids on a coastal bank. ICES J Mar Sci 52:873-885. [F]
- Arrhenius F, Hansson S 1999. Growth of Baltic Sea young-of-the-year herring is size-dependent and limited by food. MEPS 191:295-299. [F]
- Arst H et al. 1993. Proc Estonian Acad Sci Ecol 3:38 (Skagerrak optics, midsummer 1990). [F]
- Bachiller E et al. 2018. Bioenergetics modeling of the annual consumption of zooplankton by pelagic fish feeding in the Northeast Atlantic. PLoS ONE 13:e0190345. [F]
- Batty RS, Blaxter JHS, Libby DA 1986. Herring (Clupea harengus) filter-feeding in the dark. Mar Biol 91:371-375. [S]
- Batty RS, Blaxter JHS, Richard JM 1990. Light intensity and the feeding behaviour of herring, Clupea harengus. Mar Biol 107:383-388. [S]
- Bernreuther M 2007. PhD thesis, Univ. Hamburg (herring and sprat feeding, Bornholm Basin; lab functional response). [F]
- Bernreuther M et al. 2013 (diel stomach contents of herring and sprat, Bornholm Basin), via `../Sillens vinterföda/zooplankton_refuge.md`. [F]
- Blaxter JHS 1966/1968. The feeding of herring larvae and their ecology in relation to feeding. CalCOFI Rep 10. [F]
- Collard F et al. 2017. Morphology of the filtration apparatus of three planktivorous fishes and relation with ingested anthropogenic particles. Mar Pollut Bull 116:182-191. [F]
- Dalpadado P, Ellertsen B, Melle W, Dommasnes A 2000. Food and feeding conditions of Norwegian spring-spawning herring through its feeding migrations. ICES J Mar Sci 57:843-857. [F]
- Degnbol P, Munch-Petersen S 1985. ICES CM 1985/H:46 (blue whiting and krill, Skagerrak, diel). [F]
- Falk U 1968. ICES CM 1968/H:17 (wintering herring, Norwegian Channel/Skagerrak). [F, poor OCR]
- Falkenhaug T et al. 2022. Front Mar Sci 9:779335 (Calanus at Arendal St. 2). [F]
- Fernandes PG et al. 2001. ICES CM 2001/Q:08 (North Sea herring school depth, diel). [F]
- Fiksen Ø, Folkvord A 1999. Modelling growth and ingestion processes in herring Clupea harengus larvae. MEPS 184:273-289. [F]
- Gibson RN, Ezzi IA 1985. Mar Biol 88:109-116; 1990 Mar Biol 107:357-362; 1992 J Fish Biol 40:577-590. [S/A]
- Hamann L et al. 2023. Diversity of filter feeding and variations in cross-flow filtration of five ram-feeding fish species. Front Mar Sci 10:1253083. [F]
- Kaartvedt S, Røstad A, Klevjer TA, Staby A 2009. Use of bottom-mounted echo sounders in exploring behavior of mesopelagic fishes. MEPS 395:109-118. [F]
- Kulke R 2018. PhD thesis, Univ. Hamburg (sprat and herring functional responses, lab and field). [F]
- Langbehn TJ, Aksnes DL, Kaartvedt S, Fiksen Ø, Jørgensen C 2019. Light comfort zone in a mesopelagic fish emerges from adaptive behaviour along a latitudinal gradient. MEPS 623:161-174 (+ Suppl.). [F]
- Langbehn TJ, Varpe Ø 2017. Sea-ice loss boosts visual search: fish foraging and changing pelagic interactions in polar oceans. Glob Change Biol 23:5318-5330 (read in Langbehn 2019 thesis). [F]
- Link J 1996. Capture probabilities of Lake Superior zooplankton by an obligate planktivorous fish - the lake herring. Trans Am Fish Soc 125:139-142. [A]
- Lohmeyer U, Hempel G 1977 (winter herring rations), via `../Sillens vinterföda/winter_food_and_feeding.md`. [F]
- Maar M et al. 2006. MEPS 313:157 (copepod vertical distribution and grazing, Skagerrak). [F]
- Meskendahl L 2013. PhD thesis, Univ. Hamburg (sprat and herring swimming, lab). [F]
- Munk P, Kiørboe T 1985. Feeding behaviour and swimming activity of larval herring in relation to density of copepod nauplii. MEPS 24:15-21. [F]
- Nøttestad L 2002, Rose KA et al. 2008, Heath MR 2005: via the sibling notes named at the top. [F]
- Onsrud MSR, Kaartvedt S, Røstad A, Klevjer TA 2004. Vertical distribution and feeding patterns in fish foraging on the krill Meganyctiphanes norvegica. ICES J Mar Sci 61:1278-1290. [F]
- Tönnesson K, Nielsen TG, Tiselius P 2006. MEPS 314:213 (Skagerrak copepod strata, day and night). [F]
- Utne ACW 1997. The effect of turbidity and illumination on the reaction distance and search time of the marine planktivore Gobiusculus flavescens. J Fish Biol 50:926-938. [A]
- van Deurs M, Jørgensen C, Fiksen Ø 2015. Effects of copepod size on fish growth: a model based on data for North Sea sandeel. MEPS 520:235-243. [F]
- Varpe Ø, Fiksen Ø 2010. Seasonal plankton-fish interactions: light regime, prey phenology, and herring foraging. Ecology 91:311-318. [A; parameters via van Deurs 2015 and Langbehn & Varpe 2017]
- Viitasalo M et al. 1998. Predation vulnerability of planktonic copepods. MEPS 175:129-142. [A]
- Viitasalo M, Flinkman J, Viherluoto M 2001. Zooplanktivory in the Baltic Sea: prey selectivity by Clupea harengus and Mysis mixta. MEPS 216:191-200. [F]
- Data: ICES DATRAS NS-IBTS (CPUE per length, CA), HERAS acoustic logs (via `../Ekolodsdata planktivorer/`), SHARK A17 zooplankton (SMHI).
