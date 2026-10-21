# What regulates the small pelagic pool: predation, food, fishing and density dependence (task Q3)

Scope: the `pelagic_fish` pool (herring, sprat, sandeel, mackerel, horse mackerel and other
planktivores), North Sea / Skagerrak. This note builds on, and does not redo,
`../Täthetsberoendets styrka per grupp/pelagic_fish.md` (stock-recruitment steepness,
compensation elasticities, density-dependent growth at population level, spatial scale).
New here:
- predation mortality M2 by predator from the ICES SMS North Sea key run 2023 output files;
- surplus-production density dependence computed from the same output;
- weight-at-age against stock size from the same output;
- the literature on bottom-up versus top-down control of the pool.

Tags:
- **[L]** literature, quoted or paraphrased from a source I read;
- **[C]** my own arithmetic on cited data;
- **[I]** inference.

No code or library value was changed.

Model numbers used (from the brief and `fgconfig/fg_library.yaml`):
- M1 0.58 /yr: M0 0.391 plus non-FG predation 0.189.
- Budgeted M2 by FG: gadoids 0.30, porpoises 0.012, seals 0.0015, seabirds 0.024 (muted). The
  pelagic self-pair, 0.142, is unchecked and sits nowhere.
- F 0.10.
- r_max 0.648, so the derived gross production budget is r_max + M1 + M2 = 1.54 /yr.
- Realised in the 3-year run: predation 0.25-0.4 /yr, no starvation, and the pool doubles.

---

## 0. Already established (summary of the sibling note, not redone)

- **Compensation.** Stock-recruitment compensation is strong where stocks sit above their
  S-R breakpoint: NSAS herring, sprat and mackerel. It is weak or absent in the depleted
  components: WBSS herring and sandeel area 3r. Steepness is about 0.61-0.74 for herring,
  0.49-0.57 for sprat and 0.81-0.89 for mackerel. Clupeiformes have B_MSY/B0 of about 0.26
  (Thorson et al. 2012).
- **Size of the compensation.** The total recruitment-equivalent compensation at B_ref is
  about 0.3 /yr (range 0.16-0.55).
- **Growth.** Density-dependent somatic growth is weak in herring at the population level
  (Brunel & Dickey-Collas 2010), and clear in mackerel (Olafsdottir et al. 2016) and Baltic
  sprat (Casini et al. 2011).
- **Scale.** Recruitment regulation acts at stock scale, outside the 60 x 60 km box. Any
  closure should key on domain-mean (or regional) biomass, not on per-cell biomass.
- **Interference.** Per-capita intake falls only 0-5 % per doubling of stock density
  (`../Sillens vinterföda/herring_interference.md`).

---

## 1. Predation mortality M2 by predator (SMS North Sea key run 2023)

### Source and method
- **[L] Source.** ICES WGSAM, SMS North Sea key run 2023 (Vinther, 30 Nov 2023). The output
  files are in the WGSAM GitHub repository, `NorthSeaKeyRun_2023/`:
  - `summary.out`: per year, quarter, species and age, with M1, M2, F, Z, N, biomass BIO and
    yield;
  - `Input_Output/Output/WhoEatsWhom/who_eats_whom_level1.csv`: eaten biomass and partial M2
    by predator, predator age, prey and prey age, per quarter.

  Links: [repo](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023),
  [key run page](https://ices-eg.github.io/wg_WGSAM/NS_2023_key_run.html). Fetched and
  parsed in full.
- **[L] Predators and prey in SMS.**
  - Predators: cod, whiting, haddock, saithe, hake, mackerel, North Sea and western horse
    mackerel, grey gurnard, starry ray (*A. radiata*), grey seal, harbour porpoise and eight
    seabird species.
  - Prey: herring, sprat, northern and southern sandeel, Norway pout, and juvenile cod,
    whiting and haddock.
  - Mackerel and horse mackerel are **not prey** in SMS, so their M2 is 0. Harbour seal is
    not a predator in SMS.
- **[C] Method.** I computed M2 as annual biomass eaten divided by mean annual biomass (the
  mean of the four quarterly BIO values). This is the per-biomass rate the model's
  `predation_mortality` represents. It is lower than the numbers-based age M2 for young
  ages: herring age-1 numbers-based M2 is 0.40, against 0.30 by biomass.
- **[C] Grouping.** Predators are grouped as the model groups them:
  - gadoids = cod + whiting + haddock + saithe + hake;
  - "mack+hmack" = mackerel + both horse mackerel stocks, which in the model is the
    unchecked pelagic self-pair;
  - birds = all eight seabirds;
  - other fish = gurnards + starry ray.

### Results, mean 2013-2022 [C]

| Prey (biomass) | gadoids | mack+hmack | birds | porpoise | grey seal | other fish | **M2 total** | SMS M1 |
|---|---|---|---|---|---|---|---|---|
| Herring, all ages (2 320 kt) | 0.150 | 0.020 | 0.017 | 0.013 | 0.001 | 0.006 | **0.206** | 0.10 |
| Herring age 1 (267 kt) | 0.220 | 0.000 | 0.054 | 0.013 | 0.001 | 0.009 | **0.298** | 0.10 |
| Herring age 2 (477 kt) | 0.175 | 0.000 | 0.011 | 0.013 | 0.001 | 0.000 | **0.200** | 0.10 |
| Herring ages 1-2 | 0.191 | 0.000 | 0.026 | 0.013 | 0.001 | 0.003 | **0.235** | 0.10 |
| Herring age 0 (71 kt) | ~0.15 | 0.66 | 0.10 | - | - | 0.16 | **1.16** | 0.05 |
| Sprat, all ages (783 kt) | 0.327 | 0.476 | 0.149 | 0.000 | 0.000 | 0.034 | **0.987** | 0.20 |
| Sandeel N+S (987 kt) | 0.212 | 0.161 | 0.097 | 0.013 | 0.099 | 0.222 | **0.804** | 0.20 |
| Mackerel, horse mackerel | - | - | - | - | - | - | **0** (not prey) | 0.15 |

Notes on the table:
- **Herring by predator.** Saithe dominates: 0.093 of the 0.150 gadoid M2 on all herring,
  and 0.104 at ages 1-2. Whiting is 0.028, cod 0.021 and hake 0.008. Haddock eats no
  herring in SMS.
- **Herring over time.** All-age herring M2 was 0.165 in 1991, 0.224 in 2003-2012 and 0.206
  in 2013-2022, against a 1974-2022 mean of 0.232.
- **Sprat by predator.** Mackerel 0.403, whiting 0.304, birds 0.149, North Sea horse mackerel
  0.073 (2013-2022).
- **Sandeel by predator.** Northern sandeel: grey seal 0.180, haddock 0.175, gurnards 0.137,
  whiting 0.123. Southern sandeel: gurnards 0.287, mackerel 0.241, birds 0.106.
- **[L] Convention.** SMS residual M1 (from `summary.out`) is fixed by convention: herring 0.10
  /yr (0.05 at age 0), sprat and sandeel 0.20, mackerel 0.15.

### Pool-weighted M2 [C]

Mackerel and horse mackerel are given M2 = 0, as in SMS.

| Weighting | gadoids | mack+hmack | birds | porpoise | seal | other fish | **M2** | SMS M1 | **M = M1 + M2** |
|---|---|---|---|---|---|---|---|---|---|
| NS Ecopath 1991 shares (her 0.30, sandeel 0.36, mac 0.20, sprat 0.07, hom 0.07) | 0.144 | 0.097 | 0.050 | 0.008 | 0.036 | 0.084 | **0.42** | 0.16 | 0.58 |
| Site A: herring 0.8, sprat 0.1, mackerel 0.1 | 0.152 | 0.064 | 0.028 | 0.010 | 0.001 | 0.008 | **0.26** | 0.12 | 0.38 |
| Site B: herring ages 1-2 0.7, all-age herring 0.1, sprat 0.1, sandeel 0.05, mackerel 0.05 | 0.192 | 0.058 | 0.040 | 0.011 | 0.006 | 0.018 | **0.32** | 0.12 | 0.44 |
| **Model** (budget; seabirds muted) | 0.30 | 0 (unchecked 0.142) | 0 (0.024 muted) | 0.012 | 0.0015 | in M1 0.189 | **0.31** | 0.58 | **0.89** |

The site weightings A and B are illustrative and herring-dominated, following the brief and
the IBTS Q1 site data in `../Sillens vinterbudget/winter_budget.md` §4. The composition itself
is task Q1.

### Inference [I]
- **Model gadoid M2 is too high.** Its budgeted gadoid predation on the pool (0.30) is about
  2x the SMS gadoid M2 at herring-dominated site weights (0.15-0.19). It matches the North
  Sea pool value only with sandeel and sprat in the pool. This agrees with the corrected (f)
  of `../Torskfiskarnas reglering/recommendation.md`: Ecopath gives the pool M2 by gadoids as
  0.30 because of sandeel, Norway pout and sprat.
- **The missing pelagic self-pair matters little at the site.** At herring-dominated weights
  it is worth 0.06 /yr (sprat eaten by mackerel), not 0.142, because mackerel and horse
  mackerel eat sprat, sandeel and 0-group herring, not 1+ herring.
- **Seabirds** at site weights are 0.03-0.04, close to the muted 0.024.
- **Total natural mortality is about 2x SMS.** The model's M1 + M2 is 0.89 /yr, against
  0.38-0.44 from SMS at site weights and 0.58 at the North Sea 1991 Ecopath weights.
  - The model's M1 (0.58) alone exceeds SMS M1 + M2 at site weights.
  - Most of the gap is the Ecopath M0 (0.39), a balancing residual, against SMS M1 0.10 for
    herring.
  - The 0.189 non-FG predation compares with SMS gurnards + rays + mackerel and horse
    mackerel + hake of about 0.03-0.08 at site weights.
- **So option (e), "higher predation M2", has no support from SMS.** The pool is not
  under-predated relative to the literature: its literature losses are already met or
  exceeded. Realised predation in the 3-year run (0.25-0.4 /yr) is in the SMS range for
  herring ages 1-2 (0.20-0.30).
- **The unregulated growth comes from the production side.** The derived budget pays gross
  production of 1.54 /yr. For comparison:
  - SMS herring Z in 2013-2022 is 0.31 (M) + 0.21 (C/B) = 0.52 [C];
  - Ecopath 1991 herring P/B = Z is 0.80 for adults and 1.31 for juveniles (`fish.md`).
- **Caveat.** SMS M2 is conditional on SMS predator abundances and stomach data, and its M1
  is a convention. Herring also eaten by unmodelled predators (cephalopods, sharks, harbour
  seal, minke whale, tuna) would raise M. Harbour seal is a model FG whose budget here is
  0.0015.

---

## 2. Bottom-up limitation: food, condition, weight-at-age and recruitment

### Literature [L]
- **Heath 2005, North Sea 1973-2000.** Omnivorous zooplankton production of about 35 g C m-2
  y-1 had "annual fluctuations ... positively correlated with the gross production of
  planktivorous fish. The results suggest a 'bottom-up' control of the pelagic foodweb."
  "Individual planktivore species have been impacted by fishing, but the populations of other
  functionally similar species have expanded to fill the vacant niches." *Calanus
  finmarchicus* "contributed approximately 20% of the annually averaged biomass ... but
  declined to <2% by 1999". (fulltext)
  [doi:10.1016/j.icesjms.2005.01.023](https://doi.org/10.1016/j.icesjms.2005.01.023);
  [Strathprints PDF](https://strathprints.strath.ac.uk/18596/)
- **Lynam et al. 2017, PNAS** (North Sea, more than 40 years, 6 plankton, 8 fish and 1 bird
  group). (fulltext via PMC)
  [doi:10.1073/pnas.1621037114](https://doi.org/10.1073/pnas.1621037114)
  - Fishing "was the most important predictor of sandeel biomass and, after the population
    delay term, also for herring biomass".
  - Climate effects through plankton were weak for herring, which rose after 2000 "despite
    low abundance of its significant prey group: small copepods".
  - Sprat is bottom-up: "cold temperatures lead to a decrease in *C. helgolandicus* and, thus,
    sprat". Sprat was also held down by herring.
  - The lagged own-biomass term ranked first for herring, which the authors read as
    "population dependency".
- **Payne et al. 2009.** NSAS recruitment failed in 2002-2007 "despite high biomass levels",
  arising "during the larval overwintering phase". The timing coincides with a plankton regime
  shift around 2000. (fulltext, via the sibling note)
  [doi:10.1093/icesjms/fsn211](https://doi.org/10.1093/icesjms/fsn211)
- **Alvarez-Fernandez et al. 2015,** "Effect of zooplankton on fish larval abundance and
  distribution: a long-term study on North Sea herring", ICES J. Mar. Sci. 72: 2569-2577. The
  2014 working-document version reports that the larval prey *Temora* and *Pseudocalanus*
  declined sharply in the 2000s. (listing and secondary only; not read)
- **Lusseau et al. 2014.** Cumulative food intake up to February is proposed as a significant
  determinant of NSAS larval survival. (search summary only)
  [doi:10.1093/icesjms/fsu070](https://doi.org/10.1093/icesjms/fsu070)
- **Sandeel.** Recruitment is explained 30 % by SSB x age-1 interaction and **65 % with *C.
  finmarchicus* added** (van Deurs et al. 2009, via the sibling note). Bottom-up food control
  of recruitment is strong.
- **Mackerel.** Growth and weight-at-length fell with mackerel and NSS herring stock size (r2
  = 0.89 for weight-at-length; Olafsdottir et al. 2016, via the sibling note). This is
  density-dependent food competition at ocean-basin scale.

### Weight-at-age against stock size, SMS 2023 [C]
Quarter-1 weight in the sea (`west`), 1974-2023, regressed as ln W on ln (annual mean stock
biomass):

| Species, age | Elasticity d ln W / d ln B (r) | Detrended (r) | Mean W 1974-1990 -> 2013-2022 |
|---|---|---|---|
| Herring 1, 2 | 0 (fixed weights in SMS) | 0 | 12 / 75 g, constant |
| Herring 3 | -0.034 (-0.41) | +0.001 (0.01) | 136 -> 119 g |
| Herring 4 | -0.031 (-0.36) | +0.014 (0.18) | 171 -> 147 g |
| Herring 5 | -0.020 (-0.29) | +0.016 (0.25) | 192 -> 171 g |
| Sprat 1 / 2 | -0.086 (-0.20) / -0.047 (-0.15) | -0.096 / -0.058 | 8 -> 8 g; 12 -> 11 g |
| N. sandeel 1 / 2 | -0.106 (-0.71) / -0.031 (-0.34) | -0.033 / -0.036 | 5 -> 6 g; 10 -> 11 g |

- **Herring.** Herring biomass rose from 1 057 kt (1974-1990 mean) to 2 320 kt (2013-2022)
  while weight-at-age 3-5 fell 11-14 %. Between period means that is an elasticity of about
  -0.15 to -0.17. The year-to-year slope is -0.02 to -0.03 and vanishes once a linear trend is
  removed, so the decline is confounded with the long-term temperature and plankton trend.
- **Sprat and sandeel.** Their slopes are -0.03 to -0.10, weak but of the right sign.

### Inference [I]
- **Literature control.** For herring the literature puts control in fishing and recruitment
  (larval survival, set by plankton and environment), not in post-recruit food limitation. For
  sprat and sandeel, plankton (*C. helgolandicus*, *C. finmarchicus*) drives recruitment.
- **Weight-at-age elasticity.** Density-dependent individual growth is at most about -0.03
  (year-to-year) to -0.17 (decadal, confounded) per ln B. A doubling of the pool would cut
  per-capita growth by at most 2-11 %.
- **What the model's food term can do.** Only bottom-up limitation through zooplankton can
  regulate the model pool today. The literature says that limitation acts mainly on
  recruitment and larvae, which the age-less pool does not resolve. On adult growth it acts
  only weakly.
- **The model's present behaviour.** It grazes zooplankton down until intake falls. That is a
  stronger food feedback on the pool's own growth than the literature shows for herring.
  Zooplankton is depleted instead of the herring being limited. See
  `zooplankton_consumption.md`.

---

## 3. Relative roles of fishing, recruitment/environment and predation

### Literature [L]
- **Lynam et al. 2017:** "top-down impacts of fishing have dominated changes in the biomass of
  commercially exploited fish", while bottom-up processes dominate plankton.
- **Dickey-Collas et al. 2010** (via the sibling note): NSAS herring collapse and recovery were
  driven by fishing and recruitment. SMS predation mortality on 2+ herring "has been very
  stable" over a 100-fold change in herring biomass.
- **Engelhard et al. 2014** (ICES J. Mar. Sci. 71: 90-104). Climate affects forage-fish
  productivity, but single-species climate analyses can mislead because forage fish interact
  strongly. Sandeel is the most important forage-fish prey. "Apart from fisheries, the main
  predators are a few piscivorous fish (saithe, whiting, mackerel and horse mackerel), while
  seabirds and seals have a smaller impact." (abstract)
  [doi:10.1093/icesjms/fst087](https://doi.org/10.1093/icesjms/fst087)
- **Heath 2005,** planktivore guild:
  - "Landings and fish predation losses of planktivores were approximately equal to gross
    production during the mid-1970s, but declined steadily ... By the late 1990s, predation
    and landings losses were only half gross production." The rest went to unaccounted
    mortality.
  - Planktivore landings were 10-20 % of gross production.
  - Planktivore gross production was steady at 3.6-3.8 g C m-2 y-1 per decade (Table 4).

### SMS 2013-2022 loss partition [C] (eaten/B and yield/B)

| | C/B | M2 | SMS M1 | Z |
|---|---|---|---|---|
| Herring | 0.21 | 0.21 | 0.10 | 0.52 |
| Sprat | 0.20 | 0.99 | 0.20 | 1.39 |
| Sandeel | 0.27 | 0.80 | 0.20 | 1.27 |

### Inference [I]
- **Herring:** fishing and predation are about equal, and predation is about 70 % gadoid,
  mostly saithe.
- **Sprat and sandeel:** predation is 3-5x fishing, with mackerel the largest single predator
  of sprat.
- Neither stock is described in the literature as regulated by local food once recruited.

---

## 4. Density dependence quantified

### 4.1 Surplus production against biomass, SMS 2023 output [C]
Per-capita surplus production is p_t = ln(B_{t+1}/B_t) + C_t/B_t, with B the annual mean
total biomass and C the SMS yield. p is net of M1 and M2. It is the quantity the model's
"r_max - F" stands for at ideal food. I regressed p on B (Schaefer form p = r0 (1 - B/K)).

| Stock, period | n | mean B (kt) | **mean p (/yr)** | mean C/B | slope (per kt), r | r0 | K (kt) |
|---|---|---|---|---|---|---|---|
| Herring 1974-2022 | 49 | 1 852 | 0.29 | 0.29 | -1.7e-4, -0.47 | 0.60 | 3 600 |
| Herring 1990-2022 | 33 | 2 267 | 0.17 | 0.22 | -8.1e-5, -0.17 | 0.35 | 4 340 |
| Herring 2002-2022 | 21 | 2 545 | **0.06** | 0.18 | +5.2e-5, +0.09 | - | none |
| Herring 2013-2022 | 10 | 2 320 | **0.00** | 0.21 | n.s. | - | none |
| Sprat 1990-2022 | 33 | 616 | 0.28 | 0.31 | -1.4e-3, **-0.65** | 1.14 | 810 |
| Sprat 2013-2022 | 10 | 783 | 0.07 | 0.21 | -1.4e-3, -0.51 | 1.17 | 835 |
| Sandeel N+S 1974-2022 | 49 | 1 508 | 0.30 | 0.35 | ~0, -0.02 | 0.32 | none |
| Sandeel N+S 2002-2022 | 21 | 986 | 0.21 | 0.31 | -3.4e-4, -0.25 | 0.54 | 1 620 |
| Sandeel N+S 2013-2022 | 10 | 987 | 0.11 | 0.27 | n.s. | - | none |

Caveats:
- Regressing p on B with B on both sides biases the slope negative (time-series bias).
- Recruitment pulses dominate the year-to-year scatter.
- The fits describe SMS model output, not raw data.
- The absolute biomasses are North Sea-wide stock totals. Over about 570 000 km2 they are
  herring ~4.1, sprat ~1.4 and sandeel ~1.7 t/km2 in 2013-2022 [C]. This is a scale check
  only; the site density is task Q1.

### 4.2 Inference [I]
- **Long-term herring.** Over the whole series herring behaves like a Schaefer pool with r0 ≈
  0.6 /yr and K ≈ 3.6 Mt. That r0 is close to the library r_max (0.648) and the FishBase
  herring prior (0.44, CL 0.29-0.65).
- **Herring in the present regime (2002-2022).** No density dependence is detectable, and
  realised surplus production is only 0.00-0.06 /yr at today's biomass, against catches of
  0.18-0.21.
  - The stock is held near its level by low recruitment ("low productivity regime since
    2002", ICES), not by a crowding term.
  - This bears directly on task Q2: today's realised net production of the dominant pool
    member is near zero, not 0.55 (r_max - F).
- **Sprat** has the clearest density dependence: r0 ≈ 1.1-1.2 /yr, K ≈ 0.8 Mt, r = -0.5 to
  -0.65, stable since 1990. This is the classic short-lived-clupeid signature, with
  compensation through recruitment.
- **Sandeel** shows weak or no biomass density dependence. Its regulation is the age-1 versus
  age-0 cohort interaction (Arnott & Ruxton 2002; van Deurs et al. 2009) plus *Calanus*.
- **Predation is weakly depensatory.** The elasticity of herring M2 (eaten/B) against
  herring biomass across 1974-2022 is -0.10 (r -0.46): per-capita predation falls slightly as
  herring rise. Sprat gives -0.03 (r -0.16). This confirms the sibling note: predation does
  not provide compensation, and the model's saturating predators are depensatory too
  (-0.29 /yr computed there).
- **Weight-at-age** elasticity is -0.03 to -0.17 (section 2).

---

## 5. Summary for the recommendation (inference; the ranking is in recommendation.md)

1. **Predation (e).** Do not raise predation to regulate the pool. SMS M2 at
   herring-dominated site weights is 0.26-0.32 /yr, M1 + M2 is 0.38-0.44, and the model
   already applies M1 + M2 = 0.89.
   - The model's gadoid M2 budget (0.30) is about 2x the SMS gadoid M2 on site herring
     (0.15-0.19).
   - The unchecked self-pair is worth about 0.06 at site weights, not 0.142.
2. **Present productivity of the dominant member.** NSAS herring has had near-zero surplus
   production (0.00-0.06 /yr) for 20 years. Sprat compensates strongly around K ≈ 0.8 Mt.
3. **Food feedback.** Bottom-up control in the literature acts on recruitment (larvae, age-0
   sandeel, sprat via *C. helgolandicus*), not as adult food limitation. Weight-at-age
   elasticity is at most about -0.03 to -0.17.
   - A closed-box pool regulated only by grazing its own zooplankton down regulates more
     through prey depletion than the literature supports.
   - A biomass term acting at stock or regional scale (option a or d) is the form the
     evidence supports.
4. **Density dependence, where measurable.**
   - Herring: Schaefer r0 0.6, K 3.6 Mt (1974-2022). None detectable since 2002.
   - Sprat: r0 1.1-1.2, K 0.8 Mt.
   - Sandeel: cohort-level, not biomass-level.

---

## Source status

| Source | Used for | Read as |
|---|---|---|
| ICES WGSAM SMS North Sea key run 2023, `summary.out`, `who_eats_whom_level1.csv` ([GitHub](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023)) | M1, M2 by predator, biomass, yield, weight-at-age; all [C] tables | data files downloaded and parsed |
| [SMS key run 2023 page](https://ices-eg.github.io/wg_WGSAM/NS_2023_key_run.html) | species and predator list | fulltext (figures not read) |
| [WGSAM Herring M2 comparison page](https://ices-eg.github.io/wg_WGSAM/NorthSeaKeyRun_HerringM2.html) | pointer to who-eats-whom files | fulltext (code only) |
| Heath 2005, ICES JMS 62: 847-868 ([Strathprints](https://strathprints.strath.ac.uk/18596/)) | bottom-up correlation, planktivore losses vs production, Table 4 | fulltext (PDF) |
| Lynam et al. 2017, PNAS 114: 1952 ([PMC5338359](https://pmc.ncbi.nlm.nih.gov/articles/PMC5338359)) | fishing vs climate vs predation | fulltext via page summary |
| Engelhard et al. 2014, ICES JMS 71: 90 ([doi](https://doi.org/10.1093/icesjms/fst087)) | main predators of forage fish | abstract |
| Payne et al. 2009; Arnott & Ruxton 2002; van Deurs et al. 2009; Olafsdottir et al. 2016; Casini et al. 2011; Brunel & Dickey-Collas 2010; Dickey-Collas et al. 2010 | recruitment, growth, condition | via `../Täthetsberoendets styrka per grupp/pelagic_fish.md` (secondary here) |
| Alvarez-Fernandez et al. 2015, ICES JMS 72: 2569 | larval prey decline | listing only; not read |
| Lusseau et al. 2014, ICES JMS (fsu070) | larval food intake | search summary only |
| ICES 2026 advice her.27.3a47d ("low productivity regime since 2002") | regime context | via `../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md` |
