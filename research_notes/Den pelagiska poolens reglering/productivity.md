# Present-day productivity r of the pelagic pool (Q2)

Analogue of `research_notes/Torskfiskarnas reglering/r_pool_present.md` for `pelagic_fish`. Same method, so the two
pools are comparable. Nothing here is a published pool value; every pool number is computed.

Tags: **[literature]** quoted or taken from a source; **[computed]** my arithmetic on source data;
**[inference]** my reading. Data: ICES Standard Graphs (SG) source-data pages, assessment year 2026, downloaded
2026-10-21 and parsed locally (scripts and CSVs in the session scratchpad, not in the repo).

## Summary

- The library r_max 0.648 is the North Sea 1991 Ecopath mix, which is **36 % sandeel at r 1.02**. Without sandeel
  the same FishBase priors give **0.44** [computed; species priors from `reports/Litteraturbaserad growth rate per FG.md` §3.4].
- Recruitment per spawner has fallen in most components (Ricker fits with a shared density slope): NSAS herring
  1.7x, WBSS herring 2.25x, mackerel 3.6x, western horse mackerel 1.6x, sandeel (3r/2r) 2.5-4.5x. Sprat shows no
  decline (0.6-0.8x, i.e. higher today) [computed].
- Present-day pool r (Euler-Lotka, r_now = r_max (1 - ln d / 3)) [computed]:

| Weighting | r_max | r_now central | range |
|---|---|---|---|
| W-NS (Ecopath NS 1991, with sandeel) | 0.648 | **0.43** | 0.36-0.51 |
| W-noSE (same, no sandeel, no misc) | 0.436 | **0.33** | 0.28-0.39 |
| W-site (provisional Skagerrak guess) | 0.464 | **0.36** | 0.29-0.41 |
| Herring only (HERAS Skagerrak NSAS/WBSS mix) | 0.44 | 0.35 | 0.27-0.40 |

- **Empirical check at today's density:** the stocks' realised surplus production (TSB change + catch, over TSB)
  in 2016-2024 is 0.0-0.12 /yr for herring, mackerel and horse mackerel and ~0.6 /yr for sprat. Weighted, the pool
  produces **~0.12-0.14 /yr** of surplus at today's stock sizes, a little less than it is fished (~0.16 /yr at stock
  level), so the stocks are flat to declining (~-2 to -4 %/yr) [computed].
- **Reading for the model [inference]:** 0.648 is too high for today on two counts: (1) the sandeel weight, which is
  a North Sea-wide share and not a Skagerrak-site share, and (2) a 1.3-3.6x fall in recruitment per spawner in the
  dominant components. A present-day r of **~0.35 (0.29-0.41)** fits a herring-dominated site pool; 0.43 only if
  sandeel keeps its North Sea 1991 weight. And the real stocks do not grow at r at today's density: they produce
  ~0.13 /yr, about what is fished, which is the stationarity the model pool lacks.

---

## 1. Method (as in the gadoid note)

- **r_max per species:** FishBase CMSY resilience prior r, the values behind the library's 0.648
  [literature, via `reports/Litteraturbaserad growth rate per FG.md` §3.4, accessed 2026-10-02]: herring 0.44
  (95 % 0.29-0.65), sprat 0.48 (0.32-0.72), mackerel 0.43 (0.28-0.64), horse mackerel 0.39 (0.26-0.58), sandeel
  1.02 (0.67-1.53; given under *A. tobianus*, based on North Sea stocks), misc filter feeders ~0.64.
  - Cross-check: Froese et al. 2018 CMSY F_MSY (via Sparholt et al. 2021 Table 1) gives r = 2 F_MSY = 0.52 for
    NSAS, 0.62 for WBSS and 0.42 for NEA mackerel [literature, via `Litteraturbaserad growth rate per FG/fish.md` Q4].
    These are fitted to post-2000 data and are higher than the priors for herring; I keep the priors for
    consistency with the library.
- **Productivity decline d:** recruits per unit SSB, early period over recent period, from the ICES SG series.
  Two estimators, because R/SSB alone confounds productivity with compensation when SSB changed between periods:
  - raw ratio of median R/SSB;
  - **Ricker with period-specific intercepts and a shared density slope**: ln(R/S) = a_period - b S;
    d = exp(a_early - a_recent). This compares the slope at the origin (the quantity r depends on) at matched SSB.
    I use it as the central value.
- **Recruitment lag:** R is lagged to the SSB that spawned it. SG recruitment ages: age 0 for NSAS, WBSS, sprat
  (1 July, model year Jul-Jun), horse mackerel and sandeel (SMS); **age 2 for mackerel** (since the 2022 assessment;
  the 2026 advice notes "Prior to the 2022 assessment, recruitment was presented as age 0") [literature]. So mackerel
  R(y) is set against SSB(y-2); the others against SSB(y).
- **Euler-Lotka:** r ≈ ln(alpha_hat)/T, so dividing the low-density reproductive rate by d gives
  r_now = r_max - ln(d)/T, with T = 3/r_max (CMSY rule, Froese et al. 2017), i.e. r_now = r_max (1 - ln(d)/3).
  No separate warming factor (it acts through recruitment and is already in d).
- **Caveat on T [inference]:** T = 3/r_max gives 6-8 yr for herring, mackerel and horse mackerel (plausible) and
  2.9 yr for sandeel (plausible), but 6.3 yr for sprat, whose real generation time is ~2-3 yr. It does not matter
  much here because sprat d ≈ 1.

## 2. Productivity decline per species

All d values [computed] from SG 2026 series. "R/SSB" columns are medians (R in numbers of the SG unit per tonne SSB).

| Stock (SG key) | Early period | Recent period | median R/SSB early / recent | raw d | Ricker d (±1 SE) | median R early/recent | median SSB early / recent (kt) |
|---|---|---|---|---|---|---|---|
| NSAS her.27.3a47d (22468) | 1947-2001 | 2002-2025 | 22.9 / 10.9 | 2.1 | **1.71** (1.54-1.90) | 1.30 | 1478 / 1908 |
| NSAS, early = 1947-1964 | | 2002-2025 | 11.0 / 10.9 | 1.0 | 1.30 (1.14-1.49) | 1.47 | 2686 / 1908 |
| NSAS, early = 1981-2001 (recovery) | | 2002-2025 | 38.4 / 10.9 | 3.5 | 1.72 (1.48-2.00) | 1.94 | 1198 / 1908 |
| WBSS her.27.20-24 (21294) | 1991-2005 | 2011-2025 | 198 / 140 | 1.4 | **2.25** (1.88-2.69) | 2.88 | 281 / 116 |
| Sprat spr.27.3a4 (21311) | 1980-2000 | 2011-2025 | 643 / 818 | 0.79 | **0.60** (0.50-0.72) | 0.46 | 109 / 174 |
| NEA mackerel (25875), R age 2 | 2000-2012 | 2013-2025 | 2.58 / 0.77 | 3.4 | **3.59** (2.61-4.94) | 1.71 | 3379 / 10079 |
| W. horse mackerel (25892) | 1983-2000 | 2009-2025 | 9.1 / 17.8 | 0.51 | **1.96** (1.37-2.80) | 2.71 | 4637 / 813 |
| W. horse mackerel, early = 1983-2008 | | 2009-2025 | 8.4 / 17.8 | 0.47 | 1.31 (0.95-1.81) | 2.45 | 3046 / 813 |
| Sandeel SA3r (21282) | 1987-2005 | 2010-2025 | 660 / 109 | 6.1 | **4.49** (2.75-7.33) | 3.34 | 149 / 322 |
| Sandeel SA2r (21281) | 1983-2000 | 2006-2025 | 391 / 426 | 0.92 | **2.51** (1.41-4.48) | 2.93 | 110 / 34 |

Notes [inference]:
- **NSAS.** Recent R/SSB equals the 1950s value, but SSB was then 1.4x higher; at matched SSB the Ricker fit gives
  1.3x lower productivity than the 1950s and 1.7x lower than the 1981-2001 recovery. The ICES regime statement
  agrees: "The stock has been in a low productivity regime since 2002" [literature, ICES 2026 her.27.3a47d advice,
  doi:10.17895/ices.advice.32133961, quoted in `Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`]. Median
  recruitment fell from ~30 billion (1947-1964) and ~40 billion (1981-2001) to ~20 billion (2002-2025) [computed].
  Central d 1.7, range 1.3-3.5.
- **WBSS.** R/SSB fell while SSB fell 2.4x, i.e. it did not rise as compensation predicts; the Ricker d (2.25) is
  therefore larger than the raw ratio. ICES: "Climate change-related effects have resulted in lower early life-stage
  survival and reduced productivity of the Rugen spawning component" [literature, ICES 2025 her.27.20-24 advice,
  doi:10.17895/ices.advice.27202614, quoted in `Täthetsberoendets styrka per grupp/pelagic_fish.md`]. Central 2.25,
  range 1.5-2.9.
- **Sprat.** No decline: R/SSB is higher now, and recruitment roughly doubled (median R 2011-2025 / 1980-2000 =
  2.2). The 2026 advice notes only year-to-year variation ("decreased incoming recruitment similar to recruitments
  observed in the period 2020-") and uncertainty in recent year classes [literature, ICES 2026 spr.27.3a4,
  doi:10.17895/ices.advice.32134120]. Central d 0.8 (0.6-1.0). Using d < 1 gives r_now above the prior; capping at
  d = 1 (r_now = r_max 0.48) is the conservative choice.
- **Mackerel.** The SG series starts in 1998, so the "early" period (2000-2012) is the high-recruitment phase that
  built the 2010s stock peak; d is relative to that, not to a long-term baseline. The Ricker slope is ~0 (no
  compensation detected), so raw and Ricker d agree (3.4-3.6). ICES uses "the years 2017-2026 ... to reflect the
  currently observed low stock productivity" in the forecast; the 2026 rise in recruitment "is based only on the 2026
  IESSNS age 2 index and remains to be confirmed" [literature, ICES 2026 mac.27.nea advice, published 30 Sep 2026,
  doi:10.17895/ices.advice.30932144, fetched full text]. Central d 3.6, range 1.7 (median-R ratio) - 4.9.
- **Horse mackerel (western stock, which now owns 3.a catches).** Raw R/SSB rose because SSB fell 4-6x; at matched
  SSB productivity is 1.3-2.0x lower. Excluding the huge 1982 year class from the early window matters little for
  the Ricker d. Central 1.6, range 1.0-2.8.
- **Sandeel.** Both areas that include the Skagerrak (SA3r: SD 20 + northern/central North Sea; SA2r: SD 20 +
  central/southern North Sea) show lower productivity. SA3r: "The zero catch advice is a result of low estimated
  recruitment and forecasted SSB"; "The geometric mean has overestimated recruitment in the recent years"
  [literature, ICES 2026 san.sa.3r advice, doi:10.17895/ices.advice.31430314, fetched full text]. Sandeel
  recruitment is dominated by single year classes (e.g. 1996, 2009, 2016) and regulated by age-1 abundance (Arnott &
  Ruxton 2002, via `pelagic_fish.md`), so Ricker on SSB is a poor model; the ±1 SE range is wide. Central d 3.4
  (geometric mean of 4.5 and 2.5), range 2.5-4.5.

## 3. r_now per species

r_now = r_max (1 - ln(d)/3) [computed]. Use this table to reweight with any site composition.

| Component | r_max | d central (range) | r_now central | r_now range |
|---|---|---|---|---|
| NSAS herring | 0.44 | 1.7 (1.3-3.5) | **0.36** | 0.26-0.40 |
| WBSS herring | 0.44 | 2.25 (1.5-2.9) | **0.32** | 0.28-0.38 |
| Herring, Skagerrak mix (NSAS 0.677 / WBSS 0.323, HERAS July 2025) | 0.44 | - | **0.35** | 0.27-0.40 |
| Sprat | 0.48 | 0.8 (1.0-0.6) | **0.52** | 0.48-0.56 |
| Mackerel | 0.43 | 3.6 (1.7-4.9) | **0.25** | 0.20-0.35 |
| Horse mackerel | 0.39 | 1.6 (1.0-2.8) | **0.33** | 0.26-0.39 |
| Sandeel | 1.02 | 3.4 (2.5-4.5) | **0.60** | 0.51-0.71 |
| Misc filter feeders | 0.64 | 1 (no data) | 0.64 | - |

**Reweighting formula:** r_pool = Σ w_i r_now,i / Σ w_i, with w_i the biomass share of component i at the site
(annual mean, or weighted by months present if the pool is meant to represent the box over a year).

## 4. Pool r_now under three weightings

[computed] Ranges combine the per-species ranges in the same direction (all low or all high), so they are wide.

| Weighting | Shares (herring / sprat / mackerel / horse mackerel / sandeel / misc) | r_max | r_now | range |
|---|---|---|---|---|
| **W-NS**: Ecopath NS 1991 (t/km2) | 2.596 / 0.579 / 1.72 / 0.579 / 3.122 / 0.030 | 0.648 | **0.43** | 0.36-0.51 |
| **W-noSE**: same, sandeel and misc dropped | 2.596 / 0.579 / 1.72 / 0.579 / 0 / 0 | 0.436 | **0.33** | 0.28-0.39 |
| **W-site** (provisional; fork 1's `stock_composition.md` sets the real site weights) | 0.65 / 0.10 / 0.18 / 0.03 / 0.04 / 0 | 0.464 | **0.36** | 0.29-0.41 |
| W-site without mackerel (winter-only box) | 0.80 / 0.12 / 0 / 0.03 / 0.05 / 0 | 0.472 | 0.38 | 0.31-0.42 |

Reading [inference]:
- The sandeel share alone moves r_max from 0.44 to 0.65. The site is ~100 m deep off the Swedish coast; sandeel
  are bound to shallow sandy banks, so a North Sea-wide 36 % sandeel share is unlikely to apply. This is the single
  largest lever, and it is a composition question, not a productivity one.
- The productivity decline lowers r by a further ~0.1 for any herring-dominated mix (0.46 -> 0.36).
- A sandeel-heavy mix keeps r_now high (0.43) because sandeel's prior is 1.02 even after a 3.4x decline.

## 5. Today's trajectory and realised surplus production

Realised surplus production SP(y) = TSB(y+1) - TSB(y) + catch(y); rate = SP/TSB. This is net of natural mortality,
so in a Schaefer pool it is r(1 - B/K) at the current state: the empirical net growth at today's density
[computed from SG 2026; sprat TSB from HAWG 2026 Table 11.1.10 as quoted in `ices_pelagic_f.md`; sprat and
sandeel otherwise only have SSB].

| Stock | SP/TSB 1985-2000 | 2001-2010 | 2011-2024 | **2016-2024** | catch/TSB 2016-2025 | ln-TSB trend 2006-2025 | 2016-2025 |
|---|---|---|---|---|---|---|---|
| NSAS herring | 0.19 | 0.11 | 0.12 | **0.12** | 0.149 | -0.4 %/yr | -1.9 %/yr |
| WBSS herring | 0.13 | 0.06 | 0.07 | **0.07** (median 0.03) | 0.081 | -5.4 %/yr | -2.7 %/yr (+7 %/yr 2021-25) |
| NEA mackerel | 0.13 | 0.28 | 0.04 | **0.00** | 0.143 | -2.9 %/yr | -14.2 %/yr |
| W. horse mackerel | 0.01 | -0.04 | 0.07 | **0.07** | 0.053 | -4.3 %/yr | -0.4 %/yr |
| Sprat (TSB, 2021-2024 only) | - | - | - | **0.63** (0.29-0.97) | 0.43 (yield/TSB) | SSB +1.3 %/yr | SSB -5.6 %/yr |
| Sandeel SA3r (SSB-based, unreliable) | 1.91 | 0.71 | 0.24 | 0.19 (median -0.05) | - | SSB +3.2 %/yr | SSB -12.5 %/yr |

Pooled [computed]:
- Herring mix SP: 0.677 x 0.12 + 0.323 x 0.07 = 0.10 /yr.
- W-noSE: (2.596 x 0.10 + 0.579 x 0.63 + 1.72 x 0.00 + 0.579 x 0.07) / 5.474 = **0.12 /yr**.
- W-site: 0.65 x 0.10 + 0.10 x 0.63 + 0.18 x 0 + 0.03 x 0.07 + 0.04 x 0.1 = **0.14 /yr**.
- Stock-level catch rate on the same W-site weights: ~0.16 /yr (herring mix 0.127, sprat 0.45, mackerel 0.143,
  horse mackerel 0.053, sandeel ~0.2). SP (0.14) < catch (0.16), consistent with the observed slow decline.

Reading [inference]:
- **No component is growing today**, apart from WBSS's small rebound under near-zero F and sprat's year-class
  noise. Mackerel is falling fast (-14 %/yr since 2016) on low recruitment and F above F_MSY; sandeel 3r SSB has
  collapsed below Blim; NSAS has drifted down ~2 %/yr. The pool as a whole is flat to declining (~-2 to -4 %/yr).
- **At today's density the real stocks produce ~0.12-0.14 /yr of net surplus**, close to what is fished. Compare the
  model pool: r_max - F = 0.648 - 0.10 = 0.55 /yr at ideal food, and a realised +0.2-0.3 /yr in the 3-year run.
- In Schaefer terms, SP/B = r(1 - B/K): with r_now 0.36 and SP/B 0.13, today's stocks sit at B/K ≈ 1 - 0.13/0.36 =
  0.64 of their present-regime capacity [computed]. This is a stock-scale statement; how it maps to the 60 x 60 km
  box is the subject of `migration_scale.md` and the recommendation.
- SP rates for herring were higher in 1985-2000 (0.13-0.19) than now (0.07-0.12), matching the R/SSB decline.

## 6. Uncertainties

- The SG recruitment of the last 1-3 years is weakly estimated (often a geometric-mean assumption); I used R up to
  2025 and excluded 2026. Medians over 10-24 years limit the effect.
- Mackerel d is relative to the 2000-2012 high-recruitment phase; against a long-term baseline it may be 1.7-2.5
  (median-R ratio), which raises mackerel r_now to 0.30-0.35.
- Sandeel recruitment is set by cohort interactions and episodic year classes, not SSB; d is uncertain by ~2x.
- T = 3/r_max is the largest structural uncertainty, as for the gadoids. A shorter T (e.g. herring 4-5 yr) lowers
  herring r_now by ~0.03-0.06.
- Sprat r_max 0.48 (FishBase "medium" resilience prior) is low for a fish with a 2-3 yr generation time; the CMSY rule
  r ≈ 3/t_gen would give ~1.0-1.5. If the site holds much sprat, its r is the most uncertain number in the pool.
- The weights are provisional; the reweighting table in section 3 lets the caller recompute.

## Source status

| Source | Used for | Status |
|---|---|---|
| ICES SG source data, assessment 2026: her.27.3a47d key 22468, her.27.20-24 key 21294, spr.27.3a4 key 21311, mac.27.nea key 25875, hom.27.2a3a4a5b6a7a-ce-k8 key 25892, san.sa.3r key 21282, san.sa.2r key 21281 — https://standardgraphs.ices.dk/ViewSourceData.aspx?key=NNNNN | R, SSB, TSB, catch, F series; recruitment ages | Fetched full table, parsed |
| ICES 2026 mac.27.nea advice (30 Sep 2026), doi:10.17895/ices.advice.30932144 (https://ndownloader.figshare.com/files/69489192) | "low stock productivity" 2017-2026, age-2 recruitment, 2026 recruitment caveat | Fetched full text (PDF) |
| ICES 2026 spr.27.3a4 advice, doi:10.17895/ices.advice.32134120 (https://ndownloader.figshare.com/files/64136503) | Recruitment statements, age 0 on 1 July | Fetched full text (PDF) |
| ICES 2026 san.sa.3r advice, doi:10.17895/ices.advice.31430314 (https://ndownloader.figshare.com/files/62226832) | Low recruitment, geometric mean overestimates recent R | Fetched full text (PDF) |
| ICES 2026 her.27.3a47d advice, doi:10.17895/ices.advice.32133961 | "low productivity regime since 2002" | Secondary (quoted in `ices_pelagic_f.md`, read in full there) |
| ICES 2025 her.27.20-24 advice, doi:10.17895/ices.advice.27202614 | Rugen early-life survival and productivity | Secondary (quoted in `pelagic_fish.md`) |
| HAWG 2026, Table 11.1.10 (sprat TSB, yield) | Sprat SP/TSB | Secondary (quoted in `ices_pelagic_f.md`) |
| FishBase CMSY resilience priors | r_max per species | Secondary (`reports/Litteraturbaserad growth rate per FG.md` §3.4) |
| Froese et al. 2017, Fish Fish 18:506, doi:10.1111/faf.12190 | r ≈ 3/t_gen, used for T | Secondary (via `fish.md`) |
| Froese et al. 2018, Mar Policy 93:159, doi:10.1016/j.marpol.2018.04.018; Sparholt et al. 2021, ICES JMS 78:55, doi:10.1093/icesjms/fsaa175 | CMSY F_MSY -> r cross-check | Secondary (via `fish.md`); Froese et al. 2016 stock-status report fetch failed (TLS error) |
| Arnott & Ruxton 2002, MEPS 238:199, doi:10.3354/meps238199 | Sandeel recruitment set by age-1 | Secondary (via `pelagic_fish.md`) |
