# Evidence A1: northern shrimp (Pandalus borealis) in the Mareld grid

Scope: the shrimp share of the mobile epifauna that `benthic_community` should hold. Builds on
`../Bentos och torskfiskar/evidence_A_standing_stock.md` (survey 0.28 t/km2 uncorrected, Skagerrak strata 11-16,
2021) and `evidence_B_production_and_food_limitation.md` (M), which are not repeated.

Tags: [F] full text read (incl. values read off a figure, "fig."), [A] abstract only, [S] search snippet or
secondary, [D] my derivation. "(sub)" = read in full by a research subagent of this task and not re-checked by
me; everything else marked [F] I checked in the downloaded PDF. Source PDFs: `data/pandalus/`.

## Short answer

| Quantity | ~100 m | ~300 m | Grid mean | Basis |
|---|---|---|---|---|
| Survey biomass, uncorrected (t/km2) | 0.12 (stratum 14, 100-200 m) | 0.46 (strata 15/16 mean; 0.57 at 200-300 m) | 0.22 | NO shrimp survey 2017-21 [F] [D] |
| Catchability q | 0.40 (0.55 low-biomass case, 0.17 high case) | | | SS3 vs survey [D]; SS3 Ln(Q) (sub); SPM q2 [F] |
| **Standing stock (t WW/km2)** | **0.30 (0.22-0.71)** | **1.15 (0.84-2.7)** | **0.56 (0.41-1.32)** | survey / q [D] |
| r (/yr) | 0.85 (0.5-1.3) | | | SPM r 0.33-0.53 [F]; Hiddink lifespan 4 yr -> 1.3; M 0.6-1.0 |
| Accessible fraction f | 0.7 (0.4-0.9) | | | epibenthic by day; partial night ascent |

The landings give an independent check: 45G0 landings ~0.6 t/km2/yr at F ~0.5 imply ~1.2 t/km2 of fishable
shrimp in the rectangle [D]. That points to the high half of the range (the rectangle also contains the deeper
trench slope, where shrimp are densest). Even the high case adds only ~1.3 t/km2 to a community of ~92, so the
choice of q does not move the community numbers.

## 1. Assessment and absolute biomass

- **Current model**: ICES advice pra.27.3a4a, 6 June 2025 [F] (`data/pandalus/adv2025.pdf`,
  https://doi.org/10.17895/ices.advice.27202809):
  - "An ensemble of three models is used, where each model differs based on the assumed level of natural
    mortality" (Stock Synthesis, quarterly, two-area, length-based; benchmark WKPRAWN 2022).
  - "*Fishing mortality is presented only in relation to FMSY, and total stock biomass is presented only in
    relation to MSY Btrigger." SSB2025/MSY Btrigger 0.37, below Blim.
  - The only absolute number is Figure 3, "Fishable biomass estimated from the reference model of the ensemble
    (median natural mortality), over 1971-2025. Estimates are from 01 July each year." Read off the figure
    [F, fig., +-2 kt]: ~55-63 kt in 1971-2002, minimum ~15 kt in 2010, **18-24 kt in 2017-2024, ~26 kt in 2025**.
  - "has a short lifespan (max. four years) ... The fishery takes place on ages 1 to 3."
  - Survey input: the Norwegian shrimp survey only (sub). No IBTS catchability exists (IBTS is not an input).
- **SS3 survey catchability** (WKPRAWN 2022 Table 3.4.8, https://doi.org/10.17895/ices.pub.19714204) (sub):
  "Ln(Q) - catchability -0.594" and "-0.681", i.e. **q 0.55 and 0.51** [D]. The survey index was rescaled in 2023
  (depth error) (sub), so these carry over only approximately.
- **Empirical survey/fishable ratio [D]** (sub, checked against SCR 21/001 Table 5 totals [F]): survey totals 2017-21
  8.80, 8.25, 5.89, 8.39, 7.59 kt over fishable biomass ~18-24 kt gives **q_eff ~0.40** (0.33-0.43).
- **Earlier Bayesian surplus-production model** (Hvingel; NAFO SCR 15/059, Table 4, medians and quartiles) [F]
  (`data/pandalus/scr15-059.pdf`, https://www.nafo.int/Portals/0/PDFs/sc/2015/scr15-059.pdf):
  - "MSY (ktons) 17 [mean] ... 14.2 16.6 19.5"; "K (ktons) 136 ... 110 127 150"; "**r 0.53** ... 0.41 0.53 0.65";
    "q1 0.22 ... 0.21"; "**q2 0.16** ... 0.14 0.16 0.18".
  - Earlier runs (sub): SCR 12/066 r 0.35, K 151 kt; SCR 13/070 r 0.33, K 158 kt.
  - Both K and q rest on informative priors: "Carrying capacity K informative ~dlnorm(4.65,3.16)"; "Catchability
    survey 2 q2 informative ~dlnorm(-1.75,11)" from "Berenboim et al. (1980) estimated a catchability of 0.173 by
    calibrating trawl catches to the results of a photo survey" [F]. The K prior was scaled by survey area,
    "57,000 km2 in the North Sea" [F]. So the SPM's r and K are not independent of the priors.
  - No SPiCT exists for this stock (sub).
- **Mackinson & Daskalov 2007** (North Sea Ecopath, Table 11.13) [F]: the Norwegian survey stock uses "a catch
  efficiency of 0.2"; mean biomass 109,887 t over 56,799 km2 = 1.93 t/km2 (1980s-2000s, a high-stock period).
- **Area**: survey strata total 15,748 nm2 = 54,000 km2; Skagerrak strata 11-16 4,648 nm2 = 15,900 km2 [F] [D].

## 2. Depth profile (survey strata, 2017-2021) [F, SCR 21/001 Table 1 and Table 5] [D]

`data/pandalus/scr21-001.pdf`, https://www.nafo.int/Portals/0/PDFs/sc/2021/scr21-001.pdf. 1 nm2 = 3.43 km2.

| Stratum | Depth | Area (nm2) | Mean biomass 2017-21 (t) | t/km2 uncorrected | / q 0.40 | / q 0.17 |
|---|---|---|---|---|---|---|
| 14 | 100-200 m | 1411 | 588 | 0.12 | 0.30 | 0.71 |
| 15 | 200-300 m | 739 | 1448 | 0.57 | 1.43 | 3.35 |
| 16 | 300-500 m | 1138 | 1366 | 0.35 | 0.87 | 2.05 |
| 11 (Norwegian side) | 100-200 m | 367 | ~620 | 0.49 | | |

- Text [F]: "In Skagerrak ... the biomass is more or less equally distributed between the depth strata of 100-200 m,
  200-300 m, and 300-500 m". That is per stratum, not per km2: per km2 the density **peaks at 200-300 m and is
  ~4-5x lower at 100-200 m on the southern/eastern side** (stratum 14, the side the grid lies on).
- The Mareld EIA: shrimp "generellt är förekomsten jämnt fördelad inom djupintervallet 100-500 meter" (mkb.txt
  3932-3933) [F]; the survey strata contradict this at 100-200 m.
- Swedish logbooks: shrimp is 98.5 % of the shellfish catch inside the Mareld park area (local_fishery.md) [F]:
  the deep part of the grid is a shrimp ground.
- The survey is in January-February; the standing stock varies seasonally (recruitment of age 0 in summer).
  Fishable biomass excludes the smallest shrimp, so the total is somewhat higher; no factor found.

## 3. Accessibility to benthivorous fish

- Skagerrak survey: "Trawling is carried out around the clock", and sun height was dropped from the index model
  "due to lack of a significant effect" (WKPRAWN 2022, s. 3.4.3) (sub). So in January-February the bottom trawl
  catches shrimp equally by day and night.
- Elsewhere: "Northern shrimp were found to be 6.33 times more abundant in day catches compared to night"
  (Frechette et al. 1983, cited in Baker et al. 2024, MEPS 740:79, https://doi.org/10.3354/meps14651) (sub);
  "Female northern shrimp are not thought to complete diel migrations to the same extent as males" (same) (sub).
- Predators in the Skagerrak: "Saithe, blue whiting, roundnose grenadier, velvet belly and haddock have been shown
  to feed on shrimp in Skagerrak (Skorda, 2018)" (NIPAG 2025) (sub); cod in Jorgensen et al. 2014 [F via
  evidence_B]. Cod stomachs Skagerrak Feb 1991: P. borealis 1.4 % by weight (Dorman et al. 1992) (sub).
- **f [D]**: shrimp sit on the sediment surface (no burial), so the whole on-bottom fraction is reachable. By day
  that is nearly all; at night males partly leave the bottom. Central **0.7**, low 0.4 (strong night ascent,
  Frechette ratio), high 0.9 (Skagerrak winter: no day/night effect).

## 4. Growth rate r

- SPM r 0.33-0.53 (medians of three runs) [F/sub]; prior-influenced through K (r = 4 MSY/K).
- Lifespan max 4 yr [F] -> Hiddink et al. 2019 r = 5.31/longevity = **1.33** (class 3-10 yr 1.24) [D].
- M (WKPRAWN Table 3.4.4, median) 1.71 / 1.02 / 0.70 / 0.58 at ages 0.5 / 1.5 / 3.5 / 7.5 [F via evidence_B];
  F ~0.5 (F2024/FMSY ~0.8 x old FMSY 0.6; advice Fig. 2) [D]. P/B = Z ~1.1-1.5 for the fishable stock [D];
  Mackinson & Daskalov shrimp P/B 3.0 (balanced), Hopkins 1988 1.7 (sub).
- **r central 0.85** (geometric mean of 0.53 and 1.33), range 0.5-1.3.

## Gaps
- No absolute total biomass is published since 2022; the fishable-biomass curve is read off a figure.
- The SS3 q (0.51-0.55) and the SPM q (0.16) differ ~3x; the SPM's q is prior-driven. This is the main
  uncertainty, a factor ~2.4 in shrimp biomass.
- Survey documents 2022-2025 (SCR 22/001 onwards) were not retrieved, so the stratum profile is 2017-21.
- No Skagerrak measurement of the time shrimp spend on the bottom.
