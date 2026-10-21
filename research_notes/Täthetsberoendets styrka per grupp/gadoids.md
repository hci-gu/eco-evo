# Density dependence in gadoid M1: where it acts, how strong, and what elasticity the `gadoids` pool should carry

Scope: about 60 tool calls (Europe PMC / OpenAlex / Crossref / Unpaywall look-ups, web searches, PDF downloads with local text extraction, two new computations on the ICES WGSAM North Sea SMS 2023 key-run files, and one extraction from the FishLife v2 database). No code or config was changed.

Builds on, and does not redo:
- `reports/Täthetsberoende och interferens hos torskfiskar.md` and its notes, mainly `regulation_and_models.md`. The SMS intra-gadoid fit `0.016 + 0.0875*B` comes from there and is re-derived below with the same script logic.
- `reports/Restdödlighet M0 hos torskfiskar.md`, for the M1 = 0.20 + 0.214 split.
- `research_notes/Litteraturbaserad growth rate per FG/fish.md`, for Myers et al. 1997 r_m.

Labels used throughout:
- **Literature** is a sourced finding, with how the source was read.
- **Calculation** is arithmetic shown in full.
- **Inference** is my own reasoning or modelling step.

## 0. Definitions and model numbers used

- **Density measure:** B in t/km2. The reference is B_ref = 0.69 t/km2, which is 2500 t on 3600 km2.
- **Other densities quoted:** 1.4 t/km2 (typical occupied cell, since the policy spreads gadoids over about 2000-3600 cells) and 2.7 t/km2 (the IBTS estimate for the area).
- **M1(B_ref)** = 0.415 /yr = M0 0.20 + 0.214 predation by non-FG predators. Per tick that is 0.000284.
- **beta_M = dM1/d ln B** at the density in question, in /yr.
- **gamma = beta_M / M1(B_ref)** (dimensionless). Two conventions:
  - **"replace":** the floor is re-set so that M1(B_ref) stays 0.415, so gamma = beta/0.415.
  - **"add":** a new term is added on top, so the denominator is 0.415 plus the term's value at B_ref.
- **Forms:**
  - Power: M1_ref (B/B_ref)^gamma. Near B_ref this is approximately M1_ref(1 + gamma ln(B/B_ref)), i.e. log-linear (Gompertz) with beta = gamma M1_ref.
  - Linear: M0 + kB. Then beta = kB, which grows in proportion to B, and gamma = kB_ref/(M0 + kB_ref).
- **Other model numbers:** r_max = 0.51 /yr; F = 0.12 /yr; 1460 ticks/yr.
- **Explicit food density dependence already in the model:** from the w = 1 probe in the prior report, gross production per biomass fell 0.872 -> 0.834 from 1250 to 2500 t and 0.834 -> 0.741 from 2500 to 5000 t.
  - **Calculation:** d(production)/d ln B = -0.038/ln 2 = **-0.055 /yr** (1250-2500 t) and -0.093/ln 2 = **-0.134 /yr** (2500-5000 t; this includes herring depletion).
  - At B_ref the explicit food-mediated beta is therefore about 0.06-0.13 /yr, central ~0.09.

---

## 1. Where and how strongly density dependence acts, species by species

### 1.1 Cod: Skagerrak coastal populations (the model's own region)

**Literature**

- **Fromentin et al. 2001, the strongest quantitative source (fetched full text from Ifremer Archimer).**
  - Data: 68 beach-seine stations in 11 fjords or areas on the Norwegian Skagerrak coast, 44-73 yr of 0-group (September/October, ~0.5 yr old) and 1-group (~1.5 yr) counts.
  - Model, with observation error explicitly included (Myers & Cadigan variance-components method): `N1,t = N0,t-1 exp[-m + a ln(N0,t-1) + eps]` (Eq. 2). Adding delayed (between-cohort) density dependence gives Eq. 6: `... + b ln(N0,t-2)`.
  - They chose a Gompertz form "because Bjørnstad et al. (1999a) showed that the survival of the Norwegian Skagerrak cod is approximately log-linearly related to the abundance (and not as often assumed linearly related ...)".
  - Table 1, direct DDM a (SE):

    | Area | a (SE) |
    |---|---|
    | Topdalsfjord | -0.30 (0.14) |
    | Grimstad | -0.07 (0.13) |
    | Sandnesfjord | -0.43 (0.11) |
    | Søndeledfjord | -0.28 (0.12) |
    | Kragerø | -0.27 (0.13) |
    | Grenland | -0.50 (0.12) |
    | Tjøme | -0.25 (0.10) |
    | Holmestrand | -0.33 (0.10) |
    | Oslofjord | -0.36 (0.12) |
    | Drøbak | -0.27 (0.10) |
    | Hvaler | +0.19 (0.19) |

  - Text: "Most of the areas displayed significant a within a range of 0.25–0.50 ... a values of 0.50 or 0.25 imply that a 100-fold increase in the 0-group cod only results in a 10-fold or a 32-fold increase in the 1-group." Grimstad and Hvaler were non-significant.
  - Delayed DDM b on detrended sub-periods (Table 5): significantly negative in 4 of 5 north-eastern areas, Tjøme -0.49, Holmestrand -0.65, Drøbak -0.75 (Oslofjord -0.20 n.s., Hvaler -0.21). Zero in the south-west.
  - They call some of these values "large if we assume that the average density-independent mortality for fish older than 1 yr is ~0.4 in these populations (an estimation based on capture–recapture modeling; Julliard et al. 2001)".
  - They note the DDM is likely **underestimated**, because the post-settlement period (May-July to Sept/Oct, when losses can be 50-~100 %) falls before the first census.
  - Mechanisms proposed: competition for shelter and food in seagrass and kelp, and cannibalism by the 1-group on the 0-group. They differ regionally with bottom-flora cover.
  - Source: [Fromentin et al. 2001, Ecology 82:567-579](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2), full text via [Archimer PDF](https://archimer.ifremer.fr/doc/00000/712/378.pdf).
- **Bjørnstad et al. 1999 (Ecology).** A model-free GAM test found the 0- to 1-group link "nonlinear and significantly convex (P < 0.05)" in both areas. "The functional form of density-dependence in the per capita survival rate is estimated to be approximately log-linear", attributed to "cannibalism, competition for habitat, and food limitation". (Abstract via OpenAlex.) [Bjørnstad et al. 1999, Ecology 80:1278-1288](https://doi.org/10.1890/0012-9658(1999)080[1278:ANTFDD]2.0.CO;2)
- **Stenseth et al. 1999 (Proc. R. Soc. B) and Bjørnstad et al. 1999 (PNAS).** Age-structured models with "asymmetrical interactions between the juvenile cohorts (0–group and 1–group)" fit 60 series from 1945-1994. Intercohort competition and cannibalism generate the 2-2.5-yr cycles and resonate recruitment noise into long-term trends. (Abstracts only. Full text 403 at PNAS, PMC and Royal Society; Europe PMC full-text API returned 500.) [Stenseth et al. 1999, Proc R Soc B 266:1645](https://doi.org/10.1098/rspb.1999.0827); [Bjørnstad et al. 1999, PNAS 96:5066](https://doi.org/10.1073/pnas.96.9.5066)
- **Fromentin et al. 2002 (reply comment, fetched full text).** Restates the model as `N1,t = N0,t-1 exp(-m + alpha ln N0,t-1 + beta ln N1,t-1 + eps)`. "The spatial structure of the short-term fluctuations is seen to take place at a local scale (< 20 km), whereas that of the long-term fluctuation occurs at a large scale." Three independent methods "all confirm significant within- and between-cohort interactions". [Fromentin et al. 2002, MEPS 241:231-234](https://doi.org/10.3354/meps241231), [Archimer PDF](https://archimer.ifremer.fr/doc/00000/10045/7599.pdf)
- **Fromentin et al. 2000 (fetched full text).**
  - 0- and 1-group cod "were significantly more abundant at sheltered stations in the inner fjord". Similarities were "significantly higher across small scales of 0–20 km".
  - "Since 2- and 3-year-old cod constitute 75% of the spawning stock, the observed 2–2.5 years' cycle is roughly equal to one generation time."
  - Source: [Fromentin et al. 2000, ICES JMS 57:330-338](https://doi.org/10.1006/jmsc.1999.0638), [Archimer PDF](https://archimer.ifremer.fr/doc/00000/10525/7082.pdf). The OUP PDF returned 403.
- **Fromentin et al. 1997 (fetched full text).** Juvenile cod, pollack and whiting at 38 stations show 2-2.5-yr cycles in optimal habitat, "which probably resulted from intrinsic interactions in age-structured populations, such as density-dependent competition and cannibalism". The heterogeneity is at the between-fjord mesoscale and the within-fjord local scale. [Fromentin et al. 1997, MEPS 155:209-222](https://doi.org/10.3354/meps155209)
- **Fromentin et al. 1998 (fetched full text).** Long-term trends in cod and pollack were not linked to the NAO or *Calanus*. The authors did not dismiss fishing and pointed to bottom-flora change. Used only for context. [Fromentin et al. 1998, MEPS 162:265-278](https://doi.org/10.3354/meps162265)
- **Julliard et al. 2001.** Capture-mark-recapture of 36,728 reared and 2415 wild cod in Risør. "Natural mortality was found to be high in the 6–12 mo range, but subsequently it decreased rapidly and was no longer age-dependent." Total mortality varied little between years except after a 1988 algal bloom. Density dependence of adult M was **not** tested. (Abstract via OpenAlex.) [Julliard et al. 2001, Ecol. Appl. 11:540](https://doi.org/10.1890/1051-0761(2001)011[0540:NMAFMI]2.0.CO;2)

**Inference**
- In Skagerrak cod the density dependence measured in mortality is a juvenile (0-1.5 yr) phenomenon. It is log-linear in numbers, so the per-capita mortality elasticity is roughly constant, not proportional to density. It acts at the fjord scale (< 20 km).
- After 1 yr, M is high (~0.4) but no density dependence has been demonstrated.

### 1.2 Cod, whiting, haddock: North Sea and other stocks

**Literature**
- **Myers & Cadigan 1993**, 17 demersal populations: "strong evidence of density-dependent mortality within cohorts during the juvenile stage for cod, plaice, sole, and whiting; density-dependent mortality appears to be related to the log of juvenile abundance. There is evidence of negative autocorrelation between adjacent cohorts of cod". Haddock was examined but is **not** in the list with strong evidence. (Abstract via OpenAlex; CJFAS full text not reachable; coefficients not seen.) [Myers & Cadigan 1993, CJFAS 50:1576](https://doi.org/10.1139/f93-179)
- **Myers et al. 1997**, cod slope at the origin from Ricker fits (Table 1, fetched full text; r_m assumes adult M = 0.2):

  | Stock | r_m (/yr) | slope at origin | age at maturity (yr) |
  |---|---|---|---|
  | Kattegat | 0.53 | 3.8 | 3 |
  | Skagerrak (north IIIa) | 0.82 | 11.2 | 3 |
  | North Sea | 0.56 | 9.0 | 4 |

  I read the column as the standardized lifetime slope alpha-hat = a·SPR(F=0) ("the number of spawners produced per spawner", as in their text and Fig. 1 axis). The OCR of the symbol is ambiguous, so this identification is mine. "There is no evidence that mortality increases at low population size, i.e. depensation". [Myers, Mertz & Fowlow 1997, Fish. Bull. 95:762](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1997/954/myers.pdf)
- **Myers 2001.** "The maximum annual reproductive rate is relatively constant for all species examined: between 1 and 7 replacement spawners are produced per spawner per year at low population size". There is "a 20-fold variation in carrying capacity per unit area for cod". "The spatial scale of recruitment correlations for marine species is approximately 500 km". (Abstract via OpenAlex.) [Myers 2001, ICES JMS 58:937](https://doi.org/10.1006/jmsc.2001.1109)
- **FishLife v2** (Thorson 2020 data-integrated life-history model). I extracted species predictions from the package's `FishBase_and_RAM.rda` (GitHub, commit 2019-11-08) with the `rdata` Python reader. Columns h, r, G (generation time), exp(M) and exp(tm), mean (SD):

  | Species | h | r (/yr) | G (yr) | M (/yr) | tm (yr) |
  |---|---|---|---|---|---|
  | Cod | 0.79 (0.03) | 0.35 | 9.2 | 0.27 | 2.5 |
  | Haddock | 0.66 (0.06) | 0.30 | 7.3 | 0.33 | 2.3 |
  | Whiting | 0.71 (0.08) | 0.40 | 6.1 | 0.43 | 2.4 |
  | Saithe | 0.79 (0.06) | 0.27 | 12.2 | 0.23 | 5.4 |
  | Pollack | 0.76 (0.15) | 0.43 | | | |
  | Ling | 0.53 (0.20) | | | | |
  | Hake | 0.56 (0.19) | | | | |
  | Gadidae (family) | 0.73 (0.09) | 0.43 | 5.9 | | |

  (Read as data plus abstract; the paper PDF in the NOAA repository returned 403.) [Thorson 2020, Fish Fish 21:237](https://doi.org/10.1111/faf.12427); [FishLife data](https://github.com/James-Thorson-NOAA/FishLife/tree/main/data)
- **Rindorf et al. 2022**, 80 NE Atlantic stocks, density-dependent recruitment in 68 % of them (78 % excluding trending pelagics).
  - Their steepness is **R(0.2·max observed SSB)/max R**, not relative to unfished S0. The median is 0.79 among stocks with density dependence.
  - Significant overcompensation occurred in 38 %, and demersal/benthic stocks were more likely to show it (0.56) than pelagics (0.15).
  - 14 of 69 stocks showed strong year classes growing slower than average.
  - Per-stock gadoid values are in the supplement, which I did not obtain.
  - (Fetched full text from [Archimer](https://archimer.ifremer.fr/doc/00753/86468/91824.pdf); Wiley and NOAA copies 403.) [Rindorf et al. 2022, Fish Fish 23:812](https://doi.org/10.1111/faf.12650)
- **Mackinson & Daskalov 2007** (via the prior note, full text): "the compensatory response of cod and herring would be expected to be lower than that of haddock, whiting and saithe". [Cefas TR 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- **Rose et al. 2001.** Compensation "remains one of the most controversial issues in population dynamics", and they advocate process-level understanding. (Abstract only.) [Rose et al. 2001, Fish Fish 2:293](https://doi.org/10.1046/j.1467-2960.2001.00056.x)
- **Lorenzen & Camp 2019.** Density-dependent mortality ceases at roughly 10-20 % of average maximum size, which is how they define "recruited". (Secondary only, as cited in [UF/IFAS FA222](https://edis.ifas.ufl.edu/publication/FA222/pdf); the paper was not read and the number is UNVERIFIED.) [Lorenzen & Camp 2019, Fish Res 217:5-10](https://doi.org/10.1016/j.fishres.2018.09.024)
- **Lorenzen & Enberg 2002 and Hazlerigg et al. 2012** (via prior notes, abstracts): density-dependent growth in the recruited phase is common. Density-dependent mortality is strong in early juveniles and "virtually ceased prior to maturation".

**Inference by species**
- **Cod:** the clearest evidence of juvenile density-dependent mortality, including Skagerrak. Steepness is high (0.79), but the regional slopes span 3.8 (Kattegat, h ≈ 0.49) to 11.2 (Skagerrak, h ≈ 0.74).
- **Whiting:** juvenile density-dependent mortality is "strong" (Myers & Cadigan), and whiting has the largest within-species cannibalism (1.3).
- **Haddock:** the weakest direct evidence of juvenile density-dependent mortality (absent from Myers & Cadigan's list) and the lowest FishLife h (0.66). Its compensation shows up instead as strong year classes growing slowly, and it is heavily preyed on as a juvenile by whiting and saithe (1.3).
- **Saithe:** no cannibalism and no gadoid predation on it in SMS. h is 0.79, but G is long (12 yr), which spreads compensation over a long generation, so the per-year beta is small (Section 3).
- **Pollack:** Skagerrak juveniles cycle like cod (Fromentin 1997), and the same mechanism is likely but not quantified.

### 1.3 Cannibalism and intra-gadoid predation (the loss the unchecked pair removed)

**Literature**
- **NEA cod.** "Cannibalism is probably the most important and also the most variable cause of natural mortality for age 1–4" cod. "An inverse relationship between mortality induced by cannibalism and capelin abundance was found." (Abstract via OpenAlex.) [Yaragina et al. 2009, Mar Biol Res 5:75](https://doi.org/10.1080/17451000802512739)
- **Baltic cod.** 25-38 % of the 0-group and 11-17 % of the 1-group are removed by adults each year. (Abstract, via the prior note.) [Neuenfeldt & Köster 2000, ICES JMS 57:300](https://doi.org/10.1006/jmsc.2000.0647)
- **Northern cod ages 0-3.** Mortality "increasing non-linearly with [adult] abundance". (Abstract, via the prior note.) [Anderson & Gregory 2000, ICES JMS 57:349](https://academic.oup.com/icesjms/article/57/2/349/620448)
- **SMS model structure.** M2 = N_pred · RA · S / AB, with AB the suitability-weighted available biomass including "other food" (Eq. 4-5). The ration RA is fixed, so prey M2 scales with predator numbers by construction. (Fetched full text, Appendix.) [SMS stock annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)

**Calculation (new, from the SMS 2023 key-run files `summary.out` and `partial_m2.out`, 1974-2022, area 570,000 km2; one-off script `gad_sms_beta.py`, session scratchpad, not retained)**
- **Intra-gadoid predation per pooled biomass** (cod, whiting, haddock and saithe eating each other): mean 0.233 /yr, range 0.100-0.445, at a pooled density of 1.45-5.36 t/km2.
  - Linear fit: rate = 0.0159 + **0.0875**·B (SE 0.0077), r = 0.86. This reproduces the prior note.
  - Log fit: rate = const + **0.240**·ln B (SE 0.021), r = 0.86. So **beta = 0.24 /yr** across the North Sea range, centred near B ≈ 2.7 t/km2.
- **Cod on cod:** mean 0.073 /yr. beta = 0.065 /yr per ln B_cod (SE 0.003, r = 0.95); linear k = 0.169 per t/km2 of cod.
- **Whiting on whiting:** mean 0.119 /yr. beta = 0.070 (SE 0.009, r = 0.75); linear k = 0.096.
- **Haddock and saithe:** no cannibalism. Saithe is not eaten by gadoids in SMS.
- **All-gadoid predation per species biomass, vs ln(pooled density):**
  - cod: mean 0.093, beta 0.074
  - whiting: mean 0.334, beta 0.228
  - haddock: mean 0.487, beta 0.388
  - The juvenile-rich, small species carry the density-dependent loss.
- **Mean biomass shares** of the four SMS gadoids: cod 0.15, whiting 0.27, haddock 0.27, saithe 0.31.

**Inference**
- The SMS slope partly follows from the model structure (fixed ration and predator abundance), so it is not independent empirical proof of k·B. Its size, about 0.06 /yr at B_ref, is still the best available North Sea number for the loss that the unchecked `gadoids_preys_on_gadoids` pair would supply.
- It falls on ages 0-2 and is cut by alternative prey (Yaragina).

---

## 2. Conversion of each finding to beta_M and gamma

**Calculation** (one-off script `gad_calc.py`, session scratchpad, not retained). M1_ref = 0.415. "repl" means gamma = beta/0.415. "add" means beta/(0.415 + term at that density).

| Source / mechanism | Form | beta at 0.69 | beta at 1.4 | beta at 2.7 | gamma at 0.69 (repl / add) | Status |
|---|---|---|---|---|---|---|
| SMS intra-gadoid, linear fit 0.016 + 0.0875B | linear, beta = kB | **0.060** | 0.122 | 0.236 | **0.15 / 0.12** (1.4: 0.29 / 0.22; 2.7: 0.57 / 0.35) | computed from model output |
| SMS intra-gadoid, proportional 0.229/1.92 = 0.119·B | linear | 0.082 | 0.167 | 0.321 | 0.20 / 0.17 (1.4: 0.40; 2.7: 0.77) | computed |
| SMS intra-gadoid, log fit (valid 1.45-5.4 only) | Gompertz | (0.24, extrapolated) | (0.24) | 0.24 | (0.58) | computed; at 0.69 outside the data |
| Fromentin 2001 direct DDM, a median -0.28 (range -0.25 to -0.50) -> pool via beta = -a/T | Gompertz (constant) | 0.06 (T 4.7) to 0.11 (T 2.5); range 0.05-0.20 | same | same | 0.14-0.27 (range 0.13-0.48) | Lit. full text + Inference |
| Fromentin delayed DDM, NE fjords b -0.49 to -0.75 | Gompertz | +0.10-0.16 (T 4.7); +0.20-0.30 (T 2.5) | same | same | +0.25-0.39 | as above; NE fjords only |
| B-H steepness h 0.66-0.79 (FishLife), T_eff 4.0-5.3 yr, depletion d 0.2-0.5 at 0.69 | B-H equivalent | 0.14-0.19 | 0.16-0.19 | 0.18-0.20 (cap 1/T) | 0.33-0.46 | Inference (Section 3) |
| Same, Myers 1997 regional pairs (alpha-hat, r_m) | B-H equivalent | Kattegat 0.14-0.23; North Sea 0.16-0.20; Skagerrak 0.23-0.28 | | | 0.35-0.68 | Inference |
| Pool steady-state cross-check (Section 3) | logistic-recruitment pool | 0.23-0.24 (only if B_ref were equilibrium) | | | 0.54-0.58 | Inference |
| Already explicit in model (food, w = 1) | emergent | 0.055-0.134 | | | (0.13-0.32) | model probe |
| Non-FG predation (SMS gurnard + starry ray) vs ln B | confounded fit | -0.080 | | | -0.19 | computed; see 4.2 |

For the SMS linear term evaluated on its own (no floor), gamma = kB/(a + kB) = 0.79 at 0.69 t/km2. This is the "nearly proportional" property: the term is almost proportional to B, yet small at B_ref.

Notes on the arithmetic:
- **Linear SMS:** beta = kB, so 0.0875 × 0.69 = 0.060; × 1.4 = 0.122; × 2.7 = 0.236. gamma(repl) = 0.060/0.415 = 0.145. gamma(add) = 0.060/(0.415 + 0.016 + 0.060) = 0.123. Per tick, k = 0.0875/1460 = 6.0e-5 per (t/km2).
- **Fromentin:** survival from 0- to 1-group ∝ N0^a, so per-capita juvenile mortality is m - a ln N0 and dM_juv/d ln N0 = -a, per juvenile year. In the pool this becomes -a/T (Section 3).
  - Fromentin's own check: a = -0.5 means a 100-fold rise in N0 gives 100^0.5 = 10-fold N1; a = -0.25 gives 100^0.75 ≈ 32-fold. This matches the paper.
- **The key form contrast (Inference):**
  - The SMS term is linear, so its beta rises with density: about 4× from 0.69 to 2.7 t/km2.
  - Every empirical juvenile estimate (Bjørnstad 1999 "approximately log-linear", Fromentin's Gompertz, Myers & Cadigan's "log of juvenile abundance") and the B-H conversion give a roughly constant beta of 0.06-0.2 /yr.
  - That explains why 0.0875·B looks "nearly proportional" but weak at 0.69. A linear k calibrated at North Sea densities (~2.7) is about 4× weaker at B_ref than a log-linear term with the same beta at 2.7.
  - Because gamma (power/log form) is scale-free and k is not, a gamma-form term is robust to the model's B_ref being ~4× below the IBTS density. A k-form term is not.

---

## 3. Recruitment compensation expressed as a density-dependent loss on an age-less pool

**Inference (explicit derivation)**

1. **Exact equivalence (Euler-Lotka).** In an age-structured population the growth rate r solves Σ l_a f_a e^{-r a} = 1.
   - A uniform extra mortality ΔM on all ages multiplies l_a by e^{-ΔM a}, so it shifts r by exactly -ΔM.
   - A factor c on recruitment survival (or fecundity) shifts r by ≈ ln(c)/T. Here T is the generation time: the mean age of parents weighted by l_a f_a e^{-ra}.
   - So a recruitment-survival factor c(B) is equivalent, in an age-less pool, to a loss **M_eq(B) = -ln c(B) / T**.
2. **Beverton-Holt.** R = αS/(1 + βS). Relative to the density-independent slope α, c(S) = 1/(1 + βS).
   - With steepness h and unfished S0: βS0 = (5h-1)/(1-h) = alpha-hat - 1, where alpha-hat = 4h/(1-h) is the maximum lifetime spawners per spawner.
   - With depletion d = S/S0 (taking S ∝ B in the pool): βS = (alpha-hat - 1)·d.
   - Therefore M_eq(B) = ln(1 + (alpha-hat - 1)d)/T and

     **beta_M = dM_eq/d ln B = (1/T) · (alpha-hat - 1)d / (1 + (alpha-hat - 1)d)**,

     which is bounded above by 1/T. B-H recruitment compensation is therefore Gompertz-like at d > 0.2, with beta nearly constant at about 0.66-0.93 × (1/T).
3. **T consistent with the model.** At low density r ≈ ln(alpha-hat)/T, so T_eff = ln(alpha-hat)/r_max with r_max = 0.51.
   - h = 0.73 (Gadidae) gives alpha-hat = 10.8 and T_eff = 4.67 yr.
   - h = 0.66 gives T 4.0; h = 0.79 gives T 5.3.
   - Myers 1997 pairs give T = ln(alpha-hat)/r_m = 2.5 (Kattegat), 2.95 (Skagerrak), 3.9 (North Sea).
   - Coastal Skagerrak cod mature at 2-3 yr (T 2.5-3). The offshore mixed pool, with saithe and older cod, sits at 4-5 yr. That is why beta ranges are quoted for both. FishLife's G (6-12 yr) describes unfished life histories and would give lower values (0.08-0.15).
4. **Numbers** (h = 0.73, T = 4.67):

   | d | beta_M (/yr) |
   |---|---|
   | 0.2 | 0.142 |
   | 0.35 | 0.166 |
   | 0.5 | 0.178 |
   | 1.0 | 0.194 |

   - Across h 0.49-0.79 and d 0.2-0.5, beta is 0.14-0.22 /yr (gamma 0.33-0.54).
   - Doubling B from 0.69 to 1.4 (d 0.2 -> 0.41) raises beta only from 0.142 to 0.171.
   - If B_ref were the equilibrium under F = 0.12, ln(1 + βS*) = (r_max - F)·T. Then beta = (1 - e^{-0.39 T})/T = 0.25 (T 2.5), 0.20 (T 4), 0.18 (T 4.7), 0.13 (T 7). The implied depletion at h 0.73 is d = 0.53.
5. **Steepness-free cross-check (simple pool).** Let per-biomass production be ρ(B) = ρ0/(1 + bB), with density-independent loss Z = M_tot + F and ρ0 = r_max + M_tot.
   - At steady state ρ(B*) = M_tot + F, and beta = ρ(B*)·bB*/(1 + bB*) = (M_tot + F)(r_max - F)/(r_max + M_tot).
   - With M_tot = 0.415: 0.535 × 0.39/0.925 = 0.226.
   - With M_tot = 0.515 (adding ~0.1 explicit M2 from modelled porpoise, seal and bird pairs): 0.635 × 0.39/1.025 = 0.242.
   - For comparison, a Schaefer pool gives beta = r_max - F = 0.39 at equilibrium. The concave B-H form gives roughly half of that.
   - In general beta = ρ(B_ref) × elasticity. The 1/T above plays the role of ρ(B_ref). With a recruitment-production rate of 0.2-0.4 /yr, beta scales to 0.13-0.33.
6. **What part is missing from the model.**
   - The model's intake density dependence (0.06-0.13 /yr) was matched in the prior report to the literature on density-dependent *growth*, which Rindorf et al. treat as separate from recruitment density dependence.
   - Part of real recruitment compensation is food-mediated juvenile competition, which the pool's shared food partly represents. The other part, shelter-habitat limitation and 1-group-on-0-group effects (Fromentin), has no counterpart.
   - Bounds: gross 0.14-0.24 /yr if nothing is represented; net 0.14-0.24 minus 0.06-0.13 ≈ 0.01-0.18 /yr if all of the explicit density dependence counted. Central ~0.07-0.10.
   - The Fromentin-based estimate (0.06-0.11 /yr direct, more in NE fjords) is a second, independent estimate of the **same** component (0- to 1-group survival is one mechanism inside S-R compensation). It is not to be added to the steepness estimate.
   - Caveat: the alpha-hat -> h mapping assumes B-H, but Myers fitted Ricker, so the regional h values are approximate. Rindorf's h is not relative to S0 and is compared only loosely.

---

## 4. Sign of the other M1 components

### 4.1 M0 (0.20: disease, parasites, senescence, other)

**Literature**
- I found no study demonstrating density-dependent disease or parasite mortality in wild gadoids. The one result from targeted searches is a caged-cod study on *Cryptocotyle lingua* reporting density-dependent *host responses* regulating the parasite, not host mortality. (Search result snippet only, UNVERIFIED.)
- **Eastern Baltic cod liver worm** (*Contracaecum osculatum*). Prevalence and intensity rose from low levels in the 1980s-1990s to 100 % prevalence with mean intensity > 80 per fish, alongside the grey seal recovery from a few hundred to > 40,000. The seal is the final host. (Snippets only.) [J. Helminthol. article on *C. osculatum* in grey seals and cod](https://www.cambridge.org/core/journals/journal-of-helminthology/article/contracaecum-osculatum-and-other-anisakid-nematodes-in-grey-seals-and-cod-in-the-baltic-sea-molecular-and-ecological-links/052C1A4BA04B92BB24B52BBF6D55C138); [CSIC PDF](https://digital.csic.es/bitstream/10261/177212/3/Contracaecum%20osculatum.pdf)
- **SMS and ICES** use a fixed M1 of 0.2 for all four species (convention, prior notes). Atlantis advises that quadratic mortality be "low (or zero)" except for top predators (prior note, full text).
- **Kuparinen & Hutchings 2014:** adult M in NW Atlantic cod rose at *low* abundance (depensation), attributed to seals. (Abstract, via the prior note.) [doi](https://doi.org/10.1098/rsos.140075)

**Calculation (size-mediated channel)**
- Density-dependent late growth in demersal stocks is about -9 to -19 % weight-at-age per doubling (prior note, from Rindorf 2022). That is -3 to -7 % in length (W ∝ L^3).
- With Lorenzen-type M ∝ 1/L, M0 rises 3-7 % per doubling: 0.20 × 0.03-0.07 = 0.006-0.014 per doubling. Dividing by ln 2 gives **beta ≈ +0.01-0.02 /yr**.

**Inference**
- The disease and parasite part shows no evidence of being density dependent. The best-documented gadoid parasite trend is driven by the seal (an external and already explicit FG), not by cod density. Its sign is therefore 0.
- The only defensible positive channel is size-mediated, about +0.01 /yr. It is second order, because the pool has no body size.

### 4.2 Predation by non-modelled predators (0.214: squid, halibut, gurnards, long-rough dab, megrim, rays, monkfish)

**Literature**
- In SMS, prey M2 = N_pred·RA·S/AB with a fixed ration. The prey-density elasticity of per-capita M2 is therefore d ln M2/d ln B_prey = -(the prey's suitability-weighted share of the predator's available food). This is a dilution or saturation effect, depensatory, as for any type II predator near its ration. (Fetched full text, SMS annex Eq. 4-5.)
- Grey gurnard is the dominant age-0 cod and whiting predator in SMS (45 % and 58 % of age-0 M2; sibling note `ices_sms_assessments.md`). Non-FG predation falls almost entirely on ages 0-1.

**Calculation** (new, SMS 2023 key-run files, one-off script `gad_sms_nonfg.py`, session scratchpad, not retained)
- **Non-FG predation per pooled gadoid biomass** (SMS starry ray + grey gurnard, the only SMS predators in the model's non-FG set): mean 0.068 /yr, range 0.003-0.166, 2018-22 mean 0.078.
  - Fit against ln(pooled gadoid density): **beta = -0.080 /yr** (SE 0.013, r = -0.66).
  - The same fit for porpoise + seal (explicit FGs in Mareld) gives -0.065 (r = -0.91). For birds it gives -0.003.
- **Prey-density elasticity of gurnard M2 on age-0 fish:** d ln M2/d ln N0 = -0.46 (SE 0.26) for cod, -0.40 (SE 0.40) for whiting and -0.45 (SE 0.10) for haddock, over 49 years.
- Hake and horse mackerel contribute 0 gadoid M2 in these files.

**Inference**
- The fitted -0.08 is confounded: gurnards increased strongly while gadoids declined, so predator increase and prey decline coincide. The SMS structure also builds in the dilution.
- The direction is nevertheless what theory predicts for saturating predators: per-capita predation from non-modelled predators should fall, not rise, with gadoid density.
- Structurally the magnitude is 0.214 × (gadoid share of those predators' suitable food). For a 5-30 % share that is -0.01 to -0.06 /yr. The confounded SMS fit, about -0.08, is an upper bound on the magnitude.
- Central estimate: **beta ≈ -0.02 /yr (range -0.08 to 0)**. Setting it to 0 is defensible and slightly conservative. A positive value has no support.

---

## 5. Spatial scale of each mechanism

| Mechanism | Scale in the literature | Implication for a 1-km grid (Inference) |
|---|---|---|
| Juvenile within-cohort density-dependent mortality (habitat/shelter, food) | Fjord or station: < 20 km; 0- and 1-group in shallow seagrass and kelp (Fromentin 2000, 2002) | Acts in coastal nurseries, largely **outside** an offshore 60 × 60 km grid. In the pool it is a stock-level effect of spawner biomass on recruitment and is better keyed to the grid- or region-level biomass. A per-cell term gives a learnable signal but misplaces the mechanism. |
| Intercohort / cannibalism | Local overlap of adults and juveniles (Fromentin NE fjords; Anderson & Gregory); NEA and Baltic at the stock scale and modulated by alternative prey | Local. A per-cell term is defensible. |
| Stock-recruitment compensation | Recruitment correlations at ~500 km for marine species (Myers 2001) | Stock-wide (≥ grid). |
| Density-dependent growth (already explicit) | Stock-wide per Rindorf / Lorenzen & Enberg | Per cell, through intake interference and depletion. |
| Disease / parasites | Host-driven by seals (Baltic), regional | No gadoid-density term. |
| Non-modelled predator saturation | Local to regional | No positive term. |

Inference on per-cell application:
- If a density-dependent term is applied per cell and calibrated to a stock-mean rate, the realised biomass-weighted mean loss is k × E_B[B_cell] = k × mean(B) × (1 + CV²), where CV is the spatial coefficient of variation of density.
- With today's near-uniform spread (p50 1.4, p90 1.9), 1 + CV² ≈ 1.1, so the difference is small. A policy that aggregates would face a stronger term, which is the intended learnable cost.

---

## 6. Recommended gamma for gadoids

**Component-weighted estimate at B_ref = 0.69 t/km2 (Inference)**

| Component of M1 (or missing loss) | Value at B_ref (/yr) | beta_M central (range) | Basis |
|---|---|---|---|
| M0 disease/parasites/senescence | 0.20 | +0.01 (0 to +0.02) | no evidence of density-dependent disease; size-mediated channel only (4.1) |
| Non-FG predation | 0.214 | -0.02 (-0.08 to 0) | saturating predators, SMS dilution (4.2) |
| Missing intra-gadoid predation / cannibalism (unchecked pair) | +0.076 (SMS fit, 0.016 + 0.060) if added | +0.06 (0.06 at grid mean to 0.12 at local 1.4 t/km2) | SMS 0.0875·B (1.3, 2) |
| Juvenile / recruitment compensation not covered by food density dependence | — | +0.07 (0.01 to 0.18) | B-H steepness net of explicit food density dependence (3), cross-checked by Fromentin -a/T |
| **Total** | 0.415 (or 0.49 with cannibalism added) | **+0.12; judgement range 0.04-0.25** | sum of centrals = +0.12; see note below |

Why the range is not the sum of the row extremes (Inference):
- Adding the row extremes gives -0.01 to +0.32 /yr. The extremes do not co-occur:
  - The -0.08 for non-FG predation is a confounded upper magnitude (4.2).
  - The 0.12 cannibalism value applies only if the term acts at the local 1.4 t/km2 rather than the grid mean.
  - The 0.18 recruitment value assumes the model's food density dependence covers none of recruitment compensation.
- The judgement range of 0.04-0.25 drops the extremes and keeps inside the sanity bound below.

Why the cannibalism and recruitment components are added (Inference):
- Stock-recruitment curves are fitted to recruits at age 0 (SMS, quarter 3) or age 1 (ICES cod), with fixed M afterwards. Predation by gadoids on ages 1-2 therefore lies outside any fitted S-R curve, and the SMS intra-gadoid term is largely additive to steepness-based compensation.
- Only the age-0 part overlaps. Cod-on-cod M2 is about 0.09 /yr at age 0 against about 0.12 at age 1 (sibling note `ices_sms_assessments.md`), and whiting is similar. So up to roughly a third of the +0.06 (~0.02 /yr) may be double counted.
- This is in contrast to Fromentin's 0- to 1-group survival, which is the same mechanism as S-R compensation and is used only as a cross-check.

- **Recommended: beta_M ≈ 0.12 /yr, range 0.04-0.25.** Relative to M1(B_ref) = 0.415 that is **gamma ≈ 0.3 (range 0.1-0.6)**. If the cannibalism-equivalent loss is added on top (M1(B_ref) ≈ 0.49), gamma ≈ 0.25 (0.08-0.5).
- **Sanity bound.** If B_ref were the fished equilibrium, total density dependence would be 0.18-0.24 /yr (B-H) and at most 0.39 (Schaefer). Subtracting the explicit food density dependence (~0.09) leaves 0.09-0.30 /yr for M1. The recommendation sits inside that range. This is a cross-check, not a target: B_ref need not be the model's equilibrium.
- **Per tick.** In the power form, M1(B) = 0.000284·(B/B_ref)^0.3 per tick, before any floor.

**Shape (Inference)**
- The empirical juvenile estimates and the B-H conversion all point to a roughly **constant beta (log-linear / power with small gamma)**, not linear k·B.
- Only the cannibalism component has a linear SMS shape, and that shape is partly structural.
- A form that keeps a density-independent floor and is concave works. Example: M1(B) = M_floor + M_DD·(B/B_ref)^p, with p·M_DD = 0.12 /yr. Two choices:
  - M_floor ≈ 0.30 (DI part of M0 plus non-FG predation), M_DD ≈ 0.12, p ≈ 1. This is linear and grows faster at high density, giving k = 0.17 per t/km2, twice the SMS slope.
  - M_floor ≈ 0.18, M_DD ≈ 0.24, p ≈ 0.5. This is closer to the log-linear evidence and is what I would prefer.
- The linear SMS k = 0.0875 alone reproduces only the cannibalism part at B_ref: beta 0.06, gamma 0.15. It gives 0.24 at 2.7 t/km2.

**Should any part be zero? (Literature + Inference)**
- **M0:** zero (density independent) is consistent with the literature. No density-dependent disease mortality was found in gadoids, ICES and SMS keep M1 fixed, and the size channel is tiny.
- **Non-FG predation:** zero or slightly negative. Saturating predators give depensation, and adult M in cod rose at *low* abundance (Kuparinen & Hutchings 2014). No study supports a positive sign.
- **The positive part** belongs to two juvenile processes the pool cannot represent separately:
  - intra-gadoid predation, the unchecked pair
  - juvenile habitat- and cannibalism-limited survival (Fromentin, Myers & Cadigan)
- In all source populations these act on ages 0-2. Putting them on adult-dominated pool biomass is an approximation the pool cannot avoid.
- **Haddock and saithe** contribute least to the within-species positive part: no cannibalism, and haddock was absent from Myers & Cadigan's list. They contribute most as *prey* of the other gadoids (haddock: beta 0.39 from all-gadoid predation in SMS). That keeps the pooled intra-gadoid term relevant even though it is mostly interspecific.

**Confidence (Inference):**
- moderate that gamma > 0 and that the positive part sits in the missing juvenile and cannibalism losses
- low on the value: the range spans T (2.5 vs 5 yr), the unknown depletion, and how much of recruitment compensation the model's food density dependence already covers
- moderate that the shape should be concave rather than linear

---

## 7. Source log

| Source | How read | Used for |
|---|---|---|
| [Fromentin et al. 2001, Ecology 82:567](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2) | **Full text** ([Archimer](https://archimer.ifremer.fr/doc/00000/712/378.pdf)), Tables 1-5 | a, b by fjord; Gompertz form; underestimation; mechanisms |
| [Fromentin et al. 2002, MEPS 241:231](https://doi.org/10.3354/meps241231) | **Full text** ([Archimer](https://archimer.ifremer.fr/doc/00000/10045/7599.pdf)) | model restated; < 20 km scale |
| [Fromentin et al. 2000, ICES JMS 57:330](https://doi.org/10.1006/jmsc.1999.0638) | **Full text** ([Archimer](https://archimer.ifremer.fr/doc/00000/10525/7082.pdf)); OUP PDF 403 | 0-20 km scale; generation 2-3 yr |
| [Fromentin et al. 1997, MEPS 155:209](https://doi.org/10.3354/meps155209); [1998, MEPS 162:265](https://doi.org/10.3354/meps162265) | **Full text** (int-res PDFs) | cod, pollack, whiting cycles; spatial heterogeneity; trends |
| [Bjørnstad et al. 1999, Ecology 80:1278](https://doi.org/10.1890/0012-9658(1999)080[1278:ANTFDD]2.0.CO;2) | Abstract (OpenAlex) | convexity; log-linear survival |
| [Stenseth et al. 1999, Proc R Soc B 266:1645](https://doi.org/10.1098/rspb.1999.0827); [Bjørnstad et al. 1999, PNAS 96:5066](https://doi.org/10.1073/pnas.96.9.5066) | Abstracts; full text **403** (PNAS, PMC browser check, Royal Society), Europe PMC full-text API **500** | intercohort density dependence and cannibalism |
| [Julliard et al. 2001, Ecol Appl 11:540](https://doi.org/10.1890/1051-0761(2001)011[0540:NMAFMI]2.0.CO;2) | Abstract | post-1-yr M not age dependent; density dependence not tested |
| [Myers & Cadigan 1993, CJFAS 50:1576](https://doi.org/10.1139/f93-179) | Abstract only; coefficients not seen | juvenile density-dependent mortality for cod and whiting (not haddock); log form |
| [Myers et al. 1997, Fish Bull 95:762](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1997/954/myers.pdf) | **Full text** (OCR; symbol reading is mine) | alpha-hat and r_m for Kattegat, Skagerrak, North Sea |
| [Myers et al. 1999, CJFAS 56:2404](https://doi.org/10.1139/f99-201) | Abstract | max reproductive rate near-constant |
| [Myers 2001, ICES JMS 58:937](https://doi.org/10.1006/jmsc.2001.1109) | Abstract | 1-7 spawners/spawner/yr; 20× carrying capacity per area; 500 km |
| [Thorson 2020, Fish Fish 21:237](https://doi.org/10.1111/faf.12427) + [FishLife data](https://github.com/James-Thorson-NOAA/FishLife/tree/main/data) | Abstract + **data extracted** (FishBase_and_RAM.rda, commit 2019-11-08); NOAA PDF **403** | h, r, G, M by species |
| [Rindorf et al. 2022, Fish Fish 23:812](https://doi.org/10.1111/faf.12650) | **Full text** (Archimer); Wiley, NOAA **403**; supplement not obtained | recruitment density dependence 68 %; steepness definition; overcompensation |
| [Rose et al. 2001, Fish Fish 2:293](https://doi.org/10.1046/j.1467-2960.2001.00056.x) | Abstract | controversy over compensation |
| [Lorenzen & Camp 2019, Fish Res 217:5](https://doi.org/10.1016/j.fishres.2018.09.024) | **Secondary only** ([UF/IFAS FA222](https://edis.ifas.ufl.edu/publication/FA222/pdf)); UNVERIFIED | density-dependent mortality ends at 10-20 % of max size |
| [Yaragina et al. 2009, Mar Biol Res 5:75](https://doi.org/10.1080/17451000802512739) | Abstract (OpenAlex) | NEA cannibalism; inverse with capelin |
| [Neuenfeldt & Köster 2000](https://doi.org/10.1006/jmsc.2000.0647), [Anderson & Gregory 2000](https://academic.oup.com/icesjms/article/57/2/349/620448), [Kuparinen & Hutchings 2014](https://doi.org/10.1098/rsos.140075) | Abstracts via prior notes | Baltic and northern cod cannibalism; depensatory adult M |
| [SMS stock annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf) | **Full text** (Eq. 4-5, other food) | M2 structure; dilution elasticity |
| [SMS 2023 key run files](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023) (`summary.out`, `partial_m2.out`, `species_names.in`) | **Data; new computations** | beta for intra-gadoid, cod-cod, whiting-whiting, non-FG, gurnard elasticities |
| [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf) | Full text via prior note | relative compensation of cod vs haddock, whiting, saithe |
| *C. osculatum* / Baltic seals ([Cambridge](https://www.cambridge.org/core/journals/journal-of-helminthology/article/contracaecum-osculatum-and-other-anisakid-nematodes-in-grey-seals-and-cod-in-the-baltic-sea-molecular-and-ecological-links/052C1A4BA04B92BB24B52BBF6D55C138), [CSIC](https://digital.csic.es/bitstream/10261/177212/3/Contracaecum%20osculatum.pdf)) | Search snippets only; UNVERIFIED | parasite trend driven by seals |
| Model probe numbers (production per doubling, w = 1) | Prior report `Täthetsberoende och interferens hos torskfiskar.md` | explicit food density dependence 0.055-0.134 |

Failed or blocked fetches:
- PNAS (403), PMC (browser check), Royal Society PDF (403), Wiley pdfdirect (403), OUP ICES JMS PDF (403), NOAA repository PDFs (403)
- Europe PMC fullTextXML (500), Semantic Scholar API (429), Dalhousie Myers slides (HTML, not a PDF)

Not obtained:
- Myers & Cadigan 1993 coefficients
- Rindorf 2022 per-stock supplement
- Stenseth 1999 parameter values
- any Skagerrak offshore (non-coastal) gadoid stock-recruitment fit
