# Norway pout and blue whiting in `planktivorous_fish`: productivity, mortality, fishing, diet, behaviour

Evidence note for the planktivorous_fish split (`../split_brief.md`). The group is herring ~79 %, blue whiting ~10 %,
Norway pout ~7.5 %, sprat ~3 % (annual-mean biomass). This note covers Norway pout (*Trisopterus esmarkii*, stock
nop.27.3a4) and blue whiting (*Micromesistius poutassou*, stock whb.27.1-91214). Sprat is only referred to
`../../Den pelagiska poolens reglering/productivity.md` and `../../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`.

Tags: [F] full text read or data downloaded and parsed; [A] abstract only; [S] secondary; [D] computed here.
Scripts: `scripts/`; outputs: `tables/`; downloads: `data/` (see `data/README.md`). Written 2026-10-24.

## Bottom line

| Quantity | Norway pout | Blue whiting | Basis |
|---|---|---|---|
| r_max prior (FishBase CMSY) | **0.50** (95 % 0.33-0.75) | **0.47** (0.31-0.70) | FishBase "Prior r" [F] |
| Productivity decline d (Ricker, shared slope) | **1.27** (±1 SE 1.02-1.59); period variants 1.11-1.33 | **1.13** (0.87-1.46); variants 0.36-1.91 | SG 2026, keys 26011 / 25908 [D] |
| r_now, T = 3/r_max | **0.46** (0.42-0.50; 0.28-0.75 with prior range) | **0.45** (0.41-0.47, capped at r_max) | r_max (1 - ln d / 3) [D] |
| r_now, T = real generation time | **0.38** (0.27-0.49), T = 2 yr | **0.45** (0.39-0.47), T = 4.8 yr | r_max - ln d / T [D] |
| Realised SP/TSB 2016-2024 | 0.14 (median 0.00); 1984-2000: 0.70 | 0.19 (median 0.14) | TSB change + catch over TSB [D] |
| M in the ICES assessment | SMS 2023 key-run M by year, quarter, age (no spawning mortality). Annual, 2011-22: age 1 1.87, age 2 1.69, age 3 1.48; 2022 values (used for 2023-26) 2.26 / 2.12 / 1.88 | 0.2, all ages, fixed | WGNSSK 2026, whb advice 2026 [F] |
| M1 / M2 in SMS (biomass-weighted) | M1 0.20; M2 1.89 (2016-22), 1.96 (2011-22) | not an SMS species | SMS 2023 [D] |
| **M1 for the model** (M0 + non-model predators), /yr | **0.3** (0.2-0.55) | **0.15** (0.1-0.25); bounded by the stock's total M 0.2 | section 2 [D, judgement] |
| F, stock level (catch / mean TSB) | 2016-2025 mean **0.16**; 2023 0.11, 2024 0.04, 2025 0.02 | 2016-2025 mean **0.18**; 2025 0.21 | [D] |
| ICES Fbar | F1-2: 2016-25 mean 0.49; 2025 0.039; 2026 (Q4 2025-Q3 2026) 0.0165 | F3-7: 2016-25 mean 0.46; 2025 0.51; FMSY 0.32 | [F] |
| Local catch, 3.a.20 | Official 2022 805 t, 2023 888 t, 2024 13 t, 2025 < 3 t (3.a). EU FDI incl. discards 2022-24: 130-180 t/yr, > 98 % bottom otter (shrimp-trawl discards) | Official 2022 16 t, 2023 6 t (+43 t NO "3.a"), 2024 0-79 t. EU FDI incl. discards 2022-24: 33-78 t/yr, 100 % bottom otter | ICES nominal catches, FDI [D] |
| Directed fishery in the Skagerrak today | **No** (zero advice 2025-2027; bycatch-only TAC) | **No** (99 % of the stock catch is pelagic trawl far outside the Skagerrak) | [F] |
| Diet: zooplankton / benthos / fish | **0.65 / 0.35 / < 0.03** (range zoo 0.3-0.8) | **0.80 / 0.10 / 0.10** (fish 0-0.4, rising with size) | section 4 [D, judgement] |
| Q/B (/yr) | **4-5** (Ecopath 5.05; Albert 1995 1.1 % BW/d in July) | **6-9** (Bachiller 2018 ~6 zooplankton only; Ecopath 9.06) | [F] |
| dark_ratio (dark/day attack) | **1.0** (biology 0.5-1.5) | **1.0** (biology ~1-1.5 in the Skagerrak, i.e. night-biased) | section 5; the engine treats dark_ratio >= 1 as no light dependence |
| Feeding season | Year-round, dips in Jan, Mar-Apr (spawning) and Aug | Apr-Sep (Norwegian Sea); in the Skagerrak a Q3 feeding visitor | [S]/[F] |
| Storage | Income feeder, modest condition cycle (fat 4-5 %) | Capital-type: liver fat 2 % (Apr-May) -> 9 % (Aug), spent on spawning | [F] |

Main caveats, in order of weight:
1. **Pout r_max.** The CMSY prior (0.50) looks low. The Froese rule r ≈ 3/T with T = 2 yr gives 1.5. In 1984-2000,
   under F ~1.5, the realised SP/TSB was ~0.7/yr. Read as a Schaefer surplus, that needs r > 0.7. The 0.7 leans high,
   though: the TSB is a mean of start-of-quarter values that includes the Q3 0-group pulse. A defensible pout r_max is
   0.5 (prior) to ~1.0. productivity.md met the same gap for sprat (prior 0.48 vs 3/T ~1-1.5). It kept the prior for
   consistency with the library and flagged the gap. The decline d (1.1-1.3) is small next to this uncertainty.
2. **Pout spawning death.** ICES now uses SMS predation M and leaves spawning mortality out. Sparholt et al. (2002b) estimated
   non-predation M of ~2/yr at age 2 and ~3/yr at age 3, mostly between Q1 and Q2. Weighted by biomass, that is
   0.32-0.36/yr. In the model it belongs in M1.
3. **Skagerrak pout is not a closed population.** Larvae drift in, and adults leave to spawn in the northern North
   Sea (Poulsen 1968, via WKBWNG WD). Blue whiting is present only in Q3. Both are fed by recruitment from outside the box.
4. **Ecopath parse warning.** The parse in `Litteraturbaserad growth rate per FG/ecopath_ns1991/` (used by
   `scripts/planktivores_ecopath.py`) misassigns diet columns for pout and blue whiting. It gives pout 77 % benthos,
   with "large crabs" 36 % of the diet and crabs causing 0.65/yr of pout M2. The clean EcoBase XML (model 457) gives pout zooplankton 0.76
   / benthos 0.24, blue whiting 0.74 / 0.15 / fish 0.11, and no crab predation on pout. Use
   `tables/ecobase457_pout_whb_stdout.txt`.

---

## 1. Productivity (r_max, d, r_now, surplus production)

### 1.1 r_max priors [F]
- Norway pout, FishBase summary: "Resilience ... Medium, minimum population doubling time 1.4 - 4.4 years (K=0.36;
  tm=2.3; tmax=5; Fec=27,000). Rate of population increase Prior r = 0.50, 95% CL = 0.33 - 0.75, Based on 3 full stock
  assessments." Lm 14.0 cm, max reported age 5 yr. (`data/fishbase/nop_summary.html`)
- Blue whiting: "Generation time 4.8 (3.1 - 5.0) years ... Resilience ... Medium, minimum population doubling time 1.4 -
  4.4 years (K=0.18-0.22; tm=1-5; tmax=20; Fec=6000). Rate of population increase Prior r = 0.47, 95% CL = 0.31 - 0.70,
  Based on 9 full stock assessments." (`data/fishbase/whb_summary.html`)

### 1.2 Data and lags [F]
- nop.27.3a4, SAG key 26011 (assessment Oct 2026; quarterly SESAM/SAM). Advice Fig. 1: "Spawning-stock biomass (SSB) is
  estimated at the beginning of quarter 4, and the recruitment is for quarter 3." Advice text: "SSB is calculated at
  the beginning of quarter 4 as a proxy for SSB at spawning time (quarter 1 in the following year)." So R(y) is set against
  SSB(y-1) (lag 1). The 2026 benchmark rescaled the stock: "Spawning-stock biomass (SSB) and fishing mortality have been
  substantially rescaled because of the new natural mortality estimates."
- whb.27.1-91214, SAG key 25908 (assessment Sep 2026, SAM). Recruitment at age 1, so lag 1. R and SSB for 2026 excluded.

### 1.3 Decline d [D] (`scripts/ricker_d.py`, `tables/ricker_d_stdout.txt`)

| Stock | Early | Recent | median R/SSB early / recent | raw d | Ricker d (±1 SE) | median SSB early / recent (t) |
|---|---|---|---|---|---|---|
| Norway pout (lag 1) | 1985-2000 | 2010-2025 | 1334 / 1165 | 1.14 | **1.27** (1.02-1.59) | 54 942 / 46 349 |
| pout | 1985-2002 | 2003-2025 | 1306 / 1224 | 1.07 | 1.31 (1.07-1.61) | 59 100 / 42 099 |
| pout | 1985-2009 | 2010-2025 | 1300 / 1165 | 1.12 | 1.11 (0.91-1.36) | 42 560 / 46 349 |
| pout | 1985-2000 | 2016-2025 | 1334 / 1165 | 1.14 | 1.20 (0.95-1.52) | 54 942 / 43 285 |
| pout, lag 0 (check) | 1984-2000 | 2010-2025 | 1364 / 788 | 1.73 | 1.33 (1.05-1.69) | |
| Blue whiting (lag 1) | 1982-2005 | 2006-2025 | 5.46 / 4.81 | 1.14 | **1.13** (0.87-1.46) | 2.38 / 4.14 Mt |
| whb | 1982-1995 | 2006-2025 | 3.88 / 4.81 | 0.81 | 0.36 (0.25-0.52) | 2.02 / 4.14 Mt |
| whb | 1996-2005 (boom) | 2006-2025 | 10.0 / 4.81 | 2.09 | 1.91 (1.47-2.48) | 4.37 / 4.14 Mt |
| whb | 1982-2005 | 2011-2025 | 5.46 / 6.43 | 0.85 | 0.87 (0.68-1.10) | 2.38 / 4.09 Mt |

Reading [D]:
- Pout shows a small decline (1.1-1.3). The residual SD of ln(R/S) is 0.6. Recruitment is "highly variable" (advice), and
  2023-2026 recruitment is "persistently low" (advice). The last 4 years pull d up a little.
- Blue whiting shows no long-term decline. The 1996-2005 recruitment boom makes any window that includes it look
  "early-high". Against the pre-boom years, recruitment per spawner is higher today. Central d 1.13; capping d at 1 is
  equally defensible.

### 1.4 r_now [D]
r_now = r_max (1 - ln d / 3) (T = 3/r_max, productivity.md convention), and r_max - ln d / T with the real generation time.
- **Pout.** T = 3/0.50 = 6 yr. That is three times the real generation time: about 20 % of fish mature at age 1, the
  rest at age 2, and age 3+ is rare (assessment maturity 0 / 0.2 / 1). Most fish spawn once in Q1 at age 2 and
  most likely die. So T ≈ 2 yr is the real value.
  - T = 6: r_now = 0.50 (1 - ln 1.27 / 3) = **0.46** (d range 0.42-0.50).
  - T = 2: r_now = 0.50 - ln 1.27 / 2 = **0.38** (0.27-0.49).
  - The prior itself is the bigger issue. Froese et al.'s r ≈ 3/T_gen gives 1.5 for T = 2. SP/TSB was 0.70 in
    1984-2000 (below), when the stock was fished at F 1-2. That figure leans high, because the quarterly-mean TSB includes
    the Q3 0-group pulse and a one-year-generation stock is not a clean Schaefer pool. Both still point to r_max above the prior.
  - d is recruits (age 0, Q3) per spawner. It does not contain the post-2001 rise in predation M2 (SMS 1.45 -> 1.9/yr).
    For this model that is correct, because M2 is carried by the predator pairs. Do not add it to d.
- **Blue whiting.** T = 3/0.47 = 6.4 yr, close to FishBase's 4.8 yr. r_now = 0.47 (1 - ln 1.13/3) = **0.45** (0.41 at d
  1.46; capped at 0.47 for d < 1). With T = 4.8: 0.445.

### 1.5 Realised surplus production and stock-level F [D]
- **Pout** (`scripts/nop_tsb.py`, `tables/nop_tsb_stdout.txt`). Quarterly TSB from WGNSSK 2026 Table 10.3.6 ("total-stock
  biomass TSB (t)"); annual mean of the four start-of-quarter values; catch from SG.

| Period | mean TSB (t) | catch / mean TSB | SP / mean TSB, mean (median) |
|---|---|---|---|
| 1984-2000 | 371 000 | 0.59 | 0.70 (0.75) |
| 2001-2010 | 230 000 | 0.17 | 0.18 (0.19) |
| 2011-2024 | 280 000 | 0.17 | 0.23 (0.07) |
| **2016-2024** | 276 000 | **0.18** | **0.14 (0.00)** |
| 2016-2025 | 257 000 | 0.16 | 0.13 (0.01) |
| 2021-2025 | 205 000 | 0.10 | -0.10 (-0.03) |

  Mean TSB fell from 280-490 kt (2016-2021) to 93 kt (2024) and 87 kt (2025), a trend of -10.8 %/yr over 2016-2025. SSB Q4
  2026 is 12 692 t, below Blim 28 865 t. The SMS key run gives the same picture for 2016-2022: catch / mean biomass 0.19
  (`tables/sms_pout_stdout.txt`).
- **Blue whiting** (`scripts/whb_sp.py`, `tables/whb_sp_stdout.txt`).

| Period | catch / TSB | SP / TSB mean (median) | Fbar 3-7 |
|---|---|---|---|
| 1985-2000 | 0.18 | 0.26 (0.21) | 0.36 |
| 2001-2010 | 0.22 | 0.16 (0.12) | 0.42 |
| **2016-2024** | **0.18** | **0.19 (0.14)** | 0.45 |
| 2016-2025 | 0.18 | 0.17 (0.12) | 0.46 |

  2025: TSB 8.58 Mt, catch 1.77 Mt, catch/TSB 0.21, Fbar 0.51. ln-TSB trend +1.6 %/yr (2016-2025). The stock is fished above
  FMSY (0.32) but is roughly stable at a high level.

---

## 2. Natural mortality

### 2.1 Norway pout in the ICES assessment [F]
- 2026 benchmark (WKBWNG 2026) and WGNSSK 2026, section 10.2.4: "variable M at age by quarter for 1984-2022 from the SMS
  2023 key run is used in the 2026 benchmark assessment and in the September 2026 assessment where assuming that
  quarterly M-values are the same in 2023-2026 as the SMS 2022 values". And: "the new benchmark M values do not include
  spawning mortality but only the predation mortality". An M multiplier fitted by AIC was "very close to 1".
- Earlier assessments: a constant "0.4 for all age groups in all seasons" until 2012 (= 1.6/yr). In 2012-2025, constant
  quarterly M at age "taking into consideration spawning mortality and higher M values for older age groups" (WKBWNG
  2026, sect. 2.4). The 2012-2025 values are in image tables and were not extracted.
- Table 10.2.6 (WGNSSK 2026), parsed with `scripts/nop_m_table.py` [D]. The "Age 4" column equals the sum of ages 1-3
  (1984 Q1: 0.2525 + 0.1536 + 0.1532 = 0.5593), so it is ignored.

| M (sum of quarters) | Age 0 (Q3+Q4 only) | Age 1 | Age 2 | Age 3 |
|---|---|---|---|---|
| mean 1984-2000 | 0.95 | 1.33 | 1.13 | 0.97 |
| mean 2001-2022 | 1.15 | 1.80 | 1.61 | 1.41 |
| mean 2011-2022 | 1.17 | 1.87 | 1.69 | 1.48 |
| 2022 (= 2023-2026) | 1.21 | 2.26 | 2.12 | 1.88 |

  The quarterly means for 2011-2022 at ages 1 / 2 / 3 are: Q1 0.32 / 0.24 / 0.24; Q2 0.73 / 0.67 / 0.57; Q3 0.36 / 0.34 / 0.30;
  Q4 0.46 / 0.43 / 0.38. Age 0 is 0.55 in Q3 and 0.62 in Q4. Predation peaks in Q2.

### 2.2 SMS key run 2023: biomass-weighted M1, M2 and predators of pout [D]
`scripts/sms_pout.py`, `scripts/sms_pout_groups.py`; rate = eaten biomass per year / annual mean pout biomass.
SMS M1 is the convention 0.05 per quarter (`natmor1.in`; stock annex: "0.05 for ... Norway pout").

| Period | M1 | M2 | Predator shares of M2 |
|---|---|---|---|
| 1984-2000 | 0.20 | 1.45 | saithe 39 %, whiting 27 %, cod 8 %, hake 6 %, grey gurnard 6 %, mackerel 5 %, haddock 4 %, W horse mackerel 2.5 % |
| 2011-2022 | 0.20 | 1.96 | hake 34 %, saithe 25 %, whiting 19 %, grey gurnard 11 %, mackerel 4 %, cod 3 %, haddock 2.5 % |
| **2016-2022** | 0.20 | **1.89** | hake 32 %, saithe 26 %, whiting 23 %, grey gurnard 9 %, mackerel 3.5 %, haddock 3 %, cod 2.7 %, birds 0.8 %, grey seal 0.3 %, porpoise 0.2 %, starry ray 0.3 % |

By model group (2016-2022): predatory_fish 1.58, benthivorous_fish 0.29, seabirds 0.015, seals 0.006, porpoises 0.003. Every
SMS predator maps to a model group, so SMS does not name a single non-model predator of pout.

Ecopath North Sea 1991 (EcoBase 457, `tables/ecobase457_pout_whb_stdout.txt`) [D]: pout M2 1.46. Of this, predatory fish give
1.12 (horse mackerel 0.33, adult whiting 0.28, juvenile and adult saithe 0.20 + 0.20). Benthivorous fish give 0.23 (juvenile cod 0.09,
haddock 0.03, cod 0.03). Toothed whales give 0.05 and blue whiting 0.05 (within the group). Squid gives 0.012, the only non-model
predator. M0 = P/B(1 - EE) = 0.55. In the repo's 2020-25 ICES stomachs, hake, whiting, saithe, anglerfish and mackerel
contain Norway pout (`../predators/tables/ices_diet_pools_taxa.csv`). The SMS hake share is 2-3 times the Ecopath one.

### 2.3 Pout senescence and spawning death [F]/[A]
- Nielsen et al. 2012 (ICES JMS 69:197, read as full-text web page): "a significant proportion of the NP stock most
  likely dies as a direct or indirect result of spawning". Mortality is highest "from the first to the second quarter".
  In 2005, with the fishery closed, "M_age1 = 1.59", while "M_age2 and M_age3 are higher at 1.81 and 2.11". Q1 total mortality
  rises with the fraction mature (slope 1.63, SE 0.21, Table 2).
- Sparholt et al. 2002b (ICES JMS 59:1276) [A]: non-predation M "0.10 for age 1, 1.74 for age 2, 2.58 for age 3 and
  3.05 for age 4 for the 1977–1981 year classes" and "0.10 for age 1, 2.03 for age 2, 3.04 for age 3 and 4.39 for age 4
  for the 1987–1991 year classes". "the main mortality takes place between the 1st and the 2nd quarter of the year,
  pointing at spawning as the main factor."
- Lambert et al. 2009 (ICES JMS 66:1899) [F, local copy]: "The results indicate spawning mortality"; most spawning in Q1,
  around mid-February. WKBWNG 2026 WD: Norway pout is "most likely a short-lived one-time spawning species". Its
  residual misfit at age 3+ "could be related to an unmodelled variable mortality (e.g. spawning mortality)".
- Biomass-weighted [D] (`tables/nop_m0_weighted_stdout.txt`): the SMS 2011-22 age structure is 40 % age 0, 47 % age 1, 12 % age 2
  and 1 % age 3 by biomass. With Sparholt's ages (age 0 set to 0.1), this gives **0.32-0.36/yr** of non-predation M.

### 2.4 Blue whiting [F]
- Advice 2026, Table 5: "Time invariant natural mortality fixed at 0.2 for all ages, derived in the 1980s from age
  compositions before the targeted fishery started." This is total M for the whole NE Atlantic stock.
- Predators. Ecopath NS 1991 [D] gives M2 1.47 on the (then sparse, 0.08 t/km2) North Sea blue whiting. Of this,
  predatory_fish give 1.10 (horse mackerel 0.43, saithe 0.36, hake 0.13, other large gadoids 0.08, megrim 0.07,
  mackerel 0.02, spurdog 0.01) and cod gives 0.14. **Squid gives 0.21** (not in the model) and toothed whales 0.01.
  M0 = 0.27. In the same model hake eats 33 % blue whiting (`../predators/tables/ecobase457_predator_diets_stdout.txt`).
  In the Barents Sea, blue whiting were "seemingly unimportant as prey of piscivorous fish" (Dolgov et al. 2010) [F].

### 2.5 M1 for the model [D, judgement]
- **Pout: 0.3/yr (0.2-0.55).** Low end: the SMS M1 convention 0.2. Central: Sparholt's biomass-weighted spawning loss
  0.32-0.36 plus about 0.01 from squid. High end: Ecopath's unexplained M0 of 0.55. The predation M2 (~1.9/yr) should
  come from the model's pairs: predatory_fish 1.6, benthivorous_fish 0.3. Blue whiting eating pout is within the group
  and stays out of the budget.
- **Blue whiting: 0.15/yr (0.1-0.25).** The assessment's M = 0.2 is the stock's total natural mortality, all causes, so the
  model M1 should not exceed it by much. Ecopath's squid M2 of 0.21 and unexplained M0 of 0.27 refer to the sparse,
  mostly small 1991 North Sea blue whiting. In the box the fish are 20-30 cm. They are too large for squid and for most
  fish predators, except saithe and hake, which are model predators. Central: an M0 of ~0.1 plus a little non-model
  predation (squid, cetaceans other than porpoise). The upper end of 0.25 allows for squid on the smaller fish.
- Per tick: 0.3/1460 = 2.1e-4.

---

## 3. Fishing

### 3.1 Stock level [F]/[D]
- **Norway pout.** ICES catch 2016-2025: 60 241, 33 940, 36 130, 97 668, 129 333, 71 979, 35 726, 27 356, 3 899, 1 754 t.
  Catch / mean TSB is in 1.5. Fbar 1-2 (Q1-Q4): 0.75, 0.61, 0.66, 0.78, 0.76, 0.57, 0.40, 0.23, 0.095, 0.039 (mean
  0.49). Advice 2026: "there should be zero catch from 01 November 2026 to 31 October 2027". The 2025-2026 TAC "may only be
  fished exclusively for by-catches, i.e. no TAC for targeted Norway pout fishery" (Table 6). Fleet 2025: "Small-meshed
  trawl fleet 100%" (Table 7).
- Stock-level gear split. EU FDI North Sea 2022 is OTM_DEF 9 534 t vs OTB_DEF 4 061 t, about 70 % pelagic map
  (`../../Fiskeridödlighet i Skagerrak idag/catch_by_gear.md`) [S].
- **Blue whiting.** Catch 1.77 Mt in 2025; catch/TSB 0.21; Fbar 3-7 0.51 (FMSY = FPA = 0.32). "99% pelagic trawl"
  (Table 7). North Sea + 3.a catch ("Directed and mixed fisheries in the North Sea (Subarea 4; Division 3.a)", Table 9)
  was 39 509-99 955 t in 2021-2025, 2-6 % of the total, mostly Norwegian and Faroese catches in 27.4.

### 3.2 Local, Skagerrak (3.a.20) [D]
- ICES official nominal catches (`scripts/ices_nominal_3a.py`, `tables/ices_nominal_3a_stdout.txt`), t:
  - Pout 27.3.a.20, 2016-2024: 1000, 116, 160, 1143, 5598, 1944, 805, 888, 13. Almost all Danish.
  - Advice Table 8a (3.a): 2024 13 t, 2025 "<3" t.
  - Blue whiting 27.3.a (20 + NK), 2016-2024: 1268, 366, 51, 44, 121, 112, 16, 49, 79. The 2023-24 values are Norway, 3.a not split.
- EU FDI Table A, landings + discards (`scripts/fdi_gear_species.py`, `tables/fdi_3a20_nop_whb_stdout.txt`):
  - Pout 2016-2019: 380-490 t/yr, of which 93-97 % is in OTB_CRU 32-69 mm (Pandalus trawl) and almost all is discards
    (e.g. 2019: 10 t landed, 483 t discarded).
  - Pout 2020-2021: Danish industrial landings of 1 962 t and 354 t, in OTM_DEF and OTB_DEF 16-31 mm (2020: 1 273 t pelagic-coded,
    688 t bottom-coded).
  - Pout 2022-2024: 181, 131, 136 t, ≥ 98 % OTB, mostly discards.
  - FDI is EU only and gives 1 962 t for Denmark in 2020, against 5 585 t in the ICES nominal data. Treat FDI as a lower bound.
  - Blue whiting 2016-2024: 469, 123, 142, 110, 62, 34, 78, 37, 33 t. Pooled, 66 % is bottom otter (shrimp-trawl discards) and
    34 % pelagic (OTM_SPF, herring/sprat trawls); since 2022 it is 100 % bottom otter.
- **Directed fishery today: none for either species in the Skagerrak.** The pout industrial fishery in 3.a last landed
  noticeable amounts in 2020-2023 and is now closed by the zero advice. What remains in 3.a.20 is about 50-150 t/yr of
  pout and 30-80 t/yr of blue whiting discarded in the Pandalus and Nephrops trawls. All of it goes on the bottom-trawl map.
- Which F to use is the same choice as in `ices_pelagic_f.md`. A box population that stands for the stock should lose the
  stock-level rate: pout ~0.02-0.16 (today near 0.02-0.04), blue whiting ~0.18-0.21. The in-box rate is local catch over
  local biomass. The caller can compute it with the box biomass; it is probably 0.01-0.05.

---

## 4. Diet and consumption

### 4.1 Norway pout
- EcoBase 457, North Sea 1991 (Mackinson & Daskalov 2007) [D from XML]: herbivorous/omnivorous zooplankton (copepods)
  0.476, carnivorous zooplankton (incl. krill) 0.286, infaunal macrobenthos 0.191, small mobile epifauna (swarming
  crustaceans) 0.048. By pool: **zooplankton 0.76, benthic_community 0.24, fish 0**. Q/B 5.05, P/B 2.2, B 1.394 t/km2.
- Albert 1995 (ICES JMS 52:873), coastal bank SW Norway, 140 m, July, 11-20 cm, Table 4 (%W, 563 stomachs) [F, local
  copy]. Calanoids 6.6 + *Calanus finmarchicus* 10.1, hyperiids 0.6 + *Parathemisto* 3.2, euphausiids 3.0, *Pandalus
  borealis* 21.9 (+ Pandalus spp. 3.9, *P. montagui* 0.9), Caridea 11.6, Crangonidae 5.3 + *Crangon allmanni* 5.4,
  polychaetes ~1.9, unidentified Crustacea 12.5, not determined 4.1. Grouped [D]: zooplankton 23.5 %, benthic 59.8 %,
  unassigned 16.8 %, so **zooplankton ~0.28 / benthos ~0.72** of the identified part. "For Norway pout, the contribution
  of copepods and amphipods decreased, while decapods increased, with increasing predator length."
- Albert 1994 (ICES JMS 51:45, Norwegian Deep, 1985-87) [A]: "The diet was highly dominated by euphausiids and the
  feeding was more intense at night than during the day." FishBase diet records from it: zooplankton 59-60 % of the
  diet of adults [F, FishBase table].
- Raitt 1968 (FAO synopsis) [F]: "Planktonic crustaceans formed the bulk of the diet, copepods and euphausiids being of prime
  importance ... juvenile fish tended to have more copepods in the stomachs than the adults. Other food types occurring
  included Sagitta sp., appendicularians, and natant decapods (Raitt and Adams, 1965)".
- Onsrud et al. 2004 (Oslofjord) [S, via gadoid_diel_feeding.md]: pout eat *Meganyctiphanes*. Bromley et al. 1997 [A]: 0-group
  pout "stayed almost entirely on copepods".
- Mapping. Copepods, krill and hyperiids go to zooplankton. *Pandalus*, *Crangon*, mysids, gammarids and polychaetes go
  to benthic_community. *Pandalus borealis* migrates vertically but the brief maps it to benthos.
- **Central for the Skagerrak (Trench slope, 100-300 m): zooplankton 0.65 (0.3-0.8), benthos 0.35 (0.2-0.7), fish < 0.03.**
  This is a judgement: Ecopath and the Norwegian Deep put zooplankton at ~0.6-0.75, and the bank shows how large pout shift to shrimp.
- Consumption. Albert 1995: "For Norway pout, daily wet weight ration was estimated as 1.1% of body weight"
  (Jones' method agreed: 1.1 %). That is ~4/yr if it held all year. July at 140 m is near the annual temperature maximum, so
  the annual figure is probably lower. Ecopath 5.05. **Q/B 4-5/yr.**

### 4.2 Blue whiting
- EcoBase 457 (North Sea 1991) [D]: carnivorous zooplankton (krill etc.) 0.605, copepods 0.131, shrimp 0.101, Norway pout
  0.094, epifaunal macrobenthos 0.048, sprat 0.010, misc filter-feeding fish 0.010. By pool: **zooplankton 0.74, benthos
  0.15, planktivorous fish 0.11**. Q/B 9.06 (also FishBase's only Q/B record, Iceland).
- Bachiller et al. 2018 (PLoS ONE 13:e0190345), Table 2, Norwegian Sea 2005-2010, % of diet [F]: copepods 0-73,
  euphausiids 23-93, amphipods 3-19, fish 0-11. Rough mean [D]: euphausiids ~60, copepods ~25, amphipods ~11, fish ~2.
  Abstract: "blue whiting consume about 6 times their biomass in zooplankton"; "blue whiting maintain a low consumption
  rate from April to September, consuming mainly larger euphausiids."
- Dolgov et al. 2010 (ICES JMS 67:483, Barents Sea) [F, web full text]. Pooled, crustaceans 39 % and fish 60 % of stomach weight;
  krill 28 %, amphipods 7 %, copepods ~1 %. "At about 24 cm ... fish made up 42% and krill 41%"; fish > 50 % above 30 cm.
  "the proportion of fish in the diet was higher in large blue whiting and in the north of the range."
- Post et al. 2021 (MEPS 678:1, Irminger Sea, July, 22-39 cm) [F, local copy]: "The most essential prey groups consisted
  of euphausiids, copepods, amphipods and fish, in that respective order."
- Skagerrak: Degnbol & Munch-Petersen 1985 (ICES CM 1985/H:46): in the Skagerrak "the diet consists mainly of euphausiids"
  [S, via Post et al. 2021].
- **Central for the box (20-30 cm, Q3, Trench below 100 m): zooplankton 0.80 (0.6-0.9), benthos 0.10 (0.05-0.15, pelagic
  and benthic shrimps), fish 0.10 (0-0.4).** Fish prey in the box are mostly pout, juvenile herring and pearlside. Pout and
  herring are within the group. Pearlside is not in the model.
- **Q/B 6-9/yr.** Bachiller's ~6 counts zooplankton only, over a feeding season of April-September. Ecopath's 9.06 is an empirical estimate.

---

## 5. Behaviour: light, depth, season, storage

### 5.1 Diel feeding and dark_ratio
- **Norway pout:**
  - Albert 1995, July, 140 m [F]: small pout (11-15 cm, copepods) fed by day. "Mean total dry weight by day was twice that
    at night (p<0.01)", so dark/day ≈ 0.5. "The larger Norway pout showed no regular diel variation". Large pout had a
    newly-ingested peak at dusk.
  - Albert 1994, Norwegian Deep, krill [A]: "feeding was more intense at night", so dark/day > 1.
  - Onsrud et al. 2004, Oslofjord, krill [S]: "No significant differences in stomach fullness ... between day/night", so 1.
  - Raitt 1968 [F]: higher "average volume per stomach during the hours of daylight".
  - Albert 1995's reading: pout "choose to forage at night whenever they are able to capture their prey". The pattern
    follows the prey: copepods by day, migrating krill at night.
  - **dark_ratio = 1.0 (biology 0.5-1.5); no light threshold needed.**
- **Blue whiting:**
  - Skagerrak (Degnbol & Munch-Petersen 1985, via Post et al. 2021) [S]: "feeding increases during the evening and is at a
    maximum around and after midnight", and blue whiting move "towards the surface at night when feeding".
  - Irminger Sea July [F]: "highest food intake took place from noon until late evening, with minimum feeding occurring in
    the morning".
  - Norwegian Sea (Timokhina 1974, via Post) [S]: fullness peaks "around midnight and 14:00 h".
  - **For the Skagerrak, biology says dark/day ≈ 1-1.5 (0.6-2): night-biased, not a visual daytime feeder like herring
    and sprat.** `lib/world/daylight.py` treats `dark_ratio` >= 1 as "no light dependence", so the value to use is **1.0**.
- **Sprat** (brief, [S]): Cardinale et al. 2003 (Aquat. Living Resour. 16:283), Baltic: "sprat seemed to continue feeding
  during the whole day time". A Bornholm Basin study (MEPS 342:277) found sprat also fed at night. A tentative dark_ratio
  is 0.3 (0.1-0.6). This was not checked in full text.

### 5.2 Vertical habitat
- Pout: "generally 5–10 m above the sea bed, somewhat closer by day"; depth increases with age (Albert 1994) [A]. In the North Sea and
  Skagerrak it lives at 50-300 m, "very few deeper than 300 m" in the Norwegian Trench. "the main aggregations of settled
  fish are distributed around the 150 m contour" (WKBWNG 2026 WD) [F]. Oslofjord: pout approach the krill from below by day
  and are deepest at night, with none in the upper 25 m (Onsrud et al. 2004) [S]. **Demersal-midwater, no strong DVM.**
- Blue whiting: mesopelagic, 200-500 m by day in the open ocean. It does diel vertical migration with the krill and rises at night
  (Skagerrak, Degnbol & Munch-Petersen 1985) [S]. It stays below ~35 m in summer (Post et al. 2021) [F]. In the box it is a
  Q3 visitor below 100 m (caller's survey analysis).
- **Pout in the eastern Skagerrak is not a self-contained population** (WKBWNG 2026 WD, citing Poulsen 1968) [F]: "adults
  migrate out of the Skagerrak and the Kattegat to spawn, because there is no evidence of spawning there ... The main bulk
  drifts as larvae south-eastwards from more western areas to which they return mainly during the latter part of their
  second year of life before becoming mature."

### 5.3 Feeding season and storage
- **Pout**, Raitt 1968 [F]:
  - Adults may have "three periods of low feeding intensity, in January, March-April (the main spawning period) and August".
    Juveniles showed "no marked variations in feeding intensity" from September to April.
  - Condition: "Fish were shown to be losing in condition during the winter but seemed to gain rapidly in condition in the
    spring."
  - Fat 4.2-5.1 %, energy 480-520 kJ/100 g (Torry Advisory Note 38) [F].
  - Spawning in Q1, mid-February to April. After it, weight and length at age drop (Lambert et al. 2009), and most fish die.
  - **Reading: it feeds year-round (12 months, reduced in Jan-Apr) and is an income feeder with a modest storage cycle.**
    `income_growth_kappa`-style growth fits, with feeding_season_months ~10-12.
- **Blue whiting** [F]:
  - Jacobsen et al. ICES CM 2002/O:15: "The fat content in blue whiting is at a minimum in April/May (just above 2%) ...
    The fish rapidly gain fat after spawning and reaches a maximum fat content in August (nearly 9%), which remains high in
    the remainder of the year." The fat "increases mainly as an increasing liver", and before spawning "the fat is converted to
    gonads".
  - Bachiller 2018: steady consumption from April to September.
  - Barents Sea empty stomachs: Q1 42 %, Q2 63 %, Q3 26 %, Q4 43 % (Dolgov et al. 2010).
  - **Reading: it feeds for about 6 months (Apr-Sep) and is a capital-type spawner that stores liver fat.** In the box it is
    present only in the feeding season.

---

## Sources

| Source | Used for | Status |
|---|---|---|
| FishBase species summaries, *T. esmarkii*, *M. poutassou* (fishbase.se, 2026-10-24) | Prior r, generation time, diet records, Q/B record | [F] downloaded |
| ICES SAG / Standard Graphs, nop.27.3a4 key 26011, whb.27.1-91214 key 25908 (assessment 2026) | R, SSB, TSB, catch, F | [F] downloaded, parsed |
| ICES Advice 2026 nop.27.3a4, doi 10.17895/ices.advice.30932249 | Lags, catches by area, TAC history, benchmark notes | [F] |
| ICES Advice 2026 whb.27.1-91214, doi 10.17895/ices.advice.30932450 | M = 0.2, catch by area and fleet, reference points | [F] |
| ICES WGNSSK 2026, ICES Sci. Rep. 8:43, doi 10.17895/ices.pub.32676345, Norway pout section | M table 10.2.6, TSB table 10.3.6, ecology text | [F] |
| ICES WKBWNG 2026, ICES Sci. Rep. 8:27, doi 10.17895/ices.pub.32019630, + Norway pout WDs | M scenarios, spawning mortality, stock structure | [F] |
| WGNSSK 2025 Norway pout section | Old M table (image only, not extracted) | [F], table not extracted |
| WGSAM North Sea SMS key run 2023 (github.com/ices-eg/wg_WGSAM) | Pout M1, M2 by predator, mean biomass | [F] downloaded, parsed |
| EcoBase model 457 (North Sea 1991; Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142), XML from `../benthivores/data/ecobase/` | Diets, Q/B, M2 by predator | [F] parsed |
| ICES Official Nominal Catches 2006-2024 | 3.a.20 catches | [F] |
| STECF FDI 2025 data call, Table A (JRC) | 3.a.20 catch by gear, landings vs discards | [F] range-streamed |
| Nielsen et al. 2012, ICES JMS 69:197, doi 10.1093/icesjms/fss001 | Spawning-stress mortality | [F] (OUP full-text page via WebFetch) |
| Sparholt et al. 2002b, ICES JMS 59:1276, doi 10.1006/jmsc.2002.1305 | Non-predation M by age | [A] |
| Lambert et al. 2009, ICES JMS 66:1899, doi 10.1093/icesjms/fsp153 | Maturity, spawning time, spawning mortality | [F] (copy by another agent) |
| Albert 1994, ICES JMS 51:45, doi 10.1006/jmsc.1994.1005 | Norwegian Deep diet, diel feeding, depth | [A] (local PDF scanned, no text) |
| Albert 1995, ICES JMS 52:873 | Diet %W, diel feeding, daily ration | [F] (copy by another agent) |
| Raitt 1968, FAO Fisheries Synopsis 33 Rev. 1 | Food, feeding season, condition | [F] (copy by another agent) |
| Onsrud et al. 2004, ICES JMS 61:1278 | Pout krill feeding, no diel difference | [S] via `Torskfiskars ljusberoende vid jakt/gadoid_diel_feeding.md` |
| Bromley et al. 1997, ICES JMS 54:846 | 0-group pout on copepods | [A] via search / gadoid note |
| Bachiller et al. 2018, PLoS ONE 13:e0190345 | Blue whiting diet, annual consumption ~6x B | [F] |
| Bachiller et al. 2016, PLoS ONE 11:e0149238 | Blue whiting diet, qualitative | [F] web page (figures not readable) |
| Dolgov et al. 2010, ICES JMS 67:483, doi 10.1093/icesjms/fsp254 | Blue whiting diet by size, seasonal empty stomachs | [F] web full text |
| Post et al. 2021, MEPS 678:1, doi 10.3354/meps13918 | Diel feeding, DVM, Skagerrak citation | [F] (copy by another agent) |
| Degnbol & Munch-Petersen 1985, ICES CM 1985/H:46 | Skagerrak blue whiting night feeding on krill | [S] via Post et al. 2021 |
| Jacobsen et al., ICES CM 2002/O:15 | Blue whiting seasonal fat 2-9 % | [F] (copy by another agent) |
| Murray & Burt, Torry Advisory Note 38 (FAO reprint) | Fat and energy of pout and blue whiting | [F] (copy by another agent) |
| Greenstreet et al. 1997, ICES JMS 54:243 | Guild-level rations only, no species Q/B | [F], not used for numbers |
| Cardinale et al. 2003, Aquat. Living Resour. 16:283; MEPS 342:277 | Sprat diel feeding | [S] search snippets |
| Bergstad 1991, Sarsia 75:269-325 (Norwegian Deep gadoids) | Diets | not obtained (paywall) |
