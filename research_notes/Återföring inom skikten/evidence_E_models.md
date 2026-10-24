# E. How comparable models recycle N in the euphotic layer, and the values they use

Grades: [F] full text or source code read; [A] abstract only; [S] search snippet only;
[N] not found. "Derived" means I computed it from values I quote (the arithmetic is shown).
Values come from the files and papers named. Nothing is filled in from memory.

---------------------------------------------------------------------------------
## 1. ERSEM (pmlmodelling/ersem, master branch)

Sources read in full [F]:
- testcases/ersem-15.06/fabm-ersem-15.06-L4-ben-docdyn-iop.yaml
  https://raw.githubusercontent.com/pmlmodelling/ersem/master/testcases/ersem-15.06/fabm-ersem-15.06-L4-ben-docdyn-iop.yaml
- testcases/fabm-ersem-26.02-dvm.yaml (newer setup; compared with the L4 file)
- src/mesozooplankton.F90, src/microzooplankton.F90, src/bacteria_docdyn.F90,
  src/primary_producer.F90, src/shared.F90 (https://raw.githubusercontent.com/pmlmodelling/ersem/master/src/...)
- Note: there is no src/zooplankton.F90 or src/bacteria.F90 split as asked. The files
  are mesozooplankton.F90 (Z4), microzooplankton.F90 (Z5, Z6), bacteria.F90 and
  bacteria_docdyn.F90 (the L4 yaml uses `ersem/bacteria_docdyn`).

### 1a. Z4 mesozooplankton (L4 yaml, ersem/mesozooplankton) [F]
| param | value | yaml comment |
|---|---|---|
| q10 | 2.0 | |
| minfood | 12.0 mg C m-3 | MM constant to perceive food |
| chuc | 36.0 mg C m-3 | MM constant for food uptake |
| sum | 1.0 d-1 (L4 15.06); **2.0 d-1 in fabm-ersem-26.02-dvm.yaml** | max specific uptake. This is the only Z4 difference between the two files. |
| pu | 0.6 | assimilation efficiency |
| pu_ea | 0.5 | fraction of unassimilated prey excreted (not respired) |
| pu_eaR | 0.9 | same for unassimilated detritus |
| pe_R1 | 0.5 | dissolved fraction of excreted/dying matter |
| srs | 0.015 d-1 | rest respiration at reference T |
| sd | 0.05 d-1 | basal mortality |
| sdo | 0.2 d-1 | max O2-limitation mortality |
| qnc | 0.0126 mmol N/mg C (fixed) | qpc 0.000786 |
| Minprey | 300 mg C m-2 | overwintering threshold; repw 0.0025, mort 0.0025 d-1 |
| R1R2 | 1.0 | labile fraction of produced DOC |
| xR1n | 1.0 (code default 0.0) | N transfer to DOM relative to POM; xR1p 1.2 |
| prey affinities | P1 0.15, P2 0.05, P3 0.0, P4 0.15, Z4 0.25, Z5 0.25, Z6 0.05, B1 0.0, R6 (POM) 0.1 | |
| RP target | R8 (large POM) | |

How Z4 routes the flows (mesozooplankton.F90, lines ~340-490) [F]:
- `ineff = 1 - pu`; `ret = ineff * sum(fpreyc*pu_ea)` (egestion/excretion of organic matter);
  `fZIRDc = (ret + rd)*pe_R1` goes to DOM (R1/R2), `fZIRPc = (ret + rd)*(1-pe_R1)` goes to POM (R8);
  `rra = ineff*rug - ret` is activity respiration; `fZIO3c = srs*et*c + rra`.
- N in the organic losses is `(fZIRDc+fZIRPc)*qnc`, split DON/PON by `xR1n`.
- **Z4 has no `stempn` parameter.** It keeps a constant N:C (qnc 0.0126). N above
  `SZIc*qnc` goes straight to NH4:
  `excess_n = max(SZIn - SZIc*qnc, 0)`, `_SET_ODE_(self%id_N4n, excess_n)`. So the N that
  matches respired C (and the excess N in N-rich prey) is released as NH4. If C is in excess,
  the excess C goes to POM.
- Derived split of one unit of ingested prey C (for pu 0.6, pu_ea 0.5): egested 0.4*0.5 =
  **0.20** (half DOM, half POM via pe_R1 0.5); activity-respired 0.4 - 0.20 = **0.20** (its N goes
  to NH4 through the stoichiometry balance when the prey N:C equals the Z4 N:C); **0.60** assimilated
  (minus rest respiration srs, whose N also goes to NH4; mortality sd goes to DOM/POM 50/50).

### 1b. Z5 / Z6 (ersem/microzooplankton) [F]
Z5 is covered elsewhere. Z6 heterotrophic nanoflagellates (L4 yaml):
minfood **12.0** mg C m-3, chuc **28.0** mg C m-3, sum 1.5 d-1, pu 0.4, pu_ea 0.5, pe_R1 0.5,
srs 0.025 d-1, sd 0.05 d-1, sdo 0.3, qnc 0.0167 mmol N/mg C (maximum), qpc 0.001,
**stempn 0.5 d-1**, stempp 0.5 d-1, xR1n 1.0, xR1p 1.2. Prey: B1 0.45, P2 0.15, P3 0.25, Z6 0.15.
RP target R4 (small POM).
For Z5 the yaml gives chuc **32.0** mg C m-3 (minfood 12, sum 1.25, pu 0.5, srs 0.02, stempn 0.5).
Code (microzooplankton.F90 l.376-391): `fZIRIn = ret*qnc + sd*nP`; DON share
`min(1, pe_R1*xR1n)`; excess-N excretion to NH4 `fZINIn = max(0, nP - qnc*cP)*stempn`. So
micro/nano N goes to NH4 only through quota relaxation at 0.5 d-1, unlike Z4's instantaneous
balance.

### 1c. B1 bacteria (ersem/bacteria_docdyn, L4 yaml) [F]
q10 2.0; chdo 0.31; **chn 0.5 mmol N m-3** (the yaml labels it "nitrate limitation", but the code
uses it on NH4); chp 0.1; sd 0.05 d-1; **sum 2.2 d-1**; **pu 0.6** (efficiency at high O2);
puo 0.2 (low O2); **srs 0.1 d-1**; sR1 1.0 d-1; qnc 0.0167 mmol N/mg C (maximum); qpc 0.0019;
**sR1N4 0.0** and sR1N1 0.0 (no direct DON/DOP mineralisation); nRP 3 with sRP1R1 0.01,
sRP2R1 0.0025, sRP3R1 0.001 d-1 (POM R4/R6/R8 to DOM); rR2 0.0075; rR3 0.0025; frR3 0.3.
Code (bacteria_docdyn.F90):
- respiration `rraB1 = rugB1*(1 - pu*eO2 - puo*(1-eO2))` plus `srs*B1c*et`, so the growth
  efficiency at high O2 is 0.6 of uptake, minus rest respiration.
- DON is taken up in proportion to DOC uptake: `fR1B1n = sugB1*R1nP`.
- NH4 exchange (l.424-428): if `qnB1c > qnc`: release `(qnB1c - qnc)*B1c` per day to NH4;
  otherwise uptake `(qnB1c - qnc)*B1c*N4n/(N4n+chn)` per day (B1 takes up NH4 when N-poor).
  So B1 is the DON to NH4 route. With sR1N4 = 0 there is no abiotic shortcut.

### 1d. Phytoplankton NH4 vs NO3 (primary_producer.F90 l.540-566) [F]
- `rumn3 = qun3*N3n*c`, `rumn4 = qun4*N4n*c`, `rumn = rumn3 + rumn4`;
  `fNIPIn = min(rumn, runn + misn)`; the split is **proportional**:
  `fN3PIn = fNIPIn*rumn3/rumn`, `fN4PIn = fNIPIn*rumn4/rumn`.
- **There is no NH4 preference or NO3 inhibition term.** The two N sources are used in
  proportion to affinity x concentration.
- Affinities (m3/mg C/d), L4 yaml, identical in 26.02-dvm: P1 diatoms 0.0025/0.0025;
  P2 nano 0.004/0.004; **P3 pico qun3 0.006, qun4 0.007** (the only group with a higher NH4
  affinity); P4 micro 0.002/0.002 (qun3/qun4).
- P3 other values: sum 2.0 d-1, srs 0.045, pu_ea 0.2, exulim 0.8, pu_ra 0.2, qnlc 0.006,
  xqcn 1.0, xqn 1.05, qurp 0.006, snplux 1.0, sdo 0.055.
- **qnRPIcX is not a yaml parameter.** It is a constant in shared.F90:
  `qnRPIcX = 1.26E-02` mmol N/mg C (Redfield), `qpRPIcX = 7.86E-04`. Max quota = xqn*qnRPIcX.
- Pelagic nitrification (pel_nit): sN4N3 0.5 d-1, chN4n 0.5, q10 2.0.

---------------------------------------------------------------------------------
## 2. Published North Sea / Skagerrak model results on regenerated production

- **Holt et al. 2012, Biogeosciences 9:97-117, doi:10.5194/bg-9-97-2012 (POLCOMS-ERSEM) [F].**
  - Table 3, pelagic N budget (G mol d-1), terms P = pelagic phytoplankton uptake and
    R = pelagic recycling to DIN (Eq. 4: dNi/dt = Ai - P + R + Bi + Fr + Fa):
    Northern shelves (CNS, NNS, Shetland, Irish shelf): ERA40 P 3.78, R 3.27, Bi 0.57;
    CNTRL P 3.71, R 3.18, Bi 0.57. Western/Southern: ERA40 P 4.02, R 3.62, Bi 0.30;
    CNTRL P 3.21, R 2.87, Bi 0.22.
  - Derived: R/P = 3.27/3.78 = 0.86 (northern, ERA40), 0.86 (CNTRL); 3.62/4.02 = 0.90
    (western/southern, ERA40), 0.89 (CNTRL). With benthic efflux, (R+Bi)/P is about 1.0 in the north.
    **Caveat:** R is all pelagic recycling to DIN over the whole water column. It has no NH4/NO3
    split and includes nitrified N. So it is an upper proxy for (1 - f) and not an f-ratio.
  - Table 2 (CNTRL), region 5 Skagerrak/Kattegat: netPP 138 g C m-2 yr-1; winter DIN
    1.0 mol N m-2; "Frac. P. Recyc." 2.3 yr-1; "Frac. B. Recyc." 1.3 yr-1; diffused fraction
    0.7 yr-1; growing season 195 days. Norwegian Trench: 96, 2.4, 0.9, 0.2, 0.1, 208.
    S. North Sea: 129, 0.4, 5.5, 1.2, 2.0, 253. **"Frac. P. Recyc." is a turnover rate**
    (annual pelagic remineralisation / mean total pelagic N stock), not a share of PP.
  - Text: "the winter DIN provides a lower bound on the total uptake. There is excess
    production beyond this initial supply ranging from ~20 % on the Shetland Shelf to ~4 times
    in the Southern North Sea."
- **No model f-ratio for the North Sea or Skagerrak was found** in any of ERSEM, ECOSMO, ECOHAM,
  ERGOM, NORWECOM, HBM, SCOBI or FlexSem [N]. The ECOSMO f-ratios found (0.09, 0.31, 0.42) are
  for the Atlantic stations BATS, NABE and OWS-India, not the North Sea [S].
- Petersen et al. 2017, J Mar Syst, doi:10.1016/j.jmarsys.2017.01.006 (ERGOM in FlexSem,
  Kattegat) [A]: "Microzooplankton was found to play an important role in the pelagic food web
  as mediator of nutrient and energy fluxes"; the system is "more bottom-up than top-down
  controlled". The abstract gives no share numbers. A search snippet says mesozooplankton
  grazing is imposed as a forcing series from observed abundance [S].
- Maar et al. 2011, Ecol Model, doi:10.1016/j.ecolmodel.2011.03.006 (North Sea-Baltic ERGOM
  with micro and meso zooplankton): abstract not retrieved (ScienceDirect 403) [N for numbers].
- Maar PhD thesis (DMU): "mesozooplankton grazing ... removal of 36-70% of the produced
  [algae] within the euphotic zone". This is a field estimate, not a model value, and the
  snippet is garbled [S].
- Share of PP grazed by micro vs meso zooplankton in any North Sea model: [N].
  Bacterial share of remineralisation: [N]. (In ERSEM it can only be read from diagnostics
  such as fB1NIn and fZINIn, which no paper I opened reported.)
- Moll 1998, Pätsch & Kühn 2008 (ECOHAM), Lenhart, Allen et al.: not reached [N].

---------------------------------------------------------------------------------
## 3. Fasham, Ducklow & McKelvie 1990 (FDM90) and North Sea NPZD models with NH4

- **The original FDM90 Table 1 was not read [N].** The paper is J Mar Res 48:591-639,
  doi:10.1357/002224090784984678, free on EliScholar
  (https://elischolar.library.yale.edu/journal_of_marine_research/1981), but the PDF is behind
  a Cloudflare challenge (HTTP 403). epsilon, Omega, eta, mu3 and mu4 of the original are
  therefore NOT given here.
- Second-hand FDM90-type values [S]: a Frontiers in Mar Sci 2022 NPZD "based on Fasham et al.
  (1990)" (doi:10.3389/fmars.2022.990559) lists NH4 inhibition psi 1.5 (mmol N m-3)-1,
  zooplankton assimilation efficiency 0.75 and zooplankton specific excretion 0.1 d-1.
- **FABM gotm/fasham.F90** (https://raw.githubusercontent.com/fabm-model/fabm/master/src/models/gotm/fasham.F90) [F].
  This is the Kühn & Radach (1997) modified FDM90 used for the northern North Sea (FLEX'76),
  **not the original table.** Defaults: vp 1.5 d-1; alpha 0.065; **k1 (NO3) 0.2, k2 (NH4) 0.8
  mmol m-3** (k1 < k2 here; the uptake form is `(c5/K1)/(1 + c5/K1 + c6/K2)` with **no psi
  exponential inhibition**); mu1 0.05 d-1; k5 0.2; gamma (exudation) 0.05; w_p -1 m d-1;
  gmax 1.0 d-1; k3 1.0; **beta (grazing efficiency) 0.625**; **mu2 (max zoo loss) 0.3 d-1**,
  k6 0.2 (saturating loss); **delta (zoo loss to LDON) 0.1; epsi (zoo loss to NH4) 0.7**;
  the remainder (1 - epsi - delta = 0.2) goes to detritus; r1/r2/r3 = 0.55/0.4/0.05; vb 1.2 d-1;
  k4 0.5; **mu3 (bacterial excretion) 0.15 d-1**; **eta (bacterial NH4:LDON uptake ratio) 0.0**;
  **mu4 (detritus breakdown) 0.02 d-1**; w_d -2 m d-1.
  Unassimilated grazing (1 - beta) goes to detritus.
- **ROMS bio_Fennel.in** (Fennel et al. 2006, FDM90-derived;
  https://raw.githubusercontent.com/myroms/roms/main/ROMS/External/bio_Fennel.in) [F]:
  ZooAE_N 0.75; ZooBM (basal metabolism to NH4) 0.1 d-1; ZooER (specific excretion) 0.1 d-1;
  ZooGR 0.6 d-1 (comment default 0.75); ZooMR 0.025 d-1; K_NO3 and K_NH4 are **inverse**
  half-saturations, 2.0 (m3 mmol-1), i.e. 0.5 mmol m-3; NitriR 0.05 d-1; I_thNH4 0.0095;
  D_p5NH4 0.1. fennel.h l.735-739: `inhNH4 = 1/(1 + NH4*K_NH4)`,
  `L_NO3 = K_NO3*NO3*inhNH4/(1 + K_NO3*NO3)`, a hyperbolic NH4 inhibition of NO3 uptake.
- **NORWECOM** [S]: a HYCOM-NORWECOM description lists 11 state variables: nitrate, phosphate,
  silicate, diatoms, flagellates, micro- and meso-zooplankton, N-detritus, P-detritus, biogenic
  silica and O2. No ammonium is listed, so it appears to have a single DIN pool. The rates were
  not read; the guide is Skogen & Søiland 1998 (hdl.handle.net/11250/113623).
- ECOHAM (Pätsch & Kühn 2008), Moll 1998: parameters not found [N].

---------------------------------------------------------------------------------
## 4. NEMURO (ROMS External/nemuro.in and Nonlinear/Biology/nemuro.h) [F]
https://raw.githubusercontent.com/myroms/roms/main/ROMS/External/nemuro.in ,
https://raw.githubusercontent.com/myroms/roms/main/ROMS/Nonlinear/Biology/nemuro.h

- **AlphaZS/ZL/ZP 0.70** (assimilation efficiency); **BetaZS/ZL/ZP 0.30** (growth efficiency).
- Code: egestion `EgeZ = (1 - AlphaZ)*grazing` goes to **PON** (0.30 of ingestion);
  excretion `ExcZ = (AlphaZ - BetaZ)*grazing` goes to **NH4** (0.40 of ingestion); 0.30 goes to growth.
- Mortality goes to **PON**: MorZS0/ZL0/ZP0 58.5e-3 (m3/mmol N)/d at 0 °C, which is quadratic,
  times exp(0.0693*T).
- NH4 vs NO3 (nemuro.h l.385-393): NO3 uptake `~ exp(-PusaiS*NH4)/(KNO3S + NO3)`;
  NH4 uptake `~ 1/(KNH4S + NH4)`. **PusaiS = PusaiL = 1.5 (m3/mmol N)**; KNO3S 1.0, KNO3L 3.0;
  KNH4S 0.1, KNH4L 0.3 mmol N m-3. NH4 is strongly preferred.
- Decomposition at 0 °C (d-1) x exp(0.0693*T): Nit0 0.03 (nitrification), VP2N0 0.10
  (PON to NH4), VP2D0 0.10 (PON to DON), VD2N0 0.20 (DON to NH4).
- Phytoplankton: GammaS/L 0.135 (exudation ratio); MorPS0 58.5e-3, MorPL0 29.0e-3 to PON.

---------------------------------------------------------------------------------
## 5. ECOSMO and ERGOM

### 5a. ECOSMO: FABM operational port (NERSC), not the Daewel & Schrum 2013 paper itself
https://raw.githubusercontent.com/fabm-model/fabm/master/src/models/nersc/ecosmo/ecosmo_operational.F90 [F].
Daewel & Schrum 2013 (J Mar Syst, doi:10.1016/j.jmarsys.2013.03.008) and Schrum et al. 2006:
full text not read [N].
- Defaults: GrZlP 0.80, GrZsP 1.00, GrZlZ 0.50 d-1; Rg 0.5 mmol N m-3; **mZl 0.10, mZs 0.20 d-1**
  (linear in the code: `BioC(16)*microzoo*mic_loss`); **excZl 0.06, excZs 0.08 d-1**;
  **gammaZlp = gammaZsp = gammaZd = 0.75** (assimilation); **reminD 0.003 d-1**; sinkDet 5.0 m d-1;
  **frr 0.4** (fraction of detritus production that goes to DOM); rNH4 0.20, rNO3 0.50 mmol N m-3;
  **psi 3.0 m3/mmol N** (NH4 inhibition, `up_no3 = no3/(rNO3+no3)*exp(-psi*nh4)`);
  nitrification `0.1 d-1 * exp(0.11*T) * O2/(Onitr+O2)`.
- Routing (l.616-668):
  - Excretion `excZ*Z` is **biomass-specific** and goes straight to NH4 (and PO4).
  - Unassimilated ingestion (1 - 0.75), zooplankton mortality and phytoplankton mortality form
    `dxxdet`. Of it, (1 - frr) = 0.6 goes to detritus and frr = 0.4 to DOM.
  - Detritus goes to NH4 at `frem = reminD*(1 + 20*T^2/(13^2 + T^2))`; DOM goes to NH4 at
    `fremDOM = 10*frem`.
  - Derived: frem is about 0.025 d-1 at 10 °C, 0.033 d-1 at 13 °C and 0.045 d-1 at 20 °C;
    DOM turnover is 10 times that.
- Yumruktepe et al. 2022, GMD 15:3901, doi:10.5194/gmd-15-3901-2022, Table 1 [F]: the
  "DS2013 parameter set" (EXP1) has diatom growth 1.3, flagellate growth 1.1 d-1,
  photosynthesis efficiency 0.03 m2 W-1, meso grazing on phytoplankton 0.8, meso on micro 0.5,
  micro on phytoplankton 1.0 d-1. These match the FABM defaults except aa (0.04 in the code).
  The operational sets EXP2 and EXP3 raise the grazing rates to 1.2 / 0.75 / 1.5 d-1.

### 5b. ERGOM: FABM CGT port by H. Radtke (src/models/iow/ergom/ergom_cgt.F90) [F]
https://raw.githubusercontent.com/fabm-model/fabm/master/src/models/iow/ergom/ergom_cgt.F90 .
This port may differ from Neumann 2000.
- r_zoo_graz 0.5 d-1 (Ivlev-type, food_min_zoo 4.108e-6 mol/kg, temp_opt_zoo 20 °C);
  **r_zoo_resp 0.01 d-1 to NH4**; **r_zoo_mort 0.03 d-1 to detritus**. Both act on
  `zoo_eff = zoo^2/zoo_cl` (quadratic, zoo_cl 9.0e-8 mol/kg = 0.09 mmol m-3).
- **The zooplankton gains the full grazing flux.** The `_ADD_SOURCE_(id_t_zoo ...)` line has
  only the grazing terms minus p_zoo_resp_nh4 minus p_zoo_mort_det. There is no egestion or
  assimilation-efficiency term in this version.
- Detritus to NH4: `r_det_rec 0.003 d-1 (0 °C) * exp(0.15*T)` (q10_det_rec 0.15 K-1, about
  Q10 4.5). Derived: about 0.013 d-1 at 10 °C. w_det -4.5 m d-1.
- Nitrification r_nh4_nitrif 0.1 d-1 at 0 °C.

---------------------------------------------------------------------------------
## 6. Cross-model table: where one unit of ingested N goes (derived from the parameters above)

| model / group | to biomass | direct to NH4 | to DOM | to detritus/POM | NH4 pref. of phyto |
|---|---|---|---|---|---|
| ERSEM Z4 | 0.6 (minus srs) | ~0.2 (respired-C N via stoichiometry balance) | 0.1 | 0.1 | none (proportional; P3 qun4/qun3 = 1.17) |
| ERSEM Z5 (pu 0.5) / Z6 (pu 0.4) | quota-dependent | excess N at stempn 0.5 d-1 | half of egesta | half of egesta | as above |
| NEMURO ZS/ZL/ZP | 0.30 | 0.40 | 0 | 0.30 (+ all mortality) | psi 1.5, KNH4 0.1 vs KNO3 1.0 |
| FABM Fasham (Kühn & Radach) | 0.625 of grazing, then loss mu2: 0.7 NH4 / 0.1 LDON / 0.2 det | via loss term | via loss term | 0.375 + 0.2 of loss | none; k1 0.2 < k2 0.8 |
| ECOSMO (FABM operational) | 0.75 | 0 per ingestion (biomass-specific exc 0.06/0.08 d-1) | 0.4 x 0.25 = 0.10 | 0.6 x 0.25 = 0.15 | psi 3.0, rNH4 0.2 vs rNO3 0.5 |
| ERGOM (CGT port) | 1.0 | 0 per ingestion (resp 0.01 d-1, quadratic) | none | mortality 0.03 d-1 | (not checked) |
| Fennel (ROMS) | 0.75 | ZooBM + ZooER 0.1 + 0.1 d-1 | 0 | 0.25 to small detritus | inhNH4 = 1/(1 + 2 NH4) |

Notes on the table:
- ERSEM Z4 split: pu 0.6, pu_ea 0.5, pe_R1 0.5. The NH4 share assumes prey N:C equals Z4 N:C
  (qnc 0.0126); N-rich prey (Z5/Z6/B1, qnc up to 0.0167) gives more NH4.
- ECOSMO: the 0.25 unassimilated part is split 0.4 DOM / 0.6 POM (frr).

---------------------------------------------------------------------------------
## Gaps
- The original FDM90 Table 1 (epsilon, Omega, eta, mu3, mu4, k1, k2): blocked [N].
- A model f-ratio, micro vs meso grazing share, or bacterial remineralisation share for the
  North Sea, Skagerrak or Kattegat: none found [N]. The closest number is Holt et al. 2012
  R/P ~ 0.86-0.90 (whole water column, upper proxy).
- ECOHAM, Moll 1998, NORWECOM rates, Maar et al. 2011/2016/2018, HBM, SCOBI: not reached.
