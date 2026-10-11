# Gadoid "other mortality" M0 = P/B x (1 - EE) in EwE models of the North Sea, Skagerrak/Kattegat and Baltic

Conventions used throughout:
- Ecopath master equation: P = Y + B*M2 + E + BA + P*(1 - EE), so M0 = P/B x (1 - EE) and, with E = BA = 0, M2 = P/B x EE - F, with F = (landings + discards)/B.
- "Read" = printed in the source. "Computed" = my own arithmetic, always under Inferences, with the formula shown.
- North Sea 1991 area = 570,000 km2. Check: adult cod 91.8 kt / 0.161 t km-2 = 570,186 km2 and juvenile whiting 0.222 x 570,000 = 126.5 kt, both matching the text (Mackinson & Daskalov 2007, sections 13.15-13.16). The report also gives 575,000 km2 once (section 15, benthos); the 1 % difference is immaterial.
- Scripts used for the computations: scratchpad `m0.py` (not kept in repo); all inputs are listed in the tables below so the numbers can be re-derived by hand.

## Q1. North Sea 1991 (Mackinson & Daskalov 2007): P/B, EE, F, B per gadoid group, how they were obtained, M0 per group, weighted M0, and whether M0 is a balancing residual

### Takeaway
The simulator's M0 = 0.392 /yr reproduces exactly as the biomass-weighted M0 over groups 13-21 + 24 + 25 (cod, whiting, haddock, saithe juv+adult, hake, other gadoids large+small; B = 1.9565 t km-2). It is a balancing residual: for every gadoid group except "other gadoids", EE is printed in red (estimated by Ecopath), so M0 is whatever production the diet matrix and catches leave unexplained. Two juvenile groups with very low EE (juvenile haddock EE 0.45, juvenile saithe EE 0.31) supply 66 % of the pooled 0.392; adult cod+whiting+haddock+saithe+hake have a weighted M0 of only 0.18 /yr.

### Cited Findings
Method (read):
- "Production rate (P/B) in Ecopath is assumed to be equal to total mortality Z (Allen, 1971), which can be estimated as Z = F+M2+M0 where ... M0 - natural mortality due to old age, deceases, etc. For commercial species assessed by the ICES working groups, mortality estimates (Z, F, and M) were compiled from stock assessment reports." Non-assessed species: M from Pauly (1980) with Tc = 10 C; "F was estimated from the catches (C) and biomass (B): F = C/B"; multi-species group P/B = biomass-weighted mean of species P/Bs — [Mackinson & Daskalov 2007, section 13.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Cod, whiting, haddock, saithe: "input parameters ... were taken from ICES MSVPA working group (SGMSNS, ICES 2002b) ... Production rates (P/B) were calculated from Z=F+M0+M2, consumption rates (Q/B) from consumption at age in MSVPA" — [Mackinson & Daskalov 2007, section 13.14](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Hake: biomass by Sparholt (1990) catchability method with cod/whiting/haddock/saithe as references; "P/B and Q/B were estimated using empirical formulae in FishBase" — [section 13.19](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Norway pout: "Biomass and P/B=Z were based on MSVPA results (ICES 2002b)"; other gadoids (small): species P/B and Q/B from FishBase empirical formulae, then biomass-weighted — [sections 13.21, 13.23](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Balancing rule: "ensuring that estimates of fishing mortality rates were consistent with best available estimates (this provided justification for maintaining (or changing) the biomass of groups since catches were never adjusted (NB: F=C/B))"; discards of split species "were assigned to the juvenile groups" — [section 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Table 3.3 caption: "Where balanced model values differ from initial values derived from data, the closest estimate (or range) from data is given in brackets. Values in red are estimated by Ecopath." — [Table 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

Table 3.3 values (read; red/estimated status read from the PDF's span colour, colour 0xEE332A; "(x)" = data value before balancing):

| # | Group | B t km-2 | P/B /yr | Q/B /yr | EE | Estimated by Ecopath |
|---|---|---|---|---|---|---|
| 13 | Juvenile cod (0-2, <40 cm) | 0.079 | 1.79 | 5.96 (4.89) | 0.9359 | EE, P/Q |
| 14 | Cod (adult) | 0.161 | 1.19 | 3.5 (2.17) | 0.7498 | EE, P/Q |
| 15 | Juvenile whiting (0-1, <20 cm) | 0.222 | 2.36 | 6.58 | 0.8603 | EE, P/Q |
| 16 | Whiting (adult) | 0.352 | 0.89 | 5.46 | 0.9322 | EE, P/Q |
| 17 | Juvenile haddock (0-1, <20 cm) | 0.284 | 2 (2.54) | 5.39 (4.16) | 0.4532 | EE, P/Q |
| 18 | Haddock (adult) | 0.104 | 1.14 | 4.4 (2.35) | 0.9717 | EE, P/Q |
| 19 | Juvenile saithe (0-3, <40 cm) | 0.281 | 1 | 4.94 | 0.3149 | EE, P/Q |
| 20 | Saithe (adult) | 0.22 (0.19) | 0.95 (0.883) | 3.6 | 0.621 | EE, P/Q |
| 21 | Hake | 0.014 | 0.82 | 2.2 | 0.6422 | EE, P/Q |
| 22 | Blue whiting | 0.08 (0.042) | 2.5 | 9.06 | 0.8484 | EE, P/Q |
| 23 | Norway pout | 1.394 | 2.2 (3.05) | 5.05 | 0.7505 | EE, P/Q |
| 24 | Other gadoids (large) | 0.0486 (0.015) | 1.27 | 3.2 (2.18) | 0.95 (input) | B, P/Q |
| 25 | Other gadoids (small) | 0.1909 (0.038) | 2.3 (2.5) | 6 (3.84) | 0.95 (input) | B, P/Q |

Source: [Mackinson & Daskalov 2007, Table 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf); data-derived inputs in [Table 3.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf) (e.g. Table 3.1 adult saithe B 0.191, P/B 0.883; juvenile haddock P/B 2.54, Q/B 4.16; Norway pout P/B 3.05; other gadoids large/small B 0.015/0.038).

Catches 1991, tonnes (read; total landings + discards): juv cod 0 + 29,658; adult cod 67,431 + 3,366; juv whiting 0 + 13,886; adult whiting 84,018 + 19,416; juv haddock 0 + 35,992; adult haddock 50,046 + 4,243; juv saithe 0 + 31,504; adult saithe 66,861 + 0; hake 2,337 + 662; blue whiting 34,770; Norway pout 155,895; other gadoids large 18,867 + 1,828; small 16,131 — [Table 3.5](https://www.cefas.co.uk/publications/techrep/tech142.pdf). (Gear columns of Table 3.5 do not always sum exactly to the printed total, e.g. adult cod gear sum 67,449 vs total 67,431; I used the printed totals.)

M0 is reported only graphically: Figure 3.9 shows the "Structure of mortality" split into "Natural (M0)", "Predation (M2)" and "Fishery (F)" per aggregated group; no numeric M0 table is printed — [Figure 3.9](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

Explicit evidence that biomass (hence F, hence the residual M0) was tuned for Ecosim behaviour (read): for adult saithe and monkfish "they have high fishing mortality and very little other mortality (ie P/B nearly equals F) ... It was decided that the best solution was to reduce the proportion of Z (P/B) accounted for by F. This was achieved by increasing the biomass of the groups in Ecopath, which resulted in F (C/B) decreasing"; small amounts of saithe were additionally assigned as prey of toothed whales, sharks, saithe, seals and other large gadoids — [section 4.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

Discrepancy (read, not resolved): Table 4.1 (data "from MSVPA, 2005") gives Z adult cod 1.02, whiting 1.13, haddock 1.19, saithe 0.70 and Z juvenile cod 1.27, whiting 1.53, haddock 1.96, saithe 0.25, whereas Table 3.3 uses P/B adult 1.19 / 0.89 / 1.14 / 0.95 and juvenile 1.79 / 2.36 / 2.0 / 1.0 — [Table 4.1 vs Table 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf). Probably different MSVPA vintages (ICES 2002b vs 2005) and different age splits.

Ecosim treatment of M0 (read): "Adjustments were also made to the fraction of unexplained (other) mortality sensitive to changes in feeding time. It was assumed to be 1 for all except planktonic and sessile groups, whose values were set to 0." Juvenile groups "must have a relatively high P/B rate ... or a relatively high EE ... otherwise the user must specify a high (near 1.0) value" for that parameter — [section 4.1 (stage 1)](https://www.cefas.co.uk/publications/techrep/tech142.pdf). So in the fitted Ecosim model fish M0 is not a constant rate; it scales with foraging time.

### Inferences
Per-group decomposition, **computed** from Tables 3.3 + 3.5: F = C/(B x 570,000); M0 = P/B x (1 - EE); M2 = P/B x EE - F (predation incl. cannibalism; assumes BA = E = 0); M = P/B - F.

| Group | B | P/B | EE | C (t) | F | M0 | M2 | M = P/B-F | M0/Z |
|---|---|---|---|---|---|---|---|---|---|
| 13 Cod juv | 0.079 | 1.79 | 0.936 | 29,658 | 0.659 | 0.115 | 1.017 | 1.131 | 0.06 |
| 14 Cod adult | 0.161 | 1.19 | 0.750 | 70,797 | 0.771 | 0.298 | 0.121 | 0.419 | 0.25 |
| 15 Whiting juv | 0.222 | 2.36 | 0.860 | 13,886 | 0.110 | 0.330 | 1.921 | 2.250 | 0.14 |
| 16 Whiting adult | 0.352 | 0.89 | 0.932 | 103,434 | 0.516 | 0.060 | 0.314 | 0.374 | 0.07 |
| 17 Haddock juv | 0.284 | 2.00 | 0.453 | 35,992 | 0.222 | 1.094 | 0.684 | 1.778 | 0.55 |
| 18 Haddock adult | 0.104 | 1.14 | 0.972 | 54,289 | 0.916 | 0.032 | 0.192 | 0.224 | 0.03 |
| 19 Saithe juv | 0.281 | 1.00 | 0.315 | 31,504 | 0.197 | 0.685 | 0.118 | 0.803 | 0.69 |
| 20 Saithe adult | 0.220 | 0.95 | 0.621 | 66,861 | 0.533 | 0.360 | 0.057 | 0.417 | 0.38 |
| 21 Hake | 0.014 | 0.82 | 0.642 | 2,999 | 0.376 | 0.293 | 0.151 | 0.444 | 0.36 |
| 22 Blue whiting | 0.080 | 2.50 | 0.848 | 34,770 | 0.762 | 0.379 | 1.359 | 1.738 | 0.15 |
| 23 Norway pout | 1.394 | 2.20 | 0.751 | 155,895 | 0.196 | 0.549 | 1.455 | 2.004 | 0.25 |
| 24 Other gadoids large | 0.0486 | 1.27 | 0.95* | 20,695 | 0.747 | 0.064 | 0.459 | 0.523 | 0.05 |
| 25 Other gadoids small | 0.1909 | 2.30 | 0.95* | 16,131 | 0.148 | 0.115 | 2.037 | 2.152 | 0.05 |

\* EE was an input (0.95) and B was estimated: M0 = 0.05 x P/B there is an assumption, not a residual.

Biomass-weighted aggregates, **computed** (weights = Table 3.3 B):

| Aggregate | B | P/B | F | M0 | M2 | M = P/B-F |
|---|---|---|---|---|---|---|
| Simulator pool: 13-21 + 24 + 25 | 1.9565 | 1.461 | 0.400 | **0.392** | 0.670 | 1.061 |
| Adults cod, whiting, haddock, saithe + hake (14,16,18,20,21) | 0.851 | 0.992 | 0.615 | **0.183** | 0.193 | 0.377 |
| Adults cod, whiting, haddock, saithe | 0.837 | 0.995 | 0.619 | 0.181 | 0.194 | 0.375 |
| Juveniles cod, whiting, haddock, saithe | 0.866 | 1.749 | 0.225 | 0.676 | 0.848 | 1.524 |
| 13-21 (all stages + hake, no "other gadoids") | 1.717 | 1.373 | 0.418 | 0.432 | 0.523 | 0.955 |
| Pool + Norway pout | 3.3505 | 1.769 | 0.315 | 0.457 | 0.996 | 1.453 |
| All gadoids 13-25 | 3.4305 | 1.786 | 0.326 | 0.455 | 1.005 | 1.460 |

- Reproduction check (computed): the pool value 0.392 equals the simulator's M0 (fg_library.yaml comment: "M1 = M0 (0.392) + predation by non-FG groups (0.171) + by benthos ... (0.121) = 0.683 /yr, North Sea Ecopath 1991"). The M2 split (non-FG 0.17, benthos 0.12, cannibalism 0.217 etc.) is in `reports/Litteraturbaserad growth rate per FG.md` section 3.5 and was not recomputed here.
- Contribution of each group to the pooled 0.392 (computed, B_i x M0_i / B_pool): juv haddock 0.159 (40.5 %), juv saithe 0.098 (25.1 %), adult saithe 0.041 (10.3 %), juv whiting 0.037 (9.6 %), adult cod 0.025 (6.3 %), other gadoids small 0.011 (2.9 %), adult whiting 0.011 (2.8 %), juv cod 0.005 (1.2 %), hake 0.002, adult haddock 0.002, other gadoids large 0.002. Juvenile haddock + juvenile saithe = 0.257 of the 0.392 (66 %).
- Sensitivity (computed): if juvenile haddock and juvenile saithe had EE 0.95 like the "other gadoids" (M0 = 0.05 x P/B = 0.10 and 0.05), the pool M0 would be 0.392 - 0.257 + (0.284 x 0.10 + 0.281 x 0.05)/1.9565 = **0.157 /yr**.
- Saithe counterfactual (computed): with the data biomass 0.19 instead of the balanced 0.22, F = 66,861/(0.19 x 570,000) = 0.617, and the predation flow fixed in tonnes gives M2 = 0.057 x 0.22/0.19 = 0.066; with P/B 0.95, M0 = 0.95 - 0.617 - 0.066 = **0.267** (EE 0.72) instead of 0.360; with the data P/B 0.883 as well, M0 = 0.20. So the section 4.1 biomass revision alone added ~0.09-0.16 /yr to adult-saithe M0. Section 4.1 also added new saithe predators (toothed whales, sharks, cannibalism, seals, large gadoids), which raise M2 and pull M0 the other way by an amount not quantified here; the net effect of the stability patch on M0 is therefore not known, but the biomass part is a modelling choice, not biology.
- Is M0 a balancing residual? Yes, for groups 13-23 (EE estimated). B, P/B (= assessment Z) and catches are fixed, and M0 absorbs everything the diet matrix does not attribute to predators: (i) diet-matrix under-representation of predation on juveniles (juvenile haddock M2 0.68 vs P/B 2.0; juvenile saithe M2 0.12 vs P/B 1.0), (ii) any mismatch between the assessment Z and C/B (e.g. B tuned upward), (iii) MSVPA's own residual natural mortality, which is embedded in Z (see Q2: in MSVPA M0 = 0.1-0.2 by assumption). Only (iii) is "natural mortality due to old age, diseases etc." in the report's sense.
- Juvenile EE that low is unusual in the light of best-practice guidance that most groups should have EE "approaching, but lower than 1" (Heymans et al. 2016, see Q3); juvenile haddock/saithe at 0.45/0.31 are the outliers among the gadoids.

### Gaps
- Red/black status was read from the PDF's text colour for Table 3.3 (unambiguous for gadoid rows). Biomass accumulation (BA) and migration (E) are not listed in Table 3.3; I assumed BA = E = 0. The previous session's diet-matrix parse (`research_notes/Litteraturbaserad growth rate per FG/ecopath_ns1991/`) validated M2 against EE, which supports that assumption, but I did not re-verify it here.
- No numeric M0 table in the report; Figure 3.9 is a bar chart only.
- The MSVPA (ICES 2002b, SGMSNS) per-age M1/M2 values that underlie the 1991 Z were not retrieved, so the MSVPA-assumed residual M for 1991 cannot be compared directly (see Q2 for the 1981 comparison).

## Q2. Other EwE models of the region: M0 or the inputs to compute it

### Takeaway
North Sea 1981 (Christensen 1995) gives Ecopath M0 cod 0.27, whiting 0.34, saithe 0.21, haddock 0.50 (biomass-weighted 0.34 /yr), and shows directly that Ecopath M0 exceeds the MSVPA's assumed non-predation M0 (0.20) because it absorbs MSVPA's "other predation" M1 that the Ecopath predators do not account for, plus discards. Central Baltic 1974 (Tomczak et al. 2013) gives cod M0 0.31 weighted over three stanzas (adult 0.28, small 0.19, juvenile 1.17 with EE 0.058); Harvey et al. 2003 gives a much lower 0.13. I found no Kattegat or Skagerrak EwE model with published gadoid parameters, and could not confirm whether the 2015 ICES North Sea EwE key run rebalanced the 1991 Ecopath base.

### Cited Findings
North Sea 1981 (Christensen 1995, Dana 11: 1-28). Quotations are normalised from OCR artefacts in the DTU PDF text layer (e.g. 'flot' -> 'not', 'ali' -> 'all', 'inciudes' -> 'includes'):
- Table 1 (read; B in g m-2 = t km-2, catches g m-2 y-1; "*" = estimated assuming mass balance): cod catch 0.57, B 0.57, P/B 1.11, Q/B 3.2, EE 0.77; whiting catch 0.37, B 0.66, P/B 0.84, Q/B 3.6, EE 0.60; saithe catch 0.21, B 0.65, P/B 0.58, Q/B 3.3, EE 0.64; haddock catch 0.38, B 0.83, P/B 1.37, Q/B 3.8, EE 0.64; Norway pout catch 0.81, B 1.94, P/B 2.48, Q/B 16.5, EE 0.69. Groups are not split by age — [Christensen 1995, Table 1](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)
- Table 5 "Mortality rates (all annual) based on ECOPATH and MSVPA analyses ... For ECOPATH II models Z = F + BA + M0 + M2; for the MSVPA run Z = F + BA + M0 + M1 + M2. The M2 of ECOPATH II corresponds to M1+M2 of MSVPA." Values (read; BA sign reconstructed so that each row sums to Z, see Inferences): cod Z 1.11, F 1.01, BA -0.23, Ecopath M0 0.27, M2 0.07 | MSVPA M0 0.20, M1 0.07, M2 0.06. Whiting 0.84, 0.56, -0.30, 0.34, 0.24 | 0.20, 0.17, 0.21. Saithe 0.58, 0.31, +0.06, 0.21, 0.00 | 0.20, 0.00, 0.00. Haddock 1.37, 0.46, -0.01, 0.50, 0.43 | 0.20, 0.35, 0.36. Norway pout 2.48, 0.41, +0.48, 0.77, 0.81 | 0.20, 0.57, 0.81 — [Christensen 1995, Table 5](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)
- "The efficiencies reflect that the residual mortalities in the MSVPA run have not been fully accounted for in the ECOPATH II model. The MSVPA only includes predation by the MS species. To compensate for this the MS WG assume an other predation mortality (M1) and a non-predation mortality (M0) for all groups. These mortalities are parameterized based on very limited information, especially M0 which is assumed to take values between 0.1 and 0.2 y-1 ... the ECOPATH II predation mortality, M2 is lower than the summed mortalities, M1 and M2 from the MSVPA indicating that not all of the (assumed) M1 mortality can be accounted for ... This is reflected by the other mortality, M0 values of ECOPATH II being higher than the M0 of MSVPA. The M0 values of the ECOPATH II model also include discards" — [Christensen 1995, p. 15-16](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)
- "As our knowledge of the non-predation mortality is very incomplete, this parameter" [EE] is the one usually estimated; EE "can for most groups be expected to be closer to 1 than to 0" — [Christensen 1995, p. 5](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)
- Christensen also found gross efficiencies "unrealistically high for three of the four gadoid MS species (cod with 35%, whiting 24%, and haddock 36%)", i.e. MSVPA consumption rates too low relative to Z — [Christensen 1995, p. 9](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)

Central Baltic 1974 (Tomczak et al. 2013, PLoS ONE 8: e75439), File S1 Table S1 (read, verified this session from the .docx; B t km-2, P/B and Q/B /yr, catch t km-2 /yr): JuvCod B 0.0947, P/B 1.24, Q/B 12.987, EE 0.058, catch 0; Small Cod 0.531, 0.6, 5.968, 0.676, catch 0.204; AdCod 0.49, 1.04, 3.96, 0.733, catch 0.373; Cod larvae 0.000154, 0.5, 75.89, EE 0 — [Tomczak et al. 2013, Table S1](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0075439.s001)

Baltic 1974 (Harvey et al. 2003, ICES J. Mar. Sci. 60: 939-950), Table 1 (read in a previous session, see `research_notes/Litteraturbaserad growth rate per FG/fish.md`; not re-verified now): juvenile cod B 1.40, P/B 0.45, Q/B 2.71, EE 0.63; adult cod B 0.73, P/B 1.06, Q/B 2.00, EE 0.94; "Values of B, Q/B, and total mortality (=P/B) for sprat, herring, and cod were averaged from quarterly MSVPA estimates for 1974" — [Harvey et al. 2003](https://academic.oup.com/icesjms/article/60/5/939/769206)

North Sea EwE key run (ICES WGSAM):
- WGSAM reviewed "the first Ecopath with Ecosim key run for the North Sea"; WGSAM 2015 has "Annex 6: Report on Key Run for the North Sea Ecopath with Ecosim Ecosystem Model, 1991-2013" (search snippet only) — [ICES WGSAM report (figshare)](https://ices-library.figshare.com/articles/report/Report_of_the_Working_Group_on_Multispecies_Assessment_Methods_WGSAM_/19257944/1)
- The Cefas MSE toolkit is built on "the ICES key-run of EwE for the North Sea", 69 functional groups, hindcast 1991-2013 (search snippet only) — [data.gov.uk record](https://www.data.gov.uk/api/2/rest/harvestobject/a23ef730-4f0a-4d42-9390-746d6eb37d12/html)
- ISIMIP's North Sea EwE entry cites Mackinson & Daskalov 2007 as the model description and parameterisation (search snippet only) — [ISIMIP impact model 40](https://www.isimip.org/impactmodels/details/40/)

### Inferences
- Table 5 column reconstruction (computed): with BA = -0.23 for cod, 1.01 - 0.23 + 0.27 + 0.07 = 1.12 ~ Z 1.11; whiting 0.56 - 0.30 + 0.34 + 0.24 = 0.84; haddock 0.46 - 0.01 + 0.50 + 0.43 = 1.38; saithe 0.31 + 0.06 + 0.21 + 0.00 = 0.58. Consistency with Table 1 EE: 1 - 0.34/0.84 = 0.60 (whiting), 1 - 0.50/1.37 = 0.64 (haddock), 1 - 0.21/0.58 = 0.64 (saithe). The dash printed before the BA values is therefore a minus sign.
- North Sea 1981, biomass-weighted over cod + whiting + saithe + haddock (B 2.71 t km-2), **computed**: Z 1.00, F 0.56, M0 **0.34**, M2 0.21; MSVPA M0 + M1 = 0.36. Ecopath M0 here also contains discard mortality, so it is not directly comparable with the 1991 model, where discards are in the catch (F).
- 1981 insight, applied to 1991 (inference): if P/B = assessment Z = F + M(MSVPA), then Ecopath M0 = Z - F - M2(diet) ~ M0(MSVPA, 0.1-0.2 assumed) + [M1 + M2 of MSVPA - M2 of the Ecopath diet matrix]. The large 1991 juvenile haddock/saithe M0 is most plausibly predation that MSVPA attributed but the Ecopath diet matrix does not, rather than disease/senescence.
- Central Baltic 1974 (Tomczak), **computed** with F = catch/B: juv cod F 0, M0 = 1.24 x 0.942 = 1.168, M2 0.072; small cod F 0.384, M0 0.194, M2 0.021; adult cod F 0.761, M0 0.278, M2 0.001. Biomass-weighted (B 1.116): P/B 0.848, F 0.517, M0 **0.314**, M2 0.017. Adult cod alone is the closest analogue to the North Sea adult value (0.28 vs North Sea adult cod 0.30).
- Baltic 1974 (Harvey), **computed** M0 = P/B x (1 - EE): juvenile cod 0.45 x 0.37 = 0.167, adult cod 1.06 x 0.06 = 0.064; biomass-weighted 0.131. Harvey and Tomczak disagree 4-5x on juvenile P/B for the same system and year (already flagged in the earlier fish.md notes), so the Baltic M0 range is 0.13-0.31.
- Cross-model summary (computed): weighted gadoid/cod M0 = 0.18 (NS 1991 adults), 0.34 (NS 1981, ages pooled, incl. discards), 0.39 (NS 1991 pool incl. juveniles and small gadoids), 0.43 (NS 1991 cod/whiting/haddock/saithe all stages + hake), 0.13-0.31 (Baltic 1974 cod). The MSVPA-assumed non-predation M0 was 0.20 for gadoids (Christensen 1995 Table 5).

### Gaps
- Kattegat / Skagerrak: no EwE model with gadoid parameter tables found (two searches). A Skagerrak 148-guild allometric food web (not Ecopath) exists — [Dryad dataset](https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw6c) — with no EE/M0. Record as gap.
- ICES WGSAM 2015 Annex 6 (North Sea EwE key run 1991-2013) not opened; I could not confirm whether the 1991 Ecopath base was rebalanced (69 groups vs 68 in the 2007 report suggests at least a structural change). Mackinson et al. 2018 (PLoS ONE) not found by search.
- EcoBase (ecobase.ecopath.org): not queried (the interface is a web app; no attempt made). Its North Sea/Baltic entries would mirror the published models above.
- Harvey 2003 values come from the previous session's notes, not re-read now.
- Christensen (1995b) "Ecosystem maturity" (Ecol. Model. 77) is a different paper; not needed.

## Q3. How does a fished state affect P/B and thereby M0 (M = P/B - F)? Literature on M0 as a balancing artefact

### Takeaway
Because P/B is set to the exploited Z and F = C/B is fixed by catches, M = P/B - F is the natural mortality of the fished state, and M0 is the part of M that the diet matrix does not explain. EwE best practice explicitly defines 1 - EE as unexplained mortality and expects EE close to 1 for most groups; the 1981 North Sea model and the 1991 saithe biomass adjustment show that M0 in these models carries MSVPA's assumed residual mortality, unexplained predation, discards (1981) and modeller tuning. In the fitted 1991 Ecosim model fish M0 is made feeding-time dependent, so it is not a constant rate.

### Cited Findings
- "EE is the ecotrophic efficiency of the group, and indicates the proportion of the production or total mortality (where P/B = Z) that is actually explained in the model. 1 - EE is the proportion of the production that remains not explained, i.e. the 'other mortality' rate for i (Christensen et al., 2008). Normally, biomass, P/B and Q/B, in addition to diets and catches, are inputs of the model, while EE is an output." — [Heymans et al. 2016, Ecol. Model. 331: 173-184, section 2](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf)
- "Most other functional groups should have EE values approaching, but lower than 1 if all of their production is utilized within the ecosystem that is modelled"; top predators that are not fished or preyed on have EE 0 — [Heymans et al. 2016](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf)
- Balancing diagnostic: "EE < 1.0 ... As a guideline an EE value near to 1.0 is expected when the main part of production is consumed by predators or taken by the fishery" — [Heymans et al. 2016, section 4](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf)
- PREBAL: pre-balance diagnostics "by Link (2010), who proposes a set of pre-balance (PREBAL) diagnostics" that check biomass, P/B, Q/B etc. across trophic levels before balancing; "PREBAL has not been used extensively" — [Heymans et al. 2016, section 3](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf)
- Heymans et al. also stress that Ecopath balances equally well with very different B x P/B combinations (grey seal example: 6,204 t x 0.114 vs 62,040 t x 0.0114) "but in Ecosim the dynamics will be very different" — [Heymans et al. 2016](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf)
- Icelandic EwE model set EE to 0.95 for most groups lacking biomass, citing Heymans et al. 2016 (search snippet only) — [Icelandic waters EwE, arXiv 1810.00613](https://arxiv.org/pdf/1810.00613)
- North Sea 1991: Ecosim no-fishing runs gave "unrealistic" biomass increases for groups "where fishing mortality accounted for the majority of the explained mortality" (adult saithe, cod, monkfish, other large gadoids, catfish); fixed by raising B so that F = C/B falls — [Mackinson & Daskalov 2007, section 4.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- North Sea 1991 Ecosim: fraction of other mortality sensitive to feeding time = 1 for all groups except plankton and sessile groups — [Mackinson & Daskalov 2007, section 4.1 (stage 1)](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- North Sea 1981: Ecopath M0 > MSVPA M0 because unexplained MSVPA M1 and discards end up in it; MSVPA M0 "assumed to take values between 0.1 and 0.2" — [Christensen 1995](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)

### Inferences
- Fished state and P/B (inference + computed): P/B = Z includes F. For the NS 1991 adult gadoids, F/Z = 0.62 (F 0.62 of P/B 1.00), so the unfished-state analogue M = P/B - F = 0.375, of which M0 = 0.18 (48 %) and M2 = 0.19. For the simulator pool, M = 1.061, of which M0 = 0.392 (37 %), M2 = 0.670. Taking M = P/B - F removes fishing but keeps M0 unchanged; M0 itself is not proportional to F.
- Indirect dependence of M0 on F (inference): with EE estimated, any error in F maps one-for-one into M0 with opposite sign (dM0 = -dF at fixed P/B and fixed predation flows). Underestimating B overestimates F and depresses M0; the biomass revision alone (adult saithe B 0.19 -> 0.22, predation flow held fixed) raised its M0 from ~0.27 to 0.36 (computed in Q1). The same holds if assessment Z (from a different MSVPA vintage, cf. Table 4.1 vs 3.3) is mismatched with the C/B in the model.
- For a simulator that models predation explicitly, the defensible residual is closer to MSVPA's non-predation M0 (0.1-0.2) plus predation by groups the simulator lacks, than to the Ecopath residual. Candidate values (computed above): 0.157 (pool, juvenile haddock/saithe EE set to 0.95), 0.18 (adults only), 0.20 (MSVPA assumption), 0.39 (pool as printed). Which juvenile fraction the pool implicitly represents decides the choice; 66 % of the printed 0.392 sits in two juvenile groups whose low EE best-practice guidance would flag.
- In Ecosim, the 1991 model makes M0 scale with foraging time (a density-dependent term), so a constant-rate M0 in a simulator is a simplification of how the source model itself used it.

### Gaps
- Link (2010) PREBAL (Ecol. Model. 221: 1580-1591) was not opened; cited only through Heymans et al. 2016.
- Christensen & Walters (2004) / Christensen et al. (2008) EwE user guide text on M0 in Ecosim (feeding-time sensitivity formula) not opened.
- I found no paper that quantifies, across EwE models, how much of M0 is a balancing artefact; the evidence here is model-specific (Christensen 1995 Table 5; Mackinson & Daskalov 2007 section 4.1).
- No literature found that relates EwE M0 to the fishing level directly (e.g. M0 re-estimated under different F).
