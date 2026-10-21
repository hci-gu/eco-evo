# Detritus remineralisation and sinking for a mixed-layer detritus pool

Tags: [F] field, [L] lab, [D] derived by me, model = model parameter.
Access: **FT** = full text read, **ABS** = abstract or snippet only, **SRC** = source code or parameter file read.
Context: mixed layer M = 6-17 m, temperature T = 5-14 C.

## Table

| # | Model / source (access) | Quantity | Value | Tag |
|---|---|---|---|---|
| 1 | ERSEM: Butenschön et al. 2016, GMD 9:1293; repo pmlmodelling/ersem, `testcases/ersem-15.06/fabm-ersem-15.06-L4-ben-docdyn-iop.yaml` and `src/bacteria_docdyn.F90` (SRC) | POM sinking speeds | small R4 1.0 m/d, medium R6 5.0 m/d, large R8 10 m/d | model |
| 2 | ERSEM, Blackford 2004 testcase `...Blackford2004-L4-noben-stdbac-stdlight.yaml` (SRC) | POM sinking speeds | R4 0.5, R6 2.0, R8 10 m/d | model |
| 3 | ERSEM Z4 (mesozoo), Z5/Z6 (microzoo), same yaml (SRC) | Where dead and egested matter goes | `pe_R1` = 0.5: half of mortality and excretion goes to labile DOM (R1), half to POM. Mesozooplankton POM goes to R8 (10 m/d), microzooplankton POM to R6 (5 m/d). `pu_ea` 0.5: half of unassimilated prey is excreted, the rest respired | model |
| 4 | ERSEM bacteria, docdyn (SRC) | DOM and POM uptake | Labile DOM turnover at most `sR1` = 1.0 /d, capped by bacterial capacity (`sum` 2.2 /d x B). POM goes to bacteria at most 0.01 /d (R4), 0.0025 (R6), 0.001 (R8). Semi-labile DOC availability 0.0075 | model |
| 5 | ERSEM bacteria, standard (SRC) | DOM uptake and mineralisation | `sR1` 1.0 /d, `sum` 4.0 /d, growth efficiency 0.4. DON to NH4 0.05 /d. POM availability `puRP` 0.01. R2 to R1 0.01 /d | model |
| 6 | ERSEM (SRC) | Temperature | Q10 = 2.0, T_ref = 10 C, `q10**((T-10)/10)` on bacterial uptake and respiration; nitrification 0.5 /d with Q10 2 | model |
| 7 | NEMURO: Kishi et al. 2007, Ecol. Model. 202:12-25; ROMS `External/nemuro.in` (SRC) | Decomposition rates at 0 C | PON to NH4 0.10 /d, PON to DON 0.10 /d, DON to NH4 0.20 /d, nitrification 0.03 /d | model |
| 8 | NEMURO (SRC) | Temperature | all decomposition rates x exp(0.0693 T), so Q10 = 2.0 | model |
| 9 | NEMURO (SRC) | PON sinking speed | 40 m/d | model |
| 10 | NEMURO (SRC) | Fate of zooplankton intake | assimilation 0.70, growth efficiency 0.30. So 0.40 of intake is excreted as NH4 and 0.30 egested as PON [D] | model |
| 11 | Rows 7-8 at 10 C | Effective PON to NH4 rate | 0.4 /d total PON loss; half goes directly, half via DON. Mean time to NH4 about 3.75 d, so k_eff about 0.27 /d [D] | [D] |
| 12 | Fennel et al. 2006 (shelf NPZD); ROMS `External/bio_Fennel.in` + `fennel.h` (SRC) | Remineralisation and sinking | small detritus 0.03 /d, large 0.01 /d; no temperature term (O2 factor only). Sinking: small 0.1 m/d, large 1.0 m/d. Coagulation 0.005 /d. Zooplankton mortality 0.025 /d | model |
| 13 | Fasham, Ducklow & McKelvie 1990, J. Mar. Res. 48:591 (ABS) | Detritus sinking | runs at 1 and 10 m/d. I could not read the breakdown rate (often quoted as 0.05 /d; **unverified**) | model |
| 14 | ECOSMO (Schrum et al. 2006; Daewel & Schrum 2013) | Detritus parameters | **not verified**; I found no accessible parameter table | - |
| 15 | Bendtsen & Richardson 2018, Biogeosciences (BG Discuss. bg-2018-385, FT preprint) | Summer f-ratio, NE North Sea and Skagerrak entrance, July 2016 | Norwegian Trench open water < 0.02. Shelf edge > 0.10 (nitrate flux 0.5-1.5 mmol N m-2 d-1). Shallow (< 50 m) areas: nitrate undetectable, negligible new production | [F] |
| 16 | ICES conference abstract, Kattegat to Gulf of Finland, 1990s; authors not seen (ABS) | New production after the spring bloom | detectable only near river discharge, "never exceeded some percentages" of gross production | [F] |
| 17 | Å17 summer NH4 | Concentration | not found in the literature. Å17 is an SMHI monitoring station: pull NH4 from the SHARK database | - |

## Findings

**1. ERSEM.** ERSEM makes no single remineralisation rate. A fixed half (`pe_R1` = 0.5) of zooplankton mortality and excretion goes straight to labile DOM. Bacteria turn labile DOM over at up to 1 /d at 10 C with Q10 = 2, limited by bacterial biomass. The particulate half goes to fast-sinking POM: 10 m/d for mesozooplankton, 5 m/d for microzooplankton. POM is hydrolysed only slowly, at 0.001-0.01 /d, so in a 6-17 m mixed layer it is effectively export. Nitrogen reaches NH4 through bacterial excess N, plus explicit DON mineralisation of 0.05 /d in the standard bacteria model.

**2. NEMURO.** NEMURO is a single PON pool plus DON, all with Q10 = 2. Total PON loss is 0.2 /d at 0 C (0.4 /d at 10 C), and DON to NH4 is 0.2 /d at 0 C. But PON sinks at 40 m/d, so in a 10 m mixed layer the sinking loss w/M = 4 /d beats remineralisation by 10x. NEMURO PON is essentially export in a shallow mixed layer, and recycling happens through direct NH4 excretion (0.4 of intake).

**3. Shelf NPZD models.** Fennel et al. use slow detritus: 0.01-0.03 /d with no temperature term, and 0.1-1 m/d sinking. Fasham 1990 tested sinking at 1 and 10 m/d. ECOSMO is unverified.

**4. Field.** In the stratified summer, outside fronts, new production is a few percent of production at most: the f-ratio is < 0.02 over the Norwegian Trench in July [F]. So summer phytoplankton in the A17 mixed layer runs almost entirely on N recycled within the layer. This is the strongest constraint: if the model's summer phytoplankton is too low, recycling is too weak. I found no published summer NH4 for A17.

## Recommendation

**Use a single lumped "labile dissolved N" pool.** The literature splits cleanly in two:
- The dissolved fraction recycles in days: ERSEM labile DOM up to 1 /d, NEMURO DON 0.4 /d at 10 C.
- The particulate fraction sinks at 5-40 m/d (mesozooplankton carcasses and egesta at 10-40 m/d) and is remineralised at only 0.001-0.03 /d.

In a 6-17 m mixed layer the second fraction is export whichever model you follow. Do not give the pool a POM-like k and w. Route the split at the source, and make the pool the fast part.

- **Share entering the pool: 0.5 of zooplankton and microzooplankton M1** (ERSEM `pe_R1` = 0.5). The other 0.5 is export, like the faecal pellets. Literature range 0.4-0.7 [D]:
  - The unmodelled-predator part of M1 recycles mainly as direct excretion: NEMURO gives 0.40 of intake as NH4, plus part of the predator's own turnover.
  - The non-predatory part (carcasses, a quarter to a third of copepod mortality per Hirst & Kiørboe 2002) is mostly particulate.
  - For microzooplankton M1 (lysis), 0.5-0.7 is defensible.
- **Effective k: 0.4 /d at T_ref = 10 C, Q10 = 2.0** (both ERSEM and NEMURO use Q10 = 2; ERSEM's reference is 10 C). That gives 0.28 /d at 5 C and 0.53 /d at 14 C [D]. The value lies between NEMURO's DON rate (0.4 /d at 10 C) and ERSEM's cap (1 /d), and near NEMURO's PON-to-NH4 k_eff of 0.27 /d. Plausible range 0.2-1 /d.
- **Effective w: 0.5 m/d** (ERSEM small POM 0.5-1, Fennel small detritus 0.1). Dissolved N does not sink; this small w only stands for the fine particles that the 0.5 share contains. With M = 6-17 m, w/M = 0.03-0.08 /d, so 86-90 % of the pool is recycled inside the mixed layer [D]. Overall, about 45 % of zooplankton M1 returns as dissolved N and about 55 % is exported.
- **Checks:**
  - Units: rates are per day, so divide by the ticks per day for the run's `--tick-length`; the Q10 term uses each layer's monthly temperature.
  - Calibration: a summer f-ratio of a few percent means summer phytoplankton growth should run mostly on the grazing share plus this pool. If summer phytoplankton stays too low, raise the share toward 0.7 before raising k. k mainly sets the lag, about 2-4 d, not the amount recycled.
