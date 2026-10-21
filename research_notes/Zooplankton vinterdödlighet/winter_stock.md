# What keeps a winter copepod stock at Å17? What the observations measure, the winter community, and model closures

Companion to `zoo_winter_mortality.md` and `zoo_autumn_closure.md` (same folder) and to `../Sillens vinterföda/winter_food_and_feeding.md` (the SHARK carbon table). Anything those notes already cover is cited, not repeated.

**Tags:** [F] field, [L] lab, [A] abstract or search snippet only, [S] secondary (the citing source is named), [D] derived by me, [SRC] model source code read.

**Conventions:**
- C/WW is 0.084 (low WW) to 0.06 (high WW), the sibling convention. 1 t WW/km2 = 1 g WW/m2.
- A stock in a 0-25 m haul is mg C/m2 = ug C/m3 x 25 / 1000.
- The task's "A17 obs" row (Oct 2.9 ... Mar 4.3) is exactly the C/WW 0.084 column of the herring note's table.

## Table of values

| # | Quantity | Value | Tag | Source (access: FT full text, ABS abstract, data, code) |
|---|---|---|---|---|
| 1 | Å17 zooplankton gear | WP2 (0.25 m2 mouth), 90 um mesh. Vertical haul **25-0 m** in every sample from 2007 to 2023 (one 30-0 m). Water depth 248-360 m, median 349. Volume about 6,250 l, which equals 25 m x 0.25 m2. Formalin preserved, split, analysed at Kristineberg. Method code HC-C-C7 | [D] | SHARK raw data, 2026-10-20 pull (data) |
| 2 | Å17 parameters | Abundance (ind/m3) 2007-2023, n = 5-15 visits/month. Carbon content (ug C/m3) 2019-2023 only, n = 2-5 visits/month. **No wet-weight parameter.** | [D] | SHARK (data) |
| 3 | HELCOM C7 biomass method | biomass = abundance x individual weight factors per taxon and stage (Hernroth 1985). Missing taxa come from size classes. Standard haul 25-0 m "if there is no thermocline" and in the Kattegat; otherwise stratified | [F] | HELCOM mesozooplankton guidelines (FT) |
| 4 | Å17 carbon, Calanus share | Oct 0.29, Nov 0.23, Dec 0.12, **Jan 0.70, Feb 0.71**, Mar 0.64 | [D] | SHARK, 2019-23 (data) |
| 5 | Å17 resident carbon (all minus Calanus and Metridia), ug C/m3 | Oct 6,812; Nov 8,934; Dec 3,075 (4,937 and 1,213); **Jan 710 (81 and 1,338)**; Feb 615 (223-1,296); Mar 5,252 | [D] | SHARK, 2019-23, per-visit values in brackets (data) |
| 6 | Calanus + Metridia carbon, ug C/m3 | Dec 933; **Jan 3,214** (754 and 5,673); Feb 1,863 | [D] | same (data) |
| 7 | Å17 resident copepod abundance, ind/m3, mean (median) | Nov 4,625 (1,771); Dec 4,902 (5,696); **Jan 2,440 (1,926)**; Feb 6,726 (5,907); Mar 18,069 | [D] | SHARK, 2007-23, n = 9, 10, 5, 9, 10 visits (data) |
| 8 | Calanus stages in 0-25 m, ind/m3 | Jan: AD 8.6, C5 about 14, total about 26. Feb: AD 67, C5 about 8, **nauplii 419**. Mar: nauplii 1,741 | [D] | SHARK, 2007-23 (data) |
| 9 | Carbon per resident individual, ug C | Dec 0.97, Jan 0.42, Feb 0.26 (range of 5 Jan-Feb visits: 0.11-0.70) | [D] | SHARK, same visits for carbon and abundance (data) |
| 10 | C. helgolandicus overwintering, inner Oslofjord | below the 57 m sill at 80-150 m, Dec-Apr. Rose with hypoxia from mid-January. "Mortality appeared low in early winter", then increased as the animals shallowed | [F] | Kaartvedt et al. 2021 (FT) |
| 11 | Oithona similis respiration | ln R = -3.59 + 0.114 T (Q10 3.1). **About 1.4 % body C/d at 4 C**, 23 % at 25 C. About 8x below a calanoid of equal weight | [L][A] | Castellani et al. 2005 (ABS) |
| 12 | Model zooplankton resting cost | 22.5 MJ/t/tick ÷ 4500 MJ/t = 0.005/tick = 0.020/d at the annual mean; Q10 2 gives about 0.015/d at 5 C | [SRC][D] | `fg_library.yaml` (code) |
| 13 | Temora longicornis, Helgoland winter | pelagic all winter, all stages present. EPR up to 27 eggs/f/d in Nov-Dec, **<10 eggs/f/d with 20 % spawning in Jan-Mar**; EPR tracks food | [F] | Wesche et al. 2007 (FT) |
| 14 | Acartia clausi, same site | overwinters as females; **spawning zero in Dec-Jan** (intrinsic reproductive dormancy); resumes in late Feb | [F] | same (FT) |
| 15 | Centropages hamatus, same site | absent Oct-Mar; returns in May, probably from resting eggs | [F] | same (FT) |
| 16 | Resting eggs, Kiel Bight sediment | 1.8-7.4 x 10^5 eggs/m2, 4 calanoid species, 49-94 % hatching | [A] | Madhupratap et al. 1996 (ABS) |
| 17 | Pseudocalanus acuspes, Bornholm Basin | female lipid 0.9-1.8 ug; a fatty-acid marker ties the diet to ciliates; opportunistic feeder | [A] | Peters et al. 2006 (ABS) |
| 18 | O. similis winter mortality, Kola Bay (Arctic) | C4-C5 **0.005 ± 0.001 /d in December**. Mortality correlates positively with temperature; winter abundance 129 ind/m3 | [F][A] | Dvoretsky 2012 (ABS) |
| 19 | Pseudocalanus newmani, Dabob Bay | C5 and female mortality **not significantly different** between the overwintering and growth seasons | [F][A] | Ohman & Wood 1996 (ABS) |
| 20 | ERGOM (GOTM version) | dZ/dt = G(P,T)·(Z+Z0) - (l_za+l_zd)·Z·(Z+Z0), with Z0 = 0.0225 mmol N/m3, l_za 0.0667, l_zd 0.1333 m3/(mmol N d). Ivlev-squared (sigmoid) grazing | [SRC] | FABM `gotm/ergom.F90` (code) |
| 21 | ERGOM (MSI) | loss = (nue 0.01 + sigma_b 0.03)·zcl1 50·Z^2 = 2.0 Z^2; grazing on (Z + 0.001) | [SRC] | FABM `msi/ergom1.F90` (code) |
| 22 | ECOSMO (operational) | mortality 0.10/d and excretion 0.06/d switch **off** when mesozooplankton < 0.01 mg C/m3; detritus is a possible food (prefZlD = 0 by default) | [SRC] | FABM `nersc/ecosmo/ecosmo_operational.F90` (code) |

## Q1. What the Å17 number measures

- **Gear and depth [D, row 1].** It is a WP2 90 um vertical haul from 25 m to the surface, over a water depth of about 345 m on the Norwegian Trench slope. The series is never stratified and never deep. The Dec-Jan 1.2 t/km2 is therefore entirely a **0-25 m** stock, and nothing of it sits below 30 m. What lives below 25 m at Å17 was **not measured**.
- **Carbon, not weight.** Carbon is computed, not weighed. HELCOM C7 prescribes abundance x wet-weight factors (row 3). Kristineberg's carbon is presumably abundance x per-stage carbon factors (with lengths measured), but I did not find their factors [D, not verified]. Wet weight was never reported, so the C/WW range above is an assumption.
- **Comparison with the model.** For a mixed winter column, a 0-30 m model compares with about 1.2x the 0-25 m value [D] (Jan temperature at Å17/Å16 is 5.0-7.0 C over 0-30 m, sibling temperature note). The sibling's 1.5-3x applies to the 100 m Mareld column, not to 30 m. The one depth-split haul (Anholt E, 0-50 m = 1.66 x 0-20 m) is in the herring note.
- **The winter number is not a resident stock.**
  - Jan-Feb carbon is about 70 % Calanus C5 and adults, plus about 10 % Metridia (rows 4, 6, 8).
  - These are deep-water animals. Metridia (266 lucens and 12 longa rows, 440 to genus only) is a deep vertical migrator, and the Calanus are C5 ascending or moulting at the end of diapause. Falkenhaug et al. 2022 (herring note) describe Jan adults at Arendal and overwintering in the Norwegian Trench. Kaartvedt et al. 2021 (row 10) show C. helgolandicus holding at 80-150 m and rising from mid-January.
  - A 30 m column at a 100 m site has no source for these animals.
- **The task's "Dec->Jan ~0/d" is a composition artefact [D].**
  - Visit-weighted mean dates are Dec 15 and Jan 12, 28 days apart.
  - The resident (non-Calanus, non-Metridia) carbon falls from 3,075 to 710 ug C/m3, which is **-0.052/d**: the model's rate.
  - Abundance (more visits) gives **-0.023/d** on means and -0.035/d on medians. Individual carbon also falls (row 9), as large autumn adults (Temora, Paracalanus, Centropages) are replaced by Oithona and nauplii.
  - Calanus and Metridia over the same interval rise from 933 to 3,214 ug C/m3, which keeps the total flat.
  - Caveat: January carbon rests on two visits with a 9x spread (836 vs 7,011 ug/m3 total). The high one is a Calanus patch, so the "Jan 1.2" is not a typical January.
- **Resident-stock target for the model [D].** Values are 0-25 m, in t WW/km2, at C/WW 0.084 (then 0.06):

  | Month | From carbon | From abundance x ug C/ind |
  |---|---|---|
  | Dec | 0.92 (1.28) | 1.4-1.6 (2.0-2.3) |
  | Jan | **0.21 (0.30)** | 0.18-0.30 (0.26-0.43) |
  | Feb | 0.18 (0.26) | 0.45-0.51 (0.63-0.72); February nauplii add numbers |
  | Nov | 2.7 | - |
  | Oct | 2.0 | - |

  - Against this target the model (Oct 1.5, Nov 0.75, Dec 0.16, Jan 0.05) is about 1.3x low in Oct, 3.6x in Nov, **6-8x in Dec** and 4-6x in Jan.
  - **The deficit is built in Oct-Dec.** The model's Dec->Jan rate (ln(0.05/0.16)/31 = -0.038/d [D]) lies inside the observed resident range (-0.023 to -0.052/d).
  - Spring: the model's Feb->Mar growth (0.8 to 7.0) runs ahead of the resident stock (Feb 0.2-0.5). The observed March total of 4.3 is 64 % Calanus.

## Q2. Winter species composition (Å17, 0-25 m)

- **By carbon (2019-23, n = 2-3 visits/month) [D]:**
  - Dec: Temora 0.27, Paracalanus 0.14, Centropages 0.12, Calanus 0.12, Metridia 0.11, unidentified Copepoda 0.10, Oncaea 0.04, Pseudocalanus 0.03, Oithona 0.03.
  - Jan: Calanus 0.70, Metridia 0.12, Paracalanus 0.09, Oithona 0.04.
  - Feb: Calanus 0.71, Copepoda 0.11, Metridia 0.04, Pseudocalanus 0.04, Oithona 0.03.
- **By abundance (2007-23, n = 5-10) [D]:**
  - Oithona is the most numerous copepod all winter: 1,046 / 1,934 / 879 / 1,813 ind/m3 in Nov / Dec / Jan / Feb.
  - Next are Paracalanus (814 in Dec, falling to 108 in Feb) and Pseudocalanus (rising from 239 in Dec to 1,142 in Feb and 4,835 in Mar).
  - Temora and Acartia are at 50-270 ind/m3.
  - Pteropods (Euthecosomata) and bivalve larvae are large in Jan numbers, but they are not in the carbon total.
- **Literature.** The Gullmar Fjord winter community (Calanus, Pseudocalanus and Microcalanus deep; Acartia, Temora and Oithona above the sill; a "practically uninhabited" upper layer early in the year) is in Lindahl & Hernroth 1983, sibling autumn note. I did not find another verifiable regional winter series by biomass (Kristineberg/Gullmar searches: not found).

## Q3. How the winter residents persist

- **Metabolism.** Small cyclopoids are cheap: Oithona respires about 1.4 % C/d at 4 C (row 11). The model's winter resting cost (about 0.015/d, row 12) is already at that level.
  - I found **no measured winter respiration reduction beyond temperature** for the non-diapausing small calanoids (Temora, Paracalanus, Oithona): not found.
  - Diapause-level cuts (10-20 % of active respiration, sibling) apply to Calanus at depth, not to the 0-25 m residents.
- **Reproduction stops or slows.** This saves egg costs and removes births.
  - Acartia clausi has zero spawning in Dec-Jan; Temora is below 10 eggs/f/d in Jan-Mar (rows 13-14).
  - At Å17, recruitment restarts in February (row 8: Calanus nauplii 419 ind/m3, and unidentified Copepoda 2,500 ind/m3, mostly nauplii [D]).
- **Food.**
  - Temora feeds and reproduces in proportion to food all winter (row 13). Pseudocalanus relies on ciliates (row 17). Oithona's winter gross growth efficiency is above 100 %, which implies carnivory or faecal pellets (sibling, Castellani et al. 2005b).
  - I found no measured winter ingestion in /d for Skagerrak residents: not found.
- **Lipids.** Pseudocalanus females hold 0.9-1.8 ug lipid (row 17). Temora stores little and depends on continuous food (Wesche et al. 2007 introduction). I found no starvation-survival time for Temora: not found.
- **Resting eggs.** Centropages hamatus (and in part Temora and Acartia) bridge winter in the sediment (rows 15-16). At Å17 (345 m) and Mareld (about 100 m) such a seed is unlikely to matter for a 0-30 m stock (not quantified).

## Q4. How other North Sea and Baltic models keep a winter stock

- **ERSEM** switches to an overwintering state below 300 mg C/m2 of prey, with mortality and respiration 0.0025/d (sibling).
- **ERGOM (GOTM version, row 20)** adds a **background seed Z0** to the grazers and uses quadratic loss. Per-capita loss is 0.2·(Z+Z0)/d.
  - At Z = 0.5 t WW/km2 in a 10 m layer (0.063 mmol N/m3, sibling conversion) this is 0.017/d [D].
  - In a 30 m column, Z0 corresponds to about 0.5 t WW/km2 [D].
- **ERGOM (MSI, row 21)** applies quadratic loss to respiration too.
- **ECOSMO (operational, row 22)** has a hard floor: losses turn off below 0.01 mg C/m3.
- **Already covered (siblings):** NEMURO, PISCES and Fennel (all quadratic), and ERSEM's cannibalism.
- **Not found or not read:** ECOSMO II parameter table (Daewel & Schrum 2013), NORWECOM.E2E Calanus IBM mortality (Hjøllo et al. 2012; PANGAEA supplement not opened), Daewel NPZD-fish, size-structured models.
- **Starvation-dependent mortality.** I found **no** model in which mortality falls with low activity or starvation, other than ERSEM's food-threshold switch.

## Q5. Is winter predation documented to be far lower?

- **Qualitatively, yes.**
  - Gelatinous predators and chaetognaths are gone by December (autumn note §3).
  - Herring is not the lever. The **model's** Dec-Jan herring ration (0.18-0.30 % BW/d) is already *below* the literature's winter ration (0.4-1.0 % BW/d; Lohmeyer 1977, Bernreuther 2007; herring note). Herring cannot explain the model's excess winter loss.
  - Deep Calanus enjoy a predator-free hypoxic refuge (row 10).
- **Quantitatively, weak.**
  - Field winter mortality for small copepods: 0.005/d for Oithona C4-C5 in Arctic December (row 18).
  - Pseudocalanus shows **no** winter drop (row 19).
  - Hirst & Kiørboe predict 0.048-0.062/d at 5 C, with a predation-free floor of 0.019 (sibling).
- **What Å17 itself implies.** The resident net loss is 0.02-0.05/d in Dec-Jan, with near-zero births (Q3), so total resident mortality is about 0.02-0.05/d. The evidence supports a Dec-Feb M1 of **0.02-0.03/d**, not "far below" that, and not near zero.

## Recommendation

**0. Fix the comparison first (strongest evidence, own data).**
- Å17's Dec-Mar totals are **not comparable** to a resident 0-30 m stock: Jan-Mar is 64-71 % Calanus plus Metridia carbon, imported from deep water.
- Calibrate winter against the resident 0-25 m stock (x1.2 for 0-30 m): **Dec about 1-1.6, Jan about 0.2-0.4, Feb about 0.2-0.7 t WW/km2**.
- Either treat the Calanus arrival as external forcing, or ignore it.
- With this target, "~1 t/km2 over winter" is too high for January. The real gap is a 4-8x deficit that opens in **Oct-Dec**.

**1. Winter M1 = non-predatory floor + unmodelled predation that fades with prey density** (sibling closure, re-fitted to the budget).
- Formula: `M1(T,Z) = m_np·2^((T-5)/10) + m_p·Z/(K_Z+Z)`.
- m_np = **0.019/d at 5 C** (Hirst & Kiørboe's lab longevity, Q10 about 1.85-2). m_p = **0.037/d** and K_Z = **3 t WW/km2** [D]. These give exactly 0.054/d at Z_ref = 10 t/km2 and 9.3 C, so the derived growth_rate is unchanged.
- Resulting M1 [D]:

  | Conditions | M1 (/d) |
  |---|---|
  | Dec, Z = 1, 8 C | 0.033 |
  | Jan, Z = 0.3, 5.7 C | 0.023 |
  | Feb, Z = 0.3, 4.5 C | 0.022 |
  | May, Z = 26, 13 C | 0.066 |

- Over 77 days (mid-Nov to end of Jan) about 0.027/d less mortality leaves about **8x** more stock. That matches the observed 4-8x gap and the Å17 resident net loss (Q5).
- K_Z is an areal value: rescale it if the column changes.

**2. Make winter intake able to cover the 0.015/d resting cost on the food actually present.**
- Residents feed on ciliates and dinoflagellates all winter (Temora, Pseudocalanus, Oithona; rows 13 and 17).
- The model's winter microzooplankton (2-5 ug C/l) sits on the flat part of the Type III zooplankton→microzooplankton curve. Its phytoplankton (15-30 ug C/l) is about half of Å17 chl x C:Chl 50 (Dec 0.77 ug/l gives about 40 ug C/l) [D, C:Chl assumed].
- Before adding a mechanism, check the zooplankton loss decomposition (M1 vs starvation) for Oct-Jan.
- If starvation dominates: raise winter phytoplankton food toward the observed level, and/or lower ingestion Q10 from 2.8 toward 2.0 (in situ fecundity Q10 1.4-1.6; autumn note §1).
- An ECOSMO-style detritus food pathway is possible, but **no** rate data support it.

**3. Not recommended as primary:**
- An ERSEM-style 0.0025/d overwintering switch: its evidence is Calanus diapause at depth, the wrong animals for 0-25 m.
- ERGOM's Z0 seed or ECOSMO's loss floor: numerical floors, not biology.
- Resting eggs: unlikely to matter for a 0-30 m stock at a 100-345 m site (not quantified).
- Leaving M1 at 0.054 but making it temperature-only: this cuts it only to about 0.040/d (sibling).

## References

- Castellani C, Robinson C, Smith T, Lampitt RS (2005) Temperature affects respiration rate of Oithona similis. Mar Ecol Prog Ser 285:129-135. doi:10.3354/meps285129
- Castellani C, Irigoien X, Harris RP, Lampitt RS (2005b) Feeding and egg production of Oithona similis in the North Atlantic. Mar Ecol Prog Ser 288:173-182. doi:10.3354/meps288173 (via sibling note)
- Dvoretsky VG (2012) Seasonal mortality rates of Oithona similis (Cyclopoida) in a large Arctic fjord. Polar Sci 6:263-269. doi:10.1016/j.polar.2012.09.001
- Falkenhaug T, Broms C, Bagøien E, Nikolioudakis N (2022) Front Mar Sci 9:779335. doi:10.3389/fmars.2022.779335 (via herring note)
- HELCOM (2017, web 2019) Guidelines for monitoring of mesozooplankton (COMBINE Annex C7). https://helcom.fi/wp-content/uploads/2019/08/Guidelines-for-monitoring-of-mesozooplankton.pdf
- Hernroth L (ed.) (1985) Recommendations on methods for marine biological studies in the Baltic Sea: mesozooplankton biomass assessment. Baltic Marine Biologists Publ 10 (not read; cited by HELCOM)
- Hirst AG, Kiørboe T (2002) Mar Ecol Prog Ser 230:195-209. doi:10.3354/meps230195 (via sibling note)
- Hjøllo SS, Huse G, Skogen MD, Melle W (2012) Mar Biol Res 8:508-526. doi:10.1080/17451000.2011.642805 (not read)
- Kaartvedt S, Røstad A, Titelman J (2021) Sleep walking copepods? Calanus diapausing in hypoxic waters adjust their vertical position during winter. J Plankton Res 43:199-208. doi:10.1093/plankt/fbab004
- Lindahl O, Hernroth L (1983) Mar Ecol Prog Ser 10:119-126. doi:10.3354/meps010119 (via sibling note)
- Madhupratap M, Nehring S, Lenz J (1996) Resting eggs of zooplankton (Copepoda and Cladocera) from the Kiel Bay and adjacent waters. Mar Biol 125:77-87. doi:10.1007/BF00350762 (abstract only; DOI checked via Crossref)
- Ohman MD, Wood SN (1996) Mortality estimation for planktonic copepods: Pseudocalanus newmani in a temperate fjord. Limnol Oceanogr 41:126-135. doi:10.4319/lo.1996.41.1.0126
- Peters J, Renz J, van Beusekom J, Boersma M, Hagen W (2006) Trophodynamics and seasonal cycle of the copepod Pseudocalanus acuspes in the Central Baltic Sea (Bornholm Basin). Mar Biol 149:1417-1429. doi:10.1007/s00227-006-0290-8 (abstract only; DOI, title and pages checked via Crossref)
- Wesche A, Wiltshire KH, Hirche HJ (2007) Overwintering strategies of dominant calanoid copepods in the German Bight, southern North Sea. Mar Biol 151:1309-1320. doi:10.1007/s00227-006-0560-5 (full text read at epic.awi.de/15993; DOI checked via Crossref)
- Model code: FABM (github.com/fabm-model/fabm, shallow clone 2026-10-20): `src/models/gotm/ergom.F90` (the code cites Neumann et al. 2002; see also Neumann T (2000) Towards a 3D-ecosystem model of the Baltic Sea. J Mar Syst 25:405-419, doi:10.1016/S0924-7963(00)00030-0, checked via Crossref, not read), `src/models/msi/ergom1.F90`, `src/models/nersc/ecosmo/ecosmo_operational.F90`.
- Data: SMHI national monitoring, SHARK zooplankton table, station Å17, 2007-2023 (https://shark.smhi.se/api/sample/download, `sharkdata_zooplankton`, stationName "Å17").

## Appendix: full 12-month Å17 carbon table, total vs resident (added on request)

**Data:** SHARK Å17, "Carbon content" in ug C/m3, 2019-2023. All hauls are WP2 90 um, 25-0 m.

**Method [D]:**
- Sum all taxon and stage rows per sample.
- Take the mean of the samples within each year-month, so 2020's two August visits (16 and 26 Aug) count as one.
- Take the mean over years per calendar month; n is the number of year-months.
- This grouping reproduces exactly the model's current observation row (1.2 0.7 4.3 19.3 26.0 11.4 6.9 4.6 3.7 2.9 3.5 1.2). Weighting each visit date equally instead changes only August: 14,651 ug/m3 = 4.4 t/km2.
- "Deep share" = (Calanus + Metridia carbon) / total, as a ratio of the monthly means. Resident = total minus Calanus and Metridia.
- To t WW/km2 over 0-25 m: ug/m3 x 25 / 10^6 / (C/WW). The main value uses C/WW **0.084**; the value in brackets uses 0.06.

| Month | n | Total ug C/m3 (range) | Deep share | Resident ug C/m3 (range) | Total t WW/km2, 0.084 (0.06) | Resident t WW/km2, 0.084 (0.06) |
|---|---|---|---|---|---|---|
| Jan | 2 | 3,923 (836-7,011) | 0.82 | 710 (81-1,338) | 1.2 (1.6) | 0.21 (0.30) |
| Feb | 3 | 2,477 (1,358-3,662) | 0.75 | 615 (223-1,296) | 0.7 (1.0) | 0.18 (0.26) |
| Mar | 2 | 14,490 (6,752-22,227) | 0.64 | 5,252 (2,396-8,109) | 4.3 (6.0) | 1.56 (2.19) |
| Apr | 3 | 64,867 (37,296-83,863) | 0.89 | 7,135 (4,717-9,382) | 19.3 (27.0) | 2.12 (2.97) |
| May | 4 | 87,326 (34,984-184,186) | 0.77 | 20,496 (3,403-44,869) | 26.0 (36.4) | 6.10 (8.54) |
| Jun | 4 | 38,426 (12,560-80,475) | 0.35 | 24,867 (10,634-42,937) | 11.4 (16.0) | 7.40 (10.36) |
| Jul | 4 | 23,059 (8,243-37,354) | 0.48 | 12,065 (5,453-15,738) | 6.9 (9.6) | 3.59 (5.03) |
| Aug | 4 | 15,405 (11,635-23,306) | 0.67 | 5,048 (1,102-6,871) | 4.6 (6.4) | 1.50 (2.10) |
| Sep | 2 | 12,287 (4,881-19,694) | 0.78 | 2,652 (2,006-3,298) | 3.7 (5.1) | 0.79 (1.10) |
| Oct | 3 | 9,733 (6,416-13,930) | 0.30 | 6,812 (4,989-9,287) | 2.9 (4.1) | 2.03 (2.84) |
| Nov | 3 | 11,642 (5,440-21,368) | 0.23 | 8,934 (3,503-17,810) | 3.5 (4.9) | 2.66 (3.72) |
| Dec | 2 | 4,008 (1,850-6,165) | 0.23 | 3,075 (1,213-4,937) | 1.2 (1.7) | 0.92 (1.28) |

**Annual means [D]:** total 23,970 ug C/m3 (7.1 t WW/km2 at 0.084), resident 8,138 (2.4 t WW/km2), deep share 0.66.

**Caveats:**
- n is only 2-4 per month, and the ranges span 2-50x.
- The Apr-May "deep" carbon is the new Calanus generation growing in the surface layer, not deep-water animals. The deep-origin argument (Q1) applies only to Dec-Mar.
- The resident row is therefore a winter target, not a whole-year one.
- Compare with abundance-based resident estimates for Dec-Feb in the Q1 section.
