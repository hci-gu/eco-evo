# Keeping the autumn-winter mesozooplankton stock: temperature, density dependence, seasonality

Companion to `zoo_winter_mortality.md` in this folder. Rows quoted from that note are marked "(sibling note)".

**Tags.** [F] = I read the full text, a table, a figure or the source code myself. [A] = abstract only. [S] = secondary citation (the source that cites it is named). [SRC] = model source code or parameter file read (a kind of [F]). [D] = derived by me. "Not verified" means exactly that.

**Problem.** Model vs Å17 observations (t WW/km2): Oct 0.3/2.9, Nov 0.09/3.5, Dec 0.03/1.2. Model Oct-Dec net change is -0.038 /d, observed -0.0145 /d [D]. Observed net rates between monthly means [D]: Oct-Nov +0.006, Nov-Dec -0.035, Dec-Jan 0.000, Jan-Feb -0.018, Feb-Mar +0.060, Mar-Apr +0.049, Apr-May +0.010 /d. The model loses about 0.025 /d more than observed in Oct-Dec. It also builds a tenfold deficit before October.

## 0. Unit conversion (check of the prompt)

1 t WW/km2 = 1 g WW/m2. Spread over H = 10 m that is 0.1 g WW/m3 = 100 mg WW/m3. At C:WW = 0.1 this is 10 mg C/m3 = 10 ug C/l = **0.83 umol C/l (mmol C/m3)**. At C:N = 6.6 mol/mol it is 0.83/6.6 = **0.126 mmol N/m3** [D].

**The prompt's "~1.5 mmol N/m3" is about 12x too high.** The correct value is about 0.13 mmol N/m3 per t WW/km2 in a 10 m layer. C:WW = 0.1 is the prompt's assumption, and I did not verify it here.

Every concentration-based quadratic coefficient converts to model units as **m2(area) = m2(conc) x (conversion per t/km2) / (H/10 m)**. The per-capita rate at a given areal biomass therefore scales as **1/H**. A 50 m layer gives rates 5x lower than the 10 m numbers below. Every converted number below assumes H = 10 m.

## 1. Temperature dependence of copepod ingestion and growth

| Quantity | Value | Tag / source |
|---|---|---|
| Rate normalisation used in the largest compilation (respiration, max clearance, max ingestion, growth; 2729/871/327/852 estimates) | all converted to 15 C with **Q10 = 2.8, "following Hansen et al. (1997)"**, about Ea 0.65 eV | [F] Kiørboe & Hirst 2014, Methods p. 6 of the accepted MS |
| Hansen, Bjørnsen & Hansen 1997 (Q10 2.8) | the abstract gives only the size scaling (-0.23) and no temperature values | Q10 [S] via Kiørboe & Hirst 2014; abstract [A] |
| Calanoid **maximum** feeding rates (lab) | "temperature independent" across species, scaling with mass^0.75. Field rates depend on food, temperature and size (r2 0.81), but the direct temperature effect is "difficult to ascertain" | [A] Saiz & Calbet 2007 |
| Copepod field growth, global | g = 0.045 e^(0.111 T) /d, so **Q10 = 3.0**. Gives 0.080 /d at 5.2 C and 0.208 /d at 13.8 C [D] | [S] Huntley & Lopez 1992, via Gawinski et al. 2024 Table S1 (0.0445 is also quoted; I could not verify either coefficient against the original) |
| Sac-spawner growth (Hirst & Lampitt 1998) | log10 g = -1.4647 + 0.0358 T, so **Q10 = 2.3** [D] | [S] same table |
| In situ adult **fecundity** vs lab food-saturated | Q10 1.59 (broadcast) and 1.43 (sac) in situ, vs 2.75 and 3.98 food-saturated. In situ approaches saturation at 0-10 C. Juvenile growth is more strongly temperature-dependent in situ and close to saturation | [A] Hirst & Bunker 2003. These are fecundity and growth, not ingestion |
| Functional-response meta-analysis | temperature dependence is **stronger for attack rate than for max ingestion**, and both are hump-shaped | [A] Englund et al. 2011 |
| Same, across taxa (Arrhenius fit) | attack rate E = 0.42 eV, handling time E = -0.30 eV. Over 5-15 C this is **Q10 1.84 (attack) and 1.54 (max intake 1/h)** [D] | [F] Rall et al. 2012 (text; CIs only in a figure) |
| Lab acclimation (Paracartia grani) | ingestion Q10 falls from 2.6 to 1.6 after 7 d of acclimation | [A] search-result summary only (Saiz et al., PMC9155217), not verified |
| Mauchline 1998 | not checked | - |

**Does temperature scale the whole response or only max ingestion?** In the physiology literature both parameters scale, with the attack rate scaling as much as or more than max ingestion (Englund 2011, Rall 2012). The standard ocean models scale the **whole** response: each multiplies the maximum rate by f(T) and keeps the half-saturation constant fixed, so the initial slope a = g_max/K scales by the same factor.

| Model | Mesozooplankton grazing temperature term | Q10, T_ref | Tag |
|---|---|---|---|
| ERSEM (FABM, Z4) | `et = max(0, q10^((T-10)/10) - q10^((T-32)/3))`, multiplying `sum` (2.0 /d), with a fixed chuc of 36 mg C/m3. Rest respiration (0.015 /d) is also scaled by et. Mortality sd 0.05 /d is **not** scaled | Q10 2.0, T_ref 10 C | [SRC] `src/mesozooplankton.F90`, `fabm-ersem-26.02-dvm.yaml` |
| NEMURO (ROMS) | GRmax x exp(0.0693 T) x Ivlev or Holling; GRmax 0.4 /d (large zoo on large phyto) | Q10 2.0, T_ref 0 C | [SRC] ROMS `nemuro.h`, `nemuro.in` |
| PISCES (NEMO 4.0) | grazrat2 0.75 /d x `tgfunc2 = exp(0.07608 T)`, Holling II with a fixed xkgraz2 of 20 umol C/l. Mortality and respiration also x tgfunc2 | **Q10 2.14**, T_ref 0 C | [SRC] `p4zmeso.F90`, `p4zint.F90`, `namp4zmes` |
| Fennel et al. 2006 (ROMS bio_Fennel) | ZooGR 0.6 /d, Holling III with K_Phy 2 (mmol N/m3)^2. **No temperature term** for zooplankton (only phytoplankton, via Eppley) | none | [SRC] ROMS `fennel.h`, `bio_Fennel.in` |
| ECOSMO (NERSC operational FABM version) | GrZlP 0.8 /d, Holling II with Rg 0.5 mmol N/m3. **No temperature term** on mesozooplankton grazing | none | [SRC] `fabm/src/models/nersc/ecosmo/ecosmo_operational.F90`. May differ from ECOSMO II (Daewel & Schrum 2013), which I did not read |
| NORWECOM(.E2E) | not found. Daewel et al. 2014 say only that its invertebrate predation is "basically a function of light availability" | - | [S] |

**At Å17 temperatures [D].** With T_ref = 9.3 C (the annual mean), f(5.2 C) / f(13.8 C) is 0.75 / 1.37 at Q10 2.0, 0.73 / 1.41 at 2.14, 0.66 / 1.59 at 2.8 and 0.64 / 1.65 at 3.0. Over a ±4.3 C sinusoid the annual mean of f is 1.022 at Q10 2, so the growth budget shifts by about 2 %.

## 2. Density-dependent closure terms

**Forms.** Steele & Henderson 1992 (SH92) used a mortality m_Z Z^w, with w = 1 for linear and w = 2 for quadratic. The quadratic coefficient has units d^-1 (mmol N m^-3)^-1. Their contrast between the subarctic Pacific and the North Atlantic used quadratic mortality with m_Z = 0.1 and 0.5 respectively, together with Type III grazing (g 1.0 /d, k_g 1.0 (mmol N/m3)^2). Linear mortality "generally promotes instability" [S] (SH92 via Anderson & Gentleman 2019, who re-ran the model).

Edwards & Yool 2000 compared linear, quadratic, hyperbolic and sigmoidal closures. Unforced limit cycles can occur with the quadratic closure, and "the choice of closure term can strongly affect model dynamics" [A]. Per Daewel et al. 2014 [S], they found limit cycles in 3 of 4 cases, most likely with the linear closure (both statements as reported; not reconciled against the full text). Edwards & Brindley 1999 (Bull. Math. Biol. 61:303) was not read; I checked only its metadata.

**Rationale.** The usual rationale is that a quadratic or hyperbolic term stands in for predators whose biomass, or whose encounter rate, tracks zooplankton biomass. The PISCES code itself is candid about it: "A square function has been selected with no real reason except that it seems to be more stable and may mimic predation" [SRC p4zmeso.F90]. Daewel et al. 2014 [F, read via WebFetch] note that closure parameters "are difficult to define and the parameterization is usually not empirically motivated". Atlantis uses a combined linear + quadratic term.

**Implemented values [SRC].**
- Fennel (ROMS): loss = ZooMR x Z^2, with ZooMR 0.025 (mmol N/m3)^-1 d^-1. The code's comment says "[1/day]", but the implicit update `Z/(1+dt*ZooMR*Z)` makes it quadratic.
- NEMURO: MorZ0 0.0585 (mmol N/m3)^-1 d^-1 at 0 C x exp(0.0693 T) x Z^2, for small, large and predatory zooplankton alike.
- PISCES: mzrat2 0.03 (umol C/l)^-1 d^-1 x tgfunc2 x Z^2, plus a near-linear resrat2 of 0.005 /d x Z/(0.2 umol C/l + Z).
- ERSEM: linear sd 0.05 /d. Density dependence comes in implicitly through **cannibalism**: Z4 lists Z4 as prey with suprey 0.25. Below 300 mg C/m2 of prey the group switches to an overwintering state with mortality 0.0025 /d (sibling note).
- ECOSMO (operational): linear mZl 0.10 /d plus excretion 0.06 /d.

**Per-capita mortality in model units (H = 10 m, T = 9.3 C) [D]:**

| Z (t WW/km2) | 0.3 | 1 | 3 | 10 | 26 |
|---|---|---|---|---|---|
| NEMURO quadratic | 0.004 | 0.014 | 0.042 | 0.14 | 0.37 |
| PISCES quadratic | 0.015 | 0.051 | 0.15 | 0.51 | 1.32 |
| Fennel quadratic | 0.001 | 0.003 | 0.010 | 0.032 | 0.082 |
| SH92 (0.1-0.5) | 0.004-0.019 | 0.013-0.063 | 0.04-0.19 | 0.13-0.63 | 0.33-1.6 |
| current model M1 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 |
| proposal below, 0.02 + 0.045 Z/(3+Z) | 0.024 | 0.031 | 0.043 | 0.055 | 0.060 |

A pure quadratic calibrated on any of these models gives the low winter loss, but at the May peak of 26 t/km2 it gives 0.08-1.3 /d (5x less at H = 50 m). That would remove the spring maximum, which the model currently reproduces. Any closure used here must saturate.

**Field evidence on density dependence.** Ohman & Hirche 2001 give the first evidence of "nonlinear, density-dependent mortality rates of open-ocean zooplankton": per capita *egg* mortality of C. finmarchicus at Station M depends on the abundance of adult females and juveniles, probably through egg cannibalism [A]. Ohman et al. 2004 found density-dependent early-stage mortality at 2 of 5 North Atlantic sites (Georges Bank, Station M) [A, sibling note]. Hirst et al. 2007 (L4) found that C. helgolandicus CV-adult mortality covaried with temperature (r2 0.65) and with chaetognath (0.28) and siphonophore (0.13) abundance (sibling note, [F]). That is evidence that predator abundance drives the rate.

I found **no** field study that gives per-capita copepod mortality as a function of copepod biomass for the copepodite and adult stocks that dominate biomass. Kattegat seasonal copepod mortality was 3-15 %/d [S] (Kiørboe & Nielsen 1994 via Daewel et al. 2014).

## 3. Seasonality of mortality and of invertebrate predators (Skagerrak, Kattegat, Gullmar Fjord)

| Predator / measure | Timing and magnitude | Tag / source |
|---|---|---|
| Transition zone overall | "The biomass of predators of zooplankton is generally highest from late summer to autumn" | [F] Daewel et al. 2014, citing Tönnesson & Tiselius 2005 and Tönnesson et al. 2006 |
| Kattegat copepods | biomass peaks Jun-Jul and falls from August to late October, **although Aug-Sep had the year's largest production event**. The authors infer high mortality | [A] Kiørboe & Nielsen 1994 (via the abstract on DTU Orbit) |
| *Sagitta setosa*, Gullmar 1998 | about 10 ind/m3 on 6 Aug, 10x by 14 Sep, peak of **310 ind/m3 on 1 Oct**, about 20 on 15 Dec. Present only in late summer and autumn; collapses through cannibalism (74-84 %/d potential in October). Impact on single copepod species up to **48 %/d** at the end of October. S. elegans takes 5-18 %/d of Calanus. Total prey fell from 28,000 to 1,900 ind/m3 (0-10 m) between August and December | [F] Tönnesson & Tiselius 2005 |
| *Mnemiopsis leidyi*, Gullmar 2007-09 | none Jan-Jul. Biomass appears in late Jul-Aug, peaks Sep-Oct (about 32 g WW/m3 in Sep 2008, up to about 80 in Oct 2009) and falls through Nov-Dec. In 2010 it arrived only in early September, after a cold winter. Predation impact in 2008 was 0 until late July, **5-16 %/d in Aug to early Sep** and 4-8 %/d in Sep-Oct. Copepods fell from 15,000-21,000 to 1,000-3,000 ind/m3 | [F] Møller & Tiselius, PICES 2011 slides (figures read; not peer-reviewed) |
| Same, 2007-2014 | copepods "significantly reduced during August-November" in Mnemiopsis years, with a cascade to chlorophyll | [A] Tiselius & Møller, ICES ZP6 2016 abstract |
| *Aurelia aurita*, *Cyanea capillata* (Gullmar) | historical presence Apr-Sep (Aurelia) and May-Oct (Cyanea) (approximate bars). No Aurelia in 2010 | [F] same slides |
| *Aurelia*, Gullmar 1978-81 | regulates zooplankton in summer. Copepod biomass rises rapidly "when the medusae decline in abundance in September/October" | [F] Lindahl & Hernroth 1983 |
| North Sea | in autumn, lower zooplankton production, warmer water and many 0-group fish make local depletion likely | [S] Munk & Nielsen 1994 and Nielsen & Munk 1998, via Daewel et al. 2014 |
| Temperature dependence of field mortality | Q10 about 2.0 overall (broadcast 2.06, sac 2.03); predation is 2/3-3/4 of adult mortality | [F] Hirst & Kiørboe 2002 (sibling note) |
| Plourde et al., Eiane & Ohman (seasonal) | Eiane & Ohman 2004 cover spring only (sibling note). I did not check Plourde | - |

**Pattern.** Invertebrate predation on copepods in this region peaks from **August to October** and has collapsed by December. Chaetognaths die back through cannibalism, Mnemiopsis disappears, and the scyphomedusae are gone from autumn to spring.

## 4. Autumn and winter copepod dynamics in the region

- **Gullmar Fjord 1978-81** [F] (Lindahl & Hernroth 1983).
  - Mesozooplankton biomass was **at its maximum from October to December**, mainly copepods. Deeper layers held Calanus finmarchicus, Pseudocalanus elongatus and Microcalanus. Above the sill there were Acartia longiremis, A. clausi and Temora longicornis, with Oithona similis throughout. The biomass "remains fairly high throughout autumn".
  - The first months of the year are "extremely poor". Above sill depth the water column is "practically uninhabited", and only a small diapausing population sits deep.
  - Primary production was about 500 mg C m-2 d-1 in the first half of the season and **about 800 in the second half**, with autumn blooms of Ceratium and Gyrodinium.
  - Fig. 6 is in g dry weight m-2 over **0-120 m** (3-D plot; peaks read as roughly 5-8 g DW m-2 in Oct-Dec, approximate). It is not directly comparable with Å17 units or depths, and it covers an Aurelia-dominated period.
- **Kattegat** [A] (Kiørboe & Nielsen 1994). Production was episodic, in bursts tied to three net phytoplankton blooms, one of them the largest in Aug-Sep. Microzooplankton could sustain egg production between blooms.
- **Å17 itself.** The observations (Oct 2.9, Nov 3.5, then 1.2 in Dec-Jan) look like the Lindahl & Hernroth picture, damped: a stock that survives the autumn predator peak and decays only slowly in winter. I did not query SHARK, did not check Eriksson 1973, and found no published monthly Skagerrak production series.
- **Why the stock persists Sep-Nov, by the evidence.**
  1. Production continues: there are autumn blooms, and in the Kattegat the largest production event fell in Aug-Sep.
  2. The predators **decline by themselves** from October (S. setosa, Mnemiopsis, medusae), so predation relaxes as copepods become scarce. That is a density-dependent signal, with a lag.
  3. Winter metabolism and mortality are lower at about 5 C (sibling note: Q10 about 2).
  4. The small, non-diapausing species keep feeding on microzooplankton (sibling note).

  Diapause is not the explanation for the 0-50 m stock.

## 5. Recommendation

**A seasonal M1 that is *lower* in autumn is contradicted by the literature.** Every regional source puts the predation peak in Aug-Oct. A seasonal M1 that follows the predators (high Aug-Oct, low Dec-Apr) would help the winter but make the autumn worse.

**Diagnose first.** The literature cannot say whether the Aug-Dec collapse comes from M1, from herring predation or from starvation through the energy reserve (with the hunger gate and maintenance level, a DM can lose mass without any "mortality"). Decompose the model's zooplankton loss terms by month for Aug-Dec, and check the autumn phytoplankton and microzooplankton food, before choosing a closure. The right fix differs by cause.

**Best-supported changes, in order:**

1. **A saturating (hyperbolic) density-dependent M1 that equals today's M1 at high biomass:**
   `M1(Z) = m_L + m_P * Z/(K_Z + Z)` per capita, so the loss is `m_L Z + m_P Z^2/(K_Z + Z)`.
   - m_L is the non-predatory floor, about **0.02 /d**: Hirst & Kiørboe give a predation-free adult mortality of 0.019 /d at 5 C ([F], sibling note).
   - m_P is the unmodelled predators (ctenophores, chaetognaths, medusae, fish larvae), about **0.045 /d**.
   - K_Z is about **3 t WW/km2**.
   - These values are my suggestion [D], not literature values. Edwards & Yool tested a "hyperbolic" closure (the name comes via Daewel et al. 2014; I did not read their exact form). The rationale here is predators whose impact tracks copepod abundance but saturates (Holling) at high prey density.
   - Per capita this gives 0.031 /d at 1 t/km2, 0.043 at 3, 0.055 at 10 and 0.060 at 26. **The spring and summer fit is essentially unchanged**, and at the observed Oct-Dec biomasses (1.2-3.5 t/km2) the loss is 0.033-0.044 /d, which is 18-39 % (0.010-0.021 /d) below the current 0.054.
   - The gap between model and observations is about 0.025 /d, so this closes roughly half to most of it, depending on where in Oct-Dec, if the cause is M1.
   - **Implementation:** `growth_rate` is derived from M1 (AGENTS.md sections 129 and 160). Define the M1 that enters the budget as M1(Z_ref) with Z_ref about 10 t/km2. Then M1(Z_ref) = 0.0546 ≈ the current 0.054, so the derived g barely moves. Alternatively, choose m_L + m_P Z_ref/(K_Z + Z_ref) = 0.054 exactly.
   - Note that K_Z is in areal units: if the effective layer is deeper than the one the coefficients assume, K_Z scales with H.
2. **Temperature-scale the whole functional response (both a and 1/h) with Q10 = 2.0 and T_ref = 9.3 C,** as ERSEM and NEMURO do (PISCES uses 2.14).
   - This cuts Feb-Mar intake by 25 % (factor 0.75), which addresses the too-fast growth in cold Feb-Mar, and raises the Sep ceiling by 37 %.
   - The growth evidence would allow a steeper Q10 of 2.3-3.0 (Hirst & Lampitt, Huntley & Lopez). However, in situ rates (Hirst & Bunker), interspecific lab maxima (Saiz & Calbet) and acclimation all argue for ≤2 for a seasonally succeeding community. Rall's 1.5-1.8 is the cross-taxa physiological value.
   - Use one Q10 for a and h, consistent with resting metabolism (Q10 2.0) and with microzooplankton grazing (2.6).
3. **Consistency warning.** If M1 is also made temperature-dependent (Q10 2, as NEMURO and PISCES do and the sibling note recommends), Aug-Sep losses rise about 37 % (factor 1.37 at 13.8 C). In autumn that roughly cancels the gain from temperature-scaled intake, and the net help falls in Feb-Mar, where both rates drop. Temperature scaling **alone does not fix the autumn** in either variant. Only the density-dependent part (or a model-side cause found by the diagnosis) does.
4. **Not recommended:** a pure quadratic (it destroys the May peak; see the table) and a hand-drawn seasonal M1 curve (no regional data to calibrate it beyond "high Aug-Oct, low Dec-Apr", and it points the wrong way for October).

Further work outside the agent's scope: get the Å17 and SHARK monthly zooplankton series for Aug-Sep. A tenfold model deficit already exists in October, so the Aug-Sep observations decide whether K_Z needs to be larger.

## References

- Anderson TR, Gentleman WC (2019) Remembering John Steele and his models for understanding the structure and function of marine ecosystems. J Plankton Res 41:609-620. doi:10.1093/plankt/fbz042
- Butenschön M et al. (2016) ERSEM 15.06. Geosci Model Dev 9:1293-1339. doi:10.5194/gmd-9-1293-2016 (code: github.com/pmlmodelling/ersem)
- Daewel U, Hjøllo SS, Huret M, Ji R, Maar M, Niiranen S, Travers-Trolet M, Peck MA, van de Wolfshaar KE (2014) Predation control of zooplankton dynamics: a review of observations and models. ICES J Mar Sci 71:254-271. doi:10.1093/icesjms/fst125
- Daewel U, Schrum C (2013) Simulating long-term dynamics of the coupled North Sea and Baltic Sea ecosystem with ECOSMO II. J Mar Syst 119-120:30-49 (not read; volume/pages from memory, not verified)
- Edwards AM, Brindley J (1999) Zooplankton mortality and the dynamical behaviour of plankton population models. Bull Math Biol 61:303-339. doi:10.1006/bulm.1998.0082 (not read)
- Edwards AM, Yool A (2000) The role of higher predation in plankton population models. J Plankton Res 22:1085-1112. doi:10.1093/plankt/22.6.1085
- Eiane K, Ohman MD (2004) Stage-specific mortality of Calanus finmarchicus, Pseudocalanus elongatus and Oithona similis on Fladen Ground, North Sea, during a spring bloom. Mar Ecol Prog Ser 268:183-193
- Englund G, Öhlund G, Hein CL, Diehl S (2011) Temperature dependence of the functional response. Ecol Lett 14:914-921. doi:10.1111/j.1461-0248.2011.01661.x
- Fennel K, Wilkin J, Levin J, Moisan J, O'Reilly J, Haidvogel D (2006) Nitrogen cycling in the Middle Atlantic Bight. Global Biogeochem Cycles 20:GB3007. doi:10.1029/2005GB002456 (values from ROMS code)
- Gawinski C et al. (2024) Mar Ecol Prog Ser 735:77-101, Supplement Table S1. doi:10.3354/meps14570
- Hansen PJ, Bjørnsen PK, Hansen BW (1997) Zooplankton grazing and growth: scaling within the 2-2,000-um body size range. Limnol Oceanogr 42:687-704. doi:10.4319/lo.1997.42.4.0687
- Hirst AG, Bunker AJ (2003) Growth of marine planktonic copepods: global rates and patterns in relation to chlorophyll a, temperature, and body weight. Limnol Oceanogr 48:1988-2010. doi:10.4319/lo.2003.48.5.1988
- Hirst AG, Kiørboe T (2002) Mortality of marine planktonic copepods: global rates and patterns. Mar Ecol Prog Ser 230:195-209. doi:10.3354/meps230195
- Hirst AG, Lampitt RS (1998) Towards a global model of in situ weight-specific growth in marine planktonic copepods. Mar Biol 132:247-257 (via Gawinski 2024)
- Huntley ME, Lopez MDG (1992) Temperature-dependent production of marine copepods: a global synthesis. Am Nat 140:201-242. doi:10.1086/285410
- Kiørboe T, Hirst AG (2014) Shifts in mass scaling of respiration, feeding, and growth rates across life-form transitions in marine pelagic organisms. Am Nat 183:E118-E130. doi:10.1086/675241
- Kiørboe T, Nielsen TG (1994) Regulation of zooplankton biomass and production in a temperate, coastal ecosystem. 1. Copepods. Limnol Oceanogr 39:493-507. doi:10.4319/lo.1994.39.3.0493
- Kishi MJ et al. (2007) NEMURO - a lower trophic level model for the North Pacific marine ecosystem. Ecol Model 202:12-25 (values from ROMS code)
- Lindahl O, Hernroth L (1983) Phyto-zooplankton community in coastal waters of western Sweden - an ecosystem off balance? Mar Ecol Prog Ser 10:119-126. doi:10.3354/meps010119
- Møller LF, Tiselius P (2011) Population dynamics and predation impact of the introduced ctenophore Mnemiopsis leidyi in the Gullmars Fjord. PICES/ICES Zooplankton Symposium 2011, talk S2 (slides)
- Ohman MD, Hirche H-J (2001) Density-dependent mortality in an oceanic copepod population. Nature 412:638-641. doi:10.1038/35088068
- Ohman MD et al. (2004) A comparative study of Calanus finmarchicus mortality patterns at five localities in the North Atlantic. ICES J Mar Sci 61:687-697
- Aumont O, Ethé C, Tagliabue A, Bopp L, Gehlen M (2015) PISCES-v2. Geosci Model Dev 8:2465-2513. doi:10.5194/gmd-8-2465-2015 (values from NEMO 4.0 code)
- Rall BC, Brose U, Hartvig M, Kalinkat G, Schwarzmüller F, Vucic-Pestic O, Petchey OL (2012) Universal temperature and body-mass scaling of feeding rates. Phil Trans R Soc B 367:2923-2934. doi:10.1098/rstb.2012.0242
- Saiz E, Calbet A (2007) Scaling of feeding in marine calanoid copepods. Limnol Oceanogr 52:668-675. doi:10.4319/lo.2007.52.2.0668
- Steele JH, Henderson EW (1992) The role of predation in plankton models. J Plankton Res 14:157-172. doi:10.1093/plankt/14.1.157
- Tiselius P, Møller LF (2016) Trophic cascades over three trophic levels in a coastal food web. ICES Zooplankton Production Symposium (ZP6), abstract
- Tönnesson K, Tiselius P (2005) Diet of the chaetognaths Sagitta setosa and S. elegans in relation to prey abundance and vertical distribution. Mar Ecol Prog Ser 289:177-190. doi:10.3354/meps289177
