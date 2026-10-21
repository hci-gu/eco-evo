# Temperature dependence (Q10 / activation energy) of metabolism and consumption in Skagerrak ectotherms

Scope: herring (pelagic planktivore), gadoids (cod, whiting, saithe), zooplankton (calanoid copepods, northern krill), benthic invertebrates; relevant temperature window 3-18 C. Conversion conventions used below (my own arithmetic, not from a source): Q10 <-> E via Arrhenius around 10 C (E = ln(Q10) * k / (1/T1 - 1/T2), k = 8.617e-5 eV/K, T1 = 278 K, T2 = 288 K). Resulting pairs: Q10 1.4 = 0.23 eV, 1.83 = 0.42 eV, 2.0 = 0.48 eV, 2.2 = 0.55 eV, 2.4 = 0.61 eV, 2.6 = 0.65 eV, 2.8 = 0.70 eV. Inversely E = 0.65 eV gives Q10 = 2.60 over 3-13 C and 2.51 over 8-18 C. Rates in J/g/d are numerically equal to kJ/kg/d. Oxycalorific coefficient used in the herring/blue-whiting bioenergetics model of Bachiller et al. 2018: 13,560 J per g O2 (from Elliott & Davison 1975) — [Bachiller et al. 2018, PLoS ONE, Table 6](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0190345).

---

## Q1. Q10 / activation energy for standard or routine metabolism of herring and other clupeids (sprat, Baltic herring)

### Takeaway
Measured clupeid respiration Q10 is about 2.2 (sprat, 9-21 C). The standard Wisconsin herring parametrisation (Rudstam 1988 lineage) uses an exponential coefficient of 0.0548 per C (Q10 = 1.73) for the resting term. But it also has a swimming-speed term that falls below 9 C, so the model's *total* metabolism has an effective Q10 of about 2.5 over 5-15 C and about 4 over 5-9 C. No direct respirometry of adult Atlantic herring at 4-7 C was found.

### Cited Findings
- **Sprat (Sprattus sprattus, Baltic, 3.1-9.7 g wet weight), intermittent-flow respirometry at nine temperatures from 9 to 21 C.** Routine metabolism (RR) fitted as M = a*exp(cT). Small sprat: RR c = 0.075 (Q10 2.12) and RS (standard) c = 0.081 (Q10 2.25). Large sprat: RR c = 0.080 (Q10 2.23) and RS c = 0.079 (Q10 2.20). Pooled: c_RR = 0.080 and c_RS = 0.078 (Q10 2.18). Summary statement: "Q10 = 2.2, similarly as reported for other clupeoids (Q10 = 1.8-2.2)" — [Meskendahl 2013, PhD thesis, Univ. Hamburg](https://d-nb.info/1032990570/34).
- Sprat absolute rates: small-sprat RR rose from 0.162 to 0.367 mg O2 g-1 h-1 (9 to 21 C). Large-sprat RR rose from 0.188 (10 C) to 0.504 mg O2 g-1 h-1 (19 C). RS was on average only about 12-13% below RR, meaning schooling sprat swim continuously. The author concludes that "the concept of standard metabolism may not be meaningful in schooling planktivorous fish" — [Meskendahl 2013](https://d-nb.info/1032990570/34).
- Sprat swimming-cost model: the swimming-speed temperature exponent c = 0.139 (Q10 4.0). Overall MO2 Q10 over 9-19 C is 2.1 at 0.5 BL/s and 2.8 at 0.8 BL/s — [Meskendahl 2013, Manuscript 2](https://d-nb.info/1032990570/34).
- Other clupeids cited in the same thesis: American shad Q10 2.2 (15.6-21.3 C); Atlantic menhaden about 2.1 (10-25 C); Sardinops sagax 1.82 (9.7-22.7 C) and 3.2 (15-20 C, juveniles); anchoveta 2.6 (14-20 C) — [Meskendahl 2013](https://d-nb.info/1032990570/34). Flag: most of these were measured at temperatures warmer than the Skagerrak winter.
- **Herring Wisconsin-model respiration (Pacific herring, NEMURO.FISH lineage), Rose et al. 2008, Table 1.** Respiration has intercept aR = 0.0058 / 0.0033 / 0.0033 (age-0 / age-1 / age-2+) and weight exponent bR = 0.007 / 0.227 / 0.227 (used as negative). The temperature coefficient cR is 0.083 for age-0 and 0.0548 for age-1+, i.e. Q10 = exp(10*cR) = 2.29 and 1.73. The swimming-speed coefficient dR is 0 / 0.03 / 0.03. Swimming speed has intercept aA = 3.9 (<9 C) or 15.0 (>=9 C), weight exponent bA = 0.13, and temperature coefficient cA = 0.149 (<9 C) or 0.0 (>=9 C). SDA aS = 0.125 / 0.15 / 0.175. Egestion aF = 0.125 / 0.16 / 0.16 and excretion aE = 0.078 / 0.10 / 0.10 — [Rose et al. 2008, Trans. Am. Fish. Soc. 137:278-297](https://www.pmel.noaa.gov/foci/publications/2008/rose0544.pdf). The model was configured from "modified versions of the usual formulations in the Wisconsin model" and builds on Megrey et al. 2007 (same source).
- **Norwegian spring-spawning herring, Bachiller et al. 2018 (same Rudstam-type respiration constants).** Respiration is R = alpha W^beta e^(rho T) e^(Topt*SW), with RA = 0.0033 g O2 g-1 d-1, beta = -0.227 and rho = 0.0548 per C. The swimming coefficient is 0.03 with swimming speed SW fixed at 1 body length/s. Further constants: egestion 0.16, excretion 0.10, SDA 0.175, alpha = RA*13560/ED_pred. The blue whiting column uses the Hansson cod parameters (RA 0.008, beta -0.172, RQ 1.88, Tmax 24, Topt 21, ACT 1.25) — [Bachiller et al. 2018](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0190345). The parameter labels in that table are partly garbled: the swimming coefficient is labelled "Topt".
- **Herring larvae (5-15 C), well-fed:** SMR = 0.0528 * DM^0.8859 * e^(0.1046 T), giving Q10 = exp(1.046) = 2.85 (my arithmetic) — [Linking rates of metabolism and growth in marine fish larvae, Mar. Biol. 2018](https://link.springer.com/article/10.1007/s00227-017-3252-4). This equation was seen only in a search snippet. It covers larvae only and should not be transferred to adults.
- **Juvenile herring (9-10 cm, 1.0-1.5 g dry mass), Bernreuther et al. 2013.** Metabolism here is the rate of mass loss in *food-deprived* fish (not respirometry) at 9.7, 14.2 and 17.9 C. The fit is M = -1.24*T^2 + 38.2*T - 218 (J g DM-1 d-1); the abstract text drops the exponent. The curve is hump-shaped with an optimum near 15-16 C and declines above it. The maintenance ration at 16 C was 2.0% DM/d (432 J fish-1 d-1) — [Bernreuther et al. 2013, J. Appl. Ichthyol. (abstract via OpenAlex)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jai.12045). The polynomial gives 35.9 J/gDM/d at 9.7 C, 74.4 at 14.2 C, 76.2 at the 15.4 C peak and 68.5 at 17.9 C (my arithmetic). Any Q10 computed from it (about 5 over 9.7-14.2 C) is an artefact of a quadratic that approaches zero near 8 C. It is **not** a usable Q10.

### Inferences
- The two herring parametrisations differ in how activity scales with temperature:
  - **Rose/Megrey/Rudstam form:** activity = exp(dR * U), with U = aA * W^0.13 * exp(cA*T) in cm/s. This is my reconstruction from the Wisconsin convention plus the Rose et al. 2008 Table 1 values; I did not read Rudstam 1988 itself. Swimming speed roughly halves from 9 C to 4 C, which drops the activity multiplier from about 2.27 to about 1.47 for a 100 g fish.
  - **Bachiller 2018:** holds SW = 1 BL/s, so the activity factor is the constant e^0.03 = 1.03 and the effective Q10 stays 1.73.
- Herring totals implied by the Rose/Rudstam form (my arithmetic, 13,560 J/g O2):

| Fish mass | Temperature | Total metabolism (mg O2/kg/h) | Total metabolism (kJ/kg/d) |
|---|---|---|---|
| 100 g | 4 C | 89 | 28.8 |
| 100 g | 5 C | 100 | 32.4 |
| 100 g | 7 C | 130 | 42.2 |
| 100 g | 9 C | 180 | 58.4 |
| 100 g | 12 C | 212 | 68.9 |
| 100 g | 15 C | 249 | 81.2 |
| 100 g | 17 C | 278 | 90.6 |
| 200 g | 5 C | 89 | 28.9 |
| 200 g | 15 C | 230 | 74.9 |

  - Effective Q10, 100 g fish: 2.51 over 5-15 C, 2.67 over 4-14 C, 4.4 over 5-9 C, and 1.73 above 9 C.
  - The resting term alone (100 g) gives 63.6 mg O2/kg/h (20.7 kJ/kg/d) at 5 C and 110 mg O2/kg/h (35.8 kJ/kg/d) at 15 C, a ratio of 1.73.
  - Confidence: medium. The parameters come secondhand from Pacific/Baltic herring models, the activity term is a model assumption rather than a measurement, and the step at 9 C is an artefact of the formulation.
- **Recommended value for an adult-herring "resting/maintenance" multiplier:** Q10 about 2.0-2.2 (E about 0.48-0.55 eV), anchored by the measured clupeid respirometry (sprat 2.2). Use Q10 of about 2.5 or higher only if the model's activity cost should also fall in the cold.
- **Sprat absolute rates as a herring proxy (small sprat RR = 0.077*exp(0.075T) mg O2/g/h):** 151 mg O2/kg/h at 9 C and 237 at 15 C. Extrapolating to 5 C gives 112 (my arithmetic, outside the 9-21 C measured range). Sprat are 3-10 g fish, so mass-specific rates are higher than for 100-200 g herring.

### Gaps
- No respirometry of adult Atlantic herring (any stock) at 4-7 C or 12-17 C was found. The adult winter-vs-summer numbers above are model-derived.
- I did not obtain Rudstam 1988 (Kieler Meeresforsch. Sonderh. 6:312-322) or Megrey et al. 2007 directly. Their constants are known only through Rose et al. 2008 and Bachiller et al. 2018.
- I did not obtain the parameter set in Fish Bioenergetics 4.0 (Deslauriers et al. 2017) for herring.
- A claim that Baltic herring "do not feed below 2 C" appeared in a search summary without an identifiable source. It is unverified.

---

## Q2. Q10 / activation energy for Atlantic cod, whiting (and saithe)

### Takeaway
Measured cod SMR has Q10 2.2 over 5-15 C: 2.6 over 5-10 C and 1.9 over 10-15 C. That is E about 0.55 eV over the whole range, with higher sensitivity in the cold. The widely used Hansson et al. 1996 cod bioenergetics model has a respiration "RQ" of 1.88, but through its Kitchell curve (RTO 21 C) the effective Q10 is only about 1.67 over 5-15 C. The measured data and the model disagree.

### Cited Findings
- **Juvenile Atlantic cod, Schurmann & Steffensen 1997.** SMR was 35.5, 57.0 and 78.2 mg O2 kg-1 h-1 at 5, 10 and 15 C. Active metabolic rate (AMR) was 146.6, 197.9 and 200.4 mg O2 kg-1 h-1. Scrit was 16.5, 23.2 and 30.3% saturation. "Previously reported SMR for Atlantic cod from arctic waters at 4 C was twice that measured at 5 C" — [Schurmann & Steffensen 1997, J. Fish Biol. (abstract via OpenAlex)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1095-8649.1997.tb01645.x). The derived Q10 (my arithmetic) is 2.58 for 5-10 C, 1.88 for 10-15 C and 2.20 for 5-15 C. Converted at 13.56 kJ/g O2 (my arithmetic), the SMR is 11.6, 18.6 and 25.5 kJ/kg/d.
- **Hansson et al. 1996 cod bioenergetics parameters (as tabulated for Georges Bank / Gulf of Maine cod).** Consumption: CA 0.099 g/g/d, CB 0.195, CTO 13.7 C, CTM 21.0 C, CQ 2.41. Respiration: RA 0.008 g O2/g/d, RB 0.172, RTO 21 C, RTM 24 C, RQ 1.88, ACT 1.25. Both rates use the Kitchell "equation 2" temperature function — [Mateo 2007, J. Northw. Atl. Fish. Sci.](http://journal.nafo.int/Portals/0/2007/3-mateo.pdf).
- Absolute values from the Hansson model (my arithmetic, standard Kitchell function, RB applied as a negative exponent):

| Fish mass | Temperature | Resting (mg O2/kg/h) | With ACT 1.25 (mg O2/kg/h) |
|---|---|---|---|
| 1000 g | 5 C | 51 | 64 |
| 1000 g | 10 C | 67 | 83 |
| 1000 g | 15 C | 85 | 106 |
| 100 g | 5 C | 76 | 95 |
| 100 g | 15 C | 126 | 158 |

  The effective respiration Q10 is 1.67 over 5-15 C and 1.70 over 3-13 C.
- **Whiting and saithe:** SMR of whiting was about 59.2 mg O2 kg-1 h-1, similar to saithe. The optimal swimming speed was 1.0 BL/s with a cost of transport of 0.15 J N-1 m-1 — [Steinhausen et al. 2005, Mar. Biol.](https://link.springer.com/article/10.1007/s00227-005-0055-9) (via search snippet). The measurement temperature was not recovered, and no temperature series or Q10 was found for whiting or saithe.

### Inferences
- **Recommended for the gadoid FG:** Q10 about 2.2 (E about 0.55 eV) over 5-15 C, from measured cod SMR. Over 3-8 C, Q10 is probably 2.5 or higher. The Hansson-model value (about 1.7 effective) is a lower bound.
- The 4 C arctic-cod SMR being twice the 5 C value (Schurmann & Steffensen) shows that population and acclimation history matter as much as acute Q10.

### Gaps
- Cod Q10 by temperature band (1.8-2.1 at 5-15 C, 2.6-4.3 at 2-5 C, 1.4-1.6 at 15-20 C) appeared in a search summary. The likely sources are [Tirsgaard et al. 2015, PubMed 25281351](https://pubmed.ncbi.nlm.nih.gov/25281351) or a ResearchGate figure (fig1_230040418), but I could not confirm which paper reports them, so they are unverified. They are consistent in direction with Schurmann & Steffensen.
- No whiting or saithe Q10. Claireaux et al. and Jordan & Steffensen cod data were not retrieved.

---

## Q3. Q10 / activation energy for calanoid copepods and krill (Calanus, Temora, Acartia, Pseudocalanus, Meganyctiphanes)

### Takeaway
Active epipelagic copepods have respiration Q10 about 1.8-2.1 (global multi-species regressions). Northern krill from the Kattegat, a population right next to the Skagerrak, has Q10 2.0 over 4-16 C with no thermal compensation. Diapausing Calanus finmarchicus is the exception: it shows no significant temperature response and respires at about 10-20% of the active rate.

### Cited Findings
- **Ikeda et al. 2001 (epipelagic marine copepods, multiple regression on body mass and habitat temperature).** Q10 was 1.8-2.1 for O2 consumption, 1.8-2.0 for NH4 excretion and 1.6-1.9 for PO4 excretion. Body mass and temperature together explained 93-96% of the variance in O2 consumption — [Ikeda et al. 2001, Mar. Biol. 139:587](https://link.springer.com/article/10.1007/s00227-001-0743-z) (via search snippet).
- **Ikeda 1985 (epipelagic zooplankton, global):** Q10 about 2.1 according to a secondary summary — [Ikeda 1985, Mar. Biol.](https://link.springer.com/article/10.1007/BF00396409). I could not access the original regression coefficients. Confidence: low-medium.
- Caveat: these are *habitat-temperature* (largely across-species/latitude) regressions, not acute within-animal responses.
- **Calanus finmarchicus CV, Ingvarsdottir et al. 1999.** Collected north of Scotland, in the Faroe-Shetland Channel and west of Ireland, October 1993 to June 1995, and measured at 0, 5, 7 and 12.5 C. Respiration was "very low (7-30 umol O2 gC-1 h-1) at in situ temperatures during the winter and early spring" versus 105-219 umol O2 gC-1 h-1 for the active surface population in May-June. "Animals collected from the overwintering population showed no significant response to changes in temperature." The diapause carbon loss was about 0.250 ugC/day — [Ingvarsdottir et al. 1999, Fish. Oceanogr. (abstract via OpenAlex)](https://onlinelibrary.wiley.com/doi/10.1046/j.1365-2419.1999.00002.x).
- **Meganyctiphanes norvegica, Kattegat (Alkor Deep, 140 m).** Oxygen consumption "increase[d] exponentially from 31 umolO2 h-1 gdw-1 at 4 C to 72 umolO2 h-1 gdw-1 at 16 C, giving a Q10-value of 2.0, and indicating that physiological adaptation to varying thermal conditions does not take place." The krill experience 4-6 C at depth and up to 16 C at the surface during diel vertical migration — [Saborowski, Salomon & Buchholz 2000, Hydrobiologia (AWI ePIC)](https://epic.awi.de/id/eprint/3745/). This is the one measurement found that sits squarely in the Skagerrak regime and range. Confidence: high.
- **M. norvegica across climatic zones (Kattegat, Clyde Sea, Ligurian Sea).** Q10 was about 2. Populations "exhibited approximately the same level of oxygen consumption when incubated at their respective ambient temperatures", i.e. there is between-population compensation — [Saborowski et al. 2002, Mar. Biol.](https://link.springer.com/article/10.1007/s00227-001-0730-4) (via search snippet).

### Inferences
- **Recommended for active zooplankton FGs:** Q10 about 2.0 (E about 0.48 eV), with confidence medium-high. Krill (Kattegat, 4-16 C) and the multi-species copepod regressions (1.8-2.1) agree.
- If the zooplankton FG represents Calanus, the winter behaviour is not a Q10 effect. It is a state switch (diapause) with about 80-90% metabolic suppression and temperature insensitivity, so a Q10 multiplier alone will not reproduce it. For Temora, Acartia and Pseudocalanus, which do not diapause like Calanus, a plain Q10 of about 2 is more defensible.

### Gaps
- No species-specific Q10 was recovered for Temora longicornis, Acartia clausi or Pseudocalanus elongatus. Relevant primary papers exist (e.g. Gaudy; Marshall & Orr-type studies of "respiration of some planktonic copepods II: effect of temperature"; a Pseudocalanus vs Acartia comparison in JEMBE 2016), but none were accessed.
- Two figures seen in search summaries have no identified source: a Calanus finmarchicus rate of "130 umol O2 gN-1 h-1 at 0 C with Q10 2.77", and "copepod Q10 2.05-4.5 (Arrhenius temperature characteristics 11-23)". Both are unverified.
- Hirst & Kiorboe compilations were not retrieved.

---

## Q4. Benthic invertebrates

### Takeaway
The only quantitative source found is an across-species latitudinal comparison of calcified benthic invertebrates. It gives a low median Q10 of 1.4 (E 0.16 eV), which reflects evolutionary compensation, not the acute or seasonal response.

### Cited Findings
- **Watson et al. 2013 online / 2014 print (Oecologia 174:45-54).** 17 species of calcified benthic invertebrates (laternulid bivalves, buccinid gastropods, echinoid urchins, rhynchonellid brachiopods) from 0-30 C seawater across a 70-degree latitudinal gradient. Median Q10 was 1.4 (interquartile range 0.9-1.7) within taxonomic groups, with overall Ea = 0.16 eV, "lower than predictions from metabolic theory" — [Watson et al. 2014, Oecologia (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3884134/).
- Note: by my conversion, Q10 = 1.4 corresponds to about 0.23 eV around 10 C, not 0.16 eV. The paper's Ea was presumably fitted separately; the two numbers are not mutually consistent.

### Inferences
- Interspecific/latitudinal values (Watson 2014; Clarke & Johnston 1999's 1.83 for fish) understate the seasonal response of a single local population. For a seasonal multiplier on a benthic FG, a within-species value of about 2 (E about 0.5 eV) is more defensible than 1.4. That suggestion rests on general MTE/Dell 2011 within-species results (Q5), not benthos-specific data. Confidence: low.

### Gaps
- No within-species seasonal Q10 for North Sea/Skagerrak benthos (Nephrops, brittle stars, Amphiura, bivalves) was retrieved.

---

## Q5. Within- vs across-species values, and the general MTE reference values

### Takeaway
Within-species (acclimation/seasonal) temperature responses average E about 0.66 eV (median 0.55). Teleost within-species acclimation studies give median Q10 2.40. Across-species curves are flatter (Q10 1.83 in fish). For a seasonal multiplier on a single FG, the within-species values are the relevant ones.

### Cited Findings
- **Clarke & Johnston 1999 (138 studies, 69 teleost species, about 40 C range).** The across-species Arrhenius fit for a 50 g fish is ln Rb = 15.7 - 5.02*T^-1 (T in 10^3 K), with "mean Q10 = 1.83 over the range 0-30 C", which is lower than "typical within-species acclimation studies (median Q10 = 2.40, n = 14)". The mass-scaling exponent is b = 0.79 — [Clarke & Johnston 1999, J. Anim. Ecol. (abstract via OpenAlex)](https://besjournals.onlinelibrary.wiley.com/doi/10.1046/j.1365-2656.1999.00337.x).
- **Dell, Pawar & Savage 2011 (PNAS; 1,072 thermal responses, 112 traits, 309 species).**
  - 87% of intraspecific rises are well fit by Boltzmann-Arrhenius, with mean E = 0.66 +/- 0.05 eV and median 0.55 eV (right-skewed).
  - 88% of significant rises fall between 0.2 and 1.2 eV, which the paper translates to Q10 of about 1.31-5.13.
  - Falls above the optimum have mean E = 1.15 +/- 0.39 eV.
  - By trait type: "negative motivation" (escape) 0.40 +/- 0.05 eV, positive (consumption, attack) 0.69 +/- 0.09, voluntary 0.64 +/- 0.12, autonomic (e.g. basal metabolism) 0.76 +/- 0.08.
  - Source: [Dell et al. 2011, PNAS (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3127911/).
- **Gillooly et al. 2001:** the model fits metabolic rates of microbes, ectotherms, endotherms and plants over 0-40 C — [Gillooly et al. 2001, Science (abstract via Europe PMC)](https://www.science.org/doi/10.1126/science.1061967). Its expected range of 0.6-0.7 eV is as cited in [Rall et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3479751/).
- **Brown et al. 2004 (MTE):** across-taxa E is about 0.69 eV including endotherms in hibernation/torpor. The working value is E = 0.63 eV ("We use this same value of E = 0.63 eV") with a predicted range of 0.60-0.70 eV. Brown et al. state that "rates of biomass production and consumption" follow individual metabolism — [Brown et al. 2004, Ecology (PDF)](https://michaelkaspari.org/wp-content/uploads/2014/12/2004_brown_metabolism-macarthur-lecture_ecology.pdf).

### Inferences
- Default ectotherm multiplier when no species data exist: E = 0.55-0.65 eV (Q10 about 2.2-2.6 over 3-13 C). Species data from Q1-Q3 put Skagerrak fish and zooplankton at Q10 2.0-2.2, i.e. at the lower end of that default.

### Gaps
- None beyond those noted per taxon.

---

## Q6. Herring winter metabolism, overwintering fasting, and measured winter vs summer rates

### Takeaway
Norwegian spring-spawning herring overwinter without feeding and live on fat reserves. Baltic and young herring cut consumption sharply in autumn and winter (from 20% to 4% of body weight per day in young-of-the-year Baltic herring). No study found measures a *metabolic depression beyond temperature* in adult herring; the winter metabolic drop is attributable to temperature (Q10) and to lower activity.

### Cited Findings
- **NSS herring (Slotte 1999).** Energy utilisation was studied in "non-feeding Norwegian spring-spawning herring" during wintering in a north-Norwegian fjord and the 800 km spawning migration. Energy loss was "attributed mainly to fat metabolism", and storage protein was transferred from soma to gonads. Absolute and relative length-specific energy losses were "two to three and three to four times higher, respectively, during spawning migration than during wintering". Absolute energy loss increased with fish length while relative loss decreased — [Slotte 1999, J. Fish Biol. 54 (abstract via OpenAlex)](https://onlinelibrary.wiley.com/doi/10.1111/j.1095-8649.1999.tb00834.x).
- NSS herring "hardly feed during the energy-demanding overwintering period and subsequent spawning migration" — [ResearchGate summary of Slotte's work](https://www.researchgate.net/publication/227772092_State-dependent_spawning_migration_in_Norwegian_spring-spawning_herring) (via search snippet).
- **Young Baltic herring (Arrhenius & Hansson, ICES CM 1993/L:23).** "The food consumption decreased with body sizes and temperature from 20% of body weight d-1 in the summer to 4% d-1 by the end of October for young-of-the-year." A Wisconsin bioenergetics model "based on data mainly from adult fish, was valid also for young-of-the-year herring" — [Arrhenius & Hansson 1993, ICES CM (VLIZ PDF)](https://www.vliz.be/imisdocs/publications/ocrd/275185.pdf).
- **Baltic herring seasonal feeding:** feeding activity peaks in spring and summer, and empty stomachs are most frequent in winter — [Feeding ecology of central Baltic Sea herring and sprat (ResearchGate)](https://www.researchgate.net/publication/224805653_Feeding_ecology_of_central_Baltic_Sea_herring_and_sprat) (via search snippet; numbers not verified).
- **NSS herring daily consumption over the feeding season:** about 0.05 g prey g fish-1 d-1, i.e. about 6% of body weight per day — [Bachiller et al. 2018](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0190345).
- Winter (4-7 C) vs summer (12-17 C) herring rates: see the Q1 table. In the Rose/Rudstam form a 100 g herring uses about 29-42 kJ/kg/d at 4-7 C versus about 69-91 kJ/kg/d at 12-17 C, a winter/summer ratio of about 0.45. The resting term alone gives 19.6-23.1 versus 30.4-39.9 kJ/kg/d, a ratio of about 0.6. All of these are model-derived (my arithmetic on the [Rose et al. 2008](https://www.pmel.noaa.gov/foci/publications/2008/rose0544.pdf) and [Bachiller et al. 2018](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0190345) parameters).

### Inferences
- The herring winter energy budget follows from three sourced elements: near-zero intake for NSS herring and strongly reduced intake for Baltic herring; a temperature-driven metabolic drop of about 40-55% between summer (12-17 C) and winter (4-7 C) from a Q10 of 2-2.5; and an activity reduction, which only the Rudstam-type model captures.
- A Q10 multiplier of about 2.2 applied to herring maintenance gives a winter (5 C) to summer (15 C) ratio of 0.45. I found no evidence for an additional hypometabolic state in adult herring analogous to Calanus diapause.

### Gaps
- No respirometry of overwintering adult herring.
- Slotte 1999's absolute energy-loss rates (kJ per fish per day during wintering) were not available in the abstract. The full text would give an independent check on the about 30 kJ/kg/d model value at about 5 C.
- No Skagerrak/Kattegat-specific herring winter feeding data were retrieved.

---

## Q7. Does maximum consumption / attack rate scale with the same Q10 as metabolism?

### Takeaway
No. The evidence points two ways depending on the kind of source, and both differ from "same Q10":
- Meta-analyses: Rall et al. 2012 find feeding parameters *less* temperature-sensitive than metabolism (attack E 0.42 eV, handling time -0.30 eV, vs 0.38-0.80 eV for metabolism). Englund et al. 2011 find hump-shaped responses, with attack rate more sensitive than maximum ingestion. Dell 2011 finds consumption-type traits (0.69 eV) similar to autonomic traits (0.76 eV).
- Species-specific Wisconsin parametrisations for herring and cod: consumption *flattens or declines* above about 12-14 C while respiration keeps rising. In the cold, consumption collapses toward 1-3 C faster than the *resting* respiration term. Against herring's full activity-inclusive metabolism the cold-end difference is roughly neutral (see Inferences).

### Cited Findings
- **Rall et al. 2012 (Phil. Trans. R. Soc. B 367:2923).** "We found that the activation energies of attack rates (0.42) and handling times (-0.30) were lower than those found for metabolism (0.38-0.80 for invertebrates, 0.43-0.79 for unicells to mammals). This suggests that feeding increases less strongly with temperature than metabolism" — [Rall et al. 2012 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3479751/). Only these parsimonious-model headline values are reported here; the per-group table columns could not be reliably mapped from the HTML extraction.
- **Englund et al. 2011 (Ecol. Lett., meta-analysis).** In lin-log space, both attack rate and maximal ingestion rate show "a hump-shaped relationship and not a linear one as predicted by the Arrhenius equation", and the relationship "remains significantly downward concave even when data from temperatures above the peak of the hump are discarded". "Temperature dependence is stronger for attack rate than for maximal ingestion rate, but the thermal optima are not different" — [Englund et al. 2011 (abstract via OpenAlex)](https://onlinelibrary.wiley.com/doi/10.1111/j.1461-0248.2011.01661.x).
- **Dell et al. 2011:** positive-motivation traits (consumption, attack) have E = 0.69 +/- 0.09 eV and autonomic traits 0.76 +/- 0.08 eV. Example within-species consumption-rate fits include river perch on phantom midge larvae (E = 0.99 +/- 0.25 eV) — [Dell et al. 2011 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3127911/).
- **Brown et al. 2004 (MTE) assume the opposite:** "rates of consumption and population growth are determined by rates of individual metabolism and have the same body size and temperature dependence" (E about 0.63 eV) — [Brown et al. 2004](https://michaelkaspari.org/wp-content/uploads/2014/12/2004_brown_metabolism-macarthur-lecture_ecology.pdf).
- **Herring Cmax (Wisconsin/Thornton-Lessem, Rose et al. 2008 Table 1).** CMAX intercept aC = 0.642 at the midpoint of te2 and te3, with weight exponent bC = 0.256 (used as negative). Temperature points: te1 = 1 C, te2 = 15/15/13 C, te3 = 17/17/15 C and te4 = 23/25/23 C (age-0/1/2+). Fractions: xk1 = 0.10 at te1, xk2 = xk3 = 0.98 and xk4 = 0.01 — [Rose et al. 2008](https://www.pmel.noaa.gov/foci/publications/2008/rose0544.pdf).

| Temperature | Adult herring Cmax temperature factor (fraction of max) |
|---|---|
| 3 C | 0.235 |
| 5 C | 0.458 |
| 7 C | 0.700 |
| 10 C | 0.914 |
| 12 C | 0.966 |
| 15 C | 0.973 |
| 17 C | 0.852 |
| 18 C | 0.669 |

  (my arithmetic). The equivalent consumption Q10 is 2.1 over 5-15 C, 2.9 over 4-14 C, 4.2 over 3-13 C and 1.7 over 7-13 C.
- **Cod Cmax (Hansson et al. 1996):** CQ = 2.41, CTO = 13.7 C, CTM = 21 C — [Mateo 2007](http://journal.nafo.int/Portals/0/2007/3-mateo.pdf). The Kitchell temperature factor (my arithmetic) is:

| Temperature | Cod consumption factor | Cod respiration factor |
|---|---|---|
| 3 C | 0.51 | 0.45 |
| 5 C | 0.61 | 0.50 |
| 10 C | 0.89 | 0.66 |
| 15 C | 0.98 | 0.84 |
| 18 C | 0.70 | 0.94 |

  The effective consumption Q10 is 1.60 over 5-15 C, 1.97 over 3-13 C, 1.15 over 7-17 C and 0.89 over 8-18 C.
- **Sprat:** "Such a high Q10 of metabolic rates plays a role in defining the temperature niche for optimal growth, especially if the Q10 of maximum consumption rate is lower, as indicated from unpublished results from the authors" — [Meskendahl 2013](https://d-nb.info/1032990570/34). This is unpublished and only indicative.

### Inferences
- For the ecosystem model, applying the metabolism Q10 to consumption or attack rates is the simple MTE assumption (Brown 2004), but the data do not support it as a general rule. Two defensible options:
  1. Scale attack rate a with a lower E (about 0.4 eV, Rall 2012) than maintenance (about 0.5-0.6 eV), and leave or weakly scale the handling-time ceiling.
  2. Use a Thornton-Lessem / Kitchell-shaped Cmax that peaks around 13-15 C for herring and cod.
- With either option, the surplus (intake minus maintenance) shrinks at both ends of the Skagerrak 3-18 C range:
  - **The cold end of this squeeze is conditional on which herring metabolism term is used** (all figures relative to the 15 C value; my arithmetic):
    - Herring Cmax falls to 0.24 at 3 C and 0.47 at 5 C.
    - Against the **resting term only** (RQ 0.0548, Q10 1.73), maintenance falls only to 0.52 at 3 C and 0.58 at 5 C. Intake capacity falls faster than maintenance, so the surplus is squeezed in the cold.
    - Against the **full activity-inclusive Rudstam form** (100 g fish, effective Q10 about 2.5), total metabolism falls to 0.32 at 3 C and 0.40 at 5 C. The cold-end asymmetry is then roughly neutral: at 3 C both fall comparably, and at 5 C Cmax (0.47) is slightly *above* total metabolism (0.40).
    - So "consumption is more sensitive than metabolism in the cold" holds against the resting term, not against total metabolism.
  - The warm end is unconditional: above about 15 C, Cmax declines while respiration keeps rising.
  - This combination, not the metabolic Q10 alone, is what bioenergetics models use to produce summer growth optima. The winter energy loss of NSS herring comes mainly from near-zero realised intake (Q6), not from the Cmax curve.
- Confidence: medium. The direction in the cold differs between meta-analyses (feeding less sensitive) and Wisconsin parametrisations (consumption more sensitive at low temperature), which must be acknowledged.

### Gaps
- No direct experimental Cmax-vs-temperature study for adult Atlantic herring was retrieved. The herring Cmax curve rests on Rudstam 1988 / Megrey 2007 as reproduced by Rose 2008.
- Arrhenius 1998 (food intake and seasonal energy content of young Baltic herring) was not retrieved.
- The species-group breakdown in Rall et al. 2012 (marine ectotherm vertebrates) could not be reliably extracted.

---

## Q8. Typical parameter forms in fish bioenergetics models and values for herring and cod

### Takeaway
Two families are used:
- An exponential exp(RQ*T) (Wisconsin "eq. 1") for herring respiration, with RQ = 0.0548 per C. A swimming term adds exp(dR*U(T)).
- A Kitchell/Thornton-Lessem form, used for herring consumption (Thornton-Lessem) and for both cod consumption and respiration (Kitchell eq. 2: CQ 2.41, RQ 1.88).

### Cited Findings
- Herring (Rudstam/Megrey lineage via Rose et al. 2008 and Bachiller et al. 2018): see Q1 and Q7 for all constants. Respiration R = RA W^RB exp(RQ T) * exp(dR U) with RA 0.0033 g O2/g/d (adult), RB -0.227 and RQ 0.0548. Consumption uses Thornton-Lessem with the 1, 13, 15, 23 C points and fractions 0.10, 0.98, 0.98, 0.01 (adult). SDA 0.175, F 0.16, E 0.10, oxycalorific 13,560 J/g O2 — [Rose et al. 2008](https://www.pmel.noaa.gov/foci/publications/2008/rose0544.pdf); [Bachiller et al. 2018](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0190345).
- Cod (Hansson et al. 1996): Kitchell eq. 2 with V = (TM - T)/(TM - TO) for both consumption (CTO 13.7, CTM 21, CQ 2.41) and respiration (RTO 21, RTM 24, RQ 1.88), and ACT 1.25 — [Mateo 2007](http://journal.nafo.int/Portals/0/2007/3-mateo.pdf). A sensitivity table in the same paper lists perturbed values such as RQ 2.068, CQ 2.65 and CTO 15.7 (+10%).
- The Thornton & Lessem (1978) algorithm is the product of two sigmoids, one for the rising and one for the falling limb, and fits cool- and cold-water species better at low temperature — [Thornton & Lessem 1978, Trans. Am. Fish. Soc.](https://onlinelibrary.wiley.com/doi/abs/10.1577/1548-8659(1978)107%3C284:ATAFMB%3E2.0.CO;2) (via search summary).
- Sprat (for a sprat/herring proxy): M = a e^(cT) with c_RS = 0.078 and c_RR = 0.080; a_RR = 0.074-0.094 mg O2 g-1 h-1 (global fits) — [Meskendahl 2013](https://d-nb.info/1032990570/34).

### Inferences
- Summary of recommended multipliers for an Arrhenius/Q10 maintenance multiplier m(T) = Q10^((T - Tref)/10), valid within 3-18 C:

| Functional group | Recommended Q10 | E (eV) | Basis | Confidence |
|---|---|---|---|---|
| Herring | 2.2 | 0.55 | Sprat/clupeid respirometry | Medium |
| Herring, activity cost also falling in cold | about 2.5 | about 0.6 | Rudstam swimming term | Medium-low |
| Gadoids | 2.2 | 0.55 | Measured cod SMR, 5-15 C; Q10 rises below 8 C | Medium-high for cod |
| Active copepods/krill | 2.0 | 0.48 | Ikeda 2001; Kattegat krill | Medium-high |
| Diapausing Calanus | about 1 (temperature-insensitive) | - | Ingvarsdottir 1999; about 10-20% of active rate | Medium |
| Benthos | about 2 | about 0.5 | General within-species MTE (Dell 2011) | Low (only interspecific benthic data, 1.4, available) |

- **Consumption:** do not reuse the same Q10 by default; see Q7. If a single Arrhenius term is wanted for the attack rate, E about 0.4 eV (Rall 2012) is the best-supported meta-analytic value. Note that all meta-analyses report hump-shaped responses, with optima for herring and cod Cmax around 13-15 C.

### Gaps
- Fish Bioenergetics 4.0 (Deslauriers et al. 2017) parameter tables for Atlantic herring and cod were not accessed. It may contain additional or updated parameter sets (e.g. for Atlantic herring rather than Pacific/Baltic).
