# Whiting growth and temperature: evidence (2026-11-10)

Research only. Nothing in fgconfig/, lib/, tests/, mareld2.yaml or mareld_resume.txt was changed, and
no model was run. Downloads: `data/README.md`. Derived numbers: `scripts/gadoid_proxy_growth.py` ->
`tables/gadoid_proxy_growth_stdout.txt`.

Grades: [F] field data, [L] lab data, [J] judgement or inference, [A] abstract or secondary source only.
A source can carry two grades (e.g. [L] read from an abstract = [L][A]). "Read:" says what was read.

Fish sizes are converted with the whiting length-weight relation W = 0.0042 L^3.12 (g, cm TL;
Wilhelms 2013, via Lauerburg et al. 2018 Table A2 [A]): 10 cm 5.5 g, 15 cm 20 g, 20 cm 48 g, 25 cm
97 g, 30 cm 171 g, 35 cm 276 g. Field check: a 12.1 cm whiting weighed 13.4 g (E17); 115 mm SL = 13.3 g,
215 mm SL = 89 g (E9).

## What the earlier notes already establish (not repeated here)

- `Bioenergetik för alla fiskgrupper`: whiting ingestion_q10 2.25 comes from the gastric-evacuation
  temperature term (coefficient 0.081; Andersen whiting 0.078, Q10 2.18; FB4 cod 2.14). There are no
  whiting Cmax-temperature data and no optimum. Large cod reach their satiation-ration peak at 9.8-11.6 C
  (Bjornsson 2001). The library has metabolism_q10 2.3 on the bottom layer.
- `Tillväxt mot ranson`: the cod growth-ration line is linear to satiation at 6-9 C. Andersen &
  Riis-Vestergaard 2003 (whiting, saithe) found no significant temperature or size effect on the net
  conversion efficiency (abstract only).
- `Tillväxt ur dagens föda`: material_growth_share 0.48 for whiting; feeding season 9 months.
- `Höstens tillväxt...` and `Vitlingens ranson`: in the model, whiting eats about its field ration and
  its fill reaches ~0.98 under reserve_room.

## A. Laboratory growth of whiting

**E1. Jones R, Hislop JRG 1972.** Investigations into the growth of haddock, Melanogrammus aeglefinus
(L), and whiting, Merlangius merlangus (L), in aquaria. J Cons Int Explor Mer 34:174-189. doi
10.1093/icesjms/34.2.174.
- Measured: haddock and whiting fed individually on squid at different rations, one temperature only,
  13-14 C.
- Numbers: weight gain rose more or less linearly with ration. Length growth of the larger fish (26-32
  cm) approached a maximum of 6.5 mm per 10 days in whiting (haddock 4.0 mm per 10 days). Stock
  density had no significant effect.
- Scale: tanks, Aberdeen. Sample size not in the abstract.
- Grade [L][A]. Read: abstract (OpenAlex). No temperature contrast.

**E2. Hislop JRG 1975.** The breeding and growth of whiting, Merlangius merlangus, in captivity. J Cons
Int Explor Mer 36:119-127. doi 10.1093/icesjms/36.2.119.
- Measured: spawning in captivity. Spawning cost females 0.03-0.40 % of body weight per day.
- The abstract gives no growth-temperature data, and the tank temperature is not given.
- Grade [L][A]. Read: abstract (OpenAlex).

**E3. Andersen NG, Riis-Vestergaard J 2003.** The effects of food consumption rate, body size and
temperature on net food conversion efficiency in saithe and whiting. J Fish Biol 62:395-412. doi
10.1046/j.1095-8649.2003.00030.x.
- Measured: growth experiments on whiting and saithe fed natural prey at rations up to satiation.
- Numbers: net conversion efficiency kappa = 0.426 C^0.11 (whiting) and 0.272 C^0.18 (saithe). Kappa is
  the net energy fraction of the ingested energy C.
- Abstract: "No significant effects of temperature and body size on kappa was, however, demonstrated in
  this study except for the indirect influence using feeding levels (rations expressed relatively to
  satiation rations) in bioenergetics models".
- Lauerburg et al. 2018 (E8, Table A2) quote net conversion efficiency = growth / (food intake -
  maintenance ration) = 0.6 from this paper and Temming & Herrmann 2009 [A, secondary].
- Not found: the temperatures tested, fish sizes, n, and growth at satiation per temperature.
- Grade [L][A]. Read: abstract (OpenAlex). The full text is paywalled and has no OA copy.
- This is the only whiting growth experiment found with a temperature factor.

**E4. Ross SD, Nielsen JR, Gislason H, Nielsen A, Andersen NG 2018.** Growth and food consumption of whiting
Merlangius merlangus. J Fish Biol 93:334-343. doi 10.1111/jfb.13763.
- Measured: field consumption in the western Baltic from stomachs and a gastric evacuation model, and from
  an energy budget plus growth.
- Numbers: western Baltic females grow significantly slower than North Sea females, with no significant
  difference for males. Consumption is lowest in Q1-Q2 and highest in Q3.
- Temperature is not separated from area in the abstract.
- Grade [F][A]. Read: abstract (OpenAlex).

**Not found.** A whiting growth-rate series across temperatures at ample food (juvenile or adult). No
whiting thermal optimum for growth, no growth Q10 and no lab preferred temperature. Searched: "Merlangius"
with growth, specific growth rate, temperature, thermal preference, satiation and captivity (web search and
OpenAlex, 2026-11-10). The 2011 larviculture paper (Povoa et al., Aquaculture Research, doi
10.1111/j.1365-2109.2010.02632.x) covers incubation temperature and larval rearing only [A].

## B. Gadoid proxies (lab)

**E5. Bjornsson B, Steinarsson A, Oddgeirsson M 2001.** Optimal temperature for growth and feed conversion
of immature cod (Gadus morhua L.). ICES J Mar Sci 58:29-38. doi 10.1006/jmsc.2000.0986.
- Measured: cod fed to satiation at constant temperatures between 1 and 16 C.
  - Five hatchery groups, 2-109 g, on dry feed.
  - Three wild groups, 447-2213 g, on capelin and shrimp.
- Table 3, G = a + bT - cT^2 (%/d); minus signs restored, so the printed Topt and Gmax reproduce:

  | Experiment | W (g) | Topt (C) | Gmax (%/d) |
  |---|---|---|---|
  | A | 1.5 | 17.0 | 5.63 |
  | B | 11.6 | 16.3 | 3.38 |
  | C | 11.7 | 15.5 | 3.18 |
  | D | 29.4 | 13.2 | 2.23 |
  | E | 109 | 10.1 | 0.77 |
  | F | 447 | 8.8 | 0.70 |
  | G | 874 | 8.1 | 0.63 |
  | H | 2213 | 8.6 | 0.30 |

- "The optimal temperature for growth decreased linearly with fish weight on a log scale from 17C for 2 g
  fish to 7C for 2000 g fish."
- A 2 C departure from Topt changes growth by only 1.4-5.2 %.
- 16 C had no adverse effect on survival of the smallest fish but was lethal over months for the largest.
- Derived [D] (`tables/gadoid_proxy_growth_stdout.txt` s1), satiation growth from 7 C:

  | W (g) | G(13)/G(7) | Q10 7-13 C | G(16)/G(7) |
  |---|---|---|---|
  | 12 | 1.58-1.66 | 2.15-2.32 | 1.67-1.72 |
  | 29 | 1.53 | 2.04 | 1.43 |
  | 109 | 1.03 | 1.05 | 0.49 |
  | 447-2213 | 0.75-0.89 | 0.62-0.82 | 0.33-0.63 |

- Grade [L]. Read: full text (the copy in `../Tillväxt mot ranson/data/bjornsson2001/`).

**E6. Bjornsson B, Steinarsson A 2002.** The food-unlimited growth rate of Atlantic cod (Gadus morhua).
Can J Fish Aquat Sci 59:494-502. doi 10.1139/f02-028.
- Measured: G = 0.5735 T W^(-0.1934 - 0.02001 T) (%/d) for 2-5000 g cod at 2-16 C. Minus signs restored,
  as in `../Tillväxt mot ranson`.
- "Temperature in this size range has a much greater effect on the growth rate of small juvenile cod than
  on that of larger cod." Topt is 14.3 C for 50 g and 5.9 C for 5000 g (abstract).
- Derived [D] (s2), G(13)/G(7):

  | W (g) | G(13)/G(7) | Topt from dG/dT = 0 (C) |
  |---|---|---|
  | 5 | 1.53 | 31 |
  | 10 | 1.41 | 22 |
  | 25 | 1.26 | 15.5 |
  | 50 | 1.16 | 12.8 |
  | 100 | 1.07 | 10.9 |
  | 200 | 0.98 | 9.4 |

- The restored formula gives 12.8 C for 50 g where the abstract says 14.3 C, but it reproduces 5.9 C at
  5000 g. The abstract text is probably incomplete; treat the formula as approximate.
- Grade [L][A]. Read: abstract only.

**E7. Peck MA, Buckley LJ, Caldarone EM, Bengtson DA 2003.** Effects of food consumption and temperature
on growth rate and biochemical-based indicators of growth in early juvenile Atlantic cod Gadus morhua and
haddock Melanogrammus aeglefinus. MEPS 251:233-243. doi 10.3354/meps251233.
- Measured: age-0 fish at 3 fish per tank.

  | Group | Size | Temperatures (C) | Tanks |
  |---|---|---|---|
  | codS | 0.92 g | 5, 8, 12, 15 | 8-14 per temperature |
  | codL | 7.3 g | 5, 8, 12, 15 | 6-17 per temperature |
  | haddock | 3.8 g | 8, 12 | 6-11 |

- Numbers:
  - Cmax was highest at 12 C (4.9-13.0 %/d). "The growth rate and scope for growth were maximum at 12C,
    whereas growth efficiency was greatest (26.0 to 32.2%) at temperatures between 5 and 8C."
  - At 15 C, cod had reduced growth and efficiency (GGEmax 6.2 % codS, 3.6 % codL).
  - Haddock observed Gmax 3.0-3.1 %/d at both 8 and 12 C, with Cmax 9.6-10.5 %/d.
  - Pooled haddock fit: G = -0.3 + 35.1 C - 0.07 T (n 34, R2 0.914; C as printed, evidently a fraction
    per day, not %/d). At a fixed ration the temperature term is small and negative.
- Derived [D] (s3), G at Cmax from the Table 2 fits:

  | Group | 5 C | 8 C | 12 C | 15 C |
  |---|---|---|---|---|
  | codS | 1.1 | 1.8 | 3.8 | 0.5 |
  | codL | 1.35 | 1.05 | 0.62 | 0.24 |
  | haddock | - | 2.65 | 2.49 | - |

  The codL fits fall with temperature, unlike the text's statement, so use the fits for direction only.
- Grade [L]. Read: full text (the copy in `../Fiskgrupper i rutan/benthivores/data/peck2003/`).

**E7b. Sande H, Persson A, Mehl S, Devine JA, dos Santos Schmidt TC, Karlsen O, Godiksen JA, Kjesbu OS
2019.** Temperature and age effects on latitudinal growth dynamics of the commercially valuable gadoid
Northeast Arctic saithe (Pollachius virens). Fish Res 213:94-104. doi 10.1016/j.fishres.2019.01.005.
- Measured: survey size-at-age 1992-2013 at 62-70 N, ages 0-3, plus ad libitum feeding experiments.
- Numbers (abstract): W, length, K and SGR at ages 2-3 are higher at positive local temperature
  anomalies. SGR from age 0 to 1 is highest in the warmer south, but the colder north catches up by age 3.
  This points to "a gradual displacement of response curves for growth towards lower temperatures as saithe
  become larger". Field saithe "show far from optimal growth performance".
- Grade [F][L][A]. Read: abstract (OpenAlex); the full text was bot-blocked.

**E7c. AmP DEB entry, Merlangius merlangus.** T_A 7400 K, T_ref 20 C (Add-my-Pet parameter page).
- Over 7-17 C this implies a rate Q10 of ~2.5.
- In DEB, one Arrhenius factor scales every rate, so growth at f = 1 scales by the same factor.
- This is a model assumption with a fitted or default T_A, not a whiting growth measurement.
- Grade [J][A]. Read: the AmP parameter page (via fetch tool). Listed for completeness only.

## C. Field growth vs temperature

**E8. Lauerburg et al. 2018** (citation in `data/README.md`). Forage fish control population dynamics of
North Sea whiting. MEPS 594:213-230.
- Measured: North Sea whiting length-at-age (IBTS Q1, 1990-2014) against forage fish abundance and
  against Q1 and Q3 North Sea temperature (ICES hydrography), with stomach contents from 6 data sets.
- Numbers:
  - Length-at-age is not described by temperature: GAM adjusted R2 0.08 (Q1 T) and 0.02 (Q3 T).
  - Forage fish abundance explains it: R2 0.58, linear up to a threshold.
  - In the poor-food period 2000-2007, stomach mass was ~60 % lower.
  - Ambient temperature was 0.6 C higher in the poor-food period; the authors treated it as a +4.2 %
    metabolic cost at Q10 2.
  - Q3 temperature rose +0.4 C over 1990-2014, while Q1 temperature stayed almost constant.
- Secondary statements:
  - "In the Irish Sea, the length-at-age of M. merlangus was found to be temperature-independent
    (Gerritsen et al. 2003)".
  - "No significant relationship between temperature in the North Sea and M. merlangus spatial
    distribution has been detected (Kerby et al. 2013)".
  - Within the Black Sea, whiting experience on average 9-20 C (Shulman & Love 1999).
  - Age-1 length-at-age kept falling after 2007, attributed to late-season offspring.
- Grade [F]. Read: full text.

**E9. Potter IC, Gardner DC, Claridge PN 1988.** Age composition, growth, movements, meristics and
parasites of the whiting in the Severn Estuary and Bristol Channel. JMBA 68:295-313. doi
10.1017/S0025315400052206.
- Measured: power-station intake screens, 1972-1977.
- Numbers:
  - 0+ fish enter in July at >= 38 mm SL. One cohort grew from 58.3 mm (2.1 g) in August to 114.9 mm
    (13.3 g) in November, then grew little over winter.
  - Late-autumn 0+ mean length is ~20 mm greater in the deeper Inner Bristol Channel than in the shallow
    Inner Severn Estuary. The authors attribute this partly to the larger fish of a cohort moving first
    into deeper water, so it is not a temperature contrast.
  - Late-autumn 0+ mean TL: ~140 mm at Hinkley Point; 77-125 mm (September) on the English east coast;
    larger in western Scottish sea lochs; ~190 mm off Brittany (Desbrosses 1945, secondary).
- Temperature is not measured or separated; the latitudinal contrast confounds temperature, season length
  and food.
- Grade [F]. Read: full text.

**E10. Hunter A, Speirs DC, Heath MR 2016.** Investigating trends in the growth of five demersal fish
species from the Firth of Clyde and the wider western shelf of Scotland. Fish Res 177:71-81.
- Measured: survey length-at-age 1980-2012, with von Bertalanffy L-infinity and k per cohort regressed on
  cohort, year-class strength and the mean summer SST of the first three years.
- Numbers (Table 4):
  - Clyde whiting L-infinity vs temperature: -3.19 (SE 3.69) cm per C, not significant.
  - Western shelf whiting k vs temperature: -0.15 (SE 0.095) /yr per C, not significant.
  - "Trends in temperature and year class strength tended to contribute little"; size-selective
    (Nephrops) fishing is the favoured cause.
- Grade [F]. Read: full text.

**E11. Baudron AR, Needle CL, Rijnsdorp AD, Marshall CT 2014.** Warming temperatures and smaller body
sizes: synchronous changes in growth of North Sea fishes. Glob Change Biol 20:1023-1031. doi
10.1111/gcb.12514.
- Measured: L-infinity by cohort, 1970-2008, for 8 species split into N and S sub-stocks. A dynamic
  factor analysis gives the common trend, compared with NORWECOM sea bottom temperature (SBT).
- Numbers:
  - L-infinity fell from 1973-77 to 1993-97: whiting North 13 %, whiting South 29 % (mean 16 % over 9
    sub-stocks), as SBT rose 1-2 C.
  - Trend 1 correlates with SBT at age 0 (r -0.43), age 1 (-0.45), age 2 (-0.49), ages 0-1 (-0.49) and
    ages 0-2 (-0.54); all significant after Bonferroni.
  - Yield-per-recruit loss: whiting North 3.1 %, whiting South 48.1 %.
- The authors say it is not causal proof (fishing selection and density not excluded). They also note that
  warming usually raises juvenile k while lowering L-infinity.
- This is asymptotic size of adults, not juvenile growth at ample food.
- Grade [F]. Read: full text (accepted manuscript).

**E12. Cali F, Stranci F, La Mesa M, Mazzoldi C, Arneri E, Santojanni A 2023.** Whiting (Merlangius merlangus) grows slower and smaller in the Adriatic Sea:
new insights from a comparison of two populations with a time interval of 30 years. Fishes 8:341. doi
10.3390/fishes8070341.
- Measured: commercial bottom trawl samples, 1990-91 vs 2020-21, at the species' southern limit.
- Numbers: L-infinity fell from 29.5 to 22.8 cm TL. Maturity is reached within year 1 at ~16-17 cm.
  The authors "hypothesize" a metabolic constraint from warming.
- Grade [F][A]. Read: abstract (OpenAlex); the full text was bot-blocked.

**E13. Kutsyn DN 2024.** Life history traits of the Black Sea whiting Merlangius
merlangus (Gadidae): compliance with the temperature-size rule. Environ Biol Fishes 107:91-105. doi
10.1007/s10641-024-01513-6.
- Title only; the abstract was not obtained. Grade [A]. No numbers used.

**E14. Hamerlynck O, Hostens K 1993** (authors per the KNAW repository record in OpenAlex). Growth, feeding, production, and consumption in 0-group bib and
whiting in a shallow coastal area of the south-west Netherlands. ICES J Mar Sci 50:81-91. doi
10.1006/jmsc.1993.1009.
- Measured: monthly fine-mesh sampling in 1988.
- Numbers: 0-group whiting arrive in May and reach ~170 mm SL by the end of the first season; most have
  left by the end of the year. Consumption (Jones model) is 4-6 % body weight per day in summer and 2-3 %
  in autumn. The diet is copepods in May, then mysids, amphipods and gobies, then shrimp and fish.
- No temperature analysis in the abstract.
- Grade [F][A]. Read: abstract (OpenAlex); the full text was bot-blocked.

**E15. Saulnier E et al. 2020.** Food limitation of juvenile marine fish in a coastal and estuarine
nursery. ECSS 241:106670.
- Table 1 uses 0-group whiting G 0.079 mm/d, gross conversion 0.385 and Z 0.0076 /d (Seine estuary,
  taken from earlier studies). Whiting and bib were < 5 % of fish consumption.
- Grade [F, secondary values]. Read: full text (accepted manuscript). Context only.

**Not found.**
- A whiting otolith growth chronology (increment widths vs temperature).
- A Skagerrak/Kattegat whiting length- or weight-at-age vs temperature analysis.
- 0-group whiting growth vs nursery temperature with food held apart.

## D. Thermal habitat use

**E16. Burns NM, Bailey DM, Wright PJ 2019.** A method to improve fishing selectivity through age targeted
fishing using life stage distribution modelling. PLoS ONE 14:e0214459. doi 10.1371/journal.pone.0214459.
- Measured: GAMMs of survey CPUE west of Scotland against near-bottom temperature (ICES CTD), depth,
  sediment, slope and distance from shore.
  - Age-0 from October-November surveys, 2011-2014.
  - Age-1 and mature fish from February-March surveys, 2009-2015.
- Numbers:
  - Age-0: "The greatest densities of age-0 whiting occurred over finer sediments where temperatures were
    between 12 to 13C"; densities decrease with depth. 80-96 % of age-0 fish are in 20 % of the occupied
    space.
  - Age-1 (Feb-Mar): densities peak at 60 m, shallower (50 m) at lower temperature. 8-8.5 C appeared
    optimal.
  - Mature fish: higher abundance up to 9.7 C offshore; no temperature effect near shore.
- Food is not in the models. 12-13 C is near the warm end of the Oct-Nov bottom range, but temperature,
  depth and distance from shore co-vary.
- Grade [F]. Read: full text (Europe PMC).

**E17. Tamarit-Castro E, Steinbrecher F, Pfeufer L, Lechner ER, Sauer H, Green L, Perry D, Linderholm HW,
Gullström M, Jutfelt F 2026.** Fishes in eelgrass meadows show large interspecific differences in thermal
acclimation to marine heatwaves. Glob Change Biol Commun. doi 10.1002/gcb4.70027.
- Measured: wild juvenile whiting from eelgrass near Kristineberg (Gullmarsfjord, Skagerrak coast).
  - Caught by beach seine and similar gear, 15 July - 23 September, when the water was 18.9 +- 1.2 C.
  - Size 121 +- 9 mm TL, 13.4 +- 3.3 g; n 12 per treatment.
  - CTmax measured after 5 d at 19 or 23 C.
- Numbers:
  - Gadid CTmax at 19 C: 28.6 +- 0.4 C (cod 27.8 C is the lowest of the 12 species).
  - At 23 C, 58.3 % of the juvenile whiting died during the 5-day exposure; survivors had CTmax 29.9 C.
    Acclimation was negligible.
  - In-situ eelgrass daily maxima average 20.5 C (15 Jun - 1 Sep).
  - "Preferred temperatures in the range of ~12-18C" for juvenile cod and whiting comes from AquaMaps
    envelopes, i.e. a distribution model, not measured preference [A].
  - The authors suggest juvenile gadids "may already avoid shallow eelgrass habitats during warm periods"
    (inference, not measured).
- The field fact is that 0-group whiting were present in 1-2 m eelgrass at ~19 C in the box's own coast
  in late summer.
- Grade [F][L]. Read: full text.

**E18. Henderson PA, Holmes RHA 1989.** Whiting migration in the Bristol Channel: a predator-prey
relationship. J Fish Biol 34:409-416. doi 10.1111/j.1095-8649.1989.tb03323.x.
- 0-group whiting stay in the estuary ~10 months, then move offshore. Distribution and abundance track
  the shrimp Crangon crangon, the only abundant winter prey.
- Movement is explained by food, not temperature.
- Grade [F][A]. Read: abstract (OpenAlex).

**E19. Asciutto E, Maioli F, Manfredi C, Anibaldi A, Cimini J, Isajlovic I, Marceta B, Casini M 2024.** Spatio-temporal patterns of whiting (Merlangius merlangus) in the
Adriatic Sea under environmental forcing. PLoS ONE 19:e0289999. doi 10.1371/journal.pone.0289999.
- Measured: MEDITS survey 1999-2019, spring-summer, GLMMs.
- Numbers: preference ~45 m depth, ~15.4 C bottom temperature, O2 > 5.5 ml/l. Temperature cut predicted
  density by about two thirds in the northern area. Mostly adults, at the southern edge of the range.
- Grade [F][A]. Read: abstract (OpenAlex).

**E20. Dagtekin M et al. 2026** (OpenAlex lists one author). Uncovering spatial habitat partitioning of whiting
(Merlangius merlangus) recruits and adults in the southern Black Sea. Fishes 11:112. doi
10.3390/fishes11020112.
- Measured: autumn 2024, 66 hauls at 10-125 m, GAMs.
- Numbers: recruits are concentrated at 40-75 m within 10-15 C; adults extend beyond 100 m into the Cold
  Intermediate Layer. Depth x temperature explains 74.7 % (recruits) and 69.5 % (adults) of deviance.
- Grade [F][A]. Read: abstract (OpenAlex).

**E21. Kerby TK, Cheung WWL, van Oosterhout C, Engelhard GH 2013.** Wondering about wandering whiting:
distribution of North Sea whiting between the 1920s and 2000s. Fish Res 145:54-65. doi
10.1016/j.fishres.2013.02.012.
- No clear north-south or deepening shift over the century; a ~1 degree westward shift (1940s-1960s).
  Weak links to climate.
- Grade [F][A]. Read: search summary of the abstract plus Lauerburg's citation.

**E22. Freitas C, Olsen EM, Moland E, Ciannelli L, Knutsen H 2015.** Behavioral responses of Atlantic cod
to sea temperature changes. Ecol Evol 5:2070-2083. doi 10.1002/ece3.1496.
- Proxy (cod, ~30-80 cm), Skagerrak coast, acoustic telemetry.
- Cod moved deeper as summer surface temperature rose. Small cod rose to shallow feeding areas at night,
  but above ~15 C surface temperature they stayed deeper. A follow-up reports avoidance of shallow water
  > ~16 C.
- Grade [F][A]. Read: search summary of the abstract.

**E23. Zheng X, Pierce GJ, Reid DG (and co-authors) 2001-2002.** Whiting abundance in Scottish and
northern North Sea waters is higher where the water is relatively warm, probably through Atlantic inflow.
- Grade [A]. Read: search snippet only; no numbers used.

**Not found.**
- A lab thermal preference (shuttle box or gradient) for whiting of any size.
- A tagging study of juvenile whiting depth vs temperature.
- Any study that weighs temperature-driven growth against predation risk or food in the choice of shallow
  nurseries by whiting.
