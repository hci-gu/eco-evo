# Krill escape and energy budget: evidence trail (2026-11-10)

Research only. Nothing in the library, code, tests, manifest, resume or AGENTS.md was changed. No
model output set or selected any value here. ASCII only (Swedish letters transliterated: a-ring ->
a, a-umlaut -> a, o-umlaut -> o).

Access tags: [F] full text read (local copy in data/ or read on the publisher page), [A] abstract or
repository record only, [P] project-internal note or script, [S] secondary (a review or a source quoted
by another source).

Downloads: data/<paper>/paper.pdf, paper.txt (pdftotext -layout), paper_raw.txt (pdftotext), SOURCE.txt
(DOI, URL, licence as Unpaywall reports it). Licences: data/README.md. OpenAlex abstracts:
data/openalex_abstracts_*.txt (scripts/oa_abstracts.py).

---------------------------------------------------------------------------------------------------

## Part 0. How the krill pairs' a was derived (project notes) [P]

E1. research_notes/"Krill i Skagerrak"/pairs/scripts/krill_pairs.py (2026-11-07), docstring and
parameter table. The same lines are carried unchanged into wc_depth/scripts/krill_pairs_wc.py (lines
29, 349, 497) and through it into "Krill over grunda bottnar"/rederive/scripts/krill_geometry.py (the
section 262 values).
- Formula (docstring, line 19):
  "a_k(c, t) = scan pi p_cap T / 1e6 * sum_z p_k(z) phi(z) * sum_fish n v_rel R(z, t)^2   [per t of k]"
- Capture (line 284):
  "p_cap": (0.3, 0.5, 0.7),           # krill capture (220 p_krill; tail-flip escape) [J]
- Searched field (line 285):
  "scan": (0.25, 0.5, 1.0),           # half the visual disc (Varpe & Fiksen 2010) [F/J]
- Vertical overlap / DVM (lines 300-306):
  "zc": (110.0, 95.0, 85.0),          # day layer centre over beds <= 150 m (80-100 m at 134 m) [F/J]
  "Hk": (40.0, 30.0, 20.0),           # day layer thickness (Onsrud & Kaartvedt 1998: 20-40 m) [F]
  "ztop_min": (60.0, 60.0, 60.0),     # never above ~60-70 m by day (Onsrud & Kaartvedt 1998) [F/J]
  "m_mig": (0.35, 0.45, 0.60),        # migrating share at night (Tarling 1999: 38-48 % above 65 m) [D]
- Dark detection floor (line 277):
  "R_nv": (0.05, 0.10, 0.20),         # non-visual / bioluminescence floor on krill (220 R_krill) [J]
- Prey speed (line 286): "u_krill": (0.04, 0.04, 0.04),      # m/s in situ mode (220)
- Evaluation (line 409): base = q["scan"] * math.pi * q["p_cap"] * T / 1e6

E2. research_notes/"Krill i Skagerrak"/pairs/evidence.md.
- Line 59: "| capture p on krill | 0.5 (0.3-0.7) | J | 220 p_krill (Langbehn pc 0.7; tail-flip escape) |"
- Lines 231-232: "Skagerrak krill occur as layers, not swarms, so the dispersed-layer encounter applies
  [J]; no school form was run."

E3. research_notes/"Attackrater ur motesteori" (o-umlaut)/predators_encounter.md, section 2.5:
"Krill 0.5 (0.3-0.7; J, Langbehn's piscivore pc 0.7 [F]); copepods 0.7 (0.5-0.9); bed invertebrates
0.4 (0.2-0.7; J)." Single fish prey: "Single fish 0.2 (0.1-0.36)". Section 2.4 (line 136-139): "Krill:
0.1 m (0.05-0.2) day and night (J). ... krill sit below 70 m by day (Onsrud & Kaartvedt 1998 [F]) where
vision is near its floor; bioluminescence and the non-visual floor (0.1 m) carry it."

E4. research_notes/"Vattenpelarens par i uppmatt ljus" (a-umlaut)/summary.md, section A2 (a variant,
not applied): "| krill | 0.5 | 0.6 (0.4-0.8) | x1.2 | [J]: below planktivores' 70-80 % on copepods
(tail-flip escape); no gadoid-on-euphausiid value found |" and "| fish | 0.2 | 0.45 (0.25-0.6) | x2.25 |
Juanes et al. 2002 p. 271-272 [F review]: "about half of piscivore attacks on average are successful
compared with 70-80% for planktivorous fishes"".

E5. research_notes/"Fodobegransning" (o-umlaut, a-umlaut)/fish_krill/evidence.md 3.3: "No field or lab
functional response of any fish on euphausiids was found" and "The krill pairs already carry p = 0.5
(0.3-0.7) inside a". Same section on Christiansen 2024 [A]: diving "acts on the encounter (a) and on
local density, not on the daily ceiling."

E6. research_notes/"Krill i Skagerrak"/a_revisit/summary.md 1.1: "a does not set the ration. Both
pairs are saturated at the box krill (1/(a h) 0.14 and 0.28 t/km2 against 5; sat 0.96 and 0.94). Even
a 5x lower a keeps sat > 0.8." pairs/summary.md: mackerel "sat at 5 t/km2" 0.48, "1/(a h) 5.4 t/km2";
planktivores 0.84, benthivores 0.92, water column 0.95, whiting 0.97.

E7. Booked krill M2 by predator (AGENTS.md 4, the library as of section 262; the ratios of the brief are
read against these): planktivores 1.67, whiting 0.187, water column 0.119, mackerel 0.10, benthivores
0.035 /yr; sum 2.11 /yr.

---------------------------------------------------------------------------------------------------

## Part 1. Capture success, escape and swarming

E8. Kaartvedt S, Rostad A, Fiksen O, Melle W, Torgersen T, Breien MT, Klevjer TA (2005). Piscivorous
fish patrol krill swarms. Mar Ecol Prog Ser 299:1-5. doi:10.3354/meps299001. [F] data/kaartvedt2005_meps.
- Norwegian Sea, ~70 N 4 E, 24-25 May 2004; swarms of M. norvegica at 100-200 m.
- "The major benefit of swarming is generally presumed to be protection from predation, mainly derived
  from evasion and dilution factors once an attack is launched (e.g. O'Brien & Ritz 1988). However,
  swarms may also attract and make krill vulnerable to predators capable of exploiting such dense
  concentrations (Nicol & O'Dor 1985, Ritz 1994)."
- Swarm size: "~380 000 m3, and numerical densities at ~200 ind. m-3, suggesting a total biomass of ~11 t
  per swarm."
- "Two trawl tows between 200 and 300 m depth captured four 6 to 10 kg (60 to 110 cm) large saithe
  Pollachius virens" ... "saithe captured in the trawl had fish and the cephalopod Gonatus sp. in their
  stomachs. Apparently, the saithe preyed on planktivores attracted by the krill, rather than on the
  krill itself."
- Blue whiting (~25 cm) "were the most likely deep-living predators on krill and prey for saithe."

E9. Christiansen S, Brathen MP, Moksnes FL, Schmedling CA, Thorsby SJ, Titelman J, Kaartvedt S (2024).
Extensive avoidance behaviour of krill from predators and nets. ICES J Mar Sci 81(9):1764-1773.
doi:10.1093/icesjms/fsae123 (CC BY).
[F, read on the publisher page; quotes as returned, fragments]. Oslofjord, M. norvegica, bottom-mounted
echosounders (WBAT), herring schools.
- Abstract: "During daytime, krill responded to fish schools with instantaneous coherent diving",
  "spanning tens of meters at speeds of 10-27 cm s-1".
- "Diving speeds based on these echo traces ranged between 10 and 27 cm s-1 with an average of 19 +- 5
  cm s-1" (6 echo traces, 8 and 10 April 2021); "the upper fringe of the krill layer descending 20 m".
- 226 fish schools extracted, 164 by day. Large schools (> 10 m high, n = 7): "In >75% of the daytime
  encounters with large fish schools" krill density fell by > 55 %, median 68 % (~6 dB). Schools 5-10
  m high (n = 41): density fell by > 40 % in 25 % of encounters, median ~12 % (2.7 dB). Schools < 5 m:
  "did not induce any measurable change in krill density."
- Night: "Density changes at night centred around 0, indicating limited or no influence of fish
  schools."; "some krill avoided layers richer in fish by moving towards intermediate depths."
- Voids "typically lasted 5-15 min. for the larger fish schools."
- "effective communication appeared at densities >0.2 ind. m-3"; less effective below ~0.15 ind m-3.
- Gut contents of a September 2020 trawl: krill in herring (also saithe and whiting).

E10. Onsrud MSR, Kaartvedt S, Rostad A, Klevjer TA (2004). Vertical distribution and feeding patterns in
fish foraging on the krill Meganyctiphanes norvegica. ICES J Mar Sci 61:1278-1290.
doi:10.1016/j.icesjms.2004.09.005. [F, read on the publisher page]. Oslofjord, 120 m site.
- Abstract: "Herring (Clupea harengus), whiting (Merlangius merlangus), and Norway pout (Trisopterus
  esmarkii) were foraging on krill ... during both day and night." "Krill antipredator behaviour
  comprised diel vertical migration and instantaneous escape reactions, and the krill also appeared to
  actively seek out strata with low acoustic recordings of fish."
- Discussion: "suggest that the krill performed instantaneous escape reactions during day"; "It
  appeared that M. norvegica would seek out water with reduced fish abundance at night". No escape
  speeds, distances, predation rates or capture success are reported.
- Whiting: "krill constituted up to 99% of the stomach content by weight, on average ~82% during both
  day and night".
- Table 3, % of stomachs with krill, day / night; krill per stomach day / night:
  whiting Nov 2000 42.3 / 35.2 %, 1.82 / 3.47; Mar 2001 75.5 / 79.7 %, 3.95 / 5.92; Jun 2001 42.9 /
  34.6 %, 2.17 / 1.44.
  Norway pout Nov 59.1 / 44.1 %, 1.54 / 1.54; Mar 75.0 / 47.6 %, 2.17 / 2.10.
  Herring Nov 31.0 / 17.2 %, 1.90 / 2.20; Mar 50.0 / 20.0 %, 5.13 / 1.50.
  "neither stomach fullness nor state of digestion revealed clear diel patterns in food uptake".
- "The krill layer was restricted to waters deeper than ca. 70 m during the day, mainly concentrated in
  a 20-40-m thick layer".

E11. Onsrud MSR, Kaartvedt S, Breien MT (2005). In situ swimming speed and swimming behaviour of fish
feeding on the krill Meganyctiphanes norvegica. Can J Fish Aquat Sci 62:1822-1832.
doi:10.1139/f05-090. [A] data/openalex_abstracts_1.txt.
- "Swimming speeds of deep-living fish, mainly Norway pout (Trisopterus esmarkii) and whiting
  (Merlangius merlangus), were highest during the day (speeds centred around 14-16 cm s-1) and decreased
  somewhat by night (modes around 10-11 cm s-1). Fish in the upper 10-30 m swam somewhat faster (speeds
  ranging from 16 to 24 cm s-1)." (dashes lost in the OpenAlex text and restored from the ranges.)
- "We ascribe the reduction of swimming speed in deep-living fish at night to a switch from visual
  feeding during daytime to nonvisual feeding by night."

E12. O'Brien DP (1987). Description of escape responses of krill (Crustacea: Euphausiacea), with
particular reference to swarming behavior and the size and proximity of the predator. J Crustacean
Biol 7(3):449-457. doi:10.2307/1548294. [A]
- "It is proposed that there are three levels of escape response whose manifestation is primarily
  dependant upon the aggregative state of the prey and the size and proximity of the predator." No
  numbers in the abstract; full text paywalled (not obtained).

E13. Connor AA, Webster DR (2023). Hydrodynamics of the fast-start caridoid escape response in
Antarctic krill, Euphausia superba. Sci Rep 13:5376. doi:10.1038/s41598-023-31676-8. [F, PMC10068603,
read on the page]
- "The specimen performs a single abdominal flexion-tail flip combination that leads to an acceleration
  over a 42 ms interval" ... "allowing it to reach a maximum speed of 57.0 cm/s (17.3 body lengths/s)."
  (one animal, ~3.3 cm implied). Peak acceleration 273 m/s2. Mean post-escape speed 45 cm/s (13.6 BL/s).
- Citing Kils (1982) [S]: "the maximum speed during one tail flip was measured to be 35-100 cm/s",
  "the average swimming speed can be around 50 cm/s over a distance of at least 50 cm".

E14. Foster EG, Ritz DA, Osborn JE, Swadling KM (2001). Schooling affects the feeding success of
Australian salmon (Arripis trutta) when preying on mysid swarms (Paramesopodopsis rufa). J Exp Mar Biol
Ecol 261:93-106. doi:10.1016/S0022-0981(01)00265-9. [A, figshare record 23216834; data/foster2001_figshare.json]
- "juvenile Australian salmon (Arripis trutta) had higher rates of successful attacks when foraging in
  a group of six fish (55% total advances) than when foraging alone (39% total advances). Six schooling
  fish had lower approach rates than solitary fish (25% and 37% of total advances, respectively). This
  result indicated that schooling fish were better at reducing the confusion effect of swarming prey".
- "There was no influence on the feeding success of individual fish when changes were made to the
  number of prey presented to each fish."
- Denominator (inference): the percentages are of "total advances". If the advances not counted as
  successful attacks or approaches (100 - 55 - 25 = 20 %; 100 - 39 - 37 = 24 %) are failed attacks,
  success per attack is 55/75 = 0.73 (groups of six) and 39/63 = 0.62 (solitary). Per advance it is
  0.55 and 0.39. The abstract does not define the categories; the full text was not obtained.

E15. Flynn AJ, Ritz DA (1999). Effect of habitat complexity and predatory style on the capture success
of fish feeding on aggregated prey. J Mar Biol Assoc UK 79:487-494. doi:10.1017/S0025315498000617. [A]
- "Cohesive escape responses of a large swarm are likely to be a more effective defence against a
  lunging predator than against an ambush predator". Disrupted swarms leave "a large number of more
  vulnerable small groups" (salmon: proportion of unsuccessful attacks fell with vegetation density).

E16. Ritz DA, Osborn JE, Ocken AEJ (1997). Influence of food and predatory attack on mysid swarm
dynamics. J Mar Biol Assoc UK 77:31-42. doi:10.1017/S0025315400033762. [A]
- Model lunging predator at "15 cm s-1": "Approach of the salmon was evidently detected from further
  away, escape reactions were frequent and energetic, and tailflipping was common." Ambush (seahorse,
  0.5-2 cm s-1): "No tailflips were seen among the mysids."

E17. Klevjer TA, Kaartvedt S (2011). Krill (Meganyctiphanes norvegica) swim faster at night. Limnol
Oceanogr 56:765-774. doi:10.4319/lo.2011.56.3.0765. [A]
- Oslofjord, 150 m, late autumn to spring: day and night "~3.5 cm s-1 (~1 body lengths ([bl] s-1)" at
  15-20 % O2 saturation; after renewal "nocturnal average swimming speeds increased to ~10 cm s-1 (~3 bl
  s-1), i.e., more than double that of daytime swimming speeds". Authors: "enhanced swimming at night
  ... is due to increased feeding activity under lessened danger of predation in darkness."

E18. Kaartvedt S, Melle W, Knutsen T, Skjoldal HR (1996). Vertical distribution of fish and krill
beneath water of varying optical properties. Mar Ecol Prog Ser 136:51-58. doi:10.3354/meps136051. [F]
- Norwegian shelf, ~300 m: krill (mainly Thysanoessa inermis) below Maurolicus and above Norway pout;
  across a turbid front "Mesopelagic fish ascended by about 100 m accompanied by a rise of krill.
  Demersal fish left the benthic boundary zone, with ascending Norway pout foraging in the lower part of
  the krill layer. We suggest that the intermediate light conditions inside the front provided an
  'antipredation window'".

E19. De Robertis A (2002). Size-dependent visual predation risk and the timing of vertical migration:
an optimization model. Limnol Oceanogr 47:925-933. doi:10.4319/lo.2002.47.4.0925. [A]
- Euphausia pacifica, Saanich Inlet: the model "predicts that 7-mm (larval) animals will ascend/descend
  ~35 min earlier/later than more conspicuous 24-mm adults. These predictions are consistent with
  acoustic observations of the timing of DVM in this population."

E20. Juanes F, Buckel JA, Scharf FS (2002). Feeding ecology of piscivorous fishes. In: Hart PJB,
Reynolds JD (eds) Handbook of Fish Biology and Fisheries, Vol 1, Blackwell, p. 267-283. [S via E4, E5;
the project copy is in research_notes/"Fodobegransning"/fish_krill/data/juanes2002/, never commit]:
"Usually about half of piscivore attacks on average are successful compared with 70-80% for
planktivorous fishes".

---------------------------------------------------------------------------------------------------

## Part 2. Bioenergetics

E21. Saborowski R, Brohl S, Tarling GA, Buchholz F (2002). Metabolic properties of Northern krill,
Meganyctiphanes norvegica, from different climatic zones. I. Respiration and excretion. Mar Biol
140:547-556. doi:10.1007/s00227-001-0730-4. [F, AWI repository submitted version]
data/saborowski2002_mb140_resp.
- "Oxygen consumption increased exponentially with increasing experimental temperature, which ranged
  from 4C to 16C. Overall respiration rates were between 19.9 and 89.9 umol O2 g-1 dry wt h-1."
- "Krill from the Kattegat, the Clyde Sea, and the Ligurian Sea all exhibited approximately the same
  level of oxygen consumption (30-35 umol O2 g-1 dry wt h-1) when incubated at the ambient temperatures
  found in their respective environments". (The abstract gives the ambient temperatures as 9, 5 and 12
  C in that order; the results text reads "12C in the Ligurian, 9C in the Clyde, and 5C in the
  Kattegat". The two disagree for Kattegat/Clyde.)
- Clyde: "23.2 umol O2 g-1 dry wt h-1 at 4C to 58.9 ... at 16C". Kattegat: "33.3 ... at 4C and 73.8
  ... at 16C"; no seasonal difference in Clyde or Kattegat krill.
- "The Q10-values of respiration increase for the temperature increments studied ranged between 1.55
  (4-8C) in Kattegat summer krill and 2.46 (12-16C) in summer krill from the Clyde Sea (Table 3).
  Q10-values calculated over the entire temperature range from 4C to 16C ranged between 1.87 and 2.12."
- Ammonia excretion, Kattegat: "1.06 umol NH4-N g-1 dry wt h-1 at 4C to 6.65 ... at 16C".

E22. Bamstedt U, Karlson K (1998). Euphausiid predation on copepods in coastal waters of the Northeast
Atlantic. Mar Ecol Prog Ser 172:149-168. doi:10.3354/meps172149. [F] data/bamstedt1998_meps.
Skagerrak (G.O. Sars cruise 1991) and NW Norway.
- "M. norvegica preying on Calanus finmarchicus had similar predation rates in both laboratory studies
  and field samples (average 1.2 copepods ind.-1 h-1, 12.7% daily ration) and showed that ... (2) there
  was no diel predation rhythm, and (3) ca 20% of copepods caught by the predator were only partly
  eaten".
- Daily ration defined "as the percentage of the predator body mass, on a dry-weight basis". Lab: "The
  average predation rate from all experiments was 1.2 copepods ind-1 h-1 (SD = 0.86, n = 28). The
  corresponding daily ration ranged from 0 to 33 %, ... with an average value of 12.7% (SD = 8.06)."
  Prey 1.2-2 l-1 in 45-90 l tanks "usually gave daily rations exceeding 10%".
- Digestion time "3.35 +- 1.74 h"; 2.15 h without two outliers.
- Field: "a total range in predation rate from zero to 10 copepods ind.-1 h-1 and a corresponding daily
  ration from zero to 125%"; per-station averages "1 to 20 %".
- "Carnivory was less important in the Skagerrak than in the northern area for all 3 species occurring
  in both areas. T inermis from the Skagerrak seldom fed on copepods"; "T. raschii usually had high
  stomach fullness but almost never took copepod prey." "C. finmarchicus was by far the most dominant
  prey, making up 85% of the copepod prey mass for small M. norvegica (<50 mg dry weight) and 93 to 95%
  for the larger M. norvegica."
- Share of krill feeding carnivorously: "respectively 93, 69, 35 and 6 %" (M. norvegica, T. inermis,
  T. longicaudata, T. raschii). "average degree of carnivory was 40 % or less".
- Kosterfjorden (Swedish Skagerrak coast): M. norvegica "has an annual biomass ranging from zero to ca
  1600 mg DW m-2" (Bamstedt 1988).

E23. Tarling GA (2003). Sex-dependent diel vertical migration in northern krill Meganyctiphanes
norvegica and its consequences for population dynamics. Mar Ecol Prog Ser 260:173-188.
doi:10.3354/meps260173. [F] data/tarling2003_meps. Clyde Sea.
- Arrival/departure of the scattering layer "consistently 30 min after sunset and 30 min before sunrise".
- "the daily mortality rate (beta) of the entire adult population would have been 0.003 d-1, of which
  the rate for females would have been 0.004 d-1 and that for males 0.002 d-1".
- Energy model (summer, Clyde): "males assimilating 25 J d-1, while females assimilated 35 J d-1";
  respiration "differed very little between sexes". Female DVM gives "an extra 40%, or 10 J d-1".
- Growth: modal adult length "remained at ca. 34 to 35 mm in both sexes" in summer-autumn; "2 mm smaller"
  after winter ("size-dependent mortality or shrinkage"); "increasing from 22 mm to 31 mm between early
  March and early May".
- Ovary 1147 eggs, 18 mg WW, ~115 J; spawning every 22 d; 5.2 J d-1 for eggs.

E24. Sameoto DD (1976). Respiration rates, energy budgets, and molting frequencies of three species of
euphausiids found in the Gulf of St. Lawrence. J Fish Res Board Can 33:2568-2576. doi:10.1139/f76-301.
[A] "The number of calories ingested, expressed as a percentage of the total body calories of the
euphausiid, decreased with body size and water temperature. Thysanoessa inermis ingested fewer calories
than M. norvegica in relation to total body calories at all temperatures." (Artemia nauplii as food; the
numbers were not obtained.)

E25. Sameoto DD (1980). Relationships between stomach contents and vertical migration in
Meganyctiphanes norvegica, Thysanoessa raschii and T. inermis. J Plankton Res 2:129-143.
doi:10.1093/plankt/2.2.129. [A] "M. norvegica fed on copepods while at their daytime depth of about 125
m and on phytoplankton and copepods during the night above 75 m depth." "No evidence was found of any of
the species feeding on bottom deposits".

E26. Kaartvedt S, Larsen T, Hjelmseth K, Onsrud MSR (2002). Is the omnivorous krill Meganyctiphanes
norvegica primarily a selectively feeding carnivore? Mar Ecol Prog Ser 228:193-204.
doi:10.3354/meps228193. [F] data/kaartvedt2002_meps. Oslofjord 120 and 200 m sites.
- "the highest carbon gain was from carnivorous feeding, although contribution from herbivorous feeding
  was of the same order of magnitude in spring. High daytime gut fluorescence during spring suggested
  that sedimenting algae were efficiently cropped in deep water." "In late summer, M. norvegica acted as
  a selectively feeding carnivore, primarily preying on Temora longicornis during nocturnal migrations
  to upper layers. Algal food was neglected at this time." Winter: "the ability of M. norvegica to
  visually detect and exploit overwintering Calanus spp. during daytime at the 120 m station".

E27. Buchholz F, Saborowski R (2000). Metabolic and enzymatic adaptations in northern krill,
Meganyctiphanes norvegica, and Antarctic krill, Euphausia superba. Can J Fish Aquat Sci 57(Suppl
3):115-129. doi:10.1139/f00-168. [A] "Respiration measurements showed no seasonal differences in rates
determined in krill from the thermally stable Clyde Sea (Scotland) and the cooler but variable Danish
Kattegat." "A basal level of activity of digestive enzymes ensures immediate utilization of patchy food
sources."

E28. Sibling note values (research_notes/"Krill i Skagerrak"/summary.md and evidence.md) [P, citing F]:
energy density "about 5.0 kJ/g WW as an annual mean ... About 4.3 in spring and summer, about 6 in late
autumn and winter"; Kattegat "20.5 kJ/g DW in July and 26.0 in November" (Buchholz & Prado-Fiedler 1987
Table 1); Barents Sea "4.5 kJ/g WW in summer and 6.6 in winter" (Nowicki et al. 2023 Table 3); "Dry-to-wet
weight ratio: 0.20-0.27"; adult WW "0.15-0.30 g (0.23 g at 35 mm)"; P/B "2 /yr (1.2-3.5)"; "Clyde Sea
adult mortality: 1.1 /yr"; "Lifespan: 2-2.5 years".

---------------------------------------------------------------------------------------------------

## Part 3. Behaviour (DVM, swarming)

E29. Kaartvedt S, Svendsen H (1990). Advection of euphausiids in a Norwegian fjord system subject to
altered freshwater input by hydro-electric power production. J Plankton Res 12:1263-1277.
doi:10.1093/plankt/12.6.1263. [A]
- "in most locations ~50% of M. norvegica was found in the upper 10 m" at night.
- "Estimates of advective rates of M. norvegica suggest that exchange with coastal waters was more
  important for the standing stock than local growth within the fjords." T. raschii "possibly maintained
  its horizontal position by means of vertical migrations" over a shallowing bottom (~130 m).

E30. Stromberg J-O, Spicer JI, Liljebladh B, Thomasson MA (2002). Northern krill, Meganyctiphanes
norvegica, come up to see the last eclipse of the millennium? J Mar Biol Assoc UK 82:919-920.
doi:10.1017/S0025315402006367. [A] Gullmarsfjord: "the eclipse was partial (77%) and the weather
overcast and yet krill at 70-90 m depth were able to detect, and respond to very small changes in
light."

E31. Onsrud MSR, Kaartvedt S (1998). Diel vertical migration of the krill Meganyctiphanes norvegica in
relation to physical environment, food and predators. Mar Ecol Prog Ser 171:209-219.
doi:10.3354/meps171209. [F] data/onsrud1998_meps. Oslofjord.
- Abstract: "Acoustic scattering layers of M. norvegica in the Oslofjord, Norway, typically stayed below
  ~70 m during the day. They remained below prevailing fish schools, though varying numbers of deep-living
  fish occurred among the krill." "M. norvegica followed isolumes during ascent at dusk, but upward
  migration was mainly arrested at ~10-30 m and the krill were scattered throughout the water column at
  night." "Fish schools dispersed at sunset, and fish invaded the krill layer at night. Nocturnal
  avoidance of the surface water may thus have been an antipredator response". "Feeding on copepods was
  evident both day and night". "M. norvegica carried out diel vertical migrations irrespective of
  fluctuations in distribution and abundance of food resources." (OCR of the light values garbled; the
  layer-top light is re-examined in research_notes/"Ljus under 50 m".)

E32. Tarling GA, Cuzin-Roudy J, Buchholz F (1999). Vertical migration behaviour in the northern krill
Meganyctiphanes norvegica is influenced by moult and reproductive processes. Mar Ecol Prog Ser
190:253-262. doi:10.3354/meps190253. [F] data/tarling1999_meps. Kattegat, Alkor Deep, 24 Jul - 3 Aug
1998.
- "the population concentrated in the deep (80 to 100 m) during day-time but segregated vertically
  during night-time. Moulting occurred in the deep during night-time, away from the main part of the
  non-moulting population, which was located between 80 and 50 m."
- "almost 90% of moulting krill being found below 65 m"; "The majority of non-moulting females were
  found in the upper depth intervals at night (65% above 65 m) whilst non-moulting males were more
  evenly distributed through the water column (38 % above 65 m)."
- "migration behaviour in krill was not influenced by size in either sex".

E33. Spicer JI, Saborowski R (2010). Physiology and metabolism of northern krill (Meganyctiphanes
norvegica Sars). Adv Mar Biol 57:91-126. doi:10.1016/B978-0-12-381308-4.00004-2. [F, AWI repository,
"Author's personal copy" - never commit] data/spicer2010_amb57_physiology.
- 3.1: Balsfjord "The relative amount of proteins ranged from 23% to 36% DW and the amount of lipids from
  30% to 47%. The ash content varied around 15% of the body dry weight." Kattegat recalculated to WW
  (80 % water assumed): "ash (2.2-5.8% WW), lipid (1.6-9.6% WW), carbohydrates (0.2-0.4% WW), protein
  (9.2-13.8% WW), and chitin (0.6-1.1% WW)".
- 4.1: "Q10 values ~2, indicating little thermal adaptation"; populations "displayed near identical rates
  of O2 uptake (30-35 umol g-1 DW h-1) when tested at the environmental temperatures they were each
  experiencing ... This indicates that the metabolism of M. norvegica can acclimatise to local
  temperature conditions over a time course greater than the period of acute temperature change ...,
  more likely in the order of days to weeks." "Salomon et al. (2000) showed that the starving of M.
  norvegica resulted in a reduction in O2 uptake."
- Table 4.4 (respiration, umol O2 g-1 DW h-1; the PDF prints "mmol"): "Kosterfjorden (Sweden) Dec-Sep
  5-6 [C] 21.9-40.6"; "Gullmarsfjorden Sep 6.5 46.9"; "Kattegat (Alkor Deep) Jun-Sep 5-10 50.1-73.4";
  "Gulf of St. Lawrence Feb-Aug 2-10 60.3-92.9"; Ligurian Nov 5-20 C 23.7-43.9.
- 4.2: Gullmarsfjord krill regulate O2 uptake "down to approx. 30% saturation (critical O2 tension, or
  Pc = 6-7 kPa, T = 10 C)"; "surviving not more than 1 h in anoxia".
- 3.3.1 (E. superba, the only starvation number in the chapter): "starvation for 19 days entailed a
  significant decrease in total lipids from 21% to 9% DW (Virtue et al., 1993)."
- No assimilation efficiency, maximum ingestion or starvation-to-death value for M. norvegica is given.

E34. McClatchie S (1985). Feeding behaviour in Meganyctiphanes norvegica (M. Sars) (Crustacea:
Euphausiacea). J Exp Mar Biol Ecol 86:271-284. doi:10.1016/0022-0981(85)90108-X [A, no abstract];
data set: McClatchie S (2014) Maximum ingestion and maximum clearance rate of Meganyctiphanes norvegica
determined experimentally. PANGAEA, doi:10.1594/PANGAEA.835086 (CC-BY-3.0) [F]
data/mcclatchie_pangaea835086/data.tab.
- Nova Scotia, "feeding experiment using cultures isolated from Digby Neck"; one row: T 10 C, "Biom
  C/ind [ug/#]" .1458E5, "IR C/ind [ng/#/h]" .338E5, "CR [ml/#/h]" .591E2. Derived [scripts/krill_rates.py]:
  5.6 % body C per day; 1.42 l per animal per day.

E35. Huenerlage K, Graeve M, Buchholz C, Buchholz F (2015). The other krill: overwintering physiology of
adult Thysanoessa inermis (Euphausiacea) from the high-Arctic Kongsfjord. Aquat Biol 23:225-235.
doi:10.3354/ab00622. [F] data/huenerlage2015_ab. Starvation 28 d at 4 C.
- "The survival rate during the 28 d was 80%."; dry weight "20.5 +- 1.3 mg" at day 28 vs "30.0 +- 1.6 mg"
  at the start; "total lipid decreased considerably from ~42% at the beginning (T0) to ~28%".
- "the arcto-boreal species did not reduce metabolism but utilized lipid reserves for survival"; mean
  respiration "~6 umol O2 h-1 g-1 FW"; "The daily individual energy demand was 7.6 J (~0.17 mg lipid)";
  "the lipid reserves alone could cover the energy requirements of an adult female T. inermis for ~63 d."
- Caloric equivalent "0.44 J umol-1 of oxygen consumed for lipid (Ikeda et al. 2000)".
- Discussion: "Euphausiids from other latitudes, such as the adult Antarctic E. superba and the southern
  Atlantic upwelling species Euphausia hanseni, downregulated their respiration rates by up to 70% during
  winter and/or during food absence".

E36. Huenerlage K, Buchholz F (2015). Thermal limits of krill species from the high-Arctic Kongsfjord
(Spitsbergen). Mar Ecol Prog Ser 535:89-98. doi:10.3354/meps11408. [F] data/huenerlage2015_meps.
- "The 2 Thysanoessa spp. appeared more cold-stenotherm than the other 2 krill species: the upper level
  of their respiratory capacity was reached at 12C". "Calculated Q10 values ranged from 1.5 to 2.4 in M.
  norvegica".

E37. Lass S, Tarling GA, Virtue P, Matthews JBL, Mayzaud P, Buchholz F (2001). On the food of northern
krill Meganyctiphanes norvegica in relation to its vertical distribution. Mar Ecol Prog Ser
214:177-200. doi:10.3354/meps214177. [F] data/lass2001_meps (pages 1-10 of 24 in the download).
- "Krill was found to be more carnivorous in the Kattegat than in the Clyde Sea, which correlated with
  the higher ratio of copepod to phytoplankton biomass". "Other food sources included detritus,
  terrestrial material and other euphausiids". "The fact that deep-living Calanus was not a major prey
  item suggested that there was little feeding activity during the daytime ... the majority of mandibles
  were from species that were dominant in the upper layers, e.g. the genera Temora and Pseudocalanus."
  "krill in the Clyde Sea area and the Kattegat show a diel rhythm in feeding activity that is believed
  to be an adaptive response to minimising predation risk."

E38. Project note research_notes/"Krill i Skagerrak"/exchange/summary.md [P] (sources there: Rodhe 1987
[F], Aksnes et al. 1989 [F], Sourisseau et al. 2006 [F], Aure & Dahl 1994 [S], Kaartvedt & Svendsen
1990 [S]):
- "For the krill-bearing layer, e is about 20 per year (residence about 18 days). The plausible range is
  5-50 per year (residence about 7-70 days)."
- "Retention by vertical migration needs layers flowing in opposite directions. That holds only in the
  box's offshore half"; "Even if the net drift cancelled completely, mixing alone exchanges the box at
  about 5.6-24 per year"; Masfjorden krill-versus-water ratio "0.43-0.60 (Aksnes et al. 1989 Table 4)";
  "Expect krill to be retained by a factor of about 0.5-1 relative to the water, not by orders of
  magnitude [J]."
- "Below the 270 m sill (12 % of cells, no krill by day), the density renewal every ~1-3 years (Aure &
  Dahl 1994 [S]) is not horizontal flushing. Rodhe shows the cyclonic flow reaching 400-500 m [F]."

E39. Buchholz F, Prado-Fiedler R (1987). Studies on the seasonal biochemistry of the northern krill
Meganyctiphanes norvegica in the Kattegat. Helgol Meeresunters 41:443-452. doi:10.1007/BF02365404. [S, abstract quoted in a
search snippet; the table values are in E28's sibling note]: "The amount of lipid increased to a maximum
of 48 % of the dry weight towards winter" and "Utilisation of stored reserves proceeds over winter, with
loss of lipid, and decrease in weight but not in length."

---------------------------------------------------------------------------------------------------

## Search log: looked for and not found (gaps)

G1. A measured capture probability per encounter or per attack of any fish on any euphausiid (lab or
field). Searched: OpenAlex/Crossref title and full-text queries; web searches on "capture success",
"attack success", "capture efficiency" with euphausiid / krill / Euphausia pacifica / Meganyctiphanes
and herring, pollock, salmon, cod, whiting, saithe, mackerel. Closest: mysid swarms (E14-E16) and the
Juanes review figure for piscivores in general (E20).
G2. Reaction (flight-initiation) distance of M. norvegica or any euphausiid to an approaching fish.
O'Brien 1987 (E12) and O'Brien & Ritz 1988 (J Exp Mar Biol Ecol 116:257-272,
doi:10.1016/0022-0981(88)90031-7) may hold it; full texts not obtained.
G3. Tail-flip escape speed of M. norvegica itself (E13 is E. superba).
G4. Any measurement of fish capture success on krill in dark versus light.
G5. Starvation time to death or weight loss for M. norvegica (Mayzaud 1976 Mar Biol 37, Salomon et al.
2000 not obtained; E33 has only the direction of the respiration change).
G6. An assimilation (absorption) efficiency measured for M. norvegica. Only E. pacifica (64-83 % on
marine snow, Dilling et al. 1998, MEPS 170:189 [S, search snippet]) and E. superba (63-86 % for algal
lipid [S, search snippet]) were seen.
G7. The temperature dependence of M. norvegica ingestion (Sameoto 1976, E24: only the direction).
G8. Schmidt 2010 (food and feeding chapter), Kaartvedt 2010 (DVM chapter), Tarling 2010 (population
dynamics), Simard & Harvey 2010 (predation), Jager & Ravagnan 2016 (DEB parameters), Liljebladh &
Thomasson 2001 (Gullmar ADCP): no open full text found.
