# Mackerel anti-predator behaviour - evidence (2026-11-11)

Per-source numbers for summary.md. Research only; no code, library, manifest or
resume was changed. Grades: [F] field data read in full text, [L] lab/pen data
read in full text, [A] abstract, publisher page or search snippet only,
[S] secondary citation (one paper quoting another), [D] derived here
(scripts/), [J] judgement. "Analog" = a different species.

Files: `papers/` holds the PDFs fetched with the Unpaywall helper (E1, E3, E8)
and the helper's metadata JSON for DOIs it could not fetch. E4, E14, E16, E18 and
E19 were read as PDFs through WebFetch (cached by the harness, not copied here).
Computations: `scripts/ecopath_ns1991_m2_mackerel.py` (+ `_stdout.txt`),
`scripts/tuna_take_box.py` (+ `_stdout.txt`).

---

## A. Mackerel behaviour, schooling and feeding

### E1. Macy, Sutherland & Durbin 1998, MEPS 172: 89-100, doi 10.3354/meps172089 [L]
Atlantic mackerel (Narragansett Bay, ~184 g by the end), school of 30 in a 2.4 m
tank, 50 cm deep, Calanus and smaller copepods, light 8.2e-8 to 1.6 uE/m2/s.
- Feeding mode: particulate at low concentrations of large copepods (clearance up
  to ~20 L/fish/min), filter feeding at high (switch at 24-64 copepods/L); mouth
  open 1/3-1/2 of the time when particulate feeding on large copepods.
- Light: feeding rate unchanged between 1.6 and 2.0e-6 uE/m2/s; near zero at
  8.2e-8; 50 % threshold ~1e-7 uE/m2/s. "Given a feeding threshold of about 1e-7,
  mackerel should be able to feed and school throughout the night"; feeding to
  ~70 m at full moon and ~40 m by starlight (Georges Bank light, k ~0.15).
- School shape (subtended angle of the school around the circular tank):
  not feeding ~110 deg at every light level; feeding in full light 342 deg
  (whole circumference; p < 0.001); ~70 deg at 1e-5 to 1e-4; 160 deg at the two
  lowest levels. "Changes in school dispersal were not correlated with changes in
  feeding rates"; abstract: "spacing between fish did not appear to be related to
  feeding rate". Nearest-neighbour distances could not be measured (vertical
  overlap in the image).
- Mackerel "do not break their schooling pattern" while particulate feeding:
  they feed inside the school.
- Cites the schooling light threshold of Glass et al. 1986 (E2): 1.8e-7 uE/m2/s.

### E2. Glass, Wardle & Mojsiewicz 1986, J Fish Biol 29(Suppl A): 71-81 [S, via E1]
"A light intensity threshold for schooling in the Atlantic mackerel": schooling
kept down to ~1.8e-7 uE/m2/s (the order of starlight at a few tens of metres;
E1 Table 1). Abstract not retrieved; the number is E1's quotation.

### E3. Handegard, Tenningen, Howarth, Anders, Rieucau & Breen 2017, PLoS ONE 12: e0190259, doi 10.1371/journal.pone.0190259 [L, net pen]
Wild-caught mackerel (2 t in 2014, 5 t in 2016) in 12 x 12 x 12 m pens; 950-1900
fish per pen; a predator-shaped model pulled across the school, four exposures
2 min apart, sonar sequences scored 0-3 (3 = collective response affecting a
substantial part of the fish in view).
- Mackerel chosen as "an obligatory schooling fish that can form dense schools with
  the capability to perform highly synchronized collective responses".
- Crowding (mean 2.4x density increase; pilot up to 51.5 kg/m3, limits set at
  < 30 kg/m3) strongly reduced the collective response to the predator model;
  hypoxia (down to 40 % saturation) did not. "Negligible reduction" over four
  consecutive exposures (no habituation within a session).
- Mechanism offered (not tested): at high density the visual field is restricted
  "to such an extent that only a few individuals detect and respond to the threat",
  inhibiting the propagating behavioural waves seen in intact schools.
- Schools survived crowding to ~100 fish/m3 (~88 kg/m3).
- No feeding measurement; no predator other than the model.

### E4. Diaz, Noettestad, Pena, Huse & Ferno 2013, ICES CM 2013/N:05 [F distribution; the predation statements are the authors' inference]
Norwegian Sea, July-August 2010, multibeam sonar from a purse seiner, four regions,
daytime segments plus one night segment ~3 h later.
- Mackerel in the upper 40 m and >= 6 C, "during both day and night".
- School speed 1.33 m/s relative to the current (1.44 over ground).
- Packing density NOT measured: biomass from school area assuming 1 fish/m3.
- "Larger mackerel in the northern regions formed larger schools"; "In the north,
  mackerel schools were large; a common response to predation pressure" and
  "Even without predation pressure in these four regions, the schools in the north
  probably traded off safety over maximizing feeding" - no predator data were
  analysed for this (marine mammal sightings were collected but no result is given).
  Grade the trade-off statement [J, in source].

### E5. Godoe, Hjellvik, Iversen, Slotte, Tenningen, Torkelsen 2004, ICES J Mar Sci 61: 1093-1099, doi 10.1016/j.icesjms.2004.06.009 [A]
Eastern Norwegian Sea, July 2002, two commercial vessels, 63 schools tracked >= 30 s
(sonar 24-36 kHz). All schools < 100 m, 65 % at 0-40 m. School size 1 to 7000 t
(one school ~7400 t median estimate). Migration direction and its tidal response
did not differ between day and night. Possible vessel avoidance noted, data
inconclusive. No predator observations.

### E6. Jansen, Post, Olafsdottir, Reynisson, Oskarsson & Arendt 2019, Fish Res 214: 25-34, doi 10.1016/j.fishres.2019.01.020 [A]
Irminger Current: copepods (Calanus finmarchicus, > 2 mm) main prey 16-24 h solar
time, euphausiids and myctophids 22-08 h (switch about midnight); diet by weight
copepods ~65 %, euphausiids 18 %, teleosts 5 %, hyperiids 4 %, cephalopods 2 %;
"no indication of feeding below the mixed layer". Mackerel feed through the night.

### E7. Misund 1988, ICES working paper, "Sonar observations of schooling mackerel during purse seining" [A, snippet]
Northern North Sea, August 1986: "strong avoidance of the vessel", schools escaped
capture in ~37 % of purse-seine sets; horizontal speed ~3.5 body lengths/s, large
variation among schools. (Misund 1993, Fish Res 16: 179-194, doi
10.1016/0165-7836(93)90051-8, on herring and mackerel avoidance of purse seines:
not retrieved.)

### E23. Pitcher, Magurran & Edwards 1985, Mar Biol 86: 319-322 [S]
Mackerel and herring choose school neighbours within ~15 % of their own size
(secondary statement only).

---

## B. Anti-predator responses and protection in schooling fish (analogs)

### E8. Handegard, Boswell, Ioannou, Leblanc, Tjoestheim & Couzin 2012, Curr Biol 22: 1213-1217, doi 10.1016/j.cub.2012.04.050 [F, analog: Gulf menhaden prey, spotted seatrout predators, DIDSON sonar]
- Predators attack from behind at 0.249 m/s; prey ahead flee directly away
  ("flash expansion"), then turn to face the passed predator ("fountain effect");
  prey rarely within 20 cm of a predator; 0-17 predators in view (mean 2.7).
- Predators often attack in line formation (up to 5); attacks fragment schools and
  make boundaries irregular, which degrades information transfer.
- Per-capita risk proxy (prey within 30 cm of a predator) is higher in smaller
  schools and "increases steeply for groups below ~2 m2"; above that, encounter
  rate is roughly linear in group size (turbid water), i.e. little further dilution.
- Capture success not measurable.

### E9. Noettestad & Axelsen 1999, Can J Zool 77: 1540-1546, doi 10.1139/z99-124 [A, analog: overwintering herring vs killer whales]
455 kHz multibeam sonar, Tysfjord; 10 interaction types in 54 events (one per
3.9 min). Schools with cross-section > 460 m2 were not attacked even near whales;
attacked schools were more circular and denser than unattacked ones.

### E10. Noettestad, Ferno, Mackinson, Pitcher & Misund 2002, ICES J Mar Sci 59: 393-400, doi 10.1006/jmsc.2001.1172 [A, analog: herring vs fin whales, dolphins, killer whales]
125 km2 cold-front area, April; 44 schools followed 34.8 min on average; 184 events
(one per 8.3 min; intra-school compression/reorganisation/diving every 15.3 min,
inter-school join/leave/split every 22.9 min). Predator events every ~91 min, fin
whale attacks every ~170 min. Schools large (987 m2), dense, deep (148 m), slow
(1.1 BL/s): interpreted as risk-averse. No feeding comparison in the abstract.

### E11. Axelsen, Anker-Nilssen, Fossum, Kvamme & Noettestad 2001, Can J Zool 79: 1586-1596, doi 10.1139/z01-113 [A, analog: juvenile herring vs puffins and fish]
School responses to diving puffins and predatory fish: bend, vacuole, hourglass,
pseudopodium, herd, split - the same repertoire as adult herring under killer
whales, lasting longer under puffins.

### E12. Landeau & Terborgh 1986, Anim Behav 34: 1372-1380 [A/S, analog: minnows vs largemouth bass]
Bass captured prey more easily when one or two odd (dyed) minnows were in a school
of 8; the oddity effect vanished at 15. A school of 10 was significantly less
vulnerable than single fish or groups of 2-4 (secondary).

### E13. Trachurus analogs [A]
- Cape horse mackerel (T. capensis), South Africa (Barange et al.): in one diel
  cycle dispersed in midwater at night, schools formed toward sunrise and stayed on
  the seabed all day; in another cycle near the bottom for 24 h, feeding peaking in
  late afternoon; attributed to where the prey were.
- Cunene horse mackerel (T. trecae), Angola (Vaz Velho et al.): > 40 % of daytime
  backscatter in the bottom 10 m, < 10 % at night (dispersed in scattering layers).
- Jack mackerel (T. murphyi), Chile: more aggregated at night (opposite).
No T. trachurus study of this kind was found.

---

## C. Who eats ~30 cm mackerel, and how much

### E14. Mackinson & Daskalov 2007, Cefas Sci Ser Tech Rep 142 (North Sea Ecopath, 1991 base) [F table; D derivation]
- Mackerel group: TL 3.9, B 1.72 t/km2, P/B 0.6, Q/B 1.73, EE 0.632, P/Q 0.347.
  Horse mackerel: B 0.579, P/B 1.2, Q/B 3.51, EE 0.356.
- Diet matrix, mackerel row (columns matched to predators by character offset in
  the pdftotext output; checked against the header row): baleen whales 0.094,
  toothed whales 0.015, seabirds 0.015, juvenile sharks 0.014, spurdog 0.078,
  large piscivorous sharks 0.153, small sharks 0.005, adult cod 0.012, adult saithe
  0.004, hake 0.164, monkfish 0.053, gurnards 0.001, squid & cuttlefish 0.011.
  Horse mackerel row: toothed whales 0.010, seals 0.012, seabirds 0.007, spurdog
  0.018, juvenile cod 0.001, adult cod 0.003, juvenile whiting 0.002, adult whiting
  0.010, hake 0.180, gurnards 0.001, mackerel 0.006.
- [D] M2 = sum B_j (Q/B)_j DC_j / B (scripts/ecopath_ns1991_m2_mackerel.py):
  mackerel 0.068 /yr (baleen whales 0.036, squid 0.010, seabirds 0.006, cod 0.004,
  hake 0.003, toothed whales 0.003, monkfish 0.0025, saithe 0.0018, spurdog
  0.0015); fish predators together 0.013 /yr, of which the model's water-column
  members (spurdog, saithe, hake, monkfish) 0.0087; marine mammals 0.039.
  Horse mackerel 0.10 /yr (hake dominant). Group (0.94/0.06) 0.070 /yr.
  North Sea-wide, 1991, all ages (squid and seabird columns take small fish).

### E15. ICES SMS North Sea key run (WGSAM 2024, ICES Sci Rep 6:13, doi 10.17895/ices.pub.25020968) [F, via the repository's own check]
Mackerel is a predator only: M1 = 0.0375/quarter (0.15/yr, the assessment M),
M2 = 0, no prey rows (`research_notes/Fiskgrupper i rutan/predators/
evidence_B_population.md` lines 167 and 288: "SMS gives saithe and mackerel M2 = 0
by construction"). The zero is structural, not a measurement.

### E16. Jakten Langert 2022, SLU MSc thesis "A predator's diet: prey composition analysis of Atlantic bluefin tuna in Skagerrak and Kattegat" [F]
19 tuna (18 with prey), Skagerrak and Kattegat. By number: garfish 143 (44 %),
herring 78 (24 %), mackerel 42 (13 %). By weight: garfish 40 854 g (~55 %),
mackerel 17 085 g (second; ~23 % of ~74 300 g), cod 6518 g. Mean mackerel 407 g,
32 cm (the box's mackerel size). Energy 16 kJ/g; a 250 kg tuna needs ~5.8
mackerel/day for maintenance. Scenario: at > ~1500 tuna their mackerel take would
exceed the Swedish Skagerrak-Kattegat mackerel landings (2021).

### E17. Hellstroem et al. 2025, Fish Res 291: 107501, doi 10.1016/j.fishres.2025.107501 [A, via DTU Aqua news page]
44 giant bluefin (232-295 cm) caught Aug-Sep 2020-2022 in Skagerrak, Kattegat and
the Sound; 19 fish species, 4 squid, 1 crab; garfish > 37 % by number and ~60 % by
weight; herring in 83 % of stomachs; mackerel and gadoids "prominent" (no %).
Daily intake 3-6 % of body mass; a 278 kg tuna eats 750-1500 kg over a ~90-day
stay (Aug-Oct). Tuna returned ~2015-2016.

### E16/E17 -> [D] box take (scripts/tuna_take_box.py)
Mackerel 23 % of stomach weight, 90 days, 250 kg tuna, ration 3-6 %, spread over
Skagerrak + Kattegat (~54 000 km2, [J]) at 0.85 t/km2: per tuna 155-310 kg of
mackerel per season; 500 tuna -> M2 0.002-0.003; 1500 -> 0.005-0.010; 15 000 ->
0.05-0.10 /yr; in a 3600 km2 box 5-31 t per season at 500-1500 tuna against
3060 t. Tuna numbers in the area are unknown. Upper side: the mackerel total is
~2x the annual mean in August.

### E18. Van Beveren, Smith, Smith & Pelletier 2024, DFO CSAS Res Doc 2024/018 [F, report; NW Atlantic]
Northern-contingent Atlantic mackerel: an ensemble of predators (northern gannets,
grey seals, cetaceans, bluefin tuna, spiny dogfish) "likely driving overall natural
mortality patterns". Median total consumption 10-17 kt/yr (1968-83), 18-45 kt
(1984 on), 28-44 kt in 2012-2021 (minimum 21-29 kt), "at least twice" the Canadian
landings (< 11 kt). Spiny dogfish diet share of mackerel varied 4-49 % between years
(> 100 stomachs/yr); dolphins, porpoise and pilot whales 2-30 kt/yr; tuna median
~4 kt in 2020. Shows that adult mackerel ARE heavily eaten where gannets, grey seals
and dogfish are abundant.

### E19. Garthe, Montevecchi & Davoren 2011, Limnol Oceanogr 56: 802-812 [F]
Funk Island gannets (NW Atlantic). In the year when mackerel and saury were
available (2005; capelin 10 % of prey landings) dives were shallower and shorter:
mean depth 2.7 m (capelin year 4.3), mean max 6.0 m (10.6), max 9.7 m (19.1);
duration 5.0 s (10.1). Gannets take mackerel in the top ~10 m.

### E20. Minke whale diets, Norwegian Sea and North Sea (J Cetacean Res Manage, journal.iwc.int article 888) [A]
North Sea 1999-2000, 7 forestomachs plus visual observations: sandeel 86.7 % by
weight, mackerel 9.3 %, whiting 2.4 %, herring 1.1 %, pout 0.5 %. Tiny sample.

### E21. Harbour porpoise, Dutch coast (Jansen et al. 2013, Mar Mamm Sci 29: E295) [A]
Stomachs: 90.5 % gobies, whiting, sandeel, herring, cod, sprat (mackerel not among
them); isotope mixing models put mackerel among the main prey. Danish/Swedish
porpoise stomach studies found (snippets) do not list mackerel as a main prey.

### E22. IMR cruise report 2022-5 (bluefin tuna methods, Norwegian waters) [F, negative]
No mackerel schools seen by sonar or visually during the survey; juvenile mackerel
in tuna stomachs reported only as pre-survey information. Tuna fed at the surface
to 20 m and travelled at 20-50 m.

---

## D. What was searched and not found
- Any field or lab observation of Atlantic mackerel or horse mackerel schools under
  attack by a real predator (tuna, dolphin, porpoise, gannet, whale): none found.
- A packing density (fish/m3) of wild mackerel schools: none found (E4 assumes 1).
- Mackerel diel school/disperse pattern in the North Sea or Skagerrak: none found;
  E4 and E5 show no day/night change in depth or migration in the Norwegian Sea.
- Data-storage-tag depth records for adult Atlantic mackerel: none found.
- Any measurement of feeding rate inside compact vs loose mackerel schools under
  predation risk: none. E1 is the only coupling of school shape and feeding.
- Herring hunger/school-density lab work (Robinson & Pitcher 1989, J Fish Biol 34:
  631-633): citation confirmed, content not retrieved.
