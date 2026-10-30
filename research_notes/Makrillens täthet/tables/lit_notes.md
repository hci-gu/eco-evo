# Literature notes: mackerel and horse mackerel densities in ecosystem models and surveys (North Sea, Skagerrak-Kattegat, Norwegian Trench)

Tags: [F] full text read, [A] abstract only, [S] secondary (someone else's account), [D] computed here,
J judgement. t/km2 = tonnes wet weight per km2. Downloads and URLs: `../data/lit/README.md`.
Scripts: `../scripts/lit_ecobase_mackerel.py`, `../scripts/lit_sms_mackerel_quarters.py`,
`../scripts/lit_degnbol1988_density.py` (stdout in `lit_*_stdout.txt` next to this file).

## 1. Summary table of values found

| Species | t/km2 | Area | Year | How derived | Tag | Source |
|---|---|---|---|---|---|---|
| Mackerel | 1.72 | North Sea, ICES IVa-c, 570 000 km2 | 1991 | North Sea component 57 kt (MSVPA, ICES 1997) + Western component in the North Sea 923.4 kt (SGMSNS/MSVPA 2002: whole Western stock assessment x fractions migrating into the North Sea) = 980.4 kt / 570 000 km2 | [F] text; [D] 980.4/570 = 1.72 | Mackinson & Daskalov 2007 (tech142) s. 13.28 p. 142-143, Table 3.1 / 3.3 |
| Horse mackerel | 0.579 | same | 1991 | SGMSNS MSVPA "other predator" figure 329.75 kt ("single species assessment and proportion of the horse mackerel stock migrating in the North Sea"), rounded to 330 kt / 570 000 km2; chosen over Rueckert et al. 2002 survey estimate 96.16 kt for consistency with MSVPA | [F]; [D] 330/570 = 0.579 | tech142 s. 13.29 p. 143 |
| Mackerel | 1.72 (2011 key run) -> 0.75 (2015 key run) | same | 1991 base | 2015: "New estimation of biomass from SMS 2014", NS component of NEA stock; 427 000 t | [F]; [D] 427 000/570 000 = 0.749 | ICES WGSAM 2015, Annex 6, Table 2.1 p. 107 and text p. 109-110 |
| Horse mackerel | 0.58 (2011) -> 0.75 (2015) | same | 1991 base | 2015: SMS 2014; 427 784 t | [F]; [D] 0.750 | same |
| Mackerel | 0.46 | North Sea 570 000 km2 | 1981 | ICES MS WG (MSVPA) database; group "modelled to include both the North Sea mackerel, and the West mackerel occurring in the northern North Sea in the second and third quarters of the year" | [F] | Christensen 1995 Dana 11, Table 1 (p. ~6) and Appendix 1 p. 23 |
| Horse mackerel (as "other predatory fish", ~95% horse mackerel) | 1.5 for the group, ~1.4 for horse mackerel | same | 1981 (data 1983-85) | Sparholt (1990) winter + summer trawl surveys 1983-85; "Horse mackerel were only present in noticeable quantities on the autumn survey, and the biomass estimate used here is the average annual biomass" | [F]; [D] 0.95 x 1.5 = 1.4 | Christensen 1995 Table 1, Appendix 1 p. 24-25 |
| Mackerel ("Mackerel" + "West mackerel") | 1.948 + 1.9 | North Sea 570 000 km2 | 1974 | Not documented in EcoBase; group names show a NS / Western split | [F] (XML only) | EcoBase model 413 (Christensen et al. 2002) |
| Horse mackerel | 0.275 | same | 1974 | not documented | [F] (XML only) | EcoBase 413 |
| Mackerel | 0.57 | Eastern English Channel + southern North Sea, 98 000 km2 | 2006 | input B; derivation not in the XML | [F] (XML only) | EcoBase 789 (Araignous; doi 10.1016/j.jmarsys.2026.104279) |
| Horse mackerel | 1.556 | same | 2006 | estimated by Ecopath (EE 0.95 input) | [F] (XML only) | EcoBase 789 |
| Mackerel, horse mackerel | not in the model | Kattegat (EwE, 29 groups, base year 1982) | 1982 | 11 fish species: cod, whiting, dab, plaice, sole, flounder, Norway pout, herring, sprat, sandeel, blue whiting; no mackerel or horse mackerel box | [F] | Belgrano & Tomczak 2025, SIME Report 2025:3, s. 5.2 and Fig. 2 |
| Mackerel | 0.51 | southern North Sea Ecopath | ? | not checked | [S] (search-engine snippet only) | ResearchGate "Ecopath model of the southern North Sea" (Stabler et al.?) - unverified |
| Mackerel (Western, in ICES IV) | quarterly 0.55 / 0.39 / 2.40 / 3.00 (Q1-Q4), annual mean 1.58 | ICES IV (SMS area), expressed over 570 000 km2 | 1991 | SMS 2014 key-run input N x weight in the North Sea (MSVPA residence fractions carried forward) | [D] from [F] files | wgsam_sms/NorthSeaKeyRun_2014_other_predators.out |
| Mackerel (North Sea component) | 0.22 / 0.24 / 0.14 / 0.15, mean 0.19 | same | 1991 | same | [D] | same |
| Horse mackerel (Western in ICES IV) | 0 / 0 / 0.26 / 1.99, mean 0.56 | same | 1991 | same | [D] | same |
| Horse mackerel (North Sea stock) | 0.15 / 0.34 / 0.32 / 0.21, mean 0.26 (constant every year) | same | all years | "almost pure guesswork": 7% of the 1982-1999 Western stock | [D]; [F] for the rule | same; WKMSNS 2002 s. 2.4.6 p. 8; Horse_mack.R |
| Mackerel (whole NEA stock in ICES IV) | 2022: 0.38 / 0.52 / 1.98 / 2.35, mean 1.31; 1991: mean 2.06; 2013: mean 2.56 | same | 1974-2022 | SMS 2023 key run: whole-stock BIO x prop.in | [D] from [F] files | wgsam_sms/NorthSeaKeyRun_2023_summary.out |
| Horse mackerel (Western in ICES IV) | 2022: 0.009 / 0 / 0 / 0.047, mean 0.014; 1991: mean 0.69 | same | | SMS 2023 other-predator input | [D] | NorthSeaKeyRun_2023_other_predators.out |
| Horse mackerel (North Sea stock in ICES IV) | 0.29 / 0.28 / 0.29 / 0.28, mean 0.28 (constant) | same | | 7% rule (SS3 Western numbers x 0.07/0.93), mean over years | [D]; [F] rule | Horse_mack.R |
| Mackerel | 1.95 (2024), 1.69 (2022); stratum mean catch 1.0-2.4 (2018-2024) | IESSNS North Sea stratum 13 incl. Skagerrak to 11 E, south of 59 N (Skagerrak) / 60 N (North Sea), deeper than 50 m; 285 781 km2 (2024), 278 525 km2 (2022) | July 2018-2024 | swept-area (Multpelt 832 surface trawl), StoX; 556 655 t (2024), 471 948 t (2022; re-estimated 472 627 t in the 2024 report Table A1-2); mostly ages 1-2. INDEX-TYPE, not absolute: no catchability correction, surface trawl only, excluded from the stock index | [F]; [D] for t/km2 | IESSNS 2024 report App. 1 p. 56-58; IESSNS 2022 App. 1 p. 53 |
| Mackerel | 1.61 (strata A-D), 1.78 (B-D) | Skagerrak, 27 621 / 19 674 km2 | 31 Jul - 9 Sep 1987 | acoustic, RV Dana; mackerel TS 6 dB below herring | [F] tables; [D] density; J for strata = Skagerrak | Degnbol et al. 1988 ICES CM 1988/H:40, Tables 1, 4, Fig. 2 |
| Mackerel | 0.01 | Kattegat north (Dana stratum E), 17 932 km2 | Aug 1987 | same | [D]; J | same |
| Mackerel | 1.81 | NE North Sea / Norwegian Trench south slope (Dana O+I) | Aug 1987 | same | [D]; J | same |
| Mackerel | 10.7 (sub-area 5, 58-59 N), 7.1 (sub-area 6, 57-58 N incl. Skagerrak mouth), 2.6-7.1 (sub-areas 1-4 along the Norwegian coast / Trench) | NE North Sea, Norwegian Trench | 31 Jul - 11 Aug 1987 | acoustic, RV Eldjarn (echogram scrutiny + handline); Dana gave ~40% of Eldjarn in overlapping areas | [F] Table 7; [D] | same |

Historical stock sizes without an area (no t/km2 possible):
- North Sea mackerel stock "in the Skagerak and northeastern North Sea" at 6 September 1970 "estimated to 1492 million individuals or 632 thousand tons" (tagging, Hamre 1971 ICES CM 1971/H:23, Summary p. 14) [F].
- Western stock 1987 "2 497 thousand tones ... of which about 50% is present in the North Sea in third quarter" (Degnbol et al. 1988 p. 2, citing ICES 1988/Assess:12) [F].
- Rueckert et al. 2002 (ICES JMS 59: 120-130): southern-component horse mackerel biomass in Q3 "63 000, 163 000, 213 000, 248 000, 124 000, 147 000 and 57 000 t for the years 1991-1997", northern component underestimated; whiting as reference species [A]. Mackinson & Daskalov quote "Mean biomass in the North Sea in 1991 was 96.16 thousand tons" [S].

## 2. Mackinson & Daskalov 2007 (Cefas Tech. Rep. 142): how 1.72 and 0.579 were derived

Source read from `research_notes/Refugium och utbyte i bentos/data/cefas_tech142/tech142.txt` [F].

Model area (s. 2.1, p. 15): "The North Sea is a mid-latitude, relatively shallow continental shelf
covering approximately 570,000 km2 (Jones, 1982)" ... "The limits for this study are defined by ICES
Area IV, divisions a,b,c." So the Skagerrak-Kattegat (3.a) is outside the model; the report's Figure 2.1
shows "ICES Divisions of the North Sea (Div IVa,b,c) and Skagerrak (Div IIIa)". The EcoBase record
(model 457) gives area 570 000.

Mackerel (s. 13.28, p. 142-143), verbatim: "There are two components to the mackerel Scomber
scomberstocks in the North Sea, a resident population called North Sea mackerel and a migratory
population called Western mackerel (ICES 1997). Historically the resident population has been very
large (c.a. 2.5 million tonnes, Hamre (1978)), but since the 1970's it has decreased (36 - 110 thousand
tonnes SSB, ICES (2002a)). ... The biomass of the North Sea stock component was estimated by MSVPA to
be 57 thousand tonnes in 1991 (ICES 1997). Abundance of the Western mackerel was estimated as 923.4
thousand tonnes in 1991 by the Study Group on Multispecies assessment in the North Sea (SGMSNS) based
on assessment of the whole Western stock (ICES 2002a) and fractions migrating into the North Sea
(ICES 2002b). P/B assumed as equal to Z is 0.793 for the North Sea mackerel (ICES 1997) and 0.38 for
the Western mackerel (ICES 2002a). ... North Sea and Western mackerel are practically
undistinguishable and they were aggregated into a single mackerel group with biomass 980.4 thousand
tonnes and biomass density - 1.72 t.km2, P/B=0.6." [F]
- [D] (57 + 923.4) kt / 570 000 km2 = 1.720 t/km2.
- ICES 2002a = WGMHSA ICES CM 2002/ACFM:06; ICES 2002b = "Report of the workshop on MSVPA in the North
  Sea. Charlottenlund, Denmark, 8th-12th April 2002. ICES CM 2002/D:04" (reference list p. 183) [F].

Horse mackerel (s. 13.29, p. 143), verbatim: "Rueckert et al. (2002) estimated horse mackerel abundance
applying the approach of Sparholt (1990) and using International Bottom Trawl Survey (IBTS) data using
whiting as a reference species. Mean biomass in the North Sea in 1991 was 96.16 thousand tons (Rueckert
et al. 2002). However these authors warranted that their estimate of the Western stock component
"should be regarded as significant underestimate" ... possibly for two reasons: 1. It is concentrated
in a limited area on the north of ICES area IV; 2. Schools are distributed in deeper waters over the
shelf break where they are less accessible to the bottom trawl used by IBTS. The multispecies study
group SGMSNS uses another figure: 329.75 in MSVPA. It is based on single species assessment (ICES
2002a) and proportion of the horse mackerel stock migrating in the North Sea. This estimate however is
subject to many assumptions and also does not seem very reliable. Here we used the SGMSNS estimates of
biomass 330 thousand tonnes and biomass density 0.579 t.km2 (ICES 2002b) for reasons of consistency
with other commercial species estimates also based on MSVPA." [F]
- [D] 330 kt / 570 000 km2 = 0.579 t/km2. (Sprat is also 330 kt = 0.579 t/km2 in the same report -
  a coincidence, both texts state 330 kt.)
- Conflict: WGSAM 2015 (Annex 6 p. 110) says the previous key run "used Ruekert et al. (2002) estimate of
  the North Sea stock of 96 160 t" - this contradicts the tech142 text and the 0.579 value itself
  (96 160/570 000 = 0.169) [D]. J: the tech142 text is right about 0.579.
- Table 3.1 input P/B for horse mackerel is 1.64, Table 3.3 balanced 1.2 (1.64 in brackets); EcoBase 457
  has 1.2 [F].

What the residence fraction is (from ICES CM 2002/D:04, the "ICES 2002b" source) [F]:
- MSVPA area: "The current MSVPA again applies to ICES area IV; data have been selected accordingly, by
  the subtraction of catches from areas outside of the North Sea" (s. 2.2, p. 4) - 3.a excluded.
- Mackerel (s. 2.4.5, p. 7): resident stock "since the 1970s has been low (36 - 110 thousand tonnes
  SSB...). This has been partially compensated for by influxes of the Western Mackerel stock into the
  northern North Sea during the second half of the year since the 1970s." "There are no strong
  indications for changes in the fraction of the Western stock which migrates into the North Sea, so the
  fractions applied since 1986 have been carried forwards."
- Table 2.4.5.1 "Percentage of Western Stock mackerel in the North Sea area", 1986-00, by quarter 1-4:
  age 1: 0, 20, 30, 30; age 2: 10, 10, 50, 70; age >2: 10, 5, 50, 70. (1972-1979: only age >2 in Q3-Q4,
  5-70%.)
- Table 2.4.5.2 "Percentage of North Sea mackerel stock in the North Sea area", 1986-00: age 1: 100 in all
  quarters; age 2: 80, 100, 100, 80; age >2: 80, 100, 50, 70.
- Horse mackerel (s. 2.4.6, p. 7-8): "The western stock spawns off the western slope of the European
  shelf and migrates partly into the North Sea in the autumn". Table 2.4.6.1 "Percentage of Western Horse
  mackerel stock in the North Sea area", 1991: ages 1-4 zero in all quarters; age >4: 0, 0, 5, 40
  (Q1-Q4); 1992 10/55 in Q3/Q4; 1993-95 5/65; 1996-2000 Q4 5-50. North Sea stock: "As almost pure
  guesswork, the North Sea stock size was assumed to be 7% of the average of 1982-1999 western stock,
  which gives a SSB of 230 thousand tonnes."
- Table 4.1.6 note (p. 26): "the average stock biomass of other predators is the average across quarters
  due to the large seasonal fluctuations in stock sizes". Table 4.1.6 (p. 55): average stock biomass of
  all other predators 1991 = 1626 ('000 t).
- [D] Annual mean of the adult Western-mackerel quarter fractions 1986-2000 = (10+5+50+70)/4 = 33.75%;
  of the adult (>4) Western horse mackerel 1991 = (0+0+5+40)/4 = 11.25%.
- [D] The SMS 2014 key-run input files, which carry the same residence tables, give for 1991 the
  quarter-mean biomass in ICES IV: Western mackerel 902 995 t, North Sea mackerel 107 135 t, Western horse
  mackerel 320 580 t, North Sea horse mackerel 145 931 t (`lit_sms_mackerel_quarters_stdout.txt`).
  J: the SGMSNS figures used by Mackinson & Daskalov (923.4 kt Western mackerel, 329.75 kt horse
  mackerel) match these quarter-mean Western values to 2-3%. So 1.72 and 0.579 are ANNUAL MEANS over
  quarters of the stock present in ICES IV, i.e. the residence fraction is already averaged in; the
  Q3+Q4 presence is ~6x the Q1+Q2 presence (Western mackerel 1991: 3.08 vs 0.53 Mt; Q3/Q1 4.4, Q4/Q2 7.8) and horse mackerel is essentially absent in
  Q1-Q2. The 329.75 kt appears to be the Western stock only (the North Sea horse mackerel stock, the
  7% guess, would add ~146 kt on the SMS 2014 numbers). Not confirmed by a source sentence.

## 3. Other North Sea ecosystem models

- ICES WGSAM 2015, Annex 6 (North Sea EwE Key Run 2015) [F]: "The North Sea Ecopath model originally
  published by Mackinson and Daskalov (2007) and first published as a Key Run in 2011 (ICES WGSAM 2011,
  and Mackinson 2014) was updated" (p. 105). Table 2.1 (p. 107): "31 Mackerel 0.75 (1.72) ... New
  estimation of biomass from SMS 2014"; "32 Horse mackerel 0.75 (0.58) 0.9 (1.2) ... New estimation of
  biomass from SMS 2014" (brackets = previous key run). Text (p. 109-110): "Estimates of biomass from the
  SMS Key Run 2014 were used to update biomass estimates for horse mackerel and mackerel, both of which
  represent the North Sea stock and North Sea components of the North East Atlantic (NEA) stock (WGWIDE
  2014). For mackerel, the North Sea component of NEA stock is 2.86% (123 952 t), and the resulting total
  biomass estimate (427 000 t). For horse mackerel, the North Sea component of NEA stock is 6.79%
  (273 600 t), giving a biomass estimate of 427 784 t." Also lists as a future need: "Migration - e.g.
  how much mackerel and horse mackerel in the North Sea?" (p. 89).
  - [D] 427 000/570 000 = 0.749; 427 784/570 000 = 0.750. J: the arithmetic in the text is not
    reproducible from the SMS 2014 output files (1991 quarter-means: mackerel W+N = 1.01 Mt; horse
    mackerel W+N = 0.47 Mt); the text does not say which year or season the 427 kt refers to.
  - Base year 1991, area 570 000 km2, Ecosim 1991-2013 [F] (also in Olsen et al. 2023 ICES JMS 80: 218,
    which states the North Sea model "is developed for the entire North Sea, not the Skagerrak or
    Norwegian Trench" [F via WebFetch summary, treat as S]).
- Christensen 1995 (1981, 29 groups) [F]: see table; area 570 000 km2 (p. 4, "570 000 km2 (Jones
  1982b)"). Mackerel B from the MS WG database including West mackerel present in Q2-Q3.
- EcoBase North Sea 1974 (413), 1981 (680), 1991 (457), Eastern Channel 2006 (789): values in the table
  above [F, XML]. North Sea 1970 (114), North Sea 1880 (293), Norwegian+Barents 1997 (426), English
  Channel 1973/1995 (43, 44): "Datas for this model are not available" [F].
- EcoBase list (390 models, `research_notes/Fiskgrupper i rutan/benthivores/data/ecobase/list.xml`):
  no model whose name/region/author fields contain "Kattegat" or "Skagerrak" [F, grep].
- NorthSEE simplified North Sea EwE (2019, northsearegion.eu): download refused (403). A search-engine
  summary mentioned "mackerel biomass at 3.745 with no clear unit" [S, unverified].
- Lynam et al., Atlantis North Sea, NORWECOM, ECOSMO-E2E: not checked in full; ECOSMO E2E uses fish
  functional groups, not species (J, from the GMD title/abstract only).

## 4. Kattegat / Skagerrak models

- Kattegat EwE (ICES 2019, documented in WGINOSE 2020; base year 1982, Ecosim 1982-2008), as described by
  Belgrano & Tomczak 2025 (SIME 2025:3) [F]: "The Kattegat food web model is composed of 29 biological
  groups ... Fish: 11 species, with three of them further divided into juvenile and adult life stages".
  Figure 2 shows the fish boxes and B (t/km2): Ad.Cod 0.700, Juv.Cod 0.255, Ad.Whiting 0.700, Juv.Whiting
  0.0799, Blue whiting 0.02, Norway pout 1.000, Herring 9.000, Sprat 1.600, Sandeel 1.500, Dab 1.500,
  Flounder 0.250, Ad.Plaice 0.439, Juv.Plaice 0.198, Sole 0.200. No mackerel or horse mackerel group.
  J: in the Kattegat model mackerel/horse mackerel are absent, i.e. treated as negligible (consistent
  with Degnbol's 0.01 t/km2 in the northern Kattegat in August 1987).
- Skagerrak: only a qualitative fuzzy cognitive map exists in Olsen et al. 2023 (ICES JMS 80: 218);
  no quantitative Skagerrak or Norwegian Trench Ecopath/Atlantis model was found [F/S].

## 5. Seasonal presence - mackerel

North Sea / Norwegian Trench (well documented):
- CSWG 2023 s. 2.1.1 (p. 6) [F]: "much of the Northeast Atlantic stock over-winters in the vicinity of
  the shelf edge around the northern part of the North Sea (ICES Division 4a) and onto the shelf in the
  North Sea ... The extent of adult fish overwintering along the eastern portion of the Norwegian Trench
  (along the Norwegian coastline) and to the west of the British Isles ... has not been thoroughly
  investigated. In February/March part of the stock aggregates and then starts to migrate southward
  toward spawning grounds"; "Spawning does occur in the North Sea and this tends to be later in the
  season, around June/July". s. 5.2: "At the onset of the spawning migration, large mackerel shoals
  move out of the northern North Sea ... generally occurs around the end of quarter 4 and the start of
  quarter 1. ... Prior to the onset of this migration the mackerel are overwintering, relatively static
  and are targeted by a large Norwegian purse-seine fleet."
- Jansen et al. 2012 PLoS ONE (abstract, p. 1) [F]: "mackerel aggregate and migrate distances of up to
  500 km along the continental shelf edge from mid-November to early March"; the shelf-edge current flows
  "along the western edge of the Norwegian trench and into the northern North Sea"; Q1 IBTS: "Relatively
  few mackerel were caught outside the study area, e.g. in Kattegat/Skagerrak [12]". Table 2: from 1999
  fishing allowed "in the northern North Sea up to the 15th of February"; in 2009 mackerel "had departed
  the northern North Sea east of 4 W ... by October".
- Jansen 2014 ICES JMS (p. 300) [F]: "Outside the spawning season, millions of tonnes of mackerel started
  to spend autumn and winter in the northern North Sea in the 1990s"; spawning "ends in the northern
  areas, such as the North Sea" in early July. p. 303, quoting Hamre (1980) on the winter fishery "in the
  Norwegian Trench (inside the North Sea)": "The mackerel were distributed in deeper water layers during
  winter, often below the range of the purse net". Nederlec (1958): "in the North Sea from November to
  January, the scarcity of mackerel in the trawl seems to be explained by a migration out of the North
  Sea" [S, via Jansen].
- ICES FishMap mackerel [F]: "During winter, both immature and mature mackerel tend to be more abundant
  along the edges of the continental shelf and the Norwegian Deeps as well as in the central parts of the
  North Sea. Densities increase during the summer"; "In the North Sea, mackerel overwinter in deep water
  along the edge of the continental shelf (i.e. to the north and east of Shetland, and along the edge of
  the Norwegian Deeps) and, in the spring, adult mackerel migrate south to the spawning areas in the
  central North Sea"; "In the south, mostly juveniles immigrate through the Channel during spring to
  leave the area almost completely in winter"; "The length composition of the Skagerrak/Kattegat catches
  is shifted to the larger size compared to those for the North Sea" (IBTS Q3 1985-2005); peak landings
  "in the 3rd quarter (July to September)".

Skagerrak / Kattegat specifically (thin):
- Jansen & Gislason 2011 Cont. Shelf Res. 31: 64-72 [A, paraphrased by the DTU Orbit page]: Danish coastal
  landings in the North Sea, Skagerrak, Kattegat and inner Danish waters; larger mackerel arrive at the
  feeding areas earlier and leave later; sequential arrival supports an eastward post-spawning migration;
  occasional winter catches in the Sound, Kattegat and Skagerrak indicate some overwintering there.
  Month-by-month arrival dates are in the paywalled full text - NOT read.
- Degnbol et al. 1988 [F]: acoustic surveys "in August in the eastern part of the North Sea and in
  Skagerrak and Kattegat" since 1985 found "relative high concentrations of mackerel"; Aug 1987 Skagerrak
  1.6-1.8 t/km2 [D], northern Kattegat ~0.01 t/km2 [D]; Table 9: in 3.a (IIIa) 92.0% of mackerel were
  age 1 (mean length 27.3 cm), vs IVa areas dominated by ages 3-7.
- Hamre 1971 [F]: the strong 1969 year class "is coming up in the Skagerak and southeastern North Sea";
  the purse-seine season ran from August to November (closures 7 and 12 November 1970).
- IESSNS North Sea (July) [F]: survey stratum extends into the Skagerrak (limit 59 N, 11 E); 2022
  stations "extended into the eastern Skagerrak"; catches "dominated by 1- and 2-year-olds" (CSWG 2023
  p. 23). July presence in the Skagerrak at ~1-2.4 t/km2 (stratum mean) [D].
- J (synthesis, not a source statement): in the Skagerrak mackerel are a summer-autumn visitor -
  arriving after spawning (June-July), present at survey densities of ~1.5-2 t/km2 in July-August
  (mostly juveniles), leaving by October-November for the northern North Sea / Norwegian Trench
  overwintering grounds; small winter remnants. No source gives a monthly occupancy curve for 3.a.
  See also the catch-based month profiles compiled separately in this folder (`catch_*` tables).

## 6. Seasonal presence - horse mackerel

- ICES FishMap horse mackerel [F]: "In autumn, horse mackerel leave the North Sea to return in the
  spring." "During the winter, the fish largely disappear from the North Sea." North Sea stock: "from
  October onwards they leave the North Sea through the Dover Strait to overwinter in the warmer waters of
  the Channel and Celtic Sea". Western stock "enters the North Sea from the north, and increased abundance
  is associated with a strong influx of Atlantic water in the preceding months. Return migrations occur in
  late autumn (October-November), to the overwintering area off southwest Ireland". IBTS Q3: "abundance
  of immature specimens (8-16 cm) in both the Skagerrak and North Sea, while larger, mature specimens
  (30-40 cm) are restricted to the North Sea". Stock definition: "The North Sea stock is supposed to
  comprise the eastern Channel (VIId), the southern and central North Sea (IVb,c) and Skagerrak (IIIa)."
- WKMSNS 2002 Table 2.4.6.1 [F]: Western horse mackerel present in the North Sea only in Q3 (5%) and Q4
  (40%) of the adult (>4) stock in 1991.
- Iversen, Skogen & Svendsen 2002 Fish. Oceanogr. 11: 245-250 [A]: winter Atlantic inflow "is strongly
  correlated with the Norwegian fleet's catch of horse mackerel (r2 = 0.70) in the following autumn";
  "a large fraction of the horse mackerel stock migrating into the North Sea during strong inflows".
- WGWIDE 2026 (read from `../data/catch/wgwide/`, files of the catch study) [F]: "Since the benchmark of
  2024 catches in Divisions 3a (Western Skagerrak) and 4a in all quarters are considered to be from the
  Western horse mackerel stock" (NSHM s. 6.9); Western horse mackerel Q3 catches "in the northern North
  Sea and at the Norwegian coast in the Norwegian Trench and Skagerrak area" (WHM p. 329); NS-IBTS catches
  "most abundant in the third quarter" (Rueckert et al. 2002, cited NSHM s. 6.5.1.1); "fish move into the
  Channel following quarter 3".
- Christensen 1995 [F]: in the 1983-85 surveys "Horse mackerel were only present in noticeable quantities
  on the autumn survey".
- J: horse mackerel in the Skagerrak / Norwegian Trench are a late-summer to autumn presence (roughly
  July/August-November), adults of the Western stock along the Trench and juveniles in the Skagerrak;
  near absent December-May. No month-resolved Skagerrak source found.

## 7. Stock share in the North Sea by season (numbers)

- MSVPA residence tables (ICES CM 2002/D:04, 1986-2000 values carried forward) [F]: Western mackerel
  adults 10 / 5 / 50 / 70% of the Western stock in the North Sea (Q1-Q4); North Sea mackerel adults
  80 / 100 / 50 / 70%; Western horse mackerel adults 1991 0 / 0 / 5 / 40%, juveniles 0.
- SMS 2023 key run (ICES IV), whole-NEA-stock biomass share inside ICES IV [D from F files]:
  1991 10.5 / 10.2 / 41.9 / 54.9%; 2000 9.4 / 9.5 / 37.4 / 48.9%; 2010 6.6 / 8.0 / 27.6 / 35.7%;
  2022 4.9 / 5.8 / 21.2 / 27.7% (Q1-Q4). Derivation (stock_distribution_NS_2020.r, read as text):
  component split (NorthSea vs Western share of the NEA stock, file NEA_stock_dist.CSV) x the MSVPA
  quarter residence table (macDist.CSV), with the adult in-area fraction reduced linearly after 2000
  ("North Sea component halved since 2000", fac = 0.5/17 per year). J: assumption-driven, not an
  independent measurement; the residence table itself dates from the 1980s-1990s.
- WGSAM 2015 [F]: "the North Sea component of NEA stock is 2.86%" (mackerel), "6.79%" (horse mackerel).
- Degnbol et al. 1988 [F]: Western stock 1987 2 497 kt "of which about 50% is present in the North Sea in
  third quarter".
- IESSNS July (CSWG 2023, Table 4.1.3; North Sea included from 2018) [F]: share of total survey biomass
  by EEZ (EU27 / NO / UK), e.g. 2023 1.9 / 61.8 / 11.4% - EEZs mix North Sea and Nordic Seas, so this is
  not a North Sea share. Table 4.1.2 gives the North-Sea-only split by EEZ (2023: EU27 15.3, NO 34.1,
  UK 50.6%). Survey area 2.9 million km2, North Sea 0.27 million km2 (8.5%). The North Sea July swept-area
  biomass (556 655 t in 2024) is computed separately and excluded from the stock index; no North Sea
  share of the July stock is published (not computed here).

## 8. Not found / not read

- No Skagerrak, Kattegat or Norwegian Trench Ecopath/Atlantis model with a mackerel or horse mackerel
  biomass. The Kattegat EwE has no such groups.
- No full text of Jansen & Gislason 2011 (monthly arrival in Skagerrak/Kattegat), Rueckert et al. 2002,
  Iversen et al. 2002, Iversen 2002 (ICES Mar. Sci. Symp. 215), Revheim 1969/1966, Hamre 1970
  (Skagerrak tagging), Misund 1993 (sonar survey eastern North Sea and Skagerrak, Sept 1990), Sparholt
  1990, the Swedish SLU/HaV resource overviews (makrill, taggmakrill chapters), WGWIDE 2014 North Sea
  component text.
- No source gives a month-by-month presence curve for 3.a for either species; the month statements in
  sections 5-6 are judgement from quarter-level and survey evidence.
- The origin of WGSAM 2015's 427 000 t / 427 784 t (which year/season) could not be reproduced.
- The ICES MSVPA 2002 report confirms the residence tables but does not print the 923.4 / 329.75 kt
  figures themselves; the match to the SMS 2014 quarter means is my inference.
