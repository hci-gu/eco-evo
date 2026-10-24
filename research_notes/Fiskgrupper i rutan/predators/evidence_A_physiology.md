# Evidence A - physiology of the `predatory_fish` group

Scope: rations (Cmax), standard metabolism, Q10, activity cost, swimming, energy density and
assimilation for saithe (41 % of group biomass), whiting (37 %), Atlantic mackerel (8 %),
spurdog (5 %), horse mackerel (3.4 %), European hake (2.5 %) and anglerfish (1.4 %).
Site: Skagerrak, 100-300 m. Bottom/50 m layer 6.6-11.7 C (annual mean 8.4 C). Mackerel lives
in the 0-50 m layer, Jun-Oct, at about 12-14 C.

Tags: [F] full text read or data file parsed; [A] abstract only; [S] secondary (cited via another
source); [D] my own arithmetic. Scripts: `predators/scripts/calc_physiology.py`,
`calc_ger_cmax.py`, `calc_weighted.py`, `amp_parse.py`, `amp_stat.py`, `fb_query.py`.
Downloads: `predators/data/physiology/` (listed in `predators/data/README.md`).

Units: 1 MJ/t per 6 h tick = 4 kJ/kg/d. 1 mg O2 = 13.6 J. 1 mg O2/kg/h = 0.326 kJ/kg/d =
0.0816 MJ/t/6h. Cmax is given in % body mass (wet prey per wet fish) per day, with the prey
energy density it refers to, and in kJ per kg fish per day.

Reused project notes (not redone):
- Cod Cmax from FB4 (Hansson et al. 1996): `Bentos åtkomlighet för torskfiskar/evidence_B_gadoid_foraging.md`.
- Gadoid energy density: `Torskfiskars energiinnehåll/energy_density.md`.
- Q10: `Vintermetabolism sill och tumlare/q10_metabolism.md`. Measured cod SMR there: 35.5,
  57.0 and 78.2 mg O2/kg/h at 5, 10 and 15 C (Schurmann & Steffensen 1997 [A]). Q10 is 2.2
  over 5-15 C and 2.6 over 5-10 C.

## Sources

| # | Source | Tag | Used for |
|---|---|---|---|
| 1 | Fish Bioenergetics 4 parameter file, https://raw.githubusercontent.com/jim-breck/FB4/master/Parameters_official.csv | [F] | cod, walleye pollock rows. No mackerel, hake, dogfish, horse mackerel or anglerfish rows exist. |
| 2 | ICES 2024 WGSAM report, ICES Sci. Rep. 6:13, doi:10.17895/ices.pub.25020968, https://archimer.ifremer.fr/doc/00873/98452/107912.pdf (file `diet/sms_consumption/WGSAM_archimer_107912.pdf`) | [F] | Table 3-7 gastric evacuation (GER) model for cod, saithe, whiting, mackerel. |
| 3 | ICES WGSAM North Sea SMS key run 2023, `consum_ab.in`, https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/consum_ab.in | [F] | field rations, ration = a W^b (kg per fish per quarter). |
| 4 | Velasco & Olaso 2000, ICES CM 2000/Q:11, https://www.vliz.be/imisdocs/publications/ocrd/276039.pdf | [F] | hake rations; Andersen whiting GER; prey energy densities. |
| 5 | Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142 (North Sea Ecopath 1991), https://www.cefas.co.uk/publications/techrep/tech142.pdf | [F] | annual Q/B, unassimilated fraction. |
| 6 | Bachiller et al. 2018, PLoS ONE 13:e0190345, https://pmc.ncbi.nlm.nih.gov/articles/PMC5749802/ | [F] (WebFetch extraction) | mackerel respiration, SDA, egestion, ration. |
| 7 | Brett & Blackburn 1978, J. Fish. Res. Board Can. 35:816, doi:10.1139/f78-131 | [A] | spurdog resting, routine and active MO2. |
| 8 | FishBase v24.07 tables `oxygen`, `ration`, `popqb`, `speed` (parquet, https://huggingface.co/datasets/cboettig/fishbase) | [F] data, values [S] | spurdog MO2, swimming speeds. |
| 9 | Add-my-Pet (AmP) DEB entries, https://www.bio.vu.nl/thb/deb/deblab/add_my_pet/entries_web/<Species>/ | [F] | Arrhenius T_A (gives Q10), starvation time. |
| 10 | Andersen & Riis-Vestergaard 2004, CJFAS 61:2413, doi:10.1139/f04-153 | [A] | saithe activity multiplier. |
| 11 | Andersen & Riis-Vestergaard 2003, J. Fish Biol. 62:395, doi:10.1046/j.1095-8649.2003.00030.x | [A] | saithe/whiting conversion efficiency. |
| 12 | Steinhausen, Steffensen & Andersen 2005, Mar. Biol., doi:10.1007/s00227-005-0055-9 | [S] (search snippet) | saithe/whiting SMR, Uopt. |
| 13 | Chub mackerel respirometry, Fish. Oceanogr. 2019, doi:10.1111/fog.12465 | [A] | mackerel proxy SMR and speed cost. |
| 14 | Pedersen 2000, J. Fish Biol. 57:402, doi:10.1111/j.1095-8649.2000.tb02180.x | [A] | whiting field rations. |
| 15 | Q4 and Q5 sources: see those sections (Uglem 2009, Nøttestad 2016, Sulikowski 2010, Klöcker 2024, Thorburn 2015, de Pontual 2013, He & Wardle 1988, Spitz 2010, Matvaretabellen, Ackman). | | |

## Q1. Maximum daily ration (Cmax)

Answer: no lab Cmax function exists for any of the seven species. I built one for the gadoids
and mackerel from the ICES GER model (#2). I set its stomach-fill term so that cod reproduces
the FB4 cod Cmax. That gives a sustained Cmax of 2.3-2.7 % BW/d of 4.2 kJ/g fish prey at
8.4 C, i.e. 95-115 kJ/kg/d, nearly flat with body size [D]. Field rations are 0.4-1.5 % BW/d,
so they are 30-60 % of this ceiling. Mackerel fattening needs at least about 180 kJ/kg/d in
summer. Spurdog and anglerfish are much slower feeders, about 1 % BW/d.

### 1a. Building blocks

- **FB4 cod (Hansson et al. 1996)** [F]: "North Sea cod, Hansson et al. 1996,2,0.099,-0.195,2.41,13.7,21".
  Cmax = 0.099 W^-0.195 f(T), on 4184 J/g prey. My computation [D]:
  - 500 g: 2.13 % BW/d at 7 C and 2.37 at 8.4 C.
  - 1000 g: 2.07 % BW/d at 8.4 C.
  - f(T) is 0.80 of the optimum (13.7 C) at 8.4 C.
- **FB4 walleye pollock** [F] (a possible saithe proxy). It is rejected:
  - The Buckley & Livingston row has CQ 0.3763 per C, i.e. Q10 about 43.
  - The Mason row has CTO 8 C and CTM 15 C.
  - Both give 0.5-0.9 % BW/d at 1 kg and 8.4 C [D]. That is below North Sea field rations, so they cannot be ceilings.
- **ICES gadoid/mackerel GER model** (#2, Table 3-7) [F]:
  - The equation is R = rho e^(delta T) L^lambda E^-0.85 S^0.5 (g/h). L is length in cm, E is prey kJ/g wet, S is stomach content in g, and the armament term is 1 for fish prey.
  - Parameters: "Saithe 0.00171 1.35 0.081 -0.85 0.5"; whiting is identical; "Mackerel 0.00174 1.30 0.080"; "Cod 0.00224 1.30 0.083".
  - The temperature term gives Q10 2.25 (gadoids) and 2.23 (mackerel) [D].
  - Velasco & Olaso (#4) used Andersen's whiting form, "dS/dt = -0.00132 L^1.44 e^0.078T E^-0.875 S^0.5", for hake. They state that it "describes gastric evacuation in other gadoid species", saithe included [F].
- **Calibration [D].** At steady state, intake equals evacuation. Cod at 36 cm / 500 g and 7 C, with 4.184 kJ/g prey, reproduces the FB4 Cmax (10.7 g/d) when S = 12.6 g = 2.5 % BW. I read this S as the mean stomach fill of a fish fed to satiation. I apply the same 2.5 % BW to the other species. A 5 % BW fill gives an upper bound (x 1.4).

### 1b. Cmax per species at typical size [D] (`calc_ger_cmax.py`)

| Species, size | T | Cmax % BW/d, 4.18 kJ/g prey (S 2.5 % / 5 %) | kJ/kg/d | Note |
|---|---|---|---|---|
| Whiting 28-32 cm, 190-280 g | 8.4 | 2.45-2.49 / 3.45-3.50 | 103-104 / 144-146 | On 5.5 kJ/g prey: 1.95 % BW/d, 107 kJ/kg/d. |
| Saithe 40-65 cm, 0.6-2.2 kg | 8.4 | 2.29-2.31 / 3.22-3.25 | 96-97 / 135-136 | Nearly size-independent: L^1.35 against W^-0.5. |
| Hake 40 cm, 415 g (whiting parameters) | 8.4 | 2.72 / 3.83 | 114 / 160 | Biscay field rations of 1.2-2.8 % (below) fit under this. |
| Mackerel 35 cm, 400 g | 12 / 13 | 2.61 / 2.83 (S 2.5 %) | 109 / 118 | The GER ceiling is too low for summer fattening (1c). |
| Cod 36 cm, 500 g (check) | 8.4 | 2.40 | 100 | FB4 gives 2.37 at 8.4 C. |

Prey energy density moves intake mass as E^-0.85 and intake energy only as E^+0.15. Cmax is
therefore close to an energy ceiling of about 100 kJ/kg/d at 8.4 C for all the gadoids.

### 1c. Field rations and other checks (lower bounds on Cmax)

- **SMS key run 2023, North Sea** [F data, #3; % BW/d is my conversion, D]. Values by quarter Q1-Q4:
  - Whiting 0.2 kg: 0.78, 0.83, 1.24, 1.05.
  - Saithe 1.07 kg: 0.43, 1.25, 0.66, 0.78.
  - Mackerel 0.4 kg: 0.07, 0.94, 1.05, 0.16.
  - Hake 0.415 kg: 1.04, 2.84, 1.66, 2.06.
  - N. horse mackerel 0.25 kg: 0, 1.20, 1.71, 1.64.
  - Cod 0.5 kg: 1.14-1.77.
- **Whiting, North Sea** [A, #14]: "specific daily ration ranged from 0.41 to 1.66% in 1992 and from 0.78 to 1.75% in 1993".
- **Hake, southern Bay of Biscay, 10.6-12.9 C** [F, #4, Table 5]. "Daily ration in percentage of body weight":
  - ages 2-3 (26-39 cm): 1.74-2.83;
  - ages 4-6+ (40-90 cm): 1.18-1.98.
- **Ecopath North Sea 1991** [F, #5]. Q/B per year (my conversion to % BW/d):
  - adult saithe 3.6 (0.99);
  - adult whiting 5.46 (1.50);
  - hake 2.2 (0.60);
  - mackerel 1.73 (0.47);
  - horse mackerel 3.51 (0.96);
  - spurdog 2 (0.55), "Q/B=2 based on the report by Brett & Blackburn (1978)";
  - monkfish 1.9 (0.52).
- **Mackerel, Norwegian Sea feeding season** [F, #6]: mean consumption "0.08 g prey g fish-1 day-1", i.e. 8 % BW/d of copepods at 3.6 kJ/g, about 290 kJ/kg/d.
  - Budget check [D]: fat rises from about 5 % to 25 % in about 100 days. That deposits 0.20 x 39.5 kJ/g / 100 d = 79 kJ/kg/d.
  - Add routine respiration of about 25 kJ/kg/d (Q2) and divide by (1 - egestion 0.16 - excretion 0.10 - SDA 0.17) = 0.57. That needs C of at least about 180 kJ/kg/d (4.5 % BW/d of 4 kJ/g zooplankton).
  - The GER-based 118 kJ/kg/d is therefore too low for summer mackerel. Range 120-300; I use 150 as the central value.
- **Spurdog**:
  - "after consuming a 5-6% ration of teleost fish (hake) ... Digestion and absorption were completed between 120 and 360 h" [A], doi:10.1242/jeb.02736. That is one large meal per 5-15 d, i.e. about 0.4-1.1 % BW/d sustained [D].
  - The routine metabolism of 15.9 kJ/kg/d (Q2) needs about 0.55 % BW/d of 4 kJ/g prey at AE 0.75 [D].
  - Cmax is therefore about 1-1.5 % BW/d, i.e. 40-65 kJ/kg/d at about 10 C [D, weak].
  - Lab GER was 0.063 h-1 at 15 C (exponential), slower than summer flounder [A], doi:10.1002/mcf2.10177.
- **Anglerfish**: goosefish GER is "nearly ten times slower than that of Summer Flounder", 0.018 h-1 at 15 C [A] (same paper). With the Ecopath 0.52 % BW/d, Cmax is about 0.8-1.4 % BW/d, i.e. 30-60 kJ/kg/d [D, weak].
- **Horse mackerel**:
  - Cape horse mackerel, mostly crustaceans: "daily ration was estimated as 3.8% of wet body mass" [A], doi:10.2989/025776198784126683.
  - Using the mackerel GER as a proxy at 13 C gives 2.8 % BW/d of 4.2 kJ/g prey. Cmax is therefore about 2.8-4 % BW/d, i.e. 90-170 kJ/kg/d [D].
- **AmP DEB implied maximum food intake** [D from #9]: saithe 38, whiting 57, mackerel 31, spurdog 6, hake 11 kJ/kg/d.
  - These are below the field rations.
  - DEB fits to growth data do not constrain per-kg ingestion (the hake and anglerfish fits are reserve-dominated, with del_V 0.04 and 0.03).
  - Not used.

## Q2. Standard metabolic rate (SMR) and Q10

Answer: SMR is about 40-60 mg O2/kg/h for whiting and saithe at 8-10 C, i.e. 3.3-4.8 MJ/t/6h.
Spurdog is about 2.1 and mackerel's true SMR at 13 C is about 3.4. Mackerel has no swim
bladder and must swim, so its floor is the routine rate at 1 BL/s, about 6 MJ/t/6h. The
weighted group SMR is 3.65 MJ/t/6h, well below the old 6.0. The old 6.0 equals SMR x 1.6,
close to the routine rate. Q10 is about 2.3.

| Species | Measurement | MJ/t/6h [D] | Tag, source |
|---|---|---|---|
| Saithe, whiting | "SMR of whiting was about 59.2 mg O2 kg-1 h-1, similar to saithe"; temperature and size not recovered | 4.83 | [S] #12, via `q10_metabolism.md` |
| Cod (gadoid analogue), juveniles | 35.5 (5 C), 57.0 (10 C); 49 at 8.4 C by exponential interpolation [D] | 2.90, 4.65; 4.00 | [A] Schurmann & Steffensen 1997 |
| Cod, FB4 resting term (no ACT), 8.4 C | 0.008 W^-0.172 f(T): 81 mg/kg/h at 200 g, 61 at 1 kg | 6.6, 5.0 (x ACT 1.25: 8.3, 6.3) | [F] #1, [D]. The model term exceeds the measured SMR. |
| Spurdog, 2 kg, 10 C | "32.4 ± 2.6 SE (resting), 49.2 ± 5.0 SE (routine), and 88.4 ± 4.6 SE (active) mg O2∙kg−1∙h−1"; routine = "15.9 × 10^3 J∙kg−1∙d−1" | 2.64 / 4.01 / 7.21; at 8.4 C, Q10 2-2.5: 2.3 | [A] #7 |
| Pacific spiny dogfish S. suckleyi | SMR 20.9-21.9 mg/kg/h at 9 C (1.6-1.8 kg); 39-63 at 13 C (0.6-7.9 kg) | 1.75 (9 C), about 4.1 (13 C) | [F data] #8, values [S]. The cross-study Q10 of about 9 is not usable. |
| Mackerel 400 g, 13 C | Bachiller: R = 0.00264 W^-0.217 e^(0.06818 T) e^(0.0234 SW), SW = 1 BL/s; 74 mg/kg/h | 6.07 (12-14 C: 5.7-6.5) | [F] #6, [D] |
| Mackerel proxy, chub mackerel | "R = 0.0103W−0.490 e(0.0457T) e(0.0235U)"; at U = 0, 41 mg/kg/h; at 35 cm/s (1 BL/s), 94 mg/kg/h. U is taken as cm/s (my assumption, from Uopt "42.5 cm/s"). | 3.37 (U = 0); 7.67 (1 BL/s) | [A] #13, [D] |
| Pollack P. pollachius | routine 240-283 mg/kg/h at 18.5 C (155-870 g) | (summer only) | [F data] #8 |
| Hake, horse mackerel, anglerfish | no measurement found | proxies, see bottom line | gap |

Q10 per species:
- **Gadoids**:
  - cod SMR: 2.2 over 5-15 C and 2.6 over 5-10 C [A];
  - ICES GER delta: 2.25 [F, D];
  - AmP T_A 7400 K for saithe, whiting and hake: 2.55 over 3-13 C [F, D];
  - FB4 cod respiration effective Q10: about 1.7 (project note).
- **Mackerel**: Bachiller e^0.6818 = 1.98 [F, D]; chub mackerel e^0.457 = 1.58 [A, D]; AmP T_A 8824 K: 3.05 [D].
- **Spurdog**: AmP T_A 8000 K: 2.75 [D]. Over 10-23 C "the standard metabolic rate increased more rapidly than the maximum metabolic rate" [A], doi:10.3354/meps14586 (no number in the abstract).
- **Horse mackerel**: AmP T_A 8448 K: 2.91. **Anglerfish**: AmP T_A 8000 K: 2.75. (AmP T_A values are often defaults; low weight.)

## Q3. Routine and active metabolism relative to SMR; SDA

| Item | Value | Tag, source |
|---|---|---|
| Spurdog routine / resting | 49.2 / 32.4 = 1.52; active / resting = 2.73 | [A] #7, [D] |
| Saithe activity multiplier (field budget) | "The activity multiplier ranged from 2.0 to 2.5 at optimum cruising speed"; the budget fit gave 1.44-3.27 | [A] #10 |
| Saithe vs whiting | "SMR were similar ... AMR was twice as high for saithe" | [S] #12 |
| FB4 activity multiplier ACT | cod 1.25; walleye pollock 1.3-1.4 | [F] #1 |
| Mackerel at 1 BL/s vs U = 0 | x 2.28 (chub mackerel model, U in cm/s) | [D] from #13 |
| SDA (share of ingested energy) | cod 0.17; walleye pollock 0.125; mackerel 0.172; herring 0.175 | [F] #1, #6 |
| Saithe at fish farms | "saithe spent 8 to 10 h d-1 close to the sea-cages" (time budget only) | [F] Uglem et al. 2009 |

Implication [D]:
- Routine is about 1.5 x SMR (sit-and-wait or slow cruising), and cruising at Uopt about 2-2.5 x.
- The old `feeding_cost` of 1.75 and `movement_cost` of 2.0 x resting fit these ranges, provided resting is the true SMR of about 3.6 MJ/t/6h.
- SDA of 0.15-0.17 of intake belongs to the cost of feeding.

## Q4. Swimming speed and movement (from a sub-agent; tags as reported)

Conversion [D]: speed in m/s x 21.6 = km (1 km cells) per 6 h, in a straight line.

| Species | Value | Size, method, place | Cells/tick [D] | Source, tag |
|---|---|---|---|---|
| Saithe | Uopt 1.4 BL/s; COT 0.14 J/N/m | swim tunnel | 15 at 50 cm | #12 [S] |
| Saithe | maximum sustained 3.5 BL/s at 25 cm, 2.2 BL/s at 50 cm | lab, 14.4 C | 24 at 50 cm | He & Wardle 1988, doi:10.1111/j.1095-8649.1988.tb05468.x [A] |
| Saithe | "mean movement speed of 1.7 km h-1" between farms 1.6-4.7 km apart; 2-21 moves in 3 months | 48 cm, 1.1 kg, acoustic tags, Norway | 10 while moving | Uglem et al. 2009, MEPS 384:251, doi:10.3354/meps08001 [F] |
| Saithe | distances "levelled off around 200–300 km"; "longest in February–March" | conventional tags, NE Atlantic | hundreds of km per year | Homrum et al. 2013, doi:10.1093/icesjms/fst048 [F] |
| Whiting | Uopt 1.0 BL/s; COT 0.15 J/N/m | swim tunnel | 5.8 at 27 cm | #12 [S] |
| Whiting | sustained 1.3 BL/s at 17 cm (23 cm/s) | lab | 5 | FishBase speed table [S] |
| Mackerel | maximum sustained 3.5 BL/s at 33 cm, 11.7 C | lab | 25 | He & Wardle 1988 [A] |
| Mackerel | "Average swimming speed ... 1.33 m s−1 (3.8 BL s−1) for a 35-cm mackerel"; school vector means 0.44-0.72 m/s | sonar, Norwegian Sea, 4.8-10.3 C | 29 swimming; 9.5-15.6 net | Nøttestad et al. 2016, doi:10.1093/icesjms/fsv239 [F] |
| Spurdog | mean horizontal speed 1.36-2.49 km/h | 84 cm, satellite tags, NW Atlantic | 8-17 (geolocation error inflates this) | Sulikowski et al. 2010, MEPS 418:249, doi:10.3354/meps08821 [F] |
| Spurdog | "Nineteen out of 21 PSATs popped up ... within 40km distance from the tagging location" after up to 365 d | 85-115 cm females, Norwegian fjords | < 0.03 net | Klöcker et al. 2024, doi:10.1186/s40462-024-00498-2 [F] |
| Spurdog | females "overwintered within the loch, restricting their movements to the upper basin" | Loch Etive, Scotland (about 30 x 2 km) | resident | Thorburn et al. 2015, MEPS 526:113, doi:10.3354/meps11210 [F] |
| Hake | "most fish were recovered near their release locations"; one fish 26.1 km away after 1555 d | 27.6 cm, 27,690 tagged, Bay of Biscay | about 0.01 net | de Pontual et al. 2013, doi:10.1093/icesjms/fst102 [F] |
| Hake | sustained 3.5 BL/s at 22 cm (FishBase, 23.7 g fish) | lab | 17 | FishBase speed table [S] |
| Anglerfish | displacement 2.4-636 km; "only around 5% of the total data storage tag recordings showed more than 5 m vertical movement" | 53-124 cm, Faroes | sedentary | Frontiers Mar. Sci. 2022, doi:10.3389/fmars.2022.823066 [F] |

Implication [D]:
- Cruising at Uopt lets these fish cross 6-15 cells per tick, and burst or sustained swimming 25-29.
- Net displacement is far lower:
  - saithe at farms: a few km per day;
  - spurdog and hake: well under 1 cell per day;
  - mackerel and adult saithe on migration: hundreds of km per season.
- A 1 cell per tick limit therefore does not bind physiologically. Residence comes from behaviour.

## Q5. Energy density, lean body vs reserve, starvation (sub-agent; tags as reported)

- **Mackerel**:
  - Bay of Biscay, autumn, 25-29 cm [F]: 67.3 % water, 17.5 % protein, 10.5 % lipid, 7.9 kJ/g (range 7.1-8.5). Spitz et al. 2010, ICES JMS 67:909, https://archimer.ifremer.fr/doc/00252/36368/34909.pdf.
  - 20-25 cm, Q1, ICES VIIIc [F, #4]: 4.4 kJ/g.
  - Norwegian raw fillet [F, food table]: "Mackerel, February - June, raw" 1.9 % fat, 79 % water; "August - December" 24.0 % fat, 59 % water. Gross 5.2 and 13.6 kJ/g [D]. Source: https://www.matvaretabellen.no/api/en/foods.json.
  - NEA lipid cycle [S via #6]: "During spawning the average lipid content decrease to 2–8%", then "peaks at 25–31%" in mid-summer.
  - Bachiller Eq. 8 (11.6-32.5 over days 136-243) cannot be kJ/g wet. It looks like % lipid or kJ/g dry. Not used.
- **Horse mackerel**:
  - Bay of Biscay, autumn, 14-30 cm [F]: 72.4 % water, 5 % lipid, 6.0 kJ/g (Spitz 2010).
  - By quarter [F, #4]: 15-20 cm 5.8 / 6.9 / 6.9 / 8.0 and 20-25 cm 4.2 / 6.1 / 6.1 / 8.0 kJ/g for Q1-Q4. Lowest in Q1, highest in Q4.
- **Saithe**: no whole-body value. Raw fillet 1.3 % fat, 80 % water, 18.3 % protein, 4.8 kJ/g gross [F food table, D]. Pollack 29-30 cm, whole body, Biscay: 4.2 kJ/g, 0.7 % lipid [F] (Spitz 2010). The project note assumes 4.0.
- **Whiting and hake**: see `energy_density.md` (lean whiting 3.9, hake 3.7 kJ/g). Hake 8-30 cm is 3.9-4.2 kJ/g and flat over the quarters [F, #4].
- **Spurdog**:
  - Raw "Dogfish" muscle: 6.4 % fat, 76 % water, 17.9 % protein; gross 6.75 kJ/g [F food table, D].
  - S. acanthias "has very little or no squalene in its liver oil" [F] (Ackman et al., Proc. N.S. Inst. Sci. 44(2)).
  - No HSI or liver-oil percentage found. Other sharks' livers hold 26-60 % oil [A], doi:10.1046/j.1444-2906.2003.00668.x.
- **Anglerfish**: no value found (gap).
- **Starvation**: no measured fasting-to-death data for mackerel or spurdog. AmP DEB maximum survival when starved [D, order of magnitude only, at the AmP typical temperature]: mackerel 166 d, saithe 132 d, whiting 201 d, spurdog 1699 d. Mackerel feeds little from October to May, a gap of about 6 months bridged by its fat.
- **Suggested reserve structure [D]**:
  - Mackerel: lean body about 4.7 kJ/g (`energy_content` about 4700 MJ/t); fat 2 % -> 24-31 %. `max_energy_reserve` about 6000 MJ/t (range 3500-9000), filled Jun-Oct. The herring-like `prey_includes_reserve` applies.
  - Spurdog: lean about 6000 MJ/t; liver reserve about 1500-2500 MJ/t (low confidence; part of it is buoyancy oil).

## Q6. Assimilation efficiency

| Item | Value | Tag, source |
|---|---|---|
| Ecopath, all fish groups | unassimilated fraction 0.2, i.e. AE 0.8 | [F] #5 |
| FB4 North Sea cod | egestion FA 0.17, excretion UA 0.09; physiologically useful about 0.76 [D] | [F] #1 |
| Walleye pollock | FA 0.2, UA 0.11 | [F] #1 |
| Mackerel (Rudstam-type) | egestion 0.16, excretion 0.10 | [F] #6 |
| Crustacean prey | evacuation of brown shrimp "took almost twice as long" (whiting); krill like fish of equal energy density | [A] Andersen 1999, doi:10.1006/jfbi.1998.0865 |
| Crustacean prey | evacuation rate halves "with increasing level of the crustacean armament in terms of chitin and ash" | [S] Andersen et al., J. Fish Biol., doi:10.1111/jfb.12976 (search snippet) |
| Chitin | cod digested about 90 % of chitin from shrimp; "only 2.4% of the chitin ... recovered in the collected faeces" | [S] Danulat 1987, Helgol. Meeresunters. (search snippet), https://hmr.biomedcentral.com/articles/10.1007/BF02365402 |

Implication [D]:
- AE on fish prey is about 0.8-0.83 (energy absorbed). It is about 0.75 after nitrogen excretion.
- On krill or copepods, AE is about 0.75-0.8: thin exoskeleton, and chitin is largely digested.
- On armoured benthos (shrimp, crabs), AE is about 0.65-0.75, with slower evacuation (a lower Cmax in mass).
- The old 0.8 (fish, zooplankton) and 0.7 (benthos) are supported.

## Bottom line

Typical sizes: saithe 0.5-2 kg, whiting 0.2-0.3 kg, mackerel 0.4 kg, spurdog 2 kg, horse
mackerel 0.25 kg, hake 0.4 kg, anglerfish 4 kg. T = 8.4 C (mackerel and horse mackerel 13 C).
Cmax in kJ/kg/d; divide by 41.8 for % BW/d of 4.18 kJ/g prey. SMR is in MJ/t/6h at the
layer annual mean.

| Species (weight) | Cmax kJ/kg/d (range) | SMR MJ/t/6h (range) | Q10 | Routine / SMR | Cruise BL/s; net move | ED kJ/g (lean -> fat) | AE fish / crust. | Tags |
|---|---|---|---|---|---|---|---|---|
| Saithe (41) | 97 (75-140) = 2.3 % BW/d | 3.5 (2.9-5.0) | 2.3 (2.2-2.6) | 1.5; 2-2.5 cruising | 1.4; km/d to 100s km/season | 4.0-4.8 (no store data) | 0.8 / 0.75 | D from F, S |
| Whiting (37) | 104 (80-150) = 2.5 % | 4.2 (3.3-5.0) | 2.3 | 1.5 (proxy) | 1.0; no data | 3.9 -> 5.2 | 0.8 / 0.75 | D from F, S |
| Mackerel (8) | 150 (110-300) = 3.6 % | 3.4 SMR; 6.1 routine floor (3.0-6.5) | 2.0 (1.6-3.0) | about 1.8-2.3 (must swim) | 1-3.8; 100s km | 4.7 -> 13.6 | 0.8 | D from F, A |
| Spurdog (5) | 50 (40-70) = 1.2 % | 2.1 (1.7-2.7) | 2.5 (2.0-2.75) | 1.52 (F/A) | resident, < 0.03 cells/tick net | 6.0-6.75; liver unknown | 0.8 | A, F, D |
| Horse mackerel (3.4) | 120 (90-170) = 2.9 % | 3.4 (proxy) | 2.2 (proxy) | proxy mackerel | no data | 4.2 -> 8.0 | 0.8 | D, proxy |
| Hake (2.5) | 114 (90-170) = 2.7 % | 3.5 (proxy) | 2.3 | no data | resident | 3.7-4.2 | 0.8 | D from F |
| Anglerfish (1.4) | 45 (30-60) = 1.1 % | 1.2 (0.8-2.0; proxy) | 2.3 | low | sedentary | no data | 0.8 | D, weak |
| **Weighted** | **102 (78-154) = 2.44 % BW/d = 25.5 MJ/t/6h** | **3.65 (2.9-5.0); 3.87 with mackerel at routine** | **2.28** | 1.5 / 2-2.5 | | | 0.8 / 0.7-0.75 | D |

Comparison with the old model:
- Old gadoid resting 6.0 MJ/t/6h at Q10 2.2. The new weighted true SMR is 3.65, with Q10 2.3. 6.0 is about SMR x 1.6, i.e. close to the routine rate.
- If `resting_metabolism` keeps 6.0, the 1.75 and 2.0 multipliers double-count activity. If it is set to about 3.6, the old multipliers are about right.
- The Cmax/SMR ratio is about 102/14.6 = 7 [D].

Gaps:
1. No lab Cmax or satiation series for saithe, whiting, hake, horse mackerel or anglerfish. Andersen & Riis-Vestergaard 2003 and Jones & Hislop (satiation feeding of gadoids) were not accessible. The GER-plus-cod-calibration route is my construction.
2. The temperature and fish size behind the Steinhausen et al. 2005 saithe/whiting SMR are unknown.
3. No SMR at all for hake, horse mackerel or anglerfish. Spurdog has only one temperature (Brett & Blackburn); the 2024 MEPS temperature series was paywalled.
4. Mackerel SMR is a chub-mackerel proxy. The unit of the speed term (cm/s) is assumed.
5. Whole-body ED and seasonal fat for saithe, spurdog and anglerfish; spurdog HSI and liver oil.
6. No measured starvation tolerance for any species; the DEB values are model-implied.
7. Whiting home range; hake swimming speed.
8. The primary NEA mackerel monthly lipid series (dos Santos Schmidt et al. 2021, doi:10.1016/j.pocean.2021.102658) was blocked.
