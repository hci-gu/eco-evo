# Literature evidence - energy budget of `benthivorous_fish`

Question: the engine's benthivores need Q/B ~7.5-8 /yr (by mass, on infauna) to hold their reserve,
while the growth budget assumes Q/B 3.48 /yr (mixed diet). Which side is wrong?

Tags: [F] full text or data file read; [A] abstract only; [S] secondary (cited via another source or a
search summary); [D] my arithmetic (scripts kept in the session scratchpad; the numbers are reproduced
below); **J** my judgement. Model units: MJ per tonne wet weight (WW); 1 MJ/t/tick = 4 kJ/kg/d;
1 mg O2/kg/h = 0.326 kJ/kg/d = 0.0816 MJ/t/tick (13.6 J/mg O2). Prior project notes are cited by
folder name; their own sources are not repeated here.

Downloads (untrusted, parsed only): `../data/chung2019/` (PMC OAI XML of Chung et al. 2019),
`../data/brey2010/` (only the first page; the conversion tables are not in it), `../data/sealifebase/`
(Brey "mass conversion" pages: empty templates for every taxon tried, no values).

---

## Where the engine values came from (prior notes)

| Engine value | Origin | Note |
|---|---|---|
| Q/B 3.48 /yr (2.47-4.71) | `Fiskgrupper i rutan/benthivores/evidence_benthivores_C_diets_consumption.md` s5, `tables/qb_pooled.csv`: biomass-weighted species Q/B from EcoBase North Sea models (Mackinson & Daskalov 2007 = M&D, Christensen 1995), Greenstreet 1996/1997 rations x 365, FishBase empirical formula (10 C) | per tonne of the whole species group; M&D adult and juvenile groups are separate; prey wet weight as in stomachs |
| SMR 3.0 MJ/t/tick, RM 3.6 | same folder, `evidence_benthivores_A_bioenergetics.md`, `tables/bioenergetics_A_smr.csv`: Schurmann & Steffensen 1997 (cod), Norin 2019 (haddock), Fonds et al. 1985 DOCmin (plaice), Moser 1996 (witch), Rijnsdorp et al. 1983 field winter metabolism (LRD/plaice) | annual mean over the 50 m series (8.4 C) |
| feeding_cost 2.0 x RM | A file s3c: Fonds et al. 1985 Table 7 DOCmax/DOCmin 1.73-2.91 (fully fed vs unfed), Table 8 "score max-min" 15 % of the MAXIMUM ration energy; Perez-Casanova 2009 haddock SDA peak 30-50 % above RMR | both measured at satiation, see Q2/Q3 |
| infauna 2600 MJ/t WW | `Infauna och rörlig epifauna/params_split.md` s1.5: Ricciardi & Bourget 1998 AFDW/WW x Steimle & Terranova 1985 kJ/g AFDW, weighted by the fish intake | brittle stars set to 2.0 kJ/g WW "arms are cropped" |
| assimilation 0.7 / 0.8 | gadoid library value (section 208.2); Ecopath unassimilated 0.2; FB4 cod (predators note Q6) | |

---

## Q1. Consumption: Q/B and rations

| Quantity | Species | Value | Model units | T | Source | Quality / notes |
|---|---|---|---|---|---|---|
| Ration = a W^b, kg per fish per quarter, W in kg | cod | a = 0.900 / 1.212 / 1.247 / 1.390 (Q1-Q4), b = 0.786 | Q/B at W: 0.3 kg 6.1; 1.0 kg 4.75; 1.19 kg (box W_eff) **4.57**; 2.4 kg 3.9; 5 kg 3.4 /yr [D] | North Sea ambient | ICES WGSAM 2024 (ICES Sci. Rep. 6:13), Table 3-8; SMS key run 2023 `consum_ab.in` [F] (`Fiskgrupper i rutan/predators/data/diet/sms_consumption/`) | evacuation-based field rations from stomach contents (Andersen & Beyer 2005 cylinder model, bias-corrected for prey energy density and armament, Table 3-7). Units checked: grey seal a 478 kg/quarter, fulmar 34 |
| same | haddock | a = 0.323 / 0.446 / 0.594 / 0.588, b = 0.714 | 0.1 kg 3.8; 0.27 kg (box W_eff) **2.84**; 0.44 kg 2.5; 1 kg 1.95 /yr [D] | same | same [F] | W_eff = biomass-weighted mean of W^(b-1) over the 2001-25 GOV length mix (`benthivores/tables/size_factors_longrun.csv`) |
| SMS-based Q/B, cod + haddock biomass-weighted (0.31 : 0.39) | gadoid 70 % of group | **3.6 /yr** [D] | 3.6 | | from the two rows above | stock level for survey-catchable fish; consistent with the pooled 3.53 for haddock + cod in qb_pooled.csv |
| Q/B adult (data value before balancing) | haddock / cod | 4.4 (2.35) / 3.5 (2.17); juveniles 5.39 / 5.96 | /yr | NS mean, 10 C assumed for formula groups | Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142, Table 3.3 [F, prior note] | balancing raised gadoid Q/B by 1.6-1.9x; Christensen 1995 found MSVPA gross efficiencies "unrealistically high ... cod with 35%, ... haddock 36%" [F, prior note] |
| Daily ration x 365 | cod / haddock | 0.0144 / 0.0130 per day | 5.26 / 4.75 /yr | North Sea | Greenstreet et al. 1997 via Heath 2005 Table 3 [S/F, prior note] | population means |
| Q/B | plaice, dab, LRD, witch, lemon sole | 3.42, 3.36-4.0, 3.4 (FishBase formula 4.0 at 10 C), 3.0, 4.32 | /yr | 10 C for formula values | M&D 2007 from Greenstreet 1996 and Palomares & Pauly 1998 [F, prior note] | flatfish values are not field rations; FishBase formula x 0.91 at 7.5 C |
| Field ration | haddock 40 cm | 1.30 %BW/d (cod 1.44) | 4.7 (5.3) /yr | | via Hall 1987 / prior A note [S] | "feed at between 15 and 22% of their maximum capacity" (Hall 1987, J. Fish Biol. 31:479) |
| Pooled group Q/B | benthivores | **3.48 (2.47-4.71)**; 3.0-3.5 at 6-9 C | /yr | | `qb_pooled.csv` [D, prior] | |
| Field ration as share of Cmax | group | 3.48 x 3440 MJ/t = 11 970 MJ/t/yr vs Cmax 108 kJ/kg/d = 39 420 MJ/t/yr: **0.30** [D] | | | Cmax from params_benthivorous_fish.md s2 | consistent with Hall's 15-22 % |

Prey basis: SMS and Greenstreet rations are WET weight of prey as eaten (stomach contents corrected for
digestion state by the evacuation model). M&D Q/B are wet-weight flows. The diet is mixed: 0.511
infauna, 0.209 epifauna, ~0.28 fish and other prey by mass (consumption-weighted, evidence C).

### Synthesis Q1
- Stock-level Q/B for an adult-dominated group is **3.5 /yr by mass on the mixed diet (range 2.5-5.3)**;
  the SMS evacuation rations at the box's size mix give cod 4.6 and haddock 2.8, i.e. 3.6 for the gadoid
  70 %. W_eff is biomass-weighted over the survey (GOV) length mix, so juveniles from ~10 cm are already
  in; only the 0-group is missing, a small biomass share (cod < 30 cm is 9 % of cod biomass) **J**.
- The published Q/B is likely on the LOW side: M&D's raw MSVPA data gave gross efficiencies of 35 %
  that Christensen called unrealistic, and the energy budget below (Q2) needs ~4-5.5 /yr (mixed diet)
  to close at measured SMR and a realistic production. **J**: Q/B 3.48 is a floor-to-central value,
  not a central one; 4-5 is defensible.
- Apples to oranges: the engine's "7.5-8 on infauna" must be compared in energy. Q/B 3.48 on the mixed
  diet assimilates 3.48 x (0.511 x 2600 x 0.7 + 0.209 x 3750 x 0.8 + 0.28 x 4750 x 0.8) = **9 120
  MJ/t/yr** [D], equal to **5.0 /yr on infauna alone** (at 1820 MJ assimilated per t). The engine
  needs 7.5-8 on infauna = 13 650-14 560 MJ/t/yr: **1.5-1.6x** the published ration.

---

## Q2. Standard, routine and field metabolism

| Quantity | Species | Value | Model units (MJ/t/tick) | T | Source | Quality / notes |
|---|---|---|---|---|---|---|
| SMR | cod | 35.5 / 57.0 / 78.2 mg O2/kg/h at 5 / 10 / 15 C | 2.9 / 4.65 / 6.4; 4.1 at annual mean, 3.5 at 6.5 C | | Schurmann & Steffensen 1997, J. Fish Biol. 50:1166 [A, prior note] | test-fish size not in the abstract |
| Respiration (FB4, = SMR-type term) | North Sea cod 1.2 kg | RA 0.008 g O2/g/d, RB -0.172, Kitchell T-function (RQ 1.88, RTO 21, RTM 24): 56.6 mg O2/kg/h at 7.5 C (53.6 at 6.5, 59.4 at 8.4) [D] | **4.6** (4.4-4.8); x ACT 1.25 = 5.75 | 6.5-8.4 | Hansson et al. 1996 via Fish Bioenergetics 4.0 `Parameters_official.csv` [F] | FB4 adds SDA 0.17 of (C-F) on top; 300 g cod 5.9, 2.4 kg 4.1 |
| SMR | haddock 300 g | 2.2-3.8 (central 3.0) | 3.0 | annual mean | Norin 2019 scaled; ratio route [F, prior A note] | RMR/SMR 1.42, cruising 1.48 x SMR |
| SMR (DOCmin) | plaice 150 / 400 g | 39.9 / 32.1 mg/kg/h at 8.4 C | 3.25 / 2.6 | | Fonds et al. 1985 Table 7 [F, prior] | |
| Field winter metabolism, fasting | plaice females 487 g | 310 kJ per Dec-Mar season | 1.45 (~0.7 x lab SMR) | 5-7 | Rijnsdorp et al. 1983, ICES CM 1983/G:31 [F, prior] | fasting fish depress metabolism |
| Field winter metabolism | LRD | | 1.0-1.3 at ~3 C | ~3 | prior A note [F] | |
| SMR | witch | | 1.1-1.5 | | Moser 1996 scaled [F, prior] | |
| RMR fed / SMR | cod | **1.55 (1.44-1.65)** at 6 and 14 C | | 6, 14 | Jordan et al. (cited by Chung et al. 2019) [S] | routine = maintenance + regular tank activity + digestion |
| FMR / SMR (otolith d13C) | wild cod, Newfoundland, ages 0-14 | **1.9-3.5**, falling with increasing mass-specific O2 use (small fish 1.9, large fish 3.5) | at SMR 3.0: 5.7-10.5 | field | Chung, Trueman, Godiksen, Holmstrup & Grønkjær 2019, Commun. Biol. 2:24, doi:10.1038/s42003-018-0266-5 [F] | denominator is MTE-predicted SMR; measured SMR exceeds the MTE value below ~75 mg O2/kg/h, so for large fish the ratio to MEASURED SMR is lower **J** |
| Activity share of FMR | wild cod | 37-59 % of FMR (activity 0.73-2.0 x SMR), SDA assumed 0.12 x FMR | | | Chung et al. 2019 [F] | "FMR oto was always higher than the sum of SMR and SDA, even when ... 0.4 times total energy intake" was used |
| MMR / SMR | haddock / cod | 3.4 / 2.0 (68 g, 12 C); literature 2.3-3.5 | | 12 | Norin 2019 [F, prior]; Chung 2019 [F] | sets the ceiling of any field multiplier |

### Synthesis Q2
- Group SMR at 7-8 C: 3.0 MJ/t/tick at the 8.4 C annual mean x Q10 2.25^(-0.09) = **2.8 MJ/t/tick =
  11.2 kJ/kg/d = 4 100 MJ/t/yr** [D]. Real bottom water at 100-300 m in the Skagerrak is 7-8 C all year
  ("below 100 m the temperature was around 7-8 C", SMHI via `Vintermetabolism sill och tumlare/
  skagerrak_temperature.md`; Å17 100 m annual mean 7.6 C). The 50 m layer (6.6-11.7, mean 8.4 C) is
  warmer in autumn; on the annual mean the difference is only x 0.93 on RM.
- Field metabolic rate (FMR = SMR + activity + SDA): cod 1.9-3.5 x SMR (Chung 2019, upper values for
  large migratory cod, likely overstated against measured SMR); routine-fed cod 1.55 x SMR; sedentary
  flatfish and benthic-foraging haddock lower (**J**: 1.4-2.0; plaice fast through winter at 0.7 x lab
  SMR). Biomass-weighted **J**: **FMR 1.5-2.5 x SMR, central ~2.0**.
- In model units at 7-8 C: **8 200 MJ/t/yr central (6 100-10 200)**, i.e. 5.6 (4.2-7.0) MJ/t/tick,
  SDA included. The engine's realised cost, 3.6 x (0.94 x 2 + 0.06) x 1460 = **10 200 MJ/t/yr** at the
  8.4 C mean (~9 500 at 7.5 C) = 2.4 x SMR, sits at the TOP of this range: plausible for cod, high for
  haddock and flatfish.
- FB4 North Sea cod (Hansson et al. 1996) would give 5.75 MJ/t/tick before SDA at 1.2 kg and 7.5 C,
  i.e. ~1.9 x the group SMR; with SDA at an SMS cod ration (Q/B 4.6 on fish prey) ~7.9 MJ/t/tick. So
  the engine's 7.2 MJ/t/tick "eat" cost is in line with a cod bioenergetics model, but cod is 31 % of
  the group and is the most active member.

---

## Q3. SDA (heat increment) and the energy budget convention

| Quantity | Species | Value | Model units | T | Source | Quality / notes |
|---|---|---|---|---|---|---|
| SDA coefficient (SDA / meal energy) | juvenile cod, 5 % BM herring meal | 5.4-6.3 % at 2 and 10 C; 10.4-12.4 % at 5, 15, 20 C; lasted 80 h (10 C) to 250 h (5 C) | | 2-20 | Tirsgaard, Svendsen & Steffensen 2015, Fish Physiol. Biochem. 41:41-50 [A] | |
| SDA | haddock juveniles, pellets ~2 % BM | 2.3-5.2 % of meal energy; peak 30-50 % above RMR; 55-85 h | | 2-11 | Perez-Casanova et al. 2009 [F, prior] | small meals |
| Feeding + SDA | plaice | 15 % (11-19 %) of the MAXIMUM ration energy (fed max minus unfed min respiration); fed/unfed ratio 1.73-2.91 | | 6-18 | Fonds et al. 1985 Tables 7-8 [F, prior] | measured AT SATIATION; this is the source of "2.4" |
| Post-feeding peak | plaice | "approximately twice the resting rate", back to rest in 24-72 h; rises with ration | | | Jobling & Davies 1980, J. Fish Biol. 16:629 [A, prior] | a peak, not a mean |
| Mean peak rise | fishes (review) | peak SDA rise 136 % above SMR on average | | | Secor 2009, J. Comp. Physiol. B 179:1-56 [A] | peak, not time-averaged |
| SDA in FB4 | cod | 0.17 x (C - F) = 0.14 of C | | | Hansson et al. 1996 via FB4 [F] | |
| Egestion / excretion in FB4 | cod | FA 0.17 of C; UA 0.09 of (C - F) | ME = 0.755 C | | FB4 [F] | "assimilation" net of F and U = 0.755 |
| Egestion / excretion | walleye pollock / mackerel | 0.20 / 0.11; 0.16 / 0.10 | | | FB4; Bachiller et al. 2018 [F, predators note] | |
| Ecopath convention | all fish | unassimilated 0.20 (egestion + excretion); respiration = everything else, SDA inside R | | | M&D 2007 [F] | |
| SDA at the field ration | benthivores | 0.10-0.15 x 11 970 MJ/t/yr = 1 200-1 800 MJ/t/yr = 0.8-1.2 MJ/t/tick = **0.23-0.34 x RM** (at Q/B 4.5: 0.29-0.44 x RM) [D] | | | | |

Standard budget (Kitchell et al. 1977; FB4): C = F + U + SDA + R + G, with R = SMR x activity
multiplier and SDA proportional to (C - F). "Assimilation efficiency" in fish bioenergetics usually
means (C - F)/C (absorption, egestion only, ~0.8-0.85 for carnivores); "physiologically useful" or
metabolisable energy is (C - F - U)/C (~0.75); neither nets SDA. Ecopath's 0.8 nets F + U and leaves SDA
inside respiration.

### Synthesis Q3
- SDA scales with meal energy (Jobling & Davies 1980; Tirsgaard 2015; Fonds 1985). A fixed per-tick
  surcharge is equivalent to an SDA only at one ration. The engine's surcharge, 1.0 x RM on 94 % of the
  ticks = **4 950 MJ/t/yr**, equals SDA at a ration of 33 000-50 000 MJ/t/yr, i.e. **0.85-1.25 x Cmax**
  [D]. The fed/unfed 2.4 and the 15 %-of-max-ration of Fonds 1985 describe fish at satiation; wild
  fish eat ~0.3 of Cmax (Q1), where SDA is 0.23-0.44 x RM.
- Double counting: the engine's assimilation 0.7/0.8 nets egestion (+ excretion, by the Ecopath
  reading). It does not net SDA, so a separate SDA term is legitimate, but it must be ~0.14 of the
  ingested energy (FB4) or 0.10-0.15 (Tirsgaard, Fonds), not a fixed 1.0 x RM. A defensible "eat"
  multiplier is **1.3-1.5 x RM** (SDA at the field ration 0.25-0.45 plus some foraging swimming above
  routine; haddock cruising 1.48 vs routine 1.42 x SMR) **J**, or better an SDA proportional to intake.
- The routine factor 1.2 in RM already holds some activity; Chung's activity share (37-59 % of FMR)
  says activity is large in wild cod, but it is mostly paid while moving, which the engine charges via
  `movement_cost` **J**.

---

## Q4. Prey energy density (wet weight, as eaten)

| Taxon | kJ/g WW (range) | kJ/g AFDW | AFDW/WW | Source | Notes |
|---|---|---|---|---|---|
| Polychaetes, errant | Nephtys incisa 4.4, Glycera americana 4.0, Ophioglycera 3.3, Aphrodita hastata 2.4 | 21.8-23.3 (Aphrodita 20.1) | errant 0.166 (0.157-0.185), sedentary 0.140 (0.134-0.166), all 0.156 | Steimle & Terranova 1985, J. Northw. Atl. Fish. Sci. 6:117, appendix [F]; Ricciardi & Bourget 1998, MEPS 163:245, Table 1 [F] | Table 1 polychaete mean 21.85 kJ/g AFDW; 0.156 x 21.85 = 3.4 |
| Ophiuroids, whole | Ophiopholis aculeata 1.8, Ophioderma brevispina 2.1 (ash 68-79 % of DW) | 14.7-17.6 | 0.065 (0.048-0.100) | Steimle & Terranova appendix [F]; Ricciardi & Bourget [F] | Ricciardi route gives 1.05; measured whole animals 1.8-2.1. Stelleroid mean 19.1 kJ/g AFDW |
| Amphiura filiformis | 2.6-3.2 (AFDW 13.7-17 % WW, Rumohr et al. 1987) | 19.1 | | prior split note [S] | Skagerrak population: arm regeneration 13 % of production, attributed to cropping (Sköld et al. 1994, MEPS 103:81 [A]) |
| Heart urchins / echinoids | Moira atropos 0.7; Echinarachnius 2.0; Arbacia 0.6 | 23.3-28 | 0.027 (0.014-0.056) | Steimle & Terranova [F]; Ricciardi & Bourget [F] | |
| Bivalves, whole | Nucula proxima 2.2 (whole), 3.3 (meat) | 21.7 (Table 1 mean) | 0.055 (0.052-0.064) | same [F] | 0.055 x 21.7 = 1.2 |
| Gastropods (prosobranchs) | ~1.6 | 21.2 | 0.075 | Ricciardi & Bourget; Steimle Table 1 [F] [D] | |
| Amphipods | 3.4 | 20.7 | 0.165 | same [F] | |
| Crabs | Cancer irroratus 3.7 | 20.7 | 0.165 (decapods) | Steimle [F] | |
| Hermit crabs (shell-free) | Pagurus bernhardus 6.0, P. acadianus 6.9 | 21.3-22.7 | | Steimle [F] | without gastropod shell |
| Crangon / Dichelopandalus | 3.7 / 5.5 | 18.4 / 24.5 | | Steimle [F] | |
| Northern shrimp | 4.7 (4.6-5.2) | | | Lawson et al. 1998, MEPS 164:13 [F, prior] | |
| Munida, Nephrops | no direct value; decapod mean 3.4, Homarus 4.8 | | | prior split note **J** | gap |

Diet-weighted infauna [D] (weights from `Infauna och rörlig epifauna`: polychaete 0.487, ophiuroid
0.303, other echinoderm 0.044, bivalve 0.093, other mollusc 0.028, amphipod 0.013, other crustacean
0.030, other 0.003; kJ/g WW central (low-high): 3.5 (2.4-4.4), 1.9 (1.05-2.6), 0.8 (0.6-2.0), 1.2
(0.8-2.2), 1.6 (0.8-2.2), 3.4 (2.9-3.6), 3.2 (2.5-3.7), 2.0):

| | MJ/t WW |
|---|---|
| central | **2 620** |
| all-low corner | 1 730 |
| all-high corner | 3 450 |

Energy shares of the eaten infauna: polychaetes 65 %, ophiuroids 22 %, the rest 13 % [D]. Ash-free,
the infauna mix is ~21 kJ/g AFDW (polychaetes 21.9, stelleroids 19.1, bivalves 21.7, Steimle Table 1).

### Synthesis Q4
- **2600 MJ/t WW is supported**: 2 620 central, **1 900-3 300** as a realistic range **J** (the corner
  values need every taxon at its extreme). The brittle-star value 2.0 does not need the "arms cropped"
  argument: measured whole ophiuroids are 1.8-2.1 kJ/g WW (Steimle); only the Ricciardi factor route
  gives ~1.05. Cropped arms are mostly ossicles, so a cropped-arm meal is, if anything, lower per gram
  than the whole animal **J**.
- The as-eaten basis is right for stomach-based Q/B: stomach WW of molluscs and echinoderms includes
  shell and skeleton, and the energy per WW above includes that ash in the denominator.

---

## Q5. Assimilation (absorption) efficiency on benthos

| Quantity | Value | Source | Notes |
|---|---|---|---|
| FB4 cod egestion + excretion | F 0.17, U 0.09 -> ME 0.755 | FB4 / Hansson et al. 1996 [F] | mixed natural diet |
| Ecopath fish | unassimilated 0.20 | M&D 2007 [F] | |
| Chitin digestion, cod on shrimp | ~90 % digested; "only 2.4% of the chitin ... recovered in the collected faeces" | Danulat 1987, Helgol. Meeresunters. [S, predators note] | chitin is not a large energy loss |
| Evacuation of armoured prey | rate halves with "chitin and ash" armament; brown shrimp evacuated ~2x slower than fish | Andersen 1999 [A]; Andersen et al. (J. Fish Biol.) [S] (predators note) | affects Cmax in mass, built into the SMS rations |
| Organic AE on shelled / ophiuroid prey | no measurement found for cod, haddock, plaice or dab | searches 2026-10-26 | gap |

### Synthesis Q5
- Ash carries no energy, so an energy AE applies to the organic part only. On fish and soft prey the
  energy absorption of carnivorous fish is ~0.8-0.85 (F only) and ~0.75 after excretion (FB4).
- For skeleton-rich prey, AE on energy of ~0.65-0.75 is plausible (lower digestibility of tissue bound
  in calcite, more egested organic matter) **J**; no measurement on ophiuroids or bivalves by North Sea
  fish was found. 0.7 on infauna is therefore within the evidence and is NOT an extra shell penalty on
  top of an energy value that already counts the shells as zero.
- Assimilated energy per tonne of infauna eaten: 2 600 x 0.7 = **1 820 MJ/t** (central); realistic
  range **1 300-2 500 MJ/t** (E 1 900-3 300 x AE 0.65-0.75) [D].

---

## Q6. Benthos production vs benthivorous-fish consumption (prior notes, cited)

| Quantity | Value | Source |
|---|---|---|
| Fish consumption of benthos / benthic production, North Sea 1991 | 0.056 of total benthic production (0.077 excluding sessile and meiofauna); infauna 0.04, polychaetes 0.08, crabs 0.82, Nephrops 0.71 | M&D 2007, EcoBase 457 [F, `Bentos och torskfiskar/evidence_C_diets_and_consumption.md`] |
| Fish eating benthic carnivore production | 72 / 53 / 33 % (1970s / 80s / 90s) | Heath 2005, ICES JMS 62:847, Table 4 [F, same] |
| Fish consumption of benthos | average 2.4 g C/m2/yr (~30 g WW) | Heath 2007 [F, same] |
| Cod + haddock benthos consumption | 0.8 (south) to 0.5 (north) g C/m2/yr | Cramer & Daan 1986, ICES CM 1986/G:56 [F, same] |
| Secondary production "is sufficient to supply the food requirements of ... benthivorous demersal fish" | | Greenstreet et al. 1997, ICES JMS 54:243 [A, same] |

### Synthesis Q6
Fish take a few per cent of total macrobenthic production but most of the production of the large
mobile epifauna (crabs, Nephrops). Doubling the benthivores' ration would be supportable by infauna
production in a mass balance; it would not be by the epifauna **J**. No Skagerrak/Kattegat budget with
benthos consumption was found.

---

## Overall budget check and verdict [D, J]

Per tonne of fish per year, 7-8 C:

| Term | Literature-consistent | Engine |
|---|---|---|
| SMR | 4 100 (SMR 2.8 MJ/t/tick) | 5 260 at RM incl. routine 1.2 (8.4 C) |
| Field metabolism incl. SDA | 8 200 (6 100-10 200) | 10 200 (2.4 x SMR at 8.4 C; ~9 500 at 7.5 C) |
| of which SDA | 1 200-2 300 (0.10-0.15 of C) | 4 950 (fixed surcharge) |
| Production P (Z ~0.64 /yr x 4 500 MJ/t) | ~2 900 | ~3 500: derived g pays (r_max 0.31 + M1 0.29 + sum M2 ~0.175) x 4 500 **J** |
| Assimilated need, R + P | 11 100 (9 000-13 100) | predicted 10 200 + 3 500 = 13 700; observed 13 650-14 560 (Q/B 7.5-8 on infauna) |
| Infauna-equivalent Q/B at 1 820 MJ/t | **6.1 (4.9-7.2)** | 7.5-8 |
| Mixed-diet Q/B at ~2 620 MJ assimilated per t | **4.2 (3.4-5.0)** | |
| Published Q/B (mixed diet) | 3.48 (2.5-4.7); SMS gadoids 3.6 | |

Gross efficiency at Q/B 3.48: P / C = 2 900 / 11 970 = 0.24 in energy (0.18 by mass), which is high
for an adult stock (Christensen's critique of the raw MSVPA values); at K1 0.15 the Q/B would be 5.6.

The engine is internally consistent: cost 10 200 + production demand ~3 500 = 13 700 MJ/t/yr / 1 820 =
**7.5 /yr on infauna**, the lower end of what the simulation shows. The 1.5-1.6x gap to the published
ration is therefore two terms: the feeding surcharge (~2 600-3 700 above a field SDA) and production
~600 above a Z-based P; the rest is the low-end published Q/B.

Verdict: neither side alone. The engine total (2.4 x SMR) lies INSIDE the field range, at its top, so
"2x RM is impossible" is not the claim; Chung 2019 and FB4 cod would allow it for cod. (1) The engine's **feeding_cost 2.0** is the most out-of-line single
quantity: it applies a satiation SDA (Fonds 1985 fed/unfed at maximum ration) to a field ration of
~0.3 Cmax, charging ~5 000 MJ/t/yr where SDA is 1 200-2 300; that alone is ~1.5-2.1 Q/B units on
infauna. A 1.3-1.5x eat multiplier, or SDA proportional to intake (0.14 of C), brings the engine to
~5.6-6.6 on infauna (the rest of its need is close to the literature budget). (2) The published Q/B 3.48 is at the low end: an energy budget at measured SMR,
a field multiplier ~2 and a realistic production needs ~4-5 /yr on the mixed diet (~6 on infauna).
(3) Infauna energy (2600) and assimilation (0.7) are supported within +-25 %; they are not the cause.
(4) The 50 m layer temperature costs only ~7 % against real 7-8 C bottom water.

## Gaps

- Brey et al. 2010 (J. Sea Res. 64:334) conversion tables not obtained: the AWI PDF holds only the first
  page; Brey's virtual handbook (thomas-brey.de) is offline; SeaLifeBase "Brey mass conversion" pages
  for Amphiura, Ophiura, Nephtys, Glycera, Lagis, Abra, Echinocardium, Pagurus, Crangon are empty.
- No new primary SMR papers fetched (Saunders 1963, Edwards et al. 1972, Claireaux et al. 2000,
  Duthie 1982, Fonds et al. 1992); SMR values reused from the prior benthivore A note.
- Secor 2009 and Tirsgaard et al. 2015 abstract only; Jordan & Steffensen 2007 (Physiol. Biochem. Zool.
  80:178) blocked (403); the RMR-fed 1.55 x SMR comes via Chung 2019.
- No measured energy AE of cod, haddock, plaice or dab on ophiuroids or bivalves was found.
- Greenstreet et al. 1997, Daan 1973/1989, Jones 1974 and Temming & Herrmann 2003 full texts not
  obtained (Temming & Herrmann 2003, Fish. Res. 63:21, is a cod evacuation paper); the SMS key-run
  rations stand in for evacuation-based field rations.
- Chung 2019 FMR is for Newfoundland cod (migratory, large), with an MTE-predicted SMR denominator; no
  FMR exists for haddock or North Sea flatfish.
- No Skagerrak/Kattegat mass balance of benthos consumption exists.
