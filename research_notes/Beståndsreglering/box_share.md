# WP3: the box's share of each stock, and where its recruits come from (2026-10-29)

Box: 60 x 60 km = 3600 km2, 57.81-58.31 N, 10.07-11.15 E. Grades: **D** computed from data,
**F** stated in a source, **A** analogy, **J** judgement.

The box lies in ICES rectangles 44G0 (north part) and 45G0 (south part). The 11.0-11.15 E edge
reaches 44G1/45G1, which there are mostly land and skerries. It is offshore in the central-eastern
Skagerrak, at the latitude of the Bohuslän coast (Lysekil-Smögen). Hvaler and Koster lie 70-100 km to
the north. [D]

## Takeaway

- **The box holds a small share of every stock.**
  - By survey index (q cancels within a species) it holds **0.1-0.6 %** of the North Sea + 3a
    biomass of herring, haddock, whiting, saithe, Norway pout and sprat in Q1 [D].
  - In Q3 the share is higher for the species that feed here in summer: saithe 1.9 %, Norway pout
    0.8-0.9 %, sprat up to 1.0 % [D].
  - Three members are concentrated here: cod (~2 % of North Sea cod; 5-7 % of the Viking substock),
    long rough dab (~1.7 %) and witch (6-9 %) [D].
  - The SAG checks agree within a factor ~2 (box biomass / stock TSB): herring NSAS 0.22 %,
    haddock 0.44 %, whiting 0.58 %, saithe 0.44 %, cod 1.3 % (whole stock) and 4.5 % (Viking) [D].
- **The box is a nursery and a feeding area, not a spawning area.**
  - Its share of each stock's recruits is 2-5x its share of the spawners. Haddock: age-0 0.75-1.27 %
    against spawners 0.16-0.22 %. Cod: age-1 3.7-5.2 % against 1.7-2.0 %. Herring: age-1
    0.5-0.9 % against 0.04-0.20 %. Whiting: age-1 0.6-1.1 % against 0.25-0.32 % [D].
  - The literature says the same, in so many words, for most members [F]:
    - herring spawns in the western North Sea and at Rügen;
    - "North Sea cod do not appear [to] spawn in the Skagerrak";
    - haddock spawning is "mainly around the east and west coasts of Scotland";
    - Norway pout shows "no evidence of spawning" in the eastern Skagerrak;
    - saithe spawns on the northern shelf edge;
    - blue whiting, mackerel and horse mackerel spawn elsewhere.
- **Almost none of the box's recruits are spawned by the box's own adults.**
  - Data bound [D]: even if every offspring of the box's spawners stayed in the box, they could make
    at most 12-30 % of the box's haddock recruits, 32-54 % of its cod and 4-40 % of its herring.
  - Literature on where spawning actually happens [F]: it puts the true local share near 0 for
    herring, haddock, Norway pout, saithe and the pelagic visitors.
  - Flatfish and perhaps whiting are the exception: plaice has a resident Skagerrak component, and
    witch and long rough dab are concentrated on the Skagerrak and Norwegian Trench slope.
- **Recruitment weight on the box's own adults is small** [D+J]. Define w as the box's share of the
  spawning stock that supplies its recruits. It works out to:

  | Group | w | Range |
  |---|---|---|
  | planktivores | 0.003 | 0.001-0.01 |
  | benthivores | 0.03 | 0.015-0.065 |
  | water column | 0.004 | 0.002-0.008 |
  | whiting | 0.02 | 0.003-0.06 |

- **Conclusion for every group: (ii), the stock SSB as an external driver.**
  - The box receives s_rec x R_stock(SSB_stock(t)). s_rec is the box's share of the stock's recruits
    (table in section 3).
  - Benthivores and whiting could carry a small (iii) term with w <= 0.06. Its effect is well below
    the recruitment noise.
  - Stock-scale compensation (Beverton-Holt in SSB) lives **outside the box**, in R_stock, and is not
    represented by the box adult biomass A. The plan's rho/(1 + A_bar/K_R) on the box's A is the
    wrong variable.
  - Any density dependence the box can carry acts **after settlement, on the box juveniles J, at box
    density**: nursery survival and growth, and density-dependent emigration of J as it grows (WP2).

## 1. Data and method

Scripts in `scripts/`, run with `python3 -I`, paths as arguments. Downloads in `data/share/`
(untracked, untrusted).

**Box biomass.** Model box density x 3600 km2 x member share. The densities in t/km2 are
planktivores 3.3, benthivores 1.53, water column 0.754 and whiting 0.446. Herring is split
NSAS 0.68 / WBSS 0.32, the HERAS July 2025 Skagerrak share (`../Fiskeridödlighet i Skagerrak
idag/ices_pelagic_f.md`). That split is an upper bound for juveniles: HAWG 2026 Table 3.2.4 gives
95-99.5 % NSAS among Skagerrak 0-ringers and 88-98 % among 1-ringers. [D/F]

**Stock.**
- `share_fetch_sag.py` fetches the ICES SAG SummaryTable for each member stock. `share_sag.py`
  computes the box share of the 2001-25 mean SSB, plus the 2021-25 mean (`tables/share_sag.csv`).
- North Sea cod is the sum of the three 2026 substocks, because the whole-stock key 26010 carries no
  SSB. The Viking substock (3.a.20 plus eastern 4.a/4.b) averages 30 % of the summed SSB.
- TSB = SSB x the TSB/SSB ratio from the WGSAM SMS 2023 key run, 2001-22 mean
  (`share_sms_tsb.py`, `tables/share_sms_tsb.csv`). The ratios are cod 2.13, haddock 1.59, whiting
  1.42, saithe 2.08, herring 1.51, Norway pout 3.45, sprat 3.35 and plaice 1.32. Hake uses the SS3
  TBiomass. Other stocks use a [J] ratio of 1.0-1.3 (shares there are < 0.1 % and insensitive to it).

**Survey.**
- `share_extract.py` re-reads the raw NS-IBTS CPUE-per-length XML for the whole survey area: 2000-20
  from `../Fiskgrupper i rutan/data/datras`, 2021-25 from `../Torskfiskarnas
  artsammansättning/data/datras`. It keeps 16 species.
- `share_survey.py` uses valid GOV hauls in Q1 and Q3, 2001-25: 9322 Q1 and 8392 Q3 hauls, of which
  171 / 164 fall in 44G0+45G0.
- Haul key includes the position (the 26D4 collision). Length-weight comes from `../Fiskgrupper i
  rutan/tables/length_weight_all.csv`.
- Estimator:
  - rectangle mean per year and quarter, then the mean over years;
  - stock index = sum over rectangles of the mean x the rectangle's sea area (ETOPO1 2', from NOAA
    ERDDAP);
  - box, two estimators: R = mean of 44G0 and 45G0 x 3600; S = Skagerrak + box-rectangle hauls in
    depth bins, weighted by the box's depth-bin areas (the convention of `../Fiskgrupper i rutan`) x
    3600. The two bound each range below.
- Stage windows [J; the age-0 lengths of `../Årsungarnas andel/summary.md` guided them]:
  - age-0 = Q3 (Aug), small fish: herring < 16 cm, haddock and cod < 20, whiting < 19, Norway pout
    < 11;
  - age-1 = Q1 (Jan-Feb): herring 10-20, haddock 10-25, cod 10-27, whiting 8-22, pout 8-14, sprat
    6-11, plaice 5-15;
  - spawners = length >= L50 in Q1 (pre-spawning for the winter-spring spawners): cod 50, haddock 30,
    whiting 22, saithe 60, pout 14, sprat 11, plaice 27, dab 15, long rough dab 18, witch 32, hake 45,
    spurdog 80. For herring (autumn spawners) and blue whiting the spawner window is >= 24 / 22 cm in
    Q3.
- The age-0 windows depend on the cut-offs and the GOV hardly catches 0-group flatfish or saithe.
  Age-1 in Q1 is the better recruit measure for cod, whiting and herring; age-0 is better for haddock
  (its Q3 0-group is well caught after settlement).
- Survey areas follow each stock's divisions: "4+3a", "4+3a20" (the North Sea stocks without the
  Kattegat), "3a (20+21)" (whg.27.3a), and "Viking>2E", a [J] proxy for the Viking substock (3a.20 +
  4a/4b east of 2 E and north of 56 N; ICES defines it by divisions only).
- **Cross-check:** that proxy holds 33 % of the 4+3a cod biomass index, against the Viking share of
  30 % of SAG SSB [D].
- The sampled rectangles cover 84 % of the 4+3a sea area and 72 % of 3a.20. Unsampled coastal and
  fjord rectangles are missing from the denominators, so the shares are those of the offshore
  survey stock.
- `share_summary.py` joins everything and adds the bound f_self_max = min(1, s_spawners /
  s_recruits). `share_skagerrak.py` gives the Skagerrak and Kattegat shares of each index.

## 2. Per stock

Shares are % of the stock. "Survey" ranges are the R and S estimators, climatology 2001-25.
Per-year medians are in `tables/share_survey_by_year_summary.csv`; single years swing 3-10x.

### Planktivorous fish (3.3 t/km2, 11 880 t)

**Herring, NSAS her.27.3a47d (0.79 x 0.68 = 6380 t) and WBSS her.27.20-24 (3000 t).**
- Box share [D]:
  - NSAS: 0.33 % of SSB (2001-25; 0.38 % in 2021-25) and 0.22 % of TSB.
  - Survey 4+3a (NSAS + WBSS in 3a): biomass 0.42-0.58 %, spawners (Q3 >= 24 cm) 0.04-0.20 %, age-0
    0.69-0.94 %, age-1 0.49-0.89 %.
  - WBSS: 1.6 % of SSB (3.1 % in 2021-25, a depleted stock).
  - The box holds 10-12 % of the Skagerrak herring index. HERAS gives the same scale: 9.4 kt in the
    box against 66-98 kt in the Skagerrak strata in 2024-25.
- Origin [F]:
  - NSAS larvae drift 3-4 months from Buchan, Banks and Downs, 730-1030 km away (`../Den pelagiska
    poolens reglering/migration_scale.md`, not redone).
  - Skagerrak 0-ringers are 95-99.5 % NSAS by genetics (HAWG 2026 Table 3.2.4).
  - WBSS spawn at Rügen. A local spring-spawning "WBSS-Skagerrak" population exists (her.27.20-24
    annex 2026, Bekkevold et al. 2023), but it is small and coastal.
- Self-recruitment: bound f_self_max 0.04-0.40 [D]; actual ~0 at box scale [F/J].
- The 3a share of NSAS 1-ringers rises with year-class strength (NSAS annex, via migration_scale.md):
  a stock-scale spill-over, not local density dependence. [F]

**Blue whiting, whb.27.1-91214 (1250 t).** 0.03 % of SSB [D]. Spawning is "along the shelf edge and
the banks west of the British Isles" (annex 2026) [F]. Survey shares (3 % of the Q3 4+3a index) only
measure the North Sea fringe of the stock. Local recruits: 0.

**Norway pout, nop.27.3a4 (890 t).**
- Box share [D]:
  - 1.7 % of SSB, 0.50 % of TSB (SMS ratio 3.45: the stock is mostly juveniles).
  - Survey: biomass 0.10-0.21 % (Q1), 0.81-0.88 % (Q3); spawners 0.06-0.13 %; age-0 0.32-0.49 %;
    age-1 0.13-0.27 %.
- Origin [F]: "Norway pout in the eastern Skagerrak is only to a very small degree a self-contained
  stock and adults migrate out of the Skagerrak and the Kattegat to spawn, because there is no
  evidence of spawning there ... The main bulk drifts as larvae south-eastwards from more western
  areas" (annex 2025, Sec. 1.1). Munk et al. 1999 find pout larvae offshore of the Skagerrak shelf-
  break front: the highest of all gadoids in May 1992, and none in 1994 (advective loss).
- Local recruits: ~0. The box is a sink supplied by drift.

**Sprat, spr.27.3a4 (360 t).**
- Box share [D]: 0.22 % of SSB, 0.07 % of TSB. Survey: biomass 0.06-0.29 % (Q1), 0.22-1.0 % (Q3);
  spawners 0.05-0.15 %.
- Origin [F]: offshore Skagerrak-Kattegat sprat "all grouped with the North Sea, irrespective of
  the time they were collected" (Quintela et al. 2020). Hybrids occur near the Swedish coast; fjord
  sprat are distinct. Spawning is protracted and batch-wise, so local spawning cannot be excluded.
- Local share unknown, <= 0.1 [J].

### Benthivorous fish (1.53 t/km2, 5508 t)

**Haddock, had.27.46a20 (0.386, 2130 t).**
- Box share [D]:
  - 0.70 % of SSB (0.33 % in 2021-25, the haddock peak) and 0.44 % of TSB.
  - Survey 4+3a20: biomass 0.15-0.24 % (Q1), 0.34-0.49 % (Q3); spawners 0.16-0.22 %; age-0
    0.75-1.27 %; age-1 0.23-0.41 %.
  - The Skagerrak holds 1.3 % of the stock's Q1 spawners and 8 % of its Q3 age-0.
- Origin [F]: spawning "mainly around the east and west coasts of Scotland. The haddock found in the
  Skagerrak consist mainly of younger fish" (annex 2025). Juveniles lie "along the continental shelf
  at around 200 m and continuing round to the Skagerrak". Albert 1994 (abstract): mature fish
  "probably left the area in winter".
- Self-recruitment: bound f_self_max 0.12-0.30 [D]; actual ~0 [F/J].

**Cod, cod.27.46a7d20 (0.314, 1730 t).**
- Box share of SAG [D]: 2.9 % of the summed substock SSB and 1.3 % of TSB. Against the Viking
  substock, 9.5 % of SSB and 4.5 % of TSB.
- Survey [D]:
  - 4+3a20: biomass 1.9-2.3 %, spawners 1.7-2.0 %, age-1 3.7-5.2 %, age-0 0.3-0.5 % (Q3 0-group cod
    sit in the southern North Sea).
  - Viking proxy: biomass 5.4-7.5 %, spawners 5.5-6.6 %, age-1 6.0-8.4 %.
  - The Skagerrak holds 30 % of the 4+3a Q1 age-1 index (the Kattegat another 20 %), against 6 % of
    the Q1 spawners.
- Origin [F]:
  - WKNSCodID 2020: "The Skagerrak and northern Kattegat appear to be a nursing ground for Viking
    and Dogger cod, with most cod in the Skagerrak being Viking cod"; "North Sea cod do not appear
    [to] spawn in the Skagerrak and Kattegat, but they migrate out of the region to spawn".
  - WKBCOD 2023: "Skagerrak (27.3.a.20) is a nursing area for the Southern stock". Its recruitment
    model takes the Viking (Skagerrak) index to cover 100 % of Viking and 25 % of Southern
    recruitment; IBTS Q3 2021 juveniles show "a dominance of Viking juveniles in the Skagerrak".
  - Larval supply: Munk et al. 1995 find cod larvae and juveniles along the 80-140 m isobaths of the
    inner Skagerrak slope front in May. Their survey area "between 58 N and 58 30 N, east of 10 E"
    overlaps the box. Some larvae were as small as North Sea ones, so local spawning or fast
    transport (>= 10 days over 220 km) is possible.
  - Huserbråten et al. 2018 (ROMS drift): 28-66 % of North Sea-spawned particles pass through the
    Skagerrak, but only 2 % of settlers settle in the Skagerrak in a normal year. That is biased low,
    because a median 39 % of the Skagerrak particles are lost across the Kattegat boundary.
- Inshore, not the box [F]: Norwegian Skagerrak coast 0-group 35 % North Sea type and 65 % local
  fjord type (Hvaler 56 % North Sea; years <20-54 %; Knutsen et al. 2018). North Sea drift supplies
  1-10 % of the fjord 0-group (Stenseth et al. 2006). Fjord cod are sedentary ("up to 98 % ...
  within 2 km") and are not in the offshore box.
- Self-recruitment: bound f_self_max 0.32-0.54 against the whole stock [D]; actual ~0-0.1 offshore
  [F/J].
- The cod in the box are Viking (plus some Southern) cod, spawned in the northeastern North Sea,
  settled in the Skagerrak, and leaving to spawn. The box adults (5-7 % of Viking spawners) feed back
  to the box only through the Viking SSB.

**Long rough dab (0.126, 690 t; not assessed).** Survey: biomass 1.6-1.9 %, spawners 1.4-1.6 % of
4+3a, and 15 % of the Skagerrak's [D]. No Skagerrak egg or spawning source was found. The species is
concentrated on the Skagerrak and Trench slope (Skagerrak 10-13 % of the 4+3a index), so a
Skagerrak-scale local share of 0.3-1 is plausible [J].

**Dab, dab.27.3a4 (0.045, 250 t; relative SSB only).** Survey: 0.04-0.4 % of 4+3a, spawners
0.04-0.07 % [D]. No origin source. Local share unknown [J].

**Plaice, ple.27.420 (0.038, 210 t).**
- Box share: 0.024 % of SSB, but 0.34-1.9 % of the survey index [D]. The model's GOV-based plaice
  density is low against the assessment; the survey share is the relevant one.
- Origin [F]: WKPLE 2015 finds "approximately half of the fish caught in Western Skagerrak had a
  North Sea genetic print" at spawning time. There is "a resident plaice population in Skagerrak
  ... importantly mixed with the North Sea population"; "Eggs and larvae from North Sea Dogger Bank
  do likely not drift into Skagerrak".
- Local (Skagerrak) share of the box recruits ~0.5, range 0.3-0.7 [F/J]. The box holds ~9 % of the
  Skagerrak spawners [D].

**Witch, wit.27.3a47d (0.017, 94 t).**
- Box share [D]: 3.0 % of SSB. Survey: spawners 7.3-8.9 % of 4+3a; the Skagerrak holds 39 % of the
  index.
- Origin [F]: "a continuum from 3.a into the Norwegian trench" (annex 2025); no spawning-ground
  information.
- The stock is centred on this slope, so the box's own share of the supplying spawners is 0.07-0.2
  [D/J].

### Water-column predators (0.754 t/km2, 2714 t)

**Saithe, pok.27.3a46 (0.651, 1770 t).**
- Box share [D]: 0.9 % of SSB (1.4 % in 2021-25) and 0.44 % of TSB. Survey 4+3a: biomass
  0.24-0.38 % (Q1), 1.9 % (Q3); spawner-size fish in Q1 0.25-0.37 %. The stock also extends to 6.a,
  so the survey shares are upper bounds.
- Origin [F]:
  - Spawning in Jan-May "along the edge of the continental shelf". Juveniles in "inshore nursery
    grounds ... along the coast and in the fjords of western and southern Norway, the coast of
    Shetland, and the coast of Scotland", moving to the North Sea at ages 2-5 (annex A.4; FishMap).
  - "Smaller/younger fish are found in the Skagerrak compared to Subareas 4 and 6."
- Local recruits: 0. The box's trench-slope saithe are ages 1-4 passing through.

**Mackerel, mac.27.nea (0.128, 350 t): 0.005 % of SSB.** The North Sea component (~3 % of NEA)
"spawns in the North Sea and Skagerrak (ICES Subarea 4 and Division 3.aN)" (annex 2024) [F]. The box
is a summer feeding ground. Local recruits ~0.

**Spurdog, dgs.27.nea (0.08, 220 t): 0.06 % of SSB (SAG; the survey gives 0.9 % of the 4+3a Q1 index
of >= 80 cm fish).** Viviparous, with a ~2-year gestation. Pups (<= 25 cm) are caught "in very small
numbers", and pupping grounds are unknown (annex 2025) [F]. Resident females can pup locally, but the
NE Atlantic stock is the unit. w <= 0.01 [J].

**Horse mackerel, western hom.27.2a3a4a5b6a7a-ce-k8 (0.055, 150 t): 0.01 % of SSB.** "catches in
divisions 3.a and 4.a are now considered to belong to the western stock". It spawns in Biscay and SW
of the British Isles (annex 2025) [F]. Local recruits 0.

**Hake, hke.27.3a46-8abd (0.04, 110 t): 0.07 % of SSB, 0.04 % of TSB.** Survey: 0.3-0.8 % of the 4+3a
index, spawners 0.02-0.05 %, age-0 0.58-0.70 % [D]. North Sea spawning is in summer; "0-group hakes
reach the bottom in depths of more than 200 m"; the main nurseries are in Biscay and S Ireland [F].
Self-recruitment bound 0.02-0.08 [D].

### Whiting (0.446 t/km2, 1606 t)

**whg.27.3a (survey index only) or whg.27.47d.**
- Box share [D]:
  - 0.82 % of the whg.27.47d SSB and 0.58 % of TSB.
  - Survey 4+3a: biomass 0.27-0.63 %, spawners 0.25-0.32 %, age-0 0.27-0.78 %, age-1 0.61-1.12 %.
  - Survey 3a (20+21): biomass 4.1-9.8 %, spawners 4.9-6.2 %, age-1 3.7-6.7 %.
- Origin [F]:
  - "There is a paucity of information on the population structure of whiting in 3.a ... No genetic
    surveys have been conducted" (annex 2026). The early-1990s landings drop is unique to 3a, "an
    indication of local stock structure".
  - "Part of the Skagerrak population is thought to migrate into the north-eastern North Sea to
    spawn" (FishMap, citing Knudsen 1964); catches in 3a "are dominated by 1-year-old whiting".
  - Whiting larvae sit inshore of the Skagerrak slope front in every May survey 1991-94 (Munk et al.
    1999).
- Self-recruitment bound [D]: 0.23-0.53 against 4+3a, 0.72-1 against 3a.
- Origin unresolved. Local Skagerrak share of the box's recruits 0.2-0.5 [J].

## 3. The group table

Notation:
- s_B = box share of the stock's biomass;
- s_R = box share of its recruits;
- f_loc = share of the box's recruits spawned by Skagerrak-resident fish;
- w = box share of the spawning that supplies the box's recruits. For the North Sea stocks this is the
  box's share of the stock's spawners; for a Skagerrak component it is f_loc x the box's share of the
  Skagerrak spawners. Group values are weighted by member biomass share.

| Group (members) | s_B (survey; SAG TSB) | s_R | f_loc | w (range) | Recruitment form | Compensation |
|---|---|---|---|---|---|---|
| planktivorous_fish (herring 0.79, blue whiting, Norway pout, sprat) | herring 0.4-0.6 % / 0.22 %; pout 0.1-0.9 % / 0.5 %; sprat 0.06-1 % / 0.07 %; blue whiting 0.02 % | herring age-1 0.5-0.9 %, age-0 0.7-0.9 %; pout 0.13-0.49 % | ~0 (herring 95-99.5 % NSAS juveniles; pout and blue whiting spawn elsewhere; sprat <= 0.1) [F/J] | **0.003** (0.001-0.01) [D/J] | (ii): s_R x R_NSAS(t) (+ WBSS 0-5 %), external | stock scale, outside the box |
| benthivorous_fish (haddock 0.386, cod 0.314, LRD 0.126, dab, plaice, witch) | haddock 0.15-0.49 % / 0.44 %; cod 1.9-2.3 % / 1.3 % (Viking 5-7 % / 4.5 %); LRD 1.6-1.9 %; witch 6-9 % / 2.3 % | haddock age-0 0.75-1.27 %; cod age-1 3.7-5.2 % (Viking 6-8 %) | haddock ~0, cod ~0-0.1 offshore, plaice ~0.5, LRD and witch 0.3-1 [F/J] | **0.03** (0.015-0.065) [D/J]; cod 0.03-0.10, haddock 0.002-0.007, flatfish 0.02-0.2 | (ii) for haddock + cod (Viking + Southern SSB forced); flatfish (iii) at <= 0.2 | stock scale for the gadoids; Skagerrak-scale for the flatfish; post-settlement on J in the box |
| predatory_fish (saithe 0.651, mackerel, spurdog, horse mackerel, hake) | saithe 0.24-1.9 % / 0.44 %; others <= 0.06 % SSB | saithe juveniles coastal, not in the box | 0 [F] | **0.004** (0.002-0.008) | (ii): an immigration flux of ages 1-4 saithe from coastal nurseries, not recruitment | outside the box |
| whiting | 0.27-0.63 % of 4+3a; 4-10 % of 3a / 0.58 % | age-1 0.6-1.1 % of 4+3a, 3.7-6.7 % of 3a | 0.2-0.5 [J] | **0.02** (0.003-0.06) | (ii) with whg.27.3a or whg.27.47d SSB; (iii) with w <= 0.06 | stock or 3a scale; post-settlement on J |

How w is put together [D/J]:
- Planktivores: 0.79 x (0.95 x 0.003 + 0.05 x 0.02) + 0.075 x 0.001 + 0.03 x 0.001.
- Benthivores, central: 0.386 x 0.004 + 0.314 x 0.055 + 0.126 x 0.05 + 0.045 x 0.01 + 0.038 x 0.045
  + 0.017 x 0.08.
- Cod: the Viking spawner share 0.055-0.066 x the Viking part of the Skagerrak recruits (~0.8, from
  WKBCOD's 100 % Viking + 25 % Southern), plus the Southern share x 0.2.
- Water column: saithe 0.0025-0.009, spurdog <= 0.009, the rest ~1e-4.

## 4. What it means for the model [J unless tagged]

1. **Recruitment to the box is a forcing, not a feedback.**
   - Use rho(t) x R_stock(t), with R_stock from the SAG series (or its mean, or a Beverton-Holt in a
     forced stock SSB from WP1) and the box share s_R above.
   - The box's adults are 0.2-6 % of the SSB that feeds it. A local impact that halves the box adult
     biomass changes its recruitment by w/2 <= 3 %, inside the year-to-year recruitment CV (> 0.5;
     WP1).
   - Tying recruitment to the box's own A with stock-scale steepness would credit the box with a
     compensation that happens 300-1000 km away. A local loss would then be paid back by local
     recruitment that does not exist.
2. **Where compensation goes.**
   - (a) **Stock scale**: inside R_stock(SSB), outside the box, a fixed or forced series.
   - (b) **Box scale, post-settlement on J**: density-dependent survival, growth or emigration of the
     juvenile compartment at box J density. This is the only density dependence the box itself can
     carry. Its strength is a WP1/WP2 question; it is not estimated here.
   - (c) **Not on A.** Adult regulation in the box comes from emigration (spawning migrations: cod and
     haddock to the NE North Sea and Scotland, herring to the western North Sea, pout west) and
     mortality, as in the Ecosim I - eB closure of migration_scale.md.
3. **The box exports adults.** s_R is 2-5x the spawner share, so in a closed box graduation to A
   piles up adults that in reality leave. A regional K was doing that job phenomenologically. The
   replacement needs the emigration term of WP2 alongside the external recruitment.
4. **Flatfish and whiting** are the only members where a box-scale stock-recruitment term could
   matter (w up to 0.2 for witch and long rough dab, 0.06 for whiting). Together they are ~20 % of the
   benthivores plus the whiting group. A mix (iii) with w at the values above is defensible.
   Constant external recruitment is the simpler choice and loses little.
5. **The nursery-habitat claim, checked** (`../Trålning och uppväxtområden/nursery_areas.md`: grid ~0.3-0.6 % of the regional juvenile habitat, pool-weighted; cod ≈ 0).
   - The box holds 0.75-1.27 % of the surveyed haddock 0-group, 0.6-1.1 % of the whiting 1-group
     and **3.7-5.2 % of the cod 1-group** [D].
   - Weighted by the old gadoid pool (haddock 0.54, whiting 0.27, cod 0.06, saithe 0.12) that is
     ~0.8-0.9 %, 1.5-3x the habitat estimate.
   - Cod is the big miss: the offshore Skagerrak is a Q1 1-group cod hotspot (30 % of the 4+3a index;
     WKBCOD calls it a nursery). The cod ≈ 0 rested on the coastal 0-group settlement nursery only.
   - The survey cannot see inshore nurseries (< 20 m, eelgrass), so the true shares of the whole
     juvenile stock are somewhat lower than these.

## 5. Caveats

- **q and the denominators.**
  - Within a species and stage the gear catchability cancels, except through depth and length mix
    and the day-time GOV's poor catch of pelagic schools.
  - Survey shares are of the offshore 4+3a survey area. They exclude 6a (haddock, saithe, cod
    Northwestern), 7d, the coastal and fjord rectangles (16 % of the 4+3a sea area, 28 % of 3a.20),
    and depths below the trawlable range.
  - For cod and haddock the stock-wide shares are therefore slight overestimates. For saithe and
    hake they are upper bounds.
- **SAG shares mix two things.**
  - The model's member densities are GOV q = 1 values (plaice is ~20x low against its assessment;
    witch and hake are low too). The survey share is the better measure of where the stock is.
  - Share of SSB overstates the share of mature fish, because the box biomass includes juveniles. Use
    the TSB column or the survey spawner column.
- **Stage windows are fixed lengths [J]**, not age keys. Year-class and growth variation leak between
  age-1 and age-2 (cod, haddock). Age-0 in Q3 is poorly caught for cod, saithe and flatfish.
- **Q1 spawner-size fish in the box are not spawning in the box.** For cod, haddock and pout the
  literature says they leave to spawn. f_self_max is a bound under full retention and equal
  survival; the literature, not the data, puts the actual value near 0.
- **Years swing.** Per-year shares vary 3-10x (`tables/share_survey_by_year_summary.csv`; e.g. cod
  age-1 Q1 1.5-3.8 %, haddock age-1 0.03-1.25 %, p10-p90). The climatology weights the strong years.
- **Herring stock split** uses the July 2025 HERAS share (0.32 WBSS; 0.14 in 2024). Juveniles are
  95-99 % NSAS, so the box's herring recruitment is effectively NSAS.
- **Not found:**
  - an offshore (slope or trench) Skagerrak local-vs-imported share for any species;
  - the North Sea share in Jonsson et al. 2016 (abstract only);
  - Skagerrak spawning data for long rough dab, dab and witch;
  - any 3a whiting stock-identity study.

## Sources (F = read; FT = full text, AB = abstract only)

Downloads are in `data/share/lit/`.

| Source | Used for | Read |
|---|---|---|
| ICES SAG SummaryTable, assessment keys 22468, 21294, 21311, 26011, 25908, 22417, 26001-26003, 26010, 22489, 22470, 22534, 19480, 22504, 25875, 25892, 25921, 22627, 25850, 22563, 22488 (https://sag.ices.dk/SAG_API/api/) | SSB, TSB | data |
| ICES WGSAM North Sea SMS key run 2023, summary.out (repo copy in `../Fiskgrupper i rutan/predators/data/population/`) | TSB/SSB ratios | data |
| ICES DATRAS NS-IBTS HH and CPUE-per-length, 2000-25 (repo copies) | survey shares | data |
| NOAA ERDDAP etopo180 (ETOPO1) | rectangle sea areas | data |
| Huserbråten, Moland & Albretsen 2018, Prog. Oceanogr. 167:116, doi:10.1016/j.pocean.2018.07.005 | cod drift; 2 % of settlers in the Skagerrak | FT |
| ICES 2020 WKNSCodID, ICES Sci. Rep. 2:89, doi:10.17895/ices.pub.7499 | cod: Skagerrak a nursery; no spawning; tagging | FT |
| ICES 2023 WKBCOD, ICES Sci. Rep. 5:37, doi:10.17895/ices.pub.22591423 | Viking juveniles dominate the Skagerrak; index coverage 100 % Viking + 25 % Southern | FT (sections 4.1 and the substock model text) |
| Knutsen et al. 2018, Evol. Appl. 11:1527, doi:10.1111/eva.12640 | coastal 0-group ecotype shares | FT |
| André et al. 2016, BMC Res. Notes 9:63, doi:10.1186/s13104-016-1878-9 | eastern Skagerrak juveniles = eastern North Sea cod | FT |
| Stenseth et al. 2006, Proc. R. Soc. B 273:1085, doi:10.1098/rspb.2005.3290 | 1-10 % of the fjord 0-group from the North Sea | AB |
| Knutsen et al. 2004, Proc. R. Soc. B 271:1337, doi:10.1098/rspb.2004.2721 | inflow-year dependence | AB |
| Jonsson et al. 2016, Fish. Oceanogr. 25:210, doi:10.1111/fog.12146 | Kattegat supplies 30 % (72 % in the 1970s) of the Swedish Skagerrak larvae | AB |
| Munk et al. 1995, MEPS 125:21, doi:10.3354/meps125021; Munk et al. 1999, MEPS 177:221, doi:10.3354/meps177221 | gadoid larvae at the Skagerrak shelf-break front | FT |
| Barth et al. 2017, 2019; Ciannelli et al. 2010; Fox et al. 2008; Svedäng & Bardon 2003; Heath & Gallego 1998; Albert 1994; Staby et al. 2018 | context | AB |
| ICES stock annexes: had.27.46a20 (2025), nop.27.3a4 (2025), ple.27.420 (2025), wit.27.3a47d (2025), spr.27.3a4 (2019), whb (2026), whg.27.47d (2026), her.27.20-24 (2026); and the repo copies of pok.27.3a46, whg.27.3a, mac.27.nea, hom western, hke northern, dgs.27.nea | spawning grounds and stock units | FT |
| ICES 2015 WKPLE | Skagerrak plaice components | FT (section 4) |
| ICES 2026 HAWG, ICES Sci. Rep. 8:16, Table 3.2.4 | NSAS/WBSS by age in 3.a.20 | FT (table) |
| Quintela et al. 2020, Evol. Appl. 13:1906, doi:10.1111/eva.12942 | sprat genetics | FT |
| Charrier et al. 2007, MEPS 330:201, doi:10.3354/meps330201 | whiting structure (no 3a sample) | FT |
| ICES FishMap factsheets, saithe and whiting (repo copies) | nurseries, migrations | FT |
| Nordic Council 2024:032 (Skagerrak stocks) | long rough dab common, unmanaged | FT |
