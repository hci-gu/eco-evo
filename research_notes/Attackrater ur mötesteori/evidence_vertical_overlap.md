# Vertical overlap of the planktivores with the 0-25 m zooplankton pool, and light at depth

Fork C of the attack-rate derivation for `planktivorous_fish -> zooplankton` (Skagerrak box,
58.1 N 10.8 E, depth 44-529 m, mean 167 m). Written 2026-10-25. No code, library or manifest
was changed.

Tags: [F] full text / data read, [A] abstract, [S] secondary or search snippet, [D] computed
here, J = judgement (stated). Scripts in `scripts/`, their stdout in `tables/`, downloads in
`data/vertical/` (README there).

## 0. What is being converted, and the definition that fixes H

- The model's zooplankton is the **resident 0-25 m community at A17**: a WP2 0-25 m
  integral, expressed as t/km2 (resume 183.6; `../Sillens vinterföda/winter_food_and_feeding.md`
  Q1: all carbon hauls at A17 are WP2 90 um, 0-25 m). SHARK holds no other strata at A17:
  119 visits at 0-25 m and 2 at 0-30 m, 2007-2023 (`tables/shark_a17_strata_stdout.txt`) [D].
- So the pool's mean concentration in its own layer is, by construction, **c_pool = B / 25 m**.
  The prey-layer thickness for a fish foraging inside 0-25 m is **H = 25 m**. That is a
  definition, not an estimate.
- Real fish are not all inside 0-25 m, and the copepods are not evenly spread inside it. I
  therefore report an encounter-weighted overlap
  `O = sum over depth bins of p_fish(bin) x r_prey(bin)`,
  with r = prey concentration in the bin relative to B/25 m. Then **H_eff = 25 m / O** turns
  the depth-integrated pool biomass into the concentration the average foraging fish meets:
  `encounter rate = V_clear x B / H_eff`.
- O = 1 means "always inside 0-25 m among average prey". O > 1 is possible if fish forage at the
  0-10 m maximum.

## 1. Prey: where the resident copepod biomass sits

### Literature and data
- **Tonnesson, Nielsen & Tiselius 2006** (MEPS 314:213) [F]. Skagerrak stations K, H, T1-T4
  (water depth 200-600 m). 45 um pump samples at 0-10, 10-25, 25-40, 40-60 and 60-100 m in
  March 1999 and Aug-Sep 2000. Table 2 gives abundance per taxon and stratum, plus mean size.
  "Prey abundance showed no differences between day and night" (both seasons).
- **Maar et al. 2006** (MEPS 313:157) [F]. Same cruises: "In the Skagerrak, there was no
  significant diel vertical migration". Mean centre of mass of the calanoids (CIII-CVI) was
  12-22 m (Calanus spp. 18.3 m day, 13.5 m night; Pseudocalanus 12.0 / 13.9 m). In March the
  halocline under the Baltic outflow sat at 5-12 m.
- **Falkenhaug et al. 2022** (Front. Mar. Sci. 9:779335) [F]. At Arendal St. 2 a pronounced
  pycnocline sits "at 15-20 m from March to October", with fresh coastal water in the upper 30 m.
  C. finmarchicus in the northern North Sea "start the descent for diapause in July and [are]
  completely absent from the surface waters by September" (Jonasdottir et al. 2005, cited there).
  The overwintering G0 comes up in February from the Norwegian Trench.
- **Bamstedt 2000** (Mar. Biol., Kosterfjorden) [S, search snippet]. C. finmarchicus CV
  "aggregated in the deep water from October to March and performed DVM in April to June".
  Adult males do normal DVM from spring to early autumn and reverse DVM in the cold season.
- **Bagoien, Kaartvedt & Overas 2000** (Sarsia, Oslofjorden) [A]. In winter Calanus
  "became increasingly concentrated towards the bottom (120 m)". Most descended in September.
  **Kaartvedt, Rostad & Titelman 2021** (J. Plankton Res. 43:199, Bunnefjorden, 150 m) [F]: a
  Calanus layer at ~80-100 m in Jan-Feb, no diel pattern in the hypoxic layer. "sprat dived into
  the shallowing Calanus-layer later in winter".
- Winter profile: the only depth split near the site is Anholt E, 5 Nov 1996. There 20-50 m
  holds 7108 ind m-3 against 16264 in 0-20 m, so r(20-50 m) = 0.44
  (`../Sillens vinterföda/winter_food_and_feeding.md`) [F/D].

### Calculation (`scripts/skagerrak_copepod_strata.py`, `tables/skagerrak_copepod_strata_stdout.txt`) [D]
Biomass per individual is taken as L^b (b = 3; 2.7 and 3.3 give the same shares to +-0.01), with
L the table's mean size per taxon and season. The "herring prey" set drops Microsetella (a
harpacticoid on aggregates) and Oithona (small ambush feeder); that choice is J.

| Herring-prey copepods, 0-100 m | Spring (Mar 1999), 8 stations | Autumn (Aug-Sep 2000), 5 stations |
|---|---|---|
| Share of the 0-100 m biomass in 0-25 m | **0.83** (0.55-0.97) | **0.79** (0.60-0.91) |
| Concentration 0-10 m / 0-25 m mean | 1.48 (0.60-2.14) | 1.96 (1.70-2.08) |
| Concentration 10-25 m / 0-25 m mean | 0.68 (0.24-1.26) | 0.36 (0.28-0.54) |
| Concentration 25-60 m / 0-25 m mean | 0.20 (0.03-0.42) | 0.12 (0.03-0.19) |
| Concentration 60-100 m / 0-25 m mean | 0.07 (0.01-0.15) | 0.09 (0.02-0.29) |
| Centre of mass (m) | 19 | 18 |
| Calanus alone: share in 0-25 m | 0.73 (0.00-1.00) | 0.54 (0.35-0.81) |

### Inference
- In spring and late summer about 80 % of the resident copepod biomass of the upper 100 m is in
  0-25 m. Below 25 m the concentration is only ~0.1-0.2 of the pool mean, and inside 0-25 m it
  peaks in 0-10 m (1.5-2x). There is no diel migration of the resident community, so a deep fish
  does not get the prey delivered at night.
- Calanus is the exception. It is deeper in autumn (about half below 25 m) as it starts to
  descend. From October to March it sits deep (Bamstedt; Bagoien; Kaartvedt 2021; Norwegian Trench
  G0), and that part is **excluded from the pool by definition** in Dec-Mar (resume 183.6).
- Winter (Nov-Feb) has no stratified Skagerrak data here. The column is mixed below the thin
  Baltic surface layer, so J: share in 0-25 m ~0.4-0.7, r(25-60 m) ~0.3-0.6 (the Anholt 0.44 is
  the only number).
- Krill (M. norvegica, Thysanoessa) are not in the pool, since WP2 0-25 m is not quantitative
  for them (sibling note). By day they sit in a 20-40 m thick layer below ~70 m (Onsrud et al.
  2004, Oslofjord [F]) or around 100 m (Degnbol & Munch-Petersen 1985, Skagerrak [F]). At night
  the small krill go to the surface and the large ones scatter around 100 m.

## 2. The fish: depth by day, twilight and night

### 2.1 Herring (and sprat)
- **HERAS in the box, July 2010-2020** (RV Dana logs, ICES acoustic-trawl database, category MIX
  = all fish echoes, herring ~87 % of the acoustic planktivore biomass per
  `../Ekolodsdata planktivorer/acoustic_planktivores.md`) [D]
  (`scripts/heras_herring_depth.py`, `tables/heras_herring_depth_stdout.txt`). The NASC share
  is binned by channel upper depth, and integration starts at ~7-10 m:

  | Sun elevation | logs | 0-10 | 10-25 | 25-50 | 50-100 | >100 m | mean depth | share in bottom 10 m |
  |---|---|---|---|---|---|---|---|---|
  | day > 6 deg | 1365 | 0.03 | 0.39 | 0.36 | 0.21 | 0.01 | 36 m | 0.34 |
  | low sun 0-6 deg | 181 | 0.04 | 0.39 | 0.20 | 0.36 | 0.00 | 40 m | 0.13 |
  | twilight -12-0 deg | 193 | 0.03 | 0.53 | 0.41 | 0.01 | 0.01 | 25 m | 0.01 |

  The wider region (57.5-58.8 N, 9-11.5 E) gives 0.50 / 0.72 in 0-25 m by day / twilight.
  There are no logs below -12 deg, because the survey does not integrate at night. 2021+ logs
  are single-layer and were skipped. The 7-10 m below the transducer are unseen (HERAS 2023
  flagged near-surface herring, sibling note), so the 0-25 m shares are lower bounds.
- **North Sea adult herring, summer** (Fernandes et al. 2001, ICES CM 2001/Q:08) [F]: schools by
  day "either near the sea bed or in midwater"; "mean depth of schools was close to 90% of the
  water column depth for most of the day"; "These schools disperse into surface waters at night";
  break-up at dusk 18:16-20:56, settlement at dawn 04:17-04:57. In the box, July herring are
  shallower than this (mean 36 m, 34 % near the bottom), probably because the stock is juvenile
  (age 0-1 64 % of Q3 biomass, sibling `winter_budget.md` section 4). J.
- **Winter, adults, Norwegian Channel and Skagerrak** (Falk 1968, ICES CM 1968/H:17, March-April,
  poor OCR) [F]: wintering autumn-spawning herring do diel migration, but "The upper limit of the
  nightly herring traces is situated at different levels between 0 and 100 m". It lies "in or
  below the steepest gradients of the discontinuity layer" under the Baltic outflow water, which
  adult herring avoid.
- **Winter, Oslofjord** (Onsrud et al. 2004, ICES JMS 61:1278, Nov and Mar) [F]: herring by day
  in the upper-middle part of the krill layer (> ~70 m). At night "herring, whiting, and sprat
  were most abundant in the upper 50 m", herring "near the surface". 1 % light depth: 4 m in
  November, 12-15 m otherwise.
- **Winter, at the site** (IBTS Q1 GOV, all but one haul by day, sibling `winter_budget.md`
  section 4) [D]: herring are caught near the bottom by day (Q1 148 kg/h, 1-2 ringers ~90 %).
- **Juveniles** (Jensen et al. 2011, J. Fish Biol., Baltic bay) [A]: YOY herring do reverse DVM,
  "deeper at night and, on average, shallower during the day", "do not feed at night".
- **Feeding times** (Darbyson et al. 2003 [A]; Bernreuther 2013 via sibling [F]): fullness
  peaks in the morning and evening, or late afternoon. Little feeding at night.
- **Sprat** (Solberg & Kaartvedt 2015/2017, Oslofjord winter) [A]: by day schools in midwater
  (~50 m) or solitary near the bottom, "a bimodal vertical distribution with an additional part
  of the sprat population distributed in upper waters"; varied DVM. Sprat feed at night too
  (Bernreuther 2013 via sibling [F]).

### 2.2 Norway pout
- 5-10 m above the bottom, "somewhat closer by day"; main aggregations around the 150 m contour;
  50-300 m (`../Fiskgrupper i rutan/planktivores/evidence_pout_bluewhiting.md` 5.2) [A/F].
- Oslofjord (Onsrud et al. 2004) [F]: pout approach the krill "from below" by day. At night
  there were "no catches in the upper 25 m", with the maximum in midwater. In November they
  lived "pelagically below about 90 m"; in March they were bottom-associated.
- Inference: pout never enter the pool's layer. Their zooplankton is krill, hyperbenthos and deep
  Calanus, which are not in the pool (sibling: small pout eat copepods by day near the bottom,
  large pout eat krill at dusk and night).

### 2.3 Blue whiting
- **Skagerrak, NNE of Skagen, 250 m, 7-14 Aug 1984** (Degnbol & Munch-Petersen 1985, ICES CM
  1985/H:46) [F]. Thermocline at 10-25 m. Krill "forming a layer around 100 m depth at day".
  Blue whiting "forms a layer below 150 m at day, well separated from the krill", and was caught
  by day "only in the lowest layer". At night, 17-28 cm fish "dominated close to surface" and
  29-40 cm fish increased with depth, "dominating at a depth of 200 m". The large group scatters
  "around 100 m" with the large krill. "The feeding of Blue Whiting on krill takes place at
  night"; stomach content peaks "around or after midnight" and is largest "at intermediate
  depths (100 m) at night". The ascent starts ~1:40 h before sunset, and everything is diffuse
  2 h after it.
- In the box blue whiting is a Q3 visitor below 100 m (sibling).

## 3. Light at depth

### Literature
- **Skagerrak Secchi** (Aas et al. 2014, Ocean Sci. 10:177, Table 1) [F]: 57-59 N, all-year
  mean 8.3 +- 2.8 m. In the outer Oslofjord it is 4.3 m in summer and 11.1 m in winter.
  Zq(1 %) of PAR = 1.8 x Secchi (Fig. 8), so Kq = 4.61/(1.8 Zsd) = **0.31 m-1** (0.23-0.46 over
  +-1 SD) in the upper ~15 m [D].
- **Skagerrak, midsummer 1990** (Arst et al. 1993, Proc. Estonian Acad. Sci. Ecol. 3:38) [F]:
  mean z(1 %) of total solar irradiance 23 m, Jerlov coastal type 3. That gives mean K ~0.20 m-1
  [D].
- **Deeper water** (Aksnes et al. 2009, MEPS 387:39) [F]. Kd(500) = 1.059 am(500) + 0.013 for
  10-80 m, with am(500) = 0.64 - 0.016 S - 0.012 O2 + 0.15 Chl. Applying this to Skagerrak
  Atlantic water (S 35, O2 5-6 ml/l, Chl ~0.05) gives Kd(500) ~0.03-0.05 m-1. At 25-50 m
  (S 34-35, Chl 0.3-1) it gives ~0.08-0.15 m-1 [D, an extrapolation outside the fjords; J].
- **Surface illuminance** (AMS Glossary, civil and astronomical twilight) [F]: ~585-410 lx at
  sunset falling to ~3.5-2 lx at -6 deg, ~0.008 lx at -12 deg and ~6e-4 lx at -18 deg.
- **Herring biting thresholds**: 1e-3 lx (Batty et al. 1990, [A/S] via the sibling notes);
  "3.6 x 10-2 to 7 x 10-3 lux" (as cited by Onsrud et al. 2004) [F]. Herring filter-feed in the
  dark (Batty et al. 1986, [2nd] via the sibling).

### Calculation (`scripts/light_at_depth.py`, `tables/light_at_depth_stdout.txt`) [D]
K profile (0-20 / 20-50 / >50 m): low-K 0.20/0.08/0.04, central 0.30/0.12/0.06, high-K
0.45/0.15/0.08 m-1 (J, from the sources above). Daytime surface illuminance is
1.3e5 sin(h) lx, with an overcast factor of 0.2 (J).

| | central K | high-K | low-K |
|---|---|---|---|
| Noon depth of 1e-3 lx, Dec / Jun (overcast) | 143 / 171 m | 71 / 92 m | 269 / 312 m |
| Noon depth of 3.6e-2 lx, Dec / Jun (overcast) | 83 / 112 m | 37 / 49 m | 180 / 222 m |
| Noon depth of 1 lx, Dec / Jun (overcast) | 39 / 56 m | 18 / 26 m | 97 / 139 m |
| Depth of 1e-3 / 3.6e-2 lx at sun -6 deg | 37 / 15 m | 18 / 10 m | 90 / 25 m |
| Same at sun -8 deg (engine's light_threshold_deg) | 20 / 8 m | 13 / 5 m | 45 / 12 m |
| Same at sun -12 deg | 7 / 0 m | 5 / 0 m | 10 / 0 m |

Day length at 58.1 N, annual mean: sun > 0 deg 12.0 h, civil 1.9 h, nautical 2.2 h, below
-12 deg 7.9 h; sun > -8 deg 14.6 h (0.61 of the day). June-July has no night below -12 deg.
December: 6.2 h above 0 deg and 14.1 h below -12 deg.

### Inference
- **By day, light does not confine herring biting to 0-25 m.** At central K the bare threshold
  lies at 140-200 m at noon, and 1 lx at 40-80 m. Daytime herring at 25-100 m can bite, at a
  shorter reactive distance. A daytime light refuge on the bank exists only near the bottom of
  the deeper half of the box, or under high-K (turbid, low-salinity surface) conditions.
- **At night (sun < -12 deg) no depth reaches the biting threshold.** Only filtering remains.
- **The engine's -8 deg half-point fits.** At -8 deg the biting threshold falls inside the pool's
  layer (8-20 m at central K), so -8 deg marks the time when visual feeding in 0-25 m switches
  off.

## 4. Synthesis: overlap per member

### 4.1 Overlap for herring-dominated fish, July (`scripts/overlap_factors.py`, `tables/overlap_factors_stdout.txt`) [D]
O combines the HERAS fish shares (section 2.1) with the prey ratios of section 1. Spring and
autumn prey profiles bracket July. Fish in the unseen top 7-10 m would add ~+0.1-0.17 per
10 % of fish there.

| July | O, prey as spring | O, prey as autumn | H_eff = 25/O |
|---|---|---|---|
| Day, box | 0.40 (0.12-0.74) | 0.26 (0.17-0.40) | 63 / 96 m |
| Twilight, box | 0.49 (0.16-0.91) | 0.30 (0.21-0.43) | 51 / 84 m |
| Day, region | 0.44 | 0.30 | 56 / 84 m |
| Twilight, region | 0.61 | 0.42 | 41 / 59 m |

These are all-fish values. Foraging fish are probably the shallower part (J), which makes these
lower bounds for O while foraging.

### 4.2 Bottom-line table (J where marked; low / central / high)
p_layer = share of FORAGING fish inside 0-25 m. O as defined in section 0. H_eff = 25 m / O,
where 25 m is the pool layer by definition.

| Member, season | Period | p_layer | O (low / central / high) | H_eff (m) (high / central / low) | Basis |
|---|---|---|---|---|---|
| Herring, Apr-Sep | day | 0.42-0.6 | 0.25 / **0.45** / 1.0 | 100 / **55** / 25 | HERAS July [D]; foraging-fish and unseen-surface correction J |
| | twilight | 0.56-0.8 | 0.3 / **0.6** / 1.2 | 85 / **40** / 21 | HERAS twilight [D]; dusk feeding peak [A]; J |
| | night | 0.6-0.9 | 0.5 / **0.8** / 1.2 | 50 / **31** / 21 | "disperse into surface waters at night" [F]; no night data in the box; J |
| Herring, Oct-Mar | day | 0-0.15 | 0.05 / **0.15** / 0.4 | 500 / **170** / 63 | GOV day catches near bottom [D], Onsrud > 70 m [F]; winter prey r 0.3-0.6 J |
| | twilight | 0.3-0.6 | 0.3 / **0.5** / 0.8 | 85 / **50** / 31 | ascent at dusk; J |
| | night | 0.3-0.7 | 0.3 / **0.55** / 0.8 | 85 / **45** / 31 | Onsrud upper 50 m [F]; adults below the Baltic layer (Falk) [F]; J |
| Sprat | day | 0.3-0.6 | 0.25 / **0.45** / 1.0 | as herring summer | shares the HERAS echoes; Oslofjord winter bimodal [A]; J |
| | night | 0.5-0.9 | 0.5 / **0.8** / 1.2 | 50 / **31** / 21 | feeds at night [F via sibling]; J |
| Norway pout, all year | day | 0 | 0.01 / **0.03** / 0.07 | 2500 / **830** / 360 | 5-10 m above bottom at 100-300 m [A/F]; r(60-100 m) 0.07 [D] |
| | night | 0 | 0.05 / **0.1** / 0.2 | 500 / **250** / 125 | "no catches in the upper 25 m", midwater [F]; r(25-60 m) 0.1-0.2 [D] |
| Blue whiting 17-28 cm, Q3 | day | 0 | 0 / **0** / 0.01 | not feeding | below 150 m, no feeding by day [F] |
| | night | 0.5-0.9 | 0.4 / **0.7** / 1.2 | 63 / **36** / 21 | "dominated close to surface at night" [F]; J |
| Blue whiting 29-40 cm, Q3 | night | 0 | 0.03 / **0.07** / 0.15 | 830 / **360** / 170 | scatter "around 100 m" with large krill [F]; r(60-100 m) [D] |

Prey layer thickness and the share of the pool inside it:
- H = 25 m holds 100 % of the pool by definition (A17 0-25 m).
- Of the real resident herring-prey biomass of 0-100 m, 0-25 m holds 0.83 (0.55-0.97) in
  spring and 0.79 (0.60-0.91) in late summer [D]; in winter it is ~0.4-0.7 (J).
- The densest sublayer is 0-10 m (1.5-2.0x B/25 m). A fish foraging there sees an effective
  H of ~13-17 m, which is the lower bound on H.

## 5. Judgement and largest uncertainty
- **Largest uncertainty: where the FORAGING herring are relative to the prey.** The HERAS shares
  describe all fish in July and miss the top 7-10 m. There are no night acoustic logs and no
  winter acoustics in the box. The low-to-high O span (~4x for herring) comes almost entirely
  from this. A second, smaller factor is the prey ratio below 25 m (0.03-0.4).
- Winter prey profiles (no Skagerrak data) and the deep K (Aksnes regression extrapolated to the
  Skagerrak) are J.
- Structural flags for the parent:
  1. For pout and large blue whiting the encounter with the pool is near zero. Their
     zooplankton (krill, hyperbenthos, deep and diapausing Calanus) is not in the 0-25 m
     resident pool. Their share of the group `a` on this pool should be small. The real diet is
     consistent with that only if the pool is read as "everything they eat", which conflicts
     with its 0-25 m definition.
  2. The resident copepods do not migrate vertically in the Skagerrak (Maar 2006; Tonnesson
     2006). The daytime "refuge" from herring is the fish's own depth choice and the dilution
     below 25 m, not copepod hiding. That bears on the zooplankton visibility floor, which is
     not mine to set.
  3. Night overlap is high for herring (the dispersed layer rises), but biting is impossible
     below -12 deg (section 3). The night intake therefore rests on filtering, and its
     overlap O ~0.8 multiplies the filtering clearance, not the particulate one.

## 6. Failed fetches
- Cardinale et al. 2003 (Aquat. Living Resour. 16:283, Baltic herring/sprat diel distribution):
  alr-journal HTTP 403; no OA copy (Unpaywall); abstract elided in Semantic Scholar and Crossref.
- DTU Aqua report 11-96 "Acoustic monitoring of herring ..." (Sound): scanned JBIG2 PDF, no text
  layer; not OCRed (not Skagerrak).
- Bamstedt 2000 (Mar. Biol.): Springer login redirect, no OA, OpenAlex has no abstract;
  used as a search snippet [S] only.
- Bagoien et al. 2000 (Sarsia): no OA; abstract via OpenAlex [A].
- Solberg & Kaartvedt (NINA Brage page): DNS failure; abstracts via OpenAlex [A].
- Onsrud et al. 2004: characters 100,000-110,874 (the end of the article) not read.
- SHARK A17: no depth-stratified zooplankton (only 0-25 / 0-30 m).
- No source found for a measured Skagerrak Kd(PAR) profile below the pycnocline, or for
  Skagerrak herring depth at night.

## 7. References
- Aas E, Hokedal J, Sorensen K (2014) Secchi depth in the Oslofjord-Skagerrak area. Ocean Sci 10:177-199. doi:10.5194/os-10-177-2014 [F]
- Aksnes DL, Dupont N, Staby A, Fiksen O, Kaartvedt S, Aure J (2009) Coastal water darkening and implications for mesopelagic regime shifts in Norwegian fjords. MEPS 387:39-49. doi:10.3354/meps08120 [F]
- AMS Glossary of Meteorology: civil twilight; astronomical twilight. https://glossary.ametsoc.org/wiki/civil-twilight/ , https://glossary.ametsoc.org/wiki/astronomical-twilight/ [F, via search extract]
- Arst H, Kutser T, Ohvril H, Ojaveer O, Tamm J (1993) Experimental studies on the attenuation of underwater solar irradiance in some regions of the Baltic Sea and Skagerrak. Proc Estonian Acad Sci Ecol 3:38-48. doi:10.3176/ecol.1993.1.05 [F]
- Bagoien E, Kaartvedt S, Overas S (2000) Seasonal vertical migrations of Calanus spp. in Oslofjorden. Sarsia. doi:10.1080/00364827.2000.10414581 [A]
- Bamstedt U (2000) Life cycle, seasonal vertical distribution and feeding of Calanus finmarchicus in Skagerrak coastal water. Mar Biol. doi:10.1007/s002270000337 [S]
- Batty RS, Blaxter JHS, Richard JM (1990) Light intensity and the feeding behaviour of herring. Mar Biol 107:383 [A/S via sibling notes]
- Degnbol P, Munch-Petersen S (1985) On the relation between diurnal migration and feeding of blue whiting in the Skagerrak. ICES CM 1985/H:46 [F]
- Darbyson E, Swain DP, Chabot D, Castonguay M (2003) Diel variation in feeding rate and prey composition of herring and mackerel in the southern Gulf of St Lawrence. J Fish Biol. doi:10.1046/j.1095-8649.2003.00247.x [A]
- Falk U (1968) Hydrographic factors and the vertical migration of the herring in the Norwegian Channel. ICES CM 1968/H:17 [F, poor OCR]
- Falkenhaug T et al. (2022) Temporal variability of co-occurring Calanus finmarchicus and C. helgolandicus in Skagerrak. Front Mar Sci 9:779335 [F]
- Fernandes PG, Macdonald L, Aukland R, Reid D, Simmonds EJ, Shanks A (2001) Changes in the availability of herring to the North Sea acoustic survey; the impact of diurnal migration. ICES CM 2001/Q:08 [F]
- ICES Acoustic-trawl survey database, HERAS, RV Dana (DTU Aqua), 2010-2020 logs [F data, via the sibling's downloads]
- Jensen OP, Hansson S, Didrikas T, Stockwell JD, Hrabik TR (2011) Foraging, bioenergetic and predation constraints on diel vertical migration: reverse migration by YOY herring. J Fish Biol. doi:10.1111/j.1095-8649.2010.02855.x [A]
- Kaartvedt S (2000) Life history of Calanus finmarchicus in the Norwegian Sea in relation to planktivorous fish. ICES J Mar Sci 57:1819 [A here; F in sibling]
- Kaartvedt S, Rostad A, Titelman J (2021) Sleep walking copepods? Calanus diapausing in hypoxic waters adjust their vertical position during winter. J Plankton Res 43:199-208. doi:10.1093/plankt/fbab004 [F via PMC8009684]
- Maar M, Visser AW, Nielsen TG, Stips A, Saito H (2006) Turbulence and feeding behaviour affect the vertical distributions of Oithona similis and Microsetella norwegica. MEPS 313:157-172 [F]
- Onsrud MSR, Kaartvedt S, Rostad A, Klevjer TA (2004) Vertical distribution and feeding patterns in fish foraging on the krill Meganyctiphanes norvegica. ICES J Mar Sci 61:1278. doi:10.1016/j.icesjms.2004.09.005 [F, first 100k characters]
- SHARK (SMHI) zooplankton data, station A17 [F data]
- Solberg I, Kaartvedt S (2015) Ecology of overwintering sprat (Sprattus sprattus). Prog Oceanogr. doi:10.1016/j.pocean.2015.08.003 [A]; Solberg I (2017) PhD thesis, Univ. Oslo, hdl 10852/55412 [A]
- Tarling GA, Jarvis T, Emsley SM, Matthews JBL (2002) Midnight sinking behaviour in Calanus finmarchicus. MEPS 240:183-194 [F; Clyde Sea, context only]
- Tonnesson K, Nielsen TG, Tiselius P (2006) Feeding and production of the carnivorous copepod Pareuchaeta norvegica in the Skagerrak. MEPS 314:213-225 [F; Table 2 data]
- Sibling notes reused: `../Sillens vinterföda/zooplankton_refuge.md`, `../Sillens vinterföda/winter_food_and_feeding.md`, `../Sillens vinterbudget/winter_budget.md`, `../Fiskgrupper i rutan/planktivores/evidence_pout_bluewhiting.md`, `../Ekolodsdata planktivorer/acoustic_planktivores.md`.
