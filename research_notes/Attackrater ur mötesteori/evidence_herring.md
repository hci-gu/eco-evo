# Herring on zooplankton: encounter factors (R, v, capture success, filter feeding) and validation data

Evidence for `planktivores_encounter.md`: herring only (0.79 of the group). Written 2026-10-25. No code,
library or manifest was touched. Tags: [F] full text or data read here, [A] abstract only, [S] secondary
(number read in a document that cites the original, or a search snippet), [D] computed here
(`scripts/herring_encounter_checks.py`, stdout `tables/herring_encounter_checks_stdout.txt`), J = judgement.
Downloads: `data/herring/` (untrusted, only parsed with pdftotext / `python3 -I`).

Reused, not redone: `../Sillens vinterföda/herring_functional_response.md` (Arrhenius & Hansson 1999,
Bernreuther 2007 snatch rate, NEMURO K), `../Sillens vinterföda/zooplankton_refuge.md` (DVM, Bernreuther
2013 diel stomachs, light at depth), `../Sillens vinterföda/winter_food_and_feeding.md` (winter rations,
Onsrud 2004 day/night fullness, A17 copepod abundance), `../Nettoförflyttning per funktionsgrupp/herring.md`
(field school speeds), `../Fiskgrupper i rutan/planktivores/evidence_cmax_energy.md` (Cmax, field rations),
`../Vintermetabolism sill och tumlare/q10_metabolism.md` (Wisconsin swimming term, Bachiller 1 BL/s).

## 1. Bottom line: factor values for herring

| Factor | Low | Central | High | Basis (keys in section 6) |
|---|---|---|---|---|
| R, day, saturating light, on Calanus CIV-adult (2-2.7 mm) | 0.5 BL | 0.7-1.0 BL | 1.3 BL | No measured R for juvenile/adult herring exists in what was found. Modelling convention of the herring visual model: "E and K are scaled such that R is ca. 1 body length when light is not limiting" (VF10 as reported by vD15 [F] and LV17 [F]); the two reports disagree on the reference prey (1.5 mm copepod vs 4 mm, A = 3e-6 m2), which moves R on Calanus CV between 0.71 and 1.33 BL [D]. Larval herring measured 0.6-1.0 BL (MK85 [F]) |
| R, day, on small copepods (~1 mm) | 0.3 BL | 0.45 BL | 0.6 BL | R ~ sqrt(prey image area) in the Aksnes-Utne small-range limit [D, AU97 form]. Herring are size-selective, larger prey first (Ba90 [S], Da00 [F], Vi01 [A]) |
| R vs light | - | R^2 ~ I/(Ke + I) | - | Form: AU97 [A], LV17/vD15 [F]. **No Ke for herring found** (VF10 Ke not retrieved). Gobiusculus saturates at ~10 umol m-2 s-1 (Ut97 [S]). Lake herring (Coregonus artedi, not a clupeid) R significantly smaller at 2-10 lx than at 40-1500 lx (LE96 [A]) |
| Biting light threshold | - | 0.001 lx (natural zooplankton) | 0.01 lx (Artemia nauplii) | Ba90 [S] (search-snippet abstract, two independent snippets agree). = ~2e-5 umol m-2 s-1 [D, conversion J]. Below it: no biting, filtering only (Ba86 [S]) |
| v, searching (particulate) | 0.6 BL/s | 1.0 BL/s | 1.4 BL/s | Aquarium feeding herring 0.34 m/s = 1.36 SL/s (SL 24.8 cm; Ha23 [F]); juveniles in light 0.34 m/s, biting speed rises with light (Ba86/Ba90 [S]); bioenergetics convention 1 BL/s (Bachiller 2018 [F] via q10 note); field deep feeding schools 1.1 BL/s (Nøttestad 2002 [F] via herring.md); migrating schools 2.5-3 BL/s (not a search speed); Wisconsin herring activity term gives ~0.9-1.3 BL/s at 60 g (Rose 2008 [F] via q10 note) [D] |
| v, filtering in light / in dark | - | 0.34 / 0.11 m/s | - | Ba86 [S]: in light no difference filtering vs not (0.34 m/s); in dark filtering 0.11, non-filtering 0.07 m/s; dark schools disperse, fish swim in tight circles |
| Search geometry factor | 0.25 | 0.5 | 1 | VF10 clearance = 1/2 pi R^2 v (vD15 [F]); larval herring 0.5 x 0.5 (FF99 [F]: half the visual field, searching half the time - the time half is outside `a` here) |
| Capture success on copepods | 0.3 | 0.6 | 0.9 | **No herring measurement found.** Proxies: lake herring 0.30 (Limnocalanus, large copepod) to 0.65, copepod mean 0.49 (Li96 [A]); stickleback juveniles 0.92 on Temora, 0.53 on Eurytemora (Vi98 [A]); juvenile herring on frozen (non-escaping) copepods ~1 (snatch efficiency 1.11 +- 0.14, Be07 [F]). Calanus has strong escape jumps (J: lower half) |
| Gape area | - | 0.0047 SL^2 | - | 292 mm2 at SL 248 mm, 194 g (Ha23 [F], n 7, SD 155 mm2); isometric scaling [D]: 80 / 144 / 224 mm2 at TL 15 / 20 / 25 cm |
| Gill-raker gap / retention | - | GR gap 0.30-0.38 mm (adult) | - | 0.302 mm (GA1, SL 248 mm; Ha23 [F]); 0.381 mm (TL 290 mm; Co17 [F]); denticles reduce the mesh further. Observed retention lower than expected from raker spacing (Gi88 [A]). For copepods >= 1 mm retention ~0.9-1 (J); a 0.5 mm trawl mesh still keeps 50 % of 1.6 mm Calanus (Ko25 [F]) |
| Filtering clearance | 29 L/h (15 cm, dark) | 51-158 L/h (20 cm, dark-light) | 247 L/h (25 cm, light) | gape x speed x 0.9 [D]; Ha23 [F] flow through open mouth 4.91 L/min = 295 L/h (adult, 0.34 m/s) |
| Particulate clearance per t (day, saturating) | 0.8e5 m3/t/6h | 1.5e6 m3/t/6h | 1.1e7 m3/t/6h | low/central/high R, v, capture above, half-field, TL 15-25 cm (nearly size-independent per tonne) [D] |
| Night/day clearance ratio, filtering only at night | 0.003 | 0.004 | 0.09 | dark filtering vs central (high) or low day particulate [D]. Moonlit biting not included (see 2.6) |

**Which mode dominates.** Particulate (biting) clearance at saturating light is ~20-500x the filtering
clearance per fish (sections 1, 3 of the stdout) [D]. Filtering only pays where biting is limited by the
maximum bite rate (0.85-1 bites/s; Be07 [F]), i.e. above N* = bite rate / filter clearance: ~20 prey/L for a
20 cm herring in light and ~60 prey/L for an 11.5 cm fish [D]. This matches the measured switch densities:
50 % of herring filter at 18 Calanus finmarchicus/L and at 114 medium Artemia/L (Gibson & Ezzi 1990 [S] via
Be07), filtering on Artemia nauplii only above ~50/L (GE85 [S]). Field: the A17 0-25 m copepod abundance is
2.5-7 /L in winter and 25-31 /L in May-July (winter_food note [F]), mostly small copepods that are below
the size where biting a single prey is worth it; so filtering is possible in summer patches but the field
diets show size-selective particulate feeding (Da00 [F]: no small copepodids in stomachs even where
abundant; Be07 ms 1 [F]: Bornholm Basin plankton too dilute for filtering). J: particulate feeding by day
dominates the annual intake; filtering is the night mode (Ba86) and a summer high-density supplement.

**Bite-rate ceiling.** At 0.85 bites/s over 12 h a herring takes ~2.2 g/d of 0.06 mg small copepods
(= 4-11 %BW/d at 20-15 cm) but ~30 g/d of Calanus CV [D]. On small copepods the bite rate is of the same
order as the Cmax ceiling (10 %BW/d at h 40); on Calanus digestion (Cmax) binds, not handling [D].

## 2. Details by factor

### 2.1 Reactive distance R
- Larval herring [F] (MK85, Table 4): reaction distance 7.7-8.7 mm from attacks, 9.5-14.7 mm from aiming
  postures, larvae ~10-15 mm, swimming velocity set to 1 cm/s, "half-circular field of vision above the
  head"; R and search volume rose as nauplii fell from 30 to 5-7.5 /L. Search volume 36-54 cm3/min. Larvae
  only; shows R ~ 1 BL in early stages and a behavioural rise of search effort at low prey density.
- Fiksen & Folkvord 1999 [F] (larval herring IBM): visual range from AU97 "calibrated with observed
  estimates in larval herring"; prey contrast 0.4 (Palm pers. comm.); fraction of visual area perceived x
  fraction of time searching = 0.25 ("0.5 x 0.5 may be a reasonable estimate"); Calanus nauplius-CIV image
  areas 7.8e-3 to 1.21 mm2 (Table 3).
- Juvenile/adult herring: **no measured reactive distance was found**. The models that represent adult
  herring visually (VF10; LV17 "generic herring-like planktivorous fish", 20 cm; vD15 for sandeel with the
  VF10 parameters) all set R = ~1 BL at non-limiting light, i.e. R is a scaled assumption, not a datum:
  - LV17 [F] Table 1: c = 0.3 m-1 (beam attenuation), Cp = 0.3, "E' and K are scaled such that R = 1 BL when
    light is not limiting and prey image area (Prey length x prey width x 0.75) = 3 x 10-6 m2"; prey 4 mm;
    fish at 30 m; daily-mean R at high latitude 6 cm in midsummer, < 1 cm in polar night (20 cm fish).
  - vD15 [F] Table 1: "C is the contrast = 0.3 (Varpe & Fiksen 2010)"; "E and k are scaled such that R is
    ca. 1 body length when light is not limiting and prey is 1.5 mm copepods (Varpe & Fiksen 2010)";
    search rate beta = 1/2 pi R^2 v.
  - The two disagree on the reference prey. With R ~ sqrt(A): R on a Calanus CV/adult (2.5 x 0.8 mm) is
    0.71 BL (LV17 reference) or 1.33 BL (vD15 reference); on a 1 mm small copepod 0.32 or 0.60 BL [D].
- Aksnes-Utne model [A]/[F]: r^2 exp(c r) = C A E' I/(Ke + I); two eye parameters (E', Ke); saturation at high
  light. Ke for herring: not retrieved (VF10 not obtained). LV17 paper II uses Ke 4e-4 mW m-2 nm-1 for
  piscivores and 1e-8 for Benthosema [F] - not herring values.
- Light at which R collapses: biting threshold 0.001 lx on natural zooplankton, 0.01 lx on Artemia nauplii;
  "size-selective, taking the larger organisms first"; light range tested 0.0001-270 lx (Ba90 [S]).
  Larval herring feeding thresholds 0.3 lx (Artemia nauplii) and 13 lx (Balanus nauplii) (Bl68 CalCOFI
  [F]); adult retinomotor movement mainly at 10-0.1 lx (same, [F] text "a t intensities from 10-1 photopic
  lux", OCR). J: the 0.001 lx threshold is where biting stops, not the half-saturation; with no Ke the
  half-saturation is unknown and lies somewhere between ~0.01 lx and ~10 lx (Gobiusculus saturates near
  10 umol m-2 s-1 ~ 500 lx, Ut97 [S]; lake herring loses R between 40 and 10 lx, LE96 [A]).
- Turbidity: Kd/beam attenuation enter through exp(c r); at R ~ 0.2 m and c 0.3 m-1 the factor is 1.06,
  negligible at saturating light; turbidity acts mainly through light at depth [D].

### 2.2 Swimming speed while searching
- Ha23 [F] Table 1: Clupea harengus (aquarium Stralsund, n 9 filmed; dissected SL 248 mm, 194 g) swimming
  speed while feeding 1.36 SL/s = 0.34 m/s; "the swimming velocity is higher during feeding than during
  non-feeding" (difference significant only for sardine); mouth opened "frequently ... average opening
  time of 0.17 s" (particulate-like). Aquarium food was crushed regular food, not live plankton.
- Ba86 [S] (abstract via search snippets): juveniles on Artemia nauplii; in light 0.34 m/s for filtering
  and non-filtering fish alike; in darkness only filtering, school dispersed, "slower in tight circular
  paths", 0.11 m/s filtering vs 0.07 m/s non-filtering. Fish size not obtained.
- Ba90 [S]: "Swimming speed increased with increasing light intensity when the fish were actively feeding
  by biting"; filter-feeding on dense Artemia: speed not related to light.
- Field (reused from herring.md [F]): NSSH feeding/migrating schools 0.73-0.94 m/s gross (2.5-3 BL/s);
  deep risk-averse schools 1.1 BL/s; overwintering 0.25-0.42 m/s (Huse & Ona 1996 [A]); night speed above
  day speed in spring migration (1.00 vs 0.78 m/s). These include travel, not only search.
- Conventions (reused): Bachiller 2018 fixes SW = 1 BL/s [F]; vD15 sandeel 1.5 BL/s [F]; Wisconsin herring
  activity term U = aA W^0.13 exp(cA T) (Rose 2008 [F]) gives ~19 cm/s at 60 g and 7 C and ~26 cm/s at
  >= 9 C, i.e. 0.9-1.3 BL/s [D]. The Wisconsin term implies a temperature swing in v (roughly halving from
  9 C to 4 C, q10 note) - the only sourced seasonal swing in a herring encounter factor.
- Search mode: herring are treated as cruise searchers in every model found (FF99, VF10, vD15, LV17). No
  pause-travel (saltatory) description for juvenile or adult herring was found; larvae switch between
  "abrupt" and "meandering" swimming with prey density (MK85 [F]).

### 2.3 Capture success
- No herring capture-success measurement on live copepods was found (searched; record in section 5).
- Bernreuther 2007 ms 4 [F]: juvenile herring 9.9-13 cm, frozen copepods (non-escaping): snatching
  efficiency (observed/estimated ingestion per snatch) 1.11 +- 0.14 (max 1.31) - capture ~1 on dead prey.
- Viitasalo et al. 1998 [A]: stickleback juveniles captured 92 % of Temora longicornis and 53 % of
  Eurytemora affinis; capture only if they approached slowly within < 0.1 cm.
- Viitasalo et al. 2001 [A]/[F]: Baltic herring selectivity rank Temora > Eurytemora > Pseudocalanus >
  Acartia > Bosmina; deviations from size explained by escape ability (siphon capture success 9-100 % by
  taxon, an artificial-flow index, not herring capture success).
- Link 1996 [A] (lake herring Coregonus artedi, obligate particulate planktivore, video): capture
  probability copepods mean 48.9 % (Limnocalanus 30 %, Diaptomus 42.5 %, Cyclops 57.5 %, diaptomid
  copepodids 65 %), cladocerans 86.7 %.
- FF99 [F]: larval herring P(capture | attack) from prey/predator length (Heath 1993 form); not adult.
- J: central 0.6 on the copepod pool (escape-capable Calanus and Eurytemora-like taxa pull it down,
  weak escapers such as Temora and cladocerans push it up), range 0.3-0.9.

### 2.4 Filter feeding
- Mechanism: "filtering the herring swims at an increased speed with an open mouth" (Be07 [F] p. 7);
  "swimming with a wide open mouth and a flared operculum at a markedly increased swimming speed" (Be07 ms 4
  [F], citing Gibson & Ezzi 1985, 1990).
- Capture rate of filtering "directly proportional to prey concentration and above a critical prey
  concentration its capture rate exceeds that of biting"; "less than 50% of fish are filtering when capture
  rates by the two methods are equal"; filtering costs 1.4-4.6x biting (GE92 [A]).
- Thresholds: ~50 Artemia nauplii/L (GE85 [S]); 50 % filtering at 18 C. finmarchicus/L and 114 medium
  Artemia/L ("the larger the prey item the lower the concentration", Gibson & Ezzi 1990 [S] via Be07 ms 1
  [F]); juveniles 9.9-13 cm on small frozen copepods filtered from >= 900 /L (Be07 ms 4 [F]); filtering in
  the dark on Artemia nauplii, removal "much lower than in light, except in the highest prey
  concentrations" (Ba86 [S]); night filtering "would matter mainly when exploiting dense patches" (same).
- Morphology [F]: gape (mouth opening area) 292 +- 155 mm2 at SL 248 mm; GR on arch 1: 67 rakers, 11 mm,
  gap 0.302 mm; mesh 0.015-0.048 mm2; volume flow through the open mouth 4.91 L/min (Ha23). Co17: 29 cm
  herring GR gap 381 +- 37 um, GR thickness 49 um, denticles reduce the mesh. Gi88 [A]: rakers appear at
  16 mm, first arch ~60 % of filtering area, raker spacing grows through life, observed retention below
  the spacing prediction.
- [D]: filtering clearance 29-250 L/h per fish (TL 15-25 cm, dark 0.11 m/s to light 0.34 m/s,
  retention 0.9); crossover with biting at ~12-23 /L (20-25 cm, light), 38-70 /L (20-25 cm, dark-speed
  filtering), 58-69 /L (11.5 cm, light) - consistent in order with the measured switch densities.

### 2.5 Temperature
No temperature dependence of R or capture success was found for herring. Swimming speed carries the only
sourced temperature effect (Wisconsin activity term; Misund 1997 speed vs temperature r = 0.57 in one of
two years, herring.md [F]). J: no seasonal swing in `a` from temperature beyond the speed term.

### 2.6 Night
- In darkness herring do not bite; they filter (Ba86 [S]; Be07 [F] p. 7: "Herring are not able to feed by
  biting in the dark, but they are able to filter-feed").
- Encounter theory for a dark night: filtering clearance / day particulate clearance = 0.003-0.005
  (central day) up to 0.06-0.09 (low day) [D]. This is far below the engine's dark_ratio 0.25 (and below
  0.1). The field night intake is higher than this (section 3, rows V8-V10), which says night feeding in
  the field happens under moon/starlight above the 0.001 lx threshold, in twilight, or near the surface.
- Moon/starlight: the threshold 0.001 lx is of the order of clear-sky starlight at the surface and below
  full-moon light (~0.1-0.3 lx, J, textbook values not sourced here); so biting at reduced R is possible
  near the surface on moonlit nights, not at depth.

## 3. Validation data (not used to set factors)

| # | Source | Measurement | Converted [D] | What it tests |
|---|---|---|---|---|
| V1 | Arrhenius & Hansson 1999 [F] (via herring_functional_response.md), YOY Baltic herring 25-94 mm, Jul-Nov, field | WSC = 0.21 Z/(1 + 26 Z), g DW/g DW/daylight h, Z g DW m-3 (0-30 m) | initial slope 46 L per g WW fish per h = 2.8e5 m3/t/6h (daylight); measured/theory 0.12-0.16 at R 0.5 BL, 0.03-0.04 at R 1 BL (v 1 BL/s, capture 1); half-saturation 0.30 g WW m-3. As area: 0.011 km2/t/6h at 25 m overlap, 0.0055 at 50 m | The absolute daylight clearance (encounter x capture x overlap) per tonne; the only field-measured herring attack rate. Says theory with R 1 BL and capture 1 overpredicts field clearance 6-30x |
| V2 | Bernreuther 2007 ms 4 [F], lab, 20 juveniles 9.9-13 cm, 13 C, frozen copepods 0.011 mg DW, 10-1600 /L | SR = 0.849 N/(77.6 + N) snatches/s | initial slope 0.011 L/s = 39 L/h per fish = 4.7 L/g/h; measured/theory 0.01-0.05 | Lab tank clearance; a lower bound (arena, frozen prey); half-saturation in the tank 6.6 g WW m-3 |
| V3 | Gibson & Ezzi 1985, 1990, 1992 [S]/[A] | switch densities 18 Calanus/L, ~50-114 Artemia/L; capture rates of biting vs filtering from video | crossover N* from gape x speed / bite rate reproduces 12-70 /L [D] | The filtering model (gape x speed x retention) and the maximum bite rate |
| V4 | Ha23 [F] | flow through open mouth 4.91 L/min, adult | = 0.08 L/s | Filtering clearance per fish |
| V5 | Bachiller 2018 [F] (via evidence_cmax_energy.md), adult NSS herring, Norwegian Sea, Apr-Sep | realised ration 0.05 g/g/d (~6 %/d) | ~0.6-1 of Wisconsin Cmax at 2-6 C | Feeding level in the wild (P = C/Cmax) in the feeding season: says the ceiling, not encounter, sets summer intake of adults on Calanus |
| V6 | Arrhenius & Hansson 1994 [F] (via evidence_cmax_energy.md) YOY Baltic | rations 2.1-17 %BW/d; P = 0.11-0.18 of the Rudstam Cmax | - | Feeding level well below Cmax for YOY: encounter-limited regime |
| V7 | Heath 2005 / Greenstreet 1997 [F] (via evidence_cmax_energy.md); Ecopath Q/B 4.34 adult, 5.63 juvenile | North Sea annual mean ration 0.012 g/g/d | P ~0.1-0.2 annual | Annual-mean feeding level; with the A17 prey level gives an implied annual-mean `a` once overlap is fixed |
| V8 | Bernreuther 2013 [F] (via zooplankton_refuge.md), Bornholm Basin, June | stomach content night ~48-54 mg vs peak 196 / 313 mg (small / large herring); gastric evacuation says "not feeding or feeding only to a minor extent" at night | night/peak standing stock 0.17-0.24 | Upper bound of the night/day intake ratio (standing stock overstates night intake) -> tests dark_ratio |
| V9 | Onsrud et al. 2004 [F] (via winter_food_and_feeding.md), Oslofjord, Nov and Mar | fullness night/day 2.29/2.91 and 2.70/3.43 (1-5 scale), krill eaten day and night | above-empty ratio 0.68-0.70 | Night feeding on krill in a deep fjord in winter; argues dark_ratio > encounter-theory value; ordinal scale, integrates hours |
| V10 | Be07 ms 1 [F]: Bornholm Basin; "fresh prey at every station and every time of day and night ... both species feed at night, but to a lower extent than during the day"; lowest content ~02:00 | qualitative | - | Same as V8 |
| V11 | Dalpadado et al. 2000 [F], NSS herring, Norwegian Sea 1994-96 | size-selective on Calanus CIV+; "the herring perform particulate-feeding, since the smallest copepodids were not observed in the stomachs, even in areas where they were available in relatively high numbers"; stomach weight higher where zooplankton biomass higher | qualitative | Particulate mode dominates in the field; stomach fullness rises with prey density (a positive, unsaturated response across regions) |
| V12 | Lohmeyer & Hempel 1977 [F] (via winter note) | Jan-Feb ration ~0.4-1.0 %BW/d wet, feeding peaks at noon and sunset | - | Winter feeding level against the light-limited encounter rate |
| V13 | MK85 [F], larvae | search volume 36-54 cm3/min, rises at low prey density | - | Behavioural (not fixed) search effort; larvae only |
| V14 | Kiørboe & Munk 1986 [S], larvae: saturation above ~30 nauplii/L | - | - | Not usable (larvae, snippet) |

Not obtained: Utne et al. 2012 (Norwegian Sea herring IBM consumption, Zenodo record removed); Varpe & Fiksen
2010 parameter table; Segers et al. 2007 has no diel analysis (abstract).

## 4. Judgement and largest uncertainty
- J: R central 0.7-1.0 BL on Calanus and ~0.45 BL on small copepods; capture 0.6; v 1 BL/s; half-field 0.5.
- **Largest uncertainty: R (and thus R^2) for juvenile/adult herring is not measured**; the 1 BL value is
  a model scaling, and its reference prey is reported inconsistently (1.5 vs 4 mm), a factor 3.5 in R^2.
  Second: capture success on live, escaping copepods (no herring data; proxies 0.3-0.92). Together the
  low/high per-tonne clearance spans ~2 orders of magnitude (0.8e5-1.1e7 m3/t/6h).
- The only field clearance (V1, YOY) is 6-30x below the R = 1 BL, capture 1 theory and ~5x below the
  central case; this is the strongest single constraint and argues for the lower half of the theory range
  (or for an unmodelled search-efficiency factor such as schooling, where fish behind the front search
  already-depleted water - J, no herring data).
- Night: encounter theory gives dark/day ~0.004 (filter only); field stomach data imply ~0.2-0.7 standing
  stock ratios. These cannot both be the night intake ratio; standing stocks integrate the evening meal.

## 5. Failed fetches (not used for numbers)
- Varpe & Fiksen 2010 (Ecology 91:311): Wiley pdfdirect and esajournals returned HTML/403 (curl, WebFetch);
  Europe PMC has no full text; Varpe's 2007 thesis on NVA (190 kB) is the summary without the papers.
- Fiksen et al. 1998 (Fish. Oceanogr. 7): Wiley pdfdirect returned HTML.
- Batty et al. 1986, 1990; Gibson & Ezzi 1985, 1990 (Springer): idp.springer.com redirect / bot page;
  OpenAlex has no abstract; Semantic Scholar HTTP 429. Numbers are search-snippet [S] or [2nd] via Be07.
- Gibson & Ezzi 1992 and Gibson 1988 (J. Fish Biol.): abstract only (OpenAlex).
- Aksnes & Utne 1997 (Sarsia): abstract only. Utne-Palm 2002 review: not OA. Blaxter 1966 (BES symposium):
  no online copy. Blaxter 1968 (JEB): WebFetch landed on a wrong article. Huse & Fiksen 2010: not OA.
- Utne et al. 2012 (Mar. Biol. Res. 8:527): no copy found (Zenodo record removed, sibling note).
- IMR Brage host imr.brage.unit.no: DNS failure.

## 6. References (keys)
- AU97 Aksnes DL, Utne ACW 1997. A revised model of visual range in fish. Sarsia 82:137-147.
  https://doi.org/10.1080/00364827.1997.10413647 [A]
- Ba86 Batty RS, Blaxter JHS, Libby DA 1986. Herring (Clupea harengus) filter-feeding in the dark. Mar Biol
  91:371-375. https://doi.org/10.1007/BF00428631 [S]
- Ba90 Batty RS, Blaxter JHS, Richard JM 1990. Light intensity and the feeding behaviour of herring, Clupea
  harengus. Mar Biol 107:383-388. https://doi.org/10.1007/BF01313419 [S]
- Bachiller E et al. 2018. Bioenergetics modeling of the annual consumption of zooplankton by pelagic fish
  feeding in the Northeast Atlantic. PLoS ONE 13:e0190345. https://doi.org/10.1371/journal.pone.0190345 [F, via sibling notes]
- Be07 Bernreuther M 2007. PhD thesis, Univ. Hamburg (manuscripts 1, 4).
  https://ediss.sub.uni-hamburg.de/bitstream/ediss/1765/1/DissertationMatthiasBernreuther.pdf [F]
  (copy: `../Fiskgrupper i rutan/planktivores/data/ediss1765/`)
- Bl68 Blaxter JHS 1966/1968 (CalCOFI Rep. 10). The feeding of herring larvae and their ecology in
  relation to feeding. https://calcofi.org/downloads/publications/calcofireports/v10/Vol_10_Blaxter.pdf [F]
- Co17 Collard F et al. 2017. Morphology of the filtration apparatus of three planktivorous fishes and
  relation with ingested anthropogenic particles. Mar Pollut Bull. https://doi.org/10.1016/j.marpolbul.2016.12.067
  (https://www.vliz.be/imisdocs/publications/ocrd/305655.pdf) [F]
- Da00 Dalpadado P, Ellertsen B, Melle W, Dommasnes A 2000. Food and feeding conditions of Norwegian
  spring-spawning herring through its feeding migrations. ICES J Mar Sci 57:843-857.
  https://doi.org/10.1006/jmsc.2000.0573 (https://www.vliz.be/imisdocs/publications/ocrd/384908.pdf) [F]
- FF99 Fiksen Ø, Folkvord A 1999. Modelling growth and ingestion processes in herring Clupea harengus
  larvae. MEPS 184:273-289. https://www.int-res.com/articles/meps/184/m184p273.pdf [F]
- Fiksen Ø, Utne ACW, Aksnes DL, Eiane K, Helvik JV, Sundby S 1998. Fish Oceanogr 7:355-363.
  https://doi.org/10.1046/j.1365-2419.1998.00068.x [A]
- GE85 Gibson RN, Ezzi IA 1985. Effect of particle concentration on filter- and particulate-feeding in the
  herring Clupea harengus. Mar Biol 88:109-116. https://doi.org/10.1007/BF00397157 [S]
- Gibson RN, Ezzi IA 1990. Relative importance of prey size and concentration in determining the feeding
  behaviour of the herring Clupea harengus. Mar Biol 107:357-362. [S, via Be07]
- GE92 Gibson RN, Ezzi IA 1992. The relative profitability of particulate- and filter-feeding in the
  herring. J Fish Biol 40:577-590. https://doi.org/10.1111/j.1095-8649.1992.tb02607.x [A]
- Gi88 Gibson RN 1988. Development, morphometry and particle retention capability of the gill rakers in the
  herring. J Fish Biol 32:949-962. https://doi.org/10.1111/j.1095-8649.1988.tb05438.x [A]
- Ha23 Hamann L et al. 2023. Diversity of filter feeding and variations in cross-flow filtration of five
  ram-feeding fish species. Front Mar Sci 10:1253083. https://doi.org/10.3389/fmars.2023.1253083 [F]
- Ko25 Kostak EN et al. 2025. Estimating size selectivity of Calanus spp. in plankton trawls. Rev Fish
  Biol Fish 35:963-977. https://doi.org/10.1007/s11160-025-09943-w [F]
- LE96 Link J, Edsall TA 1996. The effect of light on lake herring (Coregonus artedi) reactive volume.
  Hydrobiologia 332:131-140. https://doi.org/10.1007/BF00016692 [A]
- Li96 Link J 1996. Capture probabilities of Lake Superior zooplankton by an obligate planktivorous fish -
  the lake herring. Trans Am Fish Soc 125:139-142. [A]
- LV17 Langbehn TJ, Varpe Ø 2017. Sea-ice loss boosts visual search: fish foraging and changing pelagic
  interactions in polar oceans. Glob Change Biol 23:5318-5330; read in Langbehn 2019 PhD thesis, Univ.
  Bergen (NVA). [F]
- MK85 Munk P, Kiørboe T 1985. Feeding behaviour and swimming activity of larval herring in relation to
  density of copepod nauplii. MEPS 24:15-21. https://www.int-res.com/articles/meps/24/m024p015.pdf [F]
- Ut97 Utne ACW 1997. The effect of turbidity and illumination on the reaction distance and search time of
  the marine planktivore Gobiusculus flavescens. J Fish Biol 50:926-938. [S]
- VF10 Varpe Ø, Fiksen Ø 2010. Seasonal plankton-fish interactions: light regime, prey phenology, and
  herring foraging. Ecology 91:311-318. https://doi.org/10.1890/08-1817.1 [A]; parameters [S] via vD15, LV17
- vD15 van Deurs M, Jørgensen C, Fiksen Ø 2015. Effects of copepod size on fish growth: a model based on
  data for North Sea sandeel. MEPS 520:235-243. https://www.int-res.com/articles/meps_oa/m520p235.pdf [F]
- Vi98 Viitasalo M, Kiørboe T, Flinkman J, Pedersen LW, Visser AW 1998. Predation vulnerability of
  planktonic copepods. MEPS 175:129-142. https://doi.org/10.3354/meps175129 [A]
- Vi01 Viitasalo M, Flinkman J, Viherluoto M 2001. Zooplanktivory in the Baltic Sea: prey selectivity by
  Clupea harengus and Mysis mixta. MEPS 216:191-200. https://www.int-res.com/articles/meps/216/m216p191.pdf [F]
- Sibling-note sources (Arrhenius & Hansson 1994/1999, Bernreuther 2013, Onsrud 2004, Lohmeyer & Hempel 1977,
  Misund 1997, Kvamme 2000, Nøttestad 2002, Huse & Ona 1996, Rose 2008, Heath 2005): full citations in the
  notes named at the top.
