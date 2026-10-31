# Evidence: benthivorous_fish -> planktivorous_fish (cod and haddock on herring, pout, sprat, blue whiting)

Research only (2026-10-31); nothing in the library, manifest, code or tests was changed. Tags: [F] stated
in a source or measured in a downloaded data set, [D] derived here, [J] judgement, [S] secondary. Scripts in
`scripts/`, their stdout in `tables/`, downloads in `data/` (README there, with the command lines).

## 1. How the live pair values were built (audit trail)

| Value | Built from | Where |
|---|---|---|
| a 0.0062 | 0.019 per CAPABLE tonne (section 220, encounter pi R^2 v_rel p in a 10 m bottom band x the prey's share F inside it, Kd 0.25) x phi 0.32 (223.26 whole-group convention) | `fg_library.yaml` pair comment; resume 220.4, 223.26 |
| 0.019 (0.016 at Kd 0.30) | same-layer a 0.12 x pool overlap ~0.13 (annual); stage weights = biomass x phi stage score (cod > 50 cm 0.53, cod 30-50 0.23, haddock 0.17) | `../Attackrater ur mötesteori/benthivores_encounter.md` 1a, 1c, 2.3 |
| overlap F (day / night) | herring 0.15 / 0.02, blue whiting 0.3 / 0.15, Norway pout 0.7 / 0.4, sprat 0.1 / 0.01 [J]; herring from HERAS logs: "share of herring NASC in the bottom 10 m was 0.34 (box) / 0.26 (region) with the sun above 6 deg", reduced to 0.15 because "Those logs average about 90-120 m bottom depth. The grid's median is 142 m and herring stay mostly above 100 m" | same, 2.3 |
| phi 0.32 (0.311) | stage scores "access x capability": cod 0.3 / 0.7 / 0.9 by stage, haddock 0.1 / 0.25 ... set on GAPE: "gape k x L against the near-bottom prey lengths ... cod 40 cm swallows 96 % of the Norway pout biomass and 45 % of herring, cod 50 cm 100 / 89 % ... Scores (access x capability) J: cod 0.3 / 0.7 / 0.9" | `../Fiskgrupper i rutan/params_benthivorous_fish.md` (phi paragraph); `benthivores/scripts/group_params.py` SC |
| stage_access [0.49, 0.14] | cod p95 prey/predator length 0.30 (SMS, 45-65 cm) over box cod >= 20 cm, applied to the prey stage length tables with POOL biomass weights; blue whiting has no rows in the table (drops out) | `../Årsungarnas andel/parameters_B.md` section 4, `scripts/stage_access.py` |
| dark_ratio 0.06 at 0 deg | static band model (vision night/day 0.25 x overlap 0.30), Kd 0.25 re-run. The same note: "Defensible range: 0.1-0.35, about 0.2 J. The static 0.08 is the corner case" | `benthivores_encounter.md` 2.3; resume 220.3-220.4 |
| M2 0.145 | SMS 2023 key-run partial M2 2013-22 scaled to the box (cod 0.132, haddock 0.012, gurnards 0.001) | pair comment; resume 223.37 |

Finding 1 [D]: **the gape enters twice.** phi's planktivore scores are the prey-biomass share inside the
gape (the phi paragraph says so), and the 223.32 stage access is a second prey-size window on the same
axis. 223.26 kept phi inside a ("a x 0.32") and 223.32 added the window on top. This is the analogue of the
whiting fault 2 (window and a not covering the same predators), here in the form of a double count.
Size: replace phi by the membership share without gape - cod >= 20 cm 0.314 x 0.982 = 0.308 at 1, haddock
>= 20 cm 0.386 x 0.796 with its own window relative to cod's (f_eff 0.077 / 0.272) x a non-gape
capability c_h 0.3-1, rays + gurnard 0.024 x 0.5 - giving 0.35-0.41 (central 0.377) against 0.32:
**x1.18 (1.08-1.27)** (`tables/clearance_test_benthivores_stdout.txt`, factor 1). c_h: at equal length
SMS haddock eat about half cod's planktivore share (30-39 cm 0.071 vs 0.149; 40-49 cm 0.118 vs 0.198;
`tables/sms_diet_pred18.txt`, `sms_diet_pred16.txt`) [F/D], though 2025 Q1 haddock in the NNS >= 100 m ate
none (455 fed; `tables/benth_diet_ices_stdout.txt`) [F].

Finding 2 [D]: **the window is weighted by pool biomass, the a by encounter.** The a weights each species
by s_j F_j (pout's band share is 5x herring's), the window by s_j. With encounter weights the stage access
is [0.548, 0.180] (current F) / [0.519, 0.160] (revised F, section 3) instead of [0.494, 0.136]:
x1.09-1.18 (`tables/window_by_species_stdout.txt`). Blue whiting (11 % of the pool, 16 % of encounter) is
outside the cod window (GOV p10 17.5 cm) but absent from the window table, which over-credits f by ~x1/0.86.

Finding 3 [D]: **the window ratio itself is right.** A length-dependent p95 (SMS all planktivore prey:
cod < 30 cm 0.40, 30-45 0.34, 45-65 0.30, >= 65 0.27) gives the same f_eff as the single 0.30 (0.268 vs
0.272 POOL). The herring-specific p95 (0.38 / 0.42 / 0.39 / 0.35 / 0.21 by cod band; SMS, this note's
`sms_diet_pred16.txt`) gives x1.42 - the high end, as in the whiting audit's 0.39-0.45 band.

## 2. Field diets of cod and haddock

### 2.1 ICES stomach database, post-2000, deep areas (`scripts/benth_diet_ices.py`) [F/D]
Mass shares of identified prey, pooled over stomachs. Areas as `../Fiskgrupper i rutan/benthivores/scripts/benth_common.py`
plus depth splits (Skagerrak >= 80 m, NNS >= 100 m; depth is in the 2000+ uploads only).

| Predator, area, n fed | plk (pout / herring / sprat / blue whiting) | bed (epi + Nephrops + infauna) | whiting | notes |
|---|---|---|---|---|
| cod Skagerrak >= 80 m, 358 (Q1 180, Q3 178) | 32.5 (32.5 / 0 / 0 / 0) | 40.2 | 13.7 | P all from 50-69 cm cod in Q3; < 40 cm cod eat crustaceans |
| cod NNS >= 100 m, 53 (Q1) | 43.4 (33.7 / 1.5 / 0 / 8.0) | 29.8 | 7.9 | 50-69 cm: pout 48.7 % |
| haddock NNS >= 100 m, 455 (Q1 2025) | 0 | 88 (epi 21.5, infauna 66.6) | 0.1 | fish 7 % unidentified |
| cod Skagerrak all depths 2000+, 546 | 29.2 (24.4 / 0.8 / 4.0 / 0) | 32.6 | 12.5 | |

Box-length-weighted (cod consumption weights <30 0.19, 30-39 0.24, 40-49 0.21, 50-69 0.28, >= 70 0.08):
cod Skagerrak >= 80 m P 0.19 (all pout), NNS >= 100 m P 0.54 (pout 0.52), both P 0.34 (pout 0.32, herring
0.003, blue whiting 0.016). Group (consumption-weighted with haddock and evidence C for the flatfish): P
share of the five checked pairs 0.076 (Skagerrak), 0.187 (NNS), 0.123 (both).

**Correction to the whiting audit's area code** (`../Vitlingens fiskbyten/scripts/whiting_diet_ices.py`
`area()`): it has no eastern longitude bound, so Baltic hauls at 57-59 N, 17-21 E fall into "Skagerrak"
(3959 cod stomachs 2000+ by that rule vs 546 with lon <= 13). Checked: no whiting record is affected
(905 of 905 whiting in 57-59.2 N, > 7 E are at lon <= 13), so the whiting results stand.

### 2.2 SMS 2023 key run (North Sea, 1981, 1985-87, 1991) (`scripts/sms_cod_haddock_diet.py`) [F/D]
Cod planktivore share by length: < 30 cm 0.047, 30-39 0.149, 40-49 0.198, 50-69 0.266, >= 70 0.197; of it
pout 0.028 / 0.122 / 0.137 / 0.147 / 0.072 and herring 0.007 / 0.015 / 0.051 / 0.111 / 0.121. At the box
mix: P 0.176 (pout 0.110, herring 0.056, sprat 0.009). Haddock: pout only, 0.022 / 0.071 / 0.118 / 0.138;
box mix 0.058. Herring eaten by cod: prey p50 8.5-11 cm, p95 13.5-22.5 cm; ratio p95 0.38 / 0.42 / 0.39 /
0.35 / 0.21 by cod band.

### 2.3 Earlier notes (reused, not re-derived)
- Evidence C (`../Fiskgrupper i rutan/benthivores/evidence_benthivores_C_diets_consumption.md`): "Norway
  pout is the main fish prey of cod in and near the box (36 % of fish mass); herring and sprat together
  are only 2-7 %." Group consumption-weighted diet "B 72 %, Z 2.5 %, P 8.4 %, W 1.9 %" (P 6.5-14.7 %);
  "The P share rests almost entirely on cod." [F/D]
- CINEA 2024 (via evidence C) [F]: in the 2023 North Sea samples herring was found only in 80-90 cm cod;
  by number "sprat (2.90%), herring (1.93%), whiting (1.74%)".
- Hüssy, Andersen & Pedersen 2016 [A, via evidence C]: NE central North Sea cod: crustaceans fall with size
  "whereas that of fish and predominantly herring increased".
- Diel: cod stomachs hold herring with morning and evening peaks (Adlerstein & Welleman 2000 [S, via
  `benthivores_encounter.md`]); North Sea cod "left the seabed at nightfall and returned to it before dawn"
  in Aug-Sep (Righton et al. 2000 [F, same note]); southern North Sea cod near the seabed ~90 % of the time
  (Hobson, Righton, Metcalfe & Hays 2007, MEPS 347:101 [S, search snippet; full text not reached]).
  Norway pout in the Norwegian Deep "generally 5-10 m above the sea bed, but somewhat closer by day";
  feeding more intense at night (Albert 1994 [A, via the same note]).

### 2.4 Is the booked M2 0.145 supported for the box? [D]
Group P share of the checked pairs: booked 0.075; in-area evidence C 0.081-0.166 (central 0.102); ICES
deep 0.076-0.187; SMS at the box mix 0.104. At Q/B 3.5-4.46 x 1.53 t/km2 / 3.33 t/km2 the in-area shares
give M2 0.12-0.30 (central ~0.17). **The booked 0.145 is supported at the group level** (unlike whiting's
0.203). What the field changes is the species: in the box the take is Norway pout (83-99 % of it in the
in-area samples), not herring.

## 3. Vertical overlap: the GOV contradicts the depth discount on herring [D]
- Raw GOV (q = 1, wing spread, daytime hauls, Q1 + Q3 2001-25, depth-stratified, the box): herring 0.70
  t/km2, sprat 0.092, pout 0.25, blue whiting 0.35 (`../Fiskgrupper i rutan/tables/long_run_species_by_group.csv`)
  [F/D]. By quarter (raw for herring and sprat): herring Q1 0.53, Q3 0.87 t/km2 (`../Årsungarnas andel/tables/stage_length_biomass.csv`
  x 0.0069 t/km2 per kg/h) [D].
- Column herring: acoustic July 4.29 t/km2, annual 2.60 (summer-to-annual 0.6) (`../Fiskgrupper i rutan/params_planktivorous_fish.md`) [F].
- So the share of the column herring inside the GOV opening (~5 m) by day is ~0.27 (Q3 0.87/4.29 = 0.20;
  Q1 0.53 against a winter column of ~0.9-2 t/km2 = 0.3-0.6). The GOV under-catches small pelagics (the
  stage note: "herring, sprat ... raw (no published q; the GOV under-catches small pelagic fish)"), so 0.27
  is a floor for the bottom 5 m, and the 10 m band holds 1-2x that: **F_herring,day 0.34 (0.27-0.54)**,
  equal to the unadjusted HERAS box value 0.34 and twice the 0.15 used. The depth discount (grid median
  142 m vs logs at 90-120 m) is contradicted by the GOV hauls, which are at the grid's depths.
- Sprat: GOV 0.092 against acoustic July 0.16 / annual 0.10: ~0.6-0.9 near-bed by day: **F 0.5 (0.3-0.9)**
  [D/J] (was 0.1).
- Blue whiting: its biomass is the GOV value itself ("Q3 visitor below 100 m, missed acoustically"), so
  the share of the counted biomass in the band is high: **0.6 (0.3-1)** [J] (was 0.3). It hardly matters:
  it is outside the cod window.
- Pout: unchanged (0.7 / 0.4).
- Night: unchanged (HERAS twilight logs 0.01-0.04 for herring; "clupeids disperse at night").
- Qualitative support: juvenile ("sardine") herring "lie on or very close to the bottom on sunny, winter
  afternoons. At twilight they rise from the bottom in dense shoals and after dark become widely dispersed
  in the upper water layers" (Tibbo, ICNAF Spec. Publ. 6, Bay of Fundy, 7-22 m; `data/nafo_sp6/`) [F]; the
  IBTS Q1 herring index uses daylight hauls only (search snippet, IBTS reports) [S].
- Effect on the pool sum s_j Fbar_j: 0.0399 -> 0.0748, **x1.87** (1.6-2.6 over the herring range) [D].
  This is the whiting fault-1 analogue (vertical geometry) with the opposite sign: here the overlap was
  set too low, not too high.

## 4. Norway pout biomass and pout mortality [D]
- The pool's pout is 0.25 t/km2 (0.15-0.35): GOV long-run raw (q = 1) and acoustic 2010-19 0.2-0.3.
- Walker et al. 2017 GOV q-at-length for pout (via `../Torskfiskarnas artsammansättning/evidence_B_catchability.md`)
  [S]: 0.066 (<= 11 cm), 0.19 (>= 12 cm). The q-corrected box pout is 0.58 (Q1) / 2.70 (Q3) t/km2 in
  2001-25 (`stage_length_biomass.csv`) [D]. The project uses q = 1 for every GOV group, so this is a
  convention question; pout is the one prey for which it matters here.
- Stock scale: pout TSB 280-490 kt (2016-21), 87-93 kt (2024-25) (`../Fiskgrupper i rutan/planktivores/evidence_pout_bluewhiting.md`)
  [F] over roughly 1-2 x 10^5 km2 of 100-300 m habitat [J]: ~1-5 t/km2 (2016-21), ~0.5-1 (2024-25).
- Mortality check: the booked benthivore take 0.48 t/km2/yr with the in-area pout fraction 0.83-0.99 is
  0.40-0.48 t/km2/yr; on 0.25 t/km2 that is M2 1.6-1.9 /yr from the benthivores alone. Whiting's booked
  take (0.203 x 3.33 = 0.68 t/km2/yr; SMS: 61 % pout) adds ~1.6. Against SMS pout M2 1.89 (2016-22) and
  M1 0.2-0.3, **the field diets and the booked M2 are not compatible with 0.25 t/km2 of pout**: either the
  box pout is ~0.6-1.5 t/km2 or the takes are lower.

## 5. Concurrent-search test (`scripts/clearance_test_benthivores.py`) [D]
Anchors infauna (a 0.036 x exposed 0.2) and epifauna (0.15 x 0.62); both pass the gate and give the same
answer to within 6 %. Model c_plk/c_inf 0.234; field 3.65 (booked), 4.0-9.1 (evidence C), 5.3 (SMS),
3.7-10.6 (ICES deep): **the pair is x15 (booked) to x16-45 (field) too low.** Per species (field / model,
pout B 0.25): pout 0.014-0.039 (x25-70 too low), herring 0.9-2.4 in-area but 0.12 against SMS 1981-91,
sprat 0.013-0.044. So the deficit is the pout term.

Pout biomass scenarios on both sides (model: s_pout and the encounter-weighted window rise; field: the
pool B rises), with the proposed a and window: pout 0.25 / 0.6 / 1.3 t/km2 -> a x f_eff 0.0034 / 0.0043 /
0.0057 (x2.0 / 2.6 / 3.4 vs current), remaining clearance gap vs booked x7.8 / 5.6 / 3.5, vs the field
variants x11-14 / 8-10 / 5-6. At 1.3 t/km2 the rest is inside the encounter's own MC (95th percentile
x3.7 above central). Per species the pout term stays 5-14x low even then, and the revised herring term is
2-5x above the in-area stomachs (0.27 of SMS 1981-91): the pool-level closure is partly compensation.
The bed split 0.578 : 0.318 : 0.016 is applied to every variant, so c_epi/c_inf 13.6 is by construction:
the test is planktivores against the bed.

## 6. Not obtained
- Hobson et al. 2007 and Righton & Michalsen 2003 full texts (HTTP 429/403).
- Walker et al. 2017 primary paper (only via the earlier catchability note).
- Any Skagerrak winter acoustic herring depth profile; any post-2000 Skagerrak haddock stomachs (the 2025
  haddock are NNS >= 100 m).
- A GOV catchability for herring or sprat.
