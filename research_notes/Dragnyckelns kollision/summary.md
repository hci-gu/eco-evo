# The 26D4 haul-key collision: which outputs, notes and values it reaches (2026-11-09)

Research only. Library, code, manifest, resume, AGENTS.md and results/ unchanged; no existing script or note
edited. Corrected copies (`<name>_fixedkey.py`) and their stdout sit next to the originals. Per-script inputs,
keys, collision counts and command lines: `evidence.md`.
Tags: [D] derived here; [S] quoted from the note or table named; [J] judgement.

## The bug

- NS-IBTS GOV hauls were keyed on (Year, Quarter, Ship, HaulNo).
  - Since 2011 Sweden and Denmark both report ship 26D4.
  - In 2021-25 Denmark and Germany share it; Sweden is 77SE then.
- 131 of the box's 354 valid GOV hauls (2000-2020) share their key with another HH haul [D]. The brief and
  "Vitlingens uteblivna predation" say 130 [S]. All are 2011-2019, and the partners lie at 51.5-60.2 N.
- The key does damage in two ways, depending on the input:
  - **Raw XML (whole North Sea):** the partner's CPUEL rows are ADDED to the box haul (`gov_overlap.py`,
    `gov_by_bed.py`).
  - **Compact extracts** (`extract_datras.py` keeps CPUEL only in 56.5-59.5 N, 6.5-12.5 E): only 9 of the 137
    partners lie inside that window [D], so little catch is added. Instead, `drop_duplicates` on the 4-field
    key keeps whichever HH row comes first, usually the Danish North Sea haul in 2013-19. Hauls lost [D]:
    - 2000-20: 75 of 354 box hauls and 218 of 980 Skagerrak hauls;
    - 2021-25: 7 of 272 region hauls (Danish Skagerrak hauls of 2025);
    - 1991-96: clean.
- **Fix.** Add the shoot position to the key.
  - HH and CPUEL ShootLat/ShootLong agree to 4 decimals in every cached extract (0 mismatches in 2327 CPUEL
    hauls) [D], so an exact match orphans no catch.
  - The extract-based fixed copies assign each CPUEL row to the HH haul with the same four fields and the
    nearest position (< 0.05 deg).
  - The closest pair is 2015 Q3 26D4 haul 1: a Swedish and a Danish haul at 57.4870 / 57.4871 N, 9.1378 / 9.1254 E.
    A 4-decimal ShootLat key already separates them, but a 3-decimal rounding of latitude alone would not.

## Library values that change

**None in `fgconfig/fg_library.yaml`** [D, traced in evidence.md].

- **Water column -> whiting `predation_mortality` 0.157.**
  - It is the sum over members of diet share x Q/B x B_member / B_whiting, not an SMS partial M2 scaled to
    the box:
    - saithe 0.02 x 1.0 x 3.0 x 0.492 / 0.446 = 0.066;
    - hake 0.044; spurdog 0.030; anglerfish + ling 0.017 [S, "Vitlingens predatorer" evidence 2;
      library 223.26].
  - The SMS 0.085 in the comment is a comparison only.
  - There is no overlap or spread step. The biomasses come from position-keyed `pred_season_length.py`
    (whiting 1.20 x 0.3715 = 0.446) [S].
  - Unchanged.
- **Benthivores -> whiting** (M2 0.18, a 0.021, `stage_access` [0.68, 0.24], `dark_ratio` 0.15): no GOV overlap
  or bed share enters [S, "Torskfönster och bottenandel" 1-2]. Unchanged.
- **Porpoise pairs** (a 0.0225, `stage_access` 0.054, the re-booked M2): from `wc_window.py` (position-keyed
  lengths) and judgement [S]. Unchanged.
- **Code constant, not library:** `lib/diagnostics/field_diet.py` `FIELD_BED_SHARES['whiting']`.
  - Old 0.533 / 0.211 / 0.251 / 0.006 [S]; corrected **0.769 / 0.204 / 0.027 / 0.001** [D].
  - The working tree has the corrected value (section 264).
  - The old label "2000-19, 352 hauls" was wrong: every cached file was read (2000-2020), and the old key
    merged 2 box hauls. The fixed run has 354 hauls.
  - Saithe is robust: 0.043 / 0.127 / 0.782 / 0.048 [S] against 0.041 / 0.126 / 0.784 / 0.048 [D].

## Notes that need a correction line

1. **"Vattenpelaren på vitling"**
   - **The 0.13 (0.06-0.20) and 0.157 stand.** The field M2 is the member sum:
     - saithe 0.066 (0.023-0.10), hake 0.030 (0.008-0.057), spurdog 0.030, anglerfish + ling 0-0.017 [S];
     - no overlap step in the central value, by the note's own choice [S/J];
     - no step used the collided key or whiting's bed shares.
   - **What falls: the overlap evidence and the verdict built on it.**
     - §1 "Box GOV gives 0.61 (Q1) and 0.82 (Q3) per haul, and 0.80 from the bed classes" [S] becomes
       **0.37 / 0.20** per haul and **0.30** from the bed classes [D].
       - Per haul: saithe x all whiting, fixed key, `../Vitlingens uteblivna
         predation/tables/gov_overlap_fixedkey_box_stdout.txt` [S].
       - Bed classes: sum_c s_c w_c / a_c with cells 0.213 / 0.306 / 0.332 / 0.149. The old shares give 0.788,
         which reproduces the 0.80.
       - Window-sized whiting (10-28 cm): 0.26 / 0.17 [S].
     - §2(ii) and §4 ("L 0.6-0.8 ... lands on the field M2, so no correction"; "L = 0.80, ~0.32 of booked") [S].
       The water column -> whiting a has no depth table, so the gate encounter is linear in L.
       - ~0.32 of booked becomes ~0.12 (0.35 x 0.296 / 0.88). After the note's krill rescaling (x2.60) that is
         ~0.31 of booked, about x3 below the field M2.
       - This is scaled model-diagnostic arithmetic [D, approximate], not a value to use. Rerun `budget_gate`
         on the working tree first.
       - The 16 x 16 whiting map's L 0.29, which the note called an artefact, is close to the field's 0.30.
         A whiting-specific shortfall at field geography remains, and reading (ii) of "Vitlingens
         predatorer" (the North Sea saithe booking carries overlap the box lacks) regains weight [J].
   - **The overlap variant** ("Vitlingens predatorer" 2 and 3.2(ii), the only derivation with an overlap
     step): "~0.10 (0.05-0.17)" [S] becomes **~0.08 (0.04-0.14)** [D].
     - Saithe part = 0.066 x L: L 0.215 (whiting 10-28 cm, mean of 0.262 and 0.167) gives 0.014; L 0.284
       (all whiting) gives 0.019. The old part was 0.03-0.05 [S].
     - Pair = 0.064 + 0.014-0.019 = 0.078-0.083.
     - Range (0.008 + 0.030 + 0) + 0.023 x 0.167 = 0.042 to (0.057 + 0.030 + 0.017) + 0.10 x 0.365 = 0.141.
     - Applied to the booking it would give 0.105-0.110. Not proposed: the note chose not to apply it.
   - §5 (252): the saithe-spread whiting 0.28 / 0.30 [S] used the old shares. Superseded by a rerun.
2. **"Vitlingens predatorer"**
   - **3.1 co-occurrence.** x1.6 (1.0-2.1) [S] becomes **Q1 x1.07 (0.81-1.39), Q3 x1.84 (0.56-3.31), mean
     ~x1.5** [S, "Vitlingens uteblivna predation"].
     - Lever product 2.8 x 2.9 x 1.45 = ~x12 (was x13); residual 32 / 11.8 = ~x2.7 (was x2.5) [D].
     - Evidence 1: "L 1.5-1.6" becomes 1.1-1.8.
   - **3.3 overlap table** [S old; fixed-key table S]:
     - saithe x whiting 10-28 cm: 0.39 / 0.87 becomes 0.26 / 0.17;
     - saithe x whiting: 0.61 / 0.82 becomes 0.37 / 0.20;
     - hake x whiting: 1.24 / 0.96 becomes 0.69 / 0.96.
   - **Bed rates.** Whiting 189 / 208 / 53 / 58 kg/h [S] becomes 149 / 148 / 27 / 3.3. Cod >= 20 cm becomes
     35 / 155 / 127 / 43, saithe 1 / 39 / 22 / 127 [D, fixed box stdout].
   - **Section 2.** "saithe ~0.05" becomes ~0.013-0.024 [D].
   - **Region table.** Now `tables/gov_overlap_fixedkey_region_stdout.txt` (972 hauls, was 967). Saithe x
     whiting: 0.59 / 0.64 [S] becomes 0.32 / 0.22 [D].
3. **"Torskfönster och bottenandel" §3.** "the GOV co-occurrence x1.6 (1.0-2.1)" [S] becomes ~x1.5 (Q1 1.1, Q3
   1.8). The residual becomes ~3.4 / 1.45 = x2.3 (was "x2") [D].
4. **"Krill i Skagerrak" a_revisit**
   - **Finding 3, whiting bed shares** [S old; D new, `tables/gov_by_bed_fixedkey*_stdout.txt`]:
     - box: 0.53 / 0.21 / 0.25 / 0.006 becomes 0.769 / 0.204 / 0.027 / 0.001 (354 hauls, 2000-20);
     - region pooled: 0.404 / 0.230 / 0.213 / 0.153 becomes 0.592 / 0.343 / 0.059 / 0.007;
     - Q1: 0.306 / 0.323 / 0.138 / 0.233 becomes 0.424 / 0.429 / 0.136 / 0.011;
     - Q3: 0.518 / 0.140 / 0.254 / 0.088 becomes 0.798 / 0.191 / 0.011 / 0.000.
   - **Saithe, region.** 0.024 / 0.156 / 0.693 / 0.127 becomes 0.022 / 0.158 / 0.700 / 0.120 (robust).
   - **`weighted_gate.py`.** Its whiting rows and its saithe "GOV 2000-19" row hard-code the old outputs, and the
     sentence "krill 1.61-1.64, fish pairs 0.12-0.24, epifauna OFF" [S] rests on them. Whiting sits even more
     over < 100 m beds, so the direction holds [J]. Not rerun here (model diagnostic).
   - **Period label.** "2000-19" should read 2000-20.
5. **"Tumlarens byten" §4** (pool ratios 1991-96 / 2001-25; old [S] -> fixed [D]):

   | pool | Skagerrak | Skagerrak < 40 cm | IIIa | box rects |
   |---|---|---|---|---|
   | planktivores | 1.22 -> 1.23 | 1.22 -> 1.23 | 1.03 -> 1.05 | 1.05 -> 1.10 |
   | benthivores | 1.00 -> 1.06 | 1.22 -> 1.32 | 1.06 -> 1.10 | 1.08 -> 1.13 |
   | whiting | 0.87 -> 0.91 | 0.88 -> 0.91 | 1.09 -> 1.15 | 0.74 -> 0.78 |
   | water column | 0.30 -> 0.31 | 0.84 -> 0.88 | 0.31 -> 0.31 | 0.21 -> 0.21 |

   - Single species and lengths:
     - Norway pout 4.46 -> 4.65;
     - saithe < 40 cm in 2001-20: 0.136 -> 0.129; < 30 cm: 0.023 -> 0.034;
     - Skagerrak GOV herring 2021-25: 211 -> 267 kg/h (one restored 2025 haul holds 13.2 t/h).
   - `clearance_test_fixedkey.py`:
     - model/field: benthivores 2.2-14.5 becomes 2.3-14.5, whiting 1.7-11 becomes 1.8-11;
     - field-implied a: benthivores 0.008-0.054 becomes 0.008-0.051, whiting 0.018-0.12 becomes 0.018-0.11.
     - Verdicts and proposals unchanged; today's-density rows identical.
6. **"Torskfiskarnas artsammansättning" / "Fiskgrupper i rutan" (2021-25).**
   - `gadoid_mix.py` drops 7 Danish Skagerrak hauls of 2025.
   - Box-rectangle and strict-box rows are identical. The depth-stratified box total moves 518.7 -> 518.2 kg/h,
     and the Skagerrak haul mean 504.2 -> 498.2 kg/h [D].
   - `fish_guilds.md` already records the fix as "no material change" [S]. Rerun here: planktivores pooled
     261.0 -> 275.0 kg/h (1.81 -> 1.90 t/km2), share 32.4 -> 33.6 % [D].
   - No library value uses these 2021-25 tables.

## Results table

| script | contaminated? | quantity | old | corrected | library value affected |
|---|---|---|---|---|---|
| Vitlingens predatorer/gov_overlap.py (box) | yes, catch added; 131/354 hauls [D] | cod>=20 x whiting 10-28 L Q1/Q3; saithe x whiting 10-28 L | 1.57/1.57; 0.39/0.87 [S] | 1.07/1.84; 0.26/0.17 [S, earlier fix] | none (notes 1-3) |
| same (region) | yes; 375/972 collide, 5 merged [D] | saithe x whiting L Q1/Q3 | 0.59/0.64 [S] | 0.32/0.22 [D] | none |
| Krill i Skagerrak/a_revisit/gov_by_bed.py (box) | yes, catch added [D] | whiting bed shares | 0.533/0.211/0.251/0.006 [S] | 0.769/0.204/0.027/0.001 [D] | field_diet.py FIELD_BED_SHARES (corrected in working tree) |
| same (region) | yes [D] | whiting bed shares pooled | 0.404/0.230/0.213/0.153 [S] | 0.592/0.343/0.059/0.007 [D] | none |
| same, saithe | yes, robust | box; region | 0.043/0.127/0.782/0.048; 0.024/0.156/0.693/0.127 [S] | 0.041/0.126/0.784/0.048; 0.022/0.158/0.700/0.120 [D] | none |
| Krill i Skagerrak/a_revisit/weighted_gate.py | downstream (hard-coded) | whiting/saithe spread rows | as above [S] | not rerun | none |
| Tumlarens byten/then_now_densities.py | yes: 2001-20 lost 218/980 Skag, 105/539 box-rect; 24 Skag got other catch; 2021-25 lost 7 [D] | Skagerrak pool ratios plk/ben/wht/wc | 1.22/1.00/0.87/0.30 [S] | 1.23/1.06/0.91/0.31 [D] | none |
| Tumlarens byten/clearance_test.py | downstream (hard-coded ratios) | implied a ben; wht (ranges) | 0.008-0.054; 0.018-0.12 [S] | 0.008-0.051; 0.018-0.11 [D] | none (a 0.0225 is J) |
| Torskfiskarnas artsammansättning/gadoid_mix.py | haul loss only: 7/272 region, 0 box [D] | strat. box kg/h; Skagerrak kg/h | 518.7; 504.2 [S] | 518.2; 498.2 [D] | none |
| Fiskgrupper i rutan/scripts/fish_guilds.py | inherits; own join clean | planktivores pooled kg/h (share) | 261.0 (32.4 %) [S] | 275.0 (33.6 %) [D] | none |
| Attackrater ur mötesteori/size_structure.py | inherits; own join clean | herring pooled N/t; M2 | 2.885e4; 7.949e6 [S] | 2.709e4; 7.804e6 [D] | none recomputed (<= 2 % on M2) [J] |
| Interferens hos bottenfisk/ibts_patchiness.py | inherits; own join clean | benthivore Lloyd median; pooled | 3.06; 2.95 [S] | 3.06; 2.93 [D] | none |
| Torskfiskarnas artsammansättning/catchability.py | inherits; own join clean | - | - | not rerun (box unchanged) | none |
| Torskfiskarnas bytesandelar/*.py | inherit; own joins clean | - | - | not rerun (box unchanged) | none |
| Fiskgrupper i rutan/benthivores/scripts/prey_length_gape.py | no (CPUEL only, own position filter) | rows dropped | - | 49 of 107 225; < 0.1 % per species [D] | none |
| Krill över grunda bottnar/stomachs/scripts/link_datras.py | no (1980-91; match A needs an adjacent rectangle) | - | - | - | none |
| all other GOV scripts (key with position, Country or StNo) | no | - | - | - | - |
