# One vision rule for every visual fish-predator pair (2026-11-11)

Research subagent; saved by the main agent from the subagent's report. Nothing in the library, code,
tests, manifest, resume or AGENTS.md changed; no training or rollout; no value set from model output.
Temporary library copies only in the session scratchpad.

Files:
- `scripts/one_rule_pairs.py` -> `tables/one_rule_pairs_stdout.txt` / `.json`: reproduction of every
  library derivation, R_sat per member, a_new / a_lib for 13 variants, class ratios, implied night/day,
  term decomposition of the in-layer fish pairs.
- `scripts/gate_variants.py` -> `tables/gates/budget_gate_<variant>.txt` (40 runs), `tables/gate_variants.json`:
  `tools/probes/budget_gate.py` exec'd from its own file with one textual change (`library_path=<temp copy>`;
  the `lib` run is byte-identical to today's output), plus a per-bed-class gate.
- `scripts/tabulate.py` -> `tables/tabulate_stdout.txt`: predator side, prey side, per-class krill share.
- `scripts/band_reading.py` -> `tables/band_reading_stdout.txt` / `.json`: water-column cross-check with the
  fish pairs on the band overlap (the sibling's V_one convention; reproduces V_one 0.687 / 0.705 / 0.693).

```
S=<scratch>
python3 -I scripts/one_rule_pairs.py <repo> $S/tmp tables/one_rule_pairs.json
python3 scripts/gate_variants.py <repo> tables/one_rule_pairs.json $S/libs tables
python3 -I scripts/tabulate.py <repo> tables
python3 -I scripts/band_reading.py <repo> tables/band_reading.json
```

Tags: [S] source, [D] derived here, [J] judgement.

How the rule was applied: research_notes/"Synregel i vattenpelaren" ("Proposed single rule"); every other
factor of each pair as in the library (scan, speeds, N/t, membership, overlap - band O for krill, in-layer
1/H for fish, the 243 time-near-bed factors - prey shares, depth_access_table geometry, windows, p); only the
reactive-distance term replaced. Light = the section 253 measured light over the prey's own vertical profile
(the "literal reading": band light, library overlap; not V_one). Krill L 25 mm (library; 30 mm sensitivity),
fish prey 15 cm, R_nv krill 0.1 / fish 0.3 m. The clupeid eye (herring, mackerel) keeps its Ke 1 lx and biting
threshold; non-visual members (spurdog, velvet belly) keep their light-free R. Rule B: RD_max 1.2 m (1.0-1.9).
Review p: krill 0.6/0.5 (x1.2), 220-convention fish 0.45/0.2 (x2.25), benthivore fish 0.45/0.3 (x1.5).

## Findings

1. **The two-rule inconsistency exists in every visual fish predator, not only the water column (WC).**
   Whiting and mackerel have the WC's form (krill pair light-driven under the measured light; fish pairs a
   fixed R: 0.2 m dark, 0.25 m at the bed by day, 0.5 m on herring by day). Benthivores are light-driven on
   both sides but under two light climates (krill: measured light, R_nv 0.1, p 0.5; fish: the old central
   profile, ~x6 darker at 80-150 m, R_nv 0.6 L, p 0.3). Planktivores have no fish pair; their krill and
   zooplankton pairs share the image-area k_v x g form (consistent today). For whiting, planktivores and
   mackerel the krill R_sat already sits at the 2 L_pred cap (0.24-0.53 m): their fixed fish R is the outlier.
2. **Rule A changes a krill pair only where 2 L_pred > R0_krill = 0.57 m** (|C0| 0.4, 25 mm; 0.675 m at 30 mm):
   WC x0.45, benthivores x0.73 (cod > 28.5 cm); whiting, planktivores, mackerel x1.00. Rule B lowers every
   krill pair to x0.29-0.39 [D].
3. **Every fish pair rises under the one rule; light, not the floor, carries it.** WC pairs x4.0-5.7 (Ke
   bracket x3.3-8.2), whiting x2.2-2.5, mackerel -> planktivores x1.29, benthivore fish pairs x1.19-1.25.
   The section 220 premise ("bed below ~0.5 lx even at noon") fails under the 253 light on beds < 150 m: the
   pout layer has a daytime median 2.3 lx over < 100 m beds, 0.20 lx over 100-150 m [D]. WC -> planktivores:
   pout by day x11.3 (x6.1 at 150-250 m), herring by day x3.5 (saithe R_sat 0.86 / 1.40 m vs fixed 0.5 m at
   ~150 lx in the 10-100 m herring layer). With the library's fish floor 0.2 m instead of 0.3, still x5.0.
4. **Predator-side gate: the library's OFF pairs pass under rule A.** Equal cells: WC -> planktivores 0.21 ->
   1.07, WC -> benthivores 0.18 -> 0.66, whiting -> benthivores 0.25 -> 0.58. Field spread under A, library p:
   the WC passes on every prey; OFF remain whiting -> benthivores (0.29) and whiting -> mobile_epifauna (0.23;
   an unchanged benthic pair moved only through whiting's other a's). Rule B passes everything in both
   spreads; with review p, B puts mackerel -> planktivores OFF (3.25x).
5. **Prey side: the fish-prey M2 deficits shrink.**

   | prey | library | rule A | rule B | status |
   |---|---|---|---|---|
   | planktivores | 0.38 | 0.86 | 1.04 | toward booking |
   | benthivores | 0.37 | 0.69 | 1.03 | toward booking |
   | whiting | 0.27 | 0.63 | 0.66 | toward booking |
   | mackerel | 0.10 | 0.24 | 0.28 | still OFF |
   | krill pool | 1.10 | 1.02 | 0.51 | B moves away (planktivores' krill a x0.32) |

   Still OFF under A: benthivores -> whiting (0.21 -> 0.25), WC -> mackerel (0.10 -> 0.24), seals ->
   predatory_fish (out of scope).
6. **WC krill share over 150-250 m beds**: library 0.74; literal rule A 0.25 class-resolved (0.20 with flat
   fish a's); every variant 0.09-0.32. Field: saithe all seasons 0.19 (0.08-0.27), group Q3 0.47
   (0.32-0.69); booked 0.36. The band reading gives 0.45 (review p 0.30; Ke bracket 0.35-0.40). **Under one
   vision rule the remaining spread at 150-250 m (0.25 vs 0.45) is the fish pairs' overlap convention, not
   vision** [D].
7. **The juvenile-benthivore pair passes in the literal reading** (WC -> benthivores x3.98 under A: 0.66 of
   booked at equal cells, 0.80 at the field spread), reversing the sibling's V_one 1.1-1.6x: the literal
   reading keeps section 243's time-near-bed factors (saithe 0.05). The band reading (saithe's 15 m
   exponential profile over the bottom 2 m) puts it at 2.0x (2.6x review p). The pair hinges on the overlap
   convention, not on vision.
8. **dark_ratio**: no double count if max_intake_rate is the rule's annual mean and dark_ratio is REPLACED by
   the rule's own night/day (the engine normalises m(t) to mean 1; the gate does not depend on it). The
   rule's night/day for the WC fish pairs is 0.14-0.21 vs the library's 0.41-0.48. It contradicts three
   stomach-derived values: planktivores -> krill 0.57 (rule 0.10), mackerel -> krill 0.42 (rule 0.26),
   mackerel -> planktivores 0.16 (rule 0.70; from R_nv 0.3 m on a clupeid eye, unsupported; with R_nv 0.2 m
   the pair goes x0.97).

## Current R and p conventions per group

| group | krill pair (band, measured light) | fish pairs (library) | p krill / fish | dark_ratio krill / fish |
|---|---|---|---|---|
| water column | R^2 = R_nv^2(1-g) + R_sat^2 g; R_sat = min(R0 0.869 (k_v 100, 25 mm, \|C0\| 1 implied), 2 L); R_nv 0.1; Ke 0.02; non-visual members R_nv only | in-layer 1/H, fixed R: herring day min(0.5, 2L), night 0.2; pout / blue whiting 0.2; bed fish 0.25 / 0.2; hake day 0.4; elasmobranchs 0.3 (light-free); anglerfish strike 0.4 | 0.5 / 0.2 | 0.02 / 0.41-0.48, mackerel prey 0.92 |
| whiting | same rule; R_sat = 2 L binds (0.24 / 0.50 m); band within 10 m of the bed | in-layer fixed: herring day 0.50, pout / blue whiting 0.2, juvenile benthivores 0.25 / 0.2 (H 2 m) | 0.5 / 0.2 | none / 0.48 (planktivores), 0.64 (benthivores) |
| benthivores | same rule over length classes; band 10 m | light-driven: R_sat 2 L, R_nv 0.6 L, Ke 0.02, but the old central light at the bed; near-bed band with prey share F | 0.5 / 0.3 | 0.03 / 0.06, 0.15 (0 deg) |
| mackerel | clupeid eye (Ke 1 lx, threshold 0.016 lx, R_nv 0.05); R_sat = 2 L (0.53); horse mackerel gadoid | in-layer fixed: herring day 0.5, night 0.2; herring and sprat only | 0.5 / 0.2 | 0.42 / 0.16 (stomachs) |
| planktivores | herring clupeid, pout / blue whiting gadoid; R_sat = 2 L (0.24-0.50) | none; zooplankton pair k_v sqrt(A) x I/(Ke + I) (k_v 115 / 100, Ke 1 lx), same form as krill, no contrast term | 0.5 / (copepods 0.6-0.7) | 0.57 (stomachs) / 0.03 |

Adding |C0| 0.4 on krill without a copepod contrast would create a new asymmetry inside the planktivores;
their zooplankton pair was left alone [J].

## a_new / a_lib (equal-cell annual mean, library p)

| pair | lib a | A | A Ke 0.1-2e-4 | B | B Ke 0.1-2e-4 | other (A) | night/day A vs lib |
|---|---|---|---|---|---|---|---|
| WC -> krill | 0.026 | 0.450 | 0.38-0.52 | 0.312 | 0.27-0.36 | k_deep 0.05/0.09: 0.46/0.41; 30 mm 0.63 | 0.025 vs 0.02 |
| whiting -> krill | 0.043 | 1.000 | 0.88-1.13 | 0.289 | 0.26-0.32 | | 0.041 vs none |
| planktivores -> krill | 0.023 | 1.000 | 0.95-1.06 | 0.318 | 0.31-0.33 | | 0.10 vs 0.57 |
| benthivores -> krill | 0.013 | 0.727 | 0.64-0.82 | 0.306 | 0.27-0.34 | 30 mm 0.86 | 0.036 vs 0.03 |
| mackerel -> krill | 0.0039 | 1.000 | 0.96-1.05 | 0.390 | 0.39-0.40 | | 0.26 vs 0.42 |
| WC -> planktivores | 0.0060 | 5.43 | 4.65-7.26 | 4.99 | 4.29-6.63 | R_nv 0.2/0.5: 4.98/6.88 | 0.16 vs 0.41 |
| WC -> benthivores | 0.0055 | 3.98 | 3.30-5.55 | 3.81 | 3.17-5.30 | 3.42/5.80 | 0.21 vs 0.48 |
| WC -> whiting | 0.013 | 5.73 | 4.64-8.22 | 5.29 | 4.31-7.51 | 5.11/7.73 | 0.14 vs 0.48 |
| WC -> mackerel | 0.00088 | 2.70 | 2.63-3.01 | 2.66 | 2.59-2.96 | | 0.42 vs 0.92 |
| whiting -> planktivores | 0.0212 | 2.17 | 2.01-2.56 | 2.17 | same (cap) | 1.68/3.73 | 0.58 vs 0.48 |
| whiting -> benthivores | 0.21 | 2.55 | 2.33-3.08 | 2.55 | same | 1.81/4.89 | 0.56 vs 0.64 |
| mackerel -> planktivores | 0.0266 | 1.29 | 1.29 | 1.29 | same | 0.97/2.33 | 0.70 vs 0.16 |
| benthivores -> planktivores | 0.0198 | 1.25 | 1.02-1.75 | 1.16 | 0.95-1.61 | 1.09/1.81 | 0.09 vs 0.06 |
| benthivores -> whiting | 0.021 | 1.19 | 0.97-1.71 | 1.10 | 0.90-1.56 | 1.03/1.72 | 0.18 vs 0.15 |

- Review p multiplies these by x1.2 (krill), x2.25 (220 fish), x1.5 (benthivore fish). Rule B's RD_max
  1.0-1.9 m moves only members with 2 L > 1 m.
- WC krill class a (< 100 / 100-150 / 150-250 / > 250 m): A 0.0306 / 0.0141 / 0.0026 / 0.00007;
  B 0.0212 / 0.0098 / 0.0019 / 0.00005. Under the rule the fish pairs become depth-dependent; WC class a /
  equal-cell a 1.37 / 1.23 / 0.75 / 0.55 (planktivores), 1.58 / 1.34 / 0.60 / 0.37 (benthivores).
- Reproduction: krill tables all five exact; in-layer encounters WC 0.00595 / 0.00569 / 0.01301, whiting
  0.02128 / 0.2091, mackerel 0.02653. Only WC -> mackerel deviates (0.00117 vs 0.00088, a known
  discrepancy); every result uses ratios.

## Gate, predator side

Encounter share / booked share; equal cells (field spread in brackets, WC and whiting only); `*` = OFF.

| variant | WC krill / plk / ben / whi | whiting krill / ben / epi / plk | benthivores krill / plk | mackerel krill / plk / zoo | planktivores krill |
|---|---|---|---|---|---|
| library | 2.41 (2.25) / 0.21* (0.32*) / 0.18* (0.27*) / 0.18 (0.09) | 1.48 (1.65) / 0.25* (0.12*) / 0.51 (0.24*) / 0.35 (0.17*) | 1.22 / 0.41 | 1.67 / 0.83 / 0.54 | 0.95 |
| library, review p | 2.17 (1.94) / 0.36 (0.51) / 0.30* (0.43) / 0.31 (0.14) | 1.41 (1.61) / 0.45 (0.22*) / 0.40 (0.19*) / 0.63 (0.31*) | 1.43 / 0.60 | 1.68 / 1.57 / 0.45 | 1.06 |
| A | 1.02 (0.76) / 1.07 (1.29) / 0.66 (0.80) / 0.97 (0.37) | 1.35 (1.57) / 0.58 (0.29*) / 0.46 (0.23*) / 0.69 (0.35) | 0.88 / 0.51 | 1.65 / 1.06 / 0.53 | 0.95 |
| A, review p | 0.66 (0.46) / 1.30 (1.48) / 0.80 (0.92) / 1.18 (0.42) | 1.20 (1.48) / 0.97 (0.52) / 0.34 (0.18*) / 1.15 (0.61) | 1.04 / 0.74 | 1.64 / 1.98 / 0.44 | 1.06 |
| A, fish depth tables (T) | 1.02 (0.85) / 1.07 (1.22) / 0.66 (0.67) / 0.97 (0.58) | 1.35 (1.55) / 0.58 (0.37) / 0.46 (0.23*) / 0.69 (0.39) | 0.88 / 0.51 | as A | 0.95 |
| B | 0.84 (0.62) / 1.17 (1.37) / 0.76 (0.89) / 1.07 (0.39) | 0.84 (1.20) / 1.25 (0.77) / 0.98 (0.61) / 1.48 (0.92) | 0.38 / 0.48 | 1.07 / 1.74 / 0.88 | 0.43 |
| B, review p | 0.53 (0.37) / 1.37 (1.54) / 0.89 (0.99) / 1.25 (0.44) | 0.66 (1.03) / 1.84 (1.24) / 0.64 (0.43) / 2.18 (1.47) | 0.44 / 0.70 | 1.07 / 3.25* / 0.73 | 0.49 |

The Ke bracket, k_deep, floor and L_krill variants do not change the verdict (OFF only on whiting's field
row, plus B review p on mackerel -> planktivores); exceptions: B Ke 0.1 review p puts WC -> krill at the field
spread OFF (0.31); A Ke 2e-4 review p puts whiting -> epifauna OFF at equal cells too. All in `tables/gates/`.

## Gate, prey side

Implied M2 / booked M2, equal cells (field spread); totals count mammal and seabird pairs at their booking.

| prey | library | A | A, review p | B | B, review p |
|---|---|---|---|---|---|
| krill (total) | 1.10; WC 2.41 (2.25), plk 0.95 | 1.02; WC 1.02 (0.76), whi 1.35, mac 1.65, ben 0.88 | 1.08 | 0.51; plk 0.43 | 0.53 |
| planktivores | 0.38; WC 0.21* (0.32*), whi 0.35 (0.17*), ben 0.41 | 0.86; WC 1.07 (1.29), whi 0.69 (0.35), ben 0.51, mac 1.06 | 1.14 | 1.04; whi 1.48 (0.92), mac 1.74 | 1.37 |
| benthivores | 0.37; whi 0.25* (0.12*), WC 0.18* (0.27*) | 0.69; whi 0.58 (0.29*), WC 0.66 (0.80) | 0.92; 0.97 (0.52), 0.80 (0.92) | 1.03; 1.25 (0.77), 0.76 (0.89) | 1.34 |
| whiting | 0.27; ben 0.21*, WC 0.18* (0.09*) | 0.63; ben 0.25*, WC 0.97 (0.37) | 0.77; ben 0.37, WC 1.18 (0.42) | 0.66; ben 0.24* | 0.79 |
| mackerel | 0.10* | 0.24* | 0.29* | 0.28* | 0.33* |

## WC krill share at 150-250 m

Bed class 2 on the 16 x 16 probe grid (library 0.742; 60 x 60: 0.728). "flat" = fish a's as the engine
would carry them; "cls" = fish a's class-resolved by the rule (the T flavour).

| variant | flat | cls |
|---|---|---|
| library | 0.74 | 0.74 |
| library, review p | 0.61 | 0.61 |
| A | 0.20 | 0.25 |
| A, review p | 0.12 | 0.15 |
| A, Ke 2e-4 / 0.1 | 0.24 / 0.14 | 0.24 / 0.19 |
| A, k_deep 0.05 / 0.09 | 0.22 / 0.11 | 0.26 / 0.15 |
| A, krill 30 mm | 0.26 | 0.32 |
| A, R_nv fish 0.2 / 0.5 | 0.21 / 0.17 | 0.28 / 0.20 |
| B | 0.16 | 0.20 |
| B, review p | 0.09 | 0.12 |
| band reading, A (lib p / review p) | 0.45 / 0.30 | |
| field | all seasons 0.19 (0.08-0.27); group Q3 0.47 (0.32-0.69) | |
| booked | 0.36 | |

The equal-cell krill ratio 1.02 under A is two factors crossing, not a calibration; quote the field spread
(0.76) and class 2 (0.69 of booked).

## What moves toward and away from the bookings

- Toward (A, library p): WC -> krill 2.41 -> 1.02 (field 0.76); WC -> planktivores 0.21 -> 1.07; WC ->
  benthivores 0.18 -> 0.66; WC -> whiting 0.18 -> 0.97 (field 0.09 -> 0.37); whiting -> benthivores 0.25 ->
  0.58; whiting -> planktivores 0.35 -> 0.69; benthivores -> krill 1.22 -> 0.88; benthivores -> planktivores
  0.41 -> 0.51; mackerel -> planktivores 0.83 -> 1.06; every fish prey's total implied M2.
- Unchanged: under A planktivores, mackerel and whiting on krill (cap binds); all benthic and zooplankton pairs.
- Away: under B planktivores -> krill 0.95 -> 0.43 and the krill pool's total 0.51 (B carries a piscivore
  prey-size exponent onto planktivores whose k_v was anchored on copepods, outside its domain [J]); whiting ->
  planktivores 1.48 (2.18 review p); mackerel -> planktivores 1.74 (3.25 OFF review p). Under A review p,
  WC -> krill at the field spread falls to 0.46 (0.37 under B review p).
- Persisting OFF pairs vision cannot fix: benthivores -> whiting (x1.19 only; prey side 0.25; the section 251
  residual); WC -> mackerel (x2.7, still 0.24; windows and geometry); whiting -> epifauna at the field spread;
  seals -> predatory_fish (out of scope).

## Unchanged and out of scope

Benthic pairs (exposure factors): benthivores -> infauna, mobile_epifauna, norway_lobster; whiting ->
mobile_epifauna. Zooplankton pairs: planktivores -> zooplankton, WC -> zooplankton (R_cop 0.02), whiting ->
zooplankton, mackerel -> zooplankton (measured clearance). Mammal pairs.

## Caveats

1. [D] Literal reading: in-layer O = 1/H kept, light over the prey's profile. Under concurrent search (242)
   the in-layer form credits a predator with full time in every prey's layer; section 243 replaced it for
   WC -> benthivores only. The herring-by-day term (saithe at 150 lx) is where this convention matters most.
   The band cross-check covers the WC and rule A only.
2. [J] |C0| 0.4 / 0.7, R_nv fish 0.3 m and review p are judgements; rule B's RD_max and prey-size exponents
   are salmonid analogues [S]. No gadoid R on krill or fish exists.
3. [D/J] Light below 150 m is extrapolated (K_deep 0.05-0.09); it moves the class-2 krill share 0.15-0.26,
   more than the Ke bracket.
4. [D] R_nv fish 0.3 m exceeds 2 L_pred for juvenile whiting, juvenile hake and horse mackerel (floor used);
   none carries more than 0.05 of any fish pair.
5. [J] The clupeid eye is kept (Ke 1 lx; krill floor 0.05 m); the rule's fish floor on mackerel contradicts
   the stomach night/day 0.16.
6. [D] Benthivore fish pairs approximated by the ratio of benth_encounter's vision term with approximated
   member weights; the 243.3 / 251 member scripts were not re-exec'd.
7. [D] The gate runs on the 16 x 16 probe grid (class shares 0.203 / 0.289 / 0.340 / 0.168); the fish pairs'
   night/day read at -5 deg with the derivations' 0-deg speed and layer switch.
8. [D] Shares only: annual-mean shares, not totals; booked M2 and rations untouched; the Q3 field value is
   seasonal.
9. [J] WC -> mackerel prey geometry is the planktivore geometry (0.00117 vs library 0.00088).
10. Not fitted: no contrast, p, Ke or floor was chosen to hit 0.36, 0.47 or any booking.

## User decisions (2026-11-11)

- Before rule A goes into the library, find out which overlap convention for the fish pairs (in-layer 1/H
  or the band, as the krill pairs) biology and section 242's concurrent search support; then rule A and the
  overlap together as one section.
- When the rule goes in, the stomach-derived night/day ratios win where they exist (planktivores -> krill
  0.57, mackerel -> krill 0.42, mackerel -> planktivores 0.16); the rule's own night/day only for pairs
  without diet data.
