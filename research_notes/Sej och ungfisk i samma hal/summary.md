# Saithe and the benthivores' juveniles in the same haul; re-booking water column -> benthivores (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library, code, tests, manifest,
AGENTS.md, resume or existing notes; no training; library copies only in the session scratchpad. Every value is
read from field data (NS-IBTS GOV, stomach-based diets) or the SMS 2023 key run, never from model output; the gate
is a check and chose nothing. Sources E1-E9 and model facts M1-M7: `evidence.md`. Grades: [F] field data or full
text, [L] lab, [A] abstract only, [S] secondary / via another note, [D] derived here, [J] judgement.

## Files

Run from `scripts/`; each script's stdout is saved beside it.

- `gov_within_class.py` -> `gov_within_class_stdout.txt`, `../tables/within_class.json`: the within-class
  co-occurrence factor W of adult saithe (>= 35 cm) and the benthivores' J (also Js < 19 cm, A, B), box and window,
  Q1 / Q3. Haul construction copied from research_notes/"Rekryternas bottendjup"/scripts/gov_juv_bed.py (same J
  cuts, Walker q, member weights) with saithe added; decomposition by 25 m depth band, ICES rectangle and year;
  permutation nulls (1000) and year-block bootstrap (1000); robustness variants.
- `sms_rebook.py` -> `sms_rebook_stdout.txt`, `../tables/rebook.json`: the booking's origin member by member; the
  field route (member B x Q/B x share on the model's benthivore members x J / A, Monte Carlo over the ranges); SMS
  2023 saithe on haddock, predator side and prey side at the box's saithe density.
- `growth_check.py` -> `growth_check_stdout.txt`: derived g of every group on in-memory library copies with the
  booking varied; the benthivores' booked M2 sum; the water column's booked ration.
- `gate_rebook.py` -> `gate_rebook_stdout.txt`, `../tables/gates/budget_gate_<run>.txt`,
  `../tables/gates/gate_rebook.json`: a copy of research_notes/"Vattenpelarens vuxna bottenfisk"/scripts/
  gate_adult.py; every run starts from its central (pA+Aprof+hake_out) and varies the booking and, as a
  diagnostic, f_J x W. `book_0.0833` reproduces 1.54 / 1.41 / 2.13 and 0.89 / 0.87 exactly.
- `papers/README.md`: one Unpaywall record (closed access).

```
R=<repo>/research_notes; S=<session scratchpad>/sejungfisk
python3 -I gov_within_class.py "$R/Fiskgrupper i rutan/data/extract" "$R/Torskfiskarnas artsammansattning/data/extract" \
    "$R/Fiskgrupper i rutan/tables/length_weight_all.csv" <repo>/maps/depth.npz <repo>/fgconfig/fg_library.yaml ../tables
    # a-umlaut in the real folder name; ~3.5 min (NBOOT=NPERM=1000)
python3 -I sms_rebook.py "../../Fiskgrupper i rutan/predators/data/population" ../tables/rebook.json
python3 growth_check.py <repo>
python3 gate_rebook.py <repo> "$R/Rekryternas bottendjup/tables/gov_juv_bed.json" \
    "$R/Vattenpelarens vuxna bottenfisk/tables/decompose_members.json" $S/libs ../tables/gates ../tables/within_class.json
```

## Question

1. Inside a bed-depth class, do adult saithe and the benthivores' juveniles co-occur at haul level less (or more)
   than the class means say? (The class-level part is already carried by the pair's depth_access_table and the
   decided recruit_depth_table; research_notes/"Sejens hojd over botten" quoted haul-level F 0.086 / 0.296 without
   separating the two levels.) Can the engine represent a within-class factor, would trained policies produce it,
   and does it belong in a parameter?
2. What booking (predation_mortality, today 0.0833) do the members' field diets and rations support, with the J / A
   split, against SMS 2023 at the box's predator density? What does a change do to the benthivores' g?
3. Gate on library copies with all decided corrections: WC -> benthivores and the benthivore prey total with the
   re-booking, and with the within-class factor as a diagnostic.

## Model (M1-M7)

- The pair after the decided corrections: a 0.117 (combined x2.96, hake out), stage_access [0.3818, 0.01653], depth
  table (mean 1) 1.63 / 1.34 / 0.55 / 0.30, booking 0.0833 (flow 0.127 t/km2/yr on B 1.53).
- J rides inside B (one J/B per cell); recruits arrive uniformly (the decided table would shape the arrival by bed
  class); the water column and the benthivores move by learned policies on 1 km cells (M5, M6).
- Benthivores and the water column are open populations: g = P / ((s* - u) x 1460), P 0.74 and 0.42; since section
  250 the booked M2 is not in g (M2).

## Findings

### 1. Within-class co-occurrence [F data, D]

W = sum_h S_h Y_h / sum_h S_h Ybar_k inside each bed class k (haul level over class level, ratio of sums, zeros
kept), aggregated over the classes with the pair's class-level encounter weights (cell share x depth table x
saithe x prey). E(g) is the W that co-occurrence only at the scale of strata g would give; W / E(g) is what remains
inside g. Saithe presence weights Q3 0.9 / Q1 0.1. Brackets: 95 % year-block bootstrap.

Identity check: F_haul = F_class x W exactly (box Q3 J: 0.316 x 0.668 = 0.211; with the depth table F_class 0.447).

| reading (J, haddock + cod member-weighted) | W | permutation null inside class: 95 %, p(null <= obs) | E_band (25 m) | E_rect | within-year W / E_year (plain / LOO) |
|---|---|---|---|---|---|
| box Q3 | 0.78 [0.58-1.07] | 0.60-1.68, p 0.22 | 1.06 | 0.98 | 0.95 / 0.99 |
| box Q1 | 0.57 [0.35-0.90] | 0.54-1.69, p 0.036 | 1.09 | 0.98 | 0.85 / 1.48 (1 haul per stratum) |
| box, Q3 0.9 / Q1 0.1 | 0.76 [0.57-1.02] | - | 1.06 | 0.98 | 0.94 / 1.04 [0.71-1.45] |
| window Q3 | 0.91 [0.48-1.55] | 0.53-1.98, p 0.50 | 1.01 | 1.25 | 0.96 / 1.00 |
| window Q1 | 0.93 [0.62-1.36] | 0.66-1.62, p 0.44 | 0.91 | 0.91 | 0.81 / 0.72 |
| window, Q3 0.9 / Q1 0.1 | 0.91 [0.52-1.51] | - | 1.00 | 1.22 | 0.95 / 0.97 |
| Js (< 19 cm) box Q3 / Q1 | 0.84 [0.56-1.13] / 0.38 [0.19-0.85] | p 0.40 / 0.020 | 1.17 / 1.16 | | 0.90 / 0.63 |
| Js window Q3 / Q1 | 0.99 / 0.54 [0.38-0.82] | p 0.68 / 0.014 | | | |

- Not distinguishable from random in Q3, the quarter in which saithe take the 0-group (E6): the bootstrap includes
  1 in both regions, permutation p 0.22 (box) / 0.50 (window).
- Only Q1 is significant (box J 0.57, Js 0.38, window Js 0.54): 1-group haddock in winter away from the large
  saithe hauls; little weight for this pair (saithe box presence 0.1 in Q1, ~0 haddock in Q1 stomachs, E6).
- Not a depth gradient inside the class (E_band 1.00-1.17), not position at rectangle scale (E_rect 0.98 in the
  box): no finer recruit or depth table follows.
- In the box the pooled W is mostly between-year covariation (E_year 0.73-0.81 vs pooled 0.76); within-year 0.94
  (plain strata, biased toward W at 2-3 hauls per stratum) / 1.04 LOO (0.71-1.45); the window 0.95 / 0.97.
- The magnitude rides on a few heavy saithe hauls: presence/absence W 1.10 (box) / 1.05 (window); largest saithe
  haul per class dropped 0.81 / 1.14; raw without Walker q 0.68, per swept km2 0.78, saithe >= 40 cm 0.76.
- Catchability noise [D]: independent multiplicative noise leaves the ratio of means unbiased and only widens it
  (the permutation range); a shared haul effect would bias W upward, so W < 1 is if anything conservative.
- W(J) = W(B) x R: box W(B) 1.30 [0.90-1.80] x R 0.59 [0.37-0.98]; window 1.11 x 0.82 [0.49-1.28]; within year
  W(B) 1.25 [0.81-1.72] with W(J) 1.04 LOO (R ~0.83), plain 1.01 / 0.94 (R ~0.93); all CIs include 1. Juveniles vs
  adult benthivores inside the class: W(J vs A) 1.1-1.8 (the engine's J-inside-B direction).
- Scale: a GOV tow is 3.43 km x 20 m (median), ~3 model cells: W measures co-occurrence at 3-4 km.
- The saithe note's 0.086 / 0.296 = class level (now carried by the depth and recruit tables) x a within-class
  remainder (0.53-0.98 cell-weighted without the depth table; 0.76 / 0.91 with the encounter weights); the old
  figure was haul-weighted, haddock < 19 cm only, another extract.

### 2. Can the engine represent it, and does it belong in a parameter? [D/J]

| part of W | field value (box, presence-weighted) | what the engine can do | parameter? |
|---|---|---|---|
| class level (bed classes) | F_class 0.32-0.45 (Q3) | depth_access_table + the decided recruit_depth_table | already carried (decided) |
| depth inside the class | E_band ~1.0-1.17 (none) | finer table rows | no: nothing to carry |
| between-year covariation | E_year 0.73-0.81 (of the pooled 0.76) | none: no interannual saithe level or year-class strength | no: an aggregation effect, not a process |
| within year, B part (saithe at the benthivores' hauls) | 1.01-1.25 (CI to 0.81-1.72) | yes: WC and benthivore policies on 1 km cells | no: position is behaviour (learned) |
| within year, J/B part | 0.83-0.93 (CI includes 1) | no: one J/B per cell, J moves with B | no: not significant; would need J movement, not a factor |

The within-year spatial part is ~1 and an outcome of where the trained water column searches and the trained
benthivores stand or hide; a multiplier on a would pre-empt or double-count it. The between-year part is real in
the field and already inside the stomach-based booking (diet shares average over years); the engine has no
process that makes it, so it is a gate-reading diagnostic only: pooled W 0.76 [0.57-1.02] box, 0.91 [0.52-1.51]
window. The long-run probe's realised diets after retraining test the within-year part.

### 3. Re-booking [F/S/D]

Origin of 0.0833 (split_fields_223, the pre-228 water column; flow t/km2/yr = member Q x its "Bf" share):

| member | origin flow | today's field flow (model's benthivore members) | what changed |
|---|---|---|---|
| saithe | 0.0737 | 0.0738 (J 0.068, A 0.006) | nothing |
| spurdog | 0.0264 (Bf 0.22) | 0.0072 (share 0.06) | the 223 "Bf" pool held small gadoids (poor cod, rocklings), gurnards, rays not in the benthivores' B (E8) |
| horse mackerel + mackerel | 0.0125 | 0 | now the mackerel group, which has no benthivore pair; the 228 split kept their M2 on the rest; horse mackerel's 0.07 came from 20-29 cm fish in the eastern North Sea in 1985, box horse mackerel are 8-16 cm and 87 % krill (E8) |
| anglerfish | 0.0089 (Bf 0.30) | 0.0057 (0.19) | the members-only share |
| ling | 0.0050 (anglerfish proxy) | 0.0017 (0.10 [J]) | |
| hake | 0.0010 | (0.0010; excluded) | out of this pair (user 2026-11-11) |
| velvet belly, pollack | 0.0015 | 0 | no benthivore item in the stomachs |
| total | 0.1291 -> 0.0833 | 0.0884 -> 0.0578 | |

Field route [S/D]: flow 0.0884 t/km2/yr (J 0.0707, A 0.0177; A share 0.20); Monte Carlo 5-95 % 0.063-0.139
(all-lo / all-hi 0.032 / 0.200). Booking = flow / 1.53 = 0.058 (0.041-0.091); 0.0833 sits at the 87th percentile.
Saithe carries 0.83 of the flow; its share 0.05 (0.02-0.09) alone spans 0.029-0.096. Prey side: M2 on J 0.30 /yr
(J 0.237 t/km2), on A 0.014 /yr.

SMS 2023 cross-check [F data of an ICES model, D] (2001-22: saithe Q/B 2.87; haddock 4.1 % of saithe food, J 3.5,
A 0.6; no cod; M2 by saithe on haddock < 27 cm 0.43 /yr at 0.637 t/km2 of saithe):

| SMS reading (saithe) + the other members at their field flow (0.0146) | flow t/km2/yr | booking /yr |
|---|---|---|
| predator side (haddock eaten per saithe tonne x box saithe 0.492) | 0.0725 | 0.047 |
| prey side (partial M2 x box/North Sea saithe density 0.77 x box haddock J, A) | 0.0990 | 0.065 |
| the same at the stomach level (SMS overshot 1980-91 stomachs x0.45) | 0.041 / 0.053 | 0.027 / 0.034 |

The prey side is the more box-like reading (the box holds 0.42 t of haddock J per saithe tonne vs 0.25 in the North
Sea). The field route's M2 on J (0.30) lands on SMS's 0.29; the field J share of the flow 0.80 (MC 0.71-0.87)
matches the gate's encounter J share 0.76.

Supported booking [F/S/D; moderate]: 0.058 /yr (0.04-0.09); flow 0.088 t/km2/yr, J share 0.80; SMS 0.047-0.065
(0.027-0.034 at the stomach level). 0.0833 carries 0.0125 t/km2/yr from members that left the group and ~0.025 from
broad-pool shares (spurdog, anglerfish, ling proxy) on prey outside the benthivores' B.

### 4. What a re-booking moves [F code, D]

- Derived g: unchanged for every group across bookings 0.0266-0.1308 (benthivores g 0.000550 = P 0.74 / (0.922 x
  1460); the water column pays its own P 0.42).
- Moves: the benthivores' booked M2 sum 0.2341 -> 0.2086 (in the engine used only by `habitat.predation_m2` for
  `spawn: {mode: equilibrium}`, not the benthivores' colony spawn); budget_gate's booked shares, prey-side check and
  total-ration row (the water column's booked ration 2.67 -> 2.61 t/t/yr vs field Q/B 2.83, inside 2.6-3.6);
  probe_diets' and ideal_growth's booked columns; the FG editor's matrix (not its growth display).
- Tests: none pins this pair's predation_mortality or the benthivore M2 sum (grepped 0.0833, 0.234, 0.2086, the
  pair name, predation_m2; test_half_saturation, test_growth_budget, test_open_pool, test_pair_shares,
  test_appetite_sda checked and unaffected).

### 5. Gate [D]

All runs on the decided central (combined factors, hake out, stage p, adult profile; a 0.117, stage_access
[0.3818, 0.01653]; J on the recruit_depth_table, A on the GOV annual shape). WC -> benthivores implied / booked M2
at equal cells / field spread / P2 (predator uniform, prey shaped); benthivore prey total equal / field (P2);
* = OFF (band 3); equal-cell values in brackets are not like-for-like for W runs.

| run | booking | f_J | eq / field / P2 | ben total eq / field (P2) |
|---|---|---|---|---|
| control (= gate_adult central) | 0.0833 | 0.3818 | 1.54 / 1.41 / 2.13 | 0.89 / 0.87 (1.13) |
| re-booked, field central | 0.0578 | 0.3818 | 2.17 / 1.98 / 3.00 | 0.99 / 0.96 (1.24) |
| re-booked, MC 95 % | 0.0910 | 0.3818 | 1.42 / 1.30 / 1.96 | 0.87 / 0.85 (1.09) |
| re-booked, SMS prey side | 0.0647 | 0.3818 | 1.95 / 1.78 / 2.69 | 0.96 / 0.94 (1.21) |
| re-booked, MC 5 % | 0.0414 | 0.3818 | 2.98 / 2.72 / 4.12* | 1.06 / 1.04 (1.33) |
| re-booked, SMS stomach level | 0.0266 | 0.3818 | 4.57* / 4.18* / 6.32* | 1.14 / 1.11 (1.43) |
| diagnostic W box 0.76 | 0.0833 | x0.76 | (1.27) / 1.17 / 1.76 | 0.80 / 0.79 (0.99) |
| diagnostic W box 0.76, re-booked | 0.0578 | x0.76 | (1.79) / 1.65 / 2.47 | 0.88 / 0.87 (1.10) |
| diagnostic W box lo 0.57, re-booked | 0.0578 | x0.57 | (1.48) / 1.38 / 2.05 | 0.80 / 0.80 (0.98) |
| diagnostic W box hi 1.02, re-booked | 0.0578 | x1.02 | (2.21) / 2.01 / 3.05* | 1.00 / 0.97 (1.26) |
| diagnostic W window 0.91, re-booked | 0.0578 | x0.91 | (2.03) / 1.86 / 2.80 | 0.95 / 0.93 (1.19) |
| diagnostic W within-year 1.04, re-booked | 0.0578 | x1.04 | (2.23) / 2.03 / 3.08* | 1.01 / 0.98 (1.26) |

- The encounter-implied flow at the field spread is ~0.18 t/km2/yr with either booking (1.41 x 0.127 = 1.98 x
  0.088), about twice the field flow 0.088: the re-booking moves the ratio, not the flow. At the field-central
  booking P2 sits on the band (3.00); lower re-bookings go OFF. The benthivore prey total moves to 0.99 / 0.96
  because the booked total falls; the implied total stays ~0.21.
- Unchanged in every run: planktivores 0.86-0.92, krill 1.01-1.03, whiting 0.99-1.07 / 0.88-0.92, epifauna 0.92 /
  0.89, Nephrops 2.14, infauna 1.01; whiting -> benthivores 0.29 / 0.34, seals 0.90, porpoises 2.44; the water
  column's other diet rows at the field spread (krill 1.03-1.06, planktivores 0.90-0.93, whiting 0.99-1.02).
- OFF pairs as before (WC -> mackerel prey side, seals -> WC prey side, whiting -> benthivores at equal cells and
  prey side, whiting -> epifauna at the field spread), plus WC -> benthivores in the runs marked *.

## If applied (not applied; the user's call)

- `predatory_fish_preys_on_benthivorous_fish`: predation_mortality 0.0833 -> 0.058 (0.04-0.09). Comment: the
  members' field diets on the model's benthivore members x Q/B x B (saithe 0.074, spurdog 0.007, anglerfish 0.006,
  ling 0.002 t/km2/yr; hake out), J share 0.80; SMS 2023 at the box's saithe density 0.047-0.065; the old value was
  the 223 water column's "Bf" share, including horse mackerel and mackerel (now another group) and prey outside the
  benthivores' B. No derived g changes and no test needs an update. The water column's booked ration falls to
  2.61 t/t/yr (field 2.83); whether the 0.039 t/km2/yr is re-booked on its other prey is a separate question (its
  other shares were set from their own evidence, and 0.0125 of it was never this group's food).
- Within-class W: no parameter. The pooled W (0.76, 0.57-1.02) could be reported beside budget_gate's WC ->
  benthivores row as a diagnostic band.

## Gaps

1. Within-year co-occurrence in the box rests on 2-3 hauls per (year, class) stratum (150-250 m in Q1: 1); plain
   and LOO bracket it (0.94 / 1.04) with wide CIs; the window is better sampled but not the box.
2. A tow integrates 3.4 km: co-occurrence inside a 1 km cell is unmeasured; GOV samples only the bottom 4-5 m
   (saithe Hs 15 m).
3. All Q3 GOV hauls are day hauls; night co-occurrence unmeasured.
4. Saithe's diet share (0.05, 0.02-0.09) rests on pre-2000 NW North Sea stomachs plus Mehl 2005 (Norway); no
   Skagerrak saithe diet with numbers; it carries 0.83 of the field flow and most of the range.
5. SMS: the North Sea area (575 000 km2) and the 27 cm J/A cut on Lsea are judgements; SMS overshot the 1980-91
   stomachs (x0.45), so its readings are a bracket.
6. Ling's share, A fraction and Q/B are judgements; spurdog's and anglerfish's rest on Ecopath 1991 and few
   stomachs.
7. W(B) x R is not resolved within year (CIs include 1); juvenile avoidance of saithe inside a haul-scale patch
   cannot be tested with GOV.

## Verdict

- Within-class co-occurrence [F data / D; moderate for box Q3, low for the within-year split]: 0.76 (0.57-1.02) in
  the box, 0.91 (0.52-1.51) in the window, saithe presence-weighted; in Q3, when saithe eat the 0-group, not
  distinguishable from random (p 0.22 / 0.50); not a depth gradient; rides on a few heavy saithe hauls; in the box
  mostly between-year covariation (within-year 0.94-1.04). Only Q1 is significant, when saithe are largely absent
  and not eating haddock. The class-level part of the old haul-level F is carried by the decided depth and recruit
  tables.
- It belongs in no parameter [J, project rules]: the within-year part is learned position on 1 km cells and ~1;
  the between-year part has no engine process and is averaged in the stomach-based booking. Gate diagnostic only:
  field spread with W 0.76 and the re-booking 1.65 (1.38-2.01).
- Booking [F/S/D; moderate]: 0.058 /yr (0.04-0.09); flow 0.088 t/km2/yr, 0.80 on J (M2 on J 0.30 /yr vs SMS 0.29);
  SMS at the box's saithe density 0.047-0.065; 0.0833 at the 87th percentile, booking 0.0125 t/km2/yr for members
  now in the mackerel group and ~0.025 for broad-pool prey outside the benthivores' B. No derived g or test moves.
- Gate [D]: with the re-booking the decided central reads 2.17 (equal cells) / 1.98 (field spread) / 3.00 (P2, on
  the band); benthivore prey total 0.99 / 0.96. The encounter-implied flow (~0.18 t/km2/yr) is unchanged and about
  twice the field flow: the pair's field-spread excess is ~2.0x, neither a booking artefact nor a within-class
  co-occurrence effect. What remains is the J encounter itself (the saithe a on J at the bed, 0.80 of the flow);
  the long-run probe's realised diet after retraining is the next check, as the user decided.

## User decisions

- 2026-11-11: re-book water column -> benthivores 0.0833 -> 0.058 in section 270;
  within-class W is no parameter.
