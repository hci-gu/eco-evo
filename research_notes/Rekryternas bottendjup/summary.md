# Bed depth of the benthivores' recruits and juveniles (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library, code, tests, manifest,
AGENTS.md, resume or existing notes; no training; library copies only in the session scratchpad; budget_gate.py
and field_diet.py were patched in memory only. Every shape is read from NS-IBTS GOV catches (field data), never
from model output; the gate is a check, nothing was chosen to move it. Sources E1-E10 and model facts M1-M5:
`evidence.md`. Grades: [F] field data or full text, [A] abstract only, [S] secondary / via another note,
[D] derived here, [J] judgement.

## Files

Run from `scripts/`; each script's stdout is saved beside it.

- `flatfish_age1_length.py` -> `flatfish_age1_length_stdout.txt`: length at age 1 in Q3 from DATRAS CA 2021-25
  (the stage cut); reproduces the library's 27.0 / 27.1 / 16.0 / 14.8 cm; dab, long rough dab and lemon sole have
  no age-1 readings.
- `gov_juv_bed.py` -> `gov_juv_bed_stdout.txt`, `../tables/gov_juv_bed.json`, `../tables/gov_juv_bed_shapes.csv`:
  J, J < 19 cm, A, B and recruit shapes by bed class, box and window, Q1 / Q3 / annual, member-weighted, Walker q,
  per hour and per swept km2, 1000 year-resamples; J/B by class; member shares of J; F against the predators'
  field spreads.
- `gov_q2q4_check.py` -> `gov_q2q4_check_stdout.txt`: haddock + cod J and A shapes in Q2 (May) and Q4 (Nov),
  1991-97.
- `gate_ben_shape.py` -> `gate_ben_shape_stdout.txt`, `../tables/gates/budget_gate_<run>.txt`,
  `../tables/gates/gate_ben_shape.json`: budget_gate on library copies (combined capture-p x schooling factors,
  central) with the benthivores placed by stage. Controls: `library` and `combined` are byte-identical to
  research_notes/"Sejens hojd over botten" budget_gate_library / budget_gate_combined_central.
- `decompose_wc_pair.py` -> `decompose_wc_pair_stdout.txt`: the class-level co-occurrence factor with and
  without the pair's depth_access_table.
- `../tables/recruit_depth_table_proposal.csv`: the proposed table and its two brackets.

```
R=<repo>/research_notes; S=<session scratchpad>
python3 -I flatfish_age1_length.py "$R/Torskfiskarnas artsammansattning/data/datras"          # a-umlaut in the real name
python3 -I gov_juv_bed.py "$R/Fiskgrupper i rutan/data/extract" "$R/Torskfiskarnas artsammansattning/data/extract" \
    "$R/Fiskgrupper i rutan/tables/length_weight_all.csv" <repo>/maps/depth.npz ../tables      # ~4 min (NBOOT=1000)
python3 -I gov_q2q4_check.py "$R/Sejens djuplage/data/gov_q2q4_1991_97" <repo>/maps/depth.npz # a-umlaut
python3 gate_ben_shape.py <repo> ../tables/gov_juv_bed.json $S/libs ../tables/gates             # ~10 s
python3 -I decompose_wc_pair.py <repo>/fgconfig/fg_library.yaml ../tables/gov_juv_bed.json
```

## Question

The saithe note found the water column -> juvenile benthivores pair at ~2x its booking and traced it to the
juveniles' horizontal position: the engine imports recruits uniformly and the gate places the benthivores
uniformly, while in-gape haddock sit over shallow beds. The user chose to give the benthivores' `recruit_import`
a bed-depth shape from GOV over all members, and the same shape in budget_gate, after investigating the shape.
What is the shape, does it need a season, what does the gate say with it, and what would the key look like?

## Model (what the shape acts on)

- J = the fish below the species' length at age 1 in Q3, J rides inside B, recruits (haddock age 0 in Q3, cod
  age 1 in Q1, SMS x box share, 0.39 / 0.61 of the import tonnes) enter J uniformly over the cells in Jun-Sep at
  the month's fill; graduation 1.02 /yr, emigration e_J 1.1 /yr, so a J tonne stays ~4-5 months (M1-M3).
- J moves with B: every movement fraction of a cell carries its J along (stages.py). A recruit table therefore
  sets where recruits ARRIVE; where J then sits depends on the adults' learned moves.

## Findings

### 1. J definition and member shares [F data, D]

- The library cuts are reproduced from CA 2021-25 in the Skagerrak rectangles: haddock 27.0 cm (465 aged fish),
  cod 27.1 (822), plaice 16.0 (250), witch 14.8 (25). Dab, long rough dab and lemon sole have no age-1 Q3
  readings in the whole NS-IBTS CA 2021-25 (5 / 0 / 3 fish): no field cut; they count as adult (the J/B 0.15
  convention). A 12 cm proxy cut [J] is carried as a sensitivity.
- Member shares of J biomass (member weight x the species' own J/B, box, Walker q):

  | J definition | Q1 | Q3 | annual |
  |---|---|---|---|
  | library (haddock, cod, plaice, witch) | haddock 0.93 (0.86-0.97), cod 0.07, plaice 0.003 | haddock 0.81 (0.71-0.89), cod 0.19 (0.11-0.30) | haddock 0.87, cod 0.13, plaice 0.002 |
  | + dab, long rough dab, lemon sole < 12 cm [J] | haddock 0.91, cod 0.07, LRD 0.02 | haddock 0.80, cod 0.19, LRD 0.01 | haddock 0.85, cod 0.13, LRD 0.016 |

  The flatfish are < 2.5 % of J by any definition, so "all members" and "haddock + cod" give the same J shape to
  three decimals. The import tonnes are weighted differently (cod 0.61): the SMS import counts cod at age 1 in
  Q1, while the GOV J biomass is mostly haddock.
- Box J/B from these data (pooled haul means on the model map's cell shares): Q1 0.18 (0.12-0.28), Q3 0.23
  (0.14-0.32) with Walker q; raw 0.16 / 0.15. The library's 0.15 / 0.16 (depth-stratified, another box polygon)
  is inside the range. J/B by class, Q3: 0.375 / 0.137 / 0.103 / 0.001 - the juvenile share itself falls
  with depth.

### 2. GOV bed-class shapes [F data, D]

Shape f per class (< 100 / 100-150 / 150-250 / >= 250 m), equal-cell mean 1 on the model's 60 x 60 depth map
(cell shares 0.232 / 0.299 / 0.324 / 0.146); 95 % year-block (whole-year) bootstrap range in brackets;
member-weighted, Walker q, per hour. "share" = f x cell share.

| quantity (box) | f | 95 % range | class shares |
|---|---|---|---|
| J, Q1 | 2.35 / 0.80 / 0.56 / 0.25 | 1.44-3.07 / 0.45-1.35 / 0.26-1.02 / 0.08-0.52 | 0.55 / 0.24 / 0.18 / 0.04 |
| J, Q3 | 2.92 / 0.87 / 0.20 / 0.00 | 2.41-3.28 / 0.61-1.27 / 0.10-0.30 / 0-0 | 0.68 / 0.26 / 0.06 / 0 |
| J, annual (Q1/Q3 mean) | 2.64 / 0.83 / 0.38 / 0.13 | 2.18-2.96 / 0.63-1.13 / 0.22-0.60 / 0.04-0.26 | 0.61 / 0.25 / 0.12 / 0.02 |
| J < 19 cm (in-gape part), annual | 2.97 / 0.59 / 0.36 / 0.12 | 2.43-3.36 / 0.41-0.97 / 0.17-0.63 / 0.03-0.26 | 0.69 / 0.18 / 0.12 / 0.02 |
| A, Q1 | 1.24 / 1.35 / 0.68 / 0.61 | | 0.29 / 0.41 / 0.22 / 0.09 |
| A, Q3 | 1.42 / 1.60 / 0.50 / 0.21 | | 0.33 / 0.48 / 0.16 / 0.03 |
| A, annual | 1.33 / 1.48 / 0.59 / 0.41 | 1.09-1.60 / 1.28-1.66 / 0.51-0.68 / 0.33-0.50 | 0.31 / 0.44 / 0.19 / 0.06 |
| B, annual | 1.60 / 1.34 / 0.54 / 0.36 | 1.29-1.90 / 1.15-1.55 / 0.47-0.62 / 0.28-0.46 | 0.37 / 0.40 / 0.18 / 0.05 |
| recruits: haddock age 0, Q3 (< 20 cm) | 3.36 / 0.53 / 0.20 / 0.00 | 2.97-3.58 / 0.39-0.78 / 0.06-0.38 / 0-0 | 0.78 / 0.16 / 0.06 / 0 |
| recruits: cod age 0, Q3 (< 20 cm) | 3.27 / 0.63 / 0.16 / 0.00 | 2.29-3.67 / 0.43-1.22 / 0.06-0.39 / 0-0.01 | 0.76 / 0.19 / 0.05 / 0 |
| recruits: cod age 1, Q1 (10-27 cm) | 2.29 / 1.30 / 0.21 / 0.09 | 1.83-2.66 / 1.06-1.55 / 0.10-0.37 / 0.04-0.15 | 0.53 / 0.39 / 0.07 / 0.01 |
| **import-weighted (had0 Q3 0.39 + cod1 Q1 0.61)** | **2.71 / 1.00 / 0.21 / 0.05** | 2.42-2.93 / 0.85-1.17 / 0.12-0.32 / 0.03-0.09 | 0.63 / 0.30 / 0.07 / 0.01 |

- Robust within the box [D]: raw (no Walker q) J annual 2.48 / 0.97 / 0.37 / 0.12 (the smallest fish, with the
  lowest q, sit shallowest, so q sharpens the shape); per swept km2 2.74 / 0.79 / 0.34 / 0.12 (wing spread rises
  20 -> 24 m with depth); day-only Q3 identical (every Q3 haul is a day haul); >= 250 m by neighbour copy
  2.59 / 0.82 / 0.37 / 0.27; the 12 cm flatfish proxy and haddock + cod only identical to three decimals.
- The window (57.3-59.2 N, 8-12 E) is flatter: J annual 1.50 / 1.66 / 0.42 / 0.13, Q3 1.54 / 1.88 / 0.24 / 0;
  its 100-150 m beds hold more juveniles than the box's (shelf edges outside the box [J]).
  Both agree on the decisive contrast: J are scarce over 150-250 m and nearly absent over >= 250 m.
- Season [F data, S]: Q2 1991-97 (May, before settlement, gadoids only) box J 2.74 / 0.72 / 0.35 / 0.24, Q4
  (Nov) 3.28 / 0.78 / 0.01 / 0.01 (few hauls: 4 / 8 / 6 / 0). The box's J is over < 100 m beds in every quarter
  (0.55-0.76 of J); in winter (Q1) part of it moves to 150-250 m (f 0.56 vs 0.20 in Q3), as Albert 1994 found on
  the Trench slope ("all age-groups ... deeper in winter", E5). Albert sampled only 80-300 m, where the 0-group
  summer occurrence falls from 40 % (80-149 m) to 21 % (200-250 m).
- Settlement timing [S] (E6): haddock settle mid-July to mid-August (pelagic 4 cm in June, demersal 12 cm in
  August), cod 0-group in July; the library's recruitment months (Jun 0.2, Jul 0.5, Aug 0.5, Sep 0.2) fit
  haddock. Cod age 0 are almost absent from the box in Q3 (0.3-0.5 % of the stock's age 0 against 3.7-5.2 % of
  its age 1 in Q1, M2): most cod recruits arrive in autumn-winter, from the inner Skagerrak front where Munk et
  al. 1995 found cod larvae and juveniles along the 80-140 m isobaths (E7) - the same shallow classes.
- Catchability [S/J] (E8): no depth-dependent GOV q is published; Walker q is by length only. In Q1 Norway and
  Sweden lengthen the sweeps below 70 m, which herds more large roundfish at depth; Q3 uses 60 m sweeps
  everywhere. The Q1 A shape may therefore be biased deep, the J shape little (small fish are herded weakly);
  the Q3 shape is the cleaner arrival estimate. GOV standard stations stop at 250 m: the >= 250 m class (0.146
  of the model's cells, beds to 529 m) is extrapolated from the window (24 Q1, 13 Q3 hauls).

### 3. One table, the arrival shape, no season [D/J]

- The import is non-zero only in Jun-Sep, so a single arrival table already acts in one season; a monthly table
  would only reshape arrivals that do not happen. The Q1 deepening is the juveniles' own winter movement, which
  the engine cannot do separately from the adults (J moves with B): that is a structural limit, not a reason for
  a seasonal import shape.
- Central [J]: the import-weighted shape, 2.71 / 1.00 / 0.21 / 0.05 - each recruit component at the tonnes and
  the survey state at which M2 counts it (haddock age 0 in August, cod age 1 in January-February, since the box
  holds almost no cod 0-group in August). Brackets: Q3 J (2.92 / 0.87 / 0.20 / 0) and haddock 0-group alone
  (3.36 / 0.53 / 0.20 / 0). The >= 250 m value 0.05 (not 0) comes from the cod component; recruits are absent
  from the 13 window Q3 hauls on >= 250 m beds (3 of them in the box).
- Graduation and adults: no shape. Graduation is internal to B (J <- J exp(-gamma) in place); adults move by their
  learned policy (behaviour must be learned). The adults' FIELD bed shape (A annual 1.33 / 1.48 / 0.59 / 0.41;
  class shares 0.31 / 0.44 / 0.19 / 0.06) is a gate input only: it is where the field puts the A that predators
  meet, for a like-for-like gate.

### 4. Gate [D]

budget_gate (16 x 16 grid; cell shares 0.203 / 0.289 / 0.340 / 0.168) on library copies with the combined
capture-p x schooling factors (central, E10). Benthivores as prey at density x [js f_J shape_J + (1 - js) f_A
shape_A] per cell (field spread; shapes renormalised on the gate grid as the loader treats depth tables). Ratios
are encounter share / booked share (predator side) and implied / booked M2 (prey side); equal cells | field
spread; * = OFF.

| run | WC -> ben | whiting -> ben | seals -> ben | porpoises -> ben | ben prey total eq / field |
|---|---|---|---|---|---|
| library (no factors) | 0.97 / 1.13 | 0.12* / 0.07* | 0.90 / - | 2.44 / - | 0.62 / 0.65 |
| library + J/A shapes | 0.97 / 0.95 | 0.12* / 0.14* | 0.90 | 2.44 | 0.62 / 0.61 |
| combined (control, = saithe note) | 2.00 / 2.33 | 0.29* / 0.17* | 0.90 | 2.44 | 1.06 / 1.12 |
| combined, A shape only (J uniform) | 2.00 / 2.27 | 0.29* / 0.21* | 0.90 | 2.44 | 1.06 / 1.12 |
| **combined + J annual + A annual** | 2.00 / **1.98** | 0.29* / 0.33* | 0.90 | 2.44 | 1.06 / 1.07 |
| combined + J Q3 + A Q3 | 2.00 / 1.87 | 0.29* / 0.37 | 0.90 | 2.44 | 1.06 / 1.05 |
| combined + J Q1 + A Q1 | 2.00 / 2.08 | 0.29* / 0.30* | 0.90 | 2.44 | 1.06 / 1.09 |
| combined + J import shape (proposal) | 2.00 / 1.90 | 0.29* / 0.34 | 0.90 | 2.44 | 1.06 / 1.05 |
| combined + J haddock 0-group | 2.00 / 1.83 | 0.29* / 0.39 | 0.90 | 2.44 | 1.06 / 1.04 |
| combined + J / A window shapes | 2.00 / 2.16 | 0.29* / 0.23* | 0.90 | 2.44 | 1.06 / 1.09 |
| combined + J raw / per swept km2 / >= 250 copy | 2.00 / 1.99 / 1.95 / 1.96 | 0.29* / 0.32* / 0.34 / 0.33* | 0.90 | 2.44 | 1.06 / 1.06-1.07 |
| combined + J < 19 cm only (diagnostic) | 2.00 / 1.94 | 0.29* / 0.35 | 0.90 | 2.44 | 1.06 / 1.07 |
| combined + J annual, predator uniform (P2) | **2.59** / 1.98 | 0.33* / 0.33* | 0.90 | 2.44 | 1.29 / 1.07 |
| combined + J import shape, predator uniform (P2) | **2.66** / 1.90 | 0.34 / 0.34 | 0.90 | 2.44 | 1.31 / 1.05 |

- The J shape lowers WC -> benthivores at the field spread from 2.33 to 1.83-2.16 (central 1.90-1.98), not to
  the 1.2-1.3 the saithe note estimated. `decompose_wc_pair.py`: the pair's depth_access_table (mean-1 values
  1.63 / 1.34 / 0.55 / 0.30) already puts the encounter on shallow beds; the saithe field spread x the table
  weights the classes 0.10 / 0.25 / 0.63 / 0.02 (saithe alone 0.04 / 0.13 / 0.78 / 0.05). The class-level F
  is 0.52 without the table and 0.72 with it (import shape 0.41 / 0.66; J < 19 cm 0.49 / 0.68); with the A part
  (F_eff 0.88, 0.35 of the pair's encounter) the pair falls x0.77, and the share renormalisation gives 1.98. The
  saithe note's F (0.19-0.55) did not carry the depth table and is superseded for this pair.
- The equal-cell reading with the prey shaped and the predator uniform (P2) RISES to 2.59-2.66: a water column
  spread evenly meets the shallow juveniles exactly where its depth table is largest. In the engine the trained
  WC policy decides which reading applies.
- Whiting -> benthivores roughly doubles at the field spread (0.17 -> 0.33-0.39; F_eff 2.25-2.48, whiting 0.77
  on < 100 m beds) and passes the predator-side band in six runs, but stays OFF on the prey side in every run
  (0.29-0.39 of its booked M2 0.1067; stage window [0.038, 0.0064]).
- Seals and porpoises do not move (0.90, 2.44): in the gate they are uniform, Type II, without a depth table,
  so a prey shape cancels. In the engine their answer depends on their learned positions.
- The benthivores' prey-side total stays near its booking (1.04-1.09 field, 1.06 equal cells; 1.29-1.31 under
  P2).
- Other prey (combined + shapes): planktivores 0.89 / 0.86, krill 1.01 / 1.02, whiting 1.03 / 0.88 (equal /
  field), unchanged by the J/A shapes. Adding the benthivores as a PREDATOR at their field B spread (class shares
  0.370 / 0.401 / 0.176 / 0.052; a separate patch, P3) moves the field-spread totals: whiting 0.88 -> 1.37
  (benthivores -> whiting share 1.64), krill 1.02 -> 1.03 (benthivores -> krill share 1.31), planktivores 0.86 ->
  0.92, Nephrops 2.14 -> 2.05, epifauna 0.89 -> 0.85; no new OFF pair. This is the benthivores' own spread, not
  the J effect.
- OFF in every combined run as before: WC -> mackerel (prey side), seals -> WC (prey side), whiting -> benthivores
  (equal cells and prey side), whiting -> mobile_epifauna (field). The J shape adds no OFF pair and removes
  whiting -> benthivores from the field-spread diet OFF list in the Q3, import, haddock 0-group, swept-km2, J < 19
  and P2-import runs.

## Proposal (not applied)

Library key, in the benthivores' `juvenile_stage` block (rows as the pairs' depth_access_table; the loader
renormalises to an equal-cell mean 1, as `open_pool.depth_weight_map`; values here are already mean 1 on the
60 x 60 map):

```yaml
      # Bed-depth shape of the recruit import (equal-cell mean 1): NS-IBTS GOV
      # 2000-25 in the box, haddock age 0 in Q3 (0.39 of the import tonnes)
      # and cod age 1 in Q1 (0.61), Walker q; 95 % year-bootstrap
      # 2.42-2.93 / 0.85-1.17 / 0.12-0.32 / 0.03-0.09. >= 250 m from the
      # window (GOV stations stop at 250 m). research_notes/"Rekryternas
      # bottendjup" summary.md.
      recruit_depth_table:
        - {depth_m: 99.9, weight: 2.71}
        - {depth_m: 100.0, weight: 1.00}
        - {depth_m: 149.9, weight: 1.00}
        - {depth_m: 150.0, weight: 0.21}
        - {depth_m: 249.9, weight: 0.21}
        - {depth_m: 250.0, weight: 0.05}
```

Brackets for a sensitivity run: Q3 J 2.92 / 0.87 / 0.20 / 0.00; haddock 0-group 3.36 / 0.53 / 0.20 / 0.00.

Code paths:
- `lib/environments/ecosystem_env/stages.py`: `settings` parses the key with `open_pool._parse_rows(rows,
  "weight", ...)`; `_open_population_tables` stores `env.stage_import_map[j]` = `open_pool.depth_weight_map(table,
  env depth map, (H, W))` (ones without a table or without a depth map, so every test world and the MVP keep the
  uniform import); `exchange` multiplies `per_cell` by the map and books `imported` as `per_cell x map.sum()`
  (= I dt x cells, unchanged), the arriving reserve likewise.
- `lib/gpu/ecosystem.py`: `stage_import` gets a cell dimension (or a separate [1, N_all, C] map tensor multiplied
  in `_stage_exchange`); the record's `inn = (imported * self.C)[..., 0]` assumes a uniform import and must
  become the sum over cells.
- `fgconfig/fgconfig.py`: the FG editor rebuilds the `juvenile_stage` block from its fields (lines ~4985-5020),
  so an unknown key would be DROPPED on save; it needs a field (as `outside_depth_table`) or a pass-through.
- `lib/diagnostics/field_diet.py` and `tools/probes/budget_gate.py`: a stage-aware `field_prey_shape` - J on the
  recruit table (one source of truth, as an open pool's `outside_weight_map`), A on a field constant
  `FIELD_STAGE_BED_SHARES['benthivorous_fish']['A'] = (0.308, 0.441, 0.191, 0.060)` (GOV box, annual), each
  weighted by js f_J and (1 - js) f_A per pair; the equal-cell gate unchanged (or reported with P2). The annual
  field J (2.64 / 0.83 / 0.38 / 0.13) instead of the recruit table gives 1.98 vs 1.90: not load-bearing.
  Optional: the benthivores as a predator at their field B shares (0.370 / 0.401 / 0.176 / 0.052).
- `tests/test_open_population.py`: no depth map -> uniform import (today's numbers); with a map -> the per-cell
  import sums to I dt x cells, the ledger's import total is unchanged, CPU and GPU equal. Optional: `initialise`
  could start J on the same shape (clipped to B); left out here - the start J relaxes within ~5 months.

What else moves (engine; judge with the long-run probe after retraining, not the gate):
- The benthivores' own dynamics: J/B becomes a gradient (field Q3 0.38 / 0.14 / 0.10 / 0 by class), so the
  per-cell growth split kappa J / (kappa J + A), J's share of bottom-trawl losses (s_J 0.15) where effort sits,
  and the J observation channel change; the recruits land where the infauna K is highest (132 t/km2 at 100 m).
  The import total and the reward ledger are unchanged. Checkpoint shapes stay valid; the policies were trained
  with uniform recruits, so retrain before judging.
- WC -> benthivores: down if the WC stays over 150-250 m beds (field-spread reading 1.9), up if it spreads
  (2.6-2.7).
- Whiting -> benthivores: up ~2x where whiting stand shallow, still under its booking.
- Seals / porpoises -> benthivores: f_J 1, so their J take follows where they forage; unchanged in the gate.
- The adults: unchanged by the key (policy), but they carry the shallow J wherever they move it.

## Gaps

1. The engine can only place arrivals: J moves with B, so the field winter deepening and any J persistence on
   shallow beds depend on the adults' policy. Whether the engine's J stays shallow is a long-run-probe question.
2. Within-class co-occurrence is not representable at class level: haul-level F (saithe note: 0.086 box, 0.30
   window) is lower than class-level F; it is a candidate for the residual WC -> benthivores overshoot, not a value
   to fit.
3. >= 250 m beds (0.146 of the cells) are outside GOV standard stations; the class is extrapolated (0.05; 0.00-0.27
   across variants).
4. No depth-dependent GOV catchability is published; Q1 sweep length varies with depth (Norway, Sweden), so the
   Q1 shapes carry a possible herding bias; Walker q for haddock < 11 cm (0.012) is the least certain value and
   moves the shape (raw vs q: 2.48 vs 2.64 in < 100 m).
5. Dab, long rough dab, lemon sole have no field length at age 1 (no CA ages); their J is < 2.5 % of J under any
   proxy.
6. The box rectangle's GOV densities per class are applied to the model map's classes; the 1991-97 Q2/Q4 check
   rests on 18-48 box hauls.
7. The import mix (cod 0.61 of the tonnes) comes from SMS x survey shares, while GOV J biomass is 0.87 haddock;
   the arrival shape follows the import's composition [J].

## Verdict

- Shape [F data / D; high for the contrast, moderate for the values]: the benthivores' juveniles and recruits sit
  over < 100 m beds in every quarter and survey period (0.55-0.78 of J biomass on 0.23 of the cells), are scarce
  over 150-250 m and nearly absent over >= 250 m. Members do not matter (haddock 0.87, cod 0.13 of J; flatfish
  < 0.025). Proposed `recruit_depth_table` 2.71 / 1.00 / 0.21 / 0.05 (import-weighted [J]; brackets Q3 J and
  haddock 0-group). One table: the import is seasonal already; the winter deepening is juvenile movement the
  engine cannot represent separately.
- Effect [D]: the shape is a correct physical input, but it does NOT close the water column overshoot. At the
  field spread WC -> benthivores goes 2.33 -> 1.83-2.16 (central ~1.9), because the pair's depth_access_table
  already weights shallow beds; the saithe note's 1.2-1.3 estimate (class F without the depth table) is
  superseded. With the predator uniform the pair rises to 2.6-2.7, so in the engine the outcome depends on where
  the trained water column stands. Whiting -> benthivores roughly doubles and stays short of its booking; seals
  and porpoises unchanged in the gate.
- What remains open for this pair: within-class (haul-level) co-occurrence and the adult part of the flow; the
  booking itself is already at the top of the field range (saithe note). Applying the key is the user's call; if
  applied, budget_gate's field spread should use it (J) with the GOV A shares, and the long-run probe after
  retraining is the check.

## User decisions

- 2026-11-11: the recruit_depth_table (2.71 / 1.00 / 0.21 / 0.05) and the gate's
  stage field shapes (J on the table, A on the GOV annual shares) go into the
  combined section; the adult part of the water column -> benthivores flow
  (hake, anglerfish, ling on adult benthivores) is investigated against those
  members' field diets.
