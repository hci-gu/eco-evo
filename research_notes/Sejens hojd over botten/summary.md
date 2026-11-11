# Saithe height above the bed, and the water column -> juvenile benthivores pair (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library, code, tests, manifest,
AGENTS.md, resume or existing notes; no training; library copies only in the session scratchpad. Hs and every
co-occurrence factor are fitted to FIELD catches (HERAS, GOV), never to model output; nothing was chosen to hit
a gate value of 1. Sources E1-E14 and model facts M1-M3: `evidence.md`. Grades: [F] field data or full text,
[L] lab, [A] abstract only, [S] secondary / read via another note, [D] derived here, [J] judgement.

## Files

Run from `scripts/`; each script's stdout is saved beside it.

- `heras_saithe_height.py` -> `heras_saithe_height_stdout.txt`: adult saithe by fished height above the bed in
  the HERAS/IESSNS hauls (54.6-59.1 N, 5-12.4 E, June-July); exp, uniform and hump profiles, with and without the
  GOV bed-class density offset.
- `neilson_hs.py` -> `neilson_hs_stdout.txt`: Hs implied by Neilson et al. 2003 and DFO 2003/075.
- `band_hs.py` -> `band_hs_stdout.txt`, `../tables/band_hs.json`: the WC fish pairs and WC -> krill on the band
  overlap + rule A with the saithe profile and the juvenile layer varied; execs research_notes/"Overlapp i
  fiskparen"/scripts/band_pairs.py unchanged and reproduces section 269 exactly (2.904 / 7.604 / 2.764 / 0.155;
  krill 0.01169).
- `window_hs.py` -> `window_hs_stdout.txt`, `../tables/window_hs.json`: the stage window re-weighted per variant,
  saithe identity gadoids (library) or haddock only; a modified copy of research_notes/"Bottenfiskens
  hojdprofil"/scripts/wc_benth_window.py.
- `gate_hs.py` -> `gate_hs_stdout.txt`, `../tables/gates/budget_gate_<run>.txt`, `../tables/gates/gate_hs.json`:
  tools/probes/budget_gate.py unchanged on library copies, each variant also carrying the combined capture-p x
  schooling factors (research_notes/"Fangst och stim i fiskparen", central). `library` is byte-identical to
  today's gate; `repro_central` gives 2.01 / 2.34 vs the combined note's 2.00 / 2.33 (rounding: 4 vs 6
  significant figures in the edits).
- `gov_juv_overlap.py` -> `gov_juv_overlap_stdout.txt`: adult saithe vs in-gape haddock (< 19 cm) by bed depth in
  NS-IBTS GOV, box and a Skagerrak/Trench window.
- `prey_side.py` -> `prey_side_stdout.txt`: booked and implied M2 split into J / A vs SMS 2023 saithe partial M2.
- `cooccurrence_check.py` -> `cooccurrence_check_stdout.txt`: the gate ratio with the J part scaled by the field
  co-occurrence factor.
- `papers/README.md`: one PDF (Kaartvedt 2005, not committed) and two Unpaywall records; the sibling note's PDFs
  (Armannsson 2012, Neilson 2003, Lough 1989) were read in place.

```
S=<session scratchpad>
python3 -I heras_saithe_height.py "../../Ekolodsdata planktivorer/data/derived/biotic_region_"{hauls,catch,individuals}.tsv \
    "../../Krill i Skagerrak/wc_depth/tables/heras_haul_meta.tsv"          # ~3 min
python3 -I neilson_hs.py
python3 -I band_hs.py <repo> $S/tmp ../tables/band_hs.json                   # ~3 min
python3 -I window_hs.py <repo> $S/tmp ../tables/band_hs.json ../tables/window_hs.json
python3 gate_hs.py <repo> ../tables/band_hs.json ../tables/window_hs.json $S/libs ../tables/gates   # ~10 s
python3 -I gov_juv_overlap.py "../../Sejens djuplage/data/gov"   # the real folder name has an a-umlaut
python3 -I prey_side.py <repo> ../tables/gates/gate_hs.json
python3 -I cooccurrence_check.py ../tables/gates/gate_hs.json
```

## Question

Under the combined capture-p x schooling derivation budget_gate puts WC -> benthivores at 2.00 of its booked M2
(2.33 at the field spread). Earlier notes named saithe's scale height above the bed (Hs; exp 15 m, 5-40 m, a July
judgement) as the controlling unknown. What does the field say about saithe's height above the bed, the juvenile
prey's position, the stage window and the booking, and does the overshoot survive?

## Findings

### 1. Saithe's vertical profile above the bed

- In-area, June-July, day [F data, D fit] (E1): adult saithe (>= 35 cm) are caught from the bed up to ~35 m above
  it, in none of 75 day hauls fished more than 60 m above it. Exp from the bed: Hs 14-15 m by quasi-Poisson on
  beds >= 60 / 100 m with the GOV bed-class density as offset (9 m over beds >= 140 m); by presence/absence Hs
  14-31 m; 95 % profile ranges 5.5-85 m. Bottom-2-m share 0.12-0.20 (catch-rate fits), 0.06-0.13 (presence).
- Profile shape [D/J]: a hump (h exp(-h/s), mode 4.5-7 m) is a bracket, not the central. Bottom-2-m share
  0.03-0.07; it beats exp on the catch-rate likelihood (2 dll/phi +5 to +14) but loses on presence (-5.7 to
  +1.0); it rests on 18 adult-saithe hauls, 8 over beds >= 140 m, one of 1834 kg/h. The cross-gear check
  contradicts it (E2): GOV takes 172 kg/h at the bed over 150-250 m beds (Q3, 54 hauls), the pelagic trawl
  reaches that only 3-10 m above the bed (184 kg/h) and takes 22 kg/h at footrope 0-3 m, so the on-bed deficit is
  plausibly a pelagic-gear artefact.
- Winter and spawning aggregations, other areas [F/S] (E3, E4), a winter bracket only [J]: Neilson 2003 (Scotian
  Shelf, October, pre-spawning, 1-30 m integrated) implies Hs 15 / 9 m (day / night) at one site and 3.5 / 2.5 m
  at the other; DFO 2003/075 (Dec-Jan) Hs 2-9 m. Neilson's saithe were nearer the bed at night (DFO gives no
  day/night split). Both are non-feeding aggregations; the annual-mean SEARCH profile weights the feeding months
  (saithe in the box are 0.9 Q3 fish; Q1 stomachs hold ~0 haddock; Armannsson & Jonsson 2012 (E5): activity much
  reduced at night and in winter).
- Night [F/D]: no bed-referenced night haul exists; 11/77 night surface hauls over beds >= 100 m held adults vs
  0/60 by day. Nothing contradicts the library's night form (0.07 in 0-30 m, the rest nearer the bed); it barely
  matters here (night/day 0.165; night variants x0.97-0.98).
- Other members: hake, anglerfish, ling as in section 269 (bed-bound or exp 5-8 m; HERAS: hake in 32/43 day bed
  hauls). Their share of the pair's a rises as saithe's f(bed) falls (0.10 at Hs 15, 0.20 at Hs 40, 0.21 for the
  hump), and they carry adult-sized windows.

Supported profile [D/J]: exponential from the bed, Hs 15 m (in-area 9-31 m; hump and winter Hs 2.5-9 m outer
brackets); bottom-2-m share 0.125 (0.06-0.20), f(bed) ~0.063 /m (0.03-0.10). This is the library value: the
evidence does not move Hs.

Hs is not the controlling unknown of this pair: across exp Hs 7-40 m it stays at 1.85-2.50 of its booking (equal
cells). As saithe's f(bed) falls, the bed-bound members' share of the a rises and their adult windows lift f_A
(window [0.375, 0.021] at Hs 7 -> [0.444, 0.083] at Hs 40), offsetting most of the drop. The earlier "at Hs 40 m
~1.0" held before the capture-p change, not under it.

### 2. Saithe diet on juvenile gadoids, and the booking

- Booking origin [F] (E12): predation_mortality 0.0833 = the water column's consumption-weighted member diet
  share on benthivores (0.061) x its ration / prey B (params_split_223.md 3.1-3.2; unchanged at 228). Field:
  saithe 5 % (2-9) of food, almost all 0-1 group haddock; the group ~4 % (2-9). The booking is at the top of the
  field central, not low, so it cannot absorb a 2.0.
- SMS cross-check [F/D] (E13, `prey_side.py`): SMS 2023 saithe partial M2 on haddock 2001-22, age 0 0.385, age 1
  0.519 /yr at 0.66 t/km2 of saithe; at the box's 0.49 t/km2 0.29 / 0.39 /yr. SMS-based flow at the box density
  ~0.056 t/km2/yr from saithe (0.49 x Q/B 2.78 x 4.1 %) + ~0.01 from anglerfish and spurdog = ~0.066. The booked
  flow 0.127 t/km2/yr is already ~1.9x that. SMS itself overshot the stomachs in 1980-91 (9.1 vs 4.1 %, x0.45).
- Where and when [S] (E14): haddock in saithe food ~0 in Q1, peaking in Q3 at settlement, scaling with year-class
  strength; saithe feed mainly by day. Gap: no stomach study separates captures at the bed from off-bed or night.

### 3. The juvenile prey's position

- Vertical [F] (E9, E10): settled 0-group cod and haddock hold "within a few centimeters of the bottom" by day
  (Lough 1989); at night most stay on the bed, some rise 3-5 m. Pelagic 0-group (< ~8 cm) migrate up at night;
  Barents small haddock rise 30-40 m. The band derivation's phi_j (uniform bottom 2 m, day and night) is right in
  kind. A 0.5 m layer lowers the pair x0.83 (slab clipping: half the detection disc below the bed) [D]; night-rise
  variants x0.96-0.99 (the pair's night a is 0.165 of the day value) [D].
- Horizontal, the decisive field fact [F data, D] (E11, `gov_juv_overlap.py`): in-gape haddock (< 19 cm) sit over
  shallow beds (box Q3 class shares 0.82 / 0.14 / 0.04 / 0 for < 100 / 100-150 / 150-250 / > 250 m), adult saithe
  over deep beds (0.005 / 0.17 / 0.80 / 0.03). F_gate, the juvenile density the mean adult saithe meets relative
  to the box mean:

  | reading | F |
  |---|---|
  | box, Q3 | 0.19 |
  | box, Q1 | 0.55 |
  | box, presence-weighted | 0.225 |
  | Skagerrak/Trench window | 0.49 |
  | haul level, box / wide | 0.086 / 0.296 |
  | sibling note's haul-level Williamson overlap, Q1 / Q3 | 0.06 / 0.26 |

- Engine and gate [D]: recruits are imported into J uniformly over the cells and J rides inside B (section 250);
  budget_gate places the benthivores uniformly (only the WC and whiting have field bed shares). Both credit saithe
  with the box-mean juvenile density, ~2-4x what the field shows at class level, 3-12x at haul level. The booked
  share comes from stomachs of saithe caught where saithe are.

### 4. The stage window [0.392, 0.0388]

- Consistent with the capture-p note [D]: the 232.2/243 aw form re-weighted on the band a's members (reproduced
  [0.394, 0.0391]); the capture note took PPR inside the window (0.25 central), so p and the window do not double
  count. August 0-group (8-17 cm, mode 12-13) on 43 cm saithe spans PPR 0.19-0.40 (p_s 0.73-0.48), ~0.25
  biomass-weighted over the two saithe stages.
- Saithe alone: window [0.354, 0.0008], f_eff 0.054; haddock only 0.047 (the diet note's 0.043 for settled
  in-gape haddock). A haddock-only identity lowers the pair x0.93-0.95 [F stomachs: cod is a trace].
- The pair's f_eff 0.092 comes from the bed members' adult windows: the A part is 0.36 of the pair's encounter
  flow at Hs 15 (0.49-0.52 at Hs 40 or with the hump), carried by members holding ~0.10 of the a.

### 5. Recomputed pairs and the gate

Band + rule A as a ratio to the section 269 central (equal-cell a; `band_hs_stdout.txt`):

| variant | WC -> krill | -> planktivores | -> benthivores | -> whiting | saithe share of ben a | bottom-2-m share |
|---|---|---|---|---|---|---|
| exp Hs 15 (library) | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.125 |
| exp Hs 7 / 10 | 1.04 / 1.03 | 1.29 / 1.16 | 1.86 / 1.39 | 1.48 / 1.26 | 0.95 / 0.93 | 0.25 / 0.18 |
| exp Hs 20 / 30 / 40 | 0.96 / 0.90 / 0.85 | 0.90 / 0.78 / 0.72 | 0.80 / 0.59 / 0.49 | 0.84 / 0.65 / 0.55 | 0.87 / 0.83 / 0.80 | 0.10 / 0.07 / 0.05 |
| hump s 6 / 10 m | 1.08 / 0.98 | 1.04 / 0.76 | 0.48 / 0.25 | 1.05 / 0.61 | 0.79 / 0.60 | 0.045 / 0.018 |
| uniform 35 m | 1.00 | 0.78 | 0.52 | 0.63 | 0.80 | 0.057 |
| juveniles in 0.5 m | 1.00 | 1.00 | 0.83 | 1.00 | 0.88 | 0.125 |
| night rise (Aglen) | 1.00 | 1.00 | 0.96 | 1.00 | 0.90 | 0.125 |

Gate after the combined capture-p x schooling factors (`gate_hs_stdout.txt`); prey side = implied / booked M2 at
equal cells; predator side = encounter share / booked share, equal cells with the field spread in brackets;
* = OFF:

| run | prey plk / ben / whi / krill | WC -> ben | WC -> krill | WC -> whiting | WC -> plk |
|---|---|---|---|---|---|
| library (no factors) | 0.69 / 0.62 / 0.60 / 1.04 | 0.97 (1.13) | 1.39 (1.40) | 0.64 (0.46) | 0.78 (0.76) |
| combined central (exp 15) | 0.89 / 1.06 / 1.03 / 1.01 | 2.00 (2.33) | 0.97 (0.97) | 1.31 (0.94) | 0.88 (0.85) |
| exp Hs 7 | 0.92 / 1.24 / 1.15 / 1.00 | 2.50 (3.13*) | 0.74 (0.59) | 1.60 (1.23) | 0.94 (0.98) |
| exp Hs 10 | 0.91 / 1.14 / 1.10 / 1.00 | 2.23 (2.72) | 0.85 (0.75) | 1.48 (1.11) | 0.92 (0.94) |
| exp Hs 20 | 0.87 / 1.03 / 0.97 / 1.01 | 1.93 (2.15) | 1.04 (1.12) | 1.17 (0.81) | 0.84 (0.78) |
| exp Hs 30 | 0.85 / 1.01 / 0.90 / 1.02 | 1.87 (1.94) | 1.13 (1.29) | 1.00 (0.66) | 0.81 (0.70) |
| exp Hs 40 | 0.85 / 1.00 / 0.86 / 1.02 | 1.85 (1.82) | 1.16 (1.37) | 0.90 (0.58) | 0.80 (0.67) |
| hump s 6 | 0.90 / 0.87 / 1.06 / 1.01 | 1.47 (1.81) | 1.02 (0.96) | 1.38 (1.02) | 0.91 (0.94) |
| hump s 10 | 0.85 / 0.85 / 0.88 / 1.02 | 1.42 (1.58) | 1.24 (1.34) | 0.95 (0.63) | 0.80 (0.72) |
| uniform 35 m | 0.84 / 0.95 / 0.87 / 1.02 | 1.69 (1.87) | 1.22 (1.29) | 0.93 (0.62) | 0.78 (0.72) |
| juveniles 0.5 m | 0.89 / 1.00 / 1.03 / 1.01 | 1.85 (2.23) | 0.98 (0.98) | 1.32 (0.94) | 0.89 (0.86) |
| juveniles 0.5 m, haddock only | 0.90 / 0.97 / 1.04 / 1.01 | 1.75 (2.11) | 0.99 (0.99) | 1.34 (0.95) | 0.90 (0.87) |
| hump s 6 + juv 0.5 m + haddock | 0.91 / 0.81 / 1.07 / 1.01 | 1.30 (1.67) | 1.04 (0.97) | 1.40 (1.04) | 0.93 (0.95) |

OFF in every run: WC -> mackerel, seals -> WC, whiting -> benthivores, whiting -> mobile_epifauna (field
spread); in one run only: WC -> benthivores at the field spread at Hs 7 (3.13). Benthivores -> whiting passes in
every combined run. The WC krill share stays on its booking at the supported profile (0.97), 0.74-1.16 across Hs
7-40 m (0.59-1.37 at the field spread). WC -> whiting 1.31 at the central (0.94 at the field spread, which already
places whiting where it is). Planktivores 0.89 (0.84-0.92).

Co-occurrence check [D] (`cooccurrence_check.py`): only the J part of the WC -> benthivores flow (share 0.64 at the
central) scaled by the field F; the adult part as the gate has it.

| run | field spread as gated | x F class: box 0.225 / wide 0.49 | x F haul: box 0.086 / wide 0.296 |
|---|---|---|---|
| combined central | 2.33 | 1.17 / 1.57 | 0.97 / 1.28 |
| juveniles 0.5 m + haddock only | 2.11 | 1.19 / 1.50 | 1.02 / 1.27 |
| exp Hs 40 | 1.82 | 1.13 / 1.37 | 1.01 / 1.20 |
| hump s 6 + juv 0.5 m + haddock | 1.67 | 1.21 / 1.37 | 1.13 / 1.26 |
| library (no factors) | 1.13 | 0.57 / 0.76 | 0.47 / 0.62 |

Prey-side coherence [D]: with F in, predator and prey sides agree.

| quantity | M2 on J (/yr) |
|---|---|
| booking split by the window | 0.36 (0.46 if all on the haddock part) |
| SMS saithe on haddock age 0 / 1, at the box's saithe density | 0.29 / 0.39 |
| combined central, implied | 0.71 (2x SMS) |
| combined central x F_gate 0.225-0.49 | 0.16-0.35 (back on SMS) |

## Gaps

1. No bed-referenced saithe profile outside June-July in the area; the in-area fit rests on 18 adult hauls aimed
   at herring marks, nominal footrope heights, mixed gears.
2. Search time vs occupancy: every source measures occupancy; saithe search mostly by day (E5); hump vs exp for
   the bottom 2 m stays open.
3. No stomach study places saithe's juvenile-haddock captures at the bed or off it, by day or night.
4. F built from haddock < 19 cm and applied to the whole J share; juvenile cod and flatfish by bed depth not
   checked (the bed members' J take would follow their own distributions). GOV q at < 19 cm is low but equal in
   every bed class, so the ratio holds; the Q1 > 250 m class is imputed; one gear.
5. The adult part (0.36-0.5 of the pair's flow: anglerfish / ling / hake on adult benthivores) not checked against
   those members' own field diets.
6. The SMS comparison assumes a North Sea area of 575 000 km2 and the box's saithe at 0.49 t/km2.

## Verdict

- Profile [D/J; moderate for July days, low otherwise]: exponential from the bed, Hs 15 m (9-31 m in-area; hump
  and winter readings outer brackets), bottom-2-m share 0.125 (0.06-0.20). The library value stands; if anything
  the winter acoustics lower Hs.
- The overshoot does not come from the profile: at any evidence-supported profile the pair stays 1.8-2.5 of its
  booking (equal cells), because the bed-bound members' adult windows replace saithe's lost f(bed). Hs is not the
  controlling unknown of this pair; the earlier framing is superseded.
- The overshoot remains, and the evidence points to the juveniles' horizontal distribution [F data, D]: adult
  saithe stand over 150-250 m beds, in-gape juvenile haddock over < 100 m beds; saithe meet 0.2-0.5 (class level)
  to 0.1-0.3 (haul level) of the box-mean juvenile density, while the engine (recruits imported uniformly, J
  inside B) and the gate (benthivores uniform) credit 1. With F in, the pair lands at 1.0-1.6 at the field spread
  (central 1.2-1.3) and the prey-side M2 on juveniles lands on SMS. Second-order and field-supported: juveniles
  within a few cm of the bed (x0.83-0.92), the haddock-only saithe identity (x0.93-0.95). Not supported as causes:
  the booking (top of the field range, ~1.9x the SMS flow), the window (consistent with p by construction), the
  night (x0.96-0.99).
- Levers, not chosen here (user's call):
  (a) a bed-depth shape on the benthivores' `recruit_import` - settlement habitat is a physical fact, like
      `capacity_depth_table` or an open pool's entry cells, so it fits the behaviour rule; it moves the J of every
      predator pair and the benthivores' own dynamics;
  (b) a per-pair `depth_access_table` term carrying J's bed-class density relative to B - it acts on both stages,
      while A is 0.36-0.5 of this pair's flow;
  (c) for the gate only, a field bed shape for J (as section 252 gave whiting), so the check is like-for-like.
  Whichever is taken, budget_gate's equal-cell WC -> benthivores ratio (2.00) is not the like-for-like test for
  this pair; the field-spread form with a J shape is.

## User decisions

- 2026-11-11: lever (a): a bed-depth shape on the benthivores' recruit_import
  from GOV (all members: cod, haddock, flatfish), and the same shape in the
  gate. The shape is investigated first; nothing applied.
