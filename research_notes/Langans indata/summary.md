# Ling's inputs in the water-column pairs (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library, code, tests, manifest,
AGENTS.md, resume or existing notes; no training; library copies only in the session scratchpad; budget_gate.py and
field_diet.py were patched in memory only (as research_notes/"Vattenpelarens vuxna bottenfisk"). Every value below
is read from field data, an abstract, a figure or the exec'd derivations; the gate is a check and chose nothing.
Sources E1-E15 and model facts M1-M5: `evidence.md`. Grades: [F] field data or full text / figure read, [L] lab,
[A] abstract only, [S] secondary / via another note, [D] derived here, [J] judgement.

## Files

Run from `scripts/`; each script's stdout is saved beside it.

- `ling_gov_box.py` -> `ling_gov_box_stdout.txt`: NS-IBTS GOV 2000-25 (DATRAS extracts on disk), ling hauls, length
  composition and bed-class shape in the box and the wider Skagerrak / Trench window; co-occurrence with the gate's
  benthivore J / A shapes against the water-column group's shape.
- `ling_ices_all.py` -> `ling_ices_all_stdout.txt`, `../tables/ling_ices_items.csv`: every ling stomach in the ICES
  stomach database on disk, every prey item with length and prey / predator length ratio (PPR).
- `qb_ling.py` -> `qb_ling_stdout.txt`: Q/B proxies (Palomares & Pauly 1998; SMS 2023 cod ration at ling weights).
- `decompose_ling.py` -> `decompose_ling_stdout.txt`, `../tables/decompose_ling.json`: COPY of
  research_notes/"Vattenpelarens vuxna bottenfisk"/scripts/decompose_members.py with a ling hook (duty cycle with the
  stationary part's scan, night, profile) and ling's own windows on planktivores and whiting; the baseline still agrees with band_fish_pair.
- `gate_ling.py` -> `gate_ling_stdout.txt`, `../tables/gates/budget_gate_<run>.txt`, `../tables/gates/gate_ling.json`:
  COPY of that note's gate_adult.py; every run starts from its central (stage p + adult profile + hake out) and
  changes only ling: its g and A-profile factor on WC -> benthivores, its g and own window on WC -> whiting and WC ->
  planktivores, its g on WC -> mackerel.
- `ling_flow_check.py` -> `ling_flow_check_stdout.txt`: ling's implied flow per prey (field spread) against its
  field ration and diet; the WC pair ratios per prey.
- `../tables/hywind2022_ling_stomach_counts.csv`: ling stomach categories read from the IMR 2022 report figure.
- `papers/README.md`: what was fetched (Unpaywall PDFs, IMR report figures) and what was not.

```
R=<repo>/research_notes; S=<session scratchpad>
python3 -I ling_gov_box.py "$R/Fiskgrupper i rutan/data/extract" "$R/Torskfiskarnas artsammansattning/data/extract" \
    "$R/Fiskgrupper i rutan/tables/length_weight_all.csv" <repo>/maps/depth.npz             # a-umlaut in the real name
python3 -I ling_ices_all.py "$R/Torskfiskarnas bytesandelar/data/diet/ices_stomach" ../tables/ling_ices_items.csv
python3 -I qb_ling.py "$R/Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv" \
    "$R/Fiskgrupper i rutan/predators/data/diet/sms_consumption/NS2023_consum_ab.in"
python3 -I decompose_ling.py <repo> $S/tmp ../tables/decompose_ling.json                    # ~2 min
python3 gate_ling.py <repo> "$R/Rekryternas bottendjup/tables/gov_juv_bed.json" ../tables/decompose_ling.json $S/libs ../tables/gates
python3 -I ling_flow_check.py ../tables/decompose_ling.json ../tables/gates/gate_ling.json
```

## Question

research_notes/"Vattenpelarens vuxna bottenfisk" found that ling (0.014 of the water column, 0.0086 t/km2) carries
~0.36 of the adult part of water column (WC) -> benthivores at 1.84x the group's a per tonne, crediting it with
~2.2x its whole field ration on adult benthivores alone, and that every ling input is a judgement. What does the
evidence say about ling's length mix, diet, ration and behaviour, and what do ling's inputs become when re-derived
from it?

## Model (M1-M5)

- Ling's a per tonne comes from the section 220 encounter terms on the section 269 band + rule A: TYPE "midambush",
  speed v_hake 0.5 BL/s day and night ("J: no measurement"), profile exp(5 m) above the bed ("[J] no field profile"),
  R_sat = 2 L = 1.26 m (binds), N/t 550, number-mean L 62.9 cm (GOV, Skagerrak + box, depth-stratified).
- Membership: capability scores (research_notes/"Fiskgrupper i rutan"/predators/inputs/pred_scores.csv) benth_fish
  0.60 "judgement: demersal piscivore" (also read for the whiting pair), plank_fish 0.60, zoo_krill 0.05. A 0/1
  switch at >= 0.1: ling is in WC -> benthivores, whiting, planktivores, mackerel; NOT in WC -> krill.
- Windows: the library window sets do not list ling; the sibling decomposition gave it anglerfish's (p95 PPR 0.57).
- Ling's share of each pair's a at the band central: benthivores 0.0249 (day share 0.88), whiting 0.0177,
  planktivores 0.0152, mackerel 0.0082 (decompose_ling_stdout.txt).

## Findings

### 1. Length and age in the box [F data] (E1, E10)

| source | hauls with ling | length range | number-mean L | biomass-median L | note |
|---|---|---|---|---|---|
| model N/t table (GOV Q1+Q3 2001-25, Skagerrak + box rectangles, depth-stratified) | - | 5-150 cm | 62.9 cm | 77.5 cm | 0.16 of biomass in 140-155 cm classes (deep strata) |
| box 57.7-58.4 N, 9.9-11.3 E (GOV 2000-25) | 73 of 436 (17 %), ~190 fish | 8-90 cm | 56.6 cm | 64.5 cm | no ling > 90 cm |
| wider window 57.3-59.2 N, 8-12 E | 160 of 1533 | 8-150 cm | 58.1 cm | 69.5 cm | |
| Hywind Tampen gillnets (E3, E4; 61 N, 136-286 m) | - | 0.5-1.45 m | - | mode ~0.95 m | gillnet-selected, spawning aggregation |

- Length-weight: CA fit used for N/t 0.00371 L^3.088; stock annex 0.0055 L^3.012; FishBase 0.0038 L^3.08 - the same
  within ~3 % over 40-150 cm. Age: L50 63.6 cm at 4.8 yr (annex); Linf / K 183-189 cm / 0.08-0.12 in 4.a; so the box
  ling (median 57-65 cm) sit around the length at 50 % maturity (63.6 cm, 4.8 yr) [D from E10].
- Bed depth (box, kg/h by class < 100 / 100-150 / 150-250 / >= 250 m): 0.16 / 0.45 / 1.53 / 0 (3 hauls); class shares
  0.06 / 0.20 / 0.74 / 0, the water-column group's own GOV shape (saithe 0.04 / 0.13 / 0.78 / 0.05). Co-occurrence
  with the gate's benthivore J / A shapes relative to the group: 1.25 / 1.12 (box), 0.89 / 0.98 (window). Ling is on
  the box's trawlable ground, mostly at 150-250 m, despite the literature's "prefers hard seabed or sand with large
  rocks" (E10, E11 [S]); no member-level spatial correction is indicated.

### 2. Diet (E2-E5, E8, E9, E14, E15)

| source | n | fish | benthivore members by mass | other main prey | grade |
|---|---|---|---|---|---|
| ICES stomach DB, North Sea + Skagerrak (all years) | 12 stomachs with prey, 30-88 cm | ~all | 0.39 of 235 g (one 16.6 cm flatfish, one 6 cm cod) | Norway pout, Trisopterus, herring 0.61 | [F data, anecdotal] |
| ICES extract by area (via sibling E11) | NNS 25 fed < 40 cm; Skagerrak 16 | ~all | NNS cod 5.7 %; Skagerrak 0 | Norway pout 92 %; herring 81 %, Trisopterus 19 % | [S] |
| Hywind 2022 (gillnet, 0.7-1.4 m ling) | ~128 category records (114 stomach samples listed: records are not stomachs), 58 empty | 61 of 70 non-empty records | demersal teleost 3 of 11 identified fish (species not given per predator) | pelagic teleost 8 of 11 (mackerel, herring, blue whiting) | [F figure, counts] |
| Hywind 2023 | ~85 category records read from bars, mostly empty / inverted | unidentified teleost dominates | <= 2 demersal teleost records | <= 3 pelagic | [F figure, counts] |
| Iceland MFRI 1996-2023 (all sizes) | not on the excerpt pages | - | haddock 8.2 % + cod 6.7 % = ~15 % (flatfish not counted) | - | [F figure] |
| FishBase, Bergstad 1991, Sara 2009, stock annex | - | "mainly fish" | cod (adults and juveniles), flatfish, haddock named | herring, blue whiting, saithe, Norway pout | [S] |

- Ling is a piscivore that eats benthivores: by mass ~0.15 (0.05-0.30) of its diet [F figure / F anecdotal],
  enough to stay a member (score >= 0.1). Removing ling from the pair ("ling out") is NOT supported by its diet.
- Prey / predator length ratio (E2, n 7 fish items): 0.07, 0.07, 0.20, 0.22, 0.25, 0.26, 0.28 - median 0.22, max
  0.28. No ling item reaches anglerfish's 0.57. Adult benthivores in ling's diet are named [S] but no in-area adult
  item and no prey sizes from Iceland are available; the adult fraction stays [J] 0.5 (0.2-0.9).
- Counts from Hywind are of stomach categories, not mass; feeding in the nets cannot be excluded; March is a
  spawning aggregation.

### 3. Ration Q/B [D / S proxy] (E12, E13; `qb_ling_stdout.txt`)

| proxy | Q/B /yr | basis |
|---|---|---|
| Palomares & Pauly 1998 (carnivore, T 7.4 C) | 1.23-1.59 (A 0.8-1.6 [J]); 1.22-1.61 at 6-9 C | Linf 160-189 cm, Winf 24-40 kg (annex growth and length-weight) |
| SMS 2023 cod ration at ling's box weights | 3.8 (2 kg 4.1, 15 kg 2.7) | a more active gadoid at the same size [S proxy] |
| used here | 2.0 (1.3-3.8) [J] | inside the two proxies; Lokkeborg's 65 % inactivity (E6) points to the low side |

No ling Q/B, ration, evacuation rate or bioenergetics model exists in the sources reached; no Ecopath model on disk
has a ling group.

### 4. Behaviour (E1, E3, E4, E6, E7, E14)

- Activity [A]: 5 tracked ling were inactive ~65 % of the time inside a small core area, crepuscular peaks, moved
  slowly in a restricted area near bait; they located bait in active and inactive periods by day, never reacted at
  night; "less active at night" than cod, "more dependent on visual stimuli", taking "more mobile prey" (E6).
  So ling is neither a continuous cruiser (the model's 0.5 BL/s day and night) nor a bed ambusher like anglerfish:
  a mixed searcher, active ~0.35 of the time, sit-and-wait (still responsive by day) the rest.
- Speed: no number reachable (the tracking speeds are in the closed full text). v_hake 0.5 BL/s stays [J] for the
  active part.
- Vertical position: tagged "close to the sea bed" (E7 [A]); caught by bottom gillnets and GOV (E1, E3, E4); one
  ambiguous ling / tusk tag set (E14 [S]). No depth-tag profile: exp(5 m) stays [J], with exp(2 m) as the bracket the
  "near the bed" evidence points to (it RAISES ling's slab time and a).
- Habitat: literature "rocky / sand with large rocks" (E10, E11 [S]), but in the box ling is caught on trawlable
  150-250 m ground in 17 % of GOV hauls (E1), and at Hywind on soft slope and even shallow ground (E4).

### 5. Re-derived ling inputs and the recomputation [D]

Ling's inputs from the evidence:

| input | before | from evidence | grade |
|---|---|---|---|
| encounter mode | cruising 0.5 BL/s, 24 h | 0.35 active cruising at 0.5 BL/s + 0.65 sit-and-wait: prey passes at its own speed u (vrel = u, the derivation's ambush branch) and the stationary fish scans the whole disc (scan 1, as that branch; the cruiser's 0.5 as the low bracket); visual R unchanged. Factor (0.35 vrel(v,u) + 0.65 u x 1/0.5) / vrel(v,u) = x0.75 on the fish pairs (u 0.1 m/s), x0.85 on planktivores (herring u 0.2) | [A] split; [J] speed and stationary scan; [D] factor |
| night | as day | as day (central); bracket no night foraging (x0.88 on benthivores: day share 0.88) | [A] bracket |
| window | anglerfish's p95 0.57: ben [0.970, 0.587], plk [0.993, 0.930], wht [0.990, 0.834] | own length mix at p95 0.30: ben [0.683, 0.260], plk [0.920, 0.483], wht [0.838, 0.332]; bracket 0.40: ben [0.907, 0.420], plk [0.994, 0.782], wht [0.981, 0.621] | [F data, n 7] ratio (max 0.28); [D] window |
| capture p inside the window | anglerfish's (J 0.56 / A 0.48) | own window: J 0.63 / A 0.62 | [D] |
| profile | exp 5 m | exp 5 m kept; bracket exp 2 m (alone x1.85 on WC -> benthivores) | [J]; no field profile |
| membership | benth_fish score 0.60 [J] | member: benthivores ~0.15 (0.05-0.30) of the diet by mass, >= 0.1 | [F figure / F anecdotal] |
| spatial co-occurrence | group shape | ling's GOV bed shape = the group's (factor 0.89-1.25) | [F data, 73 box hauls]; no change |
| Q/B (ration check only) | 2.0 (1.5-3.0) [J] | 2.0 (1.3-3.8) [J] inside two proxies | [D / S proxy] |

Ling's g per pair (ling a under the variant / ling a before; `decompose_ling_stdout.txt`):

| variant | benthivores | whiting | planktivores | mackerel |
|---|---|---|---|---|
| act 0.35, stationary scan 1 (central) | 0.750 | 0.750 | 0.845 | 0.946 |
| act 0.35, stationary scan 0.5 | 0.550 | 0.550 | 0.597 | 0.648 |
| act 0.50, scan 1 | 0.808 | 0.808 | 0.880 | 0.959 |
| idle 0.35 (inactive time not foraging) | 0.350 | 0.350 | 0.350 | 0.350 |
| act 0.35 scan 1 + night 0 | 0.660 | 0.667 | 0.813 | 0.578 |
| exp 2 m profile alone / with act 0.35 scan 1 | 1.848 / 1.386 | 1.089 / 0.816 | 1.085 / 0.916 | 0.333 / 0.315 |

Gate (all runs: combined factors, J on the recruit_depth_table, A on the GOV annual shape, and the sibling central
stage p + adult profile + hake out; only ling changes; library copies made from the UNEDITED library, and
prev_central reproduces the sibling's 1.54 / 1.41 / 2.13). WC -> benthivores implied / booked M2: equal cells | field
spread | P2; the benthivores' prey-side total; the other WC pairs (prey side, equal / field); the pair's adult flow
against the field adult take of all members (0.0177 t/km2/yr, sibling field_vs_model) and ling's part of it.

| run | WC -> ben a, stage_access | eq / field / P2 | ben total eq / field | WC -> whiting | WC -> plankt. | pair A flow (x field A); ling's part |
|---|---|---|---|---|---|---|
| sibling central (ling as before) | 0.117, [0.382, 0.0165] | 1.54 / 1.41 / 2.13 | 0.89 / 0.87 | 1.36 / 1.02 | 0.91 / 0.93 | 0.0446 (2.5); 0.58 |
| own window 0.30 only | 0.117, [0.377, 0.0124] | 1.47 / 1.33 / 2.05 | 0.87 / 0.84 | 1.32 / 0.98 | 0.92 / 0.93 | 0.0341 (1.9); 0.44 |
| activity (scan 1) only | 0.1162, [0.378, 0.0142] | 1.49 / 1.36 / 2.07 | 0.88 / 0.85 | 1.34 / 1.00 | 0.92 / 0.93 | 0.0386 (2.2); 0.51 |
| **central: activity (scan 1) + own window 0.30** | **0.1162, [0.375, 0.0111]** | **1.44 / 1.30 / 2.01** | **0.86 / 0.83** | **1.31 / 0.98** | **0.92 / 0.93** | **0.0306 (1.7); 0.37** |
| central + night 0 | 0.116, [0.374, 0.0106] | 1.43 / 1.29 / 2.00 | 0.85 / 0.83 | 1.30 / 0.97 | 0.92 / 0.93 | 0.0293 (1.7); 0.34 |
| stationary scan 0.5 + window 0.30 | 0.1156, [0.373, 0.0101] | 1.42 / 1.27 / 1.98 | 0.85 / 0.82 | 1.30 / 0.97 | 0.92 / 0.93 | 0.0278 (1.6); 0.30 |
| idle 0.35 + window 0.30 (low bracket) | 0.115, [0.371, 0.0090] | 1.39 / 1.25 / 1.96 | 0.84 / 0.81 | 1.30 / 0.97 | 0.92 / 0.93 | 0.0249 (1.4); 0.21 |
| act 0.50 scan 1 + window 0.40 (high bracket) | 0.1164, [0.379, 0.0136] | 1.49 / 1.35 / 2.07 | 0.88 / 0.85 | 1.33 / 0.99 | 0.92 / 0.93 | 0.0371 (2.1); 0.48 |
| central + exp 2 m profile | 0.1182, [0.380, 0.0139] | 1.52 / 1.38 / 2.11 | 0.89 / 0.86 | 1.30 / 0.97 | 0.91 / 0.92 | 0.0383 (2.2); 0.50 |
| ling out (not supported) | 0.1139, [0.367, 0.0072] | 1.32 / 1.17 / 1.86 | 0.81 / 0.79 | 1.39 / 1.04 | 0.93 / 0.94 | 0.0194 (1.1); 0 |

- Central edits on the other pairs: WC -> whiting a x0.9956, stage_access [0.41, 0.08] -> [0.407, 0.072]; WC ->
  planktivores a x0.9976, [0.65, 0.18] -> [0.648, 0.173]; WC -> mackerel a x0.9996. WC -> krill: ling is not a
  member (zoo_krill 0.05), unchanged (prey total 1.01 / 1.03 in every run). Prey totals (equal / field):
  planktivores 0.91 / 0.88, whiting 1.03 / 0.89; epifauna, Nephrops, infauna unchanged. OFF pairs unchanged in
  every run (WC -> mackerel and seals -> WC on the prey side, whiting -> benthivores, whiting -> mobile_epifauna at
  the field spread); none added.
- The whiting / planktivore re-weighting takes ling's "before" window as anglerfish's (the sibling's convention);
  the library windows of those pairs were derived from member sets without ling (aw_stage_access WC_WHT / WC_PLK),
  so the edits there are the ling-share-sized change (0.013-0.018 of the a), not a re-derivation of the windows.
- The WC -> benthivores a falls only x0.994 (ling's share 0.0249 x 0.25); the adult flow falls through f_A
  (0.0165 -> 0.0111), because ling held 0.58 of the adult flow.

Ling's own flows at the field spread against its field ration (B 0.0086 t/km2, ration 0.0173 t/km2/yr at Q/B 2.0;
field benthivore J 0.0013 / A 0.0013 t/km2/yr; `ling_flow_check_stdout.txt`):

| run | ben J | ben A | whiting | plankt. | total | total / ration | ben A / ration | ben A / field A | ben share |
|---|---|---|---|---|---|---|---|---|---|
| sibling central | 0.0085 | 0.0258 | 0.0065 | 0.0293 | 0.0701 | 4.06 | 1.49 | 19.9 | 0.49 |
| central | 0.0052 | 0.0112 | 0.0024 | 0.0172 | 0.0360 | 2.09 | 0.65 | 8.7 | 0.46 |
| stationary scan 0.5 + window 0.30 | 0.0038 | 0.0083 | 0.0018 | 0.0122 | 0.0261 | 1.51 | 0.48 | 6.4 | 0.46 |
| idle 0.35 + window 0.30 | 0.0024 | 0.0053 | 0.0011 | 0.0072 | 0.0161 | 0.93 | 0.31 | 4.1 | 0.48 |
| act 0.50 scan 1 + window 0.40 | 0.0070 | 0.0180 | 0.0042 | 0.0236 | 0.0527 | 3.05 | 1.04 | 13.9 | 0.47 |

- Over the Q/B bracket 1.3-3.8 the central total is 3.2x-1.1x ling's ration (x1.54 / x0.53 of the 2.09): the
  activity + window re-read halves ling's credited intake (4.1x -> 2.1x at Q/B 2.0) without using the gate, but it
  stays above a ration unless Q/B is near the cod-proxy end.
- What stays wrong is the composition: ling's encounter diet is ~0.46-0.48 benthivores (A ~0.32) in every variant,
  against ~0.15 in the field (Iceland, Hywind's few identified items, the ICES North Sea items). The activity and
  window factors scale all of ling's pairs nearly alike; the share comes from geometry - benthivore prey met in a
  2 m bed slab (H_juv [J], the group convention) by a fish spending 0.33 of its time there (exp 5 m), against
  herring / pout / blue whiting dispersed over tens of metres. A tighter profile (the direction the near-bed
  evidence points) raises it to 0.58. Ling's adult benthivore take stays ~9x its field A (4-14x over the brackets;
  the field A itself spans ~80x through the [J] adult fraction and the Q/B bracket).

## Gaps

1. Ling's swimming speed and the tracking details (sizes, depth, bed type) are in the closed full text of Lokkeborg
   et al. 2000; the 0.35 / 0.65 split is from its abstract; the speed while active stays [J] 0.5 BL/s, and the
   stationary part's scan (1, as the derivation's ambush branch; 0.5 as a cruiser) is a judgement worth x0.75 vs
   x0.55 on ling's a (gate 1.30 vs 1.27 at the field spread).
2. No ling depth-tag profile: exp(5 m) is [J]; the evidence (tagged near the bed, bottom gillnets, GOV) would put it
   closer, which raises its benthivore share and a.
3. Diet by mass and size: 12 in-area stomachs with prey (7 PPRs), Hywind counts with species unresolved per
   predator, Iceland bars for haddock and cod only. The MFRI HV 2024-01 ling chapter (open, hafogvatn.is, no DOI) and
   Bergstad 1991 (closed) would give mass shares and sizes; not fetched under the PDF rule. The p95 PPR 0.30 rests on
   n 7 (max 0.28).
4. No ling Q/B: two proxies 1.3 and 3.8 differ ~3x; the ration check is bounded, not measured.
5. Length mix: the model's N/t table (Skagerrak + box, depth-stratified) holds ling to 150 cm (0.16 of biomass at
   140-155 cm); the box itself has none above 90 cm in 73 hauls. On the box mix the window and R_sat would be smaller
   (not recomputed).
6. The composition residual (benthivores 0.46 of ling's encounter diet vs ~0.15) sits in the prey-slab geometry
   (H_juv 2 m, adults placed by the sibling's profile), a group convention, not a ling input.

## Verdict

- Evidence-based ling inputs:
  (1) Encounter mode [A for the split; J for speed and stationary scan; D for the factor]: a mixed searcher, ~0.35 of
      the time cruising and ~0.65 sit-and-wait but responsive by day (Lokkeborg et al. 2000), neither a 24-h cruiser
      nor a bed ambusher like anglerfish: x0.75 on ling's terms on the fish pairs (x0.85 planktivores, x0.95
      mackerel) with the stationary part scanning the whole disc; x0.55 if it scans as a cruiser (low bracket).
      Night: no night response to bait [A] is a bracket (gate -0.01), not applied centrally.
  (2) Window [F data, n 7; D]: its own length mix at p95 PPR 0.30 (observed max 0.28), not anglerfish's 0.57:
      benthivores [0.683, 0.260], planktivores [0.920, 0.483], whiting [0.838, 0.332].
  (3) Membership [F figure / F anecdotal]: ling stays a member of WC -> benthivores and WC -> whiting (benthivores
      ~0.15 by mass; cod, haddock and flatfish in its diet wherever sampled); "ling out" is not supported. Its 0.60
      score is a judgement, but only the >= 0.1 switch matters and the diet clears it.
  (4) Profile, speed: no field value; exp(5 m) and 0.5 BL/s stay [J]; the near-bed evidence argues for a tighter
      profile (bracket exp 2 m: gate 1.52 / 1.38).
  (5) Spatial: ling's GOV bed shape equals the group's [F data]; no member correction.
  (6) Q/B: 2.0 (1.3-3.8) [J inside D / S proxies]; for checks only.
- If (1) + (2) are applied (central): WC -> benthivores 1.44 (equal) / 1.30 (field spread) / 2.01 (P2), from 1.54 /
  1.41 / 2.13; benthivore prey total 0.86 / 0.83; WC -> whiting 1.31 / 0.98 (from 1.36 / 1.02); WC -> planktivores
  0.92 / 0.93 (unchanged); WC -> krill unchanged; no OFF pair added or removed. The pair's adult flow 1.7x the field
  adult take (from 2.5x); ling's part of it 0.37 (from 0.58). Brackets: 1.39-1.52 (equal) / 1.25-1.38 (field).
- Ling's credited intake falls from 4.1x to 2.1x its ration at Q/B 2.0 (1.1-3.2x over 1.3-3.8); its credited adult
  benthivore take from 1.5x its whole ration to 0.65x, still ~9x its field adult take, because its encounter diet is
  ~0.46 benthivores against ~0.15 in the field - a residual of the group's prey-slab geometry (and of the [J]
  profile), not of a ling input that evidence can move; not to be closed on ling.
- Grade: moderate for the window and membership (small n, consistent across areas), low-moderate for the activity
  split (one abstract, 5 tracked fish), judgement for the stationary scan, speed and profile.

## If applied (not applied; the user's call)

Values against the unedited library (every gate run here starts from it; prev_central reproduced the sibling's
1.54 / 1.41 / 2.13 exactly). If the sibling's edits (hake out, stage p, adult profile) have landed, these values
REPLACE theirs; they are not multipliers on top.

- `predatory_fish_preys_on_benthivorous_fish`: max_intake_rate 0.0418 x 0.939 = 0.0393 before the combined factor
  (0.1162 after x2.96, as gated); stage_access [0.375, 0.0111] (the sibling's [0.382, 0.0165]). The comment must add
  that ling enters with its own length mix at p95 PPR 0.30 (not anglerfish's 0.57) and a duty cycle (0.35 cruising
  at 0.5 BL/s, 0.65 sit-and-wait at the prey's speed with scan 1; Lokkeborg et al. 2000, abstract).
- `predatory_fish_preys_on_whiting`: max_intake_rate 0.0359 x 0.9956 = 0.0357; stage_access [0.407, 0.072].
- `predatory_fish_preys_on_planktivorous_fish`: max_intake_rate 0.0174 x 0.9976 = 0.0174 (no change at the
  library's precision); stage_access [0.648, 0.173].
- `predatory_fish_preys_on_mackerel`: unchanged at the library's precision (x0.9996).
- The derivation scripts that define ling (research_notes/"Attackrater ur motesteori" pred_encounter_a.py TYPE /
  SPEED, research_notes/"Overlapp i fiskparen" band_pairs.py profile) are not edited here; the factors live in
  `scripts/decompose_ling.py` (ling_factor) and `scripts/gate_ling.py` (pair_edit, other_pairs).

## User decisions

- 2026-11-11: the activity split and ling's own window go into the combined
  section (section 270): water column -> benthivores a 0.0393 before the combined
  factor, stage_access [0.375, 0.0111], replacing the adult note's values; the
  small whiting and planktivore pair edits with it. The pair is also re-booked to
  0.058 (research_notes/"Sej och ungfisk i samma hal").
