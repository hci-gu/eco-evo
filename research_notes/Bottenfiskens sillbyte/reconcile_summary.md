# Herring's daytime near-bed share: the two methods reconciled (2026-10-31)

Research only; library, manifest, code and tests not edited. Tags: [F] field data or a source, [D] derived here, [J] judgement. Written by the research subagent, saved by the main agent.

**Result:** on the grid footing both methods agree. Herring's daytime share in the bottom 10 m, as a box mean, is **0.16 (0.08-0.26)** [D/J]. The proposal's 0.34 (x1.87 -> a 0.013, `summary.md`) is not supported. The benthivore -> planktivore a becomes **~0.0074 (0.0054-0.0100)** with the other members as now.

## Files
Each script's stdout is in `tables/reconcile_*_stdout.txt`.
- `scripts/reconcile_heras_profile.py` -> `tables/reconcile_heras_by_class.csv`: HERAS (RV Dana, July) near-bed profile by bed-depth class; column backscatter per grid log; herring share of near-bed backscatter in Dana bottom-trawl (EXPO) hauls; backscatter -> t/km2.
- `scripts/reconcile_gov_nearbed.py` -> `tables/reconcile_gov_by_class.csv`, `tables/reconcile_gov_hauls.csv`: GOV herring, benthivore and whiting t/km2 per haul inside the grid, by class.
- `scripts/reconcile_combine.py`: one footing; readings H1/H2; three weightings; bootstrap; sensitivities; Method 1 recomputed; consequences for a.

Commands (scripts run with `python3 -I`):
```
R=/home/simon/eco-evo/research_notes; S="$R/Bottenfiskens sillbyte"
python3 -I "$S/scripts/reconcile_heras_profile.py" "$R/Ekolodsdata planktivorer/data/ices_zip" "$R/Ekolodsdata planktivorer/data/derived" /home/simon/eco-evo/maps/depth.npz "$S/tables" > "$S/tables/reconcile_heras_profile_stdout.txt"
python3 -I "$S/scripts/reconcile_gov_nearbed.py" "$R/Fiskgrupper i rutan/data/extract" "$R/Torskfiskarnas artsammansättning/data/extract" "$R/Fiskgrupper i rutan/inputs/guilds.csv" "$R/Fiskgrupper i rutan/tables/length_weight_all.csv" "$S/tables" > "$S/tables/reconcile_gov_nearbed_stdout.txt"
python3 -I "$S/scripts/reconcile_combine.py" "$S/tables" /home/simon/eco-evo/maps/depth.npz > "$S/tables/reconcile_combine_stdout.txt"
```

## Common footing
- **Cells:** the model grid (SWEREF 99 TM E 211-271 km, N 6414-6474 km). Method 2 used the depth-map download area (57.7-58.4 N, 9.9-11.3 E); Method 1 used depth-stratified hauls from the whole Skagerrak.
- **Bed-depth classes, share of grid cells** (`maps/depth.npz`): <80 m 0.096, 80-110 0.213, 110-130 0.116, 130-150 0.106, 150-200 0.192, 200-300 0.191, >=300 0.087. Beds >=150 m are 47 % of cells.
- **Season:** HERAS late June to mid July. GOV Q3 hauls in the grid are August-September (102 day hauls). Q1 (Jan-Feb, 159 hauls) has no acoustic survey to compare with.
- **Time of day:** day only (sun above 6 deg; GOV day hauls).
- **Column herring per class (July)** [D]: backscatter per grid log 2010-26, scaled so the log mean is the grid's 4.3 t/km2 (`Ekolodsdata planktivorer`, grid correction): 7.35 / 6.80 / 3.22 / 2.80 / 2.64 / 1.28 / 2.46 t/km2. Deep cells hold a comparable column but almost no herring near the bed.
- **Vertical shares:** only the six cruises with 1 m channels (2015-20): 851 logs, 373 by day in the grid. Bed depth = deepest channel (matches EMODnet, median +0.8 m). The lowest 1-2 channels are about empty (1 m backstep plus dead zone), so "bottom 10 m" means 0-10 m above the bed with the lowest ~2 m blind.

## Measured, per class (grid, day, summer)

| Bed class | HERAS bottom-10 m share | blind 0-2 m filled at the 2-4 m density | GOV Q3 herring t/km2 (size-corrected) | GOV / column | GOV / HERAS band |
|---|---|---|---|---|---|
| <80 | 0.369 | 0.395 | 1.66 | 0.225 | 0.57 |
| 80-110 | 0.152 | 0.159 | 1.67 | 0.246 | 1.54 |
| 110-130 | 0.046 | 0.052 | 0.15 | 0.046 | 0.90 |
| 130-150 | 0.105 | 0.132 | 0.35 | 0.126 | 0.95 |
| 150-200 | not seen | | 0.099 | 0.037 | |
| 200-300 | not seen | | 0.023 | 0.018 | |

- **The GOV catch per wing-swept area equals the herring biomass HERAS measures in the 0-10 m band.** Herring-biomass-weighted over the four classes where HERAS sees the bed, q_eff = 1.00 (bootstrap 10-90 % 0.61-1.44) [D].
- **Beds >=150 m:** HERAS herring backscatter at 134-154 m is 0.6-1 % of the column; GOV Q3 0.10 and 0.02 t/km2 near the bed, shares 0.037 / 0.018. Beds >=300 m (no Q3 hauls) take the 200-300 m value [J], immaterial at 8.7 % of cells.

## Which weighting the encounter needs
The encounter is a x sum_c F(c) B_herring(c) B_benthivore(c), so a constant F should be the **herring-biomass-weighted** share. Equal-cell weighting is the wrong quantity.

| Reading | Equal-cell | Herring-weighted | Herring x benthivore | Herring x whiting |
|---|---|---|---|---|
| H2 (central) | 0.104 | **0.157** (0.142-0.180) | 0.182 (0.165-0.208) | 0.276 |
| H2 + length correction of backscatter per kg | | 0.148 | 0.169 | 0.240 |
| H1, in-net efficiency q_in 1 | 0.146 | 0.228 | 0.264 | 0.355 |
| H1, q_in 0.7 | | 0.264 | 0.304 | 0.398 |

- **H2 (acoustic profile real):** band share = HERAS share; deep beds from GOV / q_eff; the GOV surplus is fish brought into the net from above.
- **H1 (dead-zone layer):** the surplus lies in the 0-2 m dead zone.
- H2 is central [J]: the seen profile does not rise toward the bed, and under H1 GOV/band ~1 in every class would be a coincidence.

The fish groups have no depth habitat in the model, so the herring-weighted 0.16 is the constant to use. The co-weighted values hold only if the model's predators sit at the field's depths.

Sensitivities: column weights 2015-20 only 0.158; GOV hauls from the whole download area 0.166 (q_eff 1.40).

## Reconciled value: 0.16 (0.08-0.26) [D/J]
- Central: H2, herring-weighted. Sampling range 0.14-0.18.
- Low 0.08: GOV catch 2x the band, Dana near-bed composition gadoid-heavy (bottom hauls give herring 0.42 / 0.13 of near-bed backscatter at <80 / 80-110 m; face value ~0.07 cannot hold since the GOV took 1.67 t/km2 at 80-110 m).
- High 0.26: H1 with q_in 0.7 (q_in 1: 0.23).
- Walker check [F]: adult herring q 0.22-0.30 ("ignore the vertical component") matches GOV/column in the shallow classes, as H2 predicts.
- Per-class F is the durable result: 0.40 / 0.16 / 0.05 / 0.13 / 0.04 / 0.02 / 0.02.

## What each method got right and wrong
**Method 1 (0.34):** right to compare GOV with the acoustic column and that deep beds are not empty. Wrong hauls (Skagerrak-wide 0.70 annual, Q3 0.87; grid Q3 0.67, Q1 0.28 -> 0.156 against July, 0.182 annual); Q1 mixed against an assumed winter column; "0.27 is a floor" wrong in kind - Walker's level already contains vertical availability, and GOV/acoustic = q x share has two unknowns, which the acoustic profile closes with q_eff ~1 for the whole 10 m band. Only under H1 does it approach 0.23-0.26.

**Method 2 (0.10 / 0.31):** right that the share falls with bed depth and that 0.15 had no measured basis. Wrong: its ">=150 m, bed not seen" bin held every 2010-14 coarse-channel log; the per-class shares and 0.31 come from the download area incl. the dense Skagen / coastal strip (backscatter per log at <80 m 1425 there vs 273 in the grid); equal-cell weighting; dead zone and non-herring fish not quantified; >=150 m left unmeasured although the GOV measures it.

## Caveats
- Season: GOV Aug-Sep vs HERAS July. Q1 grid GOV is 0.11-0.21 of a judged winter column (0.3-0.6 x July), so annual ~ summer. In winter herring sit on deep beds (Q1 200-300 m: 0.33 t/km2).
- Light weighting (flag, not applied): F is for sun > 6 deg; at 0-6 deg the share collapses to 0.002-0.03, so the encounter's w_day = 0.5 x 0.36 over-credits ~10-15 % [J].
- Deep columns: at beds >=150 m the acoustic column stops at 154 m, so the deep F is if anything lower.

## Consequences
**Benthivores -> planktivores** (a = 0.019 x overlap factor x membership 0.377; pout 0.7, blue whiting 0.3, sprat 0.1 as now):

| Herring F | Overlap factor | a |
|---|---|---|
| 0.16 (central) | x1.04 | **0.0074** |
| 0.08 (low) | x0.75 | 0.0054 |
| 0.26 (high) | x1.39 | 0.0100 |
| 0.18 (co-weighted) | x1.12 | 0.0080 |

With the proposed sprat 0.5 and blue whiting 0.6: x1.24 -> 0.0088 (0.0068-0.0114); the sprat 0.5 came from the same ratio-as-floor logic and needs redoing on this footing. The x1.87 -> 0.013 is not supported.

**Whiting -> planktivores (band convention, not adopted):** step 2's band share 0.15 confirmed (0.16); a ~0.0121 (0.0084-0.0166); 0.0175 at 0.28 only with a whiting depth habitat.
