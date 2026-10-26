# Splitting `benthic_community` into infauna and `mobile_epifauna`: library parameters

Written 2026-10-26. Biology-first: every value comes from measurements, mass balance of measured rations
and diets, or stated judgement. No model output, no simulator run, nothing in the code, library, manifest,
AGENTS.md or resume changed. Tags: [F] full text or data read, [A] abstract, [S] secondary, [D] computed here,
J judgement.

Scripts (`scripts/`, all run with `python3 -I`, outputs in `tables/`):

| Script | What it does | Output |
|---|---|---|
| `benth_diet_split.py` | ICES stomach DB sub-pools -> epifauna share of the benthivores' benthos diet, consumption weighted | `benth_diet_split*_stdout.txt`, `.csv` |
| `pred_benthos_split.py` | the same for predatory_fish (whiting, hake, saithe ...) by taxon name | `pred_benthos_split_stdout.txt` |
| `energy_content.py` | energy as eaten per group, weighted by the fish intake | `energy_content_stdout.txt` |
| `benth_encounter_split.py` | copy of section 1 of `../Attackrater ur mötesteori/scripts/benth_encounter.py`, split into two pairs, no stage scores | `benth_encounter_split_stdout.txt` |
| `split_budget.py` | per-taxon stocks per cell (depth map, UWTV strata, trawl map), flows, mass balance, F, M2, r, K tables, equilibrium maps and start totals; variants `central`, `bottomup`, `codnns` | `split_budget_{central,bottomup,codnns}_stdout.txt` |

Downloads in `data/` (each in its own folder): Steimle & Terranova 1985, Ricciardi & Bourget 1998,
Lawson et al. 1998.

---

## 0. Bottom line

### 0a. Groups

| Field | `benthic_community` = infauna | `mobile_epifauna` (new) |
|---|---|---|
| Members (share of today's stock) | 100 m (SK3 without heart urchins): Amphiura/ophiuroids ~65 %, polychaetes ~17 %, bivalves ~12 %, priapulids/nemerteans/others ~4 %, crustaceans and gastropods ~1 % each. 300 m (LYSE 8, by numbers): polychaetes 44 %, small bivalves 44 %, cumaceans/amphipods 5 %, ophiuroids 4 % | Norway lobster 0.60 (4.57 t/km2), crabs + hermit crabs + Munida 0.26 (2.0), northern shrimp 0.073 (0.56), small shrimps 0.039 (0.3), mysids/hyperbenthos 0.026 (0.2) |
| Members (share of what the fish eat) | polychaetes 0.49, brittle stars 0.30, bivalves 0.09, other echinoderms 0.04, other small crustaceans 0.03, other molluscs 0.03, amphipods 0.01 | crabs + hermits + Munida 0.71, northern shrimp 0.14, small shrimps 0.12, mysids 0.02, Norway lobster 0.002 (0.09 with the NNS cod diet) |
| B* today, t/km2 at 100 / 300 m | **110 / 55** (Josefson 1990, resume 207) | **9.45 / 3.80** (LSQ line over the grid; 7.96 / 3.06 bottom-up) |
| B* grid mean today | 92.3 | 7.63 (6.39 bottom-up) |
| `r_max` (/yr), derived g = (r_max + sum M2)/1460 | **1.0** (0.8-1.5) | **0.70** (0.60-0.75; biomass-weighted 0.65) |
| engine r_gross, `growth_rate` per tick | 1.029, 0.000704846 | 0.879, 0.000601734 |
| `energy_content` (MJ/t WW, as eaten) | **2600** (1700-3600) | **3750** (3100-4700) |
| Trawl loss at mean effort (/yr) | **0.136** = kill only, d 0.06 x SAR 2.26 | **0.136** = catch 0.672 + kill of the uncaught 0.365 t/km2/yr over 7.63 |
| `bottom_trawling` biomass_factor at table value 100 (per tick; 0 at 0) | **0.003082** (d 0.03-0.115: 0.00154-0.00591) | **0.003093** (0.0025-0.0055, section 1.2.3) |
| Sum of M2 (/yr) | 0.0291 (benthivores only) | 0.177 (benthivores 0.141, predators 0.036) |
| Equilibrium share 1 - (F + M2)/r_gross | 0.840 | 0.643 |
| `capacity_depth_table` K at 100 / 300 m | **131.0 / 65.5** | **14.78 / 5.95** |
| Box-mean K = `max_carrying_capacity` | **110.0** | **11.94** |
| Engine B*(c) from the table: mean, range | 93.6, 39-127 | 7.88, 2.1-11.7 |
| `inference_initial_biomass` = sum of B*(c) | **337 000 t** | **28 400 t** |
| `initial_biomass_min` / `_max` (+/-25 %) | 252 750 / 421 250 | 21 300 / 35 500 |
| `spawn` | equilibrium | equilibrium |
| `seed_rate`, `movement_speed`, `is_decision_maker` | as today (1e-6, 0, false) | same |

**Infauna d is a choice for the user.** Resume 218.3 kept d 0.10 ("infauna-only d may be ~0.06; d 0.10
kept"). The table uses 0.06 (section 1.2.1). Keeping d 0.10 instead (`split_budget.py ... central 0.10`,
`tables/split_budget_central_dinf0.10_stdout.txt`): F 0.226 /yr, bf100 **0.005137** (today's slope), M2
0.0287, s 0.752, K **146.3 / 73.2**, box K 122.8, engine B* mean 94.7 (26-141), start **341 000 t**
(255 750-426 250); r_max, E and the pairs unchanged.

Optional for `mobile_epifauna`: a six-row K table that keeps today's depth structure under the
effort map (section 1.4): depth_m 81: 14.31, 123: 10.77, 173: 12.27, 221: 17.43, 272: 9.38, 387: 3.75
(start 27 900 t). The two-row table puts too little epifauna at 200-300 m (engine 4.3 vs today 7.9 t/km2
at 200-250 m) and too much below 300 m.

### 0b. Pairs

| Pair | Diet share of the predator's consumption | M2 on the prey (/yr) | assimilation | h = E / ceiling | a (km2/t/tick, per tonne of the whole predator group) | 1/(a h), accessible t/km2 | accessible_fraction | light |
|---|---|---|---|---|---|---|---|---|
| benthivorous_fish -> benthic_community (infauna) | **0.511** (benthos 0.72 x 0.71) | **0.0291** | 0.7 | **95.9** | **0.25** (MC 5-95 % 0.12-0.64) | 0.042 (0.21 in total-stock units) | **0.2** | none |
| benthivorous_fish -> mobile_epifauna | **0.209** (0.72 x 0.29; 0.27-0.36 of benthos) | **0.141** | 0.8 | **138.4** | **0.44** (0.15-1.2) | 0.017 (0.053 total) | **0.31** (0.2-0.45) | none |
| predatory_fish -> mobile_epifauna | **0.072** (benthos 0.11 x 0.65) | **0.036** | 0.8 | **147.1** | **0.33** (0.10-1.5), all members at their own encounter rate; 0.086 if the members' capability scores are folded in | 0.021 | **0.13** (0.06-0.25) | none |
| predatory_fish -> benthic_community (infauna), NOT recommended; values only if checked | 0.039 | 0.0016 | 0.7 if checked | 102 if checked | not derived | - | ~0.01 if checked | - |
| planktivorous_fish -> either | ~0.06 benthos, ~0.04 on mysids/small shrimps | (0.06-0.09 on epifauna if checked) | - | - | - | - | - | - |
| seals, porpoises -> either | not reported (fish diets) | - | - | - | - | - | - | - |

Recommended: check the first three pairs; leave predators -> infauna, planktivores -> both, seals -> both
and porpoises -> both unchecked (section 2). Every checked pair is Type II (default), unmodulated by light.

---

## 1. Derivations: the groups

### 1.1 Members, stocks and depth anchors

**Infauna.** The grab infauna of Josefson (1990) at the in-grid stations, "without large species" (heart
urchins), taken in the lower half of the station range (110 at 100 m, 55 at 300 m; resume 207) [F via
`../Bentos och torskfiskar/evidence_A_standing_stock.md`]. These anchors already were the infauna; section
218 only added the epifauna on top. Composition: SK3 (100 m) biomass ophiuroids (Amphiura) 252.6 g/m2,
echinoids (Brissopsis) 248.1, polychaetes 63.9, bivalves 46.9, priapulids/nemerteans 15.3, crustaceans
4.0, gastropods 3.2 [F via that note]; without the echinoids the shares in table 0a. LYSE 8 (300 m) is a
community of small deposit feeders (shares by number) [F via that note]. The engine's anchors are prey-sized
infauna; the heart urchins stay outside K.

**Mobile epifauna.** Per taxon from `../Bottensamhällets epifauna/` (scripts/combine.py structure,
reproduced in `split_budget.py`):
- Norway lobster: ICES UWTV 2024 stratum densities x 30 g per animal on the per-cell survey share; 4.57
  t/km2 grid mean, flat to ~250 m, ~0 below 300 m [F/D there].
- Northern shrimp: SS3 survey strata 14-16 / q 0.40; 0.56, peak 200-300 m [F/D there].
- Crabs, hermit crabs, Munida: evidence B central 0.9 at 100 m, 0.45 at 300 m (grid mean 0.76), a North
  Sea proxy with a factor ~3 either way [J there]. **Raised to 2.0 t/km2 here by mass balance (section 1.3).**
- Small shrimps (C. allmanni, P. montagui, Processa, Eualus) 0.3 and mysids/hyperbenthos 0.2, flat with
  depth [J there].

Least-squares lines over the 3600 cells (depth clipped to 100-300 m, the table's hold rule) give the B*
anchors 9.45 / 3.80 (bottom-up 7.96 / 3.06 = section 218's 8.0 / 3.1).

### 1.2 Trawl loss

The library's form is a linear `biomass_factor` table in effort, per tick, 0 at 0. With mean effort
E = 3.0142 MW-h/km2/yr and k = 0.75 SAR per MW-h/km2 (`../Bottentrålning och bentos/swept_area.md`):
bf100 = F_mean x 100 / (1460 x 3.0142). Check: section 218's 0.233 /yr gives 0.005294, the library's
0.005292.

**1.2.1 Infauna: kill only, d 0.06.**
- Hiddink et al. 2017 (PNAS 114:8301) otter trawl d 0.06 per pass (0.02-0.16), pooled over the sampled
  community and "primarily ... epi-fauna" (Pitcher et al. 2022) [F via evidence_D].
- The per-pass infauna evidence on mud is at or below that: Sciberras et al. 2018 otter trawl -3 % per pass
  [A/S]; Kaiser et al. 2006 annelids -24 % per multi-pass experiment on mud [F]; chronic otter trawling on
  Nephrops mud lowers infauna biomass (Hinz et al. 2009) [A], Kattegat Nephrops mud shows almost no total
  biomass signal (Sköld et al. 2018) [F via the sister note].
- The 0.10 of today's library is Rijnsdorp et al. 2020's OT_CRU gear-geometry scaling of 0.06, not a
  taxon measurement, and was argued for the pooled community that now leaves. With the epifauna out, the
  infauna takes **d 0.06 (0.03-0.115)** [D/J].
- F = 0.06 x 0.75 x 3.0142 = **0.136 /yr**; bf100 = 0.06 x 0.75 x 100 / 1460 = **0.003082**.
- No catch term: the trawl lands no infauna; its dead discards of echinoderms (~0.008 t/km2/yr of the 0.044
  "other invertebrates") are inside d (resume 218 note).

**1.2.2 Epifauna: catch plus kill of the uncaught part.** Catch at mean effort (t/km2/yr, STECF FDI,
Norwegian ERS, ICES 2025; `../Bottensamhällets epifauna/summary_trawl_removal.md`): northern shrimp 0.336
(landings x 1.12 boiled weight + 8 % dead discards), Norway lobster 0.296 (landings + 75 % dead of discards
17.6 %), crabs and lobster landed 0.006, dead discards of other invertebrates 0.044, of which I give 0.031 to
crabs/hermits/Munida, 0.005 to small shrimps and 0.008 to echinoderms (inside the infauna d) J. Kill of the
animals in the path that are not retained, d_u x SAR (2.26 at mean effort), without counting the caught
ones twice:

| Taxon | stock | catch | d_u (J) | kill | F (/yr) |
|---|---|---|---|---|---|
| Norway lobster | 4.57 | 0.296 | 0.01 (0-0.03): in burrows, the gear takes the emerged ones | 0.103 | 0.087 |
| Northern shrimp | 0.56 | 0.336 | 0.01: escapees through the grid and meshes swim off | 0.013 | 0.62 |
| Crabs, hermits, Munida | 2.0 | 0.037 | 0.05 (0.02-0.10): epifauna d 0.10 (0.06-0.16) with mobile scavengers lower (evidence_D; Bergmann & Moore 2001 short-term mortality 2-25 % [A]) | 0.226 | 0.13 |
| Small shrimps, mysids | 0.5 | 0.005 | 0.02 | 0.023 | 0.055 |
| **Group** | **7.63** | **0.672** | | **0.365** | **0.136** |

bf100 = 0.136 x 100 / (1460 x 3.0142) = **0.003093** (bottom-up stock 0.003195; NNS cod diet 0.003130).
Range: catch 0.39-1.43 and d_u at the ends give F ~0.11-0.25, bf100 ~0.0025-0.0055.

**1.2.3 Limits of the linear table.** It spreads the loss by effort, not by where the fished taxa are, and
it gives every tonne of the group the group rate: the trawl still fishes northern shrimp at F 0.62 and Norway
lobster at 0.087 in reality, the engine at 0.136 on both. Effort is highest at 200-300 m (6.0-6.7
MW-h/km2/yr against 1.4-1.6 above 150 m), where the shrimp ground is.

### 1.3 Predation flows and the mass balance that raises the crab stock

Consumption of benthos (t/km2/yr): benthivores Q/B 3.48 x 1.53 x benthos 0.72 = 3.834
(`../Fiskgrupper i rutan/benthivores/tables/group_diet_consumption.csv`); predators 3.34 x 1.20 x 0.11 =
0.441 (`params_predatory_fish.md`).

**Epifauna share of the benthivores' benthos diet** (`benth_diet_split.py`, ICES stomach DB, size-class
diets weighted by the box length mix for cod and haddock, consumption weights share x Q/B x B):
haddock NNS + trench 2025 0.15, cod Skagerrak + box 2023-25 0.64 (Brachyura 31 %, Pagurus 22 %, Aphrodita 36
% of its benthos), long rough dab 1991 0.36, plaice Skagerrak 2024 0.01, starry ray 0.83, grey gurnard 1.0;
judgement for dab 0.35, witch and lemon sole 0.05, hagfish 0.3, weever 0.6 [F/D/J]. **Group 0.29**
(judgement corners 0.27-0.31; cod from the northern North Sea, where it eats Norway lobster 18 % and
northern shrimp 34 % of its benthos: 0.34). Dorman et al. 1992 (cod, Skagerrak, Feb 1991, 336 stomachs):
crabs and hermits 25.6 % of total stomach mass [S via evidence_B], the same picture. Composition of the
epifauna eaten: crabs/hermits/Munida 0.87, northern shrimp 0.05, small shrimps 0.07, Norway lobster 0.003.

**Predators** (`pred_benthos_split.py`): whiting is the benthos eater; box + Skagerrak all periods 0.61
epifauna (Pandalus 0.44 of the benthos, Nereididae and bivalves the infauna), 2022 0.74, all areas 2022-24
0.71; hake 0.99 (Pagurus), saithe one fish of Crangon. **0.65** (0.55-0.75); composition J: northern shrimp
0.5, small shrimps 0.3, mysids 0.1, crabs 0.1.

**Flows** (central): benthivores 2.722 on infauna and 1.112 on epifauna; predators 0.154 on infauna and
0.287 on epifauna. By epifauna taxon: crabs 0.998, northern shrimp 0.200, small shrimps + mysids 0.196,
Norway lobster 0.003.

**Mass balance per taxon.** Production must cover what the fish eat plus the catch:
B (EE x P/B - d_u SAR) >= C + catch, EE 0.9 (Ecopath convention) [D]:

| Taxon | stock today | C + catch | P/B (literature) | B_min |
|---|---|---|---|---|
| Norway lobster | 4.57 | 0.30 | 0.3-0.4 (M&D 0.37; Z 0.26-0.36) | 0.9-1.2, met |
| Northern shrimp | 0.56 | 0.54 | 1.2-1.7 (Z from WKPRAWN M + F; Hopkins 1.7) | 0.36-0.51, met |
| Crabs, hermits, Munida | **0.76** | 1.04 | 0.4-0.9 (Brey 2001 runs: hermits 0.66-0.70, Munida 0.42-0.54, Atelecyclus/Goneplax 0.44-0.46, L. depurator 0.37, Pisidia 0.94; M&D large crabs 0.55) | **1.5-4.2, not met** |
| Small shrimps, mysids | 0.5 | 0.20 resolved + ~0.5 planktivores (J) | 1.4-2.5 | 0.32-0.58, met |

The bottom-up crab stock (a North Sea proxy, x3 either way, no in-area data) cannot produce what the fish
are measured to eat. Measured rations x biomass x diet are firmer than the proxy, so **the crab stock is
raised to the mass-balance value at P/B 0.7: 2.0 t/km2** (x 2.65; inside evidence B's high bracket ~2.3)
[D]. Its depth shape (2:1 from 100 to 300 m) is kept. The `bottomup` variant keeps 0.76: the engine would
still run (the derived growth pays the M2), but the crabs would need a gross production of 2.2 /yr, 3-5x
their P/B.

### 1.4 r, equilibrium share and K

The library's NDM convention (`lib/world/growth_budget.py`): `r_max` is the net growth after natural
mortality and the predation of groups the model does not resolve, and the derived growth adds the checked
pairs' M2: r_gross = r_max + M2. Today's equilibrium share is s = 1 - (F + M2)/r_gross = (r_max - F)/(r_max +
M2). Field r values (Hiddink et al. 2019 recovery after trawling, r = 5.31/longevity; assessment surplus
production) are net of the predation they had, so they are `r_max` here.

| Taxon | B* | F | M2 | r_max | r_gross | s | K |
|---|---|---|---|---|---|---|---|
| Infauna | 92.3 | 0.136 | 0.031 | 1.0 (0.8-1.5; resume 207) | 1.031 | 0.838 | 110.1 |
| Norway lobster | 4.57 | 0.087 | 0.0007 | 0.34 (0.25-0.5) | 0.341 | 0.741 | 6.17 |
| Northern shrimp | 0.56 | 0.62 | 0.358 | **1.2** (0.85-1.33) | 1.558 | 0.370 | 1.51 |
| Crabs, hermits, Munida | 2.0 | 0.132 | 0.499 | 0.9 (0.6-1.24; brachyurans 0.8, hermits 1.0, Munida 0.6) | 1.398 | 0.550 | 3.64 |
| Small shrimps, mysids | 0.5 | 0.055 | 0.392 | 1.86 (shrimps 1.5, mysids 2.4) | 2.252 | 0.802 | 0.62 |

- Northern shrimp r_max 1.2 instead of evidence A1's 0.85: today's F ~0.5 is ~0.8 F_MSY with the stock above
  MSY B_trigger (ICES 2025 [F via A1]); a Schaefer stock in that state needs r >= F/(1 - B/K) ~1.0-1.25, and
  the longevity rule gives 1.33. The SPM r 0.33-0.53 is prior-driven (A1). J.
- **Group r_max.** One logistic reproduces the sum of the taxon K only at r_max = (F_g + M2_g)/(1 - B*/sum K)
  - M2_g = **0.70** (bottom-up 0.69, NNS cod 0.65); the biomass-weighted mean is 0.65 (section 218's 0.55
  was with northern shrimp at 0.85 and the crabs at a third of the stock). K tables are built from the group
  values: K(c) = today(c) / s_g, least squares in depth.
- **M2 iteration.** M2 is recomputed once on the engine's own equilibrium map (resume 218 practice): 0.0291
  on the infauna (B* mean 93.6), 0.1409 + 0.0364 on the epifauna (B* mean 7.88).
- **Start totals** (sum over the 3600 cells of K_table(depth) x max(0, 1 - (F(c) + M2)/r_gross), F(c) =
  1460 x slope x E(c) from `maps/fishing_bottom_otter_trawls.npz`): infauna 337 001 t, epifauna 28 374 t. No
  cell at zero (effort max 9.6, F max 0.43 on both).
- By depth (today / engine B*): infauna 0-100 m 110 / 118, 200-250 m 77 / 62, > 300 m 55 / 57; epifauna 0-100
  m 10.3 / 10.6, 200-250 m 7.9 / 4.3, > 300 m 2.5 / 4.0. The optional six-row table (0a) brings the
  epifauna within 0.3-0.7 t/km2 of today in every band; an equivalent six-row infauna table (81: 121.8, 123:
  114.1, 173: 113.0, 221: 113.1, 272: 88.6, 387: 63.5) exists in the stdout if wanted.
- Variants (`tables/split_budget_*_stdout.txt`): bottom-up crabs K 13.18 / 5.06, start 23 900 t, M2 0.211, r_max
  0.69; NNS cod diet K 15.06 / 5.96, start 26 700 t, M2 0.214, r_max 0.65.

### 1.5 Energy content as eaten

kJ/g WW = AFDW/WW (Ricciardi & Bourget 1998, Table 1 medians [F]) x kJ/g AFDW (Steimle & Terranova 1985
Table 1 [F]), checked against Steimle & Terranova's appendix wet values [F], plus Lawson et al. 1998 for
northern shrimp [F]. Wet weights of molluscs and echinoderms include shells (as in grab biomass and in
stomach contents).

| Taxon | kJ/g WW (low-high) | basis |
|---|---|---|
| Polychaetes | 3.4 (2.4-4.4) | 0.156 x 21.85; Aphrodita 2.4; appendix median ~4.2 |
| Brittle stars | 2.0 (1.0-3.0) | Ophiura-type whole 0.065 x 16.2 = 1.05; Amphiura filiformis AFDW 13.7-17 % WW (Rumohr et al. 1987 via evidence_A) x 19.1 = 2.6-3.2; arms are cropped |
| Heart urchins / other echinoderms | 0.9 (0.6-2.0) | 0.027 x 25.8; appendix 0.6-2.0 |
| Bivalves, whole | 1.2 (0.8-2.2) | 0.055 x 21.7; Nucula whole 2.2 |
| Amphipods / isopods, cumaceans | 3.4 / 2.9 | 0.165 / 0.142 x 20.7; cumaceans 1.6 |
| Crabs, hermits, Munida | 3.6 (2.8-4.5) | decapods 0.165 x 20.7 = 3.4; Cancer 3.7, Hyas 2.8, Libinia 5.4, Pagurus shell-free 6.0-6.9 |
| Northern shrimp | 4.7 (4.4-5.2) | Lawson et al. 1998: whole 4.72, females 4.57, males 5.19 |
| Norway lobster | 4.0 (3.4-4.8) J | decapod mean 3.4, Homarus 4.8 |
| Small shrimps / mysids | 3.7 / 3.1 | Crangon 3.7, Dichelopandalus 5.5 / 0.155 x 20.7, Neomysis 2.9 |

Weighted by the fish intake: **infauna 2580 -> 2600 MJ/t** (1700-3600), **epifauna 3760 -> 3750 MJ/t**
(3100-4700; stock-weighted 3910). Check: 0.71 x 2580 + 0.29 x 3760 = 2920, the pooled 3000 of today. The
infauna value sits on the brittle-star and polychaete energies; the epifauna on the crabs.

---

## 2. Derivations: the pairs

M2 = Q/B x B_pred x diet share / B*_prey (B* = the engine's equilibrium-map mean). h = E_prey / ceiling,
ceilings benthivores 27.1, predators 25.5 MJ/t/tick (section 148 energy convention).

### 2.1 benthivorous_fish -> infauna and -> mobile_epifauna

- Diet: benthos 0.72 of the consumption (0.62-0.73 over the diet variants), of which 0.29 epifauna (1.3).
- M2 infauna 2.722 / 93.6 = **0.0291**; epifauna 1.112 / 7.88 = **0.141** (bottom-up 0.168, NNS cod 0.176).
- h infauna 2600 / 27.1 = **95.9**; epifauna 3750 / 27.1 = **138.4**.
- Assimilation 0.7 on infauna (today's value; skeleton and shell) and 0.8 on epifauna (the library's
  crustacean value on predators -> zooplankton; Ecopath unassimilated 0.2) J.
- **Attack rates** (`benth_encounter_split.py`): 2-D swept area v x w x p x 21 600 s per fish, summed per
  tonne over the 1-cm length classes of the long-run GOV catch, members weighted by biomass only (the stage
  scores of the original are dropped with predator_share).
  - Infauna: w and p as in the original (gadoid contact path 0.15 L, capture 0.5; flatfish visual 0.6 L,
    0.7). **a 0.25** (MC 5-95 % 0.12-0.64); was 0.26 pooled.
  - Epifauna: the original note found the measured diet needs an effective a on epifauna ~4x that on
    infauna, from detection width: moving shrimps and crabs are felt by the lateral line, smelt or seen at
    short range, a siphon only by contact. Gadoids, rays, gurnard w 0.6 L (0.3-1.2; R 0.3 L) J, flatfish
    0.8 L (0.3-1.5) J, capture 0.3-0.4 (original). **a 0.44** (0.15-1.2).
  - Both saturated at today's accessible stock (x = a h f B 440 and 143; 0.998 and 0.993 of the ceiling).
    The epifauna pair is still at 0.59 of the ceiling at 1 % of today's accessible stock. **So a does not
    set the diet split; the policy's eat shares and the depletion point do.** Unsaturated, the intake ratio
    per eat share would be 0.23 epifauna/infauna against the measured 0.41.
- **accessible_fraction.** Infauna 0.2 (resume 208, kept). Epifauna by taxon (evidence A1, A2, B): Norway
  lobster 0.10 (0.09 visible per burrow at peak emergence, Aguzzi et al. 2021), northern shrimp 0.7,
  crabs/hermits/Munida 0.6 (Munida 0.3, hermits 0.8), small shrimps/mysids 0.6, weighted by the stock:
  **0.31** (0.2-0.45). The benthivores (cod) take every taxon, Norway lobster included.
- Light: none, see 2.5.

### 2.2 predatory_fish -> mobile_epifauna

- Diet: benthos 0.11 of the consumption (evidence C), 0.65 of it epifauna: **0.072**; M2 = 4.00 x 0.072 /
  7.88 = **0.036**.
- h = 3750 / 25.5 = **147.1**; assimilation 0.8.
- **accessible_fraction 0.13** (0.06-0.25): what these fish take is exposed small prey: northern shrimp
  0.56 x 0.7, small shrimps/mysids 0.5 x 0.6, small crabs ~a quarter of the crab stock x 0.6; no Norway
  lobster (burrows, size) J. 0.99 t/km2 of 7.63 (the predator note's "edible exposed bed prey" was 1.0).
- **Attack rate.** The library 0.030 is NOT this pair's value as it stands: it was derived on the pooled
  accessible benthos 20.3 t/km2 with the edible exposed share s = 0.05 inside a, and per tonne of the
  phi = 0.145 share (`../Attackrater ur mötesteori/tables/pred_encounter_stage_detail.csv`). On a Holling
  biomass that is all edible (f x B_epi), from the same per-member a [D]:
  - all members at their own encounter rate, per tonne of the whole group (no scores): **0.33** (MC
    ~0.10-1.5; juvenile whiting 1.6, adult whiting 0.40, saithe 0.05-0.12 per edible tonne);
  - folding in the members' capability scores (what phi was built from): 0.086;
  - per tonne of the capable share: 0.59.
  Saturated in every case: x = a h f B = 48 (0.98 of the ceiling) at 0.33, 12.6 at 0.086. I give 0.33 because
  the brief removes phi everywhere; see section 3 for what that implies.
- Light: none (night/day 0.99 in the predator note).

### 2.3 Pairs not to check

- **predatory_fish -> infauna.** Whiting's benthos is 0.3-0.4 infauna (Nereididae, bivalves, Ensis, Annelida;
  `pred_benthos_split_stdout.txt`), i.e. 0.039 of the group's consumption and M2 0.0016 on the infauna.
  The pair is real but negligible for the prey, and the predator note found the exposed share these
  non-digging fish can reach ~0.01 of the infauna. A checked, saturated pair would pay the whole predatory
  group its full ceiling on infauna at h 102 (energy per eat share is the same for every saturated pair),
  which the diets do not show. Leave unchecked; the 4 % of the diet goes to the other pairs or is lost.
- **planktivorous_fish -> either.** Benthos ~6 % of the group's ~16.5 t/km2/yr (pout 0.3 of its diet, blue
  whiting 0.12, herring 2-5 %; `params_planktivorous_fish.md`), ~0.5 t/km2/yr of it mysids and small shrimps
  J: M2 ~0.06-0.09 /yr on the epifauna group if checked. A checked pair would let the 79 % herring group feed
  at the ceiling on the bottom. Leave unchecked: the flow is unresolved predation inside the small taxa's
  r_max (field P/B 1.4-2.5 covers it: B_min 0.32-0.58 against 0.5).
- **seals, porpoises.** Their Skagerrak diets are fish (`params_top_predators.md`: otolith and hard-part
  studies split over planktivores, benthivores and predators; the 11-27 % "not in a pool" is named fish such
  as wrasses, blennies, eel and sandeel). Benthic crustaceans are not reported as a diet item there; any
  share is small J. No pair.

### 2.4 Diet check across the two benthivore pairs

The group eats 2.72 t/km2/yr of infauna (0.031 of its stock per year) and 1.11 of epifauna (0.14 per year,
but 0.5 per year on the crabs that make up 87 % of it). Selectivity ~28-34 % epifauna in the diet against
8 % of the reachable stock (0.31 x 7.6 / (0.31 x 7.6 + 0.2 x 92) = 0.11 here) is now two pairs, so the
epifauna can be depleted while the infauna is not.

### 2.5 Light

f is a time-averaged exposure (Norway lobster: visible share at peak emergence and below; northern shrimp:
males partly off the bottom at night). A dark_ratio on top of it would count the same effect twice. The
evidence does not give one sign either: Norway lobster emergence shifts from dusk/dawn above 50 m to midday
at 340-570 m (Aguzzi et al. 2021 [F via A2]), northern shrimp are more available by day (6.3x in day trawl
catches elsewhere, Frechette et al. 1983 [S]; no day/night effect in the Skagerrak winter survey), while
gadoids feed on bottom prey at least as much at night (Rindorf 2003 [A]; Brawn 1969 [A]) and the bottom at
100-300 m is below ~0.01 lux even at noon (benth_encounter stdout). **No dark_ratio / threshold on any of
the pairs.**

---

## 3. Conflicts with the engine's conventions, uncertainties, judgements

1. **Removing predator_share leaves the diet split to the policy.** Every pair here is saturated at today's
   stock, so at saturation each pays its ceiling per eat share (energy per eat share differs only by
   assimilation). Without phi nothing but the learned eat shares and the depletion of the accessible pool
   (1/(a h) 0.017-0.04 t/km2 accessible) stops the predatory group from feeding entirely on epifauna, or
   the benthivores on either. The library 0.030 on predators -> benthos cannot be moved over unchanged
   (section 2.2).
2. **r_max versus hand-set growth.** The values assume each group carries `r_max` and the loader derives
   growth_rate = (r_max + sum M2)/1460 (as for the other NDMs). `benthic_community` is hand-set today
   (resume 136). `lib/world/habitat.build` reads `spec["growth_rate"]` for the equilibrium map: it must
   see the derived value, or the hand-set growth_rate must be written as in table 0a (0.000704846 and
   0.000601734).
3. **One logistic for the epifauna.** Norway lobster is 60 % of the stock and 0.2 % of the diet; crabs are
   26 % and 71 %. The group's single r and K let the fish eat "epifauna" at the group's rate, so in the
   engine the crabs' depletion is spread over the Norway lobster; the trawl's selectivity (northern shrimp
   F 0.62, Norway lobster 0.087) is still not held. Two pairs fix the infauna/epifauna selectivity only.
4. **Crab stock from mass balance.** Raised x 2.65 above the only (proxy) estimate. It rests on the
   benthivore biomass 1.53 t/km2 (top of the long-run span, user's choice), Q/B 3.48, and two cod diet sets
   (2023-25, n ~100; Dorman 1991, n 336). If the ICES records weigh hermit crabs with their gastropod shell,
   the crab flow is overstated. An in-area crab/hermit/Munida biomass (IMR shrimp-survey by-catch, DATRAS
   invertebrates, Zühlke et al. 2001) is the missing measurement.
5. **Aphrodita aculeata** (sea mouse) is classed infauna by the brief's rule but lives on the mud surface; it
   is 36 % of the Skagerrak cod's benthos. Moved to the epifauna it would raise the group epifauna share to
   ~0.36 and make the accessible fraction of the infauna for cod higher than 0.2.
6. **Spatial limits.** The two-row K table plus an effort-linear loss misplace the epifauna by depth (1.4);
   use the six-row table if the depth structure matters. The EMODnet effort may be low (Danish VMS missing,
   resume 218): every slope calibrated to it would then be high per unit effort, the totals unaffected.
7. **New group:** observation channels (Observability matrix) and the FG set change, so existing checkpoints
   become invalid; all `mobile_epifauna_*` and `*_preys_on_mobile_epifauna` / `*_observes_*` entries and
   the impact entries (`mobile_epifauna_impacted_by_bottom_trawling`; none for pelagic trawling, rotor,
   noise) need adding.
   Tests that pin today's numbers will fail and need the new values: `tests/test_habitat.py` (manifest
   start = sum of B*, the benthos K map and spawn pattern) and `tests/test_fgconfig_depth_map.py` (K rows
   164.5 / 81 and the preview mean 137.7), per resume 218.3-218.4.
8. **Knock-on to the predators' own derived growth.** `growth_budget.ideal_intake` fills the eat share over
   the checked pairs in order of energy per unit eat share; with phi gone the best pair alone sets s*. For
   the benthivores that becomes the epifauna pair (assimilation 0.8 x 27.1 against 0.7 x 27.1 on
   today's pooled pair), so s* and their derived g move, and the same for the predators. Expect
   `tools/probes/budget_gate.py` output to shift when the pairs are entered (not derived here).
9. **Judgements that matter most:** the crab P/B 0.7 (0.4-0.9 moves the crab stock 1.5-4.2), the epifauna
   detection width (a only, saturated), d_u on crabs (F 0.11-0.18), northern shrimp r_max 1.2, the predator
   epifauna composition, infauna d 0.06.
10. **Seasonality** (whiting's Pandalus is pre-2000 and spring; haddock's diet is Q1 2025) is not resolved.

## 4. References

- Aguzzi J. et al. 2021. Sci. Rep. 11:5797, doi:10.1038/s41598-021-85240-3 [F via evidence_A2].
- Bergmann M., Moore P.G. 2001. ICES J. Mar. Sci. 58:163 [A via evidence_C].
- Brey T. 2001. Population dynamics in benthic invertebrates (virtual handbook), runs via evidence_B [S].
- Dorman J.A. et al. 1992. ICES CM 1992/G:70 [S via evidence_B].
- Hiddink J.G. et al. 2017. PNAS 114:8301; 2019, J. Appl. Ecol. (longevity rule) [F via sister notes].
- Hinz H., Prieto V., Kaiser M.J. 2009. Ecol. Appl. 19:761 [A].
- ICES 2025. Advice pra.27.3a4a, nep.fu.3-4 [F via the epifauna notes].
- ICES stomach content database (DATRAS), extracts in `../Fiskgrupper i rutan/benthivores/tables/` and
  `predators/tables/` [F data].
- Josefson A.B. 1990. Mar. Ecol. Prog. Ser. 66:117 [F via `../Bentos och torskfiskar/`].
- Kaiser M.J. et al. 2006. Mar. Ecol. Prog. Ser. 311:1 [F via evidence_D].
- Lawson J.W., Magalhães A.M., Miller E.H. 1998. Important prey species of marine vertebrate predators in the
  northwest Atlantic: proximate composition and energy density. Mar. Ecol. Prog. Ser. 164:13-20 [F]
  (`data/lawson/`).
- Mackinson S., Daskalov G. 2007. Cefas Sci. Ser. Tech. Rep. 142 [F] (P/B and group definitions, Table 11.6,
  11.9; `../Bottensamhällets epifauna/data/epifauna/tech142.pdf`).
- Ricciardi A., Bourget E. 1998. Weight-to-weight conversion factors for marine benthic macroinvertebrates.
  Mar. Ecol. Prog. Ser. 163:245-251 [F] (`data/ricciardi/`).
- Rindorf A. 2003 (whiting diel feeding) [A via the encounter notes].
- Sciberras M. et al. 2018. Fish Fish. 19:698 [A/S].
- Sköld M. et al. 2018. Mar. Ecol. Prog. Ser. 586:41 [F via sister note].
- Steimle F.W., Terranova R.J. 1985. Energy equivalents of marine organisms from the continental shelf of the
  temperate Northwest Atlantic. J. Northw. Atl. Fish. Sci. 6:117-124 [F] (`data/steimle/`).
- Sister notes reused: `../Bottensamhällets epifauna/`, `../Bentos och torskfiskar/`,
  `../Fiskgrupper i rutan/` (benthivores, predators, planktivores, top predators),
  `../Attackrater ur mötesteori/`.
