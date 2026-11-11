# The benthivores' vertical search profile, and the final rule A + band-overlap table (2026-11-11)

Research subagent; saved by the main agent from the subagent's report. Nothing in the library, code, tests,
manifest, resume or AGENTS.md changed; no training or rollout; no model output read. No profile, window or p was
chosen to hit 0.073, 0.012, 0.027, 0.36 or any other booked or field share. Temporary library copies only in the
session scratchpad. Tags: [S] source, [D] derived here, [J] judgement.

## Files

- `scripts/benth_profile.py` -> `tables/benth_profile_stdout.txt`, `tables/benth_profile.json`: member profiles
  f_m(z), nine variants, every benthivore pair under the band theorem (krill, planktivores, whiting, infauna,
  epifauna, Nephrops). Execs `../En synregel for fiskparen/scripts/one_rule_pairs.py` up to its run block (krill
  machinery, section 253 light, section 262 layer, rule A, gmat) and runs `../Bottenfiskens bytesbalans/scripts/
  pout_vs_bed.py` quietly (243.3 members and windows, benthic member weights).
- `scripts/wc_benth_window.py` -> `tables/wc_benth_window_stdout.txt`, `.json`: re-weights the water column
  (WC) -> benthivores stage_access window (execs `../Fönsterviktning/scripts/aw_stage_access.py`).
- `scripts/final_gate.py` -> `tables/yaml_edits.txt`, `tables/final_gate_stdout.txt`, `tables/final_gate.json`,
  `tables/gates/budget_gate_<variant>.txt` (12 runs): final table, YAML block, budget_gate on temporary copies,
  loader check.
- `papers/README.md`: one PDF (Hunter et al. 2009, doi:10.3354/meps08213; MEPS, "resale or republication not
  permitted"); one Unpaywall record without PDF (Boudreau 1992). Never committed.

```
S=<session scratchpad>
python3 -I scripts/benth_profile.py <repo> $S/tmp tables/benth_profile.json          # ~15 s
python3 -I scripts/wc_benth_window.py <repo> $S/tmp                                   # ~1 s
python3 scripts/final_gate.py <repo> tables/benth_profile.json $S/libs tables        # ~10 s (imports lib/, so no -I)
```

## Findings

1. **[D] The band theorem cuts the benthivore fish pairs only ~x0.87-0.90, not the x0.3-0.6 of "Overlapp i
   fiskparen" finding 11.** (a) Off-bed time is not lost under the integral: a cod 20 m off the bed still meets
   herring, pout's upper part and the krill layer; a band time share is right only if the prey were uniform in
   the band and absent above it. (b) The central cod profile is anchored on the North Sea stock (cod.27.47d20
   includes the Skagerrak): Hobson et al. 2007 (92 DSTs) SB 70 %, SW 20.5 %, midwater + continuous search 9.5 %;
   the Barents numbers come from capelin-feeding NEA cod. The Barents profile for every gadoid ("deep") gives
   x0.59 (planktivores), x0.54 (whiting). **The North Sea vs Barents split is the decisive judgement.**
2. **[D] Krill: 0.013 -> 0.0068 (x0.52)**: rule A on the old band x0.727, the profile x0.72. The < 100 m class
   falls (0.0325 -> 0.0225 at rule A: long rough dab (LRD) bed-bound, cod and 20-34 cm haddock 70 % in the bottom
   2 m, below most krill); the 150-250 m class rises (0.00043 -> 0.00070: haddock >= 35 cm rise into the day
   layer). Check, not input: Albert 1994 found summer euphausiids in Norwegian Deep haddock (FO 18-56 % by
   length). Night/day 0.036.
3. **[D] Planktivores 0.0198 -> 0.0214** (vision x1.247 x profile x0.865; pout 0.85 / herring 0.13 of a, cod 0.68
   / haddock 0.32; night/day 0.141, library 0.06). **Whiting 0.021 -> 0.0226** (x1.188 x 0.904; cod 0.94 of a;
   night/day 0.168, library 0.15).
4. **[D] Benthic pairs: bed-contact share of search time (0-2 m) tau 0.71 (infauna), 0.74 (epifauna), 0.70
   (Nephrops)**; by member flatfish and LRD 1, cod 0.70, haddock 0.56 (size-weighted); bracket 0.43
   ("deep_rest50") - 0.81 ("sw_contact"). Implied a: infauna 0.036 -> 0.0256, epifauna 0.15 -> 0.110, Nephrops
   0.44 -> 0.308; night/day ~1.0 (no dark_ratio).
5. **[D] The benthic scaling changes the diet, not the intake.** With fixed attack (265.1) and shared handling
   (242) the diet is the encounter shares a_j B_vis_j; saturation no longer decouples a from the diet. Bed a's
   at full time while the fish and krill pairs carry the profile credit each benthivore with a search budget of
   1 + its off-bed share (~x1.36 on every non-bed share). Final vs final_nobenthic, encounter / booked share:
   krill 0.024 / 0.027 (0.91) vs 0.018 (0.67); planktivores 0.045 / 0.073 (0.61) vs 0.033 (0.45); whiting 0.004
   vs 0.003. Total intake moves < 1 %; infauna a h B_vis 75 -> 54, saturation 0.987 -> 0.982.
6. **[D] WC -> benthivores window [0.42, 0.12] -> [0.392, 0.0388]** (f_eff 0.165 -> 0.092), re-weighted on the
   band a's member shares (saithe 0.892, pollack 0.007 with saithe's window, hake 0.051, anglerfish 0.015, ling
   0.025 with anglerfish's window [J]). The a(L) exponent is derived from the band a: saithe 43 / 70 cm carry
   0.601 / 0.291 of the a at stage weights 0.259 / 0.151; per-tonne ratio 1.20 gives a band-A exponent -0.38 vs
   -2.11 in the 220 form, so kexp +1.7 (+1.5 from the per-stage 2.55). Bracket [0.349, 0.0099] (Hs 5 m, ling
   excluded, k +1.5) to [0.461, 0.0868] (Hs 40 m, ling as anglerfish, k +2).
7. **[D] Gate with the full final table.** Predator side OFF: only whiting -> benthivores (0.12 equal cells,
   0.07 field; flagged open, user decision) and the pre-existing whiting -> mobile_epifauna at the field spread
   (0.23). WC (field in brackets): krill 1.39 (1.40), planktivores 0.78 (0.76), benthivores 0.97 (1.13), whiting
   0.64 (0.46), zooplankton 0.38 (0.62), mackerel 0.02 (not judged). Benthivores: infauna 1.03, epifauna 1.01,
   Nephrops 2.19, krill 0.91, planktivores 0.61, whiting 0.31 (share < 0.05, not judged). WC krill share by bed
   class (< 100 / 100-150 / 150-250 / >= 250 m) 0.41 / 0.59 / **0.49** / 0.06 (Overlapp 0.45, library 0.74;
   booked 0.36; field saithe all seasons 0.19, group Q3 0.47). Prey side (implied / booked M2, equal cells):
   krill 1.04; planktivores 0.69; benthivores 0.62 (whiting pair 0.12 OFF, WC 0.97); whiting 0.60 (benthivores
   pair 0.31 OFF, the section 251 residual, x1.24 better than Overlapp); mackerel 0.02 (OFF); infauna 1.03;
   epifauna 0.95; Nephrops 2.19.
8. **[D] The loader loads the final library** (`load_project_config("mareld2.yaml", library_path=<copy>)`): 15
   groups, 0 warnings, no refusal (no open pool at g >= exp(k) - 1); every derived growth_rate identical to the
   live library's (a does not enter g).
9. **[D] Pout's near-bed day layer (Albert 1994; sensitivity, not a proposal)**: pout within 1.5 m of the bed by
   day raises the planktivore pair x2.5 (a 0.062), the benthivore planktivore share to 1.62 of booked (ok),
   night/day 0.014; overstated (0-2 m predator slab vs 0-1.5 m prey slab without slab clipping, R 0.6-1.2 m).

## Member profiles [S/D/J]

Three components by height above the bed: bed (0-2 m, uniform), near-bed (2-10 m, uniform), off-bed (> 10 m,
exponential with scale H, truncated at the surface); day/night at sun elevation 0 deg. Group values
biomass-weighted over members. Member weights per pair: krill - haddock 0.386 (by size), cod 0.314, LRD 0.126,
gurnard 0.009, other flatfish 0; planktivores - the 243.3 rebuild (cod >= 20 cm W 0.310, haddock >= 20 cm W
0.353 x c_h 0.65, per-species windows and pool shares); whiting - section 251 membership (cod >= 20 cm 0.308;
rays and gurnard 0.024 x 0.5); benthic - a_enc x e per species from pout_vs_bed section D (infauna: haddock
0.517, cod 0.183, LRD 0.118, dab 0.070, plaice 0.045, other 0.067; epifauna: haddock 0.430, cod 0.229, LRD 0.135,
dab 0.080, plaice 0.052, other 0.074; Nephrops: cod 1.0 [J], bracket cod 0.8 / haddock 0.15 / LRD 0.05 gives
x0.69).

Central profiles (within 2 m / within 10 m / mean height m, day | night; tau_bed):

| member (weight) | central | basis |
|---|---|---|
| cod (0.314) | 0.70/0.91/5 \| same; tau 0.70 | Hobson et al. 2007 all areas (SB 70, SW 20.5, MW+CS 9.5 %; > 55 % within 5 m; mean height up to 35.8 m per day, H 20 m [J]); no diel difference [J] |
| haddock < 20 cm (0.086 of haddock) | 0.60/0.95/4 \| 0.30/0.60/12; tau 0.45 | Aglen et al. 1997: small haddock near the bottom by day, lifted at night to 30-40 m |
| haddock 20-34 cm (0.462) | as cod; tau 0.70 | Aglen 1997: medium groups "fairly stable" in bottom catches [J: cod-like] |
| haddock >= 35 cm (0.452) | 0.30/0.50/22 \| 0.50/0.80/7; tau 0.40 | [J] midway between the Barents reading (Aglen 1997 up to 100 m by day; Engas 1988 up to 150 m) and a bed-bound one (Boudreau 1992: 50 % < 2.5 m) |
| LRD (0.126), dab, plaice, witch, lemon sole, starry ray | 1.0 in 0-2 m; tau 1 | Hunter et al. 2009: plaice rarely leave the bed in summer, < 5 h/d midwater only on migration; others by analogy [J] |
| grey gurnard (0.009) | as cod [J] | |

Variants (`benth_profile_stdout.txt` section 1): **deep** - Barents profiles for every gadoid (cod 0.19 / 0.46
day, 0.34 / 0.57 night within 2 / 10 m, Stensholt & Stensholt 2004, Hjellvik et al. 2003 Table 6; H 50 m day from
Hjellvik Table 4; haddock >= 35 cm 0.15 / 0.30 day, 0.41 / 0.75 night, Engas 1988, Aglen 1997); **bedbound** -
Hobson's Jutland Bank cod (SB 63.9, SW 35.9, MW 0.2 %) and Boudreau-like haddock (lower tau 0.64 than central,
SW goes to 2-10 m); **sw_contact** - Hobson's SW counted as bed contact (cod tau 0.905); **rest50 / deep_rest50**
- half of the gadoids' 0-2 m occupancy counted as rest; **flat_lrd_off** - flatfish 0.9 and LRD 0.7 at the bed;
**had_barents / cod_barents** - one species at a time.

### Evidence

| species | quantity | value | area, method | source | tag |
|---|---|---|---|---|---|
| cod | behaviour classes | SB 70 %, SW 20.5 %, MW+CS ~10 %; > 55 % within 5 m; Jutland Bank SB 63.9 / SW 35.9 / MW 0.2 %; May-Sep > 95 % near the bed (S North Sea) | North Sea, 92 DSTs, 11 007 days | Hobson et al. 2007, doi:10.3354/meps347101 | [S] |
| cod | midwater vs seabed | 43 / 57 % overall; resident periods 17 % midwater, 83 % near the bed (69 % strongly associated) | Channel / North Sea, 14 tags | Hobson et al. 2009, doi:10.3354/ab00144 | [S] |
| cod | seasonal switching | on the bed Jun-Jul; leave it at nightfall Aug-Sep; midwater Apr-May | North Sea tags | Righton et al. 2000, ICES CM 2000/Q:09 | [S] |
| cod | within 2 m of daily max depth | 28.9 % (day 18.7, night 34) | Barents, winter, beds > 200 m | Stensholt & Stensholt 2004 | [S] |
| cod | > 10 m above the bed | 13-87 % by tag and season | same | same, Table 2 | [S] |
| cod | acoustic share 0-10 m | 17-91 % by survey (winter 25-48 %); winter large fish 0.46 day / 0.57 night; > 10 m profile declines to ~100 m | Barents 1995-2002; dead zone excluded | Hjellvik et al. 2003, doi:10.1016/s1054-3139(03)00116-4 | [S] |
| cod | all sizes on the bottom day and night | qualitative | Barents station | Aglen et al. 1997, ICES CM 1997/Y:16 | [S] |
| haddock | diel layer | >= 35 cm from the bottom up to 100 m by day, down at night; < 20 cm the reverse | Barents station | Aglen et al. 1997 | [S] |
| cod + haddock | 0-2 m share of the lowest 22 m | day 0.24 / night 0.41 at 350 m; up to ~150 m above the bed by day; haddock migrates far more than cod | Barents | Engas, Jacobsen & Soldal 1988, ICES CM 1988/B:32 | [S] |
| cod + haddock | share in the bottom 10 m | two peaks per day, tidal | Barents, 267-290 m | Michalsen et al. 1996, doi:10.1006/jmsc.1996.0054 | [S] |
| haddock | 50 % within 2.5 m | spawning aggregation, Scotian Shelf | acoustics, 24 h | Boudreau 1992, doi:10.1139/f92-003 | [S, abstract via search index, unverified] |
| haddock | diet | summer euphausiids, decapods, ophiuroids, teleosts; winter polychaetes, bivalves | Norwegian Deep incl. Skagerrak 1984-87 | Albert 1994, doi:10.1006/jmsc.1994.1004 | [S] (diet only) |
| plaice | midwater swimming | rarely leave the bed in summer; > 5 h/d only Dec-Jan (west) | North Sea, 139 tags | Hunter et al. 2009, doi:10.3354/meps08213 | [S] |
| demersal NW Atlantic | haddock sometimes far off the bed; lower night catches | qualitative | Canadian banks | Beamish 1966, doi:10.1139/f66-009 | [S, search summary] |

Gaps: no haddock height-above-bed time budget for the North Sea or Skagerrak (Hjellvik 2003 has no haddock);
every source measures occupancy, not search; tags take the daily maximum depth as the bed and resolve 1.3-2 m;
0-2 m used as "bed contact" by that resolution [J].

## Per-pair results

Central (brackets over deep, bedbound, rest50, deep_rest50, sw_contact):

| pair | a library -> new | bracket | class a (< 100 / 100-150 / 150-250 / >= 250 m) | night/day |
|---|---|---|---|---|
| benthivores -> krill | 0.013 -> 0.00679 | 0.0062-0.0076 | 0.0225 / 0.00583 / 0.000701 / 1.5e-5 | 0.036 |
| benthivores -> planktivores | 0.0198 -> 0.0214 | 0.0134-0.0234 (pout_albert 0.062) | 0.0401 / 0.0262 / 0.0114 / 0.00685 | 0.141 |
| benthivores -> whiting | 0.021 -> 0.0226 | 0.0122-0.0245 | 0.0366 / 0.0306 / 0.0129 / 0.00738 | 0.168 |
| benthivores -> infauna | 0.036 -> 0.0256 | 0.0156-0.0290 | - | ~1.0 (none) |
| benthivores -> epifauna | 0.15 -> 0.110 | 0.069-0.125 | - | ~1.0 (none) |
| benthivores -> Nephrops | 0.44 -> 0.308 | 0.068-0.398 | - | ~1.0 (none) |

Method: fish pairs a_new = a_lib x V_A x P; V_A the rule A vision factor at the bed from Overlapp's A_band (1.247
/ 1.188); P = (rule A integral with f_m and the light at each depth) / (the same with f uniform in the 10 m band).
Reproductions: krill reproduces 0.013 (classes 0.0448 / 0.0107 / 0.000583 / 0, night/day 0.030) and rule A on
the band x0.727; the planktivore reference reproduces 243.3's structure (pout 0.86 / herring 0.12, cod 0.65 /
haddock 0.35 vs 0.87 / 0.11, 0.67 / 0.33); the gate builder reproduces Overlapp's A_band_T byte for byte. Night/day
from the integral itself (the sibling's 0.090 / 0.180 kept the library's day/night split). Prey profiles from
krill_pairs_wc.py (pout 0.7 / 0.4 in the bottom 10 m); whiting as prey 0.7 day / 0.5 night in the bottom 10 m
(section 251), the rest 10-40 m (Onsrud et al. 2004) [J].

## Benthic pairs: what the theorem implies, and recommendation [J]

- Theorem [D]: bed prey sit at the bed, so a_j = a_lib x tau_m (x0.71 / 0.74 / 0.70).
- e carries no time term ("Exponering av bottenbyten": "search share not overstated" referred to cod's diel
  activity, not its height above the bed).
- Saturation at today's beds untouched (a h B 75 -> 54 infauna, 90 -> 66 epifauna); the diet shares change
  (finding 5).
- Half-saturation 1/(a h f): infauna 0.25 -> 0.35 t/km2 (stock 18.7 accessible), epifauna 0.042 -> 0.057 (3.78),
  Nephrops 0.40 -> 0.57 (4.57).
- Behaviourally it matters on depleted beds: in the `--depleted-patches` worlds saturation on a 1 % bed falls
  0.43 -> 0.35 (infauna), 0.47 -> 0.40 (epifauna), so emptied beds pay less and are left sooner.
- **Recommendation [J]: apply the three benthic a's with the fish and krill rows**: one f_m per member shared by
  all its prey; scaling only the off-bed pairs credits more than one search budget. Every benthivore pair passes
  the predator-side gate either way (Nephrops 2.19 with, 2.29 without).

## Final table: the whole section, every pair at once

Exact YAML in `tables/yaml_edits.txt` (17 entries, one-line derivation comment and old values each).

| pair | max_intake_rate | dark_ratio (threshold) | depth_access_table class a (< 100 / 100-150 / 150-250 / >= 250) | stage_access |
|---|---|---|---|---|
| predatory_fish -> krill | 0.026 -> 0.0117 | 0.0245 (0), rule | 0.0306 / 0.0141 / 0.00262 / 6.91e-5 | - |
| whiting -> krill | 0.043 | none (stomachs) | 0.148 / 0.0354 / 0.00199 / 0 | - |
| planktivorous_fish -> krill | 0.023 | 0.57 (0), stomachs | 0.0707 / 0.0211 / 0.00377 / 0.00171 | - |
| benthivorous_fish -> krill | 0.013 -> 0.00679 | 0.036 (0), integral | 0.0225 / 0.00583 / 0.000701 / 1.52e-5 | - |
| mackerel -> krill | 0.0039 | 0.42 (-5), stomachs | 0.00532 / 0.00429 / 0.0031 / 0.00284 | - |
| predatory_fish -> planktivorous_fish | 0.006 -> 0.0174 | 0.0594 (-5), rule | 0.0384 / 0.0211 / 0.00716 / 0.00286 | kept |
| predatory_fish -> benthivorous_fish | 0.0055 -> 0.0418 | 0.165 (-5), rule | 0.0693 / 0.057 / 0.0234 / 0.0128 | [0.42, 0.12] -> [0.392, 0.0388] |
| predatory_fish -> whiting | 0.013 -> 0.0359 | 0.137 (-5), rule | 0.0599 / 0.0495 / 0.0199 / 0.0097 | kept |
| predatory_fish -> mackerel | 0.00088 -> 0.000137 | 0.923 (-5), rule | 0.000229 / 0.000127 / 0.000102 / 0.000102 | kept |
| whiting -> planktivorous_fish | 0.0212 -> 0.0329 | 0.212 (-5), rule | 0.0626 / 0.0341 / 0.0205 / 0.0159 | kept |
| whiting -> benthivorous_fish | 0.21 -> 0.103 (flagged open) | 0.565 (-5), rule | 0.132 / 0.119 / 0.0825 / 0.0715 | kept |
| mackerel -> planktivorous_fish | 0.0266 -> 0.0248 | 0.16 (-5), stomachs | none (flat +-1 %) [J] | kept |
| benthivorous_fish -> planktivorous_fish | 0.0198 -> 0.0214 | 0.06 -> 0.141 (0), integral | 0.0401 / 0.0262 / 0.0114 / 0.00685 | kept |
| benthivorous_fish -> whiting | 0.021 -> 0.0226 | 0.15 -> 0.168 (0), integral | 0.0366 / 0.0306 / 0.0129 / 0.00738 | kept |
| benthivorous_fish -> benthic_infauna | 0.036 -> 0.0256 [J rec.] | none | - | - |
| benthivorous_fish -> mobile_epifauna | 0.15 -> 0.110 [J rec.] | none | - | - |
| benthivorous_fish -> norway_lobster | 0.44 -> 0.308 [J rec.] | none | - | - |

light_threshold_deg unchanged on every pair; table rows at 99.9 / 100 / 149.9 / 150 / 249.9 / 250 m (the loader
renormalises to an equal-cell mean of 1). Unchanged: zooplankton pairs, mammal pairs, every predation_mortality,
handling time and growth rate.

## Gate results

Encounter share / booked share, equal cells (field spread); `*` = OFF.

| variant | WC krill / plk / ben / whi | whiting ben / epi (field) | benthivores inf / epi / Neph / krill / plk / whi | WC krill share 150-250 m | prey side plk / ben / whi |
|---|---|---|---|---|---|
| Overlapp A_band_T (repro) | 1.32 (1.31) / 0.74 (0.71) / 1.64 (1.89) / 0.61 (0.43) | 0.12* / 0.23* | 1.05 / 1.00 / 2.26 / 0.89 / 0.51 / 0.25 | 0.45 | 0.64 / 0.85 / 0.55 |
| **final** | **1.39 (1.40) / 0.78 (0.76) / 0.97 (1.13) / 0.64 (0.46)** | **0.12* / 0.23*** | **1.03 / 1.01 / 2.19 / 0.91 / 0.61 / 0.31** | **0.49** | **0.69 / 0.62 / 0.60** |
| final, benthic a's unchanged | same | same | 1.06 / 1.01 / 2.29 / 0.67 / 0.45 / 0.23 | 0.49 | 0.65 / 0.62 / 0.55 |
| final, 243 window kept | 1.32 / 0.74 / 1.64 (1.89) / 0.61 | same | as final | 0.45 | 0.67 / 0.85 / 0.58 |
| window low / high | ben 0.66 (0.78) / 1.45 (1.68) | same | as final | 0.51 / 0.46 | ben 0.51 / 0.78 |
| deep | as final | same | 1.04 / 1.02 / 1.13 / 1.41 / 0.56 / 0.25 | 0.49 | 0.68 / 0.62 / 0.57 |
| bedbound | as final | same | 1.01 / 1.00 / 2.16 / 1.07 / 0.72 / 0.37 | 0.49 | 0.73 / 0.62 / 0.62 |
| rest50 / deep_rest50 | as final | same | Neph 1.96 / 0.78; krill 1.17 / 1.79; plk 0.67 / 0.62 | 0.49 | - |
| sw_contact | as final | same | 1.04 / 1.02 / 2.51 / 0.74 / 0.53 / 0.28 | 0.49 | - |
| pout_albert (sensitivity) | as final | same | 0.95 / 0.93 / 2.02 / 0.84 / 1.62 / 0.29 | 0.49 | plk 0.98 |

Every benthivore pair passes the predator-side gate in every variant; the same two whiting rows are OFF in every
variant. Prey-side OFF in every variant: whiting -> benthivores, WC -> mackerel, benthivores -> whiting (0.25-0.37,
the 251 residual). WC krill class shares in final 0.41 / 0.59 / 0.49 / 0.06.

## Loader check

`load_project_config(mareld2.yaml, library_path=<final copy>)` and the variant without the benthic changes: 15
groups, 0 warnings, no refusal; open-pool stability holds; every derived growth_rate equals the live library's.

## Caveats

1. [J] The profiles are field occupancy; search is bracketed (rest50), not measured (no feeding-rate-by-behaviour
   data; Hobson 2007 says so).
2. [J] Haddock >= 35 cm central is a midpoint between a Barents analogue and a bed-bound reading; no North Sea or
   Skagerrak haddock vertical data (one targeted search). It moves P by ~x0.95 and the krill classes visibly.
3. [J] Cod anchored on the North Sea stock (Hobson); Hobson's tags are from shallow (< 100 m) grounds, the box
   beds 100-300 m. "deep" is the Barents bracket.
4. [J] 0-2 m is "bed contact" by tag resolution; a 0.5 m bed-search mode would lower tau; SW as contact raises it
   to 0.81-0.83.
5. [D] Point form, no slab clipping (matters only where the predator slab meets a thin prey slab; pout_albert
   overstated).
6. [D] Ratio method throughout; the benthivore fish pairs use Overlapp's bed-light vision factor times this
   note's profile factor; the 243.3 / 251 absolute derivations were not re-executed (structure reproduced).
7. [J] LRD and flatfish bed-bound (Hunter 2009 is plaice); LRD's krill credit falls to the cells where the krill
   layer meets the bed.
8. [J] The window maps ling to anglerfish and pollack to saithe; the kexp derivation uses two saithe stages.
9. [D] Annual-mean equal-cell shares on the 16 x 16 probe grid; seasonal cod behaviour (Righton 2000: night
   ascent Aug-Sep) not carried (no box data).
10. Not fitted: nothing chosen against 0.073, 0.012, 0.027, 0.36 or any booking.

## References

- Aglen A, Engas A, Huse I, Michalsen K, Stensholt BK 1997. ICES CM 1997/Y:16 (ICES J Mar Sci 56:345, 1999, doi:10.1006/jmsc.1999.0449)
- Albert OT 1994. Ecology of haddock in the Norwegian Deep. ICES J Mar Sci 51:31-44. doi:10.1006/jmsc.1994.1004
- Albert OT 1994. Norway pout in the Norwegian Deep. ICES J Mar Sci 51:45. doi:10.1006/jmsc.1994.1005
- Beamish FWH 1966. J Fish Res Board Can 23:109. doi:10.1139/f66-009 (search summary only)
- Boudreau PR 1992. CJFAS 49:23-31. doi:10.1139/f92-003 (abstract via search index, unverified)
- Engas A, Jacobsen JA, Soldal AV 1988. ICES CM 1988/B:32
- Hjellvik V, Michalsen K, Aglen A, Nakken O 2003. ICES J Mar Sci 60:967. doi:10.1016/s1054-3139(03)00116-4
- Hobson VJ, Righton D, Metcalfe JD, Hays GC 2007. MEPS 347:101. doi:10.3354/meps347101
- Hobson VJ et al. 2009. Aquat Biol 5:133. doi:10.3354/ab00144
- Hunter E, Cotton RJ, Metcalfe JD, Reynolds JD 2009. MEPS 392:167-178. doi:10.3354/meps08213
- Michalsen K, Godo OR, Ferno A 1996. ICES J Mar Sci 53:389. doi:10.1006/jmsc.1996.0054
- Onsrud MSR et al. 2004. ICES J Mar Sci 61:1278. doi:10.1016/j.icesjms.2004.09.005
- Righton D et al. 2000. ICES CM 2000/Q:09
- Stensholt BK, Stensholt E 2004. GIS/Spatial Analyses in Fishery and Aquatic Sciences 2:109-126
- Sibling notes: "Overlapp i fiskparen", "En synregel for fiskparen", "Bottenfiskens bytesbalans", "Fönsterviktning", "Exponering av bottenbyten", "Attackrater ur mötesteori" benthivores_encounter.md.
