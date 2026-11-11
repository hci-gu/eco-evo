# Overlap convention for the fish pairs under concurrent search (2026-11-11)

Research subagent; saved by the main agent from the subagent's report. Nothing in the library, code, tests,
manifest, resume or AGENTS.md changed; no training or rollout; no value set from model output. Temporary
library copies only in the session scratchpad.

Files:
- `scripts/band_pairs.py` -> `tables/band_pairs_stdout.txt`, `tables/band_pairs.json`: every in-layer fish
  term of the 220 / 223.26 / 228 / 243.3 derivations re-evaluated on the BAND overlap, one time-at-depth
  density per predator member (the one its krill pair already uses), with each term's library R ("lib_band")
  or vision rule A ("A_band"); overlap geometry per member and prey layer (section 0) and brackets. It execs
  `../En synregel for fiskparen/scripts/one_rule_pairs.py` up to that script's run block, so every other factor
  is the library's.
- `scripts/gate_band.py` -> `tables/gates/budget_gate_<variant>.txt` (21 runs), `tables/gate_band.json`:
  `tools/probes/budget_gate.py` with `library_path=<temp copy>` (the `lib` run byte-identical to today's).
  "_T" = each fish pair also carries its own class a's as a `depth_access_table`. Bracket variants with extra
  edits: the WC -> benthivores window; whiting's epifauna a scaled with its band.
- `scripts/tabulate_band.py` -> `tables/tabulate_band_stdout.txt`.
- `scripts/presence_weight.py` -> `tables/presence_weight_stdout.txt`: each a weighted by the prey's season.
- `scripts/proposal_table.py` -> `tables/proposal_table_stdout.txt`: the per-pair table.
- `papers/README.md`: no PDF downloaded; 11 Unpaywall JSON records (no contact details).

```
S=<session scratchpad>
python3 -I scripts/band_pairs.py <repo> $S/tmp tables/band_pairs.json            # ~2 min
python3 scripts/gate_band.py <repo> tables/band_pairs.json ../"En synregel for fiskparen"/tables/one_rule_pairs.json $S/libs tables   # ~10 s
python3 -I scripts/tabulate_band.py <repo> tables
python3 -I scripts/presence_weight.py <repo> tables/band_pairs.json
python3 -I scripts/proposal_table.py <repo> tables/band_pairs.json
```

Tags: [S] source, [D] derived here, [J] judgement.

## Findings

1. **[D] Only the band overlap is consistent with the section 242 engine.** Since 242 every prey is searched
   at once during one foraging time T; since 265.1 the fish attack every prey met (`attack_choice: fixed`);
   the engine has no vertical action. The search depth cannot be allocated by the policy: it must sit in a,
   as one density f_m(z) per member (integral 1) shared by all of that member's prey. In-layer O = 1/H_j is
   the upper bound of the band O_j, reached only if the member spends all its search time inside prey j's
   layer; crediting it on every pair assumes a search time equal to the number of layer-separated prey (6 for
   saithe). The field profile gives saithe day time shares inside each prey's layer: herring 0.69, pout 0.87,
   blue whiting 0.82, juvenile benthivores 0.13, the whiting layer 0.49, mackerel 0.02. Before 242 in-layer
   was consistent (232.1: the eat share was "the time spent in the prey's layer").
2. **[D] Several pairs are already band special cases**: every krill pair (245/262); the benthivore fish pairs
   (`benth_encounter.py`: pi R^2 v p F_j/H_band); whiting -> mobile_epifauna (232: H_bed/H_whiting); section
   243's WC -> benthivores (O_m a judged time near the bed). Only the section 220-convention fish pairs of the
   water column (WC), whiting and mackerel are in-layer. 243's judged saithe 0.05 "within ~2 m" is not the
   share implied by the profile the WC krill pair uses (exp 15 m: 0.125 day, 0.16 night; Hs 40-5 m:
   0.05-0.33): two profiles for one fish.
3. **[D] The band alone (library vision) lowers every in-layer fish pair except WC -> benthivores**: WC ->
   planktivores x0.49, -> whiting x0.46, -> mackerel x0.13; whiting -> planktivores x0.75, -> benthivores
   x0.20; mackerel -> planktivores x0.61; WC -> benthivores x1.45 (profile time shares replace 243's judged
   ones). Rule A on the band (A_band) vs the library a: WC -> planktivores x2.90, -> benthivores x7.60, ->
   whiting x2.76, -> mackerel x0.155; whiting -> planktivores x1.55, -> benthivores x0.49; mackerel ->
   planktivores x0.93; benthivore fish pairs x1.25 / x1.19 (vision only); krill pairs as rule A (WC x0.45,
   benthivores x0.73, others x1.00). Against rule A in-layer, the WC fish pairs fall to about half
   (planktivores 0.54, whiting 0.48); WC -> benthivores x1.9; whiting -> benthivores x0.19; whiting ->
   planktivores x0.72; mackerel -> planktivores x0.72; WC -> mackerel x0.057.
4. **[D] For a predator profile smooth over a thin bed layer, O_j ~ f(bed)**, independent of the prey layer's
   thickness: whiting uniform in a 10 m band meets juvenile benthivores at 1/10 per m vs 1/2 in-layer (a
   robust x0.2); saithe at ~1/Hs. Under the band the predator's density at the bed sets WC -> benthivores,
   the WC krill share and whiting -> benthivores.
5. **[D] Gate under rule A + band (A_band_T, the engine form with class tables).** Predator side: two OFF
   pairs, whiting -> benthivores (0.12 equal cells, 0.07 field spread) and the pre-existing whiting ->
   epifauna at the field spread (0.23). The WC passes every judged prey: benthivores 1.64 (1.89), krill 1.31
   (1.31), planktivores 0.74 (0.71), whiting 0.61 (0.43), zooplankton 0.36 (0.58); WC -> mackerel 0.02 (both
   shares < 0.05, not judged on the predator side). Prey side (implied / booked M2):

   | prey | A_band_T | A in-layer_T (sibling) | library |
   |---|---|---|---|
   | krill | 1.04 | 1.02 | 1.10 |
   | planktivores | 0.67 | 0.86 | 0.38 |
   | benthivores | 0.83 (whiting pair OFF 0.12) | 0.69 | 0.37 |
   | whiting | 0.47 (benthivores pair OFF 0.25, the section 251 residual) | 0.63 | 0.27 |
   | mackerel | 0.02 OFF | 0.24 | 0.10 |

6. **[D] WC krill share over 150-250 m beds: 0.45 under A_band_T (flat fish a's 0.28).**

   | reading | share |
   |---|---|
   | A_band_T, central | 0.45 |
   | review p | 0.31 |
   | Ke 2e-4 / 0.1 | 0.36 / 0.41 |
   | saithe-weighted juvenile windows | 0.50-0.51 |
   | Hs 5 m / 40 m | 0.12 / 0.68 |
   | literal in-layer A (sibling) | 0.25 |
   | library | 0.74 |
   | field: saithe, all seasons | 0.19 (0.08-0.27) |
   | field: group, Q3 | 0.47 (0.32-0.69) |
   | booked | 0.36 |

   Saithe's near-bed scale height Hs, not vision, is the controlling unknown.
7. **[D] WC -> benthivores: a 0.0055 -> 0.0418, ratio 1.64 / 1.89 at 243's stage_access [0.42, 0.12].** Those
   windows were a-weighted on the 243 member shares (saithe 0.45, spurdog 0.18, hake 0.21, ...); the band a is
   saithe 0.89, hake 0.05, anglerfish 0.02 (which answers 243's hake flag). With saithe-weighted windows:
   0.83 / 0.97 (pre-243 [0.30, 0.038]) and 0.58 / 0.68 (the audit's aw [0.312, 0.0070]). OFF only at Hs 5 m
   (3.08 / 3.94); at Hs 40 m 1.01 / 0.96. Inside the gate at every window for Hs >= 15 m. If the band a goes
   in, the window needs re-weighting on the new member shares.
8. **[D] Whiting -> benthivores is the one pair the band breaks: OFF in every variant, 0.04-0.26 at equal
   cells.** Hw 5 m with whiting's epifauna a scaled consistently (x10/Hw) lifts the epifauna pair to 0.96
   (0.43 field) and leaves benthivores at 0.24 (0.13). Whiting's near-bed density alone cannot carry the
   booked 0.1067 unless whiting sit closer to the bed than a 5 m band. No new literature on whiting's height
   above the bed. Choices: a measured whiting profile, or the booking.
9. **[D] WC -> mackerel collapses under the band: a 0.00088 -> 0.000137, ratio 0.02.** Saithe is not a member;
   under the band it is spurdog's summer surface pair (0.76 of the a) plus hake. Weighting by mackerel's
   season gives x1.53 (0.03), still OFF (OFF in every reading, library included). The equal-time annual mean
   is the engine convention since 220; presence weighting is an open library-wide question (x1.53 here, at
   most x1.06 elsewhere).
10. **[D] Night/day under band + rule A** (the rule's values; the stomach-derived values win where they exist):
    WC fish pairs 0.06-0.17 (library 0.41-0.48), WC -> mackerel 0.92; whiting -> planktivores 0.21, ->
    benthivores 0.57; mackerel -> planktivores 0.83 by the rule, the stomach 0.16 wins (user decision);
    benthivore fish pairs 0.09 / 0.18; WC -> krill 0.025, benthivores -> krill 0.036. Whiting -> krill "none"
    (stomachs, no diel difference, Onsrud et al. 2004) kept as a fourth stomach-derived case [J].
11. **[S/D] Benthivores' time near the bed (Barents analogue)**: Stensholt & Stensholt 2004 18.7 % of day and
    34 % of night within 2 m; Hjellvik et al. 2003 and Engas et al. 1988 ~30-60 % within 10 m; large haddock
    rise 100-150 m by day (Aglen et al. 1997). Under the band theorem a time-share factor ~0.3-0.6 against
    benth_encounter's full time in the 10 m band, for the benthivores' krill, planktivore and whiting pairs.
    Not re-derived here.
12. **[S/D] Prey profiles weigh more under the band**: Albert 1994 puts Norway pout within ~1.5 m of the bed
    by day, in the lowest 5-10 m at night (library: 0.7 in the bottom 10 m). For exp(15 m) saithe that
    roughly doubles the pout term by day [D, estimate] and brings slab clipping in. A sensitivity, not a
    proposal.

Recommendation [J]: rule A and the band overlap in together, every member's f_m from field profiles and shared
by all of its prey. Decisive unknowns: saithe's near-bed density (Hs), whiting's (Hw), the benthivores' time
share in their band (finding 11), pout's day layer (finding 12).

## Theory [D]

Encounter under concurrent search: during search member m is at depth z with density f_m(z) (SEARCH time,
integral 1, one f per member and period, shared by every prey). Prey j has column biomass B_j and vertical
density phi_j(z) (integral 1), search volume rate S_j(z,t) = scan pi R_j(z,t)^2 v_rel p_j, R set by the light
at z. Then lambda_j = B_j a_j, a_j = integral f_m(z) phi_j(z) S_j(z,t) dz (per fish; summed over N/t and members
with biomass weights): the band overlap with the light inside the integral.

The Holling form stays exact: the predator samples depth independently of handling events, so intake per
foraging time is a renewal-reward ratio sum_j e_j lambda-bar_j / (1 + sum_k h_k lambda-bar_k), lambda-bar the
search-time average; 242's demand_j = B T p_j lambda_j / (1 + sum_k p_k h_k lambda_k + w B) is exact with the
band-averaged lambda. f must be the search-time density, not occupancy (Hobson et al. 2007: ~70 % of cod time in
a near-inactive on-bed class).

In-layer is an upper bound: O_j = integral f phi_j <= max phi_j = 1/H_j for a uniform layer, equality only when
f lies inside layer j. In-layer (a3d: A(R,H)/H) sets f = phi_j for every prey separately; with K
layer-separated prey it credits K full search budgets; the double count on pair j is 1/tau_j, tau_j the
member's search-time share in layer j. Before 242, demand_j = B eat_j f_j(B_j) with eat_j the time in prey j's
layer, so in-layer was consistent (232.1). Since 242 / 265.1 T = sum eat_j and p = 1 on every fish pair; no
action allocates depth. A vertical action would bring in-layer back as the per-layer bound.

Normalisation: per member and period sum_z f_m(z) dz = 1; the group a = sum_m W_m a_m, W_m the biomass share in
the whole group, members that cannot take the prey in the denominator with a = 0 (223.26); absent members
(velvet belly over beds < 150 m) contribute 0 there.

Finite R near the bed: for prey in a slab at the bed (juvenile benthivores 2 m; whiting as prey 10 m) the point
form overstates the disc area inside the slab; the band analogue of `area_in_slab` averages it over the
predator's height (-3 to -15 % at R 0.25-1.4 m; the `noclip` variant shows it).

Consistent special cases: krill pairs (the formula with field f_m); benthivore fish pairs (F_j/H_band = f
uniform in the band, or a bed-bound cod reading phi_j at the bed); whiting -> epifauna (tau = H_bed/H_whiting);
243's WC -> benthivores (a = in-layer a x sum share_m O_m, O_m a judged tau_m). Smooth-profile corollary: when f
varies slowly over prey layer j, O_j ~ f(z_j).

## Biology evidence

| species | quantity | value | area, method | source | tag |
|---|---|---|---|---|---|
| saithe | day position | adults within ~0-35 m of the bed over 100-250 m beds; 0/22 day surface hauls; largest catch 12-35 m above a 166 m bed | HERAS/IESSNS, box +-50 km, July, 117 hauls | `Krill i Skagerrak/wc_depth/evidence.md` 1.1 | [D] |
| saithe | share near bed | "greatest proportion remains within 1-5 m off bottom during both day and night"; higher at night | Scotian Shelf, pre-spawning | Neilson et al. 2003, doi:10.1016/S1054-3139(03)00068-7 | [S] |
| saithe | biomass within 10 m of bed | 75 % Dec-Jan; 3-13 m by bimonth | DFO echo logger | DFO CSAS 2003/075 (via wc_depth/evidence.md) | [S] |
| saithe | best acoustic match layer | 30-100 m above the bed (calibration height) | Norwegian shelf | Aglen 1996 (via wc_depth/evidence.md) | [S] |
| demersal gadoids | vertical | "bottom half of the water column", lower at night | Barents acoustics | Stensholt et al. 2002, doi:10.1006/jmsc.2002.1249 | [S] |
| saithe | depth | median ~140 m Mar-Apr, < 50 m summer-autumn; less active at night | Iceland, 51 DSTs | Armannsson & Jonsson 2012, doi:10.1093/icesjms/fss121 | [S] |
| saithe | depth range | ~100 m of column; 25-50 m summer, 60-90 m winter | farm-attracted, Norway | Skilbrei & Ottera 2016, doi:10.1093/icesjms/fsv261 | [S] |
| saithe | under krill swarms | stacked below swarms at 100-200 m; ate fish, not krill | Norwegian Sea | Kaartvedt et al. 2005, doi:10.3354/meps299001 | [S] |
| saithe | GOV bed classes | adults 0.80-0.91 over 150-250 m beds; size rises with bed depth | GOV box, Q1/Q3 2000-25 | `Sejens djuplage` summary | [D] |
| cod | within 2 m of daily max depth | 28.9 % (34 % night, 18.7 % day) | Barents, winter, 6 DSTs, beds > 200 m | Stensholt & Stensholt 2004 | [S] |
| cod | > 10 m above daily max depth | Sep-Nov day 13-67 %, night 23-74 %; Feb-Mar day 54-87 %, night 31-65 % | same | same | [S] |
| cod | acoustic share 0-10 m above bed | summer median ~39 %; winter large fish 0.46 day / 0.57 night; > 75 % within 50 m | Barents 1995-2002 | Hjellvik et al. 2003, doi:10.1016/s1054-3139(03)00116-4 | [S] |
| cod | time near the seabed | > 55 % within 5 m; ~90 % "closely associated"; on-bed class ~70 % (near-inactive) | S/central North Sea, 92 DSTs | Hobson et al. 2007, doi:10.3354/meps347101 | [S] |
| cod | midwater vs seabed | 43 % midwater, 57 % near the seabed | North Sea, 14 tags | Hobson et al. 2009, doi:10.3354/ab00144 | [S] |
| cod + haddock | sA share 0-2 m of the lowest 22 m | day 0.24 / night 0.41 (350 m bed); 0.36 / 0.52 (250 m) | Barents | Engas, Jacobsen & Soldal 1988, ICES CM 1988/B:32 | [S] |
| haddock | diel layer | large and medium from the bottom up to 100 m above it by day, down at night; small the reverse | Barents | Aglen et al. 1997, ICES CM 1997/Y:16 (doi:10.1006/jmsc.1999.0449) | [S] |
| whiting | position | only in day bottom hauls, none in night surface hauls beside herring | Danish HERAS 2017-19 | `Vitlingens fiskbyten` step2_summary (232.1) | [D] |
| whiting | position | 20-45 m above a 120 m bed | Oslofjord | Onsrud et al. 2004, doi:10.1016/j.icesjms.2004.09.005 | [S] |
| whiting | diel catch | no adult day/night difference | Scottish survey 1976-93 | Petrakis et al. 2001, doi:10.1006/jmsc.2000.0989 | [S, abstract] |
| hake | position | day near-bed (32/43 bed-referenced day hauls); 1/37 night surface hauls | HERAS | wc_depth/evidence.md | [D] |
| spurdog | depth | Nov-May day 162-231 m, night 37.5-88 m; Jun-Sep upper 25 m | Norwegian fjords, DSTs | Klocker et al. 2024 (via wc_depth/evidence.md) | [S] |
| anglerfish | activity | 95 % inactive on the bed | - | Ofstad et al. 2022 (via wc_depth/evidence.md) | [S] |
| mackerel | school depth | 65 % in 0-40 m; all < 100 m | Norwegian Sea, July | Godo et al. 2004, doi:10.1016/j.icesjms.2004.06.009 | [S] |
| mackerel | school depth | very few deeper than 30 m | Nordic Seas, Jul-Aug | Nottestad et al. 2016, doi:10.1093/icesjms/fsv218 | [S] |
| mackerel | feeding layer | no feeding below the mixed layer | Irminger Sea | Jansen et al. 2019, doi:10.1016/j.fishres.2019.01.020 | [S, abstract] |
| Norway pout (prey) | height above bed | day: below the 1.2-1.5 m detection height; night: lowest 5-10 m | Norwegian Deep, Skagerrak 1984-87 | Albert 1994, doi:10.1006/jmsc.1994.1005 | [S] |
| juvenile gadoids (prey) | height above bed | within a few cm of the bottom by day; 3-5 m off at night | Georges Bank | Lough et al. 1989, doi:10.3354/meps056001 | [S] |
| herring, blue whiting, krill (prey) | layers | herring 10-100 m by day, bottom-10-m share by bed 0.66 -> 0.05; blue whiting below 150 m by day; krill day layer 95-145 m centre (262) | box | `krill_pairs_wc.py` profiles; `Krill over grunda bottnar` | [S/D] |

Evidence caveats: acoustic shares exclude the bottom dead zone (~1.5-2 m at 200-300 m), so they are lower bounds
near the bed; DSTs take the daily maximum depth as the bed (biased toward 0); every source measures all time,
not search time; cod and haddock are Barents analogues; no saithe or whiting height-above-bed time budget
exists for the box.

## Gate results

Encounter share / booked share, equal cells (field spread); `*` = OFF.

| variant | WC krill / plk / ben / whi | whiting krill / ben / epi / plk | WC class-2 krill share |
|---|---|---|---|
| library | 2.41 (2.25) / 0.21* (0.32*) / 0.18* (0.27*) / 0.18 (0.09) | 1.48 (1.65) / 0.25* (0.12*) / 0.51 (0.24*) / 0.35 (0.17*) | 0.74 |
| A in-layer, T (sibling) | 1.02 (0.85) / 1.07 (1.22) / 0.66 (0.67) / 0.97 (0.58) | 1.35 (1.55) / 0.58 (0.37) / 0.46 (0.23*) / 0.69 (0.39) | 0.25 |
| band, library vision, T | 2.54 (2.49) / 0.11* (0.12*) / 0.27* (0.43) / 0.09 (0.04) | 1.53 (1.64) / 0.05* (0.02*) / 0.52 (0.24*) / 0.27* (0.24*) | 0.86 |
| **A band, T** | **1.31 (1.31) / 0.74 (0.71) / 1.64 (1.89) / 0.61 (0.43)** | **1.46 (1.59) / 0.12* (0.07*) / 0.50 (0.23*) / 0.54 (0.43)** | **0.45** |
| A band flat (no fish tables) | 1.31 (1.02) / 0.74 (0.92) / 1.64 (2.05) / 0.61 (0.24) | 1.46 (1.63) / 0.12* (0.06*) / 0.50 (0.24*) / 0.54 (0.26*) | 0.28 |
| A band, review p, T | 0.90 (0.90) / 0.95 (0.92) / 2.11 (2.44) / 0.78 (0.56) | 1.37 (1.50) / 0.21* (0.13*) / 0.39 (0.18*) / 0.95 (0.77) | 0.31 |
| Hs 5 / 40 m | ben 3.08* (3.94*) / 1.01 (0.96); krill 0.90 (0.65) / 1.57 (1.77) | unchanged | 0.12 / 0.68 |
| WC->ben window pre-243 / aw | ben 0.83 (0.97) / 0.58 (0.68) | unchanged | 0.50 / 0.51 |
| Hw 5 / 30 m (epifauna a x10/Hw) | whi 0.69 (0.49) / 0.37 (0.26) | ben 0.24* (0.13*) / 0.04* (0.03*); epi 0.96 (0.43) / 0.16* (0.11*); plk 0.52 / 0.27* | 0.45 |
| Ke 2e-4 / 0.1 | ben 1.84 (2.25) / 1.61 (1.96) | ben 0.13* / 0.13* | 0.36 / 0.41 |
| no slab clipping | ben 1.79 (2.03) | ben 0.13* | 0.45 |

Benthivores, mackerel and planktivores pass in every band variant (benthivores: krill 0.88, planktivores 0.51,
whiting 0.25 unchanged, already band; mackerel: krill 1.67, planktivores 0.77, zooplankton 0.54; planktivores:
krill 0.95). Prey side A_band_T: krill 1.04; planktivores 0.67 (WC 0.74, ben 0.51, whi 0.54, mac 0.77);
benthivores 0.83 (whi 0.12*, WC 1.64); whiting 0.47 (ben 0.25*, WC 0.61); mackerel 0.02*; seals ->
predatory_fish OFF (out of scope). All annual-mean shares on the 16 x 16 probe grid.

## Proposed per-pair parameters (A_band, central, library capture p) - NOT applied

a_new = a_lib x ratio (band alone x vision A_band/lib_band, relative to the reproduced library derivation);
depth_access_table = the pair's own class a's for beds < 100 / 100-150 / 150-250 / >= 250 m (rows at 99.9 / 100
/ 149.9 / 150 / 249.9 / 250 m as the krill pairs; the loader renormalises to an equal-cell mean of 1);
light_threshold_deg kept.

| pair | a_lib -> a_new (ratio = band x vision) | depth_access_table class a | dark_ratio (thr) | review-p a | bracket (Hs, Hw, Ke) |
|---|---|---|---|---|---|
| predatory_fish -> krill | 0.026 -> 0.0117 (x0.45; band already; saithe R_sat on 25 mm krill \|C0\| 0.4 = 0.57 m vs 0.87) | 0.0306 / 0.0141 / 0.00262 / 6.9e-5 | 0.0245 (0), rule | 0.014 | 0.0100-0.0134 |
| whiting -> krill | 0.043 -> 0.043 (x1.00; 2 L cap binds) | 0.148 / 0.0354 / 0.00199 / 0 | none, stomachs [J] | 0.052 | 0.038-0.048 |
| planktivorous_fish -> krill | 0.023 -> 0.023 (x1.00) | 0.0707 / 0.0211 / 0.00377 / 0.00171 | 0.57 (0), stomachs | 0.028 | 0.022-0.025 |
| benthivorous_fish -> krill | 0.013 -> 0.00945 (x0.73; cod > 28.5 cm capped at R0) | 0.0325 / 0.00778 / 0.000432 / 0 | 0.036 (0), rule | 0.011 | 0.0083-0.0106 |
| mackerel -> krill | 0.0039 -> 0.0039 (x1.00) | 0.00532 / 0.00429 / 0.0031 / 0.00284 | 0.42 (-5), stomachs | 0.0047 | 0.0038-0.0041 |
| predatory_fish -> planktivorous_fish | 0.006 -> 0.0174 (x2.90 = 0.49 x 5.93; saithe 0.92 of a; R_sat on 15 cm fish 0.86 / 1.40 m vs fixed 0.5 / 0.2) | 0.0384 / 0.0211 / 0.00716 / 0.00286 | 0.059 (-5), rule | 0.039 | 0.0125-0.0243 |
| predatory_fish -> benthivorous_fish | 0.0055 -> 0.0418 (x7.60 = 1.45 x 5.24; saithe 0.89, hake 0.05; 243 time factors dropped) | 0.0693 / 0.057 / 0.0234 / 0.0128 | 0.165 (-5), rule | 0.094 | 0.0206-0.101 |
| predatory_fish -> whiting | 0.013 -> 0.0359 (x2.76 = 0.46 x 6.02) | 0.0599 / 0.0495 / 0.0199 / 0.0097 | 0.137 (-5), rule | 0.081 | 0.0197-0.0592 |
| predatory_fish -> mackerel | 0.00088 -> 0.000137 (x0.155 = 0.13 x 1.19; spurdog 0.76 summer surface); presence-weighted 0.00021 (open) | 0.000229 / 0.000127 / 0.000102 / 0.000102 | 0.92 (-5), rule | 0.00031 | 0.000133-0.000147 |
| whiting -> planktivorous_fish | 0.0212 -> 0.0329 (x1.55 = 0.75 x 2.06) | 0.0626 / 0.0341 / 0.0205 / 0.0159 | 0.21 (-5), rule | 0.074 | 0.0172-0.0392 |
| whiting -> benthivorous_fish | 0.21 -> 0.103 (x0.49 = 0.20 x 2.51) | 0.132 / 0.119 / 0.0825 / 0.0715 | 0.57 (-5), rule | 0.23 | 0.034-0.205 |
| mackerel -> planktivorous_fish | 0.0266 -> 0.0248 (x0.93 = 0.61 x 1.53) | none (derived shape flat +-1 %) [J] | 0.16 (-5), stomachs | 0.056 | 0.0248 |
| benthivorous_fish -> planktivorous_fish | 0.0198 -> 0.0247 (x1.25; vision only, band already) | 0.0401 / 0.0335 / 0.0141 / 0.00845 | 0.090 (0), rule | 0.037 | 0.020-0.035 |
| benthivorous_fish -> whiting | 0.021 -> 0.0249 (x1.19; vision only, band already) | 0.0408 / 0.0338 / 0.0141 / 0.00843 | 0.18 (0), rule | 0.037 | 0.020-0.036 |

Unchanged: the benthic pairs; every zooplankton pair (WC -> zooplankton and whiting -> zooplankton are still
in-layer and the same theorem applies; whiting's 10 m band never reaches the 0-25 m copepod pool over the box's
beds, minimum 44 m; flagged, not derived); the mammal pairs; every predation_mortality, handling time and growth
rate; stage_access (WC -> benthivores' window should be re-weighted on the band a's member shares if the a goes
in, finding 7).

## Caveats

1. [J] f_m is field occupancy (HERAS July, GOV day hauls, DSTs), not search-time density (Hobson 2007's ~70 %
   inactive on-bed class shows they differ).
2. [J] Small-scale co-aggregation (saithe under krill swarms; Kaartvedt 2005) is outside f x phi; it would
   raise O.
3. [J] Saithe's day scale height (Hs 15 m, 5-40) is July only and not seasonal; it controls the juvenile pair
   (x5 across the bracket) and the krill share (0.12-0.68).
4. [J] Anglerfish (bottom 1 m) and ling (exp 5 m) profiles judged; together <= 0.04 of any WC pair.
5. [D] Slab clipping only on bed-slab prey; elsewhere the point form errs by <~ R/2H.
6. [D/J] The benthivores' profile (uniform 10 m band or bed-bound) conflicts with the cod and haddock field
   time shares (finding 11); their benthic 2-D pairs assume full bed time.
7. [D] WC -> mackerel reuses the planktivore term structure (reproduction 0.00117 vs 0.00088); ratio method
   throughout.
8. [D] 16 x 16 probe grid, annual-mean shares; presence weighting reported, not applied.
9. [J] Rule A's |C0|, R_nv fish 0.3 m and review p are the sibling's judgements; light below 150 m
   extrapolated.
10. Not fitted: no profile, Hs, Hw, window or p chosen to hit 0.36, 0.47 or any booking.

## References

- Aglen A, Engas A, Huse I, Michalsen K, Stensholt BK 1999. ICES J Mar Sci 56:345. doi:10.1006/jmsc.1999.0449 (CM 1997/Y:16 version read)
- Albert OT 1994. Ecology of haddock in the Norwegian Deep. ICES J Mar Sci 51:31. doi:10.1006/jmsc.1994.1004
- Albert OT 1994. Biology and ecology of Norway pout in the Norwegian Deep. ICES J Mar Sci 51:45. doi:10.1006/jmsc.1994.1005
- Armannsson H, Jonsson ST 2012. ICES J Mar Sci 69:1372. doi:10.1093/icesjms/fss121
- Engas A, Jacobsen JA, Soldal AV 1988. ICES CM 1988/B:32
- Godo OR et al. 2004. ICES J Mar Sci 61:1093. doi:10.1016/j.icesjms.2004.06.009
- Godo OR, Michalsen K 2000. Fish Res 48:127. doi:10.1016/S0165-7836(00)00177-6 (snippet)
- Hjellvik V, Michalsen K, Aglen A, Nakken O 2003. ICES J Mar Sci 60:967. doi:10.1016/s1054-3139(03)00116-4
- Hobson VJ, Righton D, Metcalfe JD, Hays GC 2007. MEPS 347:101. doi:10.3354/meps347101
- Hobson VJ et al. 2009. Aquat Biol 5:133. doi:10.3354/ab00144
- Jansen T et al. 2019. Fish Res 214:25. doi:10.1016/j.fishres.2019.01.020
- Kaartvedt S et al. 2005. MEPS 299:1-5. doi:10.3354/meps299001
- Lough RG et al. 1989. MEPS 56:1-12. doi:10.3354/meps056001
- Neilson JD et al. 2003. ICES J Mar Sci 60:860. doi:10.1016/S1054-3139(03)00068-7
- Nottestad L et al. 2016. ICES J Mar Sci 73:359. doi:10.1093/icesjms/fsv218
- Onsrud MSR, Kaartvedt S, Rostad A, Klevjer TA 2004. ICES J Mar Sci 61:1278. doi:10.1016/j.icesjms.2004.09.005
- Petrakis G, MacLennan DN, Newland AW 2001. ICES J Mar Sci 58:50. doi:10.1006/jmsc.2000.0989
- Skilbrei OT, Ottera H 2016. ICES J Mar Sci 73:1186. doi:10.1093/icesjms/fsv261
- Stensholt BK et al. 2002. ICES J Mar Sci 59:679. doi:10.1006/jmsc.2002.1249
- Stensholt BK, Stensholt E 2004. GIS/Spatial Analyses in Fishery and Aquatic Sciences 2:109-126
- Via research notes: Aglen 1996; DFO CSAS 2003/075; Klocker et al. 2024; Ofstad et al. 2022 (`Krill i Skagerrak/wc_depth/evidence.md`). Sibling notes: "En synregel for fiskparen", "Synregel i vattenpelaren", "Vattenpelarens bottenfiskbyte", "Sejens djuplage".
- Sought, not obtained: Neat et al. 2006 (doi:10.1007/s00227-005-0110-6), Casey & Myers 1998 (doi:10.1139/f98-120), Rosen & Holst 2013 (doi:10.1016/j.fishres.2013.08.002), Skaret et al. 2020 (doi:10.3354/meps13254; abstract only).

## User decisions (2026-11-11)

- Rule A + the band overlap go into the library together (proposal table above, central, library p,
  stomach-derived night/day where it exists, per-pair class tables, WC -> benthivores window re-weighted on
  the band a's member shares), but only after the benthivores' time share in their band (finding 11) is
  derived, so every pair goes in at once and training runs once.
- Whiting -> benthivores: the band a (0.103) goes in and is flagged open until whiting's height above the bed
  is measured (the prey's g pays its somatic production, so the shortfall moves only the yardstick).
