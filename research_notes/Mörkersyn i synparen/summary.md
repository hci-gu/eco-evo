# Dark vision in the visual pairs: gadoid-eye Ke 0.02 lx judged -> 0.002 lx (2026-11-22)

Derivation subagent; the main agent saved this text.
- Nothing in `fgconfig/fg_library.yaml`, `lib/`, `tests/`, the manifest, `AGENTS.md`, `mareld_resume.txt` or other
  notes was edited.
- Nothing was committed.
- The new values are on a library COPY in the session scratchpad.
- CPU only.

**User choice.** The "Ke 2e-3, F 1" corner of "Djupgradientens bärare", in every visual fish pair the section 269
rule-A chain derives:
- the source is the Aksnes & Utne 1997 fit of Anthony 1981's cod thresholds, in downwelling lx at F = 1
  ("Torskfiskars syn i morker" E1);
- no value was chosen against a gate, booking, stomach share or model run.

Tags: [D] derived here, [J] judgement.

## Verdict

- **Reproduction gate: PASS.** The section 280 chain at Ke 0.02, K_deep 0.09, through this note's Ke-aware drivers
  and x the 296 factors:
  - all 130 values of the 17 pairs are exact (largest relative difference 2.2e-16; `tables/reproduction_gate.txt`);
  - every step's stdout is identical to the "Djupljus i synparen" K_deep 0.09 run.
- **Cross-checks.**
  - At Ke 2e-3 the band stage reproduces the Ke_2e-3_F1 row of "Djupgradientens bärare" exactly: WC -> krill
    0.0322 / 0.0151 / 0.0026 / 9.14e-05; WC -> planktivores 0.04 / 0.0224 / 0.00609 / 0.00289.
  - Steps 5 and 7 at Ke 2e-3 return step 1's eq per pair, so no Ke site was missed.
- **Scope A (gadoids only) was not built [D].** The chain carries Ke per EYE KEY, not per member.
  - The band-stage decomposition (`tables/scope_a_shares_stdout.txt`) gives 12 changed pairs under A.
  - Mackerel -> krill stays exactly live: horse mackerel, 0.30 of its a, is its only gadoid-key member; B gives
    x1.042.
  - Three benthivore pairs differ from B at the band stage by about 1 %:
    - -> krill x1.097 vs x1.111 (long rough dab and grey gurnard, 0.14 of the a);
    - -> planktivores x1.228 vs x1.237;
    - -> whiting x1.235 vs x1.245 (long rough dab, minor flatfish, gurnard and starry ray in benth_V).
  - Downstream steps borrow benthivore radii, so A is not claimed equal to B there.
- **Scope B = every member on the gadoid key.**
  - Gadoids, with hake and ling counted as Gadiformes.
  - Non-gadoids on the key: horse mackerel, long rough dab, grey gurnard, starry ray and the minor flatfish.
  - Untouched: herring and mackerel are on the clupeid key (Ke_clup 1 lx, 0.016 lx threshold). Spurdog and velvet
    belly are non-visual; anglerfish has a fixed strike radius.
- **13 of 17 pairs change [D]** (`tables/yaml_edits_ke0.002_scopeB.txt`). Every changed a rises:
  - fish -> fish x1.09-1.25;
  - krill pairs x1.04-1.15;
  - WC -> mackerel x1.02.
  - Unchanged: mackerel -> planktivores and the three benthic pairs.
- **The class tables gain most over 100-250 m beds**, the reverse of section 280.
  - Day light over a 160 m bed is ~7e-4 lx (median of sun-up ticks, "Djupgradientens bärare").
  - g there goes ~0.03 -> ~0.26.
- **Night/day changes on every rule-derived pair:**
  - it falls on every fish -> fish pair, x0.79-0.99;
  - it rises on the WC and benthivore krill pairs, x1.42-1.43. A reading [J], not computed: the night krill layer
    sits in light between the two Ke values.
- **The gate takes no pair OFF and brings none back ON.**
- **No derived growth_rate moves** (15 groups, no loader warnings).
- **Water column, section 298 potential ration:** 1.10x field Q/B with 5 short months -> 1.44x with none.
- **Water column, policy-free intake gradient** (< 150 m vs 150-250 m beds): 2.0-2.8 -> 1.5-1.9.

## Edit table (scope B; a includes the 296 factor)

| pair | a | dark_ratio | class factors < 100 / 100-150 / 150-250 / >= 250 m | stage_access |
|---|---|---|---|---|
| whiting -> krill | 0.00656 -> 0.00728 (x1.110) | none | 0.146/0.0317/0.0006/0 -> 0.157/0.0379/0.00174/0 | - |
| planktivores -> krill | 0.007704 -> 0.0081 (x1.051) | 0.57 kept (stomachs) | 0.0689/0.018/0.00281/0.00166 -> 0.0707/0.0197/0.00364/0.00176 | - |
| WC -> krill | 0.001712 -> 0.001968 (x1.150) | 0.0257 -> 0.0367 | 0.0302/0.0129/0.0011/6.8e-05 -> 0.0322/0.0151/0.0026/9.14e-05 | - |
| mackerel -> krill | 0.00382 -> 0.00398 (x1.042) | 0.42 kept (stomachs) | 0.0053/0.0042/0.00297/0.00284 -> 0.00543/0.00441/0.00314/0.00288 | - |
| benthivores -> krill | 0.00641 -> 0.0072 (x1.123) | 0.0366 -> 0.0519 | 0.0221/0.00523/0.000322/7.91e-06 -> 0.0237/0.00625/0.000697/1.57e-05 | - |
| WC -> planktivores | 0.0037568 -> 0.0045824 (x1.220) | 0.14 -> 0.125 | 0.038/0.0167/0.00368/0.00284 -> 0.04/0.0224/0.00609/0.00289 | [0.6484, 0.1732] -> [0.6483, 0.1733] |
| WC -> benthivores | 0.0151088 -> 0.018864 (x1.249) | 0.209 -> 0.164 | 0.0676/0.0426/0.0141/0.0127 -> 0.0732/0.0602/0.0197/0.0127 | [0.3711, 0.01256] -> [0.3746, 0.01104] |
| WC -> whiting | 0.0137728 -> 0.01704 (x1.237) | 0.172 -> 0.138 | 0.0588/0.0379/0.0112/0.00966 -> 0.0632/0.0526/0.0168/0.00966 | [0.4066, 0.07179] -> [0.4067, 0.07183] |
| WC -> mackerel | 5.3536e-06 -> 5.4512e-06 (x1.018) | 0.916 -> 0.902 | 0.000229/0.000127/0.000102/0.000102 -> 0.000246/0.000132/0.000102/0.000102 | - |
| whiting -> planktivores | 0.0094608 -> 0.01028 (x1.087) | 0.375 -> 0.34 | 0.062/0.0293/0.0169/0.0159 -> 0.0642/0.0354/0.0193/0.0159 | - |
| whiting -> benthivores | 0.03928 -> 0.042992 (x1.095) | 0.641 -> 0.56 | 0.13/0.103/0.0727/0.0714 -> 0.137/0.123/0.0787/0.0714 | - |
| benthivores -> planktivores | 0.02431 -> 0.02939 (x1.209) | 0.28 -> 0.248 | 0.0379/0.0195/0.00716/0.00657 -> 0.0417/0.0275/0.00987/0.00679 | - |
| benthivores -> whiting | 0.03745 -> 0.0464 (x1.239) | 0.21 -> 0.182 | 0.0353/0.0229/0.00794/0.00726 -> 0.0386/0.0324/0.0111/0.00735 | - |

Notes on the table:
- **Section 296 re-applied as live.** One factor per predator, on a only: planktivorous_fish x0.36, whiting and
  predatory_fish x0.16. The product is rounded to 10 significant figures, as the live file stores it.
- **WC -> benthivores window.**
  - K_C (from the band a's saithe exponent) moves 1.6 -> 1.7.
  - The members' shares of the band a move: saithe 0.887 -> 0.893, hake 0.056 -> 0.050.
  - Net f_A 0.01256 -> 0.01104.
  - The ling and member corrections go x0.9377 -> x0.9392.
- **The 270 capture p x schooling F is re-run on the new R_eff.** For example, WC -> planktivores F 0.545 (day 0.512).
- **Benthivore bed radii** go to 0.602 / 0.307 m (0.6 / 0.3 at Ke 0.02, K_deep 0.06).

## Files

- **`scripts/ke_step.py <step> <Ke> <args>`** runs one "Djupljus i synparen" driver (s1-s8) with asserted Ke
  replacements, on the rule-A side only:
  - s1: the A_band Ke;
  - s2: G_LIGHT and the two rule-A krill runs;
  - s5: band_fish_pair's default and the krill state;
  - s7: decompose_ling's light and its reference.
  - Kept at 0.02 as ratio denominators: KR_REPRO, benth_V "lib" with TR_LIB, and lib_band.
- **`scripts/run_chain_ke.sh`** runs the chain. Step 6's radius reference stays the Ke 0.02, K_deep 0.06 band.
- **`scripts/run_all.sh`** documents the order. It was not run end to end; each step was run by hand with the same
  arguments.
- **`scripts/make_copy.py`** writes the edit list and applies it to the copy as text. It refuses fgconfig/.
- **`scripts/repro_gate.py`, `scope_a_shares.py`, `growth_check.py`, `compare_gates.py`.**
- **`scripts/gate_copy.py`** is copied from "Djupgradientens bärare". On the live library it prints exactly the plain
  budget_gate text.
- **`scripts/intake_by_class.py`** is that note's script with the predator as an argument.
- **Tables:**
  - `tables/chain/`
  - `tables/reproduction_gate.txt`
  - `tables/scope_a_shares_stdout.txt`
  - `tables/yaml_edits_ke0.002_scopeB.txt`
  - `tables/growth_check_stdout.txt`
  - `tables/gate/` (live and copy)
  - `tables/intake/` (3 predators, live and copy)
  - `tables/compare_scopeB_stdout.txt`
- **Copy:** session scratchpad `mork/libs/fg_library_ke0.002_scopeB.yaml` (rebuild with `scripts/make_copy.py`).

## Checks on the copy

**OFF sets, unchanged live -> copy:**
- equal cells: porpoises -> benthivores, porpoises -> whiting, whiting -> benthivores;
- field spread: whiting -> epifauna;
- prey side: whiting -> benthivores, WC -> mackerel, seals -> WC, and three porpoise pairs.
- **Goes OFF: none. Comes back ON: none.**
- **Nearest the edge:** whiting -> benthivores at the field spread, 0.35 -> 0.34 (OFF below 1/3). In the "all"
  scope of "Djupgradientens bärare" it went OFF (~0.32), because whiting's krill a rose while its fish pairs did
  not. Rewriting both keeps it ON.

**Diet gate, encounter / booked share, live -> copy:**

| pair | equal cells | field spread |
|---|---|---|
| benthivores -> planktivores | 0.69 -> 0.82 | - |
| benthivores -> whiting | 0.52 -> 0.63 | - |
| benthivores -> krill | 0.82 -> 0.95 | - |
| WC -> benthivores | 1.95 -> 1.98 | 2.13 -> 1.85 |
| WC -> whiting | 1.24 -> 1.26 | 1.24 -> 1.01 |
| WC -> krill | 1.03 -> 1.02 | 1.02 -> 1.10 |
| WC -> zooplankton | 0.33 -> 0.27 | 0.68 -> 0.47 |
| whiting -> epifauna | 0.49 -> 0.43 | 0.22 -> 0.20 |

- **Prey totals** (implied / booked M2): planktivores 0.95 -> 0.98, whiting 0.67 -> 0.72, krill 0.96 -> 1.00,
  benthivores 0.74 unchanged.
- **Total ration:** unchanged; the bookings were not touched.

**Section 298 potential ration at the field spread (x field Q/B), live -> copy:**

| group | annual | short months | month range | ceiling share | krill share |
|---|---|---|---|---|---|
| water column (saithe GOV) | 1.10 -> 1.44 | 5 -> 0 | 0.85-1.36 -> 1.12-1.78 | 0.41 -> 0.53 | 0.36 -> 0.41 |
| whiting (GOV) | 1.77 -> 1.79 | 0 | 1.54-2.15 -> 1.57-2.18 | 0.85 -> 0.86 | 0.73 |
| benthivores (GOV adults) | 1.90 -> 1.89 | 0 | 1.82-1.99 -> 1.81-1.98 | 0.99 | 0.03 |
| planktivores (equal cells) | 3.16 -> 3.18 | 0 | 2.07-4.59 -> 2.08-4.62 | 0.72 -> 0.73 | 0.31 -> 0.33 |
| mackerel (equal cells) | 4.72 -> 4.76 | 0 | 3.04-7.00 -> 3.07-7.07 | 0.66 -> 0.67 | 0.61 -> 0.62 |

- **Water column by month on the copy:** 1.12 1.15 1.19 1.25 1.35 1.48 1.69 1.72 1.78 1.75 1.49 1.26.
- **Against the scope "wc" reading (1.43):** the difference comes from re-deriving its own 270 p x F, ling terms,
  stage_access and dark_ratio, which that scope skipped.

**Policy-free intake by bed class (60x60).** G = I(< 150) / I(150-250); I(< 100) / I(150-250) in brackets.

| predator | Jan | Apr | Jul | Oct |
|---|---|---|---|---|
| water column | 2.76 (3.02) -> 1.91 (2.03) | 2.35 (2.53) -> 1.72 (1.81) | 2.02 (2.09) -> 1.51 (1.54) | 2.40 (2.56) -> 1.69 (1.77) |
| whiting | 2.09 (2.21) -> 1.83 (1.91) | 1.86 (2.00) -> 1.72 (1.83) | 1.73 (1.67) -> 1.57 (1.50) | 2.04 (2.14) -> 1.76 (1.83) |
| benthivores | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.00 -> 1.00 |

- **Water column at 150-250 m beds:** its share of the ceiling rises from 0.28-0.42 to 0.42-0.58. Over < 100 m it
  barely moves (0.85-0.88 -> 0.86-0.89).
- **Benthivores are saturated** at 0.99 of the ceiling on infauna and epifauna in every class, so the intake shows no
  gradient.

## Caveats

1. **One corner of an unresolved bracket [J].**
   - The cod data fit 2e-4 to ~0.07 lx, depending on the fitted light range and the geometry factor F (1-30).
   - 2e-3 is the A&U single fit at F 1; the live 0.02 corresponds to F ~10.
   - No gadoid R-vs-light curve exists on fish or krill prey. Confidence is low.
2. **Scope B gives non-gadoids the gadoid Ke.** A real per-member Ke would need:
   - a third eye key in one_rule_pairs.gmat and krill_pairs_wc.light_g;
   - a member-resolved benth_V and benth_profile.member_A.
3. **The 296 level [D/J].**
   - Whiting's x0.16 is field slope / chain a for whiting -> planktivores at Rindorf's bed mix (0.8 < 100 m, 0.2
     100-150 m; "Mötesfaktorn per grupp").
   - At Ke 2e-3 that a rises x1.040 at the same mix (class tables, cell shares 0.23 / 0.30 / 0.32 / 0.15).
   - Holding the anchor therefore needs x0.154. The water column's transferred 0.16 would follow.
   - The planktivores' 0.36 is anchored on their zooplankton pair. That derivation gives the gadoid members (pout,
     blue whiting) a judged day / twilight / night R^2 table, not Ke, so the factor is unaffected.
   - Not applied: the task said to re-apply 296 as live.
4. **Pairs not re-derived.**
   - WC, whiting and mackerel -> zooplankton (section 245): meas_pairs.py used the gadoid Ke in the 0-25 m light. The
     light there exceeds both Ke values most of the day, so the change is expected to be small and confined to
     twilight and night. It was not computed.
   - The mammal pairs and the section 275 encounter lists have no light term.
5. **Light, as in section 280.**
   - One profile serves every bed.
   - The absolute level gap the gliders show is not folded in.
   - The Jan/Dec Secchi fix (270.9) is still pending. It touches the top 20 m, and with a lower Ke more twilight light
     counts, so it matters slightly more now.
6. **Checks are policy-free.**
   - The diet gate runs at 16 x 16.
   - Intake assumes fixed attack, T = 1, and no depletion or interference.
   - Night/day ratios change on every rule-derived pair, so existing checkpoints meet different day/night payoffs.
