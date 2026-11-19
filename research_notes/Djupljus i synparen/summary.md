# Deep light in the visual pairs: K_deep 0.06 judged -> 0.09 measured (section 280, 2026-11-19)

Research subagent; written to this note only. Nothing in `fgconfig/fg_library.yaml`, `lib/`, `tests/`, the manifest,
`mareld_resume.txt`, `AGENTS.md` or any other note was edited. The new values are applied only to a library COPY
in the session scratchpad. User decision: one flat K = 0.09 /m below 65 m for every pair and bed class (the
"K_deep" parameter of the section 253 light), from research_notes/"Ljus under 65 m" (VOTO gliders, K_PAR 0.092
over 150-250 m beds). No value was chosen against a gate, booking, stomach share or clumping result.
Tags: [D] derived here, [J] judgement.

## Verdict

- **The whole section 269 -> 270 chain reproduces the live library exactly at K_deep 0.06.** All 130 values of the
  17 pairs match with a relative difference of 0 (`tables/reproduction_0.06.txt`), against the live file and git
  HEAD. Every link also matches its own note's saved json or stdout.
- **At K_deep 0.09, 12 pairs change** [D] (`tables/yaml_edits_280.txt`). Five visual pairs do not: WC -> mackerel
  and mackerel -> planktivores (prey above 65 m), and the three benthic pairs (bed-contact tau, no light).
- **Every changed attack rate falls.** The fish -> fish pairs fall x0.81-0.93, the krill pairs x0.92-0.98.
- **Night/day rises on every rule-derived pair.** The day light at the prey falls, while night stays at R_nv.
- **The class tables lose most at 100-250 m beds.** The < 100 m and >= 250 m classes barely move: the first stays
  bright, the second was already dark.
- **The gate takes no pair OFF and brings none back ON.**
- **No derived growth_rate moves.**

| pair (WC = water column) | a | dark_ratio | class a's < 100 / 100-150 / 150-250 / >= 250 m | stage_access |
|---|---|---|---|---|
| whiting -> krill | 0.043 -> 0.041 (x0.953) | none (kept) | 0.148 / 0.0354 / 0.00199 / 0 -> 0.146 / 0.0317 / 0.0006 / 0 | - |
| planktivores -> krill | 0.023 -> 0.0214 (x0.930) | 0.57 (stomachs, kept) | 0.0707 / 0.0211 / 0.00377 / 0.00171 -> 0.0689 / 0.018 / 0.00281 / 0.00166 | - |
| WC -> krill | 0.0117 -> 0.0107 (x0.915) | 0.0245 -> 0.0257 | 0.0306 / 0.0141 / 0.00262 / 6.91e-05 -> 0.0302 / 0.0129 / 0.0011 / 6.8e-05 | - |
| mackerel -> krill | 0.0039 -> 0.00382 (x0.979) | 0.42 (stomachs, kept) | 0.00532 / 0.00429 / 0.0031 / 0.00284 -> 0.0053 / 0.0042 / 0.00297 / 0.00284 | - |
| benthivores -> krill | 0.00679 -> 0.00641 (x0.944) | 0.036 -> 0.0366 | 0.0225 / 0.00583 / 0.000701 / 1.52e-05 -> 0.0221 / 0.00523 / 0.000322 / 7.91e-06 | - |
| WC -> planktivores | 0.0279 -> 0.02348 (x0.842) | 0.116 -> 0.14 | 0.0384 / 0.0211 / 0.00716 / 0.00286 -> 0.038 / 0.0167 / 0.00368 / 0.00284 | [0.6483, 0.1732] -> [0.6484, 0.1732] |
| WC -> benthivores | 0.1162 -> 0.09443 (x0.813) | 0.165 -> 0.209 | 0.0693 / 0.057 / 0.0234 / 0.0128 -> 0.0676 / 0.0426 / 0.0141 / 0.0127 | [0.3745, 0.01112] -> [0.3711, 0.01256] |
| WC -> whiting | 0.1045 -> 0.08608 (x0.824) | 0.139 -> 0.172 | 0.0599 / 0.0495 / 0.0199 / 0.0097 -> 0.0588 / 0.0379 / 0.0112 / 0.00966 | [0.4067, 0.07186] -> [0.4066, 0.07179] |
| whiting -> planktivores | 0.06357 -> 0.05913 (x0.930) | 0.342 -> 0.375 | 0.0626 / 0.0341 / 0.0205 / 0.0159 -> 0.062 / 0.0293 / 0.0169 / 0.0159 | - |
| whiting -> benthivores | 0.2687 -> 0.2455 (x0.914) | 0.565 -> 0.641 | 0.132 / 0.119 / 0.0825 / 0.0715 -> 0.13 / 0.103 / 0.0727 / 0.0714 | - |
| benthivores -> planktivores | 0.02905 -> 0.02431 (x0.837) | 0.238 -> 0.28 | 0.0401 / 0.0262 / 0.0114 / 0.00685 -> 0.0379 / 0.0195 / 0.00716 / 0.00657 | - |
| benthivores -> whiting | 0.04581 -> 0.03745 (x0.818) | 0.171 -> 0.21 | 0.0366 / 0.0306 / 0.0129 / 0.00738 -> 0.0353 / 0.0229 / 0.00794 / 0.00726 | - |

Unchanged:
- `light_threshold_deg`, `handling_time`, `predation_mortality`, and every `stage_access` except the three WC pairs.
- The whiting, mackerel and benthivore windows were never re-weighted on the band a in section 269, and are not
  re-weighted here.

## Files

- `scripts/k280_common.py`: the helper every driver uses. It execs an older note script from its own file, with
  exact string replacements, each anchor asserted. No existing note was edited, and none of its tables were written.
- `scripts/run_chain.sh <repo> <scratch> K [K ...]` runs the chain once per K value. Each step's outputs go to the
  scratch directory; the K 0.06 and 0.09 copies are in `tables/chain/`.

| step | script | runs | what changes at K_deep |
|---|---|---|---|
| s1 | `s1_band.py` | "Overlapp i fiskparen" `band_pairs.py`, whole: WC, whiting and mackerel fish pairs, the krill pairs, the benthivores' vision ratio | Its A_band variant carries kdeep |
| s2 | `s2_benth.py` | "Bottenfiskens hojdprofil" `benth_profile.py`: benthivore pairs and member profiles | G_LIGHT and the rule-A krill runs at K; V_A from s1 |
| s3 | `s3_window.py` | `wc_benth_window.py`: the WC -> benthivores window | Member shares and the saithe kexp from s1. K_C 1.7 -> 1.6 at 0.09 |
| s4 | `s4_rows269.py` | `final_gate.py`'s row builders: g3 rounding, tables, stomach ratios | The window key follows the derived K_C |
| s5 | `s5_terms.py` | "Stimskydd i motes-a" `pair_terms.py`: term shares and R_eff | band_fish_pair's default kdeep and its krill state at K |
| s6 | `s6_combined.py` | "Fangst och stim i fiskparen" `combined.py`'s factor loop, on "Vitlinglojans stim" `school_factor_pout.py` and `p_ppr.json` | F on s5's R_eff; the benthivore rows follow K (see below) |
| s7 | `s7_decompose.py` | "Langans indata" `decompose_ling.py` (~2 min) | gmat and band_fish_pair at K; its window exponent KEXP 1.73 (the band-A saithe a(L) exponent + 2.11) = round(s3's k_band + 2.11, 2): 1.73 at 0.06, 1.61 at 0.09 |
| s8 | `s8_compose.py` | `gate_ling.py`'s library builder, "central" run | LIB0 = the 269 state at K (git 7153a6c + s4 rows); the 0.392 / 0.0388 window = s3's |

- `s9_reproduction.py` writes `tables/reproduction_0.06.txt`.
- `cap_check.py` writes `tables/cap_check_stdout.txt`.
- `s10_edits.py` writes `tables/yaml_edits_280.txt`.
- `apply_280.py <library.yaml>` takes its target as an argument, so a run cannot hit the live file by accident.
- `gate_280.py` writes `tables/gate_280_stdout.txt` and `tables/gates/`.
- `sensitivity.py` writes `tables/sensitivity_stdout.txt` and `.json`.

What never changes with K:
- The library-rule krill reproduction (`KR_REPRO`) and the benthivores' section 220 library light (`TR_LIB`). Both
  are ratio denominators that reproduce how the old a's were derived, so they stay at 0.06. `KR_REPRO` returns the
  section 262 krill a's within 2 % (in s1's stdout).
- The other sibling notes (Sejens hojd over botten, Vattenpelarens vuxna bottenfisk, Sej och ungfisk i samma hal,
  Rekryternas bottendjup) feed only the booking and the gate's field shapes. They hold no pair value, so they were
  not re-run.

### Three links with hard-coded inputs that follow K [J]

`school_factor_pout.py` hard-codes the benthivore pairs' schooling rows. Here they follow K:
- **night/day:** 0.141 / 0.168 become s4's values (0.169 / 0.21 at 0.09);
- **pout share:** 0.85 becomes s2's derived share (0.84 at 0.09);
- **rule-A bed radii:** 0.6 / 0.3 m are scaled by benth_pair's light-weighted radius at the bed, R_eff(K) / R_eff(0.06):
  0.537 / 0.299 m at 0.09. The day ratio also scales the WC bed-slab default.

At 0.06 every one of these is the original number. Without the radius scaling (`tables/s6_0.09_bed_radii_unscaled_stdout.txt`):
- WC -> whiting 0.0850 instead of 0.08608;
- benthivores -> planktivores 0.02378 instead of 0.02431;
- benthivores -> whiting 0.03691 instead of 0.03745 (1-2 %).

### The 2 L cap (`tables/cap_check_stdout.txt`)

- It still binds on whiting, planktivores and mackerel -> krill: rule A equals the library rule at the same light.
- Those three pairs move only through the light, which section 269 did not show: x0.953 / x0.928 / x0.980.

## Reproduction at K_deep 0.06

| link | check |
|---|---|
| s1 | `band_pairs.json` A_band, every float identical |
| s2 | `benth_profile_stdout.txt` identical (except the output file name) |
| s3 | `wc_benth_window` [0.392, 0.0388] at k+1.7 |
| s4 | `yaml_edits.txt` (269) and the git 7153a6c library: 0 mismatches |
| s5 | `pair_terms.json`: rel. diff 0 |
| s6 | `combined.json` central factors: rel. diff 0; night/day identical |
| s7 | `decompose_ling.json`: 0 diffs; stdout identical |
| s8 | the live library: 130 / 130 values exact |

- s8 needed one rounding rule. Section 270 wrote WC -> planktivores night/day as 0.116, from combined's printed
  0.1155: 4 decimals, then 3 significant figures.
- The live file differs from git HEAD only by the uncommitted section 279 spawning blocks and growth_rate values;
  none of the reproduced keys.

## Gate on the copy (budget_gate, `library_path` patched as gate_p.py; `tables/gate_280_stdout.txt`)

The runner was checked first: the live library through `library_path` prints exactly the plain run's text.

**OFF before and after (unchanged):**
- whiting -> benthivores (equal cells and prey side; flagged open);
- whiting -> epifauna (field spread);
- WC -> mackerel and seals -> WC (prey side);
- three porpoise pairs (prey and diet side; present since sections 272/273).

**Goes OFF:** none. **Comes back ON:** none.

Largest moves (encounter / booked; prey side = predator side at equal cells):

| pair | live | 0.09 |
|---|---|---|
| benthivores -> planktivores | 0.81 | 0.69 |
| benthivores -> whiting | 0.62 | 0.52 |
| benthivores -> krill | 0.89 | 0.82 |
| planktivores -> krill | 0.95 | 0.89 |
| WC -> benthivores (equal / field) | 2.02 / 1.81 | 1.95 / 2.13 |
| WC -> whiting (equal / field) | 1.28 / 0.95 | 1.24 / 1.24 |
| WC -> zooplankton (equal / field) | 0.28 / 0.47 | 0.33 / 0.68 |
| WC -> krill (equal / field) | 1.01 / 1.06 | 1.03 / 1.02 |

Prey totals (implied / booked M2), live -> 0.09:

| prey | live | 0.09 |
|---|---|---|
| planktivores | 0.99 | 0.95 |
| whiting | 0.72 | 0.67 |
| krill | 1.01 | 0.96 |
| benthivores | 0.75 | 0.74 |
| zooplankton | 1.00 | 1.04 |

- At the field spread, the WC pairs on bed prey rise. The class tables now put more of the a over < 150 m beds,
  which is where the field spread places the water column.
- **Loader:** mareld2.yaml, seed 1, live vs copy: all 15 derived growth_rate identical, no warnings.

## Sensitivity, not to be applied (`scripts/sensitivity.py`)

- The chain has one light profile for all beds: gmat is (time, depth).
- A bed-dependent light would need g(t, z, bed) through `point_series`, `slab_series`, `benth_V`, `member_A` and the
  krill `lux`.
- The modest version is a per-class composite. Each bed class takes its a from a whole-chain run at that class's K;
  each run assumes that K everywhere [J].
- Readings (K < 100 / 100-150 / 150-250 / >= 250 m):
  - **class:** 0.16 / 0.13 / 0.092 / 0.082, the glider mid-water K by EMODnet bed class.
  - **class+bed:** 0.16 / 0.14 / 0.11 / 0.092. This is the bed-light effective K, with the 0-100 m near-bed excess
    (optical depth 2.15) spread over bed - 65 m; the near-bed layer over >= 250 m beds is assumed.

Equal-cell a for the live library, flat 0.09, class and class+bed:

| pair | live | flat 0.09 | class | class+bed |
|---|---|---|---|---|
| WC -> planktivores | 0.0279 | 0.0235 | 0.0202 | 0.0195 |
| WC -> benthivores | 0.1162 | 0.0944 | 0.0769 | 0.0740 |
| WC -> whiting | 0.1045 | 0.0863 | 0.0712 | 0.0682 |
| whiting -> planktivores | 0.0636 | 0.0592 | 0.0557 | 0.0549 |
| whiting -> benthivores | 0.2687 | 0.2451 | 0.2276 | 0.2246 |
| benthivores -> planktivores | 0.0291 | 0.0243 | 0.0205 | 0.0199 |
| benthivores -> whiting | 0.0458 | 0.0375 | 0.0307 | 0.0294 |
| krill: whiting / planktivores / WC / benthivores / mackerel | 0.043 / 0.023 / 0.0117 / 0.00679 / 0.0039 | 0.041 / 0.0213 / 0.0107 / 0.00641 / 0.00382 | 0.0373 / 0.0193 / 0.0097 / 0.00578 / 0.00374 | 0.0365 / 0.0189 / 0.0093 / 0.00563 / 0.00371 |

- **What drives it:** the murkier water over < 150 m beds (K 0.13-0.16) takes a further x0.80-0.85 off the
  fish -> fish pairs. That is where the flat 0.09 left the shallow classes almost untouched.
- **Night/day:** not composited. Per K, it rises steadily with K (e.g. WC -> benthivores 0.209 at 0.09, 0.262 at
  0.13, 0.296 at 0.16; `tables/sensitivity_stdout.txt`).
- **Reach:** the composite stays an approximation. The 270 factors and the ling decomposition are pair-level.
- **Both readings understate the shallow classes.** The composite changes K only below 65 m, but the gliders
  also show K 0.15-0.16 at 20-65 m over beds < 100 m, against the convention's 0.109 / 0.091 there.

## Left out

- **The near-bed layer and the bed-class K.** They are in the sensitivity only; the user approved one flat K.
- **The absolute level gap.** The gliders read 0.31-0.41 of the convention at 50-120 m (section 3 of "Ljus under
  65 m"); it is unexplained and not folded in. Its sign is the same as the K change (darker), so it would lower the
  visual a's further.
- **The pending Jan/Dec Secchi fix of section 270.9 (research_notes/"Januaris ljusunderskott").** It is not in this
  diff. The chain still uses `one_rule_pairs.py`'s SECCHI table, with January 5.0 m (the n = 1 artefact) and
  December 11.5 m; both affect only the top 20 m. It should be fixed when the chain is next applied, as a separate
  change.
- **Pairs not re-derived.** None has light below 65 m in its derivation:
  - planktivores -> zooplankton: 0-25 m pool, its own 0.30 / 0.12 light;
  - WC, whiting and mackerel -> zooplankton: copepods 0-25 m, section 245 scalings;
  - the mammal pairs: no light term in the engine;
  - the section 275 observation encounter lists: no light term.
- **Ke, R_nv and bioluminescence.** These are open in "Ljus under 65 m" (gap 4) and unchanged.
