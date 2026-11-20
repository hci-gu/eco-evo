# The excess of encounter theory: which term carries it? (2026-11-20)

Research only, by a research agent with three literature subagents; saved by the main agent.
- Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed. Nothing was committed.
- The only model call was the read-only, policy-free `tools/probes/budget_gate.py`.
- The frozen fix105 decomposition ("Mättnad hos rovfisk"/model/tables) is used only as a yardstick that maps a
  correction onto a feeding level. No value is chosen from it.

**Files:**
- `evidence.md`: M1-M4 (model side), L1 (per term), D1 (arithmetic), J1 (reading).
- `lit_search_time.md`, `lit_detection_capture.md`, `lit_overlap.md`: the subagent trails, with quotes.
- `tables/search_time_raw.csv`, `detection_capture_raw.csv`, `overlap_raw.csv`: one row per sourced number.
- `tables/term_corrections.csv`: the per-term factors used.
- `scripts/s1_yardstick.py`, `s2_forward.py`, `s3_gut_window.py`, with output in `tables/*_stdout.txt`.
- `tables/budget_gate_stdout.txt`.
- `data/`: untrusted downloads; do not commit.

**Tags:** [F] full text, [A] abstract, [S] secondary or held note, [D] derived here, [J] judgement. PROXY = not a
model species.

Side findings: Onsrud et al. 2005 (doi:10.1139/f05-090) is held in full at "Krillens fläckighet"/data/onsrud2005.
Two chain speeds are off: Norway pout's 0.33 BL/s is ~2.6x below the in situ ~1 BL/s by day (the chain is LOW);
whiting's correction depends on the length assumed (x0.75 at adult length, x1.25 at the tracked juveniles).

## Question
Every fish group could eat 1.65-2.3x its field ration while foraging. The attack rate is derived as a = scan x area
x v_rel x T_tick x N_t x p x F x light x overlap; on the bed pairs it is a_enc x e.
- Which term of that derivation carries the error?
- Is it common to all of a predator's pairs, or specific to a prey type?

## What has to move [D] (s1)
The uncapped feeding level is T X/(1 + X).

| group | X_eff | need / field | generic a-factor, need (field) | time share, need |
|---|---|---|---|---|
| planktivores | 1.34 | 0.262 / 0.231 | 0.29 (0.25) | 0.49 |
| benthivores | 36 | 0.633 / 0.521 | 0.054 (0.033) | 0.68 |
| whiting | 5.1 | 0.390 / 0.455 | 0.15 (0.20) | 0.52 |
| water column | 3.5 | 0.397 / 0.375 | 0.20 (0.18) | 0.53 |
| mackerel | 1.3 | 0.282 / 0.137 | 0.31 (0.13) | 0.51 |

- **The spread in the needed a-factor (5.8x) is the benthivores alone.** The four other groups need x0.15-0.31, a
  ~2x spread that is inside the yardstick's roughness. A time share would be nearly uniform (0.49-0.68).
- **Per-prey test** (exact, from the decomposition). Removing any single prey entirely still leaves the potential
  above both targets in every group, with one exception: mackerel without krill gives 0.272 against a need of
  0.282, which would need a krill a of about 0. Whiting without krill gives 0.470, above both its need and its
  field ration of 0.455.
- **X_eff is a rough group mean** taken over cells and ticks.

## Per term: field vs chain (Q1)

| term | chain assumes | field (best) | tag | already in a? | correction, central (range) | shared or specific |
|---|---|---|---|---|---|---|
| (a) search time within foraging | the whole 6-h tick is searched at v; the flatfish v is already a move+pause mean | Intermittent movers pause ~50 % (6-94 %) of locomotion time (all taxa, PROXY). Gadoids, herring and mackerel swim continuously: "few tracks had swimming speeds below 5 cm s-1" in situ. A herring-like up-down cycle that searches only on the ascent gives 0.5-0.6 [D on J]. No foraging-time budget exists for any wild adult | A, F, S | flatfish yes, round fish no | x0.9 (0.7-1) for round fish; herring x0.75 (0.5-1) | shared |
| (a') whole-day idle time | none (the policy forages 0.90-0.97 of the time) | Cod DST records are vertically active 0.13-0.5 of the time, with ~0.7 'seabed' days; these measure vertical activity, not foraging. Lab cod swim 0.2 BL/s satiated or unfed vs 0.6 at half ration | S (grey), F | no | engine form 0.5-1 [J]; ~1 for a gut-buffered saturated fish | shared |
| (b) detection R | rule A; fish prey sit at the 2 L cap; krill R_nv 0.1 m | No in situ R for any model species. Juvenile cod keep R by chemoreception in turbid water. Inside rule A, fish prey are unaffected below c 1.7 /m, and krill R^2 falls x0.86-0.90 per doubling of c. Attention: a bird PROXY only | A, D, F (PROXY) | yes | fish x1; krill x0.85-1; copepods x0.9 (0.7-1) | shared in form, bounded near 1 |
| (c) search speed | whiting 0.8, saithe 1.0, herring 1.0, pout 0.33, cod 0.4 (modal), mackerel 1.8 BL/s | Krill-layer fish 15-16 cm/s by day and 10 at night (~1 / 0.7 BL/s at 14-15 cm; "larger fish somewhat lower"). Cod hourly means 0.7-0.9 BL/s. Saithe split-beam >= 0.9 m/s (artefact-prone). Herring ~1 BL/s | F, S | yes | whiting x0.9 (0.6-1.25). Pout x2.6 (1-3.9): the chain is LOW, which raises the planktivore krill a x1.72 and the copepod a x1.05. Others ~x1 | shared |
| (d) fine-scale overlap | mean field within a cell and depth band | Cod and capelin (PROXY prey) are out of phase only at 2-3.5 km and in phase at 3.5 m while cod feed. "No association" from 20 m to 10 km. GOV Lloyd index 1.03-1.59. Krill voids under herring schools by day give x0.83-0.96. Reaching x0.3 needs r ~ -0.7 with CVs ~1 | A, F, S, D | no (vertical band only) | ~1 (0.7-1.5); planktivores -> krill 0.9 | prey-specific |
| (e) capture per encounter | fish 0.52-0.63 (hungry predator, q 0.8); krill 0.5 [J]; copepods 0.6-0.7 | No in situ value for any model species. Reef proxies 0.17-0.26 per strike become x0.29-0.52 through q and r, with refuge and fullness caveats. Bluefish lab lines differ ~2x between prey species at equal size. Field herring diets show no copepod-escape effect | F (PROXY), D | yes | fish x0.8 (0.3-1) [J]; krill, copepods and bed prey x1 | prey-specific |
| (f) scan | 0.5 of the visual disc | supported by field sprat clearance; no new number | S | yes | x1 | shared |
| vulnerability-window residual | rule A night/day ratio; F by day | intake concentrated at twilight on fish and krill (Rindorf & Gislason 2005; Rindorf 2003) | J (held) | partly | fish x0.5 (0.25-1); krill x0.75 (0.5-1) | prey-specific |

## Q2. Shared or prey-specific?

**Mostly shared within each predator [D/J].** Two independent lines support this.

1. **The per-prey test.** It needs no share comparison. Only mackerel minus krill reaches its need, and only at a
   krill a of about 0.
2. **The policy-free diet gate.** On the pairs it judges (either share >= 0.05), encounter/booked ratios run
   0.28-1.95.
   - Two pairs are OFF: whiting -> benthivores 0.28 (0.35 at the field spread), and whiting -> mobile_epifauna 0.22
     at the field spread only. Both are UNDER-encounters, the opposite sign of a prey-specific excess.
   - Unjudged minor pairs scatter more: water column -> mackerel 0.00, zooplankton pairs 0.33-0.53, Nephrops 2.17.
   - Krill on judged pairs runs 0.89-1.60. No prey type sits consistently above its booking.
   - Whiting and mackerel krill at 1.3-1.6 allow a krill-specific residual of at most ~1.5-2x [J].

**Limits.**
- The gate compares shares and inherits the booked level, so a factor common to all of a predator's pairs is
  invisible to it by construction.
- Its 1/3-3 band is as wide as the needed factors.
- The booked diets are themselves stomach-based.
- The evidenced prey-specific terms (capture, overlap) are prey-specific in kind but near 1 in size.

## Q3. Forward sizing per group [D] (s2)
The "corner" column sets every term at its strongest at the same time. That is not a plausible combination.

| group | central potential (need) | share of the needed reduction carried | residual a-factor / time share after central | corner |
|---|---|---|---|---|
| planktivores | 0.509 (0.262) | 0.06 (the pout speed raises a) | 0.32 / 0.51 | 0.181 |
| benthivores | 0.926 (0.633) | 0.01 | 0.059 / 0.68 | 0.262 (needs bed structure x0.2 and idle 0.5) |
| whiting | 0.683 (0.390) | 0.15 (of which the krill window [J] 0.06) | 0.24 / 0.57 | 0.275 |
| water column | 0.638 (0.397) | 0.26 (of which the fish window [J] 0.13) | 0.36 / 0.62 | 0.125 |
| mackerel | 0.511 (0.282) | 0.11 | 0.37 / 0.55 | 0.279 |

**Candidate model forms:**
- **Factor on a** (speed, pauses, R, scan, p, overlap): fixed and derived.
  - One factor of ~x0.25-0.35 would fit the four non-benthic groups.
  - It cannot reach the benthivores: at X ~36, halving a changes their potential by less than 3 %.
- **Time share on the whole response** (the engine's T): one number of 0.5-0.68 fits all five groups. Against it:
  - A fixed cap is the removed `predator_share` (AGENTS.md: do not reintroduce).
  - A learned share needs a rest payoff, which was ruled out ("Vilans avkastning").
  - Biologically, a daily idle share does not lower the ration of a gut-buffered, saturated fish. In s3, any window
    longer than ~1 h fills a benthivore's gut; satiation meals are 1-8.5 days of ceiling (Arnott & Pihl 2000).
  - It would hold only for idle spells longer than the gut buffer, which no tag study shows as a foraging state.
- **Per-prey ceiling** (Rindorf plateaus): outside the encounter chain, and field-sized only for whiting on fish.

## Verdict [J on D]
1. **No individual term carries the excess on field evidence.**
   - Speed, scan and detection are at or above the field values.
   - Within-bout pauses give about x0.9.
   - Capture and fine-scale overlap are prey-specific and near 1.
   - The central product of all sized corrections carries only 1-26 % of the needed log-reduction, mostly through
     the [J] window residual.
2. **The excess is shared within each predator.**
   - For the planktivores, whiting, water column and mackerel, one shared a-side factor of ~x0.25-0.35 (after the
     central corrections) cannot be rejected.
   - Its term is the step the chain takes as 1: the fish searches during the whole foraging time, continuously, a
     mean-field prey, at its swimming speed, and every contact inside R is an encounter (search effort and
     encounter realisation).
   - This step is not measured for any model species.
   - The field constrains only the whole product:
     - herring clearance at x0.5-0.67 of theory;
     - January krill at x1/50-1/260;
     - cod-capelin stomachs emptier than the encounter model;
     - whiting fish-prey plateaus.
3. **The benthivores are a separate case.**
   - They need x0.03-0.06 on a, or a time share of 0.56-0.68 that a gut buffer would absorb.
   - The bed-side terms are judged (chemo-tactile width 0.15 L, capture, exposure e) or proxy (habitat structure
     x0.11-0.2, lab juvenile cod).
   - The ceiling side (armour, the bottom-temperature Cmax) and acquisition limits outside the encounter are
     equally open.
   - The encounter chain cannot be shown to be their carrier.

## Settling datum
1. **A wild adult gadoid's foraging-time budget plus its in situ attack rate inside a measured prey field.**
   Examples: accelerometer or acoustic-tag time in a foraging state together with stomachs, or CODFUN-type target
   tracking (Skaret et al. 2016).
   - If attacks inside the prey field match the chain and the fish forages most of the day, the error sits on the
     ceiling or appetite side.
   - If attacks are far below theory, the error is the search-effort / encounter-realisation factor on a.
2. **An in situ capture or reaction distance for any model species on krill.**

## Load-bearing sources
- Onsrud, Kaartvedt & Breien 2005, CJFAS 62:1822, doi:10.1139/f05-090 [F]. Held copy: "Krillens fläckighet"/data/onsrud2005.
- Winger 2004, PhD [F held].
- Kramer & McLaughlin 2001, doi:10.1093/icb/41.2.137 [A].
- Meager et al. 2005, doi:10.1139/f05-104 [A].
- Scharf et al. 2003, doi:10.1016/S0022-0981(03)00260-0 [F].
- Scott et al. 2017, Sci Rep 7:12966 [F, PROXY].
- Viitasalo et al. 1998, MEPS 175:129, and 2001, MEPS 216:191 [F].
- Rose & Leggett 1990, doi:10.2307/1940245 [A].
- Kaartvedt et al. 2005, MEPS 299:1 [F].
- Onsrud et al. 2004, ICES JMS 61:1278 [F held]; Christiansen 2024 [S].
- Rindorf & Gislason 2005, doi:10.1016/j.jembe.2005.04.013 [F held].
- Fall & Fiksen 2020, doi:10.1111/faf.12415 [F held].
- Arrhenius & Hansson 1999 [F held].
- Hobson et al. 2007, doi:10.3354/meps07047 [F held].

## Gaps and caveats
- **No measurement exists for any model species of:**
  - in situ R;
  - capture per encounter;
  - fish-euphausiid capture;
  - hake and haddock in situ speed or activity;
  - adult whiting speed.
- **Closed or unread:** O'Brien 1989/1990, Hill et al., Løkkeborg 1998, the Rose & Leggett full text, Handegard 2012
  (its predator is spotted seatrout, not saithe).
- **Grey literature:** Righton 2000 is used as a pointer only.
- **Whiting speed** hinges on the length assumed: x0.75 at adult length, x1.25 at the tracked juveniles.
- **The s3 gut-buffer form min(1, wX, G + w)** is an upper bound used for the window argument. It is not a second
  estimate of the need.
- **[J] values** in `term_corrections.csv`, especially the window residual and how far bed structure applies, are
  judgement.
