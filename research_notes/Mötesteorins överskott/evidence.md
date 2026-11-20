# Evidence trail: which term of the encounter derivation carries the excess (2026-11-20)

Research only. Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed; no model was
trained or run except the read-only `tools/probes/budget_gate.py` (policy-free, start densities). The frozen fix105
decomposition ("Mättnad hos rovfisk"/model/tables) is used ONLY as a yardstick that maps a correction onto a feeding
level; no value is chosen from it. Tags: [F] full text, [A] abstract, [S] secondary (incl. held notes), [D] derived
here, [J] judgement. PROXY = not a model species.

Literature trails (three subagents): `lit_search_time.md`, `lit_detection_capture.md`, `lit_overlap.md`, with
`tables/search_time_raw.csv`, `tables/detection_capture_raw.csv`, `tables/overlap_raw.csv`. Scripts in `scripts/`,
stdout in `tables/*_stdout.txt`. `data/` = untrusted downloads, not to be committed.

## M1. What has to move (`scripts/s1_yardstick.py` -> `tables/s1_yardstick_stdout.txt`) [D]

Shared handling: uncapped feeding level unc = T X/(1+X), X = sum_j a_j B_vis,j h_j (group-mean X_eff = S/(1-S),
S = unc/T; rough, the mean of a saturating function over cells and ticks is not the function of the mean).
Two targets: the realised need (cost + field P at fill ~1, the capped level) and the field ration (Q/B in the
ceiling's currency, pair_vs_field_fix105.txt).

| group | T | unc | X_eff | need | field | generic a-factor need / field | time share need / field |
|---|---|---|---|---|---|---|---|
| planktivores | 0.931 | 0.532 | 1.34 | 0.262 | 0.231 | 0.29 / 0.25 | 0.49 / 0.43 |
| benthivores | 0.954 | 0.928 | 36.3 | 0.633 | 0.521 | 0.054 / 0.033 | 0.68 / 0.56 |
| whiting | 0.898 | 0.751 | 5.13 | 0.390 | 0.455 | 0.15 / 0.20 | 0.52 / 0.61 |
| water column | 0.970 | 0.756 | 3.53 | 0.397 | 0.375 | 0.20 / 0.18 | 0.53 / 0.50 |
| mackerel | 0.972 | 0.550 | 1.30 | 0.282 | 0.137 | 0.31 / 0.13 | 0.51 / 0.25 |

- Across groups the needed a-factor spans 5.8x (need; 7.4x field), but that spread is the benthivores alone: the
  four non-benthic groups need x0.15-0.31 (need) / x0.13-0.25 (field), a ~2x spread inside the yardstick's
  roughness. The time share spans 1.4x (need). So one a-side factor of ~x0.2-0.3 cannot be rejected for the four
  non-benthic groups; the benthivores (x0.03-0.05 on a) are a separate case.
- Per prey (exact, from the decomposition's 'potential without prey j'): removing any single prey entirely leaves
  the potential above both targets in every group, with ONE exception: mackerel without krill 0.272 is just below
  its need 0.282 (it needs krill a ~0, not a plausible correction). Whiting without krill 0.470 stays above both its
  need 0.390 and its field ration 0.455. So no prey-specific correction can carry the excess; whatever carries it
  must reach (nearly) every prey of each predator, i.e. act as a shared term, or be several prey-specific terms of
  similar size on every prey.

## M2. A time share and the gut buffer (`scripts/s3_gut_window.py`) [D]

The engine's ceiling is Holling's h = E/Cmax per foraging time (the SSS limit, Jeschke et al. 2002 [S held]): a
fish cannot ingest faster than it digests, so the engine's foraging share T scales a AND the ceiling. A real fish
swallows a meal and digests it while resting: satiation meals 17.8 / 9.1 / 2.8 %BW (fish / Crangon / crabs; Arnott
& Pihl 2000 [F], via "Mättnad hos bottenbyten") are G = 8.5 / 3.3 / 1.0 days of the benthivore ceiling (104.7
kJ/kg/d at 7.4 C [S held]). With a daily foraging window w the gut-buffered daily level is ~min(1, w X, G + w):

| group | X_eff | w giving the need under the engine form | w giving it under the gut buffer |
|---|---|---|---|
| benthivores | 36 | 0.68 | < 0.03 (any window of > 1 h fills the gut) |
| whiting | 5.1 | 0.52 | 0.08 |
| water column | 3.5 | 0.53 | 0.11 |
| planktivores, mackerel | 1.3 | 0.49-0.51 | 0.20-0.22 |

So a time share on the whole response is biologically valid only if (i) the non-foraging spells outlast the gut
buffer (rest of several days: 2-8 days on crustaceans, > 8 days on fish), or (ii) the in-window encounter is
itself low (w X < 1), which is an a-side error under another name. A daily activity budget ("forages 50 % of the
day") does not by itself lower a saturated fish's ration. This agrees with "Födonivå och mättnad i fält"
(gut_buffer_window.py) and "Mättnad hos rovfisk" M1 [S].

## M3. What the chain already contains (current library chain, sections 220-280) [S, checked in the scripts]

| term | chain value | where |
|---|---|---|
| search time | the whole 6-h tick (T_tick 21 600 s) at speed v, times the policy's foraging share (0.90-0.97); no pause share for the round fish; flatfish v 0.12 BL/s is already a move+pause mean (Holmes 1979); relocation tau (276) only for moving cells | pred_encounter_a.py, krill_pairs_wc.py, benth_encounter.py |
| search speed | saithe 1.0 BL/s (0.5-1.4), night x0.6; whiting 0.8 (0.5-1.3); hake/ling 0.5 (J); spurdog/velvet belly 0.5; mackerel 1.8 (Macy feeding); horse mackerel 1.3; herring 1.0 (0.6-1.4); pout / blue whiting 0.33; sprat 0.6; cod/haddock on the bed 0.4 BL/s (modal, Winger; hourly means 0.67-0.9) | same |
| detection R | rule A: R^2 = R_nv^2 (1-g) + R_sat^2 g, g = E/(E + 0.02 lx), R_sat = min(image-area R0 (|C0| 0.4 krill / 0.7 fish, k_v 100, c 0.25/m), 2 L); R_nv 0.1 m krill, 0.3 m fish; measured light, K_deep 0.09; light-weighted R_eff on fish 0.3-0.8 m by day; bed: chemo-tactile width 0.15 L (gadoids), 0.6 L (flatfish) [J] | one_rule_pairs.py, s5 chain |
| scan | 0.5 of the visual disc (fish and krill pairs; planktivores f 0.5, Kulke field sprat clearance supports 0.5 against 1) | P tables |
| overlap | vertical band overlap (member time-at-depth x prey layer), bed-class depth_access tables; mean-field within a 1-km cell and depth band; no within-layer avoidance | band_pairs.py |
| schooling | F on schooled clupeids / pout by day (0.51-0.68 on the planktivore pairs) | "Fangst och stim i fiskparen" |
| capture | per encounter of a HUNGRY predator: fish prey 0.52-0.63 (q 0.8 pre-strike escape included); krill 0.5 (0.3-0.7, J); copepods herring 0.6, pout/sprat 0.7; bed 0.3-0.7 x exposure e 0.165 (infauna) / 0.43 (epifauna) | same |
| light / windows | rule A night/day; stomach ratios where measured; no midday vulnerability low | s4 chain |
| not encounter theory | mackerel -> zooplankton: measured clearance (Macy lab) | 245 |

Validation data already held that compare the whole chain with the field (not single terms):
- Herring (YOY, Baltic) field clearance: theory with R proportional to length over-predicts 1.5-2x; size-free R
  9-49x (Arrhenius & Hansson 1999 [F, via "Attackrater ur mötesteori" V1]).
- Krill pairs: library a 50-260x the January field-implied a on 100-150 m beds ("Krillparen mot vintermagar" [S]).
- Cod-capelin: a visual encounter model's stomachs fuller than observed (Fall & Fiksen 2020 [F held]).
- Whiting fish-prey plateaus 0.07-0.29 of the model ceiling per prey group (Rindorf & Gislason 2005 [F held]).
- Bed pairs: Irish Sea gradient gives only a lower bound on a, 6.5-19x below the library ("Födobegränsning" [S]);
  lab juvenile cod on sand Crangon a 0.065-0.13 vs library 0.15, habitat structure x0.11-0.2 (Persson et al. 2012
  [F held], PROXY stage).

## M4. Q2 from the diet gate (`tables/budget_gate_stdout.txt`, policy-free) [D]

- The gate's 'implied M2' is booked M2 x (encounter share / booked share): it compares SHARES and inherits the
  booked LEVEL. A factor common to all of a predator's pairs is invisible to it by construction.
- Its pass band is 1/3-3 per pair, the same size as the needed generic factors (x0.15-0.3), so it cannot exclude a
  prey-specific error of that size on one pair either. What it shows, on the pairs it judges (either share >= 0.05),
  equal cells: ratios 0.28-1.95. Two pairs are OFF: whiting -> benthivores 0.28 (0.35 at the field spread) and,
  at the field spread only, whiting -> mobile_epifauna 0.22 (0.49 equal cells). Both are UNDER-encounters, the
  opposite sign of a prey-specific excess. Unjudged minor pairs scatter more (water column -> mackerel 0.00,
  zooplankton pairs 0.33-0.53, benthivores -> Nephrops 2.17). Krill on judged pairs: 0.89 (planktivores), 1.03
  (water column), 1.32 (whiting; 1.45 field spread), 1.60 (mackerel); benthivores 0.82 (unjudged, shares < 0.05).
  No prey type sits consistently above its booking. A pair-specific excess of x3-20 confined to krill would put
  krill near the 3x edge and the other prey near 0.4-0.7; whiting and mackerel krill at 1.3-1.6 allow a
  krill-specific excess of at most ~x1.5-2 relative to their other prey, not x5-7 [J on D].
- Stronger than the gate: M1's per-prey test, which needs no share comparison. Even a prey-specific factor of 0
  on the dominant prey leaves every predator above its target. Combined with the gate (no pair-specific outlier
  beyond ~2x), the excess is mostly SHARED within each predator, with a krill-specific residual of at most ~1.5x
  in whiting and mackerel.
- Limits: (i) shares, not levels; (ii) the 3x band; (iii) the booked diets themselves are field-derived stomach
  shares, so "shared" here means "the same factor on every prey within the precision of stomach-based diets"; (iv)
  the argument does not say the shared factor is the same ACROSS predators.

## L1. Per term: field vs chain (Q1) [details and quotes in the three lit_*.md files]

(a) Search time inside a foraging period.
- Pause-travel: animals with intermittent locomotion pause ~50 % (6-94 %) of locomotion time (Kramer & McLaughlin
  2001 [A], PROXY all taxa). Algebra [D, lit_search_time]: effective clearance = scan area x MEAN speed over move +
  pause, so a correction applies only where the chain uses a MOVING speed of a pause-travel species.
- The model's gadoids, herring and mackerel swim continuously: juvenile walleye pollock "swam continuously ... periods
  of passive sinking were not observed" (Sogard & Olla 1996 [S held], PROXY); in the Oslofjord krill layer "few tracks
  had swimming speeds below 5 cm s-1" (Onsrud et al. 2005 [F]); mackerel lacks a swim bladder. Flatfish v 0.12 BL/s
  is already a move+pause mean (Holmes 1979 [S held]); plaice search more efficiently stationary (Hill et al. [S]).
- One herring-like up-down cycle (search on the ascent only, Janssen 1981 [S] PROXY): effective search 0.50-0.61 of
  continuous [D on one cycle, species J] -> herring 0.5/0.75/1.0 [J].
- Whole-day idle time: cod DST vertically active 0.13 (feeding grounds, Jun-Aug) to 0.44-0.51 (migration) (Righton et
  al. 2000, grey [S held]); ~0.70 of cod-days 'seabed' (Hobson et al. 2007 [F held]); coastal cod active round the clock
  (Winger 2004 [F held]); lab cod swim 0.2 BL/s satiated or unfed vs 0.6 at half ration (Björnsson 1993 [S]). These
  measure vertical or lab activity, not foraging time. No direct foraging-time budget for any wild adult model species
  was found [gap]. Under a gut buffer, daily idle time does not lower a saturated fish's ration (M2).
- Chain: none for round fish (whole tick searched); in for flatfish (inside v). Verdict: within-bout pauses x0.9
  (0.7-1.0) [J] for continuous swimmers; not a large error source.

(b) Detection distance in situ vs lab.
- No in situ reactive distance for any model species [gap]. Turbidity: juvenile cod keep R by chemoreception, visual
  R "only slightly" reduced over 0.4-17 /m beam attenuation (Meager et al. 2005 [A]); planktivorous walleye pollock
  insensitive to 0-40 NTU (De Robertis et al. 2003 [A], PROXY). Inside rule A [D]: fish prey sit at the 2 L cap
  (R00 ~4.9 m), so turbidity changes nothing until c > 1.7 /m; krill R^2 x0.86-0.90 per doubling of c 0.25 -> 0.5 /m.
- Attention: divided attention cut peripheral detection 0.37x in blue jays (Dukas & Kamil 2000 [F], PROXY bird, a
  predator-detection task); two cryptic prey types searched at once lowered detection (Dukas & Kamil 2001 [S]); no
  fish number. Scan 0.5 is in the chain already; no new number moves it. Lab R can understate the search volume
  (bluegill located prey beyond R, Luecke & O'Brien 1981 [S]).
- Verdict: x1.0 on fish prey, x0.85-1.0 on krill, x0.7-1.0 on copepods [J]; shared in form (attention, turbidity) but
  bounded near 1.

(c) Search speed.
- Whiting / pout in the krill layer: modes 15-16 cm/s by day, 10 at night, "about 1 body length s-1 by day and 0.7
  ... by night", most fish ~14-15 cm (Onsrud et al. 2005 [F]); "for the larger fish, somewhat lower swimming speeds"
  [F]. The model whiting is 0.84 adult at 25 cm: the same 15-16 cm/s is 0.6 BL/s (x0.75 on the chain's 0.8, the
  held reading of "Intag under födosök" E3), while a size-scaled cm/s (v ~ L^0.5 [J]) gives 0.79 BL/s (x1.0); juveniles
  x1.25 by day [D on F]. Central x0.9 (0.6-1.25). Pout (and the 80-110 m mixture) ~1 BL/s by day, 0.7 by night vs
  the chain's 0.33: the chain is LOW by x2.6 (1-3.9) [F/D].
- Cod: modes 0.1-0.4, hourly means 0.7-0.9 BL/s (Winger [F held]); chain 0.4 = the mode, conservative.
- Saithe: split-beam 0.9-1.05 m/s (2.7-4 BL/s at 25-29 cm; Pedersen 2001 [S], artefact-prone) vs chain 1.0 BL/s.
- Herring ~1 BL/s in situ [F], 1-1.5 BL/s biting (Gibson & Ezzi [S]) vs chain 1.0. Mackerel: lab feeding 1.3-1.8
  BL/s (Macy 1998 [S held]) vs chain 1.8; field school speeds unusable. Hake, haddock: no data [gap].
- Verdict: in situ speeds are at or ABOVE the chain's in every species measured; no correction below x1 is evidenced.

(d) Fine-scale overlap (lit_overlap).
- Cod-capelin out of phase below ~3.5 km, in phase above 4-10 km and at 3.5 m while cod feed (Rose & Leggett 1990
  [A], PROXY prey); no association 20 m-10 km in another predator-prey set (Horne & Schneider [A]); box GOV Lloyd
  1.03-1.59 for predators with pout/juvenile gadoids (haul scale [S/D]); whiting-Trisopterus collocation 0.51-0.59,
  aggregative [S]; R&G 2005 slight negative catch correlation [F held].
- The one measured within-layer void: krill escaping fish SCHOOLS by day (Onsrud 2004 [F held]; Christiansen 2024
  [S]) -> x0.83-0.96 on the planktivore krill pair [D on S]; dispersed gadoids "distributed together with the krill"
  [S]; saithe stack under krill swarms (Kaartvedt et al. 2005 [F]).
- Verdict: x0.7-1.5 per pair type, central ~1; prey-specific (krill near schools); it cannot reach x0.3, which
  needs within-cell segregation at r ~ -0.7 with CVs ~1 [D, lit_overlap section 0]. Patchiness without tracking
  lowers saturated intake (up to x0.36-0.68 at occupancy 0.2) but not a, and krill cell occupancy is 0.77-0.9.

(e) Capture per encounter in situ.
- No in situ capture for any model species [gap]. Lab per-attack bluefish lines differ ~2x between prey species at
  equal relative size (Scharf et al. 2003 [F], PROXY): prey-specific; the chain's bluefish line was measured in a
  449-L tank, on the tank-size plateau (475-900 L), so no tank correction [D on F]. In situ per-strike PROXIES:
  coral trout 0.17 (Scott et al. 2017 [F]), dottyback 0.26 ambush / 0.05 pursuit [S], sailfish on schools 0.24
  [A] (a school effect, i.e. F). Through the chain's q 0.8, r 0.3: per-encounter 0.18-0.27 = x0.29-0.52 of the
  chain's 0.52-0.63 [D], with reef-refuge and fullness caveats (the fullness part is Holling h already).
- Copepods: stickleback 0.53 (E. affinis) - 0.92 (Temora) per interaction (Viitasalo 1998 [F], PROXY); escape ability
  spans 10x among Baltic copepods but field herring selectivity does not follow it (Viitasalo 2001 [F]). Krill: no
  datum.
- Attack refusal of hungry fish ~0 in tanks (Juanes & Conover 1994; Staniland [S held]).
- Verdict: fish prey x0.8 (0.3-1.0) [J]; krill x1 (0.6-1.4) [J]; copepods x1 (0.8-1.2); prey-specific.

(f) Scan.
- Chain 0.5 (Varpe & Fiksen 2010, Langbehn 2019; Kulke 2018 field sprat clearance supports 0.5 over 1 [S held]).
  Saltatory and attention terms are the same kind of term; no new number moves it. Verdict x1.

## D1. Forward combination (Q3) (`scripts/s2_forward.py`, input `tables/term_corrections.csv`) [D]

Each term's weak / central / strong factor (in situ / chain) is mapped through the yardstick; 'share of needed' =
log(P/unc) / log(need/unc).

| group | central product: potential (need) | share of needed carried, central | residual generic a-factor after central | residual time share | all-strong corner |
|---|---|---|---|---|---|
| planktivores | 0.509 (0.262) | 0.06 (pout speed, weighted per pair: krill x1.72, copepods x1.05) | 0.32 | 0.51 | 0.181 |
| benthivores | 0.926 (0.633) | 0.01 | 0.059 | 0.68 | 0.262 (structure x0.2 and idle 0.5 needed) |
| whiting | 0.683 (0.390) | 0.15 (krill window [J] 0.06, speed 0.04) | 0.24 | 0.57 | 0.275 |
| water column | 0.638 (0.397) | 0.26 (fish window [J] 0.13) | 0.36 | 0.62 | 0.125 |
| mackerel | 0.511 (0.282) | 0.11 | 0.37 | 0.55 | 0.279 |

- Every evidenced term (tags F, A, D on F) is at or near x1 or raises a; what moves the central product is the [J]
  vulnerability-window residual carried over from "Intag under födosök". The all-strong corner reaches the need
  only by setting every term at its judged extreme at once (not a plausible combination).
- The single largest evidenced speed finding is in the opposite direction: pout's chain speed is ~2.6x (1-3.9)
  below the in situ speed; weighted by pout's share of each pair's a (krill 0.45, copepods ~0.03) it raises the
  planktivore krill a x1.72 and the copepod a x1.05.
- The 'gut-buffered' a-factor printed by s2 uses the upper-bound form min(1, wX) of s3; it is a bound for the
  window argument at X >> 1, not a second estimate of the need (the SSS form X/(1+X) is the engine's and the one
  used for every sizing here).

## J1. Reading [J]

1. The excess is shared WITHIN each predator (M1 per-prey test; M4 gate, krill residual <= ~1.5x in whiting and
   mackerel).
2. No single term of the derivation has field evidence of the needed size. Speed, scan and detection are at or above
   the field; within-bout pauses x0.9; capture and fine-scale overlap are prey-specific and near 1 (fish-prey capture
   x0.3-1 by reef proxies only).
3. Without the benthivores, one a-side factor of ~x0.2-0.35 (residual after the central corrections 0.24-0.37) fits
   the four other groups within the yardstick's roughness: a shared a-side term is possible there. The benthivores
   need x0.03-0.06 on a (X ~36), which no sized term reaches. In engine form a foraging-time share of 0.49-0.68
   (need) fits every group including the benthivores. But (i) no foraging-time budget of a wild adult exists; (ii) a daily idle
   share does not lower a gut-buffered, saturated fish's ration (M2), so the uniformity is an artefact of the
   engine's SSS ceiling; (iii) a fixed cap on the eat share is the removed `predator_share` (AGENTS.md: do not
   reintroduce); a learned one needs a rest payoff, ruled out as the main fix ("Vilans avkastning").
4. What the field data constrain is the PRODUCT: whole-chain field/theory ratios of x0.5-0.67 (herring clearance,
   R proportional to length), x1/50-1/260 (krill, January), stomachs below a cod-capelin encounter model, whiting
   fish plateaus. The missing factor is between "a fish that is foraging" and "a fish searching a mean-field prey
   at its swimming speed with its reactive distance" - i.e. the search-effort / encounter-realisation step the chain
   takes as 1 (search effort = 1 for the whole foraging time, every reactive-distance contact an encounter) - and no
   field study measures it for these species.
