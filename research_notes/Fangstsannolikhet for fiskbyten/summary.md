# Capture probability on fish prey - summary (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library,
code, tests, manifest, AGENTS.md or resume; no training; nothing set from model
output (library copies only in the session scratchpad). Per-source numbers:
`evidence.md` (E1-E15, model facts M1-M6, derived D1-D4). Grades: [F] field,
[L] lab (full text read), [A] abstract or search summary only, [S] secondary,
[D] derived here, [J] judgement. PPR = prey : predator total length.

## Files

Run from `scripts/`; stdout saved next to each.

- `p_ppr.py` -> `p_ppr_stdout.txt`, `p_ppr.json`: per-strike success vs PPR fitted
  on the bluefish data (E5); cross-species per-strike values (E1, E3, E4, E7); the
  follow-up probability r inverted from the tank sequences (E1); p per encounter
  per pair and reading; the gut-fullness factor (E4).
- `gate_p.py` -> `gate_p_stdout.txt`, `../tables/gates/budget_gate_<reading>.txt`,
  `../tables/gates/gate_p.json`: tools/probes/budget_gate.py unchanged with
  `library_path=<copy>`, the copies carrying max_intake_rate x (p_new / p_library)
  on the nine fish-prey pairs; the `library` run is byte-identical to
  research_notes/"Stimskydd i motes-a" `budget_gate_library.txt`. Also recomputes
  that note's schooling factor F at the new dispersed p.
- `papers/`: Unpaywall-helper downloads (Juanes & Conover 1994, Herbert-Read et
  al. 2016, Langbehn et al. 2019) and records for DOIs without an open copy
  (Scharf 1998 and 2003, Webb 1986, Christensen 1996). Not committed.

```
S=<session scratchpad>
python3 -I p_ppr.py p_ppr.json                                    # < 1 s
python3 gate_p.py <repo> p_ppr.json $S/libs ../tables/gates       # ~1 min, 11 gate runs
```

## Question

Every visual fish-predator pair carries a capture probability p: 0.2 for the water
column (WC), whiting and mackerel (the section 220 convention), 0.3 for the
benthivores. The 0.2 is Turesson & Bronmark's perch on single roach, 34/168 per
ATTACK; a review figure 0.45 circulates in two sibling notes. (1) What must p be in
the encounter formula? (2) Field and lab values for piscivores on dispersed fish?
(3) Mackerel on fish? (4) Group values, and how do a and budget_gate move? (5) Is
0.2 / 0.3 right?

## Model (as given and checked)

- Form (M1): a = scan x A(R, H) x v_rel x T x N_t x p x f / H, scan 0.5
  (`pred_encounter_a.py` a3d); no pursuit, abort or attack-choice term.
- R (M2): rule A's DETECTION distance (section 269): image-area visual range capped
  at 2 L_pred, lateral-line floor 0.3 m; calibrated on salmonid reactive distance
  (first orientation). Light acts on R, not on p.
- Engine (M3): fish attack every prey they meet (section 265); appetite is the
  reserve-room cap (h = 1), saturation is Holling h. Schooling belongs in a separate
  factor on the schooled share (k prey per school met replaces p; Stimskydd), so
  this p is for dispersed or single prey only.
- Member shares of today's a (M4): WC fish pairs saithe 0.89-0.92; WC -> mackerel
  spurdog 0.76, hake 0.22; whiting pairs whiting 1.00; mackerel pair mackerel 0.96;
  benthivore fish pairs cod-weighted.

## Findings

### 1. Definition: p is per ENCOUNTER (per detection) of a hungry predator

p must be P(prey eaten | prey crossed the detection disc, predator hungry); several
strikes on one prey inside one detection are one encounter. Three parts [D]:
p = q x p_s / (1 - r (1 - p_s)), q = P(strike | detected) = 1 - pre-strike escape,
p_s = success per strike, r = P(another strike | miss).

- q: close to 1 in tanks (an empty whiting "struck at almost every shrimp they
  orientated towards", E4; bluefish "Prey items were rarely ignored", E5). In open
  water the prey can see the predator first ("At high light, bluegills can spot
  modest-sized bass well before being spotted", E13); prey reaction distance drives
  vulnerability in Scharf 2003 (E9) [A].
- r: inverted from per-strike vs per-sequence success (E1) [D]: perch 0.93, pike
  0.47, pikeperch 0.09, all in a 2.44 m pool the prey could not leave. Bluefish
  "Unsuccessful attacks were not usually followed by pursuits" (E5); whiting "would
  either rapidly reorientate, or simply turn away" (E4); cod "would often continue
  to pursue prey that escaped" (E3, 60 x 33 cm arena).
- Central [J]: q 0.8 (0.6-1.0), r 0.3 (0-0.6).

Consequences:
- Per-strike values are the lower lab bracket (0.20 perch single prey, cod 0.05-1,
  bluefish 23-87 %); per-sequence values in confined tanks the upper one (perch
  0.74, pike 0.25, pikeperch 0.093 - the PDF text drops decimal points: "93" is
  9.3 %). "Laboratory tanks ... are generally too small ... all prey are usually
  within the visual field" (E2).
- Section 220 applied a per-strike figure as if per encounter (M1). Its argument
  "per encounter is lower still because not every encounter is attacked" conflicts
  with section 265 and the appetite model: the satiation part of "not attacked" is
  already in the engine (Holling h, reserve room); the pre-strike escape part is
  not, and that is q. Today's 0.2 is self-consistent only under a defence 220 did
  not make: per-strike 0.2 (perch) with the central q and r gives 0.21 (`percid`).
- The "review 0.45" comes from Juanes, Buckel & Scharf 2002 (E2): "Usually about
  half of piscivore attacks on average are successful", per ATTACK and secondary.
  The sibling notes' 0.45 vs 0.2 was per strike against per strike, not a
  definitional correction.
- Boundaries (no double count): R carries detection and light, so p stays
  light-independent, and dark_ratio and the stomach night/day ratios are untouched
  (lab values are bright; E12 bass > 95 % success down to moonlight and E3 escape
  higher in dim open water point opposite ways). Schooling stays in k. PPR is taken
  inside the stage-access window, so the gape-limit zero is not counted twice. q vs
  the hide floors: if the planktivore pair floors (0.25) stay after schooling moves
  into a, q double-counts prey vigilance; if they go to 1, q is the only place for
  it.

### 2. Values for piscivores on dispersed fish (per strike unless stated)

| predator / prey | PPR | value | grade |
|---|---|---|---|
| perch / roach, single prey (pool) | 0.27 | 0.20 per strike; 0.74 per sequence | [L] E1 |
| pike / roach | 0.30 | pursuit 0.12, sit-and-wait 0.36; sequence 0.25 | [L] E1 |
| pikeperch / roach, dusk | 0.30 | 0.085; sequence 0.093 | [L] E1 |
| 0-group cod / 0-group flatfish (60 x 33 cm arena) | 0.20-0.33 / 0.35-0.45 | most trials 1.0 (rest 0.33-0.5) / 0.05-0.3 | [L] E3 |
| whiting 23-30 cm / shrimp, empty | 0.2-0.34 | 0.8 (model: 0.8 + s x stomach g) | [L] E4 |
| juvenile bluefish / silverside (17 points) | 0.16-0.62 | p_s = 0.973 - 1.230 PPR (SD 0.11): 0.73 / 0.67 / 0.60 / 0.48 at 0.20 / 0.25 / 0.30 / 0.40 | [L] E5, [D] |
| bluefish, two prey species | 0.20-0.65 | linear decline; lower on striped bass than silverside | [A] E6 |
| esocids / gizzard shad (a clupeid), minnow, bluegill | "optimal" | 0.78 / 0.67 / 0.14 | [A] E7 |
| trout, bass, rock bass, tiger musky / minnow | - | ~0.3-0.6; musky ~0.8 (1 - response x evasion) | [A] E8, [D] |
| sailfish / sardine SCHOOLS (field) | - | 51/210 = 0.24 | [F] E10 (school, not dispersed) |
| generic piscivore model | - | 0.7 per encounter | [S] E11 (assumption) |
| piscivores in general | - | "about half"; planktivores 0.70-0.80 | [S] E2 |

- PPR dominates: every size-resolved set declines with PPR (E2, E3, E5, E6).
- At PPR 0.25-0.33 two clusters separate: percids and pike 0.09-0.36; gadoid tanks
  (cod, whiting), bluefish and esocids on soft-rayed prey 0.6-0.8.
- No gadoid capture success on clupeids or other pelagic fish, in tank or open
  water, was found.
- Gut fullness lowers per-strike success strongly (E4): x0.77-0.47 on small shrimp,
  x0.62-0.11 on large shrimp, at a mean stomach 0.22-0.53 of the maximum; central
  x0.6 [L one species and prey type; D/J on the stomach mapping].

### 3. Mackerel on fish

No mackerel capture success on fish found (Folkvord & Hunter 1986, chub mackerel on
anchovy LARVAE, is outside the size domain). Mackerel takes age-0 sprat and herring
at PPR ~0.15-0.31 (window p95 0.31-0.33, M5). The bluefish line (a pelagic lunger,
same prey type and PPR) gives p_s 0.70 at PPR 0.22 [J on L-analog]. Nothing new on
krill.

### 4. Pair and group values, and the effect on a

p_s from the bluefish line at each pair's PPR (D3: the encountered prey inside the
window, from the realised stomach ratios and window p95 in research_notes/
"Arsungarnas andel" (A-ring)). The bluefish line is the central because it is the
only size-resolved set and sits between the gadoid-tank cluster (0.8) and the
percid cluster (0.1-0.36), not because bluefish resembles a gadoid [J on L]. a is
linear in p in every section 269 term, so the factor is p_new / p_library.

| pair | p_lib | PPR | p_s line | low | percid | central | per strike | high | factor central |
|---|---|---|---|---|---|---|---|---|---|
| WC -> planktivores | 0.20 | 0.25 (0.19-0.33) | 0.67 | 0.12 | 0.21 | 0.59 | 0.67 | 0.91 | x2.96 |
| WC -> benthivores, WC -> whiting | 0.20 | 0.25 | 0.67 | 0.12 | 0.21 | 0.59 | 0.67 | 0.91 | x2.96 |
| WC -> mackerel (spurdog at PPR ~0.4, no elasmobranch data) | 0.20 | 0.30 | 0.60 | 0.12 | 0.21 | 0.55 | 0.60 | 0.91 | x2.74 |
| whiting -> planktivores | 0.20 | 0.28 (0.26-0.40) | 0.63 | 0.12 | 0.21 | 0.57 | 0.63 | 0.91 | x2.83 |
| whiting -> benthivores | 0.20 | 0.33 (0.27-0.42) | 0.57 | 0.12 | 0.21 | 0.52 | 0.57 | 0.91 | x2.61 |
| mackerel -> planktivores | 0.20 | 0.22 (0.15-0.31) | 0.70 | 0.12 | 0.21 | 0.62 | 0.70 | 0.91 | x3.09 |
| benthivores -> planktivores | 0.30 | 0.20 (0.16-0.30) | 0.73 | 0.12 | 0.21 | 0.63 | 0.73 | 0.91 | x2.11 |
| benthivores -> whiting | 0.30 | 0.22 (0.16-0.33) | 0.70 | 0.12 | 0.21 | 0.62 | 0.70 | 0.91 | x2.06 |

Readings: low p_s 0.2, q 0.6, r 0; percid p_s 0.2, q 0.8, r 0.3; central line,
q 0.8, r 0.3; per strike line, q 1, r 0; high p_s 0.8, q 1, r 0.6. A confined-tank
sequence (r 0.9) gives 0.93-0.96.

Group values (central): WC 0.59, whiting 0.54, mackerel 0.62, benthivores 0.63;
range per pair 0.12-0.91. Biomass- and a-weighted values coincide, since one member
carries each pair under both (saithe 0.80 of the WC biomass and ~0.9 of its a;
whiting 1.0; mackerel 0.96; cod the only benthivore fish-eater of weight). The PPR
rule is per member; a biomass mean over haddock and flatfish would be meaningless.

Field-mean level (the fullness caveat): x0.6 on p_s gives central 0.34-0.42, x1.4-2.0
the library value. That is a LEVEL correction; fullness is a predator state, so it
would act on that predator's krill and benthic pairs too and cannot be applied to
the fish pairs alone. Whether a carries the hungry rate (the engine's fish are
hungry until the reserve is full; saturation from Holling h) or the field-mean rate
is a convention for all pairs: a user decision.

### 5. budget_gate (equal cells unless noted; * = OFF)

| | library | low | percid | central | per strike | high | school only | central x school |
|---|---|---|---|---|---|---|---|---|
| planktivores, prey total | 0.69 | 0.45 | 0.66 | 1.24 | 1.33 | 1.55 | 0.50 | 0.81 |
| WC -> planktivores | 0.78 | 0.59 | 0.80 | 1.18 | 1.21 | 1.30 | 0.54 | 0.82 |
| whiting -> planktivores | 0.54 | 0.33 | 0.56 | 1.30 | 1.41 | 1.83 | 0.45 | 0.86 |
| mackerel -> planktivores | 0.78 | 0.47 | 0.81 | 2.20 | 2.46 | 3.06* (predator side) | 0.56 | 1.38 |
| benthivores -> planktivores | 0.61 | 0.25* | 0.43 | 1.22 | 1.38 | 1.68 | 0.42 | 0.70 |
| benthivores, prey total | 0.62 | 0.51 | 0.63 | 0.86 | 0.88 | 0.98 | 0.69 | 1.10 |
| WC -> benthivores (field spread) | 0.97 (1.13) | 0.73 | 1.00 | 1.46 (1.71) | 1.51 | 1.61 | 1.17 | 2.11 (2.45) |
| whiting -> benthivores (field) | 0.12* (0.07*) | 0.08* | 0.13* | 0.27* (0.17*) | 0.29* | 0.42 | 0.12* | 0.29* |
| whiting, prey total | 0.60 | 0.44 | 0.56 | 0.88 | 0.93 | 1.04 | 0.65 | 1.06 |
| WC -> whiting | 0.64 | 0.48 | 0.66 | 0.97 | 0.99 | 1.06 | 0.77 | 1.37 |
| benthivores -> whiting | 0.31* | 0.13* | 0.22* | 0.61 | 0.69 | 0.87 | 0.32* | 0.63 |
| WC -> krill (field spread) | 1.39 (1.40) | 1.74 | 1.36 | 0.71 (0.72) | 0.65 | 0.51 | 1.67 | 1.02 (1.02) |
| WC -> zooplankton (field) | 0.38 (0.62) | 0.48 | 0.37 | 0.20 (0.32) | 0.18 | 0.14 | 0.46 | 0.28 |
| mackerel, prey total (WC -> mackerel OFF) | 0.02 | 0.01 | 0.02 | 0.03 | 0.03 | 0.03 | 0.01 | 0.00 |

Two further columns in the tables, `central_full` and `full_x_school` (x0.6 on the
fish pairs only), are excluded: an unsupported asymmetric sensitivity (the gate
judges encounter shares, and fullness would act on krill too); their planktivore
total 0.66 is an artifact, not a result.

- Central passes the planktivore booking (0.69 -> 1.24), every planktivore pair
  above 1 (mackerel 2.2). The level is a hungry-predator level; the field-mean one
  would sit lower.
- Schooling and p interact: S = ... k / (N p), so the schooled share keeps its
  absolute rate while the dispersed share rises with p, and F falls as p rises (WC
  -> planktivores 0.57 -> 0.48; whiting 0.83 -> 0.61; mackerel 0.71 -> 0.60;
  benthivores 0.69 -> 0.55). Central x school gives planktivores 0.81 (WC 0.82,
  whiting 0.86, benthivores 0.70, mackerel 1.38). The two notes cannot be applied
  independently. Stimskydd's S used P_I 0.2 on the benthivore pairs where the
  library p is 0.3; `school_only` reproduces its numbers, so that inconsistency is
  carried over.
- Krill moves toward the field value: more fish encounters shift the WC diet off
  krill under shared handling; the WC krill share at the field spread falls 0.51 ->
  0.26 (booked 0.36; field saithe all seasons 0.19, 0.08-0.27; group Q3 0.47);
  central x school 0.37.
- One pair moves the wrong way: WC -> juvenile benthivores, 0.97 -> 1.46 at equal
  cells (1.71 at the field spread), 2.11 / 2.45 with schooling (lost planktivore
  encounters redistributed). The vision and overlap notes already flagged this pair
  as hinging on the overlap convention: a consistency failure of that pair, not a
  reason for a lower p.
- OFF: central adds none, and benthivores -> whiting leaves OFF (0.31 -> 0.61);
  whiting -> benthivores stays OFF except in high (0.42); high puts mackerel ->
  planktivores OFF on the predator side (3.06); low puts benthivores ->
  planktivores OFF (0.25) and whiting -> planktivores at the field spread OFF
  (0.27). WC -> mackerel stays OFF throughout (spurdog on mackerel at PPR ~0.4, the
  rule outside its evidence).
- The gate is a check, not a target; no value here was chosen to hit 1.

## Gaps

- No gadoid (cod, saithe, whiting, haddock, pollack, hake) capture success on
  pelagic fish, tank or open water; the only gadoid data are 0-group cod on 0-group
  flatfish (E3) and whiting on shrimp (E4).
- q and r unmeasured outside confined tanks (perch r 0.93 from a 2.44 m pool;
  Staniland's re-orientation counts only as flow diagrams).
- The fullness effect rests on one species and prey type, with an approximate
  stomach mapping.
- No full texts for Scharf 1998 and 2003, Webb 1982 and 1986, Wahl & Stein 1988,
  Christensen 1996 (no open copy); Scharf 2003 would give prey-specific attack and
  capture probabilities for marine forage fish.
- No mackerel-on-fish datum (mackerel -> planktivores [J on L-analog]); no
  elasmobranch capture data for WC -> mackerel.
- Light: all lab values bright; dim-light p_s could be lower (more escape, E3) or
  higher (prey see less, E13); kept out of p by convention.

## Verdict

Definition [D on L]: p must be P(capture | detected) for a hungry predator - per
encounter, repeated strikes and pre-strike escape included, light and schooling
excluded. Today's 0.2 is a per-strike percid figure applied as if per encounter;
it is self-consistent only as "percid per strike with open-water corrections",
which section 220 did not argue. The review 0.45 is also per strike.

Value: p rises as PPR falls; at the box pairs' PPR (0.20-0.33) the hungry
per-encounter p is central 0.55-0.63 (WC 0.59, whiting 0.54 (0.52 on juvenile
benthivores), mackerel 0.62, benthivores 0.63), defensible range 0.2-0.9, floor
0.12 [J on L: bluefish line, gadoid tanks above it, percids below, q and r judged].
At the field-mean gut fullness the level would be ~0.35-0.42 [L one species, J], a
convention for every pair (krill and benthic included), not the fish pairs alone.
Both library values are low by x2-3 (hungry) or x1.4-2 (field-mean) in every
reading above `percid`.

Benthivores vs the 220 convention: the ordering survives (cod's prey relatively
smaller, PPR 0.20 vs 0.25-0.33), but the gap is ~10-15 %, not x1.5; 0.3 is as low
as 0.2 relative to the evidence.

Gate: central p takes the planktivores past the booking (1.24); with Stimskydd's
schooling factor recomputed at the new p they land at 0.81. WC -> juvenile
benthivores moves further over its booking and needs its own fix (overlap
convention), not a lower p.

## User decisions

- 2026-11-11: a carries the HUNGRY predator's rate (satiation stays in the
  engine's Holling h and reserve-room cap); no field-mean fullness factor.
- 2026-11-11: next, one combined derivation (new p, schooling factor with the
  Norway pout values at the new p, the benthivore pairs' p made consistent),
  gated on library copies; nothing applied.
