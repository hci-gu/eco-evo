# Benthivore and water-column hide floors - summary (2026-11-11)

Research subagent; the main agent saves this text. Nothing in the library, code, tests, manifest, AGENTS.md
or resume changed; no training; no value set from model output (rest shares from training.jsonl are quoted
descriptively). Sources E1-E19, model facts M1-M7 and derived D1-D3 in `evidence.md`. Grades: [F] field,
[L] lab, [A] abstract or catalogue record only, [S] secondary, [D] derived here, [J] judgement. Seal and
porpoise pair floors are decided (research_notes/"Sal och tumlare golv") and out of scope.

## Files

Run from `scripts/`; stdout saved beside each.
- `take_composition.py` -> `take_composition_stdout.txt`, `take_composition.json`: prey species and stage
  composition of the WC -> benthivores and whiting -> benthivores takes, with the library window machinery
  (execs research_notes/"Fonsterviktning" (o-umlaut) aw_stage_access.py up to its run block, as
  "Bottenfiskens hojdprofil" wc_benth_window.py does; reproduces [0.392, 0.0388] and [0.038, 0.0064]); the
  pair-equivalent floor of a flatfish-only refuge; hake-identity sensitivity.
- `floor_mechanics.py` -> `floor_mechanics_stdout.txt`: pairs, pair or species floor, observers, the shared
  cap, the visible share at the brief's and training.jsonl's rest shares.
- `papers/`: Ellis & Gibson 1995 and 1997, Ryer et al. 2004 (PDFs, not to be committed); Unpaywall records
  for six others (README.md).

```
python3 -I take_composition.py <repo> <scratch dir>            # < 1 s
python3 -I floor_mechanics.py <repo> <repo>/results/pzbpgp248nsrpfix97
```

## Question

1. Benthivores (haddock, cod, flatfish), species floor 0.55 ("gadoids 0.6, flatfish and rays bury and are
   cryptic (0.4)"): is there a real refuge state that (a) costs feeding and (b) is not already in a (band
   profile, rule A, F, q, recruit shape), against the water column (saithe, hake, anglerfish, ling) and
   whiting? Weighted by the stages and members the predators actually take; the engine applies one floor
   to J and A.
2. Water column, species floor 0.7 ("midwater saithe schools hide badly"): no fish eats it, so the floor
   acts only on what benthivores, whiting, mackerel and planktivores (all its prey) observe of a resting
   water column. What is a resting water column, and what can its prey sense of it at km scale?
3. Proposed floors with grades and consequences; any pair key needed to keep a decided mammal value.

## Model (verified, M1-M7)

- Visible = B (1 - rest (1 - floor)), predation this tick and every observer's view next tick, the pair
  floor on a checked pair else the prey's species floor; one rest share per cell for J and A
  (`interactions.py` visible_prey_biomass, `observations.py:39-78`). The shared cap is the best detecting
  checked predator's visible share.
- Benthivores: WC -> benthivores (a 0.0418, [0.392, 0.0388], M2 0.0833) and whiting -> benthivores (a 0.103,
  [0.038, 0.0064], M2 0.1067, OFF on the prey side, flagged open) carry no pair floor -> 0.55. Seals 0.85
  (decided 0.9) and porpoises 0.75 have pair keys; cap today 0.85 (seals). Observers through the species
  floor: WC, whiting (predators), planktivores (their prey: the benthivores eat them), mackerel (no link).
- Water column: eaten only by seals (key 0.75, decided 0.7) and porpoises (key 0.9); cap 0.9. Observed
  through the species floor by planktivores, benthivores, whiting and mackerel, for all of whom it is a
  predator: today a resting water column looks 0.3 x rest smaller to its prey (the predator-role leak of
  "Gom och sinnen" section 3).
- Costs: rest pays 1.0 x RM. Benthivores: RM 3.33, feeding_cost 1.02, so rest saves only 0.07 MJ/t/tick
  against eating; WC: RM 3.29, feeding_cost 2.14, rest saves 3.75 MJ/t/tick.
- Already in a (combined derivation, not yet applied): benthivores as prey in a 2 m bed slab, rule A with
  ONE contrast |C0| 0.7 for every fish prey, night/day from the rule (WC 0.165), F = 1 for benthivores,
  capture p with q 0.8 = pre-strike escape of a DETECTED prey, recruit_depth_table. Nothing lowers the
  detection of a buried or cryptic flatfish below that of a gadoid.
- Rest shares (training-world means, descriptive; training.jsonl is live, so a rerun of floor_mechanics.py
  prints newer values): brief benthivores 0.026, WC 0.032; training.jsonl at 624
  updates all-update mean 0.026 / 0.018, last 20 updates 0.006 / 0.003, max 0.11 (early training). The brief's
  WC 0.032 does not match the file's 0.018 (footnote only). No monthly action series exists: seasonal swings
  are undocumented.

## Findings - benthivores

### 1. What the two fish predators take (D1) [D on the library windows]

| share of the take | WC -> benthivores | WC, hake identity haddock + cod (sensitivity) | whiting -> benthivores |
|---|---|---|---|
| gadoid J (haddock, cod) | 0.641 (haddock 0.555) | 0.709 | 0.511 (haddock 0.441) |
| gadoid A | 0.140 | 0.155 | 0.000 |
| flatfish (all A by the engine's J/B) | 0.210 | 0.129 | 0.486 |
| of which long rough dab | 0.127 (0.60 of flatfish) | 0.074 | 0.399 (0.82 of flatfish) |
| rays, gurnard | 0.010 | 0.007 | 0.004 |

- Saithe (0.89 of the WC pair's a) takes juvenile haddock and cod only (identity). The WC's flatfish come
  through hake 0.093, ling (on anglerfish's window) 0.073 and anglerfish 0.044 - the adult part that
  research_notes/"Vattenpelarens vuxna bottenfisk" is re-deriving against those members' field diets; hake's
  in-area diet holds 0.4 % benthivores (sensitivity column).
- Whiting's flatfish share is the "all adult" bookkeeping of 12-15 cm long rough dab and dab
  (research_notes/"Fonsterviktning" open point 3); its booking is SMS whiting on haddock, juveniles only.

### 2. Candidate refuge states

| candidate | members / stage | (a) costs feeding? | (b) outside a? | verdict |
|---|---|---|---|---|
| lying still on the bed | all | partly | NO: Hobson's stationary-on-bed class (cod 70 %) is the occupancy behind the band profile; J sit within cm of the bed day and night, the 2 m slab (E13, E14) | in a [S/D] |
| shelter in cobble, kelp, sponge, eelgrass | 0-group cod (haddock, whiting over mud) | yes (hiding stops foraging, E7) | yes, but the box is mud and clay with sparse sea pens (E10); no study links mud epifauna to juvenile gadoid survival (E9) | not a box state [S/A] |
| spatial avoidance of a predator on fine substrate | juvenile cod | - | it is a move, which the engine has (E7) | not a floor [A] |
| night inactivity of juveniles | 0-group cod | yes (strict diurnal foraging, E11) | darkness is in rule A (night/day 0.165); gadoids eat 0-group most at dusk and night (E12) | no refuge [A/S] |
| schooling | J gadoids | - | F = 1 (sigma 0) | none to count |
| burial / crypsis | flatfish (0.21 WC, 0.49 whiting) | YES: buried = inactive (E5); age-2 halibut buried 50-60 % after a meal, 20-35 % when hungry, "a hunger-mediated trade-off between refuging and foraging" (E4) | YES: rule A uses one contrast for every fish prey; q is escape after detection | the one residual [L/A, J] |

Protection magnitude of burial: the one buried-vs-not design found (E3, Ansell & Gibson 1993, aquaria, 0-group
plaice) found an effect of sand ONLY for pollack, none for cod; "only a partial refuge"; magnitudes not
available. Ryer et al. 2004 (E4) give a SPECIES contrast, a cryptic burier (rock sole) eaten 0.75 as often per
24 h and 0.43 as often per unit predator effort as a flusher (halibut) - not a buried-vs-exposed value. On a
sand nursery gadoids, whiting among them, eat small flatfish at night; protection there is by size, > 45 mm
(E1). So the flatfish floor x_flat is [J] with direction only: ~1 against cod-like, tactile or ambush
predators, below 1 against visual searchers (pollack-like; hake and whiting are visual). Central x_flat 0.6
[J], range 0.3-1.0. Against anglerfish (an ambusher; 0.044 of the WC take) burial and motionlessness are the
same physics as the rejected "resting prey is motionless" convention; not separated here.

### 3. From a member refuge to a pair floor

The engine has one rest share per cell and one floor per pair for J and A: a flatfish floor would be paid to
the haddock juveniles in the same cell. The pair-equivalent floor is therefore the take-weighted

    floor_pair = 1 - w_flat (1 - x_flat)

weighted by the predator's TAKE (the floor acts in that pair's Holling term), not by biomass (biomass
weighting fits only the observation, where the answer is 1 anyway). WC: 0.92 at x 0.6 (0.85 at x 0.3, 1.0 at
x 1); hake sensitivity 0.95. Whiting: 0.81 at x 0.6. Both w_flat values are unstable inputs (window
conventions, the parallel note); recompute when the WC adult part lands.

### 4. Observation (blend of senses)

A buried flatfish or a gadoid on the bed still smells; at km scale smell and sound carry the blend, sight and
the lateral line reach metres (E18). For mackerel (no link) and planktivores (prey of the benthivores; the
predator-role leak) the species floor has no sense behind it below 1. With the coupling kept, a WC pair floor
0.92 also lowers what the WC observes of a resting benthivore cell by 0.08 x rest.

## Findings - water column

- What rest is, per member (D3; M6 biomass): saithe + pollack 0.805 - non-feeding aggregations sit nearer the
  bed (Hs 2.5-9 m vs 15 m foraging) and activity drops at night and in winter (E15), a 6-12 m shift inside the
  1-21 m per-tick vertical spread of a cue (E18) -> factor 1.0; spurdog 0.097 obligate swimmer -> 1.0 [J];
  anglerfish 0.027 and ling 0.014 bed-bound ambushers, rest = forage -> 1.0 [J]; hake 0.048 on the bed by
  day, up at night to feed (E17, [S/J]) -> 0.5 (0-1) for pelagic observers, 1.0 for bed observers; velvet
  belly 0.009 -> 1.0 [J].
- Weighted: pelagic observers (planktivores, mackerel) 0.976 (0.952-1.0); bed observers (benthivores,
  whiting) 1.0 [D on S/J].
- Activity state is read at metres, not km: juvenile cod ignored a non-foraging large cod but hid from a
  foraging one (tank, E7); herring dived from a fast attacking saithe school seen by sonar at tens of metres
  (E16). No source shows a fish reading a predator's activity at km scale or a non-foraging saithe school
  being less detectable by smell or sound. A floor below 1 therefore leaks the predator's last activity
  state into its prey's km-scale observation, with no sense behind it [J].
- The WC floor never acted on predation (no fish eats the WC; the mammals have keys); "midwater saithe
  schools hide badly" was a predation argument that has no pair to act on.

## Mechanical consequences

- Size today (D2): benthivores at rest 0.026, visible 0.988 (floor 0.55) -> 1.000 (1.0) or 0.998 (0.92);
  latest rest 0.006: 0.997 -> 1.000. WC at rest 0.032: 0.990 -> 1.000. A fully resting cell: 0.55 -> 1.0 /
  0.92 (benthivores), 0.7 -> 1.0 (WC). Every annual-mean change < 1.2 % of B.
- Who is affected: WC and whiting predation on resting benthivores (pair terms), their observation of
  benthivores, and planktivores' and mackerel's observation of benthivores; planktivores', benthivores',
  whiting's and mackerel's observation of the WC. Mammals: none (all four mammal pairs carry explicit keys:
  seals -> benthivores 0.85 key, decided 0.9; porpoises -> benthivores 0.75; seals -> WC 0.75 key, decided
  0.7 - equal to today's WC species floor, so it MUST stay an explicit key; porpoises -> WC 0.9). No new pair
  key is needed for the mammals.
- Shared cap: benthivores' cap moves from seals' 0.85 (0.9 decided) to 1.0 (whiting's floor 1.0); WC's
  cap stays 0.9 (porpoises).
- Rest payoff: for the benthivores rest saves 0.07 MJ/t/tick against eating; at floors ~1 rest buys almost
  nothing, so the learned rest share (already 0.006) should go to ~0 - except where a WC pair floor < 1 keeps
  a small flatfish refuge. For the WC rest stays a feeding pause (3.75 MJ/t/tick saved).
- The exposure-weighted M1 (section 139, `natural_mortality.py`) reads a DM's own species floor, but no
  library group sets `m1_visual_share`, so the species floors move no M1.
- budget_gate has no floor term: no gate number moves. Growth budgets unchanged (floors are not in s*).
- Order: into the combined section. For the benthivores F = 1, so the floor never carried schooling and the
  ordering holds by convention, not dependence; a flatfish burial floor acts on DETECTION and does not
  double-count q (escape after detection), unlike the planktivore case. Shapes unchanged; observation inputs
  change meaning: retrain.

## Proposed floors (not applied)

| floor | today | central | range | grade | basis |
|---|---|---|---|---|---|
| benthivorous_fish species floor (mackerel, planktivores observe; any pair without key) | 0.55 | 1.0 | 0.9-1.0 | [J on F/L/A] | gadoid J (0.51-0.71 of the fish take) have no refuge state on mud outside a (E7, E9-E14); km-scale blend of senses not cut by a rest (E18) |
| predatory_fish -> benthivorous_fish (pair key) | 0.55 (species) | 0.92 = 1 - 0.21 (1 - 0.6) (x_flat [J], w_flat from today's windows) | 0.85-1.0 (0.95 at hake's field identity) | [D on J; L/A] | flatfish burial is a feeding-exclusive, detection-lowering refuge not in a (E3-E5); x_flat 0.6 [J]; recompute with research_notes/"Vattenpelarens vuxna bottenfisk" |
| whiting -> benthivorous_fish (no key; species) | 0.55 (species) | 1.0 | 0.8-1.0 (0.81 if the 12-15 cm LRD/dab window is real whiting prey) | [D/J] | the flatfish share 0.49 is window bookkeeping (Fonsterviktning open 3); the booking is haddock juveniles; pair OFF, a floor is not its fix |
| predatory_fish species floor (observation by planktivores, benthivores, whiting, mackerel) | 0.7 | 1.0 | 0.95-1.0 | [D/J on S/A] | resting members stay in their prey's reach (D3: 0.976 pelagic, 1.0 bed); activity state readable at metres only (E7, E16); ends the predator-role leak |
| seals / porpoises pair keys on both groups | 0.85 / 0.75 / 0.75 / 0.9 | as decided: 0.9 / 0.75 / 0.7 / 0.9 | - | decided | explicit keys exist; seals -> WC 0.7 must stay a key |

Alternative if the user prefers no member-weighted pair floor: WC -> benthivores 1.0 (no key; the flatfish
refuge dropped as unmeasured), consistent with the whiting and mackerel decisions; the cost is ~0.002 of B at
today's rest share.

## Gaps

1. Long rough dab (0.60 of the WC's flatfish take, 0.82 of whiting's): no burial or crypsis data found (one targeted
   search); whether it buries when resting is unknown. Dab, plaice on 100-300 m mud: no data either.
2. No burial protection magnitude against hake, ling, anglerfish or whiting; the only design (E3) is behind a
   paywall and tested pollack and cod only. x_flat is judgement.
3. w_flat depends on window conventions (hake's identity "all", the unkeyed "all adult" flatfish) and on the
   parallel adult-benthivore note; the WC pair value must be recomputed after it.
4. Anglerfish (ambusher): burial and motionlessness are inseparable from the rejected motionless convention.
5. Hake's diel position in the Skagerrak: not read at source (E17 [S]); it moves only the WC's pelagic factor
   between 0.952 and 1.0.
6. No study of a fish sensing a non-foraging vs foraging predator school at km scale.
7. Monthly rest shares are not recorded; the floors' seasonal weight is unknown.

## Verdict

Benthivores: for the gadoid part of what the fish predators take (0.78 of the WC's, 0.51 of whiting's;
mostly juvenile haddock) no refuge state survives once the bed slab, rule A, F and q are in a: "on the bed" is
the band occupancy, structure shelter needs substrate the box lacks, and night inactivity coincides with
darkness that is already in a. The flatfish part meets both criteria - burial is inactivity, traded against
foraging, and nothing in a lowers a buried flatfish's detection - but its protection is measured only as
"partial" against a pollack and nil against cod, and the flatfish the windows hand the predators are mostly
long rough dab, whose burial is unmeasured. Proposed: species floor 1.0; WC -> benthivores as a take-weighted
pair key 0.92 (0.85-1.0), or 1.0 if the unmeasured flatfish refuge is dropped; whiting -> benthivores 1.0.
Water column: a resting saithe school, hake, anglerfish or ling stays within the cue range of its prey (D3:
0.976-1.0), and prey read a predator's activity only at metres; the floor acts on nothing but that leak.
Proposed 1.0 (0.95-1.0). The mammal keys stay as decided; none needs adding. All effects < 1.2 % of B at
today's rest shares; apply in the combined section and retrain.

## User decisions

- 2026-11-11: all 1.0 in the combined section: benthivores species floor,
  whiting -> benthivores, water column -> benthivores (the unmeasured flatfish
  burial refuge dropped; its share is small after the adult-part corrections),
  water column species floor. Mammal keys as decided; seals -> water column 0.7
  stays an explicit pair key.
