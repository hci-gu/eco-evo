# Planktivore hide floors - summary (2026-11-11)

Research subagent. The main agent saves this text. Nothing in the library, code, tests, manifest,
AGENTS.md or resume changed; no training; nothing set from model output. Per-source numbers are in
`evidence.md` (model facts M1-M11, literature E1-E21, derived D1-D3). Grades: [F] field, [L] lab,
[A] abstract or catalogue record only, [S] secondary, [D] derived here, [J] judgement.

Files (run from `scripts/`; stdout saved next to each):
- `sms_quarter_m2.py` -> `sms_quarter_m2_stdout.txt`: SMS 2023 key run, predation M2 on herring,
  Norway pout and sprat by quarter, predator class and age (E21).
- `winter_feeding_payoff.py` -> `winter_feeding_payoff_stdout.txt`: does eating pay the planktivores
  in January vs July on today's library pairs, before predation (D1).
- `floor_readings.py` -> `floor_readings_stdout.txt`: the activity-schooling floor on the combined
  derivation's pair terms (D2) and the visible share at the winter rest shares (D3).
- `papers/`: empty - the Unpaywall helper found no open PDF for the six DOIs tried (evidence.md).

```
python3 -I sms_quarter_m2.py "<repo>/research_notes/Fiskgrupper i rutan/planktivores/data/sms2023"
python3 -I winter_feeding_payoff.py <repo>
python3 -I floor_readings.py
```

## Question

The planktivores' rest share is also a hide: a resting fish does not eat and predators see
B (1 - rest (1 - floor)); the same floor sets what that predator observes of them next tick (user kept
this coupling, research_notes/"Gom och sinnen"). Floors today: 0.25 for the four fish predators (pair
overrides, never motivated, resume 73), 0.35 species floor (used by seals, the only observer without an
override), 0.95 porpoises (tuned from 1.0, resume 72-73). Once the combined derivation
(research_notes/"Fangst och stim i fiskparen") puts schooling (F) and the per-encounter capture p with
pre-strike escape q into the fish pairs' a, is there a real refuge state left for the hide floor, and
what should each floor be?

## Model (checked)

- One floor per (observer, prey) acts on that pair's Holling term and on that observer's view of the
  prey (M1; `interactions.py:764-809`, `observations.py:39-78`). The shared harvest cap is the best
  detector's visible biomass, i.e. the porpoises' >= 0.95 B, so the fish floors only move each fish
  pair's own term (M2).
- Depth, bed and darkness are in a through the band profiles and vision rule A; F is per prey species
  and period (herring and sprat sigma 0.9 by day, 0 at night; pout 0.4 / 0; blue whiting 0.2 / 0); q 0.8
  is the pre-strike escape of a detected individual (M6). None of the three depends on whether the prey
  is feeding.
- The planktivore band profiles are NOT seasonal: herring and sprat by day use the HERAS July bottom-10-m
  share (0.04-0.06 over beds >= 120 m) plus 10-100 m, by night 0-50 m (M5). A winter position is not in
  phi_j.
- The mammal pairs carry no F or q: seals' a 0.002 (Type III) is a section 166 starting value, porpoises'
  a 0.0225 the section 151/206 diet fit (M7).
- budget_gate applies no hide floor (M9), so the floors are judged by mechanism here.

## Findings

### 1. Is there a refuge state that costs feeding and is not already in a?

A hide-floor refuge must be a STATE of the individual (it stops eating) that lowers its per-capita risk
in the same cell and 6 h tick beyond what F, q, the profiles and the light already give. Candidates:

| candidate | what the evidence says | in the box? | already in a? | verdict |
|---|---|---|---|---|
| Overwintering herring layer (non-feeding, dense, deep) | NSS herring wintered in Ofotfjord/Tysfjord in dense layers 100-400 m by day, at 1-2 ml/l O2 (E7-E9); WBSS keep their schools through the winter in the S Kattegat and the Sound (E1) | No: the WBSS wintering grounds are outside the box (E1); the box's winter herring are NSAS juveniles (E2) | - | not a box state [F on E1, E2] |
| Is a wintering layer a refuge at all? | Killer whales (~500) and saithe attack the overwintering layers (E10-E12); attacked schools become denser (E10) | - | protection by schooling = F | no per-capita evidence of a refuge; predators follow the layer [A] |
| Winter feeding in the region | Oslofjord herring > 25 cm feed day and night in Nov and Mar, 0-6 % empty, krill by day (E3); North Sea herring feed in February (E4) | yes (adjacent) | - | the box's herring do not enter a non-feeding state [F there] |
| Winter daytime position near the bed / deeper | IBTS Q1 catches of 1-2 ring herring higher by day, lower in early-morning dark (E5); Baltic herring aggregate at the bed by day, disperse at night (E6, October); wintering NSS herring descend at dawn (E7) | plausibly [J]; no winter acoustics for the box | NO: the profiles are July (M5) | a profile gap, not a hide: the depth belongs in phi_j (it raises overlap with the bed-band gadoids and lowers rule-A light) [J] |
| Hypoxic deep-water refuge (sprat) | gadoids avoid < 15-20 % O2 saturation, sprat tolerate ~7 % (E13, E14) | No: box water at 100-300 m 5.2-6.3 ml/l all year (M10) | - | fjord only [A; F/D on M10] |
| Reduced activity of a resting fish | wintering herring keep swimming 0.25-0.42 m/s (E8) | - | encounter speed is in a | small; no evidence [A] |
| Tighter schooling while not feeding | feeding herring schools stay schooled under cod and haddock attacks and keep feeding (E15) | - | F (population sigma 0.9 by day) | at most floor 0.93-0.98 on the clupeid day terms (D2, upper bound) [D on J] |
| Foraging-vigilance trade-off | foraging guppies react later and are attacked preferentially (E16, E17) | analog only | q is one average value | direction only, no herring or gadoid magnitude [A, L] |
| Norway pout near-bed by day; blue whiting below 150 m | - | yes | in the profiles (M5) | no residual |
| Pre-strike escape, evasive dives | saithe attacks make herring dive and stop feeding for minutes (E15) | yes | q and k (minutes, inside an encounter) | not a 6 h state |

No documented member state of the box's planktivores gives a residual refuge against fish predators
once F, q, the profiles and rule A are in a. The classic non-feeding overwintering layer exists but
belongs to adult NSS and WBSS herring outside the box, and even there its predators (killer whales,
saithe) attack it. The one real gap - the winter daytime depth - is a seasonal profile, which acts on
feeding and non-feeding fish alike.

Cross-check (E21, SMS 2023, confounded by predator presence, ration and other food): fish-predator M2 on
1-ring herring is highest in Q1 (0.38 /yr, 1.6 x the annual mean); 2+ herring Q1 0.13 vs annual 0.15. The
stomach-based model has no winter low on juvenile herring, so it does not support a winter refuge.

### 2. Protection the floor could still stand for, per predator type

- Gadoid and mackerel predators (saithe-dominated water column, whiting, cod/haddock, mackerel): F and
  q carry schooling and vigilance. If a resting herring were fully schooled by day while the population
  mean is 0.9, the pair floor would be 0.93-0.98 (WC 0.95, whiting 0.97, mackerel 0.93, benthivores
  0.98; D2). That is an upper bound on the protection: the population sigma already contains resting
  fish and feeding herring stay schooled (E15). Pout schooling at rest would give 0.39-0.63, but pout's
  day sigma 0.4 is itself a judgement with no state dependence. Range for the fish pairs: 0.9-1.0.
- Seals: visual and vibrissal pursuit predators; schooling protects against them somewhat (E20), and the
  seal pair's a carries no F. The floor is the only place that protection lives for seals until the seal
  a is derived. The seal note's 0.4 (0.3-0.5) [J] stands; 0.35 is inside it.
- Porpoises: biosonar works in the dark and through school and depth, capture success > 90 % on small fish
  (E19). The porpoise a is a diet fit, so realised schooling is in it. 0.95-1.0; the 0.05 "residual for
  schooling" was a tuning (M4) [J on A].

### 3. Observation under the blend-of-senses reading

With option A one floor serves both. A non-feeding herring layer still smells and sounds (schools are
heard at hundreds of m; "Gom och sinnen" section 3); sight and the lateral line do not reach at the km
scale. A floor of 1 on the fish pairs therefore makes their observation of the planktivores the whole
biomass, which is what the blend reading supports, and it removes the one-tick lag of a hidden share for
those observers (resume 239.3, the diet-gap lag). Seals keep observing 0.35-0.4 of a hiding planktivore:
defensible only as the schooling stand-in on predation; on the senses it is an under-statement, a cost
of keeping one floor for both (user decision, option A).

### 4. Seals 0.35 and porpoises 0.95

- Seals: keep 0.35 (or 0.4 per the seal note), [J]. Do not move it by the fish-pair argument: that
  argument holds only where F is in a. If the seal pair's a is ever derived from the field diet or an
  encounter model with schooling in it, its floor goes to 1 too.
- Porpoises: 1.0 (0.95-1.0) [J on A]. Effect on visible prey at most 0.05 x rest.

### 5. Consequences

Arithmetic (D3): the trained runs that hid most in winter showed visible 0.28-0.50 at floor 0.25, i.e.
rest 0.67-0.96 (fix78, fix80; descriptive). At floor 1 the same rest leaves the whole biomass to the fish
predators: x2.0-3.6 per-capita fish predation in those cells below saturation. At the annual mean rest
0.043 (fix97) the floor moves visible prey only from 0.968 to 1.000.

Energetics (D1, an upper bound: prey fully visible, krill at the open pool's per-cell target, no
depletion, today's pair values): in January eating pays in every cell, 24-40 MJ/t/tick net against an eat
premium of 3.8 (6.5-10.5 x); zooplankton alone 16.5 (4.3 x); July 14 x. Two separate statements:
(a) today's library: a floor of 1 does not strand the planktivores energetically in winter, and with
fish floors 1 hiding still buys a little against seals (booked M2 0.005 /yr) and almost nothing against
porpoises (0.0095 /yr at floor 0.95-1), against a booked fish M2 of ~0.35 /yr (~70 x larger). Expect the
learned winter hiding against fish to fall toward zero, winter intake to rise and the reserve to stay
higher through winter. (b) Why fix78/fix80 hid in winter is NOT answered here: those worlds had no krill
and a 0.25 floor, and the zooplankton-only payoff above still exceeds the premium, so their hiding was
not plainly energetic either (unknown; descriptive only).
The test is the long-run probe after the change (realised fill against the field start_fill cycle Jan
0.50 -> Apr 0.20; realised planktivore M2 against the booking), not a floor tuned to it: a floor below 1
would sell a protection no field source shows.

Two consistent pairings with q (no double count):
- Central: fish floors 1.0, q 0.8 the average pre-strike escape (vigilance lives in q only).
- Alternative [J]: fish floors 0.9 with q set for the feeding fish (q -> ~0.9-1.0); defensible only if a
  herring or gadoid study gives the foraging-vigilance magnitude (E16 is a guppy analog, direction only).

## Proposed floors

Ranges: the lower end 0.9 of the fish pairs is D2's clupeids-only bound (0.93-0.98) rounded down; D2
assumes a resting herring is fully schooled by day (sigma 1 vs the population 0.9), which no source
supports (E15 cuts against it). It is the far end of a bound, not an evidence-backed floor.

| observer -> planktivores | today | proposed (range) | grade | basis |
|---|---|---|---|---|
| water column (saithe-dominated) | 0.25 | 1.0 (0.9-1.0) | [D/J on F, A] | no residual refuge in the box (E1-E4, E8, E13-E15, M10); schooling in F, vigilance in q; D2 bound 0.95 |
| whiting | 0.25 | 1.0 (0.9-1.0) | [D/J] | as above; D2 bound 0.97 |
| mackerel | 0.25 | 1.0 (0.9-1.0) | [D/J] | as above; mackerel mostly absent in winter (x0.1); D2 bound 0.93 |
| benthivores (cod, haddock) | 0.25 | 1.0 (0.9-1.0) | [D/J] | as above; D2 bound 0.98 |
| seals (species floor) | 0.35 | 0.35-0.4 (0.3-0.5), keep | [J] | seal a has no F; schooling protects against a visual/vibrissal predator (E20) |
| porpoises | 0.95 | 1.0 (0.95-1.0) | [J on A] | biosonar, > 90 % capture (E19); a is a diet fit |

Applied together with the combined derivation (F and the new p), never before it: on today's a a floor of
1 would drop the only stand-in for schooling. Retrain afterwards (the observation of the fish predators
changes meaning). The species floor stays 0.35 for seals; the four fish pairs need the explicit pair key
visibility_floor 1.0 (required: omitting the key would give them the species 0.35).

## Gaps

- No per-capita predation rate for a herring wintering layer against the same fish feeding; the
  wintering evidence is NSS fjords (abstracts) and shows predators following the layer.
- No winter (Q1) acoustics or depth distribution of the box's NSAS juveniles; the July profile is used
  all year. A seasonal (winter) herring and sprat profile is the open item this note points to - it would
  move the overlap with the bed-band gadoids and rule A's light, for feeding and resting fish alike.
- No herring or gadoid magnitude for the foraging-vigilance trade-off (q_rest vs q_feed).
- Norway pout's state-dependent schooling (pout carries 0.52-0.75 of the fish pairs' planktivore a).
- Seal and porpoise pairs not derived from encounter theory; their floors stay judgement.
- The winter payoff (D1) ignores prey hiding, krill and zooplankton depletion and interference; a
  realised check needs a probe.

## Verdict

The hide floor of the planktivores against fish predators has no remaining biological content once
schooling (F) and the capture probability with prey escape (q) are in the encounter a. The one documented
non-feeding refuge state of herring - the dense winter layer - belongs to adult stocks that winter outside
the box (WBSS in the S Kattegat and the Sound, NSS in northern fjords), it is attacked there, and the
region's herring (Oslofjord, North Sea) keep feeding in winter. Hypoxic refuges need fjord water the box does not have. The winter
deep position is real but is a profile matter. Proposed: the four fish -> planktivore pair floors 1.0
(0.9-1.0), seals keep 0.35 (0.3-0.5) as the schooling stand-in until their a is derived, porpoises 1.0
(0.95-1.0), all applied with the combined derivation. Expect the learned winter hiding against fish to
disappear; judge the result with the long-run probe's realised fill and planktivore M2, not by tuning a
floor.

## User decisions

- 2026-11-11: the four fish -> planktivore pair floors go to 1.0 (explicit pair
  keys) in the combined section, with F and the new p, never before. Seals and
  porpoises on the planktivores are decided together with research_notes/"Sal
  och tumlare golv".
- 2026-11-11: investigate a winter (seasonal) profile for herring and sprat in
  the box.
