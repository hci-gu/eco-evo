# Smell and predator avoidance - summary

Question: does the literature support the model's km-scale observation of other
groups' biomass, read as a loosely modelled sense of SMELL, as a basis for whiting
avoiding the benthivores (cod, haddock, flatfish) within bed-depth classes
(model Lloyd's L ~0.1 vs field 1.3-1.9; resume 266.7)? Research only, nothing in
the code or library changed. Per-source numbers: evidence.md (E1-E29, E14b).
Grades: [F] field, [L] lab, [J] judgement, [A] abstract/secondary only.

## 0. The model's observation (verified 2026-11-10)

Each tick, before any process, every decision maker reads the exact tonnes per cell
of each group in its `observes` list in its own cell and the N/E/S/W neighbours
(zeros beyond the edge), scaled only by last tick's hiding (benthivores' visibility
floor 0.55, so whiting sees 0.55-1.0 of their biomass). It is z-scored with training
statistics: linear in density, no saturation, no noise, no lag beyond the tick, no
memory in the policy. All four neighbours are read the same way; currents move only
plankton groups and never enter the observation.

Code: `lib/environments/ecosystem_env/environment.py:213-226` (built once per tick
from the previous state); `observations.py:7-17, 20-127` (layout, one-cell shifts,
zeros at the edge), `39-47, 69-78` (visible fraction `1 - rest_prev*(1 - floor)`),
`82-100` (juvenile channels); `decisions.py:29-48` (z-score, clip +-10), `211-237`;
`state.py:314-326, 359-366`; `trainer.py:258-262`, `parallel_worker.py:163-168`
(training statistics, fixed within a rollout); `movement.py:104-164` and
`currents.py:36-39` (currents off by default, fix95 strength 0.1, period 20; only
groups with `current_response` > 0 drift, `lib/world/functional_group.py:50-53`);
GPU `lib/gpu/ecosystem.py:560-600` (same layout).

## 1. Detection range, thresholds and persistence of predator cues

- Juvenile cod (9 cm) avoided water from a tank with one 500 g trout, in a
  32 x 40 cm flume at 1 cm/s (E1) [L]. No study found on predator odour range,
  threshold or density discrimination in cod, haddock, whiting, herring or sprat.
- Cod as the predator: wild gobies avoided a feeder with cod image and/or odour for
  1-2 days (E2); naive gobies ignored cod odour until a live cod had been present at
  3 sessions, after which the avoidance lasted 6 months (E3) [L]. Predator odour is
  partly a learned label.
- Club-cell Schreckstoff is characteristic of Ostariophysi (E12) [A]; damage-released
  alarm cues are documented in other taxa, e.g. reef damselfish (E10) [F]; none found
  for gadoids or clupeids, and goby skin extract had no effect (E2) [L].
- Measured ranges are metres: freshwater minnow alarm cue 2.3 m (lab dilution),
  2-8 m in wild populations (E9); reef predator odour cut settlement 24-43 % at patch
  scale (E15) [F]. The only gadoid km-order chemical range is FOOD odour: cod reached
  baits from "several hundred metres" (700 m in a secondary citation), up-current,
  7 h after setting (E5, E7) [F/A]; cod and haddock approach baits upstream (E6) [F].
- Persistence: marine reef alarm cue inactive after 30 min in ocean water (E10) [F];
  freshwater alarm cue 3-6 h, other cues 1-72 h (E9) [L/F]. Advection dilutes cues
  below threshold "long before 2 h" even in a sheltered lake (E9).
- Thresholds: no gadoid-predator kairomone is chemically identified, so none exists;
  responses are graded with concentration and then plateau (E11, E14b) [A].

## 2. Plume physics at the model's scales

- Advection [J]: at 0.05-0.2 m/s (an assumed near-bed range, not checked for the box)
  a cue moves 1.1-4.3 km per 6 h tick, only downstream. A 30 min cue dies within
  90-360 m; a 3-6 h cue could reach 0.5-4 km.
- Spreading [A/J]: Okubo's surface diffusion gives patch variance ~5.15e-5 t^3 cm2
  (E19), sigma ~15 m after 1 h and ~230 m after 6 h. Plumes 1-4 km downstream are a
  few hundred m wide; cells upstream or 1 km across the flow receive nothing.
- Structure [L/F]: filaments separated by clean water, streaky near the bed, chaotic
  at large scales (E16-E18). The time-averaged concentration "converges far too
  slowly to be useful"; burst size and duration vary only mildly with distance (E16).
- Density [J, from E16-E17]: received concentration scales with source strength
  (biomass x unknown emission rate) / (current x plume cross-section); all three
  vary several-fold, so a sample cannot separate many-far from few-near or slow
  water. Density discrimination is shown only in well-mixed tanks (E11) [A].
- Km-scale chemical orientation is known for diffuse sources (salmon home waters)
  and for approaching food plumes up-current (sharks, cod) (E18, E5).

## 3. Area avoidance or local response; why prey and predators co-occur

- Measured marine responses are local and short: metre-scale feeder/patch avoidance
  (E2, E3, E15), graded by threat (E27, visual) [L/F]. No study found of a marine
  fish avoiding predator odour over km2, none for whiting; the one whiting study is
  a tank habitat choice under a (probably visual) predator stimulus (E4) [A].
- Chronic cues: constant exposure attenuates the response within 2-4 h, recovery
  30-60 min (E13, E14) [L]; risk allocation predicts LESS wariness where predators
  are always present, since prey must eat (E26) [J; tests mixed].
- Overlap: cod-capelin positive at > 4-10 km, negative at 2-3 km, in phase down to
  3.5 m when cod feed (E20) [F]; coherence episodic at any scale (E21) [J]. North Sea
  cod/whiting predator-prey overlap varies at 55 km scale with prey abundance and NAO
  (E22) [F]; Celtic Sea predators and prey largely independent at haul scale,
  whiting tracking its own prey (E23) [F]; predators aggregate on prey aggregations
  (E24) [A]; predators settle on the prey's resources, prey avoidance drives
  separation (E25) [A].
- Why they co-occur [J]: shared habitat and prey patches, predators following the
  prey's food, shoal dilution, risk allocation under ubiquitous risk. The box's GOV
  hauls show whiting and cod >= 20 cm together within depth classes (E29) [F].

## 4. Properties a smell channel could carry, if the evidence is followed

- Range: m to a few hundred m for predator cues [L/F]; to ~1 km only for strong
  persistent food plumes, approached up-current [F/A].
- Direction: downstream of the source only [F food plumes; J predators].
- Lag: reports where predators were 0.5-6 h ago, carried with the water [F/L].
- Noise: intermittent filaments; one sample is a poor mean [L/F].
- Density: presence/proximity yes; density confounded with distance and flow [A/J];
  responses saturate [A].
- Habituation / allocation: decline under constant exposure; chronic risk accepted
  [L/J]. Learning: a predator's odour is avoided after pairing with attacks [L].
- Hiding: odour does not vanish when the emitter hides; the model's channel does [J].

## Table

| Cue / species | Value | Setting | Grade | Src |
|---|---|---|---|---|
| Trout odour, juvenile cod | avoided, 32x40 cm arena | flume, marine | L | E1 |
| Cod odour, wild goby | avoided 1-2 d (with/without image) | 0.32 m2 tank | L | E2 |
| Cod odour, naive goby | no effect until 3 sessions; 6 mo after | tank | L | E3 |
| Minnow alarm cue | 2.3 m (lab), 2-8 m (field) | freshwater | L/F | E9 |
| Predator odour, reef settlers | -24 to -43 % settlement | patch, tropical | F | E15 |
| Bait odour, cod | several hundred m (700 m cited), up-current | fjord, tagged | F/A | E5, E7 |
| Alarm cue activity | 30 min / 3-6 h | ocean / freshwater | F / L | E10, E9 |
| Habituation, constant cue | 2-4 h; recovery 30-60 min | lamprey | L | E13 |
| Plume spread, 6 h | sigma ~230 m (surface) | Okubo diagrams | A/J | E19 |
| Cod-capelin correlation | - at 2-3 km, + at > 4-10 km | acoustic | F | E20 |

## Gaps

No predator-odour study in whiting, haddock, herring or sprat; no range or threshold
for any marine predator kairomone; no emission rate per tonne of any gadoid; no
near-bed plume data in the Skagerrak; the current speeds are an assumed range; no field
test of km-scale odour avoidance in a marine fish. E5, E7, E11, E24, E25 read as
abstracts/snippets only. Not reviewed (scope): Rindorf 2006, Hinz 2005, Langoy 2012.

## Verdict

Km-scale, density-resolving, direction-free predator smell has NO support. The
evidence supports cm-m predator-odour detection in gadoids, food-plume following
up-current over hundreds of m, cues lasting minutes to hours downstream only as
filaments, and local, learned, habituating responses; in the field gadoid predators
and prey co-occur at km scales. The model's channel (exact tonnes, four neighbours,
no flow, no noise) is a better sense than any measured fish sense and plausibly lets
whiting avoid predators real whiting cannot locate at that scale [J]. That the
channel drives the segregation is shown by the mask arm (266.2, 266.6); whether a
weaker channel would end it is untested.
