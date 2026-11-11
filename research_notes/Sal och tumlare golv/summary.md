# Seal and porpoise floors - summary (2026-11-11)

Research subagent; the main agent saved this text. Nothing in the code, library, manifest, tests,
AGENTS.md or the resume was changed. Sources E1-E27 in evidence.md; derived numbers D1-D6 from
`scripts/mammal_floors.py` (`python3 -I mammal_floors.py <repo>`, stdout `scripts/mammal_floors_stdout.txt`).
PDFs in `papers/` (Unpaywall helper; not committed). Grades: [F] field, [L] lab, [A] abstract only,
[S] secondary, [D] derived, [J] judgement. Out of scope (other notes): seals -> planktivores and
porpoises -> planktivores ("Planktivorernas golv"); recruit depths ("Rekryternas bottendjup").

## Question

A. The mammals' own species floors (seals 0.0, porpoises 0.0). Nobody eats a mammal, so these floors act
only on what other groups observe of a resting mammal: today a resting seal or porpoise vanishes from the
observation of the benthivores, whiting, water column, planktivores and mackerel, and of the other mammal.
What does rest correspond to for a harbour seal and a harbour porpoise in this offshore box (beds 44-450 m,
median 143 m; haul-outs 5-30 km outside the east edge)? What can fish sense of a resting seal or porpoise,
which floor does the evidence support, and should seal rest mean haul-out?

B. The mammals' pair floors on fish prey other than the planktivores:

| pair | today |
|---|---|
| seals -> benthivores | 0.85 |
| seals -> whiting | 0.85 |
| seals -> water column | 0.75 |
| porpoises -> benthivores | 0.75 |
| porpoises -> whiting | 0.75 |
| porpoises -> water column | 0.9 |

All six are judgements (resume 167.3, 149.3, 223.26). What can a fish's hide stand for against a seal
(vibrissae and vision, no sonar) and against a porpoise (biosonar)?

## The model (verified 2026-11-11)

- Visible share B (1 - rest (1 - floor)), used for predation this tick and observation next tick, one
  floor per (observer, hider): the pair's `visibility_floor` if `<obs>_preys_on_<hider>` is checked, else
  the hider's species floor (`interactions.py` ~95-162 reads the pair key only on checked pairs;
  `state.py` 327-336 builds `_obs_vis_floor` from the same matrix; `observations.py` 65-78). User decision
  2026-11-11 ("Gom och sinnen", option A): keep this coupling.
- Fish do not eat mammals, so every fish observer sees a mammal at the mammal's species floor. Seals and
  porpoises are each observed by the benthivores, whiting, water column, planktivores and mackerel;
  porpoises observe seals and seals porpoises, each at the other's species floor.
- A pair floor on a mammal -> fish pair sets how much of the resting fish the mammal can eat, what the
  mammal observes of it, and enters the shared availability cap (the total removal from a prey is bounded by
  what its best-detecting predator sees, `interactions.py` ~800).
- Size is not in the floor: the prey-size windows are `stage_access` (seals -> water column 0.029,
  porpoises -> water column 0.054; [1, 0.5] / [1, 0.63] on the benthivores). The floor is the refuge value of
  hiding only.
- Learned rest shares are small (fix97 annual means, "Gom och sinnen" section 4): seals 0.020, porpoises
  0.016, benthivores 0.026, whiting 0.066, water column 0.032.

## A. The mammals' species floors

### What rest is

Harbour seal (the seals group is harbour seals only):
- Haul-out is 10-30 % of time outside the moult (E1) [F/S]: about 17 % in the Norwegian Skagerrak in
  Aug-Nov, up to about 57 % at the August moult count. It cannot happen in a box cell (no land in the grid).
- At sea, seals rest about 14 % of the time (E2) [F]: median 206 min/day, at the surface and in slow, still
  dives to the bed; resting time rises by 37 min/day over a trip; resting at sea "is not equivalent to
  hauling out" (E2).
- Resting on the bed is 8-12 % of at-sea time in the North Sea (E3) [F], seen down to about 35 m, in dives
  of 4.0-5.3 min.
- Foraging is benthic and almost continuous (E2, E4) [F/A].
- No bed in the box is as shallow as 35 m (D1). A 4-5 min resting dive leaves time on the bed only where the
  bed is shallower than 54-181 m, depending on descent speed (D2) [D]. In the box at-sea rest is therefore
  mostly at or near the surface [D/J].

Harbour porpoise:
- In 20 tagged porpoises feeding took 62 % of diving time: 37 % pelagic, 25 % on the bottom (E6) [F]; the
  rest non-feeding (travel, nursing, sleep), in dives about 4 m deep.
- Candidate sleep: shallow, quiet "parabolic" dives, nearly half without clicks (E5) [A].
- The "continuous foraging" result of E7 comes from four juveniles and one adult just released from a pound
  net (E7b) [F].
- In the box porpoises reach the bed only where it is shallower than about 130-150 m (E23); a resting
  porpoise is in the top few metres.

### What fish can sense of a resting mammal

- Sound: neither a clicking nor a silent mammal gives fish an audible cue [L/A/D]. Porpoise clicks are at
  130-140 kHz (E11), outside the hearing of every model fish: cod showed no response to clicks at 204-213 dB
  (E8), their conditioned threshold is reached only within 1-3 m of a click at the maximum wild source level
  (E9, D4); Atlantic herring startle up to 4 kHz and not above (E10); Pacific herring hear no ultrasound up
  to 185 dB (E12). The narrow-band high-frequency click is thought to be crypsis against killer whales
  (E19); the ultrasound-hearing shad (E27) is not a model species. Harbour seals are nearly silent
  underwater outside the males' summer roars (E20) [A]. Fish reactions to mammal sounds are documented only
  for LOW-frequency FORAGING sounds of other species (herring to killer whale feeding sounds E17, toadfish
  to dolphin pops E18) [A]; low relevance here. E26 (seals hearing killer whales) has the seal as listener.
  Surface blows are in air; no study of fish detecting them [J, gap].
- Smell: a resting mammal keeps emitting odour. Predator odour reaches cm to m, lasts minutes to hours and
  travels downstream (E24) [S]. A cue released at the surface spreads only 1-21 m vertically in a tick (D3)
  [D], so it does not reach a 100-300 m bed.
- Hydrodynamic and visual cues: rest cuts these (no wake, no pursuit), but they reach only metres.
- No study shows a marine fish avoiding seals or porpoises at km scale; grey seal-cod telemetry finds only
  brief, rare encounters (E21) [A/S].

So the floor is set by WHERE the resting animal is relative to the observer, not by a sense that switches
off. Floor 0 (the resting mammal vanishes) has no sense behind it for any observer; floor 1 is right only for
an observer in the resting mammal's own layer.

### Derivation (D, J)

Factor per observer = cue of a resting mammal that reaches that observer / cue of the same mammal foraging.
- Bed observers (benthivores, whiting): seal - the bed share of at-sea rest (0.57-0.87, E3 over E2) x the
  share of the seals' cells where a resting dive reaches the bed (D2): central 0.24 (0.02-0.76) [D].
  Porpoise - never at the bed when resting, rarely when foraging in this box: 0.05 (0-0.1) [J].
- Pelagic observers (planktivores, water column, mackerel): 0.75 (0.5-1) [J]. A resting mammal shares their
  layer at night but is 50-150 m above herring and Norway pout by day, and a surface cue does not cross that
  gap in a tick (D3). A foraging mammal does meet them at depth (porpoises reach the herring's daytime depth;
  seals take pelagic prey where it is - in E15 schooling prey were attacked near the bottom; Norway pout,
  0.35 of the planktivores, lies near the bed by day). If a foraging seal were counted as never meeting
  pelagic fish, its pelagic factor would be 1 and the seal floor 0.46 (flow weights), inside the range below.
- Weights: the booked flows M2 x B of the mammal on each observer, or the observers' biomass:

| mammal | flow weights | biomass weights |
|---|---|---|
| seals | 0.38 (0.16-0.83) | 0.63 (0.38-0.94) |
| porpoises | 0.54 (0.35-0.73) | 0.58 (0.38-0.78) |

### Haul-out

Rest in a cell should not stand for haul-out [F: E1, E2]: haul-out is absence from the box (5-30 km of
travel plus ~5-6 h ashore). Structurally that would be a seasonal at-sea fraction, an outside exchange like
`open_pool.migrate` on a DM, not a floor and not the Rest action; the share ashore is ~0.15-0.3 most of the
year, ~0.55-0.6 at the August moult count. Today the start density already assumes 75-90 % of the seals at
sea, and the seasonal absence is missing (E1). A flagged option, not a recommendation to build now: the
seals' M2 on fish is booked as annual flows, and a summer absence would move that flow in time.

## B. Pair floors on fish prey

What hiding can mean against the mammals [J from F/L/D]:
- Burying (flatfish). Against seals no refuge [F/L/S]: seals took buried flounders at the highest capture
  success of any prey, 79.9 %, and in 14 of 15 cases the flounder stayed hidden until the strike (E15);
  buried sand lance 63.7 % vs 37.5 % for schooling sand lance (E15); vibrissae find flatfish breathing
  currents (E16) [L]; flatfish are 46 % of a Skagerrak seal diet's energy (E22) [S]. Against porpoises a
  partial refuge [A/D/F]: a buried flatfish lies inside the sonar's bed dead zone (D5: a few cm close in,
  ~1-2 dm at 5 m) and has no swimbladder (90-95 % of a gadoid's echo, E13); but porpoises do excavate
  sandeels (E14) [F/D].
- Lying still on the bed (cod, haddock, whiting, hake): seals find such fish with their vibrissae, without
  light dependence (E16); the box's 100-450 m beds are soft, without kelp or crevices (E22). For porpoises a
  swimbladdered fish on the bed stands above the dead zone in the final approach (D5); porpoises feed on the
  bottom (E6) and aim clicks at the bed (E23).
- Schooling (juvenile saithe, whiting in part): against seals the one measured refuge (E15); against
  porpoises schools are strong sonar targets (consistent with porpoises -> planktivores 0.95).
- Darkness is not what a hide buys against these predators: neither mammal has a light term (seals none,
  resume 167.2; porpoises sonar); at 100-450 m it is dark anyway.

Member weights inside the prey-size windows (E25, D6): benthivores in the seal window 0.57 gadoids / 0.43
flatfish, porpoise window 0.64 / 0.36; water column, accessible part: saithe 0.70 / hake 0.30 for seals,
0.73 / 0.27 for porpoises.

| pair | member floors (central, range) [J] | weighted |
|---|---|---|
| seals -> benthivores | gadoids 0.85 (0.75-0.95), flatfish 0.95 (0.85-1.0) | 0.89 (0.79-0.97) |
| seals -> whiting | 0.85 (0.6-0.95): bed contact vs schooling unresolved | 0.85 |
| seals -> water column | saithe 0.6 (0.35-0.75, schooling), hake 0.85 (0.75-0.95) | 0.68 (0.47-0.81) |
| porpoises -> benthivores | gadoids 0.9 (0.75-1.0), flatfish 0.5 (0.3-0.8) | 0.75 (0.59-0.93) |
| porpoises -> whiting | 0.9 (0.75-1.0) | 0.9 |
| porpoises -> water column | saithe 0.95 (0.85-1.0), hake 0.9 (0.75-1.0) | 0.94 (0.82-1.0) |

## Proposed floors

| floor | today | proposed | range | grade | what it changes | who is affected |
|---|---|---|---|---|---|---|
| seals (species) | 0.0 | 0.4 | 0.15-0.85 | D/J (F pieces: E2, E3, D1-D3) | observation only: a resting seal stays 0.4 visible | benthivores, whiting, water column, planktivores, mackerel, porpoises as observers of seals |
| porpoises (species) | 0.0 | 0.55 | 0.35-0.8 | D/J (F pieces: E5, E6, E8-E12) | observation only | the same five fish groups and seals as observers of porpoises |
| seals -> benthivores | 0.85 | 0.9 | 0.8-0.95 | F (E15) / L (E16) / J | seal predation on resting benthivores, seals' observation of them, the availability cap | seals |
| seals -> whiting | 0.85 | 0.85 (keep) | 0.6-0.95 | J | - | - |
| seals -> water column | 0.75 | 0.7 (optional) | 0.5-0.8 | F (E15, schooling) / J | as above, on the 0.029 accessible part | seals |
| porpoises -> benthivores | 0.75 | 0.75 (keep) | 0.6-0.9 | D (D5, D6) / J | - | - |
| porpoises -> whiting | 0.75 | 0.9 | 0.75-1.0 | D (D5, E13) / J | porpoise predation on resting whiting, porpoises' observation of whiting, the whiting availability cap | porpoises |
| porpoises -> water column | 0.9 | 0.9 (keep) | 0.8-1.0 | J | - | - |

Size at today's rest shares (D, script part 8): a resting seal as observed 0.980 -> 0.988 of B; a resting
porpoise 0.984 -> 0.993; porpoises -> whiting 0.984 -> 0.993; seals -> benthivores 0.996 -> 0.997. Every
change is below 1 % in annual means (more in seasons or ticks in which a group hides more). No growth-budget
or diet-gate number moves (floors are not in s* or budget_gate). The species floors change the meaning of an
observation input and the pair floors predation and the mammals' observation of fish: retrain (shapes
unchanged). Both engines read the same keys; the GPU mirrors the floor matrix (resume 77.3).

## Gaps

- The bed share of harbour seal rest over 100-300 m beds is unmeasured, as is where seals rest offshore in
  the Skagerrak; the resulting bed factor (0.02-0.76) is the widest input.
- No study of any fish sensing a seal or porpoise that is not foraging; the smell reading (E24) is the
  project's own and untested on mammals.
- E5, E7-E13, E17-E21, E26-E27 read as abstracts or secondary sources only; the 18 % time share for Wright
  2017 is a press figure.
- No measured capture success of seals or porpoises on resting vs active gadoids; Bowen 2002 covers NW
  Atlantic sand lance and flounders at 30-37 m.
- The model does not define whether whiting's hide is bed contact or a tight school; the seal floor on
  whiting depends on it (0.6 vs 0.95).
- Weighting a species floor over observers is a choice (flows 0.38 / 0.54, biomass 0.63 / 0.58); one species
  floor cannot be right for both bed and pelagic observers.

## Verdict

A. Floor 0 is the one value the evidence rules out: no resting seal or porpoise in the water vanishes from
every sense. What a rest really cuts is the vertical overlap: a seal resting at the surface is out of reach
of the bed fish it eats, while a pelagic fish shares its layer at night; porpoise clicks are inaudible to all
model fish whether the animal rests or forages. Proposed: seals 0.4 (0.15-0.85), porpoises 0.55 (0.35-0.8)
[D/J]. The floor really depends on the observer (~0.05-0.25 for bed observers, ~0.75 for pelagic ones); one
species floor cannot express that - under option A a tension, not a proposal.

Haul-out: seal rest should not mean haul-out; haul-out is absence from the box. If modelled, a seasonal
at-sea fraction (an outside exchange), not the Rest action or a floor [F/J].

B. The evidence supports today's values within their ranges, with two exceptions: seals -> benthivores 0.85
-> 0.9 (buried flatfish are no refuge from seals, E15 [F]); porpoises -> whiting 0.75 -> 0.9 (whiting neither
buries nor leaves the sonar's reach on the bed, D5 [D], J). Seals -> water column 0.75 -> 0.7 is optional
(schooling is the one refuge measured against seals). The other floors stay. All changes are small at
today's rest shares; the decisions are the user's.

## User decisions

- 2026-11-11: all into the combined section: species floors seals 0.4 and
  porpoises 0.55; seals -> benthivores 0.9; porpoises -> whiting 0.9; seals ->
  water column 0.7; porpoises -> planktivores 1.0 (research_notes/"Planktivorernas
  golv"); seals -> planktivores keep the species floor 0.35. Haul-out stays a
  flagged option.
