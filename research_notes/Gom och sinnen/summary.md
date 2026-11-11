# Hiding and the senses - summary

Question (user, 2026-11-11): the hide action (rest) also lowers how much of the
hider the other groups OBSERVE. Is that coupling right, for every group? The
observation is a diffuse blend of all senses - smell, sound, the lateral line,
sight - not smell alone (user, 2026-11-11). Research and code reading only;
nothing in the code or library changed.

## 1. The mechanism (verified 2026-11-11)

One share per DM, cell and tick, `rest` (action 4), is both rest and hide:

- Predation, this tick: visible prey = B (1 - rest_now (1 - floor)), floor = the
  pair's `visibility_floor` if the pair carries one, else the prey's species floor
  (`interactions.py:785-800`, `predation.py:235-262`; GPU `ecosystem.py:917-943`).
- Observation, next tick: every observer sees B (1 - rest_prev (1 - floor)) of
  the hider in its own cell and the four neighbours, with the SAME floor matrix
  (`state.py:330-336`: "what I can see and what I can eat stay consistent per
  observer"; `observations.py:39-78`, `decisions.update_hidden_state`; GPU
  `ecosystem.py:753-796`). For an observer that does not eat the hider the
  hider's species floor applies. The DM's own biomass is never attenuated.
- Origin: section 41 (perfect hide, floor 0) and section 51.2, where the floor
  was put on the observation so predators "CAN learn to track/stalk hiding prey
  instead of treating them as truly absent". The reading of the
  observation as a loose blend of the senses came later (user, 2026-11-10 and
  2026-11-11; resume 266.9).

## 2. Who sees whom through a hide (mareld2.yaml, seabirds muted)

46 observer -> hider pairs between active DMs (zooplankton observes no DM).

| Role of the hider for the observer | Pairs | Floors |
|---|---|---|
| prey (observer eats it) | 21 | planktivores 0.25 (fish), 0.35 seals, 0.95 porpoises; zooplankton 0.2 / 0.5; benthivores 0.55-0.85; whiting 0.5-0.85; water column 0.7-0.9; mackerel 0.7 |
| predator (hider eats observer), incl. 2 both ways | 17 | seals and porpoises 0.0 (vanish when resting), water column 0.7, mackerel 0.7, whiting 0.5, benthivores 0.55 |
| neither (no food link) | 8 | mackerel 0.7 to benthivores, whiting, porpoises, seals; mammals 0.0 to each other and to mackerel |

## 3. What each role means under a blend of senses

Each sense is cut by hiding to its own degree. Smell persists through a hide:
predator odour is detected at cm-m in gadoids and cues last minutes to hours,
travelling downstream as filaments; "odour does not vanish when the emitter
hides" [J] (research_notes/"Lukt och rovdjursundvikande", E1-E29). Sight and the
lateral line are cut by a hide: a school that goes deep, below the light or
tight against the bed is harder to see or feel. Sound is mostly not cut: a school
or a porpoise is heard at hundreds of m, and porpoise biosonar reaches about
100 m [J, not reviewed here]. At the observation's km scale, sight and the
lateral line hardly reach at all; there the channel is mostly smell and sound,
which a hide barely changes. So a partial cut on the observation can be
defended; a cut to zero cannot.

- Prey role. The observation reports the EATABLE part of the prey, lagged one
  tick. The part a hide cuts by sight and touch is defensible; the exact share is
  not, because it is the prey's own last decision copied at the same floor as the
  predation. Odour and sound of a herring school that goes deep in winter, or of
  zooplankton that migrate below the light, stay in the water. Whether they can be
  caught is a matter of light and depth, which the predator also observes (the
  light channel, the season). The lag is measurable: after a dim tick a herring
  cell shows 0.50 of B while 0.31 is hidden now (resume 239.3, "Observation
  lag"); "real but not the main cause" of the diet gap.
- Predator role. A resting predator looks smaller or, for seals and porpoises
  (floor 0), absent. Rest is also "not foraging", so the prey reads its predators'
  activity state from last tick. A resting cod still smells; a resting porpoise
  still breathes at the surface and moves; neither vanishes from every sense.
  The one case where absence is real is a seal hauled out on land, but the
  model's resting seal stays in its water cell. Floor 0 on the observation has
  no sense behind it.
- Neither. Pure attenuation of an unrelated group's density (mackerel to the
  benthivores, mammals to each other); no ecological meaning.

## 4. How much it matters today

Mean action shares over the training worlds at fix97's last update
(results/pzbpgp248nsrpfix97/training.jsonl, `actions` = entropy, move, rest, eat;
order benthivores, mackerel, planktivores, porpoises, water column, seals,
whiting, zooplankton):

| DM | rest share | max observation cut (1 - floor) x rest |
|---|---|---|
| zooplankton | 0.41 | 0.33 (fish floor 0.2) / 0.21 (0.5) |
| whiting | 0.066 | 0.03 |
| planktivores | 0.043 | 0.03 |
| water column | 0.032 | 0.01 |
| mackerel | 0.028 | 0.01 |
| benthivores | 0.026 | 0.01 |
| seals | 0.020 | 0.02 |
| porpoises | 0.016 | 0.02 |

These are annual means over training worlds; hiding is seasonal and diel (herring
visible 0.28-0.50 in winter in fix78 and fix80, resume 223.27, 223.31), so the cut is larger
in some months. Zooplankton's diel hide is the only large one today.

## 5. Options (decision is the user's)

A. Keep the coupling (status quo): observation = eatable part, lagged.
B. Decouple: observation always the whole biomass (smell and sound stay);
   hiding acts on predation only. Predators must learn catchability from light, depth and
   season, as wild ones do. Small change in both engines (the observation's
   visible fraction set to 1); action and observation shapes unchanged, but the
   input means something new, so retrain.
C. Decouple the predator and neither roles only (prey see their predators'
   whole biomass; a predator still sees the eatable part of its prey): removes
   the leaked activity state, keeps 51.2's stalking signal. Mixed semantics.

D. Use the existing pair floors (`visibility_floor` on a pair, section 51 /
   `interactions.build_interaction_matrices`), with two limits lifted: (1) today the pair key
   is read only on a CHECKED predation pair (`<obs>_preys_on_<hider>`, preys_on
   true, prey on the menu), so the 25 predator-role and no-link pairs always
   fall back to the hider's species floor (0 for seals and porpoises); (2) one
   number sets both what the predator can eat and what it observes. Lifting (1)
   means reading a floor on the observation pair (`<obs>_observes_<hider>`) as
   well; lifting (2) means an observation floor that may differ from the
   predation floor, set from what the hide cuts across the senses at km scale,
   never 0 for an animal that stays in the water.

Grade: the mechanism and numbers are verified [F, code]; how much each sense is
cut by a hide is [J] (smell from the smell note's evidence; sound and sight not
reviewed).

## User decisions

- 2026-11-11: option A, keep the coupling (one floor for predation and
  observation; pair floors on checked predation pairs only). Nothing changed.
