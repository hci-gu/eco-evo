# Observation pairs: what should observe what, in the own cell and in the neighbours (2026-11-13)

Research note. Nothing in code or library changed; no training. Sources and grades are in `fish.md` (fish
observers; section 7 corrects sections 1-4), `mammals.md` (porpoises, seals) and `code.md` (layout and the cost of
a split). [F] measured, [J] judgement.

## 1. The question

Every observation pair `<obs>_observes_<target>` gives the observer the target's density in its own 1 km cell
AND in the four neighbours (code.md). About 3/4 of every policy's inputs are neighbour channels. A neighbour
channel can mean:
- **A**: the animal senses the neighbour (0.5-1.5 km away) from where it is;
- **B**: the animal ranged through the neighbourhood during the 6 h tick and met what was there.

## 2. What the evidence says, in one paragraph

**Under A almost nothing survives.** No fish or mammal sense reaches 0.5 km for the model's targets: vision and
lateral line are metres, porpoise sonar 20-65 m for single fish and 140-360 m for schools, seal vibrissae tens of
m, porpoise clicks are inaudible to every model fish (the ~2 kHz by-product ~60 m, herring only), masked fish
sounds carry metres to ~20 m, seal roars ~50-140 m against the box's ship noise (15-20 dB over natural at
20-200 Hz). The one remaining cue is the downstream food-odour plume of bed prey (hundreds of m, directional; the
model's neighbours are isotropic). No study shows any model fish detecting a plankton patch, a school or a
predator from ~1 km. **Under B the matrix splits by how far the observer moves in 6 h**: porpoises (path 11-19 km)
and herring, mackerel, saithe (15-30 km) cross the neighbourhood every tick; seals do in ~70 % of diving ticks;
whiting and Norway pout swim 0.10-0.16 m/s on cyclic paths (leaning no); cod, flatfish, anglerfish, hake, ling are
sub-cell residents (no). Along a track, diffuse targets (plankton and krill layers, bed fauna) are sampled well;
point targets (schools, single mammals) only as encounters along a swath a few % of the neighbour area. The only
measured km-scale predator convergence within a 6 h tick (cod on capelin, Pednekar et al. 2024) works by chained
metre-scale senses, i.e. reading B with collective transmission, and needs densities 12-40x the box's cod mean.

## 3. The matrix

Same cell: "yes" = the observer feeds in it all tick; "partial" = encounters at metres integrated over 6 h, or
depth-conditional. Nbr A / Nbr B as above. "-" = no use (no food link either way).

| observer | target class | same cell | nbr A | nbr B |
|---|---|---|---|---|
| planktivores | zooplankton, krill (prey) | yes | no | yes (herring), leaning no (pout) |
| planktivores | benthivores, whiting, water column, mackerel (predators) | partial | no | partial |
| planktivores | porpoises, seals (predators) | partial | no | partial |
| mackerel | zooplankton, krill (prey) | yes | no | yes |
| mackerel | planktivores (prey) | yes | no | partial |
| mackerel | water column (predator) | partial | no | partial |
| mackerel | benthivores, whiting, porpoises, seals | - | - | - |
| whiting | zooplankton, krill (prey) | yes | no | leaning no |
| whiting | epifauna (prey) | yes | partial (downstream odour) | leaning no |
| whiting | planktivores (prey), benthivores (J prey, predator) | partial-yes | no | leaning no |
| whiting | water column, porpoises, seals (predators) | partial | no | leaning no |
| whiting | mackerel | - | - | - |
| benthivores | infauna, epifauna, Nephrops (prey) | yes | partial (downstream odour) | no |
| benthivores | krill (prey, beds < 150 m) | yes | no | no |
| benthivores | planktivores, whiting (prey) | partial-yes | no | no |
| benthivores | water column, porpoises, seals (predators) | partial | no | no |
| benthivores | mackerel | - | - | - |
| water column | zooplankton, krill, planktivores, benthivores, whiting (prey) | yes | no | partial-yes (saithe yes; hake, ling, anglerfish no) |
| water column | mackerel (prey, 0.02 vertical overlap) | partial | no | partial |
| water column | porpoises, seals (predators) | partial | no | partial |
| porpoises | planktivores (prey) | yes | no (schools 0.15-0.36 km, marginal) | yes |
| porpoises | benthivores, whiting (prey) | partial (beds < 150-200 m) | no | yes where reachable |
| porpoises | water column (prey, size window 0.054) | partial | no | yes |
| porpoises | mackerel, seals | - | - | - |
| seals | benthivores, whiting (prey) | partial (beds < 150-200 m) | no | partial (~70 %) |
| seals | planktivores (prey) | partial | no | partial |
| seals | water column (prey, size window 0.029) | partial | no | partial |
| seals | mackerel, porpoises | - | - | - |
| zooplankton | phytoplankton x2, microzooplankton | yes | inert | inert |

Zooplankton has movement_speed 0: its 16 neighbour inputs describe cells it can never choose (code.md); settled by
the code, not by a reading.

## 4. Pairs without a booked predation link (revised after the user's review)

Items 2-3 below have no predation pair in either direction, so no DIRECT payoff (nothing to eat, nothing to
flee). They keep an INDIRECT one: a competitor's density forecasts the depletion of shared prey (also visible in
the prey's own channel; interference acts within a group only), and any group can be a correlate of where other
things are. "No use" was too strong; they are not switch-off candidates on use alone. Under reading A they fall
with every other neighbour channel; under B the question is whether indirect information belongs in the
observation. Where a real predation link is suspected, it is the predation pair that needs the evidence:
- porpoises / seals -> mackerel were unchecked in section 228.3 (no mackerel in any Skagerrak / Kattegat harbour
  seal diet or porpoise stomach set; Sveegaard 2012: "rarely consumed", MAC-POR partial Mantel -0.127);
- mackerel -> juvenile gadoids (checked, `mackerel_0group.md`): cod, haddock and whiting are ~0.3 % of the
  mackerel group's diet by weight (0-1.5 %), June-July, almost all pelagic 0-group before settlement (Year of the
  Stomach 1991: 0.08-0.63 % in Q2, 0-0.12 % in Q3; ICES 2022 whiting 1.3 % from one 28 m haul); no SMS key run
  (2017/2020/2023) has mackerel on cod, haddock, whiting or saithe. The model's J are settled recruits imported
  at the bed (encounter a of mackerel on them 0, params_split_mack.md 4), so this predation sits in the recruit
  import, not in a pair. Norway pout is the one gadoid mackerel eat in quantity (1991: 3-16 % of the diet in Q3;
  SMS 2023 M2 0.082 /yr on age 0): it is in the planktivore group, whose mackerel pair counts herring and sprat
  only;
- porpoises <-> seals: harbour seals are not known to kill porpoises (grey seals do; not in the model); the two
  compete for herring and gadoids.

The list as first written (kept for the record):

1. **Nine self pairs** (`<x>_observes_<x>` for every DM): inert, `state.py:316-328` drops the own index and the
   own biomass is always in the centre and neighbours. Switching them off changes nothing (cosmetic; no test
   asserts them).
2. **Six fish pairs without a food link**: mackerel -> benthivores, whiting, porpoises, seals; benthivores ->
   mackerel; whiting -> mackerel. Mackerel and the bed groups do not share a layer either.
3. **Four mammal pairs**: porpoises -> mackerel (not eaten in Skagerrak/Kattegat stomachs, no co-distribution);
   seals -> mackerel (not taken); porpoises <-> seals both ways (no predation or avoidance documented between
   harbour seals and porpoises).
4. **Inert while seabirds are muted**: planktivores -> seabirds, seals -> seabirds.

A pair switch also removes a staged target's juvenile channel (code.md): mackerel -> benthivores and -> whiting
take those two J channels out of mackerel's input; no other off pair touches a staged group (planktivores,
benthivores, whiting). Items 2-3 change input widths: retrain.

## 5. The pair that started this: planktivores -> porpoises

Effect size: with the pair blinded, porpoise biomass at year 1 was 0.60 of start against 0.34 with it (frozen
fix100, section 272.4). Evidence: same cell **partial** (herring meet a porpoise at metres by vision and lateral
line; integrated over 6 h of a porpoise hunting in the cell this is an encounter record), neighbour A **no**
(clicks inaudible to herring, sprat, pout; Mann 2005, Kastelein 2007, Culik 2001), neighbour B **partial**
(single animals met along the track). Today the planktivores see porpoise density 1 km away in four directions,
which no sense supports under A and only weakly under B.

## 6. What it would take to implement

- Pair matrix only (section 4): no code, retrain.
- **Per-observer neighbour scope** (one key per DM, e.g. `observe_neighbours: false`, keeping the own biomass in
  the neighbours or not - a choice): enough for reading A (all observers off) and for the coarse reading B
  (benthivores off, whiting likely off, porpoises/planktivores/mackerel/water column/seals on). Code: state.py,
  observations.py, GPU obs_indices, train.py's width formula, the FG editor, tests (code.md list, with a per-DM
  instead of a per-pair index). Retrain.
- **Per-pair neighbour scope**: needed only for the fine reading B, where ranging fish keep neighbours for diffuse
  prey layers but drop them for point targets (predator schools, mammals). Same files, a second index list per
  DM. Retrain.

## 7. Open design questions that neither reading covers

1. **Memory**: porpoises return to an area after ~2 days, seals keep to the same areas month to month (Iorio-Merlo
   2022); information about a neighbour can come from earlier ticks. The policy is feed-forward.
2. **Collective transmission**: Pednekar 2024 (chained metre-scale senses through a dense group). Whether a cell's
   group biomass may stand for pooled experience is a modelling question.
3. **Own biomass in the neighbours**: a group's own density in the neighbour cells is a neighbour channel too; under
   A it is no better supported than the others.
4. **The km-scale blend premise**: research_notes/"Gom och sinnen" section 3 and four floor notes ("Planktivorernas
   golv", "Vitlingens golv", "Bottenfiskens och vattenpelarens golv", "Makrillens antipredatorbeteende") rest on
   smell and sound carrying the blend at km scale. For fish targets this does not hold (fish.md 1.4, 7.4): at 1 km
   the blend reduces to downstream food odour for bed feeders. Those notes' decisions concern what a hide cuts, not
   range; whether anything there changes is open.

## 8. Gaps

No audiogram for mackerel, saithe, whiting, hake, anglerfish, blue whiting, Norway pout; no field movement for
haddock, blue whiting, sprat, hake per 6 h; offshore saithe net displacement unmeasured (coastal saithe are
site-attached); no fish response to seal sounds; no test of Atlantic herring or sprat with porpoise clicks;
sonar ranges rest on one captive threshold and judged target strengths; seal dive depths offshore Skagerrak
unmeasured. Full texts not obtained: Swain 2015, Langoy 2012, Makris 2009, Chapman 1973, Hansen 2008.
