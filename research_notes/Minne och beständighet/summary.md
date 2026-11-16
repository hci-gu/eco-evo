# Memory and persistence in the neighbour weight (reading C) (2026-11-14)

Research note. Nothing in the code, library, manifest or resume changed; no training. Sources and grades
are in `memory_mammals.md`, `memory_fish.md` and `persistence.md` (research subagents). Scripts are in
`scripts/`, outputs in `tables/`. Grades: [F] measured, [S] secondary, [D] derived here, [J] judgement.
The user's email address appears in no file (checked with grep).

## 1. Question and answer in short

Section 273 set the neighbour weight w of each observation pair under reading B: w = P(the 6 h track
enters a given neighbour) x P(it meets the target there). That reading has no memory, so no w exceeds
~0.55. Reading C asks what the observer still knows of a neighbour from earlier visits within its memory
span, discounted by how much of the target's distribution there has persisted since.

**Answer:**
- Derived consistently, reading C raises the pairs on STATIONARY targets: bed fauna, the krill layer,
  resident demersal fish. It leaves the pairs on MOBILE pelagic targets about where 273 put them.
- Porpoises -> planktivores goes 0.24 -> 0.22 (0.22-0.28 over the variants).
- Planktivores -> porpoises goes 0.10 -> 0.07.
- **The porpoise problem of fix101 is therefore not carried by w.** Herring are a hard target to know at
  1 km, whatever the porpoise remembers.
- The new result that does bear on it is the OWN cell (273 open point 1.6). Under the same reading, a
  planktivore knows the porpoises in its own cell with weight 0.38-0.71, not the engine's 1 (section 4).

## 2. Method [D]

`scripts/memory_weight.py`: a correlated random walk with each observer's measured path L and net
displacement D per 6 h. These are the reading-B inputs, imported from
`../Observabilitetsparen/scripts/neighbour_weight.py`, so B and C differ only in memory and persistence.
- The walk runs M ticks before the decision; the decision cell is where it ends.
- For each neighbour of that cell, the visits within M are listed, newest first, with their age and the
  path inside each.
- w_C = E[ sum_i rho(age_i) p_i prod_{j<i} (1 - p_j) ], with p_i = 1 - exp(-n_i) (reading B's encounters)
  and rho the target's persistence. The observer knows what its most recent successful visit showed.
- Check (`--check`): M = 1 and rho = 1 reproduce reading B within Monte Carlo noise (porpoises ->
  planktivores 0.241 vs 0.239, diffuse 0.376 vs 0.373, planktivores -> porpoises 0.107 vs 0.104).
- Implied 24 h net displacement of the porpoise walk: 10.6 km, against ~11 km measured (Stalder 2020).

Inputs, central values (`scripts/library_table.py`):

| | value | basis, grade |
|---|---|---|
| memory for food / no link, ticks | porpoises 8, seals 28, planktivores and mackerel 8, gadoids 28 | DEPONS 36-60 h, AgentSeal ~7 d; stickleback 1-21 d and cod n = 1 (8 d); planktivores a proxy. All [S]/[J]: no observer has a MEASURED span |
| memory for a predator's place, ticks | 1 (benthivores 4) | gobies avoid a cod's site 1 d; freshwater 2-4 h [F proxy, J] |
| persistence rho(t) = a e^(-t/T1) + (1-a) e^(-t/T2) | fits to the "raw rho in a 1 km cell" rows of persistence.md | [F]+[J]; table below |

Two choices the script makes [J]:
- A mutual pair (whiting <-> benthivores, and every fish -> a fish that eats it) takes the PREDATOR memory
  (1 tick, benthivores 4), because the target can eat the observer. Under the food memory (28),
  whiting -> benthivores (0.31 -> 0.30 here) and benthivores -> whiting (0.19 -> 0.26) would come out
  higher.
- A self pair (own biomass in the neighbours) takes the group's OWN persistence with its food memory.
  273 used the observer's diffuse P. That is why mackerel -> mackerel falls 0.42 -> 0.36 and
  planktivores -> planktivores 0.38 -> 0.35.

Persistence fits; rho at 6 h / 1 d / 3 d / 1 wk / 1 mo:
- planktivores 0.38 / 0.34 / 0.31 / 0.26 / 0.10 (P_hot variant 0.52 / 0.47 / 0.43 / 0.34 / 0.10)
- mackerel 0.30 / 0.07 / 0 / 0 / 0
- water column 0.45 / 0.29 / 0.27 / 0.23 / 0.09
- whiting 0.77 / 0.61 / 0.53 / 0.40 / 0.08
- benthivores 0.89 / 0.75 / 0.67 / 0.57 / 0.23
- krill 0.90 / 0.76 / 0.69 / 0.62 / 0.34
- zooplankton 0.68 / 0.31 / 0.20 / 0.19 / 0.15
- porpoises 0.31 / 0.19 / 0.18 / 0.16 / 0.07 (upper variant 0.39 / 0.29 / ...)
- seals 0.35 / 0.24 / 0.23 / 0.21 / 0.12
- bed fauna 1 throughout

## 3. Results (`tables/library_reading_c.txt`, `tables/key_pairs.txt`)

Neighbour weights, B (273, = library today) -> C:

| pair | B | C | note |
|---|---|---|---|
| benthivores -> infauna, epifauna, Nephrops | 0.19 | 0.54 | bed fauna persists, cod returns to sites |
| whiting -> epifauna | 0.31 | 0.67 | |
| whiting, water column, planktivores -> krill | 0.31-0.48 | 0.53-0.60 | krill layer over the beds persists |
| seals -> benthivores, whiting | 0.30 | 0.42-0.46 | |
| porpoises -> benthivores, whiting | 0.37 | 0.43-0.45 | |
| **porpoises -> planktivores** | **0.24** | **0.22** (0.22-0.28) | herring move 4 km / 6 h; ~1 school per cell |
| **planktivores -> porpoises** | **0.10** | **0.07** | predator-place memory <= 1 tick |
| mackerel -> planktivores | 0.22 | 0.19 | |
| planktivores, whiting, mackerel -> water column | 0.31-0.40 | 0.23-0.34 | saithe move 3 km / 6 h |
| fish -> porpoises, seals, mackerel | <= 0.07 | <= 0.05 | rare point targets |

Separating the two changes, for porpoises -> planktivores (M 8):
- B 0.239;
- memory alone, with rho = 1 inside the current tick, 0.265 (P_hot 0.276);
- memory plus decay inside the tick 0.220 (0.241).
Memory adds ~0.03, and the within-tick decay takes about as much back. Neither change dominates. The
porpoise walk seldom returns to the neighbours of the cell it ends in: it moves ~11 km a day.

The independent-tick shortcut 1 - (1 - P)^M is wrong for ranging observers. An earlier chat estimate used
it (0.85 at 4 ticks), and so does the table in `memory_fish.md` section 4. Under a walk
calibrated to the measured displacement, the ticks are not independent, and porpoises -> diffuse reaches
only 0.45 at 4 ticks and 0.49 at 8 (`tables/grid_memory_persistence.txt`).

## 4. Own cell (273 open point 1.6) [D]

The engine shows every observed channel in the own cell at full strength (w = 1). Under readings B and C a
point target in the own cell is known only if it was met.

| pair | own, B (box-mean density) | own, C | own, C at x6 local density |
|---|---|---|---|
| planktivores -> porpoises | 0.42 | 0.38 | 0.71 (about 3 porpoises in the cell) |
| porpoises -> planktivores | 0.61 | 0.61 | 0.83 |
| whiting, benthivores, seals -> planktivores | 0.51-0.54 | 0.52-0.56 | 0.61-0.68 |
| any observer -> diffuse targets | 1 | 1 | 1 |

This is the only derived item here that touches the fix101 mechanism: the planktivores flee the porpoises
because they see them in their own cell at full strength. Derived, that weight is 0.4-0.7. The expected
effect is MODEST and untested:
- where the porpoises actually sit (about 3 animals per cell) the derived weight is 0.71, so the cue is
  mostly legitimate at the densities that matter;
- the blend preserves the information (the policy can invert it), so it dampens the cue rather than
  removing it;
- a planktivore cell that is being eaten sees its own biomass fall the next tick anyway;
- the blind test (0.88 vs 0.31 after one year) removed the cue entirely, own cell and neighbours. A weight
  of 0.7 is far from that. Using it
requires an engine change:
- `observations.py` and the GPU `observe` have no own-cell weight;
- design question: what does the own cell blend towards? The own cell has no "c" to blend to, so the
  candidates are the box mean, the previous tick, or 0;
- every checkpoint becomes invalid.
It also cuts the porpoises' own-cell view of planktivores (0.61-0.83), so it is not one-sided.

## 5. Limits

- **No memory span is measured.** The mammal spans come from models calibrated to track patterns (DEPONS,
  AgentSeal); the fish spans are lab proxies. Read them as upper bounds.
- **Possible double counting.** The porpoise "returns after ~2 days" is unpublished and DEPONS uses it as
  food recovery, i.e. persistence. A memory span tuned to habitat stability may also count persistence
  twice.
- **No revisits beyond the random walk.** Site fidelity (cod home sites, seal foraging grounds, DEPONS's
  memory-driven returns) is not modelled, so w_C is a LOWER bound for site-faithful observers. A home-range
  tether was tried and dropped: it broke the 6 h displacement calibration (2.9 km vs 5.5).
- **Diel cycle.** Pelagic rho can be lower at 6 h than at 24 h (day schools vs night layer). The
  exponential form cannot carry that.
- **Field vs model.** rho is field persistence, including the shot noise of ~1 school per cell. The model's
  planktivore field is a smooth biomass, and its zooplankton does not drift. w is a prior on the animal's
  knowledge, applied to a smoother field.
- **Persistence fits.** These are fitted by eye to rows that are mostly [D]/[J] (persistence.md: no fixed-site
  rho at 1 km for any pelagic fish in the North Sea or Skagerrak).

## 6. What this means for fix101 and the options

- Raising the porpoise pairs to 1 has no support: derived, they stay at 0.22-0.45.
- Reading C for ALL pairs is a consistent rebalancing. It strengthens the bed and krill feeders' view of
  their stationary food and leaves pelagic and predator pairs about where they are.
- It is a CANDIDATE for the user's decision, not a settled value: its inputs are [S]/[J] throughout (no
  measured memory span, persistence fitted by eye, no revisits). Like 273 it changes what the channels
  mean, so every checkpoint would become invalid. It does not solve the porpoises.
- The fix101 tests (frozen u1080 policies) found:
  - with neighbour information at full strength, a greedy follower finds planktivore cells 15-45x the
    mean;
  - the trained policy does not move towards them;
  - the planktivores flee via the own-cell channel.
- The levers left (none tested):
  - (a) an own-cell weight, derived 0.4-0.7 for planktivores -> porpoises; a modest expected effect
    (section 4); engine change, design, retrain;
  - (b) the unit merge, which erases sub-animal moves and flattens the ARS gradient (272 option B);
  - (c) the move time tau (273 design 2.2).

## 7. Levers (b) and (c) measured on the frozen fix101 policies (2026-11-15)

`scripts/porp_merge_arms.py` (the session's porp_coloc probe: 30x30 probe world, frozen u1080 policies,
3 years, seed 20260530). Arms on the porpoises only (other groups keep the library merge):
- `merge_base` is the library rule;
- `merge_continuous` sets `unit_splits` off;
- `merge_stochastic` rounds each move direction to whole animals with a Bernoulli on the remainder. This
  arm makes sub-unit cells (column `sub`: 35-93, against 12-39 in base), so it is not a clean option B;
- `merge_whole_animals` draws floor(n) animals multinomially over the five fractions, and the sub-unit
  remainder rides with stay. It has the same expectation as the continuous split, and `sub` stays at base
  level (10-47). This is the clean form of 272 option B.

Porpoise biomass at the end of years 1 / 2 / 3, and at the first June:

| arm | year ends | June 1 | occupied cells |
|---|---|---|---|
| base | 0.31 / 0.15 / 0.11 | 0.69 | 29-123 |
| continuous | 0.34 / 0.13 / 0.06 | 0.88 | 33-244 |
| stochastic | 0.29 / 0.12 / 0.07 | 0.78 | 25-189 |
| whole animals | 0.32 / 0.12 / 0.07 | 0.81 | 25-171 |

Honouring the policy's small moves does not help the frozen porpoises:
- the first summer is better;
- the years end the same or lower;
- the population spreads over more cells in year 1 (peaks 171-244 against 123).

The moves are not aimed (0.17-0.35 of the move mass goes to the best neighbour, against 0.25 random;
`move_base.txt` of the session), so honouring them spreads the animals, not toward food. The same held in
fix99: continuous fractions left 0.05 (272).

What this does not test: whether ARS LEARNS aimed moves once a small move has a payoff that changes
smoothly with its size. That is the remaining case for option B. It needs a training run, and a per-world
seeded draw shared by the +/- candidates (CRN).

Tau (c) is not tested here: it needs an engine change. By the arithmetic on base's budget, today a moved
porpoise tonne gives up ~81 MJ/t per tick (the 93 MJ/t an eating tonne gains, less the 0.2 RM cost
difference), about 0.9 of a tick's cost and ~10 % of the 2-day buffer. Under tau 0.071 it gives up ~7 MJ/t.
- The design's gate (4.5: build tau only if the retrained porpoises "still move heavily and starve at the
  start") is NOT met: fix101 porpoises move 0.4-3 %, and the loss is planktivore avoidance.
- Tau is one rule for every DM: the planktivores (tau 0.20, move share 0.24-0.44 in training) would forage
  through 80 % of their moving ticks too, which also makes fleeing the porpoises cheaper.
- `movement_cost` must be re-read for the tau share (design 2.2), not rescaled.
