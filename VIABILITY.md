# Long-term viability: the acceptance criterion and the rig

Before anything is asked about behaviour, reward shaping or impacts, one
question has to have a signed-off answer: **is the configured world
self-sustaining at all?** This file defines that question as a number and
describes the rig that measures it.

Nothing here involves ARS. Viability is a property of the parameters and
the mechanisms in the tick, not of the policy networks, so it must be
measurable with the behaviour frozen. If the world fails this test, no
amount of training can fix it, and every reward comparison taken in it is
uninterpretable - see `mareld_resume.txt` sections 73, 74 and 87.4.

## 1. The criterion

Let `B_f(t)` be functional group `f`'s total biomass after tick `t`, `T`
the horizon in ticks, and `W` the final window (the last `window_frac` of
the horizon). Write

    eq_f   = mean over W of B_f(t)              the *measured* equilibrium
    prev_f = mean over the window before W
    ref_f  = B_f(0)                             the spawn level

A configuration is **viable** when, for every FG present at spawn
(`ref_f > 0`) and for every seed:

| # | Name | Condition | Default |
|---|---|---|---|
| 1 | survival | `B_f(t) > 0` for all `t <= T` | - |
| 2 | floor | `eq_f >= floor * ref_f` | `floor = 0.10` |
| 3 | ceiling | `eq_f <= ceiling * ref_f` | `ceiling = 10.0` |
| 4 | stationarity | `max(eq_f/prev_f, prev_f/eq_f) <= max_drift` | `max_drift = 2.0` |

with the horizon and the sampling

| Parameter | Default | Meaning |
|---|---|---|
| `years` | `5.0` | horizon, `T = years * ticks_per_year(tick_hours)`; 1460 ticks at the default 6 h tick |
| `tick_hours` | `6` | hours per tick, read from `project_metadata.tick_hours` (section 97) |
| `seeds` | `3` | independent spawn layouts; the worst seed decides |
| `window_frac` | `0.10` | share of the horizon used as the final window |

Groups absent at spawn are outside the criterion: the project did not
seed them, so they cannot fail it.

### Why these four

* **Survival** is the only non-negotiable one. Zero is absorbing for
  every multiplicative term in the tick, so an extinction inside the
  horizon is permanent.
* **Floor and ceiling are deliberately wide.** The equilibrium is an
  *output* of the parameters, not an input, so a group is not required to
  hold its spawn level - only to avoid decaying towards zero or running
  away. A run that passes but reports `eq/spawn = 0.23` is telling you
  the spawn is miscalibrated, not that the world is dying; that is
  exactly the number needed to recalibrate `mareld2.yaml`.
* **Stationarity is what makes the claim "long-term".** A group can be
  well above the floor at `T` and still be on a monotone slide; comparing
  the last two windows catches that without needing a longer run.

### What the criterion deliberately does not say

* It does not settle **open vs closed system**. A 1 km-cell box is in
  reality open - recruitment and immigration come from outside - and
  `--migration on` models that. The rig therefore reports the verdict
  *per setting* rather than picking one. Same for `--mortality` and
  `--mortality_multiplier`.
* It says nothing about *spatial* behaviour. `occ` (occupied cells) is
  printed because a group surviving in a single cell is qualitatively
  different from one spread over the grid, but it is not a pass/fail
  condition: with no environment maps in `mareld2.yaml` the grid is
  isotropic and occupancy is not yet interpretable (section 87.5).

## 2. Who may pronounce the verdict

The criterion above is evaluated per *cell of a two-factorial design*,
not once. Two things decide the outcome besides the parameters:

* **behaviour** - what the groups do, frozen for the whole rollout, and
* **spawn geometry** - whether predator and prey even start in the same
  cells.

Both must be pinned before a FAIL means anything, and neither may be
smuggled into the other.

> **A uniform or untrained policy dying is not a result.** `neutral` and
> `random` are expected to fail; they are contrast, not evidence. The
> rig refuses to turn them into a verdict - a run of those arms alone
> reports `NO VERDICT` and exits `0`.

> **A stationary arm on an independently drawn spawn is not a result
> either.** In `mareld2.yaml`, `gadoids` and `pelagic_fish` are both
> drawn as 25 independent colonies, so the measured share of cod biomass
> that starts in a cell holding any herring is 0.24-0.44 on 60x60 -
> exactly the product of the marginals, i.e. pure chance. Under `eat`
> nothing ever moves, so two thirds of the cod can never reach food:
> that FAIL is a spawn artefact, not a statement about parameters.

**The verdict is taken from the normative corner: co-located spawn +
`greedy`.** The most favourable start and the strongest frozen
behaviour. A FAIL there is a statement about parameters and mechanisms
that no trained policy can argue with. Everything else in the table is
diagnostic, and the *differences* are the information:

| Difference | What it measures |
|---|---|
| `greedy` - `eat` | what search is worth (and what it costs) |
| `greedy_hide` - `greedy` | how much survival comes from the predation refuge rather than from energy |
| co-located - configured | how much of the outcome is spawn geometry |

The co-located corner is deliberately *generous*. It answers "can this
parameterisation carry an ecosystem given a perfect start", not "is the
layout in `fg_library.yaml` viable". Both are legitimate questions; they
just must not share a column.

## 3. The rig

    python3 tools/viability.py                              # defaults below
    python3 tools/viability.py --grid 32*32 --years 10
    python3 tools/viability.py --ticks 300 --seeds 1 --behaviour eat   # smoke
    python3 tools/viability.py --spawn configured             # geometry as drawn
    python3 tools/viability.py --migration on --mortality on  # open box
    python3 tools/viability.py --behaviour policy --run-name <run>

Defaults: `--project mareld2.yaml --grid 20*20 --years 5 --seeds 3
--behaviour greedy,greedy_hide,eat --spawn colocated --mortality off
--migration off`.

Exit code `1` means the deciding arm failed the criterion, `0` means it
passed *or* that no arm was entitled to decide, so the rig can gate a
pipeline. `--json` writes the machine-readable summary (including
`spawn`, `verdict_arm`, `normative_corner` and the tick-0 `overlap`),
`--csv` the per-FG trajectories (stride `--sample-every`).

### Behaviour arms

The behaviour is frozen for the whole rollout. None of the arms
evaluates a reward function:

| Arm | What it is | May decide |
|---|---|---|
| `greedy` | hand-coded gradient ascent: move to the neighbour with the highest reachable intake, otherwise eat. `Rest` only as the mask fallback. | yes |
| `greedy_hide` | `greedy` plus the two other uses of `Rest` in the engine - cheap metabolism and the predation refuge - under one explicit rule. | yes |
| `eat` | all action mass on the legal prey in the cell, never move; cells with nothing to eat rest. Maximal intake, zero mobility. | yes |
| `neutral` | no policy at all: zero logits, so the masked softmax is uniform over the legal actions in each cell. Deterministic. | no |
| `random` | freshly initialised `PolicyNetwork` per decision maker, seeded. | no |
| `policy` | a checkpoint from `results/<run-name>`, for comparison against the frozen arms. | no - it is a trained policy, so a FAIL could be a training problem |

Both `eat` and the `greedy` arms allocate the eat mass **by marginal
energy return**, not evenly - see "The diet" below.

#### How `greedy` decides

The field it climbs is the engine's own payoff: energy intake per unit
of predator biomass, i.e. the Holling II/III response with
Beddington-DeAngelis interference from `predation.apply_predation`, with
the predator's biomass, its hunger and the eat share divided out. Two
choices in it are load-bearing, and both were made because the naive
version measured its own artefact:

* **Payoff, not raw prey biomass.** Intake saturates at `1/h` and
  interference lowers it, so a cell that already holds the whole
  population is no longer the best cell. A raw-biomass gradient instead
  pulls every group into the single richest cell and manufactures the
  shoal density section 74 says a 1 km mean field cannot sustain.
* **A max, not a sum, and travel has to pay for itself.** The potential
  is `P(c) = max(payoff(c), 0.85 * max P(neighbour))`, the best payoff
  reachable from `c` discounted per kilometre, and a cell moves only
  when `0.85 * P(best neighbour) > payoff(c)` - the discounted gain must
  beat the feeding tick that travelling costs. A diffusive (summed)
  potential peaks at the prey's centre of mass instead of at the best
  cell: measured on 8x8, `porpoises` ended at 0.03x spawn under that
  version against 0.51x for simply standing still and eating. An arm
  that is *worse* than `eat` is not an upper bound.

`greedy` never rests voluntarily, so it cannot pass by hibernating.
`greedy_hide` may, which is why it does not outrank `greedy`.

#### The diet

`apply_predation` is **linear** in the per-prey eat share up to the
harvest cap `MAX_HARVEST_FRAC * B_prey`, so the energy-maximising diet
is water-filling in descending order of marginal yield
`energy_gain_j * a_eff_j`: fill the best prey until extra share stops
buying anything, then the next one. Whatever is left once every legal
prey is saturated goes to the best of them - it buys no energy, but the
whole population still pays the feeding cost, so the arms stay
comparable.

The arms used to spread the mass **evenly** over the legal prey present,
and that is not an upper bound on intake. For `porpoises` the
difference is decisive (section 92): the engine closes the energy budget
over the *ration*, and

    requirement          5670 MJ/t assimilated
    herring              6750 MJ/t   -> clears it by 19 %
    gadoid               4620 MJ/t   -> does not pay on its own
    even 50/50 split     5685 MJ/t   -> clears it by 0.3 %

An even split therefore fed the top predator junk food and put it three
parts per thousand from starvation, so its death said as much about the
arm as about the world. Correcting it moved `porpoises` from
`extinct@176` to `extinct@1369` in the normative corner on 20x20.

The criterion is marginal *yield*, not energy content: a cell dense in
gadoid can out-yield the same cell's thin herring, and eating the gadoid
there is the right answer. What may never happen is a split that
ignores both.

### Spawn geometry

`--spawn colocated` (default) moves every predator onto the richest
cells of its own prey, bottom up through the food web, **keeping its
total biomass and its number of occupied cells** - so per-cell density
and patchiness are untouched and only the geometry changes.
`--spawn configured` leaves the library's own layouts alone.

The predator/prey overlap at tick 0 is printed with the verdict and
stored in the JSON, per pair:

* `with prey` - share of the predator's biomass in a cell where the prey
  exists at all,
* `prey seen` - prey density seen by the average unit of predator
  biomass, divided by the plain grid mean (1.0 = indifferent layout).

A FAIL without these two numbers cannot be read.

### The report per arm

    fg                spawn   equilib  eq/spawn    min  drift  occ  extinct  verdict

`equilib` is the measured equilibrium (`eq_f` above) in tonnes -
the number to recalibrate `inference_initial_biomass` against. `min` is
the lowest point of the whole trajectory relative to spawn, which
separates "dipped and recovered" from "never left". `drift` is criterion
4. `extinct` is the first tick with zero biomass.

## 4. Reading a FAIL

* **`extinct@<tick>` early and `occ = 0`** - the group never had a chance;
  any conclusion about it from a training run is empty (section 87.5 for
  `porpoises` at 16x16).
* **`below floor` with `drift` near 1** - a settled equilibrium that sits
  far under spawn. This is a *calibration* result, not a collapse:
  recalibrate spawn against `equilib`.
* **`below floor` with `drift` well above 1** - still sliding at the end
  of the horizon. Extend `--years` before concluding anything.
* **A top predator dying under a mixed menu** - check the ration, not
  the pairs. `tools/probes/budget_gate.py` prints the minimum share of
  the best prey the diet needs (`min share`); a pair that cannot pay on
  its own is low-quality food, not pure loss.
* **Arms disagree** - the outcome depends on behaviour, so it is not a
  pure parameter statement. A group that survives under `eat` but dies
  under `neutral` is energy-limited, not structurally doomed.
* **The spawn settings disagree** - the outcome depends on geometry. A
  group that only fails under `--spawn configured` is telling you about
  `fg_library.yaml`'s spawn strategies, not about the tick.
* **A FAIL in the normative corner** is the only one that closes the
  question. There is nothing left to try: the start was as good as it
  gets and the behaviour was the strongest available without training.

## 5. Relation to `--population-stability`

`POPULATION_STABILITY.md` describes a *training* mode: bounds enforced
inside the fitness, which shapes what ARS optimises. This rig is the
opposite: a diagnostic, outside training, with no reward at all. Use the
rig to decide whether a configuration is worth training in; use
`--population-stability` only afterwards, to keep a run inside bounds.
