# Long-term viability: the criterion and the frozen-behaviour rig

Before a scenario comparison means anything, one question needs an
answer: **is the world self-sustaining with the behaviour it will
actually run with?** A group that dies out or slides towards zero inside
the horizon makes every comparison taken in that world uninterpretable
(`mareld_resume.txt` sections 73, 74 and 87.4). This file defines the
question as a number (section 1) and describes two ways of measuring it:
the frozen-behaviour rig (sections 2-4) and trained policies (section 6).

**The rig is a diagnostic for the mechanics, not the judge of the
world.** It runs with hand-coded behaviour and no ARS, which makes it
fast and deterministic - the right instrument for measuring a parameter
or mechanism change in isolation, free of training noise. Its verdict,
however, is a statement about that behaviour. The premise this file was
first written on - "if the world fails the rig, no amount of training can
fix it" - has been falsified: on the same library, trained policies keep
`gadoids`, `pelagic_fish` and `phytoplankton` viable over 10 000 ticks
where the rig's normative corner fails all three (section 6, and
`mareld_resume.txt` section 123). What still holds is the converse
direction: a group that fails **both** under frozen behaviour and with
trained policies - today `porpoises` - has a structural problem no
policy will fix.

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

  Known limitation: a two-window ratio cannot tell a slide from a
  **cycle**. Predator-prey oscillations are ordinary dynamics, and one
  whose period is comparable to the horizon fails `max_drift` depending
  on where the windows fall. Under trained policies `gadoids` run a
  cycle from 1.0x to 3.1x spawn and back over ~10 000 ticks, and the
  ratio of the last two quarters of that run is 0.44 - a FAIL at
  `max_drift = 2.0` for a population that never approached the floor.
  A cycle-tolerant test (a trend over the whole horizon, or a floor on
  the running minimum) is the proposed replacement; it is not
  implemented.

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

## 2. What the rig's verdict means

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

**The rig's verdict is taken from the normative corner: co-located
spawn + `greedy`.** The most favourable start and the strongest frozen
behaviour *per tick*. That makes a FAIL there a statement about the
parameters and mechanisms **under that behaviour** - not about every
behaviour the world could carry.

`greedy` maximises each group's own intake every tick. That is an upper
bound on short-term intake, not on long-term persistence: when every
grazer and predator takes the most it can, the shared resource is
over-exploited, and in the current library `greedy` grazes
phytoplankton down to ~0.05x spawn and drags the chain above it along. A
trained policy can hold back - under trained policies phytoplankton sits
stationary at ~0.56x - so a trained world can pass where the normative
corner fails. Section 6 has the numbers.

Everything else in the table is diagnostic, and the *differences* are
the information:

| Difference | What it measures |
|---|---|
| `greedy` - `eat` | what search is worth (and what it costs) |
| `greedy_hide` - `greedy` | how much survival comes from the predation refuge rather than from energy |
| `greedy_flee` - `greedy_hide` | what running from predators buys over hiding from them |
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
    python3 tools/viability.py --behaviour greedy_stay --mixed-fg zooplankton --run-name <run>

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
| `greedy_stay`, `greedy_hide_stay`, `greedy_flee_stay` | the arm without `_stay`, plus a stop rule: a cell that can feed does not climb while its intake at full appetite covers `STAY_COST_RATIO` (1.0) x its feeding cost. | yes |
| `greedy_flee` | "go towards food, run from predators, eat when hungry": `greedy`, but a cell whose predators outweigh it moves all its mass to the legal neighbour with the least predator biomass (if strictly less), and a cell with a reserve >= 0.95 full rests. Never hides. | yes |
| `eat` | all action mass on the legal prey in the cell, never move; cells with nothing to eat rest. Maximal intake, zero mobility. | yes |
| `eat_pays` | `eat` where the intake at the cell's appetite covers the feeding cost, rest elsewhere; never moves (section 177, the plankton calibration). | no |
| `neutral` | no policy at all: zero logits, so the masked softmax is uniform over the legal actions in each cell. Deterministic. | no |
| `random` | freshly initialised `PolicyNetwork` per decision maker, seeded. | no |
| `policy` | a checkpoint from `results/<run-name>`, measured against the same criterion. | no - the rig's exit code stays tied to the frozen arms, but see section 6: a PASS here is the stronger evidence of viability, and a FAIL may be a training problem |

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
* **A FAIL in the normative corner** says the world cannot carry that
  group under maximal per-tick greed from a perfect start. It closes the
  question only if trained policies fail too (section 6). A group that
  fails here but survives with trained policies is limited by
  over-exploitation, not by its parameters - today `gadoids`,
  `pelagic_fish` and `phytoplankton`.
* **A FAIL under both** - frozen and trained - is the structural case:
  look at the energy budget (`tools/probes/budget_gate.py`), the
  starvation window and the menu. Today that is `porpoises`.

## 5. Relation to `--population-stability`

`POPULATION_STABILITY.md` describes a *training* mode: bounds enforced
inside the fitness, which shapes what ARS optimises. This rig is the
opposite: a diagnostic, outside training, with no reward at all. Use the
rig to find mechanical faults and to measure library changes; use
`--population-stability` only during training, to keep a run inside
bounds.

## 6. Viability with trained policies

The question in the first paragraph is about the behaviour the world
will actually run with, so the criterion of section 1 applied to
trained policies is the stronger evidence. Two ways to measure it today:

    # the rig's own criterion and report, policy arm (no exit-code verdict)
    python3 tools/viability.py --behaviour policy --run-name <run> \
        --grid 60*60 --spawn configured --migration on --mortality on

    # a long inference rollout, trajectories to .npz, criterion by hand
    python3 inference.py --project mareld2.yaml --run-name <run> \
        --ticks 10000 --seed 1 --mortality on --migration on \
        --currents on --quiet --output run.npz

The rig has no `--currents` flag, so a run trained with currents is only
reproduced exactly by `inference.py`. `build_env(seed=...)` seeds the
global RNG since section 124.4, so both are reproducible per seed. Use
several seeds: a perturbation of 1e-5 t per cell and tick grows to a few
per cent over 3000 ticks, so one seed is one sample.

Measured, `results/pzbpgp248nsrpfix35` (co-evolved, 60x60, migration,
currents and mortality on; library before section 124), one seed,
10 000 ticks (~6.8 years), biomass / spawn in the final window, against
the rig's normative corner on the library of section 121:

| fg | rig, frozen `greedy` | trained policies |
|---|---|---|
| gadoids | below floor | 1.09 - cycle 1.0-3.1 |
| pelagic_fish | below floor | 1.38 - lowest 0.96 after tick 2000 |
| porpoises | extinct@71 | extinct@2365 (~1.6 years) |
| phytoplankton | 0.054, below floor | 0.56, stationary |
| benthic_community | 0.555 | 0.65 - lowest 0.49 |
| zooplankton | 0.270 | 2.29, stationary |

Three of the four groups the rig fails are viable with trained
policies; only `porpoises` fails both. The zoo:phyto ratio settles at
1.8:1 against 0.44:1 at spawn - the equilibrium is an output, so that is
a calibration fact about the spawn levels, not a failure.

(The fix58 numbers below are from a world that no longer exists: the
nutrient pool, phytoplankton M1 and microzooplankton of sections 172-177
changed the FG set and every checkpoint.)

Measured, `results/pzbpgp248nsrpfix58` (co-evolved, 1x16 tanh, one-year
rollouts, ~750 updates) against every frozen arm on its own world:
`mareld2.yaml` with seals muted, library of section 168 (seal entries
inert), 60x60, `--spawn configured --mortality on --migration on
--currents on`, 5 years, 3 seeds. Final-year mean / spawn, range over
seeds (resume section 169):

| fg | greedy | greedy_hide | greedy_flee | eat | trained |
|---|---|---|---|---|---|
| phytoplankton | 0.12 | 0.15-0.25 | 0.15-0.23 | 0.14 | 0.59-0.62 |
| zooplankton | 0.16 | 0.18-0.19 | 0.17-0.18 | 0.16 | 0.72 |
| benthic_community | 0.13-0.27 | 0.04-0.11 | 0.05-0.07 | 1.24-1.32 | 0.55-0.62 |
| pelagic_fish | 0.001 | 0.000-0.001 | 0.002 | 0.25 | 0.55-0.59 |
| gadoids | 0.92-1.03 | 0.17-0.43 | 0.18-0.32 | 0.20-0.24 | 3.2 |
| porpoises | extinct@43-49 | extinct@47 | extinct@43 | extinct@1755+ | 1.00-1.18 |

The trained policies beat every hand-coded arm for every group, the
`_stay` arms included (herring 0.01-0.02, zooplankton 0.16-0.17). The
moving arms lose herring and porpoises the same way: the reachable-payoff
potential always shows a better cell somewhere, so 90-100 % of the mass
travels every tick and starves on the way (herring reserve 0.7 -> 0.02
in 500 ticks). The trained world fails the rig criterion only on
zooplankton drift (worst seed 3.06, a cycle).

Mixed arms (`--mixed-fg FG --run-name <run>`) put one group on a
hand-coded rule and leave the rest on the trained policies. On fix58
only `porpoises` can be replaced (0.97-1.05 vs 1.00-1.18); a rule on
gadoids reaches 1.35-1.53 vs 3.2, on herring 0.06-0.26 vs 0.55-0.59, on
zooplankton 0.16 vs 0.72 (the net rests ~78 % of the zooplankton mass,
every rule eats with all of it). Resume section 169.7-169.9.

Not built yet: the policy arm as a deciding arm, with several seeds and
the cycle-tolerant stationarity test of section 1. Until then a verdict
on trained policies is taken by hand from runs like the one above.
