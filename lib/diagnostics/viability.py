"""Long-term viability rig: is the ecosystem self-sustaining *without* ARS?

Motivation
----------
``--population-stability`` is reward shaping *inside* training; it cannot
answer "is the configured world viable at all", because a trained policy
and a collapsing world are measured at the same time. Sections 73, 74 and
87.4 of ``mareld_resume.txt`` all end in the same place: the base has an
effectively absorbing zero, the top predator cannot break even in a mean
field, and the spawn levels sit away from the equilibrium - but there has
never been a *test* that turns those observations into a number that can
be signed off.

This module is that test. It runs long rollouts of the reference NumPy
tick with frozen, neutral behaviour, so nothing about the outcome depends
on a policy checkpoint or on the reward function. What it measures:

  * per-FG biomass trajectory over the whole horizon,
  * the tick at which each FG goes extinct (if it does),
  * the *measured* equilibrium level, i.e. the time mean over the final
    window - the number needed to recalibrate spawn, and
  * a PASS/FAIL verdict against an explicit criterion.

The criterion (see ``VIABILITY.md``)
------------------------------------
A configuration is viable when, for **every** functional group present at
spawn and **every** seed:

  1. survival   - the group never reaches zero total biomass,
  2. floor      - its final-window mean is >= ``floor`` x its spawn level,
  3. ceiling    - its final-window mean is <= ``ceiling`` x its spawn level,
  4. stationary - the final window differs from the preceding window by at
                  most a factor ``max_drift``, i.e. the system has settled
                  rather than still being on its way somewhere.

Criteria 2 and 3 are deliberately wide: the point is not that the world
holds its spawn level (it should not have to - the equilibrium is an
output, not an input), but that it does not decay towards zero or run
away. Criterion 4 is what makes the statement "long-term".

Neither ``--mortality`` nor ``--migration`` is baked in. The open
question of whether Mareld is modelled as a closed box or as a system
with recruitment from outside is a design decision, not a default, so
both are plain flags and the verdict is reported per setting.

Two factors, not one
--------------------
A behaviour arm on its own cannot carry the verdict, because the
outcome also depends on *where the groups start*. In ``mareld2.yaml``
``gadoids`` and ``pelagic_fish`` are both drawn as 25 independent
colonies, so only ~1/3 of the cod biomass spawns in a cell that holds
any herring at all - exactly what independence predicts. A stationary
arm can never repair that, so a FAIL says as much about the spawn
layout as about the parameters. The rig therefore separates

  * **behaviour** - ``eat`` (maximal intake, zero mobility) vs
    ``greedy`` (climbs the prey gradient, then eats) vs ``greedy_hide``
    (``greedy`` plus the hiding use of ``Rest``), and
  * **spawn geometry** - as configured vs co-located, where every
    predator is moved onto the richest cells of its own prey while
    keeping its total biomass *and* its cell count.

The verdict hangs on the normative corner, co-located + ``greedy``:
the most favourable start and the strongest frozen behaviour. A FAIL
there is a statement about parameters and mechanisms that no trained
policy can argue with. The other cells of the table are diagnostics,
and the difference between them is the information:

  * ``greedy`` - ``eat``          : how much is lost to search,
  * ``greedy_hide`` - ``greedy``  : how much survival comes from
                                    predation refuge rather than energy,
  * co-located - configured       : how much is spawn geometry.

The overlap between each predator and its prey at tick 0 is reported
alongside the verdict, because a FAIL cannot be read without it.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from lib.world.tick_time import DEFAULT_TICK_HOURS, ticks_per_year

#: Ticks in a simulated year at the historical 6 h tick. Kept as a
#: module constant because ``tools/viability.py`` converts a --ticks
#: override back into years with it; a project on another tick length
#: passes its own via ``ViabilityCriterion.tick_hours``. Section 97.
TICKS_PER_YEAR = ticks_per_year(DEFAULT_TICK_HOURS)

NEUTRAL = "neutral"
EAT = "eat"
GREEDY = "greedy"
GREEDY_HIDE = "greedy_hide"
RANDOM = "random"
POLICY = "policy"
BEHAVIOURS = (NEUTRAL, EAT, GREEDY, GREEDY_HIDE, RANDOM, POLICY)

#: Arms that are an upper bound on what behaviour can achieve, and may
#: therefore decide a verdict. ``neutral`` and ``random`` cannot: a
#: uniform or untrained policy dying proves nothing about the world.
NORMATIVE_BEHAVIOURS = (GREEDY, GREEDY_HIDE, EAT)

#: Which arm decides when several were run. ``greedy`` is the normative
#: corner; ``greedy_hide`` is more generous but can pass for the wrong
#: reason (resting through the horizon), so it only decides on its own.
VERDICT_PRECEDENCE = (GREEDY, GREEDY_HIDE, EAT, NEUTRAL, RANDOM, POLICY)

SPAWN_CONFIGURED = "configured"
SPAWN_COLOCATED = "colocated"

#: Per-cell travel discount in the potential the greedy arms climb:
#: ``P(c) = max(payoff(c), decay * max P(neighbour))``, so a cell is
#: worth the best payoff reachable from it, discounted by ``decay`` per
#: kilometre travelled. One free parameter, one meaning - what a tick
#: of travel costs relative to a tick of feeding.
TRAVEL_DISCOUNT = 0.85

#: ``greedy_hide`` rests when the predator biomass in the cell reaches
#: this multiple of the group's own biomass there ...
HIDE_PREDATOR_RATIO = 1.0
#: ... or when the energy reserve is this full, so intake buys nothing.
HIDE_SATIATION = 0.95


@dataclass(frozen=True)
class ViabilityCriterion:
    """The acceptance criterion, in one place and with explicit defaults.

    ``years`` is the horizon in simulated years, ``seeds`` the number of
    independent spawn layouts, and ``window_frac`` the share of the
    horizon used as the "final window" over which the equilibrium level
    and the drift are measured.
    """

    years: float = 5.0
    seeds: int = 3
    floor: float = 0.10
    ceiling: float = 10.0
    max_drift: float = 2.0
    window_frac: float = 0.10
    #: Hours per tick, from ``project_metadata.tick_hours``. Only the
    #: years <-> ticks conversion depends on it; the criterion itself is
    #: expressed in simulated years and so is tick-length independent.
    tick_hours: int = DEFAULT_TICK_HOURS

    def __post_init__(self):
        if not self.years > 0:
            raise ValueError("years must be positive")
        if not self.seeds >= 1:
            raise ValueError("seeds must be at least 1")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("floor must be in (0, 1]")
        if not self.ceiling >= 1.0:
            raise ValueError("ceiling must be at least 1")
        if not self.max_drift > 1.0:
            raise ValueError("max_drift must be greater than 1")
        if not 0.0 < self.window_frac <= 0.5:
            raise ValueError("window_frac must be in (0, 0.5]")

    @property
    def ticks(self) -> int:
        return max(1, int(round(self.years * ticks_per_year(self.tick_hours))))

    @property
    def window(self) -> int:
        """Length of the final window in ticks (at least one tick)."""
        return max(1, int(round(self.ticks * self.window_frac)))

    def describe(self) -> str:
        return (
            f"{self.years:g} years ({self.ticks} ticks of {self.tick_hours} h), "
            f"{self.seeds} seed(s), "
            f"floor {self.floor:g}x spawn, ceiling {self.ceiling:g}x spawn, "
            f"drift <= {self.max_drift:g}x over the last "
            f"{self.window_frac:g} of the horizon"
        )


@dataclass
class SeedResult:
    """Raw trajectory of one rollout: per-FG total biomass per tick."""

    seed: int
    reference: Dict[str, float]
    series: Dict[str, np.ndarray]
    extinct_tick: Dict[str, Optional[int]]
    occupancy: Dict[str, int]
    habitable_cells: int


@dataclass
class FGVerdict:
    fg_id: str
    reference: float
    equilibrium: float
    ratio: float
    min_ratio: float
    drift: float
    extinct_tick: Optional[int]
    survived: bool
    floor_ok: bool
    ceiling_ok: bool
    stationary_ok: bool
    occupancy: int

    @property
    def ok(self) -> bool:
        return (self.survived and self.floor_ok and self.ceiling_ok
                and self.stationary_ok)

    @property
    def reason(self) -> str:
        if not self.survived:
            return f"extinct@{self.extinct_tick}"
        if not self.floor_ok:
            return "below floor"
        if not self.ceiling_ok:
            return "above ceiling"
        if not self.stationary_ok:
            return "still drifting"
        return "ok"


@dataclass
class ArmVerdict:
    """One behaviour arm, aggregated over seeds (worst seed decides)."""

    behaviour: str
    criterion: ViabilityCriterion
    per_seed: Dict[int, List[FGVerdict]] = field(default_factory=dict)

    @property
    def fg_ids(self) -> List[str]:
        for verdicts in self.per_seed.values():
            return [v.fg_id for v in verdicts]
        return []

    def worst(self, fg_id: str) -> FGVerdict:
        """The failing verdict for ``fg_id`` if any seed fails, else the
        one with the lowest ratio - the seed a reader should look at."""
        candidates = [v for verdicts in self.per_seed.values()
                      for v in verdicts if v.fg_id == fg_id]
        failing = [v for v in candidates if not v.ok]
        pool = failing or candidates
        return min(pool, key=lambda v: v.ratio)

    @property
    def ok(self) -> bool:
        return all(v.ok for verdicts in self.per_seed.values() for v in verdicts)


# ---------------------------------------------------------------- behaviour


def _zero_logit_policies(env):
    """Neutral behaviour: no network at all.

    ``PolicyController`` emits zero logits for every DM without a policy,
    so the masked softmax is uniform over the legal actions in each cell.
    That is the closest thing to "no decision" the action layout allows,
    and it is fully deterministic.
    """
    env.policies = {}
    env.obs_mean = None
    env.obs_var = None
    env.rebuild_batched_weights()
    return None


def _random_policies(env, seed):
    """Untrained behaviour: freshly initialised networks, one per DM."""
    import torch

    from lib.runners.policy import PolicyNetwork

    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    out_dim = 5 + env.N_all
    env.policies = {
        fid: PolicyNetwork(int(env.per_dm_in_dim[i]), out_dim)
        for i, fid in enumerate(env.dm_ids)
    }
    env.obs_mean = None
    env.obs_var = None
    env.rebuild_batched_weights()
    return None


def _trained_policies(env, checkpoint_dir):
    from inference import load_policies_and_stats

    policies, mean, var = load_policies_and_stats(
        env, checkpoint_dir, verbose=False)
    env.policies, env.obs_mean, env.obs_var = policies, mean, var
    env.rebuild_batched_weights()
    return None


def _per_prey_intake(env):
    """``(N_dm, N_all, H, W)`` prey tonnes per tonne of predator per tick.

    Exactly the rate ``predation.apply_predation`` builds its demand
    from - the Holling II/III response with Beddington-DeAngelis
    interference - with the predator's own biomass, its hunger and the
    eat share divided out, and restricted to the static menu.
    """
    from lib.environments.ecosystem_env import interactions

    prey = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order],
        axis=0)[None, :, :, :]
    intake_rate = env.max_intake_mat[:, :, None, None]
    if env._has_interference:
        predator = np.stack(
            [env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
        interference = env.dm_interference * predator[:, None, :, :]
    else:
        interference = np.float32(0.0)

    if env._has_holling2 or env._has_holling3:
        per_unit = interactions.holling_a_eff(
            intake_rate, env.handling_time_mat[:, :, None, None], prey,
            env._type3_pred_mask, interference)
    else:
        # No handling time anywhere: the response is unsaturated and
        # the engine's rate does not depend on prey density at all, so
        # the gradient would be flat. Fall back to the linear form,
        # which is what that rate is capped by in practice.
        per_unit = intake_rate * prey / (np.float32(1.0) + interference)

    return (env.eat_static_mask[:, :, None, None] * per_unit).astype(
        env.dtype, copy=False)


def _saturating_share(env, per_prey_intake):
    """``(N_dm, N_all, H, W)`` eat share beyond which a prey gives nothing.

    ``apply_predation`` caps the removal from prey ``j`` at
    ``MAX_HARVEST_FRAC * B_j``, and the demand it computes is linear in
    the eat share::

        demand_j = B_pred * share_j * a_eff_j * h

    so every share above ``MAX_HARVEST_FRAC * B_j / (B_pred * a_eff_j *
    h)`` is scaled straight back down again. That ceiling is what makes
    a mixed diet worth anything at all to an energy maximiser: without
    it the optimum is always the single best prey.
    """
    from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC

    prey = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order],
        axis=0)[None, :, :, :]
    predator = np.stack(
        [env.fgs[fid].biomass for fid in env.dm_ids], axis=0)[:, None, :, :]
    hunger = np.stack(
        [env.fgs[fid].get_hunger().astype(env.dtype, copy=False)
         for fid in env.dm_ids], axis=0)[:, None, :, :]

    per_share = predator * per_prey_intake * hunger
    cap = np.where(
        per_share > 0.0,
        (MAX_HARVEST_FRAC * prey)
        / np.maximum(per_share, np.float32(1e-30)),
        np.float32(1.0))
    return np.clip(cap, 0.0, 1.0).astype(env.dtype, copy=False)


def _waterfill(score, caps):
    """Unit action mass over the prey axis, best marginal yield first.

    ``score`` ranks the prey and ``caps`` is the share above which each
    stops paying. Whatever is left once every prey is saturated goes to
    the top-ranked one - see :func:`_eat_shares` for why the mass is
    spent rather than withheld.
    """
    order = np.argsort(-score, axis=1)
    caps_sorted = np.take_along_axis(caps, order, axis=1)
    already_taken = np.cumsum(caps_sorted, axis=1) - caps_sorted
    allocated = np.clip(np.float32(1.0) - already_taken,
                        np.float32(0.0), caps_sorted)
    leftover = np.float32(1.0) - allocated.sum(axis=1)
    allocated[:, 0] += np.maximum(leftover, np.float32(0.0))
    shares = np.zeros_like(allocated)
    np.put_along_axis(shares, order, allocated, axis=1)
    return shares


def _eat_shares(env, mask):
    """``(N_dm, N_all, H, W)`` eat shares by marginal energy return.

    The engine's intake is *linear* in the per-prey eat share up to the
    harvest cap, so the diet that maximises energy is water-filling in
    descending order of marginal yield ``energy_gain_j * a_eff_j``:
    fill the best prey until extra share stops buying anything, then
    the next one. An even split - what these arms used to do - is not
    an upper bound on intake, and for ``porpoises`` it is the
    difference between a diet of herring and a 50/50 herring/gadoid
    mixture that sits within three parts per thousand of break-even
    (6750 vs 4620 MJ/t assimilated against a 5670 MJ/t requirement).
    An arm that feeds the top predator junk food cannot pronounce a
    verdict on the top predator's energy budget.

    Any share left over once every legal prey is saturated goes to the
    best of them. It buys no energy, but it keeps the arms comparable:
    the whole population still pays the feeding cost, exactly as under
    the even split, so the before/after difference is the diet and
    nothing else.

    Returns ``(shares, feeding)`` where ``feeding`` is
    ``(N_dm, 1, H, W)`` and marks the cells with anything legal to eat.
    """
    from lib.environments.ecosystem_env.constants import EAT_START

    legal = mask[:, EAT_START:EAT_START + env.N_all] > 0
    per_prey = _per_prey_intake(env)
    yield_per_share = env.energy_gain_mat[:, :, None, None] * per_prey

    # Illegal prey must sort last and may never be allocated to.
    score = np.where(legal, yield_per_share, np.float32(-1.0))
    caps = np.where(legal, _saturating_share(env, per_prey), np.float32(0.0))

    shares = _waterfill(score, caps)
    feeding = legal.any(axis=1, keepdims=True)
    shares = np.where(feeding, shares, np.float32(0.0)).astype(
        env.dtype, copy=False)
    return shares, feeding


def _eat_actions(env):
    """Hand-coded behaviour: eat whatever is legal, never move.

    The action mass is allocated over the prey that are both on the
    menu and present in the cell by marginal energy return (see
    :func:`_eat_shares`); cells with nothing to eat rest. This is the
    ecological upper bound on intake at a fixed position - if the
    system is not viable here it is not a learning problem.
    """
    from lib.environments.ecosystem_env import decisions
    from lib.environments.ecosystem_env.constants import EAT_START
    from lib.environments.ecosystem_env.state import ActionProbabilities

    num_actions = EAT_START + env.N_all
    mask = decisions.build_action_mask(env, num_actions)
    eat, feeding = _eat_shares(env, mask)
    rest = np.where(feeding[:, 0], np.float32(0.0),
                    np.float32(1.0)).astype(env.dtype, copy=False)
    move = np.zeros((env.N_dm, 4, env.H, env.W), dtype=env.dtype)
    return ActionProbabilities(move=move, rest=rest, eat=eat)


def _intake_payoff(env, shares=None):
    """``(N_dm, H, W)`` energy intake per unit of predator biomass.

    The energy a tonne of predator takes in by feeding in this cell,
    under the diet ``shares`` (defaulting to the one
    :func:`_eat_shares` would choose there). Summing the per-prey rates
    with a share of 1.0 each - what this did before - is not a
    quantity the engine can ever deliver, since the shares are a
    distribution.

    The greedy arms climb this rather than the raw prey biomass, and
    the difference is not cosmetic. Intake saturates at ``1/h`` and
    interference lowers it, so a cell that already holds the whole
    population is no longer the best cell. A raw-biomass gradient
    instead pulls every group into the single richest cell, which
    manufactures precisely the shoal density Section 74 says a 1 km
    mean field cannot sustain - the arm would then answer a question
    about its own artefact instead of about the world.
    """
    per_prey = _per_prey_intake(env)
    caps = _saturating_share(env, per_prey)
    if shares is None:
        present = per_prey > 0.0
        shares = _waterfill(
            np.where(present,
                     env.energy_gain_mat[:, :, None, None] * per_prey,
                     np.float32(-1.0)),
            np.where(present, caps, np.float32(0.0)))
        shares = np.where(present.any(axis=1, keepdims=True), shares,
                          np.float32(0.0))

    # Share above the harvest cap is scaled straight back down by
    # ``apply_predation``, so it must not be counted as energy here
    # either - otherwise the potential would send a group towards a
    # cell whose prey it has already eaten.
    effective = np.minimum(shares, caps)
    payoff = (effective * env.energy_gain_mat[:, :, None, None]
              * per_prey).sum(axis=1)
    return payoff.astype(env.dtype, copy=False)


def _predator_biomass_field(env):
    """``(N_dm, H, W)`` biomass of the groups that prey on each DM."""
    # ``eat_static_mask[k, j]`` is "DM k eats global group j"; the
    # predators of DM i are the ks with j = dm_index_in_all[i].
    menu = env.eat_static_mask[:, env.dm_index_in_all].T.astype(
        env.dtype, copy=False)                      # (N_dm, N_dm)
    biomass_dm = np.stack(
        [env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    return np.tensordot(menu, biomass_dm, axes=(1, 0)).astype(
        env.dtype, copy=False)


def _reachable_payoff(env, payoff, decay=TRAVEL_DISCOUNT, iterations=None):
    """Best payoff reachable from each cell, discounted by distance.

    Value iteration of ``P(c) = max(payoff(c), decay * max_n P(n))``,
    which converges to ``max_d payoff(d) * decay**dist(c, d)``. A bare
    4-neighbour gradient on the payoff is exactly zero outside a patch,
    so ``greedy`` would collapse into ``eat`` - the arm exists to take
    "maximal intake" and "minimal search" apart, so it needs a
    potential that reaches beyond the four neighbours.

    It has to be a *max*, not a sum. A diffusive (summed) potential
    peaks at the prey's centre of mass rather than at the best cell, so
    every group walks to the middle of the grid and starves on the way;
    measured on 8x8, ``porpoises`` ended at 0.03x spawn under a summed
    potential against 0.51x when simply standing still and eating. With
    the max form, moving away from a local optimum requires a strictly
    better cell to exist within reach - i.e. the group only gives up a
    feeding tick when the discounted payoff at the destination beats
    what it would have eaten by staying.
    """
    if iterations is None:
        iterations = env.H + env.W
    step = np.float32(decay)
    potential = payoff.astype(env.dtype, copy=True)
    for _ in range(int(iterations)):
        neighbours = np.full_like(potential, -np.inf)
        np.maximum(neighbours[:, 1:, :], potential[:, :-1, :],
                   out=neighbours[:, 1:, :])
        np.maximum(neighbours[:, :-1, :], potential[:, 1:, :],
                   out=neighbours[:, :-1, :])
        np.maximum(neighbours[:, :, 1:], potential[:, :, :-1],
                   out=neighbours[:, :, 1:])
        np.maximum(neighbours[:, :, :-1], potential[:, :, 1:],
                   out=neighbours[:, :, :-1])
        updated = np.maximum(payoff, step * neighbours)
        if np.array_equal(updated, potential):
            break
        potential = updated
    return potential


def _neighbour_potential(env, potential):
    """``(N_dm, 4, H, W)``: the destination's potential, per direction,
    aligned on the SOURCE cell - the same offsets ``apply_movement``
    uses. Off-grid directions are ``-inf`` so they can never win."""
    from lib.environments.ecosystem_env.constants import (
        EAST, NORTH, SOUTH, WEST)

    out = np.full((potential.shape[0], 4, env.H, env.W), -np.inf,
                  dtype=np.float32)
    out[:, NORTH, 1:, :] = potential[:, :-1, :]      # (y,x) -> (y-1,x)
    out[:, EAST, :, :-1] = potential[:, :, 1:]       # (y,x) -> (y,x+1)
    out[:, SOUTH, :-1, :] = potential[:, 1:, :]      # (y,x) -> (y+1,x)
    out[:, WEST, :, 1:] = potential[:, :, :-1]       # (y,x) -> (y,x-1)
    return out


def _greedy_actions(env, hide=False):
    """Hand-coded gradient ascent on prey, with eating at the local max.

    Per cell, in order:

      1. if a legal neighbour has a strictly higher intake potential, move
         there with all of the cell's action mass,
      2. else eat, allocated over the legal prey present by marginal
         energy return (:func:`_eat_shares`),
      3. else rest (the mask fallback - nothing else is legal).

    ``Rest`` is therefore never chosen voluntarily: this arm answers
    "is the energy budget enough when intake is maximal *and* search is
    optimal", and nothing else. With ``hide`` the two other uses of
    ``Rest`` in the engine are switched on as well - the cheap
    metabolism and the predation refuge - under one explicit rule, so
    the difference between the arms measures what hiding is worth.
    """
    from lib.environments.ecosystem_env import decisions
    from lib.environments.ecosystem_env.constants import EAT_START, MOVE_SLICE
    from lib.environments.ecosystem_env.state import ActionProbabilities

    num_actions = EAT_START + env.N_all
    mask = decisions.build_action_mask(env, num_actions)

    eat, feeding = _eat_shares(env, mask)
    payoff = _intake_payoff(env, shares=eat)
    potential = _reachable_payoff(env, payoff)
    neighbours = _neighbour_potential(env, potential)
    neighbours = np.where(mask[:, MOVE_SLICE] > 0, neighbours, -np.inf)

    best_direction = np.argmax(neighbours, axis=1)
    best_value = np.max(neighbours, axis=1)
    # Travelling costs a feeding tick, so the discounted value of the
    # best neighbour must beat what the cell would have eaten by
    # staying - the greedy step of the same Bellman value the potential
    # solves. Comparing potentials instead would move on an
    # infinitesimal improvement and halve the intake of anything that
    # depletes its own cell, which is how ``porpoises`` first came out
    # *worse* under ``greedy`` than under ``eat``.
    climbing = np.float32(TRAVEL_DISCOUNT) * best_value > payoff

    eat = np.where(feeding & ~climbing[:, None, :, :], eat,
                   np.float32(0.0)).astype(env.dtype, copy=False)

    move = np.zeros((env.N_dm, 4, env.H, env.W), dtype=env.dtype)
    np.put_along_axis(move, best_direction[:, None, :, :],
                      climbing[:, None, :, :].astype(env.dtype), axis=1)

    resting = ~climbing & ~feeding[:, 0]
    if hide:
        biomass_dm = np.stack(
            [env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
        reserve = np.stack(
            [env.fgs[fid].energy_reserve for fid in env.dm_ids], axis=0)
        threatened = (_predator_biomass_field(env)
                      >= np.float32(HIDE_PREDATOR_RATIO) * biomass_dm)
        full = reserve >= (np.float32(HIDE_SATIATION)
                           * env.dm_max_energy_reserve[:, None, None]
                           * biomass_dm)
        hiding = (biomass_dm > 0) & (threatened | full)
        if np.any(hiding):
            resting = resting | hiding
            eat = np.where(hiding[:, None, :, :], np.float32(0.0), eat)
            move = np.where(hiding[:, None, :, :], np.float32(0.0), move)

    rest = resting.astype(env.dtype, copy=False)
    return ActionProbabilities(move=move, rest=rest, eat=eat)


def _greedy_provider(hide):
    def provider(env):
        return _greedy_actions(env, hide=hide)
    return provider


def install_behaviour(env, behaviour, seed=0, checkpoint_dir=None):
    """Freeze ``env``'s behaviour and return a per-tick action provider.

    The provider is ``None`` for behaviours that go through the policy
    network, and a callable ``env -> ActionProbabilities`` for the
    hand-coded ones. Nothing here touches the reward path: the arms
    differ only in the actions that enter the tick.
    """
    if behaviour not in BEHAVIOURS:
        raise ValueError(
            f"unknown behaviour {behaviour!r}, expected one of {BEHAVIOURS}")
    env.build_static_caches()
    if behaviour == NEUTRAL:
        return _zero_logit_policies(env)
    if behaviour == RANDOM:
        return _random_policies(env, seed)
    if behaviour == POLICY:
        if not checkpoint_dir:
            raise ValueError("behaviour 'policy' requires a checkpoint dir")
        return _trained_policies(env, checkpoint_dir)
    if behaviour == GREEDY:
        return _greedy_provider(hide=False)
    if behaviour == GREEDY_HIDE:
        return _greedy_provider(hide=True)
    return _eat_actions


# ---------------------------------------------------------- spawn geometry


def menus(env):
    """``{dm_id: [prey ids]}`` from the project's predation matrix."""
    env.build_static_caches()
    return {
        fid: [env.global_fg_order[j]
              for j in np.flatnonzero(env.eat_static_mask[i] > 0)]
        for i, fid in enumerate(env.dm_ids)
    }


def trophic_order(menus_by_fg, fg_ids):
    """``fg_ids`` sorted so that prey comes before its predators.

    Co-location has to run bottom up: moving ``pelagic_fish`` onto the
    zooplankton changes the field ``gadoids`` and ``porpoises`` are then
    placed on. Cycles (mutual predation) fall back to declared order.
    """
    ordered, remaining = [], list(fg_ids)
    known = set(fg_ids)
    while remaining:
        progressed, pending = False, []
        for fid in remaining:
            deps = [p for p in menus_by_fg.get(fid, [])
                    if p in known and p != fid]
            if all(d in ordered for d in deps):
                ordered.append(fid)
                progressed = True
            else:
                pending.append(fid)
        remaining = pending
        if not progressed:
            ordered.extend(remaining)
            break
    return ordered


def colocate_spawn(env):
    """Move every predator onto the richest cells of its own prey.

    The spawn strategies in ``fg_library.yaml`` draw each group from an
    independent layout, so predator and prey overlap only by chance -
    measured at 0.24-0.44 for ``gadoids`` -> ``pelagic_fish`` on 60x60,
    which is exactly the product of the two marginals. This turns the
    spawn geometry into a *factor* instead of a hidden term in the
    verdict: the group keeps its total biomass and its number of
    occupied cells (so per-cell density and patchiness are unchanged)
    and is simply placed on top of its prey rather than beside it.

    It is deliberately a generous boundary condition. It answers "can
    this parameterisation carry an ecosystem given a perfect start",
    not "is the configured spawn viable" - which is why the rig reports
    the two spawn settings side by side.

    Returns the ids that were moved.
    """
    env.build_static_caches()
    menus_by_fg = menus(env)
    moved = []
    for fid in trophic_order(menus_by_fg, list(env.dm_ids)):
        prey_ids = [p for p in menus_by_fg[fid] if p in env.fgs]
        if not prey_ids:
            continue
        fg = env.fgs[fid]
        biomass = np.asarray(fg.biomass, dtype=np.float64)
        total = float(biomass.sum())
        if total <= 0.0:
            continue
        prey = np.zeros_like(biomass)
        for pid in prey_ids:
            prey += np.asarray(env.fgs[pid].biomass, dtype=np.float64)
        flat = prey.reshape(-1)
        candidates = np.flatnonzero(flat > 0.0)
        if candidates.size == 0:
            continue
        occupancy = max(1, int(np.count_nonzero(biomass > 0.0)))
        if candidates.size > occupancy:
            richest = np.argsort(-flat[candidates])[:occupancy]
            candidates = candidates[richest]
        weights = np.zeros_like(flat)
        weights[candidates] = flat[candidates]
        weights /= weights.sum()
        weights = weights.reshape(biomass.shape)

        reserve_total = float(np.asarray(fg.energy_reserve).sum())
        fg.biomass[...] = (total * weights).astype(env.dtype, copy=False)
        fg.energy_reserve[...] = (reserve_total * weights).astype(
            env.dtype, copy=False)
        moved.append(fid)
    return moved


def spawn_overlap(env):
    """Predator/prey co-location, per pair, as the world stands now.

    ``biomass_with_prey`` is the share of the predator's biomass that
    sits in a cell where the prey exists at all; ``prey_seen`` is the
    prey density the average unit of predator biomass sees, divided by
    the plain grid mean - i.e. 1.0 for an indifferent layout and higher
    when the predator sits on the patches. Without these two numbers a
    FAIL cannot be told apart from a bad starting position.
    """
    rows = []
    for fid, prey_ids in menus(env).items():
        predator = np.asarray(env.fgs[fid].biomass, dtype=np.float64)
        predator_total = predator.sum()
        if predator_total <= 0.0:
            continue
        for pid in prey_ids:
            if pid not in env.fgs:
                continue
            prey = np.asarray(env.fgs[pid].biomass, dtype=np.float64)
            if prey.sum() <= 0.0:
                continue
            rows.append({
                "predator": fid,
                "prey": pid,
                "biomass_with_prey":
                    float(predator[prey > 0.0].sum() / predator_total),
                "prey_seen":
                    float((predator * prey).sum() / predator_total
                          / prey.mean()),
            })
    return rows


def format_overlap(rows) -> str:
    """The tick-0 co-location table that a verdict must be read with."""
    lines = [f"{'predator':<20}{'prey':<20}{'with prey':>11}{'prey seen':>11}"]
    for row in rows:
        lines.append(
            f"{row['predator']:<20}{row['prey']:<20}"
            f"{row['biomass_with_prey']:>11.3f}{row['prey_seen']:>11.2f}")
    return "\n".join(lines)


# ----------------------------------------------------------------- rollout


def run_rollout(env, ticks, action_provider=None, on_tick=None):
    """Run ``ticks`` ticks and record every FG's total biomass per tick.

    Returns a :class:`SeedResult` whose ``series[fid][k]`` is the total
    biomass after tick ``k``; ``reference[fid]`` is the level at spawn.
    """
    fg_ids = list(env.fgs.keys())
    reference = {fid: float(env.fgs[fid].biomass.sum()) for fid in fg_ids}
    series = {fid: np.zeros(int(ticks), dtype=np.float64) for fid in fg_ids}
    extinct = {fid: None for fid in fg_ids}

    for k in range(int(ticks)):
        env.step(action_provider(env) if action_provider is not None else None)
        for fid in fg_ids:
            total = float(env.fgs[fid].biomass.sum())
            series[fid][k] = total
            if total <= 0.0 and extinct[fid] is None:
                extinct[fid] = k + 1
        if on_tick is not None:
            on_tick(k + 1, int(ticks))

    accessibility = env.grid.maps.get('accessibility') \
        if hasattr(env.grid, 'maps') else None
    habitable = (int(np.count_nonzero(np.asarray(accessibility) > 0))
                 if accessibility is not None else env.H * env.W)
    occupancy = {fid: int(np.count_nonzero(env.fgs[fid].biomass > 0))
                 for fid in fg_ids}
    return SeedResult(seed=getattr(env, 'current_world_seed', 0) or 0,
                      reference=reference, series=series,
                      extinct_tick=extinct, occupancy=occupancy,
                      habitable_cells=habitable)


def _window_mean(values, window, offset=0):
    end = len(values) - offset * window
    start = max(0, end - window)
    if end <= 0 or start >= end:
        return float('nan')
    return float(np.mean(values[start:end]))


def evaluate_seed(result: SeedResult, criterion: ViabilityCriterion,
                  fg_ids=None) -> List[FGVerdict]:
    """Turn one trajectory into per-FG verdicts."""
    window = criterion.window
    verdicts = []
    for fid in (fg_ids if fg_ids is not None else result.series.keys()):
        values = result.series[fid]
        reference = result.reference[fid]
        if reference <= 0.0:
            # Absent at spawn: the criterion says nothing about groups the
            # project did not seed, so they cannot fail it either.
            continue
        equilibrium = _window_mean(values, window)
        previous = _window_mean(values, window, offset=1)
        ratio = equilibrium / reference
        min_ratio = float(values.min()) / reference
        if previous > 0.0 and equilibrium > 0.0:
            drift = max(equilibrium / previous, previous / equilibrium)
        elif equilibrium <= 0.0 and previous <= 0.0:
            drift = 1.0
        else:
            drift = float('inf')
        extinct_tick = result.extinct_tick[fid]
        verdicts.append(FGVerdict(
            fg_id=fid,
            reference=reference,
            equilibrium=equilibrium,
            ratio=ratio,
            min_ratio=min_ratio,
            drift=drift,
            extinct_tick=extinct_tick,
            survived=extinct_tick is None,
            floor_ok=ratio >= criterion.floor,
            ceiling_ok=ratio <= criterion.ceiling,
            stationary_ok=drift <= criterion.max_drift,
            occupancy=result.occupancy[fid],
        ))
    return verdicts


# ------------------------------------------------------------------ report


def format_arm(arm: ArmVerdict) -> str:
    """One block per behaviour arm: the worst seed per FG, plus a verdict."""
    lines = [
        f"{'fg':<20}{'spawn':>12}{'equilib':>12}{'eq/spawn':>10}"
        f"{'min':>8}{'drift':>8}{'occ':>6}{'extinct':>9}  verdict",
    ]
    for fg_id in arm.fg_ids:
        v = arm.worst(fg_id)
        extinct = '-' if v.extinct_tick is None else str(v.extinct_tick)
        lines.append(
            f"{v.fg_id:<20}{v.reference:>12.1f}{v.equilibrium:>12.1f}"
            f"{v.ratio:>10.3f}{v.min_ratio:>8.3f}{v.drift:>8.2f}"
            f"{v.occupancy:>6}{extinct:>9}  "
            f"{'PASS' if v.ok else 'FAIL: ' + v.reason}"
        )
    lines.append("")
    lines.append(f"{arm.behaviour}: "
                 f"{'VIABLE' if arm.ok else 'NOT VIABLE'}")
    return "\n".join(lines)


def _finite(value):
    """``None`` instead of inf/nan so the summary is strict JSON."""
    return float(value) if np.isfinite(value) else None


def deciding_arm(arms: List[ArmVerdict]) -> Optional[ArmVerdict]:
    """The arm the verdict is taken from, by :data:`VERDICT_PRECEDENCE`."""
    by_name = {arm.behaviour: arm for arm in arms}
    for behaviour in VERDICT_PRECEDENCE:
        if behaviour in by_name:
            return by_name[behaviour]
    return None


def overall_verdict(arms: List[ArmVerdict]):
    """``True``/``False``, or ``None`` when no arm may decide.

    Only the normative arms carry a verdict. ``neutral`` and ``random``
    dying says nothing about the world - a uniform or untrained policy
    is expected to fail - so a run of those arms alone reports no
    verdict at all rather than a spurious NOT VIABLE.
    """
    arm = deciding_arm(arms)
    if arm is None or arm.behaviour not in NORMATIVE_BEHAVIOURS:
        return None
    return arm.ok


def summary_dict(arms: List[ArmVerdict], spawn: str = SPAWN_CONFIGURED,
                 overlap=None) -> dict:
    """JSON-serialisable summary, for gating and for later comparison."""
    decider = deciding_arm(arms)
    return {
        "spawn": spawn,
        "overlap": list(overlap or []),
        "verdict_arm": decider.behaviour if decider is not None else None,
        "normative_corner": bool(
            decider is not None and decider.behaviour == GREEDY
            and spawn == SPAWN_COLOCATED),
        "criterion": {
            "years": arms[0].criterion.years,
            "ticks": arms[0].criterion.ticks,
            "seeds": arms[0].criterion.seeds,
            "floor": arms[0].criterion.floor,
            "ceiling": arms[0].criterion.ceiling,
            "max_drift": arms[0].criterion.max_drift,
            "window_frac": arms[0].criterion.window_frac,
        } if arms else {},
        "viable": overall_verdict(arms),
        "arms": [
            {
                "behaviour": arm.behaviour,
                "viable": arm.ok,
                "groups": {
                    fg_id: {
                        "spawn": arm.worst(fg_id).reference,
                        "equilibrium": arm.worst(fg_id).equilibrium,
                        "eq_over_spawn": arm.worst(fg_id).ratio,
                        "min_over_spawn": arm.worst(fg_id).min_ratio,
                        "drift": _finite(arm.worst(fg_id).drift),
                        "extinct_tick": arm.worst(fg_id).extinct_tick,
                        "occupancy": arm.worst(fg_id).occupancy,
                        "verdict": arm.worst(fg_id).reason,
                    }
                    for fg_id in arm.fg_ids
                },
            }
            for arm in arms
        ],
    }
