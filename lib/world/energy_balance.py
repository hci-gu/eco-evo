"""Per-tick energy-balance gate for decision makers (Section 23 / 67).

This module holds the *pure* arithmetic behind the biological sanity gate
that ``fgconfig`` enforces when an FG is saved, so the exact same formula
can be asserted by ``tests/test_energy_balance_gate.py`` without pulling
in Tkinter.

Why the gate moved (Section 67)
-------------------------------
The original Section 23 gate evaluated the intake side at FULL hunger::

    intake_at_h1 = max_intake_rate * energy_gain

But ``h_X = max(0, 1 - s_X / HUNGER_SATIATION_SCALE)`` means h = 1
requires s_X = 0, i.e. an animal whose energy reserve is already empty -
a state it does not survive in. The ecologically meaningful question is
whether the FG can break even *at its own maintenance level* ``u_X``,
where growth flips sign (``q_X = s_X - u_X``). That is the point the
population is attracted to, and there the hunger gate only lets through

    h(u_X) = max(0, 1 - u_X / HUNGER_SATIATION_SCALE)

of the physiological ceiling. Evaluating at h = 1 hides exactly the
failure mode where an FG passes every gate on paper yet shrinks at
~2 %/tick with unlimited prey available.

Cost model (must mirror ``_apply_movement`` in
``lib/environments/ecosystem.py``): rest / eat / move are MUTUALLY
EXCLUSIVE population shares, each paying ``resting_metabolism`` times its
own action multiplier. ``resting_cost`` is therefore the reference (1.0 by
convention) and ``feeding_cost`` is the activity multiplier of hunting
relative to lying still - which is why ``feeding_cost < resting_cost`` is
an invalid configuration, not merely a generous one.
"""
from dataclasses import dataclass, field
from typing import List, Optional

# DEFAULT saturation scale of the hunger gate: h_X = max(0, 1 - s_X / SCALE).
# A pure model constant (no literature anchor, unlike resting_metabolism,
# max_energy_reserve, max_intake_rate and maintenance_level). Kept here so
# the runtime (FunctionalGroup.get_hunger), the GUI gate and the tests all
# read the same number.
#
# Per-FG override (Section 69). An FG may set ``satiation_scale`` in
# fg_library.yaml to raise or lower its own gate; the value falls back to
# this constant when absent. The scale is per-FG rather than global
# because the measured effect of raising it is species-specific in BOTH
# directions (Section 68): 0.9 is what lets porpoises break even at u_X
# (their ceiling/Rest_X headroom is only 3.25, against 6.6-92.9 for every
# other DM), while the same change applied to zooplankton deepens the
# phyto-zoo overshoot (zoo floor 0.36x, phyto floor 0.70x). There is no
# cross-species calibration to preserve by keeping it global - the
# headroom already spans a factor of 28 between DMs.
HUNGER_SATIATION_SCALE = 0.8

# Reference value of resting_cost. The runtime default in
# FunctionalGroup / ecosystem caches is 1.0 and no FG in fg_library.yaml
# overrides it; starve_calibration.py anchors resting_metabolism *
# resting_cost against the literature starvation window, so this is the
# fixed point the other action costs are expressed relative to.
REFERENCE_RESTING_COST = 1.0


def resolve_satiation_scale(value):
    """Per-FG ``satiation_scale`` with fallback to the module default.

    Empty / missing / non-positive values resolve to
    :data:`HUNGER_SATIATION_SCALE`, so legacy library entries that never
    declare the field behave exactly as before.
    """
    if value in (None, ""):
        return float(HUNGER_SATIATION_SCALE)
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return float(HUNGER_SATIATION_SCALE)
    if scale <= 0.0:
        return float(HUNGER_SATIATION_SCALE)
    return scale


def hunger_at(s_x, scale=None):
    """h_X at a given relative energy fill ratio s_X.

    Mirrors ``FunctionalGroup.get_hunger`` for a scalar s_X. ``scale=None``
    resolves to :data:`HUNGER_SATIATION_SCALE`.
    """
    scale = resolve_satiation_scale(scale)
    if scale <= 0.0:
        return 0.0
    return max(0.0, 1.0 - float(s_x) / scale)


@dataclass
class EnergyBalance:
    """Result of :func:`evaluate_energy_balance`.

    All energies are MJ per ton of predator per tick.
    """
    fg_id: str
    best_prey: Optional[str]
    # Inputs
    intake_ceiling: float          # max_intake_rate * energy_content * assim
    resting_metabolism: float      # Rest_X
    maintenance_level: float       # u_X
    feeding_cost: float
    resting_cost: float
    movement_cost: Optional[float]
    hunger_scale: float
    # Derived
    hunger_at_maintenance: float   # h(u_X)
    realized_intake: float         # h(u_X) * intake_ceiling
    feed_cost: float               # feeding_cost * Rest_X
    rest_cost: float               # resting_cost * Rest_X
    move_cost: Optional[float]     # movement_cost * Rest_X
    net_eat: float                 # realized_intake - feed_cost
    net_eat_at_h1: float           # intake_ceiling - feed_cost (legacy view)
    eat_minus_rest: float          # net_eat + rest_cost (eat vs. do nothing)
    eat_minus_move: Optional[float]
    headroom_ratio: float          # intake_ceiling / Rest_X  ("tak/rm")
    required_ratio: float          # feeding_cost / h(u_X)
    max_feeding_cost: float        # h(u_X) * intake_ceiling / Rest_X
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self):
        """True when no blocking gate is violated."""
        return not self.failures

    def report(self):
        """Multi-line human-readable breakdown (GUI dialog / test output)."""
        prey = self.best_prey or "(no prey with preys_on: true found)"
        lines = [
            f"  best prey                = {prey}",
            f"  intake ceiling (h=1)     = max_intake_rate * energy_content"
            f" * assim = {self.intake_ceiling:.3f}",
            f"  u_X                      = {self.maintenance_level:.3f}",
            f"  h(u_X) = 1 - u_X/{self.hunger_scale:g}"
            f"        = {self.hunger_at_maintenance:.3f}",
            f"  realized intake at u_X   = h(u_X) * ceiling"
            f" = {self.realized_intake:.3f}",
            f"  feed_cost                = feeding_cost * resting_metabolism"
            f" = {self.feed_cost:.3f}",
            f"  rest_cost                = resting_cost * resting_metabolism"
            f" = {self.rest_cost:.3f}",
        ]
        if self.move_cost is not None:
            lines.append(
                f"  move_cost                = movement_cost *"
                f" resting_metabolism = {self.move_cost:.3f}")
        lines += [
            f"  net_eat at u_X           = {self.net_eat:+.3f}"
            f"   (must be > 0)",
            f"  net_eat at h=1           = {self.net_eat_at_h1:+.3f}"
            f"   (must be > rest_cost)",
            f"  ceiling/Rest_X           = {self.headroom_ratio:.3f}",
            f"  feeding_cost/h(u_X)      = {self.required_ratio:.3f}"
            f"   (must be < ceiling/Rest_X)",
            f"  max allowed feeding_cost = {self.max_feeding_cost:.3f}",
        ]
        return "\n".join(lines)


def evaluate_energy_balance(fg_id, intake_ceiling, resting_metabolism,
                            maintenance_level, feeding_cost,
                            resting_cost=REFERENCE_RESTING_COST,
                            movement_cost=None, best_prey=None,
                            hunger_scale=None):
    """Evaluate the DM energy-balance gate at the maintenance level.

    Parameters
    ----------
    intake_ceiling
        Physiological intake ceiling in MJ/ton_pred/tick at full hunger,
        i.e. ``max_intake_rate * energy_content_of_best_prey *
        assimilation_factor``. This is the ``max_intake_mat *
        energy_gain_mat`` row maximum of the runtime caches.
    resting_metabolism, maintenance_level, feeding_cost, resting_cost,
    movement_cost
        The FG parameters, as stored in ``fg_library.yaml``.

    Blocking gates (``failures``)
        G1  ``feeding_cost >= resting_cost`` - hunting may not be cheaper
            than lying still; ``feeding_cost`` is an activity multiplier
            relative to rest.
        G2  ``h(u_X) > 0`` - with ``u_X >= hunger_scale`` the hunger gate
            closes completely at maintenance and intake is identically 0.
        G3  ``net_eat > 0`` at ``h(u_X)`` - the FG must be able to cover
            its own feeding metabolism at its maintenance level.
            Equivalent to ``ceiling/Rest_X > feeding_cost/h(u_X)``.
        G4  ``net_eat_at_h1 > rest_cost`` - the legacy Section 23 hard
            gate, kept so previously valid libraries stay valid.

    Non-blocking observations (``warnings``)
        W1  ``eat_minus_rest > 0`` at ``h(u_X)`` - eating should beat
            doing nothing at the maintenance level.
        W2  ``eat_minus_move > 0`` at ``h(u_X)`` - searching should be
            affordable when it leads to a full meal.
    """
    hunger_scale = resolve_satiation_scale(hunger_scale)
    intake_ceiling = float(intake_ceiling or 0.0)
    rm = float(resting_metabolism or 0.0)
    u_x = float(maintenance_level or 0.0)
    fc = float(feeding_cost or 0.0)
    cr = float(resting_cost or 0.0)
    mc = None if movement_cost in (None, "") else float(movement_cost)

    h_u = hunger_at(u_x, hunger_scale)
    realized = h_u * intake_ceiling
    feed_cost = fc * rm
    rest_cost = cr * rm
    move_cost = None if mc is None else mc * rm
    net_eat = realized - feed_cost
    net_eat_h1 = intake_ceiling - feed_cost
    eat_minus_rest = net_eat + rest_cost
    eat_minus_move = None if move_cost is None else net_eat + move_cost

    inf = float('inf')
    headroom = inf if rm <= 0.0 else intake_ceiling / rm
    required = inf if h_u <= 0.0 else fc / h_u
    max_fc = inf if rm <= 0.0 else h_u * intake_ceiling / rm

    res = EnergyBalance(
        fg_id=fg_id, best_prey=best_prey, intake_ceiling=intake_ceiling,
        resting_metabolism=rm, maintenance_level=u_x, feeding_cost=fc,
        resting_cost=cr, movement_cost=mc, hunger_scale=float(hunger_scale),
        hunger_at_maintenance=h_u, realized_intake=realized,
        feed_cost=feed_cost, rest_cost=rest_cost, move_cost=move_cost,
        net_eat=net_eat, net_eat_at_h1=net_eat_h1,
        eat_minus_rest=eat_minus_rest, eat_minus_move=eat_minus_move,
        headroom_ratio=headroom, required_ratio=required,
        max_feeding_cost=max_fc)

    if fc < cr:
        res.failures.append(
            f"feeding_cost ({fc:g}) is below resting_cost ({cr:g}): an "
            f"eating individual cannot burn less energy than a resting "
            f"one, since feeding_cost is the activity multiplier relative "
            f"to rest.")
    if h_u <= 0.0:
        res.failures.append(
            f"the hunger gate is fully closed at the maintenance level: "
            f"u_X = {u_x:g} >= {float(hunger_scale):g}, so h(u_X) = 0 and "
            f"realized intake is identically 0.")
    elif net_eat <= 0.0:
        res.failures.append(
            f"cannot break even at the maintenance level: realized intake "
            f"at h(u_X) = {realized:.3f} does not cover feed_cost = "
            f"{feed_cost:.3f} (net_eat = {net_eat:+.3f}). Requires "
            f"ceiling/Rest_X > feeding_cost/h(u_X), i.e. "
            f"{headroom:.3f} > {required:.3f}. Max allowed feeding_cost "
            f"is {max_fc:.3f}.")
    if net_eat_h1 <= rest_cost:
        res.failures.append(
            f"fails the legacy full-hunger gate: net_eat at h=1 = "
            f"{net_eat_h1:.3f} must exceed rest_cost = {rest_cost:.3f}.")

    if eat_minus_rest <= 0.0:
        res.warnings.append(
            f"eating is worse than resting at the maintenance level: "
            f"net_eat + rest_cost = {eat_minus_rest:+.3f}.")
    if eat_minus_move is not None and eat_minus_move <= 0.0:
        res.warnings.append(
            f"searching is unaffordable at the maintenance level: "
            f"net_eat + move_cost = {eat_minus_move:+.3f}.")
    return res
