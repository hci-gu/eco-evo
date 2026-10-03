"""Tick length: the one place that knows how long a tick is.

The engine itself is tick-agnostic - every biological parameter in
``fg_library.yaml`` is expressed *per tick* and the tick pipeline never
asks how many hours that is. The tick length is therefore purely an
*interpretive* constant: it is what converts a per-tick rate into a
real-world rate, and what lets a label say "per 6 h" instead of the
unitless "per tick".

Two consumers need it:

* Anything that converts ticks <-> real time for display or for a
  literature comparison (``tools/probes/budget_gate.py`` reporting
  % body mass per day, ``lib/diagnostics/viability.py`` turning years
  into ticks, ``tools/starve_calibration.py`` anchoring the starvation
  window).
* The FG editor in ``fgconfig/fgconfig.py``, which relabels its inputs
  and - on an explicit, confirmed action - rescales the library so the
  biology stays constant in real time when the tick length changes.

The value lives in ``project_metadata.tick_hours``. The library is
stamped with ``library_metadata.tick_hours`` recording the tick length
its numbers are calibrated at, so a project that disagrees with the
library can be detected instead of silently running miscalibrated
biology (``fg_library.yaml`` is shared between projects - rescaling it
for one project would otherwise corrupt every other).

Section 97.
"""

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

# The historical tick length. Every number in fg_library.yaml, every
# calibration in mareld_resume.txt and every "6 h/tick" in the docs
# refers to this. It stays the default so a project file that predates
# the field loads bit-identically.
DEFAULT_TICK_HOURS = 6

# What fg_library.yaml's numbers mean, always. The library has ONE
# calibration and the editor can no longer change it (section 106); a
# run at another tick length rescales in memory at load time and never
# writes back. Kept separate from DEFAULT_TICK_HOURS because they answer
# different questions - "what is the library calibrated at" and "what
# does a run use when not told otherwise" - even though both are 6.
LIBRARY_TICK_HOURS = 6

# One hour is the finest tick the engine's daily-scale rates stay
# meaningful at. Six hours - the calibration length everything in
# fg_library.yaml is expressed at - is the coarsest allowed: past a
# quarter day a tick stops resolving the within-day structure the
# decision makers are calibrated against, and the rescale would be
# carrying every parameter further from the numbers that were actually
# measured. Both bounds are enforced in the GUI and here; the GUI reads
# them from this module rather than declaring its own.
#
# The range was 1-168 h when the feature landed (section 97) and was
# narrowed on 2026-09-28 (section 103). Widening it again means raising
# MAX_TICK_HOURS and restoring the whole-days branch in ``tick_label``.
MIN_TICK_HOURS = 1
MAX_TICK_HOURS = 6


def resolve_tick_hours(value):
    """Per-project ``tick_hours`` with fallback to the module default.

    Empty / missing / unparseable / out-of-range values resolve to
    :data:`DEFAULT_TICK_HOURS`, so legacy project files that never
    declare the field behave exactly as before.
    """
    if value in (None, ""):
        return DEFAULT_TICK_HOURS
    try:
        hours = int(value)
    except (TypeError, ValueError):
        try:
            hours = int(float(value))
        except (TypeError, ValueError):
            return DEFAULT_TICK_HOURS
    if hours < MIN_TICK_HOURS or hours > MAX_TICK_HOURS:
        return DEFAULT_TICK_HOURS
    return hours


def ticks_per_day(tick_hours=None):
    """Ticks in a 24 h day. Fractional below a 24 h tick length."""
    return HOURS_PER_DAY / float(resolve_tick_hours(tick_hours))


def ticks_per_year(tick_hours=None):
    """Ticks in a 365-day year, rounded to a whole tick (>= 1)."""
    return max(1, round(DAYS_PER_YEAR * ticks_per_day(tick_hours)))


def tick_label(tick_hours=None):
    """Short unit fragment for a rate label: ``6 h``, ``1 h``.

    Always hours: every value ``resolve_tick_hours`` can return is now
    below a day. The whole-days rendering this used to carry ("7 d" for
    a 168 h tick) became unreachable when MAX_TICK_HOURS dropped to 6,
    and a branch no input can take reads as support that is not there.
    """
    return f"{resolve_tick_hours(tick_hours)} h"


def per_tick(unit, tick_hours=None):
    """Render a per-tick unit with the concrete tick length.

    ``per_tick("fraction")`` -> ``"fraction / 6 h"``. Used by every FG
    editor label so the inputs say what they actually mean.
    """
    return f"{unit} / {tick_label(tick_hours)}"


# ----------------------------------------------------------------------
# Rescaling
# ----------------------------------------------------------------------
# How each per-tick parameter converts when the tick length changes by a
# factor k = new_hours / old_hours. Three kinds, by dimension:
#
#   "flux"    A quantity transferred per tick (mass, energy, distance).
#             Genuinely accumulates within a tick, so it is linear:
#             x * k. Eating for twice as long moves twice the prey.
#
#             Being a fraction in [0, 1] does NOT make something a
#             "loss": movement_speed is bounded by 1 because the engine
#             cannot move biomass more than one cell per tick, not
#             because it is a survival probability. See its entry below.
#
#   "growth"  A per-tick fraction ADDED to a stock. Compounds:
#             (1 + r)^k - 1. Linear would overshoot badly at large k and
#             can drive `1 + rate * surplus` negative in
#             population_change._apply_decision_maker_population_change.
#
#   "loss"    A per-tick fraction REMOVED from a stock (a survival
#             probability in disguise). Compounds the other way:
#             1 - (1 - r)^k. Stays in [0, 1) for every k, which linear
#             does not.
#
#   "inverse" A quantity whose RECIPROCAL is the per-tick rate: x / k.
#             handling_time is the case: the Holling ceiling is 1/h ton
#             prey per ton predator per tick, so holding the real-time
#             ceiling fixed needs 1/h linear in the tick length, i.e. h
#             inverse in it. It also keeps the dimensionless product
#             a*h invariant, since a is a flux - which is the algebraic
#             check that the pair is classified consistently.
#
#   "period"  A duration measured in ticks: p / k, floored at 1.
#
# Anything not listed is tick-independent and is left alone: stocks
# (max_energy_reserve, energy_content, max_carrying_capacity,
# initial_biomass_*), dimensionless thresholds (maintenance_level,
# visibility_floor, extinction_threshold_factor,
# seasonal_amplitude), the action-cost multipliers (feeding_cost,
# resting_cost, movement_cost - they multiply resting_metabolism, which
# is itself rescaled), interference (1/ton) and min_split_biomass (kg).
RESCALE_RULES = {
    # flux
    "max_intake_rate": "flux",
    "resting_metabolism": "flux",
    "seed_rate": "flux",
    # Distance per tick, and the engine means it literally:
    # ``apply_movement`` computes ``b_out = moving_biomass * v`` and
    # keeps the rest in the source cell, so the moving cohort's centre
    # of mass advances v cells in one tick. Real-time speed is therefore
    # v / tick_hours, and holding it constant makes v linear in the tick
    # length. Section 97 filed it under "loss" because it is a fraction
    # in [0, 1]; that is a different thing, and the misfiling made the
    # rescale a no-op for every FG at v = 1.0 - which is four of the
    # eight in fg_library.yaml, the two fastest of them the ones the
    # user noticed. Section 104.
    "movement_speed": "flux",
    # growth
    "growth_rate": "growth",
    # loss
    "natural_mortality": "loss",
    "starve_rate": "loss",
    # period
    "seasonal_period": "period",
}

# Upper bound per parameter: what the ENGINE can represent. A rescale
# that would exceed it is clamped and reported, never silently
# truncated.
#
# These used to be described as "the FG editor's declared slider range".
# That is still true of every entry except movement_speed, whose editor
# range now follows the tick length (section 105) and is therefore
# tighter than this one at any tick below 6 h. The engine's ceiling is
# the tick-independent one: state.py clips speed to [0, 1] whatever the
# tick length is, because apply_movement moves biomass to the adjacent
# cell or not at all.
RESCALE_CLAMP = {
    "max_intake_rate": (0.0, 1.0),
    "resting_metabolism": (0.0, 1000.0),
    "seed_rate": (0.0, 1.0),
    "growth_rate": (0.0, 1.0),
    "natural_mortality": (0.0, 1.0),
    "starve_rate": (0.0, 1.0),
    "movement_speed": (0.0, 1.0),
    "seasonal_period": (0.0, 10000.0),
}


# Tick-dependent keys that do NOT live at the top level of a species
# entry. These were missed when the rescale was a rare manual action in
# the FG editor (section 97) and matter on every run now that it happens
# at load time (section 106).
INTERACTION_RESCALE_RULES = {
    # Per-(predator, prey) override of the species-level ceiling.
    "max_intake_rate": "flux",
    # Holling handling time; see the "inverse" note above.
    "handling_time": "inverse",
}

# Per-row keys inside an impact_table.
#
# biomass_factor is the fraction of standing biomass an impact removes
# per tick (``impacts.compute_impact_mortality``: contribution =
# biomass * bf), i.e. a loss rate.
#
# energy_factor is NOT here and must not be: it enters as
# ``1 + sum(energy_factor)`` multiplying the metabolic costs
# (``movement.apply_energy_costs``), exactly like feeding_cost and
# movement_cost, which 97.5 already lists as tick-independent because
# they multiply resting_metabolism - which is itself rescaled.
IMPACT_ROW_RESCALE_RULES = {
    "biomass_factor": "loss",
}


def rescale_value(key, value, old_hours, new_hours):
    """Convert one parameter from ``old_hours`` to ``new_hours`` ticks.

    Returns ``(new_value, clamped)``. ``clamped`` is True when the
    converted value hit the parameter's declared range and was cut, so
    the caller can report it rather than apply it silently.
    """
    rule = _rule_for(key)
    if rule is None:
        return value, False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value, False
    if x == 0.0:
        return value, False

    old_h = float(resolve_tick_hours(old_hours))
    new_h = float(resolve_tick_hours(new_hours))
    if old_h == new_h:
        return value, False
    k = new_h / old_h

    if rule == "flux":
        out = x * k
    elif rule == "growth":
        out = (1.0 + x) ** k - 1.0
    elif rule == "loss":
        # A rate at or above 1 already removes everything in one tick;
        # (1 - x) would go negative under a fractional power.
        out = 1.0 if x >= 1.0 else 1.0 - (1.0 - x) ** k
    elif rule in ("period", "inverse"):
        out = x / k
    else:
        return value, False

    lo, hi = RESCALE_CLAMP.get(key, (None, None))
    clamped = False
    if lo is not None and out < lo:
        out, clamped = lo, True
    if hi is not None and out > hi:
        out, clamped = hi, True
    if rule == "period" and out > 0.0:
        # A period is a whole number of ticks and must survive rounding.
        out = max(1.0, round(out))
    return out, clamped


def _rule_for(key):
    """The dimension rule for ``key`` wherever it appears."""
    if key in RESCALE_RULES:
        return RESCALE_RULES[key]
    if key in INTERACTION_RESCALE_RULES:
        return INTERACTION_RESCALE_RULES[key]
    return IMPACT_ROW_RESCALE_RULES.get(key)


def add_tick_length_argument(parser):
    """Register ``--tick-length`` on an argument parser.

    Shared by every entry point that can run at a tick length other than
    the library's, so they cannot drift apart on the flag name, the
    bounds or the default. Mirrors
    ``currents.add_current_arguments``. Section 106.
    """
    parser.add_argument(
        "--tick-length", "--tick_length", dest="tick_length", type=int,
        default=LIBRARY_TICK_HOURS,
        choices=list(range(MIN_TICK_HOURS, MAX_TICK_HOURS + 1)),
        help=(f"Hours per tick ({MIN_TICK_HOURS}-{MAX_TICK_HOURS}, default "
              f"{LIBRARY_TICK_HOURS}). fg_library.yaml is calibrated at "
              f"{LIBRARY_TICK_HOURS} h and is never rewritten: another "
              "length converts every tick-dependent parameter in memory "
              "as the run starts, so the biology stays the same in real "
              "time."))


def rescale_species(params, new_hours, old_hours=LIBRARY_TICK_HOURS):
    """Rescale one species entry, nested parameters included.

    ``params`` is what ``config_loader`` hands ``FunctionalGroup``: the
    library's species dict plus an ``interaction`` map and an ``impact``
    map built from the shared interaction definitions. Those nested
    dicts are the SAME objects the library holds and other species may
    reference, so every level that changes is copied first - rescaling
    in place would corrupt the library for the next species and for the
    next call.

    Returns ``(new_params, changes)`` with ``changes`` a list of
    ``(path, old, new, clamped)``; ``path`` is dotted, e.g.
    ``"interaction.porpoises_preys_on_gadoids.handling_time"``.

    A no-op that returns the input unchanged when the lengths agree,
    so the common case (a 6 h run against a 6 h library) costs nothing
    and is bit-identical.
    """
    if resolve_tick_hours(old_hours) == resolve_tick_hours(new_hours):
        return params, []

    out, changes = rescale_params(params, old_hours, new_hours)
    if out is params:
        out = dict(params)
    changes = [(k, a, b, c) for (k, a, b, c) in changes]

    interaction = params.get("interaction")
    if isinstance(interaction, dict):
        new_inter = dict(interaction)
        for iid, idef in interaction.items():
            if not isinstance(idef, dict):
                continue
            row = None
            for key in INTERACTION_RESCALE_RULES:
                if key not in idef:
                    continue
                value, clamped = rescale_value(key, idef[key], old_hours, new_hours)
                if value == idef[key]:
                    continue
                if row is None:
                    row = dict(idef)
                row[key] = value
                changes.append(
                    (f"interaction.{iid}.{key}", idef[key], value, clamped))
            if row is not None:
                new_inter[iid] = row
        out["interaction"] = new_inter

    impact = params.get("impact")
    if isinstance(impact, dict):
        new_impact = dict(impact)
        for iid, idef in impact.items():
            if not isinstance(idef, dict):
                continue
            table = idef.get("impact_table")
            if not isinstance(table, list):
                continue
            new_table = list(table)
            touched = False
            for n, entry in enumerate(table):
                if not isinstance(entry, dict):
                    continue
                row = None
                for key in IMPACT_ROW_RESCALE_RULES:
                    if key not in entry:
                        continue
                    value, clamped = rescale_value(
                        key, entry[key], old_hours, new_hours)
                    if value == entry[key]:
                        continue
                    if row is None:
                        row = dict(entry)
                    row[key] = value
                    changes.append(
                        (f"impact.{iid}.impact_table[{n}].{key}",
                         entry[key], value, clamped))
                if row is not None:
                    new_table[n] = row
                    touched = True
            if touched:
                new_def = dict(idef)
                new_def["impact_table"] = new_table
                new_impact[iid] = new_def
        out["impact"] = new_impact

    return out, changes


def rescale_params(params, old_hours, new_hours):
    """Rescale every tick-dependent key in one FG's parameter dict.

    Returns ``(new_params, changes)`` where ``changes`` is a list of
    ``(key, old_value, new_value, clamped)`` for the keys that moved.
    ``params`` is not mutated.
    """
    out = dict(params)
    changes = []
    for key in RESCALE_RULES:
        if key not in params:
            continue
        old_value = params[key]
        new_value, clamped = rescale_value(key, old_value, old_hours, new_hours)
        if new_value == old_value:
            continue
        out[key] = new_value
        changes.append((key, old_value, new_value, clamped))
    return out, changes
