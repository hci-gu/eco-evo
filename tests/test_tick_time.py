"""Tests for the configurable tick length (Section 97).

Two properties carry the whole feature:

* A project that never declares ``tick_hours`` behaves exactly as it did
  when the tick length was hardcoded at 6 h.
* Rescaling from A to B and back to A is the identity, and rescaling
  preserves the real-time biology the parameter encodes.
"""
import math

import pytest
import yaml

from lib.world.tick_time import (
    DEFAULT_TICK_HOURS,
    MAX_TICK_HOURS,
    MIN_TICK_HOURS,
    RESCALE_RULES,
    per_tick,
    rescale_params,
    rescale_value,
    resolve_tick_hours,
    tick_label,
    ticks_per_day,
    ticks_per_year,
)


# ----------------------------------------------------------------------
# 1. Defaults and resolution
# ----------------------------------------------------------------------

def test_default_is_the_historical_six_hour_tick():
    assert DEFAULT_TICK_HOURS == 6
    assert ticks_per_day(DEFAULT_TICK_HOURS) == 4.0
    # The constant lib/diagnostics/viability.py used to hardcode.
    assert ticks_per_year(DEFAULT_TICK_HOURS) == 1460


@pytest.mark.parametrize("bad", [None, "", "abc", 0, -1, 7, 1000, float("nan")])
def test_missing_or_out_of_range_resolves_to_the_default(bad):
    assert resolve_tick_hours(bad) == DEFAULT_TICK_HOURS


@pytest.mark.parametrize("value,expected", [(1, 1), ("3", 3), (4.0, 4), (6, 6)])
def test_valid_values_resolve_to_themselves(value, expected):
    assert resolve_tick_hours(value) == expected


def test_bounds_are_inclusive():
    assert resolve_tick_hours(MIN_TICK_HOURS) == MIN_TICK_HOURS
    assert resolve_tick_hours(MAX_TICK_HOURS) == MAX_TICK_HOURS
    assert resolve_tick_hours(MIN_TICK_HOURS - 1) == DEFAULT_TICK_HOURS
    assert resolve_tick_hours(MAX_TICK_HOURS + 1) == DEFAULT_TICK_HOURS


@pytest.mark.parametrize("hours", [7, 8, 12, 18, 24, 48, 168])
def test_the_range_is_one_to_six_hours(hours):
    """Section 103 narrowed it from 1-168.

    Six hours is the length fg_library.yaml is calibrated at, and the
    values above it were the ones whose rescale carried every parameter
    furthest from the numbers that were measured. They are not rejected
    loudly - ``resolve_tick_hours`` has always fallen back to the
    default for anything out of range - so a project file that still
    carries one now runs at 6 h.
    """
    assert (MIN_TICK_HOURS, MAX_TICK_HOURS) == (1, 6)
    assert resolve_tick_hours(hours) == DEFAULT_TICK_HOURS


# ----------------------------------------------------------------------
# 2. Labels
# ----------------------------------------------------------------------

@pytest.mark.parametrize("hours,expected", [
    (1, "1 h"), (2, "2 h"), (5, "5 h"), (6, "6 h"),
    # Out of range resolves to the default first, so the label follows.
    (24, "6 h"), (168, "6 h"), (None, "6 h"),
])
def test_tick_label(hours, expected):
    assert tick_label(hours) == expected


def test_per_tick_renders_the_unit_with_the_duration():
    assert per_tick("fraction", 6) == "fraction / 6 h"
    assert per_tick("cells", 1) == "cells / 1 h"


# ----------------------------------------------------------------------
# 3. Rescaling: identity, round trip, dimension
# ----------------------------------------------------------------------

def test_no_change_is_the_identity():
    for key in RESCALE_RULES:
        value, clamped = rescale_value(key, 0.05, 6, 6)
        assert value == 0.05 and not clamped


def test_unknown_keys_are_left_alone():
    # Stocks and dimensionless thresholds must never be touched.
    for key in ("max_energy_reserve", "energy_content", "maintenance_level",
                "interference", "min_split_biomass",
                "extinction_threshold_factor", "seasonal_amplitude",
                "feeding_cost", "resting_cost", "movement_cost",
                "max_carrying_capacity", "visibility_floor"):
        assert key not in RESCALE_RULES
        assert rescale_value(key, 0.5, 6, 3) == (0.5, False)


@pytest.mark.parametrize("key", sorted(RESCALE_RULES))
@pytest.mark.parametrize("a,b", [(6, 1), (1, 6), (6, 3), (3, 6), (2, 5)])
def test_round_trip_returns_the_original(key, a, b):
    # A period is a count of ticks, not a fraction, so it needs a
    # representative magnitude of its own: 1460 ticks is one year at the
    # 6 h tick, the scale seasonal_period actually holds.
    original = 1460.0 if RESCALE_RULES[key] == "period" else 0.02
    there, clamped_there = rescale_value(key, original, a, b)
    back, clamped_back = rescale_value(key, there, b, a)
    if clamped_there or clamped_back:
        pytest.skip("clamped; round trip is not defined through a clamp")
    if RESCALE_RULES[key] == "period":
        # Quantisation is inherent: a duration that is a whole number of
        # A-ticks need not be a whole number of B-ticks (1460 two-hour
        # ticks are 584 five-hour ticks). The round trip is therefore
        # exact to within one tick of the coarser of the two.
        assert abs(back - original) <= max(1.0, b / a)
    else:
        assert back == pytest.approx(original, rel=1e-9)


def test_flux_is_linear():
    # Eating for twice as long moves twice the prey.
    assert rescale_value("max_intake_rate", 0.035, 3, 6)[0] == pytest.approx(0.07)
    assert rescale_value("resting_metabolism", 90.0, 6, 3)[0] == pytest.approx(45.0)
    assert rescale_value("seed_rate", 0.001, 2, 6)[0] == pytest.approx(0.003)
    # Swimming for a third of the time covers a third of the distance.
    assert rescale_value("movement_speed", 0.9, 6, 2)[0] == pytest.approx(0.3)


def test_movement_speed_is_a_distance_not_a_survival_probability():
    """Section 104. It was filed under "loss" and did not rescale.

    ``apply_movement`` computes ``b_out = moving_biomass * v`` and keeps
    the rest in the source cell, so the moving cohort advances v cells
    in one tick: v is cells/tick, exactly what the FG editor's label
    says, and the real-time speed is v / tick_hours.

    Under the old "loss" rule the value compounded as a survival
    probability, and its ``x >= 1.0`` short-circuit returned 1.0
    unchanged - so pelagic_fish, porpoises, seals and seabirds, all at
    1.0 in fg_library.yaml, kept moving one whole cell per tick at every
    tick length. Going 6 h -> 1 h that is a six-fold speed-up, applied
    silently by the one operation whose entire purpose is to hold the
    biology fixed in real time.
    """
    assert RESCALE_RULES["movement_speed"] == "flux"
    # The four FGs that did not move at all under the old rule.
    assert rescale_value("movement_speed", 1.0, 6, 1)[0] == pytest.approx(1.0 / 6.0)
    # ...and one that did, but too fast: the loss rule gave 0.0468.
    assert rescale_value("movement_speed", 0.25, 6, 1)[0] == pytest.approx(0.25 / 6.0)


@pytest.mark.parametrize("v6", [1.0, 0.25, 0.02])
def test_movement_speed_over_a_fixed_real_duration_is_invariant(v6):
    """Cells per hour is what must not change."""
    baseline = v6 / 6.0
    for new_hours in (1, 2, 3, 4, 5, 6):
        v_new, clamped = rescale_value("movement_speed", v6, 6, new_hours)
        assert not clamped
        assert v_new / new_hours == pytest.approx(baseline, rel=1e-9)


@pytest.mark.parametrize("hours", [1, 2, 3, 4, 5, 6])
def test_no_library_speed_is_clamped_by_the_engine_ceiling(hours):
    """A shorter tick must never lose a speed to the [0, 1] cap.

    apply_movement moves biomass to the adjacent cell or not at all, so
    the engine cannot represent more than one cell per tick and
    RESCALE_CLAMP holds 1.0 at every tick length. Converting DOWN from
    the 6 h calibration only ever shrinks a speed, so nothing clamps -
    and fg_library.yaml has four FGs sitting exactly at 1.0, which is
    where a sign error in the rule would show up first.

    Section 105 gave the FG editor a matching tick-dependent input
    range; section 106 removed the editor's tick length entirely, so
    this is now purely the runtime property.
    """
    ceiling, clamped = rescale_value("movement_speed", 1.0, 6, hours)
    assert not clamped
    assert ceiling == pytest.approx(hours / 6.0)
    for v6 in (1.0, 0.25, 0.02, 0.0):
        rescaled, was_clamped = rescale_value("movement_speed", v6, 6, hours)
        assert not was_clamped
        assert rescaled <= ceiling + 1e-12


def test_movement_speed_clamps_when_the_tick_cannot_carry_it():
    """A cell per tick is the engine's ceiling, and it is reported.

    Coarsening is where linear scaling meets the cap: a fish crossing
    a quarter cell per hour would cross one and a half in six, which
    ``apply_movement`` cannot express - it moves biomass to the
    ADJACENT cell or not at all. The clamp is the honest answer and the
    confirmation dialog lists it; the old rule instead saturated
    smoothly to a value that was simply wrong.
    """
    value, clamped = rescale_value("movement_speed", 0.25, 1, 6)
    assert clamped is True
    assert value == 1.0


def test_loss_compounds_and_stays_below_one():
    # Survival over a 6 h tick equals survival over two 3 h ticks.
    r3 = 0.02
    r6, _ = rescale_value("natural_mortality", r3, 3, 6)
    assert (1 - r6) == pytest.approx((1 - r3) ** 2)
    # Even the widest stretch the range allows (1 h -> 6 h) cannot
    # produce a rate at or above 1.
    r_wide, clamped = rescale_value("natural_mortality", 0.5, 1, 6)
    assert 0.0 < r_wide < 1.0 and not clamped


def test_loss_rate_of_one_stays_one():
    # Already lethal in a single tick; a longer tick cannot be worse.
    assert rescale_value("natural_mortality", 1.0, 3, 6) == (1.0, False)


def test_growth_compounds():
    r3 = 0.05
    r6, _ = rescale_value("growth_rate", r3, 3, 6)
    assert (1 + r6) == pytest.approx((1 + r3) ** 2)


def test_period_divides_and_stays_a_whole_tick():
    # A 1460-tick (one year at 6 h) season becomes 2920 ticks at 3 h.
    assert rescale_value("seasonal_period", 1460, 6, 3)[0] == 2920
    # Never rounds down to zero.
    value, _ = rescale_value("seasonal_period", 3, 1, 6)
    assert value >= 1


def test_clamp_is_reported_not_silent():
    # Zooplankton intake (0.25 / 1 h) cannot be 6x that in a 6 h tick.
    value, clamped = rescale_value("max_intake_rate", 0.25, 1, 6)
    assert clamped is True
    assert value == 1.0


def test_rescale_params_does_not_mutate_and_reports_changes():
    params = {
        "natural_mortality": 5e-05,
        "max_intake_rate": 0.035,
        "max_energy_reserve": 2800.0,   # untouched
        "visibility_floor": 0.35,       # untouched
    }
    snapshot = dict(params)
    out, changes = rescale_params(params, 6, 3)
    assert params == snapshot, "input dict must not be mutated"
    assert out["max_energy_reserve"] == 2800.0
    assert out["visibility_floor"] == 0.35
    assert {c[0] for c in changes} == {"natural_mortality", "max_intake_rate"}


def test_zero_values_stay_zero():
    # movement_speed 0 (phytoplankton) must not become nonzero.
    for key in RESCALE_RULES:
        assert rescale_value(key, 0.0, 6, 1) == (0.0, False)


# ----------------------------------------------------------------------
# 4. Real-time invariance: the point of rescaling
# ----------------------------------------------------------------------

def test_mortality_survival_is_invariant_over_a_fixed_real_duration():
    """One week of survival is the same whatever the tick length."""
    r6 = 2e-4
    week_ticks_6h = 7 * 4
    baseline = (1 - r6) ** week_ticks_6h
    for new_hours in (1, 2, 3, 4, 5, 6):
        r_new, _ = rescale_value("natural_mortality", r6, 6, new_hours)
        week_ticks = 7 * 24 / new_hours
        assert (1 - r_new) ** week_ticks == pytest.approx(baseline, rel=1e-9)


def test_intake_over_a_fixed_real_duration_is_invariant():
    a6 = 0.035
    baseline = a6 * (24 / 6)  # tonnes prey per tonne predator per day
    for new_hours in (1, 2, 3, 4, 5, 6):
        a_new, clamped = rescale_value("max_intake_rate", a6, 6, new_hours)
        assert not clamped
        assert a_new * (24 / new_hours) == pytest.approx(baseline, rel=1e-9)


# ----------------------------------------------------------------------
# 6. The live project still resolves to the historical tick
# ----------------------------------------------------------------------

def test_viability_horizon_follows_the_tick_length():
    from lib.diagnostics.viability import TICKS_PER_YEAR, ViabilityCriterion
    assert TICKS_PER_YEAR == 1460
    assert ViabilityCriterion(years=5.0).ticks == 7300
    assert ViabilityCriterion(years=5.0, tick_hours=3).ticks == 14600
    assert ViabilityCriterion(years=1.0, tick_hours=1).ticks == 8760
