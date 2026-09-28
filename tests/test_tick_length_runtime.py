"""``--tick-length``: the library stays 6 h, the run converts (section 106).

fg_library.yaml has ONE calibration, LIBRARY_TICK_HOURS, and the FG
editor can no longer change it. A run at another tick length converts
every tick-dependent parameter in memory as the functional groups are
built, and never writes back.

What that moves is the correctness bar. As a rare manual action in the
editor the conversion could be incomplete and mostly not hurt; running
on every non-6 h rollout, a missed parameter is a silently different
biology. So the tests here are about the REAL-TIME quantities, not the
per-tick numbers: what a predator can eat in a day, what an impact
kills in a week, how far a fish swims in an hour.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config
from lib.world.tick_time import (
    LIBRARY_TICK_HOURS,
    MAX_TICK_HOURS,
    MIN_TICK_HOURS,
    add_tick_length_argument,
    rescale_species,
)

HOURS = [1, 2, 3, 4, 5, 6]
LIBRARY = Path(__file__).resolve().parents[1] / "fgconfig" / "fg_library.yaml"


def library():
    with open(LIBRARY, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


def porpoise_params():
    lib = library()
    params = copy.deepcopy(lib["species_definitions"]["porpoises"])
    params["interaction"] = copy.deepcopy({
        k: v for k, v in lib["interaction_definitions"].items()
        if k.startswith("porpoises_preys_on_")})
    return params


# ---------------------------------------------------------------- 1
# The nested parameters section 97 never reached


@pytest.mark.parametrize("hours", HOURS)
def test_the_holling_ceiling_is_invariant_in_real_time(hours):
    """1/h is a per-tick intake ceiling, so h must follow the tick.

    ``holling_a_eff``: f(B) = a*B / (1 + a*h*B), bounded by 1/h in ton
    prey per ton predator per TICK. handling_time lives in the
    interaction definitions, which ``rescale_params`` never walked.
    """
    params = porpoise_params()
    out, _ = rescale_species(params, hours)
    checked = 0
    for iid, idef in out["interaction"].items():
        h0 = float(params["interaction"][iid].get("handling_time", 0.0) or 0.0)
        if h0 == 0.0:
            # h = 0 is the unsaturated branch: no ceiling to hold fixed.
            continue
        per_day_0 = (1.0 / h0) * (24.0 / LIBRARY_TICK_HOURS)
        per_day_1 = (1.0 / idef["handling_time"]) * (24.0 / hours)
        assert per_day_1 == pytest.approx(per_day_0, rel=1e-9)
        checked += 1
    assert checked >= 2, f"only {checked} saturating pairs; test is thin"


@pytest.mark.parametrize("hours", HOURS)
def test_the_dimensionless_holling_product_is_invariant(hours):
    """a*h carries no unit, so it must come out untouched.

    This is the algebraic check that the pair is classified
    consistently: a is a flux (x*k) and h is its inverse (x/k), so the
    product is fixed. A rule that scaled only one of them would show up
    here even if each looked plausible on its own.
    """
    params = porpoise_params()
    out, _ = rescale_species(params, hours)
    for iid, idef in out["interaction"].items():
        a0 = params["interaction"][iid].get(
            "max_intake_rate", params["max_intake_rate"])
        h0 = float(params["interaction"][iid].get("handling_time", 0.0) or 0.0)
        a1 = idef.get("max_intake_rate", out["max_intake_rate"])
        assert a1 * idef["handling_time"] == pytest.approx(a0 * h0, rel=1e-9)


def test_the_per_pair_intake_override_is_rescaled():
    """The library really has one, so this is not hypothetical."""
    params = porpoise_params()
    pair = "porpoises_preys_on_gadoids"
    assert "max_intake_rate" in params["interaction"][pair], (
        "fixture no longer covers a per-pair override")
    out, _ = rescale_species(params, 1)
    assert out["interaction"][pair]["max_intake_rate"] == pytest.approx(
        params["interaction"][pair]["max_intake_rate"] / 6.0)


@pytest.mark.parametrize("hours", HOURS)
def test_impact_mortality_over_a_week_is_invariant(hours):
    """biomass_factor is a per-tick fraction removed: a loss rate."""
    params = {"impact": {"noise": {"impact_table": [
        {"value": 0.0, "biomass_factor": 0.0, "energy_factor": 0.0},
        {"value": 50.0, "biomass_factor": 0.3, "energy_factor": 0.4},
    ]}}}
    out, _ = rescale_species(params, hours)
    row = out["impact"]["noise"]["impact_table"][1]
    week_ticks = 7 * 24 / hours
    baseline = (1 - 0.3) ** (7 * 24 / LIBRARY_TICK_HOURS)
    assert (1 - row["biomass_factor"]) ** week_ticks == pytest.approx(
        baseline, rel=1e-9)


def test_energy_factor_is_left_alone():
    """It multiplies the costs, like feeding_cost - dimensionless.

    ``apply_energy_costs`` uses ``1 + sum(energy_factor)`` as a factor on
    resting/feeding/movement metabolism, and resting_metabolism is
    itself rescaled. Scaling the multiplier too would apply the tick
    length twice.
    """
    params = {"impact": {"noise": {"impact_table": [
        {"value": 50.0, "biomass_factor": 0.3, "energy_factor": 0.4}]}}}
    out, _ = rescale_species(params, 1)
    assert out["impact"]["noise"]["impact_table"][0]["energy_factor"] == 0.4


def test_the_library_dicts_are_never_mutated():
    """The nested dicts are shared with the library and other species."""
    params = porpoise_params()
    snapshot = copy.deepcopy(params)
    rescale_species(params, 1)
    assert params == snapshot


def test_the_calibration_tick_is_a_no_op():
    """A 6 h run must be bit-identical to not converting at all."""
    params = porpoise_params()
    out, changes = rescale_species(params, LIBRARY_TICK_HOURS)
    assert out is params and changes == []


# ---------------------------------------------------------------- 2
# The loader applies it


@pytest.mark.parametrize("hours", [1, 3, 6])
def test_the_loader_converts_the_groups(hours):
    fgs = load_project_config("mareld2.yaml", grid_size=(8, 8), seed=1,
                              tick_hours=hours)[0]
    lib = library()["species_definitions"]
    for fid, fg in fgs.items():
        expected = float(lib[fid].get("movement_speed", 0.0)) * hours / 6.0
        assert fg.speed == pytest.approx(expected), fid


def test_the_default_leaves_the_library_values_alone():
    fgs = load_project_config("mareld2.yaml", grid_size=(8, 8), seed=1)[0]
    lib = library()["species_definitions"]
    for fid, fg in fgs.items():
        assert fg.speed == pytest.approx(float(lib[fid].get("movement_speed", 0.0)))


def test_the_library_file_is_never_written():
    before = LIBRARY.read_bytes()
    load_project_config("mareld2.yaml", grid_size=(8, 8), seed=1, tick_hours=1)
    assert LIBRARY.read_bytes() == before


# ---------------------------------------------------------------- 3
# The flag


def test_the_flag_bounds_and_default():
    parser = argparse.ArgumentParser()
    add_tick_length_argument(parser)
    assert parser.parse_args([]).tick_length == LIBRARY_TICK_HOURS
    for h in HOURS:
        assert parser.parse_args(["--tick-length", str(h)]).tick_length == h
        assert parser.parse_args(["--tick_length", str(h)]).tick_length == h
    for bad in (str(MIN_TICK_HOURS - 1), str(MAX_TICK_HOURS + 1), "12", "168"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--tick-length", bad])


@pytest.mark.parametrize("module", ["train.py", "inference.py",
                                    "lib/gpu/cli.py", "tools/viability.py"])
def test_every_entry_point_registers_the_flag(module):
    """A run that cannot be told the tick length silently runs at 6."""
    source = (Path(__file__).resolve().parents[1] / module).read_text(
        encoding="utf-8")
    assert "add_tick_length_argument(parser)" in source, module


def test_the_project_file_no_longer_carries_a_tick_length():
    """One source for the number; the flag is it (section 106)."""
    root = Path(__file__).resolve().parents[1]
    project = yaml.safe_load((root / "mareld2.yaml").read_text(encoding="utf-8"))
    assert "tick_hours" not in (project.get("project_metadata") or {})
    assert "tick_hours" not in (library().get("library_metadata") or {})
