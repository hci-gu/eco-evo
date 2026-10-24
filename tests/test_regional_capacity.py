"""Regional carrying capacity of a decision maker (section 185).

Contracts asserted:

1. ``regional_carrying_capacity`` parses: empty or 0 is off, otherwise a
   positive number; ``capacity_loss_rate`` = r_max / (1460 K) and needs
   r_max.
2. The loss is outside the derived growth rate: K changes no budget term.
3. The loader derives ``capacity_loss_rate`` (the library's benthivorous_fish,
   pelagic fish and seals carry it) and rescales it as a flux for another tick length.
4. In the population step a DM with it loses the fraction c * B_region
   (grid-mean density) of its biomass and reserve, booked as a natural
   loss, with or without --mortality; another DM is untouched.
5. Refusals: a non decision maker (loader and env), K without r_max.

GPU parity: tests/test_gpu_ecosystem.py::test_regional_capacity_matches_reference.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config  # noqa: E402
from lib.environments.ecosystem_env import population_change  # noqa: E402
from lib.environments.ecosystem_env.environment import EcosystemEnvironment  # noqa: E402
from lib.world import growth_budget as gb  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402
from lib.world.tick_time import rescale_species  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"


@pytest.mark.parametrize("spec,expected", [
    ({}, None), ({"regional_carrying_capacity": ""}, None),
    ({"regional_carrying_capacity": 0}, None),
    ({"regional_carrying_capacity": 1.02}, 1.02),
    ({"regional_carrying_capacity": "2"}, 2.0),
])
def test_capacity_parses(spec, expected):
    assert gb.regional_capacity(spec) == expected


@pytest.mark.parametrize("spec,match", [
    ({"regional_carrying_capacity": -1}, ">= 0"),
    ({"regional_carrying_capacity": "x"}, "number"),
    ({"regional_carrying_capacity": float("inf")}, ">= 0"),
    ({"regional_carrying_capacity": 1.0}, "needs r_max"),
    ({"regional_carrying_capacity": 1.0, "r_max": 0}, "needs r_max"),
])
def test_invalid_capacity_raises(spec, match):
    with pytest.raises(ValueError, match=match):
        gb.capacity_loss_rate(spec)


def test_loss_rate_is_r_max_over_ticks_and_k():
    rate = gb.capacity_loss_rate({"regional_carrying_capacity": 1.02,
                                  "r_max": 0.37})
    assert rate == pytest.approx(0.37 / (1460.0 * 1.02))
    assert gb.capacity_loss_rate({"r_max": 0.37}) is None


def _library():
    return yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))


def test_capacity_is_outside_the_growth_budget():
    lib = _library()
    species, inter = lib["species_definitions"], lib["interaction_definitions"]
    base = {k: v for k, v in species["benthivorous_fish"].items()
            if k != "regional_carrying_capacity"}
    plain = gb.growth_budget("benthivorous_fish", species, inter, species=base)
    capped = gb.growth_budget("benthivorous_fish", species, inter, species=dict(
        base, regional_carrying_capacity=1.02))
    assert capped.growth_rate == plain.growth_rate
    assert capped.describe() == plain.describe()


CAPPED = ("benthivorous_fish", "predatory_fish", "planktivorous_fish", "seals")   # sections 185, 185.20, 199, 209


def test_loader_derives_the_rate_for_the_library_groups():
    lib = _library()
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    fgs = load_project_config(str(PROJECT), library_config=lib,
                              project_config=project, grid_size=(6, 6),
                              seed=1)[0]
    for fid in CAPPED:
        spec = lib["species_definitions"][fid]
        assert fgs[fid].params["capacity_loss_rate"] == pytest.approx(
            gb.capacity_loss_rate(spec)), fid
    for fid, fg in fgs.items():
        if fid not in CAPPED:
            assert "capacity_loss_rate" not in fg.params, fid


def test_the_rate_rescales_as_a_flux():
    params = {"capacity_loss_rate": 2.4e-4}
    out, _changes = rescale_species(params, 3)
    assert out["capacity_loss_rate"] == pytest.approx(1.2e-4)


def _env(rate=0.05, mortality=False, extra=None):
    """DMs ``p`` (with the loss unless ``rate`` is None) and ``o``."""
    groups = {}
    for fid in ("p", "o"):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=100.0, resting_metabolism=1.0,
                      growth_rate=0.2, maintenance_level=0.3,
                      starve_rate=0.05, max_carrying_capacity=1e6,
                      energy_content=5.0, natural_mortality=0.01)
        if fid == "p" and rate is not None:
            params["capacity_loss_rate"] = rate
        params.update((extra or {}).get(fid, {}))
        groups[fid] = FunctionalGroup(fid, params)
        groups[fid].initialize_state((1, 2), initial_biomass=np.array(
            [[1.0, 3.0]]))
        # Reserve at the maintenance line: no growth, no starvation.
        groups[fid].energy_reserve[:] = (np.float32(0.3 * 100.0)
                                         * groups[fid].biomass)
    return EcosystemEnvironment(dict(height=1, width=2), groups,
                                apply_natural_mortality=mortality)


def _step(env, fid):
    fg = env.fgs[fid]
    biomass, reserve = fg.biomass.copy(), fg.energy_reserve.copy()
    population_change._apply_decision_maker_population_change(env, fid, fg)
    return biomass, reserve, fg.biomass.copy(), fg.energy_reserve.copy()


@pytest.mark.parametrize("mortality", [False, True])
def test_population_step_removes_c_times_the_grid_mean(mortality):
    env = _env(rate=0.05, mortality=mortality)
    assert env._has_regional_capacity
    b0, r0, b1, r1 = _step(env, "p")
    after_m1 = b0 * (1.0 - 0.01) if mortality else b0
    keep = 1.0 - 0.05 * after_m1.mean()      # grid mean 2 t/cell -> 10 %
    np.testing.assert_allclose(b1, after_m1 * keep, rtol=1e-5)
    np.testing.assert_allclose(r1 / b1, r0 / b0, rtol=1e-5)
    booked = env.loss_natural["p"]
    assert booked == pytest.approx(float(b0.sum() - b1.sum()), rel=1e-5)
    o0, _r, o1, _r1 = _step(env, "o")
    expected_o = o0 * (1.0 - 0.01) if mortality else o0
    np.testing.assert_allclose(o1, expected_o, rtol=1e-6)


def test_no_capacity_is_bit_identical():
    plain = _env(rate=None)
    assert not plain._has_regional_capacity
    b0, _r0, b1, _r1 = _step(plain, "p")
    np.testing.assert_array_equal(b0, b1)


def test_refusals():
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        _env(rate=1.5)
    groups = {"n": FunctionalGroup("n", dict(
        is_decision_maker=False, movement_speed=0.0, growth_rate=0.1,
        max_carrying_capacity=10.0, energy_content=1.0,
        capacity_loss_rate=1e-4))}
    groups["n"].initialize_state((1, 1), initial_biomass=np.ones((1, 1)))
    with pytest.raises(ValueError, match="not a decision maker"):
        EcosystemEnvironment(dict(height=1, width=1), groups)
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    for sid, extra, match in (
            ("benthic_community", {"r_max": 0.5}, "not a decision"),
            ("planktivorous_fish", {"r_max": 0}, "needs r_max")):
        lib = _library()
        lib["species_definitions"][sid].update(
            regional_carrying_capacity=1.0, **extra)
        with pytest.raises(ValueError, match=match):
            load_project_config(str(PROJECT), library_config=lib,
                                project_config=project, grid_size=(6, 6),
                                seed=1)
