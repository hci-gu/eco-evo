"""Income-funded growth of a decision maker (section 184).

Contracts asserted:

1. ``income_growth_kappa`` / ``feeding_season_months`` parse: kappa 0 or
   empty is off, otherwise in (0, 1]; the season (default 12) lies in
   (0, 12] and needs a kappa.
2. The budget pays g over the feeding season: g = the plain derived g x
   12 / months, s* unchanged, and the FG editor's formula shows it.
3. Per cell the growth is G = min(kappa I - C, B g (s - u) ec, the
   settling reserve), moved from the reserve into G / ec new tonnes;
   with no intake there is no growth, whatever the reserve (the fat is
   kept).
4. In the tick: the population step never turns such a DM's reserve
   into biomass; its starvation keeps the library rate; another DM is
   untouched.
5. Refusals: a non decision maker, growth_q10 on the same DM, no
   starve_rate (loader and env).

GPU parity: tests/test_gpu_ecosystem.py::test_income_growth_matches_reference.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_project_config  # noqa: E402
from lib.environments.ecosystem_env import movement, population_change  # noqa: E402
from lib.environments.ecosystem_env.environment import EcosystemEnvironment  # noqa: E402
from lib.world import growth_budget as gb  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "mareld2.yaml"
LIBRARY = ROOT / "fgconfig" / "fg_library.yaml"


@pytest.mark.parametrize("spec,expected", [
    ({}, None), ({"income_growth_kappa": ""}, None),
    ({"income_growth_kappa": 0, "feeding_season_months": 0}, None),
    ({"income_growth_kappa": 0.75}, (0.75, 12.0)),
    ({"income_growth_kappa": 1, "feeding_season_months": 5.5}, (1.0, 5.5)),
])
def test_income_growth_parses(spec, expected):
    assert gb.income_growth(spec) == expected


@pytest.mark.parametrize("spec,match", [
    ({"income_growth_kappa": 1.2}, r"\(0, 1\]"),
    ({"income_growth_kappa": -0.1}, r"\(0, 1\]"),
    ({"income_growth_kappa": "x"}, "number"),
    ({"feeding_season_months": 5.5}, "needs income_growth_kappa"),
    ({"income_growth_kappa": 0.7, "feeding_season_months": 13}, r"\(0, 12\]"),
])
def test_invalid_income_growth_raises(spec, match):
    with pytest.raises(ValueError, match=match):
        gb.income_growth(spec)


def _library():
    return yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))


def test_budget_is_paid_over_the_feeding_season():
    lib = _library()
    species, inter = lib["species_definitions"], lib["interaction_definitions"]
    base = {k: v for k, v in species["pelagic_fish"].items()
            if k not in ("income_growth_kappa", "feeding_season_months")}
    plain = gb.growth_budget("pelagic_fish", species, inter, species=base)
    spec = dict(base, income_growth_kappa=0.75, feeding_season_months=5.5)
    seasonal = gb.growth_budget("pelagic_fish", species, inter, species=spec)
    assert seasonal.growth_rate == pytest.approx(plain.growth_rate * 12 / 5.5)
    assert seasonal.reference_fill == pytest.approx(plain.reference_fill)
    assert seasonal.per_tick(1.0) == pytest.approx(plain.per_tick(1.0) * 12 / 5.5)
    assert "12/5.5" in seasonal.describe() and "12/" not in plain.describe()
    typing = gb.growth_budget("pelagic_fish", species, inter, species=dict(
        base, income_growth_kappa="0.7x"))
    assert typing.error and "number" in typing.error
    whole_year = gb.growth_budget("pelagic_fish", species, inter, species=dict(
        base, income_growth_kappa=0.75))
    assert whole_year.growth_rate == pytest.approx(plain.growth_rate)


def _env(kappa=0.75, reserve=0.8, starve_rate=0.05, extra=None):
    """DMs ``p`` (income-funded unless ``kappa`` is None) and ``o``."""
    groups = {}
    for fid in ("p", "o"):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=100.0, resting_metabolism=1.0,
                      growth_rate=0.2, maintenance_level=0.3,
                      starve_rate=starve_rate, max_carrying_capacity=1e6,
                      energy_content=5.0)
        if fid == "p" and kappa is not None:
            params.update(income_growth_kappa=kappa, **(extra or {}))
        groups[fid] = FunctionalGroup(fid, params)
        groups[fid].initialize_state((1, 2), initial_biomass=np.full((1, 2), 2.0))
        groups[fid].energy_reserve[:] = np.float32(reserve * 100.0 * 2.0)
    return EcosystemEnvironment(dict(height=1, width=2), groups,
                                apply_natural_mortality=False)


def _arrays(env, gains, cost, reserve_fill):
    shape = (env.N_dm, 1, 2)
    b = np.full(shape, 2.0, np.float32)
    r = np.full(shape, reserve_fill * 100.0 * 2.0, np.float32)
    return (b, r, np.full(shape, gains, np.float32),
            np.full(shape, cost, np.float32), b.copy(), r.copy())


@pytest.mark.parametrize("gains,cost,fill", [
    (40.0, 10.0, 0.8),     # income-limited: 0.75 x 40 - 10 = 20 MJ
    (400.0, 10.0, 0.8),    # capped at B g (s - u) ec = 2 x 0.2 x 0.5 x 5 = 1 MJ
    (40.0, 10.0, 0.31),    # cap 2 x 0.2 x 0.01 x 5 = 0.02 MJ
    (10.0, 10.0, 0.9),     # kappa I < C: no growth on a full reserve
    (0.0, 10.0, 0.95),     # nothing eaten: the fat is kept
])
def test_growth_is_paid_from_this_ticks_income(gains, cost, fill):
    env = _env()
    b, r, g, c, sb, sr = _arrays(env, gains, cost, fill)
    new_b, new_r = movement.income_growth(env, b, r, g, c, sb, sr)
    i_p, i_o = env.dm_ids.index("p"), env.dm_ids.index("o")
    want = min(max(0.0, 0.75 * gains - cost), 2.0 * 0.2 * max(0.0, fill - 0.3) * 5.0)
    np.testing.assert_allclose(sr[i_p] - new_r[i_p], want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(new_b[i_p] - sb[i_p], want / 5.0, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(new_b[i_o], sb[i_o])
    np.testing.assert_array_equal(new_r[i_o], sr[i_o])


def test_growth_never_exceeds_the_settling_reserve():
    env = _env(kappa=1.0)
    b, r, g, c, sb, sr = _arrays(env, 400.0, 0.0, 0.9)
    sr[:] = 0.5
    _new_b, new_r = movement.income_growth(env, b, r, g, c, sb, sr)
    assert float(new_r.min()) >= 0.0


def _population_delta(env, fid):
    fg = env.fgs[fid]
    before = float(fg.biomass.sum())
    population_change._apply_decision_maker_population_change(env, fid, fg)
    return float(fg.biomass.sum()) - before


def test_population_step_never_grows_it_and_starves_it_as_before():
    fat = _env(reserve=0.9)
    assert fat._has_income_growth
    assert _population_delta(fat, "p") == 0.0
    assert _population_delta(fat, "o") > 0.0
    lean, plain = _env(reserve=0.1), _env(kappa=None, reserve=0.1)
    assert not plain._has_income_growth
    loss = _population_delta(plain, "p")
    assert loss < 0.0
    assert _population_delta(lean, "p") == pytest.approx(loss, rel=1e-6)


def test_refusals():
    with pytest.raises(ValueError, match="starve_rate"):
        _env(starve_rate=0.0)
    with pytest.raises(ValueError, match="growth_q10"):
        _env(extra={"growth_q10": 3.0})
    groups = {"n": FunctionalGroup("n", dict(
        is_decision_maker=False, movement_speed=0.0, growth_rate=0.1,
        max_carrying_capacity=10.0, energy_content=1.0,
        income_growth_kappa=0.7))}
    groups["n"].initialize_state((1, 1), initial_biomass=np.ones((1, 1)))
    with pytest.raises(ValueError, match="not a decision maker"):
        EcosystemEnvironment(dict(height=1, width=1), groups)
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    for sid, extra, match in (
            ("benthic_community", {}, "not a decision"),
            ("zooplankton", {}, "growth_q10"),
            ("pelagic_fish", {"starve_rate": 0.0}, "starve_rate"),
            ("pelagic_fish", {"r_max": 0, "feeding_season_months": 5.5}, "r_max")):
        lib = _library()
        lib["species_definitions"][sid].update(income_growth_kappa=0.7, **extra)
        with pytest.raises(ValueError, match=match):
            load_project_config(str(PROJECT), library_config=lib,
                                project_config=project, grid_size=(6, 6), seed=1)
