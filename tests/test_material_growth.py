"""Material-capped growth of a decision maker (section 244).

Contracts asserted:

1. ``material_growth_share`` parses like ``income_growth_kappa`` (0 or
   empty off, otherwise in (0, 1]); the two keys exclude each other, and
   ``feeding_season_months`` needs one of them.
2. The budget pays g over the feeding season exactly as for the income
   law (s* unchanged), and refuses a share that cannot build the budget
   under ideal conditions in the season: kappa_m h(s_n) C < P ec 12/n,
   s_n the reserve that settles at the season's production.
3. Per cell G = min(B g (s - u) p, kappa_m I p / ec, max(I - C, 0),
   R' - u ME B'): the new tonnes' material only from this tick's food,
   laid down only while the tick's energy balance is positive (244.10),
   paid from the store above maintenance. The case that tells it from the
   income law: I > C with kappa I < C grows here and not there; I < C
   grows in neither (the 244 law without the balance term grew there, on
   the fat). No intake, no growth; no growth at or below maintenance.
4. In the tick the population step never turns its reserve into
   biomass, and its starvation keeps the library rate.
5. Refusals: a non decision maker, growth_q10 on the same DM, no
   starve_rate (loader and env).

GPU parity: tests/test_gpu_ecosystem.py::test_material_growth_matches_reference.
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
LAW_KEYS = ("income_growth_kappa", "material_growth_share", "feeding_season_months")


@pytest.mark.parametrize("spec,expected", [
    ({}, None), ({"material_growth_share": ""}, None),
    ({"material_growth_share": 0.4}, ("material", 0.4, 12.0)),
    ({"material_growth_share": 1, "feeding_season_months": 8.6},
     ("material", 1.0, 8.6)),
    ({"income_growth_kappa": 0.8}, ("income", 0.8, 12.0)),
])
def test_growth_law_parses(spec, expected):
    assert gb.growth_law(spec) == expected
    material = None if expected is None or expected[0] != "material" else (
        expected[1], expected[2])
    income = None if expected is None or expected[0] != "income" else (
        expected[1], expected[2])
    assert gb.material_growth(spec) == material
    assert gb.income_growth(spec) == income


@pytest.mark.parametrize("spec,match", [
    ({"material_growth_share": 1.2}, r"\(0, 1\]"),
    ({"material_growth_share": "x"}, "number"),
    ({"material_growth_share": 0.4, "income_growth_kappa": 0.8}, "exclude"),
    ({"feeding_season_months": 6.0}, "material_growth_share"),
    ({"material_growth_share": 0.4, "feeding_season_months": 0.0001 - 1}, r"\(0, 12\]"),
])
def test_invalid_growth_law_raises(spec, match):
    with pytest.raises(ValueError, match=match):
        gb.growth_law(spec)


def _library():
    return yaml.safe_load(LIBRARY.read_text(encoding="utf-8-sig"))


def _plain(species, fid):
    return {k: v for k, v in species[fid].items() if k not in LAW_KEYS}


def test_budget_is_paid_over_the_season_and_the_material_is_checked():
    lib = _library()
    species, inter = lib["species_definitions"], lib["interaction_definitions"]
    # The hunger-gate case (whiting is on reserve_room since 244.11; the
    # room case is test_reserve_room_appetite_takes_the_whole_ceiling).
    base = dict(_plain(species, "whiting"), appetite="gate")
    plain = gb.growth_budget("whiting", species, inter, species=base)
    spec = dict(base, material_growth_share=1.0, feeding_season_months=6.0)
    seasonal = gb.growth_budget("whiting", species, inter, species=spec)
    assert not seasonal.error
    assert seasonal.growth_rate == pytest.approx(plain.growth_rate * 2.0)
    assert seasonal.reference_fill == pytest.approx(plain.reference_fill)
    income = gb.growth_budget("whiting", species, inter, species=dict(
        base, income_growth_kappa=1.0, feeding_season_months=6.0))
    assert income.growth_rate == pytest.approx(seasonal.growth_rate)
    # The share the material needs in the season: P_n ec / (h(s_n) C), s_n
    # the reserve that settles at the season's production P_n = P 12/n.
    u = seasonal.maintenance
    production = seasonal.gross_annual / gb.TICKS_PER_YEAR * 2.0
    s_n = gb.ideal_reserve(u, seasonal.ideal_intake, seasonal.ideal_cost,
                           float(base["energy_content"]),
                           float(base["max_energy_reserve"]), production)
    assert u < s_n < seasonal.reference_fill
    need = (production * float(base["energy_content"])
            / (min(1.0, (1.0 - s_n) / (1.0 - u)) * seasonal.ideal_intake))
    assert seasonal.material_need == pytest.approx(need)
    short = gb.growth_budget("whiting", species, inter, species=dict(
        spec, material_growth_share=0.99 * need))
    assert short.error and "material_growth_share" in short.error
    assert gb.derived_growth_rate("whiting", species, inter, species=dict(
        spec, material_growth_share=0.99 * need)) is None
    enough = gb.growth_budget("whiting", species, inter, species=dict(
        spec, material_growth_share=1.01 * need))
    assert not enough.error


def test_reserve_room_appetite_takes_the_whole_ceiling():
    """``appetite: reserve_room`` eats at h = 1 (section 240)."""
    lib = _library()
    species, inter = lib["species_definitions"], lib["interaction_definitions"]
    base = _plain(species, "benthivorous_fish")
    assert base.get("appetite") == "reserve_room"
    spec = dict(base, material_growth_share=1.0)
    budget = gb.growth_budget("benthivorous_fish", species, inter, species=spec)
    need = (budget.gross_annual / gb.TICKS_PER_YEAR
            * float(base["energy_content"]) / budget.ideal_intake)
    assert budget.material_need == pytest.approx(need)


def _env(share=0.4, reserve=0.8, starve_rate=0.05, extra=None):
    """DMs ``p`` (material-capped unless ``share`` is None) and ``o``."""
    groups = {}
    for fid in ("p", "o"):
        params = dict(is_decision_maker=True, movement_speed=0.0,
                      max_energy_reserve=100.0, resting_metabolism=1.0,
                      growth_rate=0.2, maintenance_level=0.3,
                      starve_rate=starve_rate, max_carrying_capacity=1e6,
                      energy_content=5.0)
        if fid == "p" and share is not None:
            params.update(material_growth_share=share, **(extra or {}))
        groups[fid] = FunctionalGroup(fid, params)
        groups[fid].initialize_state((1, 2), initial_biomass=np.full((1, 2), 2.0))
        groups[fid].energy_reserve[:] = np.float32(reserve * 100.0 * 2.0)
    return EcosystemEnvironment(dict(height=1, width=2), groups,
                                apply_natural_mortality=False)


def _arrays(env, gains, cost, fill):
    shape = (env.N_dm, 1, 2)
    b = np.full(shape, 2.0, np.float32)
    r = np.full(shape, fill * 100.0 * 2.0, np.float32)
    return (b, r, np.full(shape, gains, np.float32),
            np.full(shape, cost, np.float32), b.copy(), r.copy())


# B 2, g 0.2, u 0.3, ec = p 5, ME 100, kappa_m 0.4:
# ceiling B g (s - u) p, material 0.4 I, balance I - C, store R' - 0.3 x 100 x 2.
@pytest.mark.parametrize("gains,cost,fill,want", [
    (40.0, 10.0, 0.8, 1.0),     # ceiling 2 x 0.2 x 0.5 x 5 = 1 MJ
    (12.0, 10.0, 0.8, 1.0),     # kappa I 4.8 < C 10 < I: grows (the income law would not)
    (10.5, 10.0, 0.8, 0.5),     # balance-limited: I - C = 0.5
    (1.0, 10.0, 0.8, 0.0),      # 0 < I < C: no growth on the fat (244.10)
    (0.5, 0.0, 0.9, 0.2),       # material-limited: 0.4 x 0.5
    (40.0, 10.0, 0.31, 0.02),   # ceiling 2 x 0.2 x 0.01 x 5
    (0.0, 10.0, 0.95, 0.0),     # nothing eaten: no growth, the fat is kept
    (40.0, 0.0, 0.25, 0.0),     # below maintenance: no growth
])
def test_tonnes_come_from_this_ticks_food(gains, cost, fill, want):
    env = _env()
    b, r, g, c, sb, sr = _arrays(env, gains, cost, fill)
    new_b, new_r = movement.income_growth(env, b, r, g, c, sb, sr)
    i_p, i_o = env.dm_ids.index("p"), env.dm_ids.index("o")
    # float32: a reserve of ~160 MJ less a fraction of an MJ.
    np.testing.assert_allclose(sr[i_p] - new_r[i_p], want, rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(new_b[i_p] - sb[i_p], want / 5.0, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(new_b[i_o], sb[i_o])
    np.testing.assert_array_equal(new_r[i_o], sr[i_o])


def test_growth_is_paid_only_from_the_store_above_maintenance():
    env = _env(share=1.0)
    b, r, g, c, sb, sr = _arrays(env, 400.0, 0.0, 0.9)
    sr[:] = np.float32(0.3 * 100.0 * 2.0 + 0.1)
    _new_b, new_r = movement.income_growth(env, b, r, g, c, sb, sr)
    i_p = env.dm_ids.index("p")
    np.testing.assert_allclose(sr[i_p] - new_r[i_p], 0.1, rtol=1e-4)


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
    lean, plain = _env(reserve=0.1), _env(share=None, reserve=0.1)
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
        material_growth_share=0.4))}
    groups["n"].initialize_state((1, 1), initial_biomass=np.ones((1, 1)))
    with pytest.raises(ValueError, match="not a decision maker"):
        EcosystemEnvironment(dict(height=1, width=1), groups)
    project = yaml.safe_load(PROJECT.read_text(encoding="utf-8-sig"))
    for sid, extra, match in (
            ("benthic_infauna", {}, "not a decision"),
            ("zooplankton", {}, "growth_q10"),
            ("whiting", {"starve_rate": 0.0}, "starve_rate"),
            ("whiting", {"income_growth_kappa": 0.7}, "exclude")):
        lib = _library()
        spec = lib["species_definitions"][sid]
        for key in LAW_KEYS:
            spec.pop(key, None)
        spec.update(material_growth_share=1.0, **extra)
        with pytest.raises(ValueError, match=match):
            load_project_config(str(PROJECT), library_config=lib,
                                project_config=project, grid_size=(6, 6), seed=1)
