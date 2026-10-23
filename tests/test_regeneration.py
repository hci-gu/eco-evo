"""Regenerated nutrients from consumer losses (section 202).

Contracts asserted:

1. ``mortality_regeneration`` / ``egestion_regeneration`` parse as shares
   in [0, 1]; anything else is refused.
2. Inert without nutrient pools, and with a share of 0 the tick is
   unchanged.
3. Conservation (CPU): the pools gain exactly share x the M1 loss of the
   regenerating groups, split by where each fed (all to the one pool
   without the subsurface layer).
4. The layer split follows the intake: a group that fed only on the
   mixed producer returns everything there; a group that fed on neither
   uses the thickness share M(t) / bottom.
5. A grazing NDM's ``egestion_regeneration`` adds share x (1 - gge) x
   (1 - excretion) of its intake to the prey's pool.
6. GPU parity on the fixture (two layers; one pool with a grazer) and on
   a small mareld2 world, where the library sets the keys.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import population_change, regeneration
from lib.environments.ecosystem_env.regeneration import species_share
from test_gpu_ecosystem import (DEVICES, LIGHT_CLIMATE, NUTRIENTS, SUBSURFACE, device,
                                _assert_parity, make_env)


def test_shares_parse_and_refuse():
    assert species_share({}, "mortality_regeneration") == 0.0
    assert species_share({"mortality_regeneration": ""}, "mortality_regeneration") == 0.0
    assert species_share({"mortality_regeneration": "0.5"}, "mortality_regeneration") == 0.5
    for bad in (-0.1, 1.5, "x", float("nan")):
        with pytest.raises(ValueError):
            species_share({"k": bad}, "k")


DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 4 * 120 + 1,
            "random_start": False, "light_climate": LIGHT_CLIMATE}
POOL = (NUTRIENTS, {"nutrient_half_saturation": 0.5, "nutrient_yield": 0.02,
                    "grazing_regeneration": 0.3})


def _layered(share, start_tick=4 * 120 + 1):
    daylight = dict(DAYLIGHT, start_tick=start_tick)
    return make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                    daylight=daylight, light_saturation=150.0,
                    nutrient_pool=POOL, subsurface=(SUBSURFACE, 50.0),
                    extra_prey={"a": ["d"]},
                    regen={"a": {"mortality_regeneration": share},
                           "b": {"mortality_regeneration": share}})


def test_inert_without_pools_or_share():
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   regen={"a": {"mortality_regeneration": 0.5}})
    assert not env._has_regeneration
    env = _layered(0.0)
    assert not env._has_regeneration


def _population_step(env):
    """One population step with this tick's intake set by hand."""
    population_change.apply_population_change(env)


def test_cpu_returns_share_of_m1_split_by_feeding():
    env = _layered(0.5)
    assert env._has_regeneration
    m, d = env.regen["mixed"], env.regen["subsurface"]
    assert (m, d) == ("c", "d")
    ia, ib = env.dm_ids.index("a"), env.dm_ids.index("b")
    jc, jd = env.global_fg_order.index("c"), env.global_fg_order.index("d")
    # a fed 3:1 on the mixed and the subsurface producer, b on nothing
    intake = np.zeros((env.N_dm, env.N_all))
    intake[ia, jc], intake[ia, jd] = 3.0, 1.0
    env._tick_dm_intake = intake
    env._regen_loss = {
        "a": np.full(env.fgs["a"].biomass.shape, 2.0),
        "b": np.full(env.fgs["b"].biomass.shape, 1.0)}
    top0 = env.fgs[m].energy_reserve.astype(np.float64).copy()
    low0 = env.fgs[d].energy_reserve.astype(np.float64).copy()
    regeneration.apply(env)
    from lib.environments.ecosystem_env import interactions
    fb = float(env.regen["fallback"][interactions.light_index(env)])
    assert 0.0 < fb < 1.0
    np.testing.assert_allclose(env.fgs[m].energy_reserve - top0,
                               0.5 * 2.0 * 0.75 + 0.5 * 1.0 * fb, rtol=1e-5)
    np.testing.assert_allclose(env.fgs[d].energy_reserve - low0,
                               0.5 * 2.0 * 0.25 + 0.5 * 1.0 * (1 - fb), rtol=1e-5)
    assert env._regen_loss == {}


def test_cpu_conserves_the_m1_share_over_a_tick():
    """The pools gain share x the booked M1 loss (both layers summed)."""
    env = _layered(0.5)
    plain = _layered(0.0)
    env.policies = plain.policies = None
    for e in (env, plain):
        e.loss_natural = {}
    # Same state, same tick: the population step alone.
    for e in (env, plain):
        e._tick_dm_intake = np.zeros((e.N_dm, e.N_all))
        np.random.seed(3)
        e.ordered_fg_ids = list(e.fgs)
        population_change.apply_population_change(e)
    gained = sum(float(env.fgs[f].energy_reserve.sum() - plain.fgs[f].energy_reserve.sum())
                 for f in ("c", "d"))
    m1 = env.loss_natural["a"] + env.loss_natural["b"]
    assert gained == pytest.approx(0.5 * m1, rel=1e-4)


def test_egestion_returns_to_the_prey_pool():
    kwargs = dict(min_split=0.0, extinction_factor=0.0, mortality=True,
                  daylight=DAYLIGHT, light_saturation=150.0, nutrient_pool=POOL,
                  grazer=({"max_intake_rate": 0.4, "handling_time": 2.0},
                          {"growth_efficiency": 0.3, "excretion_regeneration": 0.4}))
    plain = make_env(**kwargs)
    kwargs["grazer"][1]["egestion_regeneration"] = 1.0
    egest = make_env(**kwargs)
    assert egest.ndm_grazers[0]["pool"] == pytest.approx(0.7 * (0.4 + 0.6))
    assert plain.ndm_grazers[0]["pool"] == pytest.approx(0.7 * 0.4)
    pools = []
    for env in (plain, egest):
        c0 = env.fgs["c"].energy_reserve.astype(np.float64).copy()
        population_change.apply_ndm_grazing(env)
        pools.append(env.fgs["c"].energy_reserve - c0)
        intake = env._tick_ndm_intake[("d", "c")]
    assert float((pools[1] - pools[0]).sum()) == pytest.approx(0.7 * 0.6 * intake, rel=1e-4)


@pytest.mark.parametrize("start_tick", [4 * 120 + 1, 4 * 290 + 2])
def test_two_layer_regeneration_matches_reference(device, start_tick):
    env = _layered(0.5, start_tick)
    model = _assert_parity(env, device, ticks=8)
    assert model.regen_on and model.regen_d is not None


def test_one_pool_with_a_grazer_matches_reference(device):
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   daylight=DAYLIGHT, light_saturation=150.0, nutrient_pool=POOL,
                   grazer=({"max_intake_rate": 0.4, "handling_time": 2.0},
                           {"growth_efficiency": 0.3, "excretion_regeneration": 0.4,
                            "egestion_regeneration": 1.0,
                            "mortality_regeneration": 0.5}),
                   regen={"a": {"mortality_regeneration": 0.5}})
    assert env._has_regeneration and env.regen["subsurface"] is None
    model = _assert_parity(env, device, ticks=8)
    assert model.regen_on and model.regen_d is None


def test_mareld2_world_matches_reference(device, tmp_path):
    """The library's keys on the project world (two layers, zooplankton
    and microzooplankton regenerating, Type III zoo -> phytoplankton).
    seed_rate is zeroed in a copy: the reference draws the seeding noise
    from np.random and the tensor engine takes it injected, so a seeded
    empty cell would differ (test_seeding_with_injected_noise covers it)."""
    import yaml
    from lib.gpu.config import EnvironmentBuilder
    root = Path(__file__).resolve().parents[1]
    library = yaml.safe_load((root / "fgconfig" / "fg_library.yaml").read_text())
    for spec in library["species_definitions"].values():
        spec["seed_rate"] = 0.0
    path = tmp_path / "library.yaml"
    path.write_text(yaml.safe_dump(library, sort_keys=False))
    env = EnvironmentBuilder(project_path=str(root / "mareld2.yaml"), grid=(5, 6),
                             library_path=str(path))(seed=3)
    assert env._has_regeneration
    assert set(env.regen["shares"]) == {"zooplankton", "microzooplankton"}
    assert env.ndm_grazers[0]["pool"] == pytest.approx(0.7 * (0.571 + 0.429))
    _assert_parity(env, device, ticks=6)
