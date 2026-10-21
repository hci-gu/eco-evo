"""Edible share of a grazed producer (section 182).

Contracts asserted:

1. With ``edibility_floor`` e0 and ``edibility_half_saturation_umol_l``
   K on a grazing NDM's pair, the grazer's rate is the plain rate at the
   edible prey e B, e = e0 + (1 - e0) N/(N + K), N the prey's pool as a
   concentration (pool x k_N / half(t)); the food threshold acts on e B.
2. e is e0 at an empty pool and tends to 1 at a full one; a pair
   without the keys is untouched.
3. One tick of ``apply_ndm_grazing`` takes the edible-share intake.
4. The keys are validated: both or neither, K > 0, e0 in [0, 1], and
   with the pools on the prey must have one; without any pools the term
   is inert, as the Q10 and the mixed-layer scale are without the
   calendar.

GPU parity: tests/test_gpu_ecosystem.py::test_ndm_edibility_matches_reference.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env import interactions, population_change  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

DEPTHS = [13, 9, 6, 6, 6, 6, 6, 7, 10, 10, 17, 15]
CLIMATE = {"light_attenuation_per_m": 0.29, "mixed_layer_depth_m": DEPTHS,
           "cloud_transmission": [0.7] * 12}
DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 4 * 190,
            "random_start": False, "light_climate": CLIMATE}
NUT = {"sub_mixed_layer_umol_l": [5] * 12, "surface_umol_l": [1] * 12,
       "exchange_m_per_day": 0.1, "mixed_layer_depth_m": DEPTHS,
       "tick_hours": 6}
K_N, YIELD = 0.5, 0.8


def _env(edibility=None, pool=True, prey_b=3.0, nutrients=True):
    """``pool=False``: the prey has no pool although another producer
    does; ``nutrients=False``: no pools at all."""
    shape = (2, 3)
    base = dict(movement_speed=0.0, growth_rate=0.0, max_carrying_capacity=1e4,
                energy_content=4500.0, min_split_biomass=0.0,
                extinction_threshold_factor=0.0, daylight=dict(DAYLIGHT))
    if nutrients:
        base["nutrients"] = dict(NUT)
    groups = {}
    if nutrients and not pool:
        groups["other"] = FunctionalGroup("other", dict(
            base, is_decision_maker=False, nutrient_half_saturation=K_N,
            nutrient_yield=YIELD))
    phyto_params = dict(base, is_decision_maker=False)
    if nutrients and pool:
        phyto_params.update(nutrient_half_saturation=K_N, nutrient_yield=YIELD,
                            grazing_regeneration=0.0)
    phyto = FunctionalGroup("phyto", phyto_params)
    pair = {"preys_on": True, "max_intake_rate": 0.3, "handling_time": 2.0,
            "mixed_layer_reference_m": 10.0, "food_threshold": 0.4}
    pair.update(edibility or {})
    micro = FunctionalGroup("micro", dict(
        base, is_decision_maker=False, growth_efficiency=0.3,
        interaction={"micro_preys_on_phyto": pair}))
    groups.update(phyto=phyto, micro=micro)
    for fg in groups.values():
        b = prey_b if fg is phyto else 1.0
        fg.initialize_state(shape, initial_biomass=np.full(shape, b))
    return EcosystemEnvironment(dict(height=2, width=3), groups,
                                apply_natural_mortality=False)


EDIBLE = {"edibility_floor": 0.1, "edibility_half_saturation_umol_l": 2.0}


def _share(env, pool):
    t = interactions.light_index(env)
    half = env.nutrient_tables["phyto"]["half"][t]      # k_N x Y x M(t)
    n = pool * K_N / half                               # umol/l
    return 0.1 + 0.9 * n / (n + 2.0)


@pytest.mark.parametrize("pool_value", [0.0, 0.5, 3.0, 500.0])
def test_rate_is_the_plain_rate_at_the_edible_prey(pool_value):
    env = _env(EDIBLE)
    entry = env.ndm_grazers[0]
    plain = dict(entry, edibility=None)
    prey = np.array([0.2, 3.0, 40.0], dtype=np.float32)
    pool = np.full(3, pool_value, dtype=np.float32)
    e = _share(env, pool_value)
    got = interactions.ndm_grazing_rate(env, entry, prey, pool)
    want = interactions.ndm_grazing_rate(env, plain, prey * np.float32(e))
    np.testing.assert_allclose(got, want, rtol=1e-5)
    if pool_value == 0.0:
        assert e == pytest.approx(0.1)
    if pool_value == 500.0:
        assert e > 0.95
    with pytest.raises(ValueError, match="pool"):
        interactions.ndm_grazing_rate(env, entry, prey)


def test_pair_without_the_keys_is_untouched():
    env = _env()
    assert env.ndm_grazers[0]["edibility"] is None
    prey = np.array([0.2, 3.0], dtype=np.float32)
    entry = env.ndm_grazers[0]
    np.testing.assert_array_equal(
        interactions.ndm_grazing_rate(env, entry, prey),
        interactions.ndm_grazing_rate(env, entry, prey, np.zeros(2, np.float32)))


def test_one_tick_takes_the_edible_share():
    env = _env(EDIBLE)
    phyto = env.fgs["phyto"]
    phyto.energy_reserve[:] = np.float32(0.7)
    entry = env.ndm_grazers[0]
    rate = interactions.ndm_grazing_rate(
        env, dict(entry, edibility=None),
        phyto.biomass * np.float32(_share(env, 0.7)))
    before = phyto.biomass.copy()
    population_change.apply_ndm_grazing(env)
    np.testing.assert_allclose(before - phyto.biomass, 1.0 * rate, rtol=1e-4)


@pytest.mark.parametrize("keys,match", [
    ({"edibility_floor": 0.1}, "go together"),
    ({"edibility_half_saturation_umol_l": 2.0}, "go together"),
    ({"edibility_floor": 1.5, "edibility_half_saturation_umol_l": 2.0}, "floor"),
    ({"edibility_floor": 0.1, "edibility_half_saturation_umol_l": 0.0}, "> 0"),
])
def test_invalid_keys_raise(keys, match):
    with pytest.raises(ValueError, match=match):
        _env(keys)


def test_prey_without_a_pool_raises_and_no_pools_is_inert():
    with pytest.raises(ValueError, match="nutrient pool"):
        _env(EDIBLE, pool=False)
    assert _env(EDIBLE, nutrients=False).ndm_grazers[0]["edibility"] is None
