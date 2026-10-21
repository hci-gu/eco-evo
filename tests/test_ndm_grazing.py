"""Grazing non decision makers (section 177: microzooplankton).

Contracts asserted:

1. One tick of grazing takes B_g m(t) a' B_p / (1 + a' h B_p) of the
   prey, with a' the attack rate times the pair's mixed-layer scale
   M_ref / M(t) and m(t) the grazer's Q10 multiplier; the grazer gains
   growth_efficiency of it, the prey's nutrient pool gets
   excretion_regeneration x (1 - growth_efficiency) of it, and the
   intake is booked as predation on the prey.
2. The take is capped at MAX_HARVEST_FRAC of the prey.
3. A decision maker's pair with mixed_layer_reference_m has its attack
   rate scaled by M_ref / M(t) through the per-tick pair table.

GPU parity: tests/test_gpu_ecosystem.py::test_ndm_grazing_matches_reference.
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
from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC  # noqa: E402
from lib.world import daylight, nutrients, temperature  # noqa: E402
from lib.world.functional_group import FunctionalGroup  # noqa: E402

DEPTHS = [13, 9, 6, 6, 6, 6, 6, 7, 10, 10, 17, 15]
CLIMATE = {"light_attenuation_per_m": 0.29, "mixed_layer_depth_m": DEPTHS,
           "cloud_transmission": [0.7] * 12}
TEMPS = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]
DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 4 * 190,
            "random_start": False, "light_climate": CLIMATE}
NUT = {"sub_mixed_layer_umol_l": [5] * 12, "surface_umol_l": [1] * 12,
       "exchange_m_per_day": 0.1, "mixed_layer_depth_m": DEPTHS,
       "tick_hours": 6}


def _env(a=0.3, h=2.0, prey_b=3.0, grazer_b=1.0, scaled=True, q10=2.6):
    shape = (2, 3)
    base = dict(movement_speed=0.0, growth_rate=0.0, max_carrying_capacity=1e4,
                energy_content=4500.0, min_split_biomass=0.0,
                extinction_threshold_factor=0.0, daylight=dict(DAYLIGHT),
                nutrients=dict(NUT))
    phyto = FunctionalGroup("phyto", dict(
        base, is_decision_maker=False, nutrient_half_saturation=1.0,
        nutrient_yield=0.8, grazing_regeneration=0.0))
    pair = {"preys_on": True, "max_intake_rate": a, "handling_time": h}
    if scaled:
        pair["mixed_layer_reference_m"] = 10.0
    micro_params = dict(base, is_decision_maker=False, growth_efficiency=0.3,
                        excretion_regeneration=0.4,
                        interaction={"micro_preys_on_phyto": pair})
    if q10:
        micro_params.update(grazing_q10=q10, grazing_t_ref=10.0,
                            temperature={"layer": "upper", "monthly_c": TEMPS})
    micro = FunctionalGroup("micro", micro_params)
    for fg, b in ((phyto, prey_b), (micro, grazer_b)):
        fg.initialize_state(shape, initial_biomass=np.full(shape, b))
    return EcosystemEnvironment(dict(height=2, width=3),
                                {"phyto": phyto, "micro": micro},
                                apply_natural_mortality=False)


def test_one_tick_of_grazing():
    env = _env()
    entry = env.ndm_grazers[0]
    t = interactions.light_index(env)
    hours = (np.arange(1460) * 6 + 3) / 24.0
    depth = daylight.monthly_value(DEPTHS, hours)[t]
    m = 2.6 ** ((temperature.tick_temperatures(TEMPS, 6)[t] - 10.0) / 10.0)
    a = 0.3 * 10.0 / depth
    expected = 1.0 * m * a * 3.0 / (1.0 + a * 2.0 * 3.0)
    pool0 = env.fgs["phyto"].energy_reserve.copy()
    population_change.apply_ndm_grazing(env)
    taken = 3.0 - env.fgs["phyto"].biomass
    assert np.allclose(taken, expected, rtol=1e-4)
    assert np.allclose(env.fgs["micro"].biomass, 1.0 + 0.3 * expected,
                       rtol=1e-4)
    assert np.allclose(env.fgs["phyto"].energy_reserve - pool0,
                       0.4 * 0.7 * expected, rtol=1e-4)
    assert env.loss_predation["phyto"] == pytest.approx(taken.sum(), rel=1e-5)
    assert env.intake_by_pred_prey["micro"]["phyto"] == pytest.approx(
        taken.sum(), rel=1e-5)
    assert entry["scale"] is not None and entry["temperature"] is not None


def test_food_threshold_makes_the_response_sigmoid():
    env = _env(scaled=False, q10=None)
    entry = dict(env.ndm_grazers[0], threshold=1.0)
    low, high = np.array([0.1]), np.array([30.0])
    plain = dict(entry, threshold=0.0)
    ratio_low = (interactions.ndm_grazing_rate(env, entry, low)
                 / interactions.ndm_grazing_rate(env, plain, low))
    ratio_high = (interactions.ndm_grazing_rate(env, entry, high)
                  / interactions.ndm_grazing_rate(env, plain, high))
    a, h = entry["a"], entry["h"]
    food = 0.1 * 0.1 / 1.1
    exact = (a * food / (1 + a * h * food)) / (a * 0.1 / (1 + a * h * 0.1))
    assert ratio_low == pytest.approx(exact, rel=1e-4)
    assert ratio_low < 0.1
    assert ratio_high > 0.95


def test_grazing_is_capped_by_the_prey():
    env = _env(a=50.0, h=0.0, prey_b=0.1, grazer_b=100.0, scaled=False,
               q10=None)
    population_change.apply_ndm_grazing(env)
    assert np.allclose(env.fgs["phyto"].biomass,
                       0.1 * (1 - float(MAX_HARVEST_FRAC)), rtol=1e-3)


def test_no_grazing_pair_is_inert():
    env = _env(a=0.0)
    assert not env._has_ndm_grazing
    b0 = env.fgs["phyto"].biomass.copy()
    population_change.apply_ndm_grazing(env)
    assert np.array_equal(env.fgs["phyto"].biomass, b0)


def test_mixed_layer_scale_is_m_ref_over_depth():
    env = _env()
    scale = interactions.mixed_layer_scale(
        env, {"mixed_layer_reference_m": 10.0})
    hours = (np.arange(1460) * 6 + 3) / 24.0
    assert np.allclose(scale, 10.0 / daylight.monthly_value(DEPTHS, hours))
    assert interactions.mixed_layer_scale(env, {}) is None
