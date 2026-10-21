"""Numerical reference tests run on CPU; also exercise CUDA when available."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem import EcosystemEnvironment
from lib.environments.ecosystem_env import movement, policies, population_change
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.world.functional_group import FunctionalGroup


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required"))]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def make_env(migration=False, mortality=False, handling=0.2, shape=(4, 5),
             pair_floors=None, min_split=200,
             extinction_factor=0.3, interference=None,
             mortality_multiplier=1.0, daylight=None, dark_ratios=None,
             light_saturation=None, m1_exposure=None, reserve_food=None,
             response_types=None, temperature=None, impacts=None,
             observable_impacts=None, nutrient_pool=None, grazer=None,
             mixed_layer_pairs=None, subsurface=None, inert_subsurface=False,
             ingestion=None, growth=None, income=None, capacity=None):
    """Reference fixture.

    ``pair_floors`` maps ``"{pred}_preys_on_{prey}"`` to a per-pair
    ``visibility_floor`` override. Absent means the cheap shared-vector
    path.

    ``min_split`` / ``extinction_factor`` default to the values above the
    fixture's biomass, i.e. splits are suppressed and the extinction sweep
    clears the grid. Set both to 0 for a fixture where biomass survives and
    actually moves between cells.

    ``daylight`` is the ``params['daylight']`` dict the loader would put
    on every FG; ``dark_ratios`` maps an interaction id to its
    ``dark_ratio`` (section 137). ``light_saturation`` makes the NDM ``c``
    light-limited (section 138; needs a light climate in ``daylight``).
    ``temperature`` is ``(q10, t_ref, monthly_c)`` for the DM ``a``: its
    Q10 metabolism on the daylight calendar (section 143).
    ``impacts`` maps a group to its ``params['impact']`` block; the caller
    adds the fields with ``env.grid.add_map`` (section 161).
    ``observable_impacts`` lists the impacts the policies observe.
    ``nutrient_pool`` is ``(params['nutrients'], species keys)``: every FG
    carries the former, the NDM ``c`` the latter (section 172).
    ``grazer`` is ``(pair keys, species keys)`` making the NDM ``d`` graze
    ``c`` (section 177).
    ``subsurface`` is ``(subsurface_layer block, light_saturation)``: the
    NDM ``d`` becomes the subsurface producer with ``c``'s nutrient keys
    (section 180; needs ``nutrient_pool``).
    ``ingestion`` is ``(q10, monthly_c)``: Q10 on DM ``a``'s whole intake
    (section 181; needs ``daylight``).
    ``growth`` is ``(q10, monthly_c)``: Q10 on DM ``a``'s growth rate
    (section 183; needs ``daylight``).
    ``capacity`` is DM ``a``'s ``capacity_loss_rate`` c: the regional
    logistic loss c * grid-mean B per tick (section 185).
    """
    rng = np.random.default_rng(42)
    pair_floors = pair_floors or {}
    groups = {}
    for i, fid in enumerate(("a", "b", "c", "d")):
        params = dict(is_decision_maker=i < 2, movement_speed=[0.7, 0.0, 0, 0][i],
                      max_energy_reserve=10 + i, resting_metabolism=0.05,
                      movement_cost=2.5, feeding_cost=1.2, resting_cost=0.3,
                      growth_rate=0.2, starve_rate=0.7, maintenance_level=0.4,
                      natural_mortality=0.1, visibility_floor=0.25,
                      min_split_biomass=min_split,
                      extinction_threshold_factor=extinction_factor,
                      max_carrying_capacity=12, seed_rate=0.0, energy_content=20,
                      observes=["b", "c"] if i == 0 else ["a"],
                      max_intake_rate=0.8,
                      menu=["b", "c"] if i == 0 else (["c"] if i == 1 else []))
        params["interaction"] = {f"{fid}_preys_on_{prey}": dict(
            preys_on=True, assimilation_factor=0.65, handling_time=handling)
            for prey in params["menu"]}
        for inter_id, floor in pair_floors.items():
            if inter_id in params["interaction"]:
                params["interaction"][inter_id]["visibility_floor"] = floor
        if interference and fid in interference:
            params["interference"] = interference[fid]
        for inter_id, kind in (response_types or {}).items():
            if inter_id in params["interaction"]:
                params["interaction"][inter_id]["functional_response"] = kind
        for inter_id, ratio in (dark_ratios or {}).items():
            if inter_id in params["interaction"]:
                params["interaction"][inter_id]["dark_ratio"] = ratio
        for inter_id, m_ref in (mixed_layer_pairs or {}).items():
            if inter_id in params["interaction"]:
                params["interaction"][inter_id]["mixed_layer_reference_m"] = m_ref
        if daylight:
            params["daylight"] = dict(daylight)
        if light_saturation and fid == "c":
            params["light_saturation"] = light_saturation
        if nutrient_pool is not None:
            params["nutrients"] = dict(nutrient_pool[0])
            if fid == "c":
                params.update(nutrient_pool[1])
            if subsurface is not None:
                params["nutrients"]["subsurface"] = dict(subsurface[0])
                if fid == "d":
                    params.update(nutrient_pool[1])
                    params.update(water_layer="subsurface",
                                  light_saturation=subsurface[1])
        if inert_subsurface and fid == "d":
            params["water_layer"] = "subsurface"
        if grazer is not None and fid == "d":
            params["interaction"] = {"d_preys_on_c": dict(
                preys_on=True, **grazer[0])}
            params.update(grazer[1])
        if temperature and fid == "a":
            q10, t_ref, monthly_c = temperature
            params.update(metabolism_q10=q10, metabolism_t_ref=t_ref,
                          temperature={"layer": "upper",
                                       "monthly_c": list(monthly_c)})
        if ingestion and fid == "a":
            params.update(ingestion_q10=ingestion[0],
                          temperature={"layer": "upper",
                                       "monthly_c": list(ingestion[1])})
        if growth and fid == "a":
            params.update(growth_q10=growth[0],
                          temperature={"layer": "upper",
                                       "monthly_c": list(growth[1])})
        if income and fid == "a":
            params.update(income_growth_kappa=income[0],
                          feeding_season_months=income[1])
        if capacity and fid == "a":
            params["capacity_loss_rate"] = capacity
        if m1_exposure and fid == "a":
            params.update(m1_exposure)
        if impacts and fid in impacts:
            params["impact"] = copy.deepcopy(impacts[fid])
        if reserve_food is not None and fid == "b":
            params["prey_includes_reserve"] = True
            params["reserve_reference_fill"] = reserve_food
        fg = FunctionalGroup(fid, params)
        fg.initialize_state(shape, initial_biomass=rng.uniform(0.01, 3, shape),
                            randomize_energy=True, rng=rng)
        fg.biomass.flat[0] = 0
        fg.energy_reserve.flat[0] = 0
        groups[fid] = fg
    env = EcosystemEnvironment(dict(height=shape[0], width=shape[1]), groups,
                               migration=migration, apply_natural_mortality=mortality,
                               mortality_multiplier=mortality_multiplier,
                               observable_impact_vars=observable_impacts)
    return env


def numpy(t):
    return t.detach().cpu().numpy()


def manual_actions(probs, env):
    p = numpy(probs[0]).reshape(env.N_dm, 5 + env.N_all, env.H, env.W)
    return ActionProbabilities(p[:, :4], p[:, 4], p[:, 5:])


# The two engines are two float32 implementations of the same map: they
# part company at round-off in tick 0 and the difference then compounds
# by up to ~3x per tick through the feedback into the policy input. The
# migration+mortality fixture sits at ~1 ULP (1.4e-7 relative) for seven
# ticks and reaches 4.9e-5 by tick 11, with the action probabilities
# still identical to 6e-8 - no branch has flipped, it is the state that
# drifts. A single constant bound therefore has to be set by the last
# tick of the longest fixture, and is then blind for the first ones,
# where a real formula difference appears at once and three orders of
# magnitude larger. Grow the bound from the ULP scale at the rate the
# divergence actually grows, and cap it where the measurements level
# off (worst seen: 6e-5 at tick 6 on the pair-visibility fixture).
PARITY_GROWTH = 3.0
PARITY_TOL_CAP = 3e-4


def _parity_tol(tick, base=1e-6):
    return min(PARITY_TOL_CAP, base * PARITY_GROWTH ** tick)


def _assert_parity(env, device, ticks=12):
    model = TensorEcosystem(env, device)
    b, r, hidden = model.import_state([env])
    bank = PolicyBank(model, hidden_dim=9, hidden_layers=2, seed=13)
    env.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    env.build_static_caches()
    packed = bank.pack([w[None] for w in bank.flat_weights()])
    starts = model.light_starts([env])
    impacts = model.import_impacts([env]) if model.has_impacts else None
    impact_obs = model.import_impact_observations([env]) if model.n_obs_imp else None
    for tick in range(ticks):
        tol = _parity_tol(tick)
        light = model.light_index(starts, torch.tensor(tick, device=model.device))
        obs = model.observations(b, r, hidden, light, impact_obs)
        observation = env.get_observation()
        np.testing.assert_allclose(numpy(obs[0]), observation.features, atol=tol, rtol=tol)
        logits = bank.forward(obs, *packed)
        probs = model.action_probabilities(logits, b, model.tensor(1.0))
        cpu_actions = env.policy_controller.forward(observation)
        ours = manual_actions(probs, env)
        # Probabilities are recomputed from the state each tick, so their
        # difference does not accumulate; it stays at the ULP scale.
        for field in ("move", "rest", "eat"):
            np.testing.assert_allclose(getattr(ours, field), getattr(cpu_actions, field), atol=2e-6, rtol=2e-5)
        b, r, hidden, _, _ = model.step(b, r, probs, model.tensor(tick), torch.zeros_like(b),
                                        light_index=light, impacts=impacts)
        env.step(cpu_actions)
        expected = model.import_state([env])
        np.testing.assert_allclose(numpy(b), numpy(expected[0]), atol=tol, rtol=tol)
        np.testing.assert_allclose(numpy(r), numpy(expected[1]), atol=tol, rtol=tol)
        assert torch.isfinite(b).all() and (b >= 0).all()
    return model


@pytest.mark.parametrize("migration,mortality", [(False, False), (True, True)])
@pytest.mark.parametrize("handling", [0.0, 0.2])
def test_full_ticks_match_reference(device, migration, mortality, handling):
    model = _assert_parity(make_env(migration, mortality, handling), device)
    assert model.pair_visibility is None, (
        "no override in the fixture must keep the cheap shared-vector path")


@pytest.mark.parametrize("multiplier", [0.0, 0.5, 2.0])
def test_scaled_mortality_matches_reference(device, multiplier):
    """``--mortality_multiplier`` must scale both engines identically.

    The fixture's ``natural_mortality`` is 0.1, so the factor is visible
    in every tick; a mirror that forgot the scale (or applied it to the
    survival fraction instead of the rate) diverges immediately.
    """
    env = make_env(mortality=True, mortality_multiplier=multiplier)
    model = _assert_parity(env, device, ticks=6)
    expected = max(0.0, 1.0 - 0.1 * multiplier)
    # Every group, the non decision makers included (section 173: their
    # natural_mortality, 0.1 in the fixture, is a real loss now).
    np.testing.assert_allclose(
        numpy(model.mortality_keep).reshape(-1),
        [expected for f in model.ids], rtol=1e-6)


@pytest.mark.parametrize("handling", [0.0, 0.2])
def test_interference_matches_reference(device, handling):
    """Beddington-DeAngelis interference must be mirrored on the GPU.

    ``handling=0.0`` also covers the unsaturated branch, where the
    reference divides the bare intake rate by ``1 + w*B_pred`` instead of
    going through ``holling_a_eff`` at all - an easy branch to forget.
    """
    env = make_env(handling=handling, interference={"a": 0.8, "b": 0.3})
    model = _assert_parity(env, device, ticks=6)
    assert model.has_interference, "interference path was not taken"
    np.testing.assert_allclose(
        numpy(model.interference).reshape(-1),
        [0.8 if fid == "a" else 0.3 for fid in env.dm_ids], rtol=1e-6)


@pytest.mark.parametrize("floor", [0.0, 0.95])
@pytest.mark.parametrize("handling", [0.0, 0.2])
def test_pair_visibility_matches_reference(device, floor, handling):
    """The GPU engine must honour per-pair detection.

    It was prey-side-only on the GPU while the reference had per-(predator,
    prey) ``visibility_floor`` (and, until section 130, per-FG
    ``satiation_scale``), so ``train_gpu.py`` optimised against a different
    biology than
    ``inference.py`` runs - invisible here until the fixture actually sets
    non-trivial values. ``floor=0.0`` also checks that rows which do NOT
    prey on a column cannot inflate the shared availability cap.
    """
    env = make_env(handling=handling,
                   pair_floors={"a_preys_on_b": floor,
                                "a_preys_on_c": 0.5})
    # Shorter horizon than the shared-path test on purpose. The two engines
    # group the same operations differently, so they diverge by one float32
    # ulp in tick 0 and that difference is then amplified chaotically by the
    # feedback through the policy input (measured on CUDA: rel 1e-7 at tick
    # 0 -> 6e-5 at tick 6, always in the one cell where prey is harvested
    # down to the MAX_HARVEST_FRAC residue). Six ticks stay two orders
    # below the tolerance while still catching a formula difference, which
    # shows up at tick 0 and three orders of magnitude larger.
    model = _assert_parity(env, device, ticks=6)
    assert model.pair_visibility is not None, "pair path was not taken"
    i, j = env.dm_ids.index("a"), env.global_fg_order.index("b")
    assert float(env.vis_floor_mat[i, j]) == pytest.approx(floor)


@pytest.mark.parametrize("start_tick", [0, 3])
@pytest.mark.parametrize("handling", [0.0, 0.2])
def test_daylight_matches_reference(device, start_tick, handling):
    """The daylight calendar must index the same year tick on both engines.

    One pair is light-modulated and one is not, so a mirror that
    modulated every pair, read the multiplier one tick off, or put the
    light channel in another slot diverges at once. 6 h ticks: the
    starts 0 and 3 sit at midnight and 18:00, so the run crosses dawn
    and dusk.
    """
    env = make_env(handling=handling,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False},
                   dark_ratios={"a_preys_on_b": 0.1})
    model = _assert_parity(env, device, ticks=6)
    assert model.daylight and model.light_pairs
    i, j = env.dm_ids.index("a"), env.global_fg_order.index("c")
    assert np.all(env.light_mult_table[:, i, j] == 1.0), (
        "an unmodulated pair must stay exactly 1")
    assert model.in_dims == tuple(int(n) for n in env.per_dm_in_dim)


LIGHT_CLIMATE = {"light_attenuation_per_m": 0.15,
                 "mixed_layer_depth_m": [40, 40, 30, 15, 12, 10,
                                         10, 12, 15, 25, 35, 40],
                 "cloud_transmission": [0.5] * 12}


@pytest.mark.parametrize("start_tick", [0, 4 * 170 + 1])
@pytest.mark.parametrize("with_pairs", [False, True])
def test_light_limited_growth_matches_reference(device, start_tick, with_pairs):
    """Producer growth must read the same year tick on both engines.

    ``with_pairs`` False is the calendar without any modulated pair,
    i.e. still with the light observation channel; the NDM ``d`` is not
    light-limited and must keep growth_rate exactly.
    """
    env = make_env(min_split=0.0, extinction_factor=0.0,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False,
                             "light_climate": LIGHT_CLIMATE},
                   dark_ratios={"a_preys_on_b": 0.1} if with_pairs else None,
                   light_saturation=150.0)
    model = _assert_parity(env, device, ticks=6)
    assert model.growth_light_on
    j_c, j_d = model.ids.index("c"), model.ids.index("d")
    assert torch.all(model.growth_light[:, j_d] == 1.0)
    # Midnight starts a night tick (no growth); the second start is noon
    # in late June (growth above the April reference).
    first = float(model.growth_light[start_tick, j_c])
    assert first == pytest.approx(0.0, abs=1e-6) if start_tick == 0 else first > 1.0


TEMPERATURES_NDM = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8,
                    10.5, 8.4]
NUTRIENTS = {"sub_mixed_layer_umol_l": [6, 7, 7, 7, 5, 4, 3, 1.4, 0.9, 0.9, 2.5, 4],
             "surface_umol_l": [5, 5, 2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 1.3, 3.6],
             "exchange_m_per_day": 0.3,
             "mixed_layer_depth_m": LIGHT_CLIMATE["mixed_layer_depth_m"],
             "tick_hours": 6}


@pytest.mark.parametrize("start_tick", [4 * 40 + 2, 4 * 300 + 1])
@pytest.mark.parametrize("regeneration,stress", [(0.0, 0.0), (0.6, 0.0),
                                                 (0.6, 0.08)])
def test_nutrient_pool_matches_reference(device, start_tick, regeneration,
                                         stress):
    """Section 172: supply, uptake, the K ceiling and grazing's return to
    the pool must agree on both engines; the pool rides in the reserve
    slot, so the ordinary state comparison checks it. The small yield
    makes the pool limiting, so the N/(N + half) term is exercised."""
    env = make_env(min_split=0.0, extinction_factor=0.0,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False,
                             "light_climate": LIGHT_CLIMATE},
                   light_saturation=150.0,
                   nutrient_pool=(NUTRIENTS, {
                       "nutrient_half_saturation": 0.5,
                       "nutrient_yield": 0.01,
                       "grazing_regeneration": regeneration,
                       "stress_loss_max": stress}))
    j_c = sorted(env.fgs).index("c")
    assert env._has_nutrients and env.nutrient_ids == ["c"]
    assert env.fgs["c"].energy_reserve.flat[0] > 0     # empty cell, full pool
    # The GPU trainer's own start (TensorEcosystem.initial_nutrients)
    # must equal the CPU environment's pool at the same start tick.
    model = TensorEcosystem(env, device)
    _, r0, _ = model.import_state([env])
    started = model.initial_nutrients(torch.zeros_like(r0),
                                      model.light_starts([env]))
    np.testing.assert_allclose(numpy(started[0, j_c]), numpy(r0[0, j_c]),
                               rtol=1e-6)
    assert float(started[0].sum()) == pytest.approx(float(started[0, j_c].sum()))
    model = _assert_parity(env, device, ticks=8)
    assert model.nutrients_on and bool(model.nutrient[0, j_c, 0])
    # An empty producer cell keeps its pool on the tensor engine too.
    b, r, _ = model.import_state([env])
    assert float(r[0, j_c].min()) >= 0.0


@pytest.mark.parametrize("start_tick", [4 * 60 + 1, 4 * 250 + 2])
def test_ndm_grazing_matches_reference(device, start_tick):
    """Section 177: a grazing NDM (Holling II, Q10, mixed-layer-scaled
    attack rate, excretion back to the prey's nutrient pool) and the
    stress loss's regeneration must agree on both engines."""
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False,
                             "light_climate": LIGHT_CLIMATE},
                   light_saturation=150.0,
                   nutrient_pool=(NUTRIENTS, {
                       "nutrient_half_saturation": 0.5,
                       "nutrient_yield": 0.02,
                       "grazing_regeneration": 0.3,
                       "stress_loss_max": 0.05,
                       "stress_regeneration": 0.5}),
                   grazer=({"max_intake_rate": 0.4, "handling_time": 2.0,
                            "mixed_layer_reference_m": 10.0,
                            "food_threshold": 0.5},
                           {"growth_efficiency": 0.3,
                            "excretion_regeneration": 0.4,
                            "grazing_q10": 2.6, "grazing_t_ref": 10.0,
                            "temperature": {"layer": "upper",
                                            "monthly_c": TEMPERATURES_NDM}}),
                   mixed_layer_pairs={"a_preys_on_c": 10.0})
    assert env._has_light_pairs      # the DM pair's M_ref / M(t) table
    assert env._has_ndm_grazing and env.ndm_grazers[0]["prey"] == "c"
    model = _assert_parity(env, device, ticks=8)
    assert len(model.ndm_grazers) == 1
    assert env.intake_by_pred_prey["d"]["c"] > 0.0     # it really grazed



@pytest.mark.parametrize("start_tick", [4 * 60 + 1, 4 * 250 + 2])
def test_ndm_edibility_matches_reference(device, start_tick):
    """Section 182: the grazer sees only the edible share
    e0 + (1 - e0) N/(N + K) of a producer with a pool, on both engines."""
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False,
                             "light_climate": LIGHT_CLIMATE},
                   light_saturation=150.0,
                   nutrient_pool=(NUTRIENTS, {
                       "nutrient_half_saturation": 0.5,
                       "nutrient_yield": 0.02,
                       "grazing_regeneration": 0.3}),
                   grazer=({"max_intake_rate": 0.4, "handling_time": 2.0,
                            "mixed_layer_reference_m": 10.0,
                            "food_threshold": 0.5,
                            "edibility_floor": 0.2,
                            "edibility_half_saturation_umol_l": 1.0},
                           {"growth_efficiency": 0.3,
                            "excretion_regeneration": 0.4}),
                   mixed_layer_pairs={"a_preys_on_c": 10.0})
    assert env.ndm_grazers[0]["edibility"] is not None
    model = _assert_parity(env, device, ticks=8)
    assert model.ndm_grazers[0]["edibility"] is not None
    assert env.intake_by_pred_prey["d"]["c"] > 0.0

SUBSURFACE = {"bottom_m": 60.0, "light_attenuation_per_m": 0.12,
              "reservoir_umol_l": [6.3, 7.4, 7.5, 7.9, 7.2, 7.2, 6.4, 4.7,
                                   4.4, 2.5, 4.6, 4.5],
              "reservoir_exchange_m_per_day": 0.3}


@pytest.mark.parametrize("start_tick", [4 * 80 + 1, 4 * 290 + 2, 4 * 310 + 3])
def test_subsurface_layer_matches_reference(device, start_tick):
    """Section 180: the two-layer exchange (entrainment, detrainment,
    diffusion, the reservoir), the subsurface light schedule and the
    layer-specific half-saturation must agree on both engines; both
    pools ride in the reserve slots, so the state comparison checks
    them. Start ticks: shoaling in spring, deepening in autumn."""
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False,
                             "light_climate": LIGHT_CLIMATE},
                   light_saturation=150.0,
                   nutrient_pool=(NUTRIENTS, {
                       "nutrient_half_saturation": 0.5,
                       "nutrient_yield": 0.02,
                       "grazing_regeneration": 0.3}),
                   subsurface=(SUBSURFACE, 50.0),
                   mixed_layer_pairs={"a_preys_on_c": 10.0})
    assert env.layer_exchange["mixed"] == "c"
    assert env.layer_exchange["subsurface"] == "d"
    model = TensorEcosystem(env, device)
    _, r0, _ = model.import_state([env])
    started = model.initial_nutrients(torch.zeros_like(r0),
                                      model.light_starts([env]))
    j_d = model.ids.index("d")
    np.testing.assert_allclose(numpy(started[0, j_d]), numpy(r0[0, j_d]),
                               rtol=1e-6)
    model = _assert_parity(env, device, ticks=8)
    assert model.layer is not None


def test_subsurface_producer_without_the_layer_is_inert(device):
    """Section 180: without the two-layer column (here: no calendar at
    all) a subsurface producer does not grow, on both engines."""
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   inert_subsurface=True)
    assert env.inert_producers == frozenset({"d"})
    model = _assert_parity(env, device, ticks=4)
    assert float(model.growth[0, model.ids.index("d"), 0]) == 0.0


TEMPERATURES = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]


@pytest.mark.parametrize("start_tick", [0, 4 * 59 + 3])
@pytest.mark.parametrize("t_ref", [14.0, "annual_mean"])
def test_temperature_metabolism_matches_reference(device, start_tick, t_ref):
    """Section 143: both engines read the same year tick's Q10 multiplier.

    Only ``a`` has a Q10, so a mirror that scaled every DM, or read the
    table one tick off, diverges. The second start crosses the 28 Feb /
    1 Mar boundary.
    """
    env = make_env(min_split=0.0, extinction_factor=0.0,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False},
                   temperature=(2.2, t_ref, TEMPERATURES))
    model = _assert_parity(env, device, ticks=6)
    assert model.temperature_on
    i_a, i_b = env.dm_ids.index("a"), env.dm_ids.index("b")
    assert torch.all(model.metabolism_temp[:, i_b] == 1.0)
    assert float(model.metabolism_temp[start_tick, i_a]) != pytest.approx(1.0)
    assert model.in_dims == tuple(int(n) for n in env.per_dm_in_dim)


@pytest.mark.parametrize("start_tick", [0, 4 * 74 + 2, 4 * 250 + 1])
def test_ingestion_q10_matches_reference(device, start_tick):
    """Section 181: both engines scale DM ``a``'s whole intake by the
    same tick's Q10 multiplier, and only ``a``'s (no metabolism Q10 here,
    so the table must be built on its own)."""
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=0.6,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False},
                   ingestion=(2.0, TEMPERATURES))
    model = _assert_parity(env, device, ticks=6)
    assert model.ingestion_temp is not None and not model.temperature_on
    i_a, i_b = env.dm_ids.index("a"), env.dm_ids.index("b")
    assert torch.all(model.ingestion_temp[:, i_b] == 1.0)
    assert float(model.ingestion_temp[start_tick, i_a]) != pytest.approx(1.0)


@pytest.mark.parametrize("start_tick", [0, 4 * 74 + 2, 4 * 250 + 1])
def test_growth_q10_matches_reference(device, start_tick):
    """Section 183: both engines scale DM ``a``'s growth branch by the
    same tick's Q10 multiplier, and only ``a``'s."""
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=0.6,
                   daylight={"latitude_deg": 58.15, "tick_hours": 6,
                             "start_tick": start_tick, "random_start": False},
                   growth=(3.0, TEMPERATURES))
    model = _assert_parity(env, device, ticks=6)
    assert model.growth_temp is not None and not model.temperature_on
    j_a, j_b = env.global_fg_order.index("a"), env.global_fg_order.index("b")
    assert torch.all(model.growth_temp[:, j_b] == 1.0)
    assert float(model.growth_temp[start_tick, j_a]) != pytest.approx(1.0)


@pytest.mark.parametrize("kappa", [0.6, 1.0])
@pytest.mark.parametrize("handling", [0.0, 0.6])
def test_income_growth_matches_reference(device, kappa, handling):
    """Section 184: both engines grow DM ``a`` out of this tick's intake
    (kappa I - C, capped at g (s - u)) and never out of its reserve."""
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=handling,
                   income=(kappa, 6.0))
    model = _assert_parity(env, device, ticks=6)
    assert model.income_on
    i_a = env.dm_ids.index("a")
    assert float(env.dm_income_kappa[i_a]) == pytest.approx(kappa)
    assert not bool(model.income_off_growth[0, model.ids.index("b"), 0])
    assert bool(model.income_off_growth[0, model.ids.index("a"), 0])


@pytest.mark.parametrize("mortality", [False, True])
@pytest.mark.parametrize("capacity", [0.02, 0.1])
def test_regional_capacity_matches_reference(device, capacity, mortality):
    """Section 185: both engines remove c * grid-mean B of DM ``a`` per
    tick after M1, with or without --mortality; ``b`` is untouched."""
    env = make_env(mortality=mortality, min_split=0.0, extinction_factor=0.0,
                   capacity=capacity)
    model = _assert_parity(env, device, ticks=6)
    assert model.capacity_on
    assert float(model.capacity_rate[0, model.ids.index("a"), 0]) == pytest.approx(capacity)
    assert float(model.capacity_rate[0, model.ids.index("b"), 0]) == 0.0


@pytest.mark.parametrize("rho", [1.0, 3.0])
@pytest.mark.parametrize("calendar", [False, True])
def test_exposure_weighted_m1_matches_reference(device, rho, calendar):
    """Section 139: per-cell M1 from this tick's hiding and light.

    Only DM ``a`` carries the split; ``b`` must keep its scalar M1.
    """
    daylight = ({"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 2,
                 "random_start": False} if calendar else None)
    env = make_env(mortality=True, min_split=0.0, extinction_factor=0.0,
                   daylight=daylight,
                   m1_exposure={"m1_visual_share": 0.2,
                                "m1_tactile_share": 0.5,
                                "depth_risk_ratio": rho})
    model = _assert_parity(env, device, ticks=6)
    assert model.m1_exposure_on
    assert bool(model.m1_flag[0, model.ids.index("a"), 0])
    assert not bool(model.m1_flag[0, model.ids.index("b"), 0])


@pytest.mark.parametrize("fill", [0.0, 0.5])
@pytest.mark.parametrize("handling", [0.0, 0.2])
def test_reserve_as_food_matches_reference(device, fill, handling):
    """Section 140: eaten tonnes of ``b`` carry its reserve to ``a``."""
    env = make_env(handling=handling, min_split=0.0, extinction_factor=0.0,
                   reserve_food=fill)
    model = _assert_parity(env, device, ticks=6)
    assert model.reserve_food
    j = env.global_fg_order.index("b")
    i = env.dm_ids.index("a")
    # Static quality = (energy_content + fill * max reserve) * assim.
    assert float(env.energy_gain_mat[i, j]) == pytest.approx(
        (20 + fill * env.fgs["b"].max_energy_reserve) * 0.65, rel=1e-6)


@pytest.mark.parametrize("kinds", [{"a_preys_on_b": 3},
                                   {"a_preys_on_b": 3, "b_preys_on_c": 2}])
def test_pair_functional_response_matches_reference(device, kinds):
    """Section 141: Holling type per pair, not only per predator.

    ``a`` eats ``b`` and ``c`` (a generalist, Type II by default) and is
    forced to Type III on ``b`` only; ``b`` is a specialist on ``c``
    (Type III by default) and may be forced back to Type II.
    """
    env = make_env(handling=0.2, min_split=0.0, extinction_factor=0.0,
                   response_types=kinds)
    model = _assert_parity(env, device, ticks=6)
    i_a, i_b = env.dm_ids.index("a"), env.dm_ids.index("b")
    j_b, j_c = env.global_fg_order.index("b"), env.global_fg_order.index("c")
    mask = env._type3_pred_mask[:, :, 0, 0]
    assert mask[i_a, j_b] == 1.0 and mask[i_a, j_c] == 0.0
    assert mask[i_b, j_c] == (0.0 if "b_preys_on_c" in kinds else 1.0)
    torch.testing.assert_close(model.type3[0, :, :, 0].cpu(),
                               torch.as_tensor(mask))


@pytest.mark.parametrize("activation", ["sig", "tanh", "relu"])
def test_packed_policies_match_individual_networks(device, activation):
    model = TensorEcosystem(make_env(), device)
    bank = PolicyBank(model, hidden_dim=7, hidden_layers=3, activation=activation)
    flat = bank.flat_weights()
    candidates = [torch.stack((w, w + 0.01)) for w in flat]
    weights, biases = bank.pack(candidates)
    obs = torch.randn(2, model.D, model.C * 3, model.F, device=device)
    result = bank.forward(obs, weights, biases)
    for candidate in range(2):
        bank.install([w[candidate] for w in candidates])
        for d, fid in enumerate(model.dm_ids):
            expected = bank.policies[fid].net(obs[candidate, d, :, :model.in_dims[d]])
            torch.testing.assert_close(result[candidate, d], expected)


def test_batch_isolation_and_empty_cells(device):
    env = make_env(True, True)
    second = copy.deepcopy(env)
    for fg in second.fgs.values():
        fg.biomass *= 0
        fg.energy_reserve *= 0
    model = TensorEcosystem(env, device)
    b, r, h = model.import_state([env, second])
    probs = model.action_probabilities(torch.zeros(2, model.D, model.C, model.A, device=device), b, 1.0)
    batched = model.step(b, r, probs, 0, torch.zeros_like(b))
    for i in range(2):
        single = model.step(b[i:i+1], r[i:i+1], probs[i:i+1], 0, torch.zeros_like(b[i:i+1]))
        for actual, expected in zip(batched, single):
            torch.testing.assert_close(actual[i:i+1], expected)
    assert batched[0][1].sum() == 0


def test_seeding_with_injected_noise(device, monkeypatch):
    env = make_env()
    for f in ("c", "d"):
        env.fgs[f].seed_rate = 0.05
    model = TensorEcosystem(env, device)
    b, r, _ = model.import_state([env])
    noise = np.random.default_rng(12).uniform(-1, 1, (model.G, env.H, env.W))
    multipliers = model.tensor(np.power(10, noise).astype(np.float32))[None].flatten(2)
    b_next, r_next, _ = model.population(b, r, multipliers)
    for j, fid in enumerate(env.global_fg_order):
        monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: noise[j])
        fg = env.fgs[fid]
        if fg.is_decision_maker:
            population_change._apply_decision_maker_population_change(env, fid, fg)
        else:
            population_change._apply_non_decision_maker_population_change(env, fid, fg)
    population_change._clip_biomass_based_on_min_thresholds(env)
    population_change._zero_energy_in_empty_cells(env)
    expected = model.import_state([env])
    torch.testing.assert_close(b_next, expected[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(r_next, expected[1], atol=3e-6, rtol=3e-6)


def sparse_env(min_split=50.0, extinction_factor=1.2, amount=0.1, shape=(4, 5)):
    """Fixture where a 4-way split lands below the extinction threshold.

    ``make_env``'s biomass fills every cell, so a split always arrives in
    a cell that is viable on its own account and the suppression never
    fires. Concentrating the biomass in one cell instead makes the
    destinations empty: at 0.1 t, velocity 0.7 and four directions each
    receives 0.0175 t, well under ``thr = 1.2 * 0.05 = 0.06``.
    """
    env = make_env(min_split=min_split, extinction_factor=extinction_factor,
                   shape=shape)
    for fg in env.fgs.values():
        fg.biomass = np.zeros(fg.biomass.shape, dtype=np.float32)
        fg.biomass[1, 2] = amount
        fg.energy_reserve = fg.biomass * 2.0
    env.build_static_caches()
    return env


def uniform_move(env, model):
    """Probabilities that put the whole cell on a 4-way split."""
    probabilities = torch.zeros(1, model.D, model.A, model.C, device=model.device)
    probabilities[:, :, :4] = 0.25
    reference = ActionProbabilities(
        np.full((env.N_dm, 4, env.H, env.W), 0.25, np.float32),
        np.zeros((env.N_dm, env.H, env.W), np.float32),
        np.zeros((env.N_dm, env.N_all, env.H, env.W), np.float32))
    return probabilities, reference


def test_subthreshold_splits_are_suppressed_like_the_reference(device):
    """Section 69's split suppression was missing from the GPU engine.

    ``population`` zeroes every cell below ``thr =
    extinction_threshold_factor * min_split_biomass``, so a splitting
    decision maker bleeds biomass through that sweep. The reference
    cancels an outflow whose destination would still be sub-threshold
    (``movement.suppress_subthreshold_splits``); without the mirror,
    ``train_gpu.py`` optimised against a diffusion loss that
    ``inference.py`` does not have. Measured on this fixture: 0.33 vs.
    0.41 t after a single tick, i.e. a fifth of the population.

    Compared per tick over several ticks so the suppression is exercised
    on an evolving grid, not just on the hand-built initial state.
    """
    env = sparse_env()
    model = TensorEcosystem(env, device)
    assert model.split_thr_any and env._dm_split_thr_any
    np.testing.assert_allclose(numpy(model.split_thr).ravel(), env._dm_split_thr,
                               rtol=0, atol=0)
    b, r, _ = model.import_state([env])
    probabilities, reference = uniform_move(env, model)
    fired = False
    for tick in range(4):
        model.split_thr_any = False              # the pre-fix code path
        unsuppressed = model.step(b, r, probabilities, model.tensor(tick),
                                  torch.zeros_like(b))[0]
        model.split_thr_any = True
        b, r, _, _, _ = model.step(b, r, probabilities, model.tensor(tick),
                                   torch.zeros_like(b))
        env.step(reference)
        expected = model.import_state([env])
        np.testing.assert_allclose(numpy(b), numpy(expected[0]), atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(numpy(r), numpy(expected[1]), atol=2e-7, rtol=2e-6)
        fired = fired or float(b.sum()) > float(unsuppressed.sum()) + 1e-3
    assert fired, "the suppression never fired; the test would be vacuous"
    assert float(b.sum()) > 0, "everything died; the comparison is against zeros"


def test_residues_and_small_emigration_match_the_reference(device):
    """Section 156's two rules, mirrored from ``movement.py``.

    A cell moving 98 % of its biomass one way keeps a residue below
    ``thr`` (``follow_residues``), and with migration a top-row cell
    sending a small share north emigrates a pool below ``thr``
    (``cancel_emigration``). Both used to end in the extinction sweep.
    Compared per tick over several ticks; ``fired`` proves the rules
    changed the outcome on this fixture.
    """
    env = make_env(migration=True, min_split=50.0, extinction_factor=1.2)
    rng = np.random.default_rng(156)
    for fg in env.fgs.values():
        b = rng.uniform(0.0, 0.3, fg.biomass.shape).astype(np.float32)
        b *= (rng.random(b.shape) > 0.5)
        fg.biomass = b
        fg.energy_reserve = b * 2.0
    env.build_static_caches()
    model = TensorEcosystem(env, device)
    direction = rng.integers(0, 4, (env.H, env.W))
    share = rng.choice(np.array([0.3, 0.98], np.float32), (env.H, env.W))
    move = np.zeros((env.N_dm, 4, env.H, env.W), np.float32)
    for d in range(4):
        move[:, d, direction == d] = share[direction == d]
    rest = (1.0 - move.sum(1)).astype(np.float32)
    reference = ActionProbabilities(
        move, rest, np.zeros((env.N_dm, env.N_all, env.H, env.W), np.float32))
    probabilities = torch.zeros(1, model.D, model.A, model.C, device=model.device)
    probabilities[0, :, :4] = model.tensor(move.reshape(env.N_dm, 4, -1))
    probabilities[0, :, 4] = model.tensor(rest.reshape(env.N_dm, -1))
    b, r, _ = model.import_state([env])
    rules = (model.follow_residues, model.cancel_emigration)
    fired = False
    for tick in range(4):
        # The section-69-only path, for the ``fired`` check.
        model.follow_residues = lambda b_out, b_total, r_total: (
            b_total, r_total, torch.zeros_like(b_total, dtype=torch.bool),
            torch.zeros_like(b_out))
        model.cancel_emigration = lambda b_out, r_out: (b_out, r_out)
        without = model.step(b, r, probabilities, model.tensor(tick),
                             torch.zeros_like(b))[0]
        model.follow_residues, model.cancel_emigration = rules
        b, r, _, _, _ = model.step(b, r, probabilities, model.tensor(tick),
                                   torch.zeros_like(b))
        env.step(reference)
        expected = model.import_state([env])
        np.testing.assert_allclose(numpy(b), numpy(expected[0]), atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(numpy(r), numpy(expected[1]), atol=2e-7, rtol=2e-6)
        fired = fired or float(b.sum()) > float(without.sum()) + 1e-4
    assert fired, "neither rule fired; the test would be vacuous"
    assert float(b.sum()) > 0, "everything died; the comparison is against zeros"


def test_immigration_matches_threshold_concentration(device):
    env = make_env(True)
    model = TensorEcosystem(env, device)
    for amount in (0.0, 0.01, 0.2, 20.0):
        b = np.full(env.N_dm, amount, dtype=np.float32)
        r = b * 2
        expected = movement._concentrate_immigration(env, b, r, env._edge_imm_weights)
        actual = model.immigration(model.tensor(b)[None], model.tensor(r)[None])
        for ours, ref in zip(actual, expected):
            np.testing.assert_allclose(numpy(ours[0]).reshape(ref.shape), ref, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("activation", ["sig", "relu", "tanh"])
def test_cpu_batched_activation_matches_individual_policies(activation):
    env = make_env()
    model = TensorEcosystem(env, "cpu")
    bank = PolicyBank(model, activation=activation)
    env.policies = bank.policies
    env.build_static_caches()
    obs = torch.randn(model.D, model.C, model.F)
    actual = policies.batched_policy_forward(env, obs, return_logits=True)
    for d, fid in enumerate(model.dm_ids):
        expected = bank.policies[fid].net(obs[d, :, :model.in_dims[d]])
        torch.testing.assert_close(actual[d], expected)
