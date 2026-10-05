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
             light_saturation=None, m1_exposure=None, reserve_food=None):
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
        for inter_id, ratio in (dark_ratios or {}).items():
            if inter_id in params["interaction"]:
                params["interaction"][inter_id]["dark_ratio"] = ratio
        if daylight:
            params["daylight"] = dict(daylight)
        if light_saturation and fid == "c":
            params["light_saturation"] = light_saturation
        if m1_exposure and fid == "a":
            params.update(m1_exposure)
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
                               mortality_multiplier=mortality_multiplier)
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
    for tick in range(ticks):
        tol = _parity_tol(tick)
        light = model.light_index(starts, torch.tensor(tick, device=model.device))
        obs = model.observations(b, r, hidden, light)
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
                                        light_index=light)
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
    np.testing.assert_allclose(
        numpy(model.mortality_keep).reshape(-1),
        [expected if env.fgs[f].is_decision_maker else 1.0 for f in model.ids],
        rtol=1e-6)


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
