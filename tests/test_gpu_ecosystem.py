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


def make_env(migration=False, mortality=False, handling=0.2, shape=(4, 5)):
    rng = np.random.default_rng(42)
    groups = {}
    for i, fid in enumerate(("a", "b", "c", "d")):
        params = dict(is_decision_maker=i < 2, movement_speed=[0.7, 0.0, 0, 0][i],
                      max_energy_reserve=10 + i, resting_metabolism=0.05,
                      movement_cost=2.5, feeding_cost=1.2, resting_cost=0.3,
                      growth_rate=0.2, starve_rate=0.7, maintenance_level=0.4,
                      natural_mortality=0.1, visibility_floor=0.25,
                      min_split_biomass=200, extinction_threshold_factor=0.3,
                      max_carrying_capacity=12, seasonal_amplitude=0.2,
                      seasonal_period=15, seed_rate=0.0, energy_content=20,
                      observes=["b", "c"] if i == 0 else ["a"],
                      max_intake_rate=0.8,
                      menu=["b", "c"] if i == 0 else (["c"] if i == 1 else []))
        params["interaction"] = {f"{fid}_preys_on_{prey}": dict(
            preys_on=True, assimilation_factor=0.65, handling_time=handling)
            for prey in params["menu"]}
        fg = FunctionalGroup(fid, params)
        fg.initialize_state(shape, initial_biomass=rng.uniform(0.01, 3, shape),
                            randomize_energy=True, rng=rng)
        fg.biomass.flat[0] = 0
        fg.energy_reserve.flat[0] = 0
        groups[fid] = fg
    env = EcosystemEnvironment(dict(height=shape[0], width=shape[1]), groups,
                               migration=migration, apply_natural_mortality=mortality)
    env._season_phase = {f: 2.0 for f in groups}
    return env


def numpy(t):
    return t.detach().cpu().numpy()


def manual_actions(probs, env):
    p = numpy(probs[0]).reshape(env.N_dm, 5 + env.N_all, env.H, env.W)
    return ActionProbabilities(p[:, :4], p[:, 4], p[:, 5:])


@pytest.mark.parametrize("migration,mortality", [(False, False), (True, True)])
@pytest.mark.parametrize("handling", [0.0, 0.2])
def test_full_ticks_match_reference(device, migration, mortality, handling):
    env = make_env(migration, mortality, handling)
    model = TensorEcosystem(env, device)
    b, r, hidden, phase = model.import_state([env])
    bank = PolicyBank(model, hidden_dim=9, hidden_layers=2, seed=13)
    env.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    env.build_static_caches()
    packed = bank.pack([w[None] for w in bank.flat_weights()])
    for tick in range(12):
        obs = model.observations(b, r, hidden)
        observation = env.get_observation()
        np.testing.assert_allclose(numpy(obs[0]), observation.features, atol=2e-5, rtol=2e-5)
        logits = bank.forward(obs, *packed)
        probs = model.action_probabilities(logits, b, model.tensor(1.0))
        cpu_actions = env.policy_controller.forward(observation)
        ours = manual_actions(probs, env)
        for field in ("move", "rest", "eat"):
            np.testing.assert_allclose(getattr(ours, field), getattr(cpu_actions, field), atol=2e-6, rtol=2e-5)
        b, r, hidden, _, _ = model.step(b, r, probs, model.tensor(tick), phase, torch.zeros_like(b))
        env.step(cpu_actions)
        expected = model.import_state([env])
        np.testing.assert_allclose(numpy(b), numpy(expected[0]), atol=3e-5, rtol=4e-5)
        np.testing.assert_allclose(numpy(r), numpy(expected[1]), atol=4e-5, rtol=4e-5)
        assert torch.isfinite(b).all() and (b >= 0).all()


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
    b, r, h, phase = model.import_state([env, second])
    probs = model.action_probabilities(torch.zeros(2, model.D, model.C, model.A, device=device), b, 1.0)
    batched = model.step(b, r, probs, 0, phase, torch.zeros_like(b))
    for i in range(2):
        single = model.step(b[i:i+1], r[i:i+1], probs[i:i+1], 0, phase[i:i+1], torch.zeros_like(b[i:i+1]))
        for actual, expected in zip(batched, single):
            torch.testing.assert_close(actual[i:i+1], expected)
    assert batched[0][1].sum() == 0


def test_seeding_with_injected_noise(device, monkeypatch):
    env = make_env()
    for f in ("c", "d"):
        env.fgs[f].seed_rate = 0.05
    model = TensorEcosystem(env, device)
    b, r, _, phase = model.import_state([env])
    noise = np.random.default_rng(12).uniform(-1, 1, (model.G, env.H, env.W))
    multipliers = model.tensor(np.power(10, noise).astype(np.float32))[None].flatten(2)
    b_next, r_next, _ = model.population(b, r, 0, phase, multipliers)
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
