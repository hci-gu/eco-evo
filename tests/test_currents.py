"""Conservation, reference parity, and deterministic training with passive drift."""

import copy
import argparse

import numpy as np
import pytest
import torch

from lib.environments.ecosystem import EcosystemEnvironment
from lib.environments.ecosystem_env.currents import (
    CurrentConfig, add_current_arguments, apply_currents, current_options,
    direction_fractions,
)
from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from lib.gpu.trainer import TensorARSTrainer
from lib.runners.population_stability import PopulationStability
from lib.runners.trainer import ARSTrainer
from lib.runners.training_progress import inference_config, build_inference_env
from lib.world.functional_group import FunctionalGroup
from test_gpu_ecosystem import make_env, DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_current_vectors_are_bounded_reproducible_and_change_smoothly(device):
    config = CurrentConfig(strength=0.4, period=20, seed=31)
    ticks = [0, 1, 19, 20, 21, 99, 10000]
    keys = torch.tensor([0, 17, 0xFFFFFFFF], device=device, dtype=torch.int64)
    for tick in ticks:
        actual = torch.stack(direction_fractions(torch.tensor(tick, device=device), keys, config), 1)
        expected = np.array([direction_fractions(tick, seed, config) for seed in (0, 17, 0xFFFFFFFF)])
        np.testing.assert_allclose(actual.cpu(), expected, atol=4e-8)
        assert (actual >= 0).all()
        assert (actual.sum(1) <= config.strength).all()
    first = np.array(direction_fractions(0, 17, config))
    assert not np.allclose(first, direction_fractions(20, 17, config))
    assert not np.allclose(first, direction_fractions(0, 18, config))
    assert np.max(np.abs(np.array(direction_fractions(19, 17, config)) -
                         direction_fractions(20, 17, config))) <= config.strength / config.period


@pytest.mark.parametrize("device", DEVICES)
def test_cloud_field_is_spatial_bounded_and_matches_batched_worlds(device):
    config = CurrentConfig(strength=0.8, scale=7.5, period=13, seed=31)
    y, x = np.indices((19, 23), dtype=np.float32)
    keys = (0, 17, 0xFFFFFFFF)
    for tick in (0, 1, 20, 99, 10000):
        expected = np.stack([direction_fractions(tick, seed, config, x, y) for seed in keys])
        actual = torch.stack(direction_fractions(
            torch.tensor(tick, device=device),
            torch.tensor(keys, device=device, dtype=torch.int64)[:, None, None], config,
            torch.tensor(x, device=device), torch.tensor(y, device=device)), dim=1)
        np.testing.assert_allclose(actual.cpu(), expected, atol=1e-7, rtol=3e-6)
        assert (expected >= 0).all()
        assert (expected.sum(axis=1) <= config.strength).all()
        assert np.ptp(expected[0, 1]) > 0.05
        assert not np.allclose(expected[0], expected[1])
        np.testing.assert_array_equal(expected[:, 0], 0)
        np.testing.assert_array_equal(expected[:, 3], 0)
        np.testing.assert_array_equal(expected[:, 1], expected[:, 2])


def test_cloud_scrolls_continuously_in_positive_xy_without_rng_consumption():
    config = CurrentConfig(strength=0.8, scale=8, period=20)
    y, x = np.indices((25, 31), dtype=np.float32)
    rng_state = np.random.get_state()
    first = np.asarray(direction_fractions(0, 17, config, x, y))
    later = np.asarray(direction_fractions(config.period, 17, config, x, y))
    # One cell of positive XY scrolling: the old pixel reappears southeast.
    np.testing.assert_array_equal(first[:, :-1, :-1], later[:, 1:, 1:])
    # A fractional tick shifts by a fractional cell, without integer rolling.
    half = np.asarray(direction_fractions(0.5, 17, config, x, y))
    assert not np.array_equal(first, half)
    assert np.max(np.abs(first - half)) < 0.005
    np.testing.assert_array_equal(first, direction_fractions(0, 17, config, x, y))
    after = np.random.get_state()
    assert rng_state[0] == after[0] and rng_state[2:] == after[2:]
    np.testing.assert_array_equal(rng_state[1], after[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("response", [0.0, 0.25, 1.0])
def test_current_response_scales_flux_and_energy_together(device, response):
    env = make_env()
    env.currents = CurrentConfig(strength=0.8)
    fg = env.fgs["c"]
    fg.current_response = response
    fg.biomass[:] = fg.energy_reserve[:] = 0
    fg.biomass[1, 1], fg.energy_reserve[1, 1] = 100, 200
    full = copy.deepcopy(env)
    full.fgs["c"].current_response = 1.0
    before_b, before_r = fg.biomass.copy(), fg.energy_reserve.copy()
    model = TensorEcosystem(env, device)
    b, r, _, _ = model.import_state([env])
    actual_b, actual_r = model.advect(b, r, model.tensor(0, torch.int64))
    apply_currents(env)
    apply_currents(full)
    expected_b, expected_r, _, _ = model.import_state([env])
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_r, expected_r)
    np.testing.assert_allclose(fg.biomass - before_b,
                               response * (full.fgs["c"].biomass - before_b), atol=1e-5)
    np.testing.assert_allclose(fg.energy_reserve, fg.biomass * 2)
    assert (fg.biomass[1, 2] > 0) == (response > 0)
    assert (fg.biomass[2, 1] > 0) == (response > 0)
    assert fg.biomass[0, 1] == fg.biomass[1, 0] == 0
    if response == 0:
        np.testing.assert_array_equal(fg.biomass, before_b)
        np.testing.assert_array_equal(fg.energy_reserve, before_r)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (4, 5)])
@pytest.mark.parametrize("migration", [False, True])
def test_current_transport_works_without_decision_makers_and_keeps_edges(shape, migration):
    fg = FunctionalGroup("drifter", {"growth_rate": 0, "max_energy_reserve": 10})
    fg.initialize_state(shape, initial_biomass=np.ones(shape, dtype=np.float32))
    env = EcosystemEnvironment(dict(height=shape[0], width=shape[1]), {"drifter": fg},
                               currents=CurrentConfig(strength=1, period=1), migration=migration)
    before = fg.biomass.copy()
    for tick in range(25):
        env.tick_count = tick
        apply_currents(env)
    assert env.N_dm == 0
    assert fg.biomass.sum() == pytest.approx(before.sum(), rel=2e-6)
    assert (fg.biomass >= 0).all()
    if shape == (1, 1):
        np.testing.assert_array_equal(fg.biomass, before)
    elif not migration:
        assert fg.biomass[-1, -1] > before[-1, -1]
    else:
        assert fg.biomass[0, 0] > 0, "Immigration keeps repopulating the upstream edge"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("migration", [False, True])
def test_current_edge_outflow_uses_swimmer_migration_weights(device, migration):
    base = make_env()
    # Unequal DM/NDM counts catch migration code that assumes DM array shapes.
    env = EcosystemEnvironment(
        dict(height=base.H, width=base.W),
        {fid: fg for fid, fg in base.fgs.items() if fid != "d"},
        migration=migration, currents=CurrentConfig(strength=1, seed=4))
    habitat = np.ones((env.H, env.W), dtype=np.float32)
    habitat[0, 0] = 0  # No immigrants may land on this boundary cell.
    habitat[0, 1] = 3  # Immigration weights, not just a binary mask.
    env.grid.add_map("accessibility", habitat)
    env.build_static_caches()
    fg = env.fgs["c"]
    fg.min_split_biomass = 0
    fg.biomass[:] = fg.energy_reserve[:] = 0
    fg.biomass[-1, -1], fg.energy_reserve[-1, -1] = 100, 200
    before = fg.biomass.copy()
    model = TensorEcosystem(env, device)
    b, r, _, _ = model.import_state([env])
    actual_b, actual_r = model.advect(b, r, model.tensor(0, torch.int64))
    apply_currents(env)
    if migration:
        assert 0 < fg.biomass[-1, -1] < 100, "Current biomass remains trapped at the edge"
        assert fg.biomass[0, 1] > 0, "Outgoing biomass must re-enter at other edges"
        assert fg.biomass[0, 1] == pytest.approx(3 * fg.biomass[0, 2], rel=1e-6)
        assert fg.biomass[0, 0] == 0
        np.testing.assert_array_equal(fg.biomass[1:-1, 1:-1], 0)
    else:
        np.testing.assert_array_equal(fg.biomass, before)
    assert fg.biomass.sum() == pytest.approx(100, rel=1e-6)
    np.testing.assert_allclose(fg.energy_reserve, fg.biomass * 2)
    expected_b, expected_r, _, _ = model.import_state([env])
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_r, expected_r)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("migration", [False, True])
def test_transport_conserves_biomass_reserves_and_respects_habitat(device, migration):
    env = make_env(migration=migration)
    env.currents = CurrentConfig(strength=1, period=3, seed=4)
    env.current_world_seed = 17
    habitat = np.ones((env.H, env.W), dtype=np.float32)
    habitat[1, 2] = 0
    env.grid.add_map("accessibility", habitat)
    env.build_static_caches()
    initial_b = {f: fg.biomass.copy() for f, fg in env.fgs.items()}
    initial_r = {f: fg.energy_reserve.copy() for f, fg in env.fgs.items()}
    model = TensorEcosystem(env, device)
    b, r, _, _ = model.import_state([env])
    for tick in range(60):
        env.tick_count = tick
        apply_currents(env)
        b, r = model.advect(b, r, model.tensor(tick, torch.int64))
    expected_b, expected_r, _, _ = model.import_state([env])
    torch.testing.assert_close(b, expected_b, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(r, expected_r, atol=5e-5, rtol=3e-5)
    for f, fg in env.fgs.items():
        assert fg.biomass.sum() == pytest.approx(initial_b[f].sum(), rel=3e-6)
        assert fg.energy_reserve.sum() == pytest.approx(initial_r[f].sum(), rel=3e-6)
        assert fg.biomass[1, 2] == fg.energy_reserve[1, 2] == 0
        assert (fg.biomass >= 0).all() and (fg.energy_reserve >= 0).all()
        if fg.is_decision_maker:
            np.testing.assert_array_equal(fg.biomass, initial_b[f])
            np.testing.assert_array_equal(fg.energy_reserve, initial_r[f])
        else:
            assert not np.allclose(fg.biomass, initial_b[f])


@pytest.mark.parametrize("config", [None, CurrentConfig(strength=0)])
def test_disabled_currents_are_exact_noop_and_do_not_consume_randomness(config):
    env = make_env()
    env.currents = config
    before = {f: (g.biomass.copy(), g.energy_reserve.copy()) for f, g in env.fgs.items()}
    state = np.random.get_state()
    apply_currents(env)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    for f, (b, r) in before.items():
        np.testing.assert_array_equal(env.fgs[f].biomass, b)
        np.testing.assert_array_equal(env.fgs[f].energy_reserve, r)


def test_cpu_and_tensor_rollouts_match_with_currents_and_population_stability():
    env = make_env()
    env.currents = CurrentConfig(period=3, seed=47)
    model = TensorEcosystem(env, "cpu")
    bank = PolicyBank(model, hidden_dim=7)
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    reference = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, hidden_dim=7,
                           population_stability=PopulationStability())
    reference.policies = {f: copy.deepcopy(p) for f, p in bank.policies.items()}
    runner = RolloutRunner(model, bank, 1, 1, execution="eager",
                           population_stability=PopulationStability())
    b, r, _, phase = model.import_state([env])
    mean, var = np.zeros((model.D, model.F), np.float32), np.ones((model.D, model.F), np.float32)
    runner.reset(b, r, phase, torch.zeros(1, dtype=torch.int64), 15,
                 model.tensor(mean), model.tensor(var), model.tensor(1.0))
    runner.run(15)
    fitness, samples, _ = reference._evaluate_coevo(list(model.dm_ids), 15,
        obs_mean=mean, obs_var=var, dm_ids_for_norm=list(model.dm_ids))
    np.testing.assert_allclose(runner.results(15)[0][0], list(fitness.values()), atol=4e-6)
    np.testing.assert_allclose(runner.obs_sum[0], samples[0], atol=0.003, rtol=3e-5)


@pytest.mark.parametrize("migration", [False, True])
def test_current_training_is_chunk_independent_and_traces_without_readback(monkeypatch, migration):
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6),
                                          currents=CurrentConfig(period=3), migration=migration))
    options = dict(device="cpu", execution="eager", n_deltas=3, worlds=2,
                   population_stability=PopulationStability())
    whole = TensorARSTrainer(spec, **options)
    chunked = TensorARSTrainer(spec, pairs_per_batch=2, **options)
    for _ in range(2):
        whole.train_step(n_eval_ticks=7)
        chunked.train_step(n_eval_ticks=7)
        torch.testing.assert_close(whole.rewards, chunked.rewards, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(whole.obs_mean, chunked.obs_mean, atol=3e-6, rtol=3e-5)
    runner = whole.runner
    snapshot = [v.clone() for v in runner.state_buffers]
    runner._tick()
    expected = [v.clone() for v in runner.state_buffers]
    for v, saved in zip(runner.state_buffers, snapshot):
        v.copy_(saved)
    torch.compile(runner._tick, backend="eager", fullgraph=True)()
    for actual, reference in zip(runner.state_buffers, expected):
        torch.testing.assert_close(actual, reference)
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor readback during training")
    for name in ("cpu", "numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    whole.train_step(n_eval_ticks=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
@pytest.mark.parametrize("execution", ["cuda-graph", "compile-graph"])
def test_current_training_captures_into_a_cuda_graph(execution):
    """Drift must be expressible inside a captured CUDA graph.

    Building the hash constants with ``torch.as_tensor`` placed them on the
    host and copied them to the device, which CUDA rejects while a stream is
    capturing (``cudaErrorStreamCaptureUnsupported``): --currents on worked
    under eager and compile but aborted capture, i.e. exactly the default
    --execution of train_gpu.py.
    """
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6),
                                          currents=CurrentConfig(period=3, seed=9), migration=True))
    options = dict(device="cuda", n_deltas=2, worlds=2)
    eager = TensorARSTrainer(spec, execution="eager", **options)
    captured = TensorARSTrainer(spec, execution=execution, graph_ticks=3, **options)
    for ticks in (7, 4):  # remainder ticks and a changed horizon
        eager.train_step(n_eval_ticks=ticks)
        captured.train_step(n_eval_ticks=ticks)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.rewards, captured.rewards, rtol=1e-4, atol=1e-5)


def test_builders_and_progress_inference_keep_current_configuration():
    config = CurrentConfig(period=7, seed=22, scale=9.5)
    builder = EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6), currents=config)
    assert builder.with_world(3).currents == config
    assert builder.with_world(3)(seed=5).current_world_seed == 6
    trainer = TensorARSTrainer(ProjectSpec(builder), device="cpu", execution="eager", n_deltas=2)
    inference = build_inference_env(inference_config(trainer, "gpu"), seed=42)
    assert inference.currents == config and inference.current_world_seed == 42
    from train import _make_env_builder, _ProbeEnvBuilder
    cpu_builder = _make_env_builder(project_path="mareld2.yaml", grid_size=(5, 6),
                                    impact_vars=[], impact_ranges={}, currents=config)
    assert cpu_builder.with_world(1, 3).currents == config
    assert cpu_builder.with_world(1, 3)(seed=5).current_world_seed == 6
    assert _ProbeEnvBuilder("mareld2.yaml", (5, 6), currents=config)().currents == config
    env = builder(seed=5)
    assert env.fgs["phytoplankton"].current_response == 1
    assert env.fgs["benthic_community"].current_response == 0


def test_current_cli_and_metadata_keep_cloud_settings():
    parser = argparse.ArgumentParser()
    add_current_arguments(parser)
    config = current_options(parser.parse_args([
        "--currents", "on", "--current-strength", "0.3", "--current-period", "5",
        "--current_scale", "9.5", "--current-seed", "31"]))
    assert config == CurrentConfig(strength=0.3, period=5, seed=31, scale=9.5)
    assert CurrentConfig(**config.metadata()) == config
    assert current_options(parser.parse_args([])) is None


@pytest.mark.parametrize("response", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_current_response_rejected(response):
    with pytest.raises(ValueError, match="current_response"):
        FunctionalGroup("drifter", {"current_response": response})


@pytest.mark.parametrize("kwargs", [{"strength": -0.1}, {"strength": 1.1},
                                   {"strength": float("nan")}, {"period": 0},
                                   {"period": 1.5}, {"seed": -1},
                                   {"scale": 0}, {"scale": 0.5},
                                   {"scale": float("nan")}, {"scale": float("inf")}])
def test_invalid_current_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        CurrentConfig(**kwargs)
