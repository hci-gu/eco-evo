"""Opt-in replenished food, reflecting trajectories, and trainer parity."""

import argparse
import copy
import pickle

import numpy as np
import pytest
import torch

from lib.environments.ecosystem_env import debug_food
from lib.environments.ecosystem_env.currents import CurrentConfig
from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.trainer import TensorARSTrainer
from lib.gpu.policy import PolicyBank
from lib.runners.training_progress import inference_config, build_inference_env
from test_gpu_ecosystem import DEVICES, manual_actions


def test_world_seeds_randomize_heading_not_just_start_position():
    y, x = np.indices((80, 80), dtype=np.float32)
    x, y = x.ravel(), y.ravel()
    config = debug_food.FoodBlobConfig(speed=.5, radius=4)
    headings = []
    for seed in (17, 42, 139, 827, 20260530, 123456789):
        centres = []
        for tick in (0, 1):
            weights = debug_food.blob_weights(x, y, 80, 80, tick, seed,
                                               config, np.ones_like(x, dtype=bool))
            centres.append(np.array([weights @ x, weights @ y]))
        delta = np.abs(centres[1] - centres[0])
        headings.append(delta / np.linalg.norm(delta))
    # Previously every world followed the same absolute 0.8/0.6 heading.
    assert np.std(np.array(headings)[:, 0]) > .05


@pytest.mark.parametrize("device", DEVICES)
def test_blob_bounces_and_cpu_tensor_fields_agree(device):
    y, x = np.indices((16, 20), dtype=np.float32)
    x, y = x.ravel(), y.ravel()
    allowed = np.ones_like(x, dtype=bool)
    config = debug_food.FoodBlobConfig(speed=1, radius=3)
    centres = []
    for tick in range(80):
        expected = debug_food.blob_weights(x, y, 16, 20, tick, 0, config, allowed)
        actual = debug_food.blob_weights(
            torch.tensor(x, device=device), torch.tensor(y, device=device), 16, 20,
            torch.tensor(tick, device=device), torch.tensor([[0]], device=device), config,
            torch.tensor(allowed, device=device))
        np.testing.assert_allclose(actual.cpu()[0], expected, atol=2e-7, rtol=5e-5)
        assert expected.sum() == pytest.approx(1, abs=2e-7)
        assert (expected >= 0).all() and np.count_nonzero(expected) < 40
        centres.append((float(expected @ x), float(expected @ y)))
    centres = np.array(centres)
    for axis in (0, 1):
        shifts = np.diff(centres[:, axis])
        assert shifts.min() < -0.2 and shifts.max() > 0.2
        assert np.max(np.abs(shifts)) < 1.01
    assert centres[:, 0].min() >= 2.9 and centres[:, 0].max() <= 16.1
    assert centres[:, 1].min() >= 2.9 and centres[:, 1].max() <= 12.1


@pytest.mark.parametrize("device", DEVICES)
def test_seeded_centres_reproduce_across_backends_and_stay_in_bounds(device):
    keys = np.array([0, 17, 42, 20260530, 2**32 - 1], dtype=np.int64)
    config = debug_food.FoodBlobConfig(speed=.1, seed=13)
    times = np.arange(5001, dtype=np.float32)
    expected = debug_food.blob_center(24, 24, times[None], keys[:, None], config)
    actual = debug_food.blob_center(
        24, 24, torch.tensor(times[None], device=device),
        torch.tensor(keys[:, None], device=device), config)
    for reference, tensor in zip(expected, actual):
        np.testing.assert_allclose(tensor.cpu(), reference, atol=1e-4, rtol=1e-5)
        assert reference.min() >= 4 and reference.max() <= 19
    steps = np.hypot(np.diff(expected[0]), np.diff(expected[1]))
    assert steps.max() <= config.speed + 1e-4
    assert np.median(steps) == pytest.approx(config.speed, abs=1e-5)
    # Random access/replay must not depend on any preceding calls or RNG state.
    for tick in (5000, 1, 97, 0):
        single = debug_food.blob_center(24, 24, tick, keys, config)
        for axis in (0, 1):
            np.testing.assert_array_equal(single[axis], expected[axis][:, tick])
    shifted = debug_food.blob_center(24, 24, times[None], keys[:, None],
                                     debug_food.FoodBlobConfig(speed=.1, seed=14))
    assert not np.allclose(shifted, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (3, 3)])
def test_seeded_blob_handles_tiny_grids(shape):
    y, x = np.indices(shape, dtype=np.float32)
    field = debug_food.blob_weights(x.ravel(), y.ravel(), *shape, 5000, 42,
                                    debug_food.FoodBlobConfig(),
                                    np.ones(x.size, dtype=bool))
    assert np.isfinite(field).all()
    assert field.sum() == pytest.approx(1)


@pytest.mark.parametrize("device", DEVICES)
def test_blob_respects_habitat_even_when_footprint_is_entirely_blocked(device):
    y, x = np.indices((8, 10), dtype=np.float32)
    allowed = np.zeros_like(x, dtype=bool)
    allowed[-1, -1] = True
    config = debug_food.FoodBlobConfig(radius=1)
    expected = debug_food.blob_weights(x.ravel(), y.ravel(), 8, 10, 0, 0, config, allowed.ravel())
    actual = debug_food.blob_weights(
        torch.tensor(x.ravel(), device=device), torch.tensor(y.ravel(), device=device),
        8, 10, torch.tensor(0, device=device), torch.tensor([[0]], device=device), config,
        torch.tensor(allowed.ravel(), device=device))
    np.testing.assert_array_equal(actual.cpu()[0], expected)
    assert expected[-1] == 1 and expected.sum() == 1


@pytest.mark.parametrize("device", DEVICES)
def test_full_ticks_keep_food_constant_and_match_cpu(device):
    env = EnvironmentBuilder("mareld2.yaml", grid=(12, 15), migration=True,
                             currents=CurrentConfig(strength=1),
                             food_blobs=debug_food.FoodBlobConfig(speed=0.5))(seed=17)
    initial = {f: (env.fgs[f].biomass.sum(), env.fgs[f].energy_reserve.sum())
               for f in debug_food.FOOD_IDS}
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    env.policies = {fid: copy.deepcopy(p).cpu() for fid, p in bank.policies.items()}
    env.build_static_caches()
    b, r, hidden, phase = model.import_state([env])
    packed = bank.pack([w[None] for w in bank.flat_weights()])
    totals = (b[:, model.food_blob_index].sum(-1, keepdim=True),
              r[:, model.food_blob_index].sum(-1, keepdim=True))
    for tick in range(40):
        logits = bank.forward(model.observations(b, r, hidden), *packed)
        actions = model.action_probabilities(logits, b, model.tensor(1.0))
        env.step(manual_actions(actions, env))
        b, r, hidden, _, _ = model.step(b, r, actions, model.tensor(tick), phase,
                                        torch.zeros_like(b), food_blob_totals=totals)
        expected_b, expected_r, _, _ = model.import_state([env])
        torch.testing.assert_close(b, expected_b, atol=5e-4, rtol=1e-4)
        torch.testing.assert_close(r, expected_r, atol=0.02, rtol=1e-4)
        for f, (b0, r0) in initial.items():
            assert env.fgs[f].biomass.sum() == pytest.approx(b0, rel=3e-6)
            assert env.fgs[f].energy_reserve.sum() == pytest.approx(r0, rel=3e-6)


def test_food_is_restored_after_complete_consumption_and_stationary_control():
    env = EnvironmentBuilder("mareld2.yaml", grid=(12, 15),
                             food_blobs=debug_food.FoodBlobConfig(speed=0))(seed=17)
    before = {f: env.fgs[f].biomass.copy() for f in debug_food.FOOD_IDS}
    for f in debug_food.FOOD_IDS:
        env.fgs[f].biomass.fill(0)
        env.fgs[f].energy_reserve.fill(0)
    debug_food.apply(env, 500)
    for f in debug_food.FOOD_IDS:
        np.testing.assert_array_equal(env.fgs[f].biomass, before[f])


def test_blob_training_is_chunk_independent_compilable_and_has_no_readback(monkeypatch):
    spec = ProjectSpec(EnvironmentBuilder("mareld2.yaml", grid=(8, 9),
                                          food_blobs=debug_food.FoodBlobConfig(speed=0.5)))
    kwargs = dict(device="cpu", execution="eager", n_deltas=3, worlds=2)
    whole = TensorARSTrainer(spec, **kwargs)
    chunked = TensorARSTrainer(spec, pairs_per_batch=2, **kwargs)
    try:
        whole.train_step(n_eval_ticks=7)
        chunked.train_step(n_eval_ticks=7)
        torch.testing.assert_close(whole.rewards, chunked.rewards, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(whole.obs_mean, chunked.obs_mean, atol=3e-6, rtol=3e-5)
        runner = whole.runner
        # Paired ARS signs see the identical scripted food trajectory.
        foods = runner.biomass[:, whole.model.food_blob_index].reshape(2, 3, 2, 2, 72)
        torch.testing.assert_close(foods[0], foods[1], atol=0, rtol=0)
        assert not torch.equal(foods[0, 0, 0], foods[0, 0, 1])
        previous_keys = runner.keys.clone()
        saved = [v.clone() for v in runner.state_buffers]
        runner._tick()
        expected = [v.clone() for v in runner.state_buffers]
        for v, before in zip(runner.state_buffers, saved):
            v.copy_(before)
        torch.compile(runner._tick, backend="eager", fullgraph=True)()
        for v, reference in zip(runner.state_buffers, expected):
            torch.testing.assert_close(v, reference)
        def forbidden(*args, **kwargs):
            raise AssertionError("Tensor readback during debug-food training")
        with monkeypatch.context() as patch:
            for name in ("cpu", "numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
                patch.setattr(torch.Tensor, name, forbidden)
            whole.train_step(n_eval_ticks=4)
        assert not torch.equal(runner.keys, previous_keys)
    finally:
        whole.close()
        chunked.close()


def test_configuration_survives_builders_pickling_and_visual_probes():
    from train import _make_env_builder, _ProbeEnvBuilder
    config = debug_food.FoodBlobConfig(speed=0.2, radius=2, seed=13)
    builder = _make_env_builder(project_path="mareld2.yaml", grid_size=(8, 9),
                                impact_vars=[], impact_ranges={}, food_blobs=config)
    builder = pickle.loads(pickle.dumps(builder.with_world(3, 7)))
    assert builder(seed=42).food_blobs == config
    probe = _ProbeEnvBuilder("mareld2.yaml", (8, 9), food_blobs=config)
    probe.b0_overrides = {"phytoplankton": 100}
    env = probe()
    debug_food.apply(env, 100)
    assert env.fgs["phytoplankton"].biomass.sum() == pytest.approx(100, rel=1e-6)
    spec = ProjectSpec(EnvironmentBuilder("mareld2.yaml", grid=(8, 9), food_blobs=config).with_world(4))
    trainer = TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2)
    try:
        assert build_inference_env(inference_config(trainer, "gpu"), seed=42).food_blobs == config
    finally:
        trainer.close()
    parser = argparse.ArgumentParser()
    debug_food.add_food_blob_arguments(parser)
    assert debug_food.food_blob_options(parser.parse_args([])) is None
    assert debug_food.food_blob_options(parser.parse_args(["--debug-food-blobs"])) == debug_food.FoodBlobConfig()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
@pytest.mark.parametrize("execution", ["cuda-graph", "compile-graph"])
def test_blob_training_supports_cuda_graph_capture(execution):
    spec = ProjectSpec(EnvironmentBuilder("mareld2.yaml", grid=(8, 9),
                                          food_blobs=debug_food.FoodBlobConfig()))
    eager = TensorARSTrainer(spec, device="cuda", execution="eager", n_deltas=2)
    captured = TensorARSTrainer(spec, device="cuda", execution=execution,
                                n_deltas=2, graph_ticks=3)
    try:
        for ticks in (7, 4):
            eager.train_step(n_eval_ticks=ticks)
            captured.train_step(n_eval_ticks=ticks)
            torch.cuda.synchronize()
            torch.testing.assert_close(eager.rewards, captured.rewards, rtol=1e-4, atol=1e-5)
    finally:
        eager.close()
        captured.close()


@pytest.mark.parametrize("kwargs", [{"speed": -1}, {"speed": float("nan")},
                                   {"radius": -1}, {"radius": 0.5}, {"radius": float("inf")},
                                   {"seed": -1}])
def test_invalid_blob_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        debug_food.FoodBlobConfig(**kwargs)
