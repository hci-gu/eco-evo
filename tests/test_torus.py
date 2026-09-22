"""Periodic topology, continuous hunting across seams and segmented debug food."""

import pickle

import numpy as np
import pytest
import torch

from lib.environments.ecosystem import EcosystemEnvironment
from lib.environments.ecosystem_env import debug_food, movement, source_tracking
from lib.environments.ecosystem_env.observations import shift_field
from lib.environments.ecosystem_env.currents import CurrentConfig, direction_fractions
from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.trainer import TensorARSTrainer
from lib.runners.training_progress import build_inference_env, inference_config
from test_gpu_ecosystem import DEVICES, make_env, manual_actions, _assert_parity


def torus_env(**kwargs):
    base = make_env(min_split=0, extinction_factor=0)
    return EcosystemEnvironment(dict(height=base.H, width=base.W), base.fgs,
                                boundary="torus", migration=True,
                                apply_natural_mortality=False, **kwargs)


@pytest.mark.parametrize("direction,axis,shift", [("N", 0, 1), ("E", 1, -1),
                                                 ("S", 0, -1), ("W", 1, 1)])
def test_observations_see_the_opposite_edge(direction, axis, shift):
    field = np.arange(20).reshape(4, 5)
    np.testing.assert_array_equal(shift_field(field, direction, "torus"),
                                  np.roll(field, shift, axis))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("direction,source,destination", [
    (0, (0, 2), (-1, 2)), (1, (2, -1), (2, 0)),
    (2, (-1, 2), (0, 2)), (3, (2, 0), (2, -1)),
])
@pytest.mark.parametrize("suppress", [False, True])
def test_active_crossing_is_local_conservative_and_tracks_source(device, direction, source, destination, suppress):
    env = torus_env(local_reward=source_tracking.LocalRewardConfig())
    assert not env.migration
    for fg in env.fgs.values():
        fg.biomass.fill(0)
        fg.energy_reserve.fill(0)
    env.fgs["a"].biomass[source] = 100
    env.fgs["a"].energy_reserve[source] = 500
    env.fgs["a"].min_split_biomass = 100 if suppress else 0
    env.fgs["a"].extinction_threshold_factor = .3
    env.build_static_caches()
    env.dm_v[0] = 1
    model = TensorEcosystem(env, device)
    b, r, _, _ = model.import_state([env])
    actions = torch.zeros(1, model.D, model.A, model.C, device=device)
    actions[:, :, 4] = 1
    index = (source[0] % env.H) * env.W + source[1] % env.W
    share = .1 if suppress else 1.
    actions[0, 0, 4, index] = 1 - share
    actions[0, 0, direction, index] = share
    actual_b, actual_r, flow = model.movement(b, r, torch.zeros_like(b[:, model.dm_index]), actions, track=True)
    source_tracking.begin_tick(env)
    movement.apply_movement(env, movement.apply_energy_costs(env, manual_actions(actions, env)))
    expected_b, expected_r, _, _ = model.import_state([env])
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_r, expected_r)
    assert env.fgs["a"].biomass.sum() == pytest.approx(100)
    assert env.fgs["a"].biomass[destination] == pytest.approx(0 if suppress else 100)
    assert np.count_nonzero(env.fgs["a"].biomass) == 1
    tracked, frac = source_tracking.tracked_end_energy(env)
    gpu_tracked, gpu_frac = model.tracked_energy(flow, actual_b, actual_r)
    np.testing.assert_allclose(gpu_tracked.cpu()[0].reshape(tracked.shape), tracked, rtol=1e-6)
    np.testing.assert_allclose(gpu_frac.cpu()[0].reshape(frac.shape), frac)
    np.testing.assert_array_equal(frac, 1)
    assert tracked[0, source[0], source[1]] == pytest.approx(
        float(model.local_energy(actual_b, actual_r).sum().cpu()), rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_full_ecology_and_observations_match_across_torus_seams(device):
    env = torus_env(currents=CurrentConfig(strength=.6, scale=2.5))
    env.fgs["c"].current_response = 1
    _assert_parity(env, device, ticks=25)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("blocked", [False, True])
def test_currents_wrap_locally_with_energy_and_respect_opposite_habitat(device, blocked):
    env = torus_env(currents=CurrentConfig(strength=1, scale=2.5))
    fg = env.fgs["c"]
    fg.current_response = 1
    fg.biomass.fill(0)
    fg.energy_reserve.fill(0)
    fg.biomass[-1, -1], fg.energy_reserve[-1, -1] = 100, 200
    if blocked:
        access = np.ones((env.H, env.W), dtype=np.float32)
        access[0, -1] = access[-1, 0] = 0
        env.grid.add_map("accessibility", access)
        env.build_static_caches()
        assert env.move_mask[1, -1, -1] == env.move_mask[2, -1, -1] == 0
    model = TensorEcosystem(env, device)
    b, r, _, _ = model.import_state([env])
    actual_b, actual_r = model.advect(b, r, model.tensor(0, torch.int64))
    movement.apply_currents(env)
    expected_b, expected_r, _, _ = model.import_state([env])
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(actual_r, expected_r)
    assert fg.biomass.sum() == pytest.approx(100)
    np.testing.assert_allclose(fg.energy_reserve, 2 * fg.biomass)
    if blocked:
        assert fg.biomass[-1, -1] == 100
    else:
        assert fg.biomass[0, -1] > 0 and fg.biomass[-1, 0] > 0
        assert np.count_nonzero(fg.biomass) == 3


@pytest.mark.parametrize("device", DEVICES)
def test_current_cloud_is_periodic_and_scrolls_over_seams(device):
    config = CurrentConfig(scale=7.5, period=13, strength=.8)
    h, w = 19, 23
    y, x = np.indices((h, w), dtype=np.float32)
    for tick in (0, 20, 5000):
        expected = np.asarray(direction_fractions(tick, 17, config, x, y, (h, w)))
        tiled = direction_fractions(tick, 17, config, x + w, y - h, (h, w))
        np.testing.assert_allclose(tiled, expected, atol=3e-6)
        actual = torch.stack(direction_fractions(torch.tensor(tick, device=device),
            torch.tensor(17, device=device), config, torch.tensor(x, device=device),
            torch.tensor(y, device=device), (h, w)))
        np.testing.assert_allclose(actual.cpu(), expected, atol=2e-6)
    first = np.asarray(direction_fractions(0, 17, config, x, y, (h, w)))
    later = direction_fractions(config.period, 17, config, x, y, (h, w))
    np.testing.assert_allclose(later, np.roll(first, (1, 1), (1, 2)), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_random_segment_duration_heading_speed_and_replay(device):
    config = debug_food.FoodBlobConfig(speed=.4, seed=13, segment_min=3, segment_max=9)
    keys = np.array([[0], [17], [42], [2**32-1]], dtype=np.int64)
    cpu = debug_food.TorusBlobMotion(24, 24, keys, config)
    gpu = debug_food.TorusBlobMotion(24, 24, torch.tensor(keys, device=device), config)
    durations, headings = [], []
    for tick in range(5000):
        before = [cpu.x.copy(), cpu.y.copy()]
        previous_remaining, previous_segment = cpu.remaining.copy(), cpu.segment.copy()
        cpu.advance()
        gpu.advance()
        for expected, actual in zip(cpu.buffers, gpu.buffers):
            np.testing.assert_allclose(actual.cpu(), expected, atol=2e-7, rtol=1e-7)
        for axis in (cpu.x, cpu.y):
            assert axis.min() >= 0 and axis.max() < 24
        dx = (cpu.x - before[0] + 12) % 24 - 12
        dy = (cpu.y - before[1] + 12) % 24 - 12
        np.testing.assert_allclose(np.hypot(dx, dy), config.speed, atol=1e-10)
        expired = previous_remaining == 0
        np.testing.assert_array_equal(cpu.segment, previous_segment + expired)
        assert ((cpu.remaining >= 0) & (cpu.remaining < config.segment_max)).all()
        if expired.any():
            durations.extend((cpu.remaining[expired] + 1).tolist())
            headings.extend(cpu.vx[expired].tolist())
    assert set(durations) == set(range(3, 10))
    assert np.std(headings) > .1
    replay = debug_food.TorusBlobMotion(24, 24, keys, config)
    gpu.reset(torch.tensor(keys, device=device))
    for expected, actual in zip(replay.buffers, gpu.buffers):
        np.testing.assert_allclose(actual.cpu(), expected, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_blob_footprint_spans_both_edges_and_is_translation_equivariant(device):
    h, w = 12, 16
    y, x = np.indices((h, w), dtype=np.float32)
    x, y = x.ravel(), y.ravel()
    allowed = np.ones(x.size, dtype=bool)
    config = debug_food.FoodBlobConfig(radius=3)
    def field(cx, cy):
        return debug_food.blob_weights(x, y, h, w, 0, 0, config, allowed,
                                       "torus", (np.array([cx], np.float32), np.array([cy], np.float32)))
    seam = field(0, 0).reshape(h, w)
    assert seam[-1, 0] > 0 and seam[0, -1] > 0 and seam[-1, -1] > 0
    np.testing.assert_array_equal(np.roll(seam, (4, 7), (0, 1)), field(7, 4).reshape(h, w))
    actual = debug_food.blob_weights(torch.tensor(x, device=device), torch.tensor(y, device=device),
        h, w, 0, 0, config, torch.tensor(allowed, device=device), "torus",
        (torch.tensor([[0.]], device=device), torch.tensor([[0.]], device=device)))
    np.testing.assert_allclose(actual.cpu()[0], seam.ravel(), atol=1e-7)
    assert seam.sum() == pytest.approx(1)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (3, 3)])
def test_torus_stationary_food_tiny_grids_and_obstacles(shape):
    config = debug_food.FoodBlobConfig(speed=0, radius=1, segment_min=1, segment_max=2)
    keys = np.array([42], dtype=np.int64)
    motion = debug_food.TorusBlobMotion(*shape, keys, config)
    before = motion.center()
    for _ in range(10):
        motion.advance()
    for a, b in zip(before, motion.center()):
        np.testing.assert_array_equal(a, b)
    y, x = np.indices(shape, dtype=np.float32)
    allowed = np.zeros(x.size, dtype=bool)
    allowed[-1] = True
    field = debug_food.blob_weights(x.ravel(), y.ravel(), *shape, 10, keys, config,
                                    allowed, "torus", motion.center())
    assert field[-1] == 1 and field.sum() == 1


@pytest.mark.parametrize("kwargs", [{"segment_min": 0}, {"segment_min": 5, "segment_max": 4},
                                   {"segment_max": 2.5}])
def test_bad_segment_ranges_are_rejected(kwargs):
    with pytest.raises(ValueError, match="segment ticks"):
        debug_food.FoodBlobConfig(**kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_segmented_blob_ecology_matches_cpu_and_replenishes(device):
    config = debug_food.FoodBlobConfig(speed=.5, segment_min=2, segment_max=7)
    env = EnvironmentBuilder("mareld2.yaml", grid=(8, 9), boundary="torus",
                             food_blobs=config, currents=CurrentConfig())(seed=17)
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    b, r, hidden, phase = model.import_state([env])
    motion = debug_food.TorusBlobMotion(env.H, env.W, torch.tensor([[17]], device=device), config)
    totals = (b[:, model.food_blob_index].sum(-1, keepdim=True),
              r[:, model.food_blob_index].sum(-1, keepdim=True))
    packed = bank.pack([w[None] for w in bank.flat_weights()])
    for tick in range(80):
        actions = model.action_probabilities(bank.forward(model.observations(b, r, hidden), *packed), b, model.tensor(1.))
        env.step(manual_actions(actions, env))
        b, r, hidden, _, _ = model.step(b, r, actions, model.tensor(tick), phase,
            torch.zeros_like(b), food_blob_totals=totals, food_blob_motion=motion)
        expected_b, expected_r, _, _ = model.import_state([env])
        torch.testing.assert_close(b, expected_b, rtol=2e-4, atol=.001)
        torch.testing.assert_close(r, expected_r, rtol=2e-4, atol=.05)
        torch.testing.assert_close(b[:, model.food_blob_index].sum(-1, keepdim=True), totals[0])
    before = {f: env.fgs[f].biomass.copy() for f in debug_food.FOOD_IDS}
    for f in debug_food.FOOD_IDS:
        env.fgs[f].biomass.fill(0)
    debug_food.apply(env, 80)
    for f in debug_food.FOOD_IDS:
        np.testing.assert_array_equal(env.fgs[f].biomass, before[f])


def torus_trainer(**kwargs):
    spec = ProjectSpec(EnvironmentBuilder("mareld2.yaml", grid=(8, 9), boundary="torus",
        food_blobs=debug_food.FoodBlobConfig(segment_min=2, segment_max=5), currents=CurrentConfig()))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=3, worlds=2, **kwargs)


def test_training_pairs_chunking_compile_no_readback_and_resume(monkeypatch):
    whole, chunks = torus_trainer(), torus_trainer(pairs_per_batch=2)
    for ticks in (17, 11):
        whole.train_step(n_eval_ticks=ticks)
        chunks.train_step(n_eval_ticks=ticks)
        torch.testing.assert_close(whole.rewards, chunks.rewards, atol=3e-6, rtol=3e-5)
        for w, c in zip(whole.theta, chunks.theta):
            torch.testing.assert_close(w, c, atol=3e-6, rtol=3e-5)
    runner = whole.runner
    for state in runner.food_blob_motion.buffers:
        torch.testing.assert_close(state[:6], state[6:], rtol=0, atol=0)
    before = [v.clone() for v in runner.state_buffers]
    runner._tick()
    expected = [v.clone() for v in runner.state_buffers]
    for value, saved in zip(runner.state_buffers, before):
        value.copy_(saved)
    # Other test modules compile this same code object with many fixed schemas.
    # Isolate this graph test from their process-wide Dynamo variant cache.
    torch._dynamo.reset()
    torch.compile(runner._tick, backend="eager", fullgraph=True)()
    for value, saved in zip(runner.state_buffers, expected):
        torch.testing.assert_close(value, saved)
    torch._dynamo.reset()
    state = whole.state_dict()
    resumed = torus_trainer()
    resumed.load_state_dict(state)
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor readback in torus training")
    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        whole.train_step(n_eval_ticks=23)
    resumed.train_step(n_eval_ticks=23)
    torch.testing.assert_close(whole.rewards, resumed.rewards, rtol=0, atol=0)
    for w, r in zip(whole.theta, resumed.theta):
        torch.testing.assert_close(w, r, rtol=0, atol=0)
    state.pop("boundary")
    with pytest.raises(ValueError, match="boundary differs"):
        resumed.load_state_dict(state)


def test_boundary_propagates_through_workers_probes_inference_and_cli():
    from train import _make_env_builder, _ProbeEnvBuilder
    from inference import build_env
    from train_gpu import build_parser
    from lib.gpu.cli import builder_from_args, parse_training_args
    builder = _make_env_builder(project_path="mareld2.yaml", grid_size=(8, 9),
        impact_vars=[], impact_ranges={}, boundary="torus")
    assert pickle.loads(pickle.dumps(builder.with_world(3, 7)))(seed=42).boundary == "torus"
    assert _ProbeEnvBuilder("mareld2.yaml", (8, 9), boundary="torus")().boundary == "torus"
    assert build_env("mareld2.yaml", (8, 9), verbose=False, boundary="torus").boundary == "torus"
    trainer = torus_trainer()
    config = inference_config(trainer, "gpu")
    assert config["boundary"] == build_inference_env(config, 42).boundary == "torus"
    args = parse_training_args(build_parser(), ["--boundary", "torus", "--debug-food-blobs",
                                               "--food-blob-segment-ticks", "3", "8"])
    builder = builder_from_args(args)
    assert builder.with_world(3).boundary == "torus"
    assert builder.food_blobs.segment_min == 3 and builder.food_blobs.segment_max == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
@pytest.mark.parametrize("execution", ["cuda-graph", "compile-graph"])
def test_torus_blob_capture_restores_and_advances_motion(execution):
    spec = torus_trainer().spec
    options = dict(device="cuda", n_deltas=2, worlds=2, graph_ticks=3)
    eager = TensorARSTrainer(spec, execution="eager", **options)
    captured = TensorARSTrainer(spec, execution=execution, **options)
    for ticks in (17, 11):
        eager.train_step(n_eval_ticks=ticks)
        captured.train_step(n_eval_ticks=ticks)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.rewards, captured.rewards, atol=1e-5, rtol=1e-4)
        for a, b in zip(eager.runner.food_blob_motion.buffers, captured.runner.food_blob_motion.buffers):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)
