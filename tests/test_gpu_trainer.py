import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from lib.gpu.trainer import TensorARSTrainer, ars_update
from lib.runners.trainer import ARSTrainer
from test_gpu_ecosystem import make_env, DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("legacy,integral", [(False, True), (False, False), (True, True), (True, False)])
def test_complete_rollout_reward_stats_and_actions_match_cpu(device, legacy, integral):
    torch.set_num_threads(1)
    env = make_env(True, True)
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    reference = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, n_workers=1,
                           hidden_dim=7, legacy_reward=legacy, integral_reward=integral,
                           alpha=0.7, beta=1.2, survival_bonus=0.4)
    reference.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    reference.softmax_temperature = 1.7
    runner = RolloutRunner(model, bank, 1, 1, execution="eager", legacy_reward=legacy,
                           integral_reward=integral, alpha=0.7, beta=1.2, survival_bonus=0.4)
    b, r, _, phase = model.import_state([env])
    mean = np.full((model.D, model.F), 0.03, dtype=np.float32)
    var = np.full_like(mean, 0.8)
    ticks = 11
    runner.reset(b, r, phase, torch.zeros(1, dtype=torch.long, device=device), ticks,
                 model.tensor(mean), model.tensor(var), model.tensor(1.7))
    runner.run(ticks)
    reward, action = runner.results(ticks)
    expected_reward, moments, expected_actions = reference._evaluate_coevo(
        model.dm_ids, ticks, obs_mean=mean, obs_var=var, dm_ids_for_norm=list(model.dm_ids))
    for d, fid in enumerate(model.dm_ids):
        assert reward[0, d].item() == pytest.approx(expected_reward[fid], rel=3e-5, abs=3e-6)
        expected = [expected_actions[fid][k] for k in ("entropy", "move_frac", "rest_frac", "eat_frac")]
        np.testing.assert_allclose(action[0, d].cpu(), expected, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(runner.obs_sum[0].cpu(), moments[0], atol=0.003, rtol=3e-5)
    np.testing.assert_allclose(runner.obs_sumsq[0].cpu(), moments[1], atol=0.003, rtol=3e-5)
    assert moments[2] == ticks * model.C


@pytest.mark.parametrize("top", [1, 3, 7])
def test_ars_update_matches_numpy(top):
    rng = np.random.default_rng(23)
    weights = rng.normal(size=31).astype(np.float32)
    deltas = rng.normal(size=(7, 31)).astype(np.float32)
    pos, neg = rng.normal(size=7), rng.normal(size=7)
    indices = np.argsort(-np.maximum(pos, neg))[:top] if top < 7 else np.arange(7)
    sigma = np.std(np.concatenate((pos[indices], neg[indices]))) + 1e-8
    step = np.zeros_like(weights)
    for i in indices:
        step += (pos[i] - neg[i]) * deltas[i]
    expected = weights + 0.03 / (top * sigma) * step
    result, actual_sigma = ars_update(torch.tensor(weights), torch.tensor(deltas),
                                      torch.tensor(pos), torch.tensor(neg), 0.03, top)
    np.testing.assert_allclose(result, expected, atol=2e-7, rtol=2e-6)
    assert actual_sigma.item() == pytest.approx(sigma)


def build_trainer(**kwargs):
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6)))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=3,
                            worlds=2, seed=16, **kwargs)


def test_chunking_is_equivalent_including_partial_final_batch():
    whole, chunks = build_trainer(), build_trainer(pairs_per_batch=2)
    for _ in range(2):
        whole.train_step(n_eval_ticks=6)
        chunks.train_step(n_eval_ticks=6)
        torch.testing.assert_close(whole.rewards, chunks.rewards, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(whole.obs_mean, chunks.obs_mean, atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(whole.obs_var, chunks.obs_var, atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(whole.obs_count, chunks.obs_count)
        for w, c in zip(whole.theta, chunks.theta):
            torch.testing.assert_close(w, c, atol=3e-6, rtol=2e-5)


def test_checkpoint_resumes_exact_next_update_and_inference_export(tmp_path):
    trainer = build_trainer()
    trainer.train_step(n_eval_ticks=4, world_epoch=7)
    state = trainer.state_dict()
    trainer.export_policies(tmp_path)
    resumed = build_trainer()
    resumed.load_state_dict(state)
    trainer.train_step(n_eval_ticks=4, world_epoch=7)
    resumed.train_step(n_eval_ticks=4, world_epoch=7)
    for w, r in zip(trainer.theta, resumed.theta):
        torch.testing.assert_close(w, r, rtol=0, atol=0)
    imported = build_trainer()
    imported.import_policies(tmp_path)
    for w, r in zip(state["theta"], imported.theta):
        torch.testing.assert_close(w, r, rtol=0, atol=0)
    torch.testing.assert_close(imported.obs_mean, state["obs_mean"], rtol=1e-6, atol=1e-7)


def test_round_robin_only_updates_target_and_refreshes_noise():
    trainer = build_trainer()
    before = [w.clone() for w in trainer.theta]
    target = trainer.model.dm_ids[0]
    trainer.train_step([target], n_eval_ticks=4, world_epoch=2)
    worlds = trainer.world_biomass.clone()
    keys = trainer.runner.keys.clone()
    assert not torch.equal(before[0], trainer.theta[0])
    for w, r in zip(before[1:], trainer.theta[1:]):
        torch.testing.assert_close(w, r, rtol=0, atol=0)
    trainer.train_step([target], n_eval_ticks=4, world_epoch=2)
    torch.testing.assert_close(worlds, trainer.world_biomass, rtol=0, atol=0)
    assert not torch.equal(keys, trainer.runner.keys)
    half = trainer.runner.E // 2
    torch.testing.assert_close(trainer.runner.keys[:half], trainer.runner.keys[half:], rtol=0, atol=0)


def test_zero_perturbation_keeps_weights_and_couples_all_randomness():
    trainer = build_trainer(entropy_coef=0.1, argmax_penalty=0.2)
    before = [w.clone() for w in trainer.theta]
    trainer.train_step(n_eval_ticks=8, deltas=[torch.zeros(3, w.numel()) for w in trainer.theta])
    torch.testing.assert_close(trainer.rewards[0], trainer.rewards[1], rtol=0, atol=0)
    for w, r in zip(before, trainer.theta):
        torch.testing.assert_close(w, r, rtol=0, atol=0)


def test_reward_modifiers_match_reference_formula():
    trainer = build_trainer(entropy_coef=0.1, argmax_penalty=0.2)
    trainer.train_step(n_eval_ticks=5)
    r = trainer.rewards.numpy()
    a = trainer.actions.numpy()
    expected = (r - r.mean(axis=(0, 1))) / (r.std(axis=(0, 1)) + 1e-8)
    expected += 0.1 * a[..., 0] / math.log(trainer.model.A)
    expected -= 0.2 * a[..., 1:].max(axis=-1)
    np.testing.assert_allclose(trainer.last_metrics["reward_mean"], expected.mean(axis=(0, 1)), atol=1e-10)


def test_full_tick_traces_without_graph_breaks():
    trainer = build_trainer()
    trainer.train_step(n_eval_ticks=2)
    runner = trainer.runner
    snapshot = [v.clone() for v in runner.state_buffers]
    runner._tick()
    expected = [v.clone() for v in runner.state_buffers]
    for v, saved in zip(runner.state_buffers, snapshot):
        v.copy_(saved)
    # Trace with the eager compiler backend: validates the complete graph on
    # machines without a CUDA compiler/device, without claiming CUDA execution.
    compiled = torch.compile(runner._tick, backend="eager", fullgraph=True)
    compiled()
    for actual, ref in zip(runner.state_buffers, expected):
        torch.testing.assert_close(actual, ref)


def test_numerical_training_loop_does_not_read_back_tensors(monkeypatch):
    trainer = build_trainer()
    trainer.train_step(n_eval_ticks=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor readback in numerical training loop")
    for name in ("cpu", "numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    trainer.train_step(n_eval_ticks=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
@pytest.mark.parametrize("execution", ["cuda-graph", "compile", "compile-graph"])
def test_cuda_execution_matches_eager_and_replay_advances(execution):
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6)))
    eager = TensorARSTrainer(spec, device="cuda", execution="eager", n_deltas=2, worlds=2)
    optimized = TensorARSTrainer(spec, device="cuda", execution=execution, graph_ticks=3, n_deltas=2, worlds=2)
    for ticks in (7, 4):  # exercise remainder and a changed horizon
        eager.train_step(n_eval_ticks=ticks)
        optimized.train_step(n_eval_ticks=ticks)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.rewards, optimized.rewards, rtol=1e-4, atol=1e-5)
        for w, r in zip(eager.theta, optimized.theta):
            torch.testing.assert_close(w, r, rtol=1e-4, atol=2e-5)
