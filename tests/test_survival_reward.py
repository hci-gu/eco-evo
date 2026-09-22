"""Survival-first reward semantics and CPU/worker/device parity."""

import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.runners.survival_reward import (
    SurvivalReward, SurvivalScore, TensorSurvivalScore,
    add_survival_reward_arguments, survival_reward_options,
)
from lib.runners.trainer import ARSTrainer
from lib.runners import parallel_worker
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from test_gpu_ecosystem import make_env, DEVICES
from test_gpu_trainer import build_trainer


class ScriptedEnvironment:
    def __init__(self, trajectory, initial=(100., 100.)):
        self.dm_ids = ["predator", "prey"]
        self.fgs = {f: SimpleNamespace(biomass=np.array([b]), energy_reserve=np.array([b]),
                                       max_energy_reserve=1., params={"energy_content": 1.})
                    for f, b in zip(self.dm_ids, initial)}
        self.trajectory, self.steps = trajectory, 0
        self.policy_controller = SimpleNamespace(forward=lambda obs: None)

    def get_observation(self):
        return None

    def step(self, actions=None):
        for f, (ratio, quality) in zip(self.dm_ids, self.trajectory[self.steps]):
            self.fgs[f].biomass[:] = 100 * ratio
            self.fgs[f].energy_reserve[:] = 100 * ratio * quality
        self.steps += 1


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("coevolution", [False, True])
def test_independent_lifetimes_tail_quality_and_no_restart(worker, coevolution, monkeypatch):
    path = [[(1., .1), (1., .1)], [(.3, .8), (1., .2)],
            [(.29, 1.), (1., .4)], [(1., 1.), (4., .6)]]
    env = ScriptedEnvironment(path)
    builder = lambda seed=None: env
    config = SurvivalReward()
    if worker:
        monkeypatch.setattr(parallel_worker, "_ENV_BUILDER", builder)
        monkeypatch.setattr(parallel_worker, "_POLICIES", {})
        task = dict(weights_dict={}, n_ticks=4, alpha=1, beta=1,
                    survival_reward=config, fg_to_train="predator", fg_list=env.dm_ids)
        evaluate = parallel_worker._evaluate_coevo_task if coevolution else parallel_worker._evaluate_task
        reward, _, _ = evaluate(task)
    else:
        trainer = ARSTrainer(builder, {}, survival_reward=config)
        reward, _, _ = (trainer._evaluate_coevo(env.dm_ids, 4) if coevolution
                        else trainer._evaluate("predator", 4))
    assert env.steps == 4
    assert reward == pytest.approx({"predator": .6, "prey": 1.0625} if coevolution else .6)


@pytest.mark.parametrize("initial,path,expected", [
    (0., [(1., 1.)], 0.),
    (100., [(.2, 1.)], 0.),
    (100., [(0., 1.)], 0.),
    (100., [(1., float("nan"))], 0.),
    (100., [(float("inf"), 1.)], 0.),
    (100., [(1., -1.)], 0.),
    (100., [(1., 10.)], 1.5),
    (100., [(1., 0.)], 1.),
])
def test_empty_failed_and_clipped_quality(initial, path, expected):
    env = ScriptedEnvironment([[point, point] for point in path], (initial, initial))
    score = SurvivalScore(env, SurvivalReward())
    for _ in path:
        env.step()
        score.step(env)
    assert score.results(len(path))["predator"] == pytest.approx(expected)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_score_matches_cpu_for_different_failure_times(device):
    b0 = torch.full((3, 2), 100., dtype=torch.float64, device=device)
    score = TensorSurvivalScore((3, 2), 9, device, SurvivalReward())
    score.reset(b0)
    envs = []
    for failure in (0, 3, 9):
        path = [[(.2 if t == failure else 1., t / 8), (1., 1 - t / 8)] for t in range(9)]
        envs.append(ScriptedEnvironment(path))
    references = [SurvivalScore(e, SurvivalReward()) for e in envs]
    for t in range(9):
        for env, reference in zip(envs, references):
            env.step()
            reference.step(env)
        b = torch.tensor([[e.fgs[f].biomass.sum() for f in e.dm_ids] for e in envs], device=device)
        r = torch.tensor([[e.fgs[f].energy_reserve.sum() for f in e.dm_ids] for e in envs], device=device)
        score.step(b, r, b0, torch.ones(1, 2, device=device), torch.tensor(t, device=device))
    expected = torch.tensor([list(s.results(9).values()) for s in references], device=device,
                            dtype=torch.float64)
    torch.testing.assert_close(score.results(9), expected)
    score.reset(b0)
    assert torch.count_nonzero(score.results(9)) == 0


def test_one_more_tick_always_beats_quality_and_early_death_uses_full_horizon():
    scores = []
    for lifetime, quality in ((20, 1.), (21, 0.), (50, 0.)):
        env = ScriptedEnvironment([[(1., quality)] * 2] * lifetime + [[(.1, 1.)] * 2])
        score = SurvivalScore(env, SurvivalReward())
        for _ in env.trajectory:
            env.step()
            score.step(env)
        scores.append(score.results(300)["predator"])
    assert scores == pytest.approx([20.5 / 300, 21 / 300, 50 / 300])
    assert scores[0] < scores[1] < scores[2]


def test_cli_and_invalid_combinations():
    parser = argparse.ArgumentParser()
    add_survival_reward_arguments(parser)
    assert survival_reward_options(parser.parse_args([])) is None
    assert survival_reward_options(parser.parse_args(["--survival-reward"])).lower == .3
    for lower in (0., -1., 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SurvivalReward(lower)
    for kwargs in ({"integral_reward": False}, {"legacy_reward": True},
                   {"entropy_coef": .1}, {"argmax_penalty": .1}):
        with pytest.raises(ValueError, match="--survival-reward"):
            ARSTrainer(None, {}, survival_reward=SurvivalReward(), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("floor", [.3, .99])
def test_complete_rollout_matches_reference_and_resizes(device, floor):
    torch.set_num_threads(1)
    env = make_env(True, True)
    config = SurvivalReward(floor)
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    reference = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, n_workers=1,
                           hidden_dim=7, survival_reward=config)
    reference.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    runner = RolloutRunner(model, bank, 1, 1, execution="eager", survival_reward=config)
    b, r, _, phase = model.import_state([env])
    for ticks in (1, 11, 4, 40):
        runner.reset(b, r, phase, torch.zeros(1, dtype=torch.long, device=device), ticks,
                     model.tensor(np.zeros((model.D, model.F))),
                     model.tensor(np.ones((model.D, model.F))), model.tensor(1.))
        runner.run(ticks)
        reward, _ = runner.results(ticks)
        expected, _, _ = reference._evaluate_coevo(model.dm_ids, ticks)
        np.testing.assert_allclose(reward[0].cpu(), list(expected.values()), atol=3e-6, rtol=3e-5)


def test_chunked_training_and_fullgraph_compilation():
    whole = build_trainer(survival_reward=SurvivalReward())
    chunked = build_trainer(survival_reward=SurvivalReward(), pairs_per_batch=2)
    for ticks in (7, 4, 40):
        whole.train_step(n_eval_ticks=ticks)
        chunked.train_step(n_eval_ticks=ticks)
        torch.testing.assert_close(whole.rewards, chunked.rewards, atol=2e-6, rtol=2e-5)
        for w, c in zip(whole.theta, chunked.theta):
            torch.testing.assert_close(w, c, atol=3e-6, rtol=2e-5)
    runner = whole.runner
    before = [v.clone() for v in runner.state_buffers]
    runner._tick()
    expected = [v.clone() for v in runner.state_buffers]
    for v, saved in zip(runner.state_buffers, before):
        v.copy_(saved)
    torch.compile(runner._tick, backend="eager", fullgraph=True)()
    for v, saved in zip(runner.state_buffers, expected):
        torch.testing.assert_close(v, saved)


def test_training_has_no_host_readback_even_when_horizon_grows(monkeypatch):
    trainer = build_trainer(survival_reward=SurvivalReward())
    trainer.train_step(n_eval_ticks=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor readback during survival-first training")
    for name in ("cpu", "numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    trainer.train_step(n_eval_ticks=4)


def test_checkpoint_retains_reward_and_rejects_objective_changes():
    trainer = build_trainer(survival_reward=SurvivalReward(.4))
    trainer.train_step(n_eval_ticks=4)
    state = trainer.state_dict()
    resumed = build_trainer(survival_reward=SurvivalReward(.4))
    resumed.load_state_dict(state)
    trainer.train_step(n_eval_ticks=4)
    resumed.train_step(n_eval_ticks=4)
    torch.testing.assert_close(trainer.rewards, resumed.rewards, rtol=0, atol=0)
    for w, r in zip(trainer.theta, resumed.theta):
        torch.testing.assert_close(w, r, rtol=0, atol=0)
    for config in (None, SurvivalReward(.3)):
        with pytest.raises(ValueError, match="Checkpoint survival reward differs"):
            build_trainer(survival_reward=config).load_state_dict(state)
    with pytest.raises(ValueError, match="Checkpoint survival reward differs"):
        resumed.load_state_dict(build_trainer().state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
@pytest.mark.parametrize("execution", ["cuda-graph", "compile-graph"])
def test_cuda_capture_replay_and_growing_horizons(execution):
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6)))
    options = dict(device="cuda", n_deltas=2, worlds=2, survival_reward=SurvivalReward())
    eager = TensorARSTrainer(spec, execution="eager", **options)
    captured = TensorARSTrainer(spec, execution=execution, graph_ticks=3, **options)
    for ticks in (1, 7, 4, 40):
        eager.train_step(n_eval_ticks=ticks)
        captured.train_step(n_eval_ticks=ticks)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.rewards, captured.rewards, rtol=1e-4, atol=1e-5)
        for w, r in zip(eager.theta, captured.theta):
            torch.testing.assert_close(w, r, rtol=1e-4, atol=2e-5)
