"""``inference.py --engine gpu``: the rollout on the tensor engine against
the CPU engine on the same world - headless (section 234), and the viewer
path with the random baselines in the same batch (section 235)."""
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import inference  # noqa: E402
from lib.environments.ecosystem_env.loss_accounting import LOSS_CAUSES  # noqa: E402
from lib.gpu.inference import GpuStepper, run_gpu_inference  # noqa: E402
from lib.runners.policy import PolicyNetwork  # noqa: E402
from test_gpu_ecosystem import DEVICES  # noqa: E402

TICKS = 6


def _world():
    return inference.build_env(str(ROOT / "mareld2.yaml"), (6, 7), seed=3,
                               verbose=False)


def _policies(env):
    torch.manual_seed(11)
    n_actions = 5 + env.N_all
    policies = {fid: PolicyNetwork(int(env.per_dm_in_dim[i]), n_actions,
                                   hidden_dim=8, hidden_layers=2,
                                   activation="tanh").eval()
                for i, fid in enumerate(env.dm_ids)}
    rng = np.random.default_rng(5)
    mean = rng.normal(0.0, 0.1, (env.N_dm, env.max_in_dim)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (env.N_dm, env.max_in_dim)).astype(np.float32)
    return policies, mean, var


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_rollout_follows_the_cpu_rollout(device):
    cpu_env, gpu_env = _world(), _world()
    policies, mean, var = _policies(cpu_env)
    expected = inference.run_inference(cpu_env, policies, mean, var, TICKS,
                                       verbose=False)
    got = run_gpu_inference(gpu_env, policies, mean, var, TICKS,
                            verbose=False, device=device)
    assert list(got) == list(expected)
    for fid in expected:
        assert len(got[fid]) == TICKS
        np.testing.assert_allclose(got[fid], expected[fid], rtol=1e-4, atol=1e-6,
                                   err_msg=fid)
    # The world handed in is only uploaded, never stepped.
    assert gpu_env.tick_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
def test_cuda_graph_replay_matches_eager():
    policies, mean, var = _policies(_world())
    eager = run_gpu_inference(_world(), policies, mean, var, 40, verbose=False,
                              device="cuda", execution="eager")
    graph = run_gpu_inference(_world(), policies, mean, var, 40, verbose=False,
                              device="cuda", execution="cuda-graph",
                              progress_ticks=[3, 9, 31])
    for fid in eager:
        np.testing.assert_allclose(graph[fid], eager[fid], rtol=1e-6, err_msg=fid)


def test_progress_lines_are_printed_on_the_cpu_schedule(capsys):
    env = _world()
    policies, mean, var = _policies(env)
    run_gpu_inference(env, policies, mean, var, TICKS, device="cpu",
                      progress_ticks=[1, 4])
    lines = [l.strip() for l in capsys.readouterr().out.splitlines()]
    assert [l for l in lines if l.startswith("tick ")] == [
        "tick 2/6", "tick 5/6", "tick 6/6"]
    assert sum(l.startswith("B/B0:") for l in lines) == 3


def test_progress_due():
    assert [t for t in range(20) if inference.progress_due(t, 20, None)] == \
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
    assert [t for t in range(10) if inference.progress_due(t, 10, Fraction(4))] == [3, 7, 9]


def test_gpu_engine_needs_cuda():
    out = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--engine", "gpu"],
                         cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode == 2
    assert "--engine gpu needs CUDA" in out.stderr


# --- The viewer path (section 235): GpuStepper mirrors into CPU envs. ---

def _close(got, expected, scale, rtol=1e-4):
    assert abs(got - expected) <= rtol * abs(expected) + 1e-6 * scale, (got, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_stepper_mirrors_the_cpu_books_for_main_and_baseline_worlds(device):
    """Trained world, an all-random and a solo-random baseline in one batch
    against three CPU envs stepped by the reference engine."""
    cpu = [_world() for _ in range(3)]
    gpu = [_world() for _ in range(3)]
    policies, mean, var = _policies(cpu[0])
    solo = cpu[0].dm_ids[1]
    for env in cpu:
        env.policies = dict(policies)
        env.obs_mean, env.obs_var = mean, var
        env.rebuild_batched_weights()
    inference._install_rnd_policies(cpu[0], cpu[1], policies, cpu[1].dm_ids)
    inference._install_rnd_policies(cpu[0], cpu[2], policies, [solo])
    stepper = GpuStepper(gpu, policies, mean, var, device=device,
                         random_dms=[set(gpu[1].dm_ids), {solo}])
    try:
        for _ in range(TICKS):
            for env in cpu:
                env.step(env.policy_controller.forward(env.get_observation()))
            stepper.step()
    finally:
        stepper.close()
    for ref, got in zip(cpu, gpu):
        assert got.tick_count == TICKS
        for fid, fg in ref.fgs.items():
            scale = float(fg.biomass.sum())
            np.testing.assert_allclose(got.fgs[fid].biomass, fg.biomass,
                                       rtol=1e-4, atol=1e-6, err_msg=fid)
            np.testing.assert_allclose(got.fgs[fid].energy_reserve, fg.energy_reserve,
                                       rtol=1e-4, atol=1e-5, err_msg=fid)
            if getattr(fg, "juvenile", None) is not None:
                np.testing.assert_allclose(got.fgs[fid].juvenile, fg.juvenile,
                                           rtol=1e-4, atol=1e-6, err_msg=fid)
            # The reference books M1 as a difference of float32 sums, hence
            # the absolute term relative to the group's biomass.
            for name, attr in LOSS_CAUSES:
                _close(getattr(got, attr)[fid], getattr(ref, attr)[fid], scale)
        for pred, row in ref.intake_by_pred_prey.items():
            for prey, value in row.items():
                _close(got.intake_by_pred_prey.get(pred, {}).get(prey, 0.0), value,
                       float(ref.fgs[prey].biomass.sum()))
        for attr in ("_action_move_frac", "_action_rest_frac", "_action_eat_frac"):
            np.testing.assert_allclose(getattr(got, attr), getattr(ref, attr),
                                       rtol=1e-4, atol=1e-6, err_msg=attr)
        np.testing.assert_array_equal(got._action_active_ticks, ref._action_active_ticks)
        np.testing.assert_allclose(got.prev_hidden_frac, ref.prev_hidden_frac,
                                   rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_run_inference_on_the_gpu_engine_follows_the_cpu_loop(device):
    """The engine switch inside ``run_inference``, with an all-random baseline."""
    cpu_env, cpu_rnd, gpu_env, gpu_rnd = (_world() for _ in range(4))
    policies, mean, var = _policies(cpu_env)
    expected = inference.run_inference(cpu_env, policies, mean, var, TICKS,
                                       verbose=False, rnd_env=cpu_rnd)
    got = inference.run_inference(gpu_env, policies, mean, var, TICKS,
                                  verbose=False, rnd_env=gpu_rnd,
                                  engine="gpu", device=device)
    for fid in expected:
        np.testing.assert_allclose(got[fid], expected[fid], rtol=1e-4, atol=1e-6,
                                   err_msg=fid)
        np.testing.assert_allclose(float(gpu_rnd.fgs[fid].biomass.sum()),
                                   float(cpu_rnd.fgs[fid].biomass.sum()),
                                   rtol=1e-4, atol=1e-6, err_msg=fid)


@pytest.mark.parametrize("device", DEVICES)
def test_live_impact_change_reaches_the_tensor_engine(device):
    envs = [_world(), _world()]
    policies, mean, var = _policies(envs[0])
    stepper = GpuStepper(envs, policies, mean, var, random_dms=[{envs[0].dm_ids[0]}],
                         device=device)
    try:
        if not stepper.model.has_impacts:
            pytest.skip("project without impact tables")
        before = stepper.runner.impact_loss.clone()
        base = {iid: np.array(envs[0].grid.get_map(iid), copy=True)
                for iid in stepper.model.impact_ids
                if envs[0].grid.get_map(iid) is not None}
        for env in envs:
            inference.apply_impact_scales(env, base, {iid: 2.0 for iid in base})
        stepper.refresh_impacts()
        expected = stepper.model.import_impacts(envs)[0]
        torch.testing.assert_close(stepper.runner.impact_loss, expected)
        assert not torch.equal(stepper.runner.impact_loss, before)
    finally:
        stepper.close()


def test_engine_defaults_to_cpu_without_cuda():
    """Default gpu where there is one (section 238); without CUDA the
    run falls back to the CPU engine instead of failing."""
    out = subprocess.run([sys.executable, str(ROOT / "inference.py"),
                          "--checkpoints", str(ROOT / "no_such_dir")],
                         cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert "running --engine cpu" in out.stderr
    assert "Engine:       cpu" in out.stdout
    assert "checkpoint directory does not exist" in out.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
def test_engine_defaults_to_gpu_with_cuda():
    out = subprocess.run([sys.executable, str(ROOT / "inference.py"),
                          "--checkpoints", str(ROOT / "no_such_dir")],
                         cwd=ROOT, capture_output=True, text=True)
    assert "Engine:       gpu" in out.stdout
