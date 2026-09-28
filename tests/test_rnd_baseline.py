"""--rnd_baseline [all|solo|none] in inference.py and train_gpu.py (section 96)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import RandomPolicy, build_env, run_inference
from lib.runners.policy import PolicyNetwork
from lib.runners.rnd_baseline import add_rnd_baseline_argument, rnd_baseline_plot_ids

GRID = (5, 6)


def parse(*argv):
    parser = argparse.ArgumentParser()
    add_rnd_baseline_argument(parser)
    return parser.parse_args(list(argv)).rnd_baseline


def test_cli_semantics_match_train_py():
    assert parse() == "none"
    assert parse("--rnd_baseline") == "all"          # bare flag, section 62.1
    assert parse("--rnd-baseline", "solo") == "solo"
    with pytest.raises(SystemExit):
        parse("--rnd_baseline", "some")


def test_plot_ids_all_reports_ndms_solo_only_dms():
    fgs, dms = ["phytoplankton", "zooplankton", "gadoids"], ["zooplankton", "gadoids"]
    assert rnd_baseline_plot_ids("all", fgs, dms) == [f + "_rnd" for f in fgs]
    assert rnd_baseline_plot_ids("solo", fgs, dms) == ["zooplankton_rnd", "gadoids_rnd"]
    assert rnd_baseline_plot_ids("none", fgs, dms) is None
    assert rnd_baseline_plot_ids(None, fgs, dms) is None


def _world():
    # The env constructor draws season phases from the global RNG, and every
    # tick shuffles the FG order with it, so identical worlds need it pinned.
    np.random.seed(0)
    env = build_env("mareld2.yaml", GRID, seed=7, verbose=False)
    env.build_static_caches()
    return env


def _trained(env):
    torch.manual_seed(0)
    policies = {fid: PolicyNetwork(int(env.per_dm_in_dim[i]), 5 + env.N_all).eval()
                for i, fid in enumerate(env.dm_ids)}
    mean = np.zeros((env.N_dm, int(env.max_in_dim)), dtype=np.float32)
    var = np.ones_like(mean)
    return policies, mean, var


class Recorder:
    """The slice of LiveVisualizer that run_inference touches."""

    def __init__(self):
        self.series = {}

    def update_series(self, tab, key, value, step=None):
        self.series.setdefault(tab, {}).setdefault(key, []).append(value)

    def pump_events(self):
        return True

    def __getattr__(self, name):
        if name.startswith("_"):          # dirty flags etc. read via getattr
            raise AttributeError(name)
        return lambda *a, **k: None       # every other viz call is a no-op


def test_solo_worlds_randomise_exactly_one_dm_and_report_only_it():
    env = _world()
    policies, mean, var = _trained(env)
    solo = {fid: _world() for fid in env.dm_ids}
    viz = Recorder()
    history = run_inference(env, policies, mean, var, 3, verbose=False, viz=viz,
                            rnd_solo_envs=solo)
    for k, world in solo.items():
        for fid, policy in world.policies.items():
            assert isinstance(policy, RandomPolicy) == (fid == k)
            if fid != k:
                assert policy is policies[fid]
    rnd_keys = {key for key in viz.series["biomass"] if key.endswith("_rnd")}
    assert rnd_keys == {fid + "_rnd" for fid in env.dm_ids}
    for fid, world in solo.items():
        # The reported curve is read from the world where fid is random.
        b0 = viz.series["biomass"][fid + "_rnd"]
        assert len(b0) == 3
        assert b0[0] > 0.0
    moved = {key for key in viz.series.get("move", {}) if key.endswith("_rnd")}
    assert moved <= {fid + "_rnd" for fid in env.dm_ids}
    assert set(history) == set(env.fgs)


def test_solo_world_with_one_random_dm_diverges_only_through_that_dm():
    """A solo world whose random DM is replaced by its trained policy again
    must reproduce the main world exactly - so any difference in a solo
    curve is caused by the one random DM, never by the setup."""
    env = _world()
    policies, mean, var = _trained(env)
    k = env.dm_ids[0]
    solo = {k: _world()}
    run_inference(env, policies, mean, var, 0, verbose=False, rnd_solo_envs=solo)
    twin = solo[k]
    twin.policies[k] = policies[k]
    twin.obs_mean, twin.obs_var = mean.copy(), var.copy()
    main = _world()
    np.random.seed(1)
    run_inference(main, policies, mean, var, 4, verbose=False)
    np.random.seed(1)
    for _ in range(4):
        twin.step(twin.policy_controller.forward(twin.get_observation()))
    for fid in env.fgs:
        # Per-DM path (twin) vs batched path (main): same maths, different
        # float32 summation order.
        np.testing.assert_allclose(twin.fgs[fid].biomass, main.fgs[fid].biomass,
                                   rtol=1e-4, atol=1e-5)


def test_all_mode_keeps_its_old_behaviour():
    env = _world()
    policies, mean, var = _trained(env)
    rnd = _world()
    viz = Recorder()
    run_inference(env, policies, mean, var, 2, verbose=False, viz=viz, rnd_env=rnd)
    assert all(isinstance(p, RandomPolicy) for p in rnd.policies.values())
    assert {k for k in viz.series["biomass"] if k.endswith("_rnd")} == \
        {fid + "_rnd" for fid in env.fgs}


# ---------------------------------------------------------------- GPU viewer


@pytest.fixture
def viewers(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import lib.viz
    from lib.viz.pygame_viz import LiveVisualizer
    created = []

    def create(**kwargs):
        viz = LiveVisualizer(**kwargs)
        created.append(viz)
        return viz

    monkeypatch.setattr(lib.viz, "LiveVisualizer", create)
    yield created
    for viz in created:
        viz.close()


def gpu_arguments(directory):
    return ["--project", "mareld2.yaml", "--device", "cpu", "--execution", "eager",
            "--grid", "5x6", "--n-deltas", "2", "--ticks", "2", "--generations", "1",
            "--iter-per-gen", "1", "--output", str(directory), "--profile", "sanity",
            "--worlds", "1", "--eval-every", "1", "--eval-ticks", "5", "--visual"]


@pytest.mark.parametrize("mode", ["all", "solo"])
def test_gpu_viewer_draws_the_baseline(tmp_path, viewers, mode):
    from train_gpu import main
    assert main(gpu_arguments(tmp_path) + ["--rnd_baseline", mode]) == 0
    viz = viewers[0]
    rnd = {key for key in viz._series["biomass"] if key.endswith("_rnd")
           and viz._series["biomass"][key]}
    dms = {"gadoids", "pelagic_fish", "porpoises", "zooplankton"}
    assert {fid + "_rnd" for fid in dms} <= rnd
    assert ("phytoplankton_rnd" in rnd) == (mode == "all")
    assert viz._series["reward"]["gadoids_rnd"]


def test_gpu_rnd_baseline_requires_visual(tmp_path):
    from train_gpu import main
    argv = [a for a in gpu_arguments(tmp_path) if a != "--visual"]
    with pytest.raises(SystemExit):
        main(argv + ["--rnd_baseline", "solo"])


def test_gpu_snapshot_carries_the_reward_settings():
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    from lib.gpu.visual import policy_snapshot
    trainer = TensorARSTrainer(ProjectSpec(EnvironmentBuilder("mareld2.yaml", grid=GRID)),
                               device="cpu", execution="eager", n_deltas=2,
                               legacy_reward=True, integral_reward=False, alpha=0.5,
                               beta=0.25, survival_bonus=2.0, survival_threshold=0.2)
    try:
        snap = policy_snapshot(trainer)
        assert (snap.legacy_reward, snap.integral_reward) == (True, False)
        assert (snap.alpha, snap.beta, snap.survival_bonus, snap.survival_threshold) == \
            (0.5, 0.25, 2.0, 0.2)
        assert snap.local_reward is None and snap.population_stability is None
    finally:
        trainer.close()
