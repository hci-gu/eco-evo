import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.trainer import TensorARSTrainer
from lib.runners.trainer import ARSTrainer
from lib.runners.training_progress import (
    TrainingProgress, build_inference_env, evaluation_randomness,
    inference_config, install_current_policies, measure_survival,
)
from train import _EnvBuilder
from train_gpu import main as gpu_main
from train_progress import main as progress_main


class BiomassTrace:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.fgs = {str(i): SimpleNamespace(biomass=np.array([value], dtype=float))
                    for i, value in enumerate(next(self.rows))}

    def tick(self):
        for fg, value in zip(self.fgs.values(), next(self.rows)):
            fg.biomass[:] = value


def test_first_breach_is_per_species_inclusive_and_never_restarts():
    env = BiomassTrace([
        [10, 10, 10, 0, 10, 10],
        [3, 30, 2, 0, np.nan, 10],  # exact bounds pass; immediate breach scores 0
        [2, 31, 10, 1, 10, 10],
        [10, 10, 10, 1, 10, 2],    # recovery does not restart; breach at cap scores 2
    ])
    result = measure_survival(env, 3, 0.3, 3.0)
    assert result["survival_ticks"] == {"0": 1, "1": 1, "2": 0, "3": None, "4": 0, "5": 2}
    assert measure_survival(BiomassTrace([[10], [10], [10]]), 2, 0.3, 3)["survival_ticks"] == {"0": 2}


def test_evaluation_rng_restores_all_cpu_generators_even_on_failure():
    np_state, py_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    with pytest.raises(RuntimeError):
        with evaluation_randomness(100):
            np.random.random(20)
            random.random()
            torch.rand(20)
            raise RuntimeError("interrupted evaluation")
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert random.getstate() == py_state
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def build_trainer(backend):
    torch.set_num_threads(1)
    if backend == "gpu":
        spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6)))
        return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2, hidden_dim=7)
    builder = _EnvBuilder(project_path="mareld2.yaml", grid_size=(5, 6))
    env = builder(seed=10)
    params = {f: (int(env.per_dm_in_dim[i]), env.N_all + 5) for i, f in enumerate(env.dm_ids)}
    return ARSTrainer(builder, params, n_deltas=2, n_workers=1, hidden_dim=7)


def update(trainer, backend):
    if backend == "gpu":
        trainer.train_step(n_eval_ticks=3)
    else:
        trainer.train_step_coevolution(list(trainer.policies), n_eval_ticks=3)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_evaluation_uses_current_weights_and_stats_without_changing_next_update(backend, tmp_path):
    trainer = build_trainer(backend)
    update(trainer, backend)
    config = inference_config(trainer, backend)
    with evaluation_randomness(100):
        env = build_inference_env(config, 100)
        install_current_policies(env, trainer, backend)
    for i, fid in enumerate(env.dm_ids):
        actual = torch.nn.utils.parameters_to_vector(env.policies[fid].parameters())
        expected = trainer.theta[i] if backend == "gpu" else torch.nn.utils.parameters_to_vector(trainer.policies[fid].parameters())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        mean = trainer.obs_mean[i].numpy() if backend == "gpu" else trainer.obs_stats[fid]["mean"]
        np.testing.assert_array_equal(env.obs_mean[i], mean.astype(np.float32))
    monitor = TrainingProgress(backend, 2, 5, 0.3, 3.0, 100, 1.0, tmp_path)
    monitor._start(trainer, 1, tmp_path, False)
    # Compare the actual next update with/without a probe using identical RNGs.
    control = copy.deepcopy(trainer)
    with evaluation_randomness(200):
        first = monitor.evaluate(trainer)
        assert monitor.evaluate(trainer) == first
        update(trainer, backend)
    with evaluation_randomness(200):
        update(control, backend)
    if backend == "gpu":
        for actual, expected in zip(trainer.theta, control.theta):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(trainer.obs_mean, control.obs_mean, rtol=0, atol=0)
        torch.testing.assert_close(trainer.obs_count, control.obs_count, rtol=0, atol=0)
        assert trainer.iterations_completed == control.iterations_completed
    else:
        for fid in trainer.policies:
            for actual, expected in zip(trainer.policies[fid].parameters(), control.policies[fid].parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            np.testing.assert_array_equal(trainer.obs_stats[fid]["mean"], control.obs_stats[fid]["mean"])
            assert trainer.obs_stats[fid]["count"] == control.obs_stats[fid]["count"]


def test_wrapper_updates_one_image_at_interval_and_resumes_history(tmp_path):
    trainer_args = ["--project", "mareld2.yaml", "--device", "cpu", "--execution", "eager",
                    "--grid", "5x6", "--n-deltas", "2", "--ticks", "2", "--generations", "1",
                    "--iter-per-gen", "3", "--output", str(tmp_path)]
    wrapper = ["--eval-every", "2", "--eval-ticks", "5", "--"]
    assert progress_main(wrapper + trainer_args) == 0
    directory = tmp_path / "progress"
    def steps():
        return [r["step"] for r in map(json.loads, (directory / "survival.jsonl").read_text().splitlines())]
    assert steps() == [0, 2]
    assert {p.name for p in directory.glob("*.png")} == {"latest.png"}
    first_image = (directory / "latest.png").read_bytes()
    with Image.open(directory / "latest.png") as image:
        assert image.format == "PNG"
        assert min(image.size) > 500
        image.verify()
    assert progress_main(wrapper + trainer_args + ["--resume"]) == 0
    assert steps() == [0, 2, 3, 4, 6]
    assert (directory / "latest.png").read_bytes() != first_image
    records = list(map(json.loads, (directory / "survival.jsonl").read_text().splitlines()))
    assert [r["evaluation"] for r in records] == [1, 2, 3, 4, 5]
    # Simulate an evaluation written after the last checkpoint, then a resume.
    with (directory / "survival.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(step=100, survival_ticks={})) + "\n")
    assert progress_main(wrapper + trainer_args + ["--resume"]) == 0
    assert steps() == [0, 2, 3, 4, 6, 8]
    assert {p.name for p in directory.glob("*.png")} == {"latest.png"}


def test_round_robin_callback_counts_each_optimizer_update(tmp_path):
    observed = []
    gpu_main(["--project", "mareld2.yaml", "--device", "cpu", "--execution", "eager",
              "--grid", "5x6", "--n-deltas", "2", "--ticks", "2", "--generations", "1",
              "--iter-per-gen", "1", "--species", "gadoids", "pelagic_fish", "--no-coevolution",
              "--output", str(tmp_path)], on_step=lambda trainer, step, *args: observed.append(step))
    assert observed == [0, 1, 2]


@pytest.mark.parametrize("flags", [
    ["--eval-every", "0"], ["--eval-ticks", "-1"], ["--biomass-bounds", "3", "0.3"],
    ["--biomass-bounds", "nan", "3"], ["--biomass-bounds", "2", "3"],
    ["--eval-temperature", "0"], ["--eval-seed", "-1"],
])
def test_invalid_options_are_rejected_before_training(flags):
    with pytest.raises(SystemExit) as error:
        progress_main(flags)
    assert error.value.code == 2


def test_resume_rejects_incomparable_evaluation_settings(tmp_path):
    trainer = build_trainer("gpu")
    monitor = TrainingProgress("gpu", 2, 5, 0.3, 3.0, 100, 1.0, tmp_path)
    monitor._start(trainer, 0, tmp_path, False)
    changed = TrainingProgress("gpu", 2, 5, 0.5, 3.0, 100, 1.0, tmp_path)
    with pytest.raises(ValueError, match="settings changed"):
        changed._start(trainer, 0, tmp_path, True)
