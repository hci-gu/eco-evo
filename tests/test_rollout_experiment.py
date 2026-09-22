"""Experiment isolation, resumability, reporting and shared progress rendering."""

import json

import numpy as np
import pytest

from lib.runners.progress_plot import plot_series, running_average
from tools.rollout_experiment import (checkpoint_progress, experiment_config, main,
                                      make_report, parser, read_records, run_job,
                                      training_command, write_json)
from train_gpu import main as gpu_main


def test_trailing_average_is_causal_and_handles_short_or_missing_windows():
    np.testing.assert_allclose(running_average([10, 20, 90, 0], 2), [10, 15, 55, 45])
    np.testing.assert_allclose(running_average([None, 20, None, 40], 2),
                               [np.nan, 20, 20, 40], equal_nan=True)
    assert running_average([]).size == 0
    with pytest.raises(ValueError):
        running_average([1], 0)


def test_raw_points_have_half_opacity_and_only_mean_is_connected():
    from matplotlib.figure import Figure
    ax = Figure().subplots()
    plot_series(ax, [0, 1, 2], [10, 30, 20], "porpoises", 2)
    assert len(ax.lines) == len(ax.collections) == 1
    assert ax.collections[0].get_alpha() == .5
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], [10, 30, 20])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [10, 20, 25])


def test_experiment_defaults_and_only_horizon_seed_output_differ(tmp_path):
    args = parser().parse_args([])
    config = experiment_config(args)
    assert config["generations"] == 25
    assert config["eval_ticks"] == 5000
    assert config["rollouts"] == [100, 300, 1000]
    first = training_command(config, dict(rollout=100, seed=0), tmp_path / "first")
    second = training_command(config, dict(rollout=300, seed=1), tmp_path / "second")
    differences = {first[i - 1] for i in range(len(first)) if first[i] != second[i]}
    assert differences == {"--ticks", "--seed", "--output"}
    assert "--survival-reward" in first and "--progress" in first
    assert "--visual" not in first


def test_headless_progress_includes_last_update_without_changing_training(tmp_path):
    import torch
    flags = ["--project", "mareld2.yaml", "--device", "cpu", "--execution", "eager",
             "--grid", "5x6", "--n-deltas", "2", "--ticks", "2", "--generations", "1",
             "--iter-per-gen", "2", "--eval-ticks", "4", "--eval-every", "50"]
    gpu_main([*flags, "--output", str(tmp_path / "plain")])
    gpu_main([*flags, "--output", str(tmp_path / "progress"), "--progress"])
    records = read_records(tmp_path / "progress/progress/survival.jsonl")
    assert [r["step"] for r in records] == [0, 2]
    assert (tmp_path / "progress/progress/progress.png").stat().st_size > 1000
    states = [torch.load(tmp_path / name / "trainer.pth", weights_only=False)["trainer"]
              for name in ("plain", "progress")]
    for actual, expected in zip(states[0]["theta"], states[1]["theta"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_two_job_experiment_report_and_resume(tmp_path):
    flags = ["--output", str(tmp_path), "--device", "cpu", "--rollouts", "2", "3",
             "--seeds", "0", "--generations", "1", "--iter-per-gen", "1",
             "--eval-ticks", "4", "--grid", "5x6", "--n-deltas", "2", "--worlds", "1"]
    assert main(flags) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [j["status"] for j in manifest["jobs"]] == ["complete", "complete"]
    for job in manifest["jobs"]:
        assert checkpoint_progress(tmp_path / job["name"], 1) == (1, 1)
    original = (tmp_path / "rollout-2_seed-0/trainer.pth").read_bytes()
    assert main([*flags, "--resume"]) == 0
    assert original == (tmp_path / "rollout-2_seed-0/trainer.pth").read_bytes()
    assert main(["--output", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert "rollout-2_seed-0/progress/progress.png" in (tmp_path / "report.html").read_text()
    rows = json.loads((tmp_path / "summary.json").read_text())
    assert all(r["candidate_world_ticks"] == 4 * r["rollout"] for r in rows)
    with pytest.raises(SystemExit):
        main(flags)
    with pytest.raises(SystemExit):
        main([*flags, "--resume", "--eval-ticks", "5"])


def test_dry_run_does_not_create_output(tmp_path):
    output = tmp_path / "absent"
    assert main(["--output", str(output), "--dry-run"]) == 0
    assert not output.exists()


def test_duplicate_horizons_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path), "--rollouts", "100", "100", "--dry-run"])


def test_resume_partial_checkpoint_runs_only_remaining_generations(tmp_path):
    args = parser().parse_args(["--device", "cpu", "--generations", "1",
                               "--iter-per-gen", "2", "--eval-ticks", "3",
                               "--grid", "5x6", "--n-deltas", "2", "--worlds", "1"])
    config = experiment_config(args)
    job = dict(name="partial", rollout=2, seed=0)
    run_job(config, job, tmp_path)
    assert checkpoint_progress(tmp_path, 2) == (1, 2)
    config["generations"] = 2
    run_job(config, job, tmp_path)
    assert checkpoint_progress(tmp_path, 2) == (2, 4)
    assert "--resume" in job["command"]
    assert job["command"][job["command"].index("--generations") + 1] == "1"
    assert [r["step"] for r in read_records(tmp_path / "progress/survival.jsonl")] == [0, 2, 4]


def test_partial_runs_are_visible_but_not_aggregated_and_missing_is_not_zero(tmp_path):
    config = experiment_config(parser().parse_args(["--rollouts", "100", "--seeds", "0", "1"]))
    jobs = []
    for seed, status, value in [(0, "complete", 10), (1, "failed", 100)]:
        name = f"rollout-100_seed-{seed}"
        jobs.append(dict(name=name, rollout=100, seed=seed, status=status))
        directory = tmp_path / name / "progress"
        directory.mkdir(parents=True)
        write_json(directory / "config.json", dict(ticks=5000, lower=.1, upper=10))
        record = dict(step=0, survival_ticks={"porpoises": value, "absent": None})
        (directory / "survival.jsonl").write_text(json.dumps(record) + "\n")
    make_report(tmp_path, dict(config=config, jobs=jobs))
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate == [dict(species="porpoises", rollout=100, completed_seeds=1, mean=10., std=0.)]
    assert "failed" in (tmp_path / "report.html").read_text()
    rows = json.loads((tmp_path / "summary.json").read_text())
    assert all(r["last_window_mean"] is None for r in rows if r["species"] == "absent")
