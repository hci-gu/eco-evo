"""GPU profile defaults and explicit overrides through the real training CLI."""

import json

import pytest

from train_gpu import main as gpu_main


@pytest.mark.parametrize("profile", ["sanity", "info", "deep"])
def test_gpu_profile_trains_and_records_resolved_settings(profile, tmp_path):
    observed = []
    def on_step(trainer, step, directory, args):
        observed.append(step)
        assert trainer.worlds == 3
        assert trainer.entropy_coef == trainer.argmax_penalty == 0
        assert float(trainer.temperature) == 1
        assert args.profile == profile
    assert gpu_main([
        "--profile", profile, "--project", "mareld2.yaml",
        "--device", "cpu", "--execution", "eager", "--grid", "5x6",
        "--generations", "1", "--iter-per-gen", "1", "--ticks", "3", "--n-deltas", "2",
        "--population-stability", "--currents", "on", "--output", str(tmp_path),
    ], on_step=on_step) == 0
    assert observed == [0, 1]
    metadata = json.loads((tmp_path / "gpu_run.json").read_text())
    assert metadata["profile"] == profile
    assert metadata["generations"] == "1"
    assert metadata["ticks"] == 3
    assert metadata["worlds"] == 3
    assert metadata["population_stability"] is True
    assert metadata["currents"] == "on"
    assert (tmp_path / "trainer.pth").is_file()


@pytest.mark.parametrize("profile,expected", [
    ("sanity", ("10", 15, 16, 100, 8)),
    ("info", ("10", 20, 16, 150, 30)),
    ("deep", ("80", 20, 20, 200, 60)),
])
def test_gpu_profile_defaults(profile, expected):
    from train_gpu import build_parser
    from lib.gpu.cli import parse_training_args
    args = parse_training_args(build_parser(), ["--profile", profile])
    assert (args.generations, args.iter_per_gen, args.n_deltas, args.ticks,
            args.temp_anneal_gens) == expected
    assert args.worlds == 3
    assert args.entropy_coef == args.argmax_penalty == 0
    assert args.temp_start == args.temp_end == 1
    assert args.coevolution is True and args.uniform_bias_init is False


@pytest.mark.parametrize("flags", [
    ["--n-deltas", "10", "--worlds", "1", "--ticks", "15", "--iter-per-gen", "50",
     "--temp-start", "3", "--temp-end", "2", "--temp-anneal-gens", "10",
     "--entropy-coef", "0.1", "--argmax-penalty", "0.3", "--uniform-bias-init", "--no-coevolution"],
    ["--n_deltas=10", "--rollouts_per_delta=1", "--n_eval_ticks=15", "--iter_per_gen=50",
     "--temp_start=3", "--temp_end=2", "--temp_anneal_gens=10",
     "--entropy_coef=0.1", "--argmax_penalty=0.3", "--uniform_bias_init", "--no_coevolution"],
])
@pytest.mark.parametrize("profile_first", [True, False])
def test_explicit_overrides_win_even_when_equal_to_ordinary_defaults(flags, profile_first):
    from train_gpu import build_parser
    from lib.gpu.cli import parse_training_args
    profile = ["--profile", "deep"]
    args = parse_training_args(build_parser(), profile + flags if profile_first else flags + profile)
    assert (args.n_deltas, args.worlds, args.ticks, args.iter_per_gen) == (10, 1, 15, 50)
    assert (args.temp_start, args.temp_end, args.temp_anneal_gens) == (3, 2, 10)
    assert (args.entropy_coef, args.argmax_penalty) == (0.1, 0.3)
    assert args.uniform_bias_init is True and args.coevolution is False


def test_no_profile_preserves_gpu_defaults_and_parser_can_be_reused():
    from train_gpu import build_parser
    from lib.gpu.cli import parse_training_args
    parser = build_parser()
    parse_training_args(parser, ["--profile", "deep"])
    args = parse_training_args(parser, [])
    assert args.profile is None
    assert (args.generations, args.iter_per_gen, args.n_deltas, args.worlds, args.ticks) == ("inf", 20, 10, 1, 15)
    assert (args.entropy_coef, args.argmax_penalty, args.temp_start) == (0.1, 0.3, 3)


def test_unknown_profile_is_rejected():
    with pytest.raises(SystemExit) as error:
        gpu_main(["--profile", "unknown"])
    assert error.value.code == 2


def test_profile_passes_through_progress_wrapper(tmp_path, monkeypatch):
    from train_progress import main as progress_main
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    assert progress_main([
        "--backend", "gpu", "--eval-every", "1", "--eval-ticks", "2", "--",
        "--profile", "info", "--project", "mareld2.yaml",
        "--device", "cpu", "--execution", "eager", "--grid", "5x6",
        "--generations", "1", "--iter-per-gen", "1", "--ticks", "2", "--n-deltas", "2",
        "--output", str(tmp_path / "run"),
    ]) == 0
    metadata = json.loads((tmp_path / "run" / "gpu_run.json").read_text())
    assert metadata["profile"] == "info"
    assert metadata["entropy_coef"] == 0
    assert (tmp_path / "run" / "progress" / "latest.png").is_file()


def test_cpu_profile_still_applies_shared_presets_with_explicit_overrides(tmp_path):
    from train import main as cpu_main
    observed = []
    def on_step(trainer, step, directory, args):
        observed.append(step)
        assert args.profile == "sanity"
        assert args.n_eval_ticks == 2 and args.rollouts_per_delta == 1
        assert args.temp_anneal_gens == 8
        assert trainer.entropy_coef == trainer.argmax_penalty == 0
    cpu_main([
        "--profile", "sanity", "--project", "mareld2.yaml", "--grid", "5x6",
        "--generations", "1", "--iter-per-gen", "1", "--n_eval_ticks", "2", "--n_deltas", "2",
        "--workers", "1", "--rollouts_per_delta", "1", "--run-name", str(tmp_path),
    ], on_step=on_step, confirm=False)
    assert observed == [0, 1]
    assert (tmp_path / "policy_gadoids.pth").is_file()
