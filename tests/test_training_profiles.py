"""Training presets must reach the tensor trainer and respect CLI overrides."""

import json
import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_gpu import main as gpu_main
from lib.training_profiles import parse_training_args


@pytest.mark.parametrize("profile", ["sanity", "info", "deep"])
@pytest.mark.parametrize("legacy_aliases", [False, True])
def test_gpu_profile_trains_and_records_effective_settings(tmp_path, profile, legacy_aliases):
    ticks = "--n_eval_ticks" if legacy_aliases else "--ticks"
    worlds = "--rollouts_per_delta" if legacy_aliases else "--worlds"
    deltas = "--n_deltas" if legacy_aliases else "--n-deltas"
    seen = []

    def on_step(trainer, step, directory, args):
        assert trainer.n_deltas == 2
        assert trainer.worlds == 1
        assert trainer.entropy_coef == 0.1  # explicit value equals the CLI default
        assert trainer.argmax_penalty == 0
        assert float(trainer.temperature) == 1
        seen.append(step)

    assert gpu_main([
        "--project", "mareld2.yaml", "--device", "cpu", "--execution", "eager",
        "--grid", "5x6", "--output", str(tmp_path),
        f"{ticks}=2", worlds, "1", deltas, "2", "--generations", "1",
        "--iter_per_gen", "1", "--entropy_coef", "0.1",
        "--no_coevolution", "--species", "gadoids", "--uniform_bias_init",
        "--profile", profile,
    ], on_step=on_step) == 0
    assert seen == [0, 1]
    metadata = json.loads((tmp_path / "gpu_run.json").read_text())
    assert metadata["profile"] == profile
    assert metadata["ticks"] == 2
    assert metadata["worlds"] == 1
    assert metadata["coevolution"] is False
    assert metadata["uniform_bias_init"] is True
    assert metadata["temp_anneal_gens"] == {"sanity": 8, "info": 30, "deep": 60}[profile]
    checkpoint = torch.load(tmp_path / "trainer.pth", weights_only=False)
    assert json.loads(json.dumps(checkpoint["options"])) == metadata
    assert checkpoint["trainer"]["iterations_completed"] == 1


def capture_arguments(monkeypatch, script, argv):
    """Use each entry point's real parser, stopping before expensive training."""
    module = importlib.import_module(script)
    captured = []

    class Parsed(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(parse_training_args(*args, **kwargs))
        raise Parsed

    monkeypatch.setattr(module, "parse_training_args", capture)
    monkeypatch.setattr(sys, "argv", [script, *argv])
    with pytest.raises(Parsed):
        module.main()
    return captured[0]


@pytest.mark.parametrize("script", ["train", "train_gpu"])
@pytest.mark.parametrize("profile,expected", [
    (None, ("inf", 20, 10, 15, 10, 1)),
    ("sanity", ("10", 15, 16, 100, 8, 3)),
    ("info", ("10", 20, 16, 150, 30, 3)),
    ("deep", ("80", 20, 20, 200, 60, 3)),
])
def test_cpu_and_gpu_effective_presets(monkeypatch, script, profile, expected):
    args = capture_arguments(monkeypatch, script, [] if profile is None else ["--profile", profile])
    ticks = args.ticks if script == "train_gpu" else args.n_eval_ticks
    worlds = args.worlds if script == "train_gpu" else args.rollouts_per_delta
    assert (args.generations, args.iter_per_gen, args.n_deltas, ticks,
            args.temp_anneal_gens, worlds) == expected
    assert args.coevolution is True
    assert args.uniform_bias_init is False
    assert args.entropy_coef == (0.1 if profile is None else 0)
    assert args.argmax_penalty == (0.3 if profile is None else 0)
    assert args.temp_start == (3 if profile is None else 1)
    assert args.temp_end == 1


def test_cpu_profile_preserves_overrides_and_optional_flag_value(monkeypatch):
    args = capture_arguments(monkeypatch, "train", [
        "--n_eval_ticks=15", "--rollouts_per_delta", "1", "--entropy_coef", "0.1",
        "--no_coevolution", "--uniform_bias_init", "--rnd-baseline", "--profile", "deep",
    ])
    assert args.n_eval_ticks == 15
    assert args.rollouts_per_delta == 1
    assert args.entropy_coef == 0.1
    assert args.coevolution is False
    assert args.uniform_bias_init is True
    assert args.rnd_baseline == "all"


def test_gpu_unknown_profile_rejected(capsys):
    with pytest.raises(SystemExit) as error:
        gpu_main(["--profile", "unknown"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


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
