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
