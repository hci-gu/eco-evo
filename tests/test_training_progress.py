import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


class BiomassTrace:
    def __init__(self, rows, dm_ids=None):
        self.rows = iter(rows)
        self.fgs = {str(i): SimpleNamespace(biomass=np.array([value], dtype=float))
                    for i, value in enumerate(next(self.rows))}
        self.dm_ids = list(self.fgs) if dm_ids is None else dm_ids
        self.tick_count = 0

    def tick(self):
        for fg, value in zip(self.fgs.values(), next(self.rows)):
            fg.biomass[:] = value
        self.tick_count += 1


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


def test_evaluation_stops_when_all_decision_makers_leave_bounds():
    env = BiomassTrace([
        [10, 10, 100],
        [2, 10, 100],  # first DM fails, second DM keeps the rollout running
        [10, 31, 100],  # second DM fails; first DM's recovery does not reset it
    ], dm_ids=["0", "1"])
    result = measure_survival(env, 5000, 0.3, 3)
    assert env.tick_count == result["ticks_run"] == 2
    assert result["survival_ticks"] == {"0": 0, "1": 1}
    assert result["initial_biomass"] == {"0": 10, "1": 10}
    assert env.fgs["2"].biomass.sum() == 100  # non-acting group is still alive


@pytest.mark.parametrize("dm_ids", [[], ["0"]])
def test_evaluation_skips_ticks_when_no_decision_makers_have_starting_biomass(dm_ids):
    env = BiomassTrace([[0, 100]], dm_ids=dm_ids)
    result = measure_survival(env, 5000, 0.3, 3)
    assert env.tick_count == result["ticks_run"] == 0


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
        assert set(first["survival_ticks"]) == set(env.dm_ids)
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


def test_round_robin_callback_counts_each_optimizer_update(tmp_path):
    observed = []
    gpu_main(["--project", "mareld2.yaml", "--device", "cpu", "--execution", "eager",
              "--grid", "5x6", "--n-deltas", "2", "--ticks", "2", "--generations", "1",
              "--iter-per-gen", "1", "--species", "benthivorous_fish", "planktivorous_fish", "--no-coevolution",
              "--output", str(tmp_path)], on_step=lambda trainer, step, *args: observed.append(step))
    assert observed == [0, 1, 2]


@pytest.mark.parametrize("flags", [
    ["--eval-every", "0"], ["--eval-ticks", "-1"], ["--biomass-bounds", "3", "0.3"],
    ["--biomass-bounds", "nan", "3"], ["--biomass-bounds", "2", "3"],
    ["--eval-temperature", "0"], ["--eval-seed", "-1"],
])
def test_invalid_options_are_rejected_before_training(flags):
    with pytest.raises(SystemExit) as error:
        gpu_main(flags)
    assert error.value.code == 2


def test_resume_rejects_incomparable_evaluation_settings(tmp_path):
    trainer = build_trainer("gpu")
    monitor = TrainingProgress("gpu", 2, 5, 0.3, 3.0, 100, 1.0, tmp_path)
    monitor._start(trainer, 0, tmp_path, False)
    changed = TrainingProgress("gpu", 2, 5, 0.5, 3.0, 100, 1.0, tmp_path)
    with pytest.raises(ValueError, match="settings changed"):
        changed._start(trainer, 0, tmp_path, True)


def _progress_args(**overrides):
    args = SimpleNamespace(eval_ticks=None, biomass_bounds=(0.3, 3.0),
                           eval_seed=20260530, eval_temperature=1.0)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("rollout", [{"n_eval_ticks": 1460}, {"ticks": 400}])
def test_the_default_probe_is_three_years(rollout):
    """Section 223.12: the probe runs three years of the probe world's
    (library) tick length, whatever the training rollout or tick length."""
    from lib.runners.training_progress import validate_progress_arguments
    args = _progress_args(**rollout)
    validate_progress_arguments(None, args)
    assert args.eval_ticks == 3 * 1460 and args.eval_ticks_default
    hourly = _progress_args(tick_length=3, **rollout)
    validate_progress_arguments(None, hourly)
    assert hourly.eval_ticks == 3 * 1460


def test_an_explicit_eval_ticks_wins():
    from lib.runners.training_progress import validate_progress_arguments
    args = _progress_args(n_eval_ticks=1460, eval_ticks=500)
    validate_progress_arguments(None, args)
    assert args.eval_ticks == 500 and not args.eval_ticks_default


def test_gpu_parser_resolves_the_three_year_probe():
    from lib.gpu.cli import parse_training_args
    from lib.runners.training_progress import validate_progress_arguments
    from train_gpu import build_parser
    parser = build_parser()
    args = parse_training_args(parser, ["--project", "mareld2.yaml", "--n_eval_ticks", "1460"])
    validate_progress_arguments(parser, args)
    assert (args.ticks, args.eval_ticks) == (1460, 4380)


def test_resume_keeps_the_runs_eval_ticks_unless_one_is_given(tmp_path):
    trainer = build_trainer("gpu")
    TrainingProgress("gpu", 2, 5, 0.3, 3.0, 100, 1.0, tmp_path)._start(trainer, 0, tmp_path, False)
    followed = TrainingProgress("gpu", 2, 9, 0.3, 3.0, 100, 1.0, tmp_path, ticks_default=True)
    followed._start(trainer, 0, tmp_path, True)
    assert followed.ticks == 5 and followed.config["ticks"] == 5
    explicit = TrainingProgress("gpu", 2, 9, 0.3, 3.0, 100, 1.0, tmp_path)
    with pytest.raises(ValueError, match="settings changed"):
        explicit._start(trainer, 0, tmp_path, True)


# ---- viability score on the probe (section 223.18) -------------------------

import json  # noqa: E402
import math  # noqa: E402

from lib.runners.training_progress import format_viability, probe_viability, run_probe  # noqa: E402

YEAR = 10  # ticks per year in the synthetic series; one sample per tick


def yearly(*factors, season=None):
    """A daily ratio series: year k sits at factors[k], times a season."""
    out = [1.0]
    for f in factors:
        out += [f * (1.0 if season is None else season(d)) for d in range(YEAR)]
    return out


def test_a_seasonal_cycle_passes_and_a_slide_fails_on_drift():
    # A tenfold seasonal swing, the same every year: whole-year means
    # cancel the season, so it is stationary (the two-window ratio of the
    # rig would read the phase).
    saw = lambda d: 0.1 + 1.8 * d / (YEAR - 1)  # mean 1, min 0.1
    series = {"plankton": yearly(1, 1, 1, season=saw),
              "slide": yearly(0.9, 0.6, 0.25),       # year 3 / year 2 = 0.42
              "absent": None}
    v = probe_viability(series, 1, YEAR)
    assert v["years"] == [2, 3] and v["present"] == 2 and v["viable"] == 1
    plankton, slide = v["groups"]["plankton"], v["groups"]["slide"]
    assert plankton["viable"] and plankton["drift"] == pytest.approx(1.0)
    assert plankton["margin"] == pytest.approx(1.0) and plankton["binding"] in ("floor", "ceiling")
    assert not slide["viable"] and slide["binding"] == "drift"
    assert slide["drift"] == pytest.approx(2.4)
    assert v["worst"] == "slide" and v["score"] == slide["margin"] < 0
    assert "absent" not in v["groups"]


def test_margins_are_distances_to_the_bounds_not_to_the_start():
    v = probe_viability({"low": yearly(0.3, 0.3, 0.3), "high": yearly(20, 20, 20),
                         "settled": yearly(0.5, 0.25, 0.23)}, 1, YEAR)
    low, high, settled = (v["groups"][f] for f in ("low", "high", "settled"))
    # 0.3x spawn: log10(3) of the one decade to the floor.
    assert low["viable"] and low["binding"] == "floor"
    assert low["margin"] == pytest.approx(math.log10(3))
    assert not high["viable"] and high["binding"] == "ceiling"
    # A slow decline above the floor passes, with an indicative horizon.
    assert settled["viable"]
    assert settled["years_to_floor"] == pytest.approx(math.log(2.3) / math.log(0.25 / 0.23))


def test_extinction_anywhere_fails_and_has_no_margin():
    series = {"gone": yearly(1, 1, 1), "ok": yearly(1, 1, 1)}
    series["gone"][5] = 0.0  # dips to zero in year 1, recovers
    v = probe_viability(series, 1, YEAR)
    g = v["groups"]["gone"]
    assert g["binding"] == "extinct" and g["margin"] is None and not g["viable"]
    assert v["worst"] == "gone" and v["score"] is None
    json.dumps(v, allow_nan=False)  # the jsonl writer refuses NaN / inf
    assert "worst: gone extinct" in format_viability(v)[0]
    ok = probe_viability({"a": yearly(0.3, 0.3, 0.3)}, 1, YEAR)
    assert "worst: a +0.48 to floor" in format_viability(ok)[0]


def test_only_complete_years_are_scored():
    assert probe_viability({"a": yearly(1)[:YEAR]}, 1, YEAR) is None
    one = probe_viability({"a": yearly(1, 0.2)[:YEAR + 5]}, 1, YEAR)
    assert one["years"] == [1] and one["groups"]["a"]["drift"] is None
    # Daily samples of a 6 h tick: 365 per year, day 0 = the start.
    days = [1.0] + [1.0] * 365 + [2.0] * 365
    two = probe_viability({"a": days}, 4, 1460)
    assert two["groups"]["a"]["annual_mean"] == [1.0, 2.0]


class YearTrace:
    """A two-group env whose biomass halves every probe year."""

    def __init__(self):
        self.fgs = {f: SimpleNamespace(biomass=np.array([10.0])) for f in ("fish", "algae")}
        self.dm_ids, self.global_fg_order, self.tick_hours, self.t = ["fish"], ["fish", "algae"], 6, 0

    def tick(self):
        self.t += 1
        self.fgs["fish"].biomass[:] = 10.0 * 0.5 ** (self.t / 1460)


def test_the_probe_record_keeps_the_viability_block(tmp_path, capsys):
    result = run_probe(YearTrace(), 2 * 1460, 0.3, 3.0)
    v = result["viability"]
    assert v["years"] == [1, 2] and v["present"] == 2
    assert v["groups"]["fish"]["drift"] == pytest.approx(2.0, rel=1e-3)
    progress = TrainingProgress("cpu", 1, 2 * 1460, 0.3, 3.0, 0, 1.0)
    progress.history = tmp_path / "probe.jsonl"
    progress.latest_path = tmp_path / "probe_latest.json"
    progress._record(1, 5, 0.0, result)
    stored = json.loads(progress.history.read_text())
    assert stored["viability"]["groups"]["fish"]["annual_mean"] == v["groups"]["fish"]["annual_mean"]
    assert "trajectory" not in stored
    assert "[progress] viability probe=1, step=5:" in capsys.readouterr().out


# ---- Ctrl+C and the live probe's worker process -----------------------------

def _live_with_running_probe(seconds=60):
    import concurrent.futures
    import multiprocessing
    import time as _time
    from lib.runners.training_progress import (
        LiveTrainingProgress, _ignore_sigint, _sigint_ignored_by_children)
    progress = LiveTrainingProgress("cpu", 1, 10, 0.3, 3.0, 0, 1.0)
    progress._pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        initializer=_ignore_sigint)
    with _sigint_ignored_by_children():  # as _submit does
        future = progress._pool.submit(_time.sleep, seconds)
    progress._pending = (future, 7, 70, _time.perf_counter())
    # No wait for the worker's imports: it must be covered from the start.
    return progress, list(progress._pool._processes.values())


@pytest.mark.parametrize("cancel", ["bool", "flag"])
def test_ctrl_c_neither_kills_the_probe_worker_nor_prints_a_traceback(cancel, capfd):
    import os
    import signal
    import threading
    import time as _time
    progress, workers = _live_with_running_probe()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored
    for _ in range(3):  # what the terminal sends the group, while it starts up and after
        for worker in workers:
            os.kill(worker.pid, signal.SIGINT)
        _time.sleep(1.0)
    assert all(w.is_alive() for w in workers)
    started = _time.perf_counter()
    if cancel == "bool":
        progress.wait(cancel=True)
    else:
        # train_gpu's handler only sets a flag; wait polls it.
        stop = threading.Event()
        threading.Timer(1.0, stop.set).start()
        progress.wait(cancel=stop.is_set)
    assert _time.perf_counter() - started < 10
    assert not any(w.is_alive() for w in workers)
    assert progress._pool is None and progress._pending is None
    out, err = capfd.readouterr()
    assert "probe 7 (update 70) cancelled" in out
    assert "Traceback" not in err and "KeyboardInterrupt" not in err
