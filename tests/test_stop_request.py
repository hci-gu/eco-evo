"""Ctrl+C in train_gpu.py stops the running loop, not after it (section 307).

1. A stop inside an update's rollout discards that update: the trainer's
   persistent state is the last completed update, and a resume from it
   continues exactly.
2. A long-run probe that has already finished is recorded when the run is
   stopped (it is collected only after the next update).
3. The viewer probe (CPU tick loop and tensor engine) stops and records
   nothing.
"""
import concurrent.futures
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.runners.stop_request import (StopRequested, raise_if_stopped, set_stop_check,
                                      stop_check)
from lib.runners.training_progress import LiveTrainingProgress
from test_gpu_trainer import build_trainer


@pytest.fixture
def stop_after():
    """Install a check that fires on its ``n``-th call; removed afterwards."""
    def install(n):
        calls = []
        set_stop_check(lambda: calls.append(1) or len(calls) >= n)
        return calls
    yield install
    set_stop_check(None)


def test_no_check_is_the_default_and_raises_nothing():
    assert stop_check() is None
    raise_if_stopped()


def test_raise_if_stopped_follows_the_check(stop_after):
    stop_after(2)
    raise_if_stopped()
    with pytest.raises(StopRequested):
        raise_if_stopped()
    # Neither a probe's except KeyboardInterrupt nor except Exception catches it.
    assert not issubclass(StopRequested, (KeyboardInterrupt, Exception))


def test_a_stop_mid_update_keeps_the_last_completed_update(stop_after):
    trainer = build_trainer()
    trainer.train_step(n_eval_ticks=4)
    before = trainer.state_dict()
    calls = stop_after(1)
    with pytest.raises(StopRequested):
        trainer.train_step(n_eval_ticks=4)
    assert calls
    set_stop_check(None)
    after = trainer.state_dict()
    assert after["iterations_completed"] == before["iterations_completed"] == 1
    for a, b in zip(after["theta"], before["theta"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in ("obs_mean", "obs_var", "obs_count", "iteration"):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    # The saved state resumes exactly where an uninterrupted run goes on.
    resumed = build_trainer()
    resumed.load_state_dict(after)
    trainer.train_step(n_eval_ticks=4)
    resumed.train_step(n_eval_ticks=4)
    for a, b in zip(trainer.theta, resumed.theta):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_check_is_read_between_blocks(stop_after):
    trainer = build_trainer()
    trainer.runner.graph_ticks = 2
    calls = stop_after(3)
    with pytest.raises(StopRequested):
        trainer.train_step(n_eval_ticks=10)
    # Five blocks of two ticks; the third check stops it.
    assert len(calls) == 3
    assert trainer.iterations_completed == 0


def _progress_with(future):
    progress = LiveTrainingProgress.__new__(LiveTrainingProgress)
    progress._pool = None
    progress._pending = (future, 4, 120, 0.0)
    progress._pending_overrides = None
    collected = []
    progress._collect = lambda block=False: collected.append(block) or setattr(progress, "_pending", None)
    return progress, collected


def test_a_finished_long_run_probe_is_recorded_on_a_stop():
    done = concurrent.futures.Future()
    done.set_result({})
    progress, collected = _progress_with(done)
    progress.wait(cancel=lambda: True)
    assert collected == [True]


def test_a_running_long_run_probe_is_cancelled_on_a_stop(capsys):
    running = concurrent.futures.Future()
    progress, collected = _progress_with(running)
    progress.wait(cancel=True)
    assert collected == []
    assert "waiting for probe" not in capsys.readouterr().out


@pytest.mark.parametrize("engine,device", [("cpu", "cpu"), ("gpu", "cpu"),
                                           pytest.param("gpu", "cuda", marks=pytest.mark.skipif(
                                               not torch.cuda.is_available(), reason="NVIDIA GPU required"))])
def test_a_stop_in_the_viewer_probe_records_nothing(tmp_path, stop_after, engine, device):
    import test_gpu_probe_engine as probes
    builder = probes.train._ProbeEnvBuilder(str(probes.ROOT / "mareld2.yaml"), (6, 7))
    trainer = probes._trainer(builder())
    out = tmp_path / "biomass.jsonl"
    calls = stop_after(2)
    with pytest.raises(StopRequested):
        probes.train._probe_biomass(trainer, builder, n_ticks=probes.TICKS, gen=0, it=0,
                                    jsonl_path=str(out), compact=False, engine=engine,
                                    device=device)
    assert len(calls) == 2
    assert not out.exists() or out.read_text() == ""
