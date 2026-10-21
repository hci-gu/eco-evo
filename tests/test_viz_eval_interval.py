"""The training viewer's "Evaluation probe interval" slider (row 4).

It sets how many training updates pass between two survival progress
evaluations (--eval-every) and between two biomass probes of the viewer:
linear 1-50, locked until its padlock is clicked, hidden in inference.
Both read the override before each update and otherwise keep the CLI
value; the first update of a session always probes.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.runners.training_progress import TrainingProgress, probe_due  # noqa: E402

GRID = (4, 4)


def make_viewer(monkeypatch, mode):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=["gadoids", "zooplankton"], grid_shape=GRID, mode=mode)
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    return viewer


@pytest.fixture
def training_viz(monkeypatch):
    viewer = make_viewer(monkeypatch, "training")
    yield viewer
    viewer.close()


def centre(rect):
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def test_slider_is_drawn_locked_with_the_cli_default(training_viz):
    training_viz.set_eval_interval_default(20)
    training_viz._draw_status_bar()
    assert training_viz._evalint_track_rect is not None
    assert training_viz._evalint_label() == "Evaluation probe interval = 20"
    training_viz._handle_click(centre(training_viz._evalint_track_rect))
    assert training_viz.get_eval_interval_override() is None, "locked slider moved"


def test_unlocked_slider_spans_one_to_fifty(training_viz):
    training_viz._draw_status_bar()
    training_viz._handle_click(centre(training_viz._evalint_lock_rect))
    assert not training_viz._evalint_locked
    x, y, w, h = training_viz._evalint_track_rect
    training_viz._handle_click((x + 2, y + h // 2))
    assert training_viz.get_eval_interval_override() == 1
    training_viz._handle_click((x + w - 3, y + h // 2))
    assert training_viz.get_eval_interval_override() == 50
    assert training_viz._evalint_label() == "Evaluation probe interval* = 50"


def test_inference_has_no_interval_slider(monkeypatch):
    viewer = make_viewer(monkeypatch, "inference")
    try:
        viewer._draw_status_bar()
        assert viewer._evalint_track_rect is None and viewer._evalint_lock_rect is None
    finally:
        viewer.close()


def _monitor(tmp_path, every, override):
    viz = SimpleNamespace(enabled=True, get_eval_interval_override=lambda: override[0])
    monitor = TrainingProgress("gpu", every, 5, 0.3, 3.0, 100, 1.0, tmp_path, visualizer=viz)
    monitor.config, monitor.records = {}, []
    monitor.history = tmp_path / "survival.jsonl"
    monitor.evaluate = lambda trainer: {"ticks_run": 5}
    return monitor


def test_progress_follows_the_slider_between_updates(tmp_path):
    override = [None]
    monitor = _monitor(tmp_path, 4, override)
    args = SimpleNamespace(resume=False)
    steps = []
    for step in range(1, 30):
        if step == 9:
            override[0] = 7
        n = len(monitor.records)
        monitor(None, step, tmp_path, args)
        if len(monitor.records) > n:
            steps.append(step)
    # multiples of 4 (--eval-every) until the slider says 7
    assert steps == [4, 8, 14, 21, 28]


def test_biomass_probe_follows_the_same_slider():
    interval = [4]
    viz = SimpleNamespace(get_eval_interval=lambda: interval[0])
    steps = []
    probed = False
    for step in range(5, 30):
        if step == 13:
            interval[0] = 7
        if probe_due(viz, step, first=not probed):
            probed = True
            steps.append(step)
    # first update of the session, multiples of 4, then of 7
    assert steps == [5, 8, 12, 14, 21, 28]
    assert all(probe_due(None, s) for s in range(1, 5)), "no viewer: every update"


def test_gpu_viewer_skips_the_probe_between_intervals(monkeypatch, tmp_path):
    visual = pytest.importorskip("lib.gpu.visual")
    calls = []
    gpu = visual.GPUTrainingVisualizer.__new__(visual.GPUTrainingVisualizer)
    status = []
    gpu.viz = SimpleNamespace(update_reward=lambda *a, **k: None,
                              update_status=lambda **extra: status.append(extra["iter"]),
                              get_eval_interval=lambda: 10)
    gpu.directory, gpu.rnd_mode, gpu.probed = tmp_path, "none", False
    gpu.builder = SimpleNamespace(PROBE_SEED=1)
    gpu.probe = lambda *a, **k: calls.append(k["viz_step"])
    gpu.pump = lambda: None
    monkeypatch.setattr(visual, "policy_snapshot",
                        lambda trainer: SimpleNamespace(softmax_temperature=1.0))
    trainer = SimpleNamespace(iterations_completed=0,
                              metrics=lambda: {"reward_mean": [0.0]},
                              model=SimpleNamespace(dm_ids=["a"]))
    for step in range(477, 502):
        trainer.iterations_completed = step
        gpu.update(trainer, ["a"], 9, step % 50, 1460)
    assert calls == [477, 480, 490, 500]
    # The status row's gen / iter follow every update, probe or not.
    assert status == [step % 50 + 1 for step in range(477, 502)]


def test_status_row_updates_without_a_probe_frame(training_viz):
    training_viz.update_status(gen=10, iter=37, T=1.0)
    training_viz._draw_status_bar()
    assert training_viz._status["iter"] == 37 and training_viz._status["gen"] == 10
