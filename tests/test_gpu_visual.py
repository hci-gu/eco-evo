"""Exercise the real viewer and GPU CLI with SDL's offscreen test driver."""

import json
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_gpu import main


@pytest.fixture
def viewers(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import lib.viz
    from lib.viz.pygame_viz import LiveVisualizer
    created = []

    def create(**kwargs):
        viz = LiveVisualizer(**kwargs)
        assert viz.enabled
        created.append(viz)
        return viz

    monkeypatch.setattr(lib.viz, "LiveVisualizer", create)
    yield created
    for viz in created:
        viz.close()


def arguments(directory, device="cpu", execution="eager"):
    return ["--project", "mareld2.yaml", "--device", device, "--execution", execution,
            "--grid", "5x6", "--n-deltas", "2", "--ticks", "2", "--generations", "1",
            "--iter-per-gen", "2", "--output", str(directory), "--profile", "sanity",
            "--worlds", "1", "--eval-every", "1", "--eval-ticks", "5"]


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_same(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("device,execution", [
    ("cpu", "eager"),
    pytest.param("cuda", "cuda-graph", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="NVIDIA GPU required")),
    pytest.param("cuda", "compile-graph", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="NVIDIA GPU required")),
])
def test_visual_cli_renders_and_preserves_training(tmp_path, viewers, device, execution):
    seen = []

    def on_step(trainer, step, directory, args):
        assert threading.current_thread() is threading.main_thread()
        seen.append(step)
        if step == 2:
            import pygame
            pygame.image.save(viewers[0]._screen, str(tmp_path / "viewer.png"))

    assert main(arguments(tmp_path / "visual", device, execution) + ["--visual"], on_step=on_step) == 0
    assert seen == [0, 1, 2]
    assert len(viewers) == 1
    viz = viewers[0]
    assert set(viz._biomass) == set(viz.fg_ids)
    assert all(array.shape == (5, 6) for array in viz._biomass.values())
    assert len(viz._series["reward"]["gadoids"]) == 2
    assert len(viz._series["biomass"]["gadoids"]) == 2
    records = [json.loads(line) for line in (tmp_path / "visual" / "biomass.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in records] == [1, 2]
    assert all(r["reward"] and r["n_ticks"] == 2 for r in records)
    assert main(arguments(tmp_path / "plain", device, execution)) == 0
    visual = torch.load(tmp_path / "visual" / "trainer.pth", weights_only=False)
    plain = torch.load(tmp_path / "plain" / "trainer.pth", weights_only=False)
    assert_same(visual["trainer"], plain["trainer"])


def test_progress_tab_continues_across_resume(tmp_path, viewers):
    def render_progress(trainer, step, directory, args):
        if step == 2:
            import pygame
            viz = viewers[-1]
            # Click the tab using the actual hit boxes, then capture its render.
            rect, _ = next((rect, i) for rect, i in viz._tab_rects if viz._tabs[i] == "progress")
            viz._handle_click((rect[0] + 2, rect[1] + 2))
            viz._render_full()
            pygame.image.save(viz._screen, str(tmp_path / "progress-tab.png"))

    flags = arguments(tmp_path) + ["--visual"]
    assert main(flags, on_step=render_progress) == 0
    history = tmp_path / "progress" / "survival.jsonl"
    assert not (tmp_path / "progress" / "latest.png").exists()
    assert [r["step"] for r in map(json.loads, history.read_text().splitlines())] == [0, 1, 2]
    # A crash may leave evaluations ahead of the saved checkpoint.
    with history.open("a") as stream:
        stream.write(json.dumps(dict(step=99, evaluation=99, survival_ticks={})) + "\n")
    assert main(flags + ["--resume"]) == 0
    records = list(map(json.loads, history.read_text().splitlines()))
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4]
    assert [r["evaluation"] for r in records] == [1, 2, 3, 4, 5]
    assert [s for s, _ in viewers[-1]._series["progress"]["gadoids"]] == [0, 1, 2, 3, 4]


def test_progress_history_outlives_rolling_plots_and_playback(tmp_path, viewers):
    def populate(trainer, step, directory, args):
        if step != 2:
            return
        viz = viewers[0]
        records = [dict(step=i * 20, survival_ticks={"gadoids": i % 1000, "porpoises": None})
                   for i in range(700)]
        viz.set_training_progress(records, dict(ticks=1000, lower=0.3, upper=3))
        viz.begin_rollout_recording()
        viz.end_rollout_recording()
        viz._active_tab = viz._tabs.index("progress")
        viz._render_full()
        assert len(viz._series["progress"]["gadoids"]) == 700
        assert viz._series["progress"]["gadoids"][0] == (0, 0)
        assert viz._series["progress"]["porpoises"] == []
        assert "No starting biomass" in viz._progress_message
        assert viz._plot_scroll_thumb_rect is None
        assert viz._plot_scroll_span_cache == (0, 13980)
        assert "phytoplankton" not in viz._active_plot_ids()

    assert main(arguments(tmp_path) + ["--visual"], on_step=populate) == 0


def test_progress_interval_and_resume_evaluation(tmp_path, viewers):
    flags = arguments(tmp_path) + ["--visual", "--eval-every", "2", "--iter-per-gen", "3"]
    history = tmp_path / "progress" / "survival.jsonl"
    assert main(flags) == 0
    assert [r["step"] for r in map(json.loads, history.read_text().splitlines())] == [0, 2]
    assert main(flags + ["--resume"]) == 0
    assert [r["step"] for r in map(json.loads, history.read_text().splitlines())] == [0, 2, 3, 4, 6]
    assert [step for step, _ in viewers[-1]._series["progress"]["gadoids"]] == [0, 2, 3, 4, 6]


def test_cpu_visual_evaluates_progress(tmp_path, viewers, monkeypatch):
    from train import main as cpu_main
    from lib.viz.pygame_viz import LiveVisualizer
    monkeypatch.setattr(LiveVisualizer, "wait_for_close", lambda *a, **kw: None)
    cpu_main(["--project", "mareld2.yaml", "--grid", "5x6", "--workers", "1",
              "--n_deltas", "2", "--n_eval_ticks", "2", "--generations", "1",
              "--iter-per-gen", "2", "--run-name", str(tmp_path), "--visual",
              "--eval-every", "1", "--eval-ticks", "5"], confirm=False)
    records = list(map(json.loads, (tmp_path / "progress" / "survival.jsonl").read_text().splitlines()))
    assert [r["step"] for r in records] == [0, 1, 2]
    assert len(viewers[0]._series["progress"]["gadoids"]) == 3


def test_events_pumped_while_training_and_close_keeps_training(tmp_path, viewers, monkeypatch):
    import pygame
    from lib.gpu.trainer import TensorARSTrainer
    from lib.viz.pygame_viz import LiveVisualizer
    started, pumped = threading.Event(), threading.Event()
    original_step = TensorARSTrainer.train_step
    original_pump = LiveVisualizer.pump_events

    def blocked_step(trainer, *args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        started.set()
        assert pumped.wait(5), "Main thread did not pump events during the update"
        return original_step(trainer, *args, **kwargs)

    def pump(viz):
        assert threading.current_thread() is threading.main_thread()
        if started.is_set() and not pumped.is_set():
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            pumped.set()
        return original_pump(viz)

    monkeypatch.setattr(TensorARSTrainer, "train_step", blocked_step)
    monkeypatch.setattr(LiveVisualizer, "pump_events", pump)
    assert main(arguments(tmp_path) + ["--visual"]) == 0
    assert pumped.is_set()
    assert not viewers[0].enabled
    state = torch.load(tmp_path / "trainer.pth", weights_only=False)
    assert state["trainer"]["iterations_completed"] == 2


def test_sliders_apply_between_round_robin_updates(tmp_path, viewers):
    def adjust(trainer, step, directory, args):
        if step == 0:
            viewers[0]._neval_override = 3
            viewers[0]._ticks_override = 4
            viewers[0]._b0_overrides = {"gadoids": 25.0}
        else:
            assert int(trainer.runner.horizon) == 3

    assert main(arguments(tmp_path) + ["--visual", "--no-coevolution",
                "--species", "gadoids", "pelagic_fish"], on_step=adjust) == 0
    records = [json.loads(line) for line in (tmp_path / "biomass.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(r["n_ticks"] == 4 and r["b0"]["gadoids"] == pytest.approx(25) for r in records)
    assert [list(r["reward"]) for r in records] == [["gadoids"], ["pelagic_fish"]] * 2


def test_unavailable_viewer_continues_without_probes(tmp_path, monkeypatch):
    import lib.viz
    from lib.viz.pygame_viz import _NullViz
    monkeypatch.setattr(lib.viz, "LiveVisualizer", lambda **kwargs: _NullViz())
    assert main(arguments(tmp_path) + ["--visual"]) == 0
    assert (tmp_path / "trainer.pth").is_file()
    assert not (tmp_path / "biomass.jsonl").exists()


def test_training_failure_propagates_and_closes_viewer(tmp_path, viewers, monkeypatch):
    from lib.gpu.trainer import TensorARSTrainer

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(TensorARSTrainer, "train_step", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        main(arguments(tmp_path) + ["--visual"])
    assert not viewers[0].enabled
    assert not (tmp_path / "trainer.pth").exists()


def test_snapshot_uses_latest_weights_and_owns_its_data():
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    from lib.gpu.visual import policy_snapshot
    trainer = TensorARSTrainer(ProjectSpec(EnvironmentBuilder("mareld2.yaml", grid=(5, 6))),
                               device="cpu", execution="eager", n_deltas=2)
    try:
        trainer.train_step(n_eval_ticks=2)
        before = trainer.state_dict()
        snapshot = policy_snapshot(trainer)
        for fid, theta in zip(trainer.model.dm_ids, trainer.theta):
            flat = torch.nn.utils.parameters_to_vector(snapshot.policies[fid].parameters())
            torch.testing.assert_close(flat, theta, rtol=0, atol=0)
            next(snapshot.policies[fid].parameters()).zero_()
            snapshot.obs_stats[fid]["mean"].fill(42)
        assert_same(trainer.state_dict(), before)
    finally:
        trainer.close()


@pytest.mark.parametrize("platform,wayland,expected", [
    ("darwin", False, None), ("win32", False, None),
    ("linux", False, "dummy"), ("linux", True, None),
])
def test_native_desktops_are_not_forced_to_dummy_display(monkeypatch, platform, wayland, expected):
    import os
    from lib.viz.pygame_viz import LiveVisualizer
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if wayland:
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(sys, "platform", platform)
    LiveVisualizer.__new__(LiveVisualizer)
    assert os.environ.get("SDL_VIDEODRIVER") == expected
