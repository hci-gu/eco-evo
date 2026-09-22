"""Fixed/dynamic colour limits in the shared training and inference viewer."""

import numpy as np
import pytest

from lib.gpu.config import EnvironmentBuilder, ProjectSpec


@pytest.fixture(params=["train", "inference"])
def viewer(request, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    viz = LiveVisualizer(["zooplankton", "phytoplankton"], (4, 4), mode=request.param)
    assert viz.enabled
    yield viz
    viz.close()


def test_colour_toggle_while_recording_and_replaying(viewer, monkeypatch):
    viz = viewer
    field = np.arange(16, dtype=np.float32).reshape(4, 4)
    viz.begin_rollout_recording()
    viz.update_biomass({"zooplankton": field, "phytoplankton": field * 10}, 0)
    viz.update_biomass({"zooplankton": field / 100, "phytoplankton": field}, 1)
    assert not viz._dynamic_heatmap
    assert viz._heatmap_max("zooplankton", field / 100) == 15
    viz._render_full()
    rect = next(rect for rect, action in viz._playback_rects
                if action == "toggle_heatmap_scale")
    # The control must work even while playback controls are disabled.
    viz._handle_click((rect[0] + rect[2] // 2, rect[1] + rect[3] // 2))
    assert viz._dynamic_heatmap
    assert viz._heatmap_max("zooplankton", field / 100) == pytest.approx(.15)
    assert viz._heatmap_max("phytoplankton", field) == 15
    viz.end_rollout_recording()
    viz._playback_mode = "paused"
    viz._playback_idx = 1
    seen = {}
    original = viz._heatmap_max

    def capture(fid, arr):
        seen[fid] = original(fid, arr)
        return seen[fid]

    monkeypatch.setattr(viz, "_heatmap_max", capture)
    for log_mode in (False, True):
        viz._log_heatmap = log_mode
        viz._render_full()
        assert seen["zooplankton"] == pytest.approx(.15)
        viz._handle_key(viz._pg.K_c)
        viz._render_full()
        assert seen["zooplankton"] == 15
        assert seen["phytoplankton"] == 150
        viz._handle_key(viz._pg.K_c)
    assert not viz._biomass_scale_manual
    viz._set_biomass_display_scale(.5)
    assert original("zooplankton", field / 100) == pytest.approx(.075)
    viz._handle_key(viz._pg.K_c)
    assert original("zooplankton", field / 100) == 7.5
    viz._reset_biomass_display_scale()
    assert original("zooplankton", field / 100) == 15


def test_empty_population_and_zero_start(viewer):
    viz = viewer
    empty = np.zeros((4, 4), dtype=np.float32)
    viz.update_biomass({fid: empty for fid in viz.fg_ids}, 0)
    for dynamic in (False, True):
        viz._dynamic_heatmap = dynamic
        assert viz._heatmap_max("zooplankton", empty) == 1e-12
        viz._render_full()
    viz.update_biomass({"zooplankton": np.ones((4, 4), dtype=np.float32)}, 1)
    assert viz._heatmap_max("zooplankton", viz._biomass["zooplankton"]) == 1
    viz._handle_key(viz._pg.K_c)
    assert viz._heatmap_max("zooplankton", viz._biomass["zooplankton"]) == 1e-12


def test_shared_zooplankton_speed_default():
    builder = EnvironmentBuilder(project_path="mareld2.yaml", grid=(4, 4))
    assert builder(seed=0).fgs["zooplankton"].speed == 1.0
    assert ProjectSpec(builder).env.fgs["zooplankton"].speed == 1.0
