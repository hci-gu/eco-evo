"""Impact maps on their own heatmap tab, with an intensity slider (section 224).

The inference viewer shows the functional groups on one heatmap tab and
the impact maps (the world's trawling fields) on another. Each map has a
slider scaling its intensity in [0, 10]x; releasing it flags a new rollout,
and inference.apply_impact_scales sets the world's map to scale x the
loaded field.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (60, 60)
DMS = ["zooplankton", "planktivorous_fish"]
NDMS = ["phytoplankton"]


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=DMS + NDMS, grid_shape=GRID, mode="inference",
                       ndm_ids=NDMS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v.set_b0_defaults({f: 10.0 for f in DMS + NDMS})
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in DMS + NDMS},
                     tick=1)
    yield v
    v.close()


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def trawl_maps():
    field = np.zeros(GRID, dtype=np.float32)
    field[10:20, 10:30] = 5.0
    return {"bottom_trawling": field,
            "pelagic_trawling": np.zeros(GRID, dtype=np.float32)}


def test_no_tab_strip_without_impact_maps(viz):
    viz._render_full()
    assert viz._hm_tab_rects == []
    assert [fid for fid, _x, _y in viz._hm_slots] == DMS + NDMS


def test_tabs_switch_between_groups_and_impacts(viz):
    viz.set_impact_maps(trawl_maps(), {"bottom_trawling": "MW-h/km2/year"})
    viz._render_full()
    tabs = {tab: rect for rect, tab in viz._hm_tab_rects}
    assert set(tabs) == {"groups", "impacts"}
    # The strip sits above the panels' viewport.
    assert tabs["groups"][1] + tabs["groups"][3] <= viz._hm_view_rect[1]
    viz._handle_click(centre(tabs["impacts"]))
    viz._render_full()
    # An all-zero map is left out.
    assert [fid for fid, _x, _y in viz._hm_slots] == ["bottom_trawling"]
    assert viz._hm_headers == []
    viz._handle_click(centre(tabs["groups"]))
    viz._render_full()
    assert [fid for fid, _x, _y in viz._hm_slots] == DMS + NDMS


def test_intensity_slider_sets_scale_and_flags_rerun(viz):
    from lib.viz.pygame_viz import _IMPACT_KEY
    viz.set_impact_maps(trawl_maps())
    viz._set_hm_tab("impacts")
    viz._render_full()
    assert viz.get_impact_scales() == {"bottom_trawling": 1.0}
    rect = next(r for r, k in viz._slider_rects
                if k == _IMPACT_KEY + "bottom_trawling")
    track_x, track_w = rect[0] + 2, rect[2] - 4
    # Half the track is 5x.
    viz._handle_click((track_x + track_w // 2, rect[1] + rect[3] // 2))
    assert viz._dragging_slider == _IMPACT_KEY + "bottom_trawling"
    viz._slider_release()
    assert viz.get_impact_scales()["bottom_trawling"] == pytest.approx(5.0, abs=0.1)
    # The impact slider never leaks into the b0 overrides.
    assert viz.get_b0_overrides() == {}
    assert viz.consume_b0_change() is False
    assert viz.consume_impact_change() is True
    assert viz.consume_impact_change() is False


def test_apply_impact_scales_scales_the_loaded_field():
    from inference import apply_impact_scales
    from lib.world.grid import Grid

    class Env:
        grid = Grid(width=GRID[1], height=GRID[0])

    base = trawl_maps()
    env = Env()
    for iid, m in base.items():
        env.grid.add_map(iid, m)
    apply_impact_scales(env, base, {"bottom_trawling": 0.5})
    apply_impact_scales(env, base, {"bottom_trawling": 0.5})
    # Scaled from the loaded field, never compounded.
    np.testing.assert_allclose(env.grid.get_map("bottom_trawling"),
                               base["bottom_trawling"] * 0.5)
    apply_impact_scales(env, base, {"bottom_trawling": 0.0})
    assert float(env.grid.get_map("bottom_trawling").max()) == 0.0


def test_each_impact_has_a_restart_box(viz):
    """Section 225: a "restart" box per impact panel, ticked by default;
    unticked, its slider flags a live change instead of a restart."""
    from lib.viz import pygame_viz
    from lib.viz.pygame_viz import _IMPACT_KEY
    maps = trawl_maps()
    maps["pelagic_trawling"][0, 0] = 1.0
    viz.set_impact_maps(maps)
    viz._set_hm_tab("impacts")
    viz._render_full()
    boxes = {iid: rect for rect, iid in viz._impact_restart_rects}
    assert set(boxes) == {"bottom_trawling", "pelagic_trawling"}
    assert viz.impact_restarts("bottom_trawling") is True
    viz._handle_click(centre(boxes["bottom_trawling"]))
    assert viz.impact_restarts("bottom_trawling") is False
    assert viz.impact_restarts("pelagic_trawling") is True
    assert pygame_viz._load_settings()["impact_restart"] == {
        "bottom_trawling": False}
    viz._render_full()
    for iid in ("bottom_trawling", "pelagic_trawling"):
        rect = next(r for r, k in viz._slider_rects if k == _IMPACT_KEY + iid)
        viz._handle_click(centre(rect))
        viz._slider_release()
    assert viz.consume_impact_live_change() is True     # bottom: unticked
    assert viz.consume_impact_change() is True          # pelagic: ticked
    # No box in the status bar any more.
    assert not hasattr(viz, "_restart_box_rect")


def test_resumed_recording_keeps_the_film_and_series(viz):
    viz.begin_rollout_recording()
    for t in range(3):
        viz.update_biomass({f: np.ones(GRID, dtype=np.float32)
                            for f in DMS + NDMS}, tick=t)
        viz.update_series("biomass", "zooplankton", 100.0, step=t)
    viz.end_rollout_recording()
    n = len(viz._current_rollout)
    viz.resume_rollout_recording()
    viz.update_biomass({f: np.ones(GRID, dtype=np.float32)
                        for f in DMS + NDMS}, tick=3)
    viz.update_series("biomass", "zooplankton", 100.0, step=3)
    viz.end_rollout_recording()
    assert len(viz._current_rollout) == n + 1
    assert len(viz._series["biomass"]["zooplankton"]) == 4


def test_sliders_snap_to_their_tick_mark(viz):
    """Section 225.4: near the tick mark (1x on an intensity slider, the
    default on a b0 slider) the slider takes exactly that value."""
    from lib.viz.pygame_viz import _IMPACT_KEY, _SLIDER_SNAP_PX
    viz.set_impact_maps(trawl_maps())
    viz._render_full()
    fid = DMS[0]
    rect = next(r for r, k in viz._slider_rects if k == fid)
    track_x, track_w = rect[0] + 2, rect[2] - 4
    tick = track_x + track_w * 0.25
    for dx in (1 - _SLIDER_SNAP_PX, 0, _SLIDER_SNAP_PX - 1):
        assert viz._slider_value_from_x(fid, int(round(tick + dx))) == 10.0
    assert viz._slider_value_from_x(fid, int(tick) + _SLIDER_SNAP_PX + 3) > 10.0
    viz._set_hm_tab("impacts")
    viz._render_full()
    key = _IMPACT_KEY + "bottom_trawling"
    rect = next(r for r, k in viz._slider_rects if k == key)
    track_x, track_w = rect[0] + 2, rect[2] - 4
    tick = track_x + track_w / 10.0
    for dx in (1 - _SLIDER_SNAP_PX, 0, _SLIDER_SNAP_PX - 1):
        assert viz._slider_value_from_x(key, int(round(tick + dx))) == 1.0
    assert viz._slider_value_from_x(key, int(tick) - _SLIDER_SNAP_PX - 3) < 1.0
    # Far from the mark the track stays linear.
    assert viz._slider_value_from_x(key, track_x) == 0.0
    assert viz._slider_value_from_x(key, track_x + track_w) == 10.0
