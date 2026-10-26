"""Folding the heatmap groups (section 214).

Each group header ("Decision makers", "Non-decision makers") has a box
with a minus (open: a click folds the group in under the header) or a
plus (folded: a click folds it out). The panels glide; the groups below
follow; the clicked header stays under the pointer.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (60, 60)
DMS = ["zooplankton", "planktivorous_fish", "benthivorous_fish",
       "predatory_fish", "porpoises", "seals"]
NDMS = ["phytoplankton", "subsurface_phytoplankton", "benthic_infauna",
        "microzooplankton"]
DM, NDM = "Decision makers", "Non-decision makers"


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=DMS + NDMS, grid_shape=GRID, mode="training",
                       ndm_ids=NDMS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v._fold_anim_s = 0.0
    v.set_b0_defaults({f: 10.0 for f in DMS + NDMS})
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in DMS + NDMS},
                     tick=1)
    v._pg.mouse.set_pos((0, 0))
    v._render_full()
    yield v
    v.close()


def header_rect(v, title):
    return next(r for r, t in v._hm_header_rects if t == title)


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def click_header(v, title):
    if all(t != title for _r, t in v._hm_header_rects):
        # Scroll the header into view first, as the user would.
        v._set_hm_scroll(dict(v._hm_headers)[title])
        v._render_full()
    v._handle_click(centre(header_rect(v, title)))
    v._render_full()


def fids_with(rects):
    return {entry[1] for entry in rects}


def test_open_layout_is_unchanged(viz):
    assert viz._group_open == {DM: 1.0, NDM: 1.0}
    cols, ph, hh = viz._cols, viz._panel_h, viz._hm_header_h
    rows = lambda n: (n + cols - 1) // cols   # noqa: E731
    assert viz._hm_block_h == 2 * hh + (rows(len(DMS)) + rows(len(NDMS))) * ph


def test_a_header_click_folds_the_group_and_another_unfolds_it(viz):
    before_block = viz._hm_block_h
    ndm_y = viz._hm_headers[1][1]
    click_header(viz, DM)
    assert viz._group_open[DM] == 0.0
    dm_rows = (len(DMS) + viz._cols - 1) // viz._cols
    assert viz._hm_block_h == before_block - dm_rows * viz._panel_h
    # The group below follows up under the folded header.
    assert viz._hm_headers[1][1] == ndm_y - dm_rows * viz._panel_h
    # Folded panels take no clicks.
    assert not fids_with(viz._slider_rects) & set(DMS)
    assert not fids_with(viz._spawn_dropdown_rects) & set(DMS)
    click_header(viz, DM)
    assert viz._group_open[DM] == 1.0
    assert viz._hm_block_h == before_block
    assert fids_with(viz._slider_rects) & set(DMS)


def test_the_box_shows_minus_open_and_plus_folded(viz):
    pg = viz._pg

    def box_pixels():
        x, y, w, h = header_rect(viz, DM)
        b = viz._FOLD_BOX
        bx, by = x + 6, y + (h - b) // 2
        return pg.surfarray.array3d(viz._screen)[bx:bx + b, by:by + b].copy()

    open_box = box_pixels()
    mid = viz._FOLD_BOX // 2
    # Minus: the vertical stroke through the middle is absent.
    assert open_box[mid, 3].sum() < open_box[3, mid].sum()
    click_header(viz, DM)
    folded_box = box_pixels()
    assert folded_box[mid, 3].sum() == folded_box[3, mid].sum()
    assert not np.array_equal(open_box, folded_box)


def test_the_clicked_header_stays_under_the_pointer(viz):
    """Scrolled down to the last header, folding its group must not slide it.

    The fold shrinks the block below the view, so without the held
    position the scroll would be clamped and the header pulled down.
    """
    viz._set_hm_scroll(viz._hm_headers[1][1])
    viz._render_full()
    assert viz._hm_scroll_y > 0, "the block does not scroll in this window"
    before = header_rect(viz, NDM)
    click_header(viz, NDM)
    assert header_rect(viz, NDM)[:2] == before[:2]
    assert viz._hm_scroll_slack > 0, "the fold did not need the held position"
    # Scrolling up gives the blank room back; nothing jumps on the way.
    viz._set_hm_scroll(0)
    assert viz._hm_scroll_slack == 0
    assert viz._hm_max_scroll() == max(
        0, viz._hm_block_h - viz._hm_view_rect[3])


def test_the_glide_paints_intermediate_frames(viz, monkeypatch):
    viz._fold_anim_s = 0.12
    seen = []
    original = viz._render_full

    def spy():
        seen.append(viz._group_open[DM])
        original()

    monkeypatch.setattr(viz, "_render_full", spy)
    viz._handle_click(centre(header_rect(viz, DM)))
    assert viz._group_open[DM] == 0.0
    inner = [f for f in seen if 0.0 < f < 1.0]
    assert len(inner) >= 2, f"no glide: {seen}"
    assert inner == sorted(inner, reverse=True), "the glide went back and forth"


def test_a_folding_group_paints_only_inside_its_span(viz):
    """Mid-glide, the sliding panels stay under their header."""
    viz._group_open[DM] = 0.5
    viz._layout_hm_block()
    viz._render_full()
    g0, g1 = viz._hm_group_spans[0]
    oy = viz._heatmap_origin[1] - viz._hm_scroll_y
    for rect, fid in viz._slider_rects:
        if fid in DMS:
            assert oy + g0 <= rect[1] and rect[1] + rect[3] <= oy + g1


def test_a_single_group_has_no_header_to_fold(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=DMS, grid_shape=GRID, mode="training")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    try:
        v._render_full()
        assert v._hm_header_rects == [] and v._group_open == {}
    finally:
        v.close()


def new_viewer(monkeypatch, fg_ids=DMS + NDMS, ndm_ids=NDMS):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=fg_ids, grid_shape=GRID, mode="training",
                       ndm_ids=ndm_ids)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v._fold_anim_s = 0.0
    v._render_full()
    return v


def test_folded_groups_are_remembered_by_the_next_viewer(viz, monkeypatch):
    open_block = viz._hm_block_h
    click_header(viz, NDM)
    folded_block = viz._hm_block_h
    viz.close()
    v = new_viewer(monkeypatch)
    try:
        assert v._group_open == {DM: 1.0, NDM: 0.0}
        assert v._group_target[NDM] == 0.0
        # It starts laid out folded, with a plus, no glide needed.
        assert v._hm_block_h == folded_block
        click_header(v, NDM)
        assert v._hm_block_h == open_block
    finally:
        v.close()
    v = new_viewer(monkeypatch)
    try:
        assert v._group_open == {DM: 1.0, NDM: 1.0}
    finally:
        v.close()


def test_a_viewer_without_the_group_keeps_its_saved_fold(viz, monkeypatch):
    import json
    import os
    click_header(viz, DM)
    viz.close()
    # One kind only: no headers, nothing folded, nothing forgotten.
    v = new_viewer(monkeypatch, fg_ids=DMS, ndm_ids=[])
    try:
        assert v._group_open == {}
        assert all(f in [s[0] for s in v._hm_slots] for f in DMS)
    finally:
        v.close()
    with open(os.environ["MARELD_VIEWER_SETTINGS"], encoding="utf-8") as f:
        assert json.load(f)["folded_groups"] == [DM]


def test_a_broken_fold_setting_is_ignored(monkeypatch):
    import json
    import os
    with open(os.environ["MARELD_VIEWER_SETTINGS"], "w", encoding="utf-8") as f:
        json.dump({"folded_groups": "Decision makers", "language": "sv"}, f)
    v = new_viewer(monkeypatch)
    try:
        assert v._group_open == {DM: 1.0, NDM: 1.0}
        assert v._lang == "sv"
    finally:
        v.close()
