"""The species tooltip over a heatmap panel's info lines.

A multi-species FG carries ``member_species`` in the library; the viewer
reads it off the FG objects in ``update_biomass`` and lists the species
when the pointer is over the panel's info lines. Below a rule the same
tooltip names the prey behind the codes on the group's diet line. A
single-species FG with no diet line has no tooltip.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)
MULTI, SINGLE = "benthivorous_fish", "porpoises"


def group(members):
    params = {} if members is None else {"member_species": members}
    return SimpleNamespace(biomass=np.ones(GRID, dtype=np.float32),
                           params=params)


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=[MULTI, SINGLE], grid_shape=GRID, mode="train")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    # Live b0 sliders, so the panel registers a slider rect to compare with.
    v.set_b0_defaults({MULTI: 100.0, SINGLE: 10.0})
    v.update_biomass({
        MULTI: group([{"name": "haddock", "share": 0.386},
                      {"name": "cod", "share": 0.314},
                      {"name": "others", "count": 43, "share": 0.3}]),
        SINGLE: group([{"name": "harbour porpoise"}]),
    }, tick=1)
    v._render_full()
    yield v
    v.close()


def rect_of(v, fid):
    return next((r for r, f in v._member_rects if f == fid), None)


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def motion(v, pos):
    pg = v._pg
    pg.mouse.set_pos(pos)
    pg.event.post(pg.event.Event(pg.MOUSEMOTION, pos=pos, rel=(1, 1),
                                 buttons=(0, 0, 0)))


def test_member_lines_format():
    from lib.viz.pygame_viz import LiveVisualizer
    lines = LiveVisualizer._member_lines(
        [{"name": "herring", "share": 0.79}, {"name": "sprat"}, "krill"])
    assert lines == ["Herring  79 %", "Sprat", "Krill"]
    # One species is no group to explain.
    assert LiveVisualizer._member_lines([{"name": "harbour seal"}]) == []
    assert LiveVisualizer._member_lines(None) == []


def test_without_a_diet_only_the_multi_species_panel_has_a_hover_rect(viz):
    assert rect_of(viz, MULTI) is not None
    assert rect_of(viz, SINGLE) is None


def test_hover_rect_covers_the_info_lines_not_the_slider(viz):
    x, y, w, h = rect_of(viz, MULTI)
    sliders = [r for r, fid in viz._slider_rects if fid == MULTI]
    assert sliders, "no b0 slider laid out"
    assert y + h <= sliders[0][1], "the hover rect reaches into the b0 slider"


def test_hovering_lists_the_species(viz, monkeypatch):
    drawn = []
    monkeypatch.setattr(viz, "_draw_tooltip_box",
                        lambda lines, *a, **kw: drawn.append(lines))
    motion(viz, centre(rect_of(viz, MULTI)))
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert viz._member_hover is True
    assert drawn, "no tooltip was drawn"
    texts = [e and e[0] for e in drawn[-1]]
    assert texts == ["Benthivorous fish", "Haddock  39 %", "Cod  31 %",
                     "Others (43 species)  30 %"]


def feed_coded_lines(v):
    """Actions, losses and a diet for both panels, then a repaint."""
    for fid in (MULTI, SINGLE):
        v.update_action_fracs(fid, 10.0, 30.0, 60.0)
    v.update_loss_breakdown({fid: {"predation": 0.5, "starvation": 0.1,
                                   "impact": 0.3, "natural": 0.1}
                             for fid in (MULTI, SINGLE)})
    v.update_diet_breakdown({MULTI: {SINGLE: 1.0}, SINGLE: {MULTI: 1.0}})
    v._render_full()


def hover_lines(v, fid, monkeypatch):
    drawn = []
    monkeypatch.setattr(v, "_draw_tooltip_box",
                        lambda lines, *a, **kw: drawn.append(lines))
    motion(v, centre(rect_of(v, fid)))
    v._last_frame_ts = 0.0
    v.pump_events()
    assert drawn, "no tooltip was drawn"
    return [e and e[0] for e in drawn[-1]]


def test_species_then_a_rule_then_the_diet_codes(viz, monkeypatch):
    feed_coded_lines(viz)
    assert hover_lines(viz, MULTI, monkeypatch) == [
        "Benthivorous fish", "Haddock  39 %", "Cod  31 %",
        "Others (43 species)  30 %", None,
        f"{viz._fg_abbrev(SINGLE)} = Porpoises"]


def test_a_single_species_panel_names_its_prey(viz, monkeypatch):
    feed_coded_lines(viz)
    assert rect_of(viz, SINGLE) is not None
    assert hover_lines(viz, SINGLE, monkeypatch) == [
        "Porpoises", f"{viz._fg_abbrev(MULTI)} = Benthivorous fish"]


def test_diet_codes_follow_the_language(viz, monkeypatch):
    viz._lang = "sv"
    feed_coded_lines(viz)
    assert hover_lines(viz, MULTI, monkeypatch)[-1] == \
        f"{viz._fg_abbrev(SINGLE)} = Tumlare"


def test_the_rule_is_drawn_inside_the_box(viz):
    pg = viz._pg
    before = pg.surfarray.array3d(viz._screen).copy()
    viz._draw_tooltip_box([("a", (255, 255, 255)), None,
                           ("b", (255, 255, 255))], 40, 40,
                          flip_rect=(0, 0, 400, 400),
                          clamp_rect=(0, 0, 400, 400))
    after = pg.surfarray.array3d(viz._screen)
    assert not np.array_equal(before, after)


def test_tooltip_is_erased_on_the_way_out(viz):
    pg = viz._pg
    bare_pos = (2, viz._status_h - 2)    # status bar: no member rect
    pg.mouse.set_pos(bare_pos)
    viz._render_full()
    bare = pg.surfarray.array3d(viz._screen).copy()

    motion(viz, centre(rect_of(viz, MULTI)))
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert not np.array_equal(bare, pg.surfarray.array3d(viz._screen))

    motion(viz, bare_pos)
    viz._last_frame_ts = 0.0
    viz.pump_events()
    assert viz._member_hover is False
    assert np.array_equal(bare, pg.surfarray.array3d(viz._screen)), \
        "the tooltip was left behind"


def test_library_members_reach_the_fg_params():
    from lib.config.config_loader import load_project_config
    root = Path(__file__).resolve().parents[1]
    fgs, *_ = load_project_config(str(root / "mareld2.yaml"),
                                  library_path=str(root / "fgconfig" /
                                                   "fg_library.yaml"),
                                  grid_size=(8, 8), seed=1)
    names = [m["name"] for m in fgs["planktivorous_fish"].params["member_species"]]
    assert names[0] == "herring"
    shares = [m["share"] for m in fgs["planktivorous_fish"].params["member_species"]]
    assert abs(sum(shares) - 1.0) < 1e-6
    for fid in ("benthivorous_fish", "predatory_fish"):
        shares = [m["share"] for m in fgs[fid].params["member_species"]]
        assert abs(sum(shares) - 1.0) < 0.01, fid
    # Single-species groups carry no list.
    assert "member_species" not in fgs["porpoises"].params
    assert "member_species" not in fgs["seals"].params


def test_the_tooltip_lists_who_eats_the_group(viz, monkeypatch):
    """Below the diet codes: the share of the group's predation loss by
    predator, largest first, in the predators' own colours."""
    feed_coded_lines(viz)
    viz.update_predation_breakdown({MULTI: {SINGLE: 0.25, MULTI: 0.75}})
    viz._render_full()
    assert hover_lines(viz, MULTI, monkeypatch) == [
        "Benthivorous fish", "Haddock  39 %", "Cod  31 %",
        "Others (43 species)  30 %", None,
        f"{viz._fg_abbrev(SINGLE)} = Porpoises", None,
        "Eaten by", "  Benthivorous fish 75%", "  Porpoises 25%"]


def test_an_uneaten_single_species_group_gets_a_tooltip_once_eaten(viz, monkeypatch):
    assert rect_of(viz, SINGLE) is None
    viz.update_predation_breakdown({SINGLE: {MULTI: 1.0}})
    viz._render_full()
    assert hover_lines(viz, SINGLE, monkeypatch) == [
        "Porpoises", "Eaten by", "  Benthivorous fish 100%"]
    viz._lang = "sv"
    viz._render_full()
    assert "Äts av" in hover_lines(viz, SINGLE, monkeypatch)
