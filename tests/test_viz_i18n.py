"""Display language of the live viewer (section 212).

The viewer's strings are English templates translated at draw time
(lib/viz/i18n.py). A flag button in the status bar opens a menu of three
flags; switching language must not move any control.
"""

import re
import string
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.viz import i18n  # noqa: E402

GRID = (4, 4)
FGS = ["benthivorous_fish", "planktivorous_fish", "predatory_fish",
       "zooplankton"]


def fields(template):
    return {f for _lit, f, _spec, _conv in string.Formatter().parse(template)
            if f is not None}


@pytest.mark.parametrize("lang", [c for c in i18n.CODES if c != i18n.DEFAULT])
def test_translations_keep_the_fields_of_their_template(lang):
    for key, value in i18n.CATALOG[lang].items():
        assert fields(value) == fields(key), (lang, key)


def test_every_catalog_key_is_a_string_the_code_uses():
    """A key that no source string matches is a typo: never shown."""
    source = "".join((ROOT / p).read_text(encoding="utf-8") for p in (
        "lib/viz/pygame_viz.py", "lib/runners/training_progress.py"))
    for lang in i18n.CATALOG:
        for key in i18n.CATALOG[lang]:
            assert f'"{key}"' in source, (lang, key)


def test_every_language_has_the_same_keys():
    keys = [set(i18n.CATALOG[c]) for c in i18n.CATALOG]
    assert all(k == keys[0] for k in keys)


def test_names_cover_the_library():
    from lib.config.config_loader import load_config
    specs = load_config(str(ROOT / "fgconfig" / "fg_library.yaml"))[
        "species_definitions"]
    for lang in i18n.FG_NAMES:
        assert set(specs) <= set(i18n.FG_NAMES[lang]), lang
        for fid, spec in specs.items():
            for m in spec.get("member_species") or ():
                if m["name"] == "others":
                    continue
                assert m["name"].lower() in i18n.SPECIES[lang], (lang, m)


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode="train",
                       plot_fg_ids=FGS)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v.set_b0_defaults({f: 16.0 for f in FGS})
    for step in range(10):
        for fid in FGS:
            v.update_reward(fid, float(step), step=step)
    v.update_status(gen=1, iter=2, T=1.0)
    v.update_biomass({f: np.ones(GRID, dtype=np.float32) for f in FGS}, tick=1)
    v._pg.mouse.set_pos((0, 0))
    v._render_full()
    yield v
    v.close()


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def controls(v):
    return {
        "norm": v._hm_norm_button_rect,
        "lang": v._lang_button_rect,
        "ticks": v._ticks_track_rect,
        "neval": v._neval_track_rect,
        "evalint": v._evalint_track_rect,
        "playback": [r for r, _a in v._playback_rects],
        "playback_track": v._playback_track_rect,
        "sliders": [r for r, _f in v._slider_rects],
        "tabs_start": v._tab_rects[0][0][:2],
    }


def test_the_flag_opens_a_menu_and_a_row_picks_the_language(viz):
    viz._handle_click(centre(viz._lang_button_rect))
    assert viz._lang_menu_open
    viz._render_full()
    codes = [c for _r, c in viz._lang_item_rects]
    assert codes == ["en", "sv", "it"]
    sv_row = next(r for r, c in viz._lang_item_rects if c == "sv")
    viz._handle_click(centre(sv_row))
    assert viz._lang == "sv" and not viz._lang_menu_open


def test_a_click_outside_the_open_menu_only_closes_it(viz):
    viz._handle_click(centre(viz._lang_button_rect))
    viz._render_full()
    rolling = viz._rolling_heatmap
    # The norm button lies outside the menu: the click must not reach it.
    viz._handle_click(centre(viz._hm_norm_button_rect))
    assert not viz._lang_menu_open
    assert viz._rolling_heatmap == rolling
    assert viz._lang == "en"


@pytest.mark.parametrize("lang", ["sv", "it"])
def test_switching_language_moves_no_control(viz, lang):
    before = controls(viz)
    viz._lang = lang
    viz._render_full()
    after = controls(viz)
    # The tab strip follows the tab names; only its start is fixed.
    assert after == before


@pytest.mark.parametrize("lang", i18n.CODES)
def test_playback_labels_fit_their_buttons(viz, lang):
    viz._lang = lang
    for label, width in (("live", 56), ("replay", 56),
                         ("resume train", 100), ("pause train", 100)):
        assert viz._font.size(viz._t(label))[0] <= width - 4, (lang, label)


def test_the_status_bar_is_redrawn_in_the_language(viz):
    pg = viz._pg
    x, y, w, h = viz._status_rect
    en = pg.surfarray.array3d(viz._screen)[x:x + w, y:y + 20].copy()
    viz._lang = "sv"
    viz._render_full()
    sv = pg.surfarray.array3d(viz._screen)[x:x + w, y:y + 20]
    assert not np.array_equal(en, sv)


def test_names_and_codes_follow_the_language(viz):
    assert viz._display_name("planktivorous_fish") == "Planktivorous fish"
    assert viz._fg_abbrev("planktivorous_fish") == "plf"
    viz._lang = "sv"
    assert viz._display_name("planktivorous_fish") == "Planktonätande fisk"
    assert viz._display_name("seals_rnd") == "Sälar rnd"
    assert viz._fg_abbrev("planktivorous_fish") == "pf"
    viz._lang = "en"
    assert viz._fg_abbrev("planktivorous_fish") == "plf"


def test_member_lines_in_each_language():
    from lib.viz.pygame_viz import LiveVisualizer
    entries = [{"name": "herring", "share": 0.79},
               {"name": "others", "count": 3, "share": 0.21}]
    assert LiveVisualizer._member_lines(entries) == [
        "Herring  79 %", "Others (3 species)  21 %"]
    assert LiveVisualizer._member_lines(entries, "sv") == [
        "Sill  79 %", "Övriga (3 arter)  21 %"]
    assert LiveVisualizer._member_lines(entries, "it") == [
        "Aringa  79 %", "Altre (3 specie)  21 %"]


def test_calendar_label_in_each_language():
    from lib.viz.pygame_viz import _calendar_label
    h = (31 + 13) * 24 + 6
    assert _calendar_label(h) == "Feb 14 06:00"
    assert _calendar_label(4 * 31 * 24, "sv") == "Maj 5 00:00"
    assert _calendar_label(4 * 31 * 24, "it") == "Mag 5 00:00"


def test_progress_texts_are_translated(viz):
    viz.set_progress_message("Progress unavailable; see terminal for details")
    viz._lang = "sv"
    assert viz._t(viz._progress_fmt[0], **viz._progress_fmt[1]).startswith(
        "Framsteg saknas")
    # The English attribute other code reads is unchanged.
    assert viz._progress_message.startswith("Progress unavailable")


def test_reward_title_keeps_its_formula(viz):
    viz.set_reward_label("ARS reward per FG — local per-cell log(B/A)")
    viz._lang = "it"
    assert viz._tab_title("reward") == "Premio ARS per FG — local per-cell log(B/A)"


def test_fg_editor_swedish_names_are_the_viewer_table():
    text = (ROOT / "fgconfig" / "fgconfig.py").read_text(encoding="utf-8")
    assert re.search(r"self\.sv_mapping = dict\(FG_NAMES\[\"sv\"\]\)", text)


def new_viewer():
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=FGS, grid_shape=GRID, mode="train")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    return v


def pick(v, code):
    v._render_full()
    v._handle_click(centre(v._lang_button_rect))
    v._render_full()
    row = next(r for r, c in v._lang_item_rects if c == code)
    v._handle_click(centre(row))


def test_the_language_is_remembered_by_the_next_viewer(viz):
    pick(viz, "it")
    viz.close()
    v = new_viewer()
    try:
        assert v._lang == "it"
    finally:
        v.close()


def test_without_saved_settings_the_viewer_starts_in_english(viz):
    assert viz._lang == i18n.DEFAULT


def test_settings_file_keeps_other_keys_and_survives_garbage(viz, tmp_path):
    import json
    import os
    path = os.environ["MARELD_VIEWER_SETTINGS"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"something_else": 1}, f)
    pick(viz, "sv")
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"something_else": 1, "language": "sv"}
    # A broken or unknown value falls back to English, never fails.
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    v = new_viewer()
    v.close()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"language": "klingon"}, f)
    v = new_viewer()
    try:
        assert v._lang == i18n.DEFAULT
    finally:
        v.close()


def test_the_test_suite_never_touches_the_real_settings_file():
    import os
    from lib.viz import pygame_viz
    assert pygame_viz._settings_path() != pygame_viz._SETTINGS_FILE
    assert os.environ["MARELD_VIEWER_SETTINGS"].endswith("viewer_settings.json")
