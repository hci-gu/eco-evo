"""The FG editor's Re-anchor checkbox (section 274): placed under the Initial
Total Biomass Range in both editors, shows and edits the project's
``simulation_settings.warmup.reanchor`` list, refuses a group whose level an
outside exchange sets, and a removed group leaves the list."""
import copy
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")

DMS = ["zooplankton", "planktivorous_fish", "benthivorous_fish", "whiting", "porpoises", "seals"]
NDMS = ["phytoplankton", "krill", "mobile_epifauna"]


@pytest.fixture
def app(monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    a = fgconfig.FGConfigApp(root)
    a.global_library = copy.deepcopy(a.global_library)
    shown = []
    monkeypatch.setattr(type(a), "save_yaml", lambda self, data, path: None)
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(fgconfig.messagebox, name,
                            lambda title, msg, *x, **k: shown.append((title, msg)))
    monkeypatch.setattr(fgconfig.messagebox, "askokcancel", lambda *x, **k: True)
    monkeypatch.setattr(fgconfig.messagebox, "askyesno", lambda *x, **k: True)
    a.project_data["decision_makers"] = [{"group_id": g} for g in DMS]
    a.project_data["non_decision_makers"] = [{"group_id": g} for g in NDMS]
    a.project_data["simulation_settings"] = {"warmup": {"reanchor": ["porpoises", "seals"]}}
    species = a.global_library["species_definitions"]
    for gid in DMS + NDMS:
        a.current_fg_configs[gid] = species[gid]
    a.update_fg_list()
    a.refresh_matrix()
    a.shown = shown
    yield a
    root.destroy()


def _select(app, category, gid):
    ids = [fg["group_id"] for fg in app.project_data[category]]
    listbox = app._listbox_for(category)
    listbox.selection_clear(0, "end")
    listbox.selection_set(ids.index(gid))
    app.on_fg_select(category)


def _apply(app, category, gid, checked):
    _select(app, category, gid)
    app.reanchor_vars["dm" if category == "decision_makers" else "ndm"].set(checked)
    app.apply_fg_changes()


def _row_of(frame, text):
    rows = [int(w.grid_info()["row"]) for w in frame.winfo_children()
            if w.winfo_class() == "TLabel" and str(w.cget("text")).startswith(text)]
    assert len(rows) == 1, (text, rows)
    return rows[0]


def test_the_checkbox_sits_under_the_initial_biomass_range(app):
    for frame in (app.editor_frame, app.ndm_editor_frame):
        max_row = _row_of(frame, "  Max:")
        assert _row_of(frame, "Re-anchor after warm-up") == max_row + 2   # a separator between


def test_the_checkbox_shows_and_edits_the_project_list(app):
    _select(app, "decision_makers", "porpoises")
    assert app.reanchor_vars["dm"].get() is True
    _select(app, "decision_makers", "benthivorous_fish")
    assert app.reanchor_vars["dm"].get() is False
    _apply(app, "decision_makers", "benthivorous_fish", True)
    assert app._reanchor_ids() == ["porpoises", "seals", "benthivorous_fish"]
    _apply(app, "non_decision_makers", "mobile_epifauna", True)
    assert "mobile_epifauna" in app._reanchor_ids()
    for gid, cat in (("porpoises", "decision_makers"), ("seals", "decision_makers"),
                     ("benthivorous_fish", "decision_makers"),
                     ("mobile_epifauna", "non_decision_makers")):
        _apply(app, cat, gid, False)
    # An emptied list leaves the manifest.
    assert "warmup" not in app.project_data["simulation_settings"]


def test_an_exchange_set_group_is_refused(app):
    for cat, gid in (("decision_makers", "planktivorous_fish"), ("non_decision_makers", "krill")):
        app.shown.clear()
        _apply(app, cat, gid, True)
        assert app.shown[-1][0] == "Invalid value" and "Outside Density" in app.shown[-1][1]
        assert gid not in app._reanchor_ids()


def test_a_removed_group_leaves_the_list(app):
    _select(app, "decision_makers", "seals")
    app.remove_fg("decision_makers")
    assert app._reanchor_ids() == ["porpoises"]
