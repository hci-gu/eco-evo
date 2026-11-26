"""The FG editor's 'Random choice' box and Random spawn share (section 308.9):
the box sits above the Spawn Strategy's Mode row in both FG editors and edits
the project's ``simulation_settings.spawn_mix.groups``; Project Info's share
field edits ``spawn_mix.share``; a removed group leaves the list."""
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

DMS = ["zooplankton", "planktivorous_fish", "benthivorous_fish", "whiting"]
NDMS = ["phytoplankton", "krill"]


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
    a.project_data["simulation_settings"] = {"spawn_mix": {"share": 0.5, "groups": ["whiting"]}}
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


def _vars(app, category):
    return app.spawn_vars if category == "decision_makers" else app.ndm_spawn_vars


def _apply(app, category, gid, checked):
    _select(app, category, gid)
    _vars(app, category)["random_var"].set(checked)
    app.apply_fg_changes()


def test_the_box_sits_above_the_mode_row_in_both_editors(app):
    for frame in (app.spawn_frame, app.ndm_spawn_frame):
        outer = frame.winfo_children()[0]
        kids = outer.winfo_children()
        assert kids[0].winfo_class() == "TCheckbutton"
        assert str(kids[0].cget("text")) == "Random choice"
        assert any(w.winfo_class() == "TLabel" and str(w.cget("text")) == "Mode:"
                   for w in kids[1].winfo_children())
    assert "random_var" not in app.impact_spawn_vars


def test_the_box_shows_and_edits_the_project_list(app):
    _select(app, "decision_makers", "whiting")
    assert app.spawn_vars["random_var"].get() is True
    _select(app, "decision_makers", "benthivorous_fish")
    assert app.spawn_vars["random_var"].get() is False
    _apply(app, "decision_makers", "benthivorous_fish", True)
    _apply(app, "non_decision_makers", "phytoplankton", True)
    assert app._spawn_random_ids() == ["whiting", "benthivorous_fish", "phytoplankton"]
    for gid, cat in (("whiting", "decision_makers"), ("benthivorous_fish", "decision_makers"),
                     ("phytoplankton", "non_decision_makers")):
        _apply(app, cat, gid, False)
    # The share stays; the emptied list leaves the block.
    assert app.project_data["simulation_settings"]["spawn_mix"] == {"share": 0.5}


def test_the_share_field_loads_saves_and_refuses_bad_values(app):
    app._load_spawn_mix_share()
    assert app.spawn_mix_share_var.get() == "0.5"
    app.spawn_mix_share_var.set("0.25")
    assert app._store_spawn_mix_share()
    assert app.project_data["simulation_settings"]["spawn_mix"]["share"] == 0.25
    app.spawn_mix_share_var.set("1.5")
    assert not app._store_spawn_mix_share()
    assert app.shown[-1][0] == "Invalid value"
    app.spawn_mix_share_var.set("")
    assert app._store_spawn_mix_share()
    assert app.project_data["simulation_settings"]["spawn_mix"] == {"groups": ["whiting"]}
    _apply(app, "decision_makers", "whiting", False)
    assert "spawn_mix" not in app.project_data["simulation_settings"]


def test_a_removed_group_leaves_the_list(app):
    _select(app, "decision_makers", "whiting")
    app.remove_fg("decision_makers")
    assert app._spawn_random_ids() == []
