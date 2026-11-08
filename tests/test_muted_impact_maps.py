"""Muted impact maps: ``baseline_map_muted`` and ``inference.impact_maps_muted``.

A muted map keeps its path in the project and is read as if it were not
there: the baseline map by the training worlds, the inference map by the
inference worlds and the long-run probe. Each mute leaves the other kind of
map alone, and the equilibrium start pattern (today's state) keeps reading
the baseline map. The FG editor's Mute boxes write both keys.
"""
import os
import sys

import numpy as np
import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "fgconfig"))

from inference import inference_impact_units, load_inference_impact_maps  # noqa: E402
from lib.world import habitat  # noqa: E402
from lib.world.impact_maps import project_baseline_maps, sample_impact_maps  # noqa: E402


def _write_map(path, value=2.0, unit=None):
    field = np.full((4, 4), value, dtype=np.float32)
    if unit:
        np.savez(path, effort=field, unit=unit)
    else:
        np.savez(path, effort=field)


def test_a_muted_baseline_map_is_absent_from_training_and_keeps_its_path(tmp_path):
    _write_map(tmp_path / "effort.npz")
    entry = {"impact_id": "fishing", "baseline_map": "effort.npz", "baseline_map_muted": True}
    project = {"impact_variables": [entry, {"impact_id": "other", "baseline_map": "effort.npz"}]}
    maps = project_baseline_maps(project, str(tmp_path / "p.yaml"), (4, 4))
    assert set(maps) == {"other"}
    assert entry["baseline_map"] == "effort.npz"
    assert set(project_baseline_maps(project, str(tmp_path / "p.yaml"), (4, 4),
                                     include_muted=True)) == {"fishing", "other"}
    fields = sample_impact_maps(["fishing", "other"], {"fishing": (0, 10), "other": (0, 10)},
                                (4, 4), seed=1, observable_impact_vars=[], baseline_maps=maps)
    assert not fields["fishing"].any()
    assert (fields["other"] == 2.0).all()


def test_a_muted_baseline_map_gives_an_observable_impact_its_spawned_field(tmp_path):
    _write_map(tmp_path / "effort.npz")
    project = {"impact_variables": [{"impact_id": "seen", "observable": True,
                                     "baseline_map": "effort.npz", "baseline_map_muted": True}]}
    maps = project_baseline_maps(project, str(tmp_path / "p.yaml"), (4, 4))
    field = sample_impact_maps(["seen"], {"seen": (100, 200)}, (4, 4), seed=1,
                               observable_impact_vars=["seen"], baseline_maps=maps)["seen"]
    assert (field >= 100).all()


def test_a_muted_inference_map_is_a_zero_field_and_leaves_the_others(tmp_path):
    _write_map(tmp_path / "a.npz", unit="MW-h")
    _write_map(tmp_path / "b.npz", 3.0, unit="kg")
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"inference": {
        "impact_maps": {"a": "a.npz", "b": "b.npz"}, "impact_maps_muted": ["a"]}}))
    maps = load_inference_impact_maps(str(path), ["a", "b"], 4, 4, verbose=False)
    assert not maps["a"].any()
    assert (maps["b"] == 3.0).all()
    assert inference_impact_units(str(path)) == {"b": "kg"}


def test_the_two_mutes_are_independent(tmp_path):
    _write_map(tmp_path / "effort.npz")
    path = tmp_path / "p.yaml"
    both = {"impact_variables": [{"impact_id": "x", "baseline_map": "effort.npz",
                                  "baseline_map_muted": True}],
            "inference": {"impact_maps": {"x": "effort.npz"}}}
    path.write_text(yaml.safe_dump(both))
    assert not project_baseline_maps(both, str(path), (4, 4))
    assert load_inference_impact_maps(str(path), ["x"], 4, 4, verbose=False)["x"].any()
    other = {"impact_variables": [{"impact_id": "x", "baseline_map": "effort.npz"}],
             "inference": {"impact_maps": {"x": "effort.npz"}, "impact_maps_muted": ["x"]}}
    path.write_text(yaml.safe_dump(other))
    assert project_baseline_maps(other, str(path), (4, 4))
    assert not load_inference_impact_maps(str(path), ["x"], 4, 4, verbose=False)["x"].any()


def test_the_equilibrium_start_pattern_keeps_reading_a_muted_baseline_map():
    project_path = os.path.join(_ROOT, "mareld2.yaml")
    if not os.path.isfile(os.path.join(_ROOT, "maps", "fishing_bottom_otter_trawls.npz")):
        pytest.skip("maps/ is untracked")
    project = yaml.safe_load(open(project_path, encoding="utf-8-sig"))
    library = yaml.safe_load(open(os.path.join(_ROOT, "fgconfig", "fg_library.yaml"),
                                  encoding="utf-8-sig"))
    ids = [g["group_id"] for key in ("decision_makers", "non_decision_makers")
           for g in project[key] if not g.get("muted")]
    fid = "benthic_infauna"
    _, before = habitat.build(project, project_path, library, (16, 16), ids)
    for iv in project["impact_variables"]:
        if iv.get("baseline_map"):
            iv["baseline_map_muted"] = True
    assert not project_baseline_maps(project, project_path, (16, 16))
    _, after = habitat.build(project, project_path, library, (16, 16), ids)
    assert before[fid]["equilibrium"] is not None
    np.testing.assert_array_equal(before[fid]["equilibrium"], after[fid]["equilibrium"])


pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")


@pytest.fixture
def app(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    a = fgconfig.FGConfigApp(root)
    a.project_path = str(tmp_path / "project.yaml")
    a.project_data["impact_variables"] = [
        {"impact_id": "pelagic_trawling", "value_min": 0.0, "value_max": 30.0}]
    a.update_impact_list()
    a.impact_listbox.selection_set(0)
    a.on_impact_select()
    yield a
    root.destroy()


def _npz(tmp_path):
    path = tmp_path / "effort.npz"
    np.savez(path, effort=np.ones((60, 60), dtype=np.float32))
    return str(path)


def _state(widget):
    return str(widget.cget("state"))


def test_the_baseline_mute_box_needs_a_map_and_keeps_the_path(app, tmp_path):
    entry = app.project_data["impact_variables"][0]
    assert _state(app.impact_baseline_mute_cb) == "disabled"
    app._impact_set_baseline_map(_npz(tmp_path))
    assert _state(app.impact_baseline_mute_cb) == "normal"

    app.impact_baseline_mute_var.set(True)
    app._impact_toggle_baseline_mute()
    assert entry["baseline_map_muted"] is True and entry["baseline_map"]
    assert app.impact_baseline_status_var.get().endswith(app.MUTED_MAP_NOTE)
    assert "[baseline map muted]" in app.impact_listbox.get(0)

    app.impact_baseline_mute_var.set(False)
    app._impact_toggle_baseline_mute()
    assert "baseline_map_muted" not in entry
    assert not app.impact_baseline_status_var.get().endswith(app.MUTED_MAP_NOTE)
    assert "[baseline map]" in app.impact_listbox.get(0)


def test_a_new_or_cleared_baseline_map_is_no_longer_muted(app, tmp_path):
    entry = app.project_data["impact_variables"][0]
    app._impact_set_baseline_map(_npz(tmp_path))
    entry["baseline_map_muted"] = True
    app._impact_set_baseline_map(_npz(tmp_path))
    assert "baseline_map_muted" not in entry
    entry["baseline_map_muted"] = True
    app._impact_clear_baseline_map()
    assert "baseline_map_muted" not in entry and "baseline_map" not in entry
    assert _state(app.impact_baseline_mute_cb) == "disabled"


def test_a_muted_baseline_map_shows_the_spawn_strategy_again(app, tmp_path):
    app.project_data["impact_variables"][0]["observable"] = True
    app.on_impact_select()
    app._impact_set_baseline_map(_npz(tmp_path))
    app.root.update_idletasks()
    assert not app.impact_spawn_frame.grid_info()
    app.impact_baseline_mute_var.set(True)
    app._impact_toggle_baseline_mute()
    assert app.impact_spawn_frame.grid_info()


def test_the_inference_mute_box_writes_the_project_list(app, tmp_path):
    app._refresh_inference_impact_maps()
    row = app.inference_impact_widgets["pelagic_trawling"]
    assert _state(row["mute_cb"]) == "disabled"
    app._inference_set_map("pelagic_trawling", _npz(tmp_path), persist=True)
    assert _state(row["mute_cb"]) == "normal"

    row["mute_var"].set(True)
    app._inference_toggle_map_mute("pelagic_trawling")
    infer = app.project_data["inference"]
    assert infer["impact_maps_muted"] == ["pelagic_trawling"]
    assert infer["impact_maps"]["pelagic_trawling"]
    assert row["status_var"].get().endswith(app.MUTED_MAP_NOTE)

    app._refresh_inference_impact_maps()  # the rows are rebuilt from the project
    row = app.inference_impact_widgets["pelagic_trawling"]
    assert row["mute_var"].get() and row["status_var"].get().endswith(app.MUTED_MAP_NOTE)

    row["mute_var"].set(False)
    app._inference_toggle_map_mute("pelagic_trawling")
    assert "impact_maps_muted" not in app.project_data["inference"]
    assert not row["status_var"].get().endswith(app.MUTED_MAP_NOTE)


def test_a_new_or_cleared_inference_map_is_no_longer_muted(app, tmp_path):
    app._refresh_inference_impact_maps()
    app._inference_set_map("pelagic_trawling", _npz(tmp_path), persist=True)
    app._inference_set_map_muted("pelagic_trawling", True)
    app._inference_set_map("pelagic_trawling", _npz(tmp_path), persist=True)
    assert "impact_maps_muted" not in app.project_data["inference"]
    app._inference_set_map_muted("pelagic_trawling", True)
    app._inference_clear_map("pelagic_trawling")
    assert "inference" not in app.project_data


def test_the_mutes_survive_save_and_load_and_reach_the_readers(app, tmp_path, monkeypatch):
    path = _npz(tmp_path)
    app._impact_set_baseline_map(path)
    app.impact_baseline_mute_var.set(True)
    app._impact_toggle_baseline_mute()
    app._refresh_inference_impact_maps()
    app._inference_set_map("pelagic_trawling", path, persist=True)
    app._inference_set_map_muted("pelagic_trawling", True)
    monkeypatch.setattr(fgconfig.messagebox, "showinfo", lambda *a, **k: None)
    app.save_project()
    saved = yaml.safe_load(open(app.project_path, encoding="utf-8-sig"))
    assert saved["impact_variables"][0]["baseline_map_muted"] is True
    assert saved["inference"]["impact_maps_muted"] == ["pelagic_trawling"]
    assert not project_baseline_maps(saved, app.project_path, (60, 60))
    assert not load_inference_impact_maps(app.project_path, ["pelagic_trawling"], 60, 60,
                                          verbose=False)["pelagic_trawling"].any()
    app.load_project_from_path(app.project_path)
    assert app.project_data["impact_variables"][0]["baseline_map_muted"] is True
    assert list(app._inference_muted_maps()) == ["pelagic_trawling"]
