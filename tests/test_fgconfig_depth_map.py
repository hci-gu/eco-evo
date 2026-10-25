"""The project's depth map and the Capacity by Depth table in fgconfig.

The depth map (simulation_settings.depth_map) is set on the Project tab
with a drop / browse zone like the baseline maps; the benthos' K table
(capacity_depth_table) is edited in the NDM FG editor, validated with the
runtime's parser, previewed on the depth map, and written to the library
with Apply Changes (section 207).
"""
import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")


@pytest.fixture
def app(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(fgconfig.messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(fgconfig.messagebox, "askyesno", lambda *a, **k: True)
    a = fgconfig.FGConfigApp(root)
    a.global_library = copy.deepcopy(a.global_library)
    a.library_path = str(tmp_path / "fg_library.yaml")       # never the repo's
    (tmp_path / "maps").mkdir()
    shutil.copy(ROOT / "maps" / "depth.npz", tmp_path / "maps" / "depth.npz")
    a.project_path = str(tmp_path / "project.yaml")
    a.project_data["simulation_settings"] = {"depth_map": "maps/depth.npz"}
    a.project_data["decision_makers"] = [{"group_id": "benthivorous_fish"}]
    a.project_data["non_decision_makers"] = [{"group_id": "benthic_community"}]
    species = a.global_library["species_definitions"]
    for gid in ("benthivorous_fish", "benthic_community"):
        a.current_fg_configs[gid] = species[gid]
    a.update_fg_list()
    a._update_depth_map_view()
    yield a
    root.destroy()


def _select_benthos(app):
    app.ndm_listbox.selection_clear(0, "end")
    app.ndm_listbox.selection_set(0)
    app.on_fg_select("non_decision_makers")


def test_the_row_shows_the_project_depth_map(app):
    status = app.depth_map_status_var.get()
    assert "depth.npz" in status and "60×60" in status
    assert "44-529 m" in status


def test_set_and_clear_write_simulation_settings(app, tmp_path):
    other = tmp_path / "maps" / "other.npz"
    np.savez(other, depth=np.full((10, 12), 150.0))
    assert app._set_depth_map(str(other))
    assert app.project_data["simulation_settings"]["depth_map"] == os.path.join("maps", "other.npz")
    assert "resampled" in app.depth_map_status_var.get()
    app._clear_depth_map()
    assert "depth_map" not in app.project_data["simulation_settings"]


def test_a_map_with_negative_depths_is_refused(app, tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, depth=np.full((5, 5), -3.0))
    assert not app._set_depth_map(str(bad))
    assert app.project_data["simulation_settings"]["depth_map"] == "maps/depth.npz"


def test_the_table_editor_shows_the_library_rows_and_k_on_the_map(app):
    _select_benthos(app)
    rows = [(r["depth"].get(), r["capacity"].get()) for r in app.capacity_rows]
    assert rows == [("100", "150.8"), ("300", "75.4")]
    preview = app.capacity_preview_var.get()
    assert preview.startswith("K on the depth map: mean 126.6")


def test_apply_writes_the_edited_table_to_the_library(app):
    _select_benthos(app)
    app.capacity_rows[1]["capacity"].set("60")
    app._add_capacity_row("500", "40")
    app.apply_fg_changes()
    table = app.global_library["species_definitions"]["benthic_community"]["capacity_depth_table"]
    assert [(float(r["depth_m"]), float(r["capacity"])) for r in table] == [
        (100.0, 150.8), (300.0, 60.0), (500.0, 40.0)]
    assert Path(app.library_path).is_file()


def test_emptied_table_leaves_the_library(app):
    _select_benthos(app)
    for row in app.capacity_rows:
        row["depth"].set("")
        row["capacity"].set("")
    assert app._collect_capacity_table() == (None, None)
    app.apply_fg_changes()
    assert "capacity_depth_table" not in app.global_library["species_definitions"]["benthic_community"]


def test_duplicate_depths_are_refused(app):
    _select_benthos(app)
    app.capacity_rows[1]["depth"].set("100")
    table, error = app._collect_capacity_table()
    assert table is None and "distinct" in error
    before = copy.deepcopy(app.global_library["species_definitions"]["benthic_community"]["capacity_depth_table"])
    app.apply_fg_changes()
    assert app.global_library["species_definitions"]["benthic_community"]["capacity_depth_table"] == before
