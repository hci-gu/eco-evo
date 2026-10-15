"""The Impact Editor's baseline map (section 161).

Browse - or a drag-and-drop onto the zone, as on the Inference tab -
stores the .npz relative to the project file on the selected impact as
``baseline_map``, the list marks the impact, and Clear removes the key.
The Inference tab keeps its own map: the baseline map is never copied
there.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Same import as tests/test_fgconfig_field_help.py.
sys.path.insert(0, os.path.join(_ROOT, "fgconfig"))

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


def test_browse_and_clear_the_baseline_map(app, tmp_path, monkeypatch):
    path = tmp_path / "maps" / "effort.npz"
    path.parent.mkdir()
    np.savez(path, effort=np.ones((60, 60), dtype=np.float32))
    monkeypatch.setattr(fgconfig.filedialog, "askopenfilename", lambda **_: str(path))
    app._impact_browse_baseline_map()
    entry = app.project_data["impact_variables"][0]
    assert entry["baseline_map"] == os.path.join("maps", "effort.npz")
    assert app.impact_baseline_status_var.get().startswith("effort.npz\nshape=60×60")
    assert "[baseline map]" in app.impact_listbox.get(0)
    assert "pelagic_trawling" not in ((app.project_data.get("inference") or {})
                                      .get("impact_maps") or {})
    app._impact_clear_baseline_map()
    assert "baseline_map" not in entry
    assert app.impact_baseline_status_var.get().endswith("browse (.npz)") or \
        app.impact_baseline_status_var.get().startswith("Drop .npz")


def test_an_invalid_file_is_not_stored(app, tmp_path, monkeypatch):
    path = tmp_path / "flat.npz"
    np.savez(path, vector=np.ones(5))
    monkeypatch.setattr(fgconfig.filedialog, "askopenfilename", lambda **_: str(path))
    monkeypatch.setattr(fgconfig.messagebox, "showwarning", lambda *a, **k: None)
    app._impact_browse_baseline_map()
    assert "baseline_map" not in app.project_data["impact_variables"][0]


def test_drop_a_file_on_the_baseline_zone(app, tmp_path):
    path = tmp_path / "my maps" / "effort.npz"
    path.parent.mkdir()
    np.savez(path, effort=np.zeros((30, 30), dtype=np.float32))
    app._impact_on_drop_baseline("{" + str(path) + "} /ignored/second.npz")
    assert app.project_data["impact_variables"][0]["baseline_map"] == os.path.join(
        "my maps", "effort.npz")


def test_dnd_paths_parse_braces_and_spaces():
    parse = fgconfig.FGConfigApp._parse_dnd_paths
    assert parse("{/a b/c.npz} /d/e.npz") == ["/a b/c.npz", "/d/e.npz"]
    assert parse("") == []
