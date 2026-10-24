"""The FG editor's impact matrices cover every group (section 161).

Bottom trawling kills benthos, a non decision maker, so the Impact
Affects / Impact Tables matrices need its row; the engine applies the
biomass factor to it on CPU and GPU.
"""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Same import as tests/test_fgconfig_field_help.py.
sys.path.insert(0, os.path.join(_ROOT, "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")


@pytest.fixture
def app():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    a = fgconfig.FGConfigApp(root)
    species = a.global_library["species_definitions"]
    a.project_data["decision_makers"] = [{"group_id": "benthivorous_fish"}]
    a.project_data["non_decision_makers"] = [{"group_id": "benthic_community"}]
    a.project_data["impact_variables"] = [
        {"impact_id": "bottom_trawling", "value_min": 0.0, "value_max": 100.0}]
    for fg in a.project_data["decision_makers"] + a.project_data["non_decision_makers"]:
        a.current_fg_configs[fg["group_id"]] = species[fg["group_id"]]
    a.update_fg_list()
    yield a
    root.destroy()


def test_benthos_has_an_impact_row_read_from_the_library(app):
    app.refresh_matrix()
    for fid in ("benthivorous_fish", "benthic_community"):
        cell = app.matrix_entries.get(f"{fid}_impacted_by_bottom_trawling", {})
        assert "impact_affects" in cell, fid
        assert cell["impact_affects"].get() is True, fid
