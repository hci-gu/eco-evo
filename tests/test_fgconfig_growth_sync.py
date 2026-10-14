"""The FG editor follows the section-160 growth budget live.

The ideal reserve s* reads the predator's energy fields and the pair's
assimilation and handling time, so editing any of them must move the
derived growth_rate the editor shows, exactly as growth_budget computes.
"""
import copy
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Same import as tests/test_fgconfig_field_help.py: the editor module is
# fgconfig/fgconfig.py, imported with its directory on the path.
sys.path.insert(0, os.path.join(_ROOT, "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")
from lib.world.growth_budget import growth_budget  # noqa: E402


@pytest.fixture
def app():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    a = fgconfig.FGConfigApp(root)
    a.global_library = copy.deepcopy(a.global_library)
    a.project_data["decision_makers"] = [
        {"group_id": "gadoids"}, {"group_id": "pelagic_fish"},
        {"group_id": "zooplankton"}, {"group_id": "porpoises"}]
    a.project_data["non_decision_makers"] = [
        {"group_id": "benthic_community"}, {"group_id": "phytoplankton"}]
    species = a.global_library["species_definitions"]
    for fg in a.project_data["decision_makers"] + a.project_data["non_decision_makers"]:
        a.current_fg_configs[fg["group_id"]] = species[fg["group_id"]]
    a.update_fg_list()
    yield a
    root.destroy()


def _select(app, fg_id):
    ids = [fg["group_id"] for fg in app.project_data["decision_makers"]]
    app.fg_listbox.selection_clear(0, "end")
    app.fg_listbox.selection_set(ids.index(fg_id))
    app.on_fg_select("decision_makers")


@pytest.mark.parametrize("key,value", [
    ("max_energy_reserve", 4000.0), ("energy_content", 4000.0),
    ("resting_metabolism", 9.0), ("feeding_cost", 2.5)])
def test_editing_an_energy_field_moves_the_shown_growth_rate(app, key, value):
    _select(app, "gadoids")
    before = float(app.prop_vars["growth_rate"].get())
    app.prop_vars[key].set(str(value))
    shown = float(app.prop_vars["growth_rate"].get())
    spec = dict(app.current_fg_configs["gadoids"], **{key: value})
    lib = app.global_library
    expected = growth_budget("gadoids", lib["species_definitions"],
                             app._live_interactions(),
                             active_predators=app._active_project_fg_ids(),
                             species=spec).growth_rate
    assert shown != pytest.approx(before, rel=1e-4)
    assert shown == pytest.approx(expected, rel=1e-5)


def test_live_interactions_overlay_handling_time(app):
    app.refresh_matrix()
    key = "gadoids_preys_on_zooplankton"
    var = app.matrix_entries.get(key, {}).get("handling_time")
    if var is None:
        pytest.skip("no handling_time cell")
    var.set("40.0")
    assert app._live_interactions()[key]["handling_time"] == pytest.approx(40.0)
    # h 130 -> 40 makes zooplankton the best pair by far (90 vs 27.7 MJ/t).
    budget = app._budget_for("gadoids")
    assert budget.best_prey == "zooplankton"
    assert budget.ideal_intake == pytest.approx(0.8 * 4500.0 / 40.0, rel=1e-6)
    var.set("")
    lib_value = app.global_library["interaction_definitions"][key]["handling_time"]
    assert app._live_interactions()[key]["handling_time"] == lib_value
