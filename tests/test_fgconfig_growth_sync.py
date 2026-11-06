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
        {"group_id": "benthivorous_fish"}, {"group_id": "planktivorous_fish"},
        {"group_id": "zooplankton"}, {"group_id": "porpoises"}]
    a.project_data["non_decision_makers"] = [
        {"group_id": "benthic_infauna"}, {"group_id": "phytoplankton"}]
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
    ("resting_metabolism", 4.5), ("feeding_cost", 2.5)])
def test_editing_an_energy_field_moves_the_shown_growth_rate(app, key, value):
    # Porpoises keep the section-133 gate: their s* is interior, so every
    # energy field moves it (section 240: a reserve_room DM sits at s* = 1).
    fg_id = "porpoises"
    _select(app, fg_id)
    before = float(app.prop_vars["growth_rate"].get())
    app.prop_vars[key].set(str(value))
    shown = float(app.prop_vars["growth_rate"].get())
    spec = dict(app.current_fg_configs[fg_id], **{key: value})
    lib = app.global_library
    expected = growth_budget(fg_id, lib["species_definitions"],
                             app._live_interactions(),
                             active_predators=app._active_project_fg_ids(),
                             species=spec).growth_rate
    assert shown != pytest.approx(before, rel=1e-4)
    assert shown == pytest.approx(expected, rel=1e-5)


def test_reserve_room_growth_rate_ignores_the_reserve_size(app):
    """Section 240: with the reserve-room appetite the benthivores' s* is 1
    (ample ideal intake), so g = gross / ((1 - u) 1460) whatever ME."""
    _select(app, "benthivorous_fish")
    before = float(app.prop_vars["growth_rate"].get())
    app.prop_vars["max_energy_reserve"].set("4000.0")
    assert float(app.prop_vars["growth_rate"].get()) == pytest.approx(before, rel=1e-6)


def test_live_interactions_overlay_handling_time(app):
    app.refresh_matrix()
    key = "benthivorous_fish_preys_on_benthic_infauna"
    var = app.matrix_entries.get(key, {}).get("handling_time")
    if var is None:
        pytest.skip("no handling_time cell")
    var.set("40.0")
    assert app._live_interactions()[key]["handling_time"] == pytest.approx(40.0)
    # h 96 -> 40 makes the infauna the best pair (39.0 vs 20.9 MJ/t);
    # without predator_share (removed, section 221) it takes the whole
    # eat share. The benthivores' SDA 0.10 is netted (section 240).
    budget = app._budget_for("benthivorous_fish")
    assert budget.best_prey == "benthic_infauna"
    assert budget.ideal_intake == pytest.approx((0.7 - 0.10) * 2600.0 / 40.0, rel=1e-6)
    var.set("")
    lib_value = app.global_library["interaction_definitions"][key]["handling_time"]
    assert app._live_interactions()[key]["handling_time"] == lib_value


def test_refuge_fields_round_trip(app, monkeypatch):
    """Exposed Share and Refuge Exchange Rate (section 223.12): shown,
    validated, written, and an emptied share removes the refuge."""
    from tkinter import messagebox
    shown = []
    monkeypatch.setattr(type(app), "save_yaml", lambda self, data, path: None)
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(messagebox, name,
                            lambda title, msg, *a, **k: shown.append(title))
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: True)
    entry = app.global_library["species_definitions"]["benthic_infauna"]

    def select():
        ids = [fg["group_id"] for fg in app.project_data["non_decision_makers"]]
        app.ndm_listbox.selection_clear(0, "end")
        app.ndm_listbox.selection_set(ids.index("benthic_infauna"))
        app.on_fg_select("non_decision_makers")

    v = app.ndm_prop_vars
    select()
    assert float(v["exposed_share"].get()) == pytest.approx(entry["exposed_share"])
    v["exposed_share"].set("1.0")
    app.apply_fg_changes()
    assert shown[-1] == "Invalid value"
    assert entry["exposed_share"] == pytest.approx(0.2)
    select()
    v["exposed_share"].set("0.25")
    app.apply_fg_changes()
    assert entry["exposed_share"] == pytest.approx(0.25)
    select()
    v["exposed_share"].set("")
    app.apply_fg_changes()
    assert "exposed_share" not in entry
    assert "refuge_exchange_rate" not in entry
