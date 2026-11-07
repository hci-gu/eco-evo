"""The FG editor carries the open pool and the pairs' depth tables
(section 245): shown, round-tripped unchanged, re-derived with r_max 0,
validated, and removed when the pool is closed or a cell emptied."""
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
from lib.world.growth_budget import growth_budget  # noqa: E402

DMS = ["zooplankton", "planktivorous_fish", "predatory_fish", "mackerel",
       "benthivorous_fish", "whiting"]
NDMS = ["phytoplankton", "krill", "mobile_epifauna"]
PREDS = ("planktivorous_fish", "predatory_fish", "mackerel", "benthivorous_fish", "whiting")


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
    a.project_data["decision_makers"] = [{"group_id": g} for g in DMS]
    a.project_data["non_decision_makers"] = [{"group_id": g} for g in NDMS]
    species = a.global_library["species_definitions"]
    for gid in DMS + NDMS:
        a.current_fg_configs[gid] = species[gid]
    a.update_fg_list()
    a.refresh_matrix()
    a.shown = shown
    yield a
    root.destroy()


def _select_krill(app):
    # The list is sorted by display name; select by the project's order.
    ids = [fg["group_id"] for fg in app.project_data["non_decision_makers"]]
    app.ndm_listbox.selection_clear(0, "end")
    app.ndm_listbox.selection_set(ids.index("krill"))
    app.on_fg_select("non_decision_makers")


def test_depth_access_cells_round_trip(app):
    pairs = app.global_library["interaction_definitions"]
    before = {p: copy.deepcopy(pairs[f"{p}_preys_on_krill"]) for p in PREDS}
    for p in PREDS:
        assert app.matrix_entries[f"{p}_preys_on_krill"]["depth_access_table"].get()
    app.apply_matrix_changes()
    assert {p: pairs[f"{p}_preys_on_krill"] for p in PREDS} == before
    cell = app.matrix_entries["whiting_preys_on_krill"]["depth_access_table"]
    cell.set("50:1, 200:0")
    app.apply_matrix_changes()
    assert pairs["whiting_preys_on_krill"]["depth_access_table"] == [
        {"depth_m": 50.0, "factor": 1.0}, {"depth_m": 200.0, "factor": 0.0}]
    cell.set("50:1, 50:2")                      # repeated depth: refused
    app.apply_matrix_changes()
    assert pairs["whiting_preys_on_krill"]["depth_access_table"][0]["factor"] == 1.0
    cell.set("")
    app.apply_matrix_changes()
    assert "depth_access_table" not in pairs["whiting_preys_on_krill"]


def test_krill_fields_round_trip_and_rederive(app):
    entry = app.global_library["species_definitions"]["krill"]
    before = copy.deepcopy(entry)
    _select_krill(app)
    v = app.ndm_prop_vars
    assert float(v["outside_density"].get()) == pytest.approx(5.0)
    assert len(v["outside_season"].get().split(",")) == 12
    assert len(app.outside_rows) == 6
    app.apply_fg_changes()
    assert not [t for t, _ in app.shown if t.startswith("Invalid")], app.shown
    for key in ("outside_density", "outside_exchange_rate", "outside_season",
                "outside_depth_table", "r_max"):
        assert entry[key] == before[key], key
    budget = growth_budget("krill", app.global_library["species_definitions"],
                           app.global_library["interaction_definitions"],
                           active_predators=DMS + NDMS)
    assert float(entry["growth_rate"]) == pytest.approx(budget.growth_rate, rel=1e-5)


def test_bad_values_are_refused_and_closing_removes_the_pool(app):
    entry = app.global_library["species_definitions"]["krill"]
    _select_krill(app)
    v = app.ndm_prop_vars
    v["outside_exchange_rate"].set("0")
    app.apply_fg_changes()
    assert app.shown[-1][0] == "Invalid value"
    _select_krill(app)
    v["outside_season"].set("1, 1, 1")
    app.apply_fg_changes()
    assert app.shown[-1][0] == "Invalid value"
    _select_krill(app)
    v["outside_exchange_rate"].set("1e-6")      # too weak to hold the pool
    app.apply_fg_changes()
    assert app.shown[-1][0] == "Invalid open pool"
    assert entry["outside_exchange_rate"] == pytest.approx(20.0 / 1460, rel=1e-6)
    _select_krill(app)
    v["outside_density"].set("")
    v["r_max"].set("0")
    app.apply_fg_changes()
    for key in ("outside_density", "outside_exchange_rate", "outside_season",
                "outside_depth_table"):
        assert key not in entry, key


def _select_dm(app, fg_id):
    ids = [fg["group_id"] for fg in app.project_data["decision_makers"]]
    app.fg_listbox.selection_clear(0, "end")
    app.fg_listbox.selection_set(ids.index(fg_id))
    app.on_fg_select("decision_makers")


def test_migrating_mackerel_round_trips_and_closes(app):
    """Section 246: the DM panel shows, keeps and re-derives the migration."""
    entry = app.global_library["species_definitions"]["mackerel"]
    before = copy.deepcopy(entry)
    _select_dm(app, "mackerel")
    v = app.prop_vars
    assert v["outside_entry"].get() == "west, north:0-0.5"
    assert len(v["outside_fill"].get().split(",")) == 12
    app.apply_fg_changes()
    assert not [t for t, _ in app.shown if t.startswith("Invalid")], app.shown
    for key in ("outside_density", "outside_exchange_rate", "outside_season",
                "outside_fill", "outside_entry", "r_max", "feeding_season_months"):
        assert entry[key] == before[key], key
    budget = growth_budget("mackerel", app.global_library["species_definitions"],
                           app.global_library["interaction_definitions"],
                           active_predators=DMS + NDMS)
    assert float(entry["growth_rate"]) == pytest.approx(budget.growth_rate, rel=1e-5)
    _select_dm(app, "mackerel")
    v["outside_entry"].set("up")
    app.apply_fg_changes()
    assert app.shown[-1][0] == "Invalid value"
    _select_dm(app, "mackerel")
    v["outside_fill"].set("0.5, 0.5")
    app.apply_fg_changes()
    assert app.shown[-1][0] == "Invalid open pool"
    _select_dm(app, "mackerel")
    v["outside_density"].set("")
    v["r_max"].set("0.255")
    app.apply_fg_changes()
    for key in ("outside_density", "outside_exchange_rate", "outside_season",
                "outside_fill", "outside_entry"):
        assert key not in entry, key


def test_start_fill_round_trips_and_empties(app):
    """Section 247: the DM panel shows, keeps, refuses and removes start_fill."""
    entry = app.global_library["species_definitions"]["planktivorous_fish"]
    before = list(entry["start_fill"])
    _select_dm(app, "planktivorous_fish")
    v = app.prop_vars
    assert len(v["start_fill"].get().split(",")) == 12
    app.apply_fg_changes()
    assert not [t for t, _ in app.shown if t.startswith("Invalid")], app.shown
    assert entry["start_fill"] == before
    _select_dm(app, "planktivorous_fish")
    v["start_fill"].set("0.5, 1.2")
    app.apply_fg_changes()
    assert app.shown[-1][0] == "Invalid value" and entry["start_fill"] == before
    _select_dm(app, "planktivorous_fish")
    v["start_fill"].set("")
    app.apply_fg_changes()
    assert "start_fill" not in entry
