"""Section 275.9: library and manifest keys the FG editor edits since then -
the member species, the reserve eaten with the prey (section 140), the
grazing pairs of a non decision maker (sections 177, 182) and the project's
nutrient block (sections 172, 180). The parsers run without a display; with
one, the editor writes each field back to the library or project.
"""
import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")
App = fgconfig.FGConfigApp
PROJECT = ROOT / "mareld2.yaml"


def _library():
    return yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml", encoding="utf-8-sig"))


def test_member_species_text_round_trips_on_the_library():
    for fid, spec in _library()["species_definitions"].items():
        members = spec.get("member_species")
        if not members:
            continue
        parsed, error = App._parse_members(App._members_text(members))
        assert error is None and App._same_members(members, parsed), fid
    assert App._parse_members("") == (None, None)
    parsed, error = App._parse_members(
        "crabs, hermit crabs and squat lobsters = 0.65; others[43] = 0.047; ciliates")
    assert error is None and parsed == [
        {"name": "crabs, hermit crabs and squat lobsters", "share": 0.65},
        {"name": "others", "count": 43, "share": 0.047}, {"name": "ciliates"}]
    for text, match in (("herring = 1.2", "[0, 1]"), ("a = 0.6; b = 0.6", "sum"),
                        ("others[x] = 0.1", "whole number"), ("= 0.3", "no name"),
                        ("herring = lots", "not a number")):
        assert App._parse_members(text)[0] is None, text
        assert match in App._parse_members(text)[1], text


@pytest.fixture
def app(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    shown = []
    saved = []
    try:
        app = App(root)
        app.load_project_from_path(str(PROJECT))
        app.library_path = str(tmp_path / "lib.yaml")
        app.save_yaml = lambda data, path: saved.append(path)
        for name in ("showinfo", "showwarning", "showerror"):
            setattr(fgconfig.messagebox, name,
                    lambda *a, _n=name, **k: shown.append((_n,) + a))
        app._shown, app._saved = shown, saved
        yield app
    finally:
        root.destroy()


def _select(app, fid):
    for cat, box in (("decision_makers", app.fg_listbox),
                     ("non_decision_makers", app.ndm_listbox)):
        ids = [fg["group_id"] for fg in app.project_data[cat]]
        if fid in ids:
            box.selection_clear(0, "end")
            box.selection_set(ids.index(fid))
            app.on_fg_select(cat)
            return app.prop_vars if cat == "decision_makers" else app.ndm_prop_vars
    raise KeyError(fid)


def test_the_reserve_field_writes_both_keys(app):
    species = app.global_library["species_definitions"]
    variables = _select(app, "planktivorous_fish")
    assert variables["reserve_reference_fill"].get() == "0.5"
    variables["reserve_reference_fill"].set("0.62")
    app.apply_fg_changes()
    assert species["planktivorous_fish"]["prey_includes_reserve"] is True
    assert species["planktivorous_fish"]["reserve_reference_fill"] == 0.62
    variables = _select(app, "planktivorous_fish")
    variables["reserve_reference_fill"].set("")
    app.apply_fg_changes()
    assert "prey_includes_reserve" not in species["planktivorous_fish"]
    assert "reserve_reference_fill" not in species["planktivorous_fish"]
    variables = _select(app, "planktivorous_fish")
    variables["reserve_reference_fill"].set("1.5")
    app.apply_fg_changes()
    assert app._shown[-1][0] == "showerror"
    assert "prey_includes_reserve" not in species["planktivorous_fish"]
    # A group without the key shows an empty field and keeps none.
    variables = _select(app, "porpoises")
    assert variables["reserve_reference_fill"].get() == ""
    app.apply_fg_changes()
    assert "prey_includes_reserve" not in species["porpoises"]


def test_the_member_field_writes_and_empties(app):
    species = app.global_library["species_definitions"]
    before = copy.deepcopy(list(species["predatory_fish"]["member_species"]))
    variables = _select(app, "predatory_fish")
    app.apply_fg_changes()
    assert [dict(m) for m in species["predatory_fish"]["member_species"]] == before
    variables["member_species"].set("saithe = 0.8; others[6] = 0.2")
    app.apply_fg_changes()
    assert [dict(m) for m in species["predatory_fish"]["member_species"]] == [
        {"name": "saithe", "share": 0.8}, {"name": "others", "count": 6, "share": 0.2}]
    variables = _select(app, "microzooplankton")
    variables["member_species"].set("")
    app.apply_fg_changes()
    assert "member_species" not in species["microzooplankton"]


def test_the_grazer_is_a_row_of_its_matrices(app):
    app.refresh_matrix()
    inter = app.global_library["interaction_definitions"]
    key = "microzooplankton_preys_on_subsurface_phytoplankton"
    cells = app.matrix_entries[key]
    for data_key in ("preys_on", "predation_mortality", "handling_time", "max_intake_rate",
                     "mixed_layer_reference_m", "food_threshold", "edibility_floor",
                     "edibility_half_saturation_umol_l"):
        assert data_key in cells, data_key
    assert "observes" not in app.matrix_entries.get(
        "microzooplankton_observes_phytoplankton", {})
    before = dict(inter[key])
    app.apply_matrix_changes()
    assert dict(inter[key]) == before
    cells["food_threshold"].set("0.9")
    cells["edibility_floor"].set("0.2")
    app.apply_matrix_changes()
    assert inter[key]["food_threshold"] == 0.9 and inter[key]["edibility_floor"] == 0.2
    # The floor and K go together: one alone is a cell error, nothing changes.
    cells["edibility_half_saturation_umol_l"].set("")
    app.apply_matrix_changes()
    assert "go together" in app._shown[-1][2]
    assert inter[key]["edibility_half_saturation_umol_l"] == 2.0
    cells["edibility_floor"].set("")
    app.apply_matrix_changes()
    assert "edibility_floor" not in inter[key]
    assert "edibility_half_saturation_umol_l" not in inter[key]


def test_relocation_speed_text_round_trips_on_the_library():
    """Section 276: a speed or 'share:speed, ...' members."""
    seen = 0
    for fid, spec in _library()["species_definitions"].items():
        value = spec.get("relocation_speed")
        if value is None:
            continue
        seen += 1
        parsed, error = App._parse_speed(App._speed_text(value))
        assert error is None and App._same_speed(value, parsed), fid
    assert seen == 7
    assert App._parse_speed("") == (None, None)
    assert App._parse_speed("0.65") == (0.65, None)
    assert App._parse_speed("0.55:0.6, 0.45:0.13") == ([[0.55, 0.6], [0.45, 0.13]], None)
    for text, match in (("0", "> 0"), ("0.5:0.6", "sum"), ("fast", "not a speed"),
                        ("0.5:0.6, 0.5", "share:speed")):
        assert App._parse_speed(text)[0] is None, text
        assert match in App._parse_speed(text)[1], text


def test_the_relocation_speed_field_writes_and_empties(app):
    species = app.global_library["species_definitions"]
    variables = _select(app, "planktivorous_fish")
    assert variables["relocation_speed"].get() == "0.55:0.6, 0.45:0.13"
    before = [list(m) for m in species["planktivorous_fish"]["relocation_speed"]]
    app.apply_fg_changes()
    assert [list(m) for m in species["planktivorous_fish"]["relocation_speed"]] == before
    variables = _select(app, "porpoises")
    variables["relocation_speed"].set("0.8")
    app.apply_fg_changes()
    assert species["porpoises"]["relocation_speed"] == 0.8
    variables = _select(app, "porpoises")
    variables["relocation_speed"].set("-1")
    app.apply_fg_changes()
    assert app._shown[-1][0] == "showerror"
    assert species["porpoises"]["relocation_speed"] == 0.8
    variables = _select(app, "porpoises")
    variables["relocation_speed"].set("")
    app.apply_fg_changes()
    assert "relocation_speed" not in species["porpoises"]


def test_the_nutrient_panel_round_trips_and_checks(app):
    block = yaml.safe_load(open(PROJECT, encoding="utf-8-sig"))[
        "simulation_settings"]["nutrients"]
    written, keep = app._nutrient_block()
    assert keep and written == block
    assert app._store_nutrient_settings()
    assert dict(app.project_data["simulation_settings"]["nutrients"]) == block
    app.nutrients_vars["exchange_m_per_day"].set("0.4")
    assert app._store_nutrient_settings()
    assert app.project_data["simulation_settings"]["nutrients"]["exchange_m_per_day"] == 0.4
    for key, text, match in (("surface_umol_l", "1, 2, 3", "12 monthly"),
                             ("bottom_m", "5", "below the deepest"),
                             ("light_attenuation_per_m", "1, 2", "one number")):
        old = app.nutrients_vars[key].get()
        app.nutrients_vars[key].set(text)
        assert not app._store_nutrient_settings(), key
        assert match in app._shown[-1][2], (key, app._shown[-1])
        app.nutrients_vars[key].set(old)
    app.nutrients_sub_enabled_var.set(False)
    assert app._store_nutrient_settings()
    assert app.project_data["simulation_settings"]["nutrients"][
        "subsurface_layer"]["enabled"] is False


def test_an_empty_panel_adds_no_block(app):
    app.project_data["simulation_settings"].pop("nutrients")
    app._load_nutrient_settings()
    assert app._nutrient_block()[1] is False
    assert app._store_nutrient_settings()
    assert "nutrients" not in app.project_data["simulation_settings"]
