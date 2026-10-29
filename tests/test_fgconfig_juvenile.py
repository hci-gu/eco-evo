"""FG editor: the Juvenile Stage rows and the Stage Access matrix (223.32).

The parsers are checked without a display; with one, an editor round trip
on the live library shows the block and the pair values and writes them
back unchanged.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")
App = fgconfig.FGConfigApp

ROWS = {"juvenile_growth_ratio": "5.3", "juvenile_graduation_rate": "0.66",
        "juvenile_recruitment": "0, 0, 0, 0, 0, 1.7, 4.1, 4.2, 1.8, 0, 0, 0",
        "juvenile_start_share": "0.38",
        "juvenile_fishing_selectivity": "pelagic_trawling = 0.7, bottom_trawling = 1.5"}


def test_juvenile_rows_make_the_library_block():
    block, error = App._collect_juvenile_block(dict(ROWS), "planktivorous_fish")
    assert error is None
    assert block == {"growth_ratio": 5.3, "graduation_rate": 0.66,
                     "recruitment": [0, 0, 0, 0, 0, 1.7, 4.1, 4.2, 1.8, 0, 0, 0],
                     "start_share": 0.38,
                     "fishing_selectivity": {"pelagic_trawling": 0.7, "bottom_trawling": 1.5}}


def test_empty_growth_ratio_means_no_stage():
    assert App._collect_juvenile_block(dict(ROWS, juvenile_growth_ratio=""), "x") == (None, None)


@pytest.mark.parametrize("key,value", [
    ("juvenile_recruitment", "1, 2, 3"),
    ("juvenile_start_share", "1.2"),
    ("juvenile_graduation_rate", ""),
    ("juvenile_fishing_selectivity", "bottom_trawling"),
])
def test_bad_juvenile_rows_are_refused(key, value):
    block, error = App._collect_juvenile_block(dict(ROWS, **{key: value}), "x")
    assert block is None and error


def test_stage_access_cells():
    assert App._parse_stage_access("", {}) == (None, None)
    assert App._parse_stage_access("0.43, 0.027", {}) == ([0.43, 0.027], None)
    for bad in ("0.5", "1.2, 0", "0, 0", "a, b"):
        assert App._parse_stage_access(bad, {})[1]
    assert App._parse_stage_access("0.5, 0.5", {"accessible_fraction": 0.3})[1]


def test_editor_round_trip_on_the_live_library(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        app = App(root)
        app.library_path = str(tmp_path / "lib.yaml")
        app.save_yaml = lambda data, path: None
        app.project_data["decision_makers"] = [
            {"group_id": "planktivorous_fish"}, {"group_id": "predatory_fish"}]
        app.project_data["non_decision_makers"] = [{"group_id": "zooplankton"}]
        species = app.global_library["species_definitions"]
        for gid in ("planktivorous_fish", "predatory_fish", "zooplankton"):
            app.current_fg_configs[gid] = species[gid]
        before = dict(species["planktivorous_fish"]["juvenile_stage"])
        pair = list(app.global_library["interaction_definitions"][
            "predatory_fish_preys_on_planktivorous_fish"]["stage_access"])
        app.update_fg_list()
        app.refresh_matrix()
        key = "predatory_fish_preys_on_planktivorous_fish"
        assert app.matrix_entries[key]["stage_access"].get() == ", ".join(f"{v:g}" for v in pair)
        index = [fg["group_id"] for fg in app.project_data["decision_makers"]].index("planktivorous_fish")
        app.fg_listbox.selection_clear(0, "end")
        app.fg_listbox.selection_set(index)
        app.on_fg_select("decision_makers")
        assert app.juvenile_vars["juvenile_growth_ratio"].get() == f"{before['growth_ratio']:g}"
        fgconfig.messagebox.showinfo = lambda *a, **k: None
        fgconfig.messagebox.showwarning = lambda *a, **k: None
        app.apply_fg_changes()
        after = species["planktivorous_fish"]["juvenile_stage"]
        assert dict(after) == before
        app.matrix_entries[key]["stage_access"].set("0.5, 0.2")
        app.apply_matrix_changes()
        assert list(app.global_library["interaction_definitions"][key]["stage_access"]) == [0.5, 0.2]
    finally:
        root.destroy()
