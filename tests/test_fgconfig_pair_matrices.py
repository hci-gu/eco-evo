"""FG Interactions: the pair attack rate, Holling type, mixed-layer reference
and accessible fraction matrices (fgconfig.PAIR_OPTIONAL_KEYS).

The cell parser is checked without a display; with one, an editor round
trip on the live library shows every pair value, writes the library back
unchanged when nothing is edited, removes a key for an emptied cell and
refuses an accessible fraction next to a stage access.
"""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")
App = fgconfig.FGConfigApp
parse = App._parse_pair_value


def test_empty_cells_remove_the_key():
    for key in fgconfig.PAIR_OPTIONAL_KEYS:
        assert parse(key, "") == (None, None)
        assert parse(key, "  ") == (None, None)


def test_valid_cells():
    assert parse("max_intake_rate", "0.01118") == (0.01118, None)
    assert parse("max_intake_rate", "2.41e-4") == (2.41e-4, None)
    assert parse("max_intake_rate", "0") == (0.0, None)
    assert parse("functional_response", "2") == (2, None)
    assert parse("functional_response", "3") == (3, None)
    assert parse("mixed_layer_reference_m", "10") == (10.0, None)
    assert parse("accessible_fraction", "1") == (1.0, None)
    assert parse("accessible_fraction", "0.029") == (0.029, None)


@pytest.mark.parametrize("key,text", [
    ("max_intake_rate", "-0.1"),
    ("max_intake_rate", "1e"),
    ("max_intake_rate", "nan"),
    ("functional_response", "1"),
    ("functional_response", "2.5"),
    ("mixed_layer_reference_m", "0"),
    ("accessible_fraction", "0"),
    ("accessible_fraction", "1.2"),
])
def test_bad_cells_are_refused(key, text):
    value, error = parse(key, text)
    assert value is None and error


@pytest.fixture
def app(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    app = App(root)
    app.library_path = str(tmp_path / "lib.yaml")
    app.save_yaml = lambda data, path: None
    fgconfig.messagebox.showinfo = lambda *a, **k: None
    fgconfig.messagebox.showwarning = lambda *a, **k: None
    dms = ["zooplankton", "planktivorous_fish", "predatory_fish", "mackerel",
           "benthivorous_fish", "seals"]
    ndms = ["phytoplankton", "subsurface_phytoplankton", "microzooplankton",
            "benthic_infauna", "mobile_epifauna"]
    app.project_data["decision_makers"] = [{"group_id": g} for g in dms]
    app.project_data["non_decision_makers"] = [{"group_id": g} for g in ndms]
    species = app.global_library["species_definitions"]
    for gid in dms + ndms:
        app.current_fg_configs[gid] = species[gid]
    app.update_fg_list()
    app.refresh_matrix()
    yield app
    root.destroy()


def _pairs(app):
    return app.global_library["interaction_definitions"]


def test_cells_show_the_library_values(app):
    shown = 0
    for key, data in app.matrix_entries.items():
        entry = _pairs(app).get(key, {})
        for data_key in fgconfig.PAIR_OPTIONAL_KEYS:
            var = data.get(data_key)
            if var is None:
                continue
            if entry.get(data_key) in (None, ""):
                assert var.get() == ""
            else:
                assert parse(data_key, var.get())[0] == pytest.approx(float(entry[data_key]), rel=0, abs=0)
                shown += 1
    assert shown >= 10  # attack rates, Holling types, references, fractions


def test_untouched_apply_leaves_the_library_unchanged(app):
    before = copy.deepcopy(dict(_pairs(app)))
    app.apply_matrix_changes()
    after = {k: dict(v) for k, v in _pairs(app).items() if k in before}
    assert after == {k: dict(v) for k, v in before.items()}


def test_edit_clear_and_conflict(app):
    key = "predatory_fish_preys_on_planktivorous_fish"
    cells = app.matrix_entries[key]
    assert cells["max_intake_rate"].get()
    cells["max_intake_rate"].set("0.02")
    cells["functional_response"].set("")
    app.apply_matrix_changes()
    assert _pairs(app)[key]["max_intake_rate"] == 0.02
    assert "functional_response" not in _pairs(app)[key]

    # Stage access present: an accessible fraction is refused and the
    # library keeps the stage access alone.
    assert cells["stage_access"].get()
    cells["accessible_fraction"].set("0.5")
    app.apply_matrix_changes()
    assert "accessible_fraction" not in _pairs(app)[key]
    assert _pairs(app)[key]["stage_access"]

    # A bad value is not written.
    seal = "seals_preys_on_predatory_fish"
    old = _pairs(app)[seal]["accessible_fraction"]
    app.matrix_entries[seal]["accessible_fraction"].set("1.5")
    app.apply_matrix_changes()
    assert _pairs(app)[seal]["accessible_fraction"] == old
