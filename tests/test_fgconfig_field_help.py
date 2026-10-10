"""Every input field in the FG editor has hover help (fgconfig/field_help.py).

Builds the editor on a real Tk root (skipped without a display) and
checks that each help key the editor binds resolves to a text, and that
every editable parameter of the FG editors, the spawn schemas and the
interaction matrices has one.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fgconfig"))

pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")
field_help = pytest.importorskip("field_help")

MATRIX_KEYS = (
    "preys_on", "predation_mortality", "observes", "assimilation_factor",
    "handling_time", "visibility_floor", "dark_ratio", "light_threshold_deg",
    "impact_affects", "impact_table",
)

FIXED_KEYS = (
    "project_name", "ref_grid", "daylight_enabled", "daylight_latitude",
    "daylight_start_day", "light_climate", "daylight_kd",
    "daylight_mixed_layer", "daylight_cloud", "temperature_enabled",
    "temperature_group_layers", "temperature_layers",
    "impact_value_range", "impact_observable",
    "spawn_mode", "impact_spawn_mode", "spawn_templates", "spawn_refs",
    "spawn_ref_name", "spawn_ref_weight", "spawn_ref_transform",
    "impact_table_value", "impact_table_biomass_factor",
    "impact_table_energy_factor",
    "inference_initial_biomass", "inference_impact_map",
    "action_costs", "feeding_cost", "movement_cost",
    "initial_biomass_range",
)


@pytest.fixture(scope="module")
def bound_keys():
    """Every key ``attach_help`` is called with while the editor is built."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    seen = []
    original = fgconfig.attach_help

    def spy(widgets, key=None, **kw):
        seen.append(key)
        return original(widgets, key, **kw)

    fgconfig.attach_help = spy
    try:
        app = fgconfig.FGConfigApp(root)
        # The spawn parameter rows only exist for the selected mode.
        for vars_store in (app.spawn_vars, app.ndm_spawn_vars,
                           app.impact_spawn_vars):
            for mode in app.SPAWN_MODES:
                vars_store["mode_var"].set(app.SPAWN_MODE_LABELS[mode])
                app._rebuild_spawn_params(vars_store)
            app._add_ref_row(vars_store, name="phytoplankton")
        # Widgets built on demand: matrices, the Inference tab and the
        # impact-table dialog need a group and an impact in the project.
        app.project_data["decision_makers"] = [
            {"group_id": "zooplankton"}, {"group_id": "pelagic_fish"}]
        app.project_data["non_decision_makers"] = [
            {"group_id": "phytoplankton"}]
        app.project_data["impact_variables"] = [
            {"impact_id": "windfarm_noise"}]
        app.refresh_matrix()
        app.refresh_inference_tab()
        app.open_impact_table_editor(
            "zooplankton_impacted_by_windfarm_noise", "zooplankton",
            "windfarm_noise")
    finally:
        fgconfig.attach_help = original
        root.destroy()
    return seen


def test_every_bound_key_has_text(bound_keys):
    missing = sorted({k for k in bound_keys
                      if k is not None and not field_help.help_text(k)})
    assert not missing, f"help keys without text: {missing}"


def test_fixed_fields_are_bound(bound_keys):
    unbound = sorted(set(FIXED_KEYS) - set(bound_keys))
    assert not unbound, unbound


def test_matrix_cells_are_bound(bound_keys):
    unbound = sorted({f"matrix:{k}" for k in MATRIX_KEYS} - set(bound_keys))
    assert not unbound, unbound


GROWTH_TERMS = ("panel", "dm_formula", "ndm_formula", "growth_rate",
                "r_max", "m1", "m2", "gate", "ticks_per_year")


def test_derived_growth_terms_are_bound(bound_keys):
    """Each term of the Derived growth_rate panel has its own help."""
    unbound = sorted({f"growth:{t}" for t in GROWTH_TERMS} - set(bound_keys))
    assert not unbound, unbound
    for term in GROWTH_TERMS + ("hand_set", "library_note", "error"):
        assert field_help.help_text(f"growth:{term}"), term


def test_every_parameter_has_help():
    keys = list(FIXED_KEYS) + [f"matrix:{k}" for k in MATRIX_KEYS]
    for schema in fgconfig.FGConfigApp.SPAWN_PARAM_SCHEMA.values():
        keys += [f"spawn_{key}" for key, *_ in schema]
    missing = [k for k in keys if not field_help.help_text(k)]
    assert not missing, f"no help text for: {missing}"


def test_fg_editor_rows_have_help(bound_keys):
    """Each DM / NDM editor row is bound to a key with text."""
    source = (ROOT / "fgconfig" / "fgconfig.py").read_text(encoding="utf-8-sig")
    import re
    params = set(re.findall(r'\(\s*"[^"]*",\s*"([a-z_0-9]+)",\s*"(?:entry|range)"',
                            source))
    assert {"max_energy_reserve", "seed_rate", "light_saturation"} <= params
    for key in params:
        ndm = field_help.ndm_help_key(key)
        assert field_help.help_text(key) or field_help.help_text(ndm), key


def test_fg_editor_growth_line_has_term_help():
    """The breakdown under the FG Editor is one helped label per term."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    seen = []
    original = fgconfig.attach_help

    def spy(widgets, key=None, **kw):
        seen.append(key)
        return original(widgets, key, **kw)

    try:
        app = fgconfig.FGConfigApp(root)
        app.project_data["decision_makers"] = [
            {"group_id": "pelagic_fish"}, {"group_id": "gadoids"}]
        species = app.global_library["species_definitions"]
        for gid in ("pelagic_fish", "gadoids"):
            # What load_project_from_path does for every project FG.
            app.current_fg_configs[gid] = species[gid]
        app.update_fg_list()
        fgconfig.attach_help = spy
        app.fg_listbox.selection_set(0)
        app.on_fg_select("decision_makers")
        texts = [w.cget("text") for w in app.growth_info_frame.winfo_children()]
    finally:
        fgconfig.attach_help = original
        root.destroy()
    assert "".join(texts).startswith("g = ")
    for key in ("growth:growth_rate", "growth:r_max", "growth:m1",
                "growth:m2", "growth:gate", "growth:ticks_per_year"):
        assert key in seen, key
