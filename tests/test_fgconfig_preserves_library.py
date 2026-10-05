"""The FG editor must not strip comments or reformat the library.

Saving a group used to replace its ruamel mapping with a plain dict,
which dropped every comment in the group's block (the literature notes
on porpoises were lost that way), rewrote ``0.10`` as ``0.1`` and added
optional fields as 0. ``_merge_species_entry`` now writes in place.
"""
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgconfig"))

ruamel_yaml = pytest.importorskip("ruamel.yaml")
fgconfig = pytest.importorskip("fgconfig")
App = fgconfig.FGConfigApp

LIBRARY = """\
species_definitions:
  porpoises:
    is_decision_maker: true
    # r_max 0.10 /yr (Lockyer 2003).
    r_max: 0.10
    # M1 from Leslie models.
    natural_mortality: 6.85e-05
    max_intake_rate: 0.035
"""


def _merge(text, config, is_dm=True):
    yaml = ruamel_yaml.YAML()
    doc = yaml.load(text)
    target = doc["species_definitions"]["porpoises"]
    App._merge_species_entry(App, target, config, set(target.keys()), is_dm)
    out = io.StringIO()
    yaml.dump(doc, out)
    return out.getvalue()


def test_unchanged_values_leave_the_text_identical():
    config = {"is_decision_maker": True, "r_max": 0.1,
              "natural_mortality": 6.85e-05, "max_intake_rate": 0.035,
              # empty optional editor fields come back as 0.0
              "m1_visual_share": 0.0, "m1_tactile_share": 0.0,
              "depth_risk_ratio": 0.0, "hide_reference": 0.0,
              "current_response": 0.0}
    assert _merge(LIBRARY, config) == LIBRARY


def test_an_edit_changes_only_its_value_and_keeps_the_comments():
    config = {"is_decision_maker": True, "r_max": 0.1,
              "natural_mortality": 7e-05, "max_intake_rate": 0.035}
    out = _merge(LIBRARY, config)
    assert "# r_max 0.10 /yr (Lockyer 2003)." in out
    assert "# M1 from Leslie models." in out
    assert "r_max: 0.10" in out
    assert "natural_mortality: 7e-05" in out


def test_a_set_optional_field_is_written():
    out = _merge(LIBRARY, {"m1_tactile_share": 0.5})
    assert "m1_tactile_share: 0.5" in out
