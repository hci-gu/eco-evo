"""tools/make_fishing_maps.py and the fishing tables (section 161).

The projection must put the grid where coordinates.txt says it is, the
rasterisation must keep the effort of a c-square (density x area), and
the library's fishing tables must be linear from zero, so an effort map
in MW-h/km2/year turns into a loss per tick proportional to the effort.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from lib.config.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_fishing_maps", ROOT / "tools" / "make_fishing_maps.py")
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)


@pytest.mark.parametrize("lat,lon,east,north", [
    (58.311573, 10.065092, 211000.0, 6474000.0),     # grid NW corner
    (57.809276, 11.145026, 271000.0, 6414000.0),     # grid SE corner
    (58.099880, 10.809982, 253118.60, 6447499.29)])  # Poseidon Nord point 1
def test_projection_matches_coordinates_txt(lat, lon, east, north):
    e, n = fm.wgs84_to_sweref99tm(np.array(lon), np.array(lat))
    assert float(e) == pytest.approx(east, abs=2.0)
    assert float(n) == pytest.approx(north, abs=2.0)


def test_rasterisation_keeps_a_csquares_effort():
    lat_min, lon_min, density = 58.0, 10.5, 3.0      # well inside the grid
    field = fm.rasterise([(lat_min, lon_min, density)])
    area = (fm.CSQ * 111.32) ** 2 * math.cos(math.radians(lat_min + fm.CSQ / 2))
    assert field.sum() == pytest.approx(density * area, rel=1e-3)
    assert field.max() == pytest.approx(density, rel=0.03), "interior cells carry the density"


def test_build_scales_the_mean_pattern_to_the_latest_window():
    def feature(window, value):
        ring = [[10.5, 58.0], [10.55, 58.0], [10.55, 58.05], [10.5, 58.05], [10.5, 58.0]]
        return {"properties": {"avgyears": window, "mw_fshn": value},
                "geometry": {"type": "MultiPolygon", "coordinates": [[ring]]}}
    today, maps, windows = fm.build([feature("2015-2018", 100.0), feature("2021-2024", 40.0)])
    assert windows == ["2015-2018", "2021-2024"]
    assert today.mean() == pytest.approx(maps["2021-2024"].mean(), rel=1e-5)


@pytest.mark.parametrize("key", ["benthivorous_fish_impacted_by_bottom_trawling",
                                 "benthivorous_fish_impacted_by_pelagic_trawling",
                                 "planktivorous_fish_impacted_by_pelagic_trawling",
                                 "planktivorous_fish_impacted_by_bottom_trawling",
                                 "benthic_infauna_impacted_by_bottom_trawling",
                                 "porpoises_impacted_by_bottom_trawling"])
def test_fishing_tables_are_linear_from_zero(key):
    table = load_config(str(ROOT / "fgconfig" / "fg_library.yaml"))["interaction_definitions"][key]
    assert table["impact_affects"]
    rows = table["impact_table"]
    assert rows[0]["value"] == 0 and rows[0]["biomass_factor"] == 0
    assert len(rows) == 2 and all(r["energy_factor"] == 0 for r in rows)
    # The catchability is small: even at the top of the table well under
    # 3 % of the cell per tick.
    assert 0 < rows[1]["biomass_factor"] < 0.03


@pytest.mark.parametrize("fg,total", [("benthivorous_fish", 0.1704),
                                      ("predatory_fish", 0.2597973656196191),
                                      ("mackerel", 0.012313058166029952 + 0.0023648464512033376),
                                      ("whiting", 0.075 + 0.005),
                                      ("planktivorous_fish", 0.10), ("porpoises", 0.017)])
def test_gear_split_keeps_the_total_f(fg, total):
    """Split over the two maps; at today's mean effort the two tables add
    up to the group's F (catch_by_gear.md; section 209: benthivores 0.17
    + 0.0004, predators 0.162 + 0.020; section 223.26 split the predators:
    water column 0.213 + 0.029, whiting 0.075 + 0.005; section 228 split
    the water column: predatory_fish 0.260 + 0, mackerel 0.012 + 0.002 at the field
    member mix: mackerel at its LOCAL 3.a F 0.013, 228.1).
    Porpoise bycatch
    sits on the bottom-trawl map alone."""
    lib = load_config(str(ROOT / "fgconfig" / "fg_library.yaml"))["interaction_definitions"]
    means = {"bottom_trawling": 3.0141578935527487, "pelagic_trawling": 0.02496212180679044}
    f = 0.0
    for impact, mean in means.items():
        rows = lib[f"{fg}_impacted_by_{impact}"]["impact_table"]
        if not rows:
            continue
        top = rows[-1]
        f += top["biomass_factor"] / top["value"] * mean * 1460
    assert f == pytest.approx(total, rel=1e-6)
