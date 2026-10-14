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


@pytest.mark.parametrize("key", ["gadoids_impacted_by_bottom_trawling",
                                 "pelagic_fish_impacted_by_pelagic_trawling"])
def test_fishing_tables_are_linear_from_zero(key):
    table = load_config(str(ROOT / "fgconfig" / "fg_library.yaml"))["interaction_definitions"][key]
    assert table["impact_affects"]
    rows = table["impact_table"]
    assert rows[0]["value"] == 0 and rows[0]["biomass_factor"] == 0
    assert len(rows) == 2 and all(r["energy_factor"] == 0 for r in rows)
    # The catchability is small: even at the top of the table well under
    # 3 % of the cell per tick.
    assert 0 < rows[1]["biomass_factor"] < 0.03
