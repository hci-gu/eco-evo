"""Section 271: the light-entrained reserve set point (lib/world/set_point.py).

The clock reads only the day length and the sign of its change; the
growth budget's s* is capped at the annual mean target; the library's
planktivore block parses and the FG editor round-trips it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.world import daylight, set_point  # noqa: E402
from lib.world.growth_budget import growth_budget  # noqa: E402

LAT = 58.15
BLOCK = {"member_share": 0.554, "depth": 0.5, "onset_day_length_h": 13.64,
         "low_day_length_h": 14.11, "top_day_length_h": 17.44}


def spec(**extra):
    return {"appetite": "reserve_room", set_point.KEY: dict(BLOCK, **extra)}


def test_clock_shape():
    g = set_point.progress_by_day(LAT, 13.64, 14.11, 17.44)
    hours = daylight.day_length_hours(LAT)
    ws, ss = int(np.argmin(hours)), int(np.argmax(hours))
    # Continuous at both solstices (one day's step at most).
    assert abs(g[ws] - g[ws - 1]) < 0.01 and abs(g[ws + 1] - g[ws]) < 0.01
    assert g[ss - 1] == g[ss] == g[ss + 1] == 0.0
    # Low (g = 1) on rising days at 14.11 h, i.e. mid-April.
    low = int(np.argmax(g))
    assert g[low] == pytest.approx(1.0, abs=0.01) and 95 < low < 115
    # Falls monotonically from the onset (early September) to the low.
    onset = int(np.flatnonzero((np.roll(hours, -1) < hours) & (hours < 13.64))[0])
    assert 235 < onset < 250
    path = np.concatenate([g[onset:], g[:low + 1]])
    assert np.all(np.diff(path) >= -1e-12)


def test_group_target_and_mean():
    by_day = set_point.group_target_by_day(set_point.settings(spec()), LAT)
    assert by_day.max() == 1.0
    assert by_day.min() == pytest.approx(1.0 - 0.554 * 0.5, abs=0.005)
    assert set_point.mean_target(spec(), LAT) == pytest.approx(by_day.mean())
    assert set_point.mean_target(spec(), None) is None
    assert set_point.mean_target({"appetite": "reserve_room"}, LAT) is None
    table = set_point.tick_table(set_point.settings(spec()), LAT, 6)
    assert table.shape == (1460,) and table.mean() == pytest.approx(by_day.mean(), abs=1e-3)


@pytest.mark.parametrize("bad", [
    {"appetite": "gate", set_point.KEY: dict(BLOCK)},
    spec(depth=1.0), spec(member_share=0.0),
    spec(low_day_length_h=18.0), spec(depth="x"),
])
def test_malformed_block_is_refused(bad):
    with pytest.raises(ValueError):
        cfg = set_point.settings(bad)
        set_point.group_target_by_day(cfg, LAT)


def test_day_lengths_outside_the_site_are_refused():
    cfg = set_point.settings(spec(top_day_length_h=19.0))
    with pytest.raises(ValueError):
        set_point.group_target_by_day(cfg, LAT)


def test_library_block_and_growth_budget():
    """The planktivores' block parses; s* drops from 1 to the annual mean
    target with the site's latitude and stays 1 without it."""
    lib = yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml"))
    species, inter = lib["species_definitions"], lib["interaction_definitions"]
    plank = species["planktivorous_fish"]
    assert set_point.settings(plank)["member_share"] == pytest.approx(0.554)
    mean = set_point.mean_target(plank, LAT)
    assert mean == pytest.approx(0.898, abs=0.002)
    active = [f for f, s in species.items() if isinstance(s, dict)]
    site = growth_budget("planktivorous_fish", species, inter, active, latitude_deg=LAT)
    none = growth_budget("planktivorous_fish", species, inter, active)
    assert not site.error and not none.error
    assert none.reference_fill == pytest.approx(1.0)
    # Form 2: P is paid inside the growth window (target not falling).
    cfg = set_point.settings(plank)
    target = set_point.group_target_by_day(cfg, LAT)
    window = set_point.growth_window_by_day(cfg, LAT)
    u = float(plank["maintenance_level"])
    assert site.window_share == pytest.approx(window.mean())
    assert 0.36 < site.window_share < 0.40            # 15 Apr - 30 Aug
    assert site.reference_fill == pytest.approx(target[window > 0].mean())
    gate = float((window * (target - u)).mean())
    assert site.growth_rate / none.growth_rate == pytest.approx((1 - u) / gate, rel=1e-6)


def test_growth_window():
    """The window is the days the target is not falling: open from the low
    (mid-April) through the summer plateau, closed from the onset."""
    cfg = set_point.settings(spec())
    window = set_point.growth_window_by_day(cfg, LAT)
    target = set_point.group_target_by_day(cfg, LAT)
    assert set(np.unique(window)) == {0.0, 1.0}
    falling = np.roll(target, -1) < target - 1e-12
    np.testing.assert_array_equal(window == 0.0, falling)
    assert window[200] == 1.0 and window[300] == 0.0 and window[30] == 0.0
    table = set_point.growth_window_table(cfg, LAT, 6)
    assert table.shape == (1460,) and table.mean() == pytest.approx(window.mean(), abs=2e-3)


def test_fg_editor_round_trip():
    pytest.importorskip("ruamel.yaml")
    sys.path.insert(0, str(ROOT / "fgconfig"))
    fgconfig = pytest.importorskip("fgconfig")
    App = fgconfig.FGConfigApp
    rows = {"set_point_depth": "0.5", "set_point_member_share": "0.554",
            "set_point_onset_day_length_h": "13.64", "set_point_low_day_length_h": "14.11",
            "set_point_top_day_length_h": "17.44", "set_point_observed": "1"}
    block, error = App._collect_set_point_block(rows, "reserve_room")
    assert error is None and block == BLOCK
    block, error = App._collect_set_point_block(dict(rows, set_point_observed="0"), "reserve_room")
    assert error is None and block["observed"] is False
    assert App._collect_set_point_block(dict(rows, set_point_depth=""), "reserve_room") == (None, None)
    _, error = App._collect_set_point_block(rows, "gate")
    assert error and "reserve_room" in error
