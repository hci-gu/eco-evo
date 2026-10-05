"""The live viewer shows the daylight calendar (section 142).

With ``simulation_settings.daylight`` on, every rollout frame carries
the month and the day length (status bar), the latter also drawn as a
thin grey frame around each heatmap: black on the year's shortest day,
white on its longest, linear in sun hours between, constant within a
day.
Both come from the ``params['daylight']`` the loader puts on the FGs,
and both follow the replayed frame during playback. Without the
calendar the viewer is unchanged.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.world import daylight
from lib.world.tick_time import DAYS_PER_YEAR, HOURS_PER_DAY

GRID = (3, 3)
FG = "mover"
LATITUDE = 58.15


def viewer(monkeypatch, **kwargs):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=[FG], grid_shape=GRID, mode="infer", **kwargs)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    return v


def config(start_day, tick_hours=6):
    return {"latitude_deg": LATITUDE, "tick_hours": tick_hours,
            "start_tick": daylight.start_tick(start_day, tick_hours),
            "random_start": False, "light_climate": None}


def fgs(cfg):
    return {FG: SimpleNamespace(biomass=np.ones(GRID, dtype=np.float32),
                                params={"daylight": cfg})}


class _SpyFont:
    def __init__(self, font, sink):
        self._font = font
        self._sink = sink

    def render(self, text, *a, **kw):
        self._sink.append(text)
        return self._font.render(text, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._font, name)


def status_line(v):
    rendered = []
    original = v._font_big
    v._font_big = _SpyFont(original, rendered)
    try:
        v._render_full()
    finally:
        v._font_big = original
    return next(t for t in rendered if t.startswith("mode = "))


def frame_pixel(v):
    """A pixel on the daylight frame, left of the heatmap's top row."""
    from lib.viz.pygame_viz import _LIGHT_FRAME_PX
    ox, oy = v._heatmap_origin
    hm_x, hm_y = ox + 4, oy + 104
    return tuple(v._screen.get_at((hm_x - _LIGHT_FRAME_PX, hm_y + 5)))[:3]


# ---- the calendar itself ---------------------------------------------

@pytest.mark.parametrize("day0,month", [
    (0, 0), (30, 0), (31, 1), (58, 1), (59, 2), (104, 3), (364, 11),
])
def test_month_of_day_boundaries(day0, month):
    assert daylight.month_of_day(day0) == month


@pytest.mark.parametrize("hours", [1, 2, 3, 4, 6])
def test_calendar_follows_the_start_day_and_the_tick_length(hours):
    cfg = config(start_day=31, tick_hours=hours)          # 31 Jan
    per_day = HOURS_PER_DAY // hours
    assert daylight.calendar_at(cfg, 0)["month_name"] == "Jan"
    assert daylight.calendar_at(cfg, per_day)["month_name"] == "Feb"
    # A whole year later it is January again: the calendar wraps.
    year = per_day * DAYS_PER_YEAR
    assert daylight.calendar_at(cfg, year)["day_of_year"] == 31


def test_calendar_light_is_the_observation_channel():
    cfg = config(start_day=172)                           # midsummer
    schedule = daylight.light_schedule(LATITUDE, 6)
    for tick in range(8):
        at = daylight.calendar_at(cfg, tick)
        assert at["light"] == pytest.approx(
            float(schedule[(cfg["start_tick"] + tick) % len(schedule)]))
    # 00-06 at midsummer, 58 N: twilight; 12-18: full light.
    assert daylight.calendar_at(cfg, 2)["light"] > 0.99


def test_day_length_spans_the_year_shortest_to_longest():
    hours = daylight.day_length_hours(LATITUDE)
    assert hours.shape == (DAYS_PER_YEAR,)
    # 58 N: about 6 h at midwinter, about 18 h at midsummer.
    assert 5.5 < hours.min() < 7.0 and 17.5 < hours.max() < 19.0
    assert abs(int(np.argmin(hours)) - 354) <= 2     # ~21 Dec
    assert abs(int(np.argmax(hours)) - 171) <= 2     # ~21 Jun
    fracs = [daylight.day_length_fraction(LATITUDE, d)
             for d in range(DAYS_PER_YEAR)]
    assert min(fracs) == 0.0 and max(fracs) == 1.0
    # Equinox: half way in hours, so half way on the scale.
    assert daylight.day_length_fraction(LATITUDE, 79) == pytest.approx(
        0.5, abs=0.03)


def test_day_length_is_constant_within_a_day_and_steps_at_midnight():
    cfg = config(start_day=60)                            # 1 Mar
    days = [daylight.calendar_at(cfg, t) for t in range(8)]
    assert len({d["day_length_frac"] for d in days[:4]}) == 1
    assert len({d["day_length_frac"] for d in days[4:]}) == 1
    # Lengthening in March.
    assert days[4]["day_length_h"] > days[0]["day_length_h"]


def test_day_length_matches_the_sun():
    """The analytic sunrise equation agrees with the elevation model."""
    day0 = 120
    t = day0 * HOURS_PER_DAY + (np.arange(24 * 60) + 0.5) / 60.0
    above = float((daylight.solar_elevation_deg(LATITUDE, t) > 0.0).sum()) / 60
    assert daylight.day_length_hours(LATITUDE)[day0] == pytest.approx(
        above, abs=0.1)


def test_polar_and_equatorial_days():
    polar = daylight.day_length_hours(80.0)
    assert polar.min() == 0.0 and polar.max() == 24.0
    assert daylight.day_length_fraction(0.0, 10) == 0.5


# ---- the viewer ----------------------------------------------------------

def test_status_bar_shows_the_month_and_day_length(monkeypatch):
    v = viewer(monkeypatch)
    try:
        v.update_biomass(fgs(config(start_day=105)), tick=2)   # 15 Apr, noon
        line = status_line(v)
        assert "month = Apr" in line
        hours = daylight.day_length_hours(LATITUDE)[104]
        assert f"daylight = {hours:.1f} h" in line
        assert line.index("time =") < line.index("month =")
    finally:
        v.close()


def _extreme_days():
    hours = daylight.day_length_hours(LATITUDE)
    return int(np.argmin(hours)) + 1, int(np.argmax(hours)) + 1


# Shortest day black, longest white, whatever the time of day: noon on
# the shortest day is still black, midnight on the longest still white.
@pytest.mark.parametrize("which,tick,expected", [
    ("short", 0, 0), ("short", 2, 0), ("long", 0, 255), ("long", 3, 255),
])
def test_heatmap_frame_is_black_on_the_shortest_day_white_on_the_longest(
        monkeypatch, which, tick, expected):
    shortest, longest = _extreme_days()
    start_day = shortest if which == "short" else longest
    v = viewer(monkeypatch)
    try:
        v.update_biomass(fgs(config(start_day=start_day)), tick=tick)
        v._render_full()
        assert frame_pixel(v) == (expected, expected, expected)
    finally:
        v.close()


def test_heatmap_frame_is_grey_in_between(monkeypatch):
    v = viewer(monkeypatch)
    try:
        v.update_biomass(fgs(config(start_day=80)), tick=0)    # equinox
        v._render_full()
        grey = round(255 * v._calendar["day_length_frac"])
        assert frame_pixel(v) == (grey, grey, grey)
        assert 110 < grey < 145
    finally:
        v.close()


def test_without_the_calendar_nothing_changes(monkeypatch):
    v = viewer(monkeypatch)
    try:
        v.update_biomass({FG: np.ones(GRID, dtype=np.float32)}, tick=2)
        assert v._calendar is None
        line = status_line(v)
        assert "month" not in line and "daylight" not in line
        # The panel background, not a frame.
        assert frame_pixel(v) == (28, 28, 34)
    finally:
        v.close()


def test_replay_shows_the_frames_calendar(monkeypatch):
    v = viewer(monkeypatch)
    try:
        cfg = config(start_day=31)                       # 31 Jan, midnight
        v.begin_rollout_recording()
        for tick in range(1, 9):                         # into 2 Feb
            v.update_biomass(fgs(cfg), tick=tick)
        v.end_rollout_recording()
        assert v._calendar["month_name"] == "Feb"
        v._playback_mode = "paused"
        v._playback_idx = 1                              # tick 2: 31 Jan noon
        line = status_line(v)
        assert "month = Jan" in line
        grey = round(255 * daylight.calendar_at(cfg, 2)["day_length_frac"])
        assert grey != round(255 * v._calendar["day_length_frac"])
        assert frame_pixel(v) == (grey, grey, grey)
        # The live calendar is restored after drawing the frame.
        assert v._calendar["month_name"] == "Feb"
    finally:
        v.close()
