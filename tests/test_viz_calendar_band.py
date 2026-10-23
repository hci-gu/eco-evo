"""The inference plot's calendar band (section 155).

With the daylight calendar on, the inference plot gets two rows under
its x range: the months as profiler-style blocks, and under them the
clock time, cut at the finest clock step that leaves room for a label
and shaded by the light level of its ticks. Plot x is the series step;
the point at step ``s`` is the state at the start of tick ``s + 1``,
which is what the status bar describes when it shows ``tick = s + 1``.
Train mode and projects without the calendar keep the old layout.
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


def viewer(monkeypatch, mode="infer"):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=[FG], grid_shape=GRID, mode=mode)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    return v


def config(start_day, tick_hours=6):
    return {"latitude_deg": LATITUDE, "tick_hours": tick_hours,
            "start_tick": daylight.start_tick(start_day, tick_hours),
            "random_start": False, "light_climate": None}


def fgs(cfg):
    params = {"daylight": cfg} if cfg else {}
    return {FG: SimpleNamespace(biomass=np.ones(GRID, dtype=np.float32),
                                params=params)}


def feed(v, cfg, n_ticks, window=None):
    """An inference rollout of ``n_ticks`` as inference.py feeds it."""
    v.begin_rollout_recording()
    for t in range(n_ticks):
        v.update_series("biomass", FG, 100.0 + np.sin(t / 5.0), step=t)
    if window is not None:
        v._plot_window_width = window
    v.update_biomass(fgs(cfg), tick=n_ticks)


def rendered_texts(v):
    class Spy:
        def __init__(self, font, sink):
            self._font, self._sink = font, sink

        def render(self, text, *a, **kw):
            self._sink.append(text)
            return self._font.render(text, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._font, name)

    out = []
    original = v._font
    v._font = Spy(original, out)
    try:
        v._render_full()          # raises, unlike the guarded live path
    finally:
        v._font = original
    return out


# ---- the x -> time mapping -------------------------------------------

@pytest.mark.parametrize("hours", [1, 2, 3, 4, 5, 6])
def test_plot_x_is_the_instant_the_status_bar_shows(hours):
    from lib.viz.pygame_viz import _date_parts, _plot_x_hours
    cfg = config(start_day=360, tick_hours=hours)       # crosses New Year
    for s in range(0, 40 * HOURS_PER_DAY // hours, 3):
        month, dom, hour = _date_parts(float(_plot_x_hours(cfg, s)))
        at = daylight.calendar_at(cfg, s + 1)
        assert month == at["month"]
        start = (at["year_tick"] * hours) % (DAYS_PER_YEAR * HOURS_PER_DAY)
        assert hour == start % HOURS_PER_DAY
        assert dom == at["day_of_year"] - sum(daylight.MONTH_DAYS[:month])


def test_calendar_label():
    from lib.viz.pygame_viz import _calendar_label
    assert _calendar_label(0) == "Jan 1 00:00"
    assert _calendar_label((31 + 13) * 24 + 6) == "Feb 14 06:00"
    year = DAYS_PER_YEAR * HOURS_PER_DAY
    assert _calendar_label(year + 18) == "Jan 1 18:00"


def test_month_blocks_cover_the_window_and_wrap():
    from lib.viz.pygame_viz import _month_blocks
    year = DAYS_PER_YEAR * HOURS_PER_DAY
    blocks = _month_blocks(year - 48, year + 48)
    assert [m for _s, _e, m in blocks] == [11, 0]
    assert blocks[0][1] == blocks[1][0] == year


def test_column_light_averages_the_ticks_exactly():
    from lib.viz.pygame_viz import _column_light
    cfg = config(start_day=1)
    sched = daylight.light_schedule(LATITUDE, 6)
    # One tick per column: the observation channel itself.
    edges = np.arange(0, 6 * 9, 6, dtype=np.float64)
    assert np.allclose(_column_light(cfg, edges), sched[:8])
    # A column over a whole year (and past it): the annual mean.
    year = DAYS_PER_YEAR * HOURS_PER_DAY
    whole = _column_light(cfg, np.array([12.0, 12.0 + year]))
    assert whole[0] == pytest.approx(float(sched.mean()))


def test_clock_step_is_never_finer_than_a_tick():
    from lib.viz.pygame_viz import _clock_step_hours
    assert _clock_step_hours(6, 100.0) == 6
    assert _clock_step_hours(1, 100.0) == 1
    assert _clock_step_hours(1, 5.0) == 6         # 1-5 h blocks too narrow
    assert _clock_step_hours(6, 0.1) is None      # a day under 18 px


# ---- the viewer ----------------------------------------------------------

def test_band_labels_months_across_a_boundary(monkeypatch):
    v = viewer(monkeypatch)
    try:
        feed(v, config(start_day=20), n_ticks=200, window=10 ** 6)
        texts = rendered_texts(v)
        # 20 Jan + 50 days: January, February and part of March.
        assert {"January", "February", "March"} <= set(texts)
    finally:
        v.close()


def test_band_shows_clock_times_when_zoomed_in(monkeypatch):
    v = viewer(monkeypatch)
    try:
        feed(v, config(start_day=105), n_ticks=200, window=12)
        texts = rendered_texts(v)
        assert "06:00" in texts and "18:00" in texts
        # The rightmost block is the live edge's: tick 200 of a run
        # that started at 15 Apr 00:00 is 15 Apr + 50 d = 4 Jun 00:00.
        assert "June" in texts
    finally:
        v.close()


def test_band_shows_days_of_the_month_at_the_default_zoom(monkeypatch):
    v = viewer(monkeypatch)
    try:
        feed(v, config(start_day=105), n_ticks=200, window=100)
        texts = rendered_texts(v)
        assert "06:00" not in texts
        # 100 ticks of 6 h = 25 days, 10 May to 4 June 00:00.
        assert "3" in texts and "28" in texts
    finally:
        v.close()


def _plot_area(monkeypatch, mode, cfg):
    v = viewer(monkeypatch, mode=mode)
    try:
        feed(v, cfg, n_ticks=50)
        v._render_full()
        return v._plot_area_rect
    finally:
        v.close()


def test_band_reserves_room_only_with_the_calendar(monkeypatch):
    from lib.viz.pygame_viz import _CAL_BAND_H
    plain = _plot_area(monkeypatch, "infer", None)
    banded = _plot_area(monkeypatch, "infer", config(start_day=105))
    assert banded[:3] == plain[:3]
    assert banded[3] == plain[3] - _CAL_BAND_H


def test_train_mode_has_no_band(monkeypatch):
    plain = _plot_area(monkeypatch, "train", None)
    with_calendar = _plot_area(monkeypatch, "train", config(start_day=105))
    assert with_calendar == plain


@pytest.mark.parametrize("window", [10 ** 6, 400, 12])
def test_every_rendered_text_is_blitted(monkeypatch, window):
    """An antialiased ``font.render`` surface that is never blitted
    leaks its pixel buffer (sections 60 and 189): zoomed out over a
    50-year rollout the band rendered three label variants for every
    month block and blitted none, ~0.25 MB per frame. Widths are
    measured with ``font.size``; only the label drawn is rendered."""
    import pygame
    v = viewer(monkeypatch)
    try:
        feed(v, config(start_day=105), n_ticks=50 * 1460, window=window)

        rendered, blitted = [], set()

        class Screen(pygame.Surface):
            def blit(self, source, *a, **kw):
                blitted.add(id(source))
                return super().blit(source, *a, **kw)

        class Font:
            def __init__(self, font):
                self._font = font

            def render(self, *a, **kw):
                surf = self._font.render(*a, **kw)
                rendered.append((a[0], surf))
                return surf

            def __getattr__(self, name):
                return getattr(self._font, name)

        screen = Screen(v._screen.get_size())
        v._screen, v._font = screen, Font(v._font)
        v._render_full()
        unblitted = [text for text, surf in rendered
                     if id(surf) not in blitted]
        assert rendered and not unblitted
    finally:
        v.close()
