"""The status bar's simulated time, as 'n y x d'.

The tick counter is the engine's unit; how much world time it stands
for is ``project_metadata.tick_hours`` (section 97), which is a project
setting and not always 6 h. The viewer therefore has to be told, and
the arithmetic has to be exact at the year boundary - "0 y 364 d" for
a full year is the failure this is guarding.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.world.tick_time import DAYS_PER_YEAR, DEFAULT_TICK_HOURS

GRID = (3, 3)
FG = "mover"


def viewer(monkeypatch, **kwargs):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=[FG], grid_shape=GRID, mode="train", **kwargs)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    return v


class _SpyFont:
    """Delegating font that records every string it is asked to render."""

    def __init__(self, font, sink):
        self._font = font
        self._sink = sink

    def render(self, text, *a, **kw):
        self._sink.append(text)
        return self._font.render(text, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._font, name)


def at_tick(v, tick):
    v.update_biomass({FG: np.ones(GRID, dtype=np.float32)}, tick=tick)
    return v._sim_time_label()


@pytest.mark.parametrize("tick,expected", [
    (0, "0 y 0 d"),
    (3, "0 y 0 d"),          # 18 h: the part-day is dropped, not rounded up
    (4, "0 y 1 d"),
    (4 * 40, "0 y 40 d"),
    (4 * DAYS_PER_YEAR, "1 y 0 d"),
    (4 * DAYS_PER_YEAR - 1, "0 y 364 d"),
    (4 * (2 * DAYS_PER_YEAR + 40), "2 y 40 d"),
])
def test_six_hour_ticks_read_as_years_and_days(monkeypatch, tick, expected):
    v = viewer(monkeypatch)
    try:
        assert v.tick_hours == DEFAULT_TICK_HOURS
        assert at_tick(v, tick) == expected
    finally:
        v.close()


# The tick length is 1-6 h (section 103), so every case here is inside
# that range; a project carrying anything else resolves to the default
# and is covered by tests/test_tick_time.py.
@pytest.mark.parametrize("hours,tick,expected", [
    (1, 24, "0 y 1 d"),
    (1, 24 * DAYS_PER_YEAR, "1 y 0 d"),
    (2, 12, "0 y 1 d"),
    (3, 8 * DAYS_PER_YEAR, "1 y 0 d"),
    (6, 4, "0 y 1 d"),
    (6, 4 * (DAYS_PER_YEAR + 5), "1 y 5 d"),
])
def test_the_project_tick_length_is_what_converts(monkeypatch, hours, tick, expected):
    """The same tick number is a different amount of time per project."""
    v = viewer(monkeypatch, tick_hours=hours)
    try:
        assert v.tick_hours == hours
        assert at_tick(v, tick) == expected
    finally:
        v.close()


def test_an_absent_tick_length_falls_back_to_six_hours(monkeypatch):
    """manual_play / simple_inference have no project file."""
    v = viewer(monkeypatch, tick_hours=None)
    try:
        assert v.tick_hours == DEFAULT_TICK_HOURS
        assert at_tick(v, 4) == "0 y 1 d"
    finally:
        v.close()


def test_the_status_bar_shows_it_next_to_the_tick(monkeypatch):
    """Simulated time is added to the bar, it does not replace the tick."""
    v = viewer(monkeypatch)
    try:
        tick = 4 * (2 * DAYS_PER_YEAR + 40)
        at_tick(v, tick)
        rendered = []
        original = v._font_big
        # pygame.font.Font.render is read-only, so wrap the font itself.
        v._font_big = _SpyFont(original, rendered)
        try:
            v._render_full()
        finally:
            v._font_big = original
        line = next((t for t in rendered if t.startswith("mode = ")), None)
        assert line is not None, rendered
        assert f"tick = {tick}" in line
        assert "time = 2 y 40 d" in line
        assert line.index("tick =") < line.index("time =")
    finally:
        v.close()


# Every construction site, and where its tick length must come from.
# train.py, inference.py and the GPU viewer can be told one and must
# pass it on; api.py, manual_play.py and simple_inference.py have no
# flag and run at the library's calibration, which the DEFAULT gives
# them - so they are allowed to pass nothing.
ENTRY_POINTS = {
    "train.py": "tick_length",
    "inference.py": "tick_length",
    "lib/gpu/visual.py": "tick_hours",
}
NO_FLAG = {"api.py", "manual_play.py", "simple_inference.py"}


def construction_sites():
    """Every module that builds a LiveVisualizer, found not listed.

    The first version of this test carried a hand-written list of entry
    points, and lib/gpu/visual.py was not on it - so train_gpu.py built
    its viewer with no tick length at all and rendered every run at 6 h
    while the engine ran at whatever --tick-length said. Discovering the
    call sites is the point: a new one cannot be forgotten.
    """
    root = Path(__file__).resolve().parents[1]
    found = {}
    for path in list(root.glob("*.py")) + list((root / "lib").rglob("*.py")):
        if path.name == "pygame_viz.py":
            continue  # its own docstring examples
        text = path.read_text(encoding="utf-8")
        if "LiveVisualizer(" not in text:
            continue
        name = str(path.relative_to(root))
        calls = [n for n in ast.walk(ast.parse(text))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "LiveVisualizer"]
        if calls:
            found[name] = calls
    return found


def test_every_construction_site_is_accounted_for():
    sites = set(construction_sites())
    assert sites == set(ENTRY_POINTS) | NO_FLAG, (
        "a LiveVisualizer is built somewhere this test does not classify: "
        f"{sorted(sites ^ (set(ENTRY_POINTS) | NO_FLAG))}")


@pytest.mark.parametrize("name,expected", sorted(ENTRY_POINTS.items()))
def test_the_entry_points_pass_the_right_tick_length(name, expected):
    """Passing the argument is not enough - the VALUE has to be the flag.

    The first version of this test only checked that ``tick_hours``
    appeared as a keyword, which a call passing a constant, a stale
    accessor or the wrong variable would have satisfied just as well.
    A viewer handed the wrong number renders a plausible, wrong
    simulated time and nothing else goes wrong, so nothing else would
    have caught it.
    """
    calls = construction_sites().get(name)
    assert calls, f"{name} no longer constructs a LiveVisualizer"
    for call in calls:
        passed = {kw.arg: kw.value for kw in call.keywords}
        assert "tick_hours" in passed, f"{name} does not pass tick_hours"
        expression = ast.unparse(passed["tick_hours"])
        assert expected in expression, (
            f"{name} passes tick_hours={expression!r}, which does not come "
            f"from {expected}")
