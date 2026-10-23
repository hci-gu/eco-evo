"""Plot y-axis labels in plain decimals (section 196).

No scientific notation (``.3g`` printed 1.23e+04 from 10000 up) and no
plus sign; one number of decimals per axis. A long label widens the
left margin instead of running into the plot.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.viz.pygame_viz import _axis_labels


def axis(lo, hi):
    vals = [hi - k * (hi - lo) / 4 for k in range(5)]
    return _axis_labels(vals, (hi - lo) / 4)


@pytest.mark.parametrize("lo, hi, expected", [
    (0, 60000, ["60000", "45000", "30000", "15000", "0"]),
    (0, 1, ["1.00", "0.75", "0.50", "0.25", "0.00"]),
    (-50.7, 29.8, ["29.8", "9.7", "-10.4", "-30.6", "-50.7"]),
    (0, 0.0012, ["0.0012", "0.0009", "0.0006", "0.0003", "0.0000"]),
    (-1, 1, ["1.0", "0.5", "0.0", "-0.5", "-1.0"]),
])
def test_axis_labels(lo, hi, expected):
    assert axis(lo, hi) == expected


@pytest.mark.parametrize("lo, hi", [
    (0, 1e7), (1234.5, 123456.7), (-3.2e-6, 1e-6), (-2e5, 3e5), (5, 5.5),
])
def test_axis_labels_never_scientific_or_signed_plus(lo, hi):
    for txt in axis(lo, hi):
        assert "e" not in txt.lower() and not txt.startswith("+")
        float(txt)
    assert len(set(axis(lo, hi))) == 5


def test_flat_axis_and_negative_zero():
    assert _axis_labels([-0.0004, 0.0], 0.0) == ["0", "0"]


def test_plot_draws_plain_labels_and_widens_its_margin(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=["mover"], grid_shape=(3, 3), mode="infer")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    try:
        tab = v._tabs.index("predation")
        v._active_tab = tab
        for t in range(50):
            v.update_series("predation", "mover", 100000.0 + 5000.0 * t,
                            step=t)
        texts = []
        font = v._font

        class Spy:
            def render(self, text, *a, **kw):
                texts.append(text)
                return font.render(text, *a, **kw)

            def __getattr__(self, name):
                return getattr(font, name)

        v._font = Spy()
        v._render_full()
        v._font = font
        labels = ["345000", "283750", "222500", "161250", "100000"]
        i = texts.index(labels[0])
        assert texts[i:i + 5] == labels
        # "345000" is wider than the 36 px default margin.
        assert v._plot_area_rect[0] - v._plot_rect[0] > 36
    finally:
        v.close()


def tooltip_texts(v, ymin, ymax):
    texts = []
    font = v._font

    class Spy:
        def render(self, text, *a, **kw):
            texts.append(text)
            return font.render(text, *a, **kw)

        def __getattr__(self, name):
            return getattr(font, name)

    v._font = Spy()
    try:
        tab = v._tabs[v._active_tab]
        px, py, pw, ph = v._plot_area_rect
        v._draw_plot_tooltip(
            px + pw // 2, py + ph // 2, active_ids=v._active_plot_ids(),
            buffers=v._series[tab], xmin=0.0, xmax=49.0,
            ymin=ymin, ymax=ymax, px0=px, py0=py, pw=pw, ph=ph,
            plot_rect=v._plot_rect)
    finally:
        v._font = font
    return [t for t in texts if ": " in t]


@pytest.mark.parametrize("log_plot", [False, True])
def test_tooltip_values_are_plain_decimals(monkeypatch, log_plot):
    """The hover tooltip formats its values like the axis: no ".3g"
    scientific notation, no plus sign (section 196)."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=["mover", "other"], grid_shape=(3, 3),
                       mode="infer")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    try:
        v._active_tab = v._tabs.index("predation")
        v._log_plot = log_plot
        for t in range(50):
            v.update_series("predation", "mover", 123456.0, step=t)
            v.update_series("predation", "other", 54321.5, step=t)
        v._render_full()
        lines = tooltip_texts(v, 50000.0, 130000.0)
        values = [line.split(": ")[1] for line in lines]
        assert len(values) == 2
        for txt in values:
            assert "e" not in txt.lower() and not txt.startswith("+")
        assert values == ["123456", "54322"]
    finally:
        v.close()


def test_tooltip_small_values_keep_their_digits(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=["mover"], grid_shape=(3, 3), mode="infer")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    try:
        v._active_tab = v._tabs.index("predation")
        for t in range(50):
            v.update_series("predation", "mover", -0.00042, step=t)
        v._render_full()
        lines = tooltip_texts(v, -0.0012, 0.0)
        assert [line.split(": ")[1] for line in lines] == ["-0.00042"]
    finally:
        v.close()
