"""The viewer's NDM panels show where their biomass goes: 'pr/im = x/y%'.

A non decision maker has no actions, diet, M1 or starvation; it loses
biomass only to predators and impacts (bottom trawling on the benthos,
section 161). Its third header line is that split; a DM keeps its
'pr/st/im/nm' line.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = (4, 4)


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    viewer = LiveVisualizer(fg_ids=["benthivorous_fish", "benthic_community"], grid_shape=GRID,
                            mode="inference", ndm_ids=["benthic_community"])
    if not viewer.enabled:
        pytest.skip("pygame display unavailable")
    yield viewer
    viewer.close()


class _Recorder:
    """Font proxy that records every rendered string."""
    def __init__(self, font, lines):
        self._font, self._lines = font, lines

    def render(self, text, *args, **kwargs):
        self._lines.append(text)
        return self._font.render(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._font, name)


def rendered_lines(viz, fid, monkeypatch):
    lines = []
    font = viz._font
    viz._font = _Recorder(font, lines)
    try:
        viz._draw_one_heatmap(fid, 0, 0)
    finally:
        viz._font = font
    return lines


def test_ndm_panel_shows_predation_and_impact_shares(viz, monkeypatch):
    viz.update_loss_breakdown({
        "benthic_community": {"predation": 0.27, "impact": 0.73,
                              "starvation": 0.0, "natural": 0.0},
        "benthivorous_fish": {"predation": 0.1, "impact": 0.2, "starvation": 0.0,
                    "natural": 0.7}})
    lines = rendered_lines(viz, "benthic_community", monkeypatch)
    assert "pr/im = 27/73%" in lines
    assert not any(line.startswith(("mv/rs/et", "pr/st/im")) for line in lines)


def test_ndm_panel_without_a_breakdown_draws_no_loss_line(viz, monkeypatch):
    lines = rendered_lines(viz, "benthic_community", monkeypatch)
    assert not any(line.startswith("pr/im") for line in lines)
