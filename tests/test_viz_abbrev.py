"""Diet-line codes in the viewer's heatmap panels must be unique.

The rule (initials of a multi-word name) gave ``planktivorous_fish`` and
``predatory_fish`` the same ``pf`` (section 209), so a diet line could
not tell the two prey apart.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.viz.pygame_viz import LiveVisualizer  # noqa: E402

FGS = ["phytoplankton", "subsurface_phytoplankton", "zooplankton", "microzooplankton",
       "benthic_community", "planktivorous_fish", "benthivorous_fish", "predatory_fish",
       "porpoises", "seals", "seabirds"]


def test_every_code_is_unique():
    codes = LiveVisualizer._unique_abbrevs(FGS)
    assert len(set(codes.values())) == len(FGS), codes


def test_clashing_names_take_more_of_their_first_word():
    codes = LiveVisualizer._unique_abbrevs(FGS)
    assert codes["planktivorous_fish"] == "plf"
    assert codes["predatory_fish"] == "prf"


def test_codes_without_a_clash_keep_the_old_rule():
    codes = LiveVisualizer._unique_abbrevs(FGS)
    assert codes["benthivorous_fish"] == "bf"
    assert codes["benthic_community"] == "bc"
    assert codes["seals"] == "sl"


def test_the_equilibrium_spawn_mode_is_shown_as_the_default(monkeypatch):
    """The benthos spawns with 'equilibrium' (section 207). The viewer's
    mode list lacked it, so the panel read 'Mode: (default)' instead of
    the mode the project uses."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from inference import _load_spawn_defaults
    v = LiveVisualizer(fg_ids=["benthic_community"], grid_shape=(4, 4), mode="train",
                       plot_fg_ids=["benthic_community"])
    try:
        v.set_spawn_defaults(_load_spawn_defaults(str(Path(__file__).resolve().parents[1]
                                                      / "mareld2.yaml")))
        assert v._spawn_defaults.get("benthic_community") == "equilibrium"
        assert v._spawn_mode_labels["equilibrium"] == "Equilibrium"
    finally:
        v.close()
