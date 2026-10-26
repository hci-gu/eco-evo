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
       "benthic_infauna", "mobile_epifauna", "planktivorous_fish", "benthivorous_fish",
       "predatory_fish",
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
    assert codes["benthic_infauna"] == "bi"
    assert codes["seals"] == "sl"


def test_the_equilibrium_spawn_mode_is_shown_as_the_default(monkeypatch):
    """The benthos spawns with 'equilibrium' (section 207). The viewer's
    mode list lacked it, so the panel read 'Mode: (default)' instead of
    the mode the project uses."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from inference import _load_spawn_defaults
    v = LiveVisualizer(fg_ids=["benthic_infauna"], grid_shape=(4, 4), mode="train",
                       plot_fg_ids=["benthic_infauna"])
    try:
        v.set_spawn_defaults(_load_spawn_defaults(str(Path(__file__).resolve().parents[1]
                                                      / "mareld2.yaml")))
        assert v._spawn_defaults.get("benthic_infauna") == "equilibrium"
        assert v._spawn_mode_labels["equilibrium"] == "Equilibrium"
    finally:
        v.close()


def test_clashing_names_with_different_second_words_extend_the_second():
    """Italian 'Pesci planctivori' / 'Pesci predatori' share the first
    word: extending it gave 'pescip' for both (section 216)."""
    codes = LiveVisualizer._unique_abbrevs(
        ["Pesci planctivori", "Pesci predatori", "Pesci bentivori"])
    assert codes == {"Pesci planctivori": "ppl", "Pesci predatori": "ppr",
                     "Pesci bentivori": "pb"}


def test_the_second_word_takes_as_many_letters_as_needed():
    codes = LiveVisualizer._unique_abbrevs(["Pesci predoni", "Pesci predatori"])
    assert codes == {"Pesci predoni": "ppredo", "Pesci predatori": "ppreda"}


def test_same_second_words_fall_back_to_the_first_word():
    codes = LiveVisualizer._unique_abbrevs(["Planktivorous fish", "Predatory fish"])
    assert codes == {"Planktivorous fish": "plf", "Predatory fish": "prf"}


def test_every_language_gives_unique_codes():
    from lib.viz import i18n
    for lang in i18n.FG_NAMES:
        names = [i18n.fg_name(lang, f) for f in FGS]
        codes = LiveVisualizer._unique_abbrevs(names)
        assert len(set(codes.values())) == len(FGS), (lang, codes)
