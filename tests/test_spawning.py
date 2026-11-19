"""Section 279: spawning in the box (lib/world/spawning.py).

The clock reads only the day length and its sign; the gametes shed integrate
to share x gamete_energy a year; the room cut is the spawners' accumulated
deficit, zero outside the window; the growth budget pays P at the lowered
fill; malformed blocks are refused.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.world import daylight, spawning  # noqa: E402
from lib.world.growth_budget import growth_budget  # noqa: E402

LAT = 58.15
ME = 1800.0


def entry(**extra):
    base = {"name": "cod", "share": 0.163, "gamete_energy": 430.0, "reserve_share": 0.5,
            "fast_cut": 0.85, "fast_loss": 20.0, "spawning_days": 60,
            "spawn_start": {"day_length_h": 6.3, "rising": True},
            "spawn_end": {"day_length_h": 14.1, "rising": True}}
    base.update(extra)
    return base


def spec(*entries):
    return {"appetite": "reserve_room", "max_energy_reserve": ME,
            spawning.KEY: list(entries) or [entry()]}


def test_day_of_reads_length_and_sign():
    hours = np.asarray(daylight.day_length_hours(LAT))
    up = spawning.day_of(LAT, 12.9, True)
    down = spawning.day_of(LAT, 12.9, False)
    assert 85 < up < 95 and 250 < down < 260
    assert hours[up] >= 12.9 > hours[up - 1]
    assert hours[down] <= 12.9 < hours[down - 1]
    with pytest.raises(ValueError):
        spawning.day_of(LAT, 19.0, True)


def test_shed_integrates_and_cut_shape():
    cut, shed = spawning.group_by_day(spec(), LAT)
    assert shed.sum() == pytest.approx(0.163 * 430.0, rel=1e-6)
    start = spawning.day_of(LAT, 6.3, True)
    end = spawning.day_of(LAT, 14.1, True)
    outside = np.ones(365, dtype=bool)
    outside[start:end] = False
    assert np.all(cut[outside] == 0.0) and np.all(shed[outside] == 0.0)
    # The deficit of one spawner that fasts all 60 days, times the share,
    # bounds the group's cut; the staggered starts keep it well below.
    per_fish = (0.85 * 20.0 + 0.5 * 430.0 / 60.0) * 60.0 / ME
    assert 0.0 < cut.max() < 0.163 * per_fish
    peak = int(np.argmax(cut))
    assert start < peak < end


def test_feeders_without_a_fast_cut_only_the_reserve_share():
    cut, shed = spawning.group_by_day(spec(entry(fast_cut=0.0, reserve_share=0.0)), LAT)
    assert np.all(cut == 0.0) and shed.sum() > 0.0


def test_window_across_the_new_year():
    late = entry(spawn_start={"day_length_h": 8.5, "rising": False},
                 spawn_end={"day_length_h": 8.0, "rising": True}, spawning_days=None)
    cut, shed = spawning.group_by_day(spec(late), LAT)
    assert shed.sum() == pytest.approx(0.163 * 430.0, rel=1e-6)
    assert shed[0] > 0.0 and shed[350] > 0.0 and shed[150] == 0.0


def test_tick_tables_conserve_the_year():
    cut_d, shed_d = spawning.group_by_day(spec(), LAT)
    for hours in (6, 3, 1):
        cut, shed = spawning.tick_tables(spec(), LAT, hours)
        assert cut.shape == shed.shape == (daylight.ticks_per_year(hours),)
        assert shed.sum() == pytest.approx(shed_d.sum(), rel=1e-6)
        assert cut.max() == pytest.approx(cut_d.max())


def test_inert_without_block_or_latitude():
    assert spawning.group_by_day({"appetite": "reserve_room"}, LAT) is None
    assert spawning.group_by_day(spec(), None) is None


@pytest.mark.parametrize("bad", [
    {"appetite": "gate", "max_energy_reserve": ME, spawning.KEY: [entry()]},
    spec(entry(share=0.0)), spec(entry(share=0.7), entry(share=0.7)),
    spec(entry(fast_cut=1.5)), spec(entry(reserve_share=-0.1)),
    spec(entry(spawning_days=200)), spec(entry(gamete_energy="x")),
    spec(entry(spawn_start={"day_length_h": 8.0, "rising": "yes"})),
    spec(entry(spawn_end=12.0)),
])
def test_malformed_block_is_refused(bad):
    with pytest.raises(ValueError):
        spawning.group_by_day(bad, LAT)


def _library():
    lib = yaml.safe_load(open(ROOT / "fgconfig" / "fg_library.yaml"))
    return lib["species_definitions"], lib["interaction_definitions"]


def test_growth_budget_pays_p_at_the_lowered_fill():
    """g = P / (1460 mean(min(1 - cut, s*) - u)) with the site's latitude."""
    species, inter = _library()
    active = [f for f, s in species.items() if isinstance(s, dict)]
    benth = dict(species["benthivorous_fish"])
    benth[spawning.KEY] = [entry()]
    species = dict(species, benthivorous_fish=benth)
    site = growth_budget("benthivorous_fish", species, inter, active, latitude_deg=LAT)
    none = growth_budget("benthivorous_fish", species, inter, active)
    assert not site.error and not none.error
    u = float(benth["maintenance_level"])
    cut = spawning.cut_by_day(benth, LAT)
    gate = float((np.minimum(1.0 - cut, none.reference_fill) - u).mean())
    assert site.growth_rate / none.growth_rate == pytest.approx(
        (none.reference_fill - u) / gate, rel=1e-6)
    assert site.window_share == pytest.approx(1.0)


def test_library_blocks():
    """The library's entries parse; their group values are the research
    note's (research_notes/"Lek i rutan" summary.md section 3)."""
    species, _ = _library()
    expected = {"benthivorous_fish": (0.401, 181.0), "whiting": (0.344, 275.0),
                "predatory_fish": (0.058, 19.0), "planktivorous_fish": (0.043, 34.0)}
    for group, (share, gametes) in expected.items():
        spec_ = species[group]
        entries = spawning.settings(spec_)
        assert sum(e["share"] for e in entries) == pytest.approx(share, abs=0.002)
        cut, shed = spawning.group_by_day(spec_, LAT)
        assert shed.sum() == pytest.approx(gametes, rel=0.02)
        floor = float(spec_["maintenance_level"])
        assert (1.0 - cut).min() > floor
    assert spawning.settings(species["mackerel"]) is None


def test_fg_editor_round_trip():
    pytest.importorskip("ruamel.yaml")
    sys.path.insert(0, str(ROOT / "fgconfig"))
    fgconfig = pytest.importorskip("fgconfig")
    App = fgconfig.FGConfigApp
    species, _ = _library()
    for group in ("benthivorous_fish", "whiting", "predatory_fish", "planktivorous_fish"):
        entries = species[group][spawning.KEY]
        back, error = App._collect_spawning_block(App._format_spawning(entries), "reserve_room")
        assert error is None and back == entries
    assert App._collect_spawning_block("", "reserve_room") == (None, None)
    _, error = App._collect_spawning_block("cod | 0.1 | 400", "reserve_room")
    assert error and "9 fields" in error
    _, error = App._collect_spawning_block(
        "cod | 0.1 | 400 | 0.5 | 0.8 | 20 | 60 | 6.3 | 14.1+", "reserve_room")
    assert error and "+ or -" in error
    _, error = App._collect_spawning_block(
        App._format_spawning(species["whiting"][spawning.KEY]), "gate")
    assert error and "reserve_room" in error


def test_fg_editor_apply_keeps_the_list_and_the_growth_rate(tmp_path):
    """An Apply on the benthivores with the live library keeps the spawning
    list (and its comments) and writes the growth rate the budget derives
    with it; an edited line is written back."""
    pytest.importorskip("ruamel.yaml")
    tk = pytest.importorskip("tkinter")
    sys.path.insert(0, str(ROOT / "fgconfig"))
    fgconfig = pytest.importorskip("fgconfig")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    from unittest import mock
    live = (ROOT / "fgconfig" / "fg_library.yaml").read_text()
    target = tmp_path / "lib.yaml"
    target.write_text(live)
    library = yaml.safe_load(live)["species_definitions"]["benthivorous_fish"]
    try:
        with mock.patch.object(fgconfig.messagebox, "showinfo"), \
                mock.patch.object(fgconfig.messagebox, "askyesno", return_value=True):
            app = fgconfig.FGConfigApp(root)
            app.load_project_from_path(str(ROOT / "mareld2.yaml"))
            app.library_path = str(target)
            ids = [fg["group_id"] for fg in app.project_data["decision_makers"]]
            app.fg_listbox.selection_clear(0, "end")
            app.fg_listbox.selection_set(ids.index("benthivorous_fish"))
            app.on_fg_select("decision_makers")
            assert app.spawning_text.get("1.0", "end").strip().startswith("cod |")
            app.apply_fg_changes()
            written = yaml.safe_load(target.read_text())["species_definitions"]["benthivorous_fish"]
            assert written[spawning.KEY] == library[spawning.KEY]
            assert float(written["growth_rate"]) == pytest.approx(
                float(library["growth_rate"]), rel=1e-6)
            assert "Section 279: spawning in the box" in target.read_text()
            text = app.spawning_text.get("1.0", "end").replace("| 430 |", "| 400 |", 1)
            app.spawning_text.delete("1.0", "end")
            app.spawning_text.insert("1.0", text)
            app.apply_fg_changes()
    finally:
        root.destroy()
    written = yaml.safe_load(target.read_text())["species_definitions"]["benthivorous_fish"]
    assert written[spawning.KEY][0]["gamete_energy"] == 400.0
