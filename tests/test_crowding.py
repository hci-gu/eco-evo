"""Section 278: the crowding diagnostic and its place in the long-run probe."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.diagnostics.crowding import FIELD_CROWDING, crowding  # noqa: E402
from lib.runners.training_progress import run_probe  # noqa: E402


def test_an_even_spread_reads_one_and_k_equal_cells_read_n_over_k():
    assert crowding(np.full((30, 30), 2.5)) == pytest.approx(1.0)
    field = np.zeros((30, 30))
    field.flat[:9] = 4.0
    assert crowding(field) == pytest.approx(900 / 9)


def test_it_ignores_the_total_and_has_no_value_without_biomass():
    rng = np.random.default_rng(0)
    field = rng.random((8, 8))
    assert crowding(field) == pytest.approx(crowding(field * 37.0))
    assert crowding(np.zeros((4, 4))) is None
    assert crowding(np.full((4, 4), np.nan)) is None


def test_the_field_table_holds_a_range_around_each_median():
    for group, (median, low, high, source) in FIELD_CROWDING.items():
        assert 1.0 <= low <= median <= high, group
        assert source


class PackingTrace:
    """Two decision makers on 4 cells: 'even' stays spread, 'packs' sits on
    one cell in the second probe year; one tick per day, 4 days per year."""

    tick_hours = 6

    def __init__(self):
        self.fgs = {"even": SimpleNamespace(biomass=np.ones(4)),
                    "packs": SimpleNamespace(biomass=np.ones(4))}
        self.dm_ids = ["even", "packs"]
        self.global_fg_order = ["even", "packs"]
        self.t = 0

    def tick(self):
        self.t += 1
        if self.t > 1460:
            self.fgs["packs"].biomass[:] = [4.0, 0.0, 0.0, 0.0]


def test_the_probe_records_each_years_mean_crowding(tmp_path):
    result = run_probe(PackingTrace(), 2 * 1460, 0.0, 10.0)
    years = result["crowding"]
    assert len(years) == 2
    assert years[0] == pytest.approx({"even": 1.0, "packs": 1.0})
    assert years[1]["even"] == pytest.approx(1.0)
    assert years[1]["packs"] == pytest.approx(4.0)
    json.dumps(result["crowding"], allow_nan=False)


def test_the_tool_reads_the_history(tmp_path, capsys):
    from tools.probes import probe_crowding
    run = tmp_path / "run"
    (run / "progress").mkdir(parents=True)
    records = [{"step": 10, "evaluation": 1},
               {"step": 20, "evaluation": 2,
                "crowding": [{"benthivorous_fish": 60.0}, {"benthivorous_fish": 64.0}]}]
    (run / "progress" / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    probe_crowding.main([str(run)])
    out = capsys.readouterr().out
    assert "u20" in out and "u10" not in out
    assert "62.0" in out and "20.3x" in out
