"""Realised diets in the long-run probe and the field-spread diet gate (section 248).

Contracts asserted:

1. ``run_probe`` differences the env's cumulative ``intake_by_pred_prey``
   book at every probe-year end: ``diets[k][pred]["tonnes"]`` is that
   year's take per prey and ``ration`` the year's total over the
   predator's mean biomass. An env without a book gives no diets; the
   record stays JSON without NaN.
2. ``tools/probes/probe_diets.py`` compares the realised shares with the
   booked M2 x B shares of the run's own library and manifest and flags a
   pair outside the diet gate's band.
3. Section 249: the probe stores every group's mean biomass per year, and
   probe_diets prints each prey's realised M2 (tonnes eaten / mean
   biomass) against its booked M2, flagging a pair that carries >= 0.1 of
   the prey's booked M2 outside the band.
4. ``field_cell_weights`` spreads each bed class's share evenly over its
   cells and renormalises when a class has no cells.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.diagnostics.field_diet import FIELD_BED_SHARES, bed_class, field_cell_weights
from lib.runners.training_progress import run_probe

YEAR = 1460


class Feeder:
    """One predator eating 2 t of krill and 1 t of fish per tick; its
    biomass doubles at the year boundary."""

    def __init__(self, book=True):
        self.fgs = {f: SimpleNamespace(biomass=np.array([10.0])) for f in ("fish", "krill")}
        self.dm_ids, self.global_fg_order, self.tick_hours, self.t = ["fish"], ["fish", "krill"], 6, 0
        if book:
            self.intake_by_pred_prey = {"fish": {}, "krill": {}}

    def tick(self):
        self.t += 1
        self.fgs["fish"].biomass[:] = 10.0 if self.t <= YEAR else 20.0
        if hasattr(self, "intake_by_pred_prey"):
            row = self.intake_by_pred_prey["fish"]
            row["krill"] = row.get("krill", 0.0) + 2.0
            row["fish"] = row.get("fish", 0.0) + (1.0 if self.t <= YEAR else 3.0)


def test_the_probe_books_each_years_diet_and_ration():
    result = run_probe(Feeder(), 2 * YEAR, 0.0, 10.0)
    first, second = result["diets"]
    assert first["fish"]["tonnes"] == pytest.approx({"krill": 2.0 * YEAR, "fish": 1.0 * YEAR})
    assert second["fish"]["tonnes"] == pytest.approx({"krill": 2.0 * YEAR, "fish": 3.0 * YEAR})
    assert first["fish"]["ration"] == pytest.approx(3.0 * YEAR / 10.0)
    assert second["fish"]["ration"] == pytest.approx(5.0 * YEAR / 20.0)
    json.dumps(result["diets"], allow_nan=False)
    assert result["year_biomass"][0]["fish"] == pytest.approx(10.0)
    assert result["year_biomass"][1]["fish"] == pytest.approx(20.0)
    assert result["year_biomass"][0]["krill"] == pytest.approx(10.0)
    # A part year is not a probe year.
    assert len(run_probe(Feeder(), YEAR + 10, 0.0, 10.0)["diets"]) == 1


def test_an_env_without_an_intake_book_has_no_diets():
    assert run_probe(Feeder(book=False), YEAR, 0.0, 10.0)["diets"] == []


def _run_dir(tmp_path, diets, year_biomass=None):
    library = {"interaction_definitions": {
        "fish_preys_on_krill": {"preys_on": True, "predation_mortality": 1.0},
        "fish_preys_on_worms": {"preys_on": True, "predation_mortality": 0.5},
        "fish_preys_on_squid": {"preys_on": False, "predation_mortality": 9.0},
    }}
    manifest = {"project_metadata": {"reference_grid_width": 10, "reference_grid_height": 10},
                "groups": [{"group_id": "fish", "inference_initial_biomass": 100.0},
                           {"group_id": "krill", "inference_initial_biomass": 300.0},
                           {"group_id": "worms", "inference_initial_biomass": 200.0}]}
    (tmp_path / "lib.yaml").write_text(yaml.safe_dump(library))
    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(manifest))
    progress = tmp_path / "run" / "progress"
    progress.mkdir(parents=True)
    (progress / "config.json").write_text(json.dumps(
        {"project": str(tmp_path / "proj.yaml"), "library": str(tmp_path / "lib.yaml")}))
    (progress / "probe.jsonl").write_text(json.dumps(
        {"evaluation": 1, "step": 10, "diets": diets, "year_biomass": year_biomass}) + "\n")
    return tmp_path / "run"


def test_probe_diets_compares_with_the_booked_shares(tmp_path, capsys):
    from tools.probes.probe_diets import main
    # Booked: krill 1.0 x 3 = 3, worms 0.5 x 2 = 1 -> 0.75 / 0.25.
    diets = [{"fish": {"ration": 4.0, "tonnes": {"krill": 75.0, "worms": 25.0}}},
             {"fish": {"ration": 4.0, "tonnes": {"krill": 97.0, "worms": 1.0, "squid": 2.0}}}]
    main([str(_run_dir(tmp_path, diets))])
    out = capsys.readouterr().out
    assert "ration 4.00 / 4.00 t/t/yr" in out
    assert "0.750  1.00 " in out                     # year 1 krill on its booked share
    assert "fish -> worms year 2" in out             # 0.01 vs 0.25: OFF
    assert "fish -> krill" not in out.split("OFF:")[1]
    assert "0.020      -" in out                     # unbooked squid shown, never judged


def test_probe_diets_refuses_a_probe_without_diets(tmp_path):
    from tools.probes.probe_diets import main
    with pytest.raises(SystemExit, match="no diets"):
        main([str(_run_dir(tmp_path, []))])


def test_field_cell_weights_follow_the_class_shares():
    depth = np.array([[50.0, 90.0, 120.0], [200.0, 220.0, 300.0]])
    assert bed_class(depth).tolist() == [[0, 0, 1], [2, 2, 3]]
    w = field_cell_weights(depth, (0.4, 0.2, 0.3, 0.1))
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w, [[0.2, 0.2, 0.2], [0.15, 0.15, 0.1]])
    # No > 250 m cell: the other three classes are renormalised.
    w = field_cell_weights(depth[:, :2], (0.4, 0.2, 0.3, 0.1))
    assert w.sum() == pytest.approx(1.0) and w[0, 0] == pytest.approx(0.4 / 0.7 / 2)


def test_field_bed_shares_are_shares():
    for pid, (shares, label) in FIELD_BED_SHARES.items():
        assert len(shares) == 4 and sum(shares) == pytest.approx(1.0, abs=1e-3), pid


def test_probe_diets_prey_side_reads_realised_m2(tmp_path, capsys):
    from tools.probes.probe_diets import main
    # Krill booked M2 1.0, worms 0.5. Year 1: krill 300 t eaten of 300 t -> 1.0 (ok);
    # worms 1 t of 200 t -> 0.005 (OFF).
    diets = [{"fish": {"ration": 3.0, "tonnes": {"krill": 300.0, "worms": 1.0}}}]
    biomass = [{"fish": 100.0, "krill": 300.0, "worms": 200.0}]
    main([str(_run_dir(tmp_path, diets, biomass))])
    out = capsys.readouterr().out.split("=== prey side")[1]
    assert "krill: booked 1; realised 1" in out
    assert "Prey side OFF: fish -> worms year 1" in out
    assert "fish -> krill" not in out.split("Prey side OFF:")[1]
