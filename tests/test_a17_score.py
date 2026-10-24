"""The A17 plankton scorer (tools/probes/a17_score.py, section 205).

1. The targets are the section-204 numbers: column biomass = chl x
   column C:chl / 1000 / C:WW, surface carbon = chl x surface C:chl.
2. A run equal to the targets scores zero error and ratio 1.
3. A short run on a small project world returns twelve months, positive
   gross production, an f-ratio in (0, 1] and leaves the population
   step's functions as it found them.
"""
import sys
from pathlib import Path
import argparse

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import a17_score as a17
from lib.environments.ecosystem_env import population_change


def test_targets_follow_section_204():
    np.testing.assert_allclose(a17.A17_COLUMN[:3], [4.83, 22.44, 21.17], atol=0.01)
    assert a17.A17_COLUMN.mean() == pytest.approx(11.2, abs=0.1)
    assert a17.A17_SURFACE_C[0] == pytest.approx(1.21 * 21)


def test_perfect_run_scores_zero():
    run = {"mixed": (a17.A17_COLUMN * 0.25).tolist(),
           "subsurface": (a17.A17_COLUMN * 0.75).tolist(),
           "surface_carbon": a17.A17_SURFACE_C.tolist(),
           "nitrate": a17.A17_NITRATE.tolist(),
           "nitrate_deep": a17.A17_NITRATE_DEEP.tolist(),
           "zooplankton": a17.A17_ZOO_RESIDENT.tolist(),
           "microzooplankton": [0.1] * 12, "pp": 150.0, "f_ratio": 0.3}
    s = a17.score(run)
    for key in ("column_err", "surface_err", "nitrate_err", "nitrate_deep_err", "zoo_err"):
        assert s[key] == pytest.approx(0.0, abs=1e-12)
    assert s["column_ratio"] == pytest.approx(1.0)
    assert "PP 150.0" in a17.report(run, "perfect")


def test_short_run_on_the_project_world():
    delta, exchange = (population_change._nutrient_limited_delta,
                       population_change.apply_layer_exchange)
    args = argparse.Namespace(project="mareld2.yaml", library=None, seed=1, years=1,
                              burn_in=0, grid=(6, 6), currents="off", checkpoints=None,
                              behaviour="eat")
    run = a17.run_world(args)
    assert population_change._nutrient_limited_delta is delta
    assert population_change.apply_layer_exchange is exchange
    assert len(run["mixed"]) == 12 and len(run["nitrate_deep"]) == 12
    assert run["pp"] > 0.0
    assert 0.0 < run["f_ratio"] <= 1.0
