"""train_gpu.py rewrites <run>/plots.html from the --visual probe log at each checkpoint."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpu

MODEL = SimpleNamespace(ids=("herring", "plankton"), dm_ids=("herring",))
OPTIONS = {"integral_reward": True, "legacy_reward": False, "population_stability": False,
           "local_reward": False}


def test_writes_plots_with_the_run_options_as_meta(tmp_path):
    record = {"gen": 1, "iter": 1, "n_ticks": 10, "b0": {"herring": 1.0, "plankton": 2.0},
              "bh": {"herring": 1.1, "plankton": 2.0}, "ratio": {"herring": 1.1, "plankton": 1.0},
              "reward": {"herring": 0.1}}
    (tmp_path / "biomass.jsonl").write_text(json.dumps(record) + "\n")
    train_gpu.regenerate_plots(SimpleNamespace(model=MODEL), tmp_path, OPTIONS)
    html = (tmp_path / "plots.html").read_text()
    assert "total-energy (integral)" in html


def test_no_probe_log_no_plots(tmp_path):
    train_gpu.regenerate_plots(SimpleNamespace(model=MODEL), tmp_path, OPTIONS)
    assert not (tmp_path / "plots.html").exists()
