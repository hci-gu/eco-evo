"""Resumed viewers get the plot history from biomass.jsonl (train.py and train_gpu.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import _hydrate_viz_history


class _Viz:
    def __init__(self):
        self.series, self.rewards = [], []

    def update_series(self, tab, fid, value, step=None):
        self.series.append((tab, fid, value, step))

    def update_reward(self, fid, value, step=None):
        self.rewards.append((fid, value, step))

    def pump_events(self):
        pass


def _write(path):
    rows = [{"__meta__": {"run": "x"}},
            {"gen": 1, "iter": 1, "ratio": {"herring": 0.5}, "reward": {"herring": 0.1},
             "rest_frac": {"herring": 2.0}},
            {"gen": 2, "iter": 3, "ratio": {"herring": 0.7}, "reward": {"herring": 0.2},
             "loss_breakdown": {"herring": {"predation": 0.4, "starvation": 0.1}}}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")


def test_steps_follow_each_trainers_x_axis(tmp_path):
    path = tmp_path / "biomass.jsonl"
    _write(path)
    cpu, gpu = _Viz(), _Viz()
    assert _hydrate_viz_history(cpu, str(path), iter_per_gen=50) == 2
    assert _hydrate_viz_history(gpu, str(path), iter_per_gen=50, step_offset=1) == 2
    # train.py plots gen g, iteration i (0-based) at g * 50 + i.
    assert cpu.rewards == [("herring", 0.1, 0), ("herring", 0.2, 52)]
    # train_gpu.py plots at iterations_completed, one later.
    assert gpu.rewards == [("herring", 0.1, 1), ("herring", 0.2, 53)]
    assert ("biomass", "herring", 50.0, 0) in cpu.series
    assert ("rest", "herring", 2.0, 0) in cpu.series
    assert ("predation", "herring", 40.0, 53) in gpu.series


def test_missing_file_is_not_fatal(tmp_path):
    assert _hydrate_viz_history(_Viz(), str(tmp_path / "none.jsonl"), 50) == 0
