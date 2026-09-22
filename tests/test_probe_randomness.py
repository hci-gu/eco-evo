"""Fresh visual probes without changing training or fixed-seed progress."""

import argparse
import random

import numpy as np
import pytest
import torch

from lib.environments.ecosystem_env.debug_food import FoodBlobConfig
from lib.runners.trainer import ARSTrainer
from lib.runners.training_progress import add_progress_arguments, measure_survival
from lib.runners.survival_reward import SurvivalReward
from test_training_progress import BiomassTrace
from train import _ProbeEnvBuilder, _probe_biomass


@pytest.mark.parametrize("mode", ["all", "solo"])
def test_probes_refresh_worlds_share_baseline_seed_and_preserve_training_rng(mode, monkeypatch, tmp_path):
    builder = _ProbeEnvBuilder("mareld2.yaml", (12, 15), boundary="torus",
                               food_blobs=FoodBlobConfig(segment_min=2, segment_max=5))
    env = builder(seed=3)
    params = {f: (int(env.per_dm_in_dim[i]), env.N_all + 5) for i, f in enumerate(env.dm_ids)}
    trainer = ARSTrainer(builder, params, hidden_dim=7, obs_normalize=False)
    captured = []
    original = _ProbeEnvBuilder.__call__
    def capture(self, seed=None):
        env = original(self, seed=seed)
        captured.append((seed, {f: fg.biomass.copy() for f, fg in env.fgs.items()}, env._season_phase.copy()))
        return env
    monkeypatch.setattr(_ProbeEnvBuilder, "__call__", capture)
    seeds = iter([11, 22])
    monkeypatch.setattr("lib.runners.training_progress.secrets.randbits", lambda bits: next(seeds))
    np_state, py_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    records, starts = [], []
    for expected_seed in (11, 22):
        captured.clear()
        record = _probe_biomass(trainer, builder, 3, 0, 0, tmp_path / "probe.jsonl",
                                compact=False, rnd_builder=builder, rnd_mode=mode)
        records.append(record)
        assert record["probe_seed"] == expected_seed and "rnd" in record
        assert len(captured) == (2 if mode == "all" else 1 + len(env.dm_ids))
        assert all(seed == expected_seed for seed, _, _ in captured)
        starts.append(captured[0][1]["phytoplankton"])
        for _, fields, phase in captured[1:]:
            assert phase == captured[0][2]
            for fid in fields:
                np.testing.assert_array_equal(fields[fid], captured[0][1][fid])
    assert not np.array_equal(*starts)
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:] and random.getstate() == py_state
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    replay = _probe_biomass(trainer, builder, 3, 0, 0, tmp_path / "replay.jsonl",
                            compact=False, rnd_builder=builder, rnd_mode=mode, probe_seed=11)
    assert replay == records[0]


def test_probe_spawn_override_uses_selected_world_seed(monkeypatch):
    builder = _ProbeEnvBuilder("mareld2.yaml", (5, 6))
    builder.spawn_overrides = {"porpoises": {"mode": "replace", "template": "test"}}
    seeds = []
    monkeypatch.setattr(builder, "_load_spawn_tpls", lambda path: {})
    monkeypatch.setattr(builder, "_apply_spawn", lambda *args, seed: seeds.append(seed))
    builder(seed=19)
    builder(seed=23)
    assert seeds == [19, 23]


def test_progress_default_is_ten_percent_to_ten_times_not_training_reward():
    parser = argparse.ArgumentParser()
    add_progress_arguments(parser)
    args = parser.parse_args([])
    assert tuple(args.biomass_bounds) == (.1, 10.)
    assert parser.parse_args(["--biomass-bounds", ".3", "3"]).biomass_bounds == [.3, 3.]
    trace = BiomassTrace([[100, 100], [10, 1000], [9.9, 1000.1]])
    result = measure_survival(trace, 2, *args.biomass_bounds)
    assert result["survival_ticks"] == {"0": 1, "1": 1}
    assert SurvivalReward().lower == .3
