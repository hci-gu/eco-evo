"""train_gpu.py's viewer probe on the tensor engine (section 237):
``train._probe_biomass(engine="gpu")`` against the CPU engine, with the
random baselines riding in the same batch."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

with torch.no_grad():
    import train  # noqa: E402
from lib.runners.policy import PolicyNetwork  # noqa: E402
from test_gpu_ecosystem import DEVICES  # noqa: E402

TICKS = 5


def _trainer(env):
    torch.manual_seed(7)
    policies = {fid: PolicyNetwork(int(env.per_dm_in_dim[i]), 5 + env.N_all,
                                   hidden_dim=8, hidden_layers=2,
                                   activation="tanh").eval()
                for i, fid in enumerate(env.dm_ids)}
    rng = np.random.default_rng(1)
    stats = {fid: dict(mean=rng.normal(0, 0.1, env.max_in_dim).astype(np.float32),
                       var=rng.uniform(0.5, 2, env.max_in_dim).astype(np.float32),
                       count=10)
             for fid in env.dm_ids}
    return SimpleNamespace(policies=policies, obs_stats=stats, obs_normalize=True,
                           softmax_temperature=1.0, legacy_reward=False,
                           integral_reward=True, alpha=1.0, beta=1.0,
                           survival_bonus=0.0, survival_threshold=0.01,
                           population_stability=None, local_reward=None)


def _probe(tmp_path, engine, device, rnd_mode):
    builder = train._ProbeEnvBuilder(str(ROOT / "mareld2.yaml"), (6, 7))
    trainer = _trainer(builder())
    out = tmp_path / f"{engine}_{device}_{rnd_mode}.jsonl"
    train._probe_biomass(trainer, builder, n_ticks=TICKS, gen=0, it=0,
                         jsonl_path=str(out), compact=False,
                         rnd_builder=builder if rnd_mode != "none" else None,
                         rnd_mode=rnd_mode, engine=engine, device=device)
    return json.loads(out.read_text().splitlines()[-1])


def _numbers(record, prefix=""):
    if isinstance(record, dict):
        for key, value in record.items():
            yield from _numbers(value, f"{prefix}/{key}")
    elif isinstance(record, (int, float)) and not isinstance(record, bool):
        yield prefix, float(record)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rnd_mode", ["solo", "all"])
def test_probe_record_matches_the_cpu_engine(tmp_path, monkeypatch, device, rnd_mode):
    from lib.gpu import inference as gpu_inference
    steps = []
    original = gpu_inference.GpuStepper.step
    monkeypatch.setattr(gpu_inference.GpuStepper, "step",
                        lambda self: (steps.append(len(self.envs)), original(self))[1])
    cpu = dict(_numbers(_probe(tmp_path, "cpu", "cpu", rnd_mode)))
    assert steps == []
    gpu = dict(_numbers(_probe(tmp_path, "gpu", device, rnd_mode)))
    n_dm = len(train._ProbeEnvBuilder(str(ROOT / "mareld2.yaml"), (6, 7))().dm_ids)
    assert steps == [1 + (n_dm if rnd_mode == "solo" else 1)] * TICKS
    assert set(gpu) == set(cpu)
    assert any("_rnd" in k or "rnd" in k for k in cpu)
    for key, value in cpu.items():
        if key.endswith("/time") or "seconds" in key:
            continue
        # Shares and ratios; the loss shares carry the CPU engine's float32
        # M1 book (section 235), hence the absolute term.
        assert gpu[key] == pytest.approx(value, rel=1e-3, abs=1e-4), key
