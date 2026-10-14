"""Impacts on the GPU engine and baseline impacts (section 161).

The GPU engine used to skip impacts altogether. It now mirrors the
reference: direct impact mortality before predation, the
``1 + sum(energy_factor)`` surcharge on every action cost, from per-world
fields that ``TensorEcosystem.impact_fields`` turns into rates once per
world, and the observable impacts as raw input channels after the
observed groups (own cell and the four neighbours, the light channel
still last). A ``baseline: true`` impact (fishing) is part of today's
world: it gets a field in training even though the policy cannot observe
it, and an inference world without a map for it samples one.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import load_inference_impact_maps
from lib.config.config_loader import project_baseline_impact_vars
from lib.environments.ecosystem_env.impacts import interp_impact
from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from lib.gpu.trainer import TensorARSTrainer
from lib.world.impact_maps import sample_impact_maps
from test_gpu_ecosystem import DEVICES, _assert_parity, make_env

ROOT = Path(__file__).resolve().parents[1]

TRAWL = {"impact_affects": True, "impact_table": [
    {"value": 0.0, "biomass_factor": 0.0, "energy_factor": 0.0},
    {"value": 10.0, "biomass_factor": 0.08, "energy_factor": 0.5}]}
# Non-zero at value 0: an absent (zero) field still costs something.
NOISE = {"impact_affects": True, "impact_table": [
    {"value": 0.0, "biomass_factor": 0.01, "energy_factor": 0.2},
    {"value": 5.0, "biomass_factor": 0.03, "energy_factor": 0.0}]}
OFF = {"impact_affects": False, "impact_table": []}


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def impact_env(migration=False, mortality=False, maps=("trawl", "noise"),
               observable=None, daylight=None):
    env = make_env(migration, mortality, min_split=0, extinction_factor=0,
                   impacts={"a": {"trawl": TRAWL, "noise": NOISE},
                            "b": {"trawl": TRAWL, "noise": OFF}},
                   observable_impacts=observable, daylight=daylight)
    rng = np.random.default_rng(7)
    # The trawl field runs past the table's last point: clamped, as on CPU.
    fields = {"trawl": rng.uniform(0, 12, (env.H, env.W)),
              "noise": rng.uniform(0, 6, (env.H, env.W))}
    for impact_id in maps:
        env.grid.add_map(impact_id, fields[impact_id].astype(np.float32))
    return env


@pytest.mark.parametrize("migration,mortality", [(False, False), (True, True)])
@pytest.mark.parametrize("maps", [("trawl", "noise"), ("trawl",)])
def test_impacts_match_reference(device, migration, mortality, maps):
    """Full ticks with impact mortality and cost, against the CPU tick.

    ``("trawl",)`` leaves the noise field out: both engines skip a
    missing field rather than reading it as zero.
    """
    model = _assert_parity(impact_env(migration, mortality, maps), device)
    assert model.has_impacts and model.impact_ids == ("noise", "trawl")


@pytest.mark.parametrize("observable", [["noise"], ["trawl", "noise"], ["noise", "seen"]])
@pytest.mark.parametrize("daylight", [False, True])
def test_observable_impacts_match_reference(device, observable, daylight):
    """Observable impact channels in the same slots as ``train.py``'s.

    ``"seen"`` has no field and no table: an observed impact that is
    pure information reads as zeros on both engines. With the daylight
    calendar the light channel must stay the last slot.
    """
    calendar = ({"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 3,
                 "random_start": False} if daylight else None)
    env = impact_env(observable=observable, daylight=calendar)
    model = _assert_parity(env, device, ticks=6)
    assert model.observed_impacts == tuple(observable)
    assert model.in_dims == tuple(int(n) for n in env.per_dm_in_dim)
    assert model.daylight == daylight


def test_impact_fields_sum_each_groups_tables(device):
    env = impact_env()
    model = TensorEcosystem(env, device)
    loss, cost = model.import_impacts([env])
    trawl = env.grid.get_map("trawl").reshape(-1)
    noise = env.grid.get_map("noise").reshape(-1)
    table = lambda block: tuple(np.array([row[k] for row in block["impact_table"]], np.float32)
                                for k in ("value", "biomass_factor", "energy_factor"))
    bf_t, ef_t = interp_impact(table(TRAWL), trawl)
    bf_n, ef_n = interp_impact(table(NOISE), noise)
    a, b = env.dm_ids.index("a"), env.dm_ids.index("b")
    np.testing.assert_allclose(loss[0, a].cpu(), bf_t + bf_n, rtol=1e-6)
    np.testing.assert_allclose(cost[0, a].cpu(), ef_t + ef_n, rtol=1e-6)
    # "b" is not affected by the noise: its table is switched off.
    np.testing.assert_allclose(loss[0, b].cpu(), bf_t, rtol=1e-6)
    np.testing.assert_allclose(cost[0, b].cpu(), ef_t, rtol=1e-6)


def test_impacts_change_the_trajectory(device):
    """The parity test would also pass if both engines ignored impacts."""
    env = impact_env()
    model = TensorEcosystem(env, device)
    b, r, _ = model.import_state([env])
    probs = torch.full((1, model.D, model.A, model.C), 1.0 / model.A, device=model.device)
    loss, cost = model.import_impacts([env])
    on = model.step(b, r, probs, model.tensor(0), torch.zeros_like(b), impacts=(loss, cost))
    off = model.step(b, r, probs, model.tensor(0), torch.zeros_like(b))
    dm = model.dm_index
    assert (on[0][:, dm].sum() < off[0][:, dm].sum()).item()
    assert (on[1][:, dm].sum() < off[1][:, dm].sum()).item()


def test_no_impact_tables_keep_the_old_graph(device):
    model = TensorEcosystem(make_env(), device)
    assert not model.has_impacts and model.impact_ids == ()
    runner = RolloutRunner(model, PolicyBank(model, hidden_dim=5), 1, 1, execution="eager")
    assert not hasattr(runner, "impact_loss")
    b, r, _ = model.import_state([make_env()])
    zero = torch.zeros((1, model.D, model.C), device=model.device)
    with pytest.raises(ValueError, match="impact table"):
        runner.reset(b, r, torch.zeros(1, dtype=torch.long, device=model.device), 2,
                     torch.zeros((model.D, model.F), device=model.device),
                     torch.ones((model.D, model.F), device=model.device),
                     model.tensor(1.0), impacts=(zero, zero))


def test_baseline_impacts_are_sampled_in_training_unobserved():
    ranges = {"fishing": (0.0, 2.0), "noise": (0.0, 140.0), "rotor": (0.0, 1.0)}
    maps = sample_impact_maps(["fishing", "noise", "rotor"], ranges, (6, 7), seed=3,
                              observable_impact_vars=["noise"],
                              baseline_impact_vars=["fishing"])
    assert maps["fishing"].max() > 0 and 0 <= maps["fishing"].min() <= maps["fishing"].max() <= 2
    assert maps["noise"].max() > 0
    assert not maps["rotor"].any(), "a scenario impact the policy cannot see stays absent"
    legacy = sample_impact_maps(["rotor"], ranges, (6, 7), seed=3)
    assert legacy["rotor"].max() > 0, "observable None keeps the sample-everything rule"


def test_project_baseline_impact_vars_skips_muted():
    project = {"impact_variables": [
        {"impact_id": "pelagic_trawling", "baseline": True},
        {"impact_id": "bottom_trawling", "baseline": True, "muted": True},
        {"impact_id": "windfarm_noise", "observable": True}]}
    assert project_baseline_impact_vars(project) == ["pelagic_trawling"]


@pytest.fixture
def baseline_project(tmp_path):
    """mareld2 with pelagic trawling un-muted as a baseline impact."""
    with open(ROOT / "mareld2.yaml", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    for iv in data["impact_variables"]:
        if iv["impact_id"] == "pelagic_trawling":
            iv["muted"] = False
            iv["baseline"] = True
    path = tmp_path / "baseline.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path), data


def test_inference_samples_a_baseline_impact_without_a_map(baseline_project):
    path, _ = baseline_project
    first = load_inference_impact_maps(path, ["pelagic_trawling", "windfarm_noise"],
                                       8, 9, seed=5, verbose=False)
    again = load_inference_impact_maps(path, ["pelagic_trawling", "windfarm_noise"],
                                       8, 9, seed=5, verbose=False)
    assert 0 < first["pelagic_trawling"].max() <= 1000.0
    np.testing.assert_array_equal(first["pelagic_trawling"], again["pelagic_trawling"])
    assert not first["windfarm_noise"].any()


def baseline_trainer(path, **kwargs):
    spec = ProjectSpec(EnvironmentBuilder(project_path=path, grid=(5, 6)))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2,
                            worlds=2, seed=16, hidden_dim=6, **kwargs)


def test_trainer_gives_each_world_its_baseline_field(baseline_project):
    path, _ = baseline_project
    trainer = baseline_trainer(path)
    m = trainer.model
    assert m.has_impacts and "pelagic_trawling" in m.impact_ids
    trainer.refresh_worlds(epoch=3)
    herring = m.dm_ids.index("pelagic_fish")
    loss = trainer.world_impact_loss.clone()
    assert (loss[:, herring] > 0).any()
    others = [d for d, fid in enumerate(m.dm_ids) if fid != "pelagic_fish"]
    assert not loss[:, others].any()
    assert not torch.equal(loss[0], loss[1]), "each world draws its own field"
    trainer.refresh_worlds(epoch=4)
    trainer.refresh_worlds(epoch=3)
    torch.testing.assert_close(trainer.world_impact_loss, loss, rtol=0, atol=0)
    trainer.train_step(n_eval_ticks=2, world_epoch=3)
    expanded = loss[None].expand(2 * trainer.pairs_per_batch, -1, -1, -1).reshape(
        trainer.runner.E, m.D, m.C)
    torch.testing.assert_close(trainer.runner.impact_loss, expanded, rtol=0, atol=0)
    resumed = baseline_trainer(path)
    resumed.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(resumed.world_impact_loss, loss, rtol=0, atol=0)


def observable_project(data, tmp_path):
    data = copy.deepcopy(data)
    for iv in data["impact_variables"]:
        if iv["impact_id"] == "windfarm_noise":
            iv["muted"] = False
    path = tmp_path / "observable.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


def test_trainer_feeds_observable_impacts_like_train_py(baseline_project, tmp_path):
    """windfarm_noise observable (colony spawn, 0-140): the input layer
    widens as on the CPU trainer, each world draws its field, and it
    reaches the runner's observation buffer."""
    _, data = baseline_project
    path = observable_project(data, tmp_path)
    trainer = baseline_trainer(path)
    m = trainer.model
    assert m.observed_impacts == ("windfarm_noise",)
    spec_cpu = trainer.spec.env
    assert m.in_dims == tuple(int(n) for n in spec_cpu.per_dm_in_dim)
    trainer.refresh_worlds(epoch=2)
    noise = trainer.world_impact_obs.clone()
    assert noise.shape == (2, 1, m.C)
    assert 0 < noise.max().item() <= 140.0
    assert not torch.equal(noise[0], noise[1])
    trainer.train_step(n_eval_ticks=2, world_epoch=2)
    expanded = noise[None].expand(2 * trainer.pairs_per_batch, -1, -1, -1).reshape(
        trainer.runner.E, 1, m.C)
    torch.testing.assert_close(trainer.runner.impact_obs, expanded, rtol=0, atol=0)
    resumed = baseline_trainer(path)
    resumed.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(resumed.world_impact_obs, noise, rtol=0, atol=0)


def test_gpu_policies_load_into_the_cpu_inference_world(baseline_project, tmp_path):
    """Exported policies have the CPU world's input width."""
    _, data = baseline_project
    path = observable_project(data, tmp_path)
    trainer = baseline_trainer(path)
    trainer.export_policies(tmp_path / "policies")
    builder = EnvironmentBuilder(project_path=path, grid=(5, 6))
    env = builder(seed=1)
    assert env.observable_impact_vars == ["windfarm_noise"]
    assert env.grid.get_map("windfarm_noise").max() > 0
    for d, fid in enumerate(env.dm_ids):
        state = torch.load(tmp_path / "policies" / f"policy_{fid}.pth",
                           weights_only=False)["state_dict"]
        first = next(v for k, v in state.items() if k.endswith("weight"))
        assert first.shape[1] == int(env.per_dm_in_dim[d])
