"""Impacts on the GPU engine and baseline impacts (section 161).

The GPU engine used to skip impacts altogether. It now mirrors the
reference: direct impact mortality before predation, the
``1 + sum(energy_factor)`` surcharge on every action cost, from per-world
fields that ``TensorEcosystem.impact_fields`` turns into rates once per
world, and the observable impacts as raw input channels after the
observed groups (own cell and the four neighbours, the light channel
still last). An impact with a ``baseline_map`` (the Impact Editor's
today's field, e.g. fishing effort per cell) gets that map in every
training world, observable or not; inference reads only its own
``inference.impact_maps``, and no map means no impact.
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
from lib.environments.ecosystem_env.impacts import interp_impact
from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from lib.gpu.trainer import TensorARSTrainer
from lib.world.impact_maps import project_baseline_maps, sample_impact_maps
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
               observable=None, daylight=None, ndm=False):
    """``ndm`` also trawls the non decision maker "c" (benthos-like)."""
    impacts = {"a": {"trawl": TRAWL, "noise": NOISE},
               "b": {"trawl": TRAWL, "noise": OFF}}
    if ndm:
        impacts["c"] = {"trawl": TRAWL}
    env = make_env(migration, mortality, min_split=0, extinction_factor=0,
                   impacts=impacts, observable_impacts=observable, daylight=daylight)
    rng = np.random.default_rng(7)
    # The trawl field runs past the table's last point: clamped, as on CPU.
    fields = {"trawl": rng.uniform(0, 12, (env.H, env.W)),
              "noise": rng.uniform(0, 6, (env.H, env.W))}
    for impact_id in maps:
        env.grid.add_map(impact_id, fields[impact_id].astype(np.float32))
    return env


@pytest.mark.parametrize("migration,mortality", [(False, False), (True, True)])
@pytest.mark.parametrize("maps", [("trawl", "noise"), ("trawl",)])
@pytest.mark.parametrize("ndm", [False, True])
def test_impacts_match_reference(device, migration, mortality, maps, ndm):
    """Full ticks with impact mortality and cost, against the CPU tick.

    ``("trawl",)`` leaves the noise field out: both engines skip a
    missing field rather than reading it as zero. ``ndm`` trawls the
    logistic group "c" too: only its biomass factor acts.
    """
    model = _assert_parity(impact_env(migration, mortality, maps, ndm=ndm), device)
    assert model.has_impacts and model.impact_ids == ("noise", "trawl")
    c = model.ids.index("c")
    assert bool(model.impact_loss_tables[c]) == ndm


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
    ga, gb = model.ids.index("a"), model.ids.index("b")
    a, b = env.dm_ids.index("a"), env.dm_ids.index("b")
    assert loss.shape[1] == model.G and cost.shape[1] == model.D
    np.testing.assert_allclose(loss[0, ga].cpu(), bf_t + bf_n, rtol=1e-6)
    np.testing.assert_allclose(cost[0, a].cpu(), ef_t + ef_n, rtol=1e-6)
    # "b" is not affected by the noise: its table is switched off.
    np.testing.assert_allclose(loss[0, gb].cpu(), bf_t, rtol=1e-6)
    np.testing.assert_allclose(cost[0, b].cpu(), ef_t, rtol=1e-6)
    assert not loss[0, model.ids.index("c")].any()


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


def test_trawling_a_logistic_group_lowers_it(device):
    env = impact_env(ndm=True)
    model = TensorEcosystem(env, device)
    b, r, _ = model.import_state([env])
    probs = torch.full((1, model.D, model.A, model.C), 1.0 / model.A, device=model.device)
    loss, cost = model.import_impacts([env])
    on = model.step(b, r, probs, model.tensor(0), torch.zeros_like(b), impacts=(loss, cost))
    off = model.step(b, r, probs, model.tensor(0), torch.zeros_like(b))
    c = model.ids.index("c")
    assert (on[0][:, c].sum() < off[0][:, c].sum()).item()


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


def test_baseline_map_is_the_training_field():
    ranges = {"fishing": (0.0, 2.0), "noise": (0.0, 140.0), "rotor": (0.0, 1.0)}
    today = np.arange(42, dtype=np.float32).reshape(6, 7)
    seen = np.full((6, 7), 7.0, dtype=np.float32)
    maps = sample_impact_maps(["fishing", "noise", "rotor", "seen"],
                              dict(ranges, seen=(0.0, 100.0)), (6, 7), seed=3,
                              observable_impact_vars=["noise", "seen"],
                              baseline_maps={"fishing": today, "seen": seen})
    np.testing.assert_array_equal(maps["fishing"], today)
    np.testing.assert_array_equal(maps["seen"], seen), "a baseline map replaces the spawned field"
    assert maps["noise"].max() > 0, "an observable impact is still spawned"
    assert not maps["rotor"].any(), "no baseline map and not observable: absent"
    legacy = sample_impact_maps(["rotor"], ranges, (6, 7), seed=3)
    assert legacy["rotor"].max() > 0, "observable None keeps the sample-everything rule"


def test_project_baseline_maps_resolve_resample_and_skip(tmp_path):
    np.savez(tmp_path / "effort.npz", effort=np.arange(16, dtype=np.float32).reshape(4, 4))
    project = {"impact_variables": [
        {"impact_id": "pelagic_trawling", "baseline_map": "effort.npz"},
        {"impact_id": "bottom_trawling", "baseline_map": "effort.npz", "muted": True},
        {"impact_id": "rotor", "baseline_map": "missing.npz"},
        {"impact_id": "windfarm_noise", "observable": True}]}
    maps = project_baseline_maps(project, str(tmp_path / "p.yaml"), (8, 8))
    assert set(maps) == {"pelagic_trawling"}
    assert maps["pelagic_trawling"].shape == (8, 8)
    assert maps["pelagic_trawling"][0, 0] == 0 and maps["pelagic_trawling"][7, 7] == 15


@pytest.fixture
def baseline_project(tmp_path):
    """mareld2 with pelagic trawling un-muted and a baseline effort map."""
    with open(ROOT / "mareld2.yaml", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    effort = np.random.default_rng(4).uniform(0, 1000, (60, 60)).astype(np.float32)
    np.savez(tmp_path / "effort.npz", pelagic_trawling=effort)
    for iv in data["impact_variables"]:
        if iv["impact_id"] == "pelagic_trawling":
            iv["muted"] = False
            iv["baseline_map"] = "effort.npz"
    data.get("inference", {}).get("impact_maps", {}).pop("rotor", None)
    data["simulation_settings"]["depth_map"] = str(ROOT / data["simulation_settings"]["depth_map"])
    path = tmp_path / "baseline.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path), data


def test_inference_reads_only_its_own_maps(baseline_project, tmp_path):
    """The baseline map is a training map; inference needs its own entry."""
    path, data = baseline_project
    maps = load_inference_impact_maps(path, ["pelagic_trawling", "windfarm_noise"],
                                      8, 9, verbose=False)
    assert not maps["pelagic_trawling"].any() and not maps["windfarm_noise"].any()
    data = copy.deepcopy(data)
    data.setdefault("inference", {}).setdefault("impact_maps", {})["pelagic_trawling"] = "effort.npz"
    with_map = tmp_path / "with_map.yaml"
    with open(with_map, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    maps = load_inference_impact_maps(str(with_map), ["pelagic_trawling"], 8, 9, verbose=False)
    assert maps["pelagic_trawling"].max() > 0


def baseline_trainer(path, **kwargs):
    spec = ProjectSpec(EnvironmentBuilder(project_path=path, grid=(5, 6)))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2,
                            worlds=2, seed=16, hidden_dim=6, **kwargs)


def test_trainer_gives_every_world_the_baseline_map(baseline_project):
    path, _ = baseline_project
    trainer = baseline_trainer(path)
    m = trainer.model
    assert m.has_impacts and "pelagic_trawling" in m.impact_ids
    trainer.refresh_worlds(epoch=3)
    herring, benthivorous_fish = m.ids.index("planktivorous_fish"), m.ids.index("benthivorous_fish")
    loss = trainer.world_impact_loss.clone()
    assert (loss[:, herring] > 0).any()
    # Pelagic trawls also take a little of the demersal F (sections
    # 161.14, 209: mackerel and horse mackerel in predatory_fish; 223.26:
    # whiting's industrial bycatch).
    predators, whiting = m.ids.index("predatory_fish"), m.ids.index("whiting")
    assert (loss[:, benthivorous_fish] > 0).any() and (loss[:, predators] > 0).any()
    assert (loss[:, whiting] > 0).any()
    assert loss[:, benthivorous_fish].sum() < loss[:, herring].sum()
    others = [g for g, fid in enumerate(m.ids)
              if fid not in ("planktivorous_fish", "benthivorous_fish", "predatory_fish", "whiting")]
    assert not loss[:, others].any()
    torch.testing.assert_close(loss[0], loss[1], rtol=0, atol=0), "one map for every world"
    expected = m.impact_fields([trainer.spec.baseline_maps])[0]
    torch.testing.assert_close(loss[0], expected[0], rtol=0, atol=0)
    trainer.train_step(n_eval_ticks=2, world_epoch=3)
    expanded = loss[None].expand(2 * trainer.pairs_per_batch, -1, -1, -1).reshape(
        trainer.runner.E, m.impact_rows, m.C)
    torch.testing.assert_close(trainer.runner.impact_loss, expanded, rtol=0, atol=0)
    resumed = baseline_trainer(path)
    resumed.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(resumed.world_impact_loss, loss, rtol=0, atol=0)


def test_without_baseline_map_a_hidden_impact_is_absent_in_training(baseline_project, tmp_path):
    _, data = baseline_project
    data = copy.deepcopy(data)
    for iv in data["impact_variables"]:
        iv.pop("baseline_map", None)
    path = tmp_path / "no_map.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    trainer = baseline_trainer(str(path))
    trainer.refresh_worlds(epoch=1)
    assert trainer.model.has_impacts
    assert not trainer.world_impact_loss.any()


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


def test_impact_fields_need_no_device_readback(baseline_project, tmp_path, monkeypatch):
    """Fresh observable fields every epoch, without reading the device back."""
    _, data = baseline_project
    trainer = baseline_trainer(observable_project(data, tmp_path))
    trainer.train_step(n_eval_ticks=2, world_epoch=1)
    first = trainer.world_impact_obs.clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor readback in numerical training loop")
    for name in ("cpu", "numpy", "item", "tolist", "__float__", "__int__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    trainer.train_step(n_eval_ticks=2, world_epoch=2)
    trainer.train_step(n_eval_ticks=2)
    monkeypatch.undo()
    assert not torch.equal(trainer.world_impact_obs, first)
