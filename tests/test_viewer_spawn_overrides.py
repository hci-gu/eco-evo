"""The viewer's start overrides (spawn dropdowns, start-biomass sliders).

1. A respawned or rescaled group keeps its reserve fill: the reserve is MJ
   per cell and used to stay on the old layout, so every respawned cell
   started near-empty and the group starved in its first days (fix101's
   mammals, 2026-11-15).
2. The spawn dropdowns reach the training worlds on both trainers.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import (_load_spawn_templates, apply_b0_overrides, apply_spawn_overrides,
                       project_with_spawn, spawn_override_configs)
from lib.config.config_loader import load_config
from lib.runners.training_progress import build_inference_env

PROJECT = "mareld2.yaml"
# The fix101 viewer choice: fish on Perlin, the mammals on their prey.
_SPAWN = {fid: {"mode": "perlin", "template": "Perlin12"}
          for fid in ("planktivorous_fish", "benthivorous_fish", "whiting")}
_SPAWN["porpoises"] = {"mode": "env_driven", "template": "Followfish"}
_SPAWN["seals"] = {"mode": "env_driven", "template": "FollowDemersalFish"}
# The same choice with the fish on colonies: the reference a changed
# pattern is judged against, whatever spawn the project itself has.
_OTHER = dict(_SPAWN, **{fid: {"mode": "colony", "template": "25Colonies"}
                         for fid in ("planktivorous_fish", "benthivorous_fish", "whiting")})


def _env(seed=3):
    return build_inference_env(dict(project=PROJECT, library="fgconfig/fg_library.yaml",
                                    grid=[6, 7], mortality=True, migration=True,
                                    mass_balance=True), seed)


def _fill(fg):
    return float(fg.energy_reserve.sum() / (fg.biomass.sum() * fg.max_energy_reserve))


# ---- 1. The reserve follows the tonnes -------------------------------------

def test_a_respawned_group_starts_every_cell_at_its_fill():
    env = _env()
    before = {fid: _fill(env.fgs[fid]) for fid in _SPAWN}
    layout = {fid: env.fgs[fid].biomass.copy() for fid in _SPAWN}
    apply_spawn_overrides(env, _SPAWN, _load_spawn_templates(PROJECT), seed=3)
    for fid, fill in before.items():
        fg = env.fgs[fid]
        assert not np.allclose(fg.biomass, layout[fid])
        occupied = fg.biomass > 0
        assert fg.energy_level[occupied] == pytest.approx(np.full(occupied.sum(), fill), rel=1e-4)
        # Nothing on a cell the group left.
        assert np.all(fg.energy_reserve[~occupied] == 0)


def test_a_rescaled_group_keeps_each_cells_fill():
    env = _env()
    for fid, factor in (("porpoises", 2.5), ("planktivorous_fish", 0.4)):
        fg = env.fgs[fid]
        level = fg.energy_level.copy()
        apply_b0_overrides(env, {fid: factor * float(fg.biomass.sum())})
        assert fg.energy_level == pytest.approx(level, rel=1e-4, abs=1e-6)


def test_an_empty_group_gets_its_start_fill():
    env = _env()
    fg = env.fgs["porpoises"]
    apply_b0_overrides(env, {"porpoises": 0.0})
    assert fg.biomass.sum() == 0 and fg.energy_reserve.sum() == 0
    apply_b0_overrides(env, {"porpoises": 2.0})
    centre = fg.start_fill_centre()
    assert _fill(fg) == pytest.approx(0.7 if centre is None else centre, rel=1e-4)


def test_a_refuge_keeps_its_accessible_share_and_a_pool_stays_put():
    env = _env()
    infauna, phyto = env.fgs["benthic_infauna"], env.fgs["phytoplankton"]
    share = infauna.energy_reserve.sum() / infauna.biomass.sum()
    pool = phyto.energy_reserve.copy()
    apply_b0_overrides(env, {"benthic_infauna": 2.0 * float(infauna.biomass.sum()),
                             "phytoplankton": 3.0 * float(phyto.biomass.sum())})
    assert infauna.energy_reserve.sum() / infauna.biomass.sum() == pytest.approx(share, rel=1e-4)
    # The nutrient pool is water, not tissue.
    assert np.array_equal(phyto.energy_reserve, pool)
    apply_spawn_overrides(env, {"benthic_infauna": {"mode": "perlin", "template": "Perlin12"},
                                "phytoplankton": {"mode": "perlin", "template": "Perlin12"}},
                          _load_spawn_templates(PROJECT), seed=3)
    assert infauna.energy_reserve == pytest.approx(infauna.biomass * infauna.exposed_share, rel=1e-4)
    assert np.array_equal(phyto.energy_reserve, pool)


# ---- 2. The spawn dropdowns as spawn blocks --------------------------------

def test_only_known_templates_become_spawn_blocks():
    templates = _load_spawn_templates(PROJECT)
    configs = spawn_override_configs(
        {"porpoises": {"mode": "env_driven", "template": "Followfish"},
         "seals": {"mode": "env_driven", "template": None},      # the project's
         "whiting": {"mode": "perlin", "template": "NoSuchTemplate"}},
        templates)
    assert list(configs) == ["porpoises"]
    assert configs["porpoises"]["mode"] == "env_driven"
    assert configs["porpoises"]["refs"][0]["name"] == "planktivorous_fish"


def test_the_project_copy_carries_the_blocks_and_refuses_unknown_groups():
    project = load_config(PROJECT)
    block = {"mode": "uniform"}
    changed = project_with_spawn(project, {"porpoises": block})
    entries = {e["group_id"]: e for e in changed["decision_makers"]}
    assert entries["porpoises"]["spawn"] == block
    assert project_with_spawn(project, {}) == project
    # The manifest itself is untouched.
    original = {e["group_id"]: e for e in project["decision_makers"]}["porpoises"]
    assert original.get("spawn") != block
    with pytest.raises(ValueError):
        project_with_spawn(project, {"no_such_group": block})


# ---- 3. The training worlds follow them ------------------------------------

def _correlation(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def test_the_gpu_training_worlds_spawn_with_the_viewers_templates():
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.ecosystem import TensorEcosystem
    from lib.gpu.spawn import WorldSpawner
    spec = ProjectSpec(EnvironmentBuilder(project_path=PROJECT, grid=(12, 12)))
    model = TensorEcosystem(spec.env, "cpu")
    spawner = WorldSpawner(spec, model)
    keys = torch.arange(4)
    plain = spawner.biomass(keys)
    configs = spawn_override_configs(_SPAWN, _load_spawn_templates(PROJECT))
    assert spawner.set_overrides(configs)
    assert not spawner.set_overrides(dict(configs))      # unchanged
    # Each env_driven group spawns after the groups it follows.
    order = list(spawner.order)
    assert order.index("porpoises") > order.index("planktivorous_fish")
    assert order.index("seals") > max(order.index("benthivorous_fish"), order.index("whiting"))
    spawned = spawner.biomass(keys)
    i = {fid: model.ids.index(fid) for fid in model.ids}
    # The chosen template sets the pattern.
    assert spawner.set_overrides(spawn_override_configs(_OTHER, _load_spawn_templates(PROJECT)))
    other = spawner.biomass(keys)
    assert not torch.equal(spawned[:, i["planktivorous_fish"]], other[:, i["planktivorous_fish"]])
    # The totals are the world's draw, whatever the pattern.
    torch.testing.assert_close(spawned.double().sum(-1), plain.double().sum(-1), rtol=1e-5, atol=1e-4)
    # Untouched groups keep their fields.
    assert torch.equal(spawned[:, i["zooplankton"]], plain[:, i["zooplankton"]])
    # The porpoises sit on the (new) planktivore field.
    for w in range(keys.numel()):
        porp, prey = spawned[w, i["porpoises"]], spawned[w, i["planktivorous_fish"]]
        assert torch.all(prey[porp > 0] > 0)
        assert prey[porp > 0].mean() > prey.mean()
    # {} restores the project's spawn exactly.
    assert spawner.set_overrides({})
    assert torch.equal(spawner.biomass(keys), plain)


def test_a_changed_spawn_redraws_the_gpu_worlds_within_an_epoch():
    from types import SimpleNamespace
    from lib.gpu.trainer import TensorARSTrainer
    calls = []
    fake = SimpleNamespace(last_world_epoch=7,
                           spawner=SimpleNamespace(set_overrides=lambda c: calls.append(c) or bool(c)))
    assert TensorARSTrainer.set_spawn_overrides(fake, {"porpoises": {"mode": "uniform"}})
    assert fake.last_world_epoch is None
    fake.last_world_epoch = 7
    assert not TensorARSTrainer.set_spawn_overrides(fake, {})
    assert fake.last_world_epoch == 7


def test_the_gpu_viewer_hands_the_dropdowns_to_the_trainer(capsys):
    from types import SimpleNamespace
    from lib.gpu.visual import GPUTrainingVisualizer
    visual = GPUTrainingVisualizer.__new__(GPUTrainingVisualizer)
    visual.spawn_templates = _load_spawn_templates(PROJECT)
    visual.training_spawn_overrides = {}
    visual.pump = lambda: None
    chosen = {"porpoises": {"mode": "env_driven", "template": "Followfish"},
              "seals": {"mode": "env_driven", "template": None}}
    visual.viz = SimpleNamespace(get_spawn_overrides=lambda: chosen)
    seen = []
    trainer = SimpleNamespace(set_spawn_overrides=lambda c: seen.append(c) or len(seen) == 1)
    active = visual.training_spawn(trainer)
    assert active == {"porpoises": {"mode": "env_driven", "template": "Followfish"}}
    assert list(seen[0]) == ["porpoises"] and seen[0]["porpoises"]["refs"]
    assert "Training worlds now spawn with porpoises Followfish" in capsys.readouterr().out
    assert visual.training_spawn(trainer) == active        # no change, no print
    assert capsys.readouterr().out == ""


def test_the_cpu_training_builder_spawns_with_the_blocks_in_every_world():
    from train import _make_env_builder
    configs = spawn_override_configs(_SPAWN, _load_spawn_templates(PROJECT))
    common = dict(grid_size=(12, 12), project_path=PROJECT, spawn_seed=5,
                  impact_vars=[], impact_ranges={})
    plain = _make_env_builder(**common)(seed=11)
    builder = _make_env_builder(spawn_configs=configs, **common)
    spawned = builder(seed=11)
    # with_world keeps the blocks (the multi-world path).
    sibling = builder.with_world(1, 6)(seed=11)
    for env in (spawned, sibling):
        prey, porp = env.fgs["planktivorous_fish"].biomass, env.fgs["porpoises"].biomass
        assert np.all(prey[porp > 0] > 0)
        assert prey[porp > 0].mean() > prey.mean()
    # The chosen template sets the pattern.
    other = _make_env_builder(
        spawn_configs=spawn_override_configs(_OTHER, _load_spawn_templates(PROJECT)), **common)(seed=11)
    assert not np.allclose(spawned.fgs["planktivorous_fish"].biomass,
                           other.fgs["planktivorous_fish"].biomass)
    assert spawned.fgs["porpoises"].biomass.sum() == pytest.approx(
        plain.fgs["porpoises"].biomass.sum(), rel=1e-5)
    # A training world draws its reserve per cell around the start fill,
    # on the new layout.
    level = spawned.fgs["porpoises"].energy_level[spawned.fgs["porpoises"].biomass > 0]
    assert np.all(level > 0)
