"""Warm-up with re-anchoring (section 274).

A fresh world is not a settled one: in fix101 the planktivores left their
spawn cells in the first weeks and the porpoises, spawned on them, lost half
their tonnes, so every scored rollout began at ~0.4 of the drawn total. After
the warm-up the groups listed in ``simulation_settings.warmup.reanchor`` are
scaled back to their start totals on the settled layout; the probes and
inference.py run the same warm-up before they score.

Contracts asserted:

1. The loader marks the listed groups, refuses an unknown id and a group whose
   level an outside exchange sets, and skips a muted one.
2. ``reanchor`` restores each listed total with the warmed layout, each
   cell's fill and the juvenile share; a group that died out gets its start
   fields back; other groups are untouched. ``run_warmup`` does it on both
   CPU training paths.
3. ``settle`` makes the settled state tick 0 on the same calendar day (the
   tensor engine uploads the same year tick), clears the books, restores the
   env's own policies and observation statistics, and is deterministic for
   one seed (the viewer's random baselines start from the main env's state).
4. GPU trainer: the warm state's listed totals are the world's drawn totals.
5. The long-run probe records the warm-up and settles before it scores; a
   history from before keeps comparing equal without a warm-up.
6. inference.py takes the run's training warm-up by default.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import load_config, load_project_config
from lib.runners.training_progress import build_inference_env
from lib.runners.warmup import reanchor, reanchor_snapshot, run_warmup, settle

PROJECT = "mareld2.yaml"
LISTED = {"benthivorous_fish", "whiting", "porpoises", "seals"}


def _env(seed=3, grid=(6, 7)):
    return build_inference_env(dict(project=PROJECT, library="fgconfig/fg_library.yaml",
                                    grid=list(grid), mortality=True, migration=True,
                                    mass_balance=True), seed)


def _with_reanchor(ids):
    project = copy.deepcopy(load_config(PROJECT))
    project["simulation_settings"]["warmup"] = {"reanchor": list(ids)}
    return project


def _policies(env):
    from lib.gpu.ecosystem import TensorEcosystem
    from lib.runners.trainer import ARSTrainer
    model = TensorEcosystem(env, "cpu")
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    torch.manual_seed(0)
    return ARSTrainer(lambda seed=None: env, params, n_workers=1, hidden_dim=7).policies


def _total(fg):
    return float(np.asarray(fg.biomass, dtype=np.float64).sum())


# ---- 1. The loader ----------------------------------------------------------

def test_the_manifest_marks_the_groups_no_exchange_sets():
    fgs = load_project_config(PROJECT, grid_size=(6, 7), seed=1)[0]
    assert {f for f, g in fgs.items() if g.warmup_reanchor} == LISTED


def test_unknown_and_exchange_set_groups_are_refused_and_muted_ones_skipped():
    for bad in ("no_such_group", "planktivorous_fish", "mackerel", "predatory_fish", "krill"):
        with pytest.raises(ValueError):
            load_project_config(PROJECT, grid_size=(6, 7), seed=1,
                                project_config=_with_reanchor([bad]))
    # seabirds are muted in mareld2.yaml: absent, so nothing to mark.
    fgs = load_project_config(PROJECT, grid_size=(6, 7), seed=1,
                              project_config=_with_reanchor(["seabirds", "seals"]))[0]
    assert "seabirds" not in fgs
    assert {f for f, g in fgs.items() if g.warmup_reanchor} == {"seals"}


# ---- 2. Re-anchoring ---------------------------------------------------------

def test_reanchor_restores_the_totals_on_the_warmed_layout():
    env = _env()
    snap = reanchor_snapshot(env)
    assert set(snap) == LISTED
    start = {f: _total(env.fgs[f]) for f in env.fgs}
    # A "warm-up": porpoises thinned unevenly, benthivores halved, seals gone,
    # zooplankton changed (not listed).
    porp = env.fgs["porpoises"]
    porp.biomass[...] = porp.biomass * np.linspace(0.1, 0.6, porp.biomass.size).reshape(porp.biomass.shape)
    porp.energy_reserve[...] = porp.biomass * 0.8 * porp.max_energy_reserve
    benth = env.fgs["benthivorous_fish"]
    benth.biomass[...] *= 0.5
    benth.energy_reserve[...] *= 0.5
    benth.juvenile = benth.juvenile * 0.5
    seals = env.fgs["seals"]
    seals.biomass[...] = 0.0
    seals.energy_reserve[...] = 0.0
    env.fgs["zooplankton"].biomass[...] *= 2.0
    warmed_layout = porp.biomass / porp.biomass.sum()
    juv_share = benth.juvenile.sum() / benth.biomass.sum()
    factors = reanchor(env, snap)
    for fid in LISTED:
        assert _total(env.fgs[fid]) == pytest.approx(start[fid], rel=1e-5)
    occupied = porp.biomass > 0
    assert porp.biomass / porp.biomass.sum() == pytest.approx(warmed_layout, rel=1e-5)
    assert porp.energy_level[occupied] == pytest.approx(np.full(occupied.sum(), 0.8), rel=1e-5)
    assert benth.juvenile.sum() / benth.biomass.sum() == pytest.approx(juv_share, rel=1e-5)
    assert factors["seals"] is None
    assert np.array_equal(seals.biomass, snap["seals"][1])
    assert np.array_equal(seals.energy_reserve, snap["seals"][2])
    assert _total(env.fgs["zooplankton"]) == pytest.approx(2.0 * start["zooplankton"], rel=1e-5)


def test_run_warmup_reanchors_the_listed_groups():
    env = _env()
    env.policies = _policies(env)
    env.rebuild_batched_weights()
    start = {f: _total(env.fgs[f]) for f in env.fgs}
    run_warmup(env, 12, seed=5)
    for fid in LISTED:
        assert _total(env.fgs[fid]) == pytest.approx(start[fid], rel=1e-5)
    moved = [f for f in env.fgs if f not in LISTED and _total(env.fgs[f]) != pytest.approx(start[f], rel=1e-6)]
    assert moved   # the rest of the web keeps what the warm-up did


# ---- 3. settle ---------------------------------------------------------------

def test_settle_rebases_the_calendar_and_clears_the_books():
    from lib.environments.ecosystem_env.interactions import light_index
    from lib.environments.ecosystem_env.loss_accounting import LOSS_CAUSES
    from lib.gpu.ecosystem import TensorEcosystem
    env = _env()
    policies = _policies(env)
    own_mean = getattr(env, "obs_mean", None)
    day = int(env.light_start)
    settle(env, 12, seed=5, policies=policies)
    assert env.tick_count == 0
    assert int(light_index(env)) == (day + 12) % env.light_period
    for _name, attr in LOSS_CAUSES:
        assert all(v == 0.0 for v in getattr(env, attr).values())
    assert all(not row for row in env.intake_by_pred_prey.values())
    # The env's own policies and statistics are back.
    assert env.policies == {} and getattr(env, "obs_mean", None) is own_mean
    # The tensor engine uploads the same year tick.
    model = TensorEcosystem(env, "cpu")
    assert int(model.light_starts([env])[0]) == int(light_index(env))


def test_settle_is_deterministic_for_one_seed():
    a, b = _env(), _env()
    policies = _policies(a)
    settle(a, 12, seed=5, policies=policies)
    settle(b, 12, seed=5, policies=policies)
    for fid in a.fgs:
        assert np.array_equal(a.fgs[fid].biomass, b.fgs[fid].biomass)
        assert np.array_equal(a.fgs[fid].energy_reserve, b.fgs[fid].energy_reserve)


def test_settle_without_ticks_changes_nothing():
    env = _env()
    before = {f: env.fgs[f].biomass.copy() for f in env.fgs}
    settle(env, 0, seed=5, policies={})
    assert env.tick_count == 0
    assert all(np.array_equal(before[f], env.fgs[f].biomass) for f in env.fgs)


# ---- 4. GPU trainer ----------------------------------------------------------

def test_reanchor_worlds_scales_restores_and_leaves_the_rest():
    from lib.gpu.trainer import reanchor_worlds
    W, G, C = 2, 3, 4
    start_b = torch.ones((W, G, C))
    start_r = torch.full((W, G, C), 2.0)
    start_j = torch.full((W, G, C), 0.5)
    warm_b = torch.tensor([[[1., 1, 0, 0], [0.5, 0.5, 0.5, 0.5], [3, 3, 3, 3]],
                           [[0., 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1]]])
    warm_r = warm_b * 3
    warm_j = warm_b * 0.25
    target = start_b.double().sum(-1)
    mask = torch.tensor([True, True, False])
    tissue = torch.tensor([True, False, True])
    b, r, j = reanchor_worlds(target, warm_b, warm_r, warm_j, (start_b, start_r, start_j), mask, tissue)
    torch.testing.assert_close(b[:, :2].sum(-1), target[:, :2].float())
    torch.testing.assert_close(b[0, 0], torch.tensor([2., 2, 0, 0]))       # layout kept
    torch.testing.assert_close(r[0, 0], b[0, 0] * 3)                       # fill kept (tissue)
    torch.testing.assert_close(r[0, 1], warm_r[0, 1])                      # no tissue reserve
    torch.testing.assert_close(j[0, 1], b[0, 1] * 0.25)                    # juvenile share kept
    torch.testing.assert_close(b[1, 0], start_b[1, 0])                     # died out: restored
    torch.testing.assert_close(r[1, 0], start_r[1, 0])
    torch.testing.assert_close(j[1, 0], start_j[1, 0])
    torch.testing.assert_close(b[:, 2], warm_b[:, 2])                      # not listed


def test_the_gpu_warm_state_starts_the_listed_groups_at_the_drawn_totals():
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    spec = ProjectSpec(EnvironmentBuilder(project_path=PROJECT, grid=(5, 6)))
    trainer = TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=3,
                               worlds=2, seed=16, warmup_ticks=8)
    trainer.train_step(None, 2, 0)
    m = trainer.model
    drawn = trainer.world_biomass.double().sum(-1)
    warm = trainer.world_warm_biomass.double().sum(-1)
    listed = [m.ids.index(f) for f in LISTED]
    torch.testing.assert_close(warm[:, listed], drawn[:, listed], rtol=1e-5, atol=1e-6)
    others = [g for g in range(m.G) if g not in listed]
    assert not torch.allclose(warm[:, others], drawn[:, others], rtol=1e-6)


# ---- 5. The long-run probe ---------------------------------------------------

def test_the_probe_config_records_the_warmup_and_old_histories_still_compare():
    from lib.runners.training_progress import comparable_config
    base = dict(project=PROJECT, grid=[6, 7], mortality=True, migration=True, mass_balance=True)
    old = comparable_config(dict(base))
    assert old["warmup_ticks"] == 0 and old["warmup_reanchor"] == []
    # No warm-up: the manifest's list does not matter.
    assert comparable_config(dict(base, warmup_ticks=0,
                                  warmup_reanchor=sorted(LISTED))) == old
    warm = comparable_config(dict(base, warmup_ticks=240, warmup_reanchor=sorted(LISTED)))
    assert warm != old and warm["warmup_reanchor"] == sorted(LISTED)


def test_the_probe_worker_settles_before_it_scores(monkeypatch):
    import lib.runners.training_progress as tp
    calls = []
    monkeypatch.setattr(tp, "build_inference_env", lambda config, seed: _FakeEnv())
    monkeypatch.setattr(tp, "settle", lambda env, ticks, seed=None: calls.append(("settle", ticks, seed)))
    monkeypatch.setattr(tp, "run_probe", lambda env, ticks, lower, upper: calls.append(("probe", ticks)) or "ok")
    threads = torch.get_num_threads()
    try:
        config = dict(project=None, warmup_ticks=240)
        assert tp._probe_in_process(config, 7, 1.0, 4380, 0.3, 3.0,
                                    tp._pack_policies({}), None, None, threads) == "ok"
    finally:
        torch.set_num_threads(threads)
    assert calls == [("settle", 240, 7), ("probe", 4380)]


class _FakeEnv:
    def rebuild_batched_weights(self):
        pass


# ---- 6. inference.py ----------------------------------------------------------

def test_inference_takes_the_runs_training_warmup(tmp_path):
    from inference import run_warmup_days
    assert run_warmup_days(str(tmp_path)) == 0.0
    (tmp_path / "biomass.jsonl").write_text(json.dumps({"__meta__": {"warmup_days": 30}}) + "\n")
    assert run_warmup_days(str(tmp_path)) == 30.0
    (tmp_path / "gpu_run.json").write_text(json.dumps({"warmup_days": 60.0}))
    assert run_warmup_days(str(tmp_path)) == 60.0


# ---- 7. The warm-up on the tensor engine (section 277) ------------------------

@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA GPU required"))])
def test_settle_on_the_tensor_engine_matches_the_cpu_warmup(device):
    """inference.py --engine gpu warms up on the tensor engine: the same
    world to float32 rounding, the listed totals re-anchored, tick 0 on the
    same day, the books cleared, progress reported."""
    from lib.environments.ecosystem_env.interactions import light_index
    from lib.environments.ecosystem_env.loss_accounting import LOSS_CAUSES
    cpu, gpu = _env(), _env()
    policies = _policies(cpu)
    start = {f: _total(gpu.fgs[f]) for f in gpu.fgs}
    day = int(gpu.light_start)
    seen = []
    settle(cpu, 8, seed=5, policies=policies)
    settle(gpu, 8, seed=5, policies=policies, engine="gpu", device=device,
           progress=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1] == (8, 8)
    assert gpu.tick_count == 0
    assert int(light_index(gpu)) == (day + 8) % gpu.light_period == int(light_index(cpu))
    for _name, attr in LOSS_CAUSES:
        assert all(v == 0.0 for v in getattr(gpu, attr).values())
    for fid in LISTED:
        assert _total(gpu.fgs[fid]) == pytest.approx(start[fid], rel=1e-5)
    for fid in gpu.fgs:
        np.testing.assert_allclose(gpu.fgs[fid].biomass, cpu.fgs[fid].biomass,
                                   rtol=1e-3, atol=1e-4, err_msg=fid)
        np.testing.assert_allclose(gpu.fgs[fid].energy_reserve, cpu.fgs[fid].energy_reserve,
                                   rtol=1e-3, atol=1e-2, err_msg=fid)
