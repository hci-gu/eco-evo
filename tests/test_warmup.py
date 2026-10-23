"""World warm-up before the scored rollout, and the log-uniform start draw (section 200).

Contracts asserted:

1. ``warmup_ticks`` converts days at the run's tick length; 0 is off and a
   negative or non-finite value is refused.
2. ``run_warmup`` advances the env and clears the observation and
   action-statistic accumulators, so the scored rollout reports only its
   own ticks; with a seed it runs on its own stream and restores the
   global one.
3. CPU trainer: the warm-up runs with the UNPERTURBED weights on a
   stream seeded from the pair seed, so two candidates with different
   weights and different global RNG histories reach the same state
   before their reward starts (sequential path and worker task), and the
   candidate's weights are back in place afterwards.
4. GPU trainer: every candidate starts from its world's warmed-up state
   (identical b0 across candidates), the warm state does not depend on
   the perturbations, and the scored calendar start is the sampled start
   plus the warm-up.
5. The start totals are drawn log-uniformly on both engines: inside the
   range, median near the geometric mean (the linear draw's median is the
   arithmetic mean), a non-positive lower bound keeps the uniform draw.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config.config_loader import _sample_total_biomass
from lib.gpu.config import EnvironmentBuilder, ProjectSpec
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.random import fold_in
from lib.gpu.spawn import WorldSpawner
from lib.gpu.trainer import TensorARSTrainer
from lib.runners import parallel_worker, trainer as cpu_trainer
from lib.runners.trainer import ARSTrainer
from lib.runners.warmup import run_warmup, warmup_ticks
from test_gpu_ecosystem import make_env


def test_warmup_ticks_follow_the_tick_length():
    assert warmup_ticks(60) == 240
    assert warmup_ticks(60, 3) == 480
    assert warmup_ticks(0) == 0
    assert warmup_ticks(None) == 0
    for bad in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            warmup_ticks(bad)


def _cpu_trainer(env):
    model = TensorEcosystem(env, "cpu")
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    trainer = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, n_workers=1,
                         hidden_dim=7)
    return trainer


def test_run_warmup_advances_and_clears_accumulators():
    env = make_env(True, True)
    trainer = _cpu_trainer(env)
    work = copy.deepcopy(env)
    work.policies = trainer.policies
    start = work.tick_count
    run_warmup(work, 3)
    assert work.tick_count == start + 3
    assert work._obs_sum is None and work._obs_count == 0
    assert work._action_entropy_sum is None


def _record_warm_states(monkeypatch, module):
    states = []
    original = module.run_warmup

    def recording(env, ticks, seed=None):
        original(env, ticks, seed=seed)
        states.append({f: env.fgs[f].biomass.copy() for f in env.fgs})
    monkeypatch.setattr(module, "run_warmup", recording)
    return states


def test_cpu_candidates_share_the_unperturbed_warm_state(monkeypatch):
    env = make_env(True, True)
    trainer = _cpu_trainer(env)
    trainer.warmup_ticks = 3
    states = _record_warm_states(monkeypatch, cpu_trainer)
    base = {f: trainer._get_weights(p) for f, p in trainer.policies.items()}
    rng = torch.Generator().manual_seed(4)
    fits = []
    for sign in (+1.0, -1.0):
        candidate = {f: w + sign * 0.5 * torch.randn(w.shape, generator=rng)
                     for f, w in base.items()}
        for f, w in candidate.items():
            trainer._set_weights(trainer.policies[f], w)
        # a different global stream before each task, as on two workers
        np.random.seed(11 if sign > 0 else 12)
        fit, _, _ = trainer._evaluate_coevo(env.dm_ids, 2, seed=7, warmup_weights=base)
        fits.append(fit)
        for f, w in candidate.items():   # the candidate's weights are restored
            torch.testing.assert_close(trainer._get_weights(trainer.policies[f]), w)
    assert len(states) == 2
    for f in states[0]:
        np.testing.assert_array_equal(states[0][f], states[1][f])
    assert fits[0] != fits[1]


def test_worker_task_warms_up_with_the_base_weights(monkeypatch):
    env = make_env(True, True)
    trainer = _cpu_trainer(env)
    monkeypatch.setattr(parallel_worker, "_POLICIES", trainer.policies)
    states = _record_warm_states(monkeypatch, parallel_worker)
    base = {f: trainer._get_weights(p).numpy().copy() for f, p in trainer.policies.items()}
    rng = np.random.default_rng(5)
    for sign in (+1.0, -1.0):
        wd = {f: w + sign * 0.5 * rng.standard_normal(w.shape) for f, w in base.items()}
        np.random.seed(11 if sign > 0 else 12)
        parallel_worker._evaluate_coevo_task(dict(
            fg_list=env.dm_ids, weights_dict=wd, n_ticks=2, alpha=1, beta=1, seed=7,
            integral_reward=True, legacy_reward=False,
            env_builder=lambda seed=None: copy.deepcopy(env),
            warmup_ticks=3, warmup_weights=base))
        for f, w in wd.items():
            np.testing.assert_allclose(
                trainer._get_weights(trainer.policies[f]).numpy(), w, rtol=0, atol=1e-6)
    assert len(states) == 2
    for f in states[0]:
        np.testing.assert_array_equal(states[0][f], states[1][f])


def test_seeded_warmup_restores_the_global_stream():
    env = make_env(True, True)
    trainer = _cpu_trainer(env)
    work = copy.deepcopy(env)
    work.policies = trainer.policies
    np.random.seed(3)
    expected = np.random.random()
    np.random.seed(3)
    run_warmup(work, 2, seed=9)
    assert np.random.random() == expected


def test_round_robin_refuses_the_warmup():
    env = make_env(True, True)
    trainer = _cpu_trainer(env)
    trainer.warmup_ticks = 2
    with pytest.raises(ValueError, match="co-evolution"):
        trainer.train_step(env.dm_ids[0], 2)


def _gpu_trainer(warm):
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6)))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=3,
                            worlds=1, seed=16, warmup_ticks=warm)


def test_gpu_candidates_start_from_the_warm_state():
    trainer = _gpu_trainer(4)
    m = trainer.model
    trainer.train_step(None, 2, 0)
    spawn = trainer.world_biomass.clone()
    warm = trainer.world_warm_biomass.clone()
    assert not torch.equal(spawn, warm)
    # every candidate's reward baseline is the warm state's DM totals
    b0 = trainer.runner.b0
    expected = warm[:, m.dm_index].sum(-1).double()
    torch.testing.assert_close(b0, expected.expand_as(b0))
    if m.calendar:
        assert int(trainer.world_scored_light[0]) == (int(trainer.world_light_start[0]) + 4) % m.light_period


def test_gpu_warm_state_ignores_the_perturbations():
    a, b = _gpu_trainer(4), _gpu_trainer(4)
    deltas_a = a.make_deltas()
    deltas_b = [torch.randn_like(d) * 3 for d in deltas_a]
    a.train_step(None, 2, 0, deltas=deltas_a)
    b.train_step(None, 2, 0, deltas=deltas_b)
    torch.testing.assert_close(a.world_warm_biomass, b.world_warm_biomass, rtol=0, atol=0)
    torch.testing.assert_close(a.world_warm_reserve, b.world_warm_reserve, rtol=0, atol=0)


def test_gpu_trainer_refuses_a_negative_warmup():
    with pytest.raises(ValueError):
        _gpu_trainer(-1)


def test_cpu_start_total_is_log_uniform():
    rng = np.random.default_rng(1)
    lo, hi = 1250, 5000
    draws = np.array([_sample_total_biomass(lo, hi, rng) for _ in range(20000)])
    assert draws.min() >= lo and draws.max() <= hi
    assert np.all(draws == np.round(draws))
    assert np.median(draws) == pytest.approx(math.sqrt(lo * hi), rel=0.03)
    # one octave each side of the geometric mean holds half of the mass
    assert np.mean(draws < math.sqrt(lo * hi)) == pytest.approx(0.5, abs=0.02)
    assert _sample_total_biomass(7, 7, rng) == 7.0
    zero = np.array([_sample_total_biomass(0, 10, rng) for _ in range(5000)])
    assert zero.min() == 0 and zero.max() == 10   # uniform fallback


def test_gpu_start_total_is_log_uniform():
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(5, 6)))
    model = TensorEcosystem(spec.env, "cpu")
    spawner = WorldSpawner(spec, model)
    keys = fold_in(torch.zeros(4000, dtype=torch.int64), torch.arange(4000))
    totals = spawner.biomass(keys).double().sum(-1)
    for fid in spec.spawn_order:
        lo, hi = spec.ranges[fid]
        if lo <= 0 or hi <= lo:
            continue
        t = totals[:, model.ids.index(fid)]
        assert float(t.min()) >= lo - 1e-3 and float(t.max()) <= hi + 1e-3
        assert float(t.median()) == pytest.approx(math.sqrt(lo * hi), rel=0.06), fid


def test_seeded_warmup_matters_on_the_project_world():
    """On mareld2 the CPU tick draws from the global stream (FG order,
    seeding): unseeded, two RNG histories give two warm states; seeded
    from the pair seed, one."""
    from inference import build_env
    env = build_env("mareld2.yaml", (6, 6), seed=1, verbose=False,
                    apply_natural_mortality=True, mass_balance=True)
    env.build_static_caches()
    trainer = _cpu_trainer(env)
    for seed, same in ((None, False), (7, True)):
        states = []
        for history in (11, 12):
            work = copy.deepcopy(env)
            work.policies = trainer.policies
            np.random.seed(history)
            run_warmup(work, 4, seed=seed)
            states.append({f: work.fgs[f].biomass.copy() for f in work.fgs})
        equal = all(np.array_equal(states[0][f], states[1][f]) for f in states[0])
        assert equal is same, seed
