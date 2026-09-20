"""The GPU mirror of the per-cell source-tracked reward (``--local_reward``).

The reference implementation lives in
``lib/environments/ecosystem_env/source_tracking.py`` and is covered by
``tests/test_local_reward.py``. What matters here is that the tensor
engine computes *the same quantity*, otherwise ``train_gpu.py`` would
optimise a different fitness than ``train.py``:

1. **Mass balance** - ``sum_c B(c,t+1) == sum_c Q(c,t+1)``. The tracked
   energy re-partitions the grid's end-of-tick energy; the whole
   construction stands or falls on that identity.
2. **Pure rest** - with no movement the tracking degenerates to the
   cell's own end-of-tick energy.
3. **Reference parity** - the rollout fitness equals the CPU trainer's
   for every metric/norm/theta combination.
4. **Neutrality** - with the flag off the tick is untouched, and the
   tracked path still traces as one graph (CUDA-graph execution).
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env.source_tracking import LocalRewardConfig
from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from lib.runners.population_stability import PopulationStability
from lib.runners.trainer import ARSTrainer
from test_gpu_ecosystem import DEVICES, make_env, sparse_env, uniform_move


CONFIGS = [
    LocalRewardConfig(),
    LocalRewardConfig(metric="ratio", norm="sum"),
    LocalRewardConfig(metric="ratio", norm="mean", theta=1.0),
    LocalRewardConfig(clip_lo=0.5, clip_hi=1.5, min_energy_factor=2.0),
    # ``grid`` divides by the constant cell count; the reference uses
    # ``env.H * env.W`` and this engine ``model.C``, so parity here also
    # asserts that the two agree on what "the grid" is.
    LocalRewardConfig(metric="log", norm="grid"),
]


def live_env(migration=False, mortality=False):
    """Fixture where biomass survives, so it actually moves and splits.

    The default fixture's ``min_split_biomass`` sits above its biomass,
    which both suppresses splits and lets the extinction sweep clear the
    grid - a tracking test would then compare zeros. The split
    suppression itself is covered by ``sparse_env`` below.
    """
    return make_env(migration, mortality, min_split=0.0, extinction_factor=0.0)


def tracked_step(model, state, probabilities, tick=0):
    return model.step(*state[:2], probabilities, model.tensor(tick), state[3],
                      torch.zeros_like(state[0]), track_source=True)


def random_probabilities(model, biomass, seed=5):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(1, model.D, model.C, model.A, generator=generator)
    return model.action_probabilities(logits.to(model.device), biomass,
                                      model.tensor(1.0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mortality", [False, True])
def test_tracked_energy_repartitions_the_end_of_tick_energy(device, mortality):
    """Mass balance: the shares neither create nor destroy energy."""
    env = live_env(mortality=mortality)
    model = TensorEcosystem(env, device)
    state = model.import_state([env])
    probabilities = random_probabilities(model, state[0])
    b, r, _, _, _, (start, tracked, frac_in) = tracked_step(model, state, probabilities)

    total = model.local_energy(b, r).double().sum()
    torch.testing.assert_close(tracked.double().sum(), total, rtol=2e-6, atol=2e-6)
    assert torch.isfinite(tracked).all() and (tracked >= 0).all()
    # ``start`` is the pre-predation state the policy observed, and no
    # move leaves the grid while migration is off.
    torch.testing.assert_close(start, model.local_energy(*state[:2]))
    torch.testing.assert_close(frac_in, torch.ones_like(frac_in))


@pytest.mark.parametrize("device", DEVICES)
def test_pure_rest_tracks_the_cell_itself(device):
    """No movement means every cell keeps exactly its own energy."""
    env = live_env()
    model = TensorEcosystem(env, device)
    state = model.import_state([env])
    rest = torch.zeros(1, model.D, model.A, model.C, device=model.device)
    rest[:, :, 4] = 1.0
    b, r, _, _, _, (_, tracked, _) = tracked_step(model, state, rest)
    torch.testing.assert_close(tracked, model.local_energy(b, r))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("migration", [False, True])
def test_rollout_fitness_matches_the_reference(device, config, migration):
    """The GPU fitness must be the same number as the CPU trainer's."""
    torch.set_num_threads(1)
    env = live_env(migration=migration, mortality=True)
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    reference = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, n_workers=1,
                           hidden_dim=7, local_reward=config)
    reference.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    reference.softmax_temperature = 1.0
    runner = RolloutRunner(model, bank, 1, 1, execution="eager",
                           obs_normalize=False, local_reward=config)
    state = model.import_state([env])
    ticks = 6
    runner.reset(state[0], state[1], state[3],
                 torch.zeros(1, dtype=torch.int64, device=model.device), ticks,
                 torch.zeros_like(runner.obs_mean), torch.ones_like(runner.obs_var),
                 model.tensor(1.0))
    runner.run(ticks)
    reward, _ = runner.results(ticks)
    expected, _, _ = reference._evaluate_coevo(model.dm_ids, ticks)
    for d, fid in enumerate(model.dm_ids):
        assert reward[0, d].item() == pytest.approx(expected[fid], rel=2e-3, abs=2e-4)
    assert (runner.occupancy > 0).any(), "no cell took part; the test is vacuous"


@pytest.mark.parametrize("device", DEVICES)
def test_tracking_follows_suppressed_splits(device):
    """A cancelled split has to be booked back onto the source cell.

    ``suppress_splits`` returns outflow to the cell it came from, so the
    tracked flow must move it from ``b_out`` into ``b_stay`` as well. Were
    it left in the outflow, the destination that never received it would
    be credited with the share and the mass balance would break - which
    is what this checks, on a fixture where the suppression actually
    fires.
    """
    env = sparse_env()
    model = TensorEcosystem(env, device)
    assert model.split_thr_any, "the fixture must exercise the suppression"
    state = model.import_state([env])
    probabilities, _ = uniform_move(env, model)
    b, r, _, _, _, (_, tracked, frac_in) = tracked_step(model, state, probabilities)
    total = model.local_energy(b, r).double().sum()
    torch.testing.assert_close(tracked.double().sum(), total, rtol=2e-6, atol=2e-6)
    assert total > 0, "everything died; the identity would be trivial"
    # The split is interior and every cancelled part stayed home, so no
    # tracked biomass leaves the grid.
    torch.testing.assert_close(frac_in, torch.ones_like(frac_in))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS[:2])
def test_rollout_fitness_matches_the_reference_with_suppression(device, config):
    """Parity has to hold with the extinction thresholds switched on."""
    torch.set_num_threads(1)
    env = sparse_env()
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    reference = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, n_workers=1,
                           hidden_dim=7, local_reward=config)
    reference.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    reference.softmax_temperature = 1.0
    runner = RolloutRunner(model, bank, 1, 1, execution="eager",
                           obs_normalize=False, local_reward=config)
    state = model.import_state([env])
    ticks = 6
    runner.reset(state[0], state[1], state[3],
                 torch.zeros(1, dtype=torch.int64, device=model.device), ticks,
                 torch.zeros_like(runner.obs_mean), torch.ones_like(runner.obs_var),
                 model.tensor(1.0))
    runner.run(ticks)
    reward, _ = runner.results(ticks)
    expected, _, _ = reference._evaluate_coevo(model.dm_ids, ticks)
    for d, fid in enumerate(model.dm_ids):
        assert reward[0, d].item() == pytest.approx(expected[fid], rel=2e-3, abs=2e-4)
    assert (runner.occupancy > 0).any(), "no cell took part; the test is vacuous"


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_tracking_leaves_the_tick_untouched(device):
    """``local_reward=None`` must not perturb the dynamics."""
    env = live_env(mortality=True)
    model = TensorEcosystem(env, device)
    state = model.import_state([env])
    probabilities = random_probabilities(model, state[0])
    plain = model.step(*state[:2], probabilities, model.tensor(0), state[3],
                       torch.zeros_like(state[0]))
    tracked = tracked_step(model, state, probabilities)
    assert len(plain) == 5 and len(tracked) == 6
    for actual, expected in zip(tracked[:2], plain[:2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("builder", [live_env, sparse_env],
                         ids=["no-threshold", "suppressed-splits"])
def test_tracked_tick_traces_without_graph_breaks(builder):
    """CUDA-graph execution requires a single fixed-shape graph.

    Both fixtures matter: ``sparse_env`` takes the split-suppression
    branch, which adds a gather and four slice subtractions to the tick.
    """
    env = builder(mortality=True) if builder is live_env else builder()
    model = TensorEcosystem(env, "cpu")
    bank = PolicyBank(model, hidden_dim=7)
    runner = RolloutRunner(model, bank, 1, 1, execution="eager",
                           local_reward=LocalRewardConfig())
    state = model.import_state([env])
    runner.reset(state[0], state[1], state[3], torch.zeros(1, dtype=torch.int64), 3,
                 torch.zeros_like(runner.obs_mean), torch.ones_like(runner.obs_var),
                 torch.tensor(1.0))
    runner._tick()
    expected = [v.clone() for v in runner.state_buffers] + [runner.local_sum.clone()]
    reference_sum = runner.local_sum.clone()
    runner.reset(state[0], state[1], state[3], torch.zeros(1, dtype=torch.int64), 3,
                 torch.zeros_like(runner.obs_mean), torch.ones_like(runner.obs_var),
                 torch.tensor(1.0))
    compiled = torch.compile(runner._tick, backend="eager", fullgraph=True)
    compiled()
    for actual, reference in zip(runner.state_buffers, expected):
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(runner.local_sum, reference_sum)
    assert torch.isfinite(runner.local_sum).all()


@pytest.mark.parametrize("kwargs", [
    dict(legacy_reward=True),
    dict(population_stability=PopulationStability()),
])
def test_runner_rejects_the_other_rewards(kwargs):
    """The local reward replaces the fitness, so it cannot be combined."""
    model = TensorEcosystem(live_env(), "cpu")
    bank = PolicyBank(model, hidden_dim=7)
    with pytest.raises(ValueError):
        RolloutRunner(model, bank, 1, 1, execution="eager",
                      local_reward=LocalRewardConfig(), **kwargs)
