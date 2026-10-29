"""Juvenile compartment inside a decision-making group (stages.py).

J rides inside B: B is unchanged by it, J follows every group-wide loss,
moves with B, takes kappa- and rho-weighted shares of the new tonnes,
graduates at exp(-gamma) per tick, is eaten per ``stage_access`` and
fished at its ``fishing_selectivity``.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.environments.ecosystem_env import impacts, predation, stages
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import daylight as daylight_module
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401

DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 0, "random_start": False}
PER_YEAR = daylight_module.ticks_per_year(6)


def block(kappa=1.0, gamma=1.0, rho=0.0, start=0.3, selectivity=None):
    out = {"growth_ratio": kappa, "graduation_rate": gamma,
           "recruitment": [rho] * 12, "start_share": start}
    if selectivity:
        out["fishing_selectivity"] = dict(selectivity)
    return out


def actions_for(env, seed=3):
    """A fixed, valid action distribution (moves, rest, eat on the menu)."""
    rng = np.random.default_rng(seed)
    H, W = env.H, env.W
    move = rng.uniform(0.0, 1.0, (env.N_dm, 4, H, W))
    rest = rng.uniform(0.0, 1.0, (env.N_dm, H, W))
    eat = rng.uniform(0.0, 1.0, (env.N_dm, env.N_all, H, W)) * env.eat_static_mask[:, :, None, None]
    total = move.sum(1) + rest + eat.sum(1)
    return ActionProbabilities((move / total[:, None]).astype(np.float32),
                               (rest / total).astype(np.float32),
                               (eat / total[:, None]).astype(np.float32))


def staged_env(**kw):
    kw.setdefault("min_split", 0.0)
    kw.setdefault("extinction_factor", 0.0)
    kw.setdefault("daylight", dict(DAYLIGHT))
    return make_env(**kw)


# ---------------------------------------------------------------- settings
def test_settings_are_validated():
    with pytest.raises(ValueError, match="start_share"):
        staged_env(stages={"b": block(start=1.0)})
    with pytest.raises(ValueError, match="growth_ratio"):
        staged_env(stages={"b": block(kappa=0.0)})
    bad = block()
    bad["recruitment"] = [1.0] * 11
    with pytest.raises(ValueError, match="recruitment"):
        staged_env(stages={"b": bad})
    with pytest.raises(ValueError, match="stage_access needs a prey"):
        staged_env(stages={"b": block()}, pair_keys={"a_preys_on_c": {"stage_access": [1, 0]}})
    with pytest.raises(ValueError, match="either accessible_fraction"):
        staged_env(stages={"b": block()},
                   pair_keys={"a_preys_on_b": {"stage_access": [1, 0], "accessible_fraction": 0.5}})


def test_without_the_block_nothing_changes():
    env = staged_env()
    assert not env._has_stages
    assert all(fg.juvenile is None for fg in env.fgs.values())
    assert all(len(idx) == 0 for idx in env.obs_juvenile_idx)


def test_start_share_and_tables():
    env = staged_env(stages={"b": block(gamma=0.8, start=0.25)})
    fg = env.fgs["b"]
    np.testing.assert_allclose(fg.juvenile, 0.25 * fg.biomass, rtol=1e-6)
    j = env.global_fg_order.index("b")
    assert env.stage_keep[j] == pytest.approx(math.exp(-0.8 / PER_YEAR))
    assert env.stage_rho.shape == (PER_YEAR, env.N_all)


# ---------------------------------------------------------------- dynamics
def _run(env, ticks, seed=3):
    for t in range(ticks):
        env.step(actions_for(env, seed + t))


@pytest.mark.parametrize("migration,mortality", [(False, False), (True, True)])
def test_juveniles_stay_inside_the_biomass(migration, mortality):
    env = staged_env(migration=migration, mortality=mortality,
                     stages={"a": block(kappa=4.0, rho=2.0), "b": block(kappa=5.0, rho=3.0)})
    for t in range(15):
        env.step(actions_for(env, t))
        for fid in ("a", "b"):
            fg = env.fgs[fid]
            assert np.all(fg.juvenile >= 0.0)
            assert np.all(fg.juvenile <= fg.biomass * (1 + 1e-5) + 1e-9)


def test_biomass_does_not_depend_on_the_compartment():
    """With f_J = f_A and s_J = 1 the group's own dynamics are unchanged:
    the same fixed actions give the same B and R with and without J."""
    kw = dict(migration=True, mortality=True)
    plain = staged_env(**kw)
    staged = staged_env(stages={"b": block(kappa=5.0, rho=3.0, gamma=2.0)}, **kw)
    for t in range(10):
        acts = actions_for(plain, t)
        plain.step(copy.deepcopy(acts))
        staged.step(copy.deepcopy(acts))
    for fid in plain.fgs:
        np.testing.assert_allclose(staged.fgs[fid].biomass, plain.fgs[fid].biomass, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(staged.fgs[fid].energy_reserve, plain.fgs[fid].energy_reserve,
                                   rtol=2e-5, atol=1e-6)


def test_neutral_compartment_keeps_its_share_and_graduates_exactly():
    """kappa 1, rho 0, f_J = f_A, s_J = 1: J is a label on a fixed share of
    every tonne, through predation, movement, M1 and growth, and only the
    graduation lowers it - by exactly exp(-gamma/T) per tick."""
    gamma, start, ticks = 3.0, 0.4, 8
    env = staged_env(migration=True, mortality=True,
                     stages={"b": block(kappa=1.0, rho=0.0, gamma=gamma, start=start)})
    _run(env, ticks)
    fg = env.fgs["b"]
    live = fg.biomass > 1e-6
    expected = start * math.exp(-gamma * ticks / PER_YEAR)
    np.testing.assert_allclose(fg.juvenile[live] / fg.biomass[live], expected, rtol=2e-4)


def test_recruitment_raises_the_juvenile_share():
    base = staged_env(stages={"b": block(rho=0.0, gamma=0.01, start=0.2)})
    recruit = staged_env(stages={"b": block(rho=20.0, gamma=0.01, start=0.2)})
    _run(base, 6)
    _run(recruit, 6)
    share = lambda e: e.fgs["b"].juvenile.sum() / e.fgs["b"].biomass.sum()
    assert share(recruit) > share(base) + 1e-3


def test_growth_share_formula():
    env = staged_env(stages={"b": block(kappa=4.0, rho=2.0)})
    j = env.global_fg_order.index("b")
    b = np.array([[10.0, 0.0]], dtype=np.float32)
    juv = np.array([[2.0, 0.0]], dtype=np.float32)
    w = stages.growth_share(env, j, b, juv)
    adult = 8.0
    assert w[0, 0] == pytest.approx((4 * 2 + 2 * adult) / (4 * 2 + 2 * adult + adult), rel=1e-6)
    assert w[0, 1] == 0.0


# ---------------------------------------------------------------- predation
def test_stage_access_takes_from_the_juveniles_only():
    env = staged_env(stages={"b": block(start=0.3)},
                     pair_keys={"a_preys_on_b": {"stage_access": [1.0, 0.0]}})
    fg = env.fgs["b"]
    adult_before = fg.biomass - fg.juvenile
    acts = actions_for(env, 1)
    env.publish_action_probabilities(acts)
    predation.apply_predation(env, acts)
    adult_after = fg.biomass - fg.juvenile
    np.testing.assert_allclose(adult_after, adult_before, rtol=1e-5, atol=1e-7)
    total = env.intake_by_pred_prey["a"].get("b", 0.0)
    assert total > 0.0
    assert env.intake_juvenile_by_pred_prey["a"]["b"] == pytest.approx(total, rel=1e-5)


def test_stage_access_matches_accessible_fraction_when_equal():
    """[f, f] is the unstaged pair with accessible_fraction f."""
    a = staged_env(stages={"b": block()}, pair_keys={"a_preys_on_b": {"stage_access": [0.4, 0.4]}})
    b = staged_env(stages={"b": block()}, pair_keys={"a_preys_on_b": {"accessible_fraction": 0.4}})
    acts = actions_for(a, 2)
    for env in (a, b):
        env.publish_action_probabilities(copy.deepcopy(acts))
        predation.apply_predation(env, copy.deepcopy(acts))
    np.testing.assert_allclose(a.fgs["b"].biomass, b.fgs["b"].biomass, rtol=1e-6)
    np.testing.assert_allclose(a.fgs["b"].juvenile, b.fgs["b"].juvenile, rtol=1e-6)


def test_adult_only_pair_with_no_adults_takes_nothing():
    env = staged_env(stages={"b": block(start=0.999)},
                     pair_keys={"a_preys_on_b": {"stage_access": [0.0, 1.0]}})
    fg = env.fgs["b"]
    fg.juvenile = fg.biomass.copy()
    acts = actions_for(env, 4)
    env.publish_action_probabilities(acts)
    predation.apply_predation(env, acts)
    assert env.intake_by_pred_prey["a"].get("b", 0.0) == 0.0
    assert np.all(np.isfinite(env.fgs["a"].temp_energy_gains))


# ---------------------------------------------------------------- fishing
@pytest.mark.parametrize("s_j", [0.0, 2.0])
def test_fishing_selectivity(s_j):
    table = {"trawl": {"impact_affects": True, "impact_table": [
        {"value": 0.0, "biomass_factor": 0.0, "energy_factor": 0.0},
        {"value": 1.0, "biomass_factor": 0.1, "energy_factor": 0.0}]}}
    env = staged_env(stages={"b": block(start=0.2, selectivity={"trawl": s_j})},
                     impacts={"b": table})
    env.grid.add_map("trawl", np.ones((env.H, env.W), dtype=np.float32))
    fg = env.fgs["b"]
    b0, j0 = fg.biomass.copy(), fg.juvenile.copy()
    env.ordered_fg_ids = list(env.fgs)
    impacts.apply_impact_mortality(env)
    np.testing.assert_allclose(fg.biomass, b0 * 0.9, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(fg.juvenile, j0 - 0.1 * s_j * j0, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- observations
def test_observation_has_the_juvenile_channels():
    env = staged_env(stages={"b": block(start=0.3)})
    plain = staged_env()
    i_a, i_b = env.dm_ids.index("a"), env.dm_ids.index("b")
    # a observes b (and c), b observes a: each sees b's J (b its own).
    assert list(env.obs_juvenile_idx[i_a]) == [env.global_fg_order.index("b")]
    assert list(env.obs_juvenile_idx[i_b]) == [env.global_fg_order.index("b")]
    assert env.per_dm_in_dim[i_a] == plain.per_dm_in_dim[i_a] + 5
    obs = env.get_observation().features.reshape(env.N_dm, env.H, env.W, -1)
    k = len(env.obs_others_idx[i_b])
    np.testing.assert_allclose(obs[i_b, :, :, 2 + k], env.fgs["b"].juvenile, rtol=1e-6)


# ---------------------------------------------------------------- GPU parity
STAGES = {"a": block(kappa=4.0, rho=2.0, gamma=1.5, start=0.3),
          "b": block(kappa=5.0, rho=3.0, gamma=0.7, start=0.4)}


@pytest.mark.parametrize("migration,mortality", [(False, False), (True, True)])
@pytest.mark.parametrize("pairs", [None, {"a_preys_on_b": {"stage_access": [0.9, 0.2]}}])
def test_gpu_matches_reference(device, migration, mortality, pairs):
    env = staged_env(migration=migration, mortality=mortality, handling=0.3,
                     stages=STAGES, pair_keys=pairs)
    # The policy forward pass differs at round-off between the engines; a
    # cell whose reserve is a small difference of large terms then sits
    # just above the harness's 1e-6 absolute floor (the unstaged fixture
    # shows the same 2e-6). The fixed-action test below is exact.
    model = _assert_parity(env, device, ticks=8, base_tol=1e-5)
    assert model.stages_on
    assert model.in_dims == tuple(int(n) for n in env.per_dm_in_dim)


def test_gpu_matches_reference_with_splits_income_and_pair_floors(device):
    env = staged_env(migration=True, mortality=True, min_split=50, extinction_factor=0.3,
                     handling=0.3, stages=STAGES, income=(0.7, 6.0), capacity=0.05,
                     pair_floors={"a_preys_on_b": 0.6},
                     pair_keys={"a_preys_on_b": {"stage_access": [1.0, 0.3]}})
    _assert_parity(env, device, ticks=8, base_tol=1e-5)


@pytest.mark.parametrize("s_j", [0.0, 3.0])
def test_gpu_fishing_selectivity_matches_reference(device, s_j):
    table = {"trawl": {"impact_affects": True, "impact_table": [
        {"value": 0.0, "biomass_factor": 0.0, "energy_factor": 0.0},
        {"value": 1.0, "biomass_factor": 0.05, "energy_factor": 0.1}]}}
    stages_ = dict(STAGES)
    stages_["b"] = block(kappa=5.0, rho=3.0, gamma=0.7, start=0.4, selectivity={"trawl": s_j})
    env = staged_env(mortality=True, stages=stages_, impacts={"b": table})
    env.grid.add_map("trawl", np.linspace(0, 1, env.H * env.W, dtype=np.float32).reshape(env.H, env.W))
    model = _assert_parity(env, device, ticks=6, base_tol=1e-5)
    assert model.impact_rows == model.G + 2


def test_gpu_fixed_actions_match_exactly(device):
    """The same fixed actions on both engines: B, R and J agree to float
    round-off over several ticks (no policy in the loop)."""
    import torch
    from test_gpu_ecosystem import TensorEcosystem, numpy
    env = staged_env(migration=True, mortality=True, handling=0.3, stages=STAGES,
                     income=(0.7, 6.0), capacity=0.05,
                     pair_keys={"a_preys_on_b": {"stage_access": [1.0, 0.3]}})
    model = TensorEcosystem(env, device)
    b, r, _ = model.import_state([env])
    juv = model.import_juvenile([env])
    starts = model.light_starts([env])
    for tick in range(6):
        acts = actions_for(env, 10 + tick)
        probs = torch.tensor(np.concatenate([acts.move, acts.rest[:, None], acts.eat], axis=1)
                             .reshape(1, env.N_dm, 5 + env.N_all, -1), device=model.device)
        light = model.light_index(starts, torch.tensor(tick, device=model.device))
        out = model.step(b, r, probs, model.tensor(tick), torch.zeros_like(b),
                         light_index=light, juvenile=juv)
        b, r, juv = out[0], out[1], out[-1]
        env.step(copy.deepcopy(acts))
        expected = model.import_state([env])
        np.testing.assert_allclose(numpy(b), numpy(expected[0]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(numpy(r), numpy(expected[1]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(numpy(juv), numpy(model.import_juvenile([env])), rtol=1e-5, atol=1e-6)


def test_without_a_calendar_the_annual_means_are_used(device):
    """Test and MVP worlds without the daylight calendar: rho and the start
    share are the annual means, gamma per library tick; both engines."""
    stage = block(kappa=3.0, gamma=1.0, start=0.25)
    stage["recruitment"] = [0.0] * 6 + [6.0] * 6
    env = make_env(min_split=0.0, extinction_factor=0.0, stages={"b": stage})
    j = env.global_fg_order.index("b")
    assert env.stage_rho.shape == (1, env.N_all)
    assert env.stage_rho[0, j] == pytest.approx(3.0)
    assert env.stage_keep[j] == pytest.approx(math.exp(-1.0 / 1460))
    _assert_parity(env, device, ticks=6, base_tol=1e-5)
