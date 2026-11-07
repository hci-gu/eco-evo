"""Migrating decision makers (section 246; open_pool.migrate, the GPU
_migrate and the reward ledger lib/runners/energy_ledger.py).

Contracts asserted:

1. The keys are checked: a migrating DM needs 12 fills in [0, 1] and an
   entry; no depth table, no juvenile stage; an NDM takes neither; the
   season has a mean of ~1.
2. The entry weights select the edge segments (row 0 north, col 0 west).
3. One step on the group total: the total relaxes as T + (B - T) exp(-k);
   an export scales every cell's B and reserve alike; an import enters on
   the entry cells with fill x max_energy_reserve; the books and the
   tick's net exchange energy follow.
4. GPU parity with a migrating DM on the daylight calendar (season, fill).
5. The reward ledger: the GPU rollout and the CPU co-evolution reward agree
   for a migrating DM (integral and final energy); a pure exchange leaves
   the index unchanged; the legacy reward is refused on both engines.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import open_pool
from lib.runners.energy_ledger import EnergyLedger
from lib.world.functional_group import FunctionalGroup
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401

SEASON = [0.12, 0.11, 0.1, 0.3, 1.05, 2.1, 2.12, 2.19, 1.85, 1.24, 0.57, 0.24]
SEASON = [v * 12 / sum(SEASON) for v in SEASON]
FILL = [0.7, 0.58, 0.45, 0.37, 0.13, 0.51, 0.57, 0.91, 1.0, 1.0, 0.94, 0.85]
ENTRY = [{"edge": "west"}, {"edge": "north", "from": 0.0, "to": 0.5}]
K = 0.2
OPEN = {"outside_density": 2.0, "outside_exchange_rate": K, "outside_season": SEASON,
        "outside_fill": FILL, "outside_entry": ENTRY}
DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 600, "random_start": False}


def _env(keys=OPEN, **kw):
    env = make_env(min_split=0.0, extinction_factor=0.0, regen={"a": keys}, **kw)
    env.build_static_caches()
    return env


def test_keys_are_checked():
    dm = dict(OPEN, is_decision_maker=True)
    s = FunctionalGroup("m", dm).open_pool
    assert s["fill"] == FILL and s["entry"][0] == ("west", 0.0, 1.0)
    for bad in (dict(dm, outside_fill=FILL[:11]),
                dict(dm, outside_fill=FILL[:11] + [1.2]),
                {k: v for k, v in dm.items() if k != "outside_fill"},
                {k: v for k, v in dm.items() if k != "outside_entry"},
                dict(dm, outside_entry=[{"edge": "up"}]),
                dict(dm, outside_entry=[{"edge": "west", "from": 0.6, "to": 0.4}]),
                dict(dm, outside_depth_table=[{"depth_m": 1, "weight": 1}]),
                dict(dm, juvenile_stage={"kappa": 1}),
                dict(dm, outside_season=[2.0] * 12)):
        with pytest.raises(ValueError):
            FunctionalGroup("m", bad)
    with pytest.raises(ValueError):
        FunctionalGroup("n", {"outside_density": 1.0, "outside_exchange_rate": 0.1,
                              "outside_fill": FILL})


def test_entry_weights():
    w = open_pool.entry_weights(open_pool.parse_entry(ENTRY), (4, 6))
    assert w.sum() == pytest.approx(1.0)
    expected = np.zeros((4, 6), dtype=bool)
    expected[:, 0] = True
    expected[0, :3] = True
    np.testing.assert_array_equal(w > 0, expected)


def _set_total(env, total):
    a = env.fgs["a"]
    a.biomass = (a.biomass * np.float32(total / float(a.biomass.sum()))).astype(np.float32)
    a.energy_reserve = (a.biomass * np.float32(4.0)).astype(np.float32)


def test_export_scales_every_cell_and_books_it():
    env = _env({k: v for k, v in OPEN.items() if k != "outside_season"})  # T = 2 x 20 = 40
    a = env.fgs["a"]
    _set_total(env, 100.0)
    b0, r0 = a.biomass.astype(np.float64).copy(), a.energy_reserve.astype(np.float64).copy()
    open_pool.migrate(env)
    total = 40.0 + (100.0 - 40.0) * math.exp(-K)
    assert float(a.biomass.sum()) == pytest.approx(total, rel=1e-5)
    keep = total / 100.0
    np.testing.assert_allclose(a.biomass, b0 * keep, rtol=1e-5)
    np.testing.assert_allclose(a.energy_reserve, r0 * keep, rtol=1e-5)
    export = 100.0 - total
    assert env.loss_exchange["a"] == pytest.approx(export, rel=1e-5)
    assert env.gain_exchange["a"] == 0.0
    ec = float(a.params["energy_content"])
    assert env.tick_exchange_energy["a"] == pytest.approx(
        -(export * ec + (1 - keep) * r0.sum()), rel=1e-5)


def test_import_enters_on_the_edges_with_the_fill():
    env = _env({k: v for k, v in OPEN.items() if k != "outside_season"})
    a = env.fgs["a"]
    _set_total(env, 10.0)
    b0, r0 = a.biomass.astype(np.float64).copy(), a.energy_reserve.astype(np.float64).copy()
    open_pool.migrate(env)
    net = (40.0 - 10.0) * (1 - math.exp(-K))
    w = env.open_dm_entry["a"].astype(np.float64)
    fill = float(np.mean(FILL))                   # no daylight: the annual mean fill
    np.testing.assert_allclose(a.biomass, b0 + net * w, rtol=1e-5)
    np.testing.assert_allclose(a.energy_reserve, r0 + net * fill * a.max_energy_reserve * w,
                               rtol=1e-5)
    assert env.gain_exchange["a"] == pytest.approx(net, rel=1e-5)
    ec = float(a.params["energy_content"])
    assert env.tick_exchange_energy["a"] == pytest.approx(
        net * (ec + fill * a.max_energy_reserve), rel=1e-5)


def test_migration_matches_reference(device):
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True, regen={"a": OPEN},
                   daylight=DAYLIGHT)
    model = _assert_parity(env, device, ticks=8)
    assert model.open_dm_on and model.open_dm_tables


def test_pure_exchange_leaves_the_ledger_index():
    class Env:
        open_dm_ids = ["a"]
        tick_exchange_biomass = {"a": 0.0}
        tick_exchange_reserve = {"a": 0.0}
        tick_exchange_energy = {"a": 0.0}
    env = Env()
    ledger = EnergyLedger(env, {"a": (10.0, 20.0, 100.0), "b": (5.0, 1.0, 50.0)})
    env.tick_exchange_biomass["a"], env.tick_exchange_reserve["a"] = 3.0, 6.0
    env.tick_exchange_energy["a"] = 30.0
    assert ledger.update("a", 13.0, 26.0, 130.0) == pytest.approx((10.0, 20.0, 100.0))
    env.tick_exchange_biomass["a"] = env.tick_exchange_reserve["a"] = 0.0
    env.tick_exchange_energy["a"] = 0.0
    assert ledger.update("a", 14.3, 28.6, 143.0) == pytest.approx((11.0, 22.0, 110.0))
    assert ledger.update("b", 6.0, 1.0, 60.0) == (6.0, 1.0, 60.0) and not ledger.tracks("b")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("legacy,integral,stability", [
    (False, True, False), (False, False, False), (True, True, False), (True, False, False),
    (False, True, True)])
def test_rollout_reward_matches_cpu_with_a_migrating_dm(device, legacy, integral, stability):
    from lib.gpu.ecosystem import TensorEcosystem
    from lib.gpu.policy import PolicyBank
    from lib.gpu.rollout import RolloutRunner
    from lib.runners.trainer import ARSTrainer
    torch.set_num_threads(1)
    env = make_env(True, True, regen={"a": OPEN}, daylight=DAYLIGHT)
    model = TensorEcosystem(env, device)
    bank = PolicyBank(model, hidden_dim=7)
    params = {f: (model.in_dims[i], model.A) for i, f in enumerate(model.dm_ids)}
    from lib.runners.population_stability import PopulationStability
    config = (PopulationStability(lower=0.05, warning_lower=0.1, warning_upper=5.0, upper=10.0)
              if stability else None)
    reference = ARSTrainer(lambda seed=None: copy.deepcopy(env), params, n_workers=1,
                           hidden_dim=7, legacy_reward=legacy, integral_reward=integral,
                           alpha=0.7, beta=1.2, survival_bonus=0.4, population_stability=config)
    reference.policies = {f: copy.deepcopy(p).cpu() for f, p in bank.policies.items()}
    reference.softmax_temperature = 1.7
    runner = RolloutRunner(model, bank, 1, 1, execution="eager", legacy_reward=legacy,
                           integral_reward=integral, alpha=0.7, beta=1.2, survival_bonus=0.4,
                           population_stability=config)
    b, r, _ = model.import_state([env])
    mean = np.full((model.D, model.F), 0.03, dtype=np.float32)
    var = np.full_like(mean, 0.8)
    ticks = 11
    runner.reset(b, r, model.light_starts([env]), ticks,
                 model.tensor(mean), model.tensor(var), model.tensor(1.7))
    runner.run(ticks)
    reward, _ = runner.results(ticks)
    expected, _, _ = reference._evaluate_coevo(
        model.dm_ids, ticks, obs_mean=mean, obs_var=var, dm_ids_for_norm=list(model.dm_ids))
    for d, fid in enumerate(model.dm_ids):
        assert reward[0, d].item() == pytest.approx(expected[fid], rel=3e-5, abs=3e-6), fid
    with pytest.raises(ValueError):
        RolloutRunner(model, bank, 1, 1, execution="eager",
                      local_reward={"metric": "log", "norm": "grid"})


def test_mareld2_mackerel_migrates():
    """Section 246 in the library and the loaded world."""
    from lib.config.config_loader import load_config, load_project_config
    root = Path(__file__).resolve().parents[1]
    spec = load_config(str(root / "fgconfig" / "fg_library.yaml"))["species_definitions"]["mackerel"]
    assert spec["outside_density"] == 0.85
    assert spec["outside_exchange_rate"] == pytest.approx(0.025)
    assert np.mean(spec["outside_season"]) == pytest.approx(1.0, abs=2e-3)
    assert len(spec["outside_fill"]) == 12 and min(spec["outside_fill"]) == pytest.approx(0.13)
    assert spec["outside_entry"] == [{"edge": "west"}, {"edge": "north", "from": 0.0, "to": 0.5}]
    assert float(spec["r_max"]) == 0.0 and spec["feeding_season_months"] == 5.5
    assert spec["max_energy_reserve"] == 6980.0 and spec["reserve_reference_fill"] == 0.672
    fgs = load_project_config("mareld2.yaml", mode="inference")[0]
    assert fgs["mackerel"].open_pool["entry"] == [("west", 0.0, 1.0), ("north", 0.0, 0.5)]
    from lib.environments.ecosystem_env.environment import EcosystemEnvironment
    env = EcosystemEnvironment({"height": 60, "width": 60}, fgs, {})
    assert env.open_dm_ids == ["mackerel"]
    assert env.open_dm_total["mackerel"] == pytest.approx(0.85 * 3600)
    w = env.open_dm_entry["mackerel"]
    assert (w > 0).sum() == 60 + 30 - 1                   # west column + half the top row
