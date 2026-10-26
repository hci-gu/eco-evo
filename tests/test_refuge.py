"""Benthic refuge with exchange (section 223.12;
lib/environments/ecosystem_env/refuge.py).

Contracts asserted:

1. ``exposed_share`` must be in (0, 1); the accessible part A starts at
   phi x B (CPU initialisation and the GPU spawner).
2. Without predation the total B grows exactly as the group's plain
   logistic while A/B stays phi; a displaced A relaxes towards phi B by
   exp(-k) per tick without changing B.
3. Predation sees and removes A only: a predator eating a refuge group
   takes what it would take from a prey of biomass A, the refuge is
   untouched; observers see A.
4. A large k with A at phi B reproduces the pre-refuge accessible
   fraction phi on the pair (the fast-exchange limit).
5. GPU parity with a refuge group.
6. mareld2: benthic_infauna carries the refuge, its benthivore pair no
   accessible_fraction, and the derived growth rate is unchanged by it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import population_change, predation, refuge
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world.functional_group import FunctionalGroup
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401

PHI, K_TICK = 0.2, 0.05
KEYS = {"exposed_share": PHI, "refuge_exchange_rate": K_TICK}


def _built(refuge_keys=KEYS, **kw):
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=2.0,
                   refuge=refuge_keys, **kw)
    env.build_static_caches()
    return env


def test_share_is_checked_and_a_starts_at_phi():
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            FunctionalGroup("x", {"exposed_share": bad})
    with pytest.raises(ValueError):
        FunctionalGroup("x", {"exposed_share": 0.3, "refuge_exchange_rate": -1})
    env = _built()
    c = env.fgs["c"]
    np.testing.assert_allclose(c.energy_reserve, PHI * c.biomass, rtol=1e-6)
    assert env._has_refuge and list(env.refuge_index) == [env.global_fg_order.index("c")]
    # A decision maker never carries the refuge.
    assert FunctionalGroup("d", {"is_decision_maker": True,
                                 "exposed_share": 0.3}).exposed_share is None


def test_growth_is_the_plain_logistic_at_phi():
    ref, plain = _built(), _built(refuge_keys=None)
    for env in (ref, plain):
        population_change._apply_non_decision_maker_population_change(
            env, "c", env.fgs["c"])
    np.testing.assert_allclose(ref.fgs["c"].biomass, plain.fgs["c"].biomass,
                               rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(ref.fgs["c"].energy_reserve,
                               PHI * ref.fgs["c"].biomass, rtol=1e-5, atol=1e-7)


def test_exchange_relaxes_a_and_keeps_b():
    env = _built()
    c = env.fgs["c"]
    c.energy_reserve = (c.biomass * 0.05).astype(np.float32)   # A eaten down
    b0, a0 = c.biomass.astype(np.float64).copy(), c.energy_reserve.astype(np.float64).copy()
    # rate 0: only the exchange acts (it runs before growth)
    refuge.grow(c, 0.0, 12.0, np.zeros_like(c.biomass), np.float32)
    np.testing.assert_allclose(c.biomass, b0, rtol=1e-6)
    expected = PHI * b0 + (a0 - PHI * b0) * np.exp(-K_TICK)
    np.testing.assert_allclose(c.energy_reserve, expected, rtol=1e-5, atol=1e-7)


def _eat_only_c(env):
    shape = (env.N_dm, env.H, env.W)
    eat = np.zeros((env.N_dm, env.N_all, env.H, env.W), dtype=env.dtype)
    eat[env.dm_ids.index("a"), env.global_fg_order.index("c")] = 1.0
    return ActionProbabilities(move=np.zeros((env.N_dm, 4, env.H, env.W), dtype=env.dtype),
                               rest=np.zeros(shape, dtype=env.dtype), eat=eat)


def test_predation_sees_and_removes_a_only():
    ref = _built()
    # The same world where c's whole biomass is A (no refuge).
    flat = _built(refuge_keys=None)
    flat.fgs["c"].biomass = (ref.fgs["c"].energy_reserve).astype(np.float32).copy()
    b0 = ref.fgs["c"].biomass.astype(np.float64).copy()
    r0 = b0 - ref.fgs["c"].energy_reserve.astype(np.float64)
    for env in (ref, flat):
        env.intake_by_pred_prey = {}
        predation.apply_predation(env, _eat_only_c(env))
    np.testing.assert_allclose(ref.fgs["a"].temp_energy_gains,
                               flat.fgs["a"].temp_energy_gains, rtol=2e-3, atol=1e-9)
    c = ref.fgs["c"]
    np.testing.assert_allclose(c.biomass - c.energy_reserve, r0, rtol=1e-5, atol=1e-6)
    assert (b0 - c.biomass).sum() > 0
    # Observers see A.
    obs_ref = ref.get_observation().features
    obs_flat = flat.get_observation().features
    np.testing.assert_allclose(obs_ref, obs_flat, rtol=1e-5, atol=1e-6)


def test_fast_exchange_is_the_old_accessible_fraction():
    fast = _built({"exposed_share": PHI, "refuge_exchange_rate": 50.0})
    old = make_env(min_split=0.0, extinction_factor=0.0, handling=2.0,
                   pair_keys={"a_preys_on_c": {"accessible_fraction": PHI},
                              "b_preys_on_c": {"accessible_fraction": PHI}})
    old.build_static_caches()
    for env in (fast, old):
        env.intake_by_pred_prey = {}
        predation.apply_predation(env, _eat_only_c(env))
        population_change._apply_non_decision_maker_population_change(env, "c", env.fgs["c"])
    np.testing.assert_allclose(fast.fgs["a"].temp_energy_gains,
                               old.fgs["a"].temp_energy_gains, rtol=2e-3, atol=1e-9)
    np.testing.assert_allclose(fast.fgs["c"].biomass, old.fgs["c"].biomass,
                               rtol=2e-3, atol=1e-6)


def test_refuge_matches_reference(device):
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True, refuge=KEYS)
    model = _assert_parity(env, device, ticks=8)
    assert model.refuge_on


def test_mareld2_infauna_refuge():
    import copy
    import yaml
    from lib.world import growth_budget
    root = Path(__file__).resolve().parents[1]
    library = yaml.safe_load((root / "fgconfig" / "fg_library.yaml").read_text(encoding="utf-8-sig"))
    spec = library["species_definitions"]["benthic_infauna"]
    assert spec["exposed_share"] == pytest.approx(0.2)
    assert spec["refuge_exchange_rate"] == pytest.approx(2.5 / 1460, rel=1e-3)
    pair = library["interaction_definitions"]["benthivorous_fish_preys_on_benthic_infauna"]
    assert "accessible_fraction" not in pair
    stripped = copy.deepcopy(library)
    for key in ("exposed_share", "refuge_exchange_rate"):
        stripped["species_definitions"]["benthic_infauna"].pop(key)
    g = [growth_budget.derived_growth_rate("benthic_infauna", lib["species_definitions"],
                                           lib["interaction_definitions"])
         for lib in (library, stripped)]
    assert g[0] == pytest.approx(g[1])


def _mareld_trainer(depleted, warmup=0):
    import torch
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    torch.set_num_threads(1)
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(12, 12)))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2, hidden_dim=7,
                            worlds=8, depleted_patches=depleted, warmup_ticks=warmup)


def test_depleted_patches_hit_the_benthos_only():
    """Section 223.12: a share of the training worlds starts with depleted
    benthic patches; the infauna loses accessible tonnes, its refuge stays."""
    import torch
    trainer = _mareld_trainer(1.0)
    trainer.refresh_worlds()
    m = trainer.model
    factor = trainer.world_depletion
    assert factor.shape == (8, 1, m.C)
    patched = factor < 1.0
    cover = patched.float().mean(-1).flatten()
    assert ((cover > 0.03) & (cover < 0.32)).all()
    assert ((factor[patched] >= 0.0099) & (factor[patched] <= 0.317)).all()
    keys = torch.zeros(8, dtype=torch.int64)
    b = trainer.world_biomass
    r = trainer.spawner.reserves(b, keys)
    nb, nr = m.deplete(b, r, factor)
    infauna = m.ids.index("benthic_infauna")
    refuge_before = b[:, infauna] - r[:, infauna]
    torch.testing.assert_close(nb[:, infauna] - nr[:, infauna], refuge_before)
    assert (nb[:, infauna] <= b[:, infauna] + 1e-6).all()
    for fid in m.ids:
        j = m.ids.index(fid)
        if not bool(m.benthic_mask[0, j, 0]):
            torch.testing.assert_close(nb[:, j], b[:, j])
    off = _mareld_trainer(0.0)
    off.refresh_worlds()
    assert (off.world_depletion == 1.0).all()


def test_depleted_patches_train_with_and_without_warmup():
    for warmup in (0, 3):
        trainer = _mareld_trainer(0.5, warmup)
        trainer.train_step(n_eval_ticks=3)
        assert trainer.iterations_completed == 1
