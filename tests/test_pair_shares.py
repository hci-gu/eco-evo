"""Predator share and accessible fraction of a predation pair (section 208).

Contracts asserted:

1. Both keys parse as shares in (0, 1], default 1; anything else is
   refused.
2. ``predator_share`` phi caps the eat share of the pair at phi (the
   capable part eats at the full a and 1/h; share above phi buys
   nothing); a and h are untouched, and the growth budget's ideal intake
   from that prey alone is phi x its ceiling.
3. ``accessible_fraction`` f scales the pair's prey biomass in the
   Holling term and in the harvest cap: with f the predator takes what
   it takes from f x B, and the prey loses no more than that.
4. Inert at 1 (bit-identical), and GPU parity with both keys set, with
   and without per-pair visibility floors.
5. mareld2: gadoids -> benthos carries phi 0.53 and f 0.2, and the
   gadoids' derived growth rate is unchanged (their best prey at the
   ceiling is zooplankton).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.environments.ecosystem_env import interactions, predation
from lib.environments.ecosystem_env.state import ActionProbabilities
from lib.world import growth_budget
from test_gpu_ecosystem import DEVICES, _assert_parity, device, make_env  # noqa: F401

PAIR = "a_preys_on_c"


def test_shares_parse_and_refuse():
    assert interactions.pair_share({}, "predator_share") == 1.0
    assert interactions.pair_share({"predator_share": "0.5"}, "predator_share") == 0.5
    for bad in (0, -0.1, 1.2, "x", float("nan")):
        with pytest.raises(ValueError):
            interactions.pair_share({"k": bad}, "k")


def _built(keys=None, **kw):
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=2.0,
                   pair_keys={PAIR: keys} if keys else None, **kw)
    env.build_static_caches()
    return env


def test_predator_share_caps_the_eat_share():
    plain, shared = _built(), _built({"predator_share": 0.4})
    i, j = plain.dm_ids.index("a"), plain.global_fg_order.index("c")
    np.testing.assert_array_equal(shared.max_intake_mat, plain.max_intake_mat)
    np.testing.assert_array_equal(shared.handling_time_mat, plain.handling_time_mat)
    assert shared._has_share_cap and shared.share_cap_mat[i, j] == pytest.approx(0.4)
    assert not shared._has_pair_access
    # all of a's eat mass on c buys what an eat share of 0.4 buys
    capped = _eat_only_c(shared)
    partial = _eat_only_c(plain)
    partial.eat[i, j] *= np.float32(0.4)
    for env, actions in ((shared, capped), (plain, partial)):
        predation.apply_predation(env, actions)
    np.testing.assert_allclose(shared.fgs["a"].temp_energy_gains,
                               plain.fgs["a"].temp_energy_gains, rtol=1e-6)
    np.testing.assert_allclose(shared.fgs["c"].biomass, plain.fgs["c"].biomass, rtol=1e-6)


def test_growth_budget_sees_the_predator_share():
    species = {"p": {"max_intake_rate": 0.5}, "q": {"energy_content": 100.0}}
    pairs = {"p_preys_on_q": {"preys_on": True, "handling_time": 10.0}}
    full, best = growth_budget.ideal_intake("p", species, pairs)
    pairs["p_preys_on_q"]["predator_share"] = 0.5
    half, _ = growth_budget.ideal_intake("p", species, pairs)
    assert best == "q" and half == pytest.approx(0.5 * full)


def _eat_only_c(env):
    """DM a eats c with its whole action mass; b does nothing (it eats c
    too, at full access, which would otherwise set the shared cap)."""
    shape = (env.N_dm, env.H, env.W)
    eat = np.zeros((env.N_dm, env.N_all, env.H, env.W), dtype=env.dtype)
    eat[env.dm_ids.index("a"), env.global_fg_order.index("c")] = 1.0
    return ActionProbabilities(move=np.zeros((env.N_dm, 4, env.H, env.W), dtype=env.dtype),
                               rest=np.zeros(shape, dtype=env.dtype), eat=eat)


def test_accessible_fraction_is_prey_biomass_out_of_reach():
    f = 0.25
    reach = _built({"accessible_fraction": f})
    assert reach._has_pair_access
    # the same world with the prey's biomass scaled by f and full access
    scaled = _built()
    scaled.fgs["c"].biomass = (reach.fgs["c"].biomass * np.float32(f)).astype(np.float32)
    c0 = reach.fgs["c"].biomass.astype(np.float64).copy()
    for env in (reach, scaled):
        env.intake_by_pred_prey = {}
        predation.apply_predation(env, _eat_only_c(env))
    # Where the cap binds, b's full access keeps the shared cap at
    # MAX_HARVEST_FRAC x B, so a's own bound f B (not MAX_HARVEST_FRAC x
    # f B) applies: 0.1 % at most, as for a per-pair visibility floor.
    np.testing.assert_allclose(reach.fgs["a"].temp_energy_gains,
                               scaled.fgs["a"].temp_energy_gains, rtol=2e-3, atol=1e-9)
    eaten = c0 - reach.fgs["c"].biomass
    assert (eaten <= f * c0 + 1e-6).all() and eaten.sum() > 0
    np.testing.assert_allclose(eaten, c0 * f - scaled.fgs["c"].biomass, rtol=2e-3, atol=1e-6)


def test_inert_at_one():
    plain, unit = _built(), _built({"predator_share": 1.0, "accessible_fraction": 1.0})
    np.testing.assert_array_equal(plain.max_intake_mat, unit.max_intake_mat)
    np.testing.assert_array_equal(plain.handling_time_mat, unit.handling_time_mat)
    assert not unit._has_pair_access


@pytest.mark.parametrize("floors", [None, {"a_preys_on_b": 0.6}])
def test_pair_shares_match_reference(device, floors):
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   pair_floors=floors,
                   pair_keys={PAIR: {"predator_share": 0.5, "accessible_fraction": 0.3},
                              "b_preys_on_c": {"accessible_fraction": 0.6}})
    model = _assert_parity(env, device, ticks=8)
    assert model.pair_access is not None


def test_mareld2_gadoid_benthos_pair():
    import yaml
    root = Path(__file__).resolve().parents[1]
    library = yaml.safe_load((root / "fgconfig" / "fg_library.yaml").read_text())
    pair = library["interaction_definitions"]["gadoids_preys_on_benthic_community"]
    assert pair["predator_share"] == pytest.approx(0.53)
    assert pair["accessible_fraction"] == pytest.approx(0.2)
    stripped = copy.deepcopy(library)
    for key in ("predator_share", "accessible_fraction"):
        stripped["interaction_definitions"]["gadoids_preys_on_benthic_community"].pop(key)
    g = [growth_budget.derived_growth_rate("gadoids", lib["species_definitions"],
                                           lib["interaction_definitions"])
         for lib in (library, stripped)]
    assert g[0] == pytest.approx(g[1])
