"""Accessible fraction of a predation pair (section 208); predator_share
is gone (section 221).

Contracts asserted:

1. ``accessible_fraction`` parses as a share in (0, 1], default 1;
   anything else is refused. A pair that still carries
   ``predator_share`` is refused (removed in section 221).
2. The growth budget's ideal intake is the best pair at the whole eat
   share (no caps).
3. ``accessible_fraction`` f scales the pair's prey biomass in the
   Holling term and in the harvest cap: with f the predator takes what
   it takes from f x B, and the prey loses no more than that.
4. Inert at 1 (bit-identical), and GPU parity with f set, with and
   without per-pair visibility floors.
5. mareld2: no predator_share anywhere; benthivorous_fish -> epifauna
   carries an f, which does not move the derived growth rate (the
   infauna pair's 0.2 is the infauna's refuge since section 223.12).
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
    assert interactions.pair_share({}, "accessible_fraction") == 1.0
    assert interactions.pair_share({"accessible_fraction": "0.5"}, "accessible_fraction") == 0.5
    for bad in (0, -0.1, 1.2, "x", float("nan")):
        with pytest.raises(ValueError):
            interactions.pair_share({"k": bad}, "k")


def _built(keys=None, **kw):
    env = make_env(min_split=0.0, extinction_factor=0.0, handling=2.0,
                   pair_keys={PAIR: keys} if keys else None, **kw)
    env.build_static_caches()
    return env


def test_predator_share_is_refused():
    with pytest.raises(ValueError, match="predator_share was removed"):
        _built({"predator_share": 0.4})


def test_ideal_intake_is_the_best_pair_at_the_whole_share():
    species = {"p": {"max_intake_rate": 0.5}, "q": {"energy_content": 100.0},
               "r": {"energy_content": 50.0}}
    pairs = {"p_preys_on_q": {"preys_on": True, "handling_time": 10.0},
             "p_preys_on_r": {"preys_on": True, "handling_time": 10.0}}
    intake, best = growth_budget.ideal_intake("p", species, pairs)
    assert best == "q"
    assert intake == pytest.approx(growth_budget.intake_ceiling(0.5, 10.0) * 100.0)


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
    plain, unit = _built(), _built({"accessible_fraction": 1.0})
    np.testing.assert_array_equal(plain.max_intake_mat, unit.max_intake_mat)
    np.testing.assert_array_equal(plain.handling_time_mat, unit.handling_time_mat)
    assert not unit._has_pair_access


@pytest.mark.parametrize("floors", [None, {"a_preys_on_b": 0.6}])
def test_pair_shares_match_reference(device, floors):
    env = make_env(min_split=0.0, extinction_factor=0.0, mortality=True,
                   pair_floors=floors,
                   pair_keys={PAIR: {"accessible_fraction": 0.3},
                              "b_preys_on_c": {"accessible_fraction": 0.6}})
    model = _assert_parity(env, device, ticks=8)
    assert model.pair_access is not None


def test_mareld2_benthivore_benthos_pair():
    # Section 221: predator_share is gone (library and engine); the
    # accessible fraction does not move the budget.
    import yaml
    root = Path(__file__).resolve().parents[1]
    library = yaml.safe_load((root / "fgconfig" / "fg_library.yaml").read_text())
    inter = library["interaction_definitions"]
    assert not any("predator_share" in v for v in inter.values() if isinstance(v, dict))
    # Section 223.12: the infauna pair's 0.2 moved into the infauna's own
    # refuge (tests/test_refuge.py); the epifauna pair keeps its share.
    pair = inter["benthivorous_fish_preys_on_mobile_epifauna"]
    assert 0.0 < pair["accessible_fraction"] < 1.0
    stripped = copy.deepcopy(library)
    stripped["interaction_definitions"]["benthivorous_fish_preys_on_mobile_epifauna"].pop(
        "accessible_fraction")
    g = [growth_budget.derived_growth_rate("benthivorous_fish", lib["species_definitions"],
                                           lib["interaction_definitions"])
         for lib in (library, stripped)]
    assert g[0] == pytest.approx(g[1])
