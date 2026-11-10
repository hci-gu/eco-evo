"""Closed grid edge for a DM with its own outside exchange (section 266.16;
state._build_edge_closure, decisions.build_action_mask, GPU action_mask).

Contracts asserted:

1. With migration on, a DM with an open pool (``outside_density``) or an open
   population's exchange (``recruit_import`` / ``emigration``) gets the
   closed-box move mask; every other DM keeps ``env.move_mask``.
2. The policy's action mask follows it: no move over the edge for a closed
   DM, so nothing it does emigrates.
3. With migration off every DM is closed, as before.
4. GPU parity with migration on and a closed DM.
5. In mareld2 exactly the five fish groups are closed; the mammals and the
   zooplankton keep the edge.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.environments.ecosystem_env import decisions, movement
from lib.environments.ecosystem_env.constants import EAST, MOVE_SLICE, NORTH, SOUTH, WEST
from test_gpu_ecosystem import _assert_parity, device, make_env  # noqa: F401

OPEN = {"outside_density": 2.0, "outside_exchange_rate": 0.2,
        "outside_fill": [0.6] * 12, "outside_entry": [{"edge": "west"}]}
STAGES = {"growth_ratio": 4.0, "graduation_rate": 1.0, "recruitment": [1.0] * 12,
          "start_share": 0.3, "emigration": [0.5, 0.2]}
DAYLIGHT = {"latitude_deg": 58.15, "tick_hours": 6, "start_tick": 600, "random_start": False}


def _env(migration=True, **kw):
    kw.setdefault("min_split", 0.0)
    kw.setdefault("extinction_factor", 0.0)
    env = make_env(migration=migration, **kw)
    env.build_static_caches()
    return env


def _edges_closed(mask):
    return (np.all(mask[NORTH, 0, :] == 0) and np.all(mask[SOUTH, -1, :] == 0)
            and np.all(mask[EAST, :, -1] == 0) and np.all(mask[WEST, :, 0] == 0))


def _edges_open(mask):
    return (np.all(mask[NORTH, 0, :] == 1) and np.all(mask[SOUTH, -1, :] == 1)
            and np.all(mask[EAST, :, -1] == 1) and np.all(mask[WEST, :, 0] == 1))


def test_open_pool_dm_is_closed_and_the_other_keeps_the_edge():
    env = _env(regen={"a": OPEN})
    assert list(env.dm_ids) == ["a", "b"]
    assert env.dm_edge_closed.tolist() == [True, False]
    assert _edges_closed(env.dm_move_mask[0])
    np.testing.assert_array_equal(env.dm_move_mask[1], env.move_mask)
    assert _edges_open(env.move_mask)
    # Inside the grid the closed mask still allows every move.
    assert np.all(env.dm_move_mask[0][:, 1:-1, 1:-1] == 1)


def test_open_population_exchange_closes_the_edge():
    env = _env(stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT))
    assert env.dm_edge_closed.tolist() == [True, False]
    plain = _env()
    assert plain.dm_edge_closed.tolist() == [False, False]
    np.testing.assert_array_equal(plain.dm_move_mask[0], plain.move_mask)


def test_migration_off_closes_every_dm():
    env = _env(migration=False, regen={"a": OPEN})
    assert _edges_closed(env.dm_move_mask[0]) and _edges_closed(env.dm_move_mask[1])


def test_action_mask_and_no_edge_emigration():
    env = _env(regen={"a": OPEN})
    mask = decisions.build_action_mask(env, MOVE_SLICE.stop + 1 + env.N_all)
    assert _edges_closed(mask[0, MOVE_SLICE])
    # A policy that puts all of a's mass on north and west: nothing leaves.
    actions = env.policy_controller.forward(env.get_observation())
    move = np.zeros_like(actions.move)
    move[0, NORTH] = 0.5 * mask[0, MOVE_SLICE][NORTH]
    move[0, WEST] = 0.5 * mask[0, MOVE_SLICE][WEST]
    actions.move = move.astype(actions.move.dtype)
    actions.rest = np.clip(1.0 - move.sum(axis=1), 0.0, 1.0).astype(actions.rest.dtype)
    seen = {}
    original = movement._edge_emigration

    def spy(b_out, r_out):
        out = original(b_out, r_out)
        seen.setdefault("b", out[0])
        return out
    movement._edge_emigration = spy
    try:
        movement.apply_movement(env, movement.apply_energy_costs(env, actions))
    finally:
        movement._edge_emigration = original
    assert float(seen["b"][0]) == 0.0


@pytest.mark.parametrize("closed", [True, False])
def test_gpu_matches_reference(device, closed):
    env = _env(regen={"a": OPEN} if closed else None, mortality=True, handling=0.3)
    assert bool(env.dm_edge_closed[0]) is closed
    model = _assert_parity(env, device, ticks=8, base_tol=1e-5)
    assert bool((model.dm_move_mask[0].reshape(4, env.H, env.W)[NORTH, 0] == 0).all()) is closed


def test_mareld2_closes_the_fish_groups():
    from lib.runners.training_progress import build_inference_env, evaluation_randomness
    cfg = {"project": str(ROOT / "mareld2.yaml"),
           "library": str(ROOT / "fgconfig" / "fg_library.yaml"), "grid": [16, 16], "mortality": True,
           "migration": True, "tick_hours": 6}
    with evaluation_randomness(1):
        env = build_inference_env(cfg, 1)
    closed = {fid for fid, c in zip(env.dm_ids, env.dm_edge_closed) if c}
    assert closed == {"planktivorous_fish", "benthivorous_fish", "whiting",
                      "predatory_fish", "mackerel"}
    for fid in ("porpoises", "seals", "zooplankton"):
        i = list(env.dm_ids).index(fid)
        np.testing.assert_array_equal(env.dm_move_mask[i], env.move_mask)
