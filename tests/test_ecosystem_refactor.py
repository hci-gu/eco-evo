import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.environments.ecosystem import EcosystemEnvironment  # noqa: E402
from lib.environments.ecosystem_env.constants import MOVE_SLICE, NORTH  # noqa: E402
from lib.environments.ecosystem_env import movement, population_change, predation  # noqa: E402
from lib.environments.ecosystem_env.state import (  # noqa: E402
    ActionProbabilities,
    ActionSettlement,
)
from lib.world.functional_group import FunctionalGroup  # noqa: E402


def _fg(group_id, params, biomass, energy_ratio=0.5):
    fg = FunctionalGroup(group_id, params)
    fg.initialize_state(
        np.asarray(biomass, dtype=np.float32).shape,
        initial_biomass=np.asarray(biomass, dtype=np.float32),
        initial_energy_ratio=energy_ratio,
    )
    return fg


def test_cache_masks_grid_edges_when_migration_disabled():
    fgs = {
        "dm": _fg(
            "dm",
            {"is_decision_maker": True, "movement_speed": 1.0},
            np.ones((3, 3), dtype=np.float32),
        )
    }
    env = EcosystemEnvironment({"width": 3, "height": 3}, fgs, migration=False)
    env.build_static_caches()

    assert np.all(env.move_mask[0, 0, :] == 0.0)
    assert np.all(env.move_mask[2, -1, :] == 0.0)
    assert np.all(env.move_mask[1, :, -1] == 0.0)
    assert np.all(env.move_mask[3, :, 0] == 0.0)
    assert env._edge_imm_weights.sum() == pytest.approx(1.0)


def test_cache_keeps_edge_moves_available_when_migration_enabled():
    fgs = {
        "dm": _fg(
            "dm",
            {"is_decision_maker": True, "movement_speed": 1.0},
            np.ones((3, 3), dtype=np.float32),
        )
    }
    env = EcosystemEnvironment({"width": 3, "height": 3}, fgs, migration=True)
    env.build_static_caches()

    assert np.all(env.move_mask[0, 0, :] == 1.0)
    assert np.all(env.move_mask[2, -1, :] == 1.0)
    assert np.all(env.move_mask[1, :, -1] == 1.0)
    assert np.all(env.move_mask[3, :, 0] == 1.0)


def test_population_change_clears_nonviable_biomass_and_empty_energy():
    fgs = {
        "dm": _fg(
            "dm",
            {
                "is_decision_maker": True,
                "min_split_biomass": 1000.0,
                "extinction_threshold_factor": 0.5,
            },
            [[0.0, 0.25, 1.0]],
            energy_ratio=0.5,
        )
    }
    env = EcosystemEnvironment({"width": 3, "height": 1}, fgs)
    env.build_static_caches()
    env.ordered_fg_ids = list(env.fgs.keys())
    env.fgs["dm"].energy_reserve[0, 0] = 99.0

    population_change.apply_population_change(env)

    np.testing.assert_allclose(env.fgs["dm"].biomass, [[0.0, 0.0, 1.0]])
    assert env.fgs["dm"].energy_reserve[0, 0] == pytest.approx(0.0)
    assert env.fgs["dm"].energy_reserve[0, 1] == pytest.approx(0.0)
    assert env.loss_starvation["dm"] == pytest.approx(0.0)


def test_predation_respects_current_hide_visibility_floor():
    prey = _fg(
        "prey",
        {
            "is_decision_maker": True,
            "visibility_floor": 0.5,
            "max_energy_reserve": 10.0,
        },
        [[10.0]],
    )
    pred = _fg(
        "pred",
        {
            "is_decision_maker": True,
            "max_energy_reserve": 10.0,
            "max_intake_rate": 1.0,
            "menu": ["prey"],
            "interaction": {
                "pred_preys_on_prey": {
                    "preys_on": True,
                    "assimilation_factor": 1.0,
                    "handling_time": 0.0,
                }
            },
        },
        [[10.0]],
        energy_ratio=0.0,
    )
    env = EcosystemEnvironment({"width": 1, "height": 1}, {"pred": pred, "prey": prey})
    env.build_static_caches()

    rest = np.zeros((env.N_dm, 1, 1), dtype=np.float32)
    rest[env.dm_ids.index("prey"), 0, 0] = 1.0
    eat = np.zeros((env.N_dm, env.N_all, 1, 1), dtype=np.float32)
    eat[
        env.dm_ids.index("pred"),
        env.global_fg_order.index("prey"),
        0,
        0,
    ] = 1.0
    actions = ActionProbabilities(
        move=np.zeros((env.N_dm, 4, 1, 1), dtype=np.float32),
        rest=rest,
        eat=eat,
    )

    predation.apply_predation(env, actions)

    assert env.fgs["prey"].biomass[0, 0] == pytest.approx(5.0)
    assert env.loss_predation["prey"] == pytest.approx(5.0)
    assert env.intake_by_pred_prey["pred"]["prey"] == pytest.approx(5.0)


def test_energy_costs_settle_actions_without_moving_biomass():
    fgs = {
        "dm": _fg(
            "dm",
            {
                "is_decision_maker": True,
                "movement_speed": 1.0,
                "max_energy_reserve": 10.0,
                "resting_metabolism": 1.0,
                "resting_cost": 1.0,
                "feeding_cost": 2.0,
                "movement_cost": 3.0,
            },
            [[10.0]],
            energy_ratio=1.0,
        )
    }
    env = EcosystemEnvironment({"width": 1, "height": 1}, fgs)
    env.build_static_caches()
    env.fgs["dm"].temp_energy_gains[...] = 3.0

    move = np.zeros((1, 4, 1, 1), dtype=np.float32)
    move[0, NORTH, 0, 0] = 0.5
    eat = np.zeros((1, 1, 1, 1), dtype=np.float32)
    eat[0, 0, 0, 0] = 0.3
    actions = ActionProbabilities(
        move=move,
        rest=np.full((1, 1, 1), 0.2, dtype=np.float32),
        eat=eat,
    )

    settlement = movement.apply_energy_costs(env, actions)

    assert env.fgs["dm"].biomass[0, 0] == pytest.approx(10.0)
    assert env.fgs["dm"].energy_reserve[0, 0] == pytest.approx(100.0)
    assert settlement.stationary_biomass[0, 0, 0] == pytest.approx(5.0)
    assert settlement.stationary_reserve[0, 0, 0] == pytest.approx(45.0)
    assert settlement.moving_biomass[0, NORTH, 0, 0] == pytest.approx(5.0)
    assert settlement.moving_reserve[0, NORTH, 0, 0] == pytest.approx(35.0)


@pytest.mark.parametrize("prey_count, sparse_intake", [(1, 0.004950495), (2, 0.045454545)])
def test_holling_intake_increases_and_saturates_with_prey_density(prey_count, sparse_intake):
    density = np.array([[0.1, 1.0, 100.0, 1000.0]], dtype=np.float32)
    prey_ids = [f"prey_{i}" for i in range(prey_count)]
    fgs = {fid: _fg(fid, {}, density) for fid in prey_ids}
    fgs["pred"] = _fg(
        "pred",
        {
            "is_decision_maker": True,
            "max_intake_rate": 0.5,
            "menu": prey_ids,
            "interaction": {
                f"pred_preys_on_{fid}": {"handling_time": 2.0}
                for fid in prey_ids
            },
        },
        np.ones_like(density),
        energy_ratio=0.0,
    )
    env = EcosystemEnvironment({"width": 4, "height": 1}, fgs)
    eat = np.zeros((1, env.N_all, 1, 4), dtype=np.float32)
    eat[0, env.global_fg_order.index("prey_0")] = 1.0
    actions = ActionProbabilities(
        move=np.zeros((1, 4, 1, 4), dtype=np.float32),
        rest=np.zeros((1, 1, 4), dtype=np.float32),
        eat=eat,
    )

    predation.apply_predation(env, actions)

    intake = density - env.fgs["prey_0"].biomass
    assert np.all(np.diff(intake[0]) > 0.0)
    assert intake[0, 0] == pytest.approx(sparse_intake)
    assert intake[0, -1] == pytest.approx(0.5, rel=0.002)
    assert np.all(intake <= 0.5)
    assert env.cache.has_holling3 == (prey_count == 1)


@pytest.mark.parametrize("prey_count, expected_intake", [(1, 0.00005), (2, 0.005)])
def test_zero_handling_time_keeps_specialist_response_and_type_i_fallback(
    prey_count, expected_intake
):
    prey_ids = [f"prey_{i}" for i in range(prey_count)]
    fgs = {fid: _fg(fid, {}, [[0.1]]) for fid in prey_ids}
    fgs["pred"] = _fg(
        "pred",
        {"is_decision_maker": True, "max_intake_rate": 0.5, "menu": prey_ids},
        [[0.01]],
        energy_ratio=0.0,
    )
    env = EcosystemEnvironment({"width": 1, "height": 1}, fgs)
    eat = np.zeros((1, env.N_all, 1, 1), dtype=np.float32)
    eat[0, env.global_fg_order.index("prey_0")] = 1.0
    actions = ActionProbabilities(
        move=np.zeros((1, 4, 1, 1), dtype=np.float32),
        rest=np.zeros((1, 1, 1), dtype=np.float32),
        eat=eat,
    )

    predation.apply_predation(env, actions)

    assert env.intake_by_pred_prey["pred"]["prey_0"] == pytest.approx(expected_intake)


@pytest.mark.parametrize("energy_ratio", [0.1, 1.0])
@pytest.mark.parametrize("move_fraction", [0.0, 0.4])
def test_action_costs_scale_with_chosen_fraction_before_movement(energy_ratio, move_fraction):
    fg = _fg(
        "dm",
        {
            "is_decision_maker": True,
            "max_energy_reserve": 10.0,
            "resting_metabolism": 1.0,
            "resting_cost": 1.0,
            "feeding_cost": 2.0,
            "movement_cost": 3.0,
            "movement_speed": 0.25,
        },
        [[10.0]],
        energy_ratio=energy_ratio,
    )
    env = EcosystemEnvironment({"width": 1, "height": 1}, {"dm": fg})
    fg.temp_energy_gains[...] = 3.0
    actions = ActionProbabilities(
        move=np.array([0.25, 0.75, 0.0, 0.0], dtype=np.float32).reshape(1, 4, 1, 1)
        * move_fraction,
        rest=np.full((1, 1, 1), 0.7 - move_fraction, dtype=np.float32),
        eat=np.full((1, 1, 1, 1), 0.3, dtype=np.float32),
    )

    settlement = movement.apply_energy_costs(env, actions)

    if energy_ratio == 1.0:
        expected_stationary = 54.0 if move_fraction == 0.4 else 90.0
        expected_moving = [7.0, 21.0, 0.0, 0.0] if move_fraction == 0.4 else [0.0] * 4
    else:
        expected_stationary = 3.0
        expected_moving = [0.0] * 4
    np.testing.assert_allclose(settlement.stationary_reserve, expected_stationary)
    np.testing.assert_allclose(settlement.moving_reserve.ravel(), expected_moving)
    assert fg.biomass[0, 0] == pytest.approx(10.0)
    assert fg.energy_reserve[0, 0] == pytest.approx(100.0 * energy_ratio)


def test_apply_movement_only_moves_settled_moving_partition():
    fgs = {
        "dm": _fg(
            "dm",
            {
                "is_decision_maker": True,
                "movement_speed": 1.0,
                "max_energy_reserve": 10.0,
            },
            np.zeros((3, 3), dtype=np.float32),
        )
    }
    env = EcosystemEnvironment({"width": 3, "height": 3}, fgs)
    env.build_static_caches()

    stationary_biomass = np.zeros((1, 3, 3), dtype=np.float32)
    stationary_reserve = np.zeros((1, 3, 3), dtype=np.float32)
    moving_biomass = np.zeros((1, 4, 3, 3), dtype=np.float32)
    moving_reserve = np.zeros((1, 4, 3, 3), dtype=np.float32)
    stationary_biomass[0, 1, 1] = 2.0
    stationary_reserve[0, 1, 1] = 4.0
    moving_biomass[0, NORTH, 1, 1] = 3.0
    moving_reserve[0, NORTH, 1, 1] = 6.0

    settlement = ActionSettlement(
        stationary_biomass=stationary_biomass,
        stationary_reserve=stationary_reserve,
        moving_biomass=moving_biomass,
        moving_reserve=moving_reserve,
    )
    movement.apply_movement(env, settlement)

    expected_biomass = np.zeros((3, 3), dtype=np.float32)
    expected_reserve = np.zeros((3, 3), dtype=np.float32)
    expected_biomass[0, 1] = 3.0
    expected_biomass[1, 1] = 2.0
    expected_reserve[0, 1] = 6.0
    expected_reserve[1, 1] = 4.0
    np.testing.assert_allclose(env.fgs["dm"].biomass, expected_biomass)
    np.testing.assert_allclose(env.fgs["dm"].energy_reserve, expected_reserve)


def test_get_observation_returns_features_and_action_mask():
    fgs = {
        "dm": _fg(
            "dm",
            {
                "is_decision_maker": True,
                "movement_speed": 0.0,
                "growth_rate": 0.0,
            },
            np.ones((2, 2), dtype=np.float32),
        )
    }
    env = EcosystemEnvironment({"width": 2, "height": 2}, fgs)

    observation = env.get_observation()

    assert observation.features.shape == (1, 4, env.max_in_dim)
    assert observation.raw_features.shape == observation.features.shape
    assert observation.action_mask.shape == (1, 6, 2, 2)
    assert np.all(observation.action_mask[:, MOVE_SLICE] == 0.0)
    assert np.all(observation.action_mask.sum(axis=1) > 0.0)


def test_policy_controller_forward_respects_action_mask():
    fgs = {
        "dm": _fg(
            "dm",
            {
                "is_decision_maker": True,
                "movement_speed": 0.0,
                "growth_rate": 0.0,
            },
            np.ones((2, 2), dtype=np.float32),
        )
    }
    env = EcosystemEnvironment({"width": 2, "height": 2}, fgs)

    observation = env.get_observation()
    actions = env.policy_controller.forward(observation)

    assert actions.move.shape == (1, 4, 2, 2)
    assert actions.rest.shape == (1, 2, 2)
    assert actions.eat.shape == (1, 1, 2, 2)
    assert np.all(actions.move == 0.0)
    np.testing.assert_allclose(
        actions.move.sum(axis=1) + actions.rest + actions.eat.sum(axis=1),
        np.ones((1, 2, 2), dtype=np.float32),
    )


def test_step_requires_actions_argument():
    fgs = {
        "dm": _fg(
            "dm",
            {
                "is_decision_maker": True,
                "movement_speed": 0.0,
                "growth_rate": 0.0,
            },
            np.ones((1, 1), dtype=np.float32),
        )
    }
    env = EcosystemEnvironment({"width": 1, "height": 1}, fgs)

    with pytest.raises(TypeError):
        env.step()


def test_step_with_manual_actions_updates_biomass_without_policy():
    producer = _fg(
        "producer",
        {
            "is_decision_maker": False,
            "energy_content": 1.0,
            "growth_rate": 0.0,
            "max_carrying_capacity": 10.0,
        },
        [[10.0]],
    )
    consumer = _fg(
        "consumer",
        {
            "is_decision_maker": True,
            "movement_speed": 0.0,
            "max_energy_reserve": 100.0,
            "growth_rate": 0.0,
            "maintenance_level": 0.0,
            "max_intake_rate": 1.0,
            "menu": ["producer"],
            "interaction": {
                "consumer_preys_on_producer": {
                    "preys_on": True,
                    "assimilation_factor": 1.0,
                    "handling_time": 0.0,
                }
            },
        },
        [[10.0]],
        energy_ratio=0.0,
    )
    env = EcosystemEnvironment(
        {"width": 1, "height": 1},
        {"consumer": consumer, "producer": producer},
    )
    env.build_static_caches()

    eat = np.zeros((env.N_dm, env.N_all, 1, 1), dtype=np.float32)
    eat[
        env.dm_ids.index("consumer"),
        env.global_fg_order.index("producer"),
        0,
        0,
    ] = 1.0
    actions = ActionProbabilities(
        move=np.zeros((env.N_dm, 4, 1, 1), dtype=np.float32),
        rest=np.zeros((env.N_dm, 1, 1), dtype=np.float32),
        eat=eat,
    )

    env.step(actions)

    assert env.tick_count == 1
    assert env.fgs["producer"].biomass[0, 0] < 10.0
    assert env.fgs["consumer"].energy_reserve[0, 0] > 0.0


def test_step_runs_through_extracted_environment_from_public_import():
    fgs = {
        "producer": _fg(
            "producer",
            {
                "is_decision_maker": False,
                "growth_rate": 0.0,
                "max_carrying_capacity": 10.0,
            },
            np.ones((2, 2), dtype=np.float32),
        ),
        "consumer": _fg(
            "consumer",
            {
                "is_decision_maker": True,
                "movement_speed": 0.0,
                "max_energy_reserve": 10.0,
                "growth_rate": 0.0,
                "maintenance_level": 0.0,
                "menu": ["producer"],
                "interaction": {
                    "consumer_preys_on_producer": {
                        "preys_on": True,
                        "assimilation_factor": 1.0,
                        "handling_time": 0.0,
                    }
                },
            },
            np.ones((2, 2), dtype=np.float32),
            energy_ratio=0.0,
        ),
    }
    env = EcosystemEnvironment({"width": 2, "height": 2}, fgs)

    observation = env.get_observation()
    actions = env.policy_controller.forward(observation)
    env.step(actions)

    assert env.tick_count == 1
    assert env.prev_hidden_frac.shape == (2, 2, 2)
    np.testing.assert_allclose(
        env.prev_hidden_frac[env.global_fg_order.index("consumer")],
        actions.rest[env.dm_ids.index("consumer")],
    )
    for fg in env.fgs.values():
        assert np.isfinite(fg.biomass).all()
        assert np.isfinite(fg.energy_reserve).all()
