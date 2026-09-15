"""Population failure timing, warning incentives, and CPU/worker consistency."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from lib.runners.population_stability import (
    PopulationStability, StabilityScore, add_population_arguments, population_options,
)
from lib.runners.trainer import ARSTrainer
from lib.runners import parallel_worker


class ScriptedEnvironment:
    def __init__(self, trajectory, initial=(100.0, 100.0)):
        self.dm_ids = ["predator", "prey"]
        self.fgs = {f: SimpleNamespace(biomass=np.array([b]), energy_reserve=np.array([b]),
                                        params={"energy_content": 1.0})
                    for f, b in zip(self.dm_ids, initial)}
        self.trajectory = trajectory
        self.steps = 0
        self.policy_controller = SimpleNamespace(forward=lambda obs: None)

    def get_observation(self):
        return None

    def step(self, actions=None):
        for f, ratio in zip(self.dm_ids, self.trajectory[self.steps]):
            self.fgs[f].biomass[:] = 100 * ratio
            self.fgs[f].energy_reserve[:] = 100 * ratio
        self.steps += 1


@pytest.mark.parametrize("ratio,expected", [
    (1.0, 0.0), (0.20, np.log(0.20)), (0.15, np.log(0.15) - 0.5),
    (0.10, np.log(0.10) - 1), (0.099, -5), (2.5, np.log(2.5)),
    (2.75, 1 - 0.5), (3.0, -5), (4.0, -5),
])
def test_warning_ramps_energy_cap_and_exact_boundaries(ratio, expected):
    env = ScriptedEnvironment([(ratio, ratio)])
    score = StabilityScore(env, PopulationStability())
    env.step()
    score.step(env)
    assert score.results(1)["prey"] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("coevolution", [False, True])
def test_failed_world_stops_and_pads_original_horizon(worker, coevolution, monkeypatch):
    # Healthy first tick, then an untrained prey species breaches the ceiling.
    env = ScriptedEnvironment([(1, 1), (1, 3), (1, 1), (1, 1)])
    builder = lambda seed=None: env
    config = PopulationStability()
    if worker:
        monkeypatch.setattr(parallel_worker, "_ENV_BUILDER", builder)
        monkeypatch.setattr(parallel_worker, "_POLICIES", {})
        task = dict(weights_dict={}, n_ticks=4, alpha=1, beta=1,
                    population_stability=config, fg_to_train="predator", fg_list=env.dm_ids)
        evaluate = parallel_worker._evaluate_coevo_task if coevolution else parallel_worker._evaluate_task
        reward, _, _ = evaluate(task)
    else:
        trainer = ARSTrainer(builder, {}, population_stability=config)
        reward, _, _ = (trainer._evaluate_coevo(env.dm_ids, 4) if coevolution
                        else trainer._evaluate("predator", 4))
    assert env.steps == 2
    if coevolution:
        assert reward == {"predator": -3.75, "prey": -3.75}
    else:
        assert reward == -3.75


def test_delaying_failure_improves_fitness_and_recovery_does_not_reset_it():
    rewards = []
    for failure_tick in (0, 1, 2):
        path = [(1, 1)] * 3
        path[failure_tick] = (0.01, 1)
        env = ScriptedEnvironment(path)
        score = StabilityScore(env, PopulationStability())
        for _ in path:
            env.step()
            score.step(env)
        rewards.append(score.results(3)["predator"])
    assert rewards == pytest.approx([-5, -10 / 3, -5 / 3])


def test_absent_species_is_excluded_and_warning_is_shared():
    env = ScriptedEnvironment([(1, 0)], initial=(100, 0))
    score = StabilityScore(env, PopulationStability())
    env.step()
    score.step(env)
    assert not score.failed
    assert score.results(1)["predator"] == 0
    env = ScriptedEnvironment([(1, 0.15)])
    score = StabilityScore(env, PopulationStability())
    env.step()
    score.step(env)
    assert score.results(1)["predator"] == pytest.approx(-0.5)


@pytest.mark.parametrize("kwargs", [{"lower": 0}, {"upper": float("nan")},
                                   {"warning_lower": 0.1}, {"warning_upper": 3},
                                   {"warning_lower": 1.1}, {"lower": float("inf")}])
def test_invalid_bounds_rejected(kwargs):
    with pytest.raises(ValueError, match="Population bounds"):
        PopulationStability(**kwargs)


def test_cli_aliases_and_incompatible_reward_flags():
    parser = argparse.ArgumentParser()
    add_population_arguments(parser)
    parser.set_defaults(integral_reward=True, legacy_reward=False)
    assert population_options(parser.parse_args([])) is None
    args = parser.parse_args(["--population_stability", "--population-min", "0.05"])
    assert population_options(args).lower == 0.05
    args.legacy_reward = True
    with pytest.raises(ValueError, match="requires integral"):
        population_options(args)
    with pytest.raises(ValueError, match="requires integral"):
        ARSTrainer(None, {}, population_stability=PopulationStability(), integral_reward=False)
