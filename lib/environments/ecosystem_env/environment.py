import numpy as np

from lib.world.grid import Grid
from lib.environments.ecosystem_env import (
    decisions,
    grid_masks,
    impacts,
    movement,
    policies as policy_module,
    population_change,
    predation,
    state,
)


class EcosystemEnvironment:
    def __init__(
        self,
        grid_config,
        functional_groups,
        policies=None,
        observable_impact_vars=None,
        apply_natural_mortality=True,
        migration=False,
    ):
        self.grid = Grid(**grid_config)
        self.apply_natural_mortality = bool(apply_natural_mortality)
        self.migration = bool(migration)
        self.fgs = functional_groups
        self.policies = policies or {}
        self.policy_controller = policy_module.PolicyController(self)
        self.observable_impact_vars = list(observable_impact_vars or [])
        self.tick_count = 0
        self.global_fg_order = sorted(self.fgs.keys())
        self._static_built = False
        self.cache = None

        self._season_phase = {
            fid: float(
                np.random.uniform(
                    0.0,
                    max(
                        1.0,
                        float(getattr(fg, "seasonal_period", 0.0) or 0.0),
                    ),
                )
            )
            for fid, fg in self.fgs.items()
        }

        self.loss_impact = {fid: 0.0 for fid in self.fgs}
        self.loss_predation = {fid: 0.0 for fid in self.fgs}
        self.loss_starvation = {fid: 0.0 for fid in self.fgs}
        self.intake_by_pred_prey = {fid: {} for fid in self.fgs}

    def build_static_caches(self):
        state.build_static_caches(self)

    def rebuild_batched_weights(self):
        policy_module.rebuild_batched_weights(self)

    def get_observation(self):
        return decisions.build_observation(self)

    def step(self, actions):
        if not self._static_built:
            self.build_static_caches()

        self.ordered_fg_ids = list(self.fgs.keys())
        np.random.shuffle(self.ordered_fg_ids)

        # impacts.apply_impact_mortality(self) currently just empty maps so we can skip it for now
        predation.apply_predation(self, actions)
        action_settlement = movement.apply_energy_costs(self, actions)
        movement.apply_movement(self, action_settlement)
        population_change.apply_population_change(self)

        decisions.update_hidden_state(self, actions)
        self.tick_count += 1

    def tick(self):
        observation = self.get_observation()
        actions = self.policy_controller.forward(observation)
        self.step(actions)
