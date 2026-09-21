import numpy as np
import torch

from lib.world.grid import Grid
from lib.environments.ecosystem_env import (
    decisions,
    currents as current_module,
    grid_masks,
    impacts as impact_module,
    interactions as interaction_module,
    movement,
    observations,
    policies as policy_module,
    population_change,
    predation,
    source_tracking,
    state,
)
from lib.environments.ecosystem_env.state import ActionProbabilities


def _looks_like_policies(candidate):
    """True when the third positional ctor argument is a policies mapping.

    The monolithic ctor took ``interactions`` third, the modular one
    ``policies``; both call styles are still in use (train.py and the
    tests pass ``{}`` or a policies dict, the ab_test_*.py scripts pass
    the interaction definitions). A policies mapping maps fg_id to a
    torch module (or None), which no interaction definition ever does.
    """
    if not isinstance(candidate, dict):
        return False
    if not candidate:
        # Ambiguous but inert: an empty mapping carries no information
        # either way, and the modular callers pass ``{}`` for policies.
        return True
    return all(
        value is None
        or isinstance(value, torch.nn.Module)
        or hasattr(value, "get_action_logits_torch")
        for value in candidate.values()
    )


class EcosystemEnvironment:
    def __init__(
        self,
        grid_config,
        functional_groups,
        interactions=None,
        policies=None,
        observable_impact_vars=None,
        apply_natural_mortality=True,
        mortality_multiplier=1.0,
        migration=False,
        currents=None,
        current_world_seed=0,
        local_reward=None,
    ):
        if policies is None and _looks_like_policies(interactions):
            policies, interactions = interactions, None

        self.grid = Grid(**grid_config)
        self.apply_natural_mortality = bool(apply_natural_mortality)
        # Global scale factor on every FG's ``natural_mortality``
        # (``--mortality_multiplier``). 1.0 leaves the tick unchanged;
        # 0.0 is equivalent to ``--mortality off``. Negative values make
        # no sense as a mortality rate and are clamped away.
        self.mortality_multiplier = max(
            0.0, float(mortality_multiplier if mortality_multiplier is not None
                       else 1.0))
        self.migration = bool(migration)
        self.currents = currents
        self.current_world_seed = int(current_world_seed or 0)
        self.fgs = functional_groups
        self.interactions = interactions
        self.policies = policies or {}
        # Ordered list of impact_ids that are exposed to the policy network
        # as observation channels (one centre slot plus one slot per
        # neighbour direction each, see ``observations``).
        self.observable_impact_vars = list(observable_impact_vars or [])
        self.policy_controller = policy_module.PolicyController(self)
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

        self.loss_predation = {fid: 0.0 for fid in self.fgs}
        self.loss_starvation = {fid: 0.0 for fid in self.fgs}
        self.loss_impact = {fid: 0.0 for fid in self.fgs}
        self._extinction_events = {fid: 0 for fid in self.fgs}
        self.intake_by_pred_prey = {fid: {} for fid in self.fgs}
        # Local reward (``--local_reward``): when set to a
        # ``LocalRewardConfig`` the tick tracks, per decision maker and
        # per cell, where the biomass that started the tick in that cell
        # ended up, and accumulates the aggregated ratio B/A. ``None``
        # (default) leaves the tick byte-identical to before.
        self.local_reward = source_tracking.LocalRewardConfig.from_dict(
            local_reward)
        source_tracking.reset(self)
        self.build_static_caches()

    # ---------- Static caches ----------
    def build_static_caches(self):
        state.build_static_caches(self)

    def rebuild_batched_weights(self):
        policy_module.rebuild_batched_weights(self)

    # ---------- Observation / decisions ----------
    def get_observation(self):
        return decisions.build_observation(self)

    def calculate_decisions(self):
        """Full observation -> policy forward pass; publishes ``pi_*``."""
        observation = self.get_observation()
        actions = self.policy_controller.forward(observation)
        self.publish_action_probabilities(actions)
        return actions

    def publish_action_probabilities(self, actions):
        """Expose the action distribution under the monolith's names.

        External probes and the legacy no-argument stage methods read
        ``pi_move`` / ``pi_rest`` / ``pi_eat`` straight off the env, with
        shapes (N_dm, 4, H, W), (N_dm, H, W) and (N_dm, N_all, H, W).
        ``pi`` keeps the legacy per-FG mapping (always None entries).
        """
        self.pi_move = actions.move
        self.pi_rest = actions.rest
        self.pi_eat = actions.eat
        if self.N_dm == 0:
            self.pi = {fid: None for fid in self.fgs}
        else:
            self.pi = {
                fid: None for fid in self.fgs
                if not self.fgs[fid].is_decision_maker
            }
        decisions.update_hidden_state(self, actions)

    def action_probabilities(self):
        """The published ``pi_*`` attributes as an ActionProbabilities."""
        return ActionProbabilities(
            move=self.pi_move,
            rest=self.pi_rest,
            eat=self.pi_eat,
        )

    # ---------- Tick ----------
    def step(self, actions=None):
        """Run one tick. ``actions=None`` runs the full monolith tick.

        With no argument the observation is built, the policies are
        evaluated and the resulting action distribution is applied, which
        is the monolithic ``step()`` contract. Passing ``actions``
        (an ``ActionProbabilities``) skips the policy forward pass and
        applies the supplied distribution instead, which is what the
        tests and the offline probes use.
        """
        if not self._static_built:
            self.build_static_caches()

        self.ordered_fg_ids = list(self.fgs.keys())
        np.random.shuffle(self.ordered_fg_ids)

        # A(c, t) for the local reward must be the state the policy saw,
        # i.e. before any mortality is applied this tick.
        if source_tracking.is_enabled(self):
            source_tracking.begin_tick(self)

        if actions is None:
            actions = self.calculate_decisions()
        else:
            self.publish_action_probabilities(actions)

        # Method.pdf steg 1-6: direct impact mortality m_X^Impact is
        # applied *before* predation, so prey biomass available to
        # predators already reflects impact losses for this tick.
        impact_module.apply_impact_mortality(self)
        predation.apply_predation(self, actions)
        action_settlement = movement.apply_energy_costs(self, actions)
        movement.apply_movement(self, action_settlement)
        current_module.apply_currents(self)
        population_change.apply_population_change(self, clip_nonviable=False)
        grid_masks.apply_accessibility_biomass_mask(self)
        # Extinction threshold: zero cells whose biomass has shrunk below
        # half an indivisible unit after every shrink step. Run last so
        # the next tick's observation never sees float32 subnormals.
        population_change.apply_extinction_threshold(self)

        # Q(c, t+1) is now final, so the source shares recorded in
        # ``apply_movement`` can be cashed in.
        if source_tracking.is_enabled(self):
            source_tracking.end_tick(self)

        decisions.update_hidden_state(self, actions)
        self.tick_count += 1

    def tick(self):
        observation = self.get_observation()
        actions = self.policy_controller.forward(observation)
        self.step(actions)

    # ---------- Legacy (monolith) API aliases ----------
    # The monolithic EcosystemEnvironment exposed these names; probes,
    # the ab_test_*.py scripts and the test suite still call them, so the
    # modular env stays a drop-in replacement. The stage methods take no
    # arguments and read the published ``pi_*`` attributes.
    _build_static_caches = build_static_caches
    _rebuild_batched_weights = rebuild_batched_weights
    _calculate_decisions = calculate_decisions
    _holling_a_eff = staticmethod(interaction_module.holling_a_eff)
    _extract_impact_table = staticmethod(impact_module.extract_impact_table)
    _interp_impact = staticmethod(impact_module.interp_impact)

    def _build_observation_batch(self):
        return observations.build_observation_batch(self)

    def _apply_predation(self):
        predation.apply_predation(self, self.action_probabilities())

    def _apply_movement(self):
        actions = self.action_probabilities()
        movement.apply_movement(self, movement.apply_energy_costs(self, actions))

    def _apply_growth(self):
        population_change.apply_population_change(self, clip_nonviable=False)

    def _compute_impact_mortality(self, fg):
        return impact_module.compute_impact_mortality(self, fg)

    def _apply_impact_mortality(self):
        impact_module.apply_impact_mortality(self)

    def _apply_accessibility_biomass_mask(self):
        grid_masks.apply_accessibility_biomass_mask(self)

    def _apply_extinction_threshold(self):
        population_change.apply_extinction_threshold(self)

    def _suppress_subthreshold_splits(self, b_out, r_out, b_total_in,
                                      r_total_in):
        return movement.suppress_subthreshold_splits(
            self, b_out, r_out, b_total_in, r_total_in)
