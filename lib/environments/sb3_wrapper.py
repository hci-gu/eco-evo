import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from lib.environments.petting_zoo import env as petting_zoo_env
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.model import INPUT_SIZE, OUTPUT_SIZE, MODEL_OFFSETS, Action

class SB3Wrapper(VecEnv):
    def __init__(self, settings: Settings, species="cod", render_mode=None):
        self.settings = settings
        self.target_species = species
        self.species_map = build_species_map(settings)
        
        # Initialize the underlying PettingZoo environment
        self.env = petting_zoo_env(settings, self.species_map, render_mode=render_mode)
        self.env.reset()
        
        # Define spaces
        # Observation: Flattened 3x3 patch for each agent
        # INPUT_SIZE is already SINGLE_CELL_INPUT * 9
        self.obs_dim = INPUT_SIZE
        observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        
        # Action: Discrete(5) -> UP, DOWN, LEFT, RIGHT, EAT
        action_space = spaces.Discrete(OUTPUT_SIZE)
        
        # Number of environments = number of grid cells
        # observe() returns (W)*(W) patches (if we consider padding is handled internally)
        # Actually, observe() returns (N-2)*(M-2). N=W+2. So W*W.
        num_envs = settings.world_size * settings.world_size
        
        self.render_mode = render_mode
        super().__init__(num_envs, observation_space, action_space)
        
        self.actions = None
        self.dones = np.zeros(num_envs, dtype=bool)
        self.species_models = {}
        self.last_total_biomass = 0.0

    def set_species_models(self, models):
        print("[SB3Wrapper] Setting species models for other species.")
        """
        Sets the models to use for other species.
        models: Dictionary mapping species name to SB3 model (or compatible predict method).
        """
        self.species_models = models
        
    def reset(self):
        self.env.reset()
        # Fast forward to the target species' turn if necessary
        # For now, we assume the order is random but we just step until we hit our species?
        # Actually, the underlying env uses an agent_selector. 
        # We might need to force the turn or just step through.
        
        # Ideally, we want to control ONLY the target species.
        # The underlying env expects us to step() for whichever agent is selected.
        
        # Let's ensure we are at the target species' turn.
        self._ensure_target_species_turn()
        
        # Initialize biomass tracking
        biomass_channel = MODEL_OFFSETS[self.target_species]["biomass"]
        self.last_total_biomass = np.sum(self.env.world[..., biomass_channel])
        
        return self._get_obs()

    def step_async(self, actions):
        self.actions = actions

    def _compute_local_rewards(self):
        """
        Simplified reward: log sum of the surrounding 3x3 grid biomass for the acting species.
        """
        # Get the 3x3 patch observations for the target species
        # Shape: (N, 3, 3, C) where N = num_envs
        raw_obs = self.env.observations[self.target_species]
        
        # Get biomass channel index for target species
        biomass_idx = MODEL_OFFSETS[self.target_species]["biomass"]
        
        # Extract biomass from all 9 cells in the 3x3 patch for each agent
        # Shape: (N, 3, 3)
        biomass_patches = raw_obs[..., biomass_idx]
        
        # Sum biomass across the 3x3 grid for each agent
        # Shape: (N,)
        total_biomass = np.sum(biomass_patches, axis=(1, 2))
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        rewards = np.log(total_biomass + epsilon).astype(np.float32)
        
        return rewards

    def step_wait(self):
        # Convert discrete actions (N,) to one-hot (W, W, 5)
        actions_grid = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
        
        # Map flat indices back to (x, y)
        # actions is (N_envs,)
        rows, cols = np.unravel_index(np.arange(self.num_envs), (self.settings.world_size, self.settings.world_size))
        
        # Set the chosen action to 1.0
        actions_grid[rows, cols, self.actions] = 1.0
        
        # Step the environment for the target species
        self.env.step(actions_grid)
        
        # Now we need to step through OTHER species until it's our turn again or done
        self._ensure_target_species_turn()
        
        obs = self._get_obs()
        
        # Calculate rewards
        # Use local rewards instead of global biomass change
        rewards = self._compute_local_rewards()
        
        if not hasattr(self, "_debug_step"):
            self._debug_step = 0
        self._debug_step += 1
        if self._debug_step % 100 == 0:
            print(f"[SB3Wrapper] Step {self._debug_step}: Avg Reward {np.mean(rewards):.6f}")
        
        # Check terminations
        done = self.env.terminations[self.target_species] or self.env.truncations[self.target_species]
        self.dones[:] = done
        
        infos = [{} for _ in range(self.num_envs)]
        
        if done:
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
            
            obs = self.reset()
        
        return obs, rewards, self.dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        # Return a list of attributes for each environment
        # Since we share the same underlying env, we repeat the value
        val = getattr(self.env, attr_name)
        if indices is None:
            return [val] * self.num_envs
        else:
            # Handle indices if it's a list or int
            if isinstance(indices, int):
                return val
            return [val] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return getattr(self.env, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    # --- Helper Methods ---

    def _ensure_target_species_turn(self):
        """
        Steps the environment through other species' turns until it is the target species' turn
        or the episode ends.
        """
        while self.env.agent_selection != self.target_species:
            if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                break
            
            agent = self.env.agent_selection
            
            # For other species, we need a policy.
            # For now, use a random/noop policy or a simple heuristic.
            # If it's plankton, the env handles it automatically if we pass None/Empty?
            # The env.step() logic:
            # if agent == "plankton": self.env.step(self.empty_action)
            
            if agent == "plankton":
                # Plankton action is ignored/handled internally usually, but we need to pass something matching shape
                # Actually `petting_zoo.py` line 60: `self.env.step(self.empty_action)`
                # We need to replicate that behavior or just pass zeros.
                empty_action = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
                self.env.step(empty_action)
            elif agent in self.species_models:
                raw_obs = self.env.observations[agent] # Shape (N, 3, 3, C)
                flat_obs = raw_obs.reshape(self.num_envs, -1)
                
                actions, _ = self.species_models[agent].predict(flat_obs, deterministic=True)
                
                # Convert to grid
                actions_grid = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
                rows, cols = np.unravel_index(np.arange(self.num_envs), (self.settings.world_size, self.settings.world_size))
                actions_grid[rows, cols, actions] = 1.0
                
                self.env.step(actions_grid)
            else:
                # No model provided, take random actions
                random_actions = np.random.randint(0, OUTPUT_SIZE, size=self.num_envs)
                
                actions_grid = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
                rows, cols = np.unravel_index(np.arange(self.num_envs), (self.settings.world_size, self.settings.world_size))
                actions_grid[rows, cols, random_actions] = 1.0
                
                self.env.step(actions_grid)

    def _get_obs(self):
        """
        Returns flattened observations for the target species.
        Shape: (Num_Envs, Obs_Dim)
        """
        # self.env.observations[self.target_species] returns (Num_Patches, 3, 3, C)
        # We need to flatten this to (Num_Patches, C*3*3)
        
        raw_obs = self.env.observations[self.target_species] # Shape (N, 3, 3, C)
        
        # Flatten the observation patch (3, 3, C) into the feature dimension
        # Note: raw_obs is already a list of patches, so shape[0] is num_envs
        
        flat_obs = raw_obs.reshape(self.num_envs, -1)
        
        return flat_obs
