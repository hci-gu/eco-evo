import numpy as np
import gymnasium as gym
from gymnasium import spaces
from lib.environments.petting_zoo import env as petting_zoo_env
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.model import INPUT_SIZE, OUTPUT_SIZE, MODEL_OFFSETS

class GlobalEcoEvoEnv(gym.Env):
    """
    A single-agent wrapper for the ecosystem where the 'agent' controls
    ALL individuals of a specific species simultaneously.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, settings: Settings, species="cod", render_mode=None):
        super().__init__()
        self.settings = settings
        self.target_species = species
        self.species_map = build_species_map(settings)
        self.render_mode = render_mode
        
        # Underlying PettingZoo environment
        self.env = petting_zoo_env(settings, self.species_map, render_mode=render_mode)
        
        # Observation Space: The entire map (H, W, C)
        # We transpose to (C, H, W) for SB3 CnnPolicy preference usually, 
        # but SB3 automatically handles (H, W, C) if we tell it.
        # Let's stick to (H, W, C) as that's what our data is.
        # Actually, SB3 CnnPolicy expects (C, H, W) or (H, W, C). 
        # We'll use (H, W, C) and ensure 'channels_last' is handled or just transpose if needed.
        # Our 'INPUT_SIZE' in model.py was for a flattened 3x3 patch. 
        # Here we want the raw channels of the map.
        
        # Calculate number of channels
        # The world tensor has shape (H, W, C).
        # We need to know C.
        self.n_channels = self.env.env.world.shape[2]
        
        # SB3 CnnPolicy expects (C, H, W)
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, # Biomass/Energy can be > 1
            shape=(self.n_channels, self.settings.world_size, self.settings.world_size), 
            dtype=np.float32
        )
        
        # Action Space: One discrete action (0-4) for EVERY cell.
        # MultiDiscrete of shape (H * W)
        self.n_cells = self.settings.world_size * self.settings.world_size
        self.action_space = spaces.MultiDiscrete(np.full(self.n_cells, OUTPUT_SIZE))
        
        self.species_models = {}
        self.last_total_biomass = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        
        # Ensure it's our turn
        self._ensure_target_species_turn()
        
        # Initialize biomass tracking
        biomass_channel = MODEL_OFFSETS[self.target_species]["biomass"]
        self.last_total_biomass = np.sum(self.env.env.world[..., biomass_channel])
        
        return self._get_obs(), {}

    def step(self, action):
        # action is an array of shape (N_Cells,) with values 0-4
        
        # 1. Convert flat action vector to (W, W, 5) one-hot grid
        actions_grid = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
        
        # Reshape action to (W, W)
        action_map = action.reshape(self.settings.world_size, self.settings.world_size)
        
        # Create indices
        rows, cols = np.indices((self.settings.world_size, self.settings.world_size))
        
        # Set ones
        actions_grid[rows, cols, action_map] = 1.0
        
        # 2. Step the environment
        self.env.step(actions_grid)
        
        # 3. Step other species
        self._ensure_target_species_turn()
        
        # 4. Calculate Reward (Global Biomass Change)
        biomass_channel = MODEL_OFFSETS[self.target_species]["biomass"]
        current_total_biomass = np.sum(self.env.env.world[..., biomass_channel])
        
        biomass_change = current_total_biomass - self.last_total_biomass
        self.last_total_biomass = current_total_biomass
        
        # Reward Scaling
        # +0.1 for survival, + change * 0.01
        reward = (biomass_change * 0.01) + 0.1
        
        # 5. Check Termination
        terminated = self.env.terminations[self.target_species]
        truncated = self.env.truncations[self.target_species]
        
        info = {}
        if terminated or truncated:
            info["terminal_observation"] = self._get_obs()
            
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        # Return the inner world (without padding if possible, or just the raw world)
        # The env.world is padded (W+2, W+2). We want the inner (W, W).
        # The petting_zoo env wrapper might handle this?
        # self.env.env.world is the raw numpy array (W+2, W+2, C)
        
        raw_world = self.env.env.world
        # Crop padding
        obs = raw_world[1:-1, 1:-1, :].copy() # Copy to avoid modifying original world if we write to it
        
        # Normalize Biomass and Energy using log1p
        b_start, b_end = MODEL_OFFSETS["biomass_range"]
        obs[..., b_start:b_end+1] = np.log1p(obs[..., b_start:b_end+1])
        
        e_start, e_end = MODEL_OFFSETS["energy_range"]
        obs[..., e_start:e_end+1] = np.log1p(obs[..., e_start:e_end+1])
        
        # Transpose to (C, H, W)
        obs = np.transpose(obs, (2, 0, 1))
        
        return obs.astype(np.float32)

    def set_species_models(self, models):
        self.species_models = models

    def _ensure_target_species_turn(self):
        """
        Steps the environment through other species' turns until it is the target species' turn
        or the episode ends.
        """
        while self.env.agent_selection != self.target_species:
            if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                break
            
            agent = self.env.agent_selection
            
            if agent == "plankton":
                empty_action = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
                self.env.step(empty_action)
            else:
                # Get global obs
                obs = self._get_obs() # (H, W, C)
                
                # Predict
                # Model expects (1, H, W, C) if vectorized, or just (H, W, C)
                # SB3 predict returns (action, state)
                # action will be (H*W,)
                action, _ = self.species_models[agent].predict(obs, deterministic=True)
                
                # Convert to grid
                action_map = action.reshape(self.settings.world_size, self.settings.world_size)
                actions_grid = np.zeros((self.settings.world_size, self.settings.world_size, OUTPUT_SIZE), dtype=np.float32)
                rows, cols = np.indices((self.settings.world_size, self.settings.world_size))
                actions_grid[rows, cols, action_map] = 1.0
                
                self.env.step(actions_grid)
