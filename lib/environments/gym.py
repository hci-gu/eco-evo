import gymnasium as gym
import numpy as np
import lib.constants as const
from lib.world import update_smell, read_map_from_file, remove_species_from_fishing, respawn_plankton, reset_plankton_cluster, move_plankton_cluster, move_plankton_based_on_current, spawn_plankton, perform_action, world_is_alive, create_map_from_noise
from lib.visualize import init_pygame, plot_generations, draw_world, plot_biomass
from gymnasium import spaces
import pygame
import torch
import random

class EcosystemEnv(gym.Env):
    metadata = {"render_modes": ["none","human","rgb_array"]}
    
    def __init__(self, render_mode="none", map_folder='maps/baltic'):
        super().__init__()
        self.render_mode = render_mode
        self.map_folder = map_folder
        
        self.reset()

        self.observation_shape = self.world.shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        if self.render_mode == "human":
            self.screen = init_pygame()

    
    def reset(self, seed=None, options=None):
        world, world_data = read_map_from_file(self.map_folder)
        self.world = torch.nn.functional.pad(world, (0, 0, 1, 1, 1, 1), "constant", 0)
        self.world_data = torch.nn.functional.pad(world_data, (0, 0, 1, 1, 1, 1), "constant", 0)

        grid_x, grid_y = torch.meshgrid(
            torch.arange(const.WORLD_SIZE),
            torch.arange(const.WORLD_SIZE),
            indexing='ij'
        )
        self.colors = (grid_x + 2 * grid_y) % 5
        
        self.step_count = 0
        self.done = False
        
        # Return the initial observation. Depending on your design, you may return the entire grid
        # or a specific subset related to the species the agent controls.
        return self._get_observation(), {}
    
    def get_obs(self, species):
        return self._get_observation(species)
    
    def _get_observation(self, species="plankton"):
        if species == "plankton":
            return torch.tensor([0], dtype=torch.float32)

        max_biomass = self.world[..., 3:7].max()
        max_smell = self.world[..., 7:11].max()

        selected_color = random.choice([0, 1, 2, 3, 4])
        selected_set_mask = (self.colors == selected_color)
        selected_positions = selected_set_mask.nonzero(as_tuple=False)
        selected_positions_padded = selected_positions + 1  # adjust if padding

        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        biomass_values = self.world[selected_positions_padded[:, 0], selected_positions_padded[:, 1], biomass_offset]
        non_zero_biomass_mask = biomass_values > 0
        selected_positions_padded = selected_positions_padded[non_zero_biomass_mask]

        if selected_positions_padded.size(0) == 0:
            empty_obs = torch.empty((0, 9 * (3 + 4 + 4)))
            return empty_obs, selected_positions_padded

        # Step 4: Extract neighborhoods for selected cells
        offsets = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1], [ 0, 0], [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ])
        neighbor_positions = (selected_positions_padded.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 2)
        neighbor_positions[:, 0].clamp_(0, const.WORLD_SIZE + 1)
        neighbor_positions[:, 1].clamp_(0, const.WORLD_SIZE + 1)

        neighbor_values = self.world[neighbor_positions[:, 0], neighbor_positions[:, 1]]
        neighbor_values = neighbor_values.view(selected_positions_padded.size(0), 9, const.TOTAL_TENSOR_VALUES)

        terrain = neighbor_values[..., 0:3]
        biomass = neighbor_values[..., 3:7] / (max_biomass + 1e-8)
        smell   = neighbor_values[..., 7:11] / (max_smell + 1e-8)
        batch_tensor = torch.cat([terrain, biomass, smell], dim=-1).view(selected_positions_padded.size(0), -1)
        
        return batch_tensor, selected_positions_padded
    
    def step(self, dict):
        species, action_batch, positions_tensor = dict["species"], dict["action"], dict["positions"]
        update_smell(self.world)
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        if species == "plankton":
            spawn_plankton(self.world, self.world_data)
            return self._get_observation(species), 0, False, False, {}
        
        self.world = perform_action(self.world, self.world_data, action_batch, species, positions_tensor)
        
        if self.step_count >= const.MAX_STEPS or world_is_alive(self.world) == False:
            self.done = True

        
        observation = self._get_observation(species)
        
        # Gymnasium expects info dict, can include additional diagnostic information.
        info = {"step_count": self.step_count}
        return observation, 1, self.done, False, info  # (obs, reward, terminated, truncated, info)
    
    def render(self):
        if self.render_mode == "none":
            return
        
        if self.render_mode == "human":
            draw_world(self.screen, self.world, self.world_data)
    
    def close(self):
        # Clean up resources if any.
        if self.render_mode == "human":
            pygame.quit()
        pass

gym.register(
    id="gymnasium_env/Ecotwin-v0",
    entry_point=EcosystemEnv,
)

