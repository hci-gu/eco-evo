import gymnasium as gym
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import lib.constants as const
from lib.world import update_smell, total_biomass, read_map_from_file, remove_species_from_fishing, respawn_plankton, reset_plankton_cluster, move_plankton_cluster, move_plankton_based_on_current, spawn_plankton, perform_action, world_is_alive, create_map_from_noise
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
        # all species except plankton
        self.possible_species = [species for species in const.SPECIES_MAP.keys() if species != "plankton"]

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(const.WORLD_SIZE, const.WORLD_SIZE, const.TOTAL_TENSOR_VALUES, 3, 3),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(const.WORLD_SIZE, const.WORLD_SIZE, const.AVAILABLE_ACTIONS * len(self.possible_species)),
            dtype=np.float32
        )

        if self.render_mode == "human":
            self.screen = init_pygame()

    def reset(self, seed=None, options=None):
        world, world_data = read_map_from_file(self.map_folder)
        self.world = np.pad(world, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)
        self.world_data = np.pad(world_data, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)

        grid_x, grid_y = np.meshgrid(np.arange(const.WORLD_SIZE), np.arange(const.WORLD_SIZE), indexing='ij')
        self.colors = (grid_x % 3) + 3 * (grid_y % 3)
        
        self.step_count = 0
        self.done = False
        
        # Return the initial observation. Depending on your design, you may return the entire grid
        # or a specific subset related to the species the agent controls.
        return self.get_obs(), {}
    
    def get_obs(self):
        max_biomass = self.world[..., 3:7].max()
        max_smell = self.world[..., 7:11].max()

        terrain = self.world[..., 0:3]
        biomass = self.world[..., 3:7] / (max_biomass + 1e-8)
        smell   = self.world[..., 7:11] / (max_smell + 1e-8)
        
        observation = np.concatenate([terrain, biomass, smell], axis=-1)

        patches = sliding_window_view(observation, (3, 3), axis=(0, 1))

        return patches
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        if (self.step_count % len(self.possible_species) == 0):
            spawn_plankton(self.world, self.world_data)
        
        color_order = list(np.arange(9))
        while len(color_order):
            # take random species from possible species
            species = random.choice(self.possible_species)

            selected_color = color_order.pop()
            selected_set_mask = (self.colors == selected_color)
            selected_positions = np.argwhere(selected_set_mask)
            selected_positions_padded = selected_positions + 1
            selected_positions_padded = selected_positions_padded.astype(np.int32)

            # slice action based on species
            action_slice = action
            if species == "cod":
                action_slice = action[..., :6]
            elif species == "herring":
                action_slice = action[..., 6:12]
            else:
                action_slice = action[..., 12:]
            
            action_part = action_slice[selected_set_mask]
            self.world = perform_action(
                self.world,
                self.world_data,
                action_part,
                species,
                selected_positions_padded
            )

        update_smell(self.world)
        self.render()
        self.step_count += 1
        
        if self.step_count >= const.MAX_STEPS or world_is_alive(self.world) == False:
            self.done = True
        
        info = {"step_count": self.step_count}
        return self.get_obs(), total_biomass(self.world) * self.step_count, self.done, False, info
    
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

    def compute_species_fitness(world, species):
        """
        Computes the fitness for a given species as the total biomass 
        in the world for that species.
        """
        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        return np.sum(world[:, :, biomass_offset])

gym.register(
    id="gymnasium_env/Ecotwin-v0",
    entry_point=EcosystemEnv,
)
