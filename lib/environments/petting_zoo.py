import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
from lib.world import (
    update_smell,
    read_map_from_file,
    remove_species_from_fishing,
    respawn_plankton,
    reset_plankton_cluster,
    move_plankton_cluster,
    move_plankton_based_on_current,
    spawn_plankton,
    perform_action,  # Make sure this is the NumPy version from the previous rewrite.
    world_is_alive,
    create_map_from_noise
)
from lib.visualize import init_pygame, plot_generations, draw_world, plot_biomass
import lib.constants as const
import random

def env(render_mode=None):
    """
    Wraps the raw environment in PettingZoo wrappers.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env_instance = raw_env(render_mode=internal_render_mode)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance

class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "pettingzoo/Ecotwin-v0"}

    def __init__(self, render_mode=None, map_folder='maps/baltic'):
        # Define agents based on species keys.
        self.possible_agents = list(const.SPECIES_MAP.keys())
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.map_folder = map_folder
        self.reset()

        # Store the shape of the world (NumPy array) for debugging.
        self.observation_shape = self.world.shape        

        # Fixed batch size:
        # For a 50x50 grid and 5 equally distributed colors, each color yields 2500/5 = 500 positions.
        self.fixed_batch_size = 500
        # Each 3x3 neighborhood with 11 channels flattens to 3*3*11 = 99.
        self.obs_vector_length = 99

        # Define observation spaces for each agent.
        self._observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.fixed_batch_size, self.obs_vector_length),
                    dtype=np.float32
                ),
                "positions": spaces.Box(
                    low=0,
                    high=const.WORLD_SIZE + 2,
                    shape=(self.fixed_batch_size, 2),
                    dtype=np.int32
                )
            })
            for agent in self.possible_agents
        }

        # Define action spaces for each agent.
        self._action_spaces = {
            agent: spaces.Dict({
                "action": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.fixed_batch_size, 6),
                    dtype=np.float32
                ),
                "positions": spaces.Box(
                    low=0,
                    high=const.WORLD_SIZE + 2,
                    shape=(self.fixed_batch_size, 2),
                    dtype=np.int32
                )
            })
            for agent in self.possible_agents
        }

        self.render_mode = render_mode
        if render_mode == "human":
            self.screen = init_pygame()

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Use the NumPy version of your map loader.
        world, world_data = read_map_from_file(self.map_folder)
        # Pad the world and world_data arrays by 1 on each side.
        self.world = np.pad(world, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)
        self.world_data = np.pad(world_data, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)

        # Create a grid of coordinates (shape: (WORLD_SIZE, WORLD_SIZE)).
        grid_x, grid_y = np.meshgrid(np.arange(const.WORLD_SIZE), np.arange(const.WORLD_SIZE), indexing='ij')
        self.colors = (grid_x + 2 * grid_y) % 5
        
        self.step_count = 0
        self.done = False

        # Initialize bookkeeping dictionaries.
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        # Initialize the agent selector.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, species):
        # Calculate maximum biomass and smell for normalization.
        max_biomass = self.world[..., 3:7].max()
        max_smell = self.world[..., 7:11].max()

        selected_color = random.choice([0, 1, 2, 3, 4])
        selected_set_mask = (self.colors == selected_color)  # Boolean array.
        selected_positions = np.argwhere(selected_set_mask)     # Shape: (n, 2)
        selected_positions_padded = selected_positions + 1       # Adjust for padding.
        selected_positions_padded = selected_positions_padded.astype(np.int32)

        # Define a 3x3 offsets array.
        offsets = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1], [ 0, 0], [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ])
        # Add offsets to each selected position (using broadcasting).
        neighbor_positions = selected_positions_padded[:, None, :] + offsets[None, :, :]
        # Clip neighbor positions so they remain within the padded world.
        neighbor_positions[:, :, 0] = np.clip(neighbor_positions[:, :, 0], 0, const.WORLD_SIZE + 1)
        neighbor_positions[:, :, 1] = np.clip(neighbor_positions[:, :, 1], 0, const.WORLD_SIZE + 1)
        # Reshape to (n*9, 2).
        neighbor_positions = neighbor_positions.reshape(-1, 2)
        
        # Gather neighbor values from the world.
        neighbor_values = self.world[neighbor_positions[:, 0], neighbor_positions[:, 1]]
        n = selected_positions_padded.shape[0]
        neighbor_values = neighbor_values.reshape(n, 9, const.TOTAL_TENSOR_VALUES)

        terrain = neighbor_values[..., 0:3]
        biomass = neighbor_values[..., 3:7] / (max_biomass + 1e-8)
        smell   = neighbor_values[..., 7:11] / (max_smell + 1e-8)
        # Concatenate the channels: shape becomes (n, 9, 11).
        batch_tensor = np.concatenate([terrain, biomass, smell], axis=-1)
        # Flatten the 3x3 grid to a vector: (n, 99).
        batch_tensor = batch_tensor.reshape(n, -1)
        
        return {"observation": batch_tensor, "positions": selected_positions_padded}
    
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            # Update the world using the NumPy version of perform_action.
            self.world = perform_action(
                self.world,
                self.world_data,
                self.state[agent]["action"],
                agent,
                self.state[agent]["positions"]
            )
            self.rewards = {agent: 1 for agent in self.agents}
            self.num_moves += 1
            self.truncations = {agent: self.num_moves >= const.MAX_STEPS for agent in self.agents}

            # Update observations for all agents (this line mimics your original logic).
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
    
    def render(self):
        if self.render_mode == "none":
            return
        if self.render_mode == "human":
            draw_world(self.screen, self.world, self.world_data)

    # Note: You must implement or override the helper methods below
    def _was_dead_step(self, action):
        # Placeholder: implement behavior for dead agents.
        pass

    def _clear_rewards(self):
        # Placeholder: clear rewards when not all agents have acted.
        pass

    def _accumulate_rewards(self):
        # Placeholder: accumulate rewards into _cumulative_rewards.
        pass
