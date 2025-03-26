import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
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
    get_movement_delta,
    apply_movement_delta,
    spawn_plankton,
    perform_eating,
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
        self.plot_data = {}
        self.possible_agents = list(const.SPECIES_MAP.keys())
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.map_folder = map_folder
        self.reset()

        self.obs_vector_length = 11

        self._observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=1,
                shape=(const.WORLD_SIZE, const.WORLD_SIZE, const.TOTAL_TENSOR_VALUES, 3, 3),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Box(
                low=0,
                high=1,
                shape=(const.WORLD_SIZE, const.WORLD_SIZE, const.AVAILABLE_ACTIONS),
                dtype=np.float32
            )
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
        world, world_data, starting_biomasses = read_map_from_file(self.map_folder)
        # Pad the world and world_data arrays by 1 on each side.
        self.world = np.pad(world, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)
        self.world_data = np.pad(world_data, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)
        self.starting_biomasses = starting_biomasses

        # Create a grid of coordinates (shape: (WORLD_SIZE, WORLD_SIZE)).
        grid_x, grid_y = np.meshgrid(np.arange(const.WORLD_SIZE), np.arange(const.WORLD_SIZE), indexing='ij')
        # Using modulo 3 in both dimensions to create 9 distinct classes.
        self.colors = (grid_x % 3) + 3 * (grid_y % 3)
        
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
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0

        # Initialize the agent selector.
        random.shuffle(self.agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.color_order = []

    def observe(self, agent):
        # Calculate maximum biomass and smell for normalization.
        max_biomass = self.world[..., 3:7].max()
        max_smell = self.world[..., 7:11].max()

        terrain = self.world[..., 0:3]
        biomass = self.world[..., 3:7] / (max_biomass + 1e-8)
        smell   = self.world[..., 7:11] / (max_smell + 1e-8)
        
        observation = np.concatenate([terrain, biomass, smell], axis=-1)

        patches = sliding_window_view(observation, (3, 3), axis=(0, 1))

        return patches
    
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return self.observations, self.rewards, self.terminations, self.truncations, self.infos

        agent = self.agent_selection
        self.state[self.agent_selection] = action

        # If the active agent is "plankton", use the hardcoded logic.
        if agent == "plankton":
            spawn_plankton(self.world, self.world_data)
        else:
            self.state[agent] = action
            # Update the world using the NumPy version of perform_action.
            total_movement_deltas = np.zeros_like(self.world)
            # total_movement_deltas = np.zeros((const.WORLD_SIZE, const.WORLD_SIZE, 1), dtype=np.float32)
            color_order = list(np.arange(9))
            while len(color_order):
                selected_color = color_order.pop()
                selected_set_mask = (self.colors == selected_color)
                selected_positions = np.argwhere(selected_set_mask)
                selected_positions_padded = selected_positions + 1
                selected_positions_padded = selected_positions_padded.astype(np.int32)

                action_part = action[selected_set_mask]
                movement_deltas = get_movement_delta(self.world, self.world_data, agent, action_part, selected_positions_padded)
                total_movement_deltas += movement_deltas
            
            apply_movement_delta(self.world, agent, total_movement_deltas)
            
            color_order = list(np.arange(9))
            while len(color_order):
                selected_color = color_order.pop()
                selected_set_mask = (self.colors == selected_color)
                selected_positions = np.argwhere(selected_set_mask)
                selected_positions_padded = selected_positions + 1
                selected_positions_padded = selected_positions_padded.astype(np.int32)

                action_part = action[selected_set_mask]
                perform_eating(self.world, agent, action_part, selected_positions_padded)

            update_smell(self.world)
            self.render()
            if not world_is_alive(self.world):
                # Abort simulation by terminating all agents.
                for ag in self.agents:
                    self.terminations[ag] = True
                # Optionally, you can also set rewards to 0 (or a penalty) here.
                self.rewards = {ag: 0 for ag in self.agents}
            else:
                # If the world is still alive, assign +1 reward to all surviving agents.
                for ag in self.agents:
                    if not self.terminations[ag] and not self.truncations[ag]:
                        self.rewards[ag] = 1
                    else:
                        self.rewards[ag] = 0

            # Accumulate rewards.
            for ag in self.agents:
                self.cumulative_rewards[ag] += self.rewards[ag]
            # self.rewards = {sp: 1 for sp in self.agents}
            self.num_moves += 1
            self.truncations = {agent: self.num_moves >= const.MAX_STEPS for agent in self.agents}

            for i in self.agents:
                self.observations[i] = self.observe(i)

        update_smell(self.world)

        if self._agent_selector.is_last():
            # Re-shuffle agents for the next round.
            random.shuffle(self.agents)
            self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self._accumulate_rewards()
    
    def render(self):
        if self.render_mode == "none":
            return
        if self.render_mode == "human":
            # print("Rendering...")
            # print(self.plot_data)
            plot_biomass(self.plot_data)
            draw_world(self.screen, self.world, self.world_data)

    def get_fitness(self, agent):
        biomass = self.world[..., const.SPECIES_MAP[agent]["biomass_offset"]]
        # divide by starting biomass to get a percentage
        biomass_growth = biomass.sum() / self.starting_biomasses[agent]

        return np.log(1 + biomass_growth)
    
    # Note: You must implement or override the helper methods below
    def _was_dead_step(self, action):
        if all(self.terminations[ag] or self.truncations[ag] for ag in self.agents):
            self.agents = []
            # Signal that the episode is over by setting agent_selection to None.
            # self.agent_selection = None
        else:
            # Otherwise, skip this agent and move to the next.
            self.agent_selection = self._agent_selector.next()

    def _clear_rewards(self):
        pass

    def _accumulate_rewards(self):
        pass