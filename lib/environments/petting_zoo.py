import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
from lib.world import update_smell, read_map_from_file, remove_species_from_fishing, respawn_plankton, reset_plankton_cluster, move_plankton_cluster, move_plankton_based_on_current, spawn_plankton, perform_action, world_is_alive, create_map_from_noise
from lib.visualize import init_pygame, plot_generations, draw_world, plot_biomass
import lib.constants as const
import torch
import random

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "pettingzoo/Ecotwin-v0"}

    def __init__(self, render_mode=None, map_folder='maps/baltic'):
        self.possible_agents = list(const.SPECIES_MAP.keys())
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.map_folder = map_folder
        self.reset()

        self.observation_shape = self.world.shape        

        # Fixed batch size:
        # For a 50x50 grid and 5 equally distributed colors, each color yields 2500/5 = 500 positions.
        self.fixed_batch_size = 500
        # Each 3x3 neighborhood with 11 channels flattens to 3*3*11 = 99.
        self.obs_vector_length = 99

        # Define observation space as a Dict with fixed shapes.
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

        # Define action space as a Dict. Each action is a probability vector of length 6.
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

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, species):
        # if species == "plankton":
        #     return {
        #         "observation": torch.tensor([0], dtype=torch.float32),
        #         "positions": []
        #     }

        max_biomass = self.world[..., 3:7].max()
        max_smell = self.world[..., 7:11].max()

        selected_color = random.choice([0, 1, 2, 3, 4])
        selected_set_mask = (self.colors == selected_color)
        selected_positions = selected_set_mask.nonzero(as_tuple=False)
        selected_positions_padded = selected_positions + 1

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
        
        return {
            "observation": batch_tensor,
            "positions": selected_positions_padded
        }
    
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            self.world = perform_action(
                self.world,
                self.world_data,
                self.state[agent]["action"],
                agent,
                self.state[agent]["positions"]
            )
            self.rewards = {agent: 1 for agent in self.agents}

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= const.MAX_STEPS for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
    
    def render(self):
        if self.render_mode == "none":
            return
        if self.render_mode == "human":
            draw_world(self.screen, self.world, self.world_data)
