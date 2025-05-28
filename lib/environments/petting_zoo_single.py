import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
from lib.world import (
    read_map_from_file,
    perform_cell_action,
    world_is_alive,
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
    metadata = {"render_modes": ["human"], "name": "pettingzoo/Ecotwin-single-v0"}

    def __init__(self, render_mode=None, map_folder='maps/baltic'):
        self.map_folder = map_folder
        self.plot_data = {}
        self.possible_agents = list([species for species in const.SPECIES_MAP.keys() if species != "plankton"])
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.reset()

        self._observation_spaces = spaces.Box(low=0, high=1, shape=(3, 3, 11), dtype=np.float32)
        self._action_spaces = spaces.Discrete(6)

        self.render_mode = render_mode

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        world, world_data = read_map_from_file(self.map_folder)
        # Pad the world and world_data arrays by 1 on each side.
        self.world = np.pad(world, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)
        self.world_data = np.pad(world_data, pad_width=((1,1), (1,1), (0,0)), mode="constant", constant_values=0)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.cumulative_rewards = {agent: 0 for agent in self.agents}

    def observe(self, agent):
        terrain = self.world[..., 0:3]
        biomass = []
        smell = []
        for species in list(const.SPECIES_MAP.keys()):
            max_biomass = self.world[..., const.SPECIES_MAP[species]["biomass_offset"]].max()
            max_smell = self.world[..., const.SPECIES_MAP[species]["smell_offset"]].max()
            biomass.append(self.world[..., const.SPECIES_MAP[species]["biomass_offset"]] / (max_biomass + 1e-8))
            smell.append(self.world[..., const.SPECIES_MAP[species]["smell_offset"]] / (max_smell + 1e-8))
        biomass = np.stack(biomass, axis=-1)
        smell = np.stack(smell, axis=-1)
        energy = self.world[..., const.OFFSETS_ENERGY:const.OFFSETS_ENERGY+4] / (const.MAX_ENERGY + 1e-8)
        observation = np.concatenate([terrain, biomass, smell, energy], axis=-1)

        patches = sliding_window_view(observation, (3, 3), axis=(0, 1))

        return patches

        max_biomass = self.world[..., 3:7].max()
        max_smell = self.world[..., 7:11].max()
        # get a random position
        x = random.randint(1, const.WORLD_SIZE)
        y = random.randint(1, const.WORLD_SIZE)

        # get the observation
        observation = self.world[x-1:x+2, y-1:y+2]

        print("Observation:", observation)
        # normalize the observation
        terrain = observation[..., 0:3]
        biomass = observation[..., 3:7] / (max_biomass + 1e-8)
        smell   = observation[..., 7:11] / (max_smell + 1e-8)

        return np.concatenate([terrain, biomass, smell], axis=-1)

    def step(self, action):
        # agent = self.agent_selection
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            obs = self.observations[self.agent_selection]
            perform_cell_action(obs, self.state)

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

            # Update observations for all agents (this line mimics your original logic).
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[self.agent_selection]]] = None
            self._clear_rewards()
        
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        if self.render_mode == "none":
            return
        if self.render_mode == "human":
            plot_biomass(self.plot_data)
            draw_world(self.screen, self.world, self.world_data)
            self.render()

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


        
