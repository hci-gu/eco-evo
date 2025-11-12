from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pbm_gym import PbmEnv


import gymnasium as gym
import numpy as np


class ChannelFlip(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        grid_shape = env.observation_space.shape[:2]
        n_channels = np.prod(env.observation_space.shape[2:])
        new_shape = (n_channels, *grid_shape)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape)

    def observation(self, observation):
        return observation.reshape(*observation.shape[:2], -1).transpose(2, 0, 1)


class PzObservation(gym.ObservationWrapper):
    """Transform the observation space to be suitable for FG based agents.
    If the observation space of the initial environment is (10, 10, 4, 3, 3) and the
    number of FGs is 3, the new observation space will be (10, 10, 3, 4 * 3 * 3) where
    the last dimension is the flattened 3x3 grid of the 3 FGs. If energy is included in
    the observation, meaning that initial observation space is (10, 10, 8, 3, 3) the
    last dimension will be 4 * 3 * 3 + 1 = 37, since the observation will only contain
    the energy of the FG in the center cell."""

    def __init__(self, env: PbmEnv):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        grid_shape = env.observation_space.shape[:2]
        self.use_energy = env.energy_in_obs
        self.n_fg = env.n_fg
        # Observation per cell is all fg in the surrounding 3x3 + energy of the fg + the
        # number of the current functional group.
        n_inputs_cell = (
            np.prod(env.observation_space.shape[-2:]) * self.n_fg + self.use_energy + 1
        )
        new_shape = (*grid_shape, self.n_fg, n_inputs_cell)
        low = env.observation_space.low.min()
        high = env.observation_space.high.min()
        dtype = env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=new_shape, dtype=dtype
        )

    def observation(self, observation):
        assert isinstance(self.observation_space, gym.spaces.Box)
        new_obs = np.zeros(
            shape=self.observation_space.shape, dtype=self.observation_space.dtype
        )
        for i in range(self.n_fg):
            new_obs[:, :, i, :-2] = observation[:, :, : self.n_fg].reshape(
                *observation.shape[:2], -1
            )
        if self.use_energy:
            new_obs[:, :, :, -2] = observation[:, :, self.n_fg :, 1, 1]
        
        # Set the FG id.
        for i in range(self.n_fg):
            new_obs[:,:,i,-1] = i

        # Check the incoming observation.
        _e = self.env._env.energy_table
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                for fg in range(self.n_fg):
                    assert np.allclose(
                        observation[i, j, self.n_fg + fg, 1, 1], _e[fg, i + 1, j + 1]
                    )
        # Check the new observation:
        for i in range(new_obs.shape[0]):
            for j in range(new_obs.shape[1]):
                for fg in range(new_obs.shape[2]):
                    assert np.allclose(
                        new_obs[i, j, fg, :-2], observation[i, j, : self.n_fg].flatten()
                    )
                    assert np.allclose(
                        new_obs[i, j, fg, -2],
                        _e[fg, i + 1, j + 1],
                    ), f"Position {(i,j)}, {new_obs[i, j, fg, -1] - _e[fg, i + 1, j + 1] }"

        return new_obs


class FlattenAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.action_space.shape
        # We flatten the last two dimensions that are the actual actions.
        new_low = env.action_space.low.reshape(*old_shape[:2], -1).copy()
        new_shape = new_low.shape
        new_high = env.action_space.high.reshape(new_shape).copy()
        self.action_space = gym.spaces.Box(new_low, new_high, shape=new_shape)

    def action(self, action):
        return action.reshape(self.env.action_space.shape)


class DropPrimaryProducerAction(gym.ActionWrapper):
    """Drop the action from the primary producer since it does not have any active
    actions anyway."""

    def __init__(self, env, pp_index=0):
        super().__init__(env)
        new_low = np.delete(env.action_space.low, pp_index, axis=2)
        new_high = np.delete(env.action_space.high, pp_index, axis=2)
        self.action_space = gym.spaces.Box(new_low, new_high, shape=new_low.shape)
        self.pp_index = pp_index

    def action(self, action):
        return np.insert(action, self.pp_index, 0, axis=2)
