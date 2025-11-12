from __future__ import annotations

import pickle
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

import lib.environments.bioMARL as brl
import lib.environments.bioMARL.default as default
from lib.environments.bioMARL.numba_fha import numba_get_observation
from lib.environments.bioMARL.wrappers import ChannelFlip, DropPrimaryProducerAction, FlattenAction

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


def env_factory(**kwargs):
    env = ChannelFlip(PbmEnv(**kwargs))
    env = FlattenAction(DropPrimaryProducerAction(env, pp_index=0))
    # env = Monitor(env)
    return env


def convertable_observation_space(shape1, shape2):
    """
    Check if the space1 is convertable to space2. Three values are possible:
    1. The spaces are equal return None.
    2. The spaces have both three dimensions and the first dimension is equal, return True.
    3. Return False.
    """
    if shape1 == shape2:
        return None
    elif len(shape1) == 3 and len(shape2) == 3:
        if shape1[0] == shape2[0]:
            return True
    return False


def load_vecenv(file_name, venv):
    """Load the VecEnv from file. We we can't use the load method since the grid shape"
    might be different."""

    with open(file_name, "rb") as file_handler:
        vec_normalize = pickle.load(file_handler)

    convertable = convertable_observation_space(
        venv.observation_space.shape, vec_normalize.observation_space.shape
    )
    if convertable is not None:
        if convertable:
            # vec_normalize.observation_space = venv.observation_space
            # vec_normalize.action_space = venv.action_space
            new_vec_normalize = VecNormalize(
                venv,
                # norm_obs=vec_normalize.norm_obs,
                # norm_reward=vec_normalize.norm_reward,
                # clip_obs=vec_normalize.clip_obs,
                # clip_reward=vec_normalize.clip_reward,
                # gamma=vec_normalize.gamma,
                # epsilon=vec_normalize.epsilon,
                # use_sde=vec_normalize.use_sde,
                # use_sde_at_warmup=vec_normalize.use_sde_at_warmup,
            )
            new_vec_normalize.obs_rms.mean[:] = vec_normalize.obs_rms.mean.mean(
                axis=(-2, -1)
            )[..., None, None]
            new_vec_normalize.obs_rms.var[:] = vec_normalize.obs_rms.var.mean(
                axis=(-2, -1)
            )[..., None, None]
            # We ignore updating the reward normalization since we don't use it.
            vec_normalize = new_vec_normalize
        else:
            raise ValueError(
                f"Model and environment have different number of channels: "
                f"{vec_normalize.observation_space.shape} vs {venv.observation_space.shape}"
            )
    else:
        vec_normalize.set_venv(venv)

    return vec_normalize


class PbmEnv(gym.Env):
    MAX_STEPS = 254

    # def __init__(self, grid_shape=(10, 10), seed=None) -> None:
    def __init__(self, **kwargs):
        # Complete with default values.
        kwargs = default.DEFAULT | kwargs

        super().__init__()
        self.n_fg = kwargs["n_fg"]
        self.fg_init_k = np.array(kwargs["fg_init_k"], dtype=float)
        self.padding = kwargs["padding"]
        self.grid_shape = tuple(kwargs["grid_shape"])
        self.seed = kwargs["seed"]
        self.rng = np.random.default_rng(self.seed)
        self.action_sigma_max = kwargs["action_sigma_max"]
        self.energy_in_obs = kwargs["energy_in_obs"]
        self.fg_k = np.array(kwargs["fg_k"], dtype=float)
        self.reproduction_freq = np.array(kwargs["reproduction_freq"])
        self.growth_rates = np.array(kwargs["growth_rates"], dtype=float)
        self.feeding_energy_reward = np.array(
            kwargs["feeding_energy_reward"], dtype=float
        )
        self.base_energy_cost = np.array(kwargs["base_energy_cost"], dtype=float)
        self.idling_energy_cost = np.array(kwargs["idling_energy_cost"], dtype=float)
        self.hiding_energy_cost = np.array(kwargs["hiding_energy_cost"], dtype=float)
        self.migration_energy_cost = np.array(
            kwargs["migration_energy_cost"], dtype=float
        )
        self.seed_bank = self._initialize_seed_bank()
        # Used to stop FGs from eating other FG which they wont get energy from.
        self.feeding_mask = (self.feeding_energy_reward > 0).astype(int)
        self.mask = None if kwargs["mask"] is None else kwargs["mask"].copy()
        self._h, self.carrying_capacity = self._initialize_habitat_and_cc()

        self._env = brl.LargeScaleRLEnv(
            grid_shape=self.grid_shape,
            functional_groups=self.n_fg,
            h=self._h,
            padding=self.padding,
            migrate_fg=self._initialize_migrate_fg(
                baselines=[0.9, 1, 0.3, 0.03],
                energy_cost=self.migration_energy_cost,
            ),
            feed_fg=self._initialize_feed_fg(),
            reproduce_fg=self._initialize_reproduce_fg(),
            low_energy_mortality=self._initialize_low_energy_mortality(),
            carrying_capacity=self.carrying_capacity,
            seed=self._get_seed(),
            mask=self.mask,
        )

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(*self.grid_shape, self.n_fg * (1 + self.energy_in_obs), 3, 3),
        )
        # The actual actions are: 3 move actions (x, y, sigma), hide, and self.n_fg eat
        # actions. hide + the eat actions should sum to one.
        shape = (*self.grid_shape, self.n_fg, 3 + self.n_fg + 1)
        lows = np.zeros(shape=shape)
        lows[..., :2] = -1  # The move actions go from -1 to 1.
        highs = np.ones(shape=shape)  # The sigma can be higher but we transform it in
        # the `step` function.
        self.action_space = gym.spaces.Box(lows, highs, shape=shape)

        self.n_steps = 0
        # Used in reward computation
        self.max_pops = self._env.total_pop

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert action.max() <= 1.0
        migrate_action = action[..., :3].copy()
        # Perhaps other transformations are better.
        migrate_action[..., 2] *= self.action_sigma_max
        hide_feed_action = action[..., 3:].copy()
        hide_feed_action[..., 1:] = hide_feed_action[..., 1:] * self.feeding_mask
        hide_feed_action = self.normalize_probs(hide_feed_action)
        # Zero out the actions for the primary producer.
        hide_feed_action[..., 0, :] = 0.0
        # Flatten the grid dimensions.
        migrate_action = migrate_action.reshape(-1, *migrate_action.shape[-2:])
        hide_feed_action = hide_feed_action.reshape(-1, *hide_feed_action.shape[-2:])
        new_action = {"migrate": migrate_action, "hide_n_feed": hide_feed_action}
        self._env.step(new_action)
        reward = self.get_reward()
        # terminated = self._env.total_pop.sum() == 0
        # Try: terminate if any species go extinct.
        terminated = (self._env.total_pop == 0).any()
        # obs = self.get_observation()
        obs = numba_get_observation(
            self._env.h,
            self._env.energy_table,
            self.padding,
            self.observation_space.shape,
            self.n_fg,
            self.energy_in_obs,
        )
        # assert np.allclose(
        #     self.get_observation(),
        #     numba_get_observation(
        #         self._env.h,
        #         self._env.energy_table,
        #         self.padding,
        #         self.observation_space.shape,
        #         self.n_fg,
        #         self.energy_in_obs,
        #     ),
        # )

        self.n_steps += 1
        if False:
            if self.n_steps and self.n_steps % 100 == 0:
                print(
                    f"System: {self.n_steps} steps, FGs alive: {np.nonzero(self._env.total_pop)[0].tolist()}"
                )
            if self.n_steps > self.MAX_STEPS:
                print(f"Done @{self.n_steps}!")
        info = self.generate_info()
        return obs, reward, terminated, self.n_steps > self.MAX_STEPS, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.n_steps = 0

        self._h, self.carrying_capacity = self._initialize_habitat_and_cc()
        self._env = brl.LargeScaleRLEnv(
            grid_shape=self.grid_shape,
            functional_groups=self.n_fg,
            h=self._h,
            padding=self.padding,
            migrate_fg=self._initialize_migrate_fg(
                baselines=[0.9, 1, 0.3, 0.03], energy_cost=[0, 1, 1, 3]
            ),
            feed_fg=self._initialize_feed_fg(),
            reproduce_fg=self._initialize_reproduce_fg(),
            low_energy_mortality=self._initialize_low_energy_mortality(),
            carrying_capacity=self.carrying_capacity,
            seed=self._get_seed(),
        )

        return self.get_observation(), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def close(self):
        """After the user has finished using the environment, close contains the code
        necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise
        an error.
        """
        pass

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

        Returns:
            A string identifying the environment
        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def _initialize_migrate_fg(self, baselines, energy_cost=10):
        try:
            migrate_fg = [
                brl.MigrationAction(baseline_s=baseline, migration_energy_cost=e_c)
                for baseline, e_c in zip(baselines, energy_cost)
            ]
        except TypeError:
            migrate_fg = [
                brl.MigrationAction(
                    baseline_s=baseline, migration_energy_cost=energy_cost
                )
                for baseline in baselines
            ]
        return migrate_fg

    def _initialize_habitat_and_cc(self):
        seed = self._get_seed()
        h = brl.rnd_init_h(
            self.grid_shape,
            n_groups=self.n_fg,
            gaussian=True,
            single_cell=False,
            min_threshold=0,
            fg_init_k=self.fg_init_k,
            sig=0.15,
            seed=seed,
        )
        carrying_capacity = brl.rnd_init_h(
            self.grid_shape,
            n_groups=self.n_fg,
            gaussian=True,
            single_cell=False,
            min_threshold=0,
            fg_init_k=np.ones(self.n_fg),
            round_to_int=False,
            sig=0.2,
            seed=seed,
        )
        # We only use the first row but needed to compute all for the same
        # seed arrangement to work.
        carrying_capacity[1:] = 1
        # Initialize the primary producer at carrying capacity.
        h[0] = np.round(carrying_capacity[0] * self.fg_init_k[0])

        # Scale with the maximal capacity.
        carrying_capacity = np.einsum("sxy, s -> sxy", carrying_capacity, self.fg_k)
        # Remove the population on land cells and set the carrying capacity to 0.
        if self.mask is not None:
            for _h, _c in zip(h, carrying_capacity):
                _h[self.mask] = 0
                _c[self.mask] = 0

        return h, carrying_capacity

    def _initialize_feed_fg(self):
        return brl.FeedingAction(
            functional_groups=self.n_fg,
            feeding_energy_reward=self.feeding_energy_reward,
            base_energy_cost=self.base_energy_cost,
            hiding_energy_cost=self.hiding_energy_cost,
            idling_energy_cost=self.idling_energy_cost,
            seed=self._get_seed(),
            verbose=0,
        )

    def _initialize_reproduce_fg(self):
        return brl.ReproduceAction(
            functional_groups=self.n_fg,
            fg_reproduction_freq=self.reproduction_freq,
            fg_growth_rates=self.growth_rates,
            fg_carrying_capacity=self.fg_k,
            seed_bank=self.seed_bank,
        )

    def _initialize_low_energy_mortality(self):
        baseline_mortality = np.zeros(self.n_fg)
        grid_shape = (
            self.grid_shape[0] + self.padding * 2,
            self.grid_shape[1] + self.padding * 2,
        )
        return brl.LowEnergyMortality(
            functional_groups=self.n_fg,
            fg_baseline_mortality=baseline_mortality,
            energy_mid_point=None,  # array of shape (n. FGs)
            mortality_logistic_growth=None,  # array of shape (n. FGs)
            grid_shape=grid_shape,
        )

    def _initialize_seed_bank(self):
        padding = self.padding
        shape = (
            self.n_fg,
            self.grid_shape[0] + padding * 2,
            self.grid_shape[1] + padding * 2,
        )
        seed_bank = np.zeros(shape, dtype=int)
        seed_bank[0, padding:-padding, padding:-padding] = 1

        return seed_bank

    def _get_seed(self):
        return self.rng.bit_generator.random_raw()

    def get_reward(self):
        """The reward is based on the number of alive FGs. To ease the training a
        smaller part of the reward is dependant on the the population sizes."""

        pops = self._env.total_pop
        self.max_pops = np.maximum(self.max_pops, pops)
        # We drop the primary producer which we assume to be the first functional group.
        pops = pops[1:]

        # return sum(pops > 0) + (pops / self.max_pops[1:] / self.n_fg).sum()
        # return (pops / self.max_pops[1:]).sum() / len(pops)
        return np.log(pops).sum() if (pops > 0).all() else -1
        # step_reward = 1 / self.MAX_STEPS
        # return step_reward if (pops > 0).all() else -1  # heart beat aka stayin' alive.
        # return 0 if (pops >0).all() else self.n_steps / self.MAX_STEPS

    def get_observation(self):
        obs = np.zeros(self.observation_space.shape)
        pad = self.padding
        h = self._env.h
        energy = self._env.energy_table
        for i in range(0, self.grid_shape[0]):
            for j in range(0, self.grid_shape[1]):
                obs[i, j, : self.n_fg] = h[:, i : i + 2 * pad + 1, j : j + 2 * pad + 1]
                if self.energy_in_obs:
                    # obs[i, j, self.n_fg:] = energy[:,i,j][...,None,None]
                    obs[i, j, self.n_fg :] = energy[:, i + pad, j + pad].reshape(
                        -1, 1, 1
                    )
        return obs

    def normalize_probs(self, ps):
        """`ps` is as matrix where each row is all zeros or all elements are >0."""
        p_sums = ps.sum(axis=-1)
        p_sums[p_sums == 0.0] = 1.0  # Any non-zero number would do.
        return ps / p_sums[..., None]

    def generate_info(self):
        info = {}
        # More interesting to report the average energy, population, and coverage.
        n_cells = self.grid_shape[0] * self.grid_shape[1]

        # Energy
        energy = self._env.avg_energy
        for i, e in enumerate(energy):
            info[f"info--energy/FG_{i}"] = e / n_cells
        population = self._env.total_pop
        for i, e in enumerate(population):
            info[f"info--population/FG_{i}"] = e / n_cells
        distribution = self._env.geo_range
        for i, e in enumerate(distribution):
            info[f"info--distribution/FG_{i}"] = e / n_cells

        # Are we done?
        total_pop = self._env.total_pop
        if (total_pop == 0).any():
            # Adjust if two or more are 0, they share the blame.
            sum = (total_pop == 0).sum()
            for i, tot in enumerate(total_pop):
                info[f"info--CoD/FG{i}"] = (tot == 0) / sum
            info["info--CoD/MaxTime reached"] = 0
        elif self.n_steps > self.MAX_STEPS:
            info["info--CoD/MaxTime reached"] = 1
        return info

    @property
    def population(self):
        return self._env.total_pop

    @property
    def habitat(self):
        return self._env.h

    @property
    def energy(self):
        return self._env.energy_table

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)


def action_from_runner_py(grid_shape):
    primary_vec = np.array([0, 0, 5, 0, 0, 0, 0, 0.0])

    herbivore1_vec = np.array([0, -0.2, 1, 0.1, 0.9, 0, 0, 0])

    herbivore2_vec = np.array([-0.8, -0.8, 5, 0.1, 0.9, 0, 0, 0])

    carnivore_vec = np.array([-1, 0.2, 1, 0, 0.4, 0.3, 0.3, 0])  # e.g. move to the left

    # init as all equal actions
    vec_cell = np.array([primary_vec, herbivore1_vec, herbivore2_vec, carnivore_vec])
    vec = np.tile(
        np.expand_dims(vec_cell, axis=(0)), (grid_shape[0] * grid_shape[1], 1, 1)
    )

    action = {
        "migrate": vec[:, :, :3],  # shape: (grid_size_flattened, n. FGs, 3)
        "hide_n_feed": vec[:, :, 3:],
    }  # shape: (grid_size_flattened, n. FGs, 1 + n. FGs)

    return vec


if __name__ == "__main__":
    env = PbmEnv(grid_shape=(25, 25))  # (25,25) is the current value in runner.py.
    print(f"Observation shape: {env.get_observation().shape}")

    terminated = False
    time_limit = False
    # See if we get the same with reset first
    env.reset()
    while not (terminated or time_limit):
        action = env.action_space.sample()
        action = action_from_runner_py(env.grid_shape)
        # Adjust sigma, it will be rescaled in env.
        action[:, :, 2] /= env.action_sigma_max
        obs, reward, terminated, time_limit, _ = env.step(action)
        print(
            f"Step {env.n_steps}: reward: {reward:.3f} obs(min): {obs.min()} obs(max):"
            f" {obs.max()}"
        )
        print(
            f"Populations: {env._env.total_pop} #populated cells/FG: "
            f"{env._env.geo_range} average energy: {env._env.avg_energy}"
        )
        print("*" * 80)
    print(f"Done after {env.n_steps} steps")
