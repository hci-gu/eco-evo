import numpy as np
import scipy.special
import scipy.stats

from .numba_fha import numba_feeding_hiding_action_no_rng
from .utilities import get_rnd_gen

np.set_printoptions(suppress=True, precision=3)


class FeedingAction:
    def __init__(
        self,
        functional_groups,
        seed=123,
        additional_actions=1,  # e.g. hiding (could add more)
        feeding_energy_reward=None,  # shape (functional_groups, functional_groups)
        base_energy_cost=None,
        hiding_energy_cost=None,
        max_energy_level=None,
        low_energy_death_rate=1,  # multiplier
        idling_energy_cost=None,  # cost of not eating
        individuals_eaten=None,  # shape (functional_groups, functional_groups) | (eating, eaten)
        verbose=0,
    ):
        self.rs = get_rnd_gen(seed)
        self.n_functional_groups = functional_groups
        self.additional_actions = additional_actions
        if feeding_energy_reward is not None:
            self.feeding_energy_reward = feeding_energy_reward
        else:
            self.feeding_energy_reward = np.ones(
                (self.n_functional_groups, self.n_functional_groups)
            )
        if base_energy_cost is not None:
            self.base_energy_cost = base_energy_cost
        else:
            self.base_energy_cost = np.zeros(
                (self.n_functional_groups, self.n_functional_groups)
            )
        if hiding_energy_cost is not None:
            self.hiding_energy_cost = hiding_energy_cost
        else:
            self.hiding_energy_cost = np.ones((self.n_functional_groups))
        if max_energy_level is not None:
            self.max_energy_level = max_energy_level
        else:
            self.max_energy_level = np.ones(self.n_functional_groups) * 100
        self.low_energy_death_rate = low_energy_death_rate
        if idling_energy_cost is None:
            idling_energy_cost = np.zeros((self.n_functional_groups))
        self.idling_energy_cost = idling_energy_cost
        if individuals_eaten is None:
            individuals_eaten = np.ones(
                (self.n_functional_groups, self.n_functional_groups)
            )
        self.individuals_eaten = individuals_eaten
        self.verbose = verbose

        self.feeding_hiding_action = self.numba_feeding_hiding_action

    def old_feeding_hiding_action(
        self,
        cell_populations,  # shape: (n_FGs)
        fg_actions,  # shape: (n_FGs, 1  + n. FGs) -> hiding, feeding on FGs
        energy_level,  # shape: (n_FGs)
        approximate=None,
    ):
        if approximate is not None:
            den = np.maximum(1, np.min(cell_populations))
            cell_populations = cell_populations / den
        else:
            den = 1

        hiding_pop = np.round(cell_populations * fg_actions[:, 0]).astype(int)
        fg_pop_n = (cell_populations - hiding_pop).astype(int)
        n_fgs = len(fg_pop_n)
        tot_pop = np.sum(fg_pop_n)
        # rnd order of action
        rnd_ord = self.rs.choice(
            np.concatenate([np.repeat(i, fg_pop_n[i]) for i in range(n_fgs)]),
            tot_pop,
            replace=False,
        )
        feed_action = fg_actions[:, 1:]
        # rm_vec = np.zeros(n_fgs)
        rm_vec = fg_pop_n.copy()

        feeding_sum_rewards = np.zeros(self.n_functional_groups)
        for ind in rnd_ord:
            p = feed_action[ind]
            if rm_vec[ind] > 0:
                # action if individual had not been previously eaten
                if (
                    p is not None and energy_level[ind] < self.max_energy_level[ind]
                ):  # don't eat if you are full!
                    p = p * (fg_pop_n > 0).astype(int)  # check if food still available
                    tot_p = np.sum(p)
                    if tot_p > 0:
                        r = self.rs.choice(range(n_fgs), p=p / tot_p)
                        rm_vec[r] = rm_vec[r] - self.individuals_eaten[ind, r]
                        fg_pop_n[r] = (
                            fg_pop_n[r] - self.individuals_eaten[ind, r]
                        )  # remove eaten individual from total
                        # print(" self.feeding_energy_reward[ind, r]", ind, r,  self.feeding_energy_reward[ind, r])
                        feeding_sum_rewards[ind] = (
                            feeding_sum_rewards[ind]
                            + self.feeding_energy_reward[ind, r]
                        )
                    else:
                        feeding_sum_rewards[ind] = (
                            feeding_sum_rewards[ind] - self.idling_energy_cost[ind]
                        )
                rm_vec[ind] -= 1
            else:
                # no action if individual was eaten in previous step
                pass  # No individual left to act.
                # rm_vec[ind] += 1  #
        if self.verbose == 2:
            print("feeding_sum_rewards", feeding_sum_rewards, fg_pop_n)
        # compute new energy levels - HIDING
        # hiding_pop_energy_loss = self.hiding_energy_cost * hiding_pop

        # energy_level is the mean for the cell population
        energy_of_hiding_pop = np.maximum(0, energy_level - self.hiding_energy_cost)

        if self.verbose == 2:
            print("energy_of_hiding_pop", energy_of_hiding_pop)

        # kill population if it couldn't afford hiding
        death_rate = (
            1 - np.minimum(1, energy_of_hiding_pop)
        ) * self.low_energy_death_rate
        # death rate > 0 when energy level < 1
        # (increases with more negative values)
        death_prob = 1 - np.exp(-death_rate)
        hiding_pop = np.round(hiding_pop * (1 - death_prob)).astype(int)
        """
        # test
        energy_of_hiding_pop = np.array([ 0,  1,  3, -1, -3])
        hiding_pop = np.ones(5) * 100
        death_rate = (1 - np.minimum(1, energy_of_hiding_pop)) * 0.5
        death_prob = 1 - np.exp(-death_rate)
        hiding_pop = np.round(hiding_pop * (1 - death_prob)).astype(int) 
        """
        # compute new energy levels - FEEDING
        feeding_avg_rewards = (
            feeding_sum_rewards / np.maximum(1, fg_pop_n) + energy_level
        )
        feeding_avg_rewards = np.minimum(feeding_avg_rewards, self.max_energy_level)
        # energy = (feeding_sum_rewards * fg_pop_n + energy_of_hiding_pop * hiding_pop
        #           ) / np.maximum(1, (fg_pop_n + hiding_pop))

        cell_populations_nz = np.maximum((fg_pop_n + hiding_pop), 1)
        energy = feeding_avg_rewards * (
            fg_pop_n / cell_populations_nz
        ) + energy_of_hiding_pop * (hiding_pop / cell_populations_nz)
        # print(feeding_avg_rewards,  (fg_pop_n / cell_populations_nz),  energy_of_hiding_pop,  (hiding_pop / cell_populations_nz))
        # print(feeding_avg_rewards * (fg_pop_n / cell_populations_nz),
        #       energy_of_hiding_pop * (hiding_pop / cell_populations_nz))
        # print(feeding_avg_rewards * (fg_pop_n / cell_populations_nz) + \
        #       energy_of_hiding_pop * (hiding_pop / cell_populations_nz))
        # print(energy)
        if self.verbose == 3:
            print("cell_populations", cell_populations)
            print("energy_level", energy_level)
            print("feeding_avg_rewards", feeding_avg_rewards, fg_pop_n)
            print("energy_of_hiding_pop", energy_of_hiding_pop, hiding_pop)
            print("energy", energy)

        return (hiding_pop + fg_pop_n) * den, energy

    def numba_feeding_hiding_action(
        self,
        cell_populations,  # shape: (n_FGs)
        fg_actions,  # shape: (n_FGs, 1  + n. FGs) -> hiding, feeding on FGs
        energy_level,  # shape: (n_FGs)
        approximate=None,
        skip_primary_producer=True,
    ):
        if approximate is not None:
            den = np.maximum(1, np.min(cell_populations))
            cell_populations = cell_populations / den
        else:
            den = 1

        hiding_pop = np.round(cell_populations * fg_actions[:, 0]).astype(int)
        fg_pop_n = (cell_populations - hiding_pop).astype(int)
        feed_action = fg_actions[:, 1:].astype(np.float64)
        r_numbers = self.rs.uniform(size=3 * int(np.sum(fg_pop_n[1:])))
        populations = numba_feeding_hiding_action_no_rng(
            fg_pop_n,
            hiding_pop,
            feed_action,
            energy_level.copy(),
            r_numbers,
            self.max_energy_level,
            self.individuals_eaten.astype(int),
            skip_primary_producer,
        )

        # energy_level is the mean for the cell population
        energy_of_hiding_pop = np.maximum(0, energy_level - self.hiding_energy_cost)

        # kill population if it couldn't afford hiding
        death_rate = (
            1 - np.minimum(1, energy_of_hiding_pop)
        ) * self.low_energy_death_rate
        # death rate > 0 when energy level < 1
        # (increases with more negative values)
        death_prob = 1 - np.exp(-death_rate)
        populations[-1] = np.round(hiding_pop * (1 - death_prob)).astype(int)

        energy_deltas = np.concatenate(
            [
                self.feeding_energy_reward.T,
                -self.idling_energy_cost[None],
                -self.hiding_energy_cost[None],
            ]
        )
        # group_energy = np.maximum(0, energy_level + energy_deltas)
        group_energy = np.clip(energy_level + energy_deltas, 0.0, self.max_energy_level)
        new_pop = populations[1:].sum(axis=0)
        energy = (populations[1:] * group_energy).sum(axis=0) / np.maximum(1, new_pop)

        return new_pop * den, energy

    def grid_feeding_hiding_action(self, actions, h, energy):
        # We assume that the padding of `h` and `energy` has been removed.

        # Do all the hiding on the whole grid in one go.
        h_hiding = self.update_hiding(h, actions)
        h_eating = (h - h_hiding).astype(int)
        # Flip the axes of all objects. We want the grid first for contiguous access in
        # the loop.

        shape_eat_actions = (*h.shape[-2:], actions.shape[-2], actions.shape[-1] - 1)
        # Drop the hide actions.
        eat_actions = (
            actions[..., 1:].reshape(shape_eat_actions).astype(np.float64, order="C")
        )
        h_hiding = h_hiding.transpose(1, 2, 0).copy()
        h_eating = h_eating.transpose(1, 2, 0).copy()
        energy = energy.transpose(1, 2, 0).copy()
        updated_pops = np.empty(
            shape=(*h_eating.shape[:2], h_eating.shape[2] + 3, h_eating.shape[2])
        )

        for row in range(h_eating.shape[0]):
            for col in range(h_eating.shape[1]):
                # Generate the number of random numbers needed. We need three times as
                # many numbers as the agents to act. For each agent we need one to
                # select the agent to act, one to select which functional group to eat,
                # and finally we need one random number to choose the "state" of the
                # eaten FG (not acted, idle, eating fg 0, eating fg 1, ...).

                # Note the "1:" is because we dont
                # use the primary producers in the simulation.
                #TODO: add  comment about the 3.
                r_numbers = self.rs.uniform(size=3 * int(sum(h_eating[row, col, 1:])))
                updated_pops[row, col] = numba_feeding_hiding_action_no_rng(
                    h_eating[row, col],
                    h_hiding[row, col],
                    eat_actions[row, col],
                    energy[row, col],
                    r_numbers,
                    self.max_energy_level,
                    self.individuals_eaten.astype(int),
                    True,
                )

        # Update the energy.
        # Kill the the ones that hid which did not have enough energy.
        # TODO: Should we also kill the ones that idled in the same way?
        energy_hiding_pop = np.clip(
            energy - self.hiding_energy_cost, 0.0, self.max_energy_level
        )
        death_rate = (1 - np.minimum(1, energy_hiding_pop)) * self.low_energy_death_rate
        # -1 is the ones that hid.
        updated_pops[:, :, -1] = np.round(h_hiding * np.exp(-death_rate)).astype(int)

        # This is used to compute the new average energy.
        energy_deltas = np.concatenate(
            [
                self.feeding_energy_reward.T,
                -self.idling_energy_cost[None],
                -self.hiding_energy_cost[None],
            ]
        )
        group_energy = np.clip(
            energy[..., None, :] + energy_deltas, 0, self.max_energy_level
        )
        new_pop = updated_pops[:, :, 1:].sum(axis=2)
        energy = (updated_pops[:, :, 1:] * group_energy).sum(axis=2) / np.maximum(
            1, new_pop
        )

        return new_pop.transpose(2, 0, 1), energy.transpose(2, 0, 1)

    @staticmethod
    def update_hiding(h, actions):
        h_hiding = h * actions[..., 0].reshape(*h.shape[-2:], -1).transpose(2, 0, 1)

        return np.round(h_hiding).astype(int)

    @property
    def get_base_energy_cost(self):
        return self.base_energy_cost


class MigrationAction:
    def __init__(
        self,
        baseline_m=None,
        baseline_s=0.3,
        max_speed=1,
        migration_energy_cost=-1,
    ):
        if baseline_m is None:
            baseline_m = [0, 0]
        self.baseline_m = baseline_m
        self.baseline_s = baseline_s
        self.max_speed = max_speed
        z = np.ones((3, 3))
        z[1, 1] = 0
        self.migration_energy_cost = -np.abs(migration_energy_cost) * z
        # Used over and over again in migration_action.
        self._mesh = np.array(np.meshgrid([-1, 0, 1], [1, 0, -1])).reshape((2, 9)).T

    def migration_action(
        self,
        m: np.ndarray = None,  # shape: (2) - ideally in range [-1, 1]
        s: float = None,
        mask: np.ndarray = None,  # shape: (3, 3) defines boundaries of where you can go
    ):
        if m is None:
            m = (
                self.baseline_m
            )  # 1st: pos -> right, neg -> left | 2nd: pos -> up, neg -> down
        if s is None:
            s = self.baseline_s  # small values make values spread more
        s = max(s, 0.1)  # logpdf breaks with too small sigmas.
        if mask is None:
            mask = np.ones((3, 3))
        m = m * self.max_speed
        cov = np.array([[1, 0], [0, 1]]) * s
        # x = np.array(np.meshgrid([-1, 0, 1], [1, 0, -1])).reshape((2, 9)).T
        x = self._mesh
        pdf = scipy.stats.multivariate_normal.logpdf(x=x, mean=m, cov=cov).reshape(
            (3, 3)
        )
        rescaled_pdf = scipy.special.softmax(pdf) * mask
        rescaled_pdf /= np.sum(rescaled_pdf)
        # print(rescaled_pdf)
        return rescaled_pdf


# class DispersalHabitatSuitability


class ReproduceAction:
    def __init__(
        self,
        functional_groups=10,
        fg_growth_rates=None,  # must be greater than 1!
        fg_reproduction_freq=None,
        fg_carrying_capacity=None,
        seed_bank=None,  # e.g. soil seeds (fg, x, y)
    ):
        self.functional_groups = functional_groups
        if fg_growth_rates is None:
            self.fg_growth_rates = np.ones((self.functional_groups)) + 0.2
        else:
            self.fg_growth_rates = fg_growth_rates
        if fg_reproduction_freq is None:
            fg_reproduction_freq = np.ones((self.functional_groups))
        self.fg_reproduction_freq = fg_reproduction_freq
        if fg_carrying_capacity is None:
            fg_carrying_capacity = np.array([np.inf for _ in range(functional_groups)])
        self.fg_carrying_capacity = fg_carrying_capacity
        self.fg_with_k = np.array(
            [i for i in range(functional_groups) if fg_carrying_capacity[i] < np.inf]
        )
        self.seed_bank = seed_bank
        if seed_bank is None:
            self.seed_bank_fg = np.zeros(functional_groups)
        else:
            self.seed_bank_fg = np.sum(seed_bank, axis=(1, 2))
        self.reset_counter()

    def reproduce_action(
        self,
        h,  # (fg, x, y)
        h_k=None,  # (fg, x, y) relative carrying capacity (ie habitat suitability)
    ):
        if self.seed_bank is not None:
            h = np.maximum(h, self.seed_bank)

        if h_k is None:
            h_k = np.ones(h.shape)
        does_reproduce = self.counter % self.fg_reproduction_freq == 0
        # print("does_reproduce", does_reproduce)
        growth = does_reproduce.astype(float) * self.fg_growth_rates
        growth[growth < 1] = 1
        new_h = np.einsum("sxy, s -> sxy", h, growth)
        delta_h = new_h - h
        delta_h[delta_h > 0] = np.maximum(
            delta_h[delta_h > 0], 1
        )  # add at least 1 individual
        h_new = np.round(h + delta_h)
        fg_k = np.einsum(
            "sxy, s -> sxy",
            h_k[self.fg_with_k],
            self.fg_carrying_capacity[self.fg_with_k],
        )
        h_new[self.fg_with_k] = np.minimum(h_new[self.fg_with_k], fg_k)
        self.counter += 1
        return h_new

    def reset_counter(self):
        self.counter = 1


class LowEnergyMortality:
    def __init__(
        self,
        functional_groups=10,
        fg_baseline_mortality=None,
        energy_mid_point=None,  # array of shape (n. FGs)
        mortality_logistic_growth=None,
        grid_shape=None,  # env.h[0].shape
    ):
        self.functional_groups = functional_groups
        if fg_baseline_mortality is None:
            fg_baseline_mortality = np.zeros(self.functional_groups)
        if energy_mid_point is None:
            energy_mid_point = np.ones(self.functional_groups) * 50
        if mortality_logistic_growth is None:
            mortality_logistic_growth = np.ones(self.functional_groups) * -0.1

        # fg_baseline_mortality, mortality_logistic_growth, energy_mid_point: (fg,)
        # expand to 3D: (fg, x, y)
        self.fg_baseline_mortality3d = np.tile(
            np.expand_dims(fg_baseline_mortality, axis=(1, 2)), (1,) + grid_shape
        )
        self.energy_mid_point3d = np.tile(
            np.expand_dims(energy_mid_point, axis=(1, 2)), (1,) + grid_shape
        )
        self.mortality_logistic_growth3d = np.tile(
            np.expand_dims(mortality_logistic_growth, axis=(1, 2)), (1,) + grid_shape
        )

    # energy: (fg, x, y)
    def fg_mortality_function(self, energy):
        delta = self.fg_baseline_mortality3d + (
            (1.0 - self.fg_baseline_mortality3d)
            / (
                1.0
                + np.exp(
                    -self.mortality_logistic_growth3d
                    * (energy - self.energy_mid_point3d)
                )
            )
        )

        return delta
