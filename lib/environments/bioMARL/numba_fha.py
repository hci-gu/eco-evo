import numpy as np
from numba import njit, types
from numba.types import Array, boolean, float64, int64

# Define the array types
int1d = Array(int64, 1, "C")  # 1D array of integers
float1d = Array(float64, 1, "C")  # 1D array of floats
float2d = Array(float64, 2, "C")  # 2D array of floats
int2d = Array(int64, 2, "C")  # 2D array of integers
rng_type = types.npy_rng


@njit(int1d(int64, float1d, rng_type))
def weighted_choice(n, p, rng):
    """This is needed because numba doesn't support np.random.random.choice or choice
    for default_rng."""
    # Flatten the grid and compute cumulative probabilities
    p = p / p.sum()
    cumulative_probs = np.cumsum(p)

    # Find the index corresponding to the random value
    sampled_index = np.searchsorted(cumulative_probs, rng.random(size=n))
    return sampled_index


@njit(  # Define the function signature
    int2d(  # Return type: 2D array of integers
        int1d,  # fg_pop_n: 1D array of integers
        int1d,  # hiding_pop: 1D array of integers
        float2d,  # feed_action: 2D array of floats
        float1d,  # energy_level: 1D array of floats
        rng_type,  # rng: NumPy default_rng
        float1d,  # max_energy_level: 2D array of floats
        int2d,  # individuals_eaten: 1D array of integers
        boolean,  # skip_primary_producer: Boolean
    )
)
def numba_feeding_hiding_action(
    fg_pop_n,
    hiding_pop,
    feed_action,
    energy_level,  # shape: (n_FGs)
    rng,
    max_energy_level,
    individuals_eaten,
    skip_primary_producer=True,
):

    # assert np.allclose(feed_action[:-1][np.triu_indices(feed_action[:-1].shape[0], k=0)], 0)
    n_fgs = len(fg_pop_n)
    ## The columns in populations are the fgs. The rows contain:
    # 0 - The number of individuals still to act
    # 1 - n_fgs The number of individuals that suceeded in eating from fg i
    # n_fgs + 1 - The number of individuals that failed to eat (idled)
    # n_fgs + 2 - The number of individuals that hid
    populations = np.zeros((n_fgs + 3, n_fgs), dtype=np.int64)
    populations[0] = fg_pop_n  # .astype(populations.dtype)
    populations[-1] = hiding_pop  # .astype(populations.dtype)
    # Quick ref to the agents to act.
    fg_pop_n = populations[0]

    # Move the primary producer direct to idle?
    if skip_primary_producer:
        populations[-2, 0] = populations[0, 0]
        populations[0, 0] = 0

    tot_pop = np.sum(fg_pop_n)
    # Since the rs.choice calls are expensive we compute all the actions in advance
    eat_actions = []

    for n in range(len(fg_pop_n)):
        pop_size = fg_pop_n[n]
        action = feed_action[n]
        s = action.sum()
        if s > 0:
            eat_actions.append(weighted_choice(pop_size, action / s, rng))
        else:
            eat_actions.append(np.zeros(shape=pop_size, dtype=np.int64))

    while tot_pop > 0:
        assert tot_pop == fg_pop_n.sum(), (tot_pop, fg_pop_n.sum())
        assert (fg_pop_n >= 0).all(), (fg_pop_n, feed_action)
        ind = weighted_choice(1, fg_pop_n.astype(np.float64), rng)[0]
        r = eat_actions[ind][int(fg_pop_n[ind]) - 1]  # Choose the associated action.
        if energy_level[ind] < max_energy_level[ind]:  # Not full.
            # NE - This (line below) opens upp for a strategy to always try to eat
            # the most nutritious food available no matter if there is any food of
            # that sort or not because you will always get something else if is not
            # there. This also implies that we will only eat individuals that has
            # not acted yet (this sentence not valid for this code)
            # If this the expected behaviour?

            # The line below is replaced with a selection where we can find food
            # from all but the hide row in populations.
            # p = p * (fg_pop_n > 0)

            # Choose a group to pick from
            p = populations[:-1, r].astype(np.float64)
            p_sum = np.sum(p)

            if p_sum > 0:  # Is there food of this FG?
                # Choose which category to eat from. n_fgs + 2 possibilities:
                # from the non-acting, failed eaters and previous eater of a fg.
                # In order for this to work self.individuals_eaten[ind,r] must be
                # equal to 1. I think the same goes for the original code.j
                group = weighted_choice(1, p, rng)[0]
                populations[group, r] -= individuals_eaten[ind, r]
                if group == 0:
                    tot_pop -= 1
                populations[1 + r, ind] += 1  # We ate a type group individual.
            else:  # No food available or no actions -> idle.
                populations[-2, ind] += 1  # Idle - No food.
        else:
            populations[-2, ind] += 1  # Idle - Full.

        fg_pop_n[ind] -= 1
        tot_pop -= 1

    return populations


@njit(int1d(float1d, float1d))
def weighted_choice_no_rng(p, rng_vec):
    """A new attempt. We supply the random numbers since there seem to be a severe
    memory leak when supplying a rng when using numba. The work around is to delete the
    returned object immediately after use and the call gc.collect. This process is so
    slow so it completely defeats the purpose of using numba in the first place.
    """
    cumulative_probs = np.cumsum(p / p.sum())

    # Find the index corresponding to the random value
    sampled_index = np.searchsorted(cumulative_probs, rng_vec)
    return sampled_index


@njit(  # Define the function signature
    int2d(  # Return type: 2D array of integers
        int1d,  # fg_pop_n: 1D array of integers
        int1d,  # hiding_pop: 1D array of integers
        float2d,  # feed_action: 2D array of floats
        float1d,  # energy_level: 1D array of floats
        float1d,
        float1d,  # max_energy_level: 2D array of floats
        int2d,  # individuals_eaten: 1D array of integers
        boolean,  # skip_primary_producer: Boolean
    )
)
def numba_feeding_hiding_action_no_rng(
    fg_pop_n,
    hiding_pop,
    feed_action,
    energy_level,
    rng_vec,
    max_energy_level,
    individuals_eaten,
    skip_primary_producer=True,
):

    # The assert checks if the action matrices are upper triangular a simple test that
    # implies an acyclic food web.
    # assert np.allclose(feed_action[:-1][np.triu_indices(feed_action[:-1].shape[0], k=0)], 0)
    n_fgs = len(fg_pop_n)
    ## The columns in populations are the fgs. The rows contain:
    # 0 - The number of individuals still to act
    # 1 - n_fgs The number of individuals that suceeded in eating from fg i
    # n_fgs + 1 - The number of individuals that failed to eat (idled)
    # n_fgs + 2 - The number of individuals that hid
    populations = np.zeros((n_fgs + 3, n_fgs), dtype=np.int64)
    populations[0] = fg_pop_n  # .astype(populations.dtype)
    populations[-1] = hiding_pop  # .astype(populations.dtype)
    # Quick ref to the agents to act.
    fg_pop_n = populations[0]

    # Move the primary producer direct to idle?
    if skip_primary_producer:
        populations[-2, 0] = populations[0, 0]
        populations[0, 0] = 0

    tot_pop = np.sum(fg_pop_n)
    rng_idx = 0  # We need to keep track of the number of random numbers we have used.

    # Since the rs.choice calls are expensive we compute all the actions in advance
    # In this version where the random numbers are passed as an argument we could remove
    # the eat_action, but we only do the normalization of the action vectors once using
    # this.
    eat_actions = []
    for n in range(len(fg_pop_n)):
        pop_size = fg_pop_n[n]
        action = feed_action[n]
        s = action.sum()
        if s > 0:
            eat_actions.append(
                weighted_choice_no_rng(action, rng_vec[rng_idx : rng_idx + pop_size])
            )
            rng_idx += pop_size
        else:
            eat_actions.append(np.zeros(shape=pop_size, dtype=np.int64))

    while tot_pop > 0:
        assert tot_pop == fg_pop_n.sum(), (tot_pop, fg_pop_n.sum())
        assert (fg_pop_n >= 0).all(), (fg_pop_n, feed_action)
        # Choose the FG to act.
        ind = weighted_choice_no_rng(
            fg_pop_n.astype(np.float64), rng_vec[rng_idx : rng_idx + 1]
        )[0]
        rng_idx += 1
        r = eat_actions[ind][fg_pop_n[ind] - 1]  # Choose the associated action.
        if energy_level[ind] < max_energy_level[ind]:  # Not full.
            # Choose a group to pick from
            p = populations[:-1, r].astype(np.float64)
            p_sum = np.sum(p)

            if p_sum > 0:  # Is there food of this FG?
                # Choose which category to eat from. n_fgs + 2 possibilities:
                # from the non-acting, failed eaters and previous eater of a fg.
                # In order for this to work self.individuals_eaten[ind,r] must be
                # equal to 1. I think the same goes for the original code.
                group = weighted_choice_no_rng(p, rng_vec[rng_idx : rng_idx + 1])[0]
                rng_idx += 1
                populations[group, r] -= individuals_eaten[ind, r]
                if group == 0:  # Did we eat an individual that has not acted yet.
                    tot_pop -= 1
                populations[1 + r, ind] += 1  # We ate a type group individual.
            else:  # No food available or no actions -> idle.
                populations[-2, ind] += 1  # Idle - No food.
        else:
            populations[-2, ind] += 1  # Idle - Full.

        fg_pop_n[ind] -= 1
        tot_pop -= 1

    return populations


### This function does not work. This is meant to replace hide_n_feed in LargeScaleRLEnv
### with a numba version.
def hide_n_feed(actions, h, energy, rng, padding=1):
    raise NotImplementedError
    new_h = np.zeros_like(h)
    new_energy = np.zeros_like(energy)
    # Do all the hiding on the whole grid in one go.
    h_hiding = update_hiding(h, actions, padding)
    h_eating = (h[:padding:-padding, padding:-padding] - h_hiding).astype(int)
    # Flip the axis of all objects. We want the grid first for contiguous access in the
    # loop.
    actions = actions.reshape(*h_hiding.shape[:-2], -1, copy=True)
    h_hiding = h_hiding.transpose(1, 2, 0).copy()
    h_eating = h_eating.transpose(1, 2, 0).copy()
    energy = energy.transpose(1, 2, 0).copy()
    return new_h, new_energy


@njit()
def numba_get_observation(h, energy, pad, obs_shape, n_fg, energy_in_obs):
    obs = np.zeros(obs_shape)
    assert obs_shape[1] == h.shape[1] - 2 * pad
    for i in range(obs_shape[0]):
        for j in range(obs_shape[1]):
            obs[i, j, :n_fg] = h[:, i : i + 2 * pad + 1, j : j + 2 * pad + 1]

    if energy_in_obs:
        # Only the energy from [i,j] cell is observable so we fill the whole
        # 3x3 grid with that.
        # When using numba this is approximately as fast as using the triple
        # loop, but ~15% faster when using plain python.
        obs[:, :, n_fg:] = energy[:, pad:-pad, pad:-pad].transpose(1, 2, 0)[
            ..., None, None
        ]
    return obs


def update_hiding(h, actions, pad):
    h_hiding = h[:, pad:-pad, pad:-pad] * actions[..., 0].reshape(
        *h.shape[-2:], -1
    ).transpose(2, 0, 1)

    return np.round(h_hiding).astype(int)
