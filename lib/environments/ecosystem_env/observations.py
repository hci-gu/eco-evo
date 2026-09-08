import numpy as np

from lib.environments.ecosystem_env.constants import DIRECTIONS


def shift_field(field, direction):
    out = np.zeros_like(field)
    if direction == "N":
        out[1:, :] = field[:-1, :]
    elif direction == "S":
        out[:-1, :] = field[1:, :]
    elif direction == "E":
        out[:, :-1] = field[:, 1:]
    elif direction == "W":
        out[:, 1:] = field[:, :-1]
    return out


def build_observation_batch(env):
    H, W = env.H, env.W
    biomass_all = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order],
        axis=0,
    ).astype(env.dtype, copy=False)

    visible_fraction = (
        np.float32(1.0)
        - env.prev_hidden_frac
        * (np.float32(1.0) - env._all_visibility_floor[:, None, None])
    )
    visible_biomass_all = biomass_all * visible_fraction

    max_dim = int(env.max_in_dim)
    obs = np.zeros((env.N_dm, max_dim, H, W), dtype=env.dtype)

    for i, pred_id in enumerate(env.dm_ids):
        pred_fg = env.fgs[pred_id]
        own_biomass = pred_fg.biomass.astype(env.dtype, copy=False)
        own_energy = pred_fg.energy_level.astype(env.dtype, copy=False)
        obs_idx = env.obs_others_idx[i]
        n_other = int(obs_idx.shape[0])

        if n_other > 0:
            observed_biomass = visible_biomass_all[obs_idx]
        else:
            observed_biomass = np.zeros((0, H, W), dtype=env.dtype)

        center_dim = 2 + n_other
        neighbor_dim = 1 + n_other

        obs[i, 0] = own_biomass
        obs[i, 1] = own_energy
        if n_other > 0:
            obs[i, 2:2 + n_other] = observed_biomass

        for direction_index, direction in enumerate(DIRECTIONS):
            base = center_dim + direction_index * neighbor_dim
            obs[i, base] = shift_field(own_biomass, direction)
            if n_other > 0:
                for k in range(n_other):
                    obs[i, base + 1 + k] = shift_field(
                        observed_biomass[k], direction)

    return obs.transpose(0, 2, 3, 1).reshape(env.N_dm, H * W, max_dim)
