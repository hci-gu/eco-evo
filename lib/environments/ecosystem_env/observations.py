import numpy as np

from lib.environments.ecosystem_env import impacts
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

    # The rest-action is a hide action, so the fraction of each FG that
    # rested last tick is invisible to observers. The floor acts on the
    # hiding fraction itself:
    #   visible = 1 - prev_hidden * (1 - floor)
    # which matches the symmetric predation-side formulation. Default
    # floor=0 => legacy parity.
    #
    # When a (predator, prey) override exists the floor is resolved per
    # OBSERVER instead (``_obs_vis_floor``), so the visible biomass stack
    # has to be built inside the per-DM loop below.
    if env._has_pair_vis_floor:
        visible_biomass_all = None
    else:
        visible_fraction = (
            np.float32(1.0)
            - env.prev_hidden_frac
            * (np.float32(1.0) - env._all_visibility_floor[:, None, None])
        )
        visible_biomass_all = biomass_all * visible_fraction

    impact_layers = impacts.observable_impact_layers(env)
    n_obs_imp = len(impact_layers)
    # Pre-shift the impact layers once (shared across all DMs).
    impact_shifts = {
        direction: [shift_field(layer, direction) for layer in impact_layers]
        for direction in DIRECTIONS
    }

    max_dim = int(env.max_in_dim)
    obs = np.zeros((env.N_dm, max_dim, H, W), dtype=env.dtype)

    for i, pred_id in enumerate(env.dm_ids):
        pred_fg = env.fgs[pred_id]
        own_biomass = pred_fg.biomass.astype(env.dtype, copy=False)
        own_energy = pred_fg.energy_level.astype(env.dtype, copy=False)
        obs_idx = env.obs_others_idx[i]
        n_other = int(obs_idx.shape[0])

        if n_other > 0:
            if visible_biomass_all is not None:
                observed_biomass = visible_biomass_all[obs_idx]
            else:
                floor_i = env._obs_vis_floor[i][:, None, None]
                observed_biomass = biomass_all[obs_idx] * (
                    np.float32(1.0)
                    - env.prev_hidden_frac[obs_idx]
                    * (np.float32(1.0) - floor_i)
                )
        else:
            observed_biomass = np.zeros((0, H, W), dtype=env.dtype)

        center_dim = 2 + n_other + n_obs_imp
        neighbor_dim = 1 + n_other + n_obs_imp

        obs[i, 0] = own_biomass
        obs[i, 1] = own_energy
        if n_other > 0:
            obs[i, 2:2 + n_other] = observed_biomass
        for k, layer in enumerate(impact_layers):
            obs[i, 2 + n_other + k] = layer

        for direction_index, direction in enumerate(DIRECTIONS):
            base = center_dim + direction_index * neighbor_dim
            obs[i, base] = shift_field(own_biomass, direction)
            if n_other > 0:
                for k in range(n_other):
                    obs[i, base + 1 + k] = shift_field(
                        observed_biomass[k], direction)
            for k, layer_shift in enumerate(impact_shifts[direction]):
                obs[i, base + 1 + n_other + k] = layer_shift

    return obs.transpose(0, 2, 3, 1).reshape(env.N_dm, H * W, max_dim)
