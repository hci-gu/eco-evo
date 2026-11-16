import numpy as np

from lib.environments.ecosystem_env import encounter, impacts, interactions, refuge
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


def _encounter_view(env, i, enc, channels, inside):
    """Section 275: the own-cell and neighbour readings of DM ``i``'s
    [B_own, observed others, J] channels under the encounter rule.

    Returns (seen, neighbour): seen[k] the own-cell reading of channel k,
    neighbour[q][k] its reading in direction q (0 off the grid)."""
    plans = getattr(env, "_encounter_plans", None)
    if plans is None:
        plans = env._encounter_plans = {}
    plan = plans.get(i)
    if plan is None:
        plan = plans[i] = encounter.build_plan(
            enc, i, len(env.global_fg_order), (env.H, env.W))
    key = encounter.tick_key(int(getattr(env, "current_world_seed", 0) or 0),
                             int(env.tick_count))
    shifted = [[shift_field(f, d) for f in channels] for d in DIRECTIONS]
    densities = (np.stack([channels[s] if r == 0 else shifted[r - 1][s]
                           for s, r in plan["draws"]]).astype(np.float32, copy=False)
                 if plan["draws"] else None)
    met_all, entered_all = encounter.plan_outcomes(plan, key, densities)
    false = np.zeros((env.H, env.W), dtype=bool)
    draws = {}
    for key_sr, how in plan["kind"].items():
        if how == "away":
            draws[key_sr] = (false, false)
        else:
            draws[key_sr] = (met_all[how[1]], entered_all[how[1]])
    zero = np.float32(0.0)
    seen = []
    for k, field in enumerate(channels):
        d = draws.get((enc["source"][k], 0))
        seen.append(field if d is None else np.where(d[0], field, zero))
    neighbour = []
    for q, direction in enumerate(DIRECTIONS):
        row = []
        for k in range(len(channels)):
            n = shifted[q][k]
            d = draws.get((enc["source"][k], q + 1))
            if d is not None:
                met, entered = d
                n = inside[direction] * np.where(
                    met, n, np.where(entered, zero, seen[k]))
            row.append(n.astype(env.dtype, copy=False))
        neighbour.append(row)
    return seen, neighbour


def build_observation_batch(env):
    H, W = env.H, env.W
    biomass_all = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order],
        axis=0,
    ).astype(env.dtype, copy=False)
    # Section 223.12: observers see a refuge group's accessible part.
    biomass_all = refuge.accessible_rows(env, biomass_all)

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

    # Section 273: in-grid mask per direction for the neighbour blend.
    inside = ({d: shift_field(np.ones((H, W), dtype=env.dtype), d)
               for d in DIRECTIONS}
              if (getattr(env, "_has_nbr_w", False)
                  or getattr(env, "_has_encounter", False)) else None)

    max_dim = int(env.max_in_dim)
    light = (interactions.light_level(env) if env._has_daylight
             else None)
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

        # Juvenile channels (stages.py): the staged groups this DM observes
        # or is, after the observed biomasses; another group's J is
        # visible-scaled as its biomass channel, the own J is not.
        juvenile_idx = getattr(env, "obs_juvenile_idx", None)
        if juvenile_idx is not None and juvenile_idx[i].shape[0] > 0:
            own_idx = int(env.dm_index_in_all[i])
            channels = []
            for j in juvenile_idx[i]:
                j = int(j)
                juv = env.fgs[env.global_fg_order[j]].juvenile.astype(env.dtype, copy=False)
                if j != own_idx:
                    floor = (env.vis_floor_mat[i, j] if env._has_pair_vis_floor
                             else env._all_visibility_floor[j])
                    juv = juv * (np.float32(1.0) - env.prev_hidden_frac[j]
                                 * (np.float32(1.0) - np.float32(floor)))
                channels.append(juv)
            observed_biomass = np.concatenate(
                [observed_biomass, np.stack(channels).astype(env.dtype)], axis=0)
            n_other = int(observed_biomass.shape[0])

        center_dim = 2 + n_other + n_obs_imp
        neighbor_dim = 1 + n_other + n_obs_imp

        obs[i, 0] = own_biomass
        obs[i, 1] = own_energy
        if n_other > 0:
            obs[i, 2:2 + n_other] = observed_biomass
        for k, layer in enumerate(impact_layers):
            obs[i, 2 + n_other + k] = layer

        channels = [own_biomass] + [observed_biomass[k] for k in range(n_other)]
        enc = env.obs_encounter[i] if getattr(env, "_has_encounter", False) else None
        if enc is not None:
            # Section 275: what the group met (encounter.py).
            seen, neighbour = _encounter_view(env, i, enc, channels, inside)
            if n_other > 0:
                obs[i, 2:2 + n_other] = np.stack(seen[1:])
            for direction_index in range(len(DIRECTIONS)):
                base = center_dim + direction_index * neighbor_dim
                for k in range(len(channels)):
                    obs[i, base + k] = neighbour[direction_index][k]
                for k, layer_shift in enumerate(impact_shifts[DIRECTIONS[direction_index]]):
                    obs[i, base + 1 + n_other + k] = layer_shift
            channels = None

        nbr_w = env.obs_nbr_w[i] if getattr(env, "_has_nbr_w", False) else None
        for direction_index, direction in enumerate(DIRECTIONS if channels is not None else ()):
            base = center_dim + direction_index * neighbor_dim
            for k, field in enumerate(channels):
                shifted = shift_field(field, direction)
                if nbr_w is not None and nbr_w[k] < 1.0:
                    # Section 273: c + w (n - c) on in-grid neighbours, 0
                    # outside the grid as before.
                    shifted = inside[direction] * (
                        field + nbr_w[k] * (shifted - field))
                obs[i, base + k] = shifted
            for k, layer_shift in enumerate(impact_shifts[direction]):
                obs[i, base + 1 + n_other + k] = layer_shift

        set_point_obs = bool(getattr(env, "dm_set_point_obs", None) is not None
                             and env.dm_set_point_obs[i])
        if env._has_daylight:
            # Light level of this tick, uniform over the grid; the last
            # slot of the DM's compact layout (section 137), or the one
            # before the set-point room (section 271).
            obs[i, int(env.per_dm_in_dim[i]) - 1 - int(set_point_obs)] = light
        if set_point_obs:
            # Section 271: the own room to the defended level,
            # max(s_T - s, 0) per cell - what the fish feels as hunger.
            target = env.set_point_table[interactions.light_index(env), i]
            obs[i, int(env.per_dm_in_dim[i]) - 1] = np.maximum(
                np.float32(target) - own_energy, np.float32(0.0))

    return obs.transpose(0, 2, 3, 1).reshape(env.N_dm, H * W, max_dim)
