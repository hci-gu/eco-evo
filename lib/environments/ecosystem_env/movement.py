import numpy as np

from lib.environments.ecosystem_env import impacts
from lib.environments.ecosystem_env.constants import EAST, NORTH, SOUTH, WEST


def _impact_energy_cost(env):
    impact_energy = np.zeros((env.N_dm, env.H, env.W), dtype=env.dtype)
    for i, entries in enumerate(env.dm_impact_tables):
        for impact_id, table in entries:
            map_data = env.grid.get_map(impact_id)
            if map_data is None:
                continue
            _, energy_factor = impacts.interp_impact(
                table, map_data.astype(env.dtype, copy=False))
            impact_energy[i] += energy_factor
    return np.float32(1.0) + impact_energy


def _transfer_to_neighbours(b_stay, r_stay, b_out, r_out):
    b_total = b_stay.copy()
    r_total = r_stay.copy()

    b_total[:, :-1, :] += b_out[:, NORTH, 1:, :]
    r_total[:, :-1, :] += r_out[:, NORTH, 1:, :]

    b_total[:, :, 1:] += b_out[:, EAST, :, :-1]
    r_total[:, :, 1:] += r_out[:, EAST, :, :-1]

    b_total[:, 1:, :] += b_out[:, SOUTH, :-1, :]
    r_total[:, 1:, :] += r_out[:, SOUTH, :-1, :]

    b_total[:, :, :-1] += b_out[:, WEST, :, 1:]
    r_total[:, :, :-1] += r_out[:, WEST, :, 1:]
    return b_total, r_total


def _edge_emigration(b_out, r_out):
    b_emig = (
        b_out[:, NORTH, 0, :].sum(axis=1)
        + b_out[:, SOUTH, -1, :].sum(axis=1)
        + b_out[:, EAST, :, -1].sum(axis=1)
        + b_out[:, WEST, :, 0].sum(axis=1)
    )
    r_emig = (
        r_out[:, NORTH, 0, :].sum(axis=1)
        + r_out[:, SOUTH, -1, :].sum(axis=1)
        + r_out[:, EAST, :, -1].sum(axis=1)
        + r_out[:, WEST, :, 0].sum(axis=1)
    )
    return b_emig, r_emig


def _concentrate_immigration(env, b_emig, r_emig, base_weights):
    weights_flat = base_weights.reshape(-1)
    edge_idx = np.flatnonzero(weights_flat > 0.0)
    b_imm = b_emig[:, None, None] * base_weights[None, :, :]
    r_imm = r_emig[:, None, None] * base_weights[None, :, :]

    if edge_idx.size == 0:
        return b_imm, r_imm

    edge_weights = weights_flat[edge_idx]
    order = np.argsort(-edge_weights)
    sorted_weights = edge_weights[order]
    cumulative_weights = np.cumsum(sorted_weights)

    for i, fid in enumerate(env.dm_ids):
        min_split = float(env.fgs[fid].min_split_biomass or 0.0)
        factor = float(
            getattr(env.fgs[fid], "extinction_threshold_factor", 0.0) or 0.0)
        if min_split <= 0.0 or factor <= 0.0:
            continue

        threshold = factor * min_split
        emigrating_biomass = float(b_emig[i])
        if emigrating_biomass <= 0.0:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            min_per_cell = (
                emigrating_biomass
                * sorted_weights
                / np.maximum(cumulative_weights, 1e-30)
            )
        viable = np.flatnonzero(min_per_cell >= threshold)
        if viable.size == 0:
            continue

        keep_count = int(viable.max()) + 1
        if keep_count >= edge_idx.size:
            continue

        keep_flat = edge_idx[order[:keep_count]]
        new_weights = np.zeros_like(weights_flat)
        kept_weights = weights_flat[keep_flat]
        new_weights[keep_flat] = kept_weights / kept_weights.sum()
        new_weights_2d = new_weights.reshape(base_weights.shape)
        b_imm[i] = emigrating_biomass * new_weights_2d
        r_imm[i] = float(r_emig[i]) * new_weights_2d

    return b_imm, r_imm


def apply_movement(env, actions):
    if env.N_dm == 0:
        return

    biomass = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    reserve = np.stack(
        [env.fgs[fid].energy_reserve for fid in env.dm_ids], axis=0)
    temp_gains = np.stack(
        [env.fgs[fid].temp_energy_gains for fid in env.dm_ids], axis=0)

    cost_factor = _impact_energy_cost(env)
    resting_metabolism = env.dm_resting_metabolism[:, None, None]
    cost_rest = env.dm_cost_rest[:, None, None]
    cost_eat = env.dm_cost_eat[:, None, None]
    cost_move = env.dm_cost_move[:, None, None]

    rest_biomass = biomass * actions.rest
    rest_reserve = np.maximum(
        0,
        reserve * actions.rest
        - biomass * actions.rest * resting_metabolism * cost_rest * cost_factor,
    )

    eat_fraction = actions.eat.sum(axis=1)
    reserve_after_eat = np.maximum(
        0,
        reserve - biomass * resting_metabolism * cost_eat * cost_factor,
    )
    eat_reserve = eat_fraction * reserve_after_eat + temp_gains
    eat_biomass = biomass * eat_fraction

    reserve_after_move = np.maximum(
        0,
        reserve - biomass * resting_metabolism * cost_move * cost_factor,
    )
    move_reserve_choices = actions.move * reserve_after_move[:, None, :, :]
    move_biomass_choices = actions.move * biomass[:, None, :, :]

    flux = env.dm_v[:, None, None, None]
    b_out = move_biomass_choices * flux
    r_out = move_reserve_choices * flux
    b_keep = move_biomass_choices - b_out
    r_keep = move_reserve_choices - r_out

    b_stay = rest_biomass + eat_biomass + b_keep.sum(axis=1)
    r_stay = rest_reserve + eat_reserve + r_keep.sum(axis=1)

    b_total, r_total = _transfer_to_neighbours(b_stay, r_stay, b_out, r_out)

    if env.migration:
        b_emig, r_emig = _edge_emigration(b_out, r_out)
        b_imm, r_imm = _concentrate_immigration(
            env, b_emig, r_emig, env._edge_imm_weights)
        b_total += b_imm
        r_total += r_imm

    max_reserve = b_total * env.dm_max_energy_reserve[:, None, None]
    new_reserve = np.clip(r_total, 0.0, max_reserve)

    for i, fid in enumerate(env.dm_ids):
        env.fgs[fid].biomass = b_total[i].astype(env.dtype, copy=False)
        env.fgs[fid].energy_reserve = new_reserve[i].astype(
            env.dtype, copy=False)
