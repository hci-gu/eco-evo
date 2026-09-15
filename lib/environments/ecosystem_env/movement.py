import numpy as np

from lib.environments.ecosystem_env import impacts
from lib.environments.ecosystem_env.constants import EAST, NORTH, SOUTH, WEST
from lib.environments.ecosystem_env.state import ActionSettlement


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


def apply_energy_costs(env, actions):
    if env.N_dm == 0:
        return None

    biomass = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    reserve = np.stack(
        [env.fgs[fid].energy_reserve for fid in env.dm_ids], axis=0)
    temp_gains = np.stack(
        [env.fgs[fid].temp_energy_gains for fid in env.dm_ids], axis=0)

    # Per-DM extra energy cost from all impacts the FG is affected by
    # (sum of energy_factor lookups over every impact with a valid table).
    # The factor (1 + sum energy_factor) scales resting / feeding /
    # movement metabolic costs.
    cost_factor = impacts.impact_energy_cost_factor(env)
    resting_metabolism = env.dm_resting_metabolism[:, None, None]
    cost_rest = env.dm_cost_rest[:, None, None]
    cost_eat = env.dm_cost_eat[:, None, None]
    cost_move = env.dm_cost_move[:, None, None]

    rest_biomass = biomass * actions.rest
    rest_reserve = np.maximum(
        0,
        reserve * actions.rest
        - biomass * actions.rest * resting_metabolism * cost_rest
        * cost_factor,
    )

    eat_fraction = actions.eat.sum(axis=1)
    eat_cost = (
        biomass * eat_fraction * resting_metabolism * cost_eat * cost_factor)
    eat_reserve = np.maximum(
        0,
        reserve * eat_fraction - eat_cost,
    ) + temp_gains
    eat_biomass = biomass * eat_fraction

    # Charge only the fraction choosing movement. Movement speed is
    # applied later when the settled population moves between cells.
    move_fraction = actions.move.sum(axis=1)
    move_cost = (
        biomass * move_fraction * resting_metabolism * cost_move * cost_factor)
    move_reserve_pool = np.maximum(
        0,
        reserve * move_fraction - move_cost,
    )
    safe_fraction = np.where(move_fraction > 0, move_fraction, np.float32(1.0))
    direction_fraction = actions.move / safe_fraction[:, None, :, :]
    move_reserve_choices = direction_fraction * move_reserve_pool[:, None, :, :]
    move_biomass_choices = actions.move * biomass[:, None, :, :]

    return ActionSettlement(
        stationary_biomass=rest_biomass + eat_biomass,
        stationary_reserve=rest_reserve + eat_reserve,
        moving_biomass=move_biomass_choices,
        moving_reserve=move_reserve_choices,
    )


def apply_movement(env, settlement):
    if env.N_dm == 0 or settlement is None:
        return

    flux = env.dm_v[:, None, None, None]
    b_out = settlement.moving_biomass * flux
    r_out = settlement.moving_reserve * flux
    b_keep = settlement.moving_biomass - b_out
    r_keep = settlement.moving_reserve - r_out

    b_stay = settlement.stationary_biomass + b_keep.sum(axis=1)
    r_stay = settlement.stationary_reserve + r_keep.sum(axis=1)

    b_total, r_total = _transfer_to_neighbours(b_stay, r_stay, b_out, r_out)

    if env.migration:
        b_emig, r_emig = _edge_emigration(b_out, r_out)
        b_imm, r_imm = _concentrate_immigration(
            env, b_emig, r_emig, env._edge_imm_weights)
        b_total += b_imm
        r_total += r_imm

    # Sub-threshold split suppression (Section 69). A move action splits
    # the moving share of a cell across up to 4 directions, and
    # ``apply_extinction_threshold`` zeroes every cell that ends the tick
    # below ``thr = extinction_threshold_factor * min_split_biomass``. A
    # diffusing DM therefore bleeds biomass through the sweep at a rate
    # that can dominate its starvation term (measured for porpoises in
    # Section 68.4: 25.1 of 40 ton lost to the sweep vs. 14.9 ton to
    # starvation).
    #
    # Fix: an outflow whose DESTINATION would still end up below thr
    # after receiving it is cancelled and returned to the source cell.
    # This is monotone-safe: sources only gain biomass, and the only
    # cells that lose inflow are ones that were going to be zeroed
    # anyway, so the number of sub-threshold cells cannot increase and
    # no viable cell is made non-viable. Same principle as the top-k
    # immigration concentration above, generalized to the ordinary
    # 4-direction split.
    if env._dm_split_thr_any:
        b_total, r_total = suppress_subthreshold_splits(
            env, b_out, r_out, b_total, r_total)

    max_reserve = b_total * env.dm_max_energy_reserve[:, None, None]
    new_reserve = np.clip(r_total, 0.0, max_reserve)

    for i, fid in enumerate(env.dm_ids):
        env.fgs[fid].biomass = b_total[i].astype(env.dtype, copy=False)
        env.fgs[fid].energy_reserve = new_reserve[i].astype(
            env.dtype, copy=False)


def suppress_subthreshold_splits(env, b_out, r_out, b_total, r_total):
    """Cancel move outflows that would land in a still-sub-threshold cell.

    ``b_out[i, d, y, x]`` is the biomass leaving cell (y, x) of DM ``i``
    in direction ``d`` (0=N, 1=E, 2=S, 3=W), and ``b_total`` is the
    tentative post-movement biomass with every transfer applied. For
    each DM with ``thr = extinction_threshold_factor *
    min_split_biomass > 0`` we look up the tentative total of the
    destination cell; where that total is below ``thr`` the transfer is
    undone -- the biomass (and its energy reserve) stays in the source
    cell instead of being swept away by ``apply_extinction_threshold``
    at the end of the tick.

    Only in-grid destinations are considered. Off-grid directions are
    treated as unblocked so the migration emigration/immigration path
    (which has its own top-k concentration) is left untouched.

    Returns the corrected ``(b_total, r_total)``.
    """
    thr = env._dm_split_thr[:, None, None, None]     # (N_dm,1,1,1)
    inf = np.array(np.inf, dtype=env.dtype)
    # Tentative destination totals, per direction, aligned on the SOURCE
    # cell. Mirrors the slice-assign offsets used above.
    dest = np.full(b_out.shape, inf, dtype=env.dtype)
    dest[:, NORTH, 1:, :] = b_total[:, :-1, :]       # N: (y,x) -> (y-1,x)
    dest[:, EAST, :, :-1] = b_total[:, :, 1:]        # E: (y,x) -> (y,x+1)
    dest[:, SOUTH, :-1, :] = b_total[:, 1:, :]       # S: (y,x) -> (y+1,x)
    dest[:, WEST, :, 1:] = b_total[:, :, :-1]        # W: (y,x) -> (y,x-1)

    blocked = (b_out > 0.0) & (dest < thr)
    if not np.any(blocked):
        return b_total, r_total

    b_cancel = np.where(blocked, b_out, np.float32(0.0))
    r_cancel = np.where(blocked, r_out, np.float32(0.0))
    # Source keeps what it was going to send away.
    b_total = b_total + b_cancel.sum(axis=1)
    r_total = r_total + r_cancel.sum(axis=1)
    # Destination no longer receives it (same offsets as the forward
    # transfer, with the sign flipped).
    b_total[:, :-1, :] -= b_cancel[:, NORTH, 1:, :]
    r_total[:, :-1, :] -= r_cancel[:, NORTH, 1:, :]
    b_total[:, :, 1:] -= b_cancel[:, EAST, :, :-1]
    r_total[:, :, 1:] -= r_cancel[:, EAST, :, :-1]
    b_total[:, 1:, :] -= b_cancel[:, SOUTH, :-1, :]
    r_total[:, 1:, :] -= r_cancel[:, SOUTH, :-1, :]
    b_total[:, :, :-1] -= b_cancel[:, WEST, :, 1:]
    r_total[:, :, :-1] -= r_cancel[:, WEST, :, 1:]
    # float32 round-off on the +/- pair can leave tiny negatives.
    np.maximum(b_total, 0.0, out=b_total)
    np.maximum(r_total, 0.0, out=r_total)
    return b_total, r_total
