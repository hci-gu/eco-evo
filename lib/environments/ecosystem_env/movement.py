import numpy as np
import torch

from lib.environments.ecosystem_env import debug_food, impacts, source_tracking
from lib.environments.ecosystem_env.constants import EAST, NORTH, SOUTH, WEST
from lib.environments.ecosystem_env.grid_masks import build_movement_mask
from lib.environments.ecosystem_env.state import ActionSettlement


def _current_xor(left, right):
    # Tensor/scalar xor needs device-local constants for fullgraph tracing
    # and CUDA graph capture (no host copies during a tick).
    for value in (left, right):
        if isinstance(value, torch.Tensor):
            if not isinstance(left, torch.Tensor):
                left = torch.full((), int(left), dtype=value.dtype, device=value.device)
            if not isinstance(right, torch.Tensor):
                right = torch.full((), int(right), dtype=value.dtype, device=value.device)
            return torch.bitwise_xor(left, right)
    return left ^ right


def _current_hash(value):
    # This multiplier fits signed int64 intermediates on NumPy and Torch.
    value = value & 0xFFFFFFFF
    value = (_current_xor(value, value >> 16) * 0x45D9F3B) & 0xFFFFFFFF
    value = (_current_xor(value, value >> 16) * 0x45D9F3B) & 0xFFFFFFFF
    return _current_xor(value, value >> 16)


def _current_noise(x, y, seed, lattice_period=None):
    """Smooth value noise in [0, 1), with matching NumPy/Torch arithmetic.

    Hash lattice corners instead of storing or repeatedly shifting a texture:
    any tick can be sampled directly, independently of the ecology's RNG.
    Coordinates and seeds follow ordinary broadcasting rules.
    """
    if isinstance(x, torch.Tensor):
        ix, iy = torch.floor(x).to(torch.int64), torch.floor(y).to(torch.int64)
        fx, fy = x - ix.to(torch.float32), y - iy.to(torch.float32)
    else:
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x - ix.astype(np.float32), y - iy.astype(np.float32)
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def corner(dx, dy):
        cx, cy = ix + dx, iy + dy
        if lattice_period is not None:
            cx, cy = cx % lattice_period[0], cy % lattice_period[1]
        key = _current_xor(_current_hash(cx), _current_hash(cy + 0x9E3779B9))
        bits = _current_hash(_current_xor(key, seed)) >> 8
        value = bits.to(torch.float32) if isinstance(bits, torch.Tensor) else bits.astype(np.float32)
        return value * (1.0 / 16777216.0)

    lower = corner(0, 0) * (1 - sx) + corner(1, 0) * sx
    upper = corner(0, 1) * (1 - sx) + corner(1, 1) * sx
    return lower * (1 - sy) + upper * sy


def current_direction_fractions(tick, world_seed, config, x=0.0, y=0.0, world_shape=None):
    """N/E/S/W fractions for a cloud field scrolling towards east/south.

    ``period`` is ticks per cell of texture scrolling, independent of biomass
    transport strength. Two noise scales give broad clouds with finer detail.
    For batched worlds, pass seeds shaped [world, 1] and coordinates [cell].
    """
    scalar_sample = False
    if isinstance(tick, torch.Tensor):
        offset = tick.to(torch.float32) / config.period
    else:
        scalar_sample = np.ndim(x) == np.ndim(y) == np.ndim(world_seed) == 0
        offset = np.asarray(tick, dtype=np.float32) / np.float32(config.period)
        # NumPy 1.x promotes zero-dimensional float32 arithmetic with Python
        # scalars to float64. Use arrays to match device-side float32 sampling.
        x = np.array(x, dtype=np.float32, ndmin=1)
        y = np.array(y, dtype=np.float32, ndmin=1)
    period = None
    if world_shape is None:
        x, y = (x - offset) / config.scale, (y - offset) / config.scale
    else:
        # Fit a whole number of noise cells to each torus circumference.
        h, w = world_shape
        period = (max(1, round(w / config.scale)), max(1, round(h / config.scale)))
        x, y = (x - offset) * (period[0] / w), (y - offset) * (period[1] / h)
    seed = _current_xor(world_seed, config.seed)
    fine_period = None if period is None else (2 * period[0], 2 * period[1])
    cloud = (_current_noise(x, y, seed, period) * 0.75
             + _current_noise(2 * x, 2 * y, _current_xor(seed, 0xA341316C), fine_period) * 0.25)
    fraction = cloud * (config.strength * 0.5)
    if scalar_sample:
        fraction = fraction[0]
    zero = fraction * 0
    return zero, fraction, fraction, zero


def apply_currents(env):
    """Conservatively drift responding non-decision makers before growth."""
    config = env.currents
    if config is None or config.strength == 0:
        return
    ids = [fid for fid in env.global_fg_order
           if not env.fgs[fid].is_decision_maker and env.fgs[fid].current_response > 0
           and fid not in debug_food.selected_ids(env)]
    if not ids:
        return
    if getattr(env, "_current_coordinates", None) is None:
        env._current_coordinates = np.indices((env.H, env.W), dtype=np.float32)
    y, x = env._current_coordinates
    fractions = np.asarray(current_direction_fractions(
        env.tick_count, env.current_world_seed, config, x, y,
        (env.H, env.W) if env.boundary == "torus" else None), dtype=env.dtype)
    # Currents share swimmers' topology and habitat mask. Blocked habitat flow
    # stays at its source; never renormalize the remaining directions.
    fractions *= build_movement_mask(env.grid, env.migration, env.dtype, env.boundary)
    response = np.asarray([env.fgs[fid].current_response for fid in ids], dtype=env.dtype)
    fractions = fractions[None] * response[:, None, None, None]
    biomass = np.stack([env.fgs[fid].biomass for fid in ids])
    reserve = np.stack([env.fgs[fid].energy_reserve for fid in ids])
    out_b, out_r = biomass[:, None] * fractions, reserve[:, None] * fractions
    b, r = _transfer_to_neighbours(
        biomass - out_b.sum(1), reserve - out_r.sum(1), out_b, out_r, env.boundary)
    if env.migration:
        b_emig, r_emig = _edge_emigration(out_b, out_r)
        b_imm, r_imm = _concentrate_immigration(
            env, b_emig, r_emig, env._edge_imm_weights, group_ids=ids)
        b += b_imm
        r += r_imm
    for i, fid in enumerate(ids):
        env.fgs[fid].biomass = b[i].astype(env.dtype, copy=False)
        env.fgs[fid].energy_reserve = r[i].astype(env.dtype, copy=False)


def _transfer_to_neighbours(b_stay, r_stay, b_out, r_out, boundary="bounded"):
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
    if boundary == "torus":
        for total, out in ((b_total, b_out), (r_total, r_out)):
            total[:, -1, :] += out[:, NORTH, 0, :]
            total[:, :, 0] += out[:, EAST, :, -1]
            total[:, 0, :] += out[:, SOUTH, -1, :]
            total[:, :, -1] += out[:, WEST, :, 0]
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


def _concentrate_immigration(env, b_emig, r_emig, base_weights, group_ids=None):
    """Redistribute emigrants for swimmers or passive groups with the same rules."""
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

    for i, fid in enumerate(env.dm_ids if group_ids is None else group_ids):
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

    b_total, r_total = _transfer_to_neighbours(b_stay, r_stay, b_out, r_out, env.boundary)

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
    b_cancel = None
    if env._dm_split_thr_any:
        b_total, r_total, b_cancel, _r_cancel = suppress_subthreshold_splits(
            env, b_out, r_out, b_total, r_total, return_cancelled=True)

    # Local reward (``--local_reward``): hand the per-source-cell biomass
    # flow to the tracker while it is still available. Cancelled splits
    # are moved from the outflow back into the stay term so the flow
    # sums to ``b_total``, exactly as the biomass does.
    if source_tracking.is_enabled(env):
        if b_cancel is None:
            b_stay_eff, b_out_eff = b_stay, b_out
        else:
            b_stay_eff = b_stay + b_cancel.sum(axis=1)
            b_out_eff = b_out - b_cancel
        source_tracking.record_movement(env, b_stay_eff, b_out_eff, b_total)

    max_reserve = b_total * env.dm_max_energy_reserve[:, None, None]
    new_reserve = np.clip(r_total, 0.0, max_reserve)

    for i, fid in enumerate(env.dm_ids):
        env.fgs[fid].biomass = b_total[i].astype(env.dtype, copy=False)
        env.fgs[fid].energy_reserve = new_reserve[i].astype(
            env.dtype, copy=False)


def suppress_subthreshold_splits(env, b_out, r_out, b_total, r_total,
                                 return_cancelled=False):
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

    Returns the corrected ``(b_total, r_total)``, or
    ``(b_total, r_total, b_cancel, r_cancel)`` when ``return_cancelled``
    is set -- the local-reward source tracking needs the cancelled
    outflow to rebuild the effective per-cell biomass flow.
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

    if env.boundary == "torus":
        dest[:, NORTH, 0, :] = b_total[:, -1, :]
        dest[:, EAST, :, -1] = b_total[:, :, 0]
        dest[:, SOUTH, -1, :] = b_total[:, 0, :]
        dest[:, WEST, :, 0] = b_total[:, :, -1]
    blocked = (b_out > 0.0) & (dest < thr)
    if not np.any(blocked):
        if return_cancelled:
            zero = np.zeros_like(b_out)
            return b_total, r_total, zero, np.zeros_like(r_out)
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
    if env.boundary == "torus":
        for total, cancelled in ((b_total, b_cancel), (r_total, r_cancel)):
            total[:, -1, :] -= cancelled[:, NORTH, 0, :]
            total[:, :, 0] -= cancelled[:, EAST, :, -1]
            total[:, 0, :] -= cancelled[:, SOUTH, -1, :]
            total[:, :, -1] -= cancelled[:, WEST, :, 0]
    # float32 round-off on the +/- pair can leave tiny negatives.
    np.maximum(b_total, 0.0, out=b_total)
    np.maximum(r_total, 0.0, out=r_total)
    if return_cancelled:
        return b_total, r_total, b_cancel, r_cancel
    return b_total, r_total
