import numpy as np

from lib.environments.ecosystem_env import interactions
from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC


def apply_predation(env, actions):
    if env.N_dm == 0:
        return

    predator_biomass = np.stack(
        [env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    hunger = np.stack(
        [env.fgs[fid].get_hunger().astype(env.dtype, copy=False)
         for fid in env.dm_ids],
        axis=0,
    )
    prey_biomass = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order], axis=0)
    visible_biomass, pair_visible_biomass = interactions.visible_prey_biomass(
        env, prey_biomass, actions)

    intake_rate = env.max_intake_mat[:, :, None, None]
    if env._has_holling2 or env._has_holling3:
        handling_time = env.handling_time_mat[:, :, None, None]
        # Holling saturation uses the *visible* prey biomass: hidden prey
        # is functionally inaccessible this tick, so it neither
        # contributes to attack-rate saturation nor to total available
        # intake. Intake per predator saturates at high prey density;
        # specialists use a quadratic response, giving sparse prey a
        # Type III refuge.
        if pair_visible_biomass is not None:
            prey_visible = pair_visible_biomass
        else:
            prey_visible = visible_biomass[None, :, :, :]
        effective_intake_rate = interactions.holling_a_eff(
            intake_rate, handling_time, prey_visible, env._type3_pred_mask)
    else:
        effective_intake_rate = intake_rate

    demand = (
        predator_biomass[:, None, :, :]
        * actions.eat
        * effective_intake_rate
        * hunger[:, None, :, :]
    )

    if pair_visible_biomass is not None:
        # Per-predator bound: no predator may demand more than the
        # fraction of the prey IT can detect. Without this, a low-floor
        # predator could feed on the hidden fraction just because a
        # better-detecting predator raised the shared cap.
        demand = np.minimum(demand, pair_visible_biomass)

    total_demand = demand.sum(axis=0)
    # Demand is capped at the *visible* prey biomass: the hidden
    # (rested) fraction is protected entirely this tick.
    #
    # Fix 2 (Section 73): the cap is RELATIVE (MAX_HARVEST_FRAC) rather
    # than an absolute 1e-9 offset in the denominator. See the comment on
    # the constant: the absolute form left a residue below float32
    # precision, so an overharvested cell went to exactly 0.0 - an
    # absorbing state no growth or refuge mechanism can escape.
    harvestable_biomass = visible_biomass * MAX_HARVEST_FRAC
    # Guard the division only; the branch itself decides when it runs, so
    # a tiny floor here cannot leak into the total_demand == 0 case.
    denominator = np.maximum(total_demand, np.float32(1e-30))
    scale = np.where(
        total_demand > harvestable_biomass,
        harvestable_biomass / denominator,
        np.float32(1.0),
    ).astype(env.dtype, copy=False)
    actual_intake = demand * scale[None, :, :, :]

    gains = (
        actual_intake * env.energy_gain_mat[:, :, None, None]
    ).sum(axis=1)
    for i, fid in enumerate(env.dm_ids):
        env.fgs[fid].temp_energy_gains = gains[i]

    intake_by_prey = actual_intake.sum(axis=0)
    intake_by_pred_prey = actual_intake.sum(axis=(2, 3))
    for i, pred_id in enumerate(env.dm_ids):
        row = env.intake_by_pred_prey.setdefault(pred_id, {})
        for j, prey_id in enumerate(env.global_fg_order):
            value = float(intake_by_pred_prey[i, j])
            if value != 0.0:
                row[prey_id] = float(row.get(prey_id, 0.0)) + value

    eps = np.float32(1e-9)
    for j, prey_id in enumerate(env.global_fg_order):
        prey_fg = env.fgs[prey_id]
        biomass_old = prey_fg.biomass
        intake = intake_by_prey[j]
        env.loss_predation[prey_id] = (
            float(env.loss_predation.get(prey_id, 0.0))
            + float(intake.sum())
        )
        reduction = np.where(
            biomass_old > eps,
            (biomass_old - intake) / (biomass_old + eps),
            np.float32(0.0),
        )
        prey_fg.energy_reserve = (prey_fg.energy_reserve * reduction).astype(
            env.dtype, copy=False)
        prey_fg.biomass = (biomass_old - intake).astype(
            env.dtype, copy=False)
