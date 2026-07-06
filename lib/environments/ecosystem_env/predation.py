import numpy as np

from lib.environments.ecosystem_env import interactions


def apply_predation(env, actions):
    predator_biomass = np.stack(
        [env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    hunger = np.stack(
        [env.fgs[fid].get_hunger().astype(env.dtype, copy=False)
         for fid in env.dm_ids],
        axis=0,
    )
    prey_biomass = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order], axis=0)
    visible_biomass = interactions.visible_biomass_from_hide(
        env, prey_biomass, actions)

    intake_rate = env.max_intake_mat[:, :, None, None]
    if getattr(env, "_has_holling2", False):
        handling_time = env.handling_time_mat[:, :, None, None]
        effective_intake_rate = (
            intake_rate
            / (1.0 + intake_rate * handling_time * visible_biomass[None, :, :, :])
        )
    else:
        effective_intake_rate = intake_rate

    demand = (
        predator_biomass[:, None, :, :]
        * actions.eat
        * effective_intake_rate
        * hunger[:, None, :, :]
    )

    total_demand = demand.sum(axis=0)
    scale = np.where(
        total_demand > visible_biomass,
        visible_biomass / (total_demand + np.float32(1e-9)),
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
