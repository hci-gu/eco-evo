import numpy as np


def _float_or_default(value, default):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def build_interaction_matrices(env):
    eat_static = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    max_intake = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    energy_gain = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    handling_time = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)

    for i, pred_id in enumerate(env.dm_ids):
        pred_fg = env.fgs[pred_id]
        menu = pred_fg.params.get("menu", [])
        interaction = pred_fg.params.get("interaction", {})
        pred_max_intake = float(pred_fg.params.get("max_intake_rate", 0.0))

        for j, prey_id in enumerate(env.global_fg_order):
            if prey_id not in menu:
                continue

            inter_id = f"{pred_id}_preys_on_{prey_id}"
            inter_def = interaction.get(inter_id, {})
            if not inter_def.get("preys_on", True):
                continue

            eat_static[i, j] = 1.0
            max_intake[i, j] = pred_max_intake

            assimilation = np.clip(
                _float_or_default(inter_def.get("assimilation_factor", 1.0), 1.0),
                0.0,
                1.0,
            )
            prey_energy_content = _float_or_default(
                env.fgs[prey_id].params.get("energy_content", 0.0), 0.0)
            energy_gain[i, j] = prey_energy_content * assimilation
            handling_time[i, j] = float(inter_def.get("handling_time", 0.0))

    return eat_static, max_intake, energy_gain, handling_time


def visible_biomass_from_hide(env, prey_biomass, actions):
    hidden_frac = np.zeros((env.N_all, env.H, env.W), dtype=env.dtype)
    visibility_floor = np.zeros(env.N_all, dtype=env.dtype)

    for i, _fid in enumerate(env.dm_ids):
        j = int(env.dm_index_in_all[i])
        hidden_frac[j] = actions.rest[i].astype(env.dtype, copy=False)
        visibility_floor[j] = env.dm_visibility_floor[i]

    visible_fraction = (
        np.float32(1.0)
        - hidden_frac * (np.float32(1.0) - visibility_floor[:, None, None])
    )
    return prey_biomass * visible_fraction
