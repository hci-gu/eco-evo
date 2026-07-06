import numpy as np


def extract_impact_table(impact_def):
    if not impact_def or not impact_def.get("impact_affects", False):
        return None

    table = impact_def.get("impact_table") or []
    if not table:
        return None

    try:
        rows = sorted(
            (
                (
                    float(row["value"]),
                    float(row["biomass_factor"]),
                    float(row["energy_factor"]),
                )
                for row in table
            ),
            key=lambda item: item[0],
        )
    except (KeyError, TypeError, ValueError):
        return None

    if not rows:
        return None

    xs = np.array([row[0] for row in rows], dtype=np.float32)
    biomass_factors = np.array([row[1] for row in rows], dtype=np.float32)
    energy_factors = np.array([row[2] for row in rows], dtype=np.float32)
    return xs, biomass_factors, energy_factors


def interp_impact(table, x):
    xs, biomass_factors, energy_factors = table
    x_arr = np.asarray(x, dtype=np.float32)
    biomass = np.interp(
        x_arr, xs, biomass_factors,
        left=biomass_factors[0], right=biomass_factors[-1],
    )
    energy = np.interp(
        x_arr, xs, energy_factors,
        left=energy_factors[0], right=energy_factors[-1],
    )
    return (
        biomass.astype(np.float32, copy=False),
        energy.astype(np.float32, copy=False),
    )


def compute_impact_mortality(env, fg):
    if "impact" not in fg.params:
        return None

    total_mortality = None
    for impact_id, impact_def in fg.params["impact"].items():
        map_data = env.grid.get_map(impact_id)
        if map_data is None:
            continue
        table = extract_impact_table(impact_def)
        if table is None:
            continue

        biomass_factor, _ = interp_impact(table, map_data)
        contribution = fg.biomass * biomass_factor
        if total_mortality is None:
            total_mortality = contribution
        else:
            total_mortality = total_mortality + contribution
    return total_mortality


def apply_impact_mortality(env):
    for fg_id in env.ordered_fg_ids:
        fg = env.fgs[fg_id]
        if not fg.is_decision_maker:
            continue

        total_mortality = compute_impact_mortality(env, fg)
        if total_mortality is None:
            continue

        loss_mask = total_mortality > 0
        reduction = np.ones_like(fg.biomass)
        reduction[loss_mask] = (
            (fg.biomass[loss_mask] - total_mortality[loss_mask])
            / (fg.biomass[loss_mask] + 1e-9)
        )
        reduction = np.clip(reduction, 0.0, 1.0)

        actual_loss = np.minimum(total_mortality, fg.biomass)
        env.loss_impact[fg_id] = (
            float(env.loss_impact.get(fg_id, 0.0)) + float(actual_loss.sum())
        )

        fg.energy_reserve = (fg.energy_reserve * reduction).astype(
            env.dtype, copy=False)
        fg.biomass = np.maximum(0.0, fg.biomass - total_mortality).astype(
            env.dtype, copy=False)
