import numpy as np


def extract_impact_table(impact_def):
    """Return (xs, biomass_factors, energy_factors) as float32 arrays sorted
    by xs, or None if the FG is not affected by this impact or the table is
    missing/empty."""
    if not impact_def:
        return None
    if not impact_def.get("impact_affects", False):
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
            key=lambda t: t[0],
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not rows:
        return None
    xs = np.array([r[0] for r in rows], dtype=np.float32)
    bf = np.array([r[1] for r in rows], dtype=np.float32)
    ef = np.array([r[2] for r in rows], dtype=np.float32)
    return xs, bf, ef


def interp_impact(table, x):
    """Linear interpolation against an impact table. Outside the support,
    the nearest endpoint value is used (no extrapolation). `x` may be a
    scalar or ndarray; returns (biomass_factor, energy_factor) with the
    same shape as `x` (or scalars if `x` is scalar)."""
    xs, bf, ef = table
    x_arr = np.asarray(x, dtype=np.float32)
    b = np.interp(x_arr, xs, bf, left=bf[0], right=bf[-1])
    e = np.interp(x_arr, xs, ef, left=ef[0], right=ef[-1])
    return (
        b.astype(np.float32, copy=False),
        e.astype(np.float32, copy=False),
    )


def build_dm_impact_tables(env):
    """Per-DM cached impact tables for ALL impacts the FG is affected by.

    Each entry is a list of (impact_id, (xs, bf, ef)) tuples; impacts
    without ``impact_affects=true`` or with an empty/invalid table are
    omitted. Used by both ``movement.apply_energy_costs`` (energy_factor
    -> extra metabolic cost) and ``apply_impact_mortality``
    (biomass_factor -> mortality, energy_factor -> reserve loss).
    """
    dm_impact_tables = []
    for fid in env.dm_ids:
        impacts = env.fgs[fid].params.get("impact", {}) or {}
        entries = []
        for impact_id, impact_def in impacts.items():
            table = extract_impact_table(impact_def)
            if table is not None:
                entries.append((impact_id, table))
        dm_impact_tables.append(entries)
    return dm_impact_tables


def impact_energy_cost_factor(env):
    """Per-DM metabolic cost multiplier (1 + sum of energy_factor lookups).

    Shape (N_dm, H, W). The factor scales the resting / feeding / movement
    metabolic costs of every decision maker that sits in an impacted cell.
    """
    impact_energy = np.zeros((env.N_dm, env.H, env.W), dtype=env.dtype)
    for i, entries in enumerate(env.dm_impact_tables):
        for impact_id, table in entries:
            map_data = env.grid.get_map(impact_id)
            if map_data is None:
                continue
            _, ef_map = interp_impact(
                table, map_data.astype(env.dtype, copy=False))
            impact_energy[i] += ef_map
    return (np.float32(1.0) + impact_energy).astype(env.dtype, copy=False)


def observable_impact_layers(env):
    """The observable impact maps, in ``observable_impact_vars`` order.

    Missing maps are substituted with zeros so the observation layout
    keeps a fixed width regardless of which maps the project spawned.
    """
    layers = []
    for impact_id in env.observable_impact_vars:
        map_data = env.grid.get_map(impact_id)
        if map_data is None:
            map_data = np.zeros((env.H, env.W), dtype=env.dtype)
        else:
            map_data = map_data.astype(env.dtype, copy=False)
        layers.append(map_data)
    return layers


def compute_impact_mortality(env, fg):
    """Return total impact-induced biomass loss m_X^Impact for the given FG
    as a per-cell array, or None if the FG has no active impact tables."""
    if "impact" not in fg.params:
        return None
    total_mortality_impact = None
    for impact_id, impact_def in fg.params["impact"].items():
        map_data = env.grid.get_map(impact_id)
        if map_data is None:
            continue
        table = extract_impact_table(impact_def)
        if table is None:
            continue
        bf_map, _ = interp_impact(table, map_data)
        contribution = fg.biomass * bf_map
        if total_mortality_impact is None:
            total_mortality_impact = contribution
        else:
            total_mortality_impact = total_mortality_impact + contribution
    return total_mortality_impact


def apply_impact_mortality(env):
    """Method.pdf steg 1-6: apply m_X^Impact before predation. Only
    decision-maker FGs carry impact tables in the current model."""
    for fg_id in env.ordered_fg_ids:
        fg = env.fgs[fg_id]
        if not fg.is_decision_maker:
            continue
        total_mortality_impact = compute_impact_mortality(env, fg)
        if total_mortality_impact is None:
            continue

        loss_mask = total_mortality_impact > 0
        reduction = np.ones_like(fg.biomass)
        reduction[loss_mask] = (
            (fg.biomass[loss_mask] - total_mortality_impact[loss_mask])
            / (fg.biomass[loss_mask] + 1e-9)
        )
        reduction = np.clip(reduction, 0.0, 1.0)

        # The actual biomass loss is min(demanded mortality, standing
        # biomass) because the biomass update below clamps at 0 the same
        # way. Tracked for the probe report in train.py.
        actual_impact_loss = np.minimum(total_mortality_impact, fg.biomass)
        env.loss_impact[fg_id] = (
            float(env.loss_impact.get(fg_id, 0.0))
            + float(actual_impact_loss.sum())
        )

        fg.energy_reserve = (fg.energy_reserve * reduction).astype(
            env.dtype, copy=False)
        fg.biomass = np.maximum(
            0.0, fg.biomass - total_mortality_impact
        ).astype(env.dtype, copy=False)
