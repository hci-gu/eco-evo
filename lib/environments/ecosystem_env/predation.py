import numpy as np

from lib.environments.ecosystem_env import interactions, refuge
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
    if env._has_reserve_food:
        reserve_offset = np.zeros_like(prey_biomass)
        for j in np.flatnonzero(env.prey_reserve_food):
            fg = env.fgs[env.global_fg_order[j]]
            density = np.where(
                fg.biomass > 1e-9,
                fg.energy_reserve / np.maximum(fg.biomass, np.float32(1e-9)),
                np.float32(0.0))
            reserve_offset[j] = (density - env.prey_reserve_fill[j]
                                 * env.prey_reserve_max[j])
    # Section 223.12: a refuge group is seen and eaten through its
    # accessible part only.
    visible_biomass, pair_visible_biomass = interactions.visible_prey_biomass(
        env, refuge.accessible_rows(env, prey_biomass), actions)
    visible_biomass, pair_visible_biomass = interactions.apply_pair_access(
        env, visible_biomass, pair_visible_biomass)
    # Juvenile compartments (stages.py): a staged prey is seen per pair as
    # f_J J + f_A (B - J) of its visible share; the parts are kept for the
    # per-compartment cap below.
    stage_parts = None
    # The per-predator bound below exists only with per-pair visibility;
    # the stage split's own per-pair tensor must not switch it on.
    per_pair_bound = pair_visible_biomass is not None
    if getattr(env, "_has_stages", False):
        visible_biomass, pair_visible_biomass, stage_parts = _stage_visible(
            env, actions, visible_biomass, pair_visible_biomass)

    # Beddington-DeAngelis interference: I = w_X * B_X(c), the predator's
    # OWN biomass in the cell. Intraspecific only - it is the standard
    # form and it keeps the predation tensor's rows independent. Shape
    # (N_dm, 1, H, W) so it broadcasts over the prey axis. Section 86.
    if env._has_interference:
        interference = env.dm_interference * predator_biomass[:, None, :, :]
    else:
        interference = np.float32(0.0)

    # Attack rate a of this tick: the library value, times the pair's
    # light multiplier under the daylight calendar (section 137).
    intake_rate = interactions.attack_rate(env)[:, :, None, None]
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
            intake_rate, handling_time, prey_visible, env._type3_pred_mask,
            interference)
    elif env._has_interference:
        # Linear (unsaturated) response: interference still divides the
        # per-predator rate, so the limit h -> 0 stays continuous.
        effective_intake_rate = intake_rate / (np.float32(1.0) + interference)
    else:
        effective_intake_rate = intake_rate

    # Section 181: the water temperature scales the whole response (a and
    # the ceiling 1/h together) of a DM with ``ingestion_q10``.
    intake_temperature = interactions.ingestion_temperature(env)
    if intake_temperature is not None:
        effective_intake_rate = (
            effective_intake_rate
            * intake_temperature[:, None, None, None]).astype(
                env.dtype, copy=False)

    eat = actions.eat
    demand = (
        predator_biomass[:, None, :, :]
        * eat
        * effective_intake_rate
        * hunger[:, None, :, :]
    )

    if per_pair_bound:
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
    juvenile_intake = {}
    if stage_parts is not None:
        actual_intake, juvenile_intake = _stage_harvest(
            env, demand, actual_intake, stage_parts)

    gains = (
        actual_intake * env.energy_gain_mat[:, :, None, None]
    ).sum(axis=1)
    if env._has_reserve_food:
        # The eaten tonnes carry the prey's reserve (section 140):
        # correct the static quality (evaluated at the reference fill)
        # to the reserve per tonne in the cell before predation.
        gains = gains + (
            actual_intake * env.assimilation_mat[:, :, None, None]
            * reserve_offset[None, :, :, :]
        ).sum(axis=1)
    for i, fid in enumerate(env.dm_ids):
        env.fgs[fid].temp_energy_gains = gains[i]

    intake_by_prey = actual_intake.sum(axis=0)
    intake_by_pred_prey = actual_intake.sum(axis=(2, 3))
    if juvenile_intake:
        # Diagnostics: the juvenile part of each pair's take (stages.py).
        book = env.__dict__.setdefault("intake_juvenile_by_pred_prey", {})
        for j, per_pred in juvenile_intake.items():
            prey_id = env.global_fg_order[j]
            sums = per_pred.sum(axis=(1, 2))
            for i, pred_id in enumerate(env.dm_ids):
                if sums[i] != 0.0:
                    row = book.setdefault(pred_id, {})
                    row[prey_id] = float(row.get(prey_id, 0.0)) + float(sums[i])
    # Section 202: this tick's intake per pair, for the layer split of
    # the regenerated M1.
    env._tick_dm_intake = intake_by_pred_prey
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
        if refuge.settings(prey_fg) is not None:
            # Section 223.12: predation removes from the accessible part.
            prey_fg.energy_reserve = np.clip(
                prey_fg.energy_reserve - intake, 0.0, None).astype(
                    env.dtype, copy=False)
        elif interactions.is_nutrient_producer(env, prey_id):
            # Section 172: the reserve slot is the producer's nutrient
            # pool, which grazing does not eat; a fraction of what it
            # removes is excreted back into it.
            prey_fg.energy_reserve = (
                prey_fg.energy_reserve
                + np.float32(env.nutrient_regeneration[prey_id]) * intake
            ).astype(env.dtype, copy=False)
        else:
            prey_fg.energy_reserve = (prey_fg.energy_reserve * reduction).astype(
                env.dtype, copy=False)
        prey_fg.biomass = (biomass_old - intake).astype(
            env.dtype, copy=False)
        if j in juvenile_intake:
            prey_fg.juvenile = np.clip(
                prey_fg.juvenile - juvenile_intake[j].sum(axis=0), 0.0,
                np.maximum(prey_fg.biomass, 0.0)).astype(env.dtype, copy=False)


def _stage_visible(env, actions, visible, pair_visible):
    """Per-pair visible prey with the staged columns split into parts.

    Returns ``(visible, pair_visible, parts)``; ``pair_visible`` is always
    the (N_dm, N_all, H, W) tensor, ``parts[j] = (vis_J, vis_A)`` each
    (N_dm, H, W): the share of the juvenile and of the adult tonnes pair i
    sees (visible fraction x f_J or f_A). With f_J = f_A = the pair's
    ``accessible_fraction`` the sum is exactly the unstaged pair.
    """
    if pair_visible is None:
        pair_visible = np.repeat(visible[None, :, :, :], env.N_dm, axis=0)
    else:
        pair_visible = pair_visible.copy()
    visible = visible.copy()
    parts = {}
    for j in env.stage_rows:
        j = int(j)
        fg = env.fgs[env.global_fg_order[j]]
        dm = env.dm_ids.index(env.global_fg_order[j])
        hidden = actions.rest[dm].astype(env.dtype, copy=False)
        if env._has_pair_vis_floor:
            floor = env.vis_floor_mat[:, j][:, None, None]
        else:
            floor = np.full((env.N_dm, 1, 1), env._all_visibility_floor[j], dtype=env.dtype)
        frac = np.float32(1.0) - hidden[None, :, :] * (np.float32(1.0) - floor)
        juv = np.clip(fg.juvenile, 0.0, fg.biomass)
        adult = np.maximum(fg.biomass - juv, np.float32(0.0))
        vis_j = (frac * env.stage_f_j[:, j][:, None, None] * juv[None]).astype(env.dtype)
        vis_a = (frac * env.stage_f_a[:, j][:, None, None] * adult[None]).astype(env.dtype)
        pair_visible[:, j] = vis_j + vis_a
        parts[j] = (vis_j, vis_a)
        eats = env.eat_static_mask[:, j][:, None, None] > 0.0
        visible[j] = np.where(eats, vis_j + vis_a, np.float32(0.0)).max(axis=0)
    return visible, pair_visible, parts


def _stage_harvest(env, demand, actual_intake, parts):
    """The harvest cap per compartment of every staged prey column.

    Each predator's demand on the column splits in the ratio of the parts
    it sees; the juvenile and the adult part are each capped at
    MAX_HARVEST_FRAC of the most that any of its predators sees of it.
    Returns the corrected ``actual_intake`` and {column: (N_dm, H, W)
    juvenile intake}.
    """
    actual_intake = actual_intake.copy()
    juvenile = {}
    for j, (vis_j, vis_a) in parts.items():
        d = demand[:, j]
        seen = vis_j + vis_a
        share = np.where(seen > 0.0, vis_j / np.where(seen > 0.0, seen, np.float32(1.0)),
                         np.float32(0.0))
        d_j = d * share
        d_a = d - d_j
        eats = env.eat_static_mask[:, j][:, None, None] > 0.0
        out = []
        for part, vis in ((d_j, vis_j), (d_a, vis_a)):
            cap = np.where(eats, vis, np.float32(0.0)).max(axis=0) * MAX_HARVEST_FRAC
            total = part.sum(axis=0)
            scale = np.where(total > cap, cap / np.maximum(total, np.float32(1e-30)),
                             np.float32(1.0)).astype(env.dtype, copy=False)
            out.append(part * scale[None])
        actual_intake[:, j] = out[0] + out[1]
        juvenile[j] = out[0]
    return actual_intake, juvenile
