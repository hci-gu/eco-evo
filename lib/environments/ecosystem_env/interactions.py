from dataclasses import dataclass

import numpy as np


@dataclass
class InteractionMatrices:
    eat_static: np.ndarray
    max_intake: np.ndarray
    energy_gain: np.ndarray
    handling_time: np.ndarray
    vis_floor_over: np.ndarray
    vis_floor_has: np.ndarray


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
    # Optional per-(predator, prey) visibility floor override. Empty /
    # missing cells inherit the prey FG's own ``visibility_floor``
    # (see the vis_floor_mat assembly in ``state.build_static_caches``,
    # after ``_all_visibility_floor`` has been built). Rationale:
    # detection is a property of the PAIR, not of the prey alone -
    # porpoises use biosonar and are unaffected by the visual crypsis
    # that protects herring from seabirds (Section 71.12).
    vis_floor_over = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    vis_floor_has = np.zeros((env.N_dm, env.N_all), dtype=bool)

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

            vis_floor = _float_or_default(
                inter_def.get("visibility_floor", None), None)
            if vis_floor is not None:
                vis_floor_over[i, j] = np.clip(vis_floor, 0.0, 1.0)
                vis_floor_has[i, j] = True

    return InteractionMatrices(
        eat_static=eat_static,
        max_intake=max_intake,
        energy_gain=energy_gain,
        handling_time=handling_time,
        vis_floor_over=vis_floor_over,
        vis_floor_has=vis_floor_has,
    )


def holling_a_eff(a, h, Bp, m3, interference=0.0):
    """Effective per-predator-unit intake rate f(B_prey) [ton_prey per
    ton_pred per tick] for the Holling Type II / Type III mixture.

        Type II:  f(B) = a*B  / (1 + a*h*B  + I)
        Type III: f(B) = a*B^2 / (1 + a*h*B^2 + I)
        f = m3 * f_III + (1 - m3) * f_II

    ``m3`` is the per-predator Type III mask (1.0 = Type III,
    0.0 = Type II) and must broadcast against ``Bp``.

    ``interference`` is the Beddington-DeAngelis term I = w_X * B_X(c),
    i.e. the predator's own biomass in the cell scaled by its
    ``interference`` parameter (Section 86). It must broadcast against
    ``Bp``. At I = 0 this reduces bit-identically to the pure Holling
    response, which is why ``interference`` defaults to 0.

    Contract (locked by tests/test_holling_response.py): for h > 0 the
    result is zero at B=0, strictly increasing in B and bounded by the
    physiological ceiling 1/h. Interference only lowers the curve, it
    never changes that shape. Do NOT drop the B factor in the numerator
    - see the BUGGFIX note in ``predation.apply_predation``.
    """
    Bp2 = Bp * Bp
    a_eff_ii = (a * Bp) / (1.0 + a * h * Bp + interference)
    a_eff_iii = (a * Bp2) / (1.0 + a * h * Bp2 + interference)
    return m3 * a_eff_iii + (1.0 - m3) * a_eff_ii


def visible_prey_biomass(env, prey_biomass, actions):
    """Visible prey biomass for the current tick's hide choices.

    ``Rest`` is a hide action: the fraction of each DM-prey that chose
    rest THIS tick is protected from predation. The floor acts ON the
    hiding fraction itself, so a population fraction pi_rest that
    chooses to hide is only (1 - floor) effectively hidden, while
    floor * pi_rest stays visible. Hence
        visible_frac = 1 - pi_rest * (1 - floor)
                     = (1 - pi_rest) + pi_rest * floor
    which means the floor is active for ANY pi_rest > 0, not only at
    saturation (pi_rest -> 1). At floor=0 this reduces to the legacy
    1 - pi_rest; at floor=1 hide is fully disabled.

    Returns ``(B_prey_visible, B_prey_pair_visible)``. The second entry
    is the (N_dm, N_all, H, W) per-pair tensor when an interaction
    carries an explicit ``visibility_floor`` (detection ability differs
    between predators: biosonar vs vision, so the same hidden herring is
    not equally cryptic to porpoises and seabirds), otherwise None and
    the cheaper legacy vector path is taken.
    """
    hidden_frac_now = np.zeros((env.N_all, env.H, env.W), dtype=env.dtype)
    for i, _fid in enumerate(env.dm_ids):
        j = int(env.dm_index_in_all[i])
        hidden_frac_now[j] = actions.rest[i].astype(env.dtype, copy=False)

    if not env._has_pair_vis_floor:
        one_minus_floor = (np.float32(1.0)
                           - env._all_visibility_floor[:, None, None])
        visible_frac = np.float32(1.0) - hidden_frac_now * one_minus_floor
        return prey_biomass * visible_frac, None

    one_minus_floor = np.float32(1.0) - env.vis_floor_mat[:, :, None, None]
    visible_frac = (np.float32(1.0)
                    - hidden_frac_now[None, :, :, :] * one_minus_floor)
    pair_visible = prey_biomass[None, :, :, :] * visible_frac
    # Shared availability cap: the total removal from prey j is bounded
    # by what the BEST-detecting predator of j can see. Rows that do not
    # prey on j are excluded so they cannot inflate the cap. Reduces
    # exactly to the legacy vector when all rows share the column
    # default.
    shared_visible = np.where(
        env.eat_static_mask[:, :, None, None] > 0.0,
        pair_visible,
        np.float32(0.0),
    ).max(axis=0)
    return shared_visible, pair_visible
