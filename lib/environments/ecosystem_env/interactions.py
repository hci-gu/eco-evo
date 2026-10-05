from dataclasses import dataclass

import numpy as np

from lib.world import daylight


@dataclass
class InteractionMatrices:
    eat_static: np.ndarray
    max_intake: np.ndarray
    energy_gain: np.ndarray
    handling_time: np.ndarray
    vis_floor_over: np.ndarray
    vis_floor_has: np.ndarray
    assimilation: np.ndarray = None


def reserve_food_settings(fg):
    """(max reserve, reference fill) if ``fg``'s reserve is eaten too.

    Section 140: with ``prey_includes_reserve`` a predator gains the
    prey's energy_content PLUS the energy reserve each eaten tonne
    carries (fat autumn herring is better food than spent spring
    herring). ``reserve_reference_fill`` (default 0.5) is the fill at
    which the static quality ``energy_gain_mat`` - used by the energy
    gates, budget_gate and the viability rig - is evaluated. None for
    every other FG, and always for non-decision makers (no reserve).
    """
    params = fg.params
    if not fg.is_decision_maker or not params.get("prey_includes_reserve",
                                                  False):
        return None
    max_reserve = float(getattr(fg, "max_energy_reserve", 0.0) or 0.0)
    fill = _float_or_default(params.get("reserve_reference_fill", None), 0.5)
    if not 0.0 <= fill <= 1.0:
        raise ValueError(f"reserve_reference_fill must be in [0, 1], got {fill}")
    return max_reserve, fill


def _float_or_default(value, default):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def build_interaction_matrices(env):
    eat_static = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    max_intake = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    energy_gain = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    assimilation_mat = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
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
            # Attack rate a of the Holling response. The species value is
            # the default; a pair may override it (section 134), which is
            # how a prey-specific half-saturation 1/(a*h) is set without
            # moving the ceiling 1/h. tick_time already rescales the pair
            # key (INTERACTION_RESCALE_RULES).
            max_intake[i, j] = _float_or_default(
                inter_def.get("max_intake_rate", None), pred_max_intake)

            assimilation = np.clip(
                _float_or_default(inter_def.get("assimilation_factor", 1.0), 1.0),
                0.0,
                1.0,
            )
            prey_energy_content = _float_or_default(
                env.fgs[prey_id].params.get("energy_content", 0.0), 0.0)
            reserve_food = reserve_food_settings(env.fgs[prey_id])
            if reserve_food is not None:
                # Quality at the reference fill; the tick adds the
                # difference to the cell's actual reserve (section 140).
                max_reserve, fill = reserve_food
                prey_energy_content += fill * max_reserve
            energy_gain[i, j] = prey_energy_content * assimilation
            assimilation_mat[i, j] = assimilation
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
        assimilation=assimilation_mat,
    )


def build_daylight_tables(env):
    """Light tables for the daylight calendar (section 137).

    Sets ``env._has_daylight`` (the light observation channel exists),
    ``env._has_light_pairs`` (at least one pair's attack rate is
    modulated), ``env.light_obs_table`` (F per year tick, the value the
    policies observe) and ``env.light_mult_table`` ((T, N_dm, N_all)
    attack-rate multiplier, exactly 1 for every unmodulated pair, or
    None when no pair is modulated). Everything is off - and the tick
    bit-identical to before - unless the loader put a
    ``params['daylight']`` on the FGs.

    Also the light-limited growth of non-decision makers (section 138):
    ``env._has_growth_light`` and ``env.growth_light`` = {fg_id: (T,)
    growth multiplier} for every NDM with ``light_saturation``.
    """
    cfg = None
    for fid in env.global_fg_order:
        cfg = env.fgs[fid].params.get("daylight")
        if cfg:
            break
    env.daylight = dict(cfg) if cfg else None
    env._has_daylight = env.daylight is not None and env.N_dm > 0
    env._has_light_pairs = False
    env._has_growth_light = False
    env.growth_light = {}
    env.light_obs_table = None
    env.light_mult_table = None
    if env.daylight is None:
        return
    latitude = float(env.daylight["latitude_deg"])
    tick_hours = int(env.daylight["tick_hours"])
    env.light_period = daylight.ticks_per_year(tick_hours)
    env.light_start = int(env.daylight.get("start_tick", 0)) % env.light_period

    for fid in env.global_fg_order:
        fg = env.fgs[fid]
        if fg.is_decision_maker:
            continue
        settings = daylight.growth_light_settings(fg.params)
        if settings is None:
            continue
        climate = env.daylight.get("light_climate")
        if not climate:
            raise ValueError(
                f"'{fid}' has light_saturation (light-limited growth) but "
                "the project's simulation_settings.daylight has no light "
                "climate (light_attenuation_per_m, mixed_layer_depth_m, "
                "cloud_transmission); section 138")
        ik, reference_day = settings
        env.growth_light[fid] = daylight.growth_light_schedule(
            latitude, tick_hours, climate, ik, reference_day)
    env._has_growth_light = bool(env.growth_light)

    if not env._has_daylight:
        return
    env.light_obs_table = daylight.light_schedule(
        latitude, tick_hours).astype(env.dtype)

    table = np.ones((env.light_period, env.N_dm, env.N_all), dtype=env.dtype)
    for i, pred_id in enumerate(env.dm_ids):
        interaction = env.fgs[pred_id].params.get("interaction", {})
        for j, prey_id in enumerate(env.global_fg_order):
            if env.eat_static_mask[i, j] <= 0.0:
                continue
            pair = daylight.pair_settings(
                interaction.get(f"{pred_id}_preys_on_{prey_id}"))
            if pair is None:
                continue
            dark_ratio, threshold = pair
            table[:, i, j] = daylight.attack_multiplier(
                latitude, tick_hours, dark_ratio, threshold)
            env._has_light_pairs = True
    if env._has_light_pairs:
        env.light_mult_table = table


def light_index(env):
    """Year tick of the current env tick (valid when ``_has_daylight``)."""
    return (env.light_start + int(env.tick_count)) % env.light_period


def growth_light(env, fg_id):
    """Light factor on ``fg_id``'s growth this tick (1.0 when unlimited)."""
    table = env.growth_light.get(fg_id) if env._has_growth_light else None
    if table is None:
        return 1.0
    return float(table[light_index(env)])


def light_level(env):
    """F for the current tick: the light observation channel's value."""
    return env.light_obs_table[light_index(env)]


def attack_rate(env):
    """``(N_dm, N_all)`` attack rate a for THIS tick.

    The library's (tick-rescaled) ``max_intake_mat`` times the pair's
    light multiplier when the daylight calendar modulates any pair;
    otherwise ``max_intake_mat`` itself, untouched.
    """
    if not env._has_light_pairs:
        return env.max_intake_mat
    return env.max_intake_mat * env.light_mult_table[light_index(env)]


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
