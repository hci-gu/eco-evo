import math
from dataclasses import dataclass

import numpy as np

from lib.world import daylight, energy_balance, nutrients, temperature


@dataclass
class InteractionMatrices:
    eat_static: np.ndarray
    max_intake: np.ndarray
    energy_gain: np.ndarray
    handling_time: np.ndarray
    vis_floor_over: np.ndarray
    vis_floor_has: np.ndarray
    assimilation: np.ndarray = None
    # Per-pair Holling type override (section 141): 0 = automatic (III
    # for a specialist, II otherwise), 2 or 3 = forced by the pair's
    # ``functional_response``.
    response_type: np.ndarray = None
    # Section 208: the share of the prey a predator can reach
    # (``accessible_fraction``, 1 = all of it), and the share of the
    # predator's eat mass that can go to the prey (``predator_share``).
    access: np.ndarray = None
    share_cap: np.ndarray = None


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
    spec = dict(fg.params)
    spec["is_decision_maker"] = fg.is_decision_maker
    spec["max_energy_reserve"] = getattr(fg, "max_energy_reserve", 0.0)
    return energy_balance.reserve_food(spec)


def _float_or_default(value, default):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def pair_share(inter_def, key, inter_id=""):
    """A pair's share in (0, 1] (``predator_share``,
    ``accessible_fraction``; section 208), 1.0 when unset."""
    raw = (inter_def or {}).get(key)
    if raw in (None, ""):
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{inter_id}: {key} must be in (0, 1], got {raw!r}")
    return value


def build_interaction_matrices(env):
    eat_static = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    max_intake = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    energy_gain = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    assimilation_mat = np.zeros((env.N_dm, env.N_all), dtype=env.dtype)
    response_type = np.zeros((env.N_dm, env.N_all), dtype=np.int8)
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
    access = np.ones((env.N_dm, env.N_all), dtype=env.dtype)
    share_cap = np.ones((env.N_dm, env.N_all), dtype=env.dtype)

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
            # Section 208: ``predator_share`` phi = the share of the
            # predator group's biomass that can feed on this prey at all
            # (the benthivores of the pooled gadoids). That part eats it at
            # the full a and 1/h; the rest cannot, so at most phi of the
            # group's eat mass reaches the prey: the tick uses
            # min(eat_j, phi_j). (A mean field a phi, h / phi made every
            # fish a half-benthivore, and an energy maximiser dropped the
            # prey altogether.)
            share_cap[i, j] = pair_share(inter_def, "predator_share", inter_id)
            # ``accessible_fraction`` f: the share of the prey within the
            # predator's reach this tick (infauna below the sediment
            # surface is out of it); scales the prey biomass of the pair.
            access[i, j] = pair_share(inter_def, "accessible_fraction", inter_id)
            forced = inter_def.get("functional_response", None)
            if forced not in (None, "", 0, 0.0):
                try:
                    forced = int(forced)
                except (TypeError, ValueError):
                    forced = -1
                if forced not in (2, 3):
                    raise ValueError(
                        f"{inter_id}: functional_response must be 2 or 3, "
                        f"got {inter_def.get('functional_response')!r}")
                response_type[i, j] = forced

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
        response_type=response_type,
        access=access,
        share_cap=share_cap,
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
        if nutrients.species_layer(fg.params) == "subsurface":
            # Section 180: light through the mixed layer, then the layer.
            sub = (fg.params.get("nutrients") or {}).get("subsurface")
            if not sub:
                # No layer in this project (e.g. daylight without
                # nutrients): the producer is inert - it does not grow.
                # With nutrients on, build_nutrient_tables refuses it.
                env.growth_light[fid] = np.zeros(env.light_period)
                continue
            env.growth_light[fid] = daylight.subsurface_light_schedule(
                latitude, tick_hours, climate, ik, sub["bottom_m"],
                sub["light_attenuation_per_m"], reference_day)
        else:
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
    # Section 177: a grazer feeds on CONCENTRATION, so a pair with
    # ``mixed_layer_reference_m`` has a(t) = a * M_ref / M(t): its areal
    # half-saturation 1/(a h) is the library's at M_ref and follows the
    # mixed layer. Folded into the same per-tick pair table.
    for i, pred_id in enumerate(env.dm_ids):
        interaction = env.fgs[pred_id].params.get("interaction", {})
        for j, prey_id in enumerate(env.global_fg_order):
            if env.eat_static_mask[i, j] <= 0.0:
                continue
            scale = mixed_layer_scale(
                env, interaction.get(f"{pred_id}_preys_on_{prey_id}"),
                prey_id)
            if scale is None:
                continue
            table[:, i, j] *= scale.astype(env.dtype)
            env._has_light_pairs = True
    if env._has_light_pairs:
        env.light_mult_table = table


def mixed_layer_scale(env, inter_def, prey_id=None):
    """(T,) M_ref / M(t) for a pair with ``mixed_layer_reference_m``, or
    None (section 177). Needs the daylight light climate. A prey in the
    subsurface layer (section 180) is diluted over that layer instead:
    M_ref / H(t), H = bottom_m - M(t)."""
    if not inter_def:
        return None
    raw = inter_def.get("mixed_layer_reference_m")
    if raw in (None, "", 0, 0.0):
        return None
    climate = (getattr(env, "daylight", None) or {}).get("light_climate")
    if not climate:
        raise ValueError("mixed_layer_reference_m needs the daylight light "
                         "climate (mixed_layer_depth_m); section 177")
    hours = int(env.daylight["tick_hours"])
    n_ticks = daylight.ticks_per_year(hours)
    mid = (np.arange(n_ticks) * hours + 0.5 * hours) / 24.0
    depth = daylight.monthly_value(climate["mixed_layer_depth_m"], mid)
    prey = env.fgs.get(prey_id) if prey_id is not None else None
    if prey is not None and nutrients.species_layer(prey.params) == "subsurface":
        sub = (prey.params.get("nutrients") or {}).get("subsurface")
        if sub:
            depth = sub["bottom_m"] - depth
    return float(raw) / depth


def build_ndm_grazing(env):
    """Non decision makers that graze another group (section 177).

    An NDM with a checked ``<grazer>_preys_on_<prey>`` pair takes, per
    cell and tick, ``I = B_g m(t) a' B_p / (1 + a' h B_p)`` of the prey
    (a' = a x the pair's mixed-layer scale, m(t) its Q10 multiplier),
    capped at MAX_HARVEST_FRAC of the prey; it gains growth_efficiency x
    I and returns excretion_regeneration x (1 - growth_efficiency) x I
    to the prey's nutrient pool when the prey has one, plus
    egestion_regeneration x the egesta (section 202). Sets
    ``env.ndm_grazers`` (a list of dicts, in global order) and
    ``env._has_ndm_grazing``. Must run after the daylight, temperature
    and nutrient tables.
    """
    env.ndm_grazers = []
    env._has_ndm_grazing = False
    for gid in env.global_fg_order:
        grazer = env.fgs[gid]
        if grazer.is_decision_maker:
            continue
        for pair_id, pair in (grazer.params.get("interaction") or {}).items():
            if not pair_id.startswith(f"{gid}_preys_on_") or not pair.get(
                    "preys_on", False):
                continue
            pid = pair_id[len(f"{gid}_preys_on_"):]
            if pid not in env.fgs:
                continue
            a = float(pair.get("max_intake_rate", 0.0) or 0.0)
            h = float(pair.get("handling_time", 0.0) or 0.0)
            if a <= 0.0:
                continue
            gge = float(grazer.params.get("growth_efficiency", 0.3))
            regen = float(grazer.params.get("excretion_regeneration", 0.0)
                          or 0.0)
            if not (0.0 < gge <= 1.0 and 0.0 <= regen <= 1.0):
                raise ValueError(f"'{gid}': growth_efficiency must be in "
                                 "(0, 1] and excretion_regeneration in "
                                 "[0, 1]")
            # Section 202: a share of the egesta returns as well.
            from lib.environments.ecosystem_env.regeneration import species_share
            egest = species_share(grazer.params, "egestion_regeneration")
            # Q10 and the mixed-layer scale are inert without the calendar
            # (e.g. setup_full_mareld_mvp), as a decision maker's Q10 and
            # pair scaling are; the loader refuses an enabled temperature
            # block without the grazer's layer.
            calendar = getattr(env, "daylight", None)
            temp = None
            q10 = grazer.params.get("grazing_q10")
            cfg = grazer.params.get("temperature")
            if q10 not in (None, "", 0, 0.0, 1, 1.0) and cfg and calendar:
                t_ref = float(grazer.params.get("grazing_t_ref", 10.0))
                temps = temperature.tick_temperatures(
                    cfg["monthly_c"], env.daylight["tick_hours"])
                temp = np.power(float(q10), (temps - t_ref) / 10.0).astype(
                    env.dtype)
            scale = (mixed_layer_scale(env, pair, pid)
                     if calendar and calendar.get("light_climate") else None)
            threshold = float(pair.get("food_threshold", 0.0) or 0.0)
            if threshold < 0.0:
                raise ValueError(f"'{pair_id}': food_threshold must be >= 0")
            env.ndm_grazers.append({
                "edibility": _edibility(env, pair_id, pair, pid),
                "threshold": threshold,
                "grazer": gid, "prey": pid, "a": a, "h": h, "gge": gge,
                "regen": regen, "egest": egest,
                # share of the intake returned to the prey's pool:
                # excretion plus the regenerated egesta (section 202)
                "pool": (1.0 - gge) * (regen + egest * (1.0 - regen)),
                "temperature": temp,
                "scale": None if scale is None else scale.astype(env.dtype)})
    env._has_ndm_grazing = bool(env.ndm_grazers)


def _edibility(env, pair_id, pair, prey_id):
    """The pair's edible share of the prey (section 182), or None.

    ``edibility_half_saturation_umol_l`` K and ``edibility_floor`` e0 make
    the grazer see only ``e = e0 + (1 - e0) N/(N + K)`` of the prey, N
    the prey's nutrient pool as a concentration: the producer's community
    shifts to cells the grazer cannot take as its pool runs dry. Returns
    ``(e0, half)`` with ``half[t]`` the pool (in the producer's units) at
    which N = K, so ``N/(N + K) = pool/(pool + half[t])``.
    """
    has_half = pair.get("edibility_half_saturation_umol_l") not in (None, "")
    has_floor = pair.get("edibility_floor") not in (None, "")
    if not has_half and not has_floor:
        return None
    if not (has_half and has_floor):
        raise ValueError(f"'{pair_id}': edibility_floor and "
                         "edibility_half_saturation_umol_l go together "
                         "(section 182)")
    k = float(pair["edibility_half_saturation_umol_l"])
    floor = float(pair["edibility_floor"])
    if not (math.isfinite(k) and k > 0.0 and 0.0 <= floor <= 1.0):
        raise ValueError(f"'{pair_id}': edibility_half_saturation_umol_l "
                         "must be > 0 and edibility_floor in [0, 1]")
    if not getattr(env, "_has_nutrients", False):
        # Inert without the pools, as the Q10 and the mixed-layer scale
        # are without the calendar (e.g. a test or MVP world).
        return None
    tables = env.nutrient_tables.get(prey_id)
    if tables is None:
        raise ValueError(f"'{pair_id}': edibility needs a prey with a "
                         "nutrient pool (simulation_settings.nutrients and "
                         "the prey's nutrient_half_saturation; section 182)")
    k_n = float(env.fgs[prey_id].params["nutrient_half_saturation"])
    half = (np.asarray(tables["half"], dtype=np.float64) * (k / k_n))
    return floor, half.astype(env.dtype)


def ndm_grazing_rate(env, entry, prey_biomass, prey_pool=None):
    """Per tonne of grazer, tonnes of prey taken this tick (section 177).

    With ``food_threshold`` F0 (t/km2 at the pair's reference depth,
    scaled with the mixed layer like the half-saturation) the prey the
    grazer perceives is F F / (F + F0): ERSEM's food-perception term,
    which makes the response sigmoid at low prey and leaves it a refuge.
    With the pair's edibility (section 182) F is first the edible share
    e(N) of the prey, N from ``prey_pool`` (the prey's nutrient pool).
    """
    a = entry["a"]
    threshold = entry.get("threshold", 0.0)
    if entry["scale"] is not None:
        s = float(entry["scale"][light_index(env)])
        a = a * s
        threshold = threshold / s
    food = prey_biomass
    edibility = entry.get("edibility")
    if edibility is not None:
        if prey_pool is None:
            raise ValueError("ndm_grazing_rate needs the prey's pool with "
                             "an edibility term (section 182)")
        floor, half = edibility
        pool = np.maximum(prey_pool, np.float32(0.0))
        h = half[light_index(env)]
        food = prey_biomass * (np.float32(floor) + np.float32(1.0 - floor)
                               * pool / (pool + h))
    if threshold > 0.0:
        food = food * food / (food + threshold)
    rate = a * food / (1.0 + a * entry["h"] * food)
    if entry["temperature"] is not None:
        rate = rate * float(entry["temperature"][light_index(env)])
    return rate


def build_nutrient_tables(env):
    """Nutrient pools of light-limited producers (section 172).

    Sets ``env._has_nutrients``, ``env.nutrient_ids`` (producers with a
    pool, in ``global_fg_order``), ``env.nutrient_config`` and, per
    producer, ``env.nutrient_tables[fid]`` = {"keep", "supply", "half"}
    (per year tick, ``nutrients.tick_tables``), ``env.nutrient_yield[fid]``
    and ``env.nutrient_regeneration[fid]``. Everything is off - and the
    tick bit-identical - unless the manifest enables
    ``simulation_settings.nutrients``. Must run after
    ``build_daylight_tables`` (the pool reads the same year tick).
    """
    cfg = None
    for fid in env.global_fg_order:
        cfg = env.fgs[fid].params.get("nutrients")
        if cfg:
            break
    env.nutrient_config = dict(cfg) if cfg else None
    env.nutrient_ids = []
    env.nutrient_tables = {}
    env.nutrient_yield = {}
    env.nutrient_regeneration = {}
    env.nutrient_stress = {}
    env.nutrient_stress_regeneration = {}
    env.nutrient_layer = {}
    env.layer_exchange = None
    env._has_nutrients = False
    if env.nutrient_config is None:
        return
    if getattr(env, "daylight", None) is None:
        raise ValueError("simulation_settings.nutrients needs the daylight "
                         "calendar (section 172)")
    for fid in env.global_fg_order:
        fg = env.fgs[fid]
        settings = nutrients.species_settings(fg.params)
        if settings is None:
            continue
        if fg.is_decision_maker:
            raise ValueError(f"'{fid}' has nutrient_half_saturation but is a "
                             "decision maker: only producers take up "
                             "nutrients (section 172)")
        k_n, yield_, rho = settings
        layer = nutrients.species_layer(fg.params)
        tables = nutrients.tick_tables(env.nutrient_config, yield_, k_n,
                                       layer)
        env.nutrient_layer[fid] = layer
        if len(tables["keep"]) != env.light_period:
            raise ValueError("nutrient tables and the daylight calendar "
                             "disagree on the tick length")
        env.nutrient_ids.append(fid)
        env.nutrient_tables[fid] = {
            key: value.astype(env.dtype) for key, value in tables.items()}
        env.nutrient_yield[fid] = yield_
        env.nutrient_regeneration[fid] = rho
        stress = nutrients.stress_settings(fg.params)
        if stress is not None:
            env.nutrient_stress[fid] = stress
        env.nutrient_stress_regeneration[fid] = float(
            fg.params.get("stress_regeneration", 0.0) or 0.0)
    if not env.nutrient_ids:
        raise ValueError("simulation_settings.nutrients is enabled but no "
                         "group has nutrient_half_saturation (section 172)")
    env._has_nutrients = True
    if env.nutrient_config.get("subsurface"):
        # Section 180: one producer per layer, one nutrient currency.
        by_layer = {}
        for fid in env.nutrient_ids:
            by_layer.setdefault(env.nutrient_layer[fid], []).append(fid)
        if sorted(by_layer) != ["mixed", "subsurface"] or any(
                len(v) != 1 for v in by_layer.values()):
            raise ValueError(
                "nutrients.subsurface_layer needs exactly one producer with "
                "water_layer 'mixed' and one with 'subsurface' (section "
                f"180), got {by_layer}")
        mixed, deep = by_layer["mixed"][0], by_layer["subsurface"][0]
        if env.nutrient_yield[mixed] != env.nutrient_yield[deep]:
            raise ValueError(f"'{mixed}' and '{deep}' must share one "
                             "nutrient_yield (section 180)")
        tables = nutrients.layer_tables(env.nutrient_config,
                                        env.nutrient_yield[mixed])
        env.layer_exchange = {
            "mixed": mixed, "subsurface": deep,
            **{key: value.astype(env.dtype) for key, value in tables.items()}}
    elif "subsurface" in env.nutrient_layer.values():
        raise ValueError("water_layer 'subsurface' needs "
                         "nutrients.subsurface_layer in the manifest "
                         "(section 180)")


def initialise_nutrient_pools(env):
    """Fill every producer's pool with the surface climatology of the
    world's start day (section 172). The pool lives in the producer's
    ``energy_reserve``; called once, when the environment is built."""
    if not getattr(env, "_has_nutrients", False):
        return
    for fid in env.nutrient_ids:
        fg = env.fgs[fid]
        pool = nutrients.initial_pool(env.nutrient_config,
                                      env.nutrient_yield[fid], env.light_start,
                                      env.nutrient_layer.get(fid, "mixed"))
        fg.energy_reserve = np.full(fg.biomass.shape, pool, dtype=env.dtype)


def is_nutrient_producer(env, fg_id):
    """True when ``fg_id``'s reserve slot holds a nutrient pool."""
    return fg_id in getattr(env, "nutrient_regeneration", {})


def build_temperature_tables(env):
    """Q10 metabolism multipliers on the daylight calendar (section 143).

    Also the ingestion multipliers of section 181
    (``env._has_ingestion_temperature``, ``env.ingestion_temp_table``,
    (T, N_dm), 1 for a DM without ``ingestion_q10``) and the growth
    multipliers of section 183 (``env._has_growth_temperature``,
    ``env.growth_temp_table``, likewise).

    Sets ``env._has_temperature`` and ``env.metabolism_temp_table``
    ((T, N_dm) multiplier on ``resting_metabolism``, exactly 1 for every
    decision maker without ``metabolism_q10``, or None when no group
    has one). Off - and the tick bit-identical to before - unless the
    loader put a ``params['temperature']`` on a decision maker. Adds no
    observation channel. Must run after ``build_daylight_tables``.
    """
    env._has_temperature = False
    env.metabolism_temp_table = None
    env._has_ingestion_temperature = False
    env.ingestion_temp_table = None
    env._has_growth_temperature = False
    env.growth_temp_table = None
    configured = [fid for fid in env.global_fg_order
                  if env.fgs[fid].params.get("temperature")]
    if not configured:
        return
    if env.daylight is None:
        raise ValueError(
            "temperature-dependent metabolism needs the daylight calendar "
            f"(groups {configured}); section 143")
    tick_hours = int(env.daylight["tick_hours"])
    table = np.ones((env.light_period, env.N_dm), dtype=env.dtype)
    for i, fid in enumerate(env.dm_ids):
        fg = env.fgs[fid]
        cfg = fg.params.get("temperature")
        settings = temperature.species_settings(fg.params)
        if not cfg or settings is None:
            continue
        q10, t_ref = settings
        table[:, i] = temperature.metabolism_multiplier(
            cfg["monthly_c"], tick_hours, q10, t_ref)
        env._has_temperature = True
    if env._has_temperature:
        env.metabolism_temp_table = table
    # Section 181: Q10 on the whole functional response, annual mean 1.
    intake = np.ones((env.light_period, env.N_dm), dtype=env.dtype)
    for i, fid in enumerate(env.dm_ids):
        fg = env.fgs[fid]
        cfg = fg.params.get("temperature")
        q10 = temperature.ingestion_q10(fg.params)
        if not cfg or q10 is None:
            continue
        intake[:, i] = temperature.metabolism_multiplier(
            cfg["monthly_c"], tick_hours, q10, temperature.ANNUAL_MEAN)
        env._has_ingestion_temperature = True
    if env._has_ingestion_temperature:
        env.ingestion_temp_table = intake
    # Section 183: Q10 on the growth rate, annual mean 1.
    growth = np.ones((env.light_period, env.N_dm), dtype=env.dtype)
    for i, fid in enumerate(env.dm_ids):
        fg = env.fgs[fid]
        cfg = fg.params.get("temperature")
        q10 = temperature.growth_q10(fg.params)
        if not cfg or q10 is None:
            continue
        growth[:, i] = temperature.metabolism_multiplier(
            cfg["monthly_c"], tick_hours, q10, temperature.ANNUAL_MEAN)
        env._has_growth_temperature = True
    if env._has_growth_temperature:
        env.growth_temp_table = growth


def metabolism_temperature(env):
    """``(N_dm,)`` multiplier on resting_metabolism this tick, or None."""
    if not getattr(env, "_has_temperature", False):
        return None
    return env.metabolism_temp_table[light_index(env)]


def growth_temperature(env, fg_id):
    """This tick's multiplier on ``fg_id``'s growth rate (section 183)."""
    if not getattr(env, "_has_growth_temperature", False):
        return 1.0
    return float(env.growth_temp_table[light_index(env),
                                       env.dm_ids.index(fg_id)])


def ingestion_temperature(env):
    """``(N_dm,)`` multiplier on every DM's intake this tick, or None."""
    if not getattr(env, "_has_ingestion_temperature", False):
        return None
    return env.ingestion_temp_table[light_index(env)]


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


def apply_pair_access(env, visible, pair_visible):
    """Section 208: scale each pair's visible prey by its
    ``accessible_fraction`` and rebuild the shared cap from the pairs.
    Unchanged (and no copy) when every pair reaches all of its prey."""
    if not getattr(env, "_has_pair_access", False):
        return visible, pair_visible
    base = pair_visible if pair_visible is not None else visible[None, :, :, :]
    pair_visible = (base * env.access_mat[:, :, None, None]).astype(env.dtype, copy=False)
    shared = np.where(env.eat_static_mask[:, :, None, None] > 0.0,
                      pair_visible, np.float32(0.0)).max(axis=0)
    return shared, pair_visible
