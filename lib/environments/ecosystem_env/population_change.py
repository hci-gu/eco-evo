import argparse

import numpy as np

from lib.environments.ecosystem_env import interactions
from lib.world import nutrients
from lib.environments.ecosystem_env import natural_mortality as natural_mortality_module
from lib.environments.ecosystem_env import regeneration
from lib.environments.ecosystem_env import refuge


DEFAULT_MORTALITY_MULTIPLIER = 1.0


def mortality_multiplier_type(value):
    """argparse type for ``--mortality_multiplier``: a finite factor >= 0."""
    try:
        factor = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--mortality_multiplier must be a number")
    if not np.isfinite(factor) or factor < 0.0:
        raise argparse.ArgumentTypeError(
            "--mortality_multiplier must be finite and >= 0")
    return factor


def add_mortality_multiplier_argument(parser):
    """Register ``--mortality_multiplier`` on a CPU or GPU parser.

    Shared so the two entry points cannot drift apart, exactly as
    ``source_tracking.add_local_reward_arguments`` does for the local
    reward flags.
    """
    parser.add_argument("--mortality_multiplier", "--mortality-multiplier",
                        dest="mortality_multiplier",
                        type=mortality_multiplier_type,
                        default=DEFAULT_MORTALITY_MULTIPLIER,
                        metavar="FACTOR",
                        help="Scale every functional group's "
                             "``natural_mortality`` by FACTOR (e.g. 0.5 halves "
                             "it, 2 doubles it). Requires --mortality on to "
                             "have any effect; 0 is equivalent to "
                             "--mortality off. Default: 1.0 (library values "
                             "unchanged).")


# ON since section 120. The library is calibrated for the mass-balanced
# growth term - zooplankton's resting_metabolism (118) and growth_rate
# (111) are both set in that world - so a run that does not pass the flag
# has to get that world, not the legacy one. ``--no-mass-balance`` is the
# way back to the pre-116 tick.
DEFAULT_MASS_BALANCE = True

# What a run recorded BEFORE the flag existed was running: the legacy
# term. Kept separate from the default above so that flipping the default
# never rewrites history (see training_progress.comparable_config).
LEGACY_MASS_BALANCE = False


def add_mass_balance_argument(parser):
    """Register ``--mass-balance`` / ``--no-mass-balance``.

    Shared between the entry points for the same reason
    :func:`add_mortality_multiplier_argument` is (section 88.3): a flag
    that changes the tick must not be able to mean two different things
    on two parsers.

    Both spellings exist because the default flipped in section 120:
    ``--mass-balance`` was the way to opt IN while the default was off
    and is now a no-op that scripts written against 116 keep working
    with, and ``--no-mass-balance`` is the way to opt out.
    """
    parser.add_argument("--mass-balance", "--mass_balance",
                        dest="mass_balance", action="store_true",
                        default=DEFAULT_MASS_BALANCE,
                        help="Make the growth term PAY for the biomass it "
                             "adds: cap it by the energy reserve above the "
                             "maintenance line and debit that reserve "
                             "(option A, section 116). ON by default since "
                             "section 120, so this flag is a no-op kept for "
                             "scripts written against the old default.")
    parser.add_argument("--no-mass-balance", "--no_mass_balance",
                        dest="mass_balance", action="store_false",
                        help="Revert to the pre-116 growth term, which adds "
                             "biomass without debiting the energy reserve. "
                             "The library is NOT calibrated for it (118, "
                             "119.4): zooplankton then converts mass at "
                             "45 %% against a 28.9 %% thermodynamic "
                             "ceiling.")


def _charge_growth_to_reserve(env, fg, biomass_delta):
    """``--mass-balance``: growth buys biomass instead of receiving it.

    Without this the growth term adds biomass and leaves
    ``energy_reserve`` untouched, so the new biomass arrives carrying an
    implied ``energy_content`` that was never withdrawn from anywhere -
    measured at 198-344 % of the mass actually eaten, against a ceiling
    of 28.9 % from the library's own numbers (sections 114, 115).

    With it, the growth term is still GATED by satiation exactly as
    ``Strategi.pdf`` specifies - ``B * MG_X * q_X`` is the wish - but the
    wish is capped by the reserve standing above the maintenance line
    and that reserve is then debited:

        E_avail = max(0, R - u_X * ME_X * B)
        dB      = min(B * MG_X * q_X, E_avail / energy_content_X)
        R      -= dB * energy_content_X

    The debit is the part that matters. The cap alone leaves the reserve
    refilling freely, so ``s_X`` falls only by dilution, which is an
    order of magnitude weaker (section 115.2).

    Only the POSITIVE part of ``biomass_delta`` is touched. Starvation is
    a loss, not a purchase, and its branch is left exactly as it was.

    Note that this makes ``max_energy_reserve / energy_content`` a hard
    ceiling on the reachable ``growth_rate``, since cap and wish are both
    linear in ``q_X``. For zooplankton that ratio is 0.150 against a
    ``growth_rate`` of 0.128 since section 160 paid the budget at the
    ideal reserve (``growth_budget`` rejects a g above the ratio).
    Section 115.3 has the table for every FG.
    """
    energy_content = float(fg.params.get("energy_content", 0.0) or 0.0)
    if energy_content <= 0.0:
        # No energy content, no price. Leaving the wish alone is the only
        # defined behaviour; an FG in this state is a configuration bug
        # the growth term is not the place to diagnose.
        return biomass_delta

    available = np.maximum(
        0.0,
        fg.energy_reserve
        - np.float32(fg.maintenance_level * fg.max_energy_reserve)
        * fg.biomass,
    )
    cap = available / np.float32(energy_content)
    gain = np.minimum(np.maximum(0.0, biomass_delta), cap)
    fg.energy_reserve = np.maximum(
        0.0, fg.energy_reserve - gain * np.float32(energy_content)
    ).astype(env.dtype, copy=False)
    return np.where(biomass_delta > 0.0, gain, biomass_delta).astype(
        env.dtype, copy=False)


def _nutrient_limited_delta(env, fg_id, fg, population_rate,
                            carrying_capacity):
    """Section 172: supply the pool, then grow on it.

    ``N' = N * keep(t) + supply(t)`` (vertical exchange and entrainment,
    ``nutrients.tick_tables``), then ``dB = min(rate B N/(N + half(t)),
    N, K - B)`` and the pool pays for it. Returns ``dB``; the pool in
    ``energy_reserve`` is updated here.
    """
    tables = env.nutrient_tables[fg_id]
    t = interactions.light_index(env)
    pool = (fg.energy_reserve * tables["keep"][t]
            + tables["supply"][t]).astype(env.dtype, copy=False)
    pool = np.maximum(pool, np.float32(0.0))
    limitation = pool / (pool + tables["half"][t])
    stress = getattr(env, "nutrient_stress", {}).get(fg_id)
    if stress is not None:
        # Section 175: lysis and aggregate sinking once the pool runs
        # dry, before growth; exported, booked as natural mortality.
        rate = nutrients.stress_loss(limitation, *stress)
        before = float(fg.biomass.sum())
        lost = fg.biomass * rate
        fg.biomass = (fg.biomass - lost).astype(env.dtype, copy=False)
        # Section 177: the lysed share is remineralised in the mixed layer.
        pool = (pool + np.float32(env.nutrient_stress_regeneration.get(
            fg_id, 0.0)) * lost).astype(env.dtype, copy=False)
        losses = getattr(env, "loss_natural", None)
        if losses is not None:
            losses[fg_id] = (float(losses.get(fg_id, 0.0))
                             + before - float(fg.biomass.sum()))
    wish = population_rate * fg.biomass * limitation
    delta = np.minimum(np.minimum(wish, pool),
                       np.maximum(np.asarray(carrying_capacity, dtype=np.float32) - fg.biomass,
                                  np.float32(0.0)))
    delta = np.maximum(delta, np.float32(0.0)).astype(env.dtype, copy=False)
    fg.energy_reserve = (pool - delta).astype(env.dtype, copy=False)
    return delta


def _apply_non_decision_maker_mortality(env, fg_id, fg):
    """Section 173: an NDM's own losses M1 (respiration, sinking, lysis),
    before its growth, as for a decision maker. Only the biomass: a
    producer's nutrient pool is the water's, and an NDM's reserve slot
    holds no energy. Booked as natural mortality."""
    rate = float(getattr(fg, "natural_mortality", 0.0) or 0.0)
    rate *= float(getattr(env, "mortality_multiplier", 1.0))
    if rate <= 0.0 or not env.apply_natural_mortality:
        return
    keep = np.float32(max(0.0, 1.0 - rate))
    before = float(fg.biomass.sum())
    start = fg.biomass
    fg.biomass = (fg.biomass * keep).astype(env.dtype, copy=False)
    if refuge.settings(fg) is not None:
        # Section 223.12: M1 takes both compartments alike.
        fg.energy_reserve = (fg.energy_reserve * keep).astype(
            env.dtype, copy=False)
    regeneration.record_m1(env, fg_id, start, fg.biomass)
    losses = getattr(env, "loss_natural", None)
    if losses is not None:
        losses[fg_id] = (float(losses.get(fg_id, 0.0))
                         + before - float(fg.biomass.sum()))


def _apply_non_decision_maker_population_change(env, fg_id, fg):
    # Section 207: a depth-habitat group carries K per cell.
    carrying_capacity = getattr(fg, "capacity_map", None)
    if carrying_capacity is None:
        carrying_capacity = fg.params.get("max_carrying_capacity", 100.0)
    _apply_non_decision_maker_mortality(env, fg_id, fg)
    # Light-limited producers grow at growth_rate * P(t)/P_ref under the
    # daylight calendar (section 138); 1.0 for everything else.
    population_rate = fg.growth_rate * interactions.growth_light(env, fg_id)
    if fg_id in getattr(env, "inert_producers", ()):
        population_rate = 0.0     # section 180: no layer to live in
    if interactions.is_nutrient_producer(env, fg_id):
        biomass_delta = _nutrient_limited_delta(
            env, fg_id, fg, population_rate, carrying_capacity)
    else:
        biomass_delta = (
            population_rate
            * fg.biomass
            * (1.0 - fg.biomass / (carrying_capacity + 1e-9))
        )

    seed_rate = float(getattr(fg, "seed_rate", 0.0) or 0.0)
    if seed_rate > 0.0:
        seed_mult = np.power(
            10.0,
            np.random.uniform(-1.0, 1.0, size=fg.biomass.shape),
        ).astype(np.float32, copy=False)
        biomass_delta = (
            biomass_delta
            + np.asarray(seed_rate * carrying_capacity, dtype=np.float32) * seed_mult
        )

    if refuge.settings(fg) is not None:
        # Section 223.12: per-compartment logistic, then the exchange.
        seed_delta = biomass_delta - (
            population_rate * fg.biomass
            * (1.0 - fg.biomass / (carrying_capacity + 1e-9)))
        refuge.grow(fg, population_rate, carrying_capacity, seed_delta,
                    env.dtype)
        return
    fg.biomass = np.clip(
        fg.biomass + biomass_delta, 0.0, carrying_capacity
    ).astype(env.dtype, copy=False)


def _apply_decision_maker_population_change(env, fg_id, fg):
    natural_mortality = float(getattr(fg, "natural_mortality", 0.0) or 0.0)
    # ``--mortality_multiplier`` scales every FG's rate by the same
    # factor; 1.0 (the default) leaves this branch bit-identical.
    natural_mortality *= float(getattr(env, "mortality_multiplier", 1.0))
    if natural_mortality > 0.0 and env.apply_natural_mortality:
        # Exposure-weighted M1 (section 139): per cell, by this tick's
        # hiding and light. None for every FG without it, which keeps
        # the scalar path below bit-identical.
        parts = natural_mortality_module.components(
            env, fg_id, natural_mortality)
        if parts is None:
            keep = np.float32(max(0.0, 1.0 - natural_mortality))
        else:
            rate = np.clip(parts[0] + parts[1] + parts[2], 0.0, 1.0)
            keep = (1.0 - rate).astype(env.dtype)
            book = getattr(env, "loss_natural_parts", None)
            if book is not None:
                row = book.setdefault(fg_id, {"visual": 0.0, "tactile": 0.0,
                                              "other": 0.0})
                biomass = fg.biomass.astype(np.float64)
                scale = np.where(parts[0] + parts[1] + parts[2] > 0.0,
                                 rate / np.maximum(parts[0] + parts[1]
                                                   + parts[2], 1e-300), 0.0)
                for name, part in zip(("visual", "tactile", "other"), parts):
                    row[name] += float((biomass * part * scale).sum())
        before = float(fg.biomass.sum())
        start = fg.biomass
        fg.energy_reserve = (fg.energy_reserve * keep).astype(
            env.dtype, copy=False)
        fg.biomass = (fg.biomass * keep).astype(env.dtype, copy=False)
        regeneration.record_m1(env, fg_id, start, fg.biomass)
        # Book M1 like the other causes, so the loss breakdown can show it
        # (section 132). ``getattr`` keeps hand-built test envs working.
        losses = getattr(env, "loss_natural", None)
        if losses is not None:
            losses[fg_id] = (float(losses.get(fg_id, 0.0))
                             + before - float(fg.biomass.sum()))

    # Section 185: the regional logistic loss c * B_region, B_region the
    # grid-mean density after M1. Outside the derived growth rate and
    # independent of --mortality (it closes r_max, which g always pays).
    if getattr(env, "_has_regional_capacity", False):
        coefficient = float(env.dm_capacity_rate[env.dm_ids.index(fg_id)])
        if coefficient > 0.0 and fg.biomass.size:
            before = float(fg.biomass.sum())
            keep = np.float32(max(0.0, 1.0 - coefficient * before
                                  / fg.biomass.size))
            fg.energy_reserve = (fg.energy_reserve * keep).astype(
                env.dtype, copy=False)
            fg.biomass = (fg.biomass * keep).astype(env.dtype, copy=False)
            losses = getattr(env, "loss_natural", None)
            if losses is not None:
                losses[fg_id] = (float(losses.get(fg_id, 0.0))
                                 + before - float(fg.biomass.sum()))

    energy_surplus = fg.energy_level - fg.maintenance_level
    starve_rate = np.float32(
        fg.starve_rate if fg.starve_rate > 0.0 else fg.growth_rate)
    # Section 183: growth_q10 scales the growth branch only.
    growth_rate = np.float32(
        fg.growth_rate * interactions.growth_temperature(env, fg_id))
    # Section 184: an income-funded DM grew in ``apply_energy_costs``;
    # it never turns its reserve into biomass here.
    if getattr(env, "_has_income_growth", False) and (
            env.dm_income_kappa[env.dm_ids.index(fg_id)] > 0.0):
        growth_rate = np.float32(0.0)
    rate = np.where(energy_surplus >= 0.0, growth_rate, starve_rate).astype(
        env.dtype, copy=False)
    biomass_delta = fg.biomass * rate * energy_surplus

    # ``--mass-balance``: charge the growth to the reserve before anything
    # downstream reads the delta, so the loss accounting below sees the
    # biomass that was actually paid for. Off by default (section 116).
    if getattr(env, "mass_balance", False):
        biomass_delta = _charge_growth_to_reserve(env, fg, biomass_delta)

    total_loss = -np.minimum(0.0, biomass_delta)
    actual_starve_loss = np.minimum(total_loss, fg.biomass)
    env.loss_starvation[fg_id] = (
        float(env.loss_starvation.get(fg_id, 0.0))
        + float(actual_starve_loss.sum())
    )

    loss_mask = total_loss > 0
    reduction = np.ones_like(fg.biomass)
    reduction[loss_mask] = (
        (fg.biomass[loss_mask] - total_loss[loss_mask])
        / (fg.biomass[loss_mask] + 1e-9)
    )
    reduction = np.clip(reduction, 0.0, 1.0)

    fg.energy_reserve = (fg.energy_reserve * reduction).astype(
        env.dtype, copy=False)
    fg.biomass = np.maximum(0.0, fg.biomass + biomass_delta).astype(
        env.dtype, copy=False)


def _clip_biomass_based_on_min_thresholds(env):
    """Silent variant of the extinction sweep, without loss accounting.

    Used only when ``apply_population_change`` is driven standalone (no
    surrounding tick), so the helper stays self-contained. Inside a full
    tick ``apply_extinction_threshold`` runs after the accessibility mask
    and owns the rule, including the starvation/event bookkeeping, so the
    growth stage must not pre-empt it there.
    """
    for fg in env.fgs.values():
        biomass = fg.biomass

        min_split = float(getattr(fg, "min_split_biomass", 0.0) or 0.0)
        factor = float(getattr(fg, "extinction_threshold_factor", 0.0) or 0.0)
        threshold = np.float32(max(0.0, min_split * factor))
        biomass = np.where(biomass < threshold, 0.0, biomass).astype(
            env.dtype, copy=False)
        fg.biomass = biomass


def apply_extinction_threshold(env):
    """Zero cells where ``0 < B < extinction_threshold_factor * min_split_biomass``.

    Below ``min_split_biomass`` the action distribution already collapses
    to a one-hot argmax (see ``decisions.collapse_subthreshold_actions``);
    once the biomass is then scaled down multiplicatively by predation /
    starvation / impact it quickly reaches float32 subnormals (~1e-38 to
    1e-45) that carry no biological meaning but add noise to the
    observations, the loss breakdown and the reward. We read "less than
    half an indivisible individual" as locally extinct and book the
    vanished biomass as a starvation loss (counting the events in
    ``_extinction_events``).

    Only FGs with ``min_split_biomass > 0`` AND
    ``extinction_threshold_factor > 0`` are affected. Continuous FGs
    (e.g. plankton, msb=0) are untouched.
    """
    for fg_id, fg in env.fgs.items():
        min_split = float(getattr(fg, "min_split_biomass", 0.0) or 0.0)
        factor = float(getattr(fg, "extinction_threshold_factor", 0.0) or 0.0)
        if min_split <= 0.0 or factor <= 0.0:
            continue
        threshold = np.float32(factor * min_split)
        biomass = fg.biomass
        if biomass is None:
            continue
        dead_mask = (biomass > 0.0) & (biomass < threshold)
        if not np.any(dead_mask):
            continue
        lost = float(biomass[dead_mask].sum())
        # Booked as starvation: semantically the closest channel, the cell
        # holds too little biomass to form a viable unit.
        env.loss_starvation[fg_id] = (
            float(env.loss_starvation.get(fg_id, 0.0)) + lost
        )
        env._extinction_events[fg_id] = (
            int(env._extinction_events.get(fg_id, 0))
            + int(dead_mask.sum())
        )
        # Zero biomass and the associated energy reserve in the same cells.
        fg.biomass = np.where(dead_mask, np.float32(0.0), biomass).astype(
            env.dtype, copy=False)
        if fg.energy_reserve is not None:
            fg.energy_reserve = np.where(
                dead_mask, np.float32(0.0), fg.energy_reserve
            ).astype(env.dtype, copy=False)


def _zero_energy_in_empty_cells(env):
    for fg_id, fg in env.fgs.items():
        # The nutrient pool stays in a cell without producers (172).
        if interactions.is_nutrient_producer(env, fg_id):
            continue
        fg.energy_reserve = np.where(
                fg.biomass <= 0.0,
                np.float32(0.0),
                fg.energy_reserve,
            ).astype(env.dtype, copy=False)


def apply_ndm_grazing(env):
    """Section 177: grazing non decision makers, before every group's
    population change (deterministic order, mirrored on the GPU)."""
    from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC
    env._tick_ndm_intake = {}
    for entry in getattr(env, "ndm_grazers", []):
        grazer, prey = env.fgs[entry["grazer"]], env.fgs[entry["prey"]]
        rate = interactions.ndm_grazing_rate(env, entry, prey.biomass,
                                             prey.energy_reserve)
        intake = np.minimum(grazer.biomass * rate,
                            np.float32(MAX_HARVEST_FRAC) * prey.biomass)
        intake = np.maximum(intake, np.float32(0.0)).astype(env.dtype,
                                                            copy=False)
        prey.biomass = (prey.biomass - intake).astype(env.dtype, copy=False)
        grazer.biomass = (grazer.biomass
                          + np.float32(entry["gge"]) * intake).astype(
                              env.dtype, copy=False)
        if interactions.is_nutrient_producer(env, entry["prey"]):
            share = entry.get("pool", entry["regen"] * (1.0 - entry["gge"]))
            prey.energy_reserve = (
                prey.energy_reserve + np.float32(share) * intake
            ).astype(env.dtype, copy=False)
        total = float(intake.sum())
        key = (entry["grazer"], entry["prey"])
        env._tick_ndm_intake[key] = env._tick_ndm_intake.get(key, 0.0) + total
        env.loss_predation[entry["prey"]] = (
            float(env.loss_predation.get(entry["prey"], 0.0)) + total)
        row = env.intake_by_pred_prey.setdefault(entry["grazer"], {})
        row[entry["prey"]] = float(row.get(entry["prey"], 0.0)) + total


def apply_layer_exchange(env):
    """Section 180: move water and algae between the mixed layer and the
    subsurface layer (entrainment / detrainment as M(t) changes, the
    diffusive exchange across the pycnocline, the reservoir below), as
    the linear per-tick map of ``nutrients.layer_tables``. Runs first in
    the population step, before grazing and growth."""
    ex = getattr(env, "layer_exchange", None)
    if not ex:
        return
    t = interactions.light_index(env)
    top, deep = env.fgs[ex["mixed"]], env.fgs[ex["subsurface"]]
    n, nd = top.energy_reserve, deep.energy_reserve
    b, bd = top.biomass, deep.biomass
    top.energy_reserve = (ex["n_mm"][t] * n + ex["n_md"][t] * nd
                          ).astype(env.dtype, copy=False)
    deep.energy_reserve = (ex["n_dm"][t] * n + ex["n_dd"][t] * nd
                           + ex["n_s"][t]).astype(env.dtype, copy=False)
    top.biomass = (ex["b_mm"][t] * b + ex["b_md"][t] * bd
                   ).astype(env.dtype, copy=False)
    deep.biomass = (ex["b_dm"][t] * b + ex["b_dd"][t] * bd
                    ).astype(env.dtype, copy=False)


def apply_population_change(env, clip_nonviable=True):
    """Growth / starvation for every FG, in ``ordered_fg_ids`` order.

    ``clip_nonviable=False`` leaves sub-threshold cells alone because the
    caller runs ``apply_extinction_threshold`` later in the tick; that is
    the monolith pipeline, where the sweep is the last step and books the
    vanished biomass as a starvation loss.
    """
    apply_layer_exchange(env)
    apply_ndm_grazing(env)
    for fg_id in env.ordered_fg_ids:
        fg = env.fgs[fg_id]
        if fg.is_decision_maker:
            _apply_decision_maker_population_change(env, fg_id, fg)
        else:
            _apply_non_decision_maker_population_change(env, fg_id, fg)
    if clip_nonviable:
        _clip_biomass_based_on_min_thresholds(env)
    _zero_energy_in_empty_cells(env)
    # Section 202: after every group's growth, as on the GPU.
    regeneration.apply(env)