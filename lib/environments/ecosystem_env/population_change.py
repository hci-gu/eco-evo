import argparse

import numpy as np


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


def _seasonal_population_rate(env, fg_id, fg):
    growth_rate = fg.growth_rate
    amplitude = float(getattr(fg, "seasonal_amplitude", 0.0) or 0.0)
    period = float(getattr(fg, "seasonal_period", 0.0) or 0.0)
    if amplitude == 0.0 or period <= 0.0:
        return growth_rate

    phase = env._season_phase.get(fg_id, 0.0)
    phase_t = (env.tick_count + phase) / period
    season = 1.0 + amplitude * float(np.sin(2.0 * np.pi * phase_t))
    return growth_rate * season


def _apply_non_decision_maker_population_change(env, fg_id, fg):
    carrying_capacity = fg.params.get("max_carrying_capacity", 100.0)
    population_rate = _seasonal_population_rate(env, fg_id, fg)
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
            + np.float32(seed_rate * carrying_capacity) * seed_mult
        )

    fg.biomass = np.clip(
        fg.biomass + biomass_delta, 0.0, carrying_capacity
    ).astype(env.dtype, copy=False)


def _apply_decision_maker_population_change(env, fg_id, fg):
    natural_mortality = float(getattr(fg, "natural_mortality", 0.0) or 0.0)
    # ``--mortality_multiplier`` scales every FG's rate by the same
    # factor; 1.0 (the default) leaves this branch bit-identical.
    natural_mortality *= float(getattr(env, "mortality_multiplier", 1.0))
    if natural_mortality > 0.0 and env.apply_natural_mortality:
        keep = np.float32(max(0.0, 1.0 - natural_mortality))
        fg.energy_reserve = (fg.energy_reserve * keep).astype(
            env.dtype, copy=False)
        fg.biomass = (fg.biomass * keep).astype(env.dtype, copy=False)

    energy_surplus = fg.energy_level - fg.maintenance_level
    growth_rate = np.float32(fg.growth_rate)
    starve_rate = (
        np.float32(fg.starve_rate)
        if fg.starve_rate > 0.0
        else growth_rate
    )
    rate = np.where(energy_surplus >= 0.0, growth_rate, starve_rate).astype(
        env.dtype, copy=False)
    biomass_delta = fg.biomass * rate * energy_surplus

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
    for fg in env.fgs.values():
        fg.energy_reserve = np.where(
                fg.biomass <= 0.0,
                np.float32(0.0),
                fg.energy_reserve,
            ).astype(env.dtype, copy=False)


def apply_population_change(env, clip_nonviable=True):
    """Growth / starvation for every FG, in ``ordered_fg_ids`` order.

    ``clip_nonviable=False`` leaves sub-threshold cells alone because the
    caller runs ``apply_extinction_threshold`` later in the tick; that is
    the monolith pipeline, where the sweep is the last step and books the
    vanished biomass as a starvation loss.
    """
    for fg_id in env.ordered_fg_ids:
        fg = env.fgs[fg_id]
        if fg.is_decision_maker:
            _apply_decision_maker_population_change(env, fg_id, fg)
        else:
            _apply_non_decision_maker_population_change(env, fg_id, fg)
    if clip_nonviable:
        _clip_biomass_based_on_min_thresholds(env)
    _zero_energy_in_empty_cells(env)