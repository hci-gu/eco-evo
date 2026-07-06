import numpy as np


def _seasonal_growth_rate(env, fg_id, fg):
    growth_rate = fg.growth_rate
    amplitude = float(getattr(fg, "seasonal_amplitude", 0.0) or 0.0)
    period = float(getattr(fg, "seasonal_period", 0.0) or 0.0)
    if amplitude == 0.0 or period <= 0.0:
        return growth_rate

    phase = env._season_phase.get(fg_id, 0.0)
    phase_t = (env.tick_count + phase) / period
    season = 1.0 + amplitude * float(np.sin(2.0 * np.pi * phase_t))
    return growth_rate * season


def _apply_non_decision_maker_growth(env, fg_id, fg):
    carrying_capacity = fg.params.get("max_carrying_capacity", 100.0)
    growth_rate = _seasonal_growth_rate(env, fg_id, fg)
    growth = (
        growth_rate
        * fg.biomass
        * (1.0 - fg.biomass / (carrying_capacity + 1e-9))
    )

    seed_rate = float(getattr(fg, "seed_rate", 0.0) or 0.0)
    if seed_rate > 0.0:
        seed_mult = np.power(
            10.0,
            np.random.uniform(-1.0, 1.0, size=fg.biomass.shape),
        ).astype(np.float32, copy=False)
        growth = growth + np.float32(seed_rate * carrying_capacity) * seed_mult

    fg.biomass = np.clip(
        fg.biomass + growth, 0.0, carrying_capacity
    ).astype(env.dtype, copy=False)


def _apply_decision_maker_growth(env, fg_id, fg):
    natural_mortality = float(getattr(fg, "natural_mortality", 0.0) or 0.0)
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
    growth = fg.biomass * rate * energy_surplus

    total_loss = -np.minimum(0.0, growth)
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
    fg.biomass = np.maximum(0.0, fg.biomass + growth).astype(
        env.dtype, copy=False)


def apply_growth(env):
    for fg_id in env.ordered_fg_ids:
        fg = env.fgs[fg_id]
        if fg.is_decision_maker:
            _apply_decision_maker_growth(env, fg_id, fg)
        else:
            _apply_non_decision_maker_growth(env, fg_id, fg)
