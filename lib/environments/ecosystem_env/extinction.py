import numpy as np


def apply_extinction_threshold(env):
    for fg_id, fg in env.fgs.items():
        min_split = float(getattr(fg, "min_split_biomass", 0.0) or 0.0)
        factor = float(getattr(fg, "extinction_threshold_factor", 0.0) or 0.0)
        if min_split <= 0.0 or factor <= 0.0:
            continue

        threshold = np.float32(factor * min_split)
        biomass = fg.biomass
        if biomass is None:
            continue

        extinct = (biomass > 0.0) & (biomass < threshold)
        if not np.any(extinct):
            continue

        lost = float(biomass[extinct].sum())
        env.loss_starvation[fg_id] = (
            float(env.loss_starvation.get(fg_id, 0.0)) + lost
        )
        env._extinction_events[fg_id] = (
            int(env._extinction_events.get(fg_id, 0)) + int(extinct.sum())
        )

        fg.biomass = np.where(extinct, np.float32(0.0), biomass).astype(
            env.dtype, copy=False)
        if fg.energy_reserve is not None:
            fg.energy_reserve = np.where(
                extinct, np.float32(0.0), fg.energy_reserve
            ).astype(env.dtype, copy=False)
