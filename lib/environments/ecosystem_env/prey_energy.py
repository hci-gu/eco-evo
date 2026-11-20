"""Seasonal energy content of a prey group (section 295).

Library key (species, non decision makers only):

``energy_season``
    12 mid-month factors (annual mean ~1) on the group's ``energy_content``
    as prey: in month t a tonne eaten carries E f(t), so every predator's
    energy gain per tonne and its handling time per tonne (h = E / ceiling,
    the library's convention) both follow f. ``energy_content`` stays the
    annual mean, which the growth budget, the diet gates and every static
    diagnostic keep reading. Krill: ~6 kJ/g WW in Q4-Q1 (lipid-rich adults),
    ~4.3 in Q2-Q3 (after spawning, the young of the year).

The benthivores' per-prey ceilings scale with E^0.25 (``Magtomning och
energitak``), so their h strictly follows f^0.75; one rule, h x f, is used
for every predator (on krill, 4 % of the benthivores' diet, the difference
is at most 4 % of h in the extreme months).

Interpolated to each tick on the daylight calendar like ``outside_season``;
inert (factor 1) without it. Refused on a decision maker, whose
``energy_content`` is also the price of its own tonnes.
"""
import math

import numpy as np

KEY = "energy_season"
MEAN_TOL = 0.05


def parse(params, group_id=""):
    """The 12 factors, or None."""
    season = (params or {}).get(KEY)
    if season in (None, "", []):
        return None
    if bool((params or {}).get("is_decision_maker", False)):
        raise ValueError(f"{group_id}: {KEY} is for non decision makers (section 295): a "
                         "decision maker's energy_content is also the price of its tonnes")
    try:
        season = [float(v) for v in season]
    except (TypeError, ValueError):
        raise ValueError(f"{group_id}: {KEY} must be 12 numbers")
    if len(season) != 12 or not all(math.isfinite(v) and v > 0.0 for v in season):
        raise ValueError(f"{group_id}: {KEY} must be 12 finite factors > 0")
    if abs(sum(season) / 12.0 - 1.0) > MEAN_TOL:
        raise ValueError(f"{group_id}: {KEY} must have a mean of ~1 (energy_content is the "
                         f"annual mean), got {sum(season) / 12.0:.3f}")
    return season


def build_table(env):
    """``env._has_prey_energy`` and ``env.prey_energy_table`` ((T, N_all)
    factor per year tick and prey, 1 for a prey without the key; None when
    no group has one or there is no daylight calendar)."""
    from lib.environments.ecosystem_env import open_pool
    env._has_prey_energy = False
    env.prey_energy_table = None
    seasons = {fid: getattr(env.fgs[fid], "energy_season", None) for fid in env.global_fg_order}
    if not any(v is not None for v in seasons.values()):
        return
    if not getattr(env, "_has_daylight", False):
        return
    hours = int(env.daylight["tick_hours"])
    table = np.ones((env.light_period, env.N_all), dtype=np.float64)
    for j, fid in enumerate(env.global_fg_order):
        if seasons[fid] is not None:
            table[:, j] = open_pool.season_table(seasons[fid], env.light_period, hours)
    env.prey_energy_table = table.astype(env.dtype)
    env._has_prey_energy = True


def factor(env):
    """``(N_all,)`` factor of this tick, or None."""
    if not getattr(env, "_has_prey_energy", False):
        return None
    from lib.environments.ecosystem_env import interactions
    return env.prey_energy_table[interactions.light_index(env)]


__all__ = ["KEY", "build_table", "factor", "parse"]
