"""Open (exchange-regulated) pool of a non decision maker (section 245).

A group whose density in the box is set outside it (krill: a 60 km window
on a Skagerrak population that the water renews every ~18 days) carries
no carrying capacity. Per cell and tick, after its M1:

    B <- T(c, t) + (B (1 + g) - T(c, t)) exp(-k)

``g`` is the derived growth rate with r_max 0, so the local production
replaces exactly the booked losses (M1 + sum M2) at T and the exchange
term is zero there: the exchange sits OUTSIDE the derived g, as impacts
do, and nothing is paid twice. No logistic, no K, no clip at K.

Library keys (species):

``outside_density``
    t/km2: the outside population's annual-mean density over the box
    (equal-cell mean), the reference state of the budget.
``outside_exchange_rate``
    k per tick (the library's 6 h tick; tick_time rescales it as a flux),
    the inverse residence time of the group in the box.
``outside_season``
    Optional, 12 mid-month factors (annual mean ~1), interpolated to each
    tick on the daylight calendar like the nutrient reservoir table. A
    boundary forcing of the outside density, not a behaviour switch.
    Without the daylight calendar the factor is 1.
``outside_depth_table``
    Optional rows ``{depth_m, weight}``: the shape of the outside density
    over the cells' depth (linear between rows, held outside them, the
    ``capacity_depth_table`` rule). The loader renormalises it to an
    equal-cell mean of 1 on the run's depth map (``depth_weight_map``), so
    ``outside_density`` stays the box mean. Without a depth map: 1.

The loader refuses g >= exp(k) - 1 (the exchange could not hold the pool:
with no predation it would grow without bound).
"""
import math

import numpy as np

DENSITY_KEY = "outside_density"
RATE_KEY = "outside_exchange_rate"
SEASON_KEY = "outside_season"
DEPTH_KEY = "outside_depth_table"


def is_open(spec):
    """True when the species entry declares an open pool."""
    return isinstance(spec, dict) and spec.get(DENSITY_KEY) not in (None, "")


def parse(params, group_id=""):
    """Validated settings ``{"density", "rate", "season"}`` or None."""
    if not is_open(params):
        for key in (RATE_KEY, SEASON_KEY, DEPTH_KEY):
            if (params or {}).get(key) not in (None, "", []):
                raise ValueError(f"{group_id}: {key} needs {DENSITY_KEY} (section 245)")
        return None
    if params.get("is_decision_maker", False):
        raise ValueError(f"{group_id}: {DENSITY_KEY} is for non decision makers (section 245)")
    try:
        density = float(params[DENSITY_KEY])
        rate = float(params.get(RATE_KEY))
    except (TypeError, ValueError):
        raise ValueError(f"{group_id}: {DENSITY_KEY} and {RATE_KEY} must be numbers")
    if not (math.isfinite(density) and density > 0.0):
        raise ValueError(f"{group_id}: {DENSITY_KEY} must be > 0")
    if not (math.isfinite(rate) and rate > 0.0):
        raise ValueError(f"{group_id}: {RATE_KEY} must be > 0 (section 245)")
    season = params.get(SEASON_KEY)
    if season in (None, "", []):
        season = None
    else:
        try:
            season = [float(v) for v in season]
        except (TypeError, ValueError):
            raise ValueError(f"{group_id}: {SEASON_KEY} must be 12 numbers")
        if len(season) != 12 or not all(math.isfinite(v) and v >= 0.0 for v in season):
            raise ValueError(f"{group_id}: {SEASON_KEY} must be 12 finite factors >= 0")
    return {"density": density, "rate": rate, "season": season}


def parse_depth_table(spec):
    """(depths, weights) of ``outside_depth_table``, or None."""
    rows = (spec or {}).get(DEPTH_KEY)
    if rows in (None, "", []):
        return None
    return _parse_rows(rows, "weight", DEPTH_KEY)


def _parse_rows(rows, value_key, label):
    try:
        pairs = sorted((float(r["depth_m"]), float(r[value_key])) for r in rows)
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{label}: rows need numeric depth_m and {value_key}")
    depths = np.array([p[0] for p in pairs])
    values = np.array([p[1] for p in pairs])
    if (not np.all(np.isfinite(depths)) or not np.all(np.isfinite(values))
            or (values < 0).any() or (np.diff(depths) <= 0).any()):
        raise ValueError(f"{label}: depths must be distinct and values finite and >= 0")
    if not (values > 0).any():
        raise ValueError(f"{label}: at least one value must be > 0")
    return depths, values


def depth_weight_map(table, depth, shape):
    """The table at each cell's depth, renormalised to an equal-cell mean of
    1; ones when there is no table or no depth map."""
    if table is None or depth is None:
        return np.ones(shape, dtype=np.float64)
    depths, values = table
    raw = np.interp(np.asarray(depth, dtype=np.float64), depths, values,
                    left=values[0], right=values[-1])
    mean = float(raw.mean())
    if not mean > 0.0:
        raise ValueError("depth table is zero over every cell of the depth map")
    return raw / mean


def season_table(season, light_period, tick_hours):
    """Per year tick factor (length ``light_period``) of the 12 monthly
    factors, at each tick's midpoint on the daylight calendar."""
    from lib.world import daylight
    if season is None:
        return np.ones(int(light_period), dtype=np.float64)
    start = np.arange(int(light_period)) * float(tick_hours)
    return daylight.monthly_value(season, (start + 0.5 * float(tick_hours)) / 24.0)


def stability_error(growth_rate, rate, group_id=""):
    """Refusal text when the exchange cannot hold the pool, else None.

    Without predation the pool's fixed point is T (1 - x) / (1 - (1 + g) x),
    x = exp(-k): it exists only for (1 + g) x < 1, i.e. g < exp(k) - 1.
    """
    g, k = float(growth_rate or 0.0), float(rate)
    if g >= math.expm1(k):
        return (f"{group_id}: derived growth_rate {g:.4g} per tick is not below "
                f"exp({RATE_KEY}) - 1 = {math.expm1(k):.4g}: the exchange cannot "
                "hold the open pool (section 245)")
    return None


def build_tables(env):
    """``env.open_ids`` and, per open group, ``env.open_target[fid]`` (H, W)
    = outside density x depth weight and ``env.open_season[fid]`` (per year
    tick factor, or None without the daylight calendar)."""
    env.open_ids = []
    env.open_target = {}
    env.open_season = {}
    for fid in env.global_fg_order:
        fg = env.fgs[fid]
        settings = getattr(fg, "open_pool", None)
        if settings is None:
            continue
        weight = getattr(fg, "outside_weight_map", None)
        if weight is None:
            weight = np.ones(fg.biomass.shape, dtype=np.float64)
        env.open_ids.append(fid)
        env.open_target[fid] = (settings["density"] * np.asarray(weight, dtype=np.float64)
                                ).astype(env.dtype)
        if getattr(env, "_has_daylight", False) and settings["season"] is not None:
            env.open_season[fid] = season_table(
                settings["season"], env.light_period,
                int(env.daylight["tick_hours"])).astype(env.dtype)
        else:
            env.open_season[fid] = None


def target(env, fid):
    """T(c, t) of this tick."""
    season = env.open_season.get(fid)
    base = env.open_target[fid]
    if season is None:
        return base
    from lib.environments.ecosystem_env import interactions
    return base * season[interactions.light_index(env)]


def grow(env, fg_id, fg, rate):
    """Growth and exchange of an open pool (its M1 already removed)."""
    keep = np.float32(math.exp(-float(fg.open_pool["rate"])))
    t = target(env, fg_id)
    grown = fg.biomass * (np.float32(1.0) + np.float32(rate))
    fg.biomass = np.maximum(t + (grown - t) * keep, np.float32(0.0)).astype(
        env.dtype, copy=False)


__all__ = ["DENSITY_KEY", "DEPTH_KEY", "RATE_KEY", "SEASON_KEY", "build_tables",
           "depth_weight_map", "grow", "is_open", "parse", "parse_depth_table",
           "season_table", "stability_error", "target"]
