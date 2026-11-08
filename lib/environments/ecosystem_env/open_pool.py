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

Decision makers (section 246: mackerel's seasonal migration). A DM with
``outside_density`` migrates on its GROUP TOTAL, never per cell, so the
policy's spatial choices stand. Per tick, after the DM growth and before
the extinction sweep:

    net = (T - B_tot) (1 - exp(-k)),  T = density x N_cells x season(t)

A negative net (export) scales every cell's B and reserve by
1 - export / B_tot: the emigrants leave with their own fill. A positive net
(import) enters on the entry cells (``outside_entry``) with the reserve
``import x fill(t) x max_energy_reserve``. Keys, DM only:

``outside_fill``
    12 mid-month reserve fills s of the arriving fish (lean after spawning,
    fat in autumn), interpolated like the season.
``outside_entry``
    Edge segments the immigrants enter on: rows ``{edge: west}`` or
    ``{edge: north, from: 0.0, to: 0.5}`` (row 0 is north, column 0 west;
    ``from`` / ``to`` the share along the edge, west -> east or north ->
    south). Uniform over the selected edge cells.

The export is ``loss_exchange``, the import ``gain_exchange``, and the net
exchange energy of the tick ``env.tick_exchange_energy[fid]`` (MJ), which
the trainers' reward ledger takes out of the group's energy: the policy is
rewarded for its local production and losses, not for the migration.
``outside_depth_table`` is for NDMs.

A DM with a ``juvenile_stage`` (section 250: the planktivores, a window on
mobile stocks) migrates on its ADULTS: ``outside_density`` is the adults'
box mean, the net is (T - A_tot)(1 - exp(-k)) with A = B - J, the export
scales A only (each cell's reserve pro rata with its biomass, J and A
sharing one fill) and the import enters A. The juveniles' own exchange is
``stages.exchange`` (recruit import and emigration).
"""
import math

import numpy as np

DENSITY_KEY = "outside_density"
RATE_KEY = "outside_exchange_rate"
SEASON_KEY = "outside_season"
DEPTH_KEY = "outside_depth_table"
FILL_KEY = "outside_fill"
ENTRY_KEY = "outside_entry"
EDGES = ("north", "south", "west", "east")
SEASON_MEAN_TOL = 0.05


def is_open(spec):
    """True when the species entry declares an open pool."""
    return isinstance(spec, dict) and spec.get(DENSITY_KEY) not in (None, "")


def parse(params, group_id=""):
    """Validated settings ``{"density", "rate", "season"}`` or None."""
    if not is_open(params):
        for key in (RATE_KEY, SEASON_KEY, DEPTH_KEY, FILL_KEY, ENTRY_KEY):
            if (params or {}).get(key) not in (None, "", []):
                raise ValueError(f"{group_id}: {key} needs {DENSITY_KEY} (section 245)")
        return None
    is_dm = bool(params.get("is_decision_maker", False))
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
        if abs(sum(season) / 12.0 - 1.0) > SEASON_MEAN_TOL:
            raise ValueError(f"{group_id}: {SEASON_KEY} must have a mean of ~1 (it scales "
                             f"the annual-mean {DENSITY_KEY}), got {sum(season) / 12.0:.3f}")
    out = {"density": density, "rate": rate, "season": season}
    if not is_dm:
        for key in (FILL_KEY, ENTRY_KEY):
            if params.get(key) not in (None, "", []):
                raise ValueError(f"{group_id}: {key} is for decision makers (section 246)")
        return out
    if params.get(DEPTH_KEY) not in (None, "", []):
        raise ValueError(f"{group_id}: {DEPTH_KEY} is for non decision makers; a migrating "
                         f"decision maker enters on {ENTRY_KEY} (section 246)")
    try:
        fill = [float(v) for v in params.get(FILL_KEY)]
    except (TypeError, ValueError):
        raise ValueError(f"{group_id}: {FILL_KEY} must be 12 reserve fills (section 246)")
    if len(fill) != 12 or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in fill):
        raise ValueError(f"{group_id}: {FILL_KEY} must be 12 fills in [0, 1] (section 246)")
    out["fill"] = fill
    out["entry"] = parse_entry(params.get(ENTRY_KEY), group_id)
    return out


def parse_entry(rows, group_id=""):
    """Validated edge segments [(edge, from, to)] of ``outside_entry``."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f"{group_id}: {ENTRY_KEY} needs at least one edge segment (section 246)")
    out = []
    for row in rows:
        if isinstance(row, str):
            row = {"edge": row}
        if not isinstance(row, dict):
            raise ValueError(f"{group_id}: {ENTRY_KEY} rows are {{edge, from, to}}")
        edge = str(row.get("edge", "")).strip().lower()
        if edge not in EDGES:
            raise ValueError(f"{group_id}: {ENTRY_KEY} edge must be one of {EDGES}, got {edge!r}")
        try:
            lo, hi = float(row.get("from", 0.0)), float(row.get("to", 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"{group_id}: {ENTRY_KEY} from / to must be numbers")
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"{group_id}: {ENTRY_KEY} needs 0 <= from < to <= 1")
        out.append((edge, lo, hi))
    return out


def entry_weights(entry, shape):
    """Uniform weights over the entry cells (sum 1); row 0 north, col 0 west."""
    H, W = int(shape[0]), int(shape[1])
    mask = np.zeros((H, W), dtype=bool)
    for edge, lo, hi in entry:
        n = W if edge in ("north", "south") else H
        a, b = int(np.floor(lo * n)), max(int(np.ceil(hi * n)), int(np.floor(lo * n)) + 1)
        idx = slice(a, min(b, n))
        if edge == "north":
            mask[0, idx] = True
        elif edge == "south":
            mask[H - 1, idx] = True
        elif edge == "west":
            mask[idx, 0] = True
        else:
            mask[idx, W - 1] = True
    return mask.astype(np.float64) / float(mask.sum())


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
    # Section 246: migrating decision makers.
    env.open_dm_ids = []
    env.open_dm_total = {}
    env.open_dm_season = {}
    env.open_dm_fill = {}
    env.open_dm_entry = {}
    env.tick_exchange_energy = {}
    env.tick_exchange_biomass = {}
    env.tick_exchange_reserve = {}
    daylight_on = getattr(env, "_has_daylight", False)
    for fid in env.global_fg_order:
        fg = env.fgs[fid]
        settings = getattr(fg, "open_pool", None)
        if settings is None:
            continue
        if fg.is_decision_maker:
            shape = fg.biomass.shape
            env.open_dm_ids.append(fid)
            env.open_dm_total[fid] = settings["density"] * float(np.prod(shape))
            env.open_dm_entry[fid] = entry_weights(settings["entry"], shape).astype(env.dtype)
            if daylight_on:
                hours = int(env.daylight["tick_hours"])
                env.open_dm_season[fid] = (season_table(settings["season"], env.light_period, hours)
                                           if settings["season"] is not None else None)
                env.open_dm_fill[fid] = season_table(settings["fill"], env.light_period, hours)
            else:
                env.open_dm_season[fid] = None
                env.open_dm_fill[fid] = float(np.mean(settings["fill"]))
            env.tick_exchange_energy[fid] = 0.0
            env.tick_exchange_biomass[fid] = 0.0
            env.tick_exchange_reserve[fid] = 0.0
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
    # The exchange per direction (the books of loss_accounting).
    flow = fg.biomass.astype(np.float64) - grown.astype(np.float64)
    for book, amount in (("loss_exchange", -np.minimum(flow, 0.0).sum()),
                         ("gain_exchange", np.maximum(flow, 0.0).sum())):
        books = getattr(env, book, None)
        if books is not None:
            books[fg_id] = float(books.get(fg_id, 0.0)) + float(amount)


def migrate(env):
    """Section 246: every migrating DM's exchange with the outside, on its
    group total (after the DM growth, before the extinction sweep)."""
    if not getattr(env, "open_dm_ids", None):
        return
    from lib.environments.ecosystem_env import interactions
    t = interactions.light_index(env) if getattr(env, "_has_daylight", False) else None
    for fid in env.open_dm_ids:
        fg = env.fgs[fid]
        settings = fg.open_pool
        season = env.open_dm_season.get(fid)
        target = env.open_dm_total[fid] * (float(season[t]) if season is not None else 1.0)
        fill = env.open_dm_fill[fid]
        fill = float(fill[t]) if t is not None and not np.isscalar(fill) else float(fill)
        juvenile = getattr(fg, "juvenile", None)
        adult = fg.biomass if juvenile is None else np.maximum(
            fg.biomass - np.minimum(juvenile, fg.biomass), np.float32(0.0))
        total = float(adult.astype(np.float64).sum())
        net = (target - total) * (1.0 - math.exp(-float(settings["rate"])))
        ec = float(fg.params.get("energy_content", 0.0) or 0.0)
        me = float(fg.max_energy_reserve)
        if net < 0.0:
            share = min(-net / total, 1.0) if total > 0.0 else 0.0
            reserve_before = float(fg.energy_reserve.astype(np.float64).sum())
            if juvenile is None:
                keep = np.float32(1.0 - share)
                fg.biomass = (fg.biomass * keep).astype(env.dtype, copy=False)
                fg.energy_reserve = (fg.energy_reserve * keep).astype(env.dtype, copy=False)
            else:
                new_b = fg.biomass - adult * np.float32(share)
                ratio = np.where(fg.biomass > 0.0, new_b / np.where(
                    fg.biomass > 0.0, fg.biomass, np.float32(1.0)), np.float32(0.0))
                fg.biomass = new_b.astype(env.dtype, copy=False)
                fg.energy_reserve = (fg.energy_reserve * ratio).astype(env.dtype, copy=False)
            leaving_reserve = reserve_before - float(fg.energy_reserve.astype(np.float64).sum())
            export, imported = share * total, 0.0
            reserve_flow = -leaving_reserve
        else:
            weights = env.open_dm_entry[fid]
            fg.biomass = (fg.biomass + np.float32(net) * weights).astype(env.dtype, copy=False)
            fg.energy_reserve = (fg.energy_reserve
                                 + np.float32(net * fill * me) * weights).astype(env.dtype, copy=False)
            export, imported = 0.0, net
            reserve_flow = net * fill * me
        for book, amount in (("loss_exchange", export), ("gain_exchange", imported)):
            books = getattr(env, book, None)
            if books is not None:
                books[fid] = float(books.get(fid, 0.0)) + float(amount)
        # The tick's net exchange, for the reward ledger (energy_ledger).
        env.tick_exchange_biomass[fid] = float(imported - export)
        env.tick_exchange_reserve[fid] = float(reserve_flow)
        env.tick_exchange_energy[fid] = float((imported - export) * ec + reserve_flow)


__all__ = ["DENSITY_KEY", "DEPTH_KEY", "ENTRY_KEY", "FILL_KEY", "RATE_KEY", "SEASON_KEY",
           "build_tables", "depth_weight_map", "entry_weights", "grow", "is_open",
           "migrate", "parse", "parse_depth_table", "parse_entry", "season_table",
           "stability_error", "target"]
