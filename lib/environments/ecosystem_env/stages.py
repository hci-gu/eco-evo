"""Juvenile compartment inside a decision-making fish group (resume 223.31,
research_notes/"Arsungarnas andel" (Swedish letters: A-ring) plan_B.md and
parameters_B.md).

A decision maker with a ``juvenile_stage`` block carries, per cell, the
biomass J of its juveniles INSIDE its biomass B (0 <= J <= B), as a refuge
group carries its accessible part. B stays the group total everywhere
(rewards, plots, K, the growth budget); J is a label on part of it, and the
energy reserve is shared pro rata, so the fill s = R/(B ME), the hunger and
the prey quality are unchanged by it.

The stage is the fish below their length at age 1 in Q3 (the size at their
second summer). Per tick:

- Recruitment and juvenile growth: new tonnes of the group (the income
  growth of an income-funded group, the positive growth delta of a
  reserve-funded one) go to J in the share
      w_J = (kappa J + rho(t) A) / (kappa J + rho(t) A + A),  A = B - J,
  kappa the per-tonne growth of a juvenile over an adult, rho(t) the
  recruitment weight per adult tonne of the month (adult-growth units).
  Read at the cell's J and B when the new tonnes are made.
- Graduation: J <- J exp(-gamma dt) after the growth, internal to B;
  gamma per year is the stage-structured biomass model's maturation rate
  (de Roos et al. 2008), dt one tick of the run's calendar.
- Losses that act on the whole group (M1, the regional capacity loss,
  starvation, the extinction sweep, the habitat mask) scale J with B.
- Fishing: each impact may carry a selectivity s_J (fishing rate on a
  juvenile tonne over the group mean). The group's loss is unchanged; J
  loses sum(bf s_J) J of it, clipped to [0, min(J, group loss)] - the clip
  bites only where s_J J/B > 1.
- Movement and currents: J moves with B (the same fractions).
- Predation: a pair ``stage_access: [f_J, f_A]`` sees f_J J + f_A (B - J)
  of the (visible) prey and takes from each part in that ratio; every
  other pair on a staged prey has f_J = f_A = its ``accessible_fraction``
  (1 by default), so the split is the prey's own J/B. The harvest cap acts
  per compartment (``predation.apply_predation``). On a prey WITHOUT a
  compartment a pair's ``stage_access`` is its one share of the whole
  group (f_J = f_A required; section 244.9), the same as an
  ``accessible_fraction``.
- Observations: every decision maker that observes a staged group (and the
  staged group itself) sees its J as one more channel after the observed
  biomasses, visible-scaled as the biomass channel is.

Open population (section 250; research_notes/"Oppen population", O-umlaut).
A block may carry ``recruit_import`` I (t/km2/yr, box mean) and
``emigration`` [e_J, e_A] (/yr). The box is an open nursery whose recruits
are spawned outside it, so recruitment is a donor-controlled supply, not
the group's own growth:

- ``recruitment`` then only shapes the import over the year (normalised
  to an annual mean of 1), and the group's own new tonnes go to J in the
  share kappa J / (kappa J + A) (rho 0).
- Per tick, after the migration (``open_pool.migrate``) and before the
  extinction sweep (``exchange``): J <- J exp(-e_J dt), A <- A exp(-e_A dt)
  (the emigrants leave with their own fill, as in M1), then I dt shape(t)
  per cell into J at the month's fill (``start_fill``, else 0.7):
  uniformly over the cells, or since section 270 by an optional
  ``recruit_depth_table`` (rows {depth_m, weight}; the loader puts the
  table at each cell's bed depth, renormalised to an equal-cell mean of
  1, on ``fg.recruit_weight_map``), so the import total stays I. It only
  places the arrivals: J then moves with B.
- The flows are booked as the exchange (``loss_exchange`` /
  ``gain_exchange``) and in the tick's net exchange that the reward ledger
  takes out (``lib/runners/energy_ledger.py``), as a migration.

With the daylight calendar rho and the start share follow the month and
gamma is converted with the run's ticks per year; without it (test and
MVP worlds) both are their annual means and the year has the library's
ticks.
"""
import math

import numpy as np

from lib.world import daylight, tick_time

KEY = "juvenile_stage"
PAIR_KEY = "stage_access"
RECRUIT_DEPTH_KEY = "recruit_depth_table"


def settings(fg):
    """Validated ``juvenile_stage`` block of ``fg`` as a dict, or None.

    Keys: ``growth_ratio`` kappa (> 0), ``graduation_rate`` gamma (/yr,
    > 0), ``recruitment`` rho by month (12 values >= 0), ``start_share``
    J/B at the start (a number or 12 monthly values in [0, 1)) and an
    optional ``fishing_selectivity`` {impact_id: s_J >= 0}.
    """
    block = (getattr(fg, "params", None) or {}).get(KEY)
    if not block:
        return None
    gid = getattr(fg, "group_id", "?")
    if not getattr(fg, "is_decision_maker", False):
        raise ValueError(f"'{gid}': {KEY} needs a decision maker")

    def number(name, lo, hi, lo_open=True):
        try:
            value = float(block[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{gid}': {KEY}.{name} is required and must be a number")
        ok = math.isfinite(value) and (value > lo if lo_open else value >= lo) and value < hi
        if not ok:
            raise ValueError(f"'{gid}': {KEY}.{name} out of range, got {block[name]!r}")
        return value

    def monthly(name, lo, hi):
        raw = block.get(name)
        if isinstance(raw, (int, float)):
            raw = [raw] * 12
        try:
            values = [float(v) for v in raw]
        except (TypeError, ValueError):
            raise ValueError(f"'{gid}': {KEY}.{name} must be a number or 12 numbers")
        if len(values) != 12 or not all(math.isfinite(v) and lo <= v < hi for v in values):
            raise ValueError(f"'{gid}': {KEY}.{name} must be 12 values in [{lo}, {hi})")
        return np.asarray(values, dtype=np.float64)

    selectivity = {}
    for impact_id, raw in (block.get("fishing_selectivity") or {}).items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"'{gid}': {KEY}.fishing_selectivity.{impact_id} must be >= 0")
        selectivity[str(impact_id)] = value
    recruit_import = block.get("recruit_import")
    if recruit_import in (None, ""):
        recruit_import = None
    else:
        try:
            recruit_import = float(recruit_import)
        except (TypeError, ValueError):
            recruit_import = float("nan")
        if not (math.isfinite(recruit_import) and recruit_import >= 0.0):
            raise ValueError(f"'{gid}': {KEY}.recruit_import must be >= 0 (t/km2/yr)")
    depth_table = block.get(RECRUIT_DEPTH_KEY)
    if depth_table in (None, "", []):
        depth_table = None
    else:
        # Section 270: bed-depth shape of the recruit import (rows {depth_m,
        # weight}; the loader renormalises it to an equal-cell mean of 1).
        if recruit_import is None:
            raise ValueError(f"'{gid}': {KEY}.{RECRUIT_DEPTH_KEY} needs recruit_import")
        from lib.environments.ecosystem_env import open_pool
        depth_table = open_pool._parse_rows(depth_table, "weight",
                                            f"'{gid}': {KEY}.{RECRUIT_DEPTH_KEY}")
    emigration = block.get("emigration")
    if emigration in (None, "", []):
        emigration = None
    else:
        try:
            emigration = [float(v) for v in emigration]
        except (TypeError, ValueError):
            emigration = []
        if len(emigration) != 2 or not all(math.isfinite(v) and v >= 0.0 for v in emigration):
            raise ValueError(f"'{gid}': {KEY}.emigration must be [e_J, e_A], each >= 0 (/yr)")
    return {
        "import": recruit_import,
        "depth_table": depth_table,
        "emigration": emigration,
        "kappa": number("growth_ratio", 0.0, math.inf),
        "gamma": number("graduation_rate", 0.0, math.inf),
        "rho": monthly("recruitment", 0.0, math.inf),
        "start": monthly("start_share", 0.0, 1.0),
        "selectivity": selectivity,
    }


def month_table(tick_hours):
    """(T,) 0-based month of every year tick of the calendar."""
    hours = int(tick_hours)
    n = daylight.ticks_per_year(hours)
    days = (np.arange(n) * hours) // 24
    return np.array([daylight.month_of_day(d) for d in days], dtype=np.int64)


def monthly_on_ticks(values, tick_hours):
    """A 12-month table read on every year tick, linear between month
    midpoints (cyclic)."""
    hours = int(tick_hours)
    n = daylight.ticks_per_year(hours)
    mid = (np.arange(n) * hours + 0.5 * hours) / 24.0
    return daylight.monthly_value(np.asarray(values, dtype=np.float64), mid)


def pair_stage_access(inter_def, inter_id=""):
    """``(f_J, f_A)`` of a pair's ``stage_access``, or None when unset."""
    raw = (inter_def or {}).get(PAIR_KEY)
    if raw in (None, "", []):
        return None
    try:
        f_j, f_a = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise ValueError(f"{inter_id}: {PAIR_KEY} must be two numbers [f_J, f_A]")
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (f_j, f_a)) or f_j + f_a <= 0.0:
        raise ValueError(f"{inter_id}: {PAIR_KEY} values must be in [0, 1] and not both 0")
    if (inter_def or {}).get("accessible_fraction") not in (None, ""):
        raise ValueError(f"{inter_id}: give either accessible_fraction or {PAIR_KEY}")
    return f_j, f_a


def unstaged_share(pair, inter_id=""):
    """The one share a ``stage_access`` pair has on a prey WITHOUT a
    juvenile compartment (section 244.9): the prey is one pool, so the
    pair sees one share of it, and f_J must equal f_A (a juvenile window
    with no juveniles to act on would be silently dropped)."""
    f_j, f_a = pair
    if not math.isclose(f_j, f_a, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{inter_id}: the prey has no {KEY}, so {PAIR_KEY} is one share "
            f"of the whole group: give f_J = f_A (got [{f_j:g}, {f_a:g}])")
    if not f_a > 0.0:
        raise ValueError(f"{inter_id}: {PAIR_KEY} on a prey without a {KEY} "
                         "must be above 0")
    return f_a


def build_tables(env):
    """Stage tables on ``env``.

    ``env._has_stages``; ``env.stage_rows`` (columns in ``global_fg_order``);
    per column (N_all): ``stage_kappa``, ``stage_keep`` (graduation survival
    per tick), ``stage_rho`` (T, N_all) by year tick, ``stage_start`` (T,
    N_all); ``stage_f_j`` / ``stage_f_a`` (N_dm, N_all) per pair (the
    ``accessible_fraction`` for an unset pair) and ``juvenile_selectivity``
    {fg_id: {impact_id: s_J}}. Must run after the interaction matrices
    (``env.access_mat``, ``env.eat_static_mask``) and the daylight tables.
    """
    env.stage_rows = []
    env.stage_exchange_ids = []
    env.stage_import = None
    env.stage_import_map = None
    env.stage_fill = None
    env.stage_emig_keep = np.ones((2, env.N_all), dtype=env.dtype)
    env.stage_kappa = np.zeros(env.N_all, dtype=env.dtype)
    env.stage_keep = np.ones(env.N_all, dtype=env.dtype)
    env.juvenile_selectivity = {}
    found = {}
    for j, fid in enumerate(env.global_fg_order):
        spec = settings(env.fgs[fid])
        if spec is not None:
            found[j] = spec
    env._has_stages = bool(found)
    env.stage_f_j = np.array(env.access_mat, dtype=env.dtype, copy=True)
    env.stage_f_a = np.array(env.access_mat, dtype=env.dtype, copy=True)
    for i, pred_id in enumerate(env.dm_ids):
        interaction = env.fgs[pred_id].params.get("interaction", {}) or {}
        for j, prey_id in enumerate(env.global_fg_order):
            inter_id = f"{pred_id}_preys_on_{prey_id}"
            pair = pair_stage_access(interaction.get(inter_id), inter_id)
            if pair is None or env.eat_static_mask[i, j] <= 0.0:
                continue
            if j not in found:
                # Section 244.9: one pool, one share; already in
                # ``access_mat`` (interactions.build_interaction_matrices).
                continue
            env.stage_f_j[i, j], env.stage_f_a[i, j] = pair
    if not found:
        env.stage_rho = None
        env.stage_start = None
        return
    calendar = getattr(env, "daylight", None)
    if calendar:
        hours = int(calendar["tick_hours"])
        per_year = daylight.ticks_per_year(hours)
        months = month_table(hours)
        rows = per_year
    else:
        # No calendar: annual means, the library's tick length.
        per_year = tick_time.ticks_per_year(tick_time.LIBRARY_TICK_HOURS)
        rows = 1
    env.stage_rho = np.zeros((rows, env.N_all), dtype=env.dtype)
    env.stage_start = np.zeros((rows, env.N_all), dtype=env.dtype)
    env.stage_import = np.zeros((rows, env.N_all), dtype=env.dtype)
    # Section 270: the import's per-cell shape (equal-cell mean 1; ones
    # without a recruit_depth_table or without a depth map).
    env.stage_import_map = np.ones((env.N_all, env.H, env.W), dtype=env.dtype)
    env.stage_fill = np.zeros((rows, env.N_all), dtype=env.dtype)
    for j, spec in found.items():
        fid = env.global_fg_order[j]
        env.stage_rows.append(j)
        env.stage_kappa[j] = spec["kappa"]
        env.stage_keep[j] = math.exp(-spec["gamma"] / per_year)
        if calendar:
            env.stage_rho[:, j] = spec["rho"][months]
            env.stage_start[:, j] = monthly_on_ticks(spec["start"], hours)
        else:
            env.stage_rho[0, j] = float(spec["rho"].mean())
            env.stage_start[0, j] = float(spec["start"].mean())
        env.juvenile_selectivity[fid] = dict(spec["selectivity"])
        if spec["import"] is not None or spec["emigration"] is not None:
            _open_population_tables(env, j, spec, per_year, months if calendar else None,
                                    hours if calendar else None)
    env.stage_rows = np.asarray(env.stage_rows, dtype=np.int64)


def _open_population_tables(env, j, spec, per_year, months, hours):
    """Section 250: the import per cell and tick, the emigration survival
    per tick and the arrivals' fill, for one staged column ``j``."""
    fid = env.global_fg_order[j]
    env.stage_exchange_ids.append(fid)
    rho = np.asarray(spec["rho"], dtype=np.float64)
    shape = rho / rho.mean() if rho.mean() > 0.0 else np.ones(12)
    imported = float(spec["import"] or 0.0) / per_year
    if spec["import"] is not None:
        # The import replaces the group's own recruitment (rho 0).
        env.stage_rho[:, j] = 0.0
    if months is not None:
        # Normalised on the ticks (months differ in length): the year's
        # import is exactly I per cell.
        per_tick = shape[months]
        env.stage_import[:, j] = imported * per_tick / per_tick.mean()
    else:
        env.stage_import[0, j] = imported
    weight = getattr(env.fgs[fid], "recruit_weight_map", None)
    if spec["import"] is not None and weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (env.H, env.W):
            raise ValueError(f"'{fid}': recruit weight map {weight.shape} does not match the "
                             f"grid {(env.H, env.W)}")
        env.stage_import_map[j] = (weight / weight.mean()).astype(env.dtype)
    e_j, e_a = spec["emigration"] or (0.0, 0.0)
    env.stage_emig_keep[0, j] = math.exp(-e_j / per_year)
    env.stage_emig_keep[1, j] = math.exp(-e_a / per_year)
    fill = getattr(env.fgs[fid], "start_fill", None)
    if fill is None:
        env.stage_fill[:, j] = 0.7
    elif hours is not None:
        env.stage_fill[:, j] = monthly_on_ticks(fill, hours)
    else:
        env.stage_fill[0, j] = float(np.mean(fill))


def exchange(env):
    """Section 250: each open-population group's emigration and recruit
    import (after ``open_pool.migrate``, before the extinction sweep)."""
    ids = getattr(env, "stage_exchange_ids", None)
    if not ids:
        return
    t = year_tick(env)
    moving = set(getattr(env, "open_dm_ids", ()) or ())
    for fid in ids:
        fg = env.fgs[fid]
        j = env.global_fg_order.index(fid)
        juvenile = np.minimum(fg.juvenile, fg.biomass)
        adult = np.maximum(fg.biomass - juvenile, np.float32(0.0))
        new_j = juvenile * env.stage_emig_keep[0, j]
        new_b = new_j + adult * env.stage_emig_keep[1, j]
        ratio = np.where(fg.biomass > 0.0, new_b / np.where(fg.biomass > 0.0, fg.biomass,
                                                           np.float32(1.0)), np.float32(0.0))
        reserve_before = float(fg.energy_reserve.astype(np.float64).sum())
        export = float((fg.biomass - new_b).astype(np.float64).sum())
        fg.energy_reserve = (fg.energy_reserve * ratio).astype(env.dtype, copy=False)
        leaving = reserve_before - float(fg.energy_reserve.astype(np.float64).sum())
        # Section 270: the per-cell shape of the import (ones = uniform).
        per_cell = np.float32(env.stage_import[t, j]) * env.stage_import_map[j]
        fill = float(env.stage_fill[t, j])
        me = float(fg.max_energy_reserve)
        fg.biomass = (new_b + per_cell).astype(env.dtype, copy=False)
        fg.juvenile = (new_j + per_cell).astype(env.dtype, copy=False)
        fg.energy_reserve = (fg.energy_reserve + per_cell * np.float32(fill * me)).astype(
            env.dtype, copy=False)
        imported = float(per_cell.astype(np.float64).sum())
        arriving = imported * fill * me
        for book, amount in (("loss_exchange", export), ("gain_exchange", imported)):
            books = getattr(env, book, None)
            if books is not None:
                books[fid] = float(books.get(fid, 0.0)) + amount
        ec = float(fg.params.get("energy_content", 0.0) or 0.0)
        base = (lambda name: float(getattr(env, name).get(fid, 0.0))) if fid in moving else (
            lambda name: 0.0)
        env.tick_exchange_biomass[fid] = base("tick_exchange_biomass") + imported - export
        env.tick_exchange_reserve[fid] = base("tick_exchange_reserve") + arriving - leaving
        env.tick_exchange_energy[fid] = (base("tick_exchange_energy")
                                         + (imported - export) * ec + arriving - leaving)


def is_staged(env, fg_id):
    return bool(getattr(env, "_has_stages", False)) and fg_id in getattr(
        env, "juvenile_selectivity", {})


def initialise(env):
    """Every staged group's J at its start share of the start day; every
    other group carries None. Called when the environment is built."""
    for fid, fg in env.fgs.items():
        fg.juvenile = None
    if not getattr(env, "_has_stages", False):
        return
    t = 0 if env.stage_start.shape[0] == 1 else (
        int(getattr(env, "light_start", 0)) % env.stage_start.shape[0])
    for j in env.stage_rows:
        fg = env.fgs[env.global_fg_order[j]]
        fg.juvenile = (fg.biomass * env.stage_start[t, j]).astype(env.dtype)


def year_tick(env):
    if env.stage_rho.shape[0] == 1:
        return 0
    return (int(getattr(env, "light_start", 0)) + int(env.tick_count)) % env.stage_rho.shape[0]


def growth_share(env, j, biomass, juvenile):
    """Share w_J of a cell's new tonnes that goes to J (column ``j``)."""
    kappa = env.stage_kappa[j]
    rho = env.stage_rho[year_tick(env), j]
    adult = np.maximum(biomass - juvenile, np.float32(0.0))
    weight_j = kappa * juvenile + rho * adult
    total = weight_j + adult
    return np.where(total > 0.0, weight_j / np.where(total > 0.0, total, np.float32(1.0)),
                    np.float32(0.0)).astype(env.dtype)


def clamp(fg, dtype):
    """Keep 0 <= J <= B (float round-off) and J = 0 in an empty cell."""
    if getattr(fg, "juvenile", None) is None:
        return
    fg.juvenile = np.clip(fg.juvenile, 0.0, np.maximum(fg.biomass, 0.0)).astype(dtype, copy=False)


def scale(fg, factor, dtype):
    """J scaled by the same per-cell factor as B."""
    if getattr(fg, "juvenile", None) is None:
        return
    fg.juvenile = (fg.juvenile * factor).astype(dtype, copy=False)
