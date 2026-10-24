"""Depth habitat of a hand-set non decision maker (section 207).

The benthos had one carrying capacity for the whole box (10 t/km2) and a
Perlin start pattern, while the stock measured inside the grid falls
from ~100-200 g WW/m2 at 100 m to ~50-85 at 300 m (Josefson 1990) and
the box spans 44-529 m. Three pieces put the depth into it:

``simulation_settings.depth_map`` (manifest)
    A .npz with one 2-D field, the mean depth per cell in metres
    (tools/make_depth_map.py -> maps/depth.npz), relative to the project
    file or absolute, nearest-neighbour resampled to the run's grid like
    an impact map. It is also the spawn reference field ``depth``.
``capacity_depth_table`` (species, library)
    Rows ``{depth_m, capacity}``: K(c) in t/km2 by depth, linear between
    rows and held at the end rows outside them (the impact-table rule).
    Needs the depth map; without one the scalar ``max_carrying_capacity``
    applies everywhere, as before.
``spawn: {mode: equilibrium}`` (species, library)
    The start pattern is today's equilibrium of the logistic under
    today's losses, cell by cell:

        B*(c) = K(c) * max(0, 1 - (F(c) + M2) / r)

    r = the hand-set growth_rate per year, F(c) = the yearly loss of the
    group's impacts read off their Impact Editor baseline maps (today's
    fishing; never the inference scenario maps, so a scenario starts
    from today and recovers from there), M2 = the checked predation
    pairs' ``predation_mortality`` on the group. The total still comes
    from the manifest (training range, inference value); B* only sets
    the pattern. Without the habitat (no depth map, no baseline map) the
    pattern is K and F as they are, i.e. uniform for a scalar K.

Everything is computed from the library's own numbers at
LIBRARY_TICK_HOURS, so the maps do not depend on ``--tick-length``.
"""
import os

import numpy as np

from lib.world.tick_time import LIBRARY_TICK_HOURS, ticks_per_year

CAPACITY_TABLE_KEY = "capacity_depth_table"
SPAWN_MODE = "equilibrium"


def depth_map_path(project, project_path=None):
    """Absolute path of the manifest's depth map, or None."""
    settings = (project or {}).get("simulation_settings") or {}
    stored = settings.get("depth_map")
    if not isinstance(stored, str) or not stored:
        return None
    if os.path.isabs(stored):
        return stored
    base = os.path.dirname(os.path.abspath(project_path)) if project_path else os.getcwd()
    return os.path.normpath(os.path.join(base, stored))


def load_depth_map(project, project_path, grid_size, verbose=False):
    """The manifest's depth map on ``grid_size`` (float64, m), or None.

    A configured map that cannot be read is an error, not a silent
    fallback: the capacity and the start pattern depend on it.
    """
    path = depth_map_path(project, project_path)
    if path is None:
        return None
    from lib.world.impact_maps import load_impact_map_npz
    H, W = int(grid_size[0]), int(grid_size[1])
    field = load_impact_map_npz(path, "depth", H, W, verbose=verbose)
    if field is None:
        raise ValueError(f"simulation_settings.depth_map: cannot read {path}")
    field = np.asarray(field, dtype=np.float64)
    if not np.all(np.isfinite(field)) or (field < 0).any():
        raise ValueError(f"{path}: depth must be finite metres >= 0")
    return field


def parse_capacity_table(spec):
    """(depths, capacities) of a species' ``capacity_depth_table``, or None."""
    rows = (spec or {}).get(CAPACITY_TABLE_KEY)
    if rows in (None, "", []):
        return None
    try:
        pairs = sorted((float(r["depth_m"]), float(r["capacity"])) for r in rows)
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{CAPACITY_TABLE_KEY}: rows need numeric depth_m and capacity")
    depths = np.array([p[0] for p in pairs])
    caps = np.array([p[1] for p in pairs])
    if (not np.all(np.isfinite(depths)) or not np.all(np.isfinite(caps))
            or (caps < 0).any() or (np.diff(depths) <= 0).any()):
        raise ValueError(f"{CAPACITY_TABLE_KEY}: depths must be distinct and "
                         "capacities finite and >= 0")
    return depths, caps


def capacity_map(table, depth):
    """K per cell: the table interpolated at each cell's depth."""
    depths, caps = table
    return np.interp(np.asarray(depth, dtype=np.float64), depths, caps,
                     left=caps[0], right=caps[-1])


def annual_impact_loss(fg_id, library, baseline_maps, grid_size):
    """F(c) per year: the group's impact tables at today's baseline maps."""
    from lib.environments.ecosystem_env.impacts import extract_impact_table, interp_impact
    H, W = int(grid_size[0]), int(grid_size[1])
    loss = np.zeros((H, W), dtype=np.float64)
    # The <group>_impacted_by_<impact> tables live with the pairs.
    defs = (library or {}).get("interaction_definitions") or {}
    n = ticks_per_year(LIBRARY_TICK_HOURS)
    for impact_id, field in (baseline_maps or {}).items():
        table = extract_impact_table(defs.get(f"{fg_id}_impacted_by_{impact_id}"))
        if table is None:
            continue
        per_tick, _ = interp_impact(table, np.asarray(field, dtype=np.float32))
        loss += n * np.clip(np.asarray(per_tick, dtype=np.float64), 0.0, None)
    return loss


def predation_m2(fg_id, library, active_ids):
    """Sum of the checked pairs' predation_mortality on ``fg_id`` per year."""
    from lib.world.growth_budget import active_predation
    return float(sum(active_predation(
        fg_id, (library or {}).get("species_definitions"),
        (library or {}).get("interaction_definitions"),
        active_predators=active_ids).values()))


def equilibrium_map(capacity, loss, m2, r_annual):
    """B*(c) = K(c) max(0, 1 - (F(c) + M2)/r); zeros when r <= 0."""
    capacity = np.asarray(capacity, dtype=np.float64)
    if not r_annual > 0.0:
        return np.zeros(np.broadcast_shapes(capacity.shape, np.shape(loss)))
    keep = np.clip(1.0 - (np.asarray(loss, dtype=np.float64) + m2) / r_annual, 0.0, 1.0)
    return capacity * keep


def _spawn_mode(spec, override=None):
    cfg = (override or {}).get("spawn") if isinstance(override, dict) else None
    if cfg is None:
        cfg = (spec or {}).get("spawn")
    if not isinstance(cfg, dict):
        return None
    return str(cfg.get("mode", "")).strip().lower()


def build(project, project_path, library, grid_size, active_ids,
          overrides=None, baseline_maps=None, verbose=False):
    """Depth map and per-group habitat maps for one project load.

    Returns ``(depth, {fg_id: {"capacity": K map or None,
    "equilibrium": B* map or None}})`` for every active group with a
    capacity table or the equilibrium spawn mode. ``capacity`` is None
    when the group keeps its scalar K (no table or no depth map).
    """
    H, W = int(grid_size[0]), int(grid_size[1])
    specs = (library or {}).get("species_definitions") or {}
    overrides = overrides or {}
    depth = load_depth_map(project, project_path, grid_size, verbose=verbose)
    wanted = {}
    for fid in active_ids:
        spec = specs.get(fid) or {}
        table = parse_capacity_table(spec)
        equilibrium = _spawn_mode(spec, overrides.get(fid)) == SPAWN_MODE
        if table is None and not equilibrium:
            continue
        if spec.get("is_decision_maker"):
            raise ValueError(f"{fid}: {CAPACITY_TABLE_KEY} and the equilibrium "
                             "spawn are for non decision makers")
        wanted[fid] = (spec, table, equilibrium)
    if not wanted:
        return depth, {}
    if baseline_maps is None:
        from lib.world.impact_maps import project_baseline_maps
        baseline_maps = project_baseline_maps(project, project_path, grid_size,
                                              verbose=verbose)
    out = {}
    for fid, (spec, table, equilibrium) in wanted.items():
        capacity = None
        if table is not None and depth is not None:
            capacity = capacity_map(table, depth)
        entry = {"capacity": capacity, "equilibrium": None}
        if equilibrium:
            k = capacity if capacity is not None else np.full(
                (H, W), float(spec.get("max_carrying_capacity", 100.0)))
            r = float(spec.get("growth_rate", 0.0) or 0.0) * ticks_per_year(LIBRARY_TICK_HOURS)
            loss = annual_impact_loss(fid, library, baseline_maps, grid_size)
            entry["equilibrium"] = equilibrium_map(
                k, loss, predation_m2(fid, library, active_ids), r)
        out[fid] = entry
    return depth, out


def summary(entry):
    """Box means of K and B* (t/km2) for reports and tests."""
    cap, eq = entry.get("capacity"), entry.get("equilibrium")
    return {"capacity_mean": None if cap is None else float(np.mean(cap)),
            "equilibrium_mean": None if eq is None else float(np.mean(eq)),
            "equilibrium_total": None if eq is None else float(np.sum(eq))}


__all__ = ["CAPACITY_TABLE_KEY", "SPAWN_MODE", "annual_impact_loss", "build",
           "capacity_map", "depth_map_path", "equilibrium_map", "load_depth_map",
           "parse_capacity_table", "predation_m2", "summary"]

