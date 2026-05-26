import yaml
import numpy as np
from lib.world.functional_group import FunctionalGroup

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_initial_biomass_range(*sources):
    """Resolve an (min, max) initial-biomass range from one or more dict sources.

    Sources are checked in the given order; the first source that provides any
    biomass information wins. Schema: ``initial_biomass_min`` and
    ``initial_biomass_max`` (both non-negative ints; max >= min).
    Returns ``(min, max)`` as floats, or ``(None, None)`` if no source provides
    a value.
    """
    def _as_num(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for src in sources:
        if not isinstance(src, dict):
            continue
        mn = _as_num(src.get("initial_biomass_min"))
        mx = _as_num(src.get("initial_biomass_max"))
        if mn is not None or mx is not None:
            if mn is None:
                mn = mx
            if mx is None:
                mx = mn
            if mx < mn:
                mn, mx = mx, mn
            return mn, mx
    return None, None


def _sample_total_biomass(min_b, max_b, rng):
    """Sample an initial total biomass uniformly from [min_b, max_b] integers.

    If both bounds are None, returns 1000.0 (legacy fallback).
    """
    if min_b is None and max_b is None:
        return 1000.0
    if min_b is None:
        min_b = max_b
    if max_b is None:
        max_b = min_b
    lo = int(round(min_b))
    hi = int(round(max_b))
    if lo == hi:
        return float(lo)
    if rng is not None and hasattr(rng, "integers"):
        return float(rng.integers(lo, hi + 1))
    return float(np.random.randint(lo, hi + 1))

def _spawn_biomass_distribution(grid_size, total_b, min_per_cell, allowed_mask=None, rng=None):
    """Distribute ``total_b`` over the grid following the cluster-spawn spec.

    Rules:
      * Eligible cells are those where ``allowed_mask`` is truthy. If
        ``allowed_mask`` is None, every cell is eligible.
      * Each picked cell receives a biomass drawn uniformly from
        ``[min_per_cell, total_b]`` (or ``min_per_cell .. remaining`` if less
        biomass than ``total_b`` is left). The pick continues until the
        running remainder drops below ``min_per_cell``.
      * Any leftover (< ``min_per_cell``) is merged into one of the already
        seeded cells so total biomass is preserved exactly.
      * If the number of eligible cells is too small to fit
        ``ceil(total_b / total_b)``-many cells while honouring the floor,
        cells are allowed to receive more than ``total_b`` (i.e. the
        remainder is absorbed and the loop exits naturally).
      * When ``min_per_cell`` is 0 or non-positive the function falls back
        to the legacy uniform-Dirichlet spread over all eligible cells.
    """
    H, W = grid_size
    if rng is None:
        rng = np.random
    initial_b = np.zeros(grid_size, dtype=np.float64)
    if total_b <= 0:
        return initial_b

    # Determine the eligible cell pool.
    if allowed_mask is None:
        flat_idx = np.arange(H * W)
    else:
        mask = np.asarray(allowed_mask).reshape(H, W)
        flat_idx = np.flatnonzero(mask > 0)
        if flat_idx.size == 0:
            # No accessible cell — fall back to the full grid so that biomass
            # never silently vanishes when accessibility is not configured.
            flat_idx = np.arange(H * W)

    # Legacy path: no clustering threshold, spread biomass across the full
    # eligible pool with a uniform Dirichlet-like draw. Preserves prior
    # behaviour for FGs whose min_split_biomass == 0.
    if min_per_cell is None or min_per_cell <= 0:
        if hasattr(rng, "random"):
            draws = rng.random(flat_idx.size)
        else:
            draws = np.random.rand(flat_idx.size)
        draws = (draws / (draws.sum() + 1e-9)) * total_b
        flat = initial_b.reshape(-1)
        flat[flat_idx] = draws
        return initial_b

    # Cluster-spawn path: pick cells one at a time, assign biomass uniformly
    # in [min_per_cell, total_b] (capped at remainder), repeat until the
    # remainder is below the floor; merge the leftover into a seeded cell.
    flat = initial_b.reshape(-1)
    seeded = []
    remaining = float(total_b)
    available = list(flat_idx)
    # `available` is the pool of eligible cells that have not yet received
    # biomass. Picking without replacement keeps the seed cells distinct.
    while remaining >= min_per_cell and available:
        # Random pick from the available pool.
        if hasattr(rng, "integers"):
            i = int(rng.integers(0, len(available)))
        else:
            i = int(np.random.randint(0, len(available)))
        cell = available.pop(i)
        # Draw biomass uniformly in [min_per_cell, min(total_b, remaining)].
        hi = min(float(total_b), remaining)
        # `hi` can equal `min_per_cell` near the end; np.random.uniform is OK
        # with hi == lo, but guard against a tiny numerical underflow.
        if hi <= min_per_cell:
            amount = remaining  # absorb the remainder, exit next iteration
        else:
            if hasattr(rng, "uniform"):
                amount = float(rng.uniform(min_per_cell, hi))
            else:
                amount = float(np.random.uniform(min_per_cell, hi))
        amount = min(amount, remaining)
        flat[cell] = amount
        seeded.append(cell)
        remaining -= amount

    # Edge case: no seed could be placed (e.g. total_b < min_per_cell). Put
    # all biomass into a single random eligible cell so it isn't lost. This
    # also allows the cell to exceed ``total_b`` if needed — see spec point 3.
    if not seeded:
        if hasattr(rng, "integers"):
            i = int(rng.integers(0, flat_idx.size))
        else:
            i = int(np.random.randint(0, flat_idx.size))
        flat[flat_idx[i]] = float(total_b)
        return initial_b

    # Merge leftover (< min_per_cell) into a randomly chosen seeded cell.
    if remaining > 0:
        if hasattr(rng, "integers"):
            j = int(rng.integers(0, len(seeded)))
        else:
            j = int(np.random.randint(0, len(seeded)))
        flat[seeded[j]] += remaining

    return initial_b


def setup_full_mareld_mvp(library_path='fgconfig/fg_library.yaml', grid_size=(60, 60), seed=None):
    lib = load_config(library_path)
    spec_defs = lib['species_definitions']
    inter_defs = lib['interaction_definitions']

    rng = np.random.default_rng(seed) if seed is not None else np.random
    fgs = {}
    for sid, specs in spec_defs.items():
        # Merge specs with interaction data
        params = specs.copy()
        params['interaction'] = {}
        params['menu'] = []
        
        # Find interactions for this species
        for iid, idef in inter_defs.items():
            if iid.startswith(f"{sid}_preys_on_"):
                prey_id = iid.replace(f"{sid}_preys_on_", "")
                if idef.get('preys_on', False):
                    params['menu'].append(prey_id)
                    params['interaction'][iid] = idef
            elif iid.startswith(f"{sid}_impacted_by_"):
                if 'impact' not in params: params['impact'] = {}
                impact_id = iid.replace(f"{sid}_impacted_by_", "")
                params['impact'][impact_id] = idef
                
        fg = FunctionalGroup(sid, params)

        # Initial total biomass: prefer library range (min/max or legacy scalar),
        # fallback to 1000. Sampled fresh on each call so spatial layouts vary.
        min_b, max_b = _resolve_initial_biomass_range(specs)
        sample_rng = rng if seed is not None else None
        total_b = _sample_total_biomass(min_b, max_b, sample_rng)

        # Cluster-aware spawn: enforce a per-cell floor of 10 * min_split
        # (kg -> tonnes already applied inside FG.__init__) so that newly
        # spawned cells start well above the sub-threshold mask. For FGs
        # with min_split == 0 this collapses to the legacy uniform spread.
        min_per_cell = 10.0 * float(getattr(fg, 'min_split_biomass', 0.0))
        initial_b = _spawn_biomass_distribution(
            grid_size, total_b, min_per_cell,
            allowed_mask=None, rng=rng)
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs

def _resolve_inference_initial_biomass(*sources):
    """Read fixed inference initial biomass (ton) from the first source that has it.

    Schema: ``inference_initial_biomass`` (non-negative number). Returns float or None.
    """
    for src in sources:
        if not isinstance(src, dict):
            continue
        v = src.get("inference_initial_biomass")
        if v is None or v == "":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f < 0:
            continue
        return f
    return None


def load_project_config(project_path, library_path='fgconfig/fg_library.yaml', grid_size=(60, 60), seed=None, mode='train'):
    project = load_config(project_path)
    rng = np.random.default_rng(seed) if seed is not None else None
    lib = load_config(library_path)
    spec_defs = lib['species_definitions']
    inter_defs = lib['interaction_definitions']
    
    # Support both the new split (decision_makers / non_decision_makers) and the
    # legacy unified functional_groups list for backward compatibility.
    # Also retain per-FG project overrides (e.g. initial_biomass).
    project_fg_ids = []
    project_fg_overrides = {}
    for key in ('decision_makers', 'non_decision_makers', 'functional_groups'):
        for fg in project.get(key, []) or []:
            if not isinstance(fg, dict):
                continue
            # Skip muted FGs: they're soft-deleted and must be invisible to
            # the simulation, while their YAML configuration is preserved so
            # the user can quickly unmute them later.
            if fg.get('muted'):
                continue
            gid = fg.get('group_id')
            if gid and gid not in project_fg_ids:
                project_fg_ids.append(gid)
                project_fg_overrides[gid] = fg
    # Likewise, muted impact variables are dropped from the active set.
    impact_vars = [iv['impact_id'] for iv in project.get('impact_variables', [])
                   if isinstance(iv, dict) and not iv.get('muted')]
    # Subset of impact_vars that the policy network observes as input layers
    # (one channel per observable impact). Order matches ``impact_vars`` to
    # keep the observation channel layout deterministic across runs.
    observable_impact_vars = [
        iv['impact_id'] for iv in project.get('impact_variables', [])
        if isinstance(iv, dict) and not iv.get('muted') and iv.get('observable')
    ]

    # Per-impact value range (vmin, vmax). Read from the project's
    # impact_variables entries: schema ``value_min`` / ``value_max``.
    # Missing/invalid values fall back to (0.0, 0.0) so the resulting map is
    # a zero field (preserving previous behaviour for unconfigured impacts).
    def _as_float(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f >= 0 else None

    impact_ranges = {}
    for iv in project.get('impact_variables', []) or []:
        if not isinstance(iv, dict) or iv.get('muted'):
            continue
        iid = iv.get('impact_id')
        if not iid:
            continue
        vmin = _as_float(iv.get('value_min'))
        vmax = _as_float(iv.get('value_max'))
        if vmin is None and vmax is None:
            vmin, vmax = 0.0, 0.0
        elif vmin is None:
            vmin = vmax
        elif vmax is None:
            vmax = vmin
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        impact_ranges[iid] = (vmin, vmax)
    
    fgs = {}
    for sid in project_fg_ids:
        if sid not in spec_defs:
            continue
            
        specs = spec_defs[sid]
        params = specs.copy()
        params['interaction'] = {}
        params['menu'] = []
        
        active_fg_set = set(project_fg_ids)
        active_impact_set = set(impact_vars)
        for iid, idef in inter_defs.items():
            if iid.startswith(f"{sid}_preys_on_"):
                prey_id = iid.replace(f"{sid}_preys_on_", "")
                # Skip predation relations whose prey is muted (or absent).
                if prey_id not in active_fg_set:
                    continue
                if idef.get('preys_on', False):
                    params['menu'].append(prey_id)
                    params['interaction'][iid] = idef
            elif iid.startswith(f"{sid}_impacted_by_"):
                impact_id = iid.replace(f"{sid}_impacted_by_", "")
                # Skip impacts that are muted or not in the project.
                if impact_id not in active_impact_set:
                    continue
                if 'impact' not in params: params['impact'] = {}
                params['impact'][impact_id] = idef
                
        fg = FunctionalGroup(sid, params)

        # Initial total biomass: per-project override range > library range >
        # 1000 fallback. The actual scalar `total_b` is sampled uniformly from
        # [min, max] on every call so each spatial layout varies even when the
        # configured range is unchanged.
        override = project_fg_overrides.get(sid, {})
        if mode == 'inference':
            # Inference uses the per-FG fixed value entered in the FG config
            # tool's "Inference" tab. No random sampling: the exact value is
            # spread spatially. Missing values fall back to 0.0.
            fixed = _resolve_inference_initial_biomass(override)
            total_b = 0.0 if fixed is None else float(fixed)
        else:
            min_b, max_b = _resolve_initial_biomass_range(override, specs)
            total_b = _sample_total_biomass(min_b, max_b, rng)

        # Cluster-aware spawn (see _spawn_biomass_distribution): per-cell
        # floor 10 * min_split_biomass eliminates the sub-threshold mask
        # lock-in that affected seals/porpoises. accessibility filtering is
        # threaded in once accessibility maps become part of the project
        # config; for now allowed_mask=None means the full grid is eligible.
        min_per_cell = 10.0 * float(getattr(fg, 'min_split_biomass', 0.0))
        initial_b = _spawn_biomass_distribution(
            grid_size, total_b, min_per_cell,
            allowed_mask=None, rng=rng)
        # Training: E_X(c) ~ Uniform(0, ME_X) per cell so policies see varied
        # initial energy fill levels. Inference keeps the deterministic
        # 0.7 * ME_X default for reproducible scenario comparisons.
        randomize_energy = (mode == 'train')
        fg.initialize_state(grid_size, initial_biomass=initial_b,
                            randomize_energy=randomize_energy, rng=rng)
        fgs[sid] = fg

    return fgs, impact_vars, impact_ranges, observable_impact_vars
