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

        # Random distribution for MVP demonstration
        initial_b = rng.random(grid_size) if seed is not None else np.random.rand(*grid_size)
        initial_b = (initial_b / (initial_b.sum() + 1e-9)) * total_b
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
            gid = fg.get('group_id')
            if gid and gid not in project_fg_ids:
                project_fg_ids.append(gid)
                project_fg_overrides[gid] = fg
    impact_vars = [iv['impact_id'] for iv in project.get('impact_variables', [])]
    
    fgs = {}
    for sid in project_fg_ids:
        if sid not in spec_defs:
            continue
            
        specs = spec_defs[sid]
        params = specs.copy()
        params['interaction'] = {}
        params['menu'] = []
        
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

        initial_b = rng.random(grid_size) if rng is not None else np.random.rand(*grid_size)
        initial_b = (initial_b / (initial_b.sum() + 1e-9)) * total_b
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs, impact_vars
