import yaml
import numpy as np
from lib.world.functional_group import FunctionalGroup

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

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

        # Initial total biomass: prefer library value, fallback to 1000.
        total_b = float(specs.get('initial_biomass', 1000) or 0)

        # Random distribution for MVP demonstration
        initial_b = rng.random(grid_size) if seed is not None else np.random.rand(*grid_size)
        initial_b = (initial_b / (initial_b.sum() + 1e-9)) * total_b
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs

def load_project_config(project_path, library_path='fgconfig/fg_library.yaml', grid_size=(60, 60), seed=None):
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

        # Initial total biomass: project override > library default > 1000.
        override = project_fg_overrides.get(sid, {})
        if 'initial_biomass' in override and override.get('initial_biomass') is not None:
            total_b = float(override['initial_biomass'])
        elif 'initial_biomass' in specs and specs.get('initial_biomass') is not None:
            total_b = float(specs['initial_biomass'])
        else:
            total_b = 1000.0

        initial_b = rng.random(grid_size) if rng is not None else np.random.rand(*grid_size)
        initial_b = (initial_b / (initial_b.sum() + 1e-9)) * total_b
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs, impact_vars
