import yaml
import numpy as np
from lib.world.functional_group import FunctionalGroup

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_full_mareld_mvp(library_path='fgconfig/fg_library.yaml', grid_size=(60, 60)):
    lib = load_config(library_path)
    spec_defs = lib['species_definitions']
    inter_defs = lib['interaction_definitions']
    
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
        
        # Initial biomass (from Mareld mini.pdf)
        initial_totals = {
            'phytoplankton': 80000,
            'zooplankton': 35000,
            'benthic_community': 160000,
            'pelagic_fish': 12000,
            'gadoids': 2500,
            'seals': 25,
            'porpoises': 12,
            'seabirds': 25
        }
        
        total_b = initial_totals.get(sid, 1000)
        
        # Random distribution for MVP demonstration
        initial_b = np.random.rand(*grid_size)
        initial_b = (initial_b / (initial_b.sum() + 1e-9)) * total_b
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs

def load_project_config(project_path, library_path='fgconfig/fg_library.yaml', grid_size=(60, 60)):
    project = load_config(project_path)
    lib = load_config(library_path)
    spec_defs = lib['species_definitions']
    inter_defs = lib['interaction_definitions']
    
    project_fg_ids = [fg['group_id'] for fg in project.get('functional_groups', [])]
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
        
        initial_totals = {
            'phytoplankton': 80000,
            'zooplankton': 35000,
            'benthic_community': 160000,
            'pelagic_fish': 12000,
            'gadoids': 2500,
            'seals': 25,
            'porpoises': 12,
            'seabirds': 25
        }
        total_b = initial_totals.get(sid, 1000)
        
        initial_b = np.random.rand(*grid_size)
        initial_b = (initial_b / (initial_b.sum() + 1e-9)) * total_b
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs, impact_vars
