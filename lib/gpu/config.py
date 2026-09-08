"""Host-side configuration boundary, shared by training and benchmarks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.config.config_loader import (
    _resolve_initial_biomass_range, load_config, load_project_config,
    setup_full_mareld_mvp,
)
from lib.environments.ecosystem import EcosystemEnvironment


DEFAULT_LIBRARY = str(Path(__file__).resolve().parents[2] / "fgconfig" / "fg_library.yaml")


@dataclass
class EnvironmentBuilder:
    """Picklable adapter for the existing multi-process CPU trainer."""
    project_path: str = None
    library_path: str = DEFAULT_LIBRARY
    grid: tuple = (60, 60)
    migration: bool = False
    mortality: bool = False
    spawn_seed: int = None

    def with_world(self, spawn_seed):
        return EnvironmentBuilder(self.project_path, self.library_path, self.grid,
                                  self.migration, self.mortality, int(spawn_seed))

    def __call__(self, seed=None):
        kwargs = dict(library_path=self.library_path, grid_size=self.grid,
                      seed=seed, spawn_seed=self.spawn_seed)
        if self.project_path:
            groups = load_project_config(self.project_path, **kwargs)[0]
        else:
            groups = setup_full_mareld_mvp(**kwargs)
        return EcosystemEnvironment(
            dict(height=self.grid[0], width=self.grid[1]), groups,
            migration=self.migration, apply_natural_mortality=self.mortality)


class ProjectSpec:
    """Read YAML once, retaining the CPU environment as a reference schema."""
    def __init__(self, builder, seed=0, allowed_mask=None):
        self.builder = builder
        library = load_config(builder.library_path)
        project = load_config(builder.project_path) if builder.project_path else None
        kwargs = dict(library_path=builder.library_path, grid_size=builder.grid,
                      seed=seed, spawn_seed=seed, allowed_mask=allowed_mask,
                      library_config=library)
        rng_state = np.random.get_state()
        try:
            np.random.seed(seed & 0xFFFFFFFF)
            if project is not None:
                groups = load_project_config(builder.project_path, project_config=project, **kwargs)[0]
            else:
                groups = setup_full_mareld_mvp(**kwargs)
            self.env = EcosystemEnvironment(
                dict(height=builder.grid[0], width=builder.grid[1]), groups,
                migration=builder.migration, apply_natural_mortality=builder.mortality)
        finally:
            np.random.set_state(rng_state)
        self.allowed_mask = allowed_mask
        self.randomize_energy = project is not None
        self.spawn_order = tuple(groups)
        self.spawn = {}
        self.ranges = {}
        overrides = {}
        if project:
            for section in ("decision_makers", "non_decision_makers", "functional_groups"):
                for fg in project.get(section, []) or []:
                    if isinstance(fg, dict) and not fg.get("muted"):
                        overrides.setdefault(fg.get("group_id"), fg)
        meta = (project or {}).get("project_metadata", {}) or {}
        def reference_size(key):
            try:
                return max(3, int(meta.get(key, 60)))
            except (TypeError, ValueError):
                return 60
        scale = (builder.grid[0] * builder.grid[1] /
                 (reference_size("reference_grid_width") * reference_size("reference_grid_height")))
        for fid, fg in groups.items():
            override = overrides.get(fid, {})
            specs = library["species_definitions"][fid]
            cfg = override.get("spawn")
            if cfg is None:
                cfg = specs.get("spawn")
            self.spawn[fid] = cfg if isinstance(cfg, dict) else None
            lo, hi = _resolve_initial_biomass_range(override, specs)
            if lo is None and hi is None:
                lo = hi = 1000
            elif project is not None:
                lo = max(lo * scale, 1.0)
                hi = max(hi * scale, lo)
            self.ranges[fid] = (int(round(lo)), int(round(hi)))
        self.policy_params = {
            fid: (int(self.env.per_dm_in_dim[i]), 5 + self.env.N_all)
            for i, fid in enumerate(self.env.dm_ids)
        }
