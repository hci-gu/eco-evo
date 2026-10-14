"""Host-side configuration boundary, shared by training and benchmarks."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lib.config.config_loader import (
    _resolve_initial_biomass_range, load_config, load_impact_spawn_specs,
    load_project_config, setup_full_mareld_mvp,
)
from lib.environments.ecosystem import EcosystemEnvironment
from lib.environments.ecosystem_env.currents import CurrentConfig
from lib.world.impact_maps import project_baseline_maps, sample_impact_maps
from lib.world.tick_time import LIBRARY_TICK_HOURS


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
    currents: CurrentConfig | None = None
    # Global scale factor on every FG's ``natural_mortality``
    # (``--mortality_multiplier``); 1.0 leaves the tick unchanged.
    mortality_multiplier: float = 1.0
    # Whether the growth term pays for the biomass it adds out of the
    # energy reserve (``--mass-balance``, section 116). True since
    # section 120; False is the pre-116 tick.
    mass_balance: bool = True
    # ``--tick-length``. The library is calibrated at LIBRARY_TICK_HOURS
    # and is never rewritten; anything else is converted in memory as the
    # groups are built. Section 106.
    tick_hours: int = LIBRARY_TICK_HOURS

    def with_world(self, spawn_seed):
        return replace(self, spawn_seed=int(spawn_seed))

    def __call__(self, seed=None):
        kwargs = dict(library_path=self.library_path, grid_size=self.grid,
                      seed=seed, spawn_seed=self.spawn_seed,
                      tick_hours=self.tick_hours,
                      apply_natural_mortality=bool(self.mortality))
        impact_vars, impact_ranges, observable = [], {}, []
        if self.project_path:
            groups, impact_vars, impact_ranges, observable = load_project_config(
                self.project_path, **kwargs)
        else:
            groups = setup_full_mareld_mvp(**kwargs)
        env = EcosystemEnvironment(
            dict(height=self.grid[0], width=self.grid[1]), groups,
            observable_impact_vars=observable,
            migration=self.migration, apply_natural_mortality=self.mortality,
            mortality_multiplier=self.mortality_multiplier,
            mass_balance=self.mass_balance,
            currents=self.currents, current_world_seed=int(seed or 0) ^ int(self.spawn_seed or 0))
        if impact_vars:
            # A training world, as train._EnvBuilder builds it: observable
            # impacts sampled, baseline maps loaded, the rest absent
            # (section 161).
            maps = sample_impact_maps(
                impact_vars, impact_ranges, tuple(self.grid), seed=seed,
                impact_spawn_specs=load_impact_spawn_specs(self.project_path),
                observable_impact_vars=observable,
                baseline_maps=project_baseline_maps(
                    load_config(self.project_path), self.project_path, tuple(self.grid)))
            for impact_id in impact_vars:
                env.grid.add_map(impact_id, maps[impact_id])
        return env


class ProjectSpec:
    """Read YAML once, retaining the CPU environment as a reference schema."""
    def __init__(self, builder, seed=0, allowed_mask=None):
        self.builder = builder
        library = load_config(builder.library_path)
        project = load_config(builder.project_path) if builder.project_path else None
        kwargs = dict(library_path=builder.library_path, grid_size=builder.grid,
                      seed=seed, spawn_seed=seed, allowed_mask=allowed_mask,
                      library_config=library,
                      tick_hours=getattr(builder, "tick_hours", LIBRARY_TICK_HOURS),
                      apply_natural_mortality=bool(builder.mortality))
        rng_state = np.random.get_state()
        try:
            np.random.seed(seed & 0xFFFFFFFF)
            impact_vars, impact_ranges, observable = [], {}, []
            if project is not None:
                groups, impact_vars, impact_ranges, observable = load_project_config(
                    builder.project_path, project_config=project, **kwargs)
            else:
                groups = setup_full_mareld_mvp(**kwargs)
            self.env = EcosystemEnvironment(
                dict(height=builder.grid[0], width=builder.grid[1]), groups,
                observable_impact_vars=observable,
                migration=builder.migration, apply_natural_mortality=builder.mortality,
                mortality_multiplier=builder.mortality_multiplier,
                mass_balance=builder.mass_balance,
                currents=builder.currents, current_world_seed=seed)
        finally:
            np.random.set_state(rng_state)
        # Impact recipe (section 161): the GPU trainer samples a field per
        # world for every observable impact and loads the baseline maps,
        # exactly as the CPU trainer; the observable ones also widen the
        # policy input.
        self.impact_vars = list(impact_vars)
        self.observable_impact_vars = list(observable)
        self.impact_ranges = dict(impact_ranges)
        self.baseline_maps = (project_baseline_maps(project, builder.project_path,
                                                    tuple(builder.grid))
                              if project is not None else {})
        self.impact_spawn_specs = (load_impact_spawn_specs(builder.project_path)
                                   if project is not None else {})
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
