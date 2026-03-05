# import lib.constants as const
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from lib.config import const
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.model import Model
import lib.model as model
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations, plot_biomass
from lib.data_manager import update_generations_data, process_data
from lib.behaviour import run_all_scenarios
from lib.evaluate import predict_years
import numpy as np
import random
import copy
import time
from lib.environments.petting_zoo import env
import optuna

# ---- Parallel worker plumbing ----
_WORKER_RUNNER = None  # one runner per proces
_WORKER_POPULATION_STATE = None
_WORKER_OPPONENT_STATE = None
_WORKER_MODEL_CACHE = None
_WORKER_OPPONENT_MODEL_CACHE = None

def noop(a, b, c):
    pass

class RandomBaselinePolicy:
    """Simple random policy adapter with a Model-like forward interface."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def forward(self, x):
        n = x.shape[0]
        logits = self.rng.random((n, model.OUTPUT_SIZE), dtype=np.float32) + 1e-6
        probs = logits / logits.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

class PettingZooRunner():
    def __init__(
        self,
        settings: Settings,
        render_mode: str = "none",
        map_folder: str = "maps/baltic",
        build_population: bool = True,
    ):
        # Create the environment (render_mode can be "none" if visualization is not needed)
        self.species_map = build_species_map(settings)
        self.env = env(
            settings=settings,
            species_map=self.species_map,
            render_mode=render_mode,
            map_folder=map_folder,
        )
        self.settings = settings
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        self.current_generation = 0
        self.rng = random.Random(None if self.settings.seed < 0 else self.settings.seed)

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.ACTING_SPECIES]
        # Create a population for each species (training only).
        self.population = {}
        if build_population:
            self.population = {
                species: [Model() for _ in range(self.settings.num_agents)]
                for species in self.species_list
            }
        # Track best fitness and best model for each species.
        self.best_fitness = {species: -float('inf') for species in self.species_list}
        self.best_agent = {species: None for species in self.species_list}
        self.agent_index = 0
        self.eval_index = 0
        self.champion_progress_history = []
        self.fixed_validation_history = []
        self.fixed_validation_by_base_history = {
            base: [] for base in const.ACTING_BASE_SPECIES
        }
        self.opponent_population_snapshot = None
        self.opponent_snapshot_generation = None
        self._last_run_diagnostics = {}
        self.fixed_validation_metric = str(
            getattr(self.settings, "fixed_validation_metric", "fitness")
        ).strip().lower()
        if self.fixed_validation_metric not in {"fitness", "survival"}:
            raise ValueError(
                f"Unsupported fixed_validation_metric '{self.settings.fixed_validation_metric}'. "
                "Use 'fitness' or 'survival'."
            )
        if self.settings.seed >= 0:
            benchmark_rng = random.Random(self.settings.seed + 9_999_937)
            self.champion_progress_seed = benchmark_rng.randrange(100000)
        else:
            self.champion_progress_seed = 137
            benchmark_rng = random.Random(self.champion_progress_seed)

        configured_fixed_seed = int(getattr(self.settings, "fixed_validation_seed", 4242))
        if configured_fixed_seed >= 0:
            self.fixed_validation_seed = configured_fixed_seed
        elif self.settings.seed >= 0:
            self.fixed_validation_seed = benchmark_rng.randrange(2**31)
        else:
            self.fixed_validation_seed = 4242
        self._base_spawn_table = None
        self._spawn_table_source_species = None
        self._cache_spawn_table_template()

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def _cache_spawn_table_template(self):
        base_env = self._base_env()
        table = getattr(base_env, "spawn_table", None)
        if table is None:
            self._base_spawn_table = None
            self._spawn_table_source_species = None
            return
        self._base_spawn_table = tuple(np.array(arr, copy=True) for arr in table)
        source_species = []
        for species_name, props in self.species_map.items():
            if not props.is_mature:
                continue
            if props.offspring_species is None:
                continue
            source_species.append(species_name)
        self._spawn_table_source_species = source_species

    def _set_training_reproduction_override(self, is_evaluation: bool):
        if self._base_spawn_table is None:
            return
        base_env = self._base_env()
        override_freq = int(getattr(self.settings, "training_reproduction_freq_override", 0))
        if is_evaluation or override_freq <= 0:
            base_env.spawn_table = tuple(np.array(arr, copy=True) for arr in self._base_spawn_table)
            return

        src_b, src_e, dst_b, dst_e, repro_freq, growth_rate = (
            np.array(arr, copy=True) for arr in self._base_spawn_table
        )

        source_species = self._spawn_table_source_species or []
        if len(source_species) == repro_freq.shape[0]:
            for idx, species_name in enumerate(source_species):
                if self.species_map[species_name].base_species == "plankton":
                    continue
                repro_freq[idx] = max(int(repro_freq[idx]), override_freq)
        else:
            # Fallback safety if table alignment changes.
            repro_freq[:] = np.maximum(repro_freq, override_freq)

        base_env.spawn_table = (src_b, src_e, dst_b, dst_e, repro_freq, growth_rate)

    def _get_alive_base_species(self):
        threshold = float(self.settings.alive_species_biomass_threshold)
        totals = self._get_base_biomass_totals()
        return [base_species for base_species, total in totals.items() if total > threshold]

    def _get_base_biomass_totals(self, world=None):
        if world is None:
            world = self.env.world
        totals = {base_species: 0.0 for base_species in const.ACTING_BASE_SPECIES}
        for species_name, props in self.species_map.items():
            base_species = props.base_species
            if base_species not in totals:
                continue
            biomass_offset = model.MODEL_OFFSETS[species_name]["biomass"]
            totals[base_species] += float(world[..., biomass_offset].sum())
        return totals

    def _resolve_fitness_target_species(self, species_being_evaluated: str, biomass_scope: str) -> str:
        if biomass_scope != "base_species":
            return species_being_evaluated
        if species_being_evaluated in self.species_map:
            return self.species_map[species_being_evaluated].base_species
        return const.base_species_name(species_being_evaluated)

    def _get_target_biomass_and_energy(self, target_species: str, world=None):
        if world is None:
            world = self.env.world

        if target_species in model.MODEL_OFFSETS:
            offsets = model.MODEL_OFFSETS[target_species]
            biomass = world[..., offsets["biomass"]]
            energy = world[..., offsets["energy"]]
            total_biomass = float(np.sum(biomass))
            weighted_energy = float(np.sum(biomass * energy) / max(total_biomass, 1e-8))
            return total_biomass, weighted_energy

        total_biomass = 0.0
        total_energy_weighted = 0.0
        for species_name, props in self.species_map.items():
            if props.base_species != target_species:
                continue
            offsets = model.MODEL_OFFSETS[species_name]
            biomass = world[..., offsets["biomass"]]
            energy = world[..., offsets["energy"]]
            bio_sum = float(np.sum(biomass))
            total_biomass += bio_sum
            total_energy_weighted += float(np.sum(biomass * energy))
        weighted_energy = total_energy_weighted / max(total_biomass, 1e-8)
        return float(total_biomass), float(weighted_energy)

    def _energy_channel_indices(self):
        indices = []
        for species_name in self.species_map.keys():
            offsets = model.MODEL_OFFSETS.get(species_name)
            if offsets is None:
                continue
            indices.append(offsets["energy"])
        return indices

    def _scale_all_species_energy(self, scale: float):
        scale = float(scale)
        if abs(scale - 1.0) < 1e-9:
            return
        energy_channels = self._energy_channel_indices()
        if not energy_channels:
            return
        world = self.env.world
        world[..., energy_channels] *= scale
        np.clip(world[..., energy_channels], 0.0, self.settings.max_energy, out=world[..., energy_channels])

    def _apply_energy_decay(self, decay_per_cycle: float):
        decay = float(decay_per_cycle)
        if decay <= 0.0:
            return
        keep = max(0.0, 1.0 - decay)
        energy_channels = self._energy_channel_indices()
        if not energy_channels:
            return
        world = self.env.world
        world[..., energy_channels] *= keep
        np.clip(world[..., energy_channels], 0.0, self.settings.max_energy, out=world[..., energy_channels])

    def _target_species_names(self, target_species: str, target_only: bool):
        if not target_only:
            return [s for s in self.species_map.keys() if self.species_map[s].base_species != "plankton"]
        if target_species in self.species_map:
            return [target_species]
        return [
            s for s, props in self.species_map.items()
            if props.base_species == target_species
        ]

    def _apply_training_low_energy_biomass_loss(
        self,
        target_species: str,
        energy_reference: float,
        floor_pct: float,
        max_loss_per_cycle: float,
        target_only: bool,
    ):
        floor_pct = float(np.clip(floor_pct, 0.0, 1.0))
        max_loss_per_cycle = float(np.clip(max_loss_per_cycle, 0.0, 1.0))
        if floor_pct <= 0.0 or max_loss_per_cycle <= 0.0:
            return

        world = self.env.world
        energy_reference = max(float(energy_reference), 1e-8)
        species_names = self._target_species_names(target_species, target_only)
        for species_name in species_names:
            offsets = model.MODEL_OFFSETS.get(species_name)
            if offsets is None:
                continue
            props = self.species_map.get(species_name)
            if props is None:
                continue

            b_idx = offsets["biomass"]
            e_idx = offsets["energy"]
            biomass = world[..., b_idx]
            energy = world[..., e_idx]
            if not np.any(biomass > 0.0):
                continue

            energy_pct = energy / energy_reference
            deficit = np.clip((floor_pct - energy_pct) / max(floor_pct, 1e-8), 0.0, 1.0)
            loss_fraction = max_loss_per_cycle * deficit
            if not np.any(loss_fraction > 0.0):
                continue

            biomass *= (1.0 - loss_fraction)
            low_biomass = biomass < props.min_biomass_in_cell
            biomass[low_biomass] = 0.0
            energy[low_biomass] = 0.0

    def _sparsify_plankton_for_training(
        self,
        cell_fraction: float,
        biomass_scale: float,
        rng_seed=None,
    ):
        cell_fraction = float(np.clip(cell_fraction, 0.0, 1.0))
        biomass_scale = float(max(0.0, biomass_scale))
        if cell_fraction >= 1.0 and abs(biomass_scale - 1.0) < 1e-9:
            return

        plankton_offsets = model.MODEL_OFFSETS.get("plankton")
        if plankton_offsets is None:
            return

        world = self.env.world
        world_data = self.env.world_data
        plankton_biomass_idx = plankton_offsets["biomass"]
        plankton_energy_idx = plankton_offsets["energy"]

        active_cells = np.argwhere(world_data[:, :, 1] > 0.5)
        if active_cells.size == 0:
            return

        rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))
        n_active = active_cells.shape[0]
        keep_n = int(np.ceil(n_active * cell_fraction))
        keep_n = min(max(keep_n, 0), n_active)
        keep_mask = np.zeros(n_active, dtype=bool)
        if keep_n > 0:
            keep_indices = rng.choice(n_active, size=keep_n, replace=False)
            keep_mask[keep_indices] = True

        removed_cells = active_cells[~keep_mask]
        if removed_cells.size > 0:
            rx, ry = removed_cells[:, 0], removed_cells[:, 1]
            world[rx, ry, plankton_biomass_idx] = 0.0
            world[rx, ry, plankton_energy_idx] = 0.0
            world_data[rx, ry, 1] = 0.0
            world_data[rx, ry, 2] = 0.0

        kept_cells = active_cells[keep_mask]
        if kept_cells.size > 0 and abs(biomass_scale - 1.0) > 1e-9:
            kx, ky = kept_cells[:, 0], kept_cells[:, 1]
            world[kx, ky, plankton_biomass_idx] *= biomass_scale
            world[kx, ky, plankton_energy_idx] *= biomass_scale
            world[kx, ky, plankton_energy_idx] = np.clip(
                world[kx, ky, plankton_energy_idx], 0.0, self.settings.max_energy
            )

    def _get_training_core_views(self):
        world = self.env.world
        world_data = self.env.world_data
        if world.shape[0] >= 3 and world.shape[1] >= 3:
            world_core = world[1:-1, 1:-1, :]
            world_data_core = world_data[1:-1, 1:-1, :]
        else:
            world_core = world
            world_data_core = world_data
        water_idx = model.MODEL_OFFSETS["terrain"]["water"]
        water_mask = world_core[:, :, water_idx] > 0.5
        return world_core, world_data_core, water_mask

    def _sync_plankton_cluster_markers(self):
        offsets = model.MODEL_OFFSETS.get("plankton")
        if offsets is None or "plankton" not in self.species_map:
            return
        world_core, world_data_core, _ = self._get_training_core_views()
        biomass = world_core[:, :, offsets["biomass"]]
        has_plankton = biomass > 0.0
        respawn_delay = float(
            self.species_map["plankton"].hardcoded_rules.get("respawn_delay", 0.0)
        )
        world_data_core[:, :, 1] = has_plankton.astype(np.float32)
        world_data_core[:, :, 2] = np.where(
            has_plankton,
            respawn_delay,
            0.0,
        ).astype(np.float32)

    def _redistribute_species_to_mask(self, species_name: str, keep_mask):
        offsets = model.MODEL_OFFSETS[species_name]
        b_idx = offsets["biomass"]
        e_idx = offsets["energy"]
        world_core, _, _ = self._get_training_core_views()
        biomass = world_core[:, :, b_idx]
        energy = world_core[:, :, e_idx]

        total_biomass = float(np.sum(biomass))
        if total_biomass <= 0.0:
            biomass.fill(0.0)
            energy.fill(0.0)
            return

        weighted_energy = float(
            np.sum(biomass * energy) / max(total_biomass, 1e-8)
        )
        biomass.fill(0.0)
        energy.fill(0.0)

        keep_n = int(np.count_nonzero(keep_mask))
        if keep_n <= 0:
            return

        biomass[keep_mask] = total_biomass / float(keep_n)
        energy_value = float(np.clip(weighted_energy, 0.0, self.settings.max_energy))
        energy[keep_mask] = energy_value

    def _sparsify_non_plankton_for_training(self, cell_fraction: float, min_cells: int, cod_spawn_cells: int, rng_seed=None):
        cell_fraction = float(np.clip(cell_fraction, 0.0, 1.0))
        min_cells = max(0, int(min_cells))
        cod_spawn_cells = max(0, int(cod_spawn_cells))
        if cell_fraction >= 1.0 and cod_spawn_cells == 0:
            return

        world_core, _, water_mask = self._get_training_core_views()
        rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

        base_to_species = {}
        for species_name, props in self.species_map.items():
            if props.base_species == "plankton":
                continue
            base_to_species.setdefault(props.base_species, []).append(species_name)

        for base_species, species_names in base_to_species.items():
            base_biomass = np.zeros(water_mask.shape, dtype=np.float32)
            for species_name in species_names:
                b_idx = model.MODEL_OFFSETS[species_name]["biomass"]
                base_biomass += world_core[:, :, b_idx]

            active_positions = np.argwhere((base_biomass > 0.0) & water_mask)
            if active_positions.shape[0] == 0:
                active_positions = np.argwhere(water_mask)
            if active_positions.shape[0] == 0:
                continue

            if base_species == "cod" and cod_spawn_cells > 0:
                keep_n = cod_spawn_cells
            else:
                keep_n = int(np.ceil(active_positions.shape[0] * cell_fraction))
                keep_n = max(keep_n, min_cells)

            keep_n = min(max(keep_n, 0), active_positions.shape[0])
            keep_mask = np.zeros(water_mask.shape, dtype=bool)
            if keep_n > 0:
                selected = active_positions[
                    rng.choice(active_positions.shape[0], size=keep_n, replace=False)
                ]
                keep_mask[selected[:, 0], selected[:, 1]] = True

            for species_name in species_names:
                self._redistribute_species_to_mask(species_name, keep_mask)

    def _clear_prey_near_cod_for_training(self, clear_radius: int, rng_seed=None):
        clear_radius = max(0, int(clear_radius))
        if clear_radius <= 0:
            return

        world_core, _, water_mask = self._get_training_core_views()
        cod_species = [
            species_name
            for species_name, props in self.species_map.items()
            if props.base_species == "cod"
        ]
        if not cod_species:
            return

        cod_total = np.zeros(water_mask.shape, dtype=np.float32)
        for species_name in cod_species:
            b_idx = model.MODEL_OFFSETS[species_name]["biomass"]
            cod_total += world_core[:, :, b_idx]
        cod_positions = np.argwhere(cod_total > 0.0)
        if cod_positions.shape[0] == 0:
            return

        close_mask = np.zeros(water_mask.shape, dtype=bool)
        for x, y in cod_positions:
            x0 = max(0, x - clear_radius)
            x1 = min(close_mask.shape[0], x + clear_radius + 1)
            y0 = max(0, y - clear_radius)
            y1 = min(close_mask.shape[1], y + clear_radius + 1)
            close_mask[x0:x1, y0:y1] = True

        prey_species = set()
        for species_name in cod_species:
            prey_species.update(self.species_map[species_name].prey)

        rng = np.random.default_rng(None if rng_seed is None else int(rng_seed) + 13)
        for prey_species_name in prey_species:
            offsets = model.MODEL_OFFSETS.get(prey_species_name)
            if offsets is None:
                continue

            b_idx = offsets["biomass"]
            e_idx = offsets["energy"]
            biomass = world_core[:, :, b_idx]
            energy = world_core[:, :, e_idx]

            remove_mask = close_mask & (biomass > 0.0)
            removed_biomass = float(np.sum(biomass[remove_mask]))
            if removed_biomass <= 0.0:
                continue

            removed_energy = float(
                np.sum(biomass[remove_mask] * energy[remove_mask])
                / max(removed_biomass, 1e-8)
            )
            biomass[remove_mask] = 0.0
            energy[remove_mask] = 0.0

            eligible_far = water_mask & (~close_mask)
            if not np.any(eligible_far):
                continue

            occupied_far = eligible_far & (biomass > 0.0)
            target_positions = np.argwhere(occupied_far)
            if target_positions.shape[0] == 0:
                fallback_positions = np.argwhere(eligible_far)
                if fallback_positions.shape[0] == 0:
                    continue
                target_positions = fallback_positions[
                    rng.choice(fallback_positions.shape[0], size=1, replace=False)
                ]

            share = removed_biomass / float(target_positions.shape[0])
            tx = target_positions[:, 0]
            ty = target_positions[:, 1]
            old_biomass = biomass[tx, ty].copy()
            old_energy = energy[tx, ty].copy()
            new_biomass = old_biomass + share
            mixed_energy = np.divide(
                old_biomass * old_energy + share * removed_energy,
                np.maximum(new_biomass, 1e-8),
            )
            biomass[tx, ty] = new_biomass
            energy[tx, ty] = np.clip(mixed_energy, 0.0, self.settings.max_energy)

    def _select_opponent_population_state(self, population_state):
        snapshot_every = max(1, int(getattr(self.settings, "opponent_snapshot_every", 1)))
        if snapshot_every <= 1:
            self.opponent_population_snapshot = population_state
            self.opponent_snapshot_generation = self.current_generation
            return population_state

        should_refresh = (
            self.opponent_population_snapshot is None
            or (self.current_generation % snapshot_every) == 0
        )
        if should_refresh:
            self.opponent_population_snapshot = {
                species: list(states)
                for species, states in population_state.items()
            }
            self.opponent_snapshot_generation = self.current_generation
        return self.opponent_population_snapshot

    def _log_trajectory_diagnostics(self, result_rows, label: str):
        if not bool(getattr(self.settings, "trajectory_diagnostics_enabled", True)):
            return
        diagnostics = []
        for row in result_rows:
            d = row.get("diagnostics")
            if d:
                diagnostics.append(d)
        if not diagnostics:
            return

        def _mean(name):
            vals = [float(d.get(name, 0.0)) for d in diagnostics if name in d]
            if not vals:
                return 0.0
            return float(np.mean(vals))

        print(
            f"Trajectory diagnostics ({label}): "
            f"bio={_mean('trajectory_component_biomass'):.3f}, "
            f"energy={_mean('trajectory_component_energy'):.3f}, "
            f"crash={_mean('trajectory_component_crash_penalty'):.3f}, "
            f"low={_mean('trajectory_component_low_biomass_penalty'):.3f}, "
            f"low_energy={_mean('trajectory_component_low_energy_penalty'):.3f}, "
            f"terminal={_mean('trajectory_component_terminal'):.3f}, "
            f"total={_mean('trajectory_total'):.3f}"
        )

    def _set_env_eating_reward_multiplier(self, multiplier: float):
        value = float(max(0.0, multiplier))
        try:
            setattr(self.env, "eating_reward_multiplier", value)
        except Exception:
            pass
        try:
            base_env = getattr(self.env, "unwrapped", None)
            if base_env is not None:
                setattr(base_env, "eating_reward_multiplier", value)
        except Exception:
            pass

    def run(
        self,
        candidates,
        species_being_evaluated = "cod",
        seed = None,
        is_evaluation = False,
        callback = noop,
        collect_plot_data = True,
        python_random_seed = None,
        simple_win_threshold = None,
        max_steps_override = None,
        initial_energy_scale = 1.0,
        energy_decay_per_cycle = 0.0,
    ):
        if python_random_seed is not None:
            random.seed(python_random_seed)
        self.env.reset(seed)
        self._set_training_reproduction_override(is_evaluation=is_evaluation)
        training_eating_reward_multiplier = float(
            max(0.0, getattr(self.settings, "training_eating_reward_multiplier", 1.0))
        )
        self._set_env_eating_reward_multiplier(
            training_eating_reward_multiplier if not is_evaluation else 1.0
        )
        if not is_evaluation:
            sparse_seed = seed if seed is not None else python_random_seed
            training_non_plankton_cell_fraction = float(
                np.clip(getattr(self.settings, "training_non_plankton_cell_fraction", 1.0), 0.0, 1.0)
            )
            training_non_plankton_min_cells = int(
                max(0, getattr(self.settings, "training_non_plankton_min_cells", 1))
            )
            training_cod_spawn_cells = int(
                max(0, getattr(self.settings, "training_cod_spawn_cells", 0))
            )
            if (
                training_non_plankton_cell_fraction < 1.0
                or training_cod_spawn_cells > 0
            ):
                self._sparsify_non_plankton_for_training(
                    cell_fraction=training_non_plankton_cell_fraction,
                    min_cells=training_non_plankton_min_cells,
                    cod_spawn_cells=training_cod_spawn_cells,
                    rng_seed=sparse_seed,
                )

            training_plankton_cell_fraction = float(
                np.clip(getattr(self.settings, "training_plankton_cell_fraction", 1.0), 0.0, 1.0)
            )
            training_plankton_biomass_scale = float(
                max(0.0, getattr(self.settings, "training_plankton_biomass_scale", 1.0))
            )
            if (
                training_plankton_cell_fraction < 1.0
                or abs(training_plankton_biomass_scale - 1.0) > 1e-9
            ):
                self._sparsify_plankton_for_training(
                    cell_fraction=training_plankton_cell_fraction,
                    biomass_scale=training_plankton_biomass_scale,
                    rng_seed=sparse_seed,
                )

            training_cod_prey_clear_radius = int(
                max(0, getattr(self.settings, "training_cod_prey_clear_radius", 0))
            )
            if training_cod_prey_clear_radius > 0:
                self._clear_prey_near_cod_for_training(
                    clear_radius=training_cod_prey_clear_radius,
                    rng_seed=sparse_seed,
                )
            self._sync_plankton_cluster_markers()
            if hasattr(self.env, "refresh_visual_scale_reference"):
                self.env.refresh_visual_scale_reference()
        fitness_method = str(self.settings.fitness_method).strip().lower()
        if fitness_method not in {"simple", "biomass_pct", "trajectory_shaped"}:
            raise ValueError(
                f"Unsupported fitness_method '{self.settings.fitness_method}'. "
                "Use 'simple', 'biomass_pct', or 'trajectory_shaped'."
            )
        biomass_scope = str(getattr(self.settings, "biomass_fitness_scope", "agent")).strip().lower()
        if (
            not is_evaluation
            and bool(getattr(self.settings, "training_force_agent_fitness_scope", False))
        ):
            biomass_scope = "agent"
        if biomass_scope not in {"agent", "base_species"}:
            raise ValueError(
                f"Unsupported biomass_fitness_scope '{self.settings.biomass_fitness_scope}'. "
                "Use 'agent' or 'base_species'."
            )

        if max_steps_override is not None:
            eval_horizon = max(1, int(max_steps_override))
        else:
            eval_horizon = self.settings.max_steps
        if fitness_method in {"biomass_pct", "trajectory_shaped"} and max_steps_override is None:
            eval_horizon = max(1, min(self.settings.max_steps, int(self.settings.fitness_eval_steps)))

        self._scale_all_species_energy(initial_energy_scale)

        fitness_target_species = self._resolve_fitness_target_species(species_being_evaluated, biomass_scope)
        initial_biomass, initial_energy = self._get_target_biomass_and_energy(fitness_target_species)
        current_biomass = initial_biomass
        current_energy = initial_energy
        prev_cycle_count = self.env.cycle_count
        episode_length = 0
        fitness = 0.0
        prev_biomass = initial_biomass
        prev_energy = initial_energy
        trajectory_return = 0.0
        trajectory_discount = 1.0
        trajectory_gamma = float(np.clip(getattr(self.settings, "trajectory_gamma", 0.98), 0.0, 1.0))
        trajectory_weight_biomass = float(getattr(self.settings, "trajectory_weight_biomass_delta", 1.0))
        trajectory_weight_energy = float(getattr(self.settings, "trajectory_weight_energy_delta", 0.05))
        trajectory_weight_terminal = float(getattr(self.settings, "trajectory_weight_terminal_biomass", 0.5))
        trajectory_crash_drop_pct = float(max(0.0, getattr(self.settings, "trajectory_crash_drop_pct", 8.0)))
        trajectory_crash_penalty = float(max(0.0, getattr(self.settings, "trajectory_crash_penalty", 2.0)))
        trajectory_low_floor_pct = float(max(0.0, getattr(self.settings, "trajectory_low_biomass_floor_pct", 30.0)))
        trajectory_low_penalty = float(max(0.0, getattr(self.settings, "trajectory_low_biomass_penalty", 0.0)))
        training_low_energy_floor_pct = float(
            np.clip(getattr(self.settings, "training_low_energy_floor_pct", 0.0), 0.0, 1.0)
        )
        training_low_energy_penalty = float(
            max(0.0, getattr(self.settings, "training_low_energy_penalty", 0.0))
        )
        training_low_energy_biomass_floor_pct = float(
            np.clip(getattr(self.settings, "training_low_energy_biomass_floor_pct", 0.0), 0.0, 1.0)
        )
        training_low_energy_biomass_max_loss_per_cycle = float(
            np.clip(getattr(self.settings, "training_low_energy_biomass_max_loss_per_cycle", 0.0), 0.0, 1.0)
        )
        training_low_energy_biomass_target_only = bool(
            getattr(self.settings, "training_low_energy_biomass_target_only", True)
        )
        training_energy_reference = float(self.settings.max_energy)
        if not is_evaluation:
            # In training we may down-scale initial energy. Evaluate "low energy" against
            # that scaled budget so floor settings remain meaningful.
            training_energy_reference *= max(float(initial_energy_scale), 1e-8)
        training_energy_reference = max(training_energy_reference, 1e-8)
        trajectory_component_biomass = 0.0
        trajectory_component_energy = 0.0
        trajectory_component_crash_penalty = 0.0
        trajectory_component_low_penalty = 0.0
        trajectory_component_low_energy_penalty = 0.0
        trajectory_component_terminal = 0.0
        callback(self.env.world, fitness, False)

        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()) and episode_length < eval_horizon:
            agent = self.env.agent_selection
            if agent == "plankton":
                self.env.step(self.empty_action)
            else:
                obs, reward, termination, truncation, info = self.env.last()
                candidate = candidates[agent]
                action_values = candidate.forward(obs.reshape(-1, model.INPUT_SIZE))
                action_values = action_values.reshape(self.settings.world_size, self.settings.world_size, model.OUTPUT_SIZE)
                # mean_action_values = action_values.mean(axis=(0, 1))

                self.env.step(action_values)

            current_cycle_count = self.env.cycle_count
            if current_cycle_count > prev_cycle_count:
                episode_length = current_cycle_count
                prev_cycle_count = current_cycle_count
                self._apply_energy_decay(energy_decay_per_cycle)
                if not is_evaluation:
                    self._apply_training_low_energy_biomass_loss(
                        target_species=fitness_target_species,
                        energy_reference=training_energy_reference,
                        floor_pct=training_low_energy_biomass_floor_pct,
                        max_loss_per_cycle=training_low_energy_biomass_max_loss_per_cycle,
                        target_only=training_low_energy_biomass_target_only,
                    )
                current_biomass, current_energy = self._get_target_biomass_and_energy(fitness_target_species)

                if fitness_method == "simple":
                    fitness = float(episode_length)
                elif fitness_method == "biomass_pct":
                    fitness = (
                        100.0
                        * (current_biomass - initial_biomass)
                        / max(initial_biomass, 1e-8)
                    )
                else:
                    biomass_delta_pct = (
                        100.0 * (current_biomass - prev_biomass) / max(initial_biomass, 1e-8)
                    )
                    energy_delta = current_energy - prev_energy
                    step_reward = (
                        trajectory_weight_biomass * biomass_delta_pct
                        + trajectory_weight_energy * energy_delta
                    )
                    trajectory_component_biomass += trajectory_discount * (
                        trajectory_weight_biomass * biomass_delta_pct
                    )
                    trajectory_component_energy += trajectory_discount * (
                        trajectory_weight_energy * energy_delta
                    )

                    if trajectory_crash_drop_pct > 0.0 and prev_biomass > 1e-8:
                        crash_drop = 100.0 * max(0.0, prev_biomass - current_biomass) / prev_biomass
                        if crash_drop >= trajectory_crash_drop_pct:
                            crash_term = trajectory_crash_penalty * (crash_drop / trajectory_crash_drop_pct)
                            step_reward -= crash_term
                            trajectory_component_crash_penalty -= trajectory_discount * crash_term

                    if trajectory_low_penalty > 0.0 and trajectory_low_floor_pct > 0.0:
                        current_pct_of_initial = 100.0 * current_biomass / max(initial_biomass, 1e-8)
                        if current_pct_of_initial < trajectory_low_floor_pct:
                            deficit = (trajectory_low_floor_pct - current_pct_of_initial) / max(
                                trajectory_low_floor_pct, 1e-8
                            )
                            low_term = trajectory_low_penalty * deficit
                            step_reward -= low_term
                            trajectory_component_low_penalty -= trajectory_discount * low_term

                    if (
                        not is_evaluation
                        and training_low_energy_penalty > 0.0
                        and training_low_energy_floor_pct > 0.0
                    ):
                        current_energy_pct = current_energy / training_energy_reference
                        if current_energy_pct < training_low_energy_floor_pct:
                            deficit = (training_low_energy_floor_pct - current_energy_pct) / max(
                                training_low_energy_floor_pct, 1e-8
                            )
                            low_energy_term = training_low_energy_penalty * deficit
                            step_reward -= low_energy_term
                            trajectory_component_low_energy_penalty -= (
                                trajectory_discount * low_energy_term
                            )

                    trajectory_return += trajectory_discount * step_reward
                    trajectory_discount *= trajectory_gamma
                    terminal_pct = (
                        100.0 * (current_biomass - initial_biomass) / max(initial_biomass, 1e-8)
                    )
                    fitness = trajectory_return + trajectory_weight_terminal * terminal_pct
                    prev_biomass = current_biomass
                    prev_energy = current_energy

                if collect_plot_data and self.settings.enable_plotting and (is_evaluation == False or self.env.render_mode != "none"):
                    process_data({
                        'species': species_being_evaluated if not is_evaluation else None,
                        'agent_index': self.agent_index,
                        'eval_index': self.eval_index,
                        'step': episode_length,
                        'fitness': fitness,
                        'world': self.env.world
                    }, self.env.plot_data)
                callback(self.env.world, fitness, False)

                if self.settings.stop_on_single_species_left:
                    alive_base_species = self._get_alive_base_species()
                    if len(alive_base_species) <= 1:
                        winner = alive_base_species[0] if alive_base_species else "none"
                        if not self.env.reason:
                            self.env.reason = f"single_species_left:{winner}"
                        break

                if fitness_method == "simple" and simple_win_threshold is not None and fitness > float(simple_win_threshold):
                    if not self.env.reason:
                        self.env.reason = "simple_win_threshold_reached"
                    break
        
        if fitness_method == "biomass_pct":
            fitness = (
                100.0
                * (current_biomass - initial_biomass)
                / max(initial_biomass, 1e-8)
            )
            self._last_run_diagnostics = {}
        elif fitness_method == "trajectory_shaped":
            terminal_pct = (
                100.0 * (current_biomass - initial_biomass) / max(initial_biomass, 1e-8)
            )
            trajectory_component_terminal = trajectory_weight_terminal * terminal_pct
            fitness = trajectory_return + trajectory_component_terminal
            self._last_run_diagnostics = {
                "trajectory_component_biomass": float(trajectory_component_biomass),
                "trajectory_component_energy": float(trajectory_component_energy),
                "trajectory_component_crash_penalty": float(trajectory_component_crash_penalty),
                "trajectory_component_low_biomass_penalty": float(trajectory_component_low_penalty),
                "trajectory_component_low_energy_penalty": float(
                    trajectory_component_low_energy_penalty
                ),
                "trajectory_component_terminal": float(trajectory_component_terminal),
                "trajectory_total": float(fitness),
            }
        else:
            self._last_run_diagnostics = {}

        if self.settings.enable_logging:
            print("end of simulation", fitness, episode_length)
        callback(self.env.world, fitness, True)
        return fitness, episode_length, self.env.reason

    def evaluate_population(self):
        """
        Evaluate each model in the population for every species.
        When evaluating a candidate for a species S, for every other species (not S),
        we use the best model from previous generations (if available) to decide their actions.
        Returns a dictionary mapping species to a list of (chromosome, fitness) tuples.
        """
        if self.settings.seed >= 0:
            # Deterministic per-generation RNG for eval setup.
            eval_rng = random.Random(self.settings.seed + self.current_generation * 1_000_003)
        else:
            eval_rng = self.rng

        # Same eval seeds for every agent in the generation.
        eval_seeds = [eval_rng.randrange(100000) for _ in range(self.settings.agent_evaluations)]
        fitnesses = {species: [] for species in self.species_list}
        fitness_method = str(self.settings.fitness_method).strip().lower()

        end_reasons = []
        tasks = []
        population_state = {
            species: [m.state_dict() for m in self.population[species]]
            for species in self.species_list
        }
        opponent_population_state = self._select_opponent_population_state(population_state)
        if self.settings.enable_logging:
            snapshot_every = max(1, int(getattr(self.settings, "opponent_snapshot_every", 1)))
            if snapshot_every > 1:
                print(
                    f"Opponent snapshot: every={snapshot_every}, "
                    f"source_generation={self.opponent_snapshot_generation}"
                )
        paired_eval = bool(getattr(self.settings, "paired_opponent_evaluation", True))
        lock_same_base_teammates = bool(getattr(self.settings, "lock_same_base_teammates", True))
        paired_context = {}
        if paired_eval:
            for species in self.species_list:
                other_species = [s for s in self.species_list if s != species]
                for eval_index in range(self.settings.agent_evaluations):
                    other_indices = {}
                    for other_species_name in other_species:
                        other_indices[other_species_name] = eval_rng.randint(0, self.settings.num_agents - 1)
                    paired_context[(species, eval_index)] = {
                        "other_indices": other_indices,
                        "python_random_seed": eval_rng.randrange(2**32),
                    }

        for idx in range(self.settings.num_agents):
            for species in self.species_list:
                other_species = [s for s in self.species_list if s != species]
                for eval_index in range(self.settings.agent_evaluations):
                    if paired_eval:
                        context = paired_context[(species, eval_index)]
                        other_indices = dict(context["other_indices"])
                        python_random_seed = int(context["python_random_seed"])
                        if lock_same_base_teammates:
                            species_base = self.species_map[species].base_species
                            for other_species_name in other_species:
                                if self.species_map[other_species_name].base_species == species_base:
                                    other_indices[other_species_name] = idx
                    else:
                        other_indices = {}
                        # pick random agent from other species
                        for other_species_name in other_species:
                            if (
                                lock_same_base_teammates
                                and self.species_map[other_species_name].base_species
                                == self.species_map[species].base_species
                            ):
                                other_species_idx = idx
                            else:
                                other_species_idx = eval_rng.randint(0, self.settings.num_agents - 1)
                            other_indices[other_species_name] = other_species_idx
                        python_random_seed = eval_rng.randrange(2**32)

                    tasks.append({
                        "agent_index": idx,
                        "species": species,
                        "eval_index": eval_index,
                        "other_indices": other_indices,
                        "map_seed": eval_seeds[eval_index],
                        "python_random_seed": python_random_seed,
                    })

        local_model_cache = {species: {} for species in self.species_list}
        local_opponent_model_cache = {species: {} for species in self.species_list}

        def _get_local_model(species, idx, for_opponent=False):
            cache = local_opponent_model_cache[species] if for_opponent else local_model_cache[species]
            model = cache.get(idx)
            if model is None:
                source_state = opponent_population_state if for_opponent else population_state
                model = Model(chromosome=source_state[species][idx])
                cache[idx] = model
            return model

        def _run_eval_task_local(task, simple_win_threshold=None):
            override_policy = str(task.get("override_policy", "")).strip().lower()
            if override_policy == "random":
                self_model = RandomBaselinePolicy(seed=int(task.get("override_policy_seed", task["python_random_seed"])))
            else:
                self_model = _get_local_model(task["species"], task["agent_index"])
            eval_species = {task["species"]: self_model}
            for sp, idx in task["other_indices"].items():
                eval_species[sp] = _get_local_model(sp, idx, for_opponent=True)

            self.agent_index = task["agent_index"]
            self.eval_index = task["eval_index"]

            map_seed = task["map_seed"]
            python_random_seed = task["python_random_seed"]
            start_time = time.time()
            fitness, episode_length, end_reason = self.run(
                eval_species,
                task["species"],
                map_seed,
                is_evaluation=False,
                callback=noop,
                collect_plot_data=True,
                python_random_seed=python_random_seed,
                simple_win_threshold=simple_win_threshold,
                max_steps_override=task.get("max_steps_override"),
                initial_energy_scale=task.get("initial_energy_scale", 1.0),
                energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
            )
            while episode_length == 0:
                if self.settings.enable_logging:
                    print("something went wrong update this seed")
                map_seed = random.randrange(100000)
                python_random_seed = random.randrange(2**32)
                fitness, episode_length, end_reason = self.run(
                    eval_species,
                    task["species"],
                    map_seed,
                    is_evaluation=False,
                    callback=noop,
                    collect_plot_data=True,
                    python_random_seed=python_random_seed,
                    simple_win_threshold=simple_win_threshold,
                    max_steps_override=task.get("max_steps_override"),
                    initial_energy_scale=task.get("initial_energy_scale", 1.0),
                    energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
                )

            end_time = time.time()
            return {
                "agent_index": task["agent_index"],
                "species": task["species"],
                "eval_index": task["eval_index"],
                "fitness": fitness,
                "episode_length": episode_length,
                "end_reason": end_reason,
                "steps_per_sec": episode_length / max(end_time - start_time, 1e-9),
                "diagnostics": dict(getattr(self, "_last_run_diagnostics", {})),
            }

        # Optional custom pipeline:
        # 1) short eval for all candidates
        # 2) long eval for top candidates
        # 3) blended score for selection
        uses_biomass_style_fitness = fitness_method in {"biomass_pct", "trajectory_shaped"}
        two_stage_enabled = bool(self.settings.two_stage_eval_enabled) and uses_biomass_style_fitness
        if uses_biomass_style_fitness:
            short_steps = int(self.settings.fitness_eval_steps)
        else:
            short_steps = int(self.settings.max_steps)
        if two_stage_enabled:
            short_steps = int(self.settings.short_eval_steps)
        short_steps = max(1, short_steps)
        long_steps = max(short_steps, int(self.settings.long_eval_steps))
        long_weight = float(np.clip(self.settings.long_eval_weight, 0.0, 1.0))
        top_fraction = float(np.clip(self.settings.long_eval_top_fraction, 0.0, 1.0))
        top_n = max(1, int(np.ceil(self.settings.num_agents * top_fraction))) if top_fraction > 0 else 0
        relative_baseline_enabled = bool(self.settings.relative_baseline_enabled) and fitness_method == "biomass_pct"
        relative_baseline_policy = str(getattr(self.settings, "relative_baseline_policy", "random")).strip().lower()
        if relative_baseline_enabled and relative_baseline_policy != "random":
            raise ValueError(
                f"Unsupported relative_baseline_policy '{self.settings.relative_baseline_policy}'. "
                "Use 'random'."
            )
        relative_baseline_fallback = bool(
            getattr(self.settings, "relative_baseline_fallback_to_raw_when_flat", True)
        )
        relative_baseline_flat_std_threshold = float(
            max(0.0, getattr(self.settings, "relative_baseline_flat_std_threshold", 0.25))
        )

        initial_energy_scale = max(0.0, float(self.settings.training_initial_energy_scale))
        energy_decay_per_cycle = float(np.clip(self.settings.training_energy_decay_per_cycle, 0.0, 0.95))

        use_custom_pipeline = (
            two_stage_enabled
            or relative_baseline_enabled
            or abs(initial_energy_scale - 1.0) > 1e-9
            or energy_decay_per_cycle > 0.0
        )

        if use_custom_pipeline:
            def _execute_tasks(task_list):
                if not task_list:
                    return []
                local_results = []
                if self.settings.num_workers <= 1:
                    for task in task_list:
                        local_results.append(_run_eval_task_local(task, simple_win_threshold=None))
                else:
                    with ProcessPoolExecutor(
                        max_workers=self.settings.num_workers,
                        initializer=_init_worker,
                        initargs=(
                            self.settings,
                            self.env.render_mode,
                            self.env.map_folder,
                            population_state,
                            opponent_population_state,
                        ),
                    ) as executor:
                        for res in executor.map(_run_eval_task_worker, task_list, chunksize=1):
                            local_results.append(res)
                return local_results

            def _baseline_seed(task):
                species_code = sum((i + 1) * ord(ch) for i, ch in enumerate(task["species"]))
                return int(
                    (
                        int(task["map_seed"]) * 1_000_003
                        + int(task["eval_index"]) * 97
                        + species_code
                        + 17
                    )
                    % (2**32)
                )

            short_tasks = []
            for task in tasks:
                t = dict(task)
                t["max_steps_override"] = short_steps
                t["initial_energy_scale"] = initial_energy_scale
                t["energy_decay_per_cycle"] = energy_decay_per_cycle
                short_tasks.append(t)

            short_results = _execute_tasks(short_tasks)
            if fitness_method == "trajectory_shaped" and self.settings.enable_logging:
                self._log_trajectory_diagnostics(short_results, "short")
            short_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in short_results}
            end_reasons.extend([r["end_reason"] for r in short_results])
            short_scores_by_key = {
                (r["agent_index"], r["species"], r["eval_index"]): float(r["fitness"])
                for r in short_results
            }
            short_raw_scores_by_key = dict(short_scores_by_key)
            short_fallback_to_raw = False

            if relative_baseline_enabled:
                baseline_short_tasks = []
                for task in short_tasks:
                    t = dict(task)
                    t["override_policy"] = "random"
                    t["override_policy_seed"] = _baseline_seed(task)
                    baseline_short_tasks.append(t)
                baseline_short_results = _execute_tasks(baseline_short_tasks)
                baseline_short_by_key = {
                    (r["agent_index"], r["species"], r["eval_index"]): r
                    for r in baseline_short_results
                }
                for key, score in list(short_scores_by_key.items()):
                    baseline_result = baseline_short_by_key.get(key)
                    if baseline_result is None:
                        continue
                    short_scores_by_key[key] = score - float(baseline_result["fitness"])

                if relative_baseline_fallback:
                    adjusted_vals = np.asarray(list(short_scores_by_key.values()), dtype=np.float32)
                    if adjusted_vals.size > 0 and float(np.std(adjusted_vals)) < relative_baseline_flat_std_threshold:
                        short_scores_by_key = dict(short_raw_scores_by_key)
                        short_fallback_to_raw = True

            short_avg = {}
            for idx in range(self.settings.num_agents):
                for species in self.species_list:
                    vals = [
                        short_scores_by_key[(idx, species, eval_index)]
                        for eval_index in range(self.settings.agent_evaluations)
                    ]
                    short_avg[(idx, species)] = float(np.mean(vals))

            long_candidates = set()
            if two_stage_enabled and long_steps > short_steps and top_n > 0:
                for species in self.species_list:
                    scored = [(idx, short_avg[(idx, species)]) for idx in range(self.settings.num_agents)]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for idx, _ in scored[:top_n]:
                        long_candidates.add((idx, species))

            long_by_key = {}
            long_scores_by_key = {}
            long_fallback_to_raw = False
            if long_candidates:
                long_tasks = []
                for task in tasks:
                    key = (task["agent_index"], task["species"])
                    if key not in long_candidates:
                        continue
                    t = dict(task)
                    t["max_steps_override"] = long_steps
                    t["initial_energy_scale"] = initial_energy_scale
                    t["energy_decay_per_cycle"] = energy_decay_per_cycle
                    long_tasks.append(t)
                long_results = _execute_tasks(long_tasks)
                if fitness_method == "trajectory_shaped" and self.settings.enable_logging:
                    self._log_trajectory_diagnostics(long_results, "long")
                long_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in long_results}
                end_reasons.extend([r["end_reason"] for r in long_results])
                long_scores_by_key = {
                    (r["agent_index"], r["species"], r["eval_index"]): float(r["fitness"])
                    for r in long_results
                }
                long_raw_scores_by_key = dict(long_scores_by_key)

                if relative_baseline_enabled:
                    baseline_long_tasks = []
                    for task in long_tasks:
                        t = dict(task)
                        t["override_policy"] = "random"
                        t["override_policy_seed"] = _baseline_seed(task)
                        baseline_long_tasks.append(t)
                    baseline_long_results = _execute_tasks(baseline_long_tasks)
                    baseline_long_by_key = {
                        (r["agent_index"], r["species"], r["eval_index"]): r
                        for r in baseline_long_results
                    }
                    for key, score in list(long_scores_by_key.items()):
                        baseline_result = baseline_long_by_key.get(key)
                        if baseline_result is None:
                            continue
                        long_scores_by_key[key] = score - float(baseline_result["fitness"])

                    if relative_baseline_fallback:
                        adjusted_vals = np.asarray(list(long_scores_by_key.values()), dtype=np.float32)
                        if adjusted_vals.size > 0 and float(np.std(adjusted_vals)) < relative_baseline_flat_std_threshold:
                            long_scores_by_key = dict(long_raw_scores_by_key)
                            long_fallback_to_raw = True

            for idx in range(self.settings.num_agents):
                self.agent_index = idx
                for species in self.species_list:
                    score = short_avg[(idx, species)]
                    if (idx, species) in long_candidates:
                        long_vals = [
                            long_scores_by_key[(idx, species, eval_index)]
                            for eval_index in range(self.settings.agent_evaluations)
                            if (idx, species, eval_index) in long_scores_by_key
                        ]
                        if long_vals:
                            long_score = float(np.mean(long_vals))
                            score = (1.0 - long_weight) * score + long_weight * long_score

                    fitnesses[species].append((self.population[species][idx].state_dict(), float(score)))
                    if self.settings.enable_logging:
                        print(f'finished eval for species: {species}, idx {idx}, fitness: {score:.3f}')

                    # Ensure plot data reflects the final score used for selection.
                    if self.settings.enable_plotting:
                        for eval_index in range(self.settings.agent_evaluations):
                            key = (idx, species, eval_index)
                            base_res = long_by_key.get(key, short_by_key.get(key))
                            if base_res is None:
                                continue
                            process_data({
                                'species': species,
                                'agent_index': idx,
                                'eval_index': eval_index,
                                'step': base_res["episode_length"],
                                'fitness': float(score),
                            }, self.env.plot_data)

            if self.settings.enable_logging:
                print(
                    f"Custom eval: short_steps={short_steps}, long_steps={long_steps}, "
                    f"top_n={top_n}, long_weight={long_weight:.2f}, "
                    f"energy_scale={initial_energy_scale:.2f}, energy_decay={energy_decay_per_cycle:.3f}, "
                    f"relative_baseline={relative_baseline_enabled}, "
                    f"baseline_fallback_short={short_fallback_to_raw}, "
                    f"baseline_fallback_long={long_fallback_to_raw}"
                )
                print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
            return fitnesses

        results = []
        noncustom_rows = []
        if self.settings.num_workers <= 1:
            tasks_by_key = {
                (task["agent_index"], task["species"], task["eval_index"]): task
                for task in tasks
            }
            best_avg_so_far = {species: -float("inf") for species in self.species_list}

            for idx in range(self.settings.num_agents):
                self.agent_index = idx
                for species in self.species_list:
                    evals_fitness = []
                    for eval_index in range(self.settings.agent_evaluations):
                        task = tasks_by_key[(idx, species, eval_index)]
                        self.eval_index = eval_index

                        simple_win_threshold = None
                        can_short_circuit_last_candidate = (
                            fitness_method == "simple"
                            and self.settings.stop_last_candidate_when_winner
                            and idx == self.settings.num_agents - 1
                            and np.isfinite(best_avg_so_far[species])
                        )
                        if can_short_circuit_last_candidate:
                            completed_sum = float(sum(evals_fitness))
                            guaranteed_min_avg = completed_sum / self.settings.agent_evaluations
                            if guaranteed_min_avg > best_avg_so_far[species]:
                                skipped = self.settings.agent_evaluations - len(evals_fitness)
                                if skipped > 0:
                                    evals_fitness.extend([0.0] * skipped)
                                    end_reasons.extend(["skipped_last_candidate_already_won"] * skipped)
                                    if self.settings.enable_logging:
                                        print(
                                            f"idx {idx}, species {species}: short-circuiting {skipped} "
                                            f"remaining eval(s); winner already guaranteed."
                                        )
                                break
                            simple_win_threshold = (
                                (best_avg_so_far[species] * self.settings.agent_evaluations)
                                - completed_sum
                            )

                        r = _run_eval_task_local(task, simple_win_threshold=simple_win_threshold)
                        noncustom_rows.append(r)
                        end_reasons.append(r["end_reason"])
                        evals_fitness.append(r["fitness"])
                        if self.settings.enable_logging:
                            print(
                                f'idx {idx}, eval {eval_index}, fitness {r["fitness"]:.1f}, '
                                f'episode_length {r["episode_length"]}, steps/sec {r["steps_per_sec"]:.2f}'
                            )

                        if can_short_circuit_last_candidate:
                            guaranteed_min_avg = float(sum(evals_fitness)) / self.settings.agent_evaluations
                            if guaranteed_min_avg > best_avg_so_far[species]:
                                skipped = self.settings.agent_evaluations - len(evals_fitness)
                                if skipped > 0:
                                    evals_fitness.extend([0.0] * skipped)
                                    end_reasons.extend(["skipped_last_candidate_already_won"] * skipped)
                                    if self.settings.enable_logging:
                                        print(
                                            f"idx {idx}, species {species}: short-circuiting {skipped} "
                                            f"remaining eval(s); winner already guaranteed."
                                        )
                                break

                    avg_fitness = float(sum(evals_fitness)) / self.settings.agent_evaluations
                    fitnesses[species].append((self.population[species][idx].state_dict(), avg_fitness))
                    best_avg_so_far[species] = max(best_avg_so_far[species], avg_fitness)
                    if self.settings.enable_logging:
                        print(f'finished eval for species: {species}, fitness: {avg_fitness:.1f}')

            if self.settings.enable_logging:
                if fitness_method == "trajectory_shaped":
                    self._log_trajectory_diagnostics(noncustom_rows, "single_pass")
                print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
            return fitnesses
        else:
            with ProcessPoolExecutor(
                max_workers=self.settings.num_workers,
                initializer=_init_worker,
                initargs=(
                    self.settings,
                    self.env.render_mode,
                    self.env.map_folder,
                    population_state,
                    opponent_population_state,
                ),
            ) as executor:
                # Small tasks improve load-balancing across workers.
                for res in executor.map(_run_eval_task_worker, tasks, chunksize=1):
                    results.append(res)

        # Aggregate results in deterministic order.
        results_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in results}
        for idx in range(self.settings.num_agents):
            self.agent_index = idx
            for species in self.species_list:
                evals_fitness = []
                for eval_index in range(self.settings.agent_evaluations):
                    r = results_by_key[(idx, species, eval_index)]
                    self.eval_index = eval_index
                    end_reasons.append(r["end_reason"])
                    evals_fitness.append(r["fitness"])
                    if self.settings.enable_logging:
                        print(f'idx {idx}, eval {eval_index}, fitness {r["fitness"]:.1f}, episode_length {r["episode_length"]}, steps/sec {r["steps_per_sec"]:.2f}')

                    # Minimal plot data update for parallel runs (preserves generation stats).
                    if self.settings.num_workers > 1 and self.settings.enable_plotting:
                        process_data({
                            'species': species,
                            'agent_index': idx,
                            'eval_index': eval_index,
                            'step': r["episode_length"],
                            'fitness': r["fitness"],
                        }, self.env.plot_data)

                avg_fitness = sum(evals_fitness) / len(evals_fitness)
                fitnesses[species].append((self.population[species][idx].state_dict(), avg_fitness))
                if self.settings.enable_logging:
                    print(f'finished eval for species: {species}, fitness: {avg_fitness:.1f}')

        if self.settings.enable_logging:
            if fitness_method == "trajectory_shaped":
                self._log_trajectory_diagnostics(results, "parallel_pass")
            print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = {}
        inject_count_cfg = max(0, int(getattr(self.settings, "early_random_injection_count", 0)))
        inject_generations = max(0, int(getattr(self.settings, "early_random_injection_generations", 0)))
        inject_active = self.current_generation < inject_generations

        for species in self.species_list:
            current_population = fitnesses[species]

            # Select elites.
            elites = evolution.elitism_selection(current_population, self.settings.elitism_selection)
            next_pop = []
            inject_count = min(inject_count_cfg if inject_active else 0, max(self.settings.num_agents - 1, 0))
            offspring_target = max(0, self.settings.num_agents - 1 - inject_count)

            while len(next_pop) < offspring_target:
                (p1, _), (p2, _) = evolution.tournament_selection(elites, 2, self.settings.tournament_selection)
                current_eta = min(10.0, self.settings.sbx_eta * (self.settings.sbx_eta_decay ** self.current_generation))
                c1_weights, c2_weights = evolution.sbx_crossover(p1, p2, current_eta)
                current_mutation_rate = max(self.settings.mutation_rate_min, self.settings.mutation_rate * (self.settings.mutation_rate_decay ** self.current_generation))
                evolution.mutation(c1_weights, current_mutation_rate, current_mutation_rate)
                evolution.mutation(c2_weights, current_mutation_rate, current_mutation_rate)
                next_pop.append(c1_weights)
                if len(next_pop) < offspring_target:
                    next_pop.append(c2_weights)

            # Update best fitness/agent.
            best_for_species = max(current_population, key=lambda x: x[1])
            if best_for_species[1] > self.best_fitness[species] + 0.01:
                self.best_fitness[species] = best_for_species[1]
                self.best_agent[species] = best_for_species[0]
                model = Model(chromosome=self.best_agent[species])
                model.save(f'{self.settings.folder}/agents/{self.current_generation}_${species}_{self.best_fitness[species]}.npy')
            elif self.best_agent[species] is None:
                self.best_agent[species] = best_for_species[0]

            # add best agent to the population
            next_pop.append(self.best_agent[species])
            for _ in range(inject_count):
                next_pop.append(Model().state_dict())

            new_population[species] = [Model(chromosome=chrom) for chrom in next_pop]
            if len(new_population[species]) != self.settings.num_agents:
                raise RuntimeError(
                    f"Population size mismatch for {species}: "
                    f"{len(new_population[species])} != {self.settings.num_agents}"
                )

        for species in self.species_list:
            # shuffle the population
            self.rng.shuffle(new_population[species])

        self.population = new_population

    def _get_generation_best_states(self, fitnesses):
        generation_best_states = {}
        best_species = None
        best_score = -float("inf")
        for species in self.species_list:
            best_state, best_species_score = max(fitnesses[species], key=lambda x: x[1])
            generation_best_states[species] = best_state
            if best_species_score > best_score:
                best_score = best_species_score
                best_species = species
        return generation_best_states, best_species, best_score

    def _build_generation_champion_models(self, fitnesses):
        generation_best_states, best_species, best_score = self._get_generation_best_states(fitnesses)

        champion_models = {}
        for species in self.species_list:
            if species == best_species:
                benchmark_state = generation_best_states[species]
            elif self.best_agent.get(species) is not None:
                benchmark_state = self.best_agent[species]
            else:
                benchmark_state = generation_best_states[species]
            champion_models[species] = Model(chromosome=benchmark_state)

        return champion_models, best_species, best_score

    def _evaluate_generation_champions(self, fitnesses):
        every = max(1, int(self.settings.champion_progress_every))
        if (
            not self.settings.champion_progress_enabled
            or (self.current_generation % every) != 0
        ):
            self.champion_progress_history.append(np.nan)
            return

        # Build a single "champion benchmark" policy set:
        # one globally best champion for this generation + strongest available opponents.
        champion_models, best_species, _ = self._build_generation_champion_models(fitnesses)

        horizon = max(1, int(self.settings.champion_progress_steps))
        episodes = max(1, int(getattr(self.settings, "champion_progress_episodes", 1)))
        lengths = []
        for episode_idx in range(episodes):
            seed = int(self.champion_progress_seed + self.current_generation * 9973 + episode_idx * 101)
            _, episode_length, _ = self.run(
                candidates=champion_models,
                species_being_evaluated=best_species if best_species is not None else self.species_list[0],
                seed=seed,
                is_evaluation=True,
                callback=noop,
                collect_plot_data=False,
                python_random_seed=seed,
                max_steps_override=horizon,
            )
            lengths.append(float(episode_length))

        self.champion_progress_history.append(float(np.mean(lengths)))

    def _evaluate_fixed_validation(self, fitnesses):
        every = max(1, int(getattr(self.settings, "fixed_validation_every", 1)))
        if (
            not bool(getattr(self.settings, "fixed_validation_enabled", False))
            or (self.current_generation % every) != 0
        ):
            self.fixed_validation_history.append(np.nan)
            for base in const.ACTING_BASE_SPECIES:
                self.fixed_validation_by_base_history[base].append(np.nan)
            return

        generation_best_states, best_species, _ = self._get_generation_best_states(fitnesses)
        validation_models = {
            species: Model(chromosome=state)
            for species, state in generation_best_states.items()
        }
        horizon = max(1, int(getattr(self.settings, "fixed_validation_steps", self.settings.fitness_eval_steps)))
        episodes = max(1, int(getattr(self.settings, "fixed_validation_episodes", 1)))
        overall_values = []
        by_base_values = {base: [] for base in const.ACTING_BASE_SPECIES}
        for episode_idx in range(episodes):
            seed = int(self.fixed_validation_seed + episode_idx * 9973)
            start_totals = None
            end_totals = None

            def _capture(world, _fitness, done):
                nonlocal start_totals, end_totals
                if start_totals is None:
                    start_totals = self._get_base_biomass_totals(world)
                if done:
                    end_totals = self._get_base_biomass_totals(world)

            fitness, episode_length, _ = self.run(
                candidates=validation_models,
                species_being_evaluated=best_species if best_species is not None else self.species_list[0],
                seed=seed,
                is_evaluation=True,
                callback=_capture,
                collect_plot_data=False,
                python_random_seed=seed,
                max_steps_override=horizon,
            )
            if self.fixed_validation_metric == "survival":
                episode_value = float(episode_length)
                overall_values.append(episode_value)
                for base in const.ACTING_BASE_SPECIES:
                    by_base_values[base].append(episode_value)
            else:
                if start_totals is None:
                    start_totals = self._get_base_biomass_totals()
                if end_totals is None:
                    end_totals = self._get_base_biomass_totals()

                base_scores = []
                for base in const.ACTING_BASE_SPECIES:
                    b0 = float(start_totals.get(base, 0.0))
                    b1 = float(end_totals.get(base, b0))
                    score = 100.0 * (b1 - b0) / max(b0, 1e-8)
                    by_base_values[base].append(score)
                    base_scores.append(score)
                overall_values.append(float(np.mean(base_scores)) if base_scores else float(fitness))

        self.fixed_validation_history.append(float(np.mean(overall_values)))
        for base in const.ACTING_BASE_SPECIES:
            vals = by_base_values[base]
            self.fixed_validation_by_base_history[base].append(float(np.mean(vals)) if vals else np.nan)

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        self._evaluate_generation_champions(fitnesses)
        self._evaluate_fixed_validation(fitnesses)
        if self.settings.enable_logging:
            print(f"Generation {self.current_generation} complete. Fitnesses: { {sp: max(fit, key=lambda x: x[1])[1] for sp, fit in fitnesses.items()} }")
            if self.champion_progress_history and np.isfinite(self.champion_progress_history[-1]):
                print(f"Champion benchmark survival: {self.champion_progress_history[-1]:.1f}")
            if self.fixed_validation_history and np.isfinite(self.fixed_validation_history[-1]):
                fixed_label = "survival cycles" if self.fixed_validation_metric == "survival" else "fitness"
                print(f"Fixed validation ({fixed_label}): {self.fixed_validation_history[-1]:.3f}")
                if self.fixed_validation_metric == "fitness":
                    by_base = ", ".join(
                        f"{base}={self.fixed_validation_by_base_history[base][-1]:.2f}"
                        for base in const.ACTING_BASE_SPECIES
                        if self.fixed_validation_by_base_history[base]
                    )
                    print(f"Fixed validation by base: {by_base}")
        if self.settings.enable_plotting:
            generations_data = update_generations_data(self.settings, self.current_generation, self.env.plot_data)
            plot_generations(
                self.settings,
                generations_data,
                champion_progress=self.champion_progress_history,
                fixed_validation=self.fixed_validation_history,
                fixed_validation_metric=self.fixed_validation_metric,
                fixed_validation_by_species=self.fixed_validation_by_base_history,
            )
        self.env.plot_data = {}
        # Evolve the population based on fitnesses.
        self.evolve_population(fitnesses)

    def train(self):
        for i in range(self.settings.generations_per_run):
            self.run_generation()
            # if i > 25 and i % 5 == 0:
            #     self.optimize_params()
            # Optionally, save best models or log additional statistics.

    def evaluate(self, model_paths = [], callback = noop):
        candidates = {}
        for model_path in model_paths:
            model = Model(
                chromosome=np.load(model_path['path'])
            )
            candidates[model_path['species']] = model

        return self.run(candidates, "cod", None, True, callback)


def _init_worker(settings: Settings, render_mode: str, map_folder: str, population_state, opponent_population_state):
    global _WORKER_RUNNER
    global _WORKER_POPULATION_STATE
    global _WORKER_OPPONENT_STATE
    global _WORKER_MODEL_CACHE
    global _WORKER_OPPONENT_MODEL_CACHE
    _WORKER_RUNNER = PettingZooRunner(
        settings=settings,
        render_mode=render_mode,
        map_folder=map_folder,
        build_population=False,
    )
    _WORKER_POPULATION_STATE = population_state
    _WORKER_OPPONENT_STATE = opponent_population_state
    _WORKER_MODEL_CACHE = {species: {} for species in population_state}
    _WORKER_OPPONENT_MODEL_CACHE = {species: {} for species in opponent_population_state}


def _run_eval_task_worker(task):
    global _WORKER_RUNNER
    global _WORKER_POPULATION_STATE
    global _WORKER_OPPONENT_STATE
    global _WORKER_MODEL_CACHE
    global _WORKER_OPPONENT_MODEL_CACHE
    runner = _WORKER_RUNNER

    def _get_worker_model(species, idx, for_opponent=False):
        cache = _WORKER_OPPONENT_MODEL_CACHE[species] if for_opponent else _WORKER_MODEL_CACHE[species]
        model = cache.get(idx)
        if model is None:
            source_state = _WORKER_OPPONENT_STATE if for_opponent else _WORKER_POPULATION_STATE
            model = Model(chromosome=source_state[species][idx])
            cache[idx] = model
        return model

    override_policy = str(task.get("override_policy", "")).strip().lower()
    if override_policy == "random":
        self_model = RandomBaselinePolicy(seed=int(task.get("override_policy_seed", task["python_random_seed"])))
    else:
        self_model = _get_worker_model(task["species"], task["agent_index"])
    eval_species = {task["species"]: self_model}
    for sp, idx in task["other_indices"].items():
        eval_species[sp] = _get_worker_model(sp, idx, for_opponent=True)

    map_seed = task["map_seed"]
    python_random_seed = task["python_random_seed"]
    start_time = time.time()
    fitness, episode_length, end_reason = runner.run(
        eval_species,
        task["species"],
        map_seed,
        is_evaluation=False,
        callback=noop,
        collect_plot_data=False,
        python_random_seed=python_random_seed,
        max_steps_override=task.get("max_steps_override"),
        initial_energy_scale=task.get("initial_energy_scale", 1.0),
        energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
    )
    while episode_length == 0:
        if runner.settings.enable_logging:
            print("something went wrong update this seed")
        map_seed = random.randrange(100000)
        python_random_seed = random.randrange(2**32)
        fitness, episode_length, end_reason = runner.run(
            eval_species,
            task["species"],
            map_seed,
            is_evaluation=False,
            callback=noop,
            collect_plot_data=False,
            python_random_seed=python_random_seed,
            max_steps_override=task.get("max_steps_override"),
            initial_energy_scale=task.get("initial_energy_scale", 1.0),
            energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
        )

    end_time = time.time()
    return {
        "agent_index": task["agent_index"],
        "species": task["species"],
        "eval_index": task["eval_index"],
        "fitness": fitness,
        "episode_length": episode_length,
        "end_reason": end_reason,
        "steps_per_sec": episode_length / max(end_time - start_time, 1e-9),
        "diagnostics": dict(getattr(runner, "_last_run_diagnostics", {})),
    }
