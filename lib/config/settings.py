# lib/config/settings.py
from dataclasses import dataclass
from typing import Final
import os
from dataclasses import fields
from typing import get_origin, get_args
# from .settings import Settings

MAP_METER_SIZE = 300 * 1000
FISH_SWIM_SPEED = 0.025
SECONDS_IN_DAY = 86400

@dataclass(frozen=True)
class Settings:
    folder: str = "results/default"

    world_size: int = 12
    speed_multiplier: float = 1.0
    multiply_death_rate: float = 1.0
    growth_multiplier: float = 1
    min_percent_alive: float = 0.1
    # Permit growth but cap runaway biomass (relative to initial total biomass)
    max_percent_alive: float = 10.0
    max_years: int = 5
    scale_fishing: float = 0.0
    base_fishing_value_cod: float = 0.002651024
    base_fishing_value_herring: float = 0.002651024
    base_fishing_value_sprat: float = 0.002651024

    num_agents: int = 16
    agent_evaluations: int = 4
    elitism_selection: int = 8
    tournament_selection: int = 4
    generations_per_run: int = 200
    num_workers: int = 1
    # Use the same opponents/seeds for all candidates in a generation/eval slot.
    paired_opponent_evaluation: bool = True
    # Keep opponent populations frozen for N generations to reduce moving-target noise.
    # 1 = refresh every generation (legacy behavior).
    opponent_snapshot_every: int = 5
    # Evaluate each age-group candidate together with same-index teammates from the
    # same base species (improves credit assignment for base-species objectives).
    lock_same_base_teammates: bool = True
    # Set to a non-negative value for deterministic runs across processes.
    seed: int = -1
    # Performance toggles.
    enable_logging: bool = True
    enable_plotting: bool = True
    mutation_rate: float = 0.12
    mutation_rate_decay: float = 0.996
    mutation_rate_min: float = 0.02
    # Optional random immigrant injection for diversity.
    early_random_injection_count: int = 0
    early_random_injection_generations: int = 0

    sbx_eta: float = 5.0
    sbx_eta_decay: float = 1.025
    # max_steps: int = 365 * 3 * 3
    max_steps: int = 5000
    # Fitness calculation mode for evolutionary evaluation:
    #   simple      -> reward by episode length (longest survives)
    #   biomass_pct -> % biomass change after fitness_eval_steps cycles
    #   trajectory_shaped -> dense per-cycle deltas + terminal biomass term
    #
    # Default to the simplest survival-based setup so a new user can train
    # without needing to understand the short-horizon curriculum first.
    fitness_method: str = "simple"
    # For biomass_pct fitness:
    #   agent        -> evaluate only the current acting species channel
    #   base_species -> aggregate all age groups of the same base species
    biomass_fitness_scope: str = "agent"
    # Training-only: force per-agent (per-age-group) fitness scope.
    training_force_agent_fitness_scope: bool = True
    fitness_eval_steps: int = 40
    # Dense short-horizon fitness shaping (used when fitness_method=trajectory_shaped).
    trajectory_gamma: float = 0.98
    trajectory_weight_biomass_delta: float = 0.5
    trajectory_weight_energy_delta: float = 2.0
    trajectory_weight_terminal_biomass: float = 0.5
    # Penalize sharp biomass crashes between consecutive cycles.
    trajectory_crash_drop_pct: float = 8.0
    trajectory_crash_penalty: float = 2.0
    # Optional extra penalty when biomass falls below a floor (% of initial biomass).
    trajectory_low_biomass_floor_pct: float = 30.0
    trajectory_low_biomass_penalty: float = 0.0
    # Print mean trajectory fitness components each generation while training.
    trajectory_diagnostics_enabled: bool = True
    # Optional speed-up: end evaluation once only one acting base species remains.
    # Disabled by default; candidate-level short-circuiting is usually safer.
    stop_on_single_species_left: bool = False
    alive_species_biomass_threshold: float = 1e-6
    # Optional speed-up (single-worker mode): if the last candidate is already guaranteed
    # to beat the current best under simple fitness, stop its remaining rollouts early.
    stop_last_candidate_when_winner: bool = True
    # Two-stage evaluation: quick screening then deeper re-eval of top candidates.
    two_stage_eval_enabled: bool = False
    short_eval_steps: int = 30
    long_eval_steps: int = 200
    long_eval_top_fraction: float = 0.2
    long_eval_weight: float = 0.6
    # Optional score normalization for short-horizon biomass objectives:
    # use (candidate_fitness - baseline_random_fitness) per matched eval task.
    relative_baseline_enabled: bool = False
    relative_baseline_policy: str = "random"
    # If baseline-adjusted scores collapse to near-constant, fall back to raw biomass
    # fitness for that generation to preserve selection signal.
    relative_baseline_fallback_to_raw_when_flat: bool = True
    relative_baseline_flat_std_threshold: float = 0.25
    # Optional harsher training-eval world (encourages eat-or-die behavior in short windows).
    training_initial_energy_scale: float = 1.0
    training_energy_decay_per_cycle: float = 0.0
    # Training-only food sparsity controls (plankton seeding).
    # 1.0 keeps all initial plankton cells, 0.1 keeps 10%.
    training_plankton_cell_fraction: float = 1.0
    # Scales initial biomass on kept plankton cells.
    training_plankton_biomass_scale: float = 1.0
    # Training-only sparsity for non-plankton (acting) base species.
    # Applied after map reset and before rollouts.
    training_non_plankton_cell_fraction: float = 1.0
    training_non_plankton_min_cells: int = 1
    # Force cod to spawn in very few random cells (1 = single-cell cod start).
    # Set to 0 to disable this override.
    training_cod_spawn_cells: int = 0
    # Keep prey away from initial cod cells so movement toward food is required.
    # Uses Chebyshev distance in grid cells (0 disables clearing).
    training_cod_prey_clear_radius: int = 0
    # If False, keep plankton near cod even when clearing cod prey radius.
    # This avoids starving juvenile cod age groups that only eat plankton.
    training_cod_prey_clear_include_plankton: bool = False
    # Training-only steep penalty when weighted mean energy falls below floor.
    # Floor is expressed as fraction of max_energy (e.g. 0.4 = 40%).
    training_low_energy_floor_pct: float = 0.0
    training_low_energy_penalty: float = 0.0
    # Training-only dynamics: extra biomass loss each cycle when energy is low.
    # Loss fraction = training_low_energy_biomass_max_loss_per_cycle * deficit,
    # where deficit = max(0, (floor_pct - energy_pct) / floor_pct).
    training_low_energy_biomass_floor_pct: float = 0.0
    training_low_energy_biomass_max_loss_per_cycle: float = 0.0
    training_low_energy_biomass_target_only: bool = True
    # Training-only multiplier on energy gained from eating.
    # >1 encourages policies to seek and eat food sooner.
    training_eating_reward_multiplier: float = 1.0
    # Training-only: set a very high reproduction frequency for non-plankton to
    # effectively disable offspring spawning during short-horizon training.
    # Set <=0 to disable this override.
    training_reproduction_freq_override: int = 0
    # Progress tracking: evaluate per-generation champions on a longer fixed horizon.
    champion_progress_enabled: bool = False
    champion_progress_every: int = 1
    champion_progress_steps: int = 1000
    champion_progress_episodes: int = 1
    # Fixed validation benchmark (deterministic seeds and horizon per generation).
    fixed_validation_enabled: bool = False
    fixed_validation_every: int = 1
    fixed_validation_steps: int = 200
    fixed_validation_episodes: int = 3
    fixed_validation_seed: int = 4242
    # fixed_validation_metric: "fitness" uses the active fitness method score,
    # "survival" uses episode length in cycles.
    fixed_validation_metric: str = "fitness"
    # Optional detail lines on plot for fixed validation by base species.
    fixed_validation_show_species: bool = False
    # Refresh biomass line chart every N simulation cycles when rendering.
    biomass_plot_update_interval_steps: int = 1

    smell_decay: float = 0.9
    smell_emission_rate: float = 0.1
    # Update smell every N environment actions (1 = every action).
    smell_update_interval: int = 1

    max_energy: float = 100.0

    # Age-group settings (applies to all non-plankton species)
    age_groups: int = 1
    # Global interval (in environment cycles) to advance age groups
    age_step_interval: int = 50
    # Optional per-age distribution weights (comma-separated in settings file).
    # Empty means "split evenly across however many age groups are configured".
    age_init_distribution = ()

    @property
    def steps_per_day(self) -> int:
        DAYS_TO_CROSS_MAP = MAP_METER_SIZE / (FISH_SWIM_SPEED * SECONDS_IN_DAY)
        return (DAYS_TO_CROSS_MAP / 50) * self.speed_multiplier

def _coerce(value: str, target_type):
    """Coerce string env values to the dataclass field type."""
    origin = get_origin(target_type)
    if origin in (list, tuple):
        target_type = origin
    elif origin is not None:  # Handles typing constructs if you add them later
        target_type = get_args(target_type)[0]

    if target_type in (list, tuple):
        raw = value.strip()
        if raw == "":
            return [] if target_type is list else ()
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        values = [float(p) for p in parts]
        return values if target_type is list else tuple(values)

    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    # default: string (paths are fine as str)
    return value

def load_settings(file_path: str = None) -> Settings:
    file_name = os.path.basename(file_path).split(".")[0]

    folder = f"results/{file_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
        if not os.path.exists(f"{folder}/agents"):
            os.makedirs(f"{folder}/agents")

    overrides = {}
    overrides["folder"] = folder
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            overrides[key] = _coerce(value, type(getattr(Settings, key)))

    # Construct a frozen Settings with env overrides applied
    return Settings(**overrides)
