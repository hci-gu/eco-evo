# lib/config/settings.py
from dataclasses import dataclass
from typing import Final
import os
from dataclasses import fields
from typing import get_origin, get_args
# from .settings import Settings

@dataclass(frozen=True)
class Settings:
    folder: str = "results/default"

    world_size: int = 18
    steps_per_day: int = 3
    speed_multiplier: float = 1.0
    multiply_death_rate: float = 1.0
    growth_multiplier: float = 1
    min_percent_alive: float = 0.2
    max_percent_alive: float = 4.0
    max_years: int = 5
    scale_fishing: float = 0.0
    base_fishing_value_cod: float = 0.002651024
    base_fishing_value_herring: float = 0.002651024
    base_fishing_value_sprat: float = 0.002651024

    num_agents: int = 16
    agent_evaluations: int = 3
    elitism_selection: int = 8
    tournament_selectin: int = 4
    generations_per_run: int = 200
    mutation_rate: float = 0.1
    mutation_rate_decay: float = 0.995
    mutation_rate_min: float = 0.01
    max_steps: int = 365 * 5 * 3

    smell_decay: float = 0.9
    smell_emission_rate: float = 0.1

    max_energy: float = 100.0

def _coerce(value: str, target_type):
    """Coerce string env values to the dataclass field type."""
    origin = get_origin(target_type)
    if origin is not None:  # Handles typing constructs if you add them later
        target_type = get_args(target_type)[0]

    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    # default: string (paths are fine as str)
    return value

def load_settings(file_path: str | None = None) -> Settings:
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
