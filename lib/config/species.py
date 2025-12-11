# lib/config/species.py
import numpy as np
from dataclasses import dataclass
from typing import Dict
from .settings import Settings

@dataclass(frozen=True)
class SpeciesParams:
    index: int
    starting_biomass: float
    original_starting_biomass: float
    min_biomass_in_cell: float
    max_biomass_in_cell: float
    activity_metabolic_rate: float  # Energy cost per unit activity (bioMARL: reduces energy, not biomass)
    standard_metabolic_rate: float  # Base energy cost per step (bioMARL: reduces energy, not biomass)
    natural_mortality_rate: float  # Base mortality rate (used in energy-dependent mortality)
    fishing_mortality_rate: float
    energy_cost: float
    individual_weight: float  # Weight per individual (used to calculate number of individuals from biomass)
    growth_rate: float  # Multiplicative growth factor (e.g., 1.2 means 20% growth per reproduction event)
    carrying_capacity: float  # Max population per cell (np.inf for uncapped, finite for plankton)
    reproduction_freq: int  # How often reproduction occurs (1 = every step, 5 = every 5 steps)
    # bioMARL-style energy-dependent mortality parameters
    baseline_mortality: float  # Minimum mortality rate when energy is high
    mortality_logistic_k: float  # Steepness of mortality curve (higher = sharper transition)
    mortality_energy_midpoint: float  # Energy level at which mortality is halfway between baseline and max
    low_energy_death_rate: float  # Death rate multiplier when energy < 1
    hardcoded_logic: bool
    color: tuple[int, int, int]
    noise_threshold: float
    noise_scaling: float
    hardcoded_rules: dict[str, float]
    prey: list[str]

    @property
    def color_ones(self) -> tuple[float, float, float]:
        return (self.color[0] / 255.0, self.color[1] / 255.0, self.color[2] / 255.0)

SpeciesMap = Dict[str, SpeciesParams]

def build_species_map(settings: Settings) -> SpeciesMap:
    spd = settings.steps_per_day
    print("settings.steps_per_day", settings.steps_per_day)
    wsq = settings.world_size * settings.world_size
    # Use per-step scaling as per your intended semantics:
    # per_step = per_day / steps_per_day
    # or if empirically tuned, keep current behavior but document it.
    cod_start = 44897
    her_start = 737356
    spr_start = 1359874
    pl_start = spr_start * 2

    base_energy_cost = 0.5
    # Target per-step base costs at speed_multiplier=1 (tuned for ~8–10% break-even):
    fish_base_cost = 0.50   # herring, sprat
    cod_base_cost = 0.80    # cod

    plankton = SpeciesParams(
        index=0,
        starting_biomass=pl_start,
        original_starting_biomass=pl_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(pl_start / wsq) * 2,
        activity_metabolic_rate=0.002 * spd,
        standard_metabolic_rate=0.0004 * spd,
        natural_mortality_rate=0.1 * spd,
        fishing_mortality_rate=0.0,
        energy_cost=base_energy_cost * spd,
        individual_weight=0.001,  # Plankton are very small (~1g per individual)
        growth_rate=1.4,  # bioMARL: 40% multiplicative growth per reproduction
        carrying_capacity=(pl_start / wsq) * 2,  # Capped at max_biomass_in_cell (like bioMARL fg_k=100)
        reproduction_freq=1,  # bioMARL: plankton reproduces every step
        baseline_mortality=0.01,  # Low baseline mortality for plankton
        mortality_logistic_k=0.1,  # Gentle mortality curve
        mortality_energy_midpoint=25.0,  # Energy midpoint for mortality curve
        low_energy_death_rate=0.5,  # Death rate when energy < 1
        hardcoded_logic=True,
        hardcoded_rules={
            "growth_rate_constant": 250,
            # "growth_rate_constant": 2.5,
            "respawn_delay": 10,
        },
        color=(0, 255, 0),
        noise_threshold=0.5,
        noise_scaling=6 * 5,
        prey=[],
    )
    herring = SpeciesParams(
        index=1,
        starting_biomass=her_start,
        original_starting_biomass=her_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(her_start / wsq) * 2,
        activity_metabolic_rate=0.007317884210714286 * spd,
        standard_metabolic_rate=0.0014635768428571428 * spd,
        natural_mortality_rate=0.01 * spd,
        fishing_mortality_rate=settings.base_fishing_value_herring * settings.scale_fishing * spd,
        # Lower base energy cost to roughly 0.7 energy/step at speed_multiplier=1
        energy_cost=fish_base_cost * settings.speed_multiplier,
        individual_weight=0.1,  # Herring ~100g per individual
        growth_rate=1.05,  # modest intrinsic growth
        carrying_capacity=np.inf,  # no cap for fish
        reproduction_freq=20,  # occasional reproduction
        baseline_mortality=0.01,  # light background mortality
        mortality_logistic_k=0.5,  # smoother rise as energy falls
        mortality_energy_midpoint=5.0,
        low_energy_death_rate=0.1,  # starvation ramp
        hardcoded_logic=False,
        hardcoded_rules={},
        color=(0, 0, 255),
        noise_threshold=0.6,
        noise_scaling=6 * 5,
        prey=["plankton"],
    )
    sprat = SpeciesParams(
        index=2,
        starting_biomass=spr_start,
        original_starting_biomass=spr_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(spr_start / wsq) * 2,
        activity_metabolic_rate=0.007317884210714286 * spd,
        standard_metabolic_rate=0.0014635768428571428 * spd,
        natural_mortality_rate=0.01 * spd,
        fishing_mortality_rate=settings.base_fishing_value_sprat * settings.scale_fishing * spd,
        # Lower base energy cost to roughly 0.7 energy/step at speed_multiplier=1
        energy_cost=fish_base_cost * settings.speed_multiplier,
        individual_weight=0.05,  # Sprat ~50g per individual (smaller than herring)
        growth_rate=1.05,
        carrying_capacity=np.inf,
        reproduction_freq=20,
        baseline_mortality=0.01,
        mortality_logistic_k=0.5,
        mortality_energy_midpoint=5.0,
        low_energy_death_rate=0.1,
        hardcoded_logic=False,
        hardcoded_rules={},
        color=(255, 165, 0),
        noise_threshold=0.6,
        noise_scaling=6 * 5,
        prey=["plankton"],
    )
    cod = SpeciesParams(
        index=3,
        starting_biomass=cod_start,
        original_starting_biomass=cod_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(cod_start / wsq) * 2,
        activity_metabolic_rate=0.014535768421428572 * spd,
        standard_metabolic_rate=0.0029071536857142857 * spd,
        natural_mortality_rate=0.003 * spd,
        fishing_mortality_rate=settings.base_fishing_value_cod * settings.scale_fishing * spd,
        # Slightly higher base energy cost for cod (~1.2 energy/step at speed_multiplier=1)
        energy_cost=cod_base_cost * settings.speed_multiplier,
        individual_weight=2.0,  # Cod ~2kg per individual (much larger than herring/sprat)
        growth_rate=1.02,  # slow apex growth
        carrying_capacity=np.inf,
        reproduction_freq=25,
        baseline_mortality=0.008,
        mortality_logistic_k=0.5,
        mortality_energy_midpoint=6.0,
        low_energy_death_rate=0.08,
        hardcoded_logic=False,
        hardcoded_rules={},
        color=(40, 40, 40),
        noise_threshold=0.7,
        noise_scaling=6 * 5,
        prey=["herring", "sprat"],
    )

    return {
        "plankton": plankton,
        "herring": herring,
        "sprat": sprat,
        "cod": cod,
    }


# Feeding energy reward matrix based on bioMARL defaults
# Row = predator index, Column = prey index
# Species order: [plankton(0), herring(1), sprat(2), cod(3)]
# Values represent energy gained when predator eats prey
# From bioMARL: [[0, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0], [1, 10, 5, 0]]
FEEDING_ENERGY_REWARD = [
    [0, 0, 0, 0],      # plankton eats nothing
    [4, 0, 0, 0],      # herring eats plankton (reward=4)
    [4, 0, 0, 0],      # sprat eats plankton (increased to match herring)
    [1, 10, 5, 0],     # cod eats plankton(1), herring(10), sprat(5)
]

def get_feeding_energy_reward(predator_species: str, prey_species: str, species_map: SpeciesMap) -> float:
    """Get the energy reward for predator eating prey based on the feeding matrix."""
    predator_idx = species_map[predator_species].index
    prey_idx = species_map[prey_species].index
    return FEEDING_ENERGY_REWARD[predator_idx][prey_idx]
