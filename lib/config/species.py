# lib/config/species.py
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
    activity_metabolic_rate: float
    standard_metabolic_rate: float
    natural_mortality_rate: float
    fishing_mortality_rate: float
    energy_cost: float
    energy_reward: float
    growth_rate: float
    hardcoded_logic: bool
    color: tuple[int, int, int]
    noise_threshold: float
    noise_scaling: float

def build_species_params(settings: Settings) -> Dict[str, SpeciesParams]:
    spd = settings.steps_per_day
    wsq = settings.world_size * settings.world_size
    # Use per-step scaling as per your intended semantics:
    # per_step = per_day / steps_per_day
    # or if empirically tuned, keep current behavior but document it.
    cod_start = 44897
    her_start = 737356
    spr_start = 1359874
    pl_start = spr_start * 2

    plankton = SpeciesParams(
        index=0,
        starting_biomass=pl_start,
        original_starting_biomass=pl_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(pl_start / wsq) * 150,
        activity_metabolic_rate=0.002 * spd,
        standard_metabolic_rate=0.0004 * spd,
        natural_mortality_rate=0.1 * spd,
        fishing_mortality_rate=0.0,
        energy_cost=0.1 * spd,
        energy_reward=100 * spd,
        growth_rate=0.2 * settings.growth_multiplier * spd,
        hardcoded_logic=True,
        color=(0, 255, 0),
        noise_threshold=0.5,
        noise_scaling=6 * 5,
    )
    herring = SpeciesParams(
        index=1,
        starting_biomass=her_start,
        original_starting_biomass=her_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(her_start / wsq) * 150,
        activity_metabolic_rate=0.007317884210714286 * spd,
        standard_metabolic_rate=0.0014635768428571428 * spd,
        natural_mortality_rate=0.01 * spd,
        fishing_mortality_rate=settings.base_fishing_value_herring * settings.scale_fishing * spd,
        energy_cost=0.2 * spd,
        energy_reward=500 * spd,
        growth_rate=0.1 * settings.growth_multiplier * spd,
        hardcoded_logic=False,
        color=(0, 0, 255),
        noise_threshold=0.6,
        noise_scaling=6 * 5,
    )
    sprat = SpeciesParams(
        index=2,
        starting_biomass=spr_start,
        original_starting_biomass=spr_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(spr_start / wsq) * 150,
        activity_metabolic_rate=0.007317884210714286 * spd,
        standard_metabolic_rate=0.0014635768428571428 * spd,
        natural_mortality_rate=0.01 * spd,
        fishing_mortality_rate=settings.base_fishing_value_sprat * settings.scale_fishing * spd,
        energy_cost=0.2 * spd,
        energy_reward=500 * spd,
        growth_rate=0.1 * settings.growth_multiplier * spd,
        hardcoded_logic=False,
        color=(255, 165, 0),
        noise_threshold=0.6,
        noise_scaling=6 * 5,
    )
    cod = SpeciesParams(
        index=3,
        starting_biomass=cod_start,
        original_starting_biomass=cod_start,
        min_biomass_in_cell=0.0,
        max_biomass_in_cell=(cod_start / wsq) * 150,
        activity_metabolic_rate=0.014535768421428572 * spd,
        standard_metabolic_rate=0.0029071536857142857 * spd,
        natural_mortality_rate=0.003 * spd,
        fishing_mortality_rate=settings.base_fishing_value_cod * settings.scale_fishing * spd,
        energy_cost=0.3 * spd,
        energy_reward=1000 * spd,
        growth_rate=0.05 * settings.growth_multiplier * spd,
        hardcoded_logic=False,
        color=(40, 40, 40),
        noise_threshold=0.7,
        noise_scaling=6 * 5,
    )

    return {
        "plankton": plankton,
        "herring": herring,
        "sprat": sprat,
        "cod": cod,
    }