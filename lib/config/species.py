# lib/config/species.py
import numpy as np
from dataclasses import dataclass
from typing import Dict
from .settings import Settings
import lib.config.const as const
import lib.model as model

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
    base_species: str
    age_index: int
    age_steps: int
    is_mature: bool
    next_species: str | None
    offspring_species: str | None

    @property
    def color_ones(self) -> tuple[float, float, float]:
        return (self.color[0] / 255.0, self.color[1] / 255.0, self.color[2] / 255.0)

SpeciesMap = Dict[str, SpeciesParams]

FEEDING_SPECIES_ORDER = ["plankton", "herring", "sprat", "cod"]

def _normalize_age_distribution(age_groups: int, distribution: tuple[float, ...]) -> list[float]:
    if age_groups <= 0:
        return []
    if len(distribution) == 0:
        return [1.0 / age_groups for _ in range(age_groups)]
    if len(distribution) != age_groups:
        print(
            f"[species] age_init_distribution length {len(distribution)} "
            f"does not match age_groups {age_groups}; using even split."
        )
        return [1.0 / age_groups for _ in range(age_groups)]
    total = sum(distribution)
    if total <= 0:
        return [1.0 / age_groups for _ in range(age_groups)]
    return [val / total for val in distribution]

def _base_index(base_species: str) -> int:
    return FEEDING_SPECIES_ORDER.index(base_species)

def build_species_map(settings: Settings) -> SpeciesMap:
    spd = settings.steps_per_day
    print("settings.steps_per_day", settings.steps_per_day)
    wsq = settings.world_size * settings.world_size

    age_groups = max(1, int(settings.age_groups))
    age_distribution = _normalize_age_distribution(age_groups, settings.age_init_distribution)

    # Configure global species lists and model offsets.
    const.configure_age_groups(age_groups)
    model.configure_species(const.SPECIES)

    cod_start = 44897
    her_start = 737356
    spr_start = 1359874
    pl_start = spr_start * 2

    base_energy_cost = 0.5
    fish_base_cost = 0.35   # herring, sprat
    cod_base_cost = 0.60    # cod

    base_params = {
        "plankton": dict(
            starting_biomass=pl_start,
            original_starting_biomass=pl_start,
            min_biomass_in_cell=0.0,
            max_biomass_in_cell=(pl_start / wsq) * 2,
            activity_metabolic_rate=0.002 * spd,
            standard_metabolic_rate=0.0004 * spd,
            natural_mortality_rate=0.1 * spd,
            fishing_mortality_rate=0.0,
            energy_cost=base_energy_cost * spd,
            individual_weight=0.001,
            growth_rate=1.4,
            carrying_capacity=(pl_start / wsq) * 2,
            reproduction_freq=1,
            baseline_mortality=0.01,
            mortality_logistic_k=0.1,
            mortality_energy_midpoint=25.0,
            low_energy_death_rate=0.5,
            hardcoded_logic=True,
            hardcoded_rules={
                "growth_rate_constant": 250,
                "respawn_delay": 10,
            },
            color=(0, 255, 0),
            noise_threshold=0.5,
            noise_scaling=6 * 5,
        ),
        "herring": dict(
            starting_biomass=her_start,
            original_starting_biomass=her_start,
            min_biomass_in_cell=0.0,
            max_biomass_in_cell=(her_start / wsq) * 2,
            activity_metabolic_rate=0.007317884210714286 * spd,
            standard_metabolic_rate=0.0014635768428571428 * spd,
            natural_mortality_rate=0.01 * spd,
            fishing_mortality_rate=settings.base_fishing_value_herring * settings.scale_fishing * spd,
            energy_cost=fish_base_cost * settings.speed_multiplier,
            individual_weight=0.1,
            growth_rate=1.03,
            carrying_capacity=np.inf,
            reproduction_freq=25,
            baseline_mortality=0.0005,
            mortality_logistic_k=0.25,
            mortality_energy_midpoint=5.0,
            low_energy_death_rate=0.05,
            hardcoded_logic=False,
            hardcoded_rules={},
            color=(0, 0, 255),
            noise_threshold=0.6,
            noise_scaling=6 * 5,
        ),
        "sprat": dict(
            starting_biomass=spr_start,
            original_starting_biomass=spr_start,
            min_biomass_in_cell=0.0,
            max_biomass_in_cell=(spr_start / wsq) * 2,
            activity_metabolic_rate=0.007317884210714286 * spd,
            standard_metabolic_rate=0.0014635768428571428 * spd,
            natural_mortality_rate=0.01 * spd,
            fishing_mortality_rate=settings.base_fishing_value_sprat * settings.scale_fishing * spd,
            energy_cost=fish_base_cost * settings.speed_multiplier,
            individual_weight=0.05,
            growth_rate=1.03,
            carrying_capacity=np.inf,
            reproduction_freq=25,
            baseline_mortality=0.0005,
            mortality_logistic_k=0.25,
            mortality_energy_midpoint=5.0,
            low_energy_death_rate=0.05,
            hardcoded_logic=False,
            hardcoded_rules={},
            color=(255, 165, 0),
            noise_threshold=0.6,
            noise_scaling=6 * 5,
        ),
        "cod": dict(
            starting_biomass=cod_start,
            original_starting_biomass=cod_start,
            min_biomass_in_cell=0.0,
            max_biomass_in_cell=(cod_start / wsq) * 2,
            activity_metabolic_rate=0.014535768421428572 * spd,
            standard_metabolic_rate=0.0029071536857142857 * spd,
            natural_mortality_rate=0.003 * spd,
            fishing_mortality_rate=settings.base_fishing_value_cod * settings.scale_fishing * spd,
            energy_cost=cod_base_cost * settings.speed_multiplier,
            individual_weight=1.0,
            growth_rate=1.02,
            carrying_capacity=np.inf,
            reproduction_freq=30,
            baseline_mortality=0.0003,
            mortality_logistic_k=0.25,
            mortality_energy_midpoint=6.0,
            low_energy_death_rate=0.04,
            hardcoded_logic=False,
            hardcoded_rules={},
            color=(40, 40, 40),
            noise_threshold=0.7,
            noise_scaling=6 * 5,
        ),
    }

    species_map: SpeciesMap = {}

    def _add_species(
        name: str,
        base_name: str,
        age_index: int,
        is_mature: bool,
        next_species: str | None,
        offspring_species: str | None,
        starting_biomass: float,
        original_starting_biomass: float,
        prey: list[str],
    ) -> None:
        params = base_params[base_name]
        species_map[name] = SpeciesParams(
            index=_base_index(base_name),
            starting_biomass=starting_biomass,
            original_starting_biomass=original_starting_biomass,
            min_biomass_in_cell=params["min_biomass_in_cell"],
            max_biomass_in_cell=params["max_biomass_in_cell"],
            activity_metabolic_rate=params["activity_metabolic_rate"],
            standard_metabolic_rate=params["standard_metabolic_rate"],
            natural_mortality_rate=params["natural_mortality_rate"],
            fishing_mortality_rate=params["fishing_mortality_rate"],
            energy_cost=params["energy_cost"],
            individual_weight=params["individual_weight"],
            growth_rate=params["growth_rate"],
            carrying_capacity=params["carrying_capacity"],
            reproduction_freq=params["reproduction_freq"],
            baseline_mortality=params["baseline_mortality"],
            mortality_logistic_k=params["mortality_logistic_k"],
            mortality_energy_midpoint=params["mortality_energy_midpoint"],
            low_energy_death_rate=params["low_energy_death_rate"],
            hardcoded_logic=params["hardcoded_logic"],
            color=params["color"],
            noise_threshold=params["noise_threshold"],
            noise_scaling=params["noise_scaling"],
            hardcoded_rules=params["hardcoded_rules"],
            prey=prey,
            base_species=base_name,
            age_index=age_index,
            age_steps=settings.age_step_interval,
            is_mature=is_mature,
            next_species=next_species,
            offspring_species=offspring_species,
        )

    juvenile_cod_names: list[str] = []
    herring_group_names: list[str] = ["herring"]
    sprat_group_names: list[str] = ["sprat"]
    if age_groups > 1:
        juvenile_cod_names = [
            const.make_age_group_name("cod", idx) for idx in range(age_groups - 1)
        ]
        herring_group_names = const.list_age_groups("herring", age_groups)
        sprat_group_names = const.list_age_groups("sprat", age_groups)

    for base_name in const.BASE_SPECIES:
        if base_name == "plankton" or age_groups <= 1:
            prey = []
            if base_name == "herring":
                prey = ["plankton"]
            elif base_name == "sprat":
                prey = ["plankton"]
            elif base_name == "cod":
                prey = ["herring", "sprat"]
            offspring = None if base_name == "plankton" else base_name
            _add_species(
                name=base_name,
                base_name=base_name,
                age_index=0,
                is_mature=True,
                next_species=None,
                offspring_species=offspring,
                starting_biomass=base_params[base_name]["starting_biomass"],
                original_starting_biomass=base_params[base_name]["original_starting_biomass"],
                prey=prey,
            )
            continue

        for age_index in range(age_groups):
            is_mature = age_index == age_groups - 1
            name = const.make_age_group_name(base_name, age_index)
            next_name = None if is_mature else const.make_age_group_name(base_name, age_index + 1)
            offspring_name = const.make_age_group_name(base_name, 0) if is_mature else None

            prey: list[str] = []
            if base_name in ("herring", "sprat"):
                prey = ["plankton"] + juvenile_cod_names
            elif base_name == "cod":
                if is_mature:
                    prey = herring_group_names + sprat_group_names
                else:
                    prey = ["plankton"]

            share = age_distribution[age_index]
            _add_species(
                name=name,
                base_name=base_name,
                age_index=age_index,
                is_mature=is_mature,
                next_species=next_name,
                offspring_species=offspring_name,
                starting_biomass=base_params[base_name]["starting_biomass"] * share,
                original_starting_biomass=base_params[base_name]["original_starting_biomass"] * share,
                prey=prey,
            )

    return species_map


# Feeding energy reward matrix based on bioMARL defaults
# Row = predator index, Column = prey index
# Species order: [plankton(0), herring(1), sprat(2), cod(3)]
# Values represent energy gained when predator eats prey
# From bioMARL: [[0, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0], [1, 10, 5, 0]]
FEEDING_ENERGY_REWARD = [
    [0, 0, 0, 0],      # plankton eats nothing
    [4, 0, 0, 0],      # herring eats plankton (reward=4)
    [4, 0, 0, 0],      # sprat eats plankton (increased to match herring)
    [1, 15, 10, 0],     # cod eats plankton(1), herring(15), sprat(10)
]

def get_feeding_energy_reward(predator_species: str, prey_species: str, species_map: SpeciesMap) -> float:
    """Get the energy reward for predator eating prey based on the feeding matrix."""
    predator_idx = species_map[predator_species].index
    prey_idx = species_map[prey_species].index
    return FEEDING_ENERGY_REWARD[predator_idx][prey_idx]
