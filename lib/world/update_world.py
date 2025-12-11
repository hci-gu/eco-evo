import numpy as np
import warnings
from enum import Enum
from lib.model import MODEL_OFFSETS
from lib.config.const import ACTING_SPECIES
from lib.config.settings import Settings
from lib.config.species import SpeciesMap, get_feeding_energy_reward
from numba import njit

# Suppress underflow warnings globally - underflow to zero is expected for dying populations
np.seterr(under='ignore', divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

def all_movement_delta(species_map: SpeciesMap, world, world_data, species_key, actions, step_counter: int = 0):
    pad = 1
    world_data[pad:-pad, pad:-pad, 4] += 1

    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]
    activity_metabolic_rate = props.activity_metabolic_rate
    standard_metabolic_rate = props.standard_metabolic_rate
    fishing_mortality_loss = props.fishing_mortality_rate
    growth_rate = props.growth_rate
    carrying_capacity = props.carrying_capacity
    reproduction_freq = props.reproduction_freq
    
    # bioMARL mortality parameters
    baseline_mortality = props.baseline_mortality
    mortality_k = props.mortality_logistic_k
    mortality_midpoint = props.mortality_energy_midpoint
    low_energy_death_rate = props.low_energy_death_rate

    move_up = actions[:, :, Action.UP.value]
    move_down = actions[:, :, Action.DOWN.value]
    move_left = actions[:, :, Action.LEFT.value]
    move_right = actions[:, :, Action.RIGHT.value]

    total_activity = move_up + move_down + move_left + move_right

    # --- Energy updates (bioMARL style: metabolic costs reduce energy, not biomass) ---
    initial_energy = world[pad:-pad, pad:-pad, energy_offset].copy()
    
    # Activity metabolic cost reduces energy
    activity_energy_cost = total_activity * activity_metabolic_rate
    # Standard metabolic cost reduces energy every step
    total_energy_cost = activity_energy_cost + standard_metabolic_rate
    
    # Apply energy costs
    world[pad:-pad, pad:-pad, energy_offset] -= total_energy_cost
    
    # Get updated energy for mortality calculation
    current_energy = world[pad:-pad, pad:-pad, energy_offset]

    # --- Biomass updates ---
    initial_biomass = world[pad:-pad, pad:-pad, biomass_offset].copy()

    # Apply fishing mortality separately (multiplicative)
    world[:, :, biomass_offset] *= 1 - fishing_mortality_loss

    # --- bioMARL-style energy-dependent mortality ---
    # Mortality rate increases as energy decreases using logistic function:
    # mortality = baseline + (1 - baseline) / (1 + exp(-k * (midpoint - energy)))
    # When energy is high: mortality ≈ baseline
    # When energy is low: mortality ≈ 1 (high death rate)
    mortality_rate = baseline_mortality + (1.0 - baseline_mortality) / (
        1.0 + np.exp(-mortality_k * (mortality_midpoint - current_energy))
    )
    
    # Additional death rate when energy < 1 (starvation)
    # death_prob = 1 - exp(-death_rate) when energy < 1
    # Clip energy deficit to avoid overflow in exp (max deficit of 100)
    starvation_mask = current_energy < 1.0
    energy_deficit = np.clip(1.0 - current_energy, 0.0, 100.0)
    starvation_death_prob = np.where(
        starvation_mask,
        1.0 - np.exp(-low_energy_death_rate * energy_deficit),
        0.0
    )
    
    # Combine mortality: use max of energy-dependent mortality and starvation death
    total_mortality = np.maximum(mortality_rate, starvation_death_prob)
    
    # Apply mortality to biomass
    world[pad:-pad, pad:-pad, biomass_offset] *= (1.0 - total_mortality)
    
    # Zero energy = death (bioMARL: h[energy <= 0] = 0)
    zero_energy_mask = current_energy <= 0
    world[pad:-pad, pad:-pad, biomass_offset] = np.where(
        zero_energy_mask, 0.0, world[pad:-pad, pad:-pad, biomass_offset]
    )
    # Also reset energy to 0 for dead cells
    world[pad:-pad, pad:-pad, energy_offset] = np.maximum(current_energy, 0.0)

    # --- bioMARL-style multiplicative growth ---
    # Growth only occurs at reproduction_freq intervals
    does_reproduce = (step_counter % reproduction_freq == 0)
    
    if does_reproduce:
        # Multiplicative growth: B_new = B * growth_rate
        current_biomass = world[pad:-pad, pad:-pad, biomass_offset]
        growth_delta = current_biomass * (growth_rate - 1.0)
        
        # Apply carrying capacity cap only for species with finite capacity (plankton)
        if carrying_capacity < np.inf:
            growth_delta = np.clip(carrying_capacity - current_biomass, 0, growth_delta)
        
        world[pad:-pad, pad:-pad, biomass_offset] += growth_delta
    # -----------------------------------------

    biomass_after_loss = world[pad:-pad, pad:-pad, biomass_offset]
    current_energy_for_movement = world[pad:-pad, pad:-pad, energy_offset]

    # --- Determine valid moves (based on terrain) ---
    water_offset = MODEL_OFFSETS["terrain"]["water"]
    valid_left_mask = world[: -2 * pad, pad:-pad, water_offset] == 1
    valid_right_mask = world[2 * pad :, pad:-pad, water_offset] == 1
    valid_up_mask = world[pad:-pad, : -2 * pad, water_offset] == 1
    valid_down_mask = world[pad:-pad, 2 * pad :, water_offset] == 1

    # Biomass to move in each direction
    biomass_up = biomass_after_loss * move_up * valid_up_mask
    biomass_down = biomass_after_loss * move_down * valid_down_mask
    biomass_left = biomass_after_loss * move_left * valid_left_mask
    biomass_right = biomass_after_loss * move_right * valid_right_mask

    total_biomass_moved = biomass_up + biomass_down + biomass_left + biomass_right

    # Movements.
    movement_deltas = np.zeros_like(world)
    movement_deltas[pad:-pad, : -2 * pad, biomass_offset] += biomass_up
    movement_deltas[pad:-pad, : -2 * pad, energy_offset] += current_energy_for_movement * biomass_up
    movement_deltas[pad:-pad, 2 * pad :, biomass_offset] += biomass_down
    movement_deltas[pad:-pad, 2 * pad :, energy_offset] += current_energy_for_movement * biomass_down
    movement_deltas[: -2 * pad, pad:-pad, biomass_offset] += biomass_left
    movement_deltas[: -2 * pad, pad:-pad, energy_offset] += (
        current_energy_for_movement * biomass_left
    )
    movement_deltas[2 * pad :, pad:-pad, biomass_offset] += biomass_right
    movement_deltas[2 * pad :, pad:-pad, energy_offset] += (
        current_energy_for_movement * biomass_right
    )

    # The stationary.
    movement_deltas[pad:-pad, pad:-pad, energy_offset] += current_energy_for_movement * (
        biomass_after_loss - total_biomass_moved
    )
    movement_deltas[pad:-pad, pad:-pad, biomass_offset] -= total_biomass_moved

    return movement_deltas

def apply_movement_delta(species_map: SpeciesMap, world, species_key, movement_deltas):
    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]

    # Apply the movement deltas to biomass and energy.
    world[:, :, biomass_offset] += movement_deltas[:, :, biomass_offset]
    denom = np.where(np.abs(world[:, :, biomass_offset]) == 0, 1, np.abs(world[:, :, biomass_offset]))
    world[:, :, energy_offset] = movement_deltas[:, :, energy_offset] / denom
    # reduce energy by base amount
    world[:, :, energy_offset] -= props.energy_cost

def matrix_perform_eating(settings: Settings, species_map: SpeciesMap, world, species_key, actions):
    pad = 1
    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]

    # Get eating actions and metabolic cost.
    eat = actions[:, :, Action.EAT.value]
    eating_energy_cost = eat * props.activity_metabolic_rate

    # Cache biomass and energy at batch positions.
    init_biomass = world[pad:-pad, pad:-pad, biomass_offset].copy()
    init_energy = world[pad:-pad, pad:-pad, energy_offset].copy()

    # --- bioMARL style: eating costs energy, not biomass ---
    # Apply eating metabolic cost to energy
    world[pad:-pad, pad:-pad, energy_offset] -= eating_energy_cost

    # Compute total eat potential.
    initial_total_eat = init_biomass * eat
    still_left_to_eat = initial_total_eat.copy()

    # Randomize prey order.
    prey_list = props.prey.copy()
    np.random.shuffle(prey_list)

    # Track total energy reward from eating (individual-based)
    total_energy_reward = np.zeros_like(init_biomass)

    # Process each prey species.
    for prey_species in prey_list:
        prey_props = species_map[prey_species]
        prey_biomass_offset = MODEL_OFFSETS[prey_species]["biomass"]
        prey_biomass = world[pad:-pad, pad:-pad, prey_biomass_offset]
        # Consume as much as possible, up to total_eat_amount.
        eat_amount = np.minimum(prey_biomass, still_left_to_eat)
        # Reduce prey biomass.
        world[pad:-pad, pad:-pad, prey_biomass_offset] -= eat_amount
        still_left_to_eat -= eat_amount

        # Calculate energy reward based on individuals eaten
        # Number of prey individuals eaten = biomass eaten / prey individual weight
        prey_individuals_eaten = eat_amount / prey_props.individual_weight
        # Energy reward per prey individual from the feeding matrix
        energy_per_prey = get_feeding_energy_reward(species_key, prey_species, species_map)
        # Total energy from eating this prey species
        total_energy_reward += prey_individuals_eaten * energy_per_prey

    # Calculate number of predator individuals to distribute energy across
    # Using the eating biomass (individuals that attempted to eat)
    eating_biomass = initial_total_eat
    predator_individuals = np.divide(
        eating_biomass,
        props.individual_weight,
        out=np.zeros_like(eating_biomass),
        where=eating_biomass > 0,
    )

    # Energy per predator individual, then scale by what fraction of biomass was eating
    # Average energy reward per individual that was eating
    energy_per_predator = np.divide(
        total_energy_reward,
        predator_individuals,
        out=np.zeros_like(total_energy_reward),
        where=predator_individuals > 0,
    )

    # Scale the energy reward by the eating percentage (how much of total biomass was eating)
    eating_percentage = np.divide(
        eating_biomass,
        init_biomass,
        out=np.zeros_like(eating_biomass),
        where=init_biomass > 0,
    )

    world[pad:-pad, pad:-pad, energy_offset] += energy_per_predator * eating_percentage

    # --- Handle low biomass: if biomass falls below minimum, set it (and energy) to zero ---
    low_biomass = world[..., biomass_offset] < props.min_biomass_in_cell
    world[low_biomass, biomass_offset] = 0
    world[low_biomass, energy_offset] = 0

    np.clip(
        world[pad:-pad, pad:-pad, energy_offset],
        0,
        settings.max_energy,
        out=world[pad:-pad, pad:-pad, energy_offset],
    )

    return world

def total_biomass(world):
    total = 0
    for species in ACTING_SPECIES:
        biomass_offset = MODEL_OFFSETS[species]["biomass"]
        total += world[:, :, biomass_offset].sum()
    return np.log(total + 1e-8)

def world_is_alive(settings: Settings, species_map: SpeciesMap, world):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, TOTAL_CHANNELS) where:
      - The first 3 channels represent the terrain (one-hot encoded).
      - The subsequent channels represent the biomass for each species.
    """
    reason = ""

    for species in ACTING_SPECIES:
        props = species_map[species]
        biomass_offset = MODEL_OFFSETS[species]["biomass"]
        biomass = world[:, :, biomass_offset].sum()
        biomass_too_low = biomass < (props.starting_biomass * settings.min_percent_alive)
        biomass_too_high = biomass > (props.starting_biomass * settings.max_percent_alive)
        if biomass_too_low or biomass_too_high:
            reason = f"{species} "
            if biomass_too_low:
                reason += f"low"
            if biomass_too_high:
                reason += f"high"
            return False, reason
    return True, ""
