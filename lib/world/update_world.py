import numpy as np
from enum import Enum
from lib.model import MODEL_OFFSETS
from lib.config.const import ACTING_SPECIES
from lib.config.settings import Settings
from lib.config.species import SpeciesMap
from numba import njit

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

def all_movement_delta(species_map: SpeciesMap, world, world_data, species_key, actions):
    pad = 1
    world_data[pad:-pad, pad:-pad, 4] += 1

    props = species_map[species_key]
    biomass_offset = MODEL_OFFSETS[species_key]["biomass"]
    energy_offset = MODEL_OFFSETS[species_key]["energy"]
    activity_metabolic_rate = props.activity_metabolic_rate
    standard_metabolic_rate = props.standard_metabolic_rate
    natural_mortality_loss = props.natural_mortality_rate
    fishing_mortality_loss = props.fishing_mortality_rate
    growth_rate = props.growth_rate
    max_biomass = props.max_biomass_in_cell

    move_up = actions[:, :, Action.UP.value]
    move_down = actions[:, :, Action.DOWN.value]
    move_left = actions[:, :, Action.LEFT.value]
    move_right = actions[:, :, Action.RIGHT.value]

    total_activity = move_up + move_down + move_left + move_right

    activity_mr_loss = total_activity * activity_metabolic_rate

    # --- Biomass updates ---
    initial_biomass = world[pad:-pad, pad:-pad, biomass_offset].copy()
    energy_at_positions = world[pad:-pad, pad:-pad, energy_offset]

    loss_factor = activity_mr_loss + standard_metabolic_rate + natural_mortality_loss
    # fished_amount = initial_biomass * fishing_mortality_loss
    # total_fished = np.sum(fished_amount)
    # const.update_fishing_amounts(species_key, total_fished)
    world[:, :, biomass_offset] *= 1 - fishing_mortality_loss

    logistic_delta = growth_rate * initial_biomass * (
        1 - initial_biomass / max_biomass
    )
    logistic_delta = np.clip(logistic_delta, 0, None)
    # -----------------------------------------

    # Apply either loss (low energy) or logistic growth (enough energy)
    world[pad:-pad, pad:-pad, biomass_offset] += np.where(
        energy_at_positions < 50,
        -initial_biomass * loss_factor,
        logistic_delta,
    )

    biomass_after_loss = world[pad:-pad, pad:-pad, biomass_offset]

    initial_energy = world[pad:-pad, pad:-pad, energy_offset]

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
    movement_deltas[pad:-pad, : -2 * pad, energy_offset] += initial_energy * biomass_up
    movement_deltas[pad:-pad, 2 * pad :, biomass_offset] += biomass_down
    movement_deltas[pad:-pad, 2 * pad :, energy_offset] += initial_energy * biomass_down
    movement_deltas[: -2 * pad, pad:-pad, biomass_offset] += biomass_left
    movement_deltas[: -2 * pad, pad:-pad, energy_offset] += (
        initial_energy * biomass_left
    )
    movement_deltas[2 * pad :, pad:-pad, biomass_offset] += biomass_right
    movement_deltas[2 * pad :, pad:-pad, energy_offset] += (
        initial_energy * biomass_right
    )

    # The stationary.
    movement_deltas[pad:-pad, pad:-pad, energy_offset] += initial_energy * (
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
    activity_mr_loss = eat * props.activity_metabolic_rate

    # Cache biomass and energy at batch positions.
    init_biomass = world[pad:-pad, pad:-pad, biomass_offset].copy()
    init_energy = world[pad:-pad, pad:-pad, energy_offset].copy()

    # --- Biomass loss from metabolic cost of eating ---
    world[pad:-pad, pad:-pad, biomass_offset] *= 1 - np.where(
        init_energy < 50, activity_mr_loss, 0
    )

    # Compute total eat potential.
    initial_total_eat = init_biomass * eat
    still_left_to_eat = initial_total_eat.copy()

    # Randomize prey order.
    prey_list = props.prey.copy()
    np.random.shuffle(prey_list)

    # Process each prey species.
    for prey_species in prey_list:
        prey_biomass_offset = MODEL_OFFSETS[prey_species]["biomass"]
        prey_biomass = world[pad:-pad, pad:-pad, prey_biomass_offset]
        # Consume as much as possible, up to total_eat_amount.
        eat_amount = np.minimum(prey_biomass, still_left_to_eat)
        # Reduce prey biomass.
        world[pad:-pad, pad:-pad, prey_biomass_offset] -= eat_amount
        still_left_to_eat -= eat_amount

    actual_eaten = initial_total_eat - still_left_to_eat

    # Avoid division by zero using np.where.
    eaten_percentage = np.divide(
        actual_eaten,
        init_biomass,
        out=np.zeros_like(actual_eaten),
        where=initial_total_eat > 0,
    )

    world[pad:-pad, pad:-pad, energy_offset] += (
        props.energy_reward * eaten_percentage
    )

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
