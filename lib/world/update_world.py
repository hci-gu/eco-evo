import numpy as np
from numba import njit
import lib.constants as const
from lib.constants import Action

def get_movement_delta(world, world_data, species_key, action_values_batch, positions):
    movement_deltas = np.zeros_like(world)

    x_batch = positions[:, 0]
    y_batch = positions[:, 1]
    world_data[x_batch, y_batch, 4] += 1

    move_up     = action_values_batch[:, Action.UP.value]
    move_down   = action_values_batch[:, Action.DOWN.value]
    move_left   = action_values_batch[:, Action.LEFT.value]
    move_right  = action_values_batch[:, Action.RIGHT.value]

    stationary  = action_values_batch[:, Action.EAT.value]

    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]
    energy_offset  = species_properties["energy_offset"]

    total_activity = move_up + move_down + move_left + move_right
    activity_mr_loss = total_activity * species_properties["activity_metabolic_rate"]
    resting_mr_loss  = species_properties["standard_metabolic_rate"]
    natural_mortality_loss = species_properties["natural_mortality_rate"]
    fishing_mortality_loss = species_properties["fishing_mortality_rate"]

    growth_rate = species_properties["growth_rate"]

    # --- Biomass updates ---
    initial_biomass = world[x_batch, y_batch, biomass_offset]
    energy_at_positions = world[x_batch, y_batch, energy_offset]
    below_threshold_mask = energy_at_positions < 50
    above_threshold_mask = ~below_threshold_mask

    loss_factor = activity_mr_loss[below_threshold_mask] + resting_mr_loss + natural_mortality_loss
    world[x_batch, y_batch, biomass_offset] *= 1 - fishing_mortality_loss
    world[x_batch[below_threshold_mask], y_batch[below_threshold_mask], biomass_offset] -= (
        initial_biomass[below_threshold_mask] * loss_factor
    )
    world[x_batch[above_threshold_mask], y_batch[above_threshold_mask], biomass_offset] += (
        initial_biomass[above_threshold_mask] * growth_rate
    )

    biomass_after_loss = world[x_batch, y_batch, biomass_offset]

    # Biomass to move in each direction
    biomass_up    = biomass_after_loss * move_up
    biomass_down  = biomass_after_loss * move_down
    biomass_left  = biomass_after_loss * move_left
    biomass_right = biomass_after_loss * move_right
    biomass_stationary = biomass_after_loss * stationary

    initial_energy = world[x_batch, y_batch, energy_offset]

    # --- Determine valid moves (based on terrain) ---
    valid_left_mask  = (x_batch > 0) & (world[x_batch - 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_right_mask = (x_batch < const.WORLD_SIZE) & (world[x_batch + 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_up_mask    = (y_batch > 0) & (world[x_batch, y_batch - 1, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_down_mask  = (y_batch < const.WORLD_SIZE) & (world[x_batch, y_batch + 1, const.OFFSETS_TERRAIN_WATER] == 1)

    total_biomass_moved = np.zeros_like(biomass_after_loss)
    movement_masks = [
        (valid_up_mask, biomass_up, (0, -1)),
        (valid_down_mask, biomass_down, (0, 1)),
        (valid_left_mask, biomass_left, (-1, 0)),
        (valid_right_mask, biomass_right, (1, 0))
    ]

    for mask, biomass_direction, (dx, dy) in movement_masks:
        if mask.any():
            x_target = x_batch[mask] + dx
            y_target = y_batch[mask] + dy
            total_biomass_moved[mask] += biomass_direction[mask]
            np.add.at(movement_deltas[:, :, biomass_offset], (x_target, y_target), biomass_direction[mask])
            np.add.at(movement_deltas[:, :, energy_offset], (x_target, y_target), initial_energy[mask] * biomass_direction[mask])

    movement_deltas[x_batch, y_batch, energy_offset] = initial_energy * biomass_stationary
    movement_deltas[x_batch, y_batch, biomass_offset] -= total_biomass_moved

    return movement_deltas

def all_movement_delta(world, world_data, species_key, actions):

    pad = 1
    world_data[pad:-pad, pad:-pad, 4] += 1

    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]
    energy_offset = species_properties["energy_offset"]
    activity_metabolic_rate = species_properties["activity_metabolic_rate"]
    standard_metabolic_rate = species_properties["standard_metabolic_rate"]
    natural_mortality_loss = species_properties["natural_mortality_rate"]
    fishing_mortality_loss = species_properties["fishing_mortality_rate"]
    growth_rate = species_properties["growth_rate"]

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
    world[:, :, biomass_offset] *= 1 - fishing_mortality_loss

    # Growth or shrink depending on the enery.
    world[pad:-pad, pad:-pad, biomass_offset] += initial_biomass * np.where(
        energy_at_positions < 50, -loss_factor, growth_rate
    )

    biomass_after_loss = world[pad:-pad, pad:-pad, biomass_offset]

    initial_energy = world[pad:-pad, pad:-pad, energy_offset]

    # --- Determine valid moves (based on terrain) ---
    valid_left_mask = world[: -2 * pad, pad:-pad, const.OFFSETS_TERRAIN_WATER] == 1
    valid_right_mask = world[2 * pad :, pad:-pad, const.OFFSETS_TERRAIN_WATER] == 1
    valid_up_mask = world[pad:-pad, : -2 * pad, const.OFFSETS_TERRAIN_WATER] == 1
    valid_down_mask = world[pad:-pad, 2 * pad :, const.OFFSETS_TERRAIN_WATER] == 1

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


def apply_movement_delta(world, species_key, movement_deltas):
    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]
    energy_offset  = species_properties["energy_offset"]

    # Apply the movement deltas to biomass and energy.
    world[:, :, biomass_offset] += movement_deltas[:, :, biomass_offset]
    denom = np.where(np.abs(world[:, :, biomass_offset]) == 0, 1, np.abs(world[:, :, biomass_offset]))
    world[:, :, energy_offset] = movement_deltas[:, :, energy_offset] / denom
    # reduce energy by base amount
    world[:, :, energy_offset] -= const.BASE_ENERGY_COST

def perform_eating(world, species_key, action_values_batch, positions):
    # Extract batch indices.
    x_batch = positions[:, 0]
    y_batch = positions[:, 1]

    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]
    energy_offset  = species_properties["energy_offset"]

    # Get eating actions and metabolic cost.
    eat = action_values_batch[:, Action.EAT.value]
    activity_mr_loss = eat * species_properties["activity_metabolic_rate"]

    # Cache biomass and energy at batch positions.
    init_biomass = world[x_batch, y_batch, biomass_offset]
    init_energy = world[x_batch, y_batch, energy_offset]

    # --- Biomass loss from metabolic cost of eating ---
    below_thresh = init_energy < 50
    if below_thresh.any():
        loss = init_biomass[below_thresh] * activity_mr_loss[below_thresh]
        world[x_batch[below_thresh], y_batch[below_thresh], biomass_offset] -= loss

    # Compute total eat potential.
    initial_total_eat = init_biomass * eat
    total_eat_amount = initial_total_eat.copy()

    # Randomize prey order.
    prey_list = list(const.EATING_MAP[species_key].items())
    np.random.shuffle(prey_list)

    # Process each prey species.
    for prey_species, _ in prey_list:
        prey_biomass_offset = const.SPECIES_MAP[prey_species]["biomass_offset"]
        prey_biomass = world[x_batch, y_batch, prey_biomass_offset]
        # Consume as much as possible, up to total_eat_amount.
        eat_amount = np.minimum(prey_biomass, total_eat_amount)
        # Reduce prey biomass.
        world[x_batch, y_batch, prey_biomass_offset] -= eat_amount
        total_eat_amount -= eat_amount

    actual_eaten = initial_total_eat - total_eat_amount
    
    # Avoid division by zero using np.where.
    eaten_percentage = np.divide(
        actual_eaten, 
        init_biomass, 
        out=np.zeros_like(actual_eaten), 
        where=initial_total_eat > 0
    )

    if species_key == "cod":
        world[x_batch, y_batch, energy_offset] += const.ENERGY_REWARD_FOR_EATING_COD * eaten_percentage
    else:
        world[x_batch, y_batch, energy_offset] += const.ENERGY_REWARD_FOR_EATING * eaten_percentage

    # --- Handle low biomass: if biomass falls below minimum, set it (and energy) to zero ---
    # Instead of indexing the entire grid, update only for batch positions.
    current_biomass = world[x_batch, y_batch, biomass_offset]
    low_biomass = current_biomass < species_properties["min_biomass_in_cell"]
    if low_biomass.any():
        world[x_batch[low_biomass], y_batch[low_biomass], biomass_offset] = 0
        world[x_batch[low_biomass], y_batch[low_biomass], energy_offset] = 0

    # --- Clamp biomass and energy for every species ---
    for species, props in const.SPECIES_MAP.items():
        species_biomass_offset = props["biomass_offset"]
        species_energy_offset = props["energy_offset"]
        world[..., species_biomass_offset] = np.clip(
            world[..., species_biomass_offset],
            0,
            props["max_biomass_in_cell"]
        )
        world[..., species_energy_offset] = np.clip(
            world[..., species_energy_offset],
            0,
            const.MAX_ENERGY
        )

def matrix_perform_eating(world, species_key, actions):

    pad = 1
    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]
    energy_offset = species_properties["energy_offset"]

    # Get eating actions and metabolic cost.
    eat = actions[:, :, Action.EAT.value]
    activity_mr_loss = eat * species_properties["activity_metabolic_rate"]

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
    prey_list = list(const.EATING_MAP[species_key].items())
    np.random.shuffle(prey_list)

    # Process each prey species.
    for prey_species, _ in prey_list:
        prey_biomass_offset = const.SPECIES_MAP[prey_species]["biomass_offset"]
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
        const.ENERGY_REWARD_FOR_EATING * eaten_percentage
    )

    # --- Handle low biomass: if biomass falls below minimum, set it (and energy) to zero ---
    low_biomass = world[..., biomass_offset] < species_properties["min_biomass_in_cell"]
    world[low_biomass, biomass_offset] = 0
    world[low_biomass, energy_offset] = 0

    # TODO: Should this be done here?
    np.clip(
        world[pad:-pad, pad:-pad, biomass_offset],
        0,
        species_properties["max_biomass_in_cell"],
        out=world[pad:-pad, pad:-pad, biomass_offset],
    )
    np.clip(
        world[pad:-pad, pad:-pad, energy_offset],
        0,
        const.MAX_ENERGY,
        out=world[pad:-pad, pad:-pad, energy_offset],
    )

    return world

def perform_action(world, world_data, species_key, action_values_batch, positions):
    pass

def remove_species_from_fishing(world):
    for species in const.SPECIES_MAP.keys():
        if species == "plankton":
            continue
        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        world[:, :, biomass_offset] *= (1 - const.FISHING_AMOUNT)
    return world

def total_biomass(world):
    total = 0
    for species in const.SPECIES_MAP.keys():
        if species == "plankton":
            continue
        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        total += world[:, :, biomass_offset].sum()
    return np.log(total + 1e-8)

def world_is_alive(world):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, TOTAL_CHANNELS) where:
      - The first 3 channels represent the terrain (one-hot encoded).
      - The subsequent channels represent the biomass for each species.
    """
    for species in const.SPECIES_MAP.keys():
        if species == "plankton":
            continue
        properties = const.SPECIES_MAP[species]
        biomass_offset = properties["biomass_offset"]
        biomass = world[:, :, biomass_offset].sum()
        if biomass < (properties["starting_biomass"] * const.MIN_PERCENT_ALIVE) or biomass > (properties["starting_biomass"] * const.MAX_PERCENT_ALIVE):
            return False
    return True
