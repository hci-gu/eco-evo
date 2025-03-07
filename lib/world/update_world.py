import numpy as np
# import numba
import lib.constants as const
from lib.constants import Action

# def perform_single_action(world, world_data, observation, action_values, species_key):
def perform_cell_action(observation, action_values, species_key):
    """
    Update a 3x3 observation (centered on the acting cell at [1,1])
    based on a single cell's action.

    Parameters:
        observation: np.array of shape (3, 3, channels) representing
                     the cell and its surrounding neighborhood.
        action_values: np.array with shape (num_actions,) giving the
                       scalar action values for the cell.
        species_key: key used to look up species properties in const.SPECIES_MAP.
    Returns:
        The updated 3x3 observation.
    """
    # Increase an action counter (assumed to be at channel index 4)
    observation[1, 1, 4] += 1

    # Extract action components (assumes an Action enum with .value)
    move_up    = action_values[Action.UP.value]
    move_down  = action_values[Action.DOWN.value]
    move_left  = action_values[Action.LEFT.value]
    move_right = action_values[Action.RIGHT.value]
    eat        = action_values[Action.EAT.value]
    rest       = action_values[Action.REST.value]

    # Get species properties and biomass channel index
    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]

    # Record the initial biomass in the central cell before losses
    initial_biomass = observation[1, 1, biomass_offset]

    # Compute losses due to activity, resting, natural and fishing mortality
    total_activity = move_up + move_down + move_left + move_right + eat
    activity_mr_loss = total_activity * species_properties["activity_metabolic_rate"]
    resting_mr_loss  = rest * species_properties["standard_metabolic_rate"]
    natural_mortality_loss = species_properties["natural_mortality_rate"]
    fishing_mortality_loss = species_properties["fishing_mortality_rate"]

    # Subtract metabolic and mortality losses from the central cell biomass
    observation[1, 1, biomass_offset] -= (
          initial_biomass * activity_mr_loss
        + initial_biomass * resting_mr_loss
        + initial_biomass * natural_mortality_loss
        + initial_biomass * fishing_mortality_loss
    )

    biomass_after_loss = observation[1, 1, biomass_offset]

    # Calculate how much biomass should move in each direction
    biomass_up    = biomass_after_loss * move_up
    biomass_down  = biomass_after_loss * move_down
    biomass_left  = biomass_after_loss * move_left
    biomass_right = biomass_after_loss * move_right

    # The neighboring cells in the 3x3 observation are:
    # Up: (1,0), Down: (1,2), Left: (0,1), Right: (2,1)
    # Check that the neighbor cell is water (assumed to be indicated by 1 in the terrain channel)
    if observation[1, 0, const.OFFSETS_TERRAIN_WATER] == 1:
        observation[1, 0, biomass_offset] += biomass_up
        biomass_after_loss -= biomass_up

    if observation[1, 2, const.OFFSETS_TERRAIN_WATER] == 1:
        observation[1, 2, biomass_offset] += biomass_down
        biomass_after_loss -= biomass_down

    if observation[0, 1, const.OFFSETS_TERRAIN_WATER] == 1:
        observation[0, 1, biomass_offset] += biomass_left
        biomass_after_loss -= biomass_left

    if observation[2, 1, const.OFFSETS_TERRAIN_WATER] == 1:
        observation[2, 1, biomass_offset] += biomass_right
        biomass_after_loss -= biomass_right

    # Update the central cell with the biomass remaining after movement
    observation[1, 1, biomass_offset] = biomass_after_loss

    # Process eating behavior:
    # For every prey species that can be consumed by the current species...
    for prey_species, prey_info in const.EATING_MAP[species_key].items():
        prey_biomass_offset = const.SPECIES_MAP[prey_species]["biomass_offset"]
        prey_biomass = observation[1, 1, prey_biomass_offset]
        # Calculate how much biomass could be eaten based on the current cell’s initial biomass
        possible_eat = initial_biomass * eat * species_properties["max_consumption_rate"]
        # Eat the minimum of what’s available and what can be consumed
        eat_amount = min(prey_biomass, possible_eat)
        # Remove a fraction of prey biomass and boost the acting cell’s biomass
        observation[1, 1, prey_biomass_offset] -= eat_amount * (const.EAT_REWARD_BOOST / 2)
        observation[1, 1, biomass_offset] += eat_amount * const.EAT_REWARD_BOOST

    # Clamp biomass for every species in the 3x3 patch to allowed bounds
    for sp, sp_info in const.SPECIES_MAP.items():
        sp_biomass_offset = sp_info["biomass_offset"]
        observation[:, :, sp_biomass_offset] = np.clip(
            observation[:, :, sp_biomass_offset],
            0,
            sp_info["max_biomass_in_cell"]
        )

    # If the acting cell's biomass falls below a minimum threshold, set it to zero
    if observation[1, 1, biomass_offset] < species_properties["min_biomass_in_cell"]:
        observation[1, 1, biomass_offset] = 0

    return observation

def perform_action(world, world_data, action_values_batch, species_key, positions):
    # positions is assumed to be a NumPy array of shape (N, 2)
    x_batch = positions[:, 0]
    y_batch = positions[:, 1]
    
    # Update world_data at the given positions
    world_data[x_batch, y_batch, 4] += 1

    # Extract actions for each move
    move_up    = action_values_batch[:, Action.UP.value]
    move_down  = action_values_batch[:, Action.DOWN.value]
    move_left  = action_values_batch[:, Action.LEFT.value]
    move_right = action_values_batch[:, Action.RIGHT.value]
    eat        = action_values_batch[:, Action.EAT.value]
    rest       = action_values_batch[:, Action.REST.value]

    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]

    total_activity = move_up + move_down + move_left + move_right + eat
    activity_mr_loss = total_activity * species_properties["activity_metabolic_rate"]
    resting_mr_loss  = rest * species_properties["standard_metabolic_rate"]
    natural_mortality_loss = species_properties["natural_mortality_rate"]
    fishing_mortality_loss = species_properties["fishing_mortality_rate"]

    initial_biomass = world[x_batch, y_batch, biomass_offset]

    # Subtract losses from biomass
    world[x_batch, y_batch, biomass_offset] -= (initial_biomass * activity_mr_loss)
    world[x_batch, y_batch, biomass_offset] -= (initial_biomass * resting_mr_loss)
    world[x_batch, y_batch, biomass_offset] -= (initial_biomass * natural_mortality_loss)
    world[x_batch, y_batch, biomass_offset] -= (initial_biomass * fishing_mortality_loss)

    biomass_after_loss = world[x_batch, y_batch, biomass_offset]

    # Biomass to move in each direction
    biomass_up    = biomass_after_loss * move_up
    biomass_down  = biomass_after_loss * move_down
    biomass_left  = biomass_after_loss * move_left
    biomass_right = biomass_after_loss * move_right

    # Determine which moves are valid based on terrain
    valid_left_mask  = (x_batch > 0) & (world[x_batch - 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_right_mask = (x_batch < const.WORLD_SIZE) & (world[x_batch + 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_up_mask    = (y_batch > 0) & (world[x_batch, y_batch - 1, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_down_mask  = (y_batch < const.WORLD_SIZE) & (world[x_batch, y_batch + 1, const.OFFSETS_TERRAIN_WATER] == 1)

    total_biomass_moved = np.zeros_like(biomass_after_loss)

    if valid_up_mask.any():
        total_biomass_moved[valid_up_mask] += biomass_up[valid_up_mask]
        world[x_batch[valid_up_mask], y_batch[valid_up_mask] - 1, biomass_offset] += biomass_up[valid_up_mask]

    if valid_down_mask.any():
        total_biomass_moved[valid_down_mask] += biomass_down[valid_down_mask]
        world[x_batch[valid_down_mask], y_batch[valid_down_mask] + 1, biomass_offset] += biomass_down[valid_down_mask]

    if valid_left_mask.any():
        total_biomass_moved[valid_left_mask] += biomass_left[valid_left_mask]
        world[x_batch[valid_left_mask] - 1, y_batch[valid_left_mask], biomass_offset] += biomass_left[valid_left_mask]

    if valid_right_mask.any():
        total_biomass_moved[valid_right_mask] += biomass_right[valid_right_mask]
        world[x_batch[valid_right_mask] + 1, y_batch[valid_right_mask], biomass_offset] += biomass_right[valid_right_mask]

    # Subtract biomass that has moved out of the original cell
    world[x_batch, y_batch, biomass_offset] -= total_biomass_moved

    # Process eating: for each prey species the current species can eat...
    for prey_species, prey_info in const.EATING_MAP[species_key].items():
        prey_biomass_offset = const.SPECIES_MAP[prey_species]["biomass_offset"]
        prey_biomass = world[x_batch, y_batch, prey_biomass_offset]
        eat_amounts = initial_biomass * eat * const.SPECIES_MAP[species_key]["max_consumption_rate"]
        # Use elementwise minimum between available prey biomass and what can be eaten
        eat_amount = np.minimum(prey_biomass, eat_amounts)
        world[x_batch, y_batch, prey_biomass_offset] -= (eat_amount * (const.EAT_REWARD_BOOST / 2))
        world[x_batch, y_batch, biomass_offset] += (eat_amount * const.EAT_REWARD_BOOST)

    # Clamp biomass in every cell for each species
    for species in const.SPECIES_MAP.keys():
        species_biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        world[:, :, species_biomass_offset] = np.clip(
            world[:, :, species_biomass_offset],
            a_min=0,
            a_max=const.SPECIES_MAP[species]["max_biomass_in_cell"]
        )

    # For the current species, if biomass is too low, set it to zero
    biomass = world[:, :, biomass_offset]
    low_biomass_mask = biomass < species_properties["min_biomass_in_cell"]
    world[:, :, biomass_offset][low_biomass_mask] = 0

    return world

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
        if biomass < (properties["starting_biomass"] * const.MIN_PERCENT_ALIVE) or \
           biomass > (properties["starting_biomass"] * const.MAX_PERCENT_ALIVE):
            return False
    return True
