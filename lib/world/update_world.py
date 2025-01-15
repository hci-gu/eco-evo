import torch
import lib.constants as const
from lib.constants import Action

def perform_action(world_tensor, action_values_batch, species_key, positions_tensor):
    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]

    move_up = action_values_batch[:, Action.UP.value]
    move_down = action_values_batch[:, Action.DOWN.value]
    move_left = action_values_batch[:, Action.LEFT.value]
    move_right = action_values_batch[:, Action.RIGHT.value]
    eat = action_values_batch[:, Action.EAT.value]
    rest = action_values_batch[:, Action.REST.value]

    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]

    total_activity = move_up + move_down + move_left + move_right + eat
    activity_mr_loss = total_activity * species_properties["activity_metabolic_rate"]
    resting_mr_loss = rest * species_properties["standard_metabolic_rate"]
    natural_mortality_loss = species_properties["natural_mortality_rate"]
    # fishing_mortality_loss = species_properties["fishing_mortality_rate"]

    initial_biomass = world_tensor[x_batch, y_batch, biomass_offset]

    world_tensor[x_batch, y_batch, biomass_offset] -= (initial_biomass * activity_mr_loss)
    world_tensor[x_batch, y_batch, biomass_offset] -= (initial_biomass * resting_mr_loss)
    world_tensor[x_batch, y_batch, biomass_offset] -= (initial_biomass * natural_mortality_loss)
    # world_tensor[x_batch, y_batch, biomass_offset] -= (initial_biomass * fishing_mortality_loss)

    biomass_after_loss = world_tensor[x_batch, y_batch, biomass_offset]

    biomass_up = biomass_after_loss * move_up
    biomass_down = biomass_after_loss * move_down
    biomass_left = biomass_after_loss * move_left
    biomass_right = biomass_after_loss * move_right

    valid_left_mask = (x_batch > 0) & (world_tensor[x_batch - 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_right_mask = (x_batch < const.WORLD_SIZE) & (world_tensor[x_batch + 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_up_mask = (y_batch > 0) & (world_tensor[x_batch, y_batch - 1, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_down_mask = (y_batch < const.WORLD_SIZE) & (world_tensor[x_batch, y_batch + 1, const.OFFSETS_TERRAIN_WATER] == 1)

    total_biomass_moved = torch.zeros_like(biomass_after_loss)

    if valid_up_mask.any():
        total_biomass_moved[valid_up_mask] += biomass_up[valid_up_mask]
        world_tensor[x_batch[valid_up_mask], y_batch[valid_up_mask] - 1, biomass_offset] += biomass_up[valid_up_mask]

    if valid_down_mask.any():
        total_biomass_moved[valid_down_mask] += biomass_down[valid_down_mask]
        world_tensor[x_batch[valid_down_mask], y_batch[valid_down_mask] + 1, biomass_offset] += biomass_down[valid_down_mask]

    if valid_left_mask.any():
        total_biomass_moved[valid_left_mask] += biomass_left[valid_left_mask]
        world_tensor[x_batch[valid_left_mask] - 1, y_batch[valid_left_mask], biomass_offset] += biomass_left[valid_left_mask]

    if valid_right_mask.any():
        total_biomass_moved[valid_right_mask] += biomass_right[valid_right_mask]
        world_tensor[x_batch[valid_right_mask] + 1, y_batch[valid_right_mask], biomass_offset] += biomass_right[valid_right_mask]

    world_tensor[x_batch, y_batch, biomass_offset] -= total_biomass_moved

    for prey_species, prey_info in const.EATING_MAP[species_key].items():
        prey_biomass_offset = const.SPECIES_MAP[prey_species]["biomass_offset"]
        prey_biomass = world_tensor[x_batch, y_batch, prey_biomass_offset]
        eat_amounts = initial_biomass * eat * const.SPECIES_MAP[species_key]["max_consumption_rate"]
        eat_amount = torch.min(prey_biomass, eat_amounts)
        world_tensor[x_batch, y_batch, prey_biomass_offset] -= (eat_amount * (const.EAT_REWARD_BOOST / 2))
        world_tensor[x_batch, y_batch, biomass_offset] += (eat_amount * const.EAT_REWARD_BOOST)

    for species in const.SPECIES_MAP.keys():
        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        world_tensor[:, :, biomass_offset] = torch.clamp(world_tensor[:, :, biomass_offset], min=0, max=const.SPECIES_MAP[species]["max_biomass_in_cell"])

    biomass = world_tensor[:, :, biomass_offset]

    low_biomass_mask = biomass < species_properties["min_biomass_in_cell"]
    world_tensor[:, :, biomass_offset][low_biomass_mask] = 0

    return world_tensor

def remove_species_from_fishing(world_tensor):
    for species in const.SPECIES_MAP.keys():
        if species == "plankton":
            continue

        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        world_tensor[:, :, biomass_offset] *= (1 - const.FISHING_AMOUNT)

def world_is_alive(world_tensor):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, 6) where:
    - The first 3 values represent the terrain (one-hot encoded).
    - The last 3 values represent the biomass of plankton, anchovy, and cod, respectively.
    """
    for species in const.SPECIES_MAP.keys():
        if species == "plankton":
            continue
        properties = const.SPECIES_MAP[species]
        biomass_offset = properties["biomass_offset"]
        biomass = world_tensor[:, :, biomass_offset].sum()
        if biomass < (properties["starting_biomass"] * const.MIN_PERCENT_ALIVE) or biomass > (properties["starting_biomass"] * const.MAX_PERCENT_ALIVE):
            return False

    return True