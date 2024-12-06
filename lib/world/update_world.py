import torch
import lib.constants as const
from lib.constants import Action

def perform_action(world_tensor, action_values_batch, species_key, positions_tensor):
    initial_tensor = world_tensor.clone()
    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]

    move_up = action_values_batch[:, Action.UP.value]
    move_down = action_values_batch[:, Action.DOWN.value]
    move_left = action_values_batch[:, Action.LEFT.value]
    move_right = action_values_batch[:, Action.RIGHT.value]
    eat = action_values_batch[:, Action.EAT.value]

    species_properties = const.SPECIES_MAP[species_key]
    biomass_offset = species_properties["biomass_offset"]
    energy_offset = species_properties["energy_offset"]

    initial_biomass = world_tensor[x_batch, y_batch, biomass_offset]
    initial_energy = world_tensor[x_batch, y_batch, energy_offset]

    # Apply energy loss to all cells
    world_tensor[x_batch, y_batch, energy_offset] -= const.BASE_ENERGY_COST

    # Apply BASE_BIOMASS_LOSS to all action-taking cells (a constant percentage loss)
    biomass_loss = initial_biomass * const.BASE_BIOMASS_LOSS
    biomass_gain = initial_biomass * const.BIOMASS_GROWTH_RATE
    energy_at_positions = world_tensor[x_batch, y_batch, energy_offset]
    energy_below_50_mask = energy_at_positions < 50
    energy_above = energy_at_positions >= 50
    
    world_tensor[x_batch[energy_below_50_mask], y_batch[energy_below_50_mask], biomass_offset] -= biomass_loss[energy_below_50_mask]
    world_tensor[x_batch[energy_above], y_batch[energy_above], biomass_offset] += biomass_gain[energy_above]

    initial_biomass = world_tensor[x_batch, y_batch, biomass_offset]
    initial_energy = world_tensor[x_batch, y_batch, energy_offset]

    biomass_up = initial_biomass * move_up
    biomass_down = initial_biomass * move_down
    biomass_left = initial_biomass * move_left
    biomass_right = initial_biomass * move_right

    energy_up = initial_energy * move_up
    energy_down = initial_energy * move_down
    energy_left = initial_energy * move_left
    energy_right = initial_energy * move_right

    valid_left_mask = (x_batch > 0) & (world_tensor[x_batch - 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_right_mask = (x_batch < const.WORLD_SIZE) & (world_tensor[x_batch + 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_up_mask = (y_batch > 0) & (world_tensor[x_batch, y_batch - 1, const.OFFSETS_TERRAIN_WATER] == 1)
    valid_down_mask = (y_batch < const.WORLD_SIZE) & (world_tensor[x_batch, y_batch + 1, const.OFFSETS_TERRAIN_WATER] == 1)

    total_biomass_moved = torch.zeros_like(initial_biomass)
    total_energy_moved = torch.zeros_like(initial_energy)

    destination_energy = torch.zeros_like(initial_energy)
    if valid_up_mask.any():
        total_biomass_moved[valid_up_mask] += biomass_up[valid_up_mask]
        total_energy_moved[valid_up_mask] += energy_up[valid_up_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_up_mask], y_batch[valid_up_mask] - 1, biomass_offset] += biomass_up[valid_up_mask]
        destination_energy[valid_up_mask] = world_tensor[x_batch[valid_up_mask], y_batch[valid_up_mask] - 1, energy_offset]

    if valid_down_mask.any():
        total_biomass_moved[valid_down_mask] += biomass_down[valid_down_mask]
        total_energy_moved[valid_down_mask] += energy_down[valid_down_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_down_mask], y_batch[valid_down_mask] + 1, biomass_offset] += biomass_down[valid_down_mask]
        destination_energy[valid_down_mask] = world_tensor[x_batch[valid_down_mask], y_batch[valid_down_mask] + 1, energy_offset]

    if valid_left_mask.any():
        total_biomass_moved[valid_left_mask] += biomass_left[valid_left_mask]
        total_energy_moved[valid_left_mask] += energy_left[valid_left_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_left_mask] - 1, y_batch[valid_left_mask], biomass_offset] += biomass_left[valid_left_mask]
        destination_energy[valid_left_mask] = world_tensor[x_batch[valid_left_mask] - 1, y_batch[valid_left_mask], energy_offset]

    if valid_right_mask.any():
        total_biomass_moved[valid_right_mask] += biomass_right[valid_right_mask]
        total_energy_moved[valid_right_mask] += energy_right[valid_right_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_right_mask] + 1, y_batch[valid_right_mask], biomass_offset] += biomass_right[valid_right_mask]
        destination_energy[valid_right_mask] = world_tensor[x_batch[valid_right_mask] + 1, y_batch[valid_right_mask], energy_offset]

    world_tensor[x_batch, y_batch, biomass_offset] -= total_biomass_moved

    # Rebalance energy for cells receiving biomass
    for direction_mask, new_x, new_y in [(valid_up_mask, x_batch, y_batch - 1),
                                         (valid_down_mask, x_batch, y_batch + 1),
                                         (valid_left_mask, x_batch - 1, y_batch),
                                         (valid_right_mask, x_batch + 1, y_batch)]:
        if direction_mask.any():
            biomass_increased_mask = (world_tensor[new_x[direction_mask], new_y[direction_mask], biomass_offset] >
                                      initial_tensor[new_x[direction_mask], new_y[direction_mask], biomass_offset])

            if biomass_increased_mask.any():
                prev_biomass = initial_tensor[new_x[direction_mask][biomass_increased_mask], 
                                          new_y[direction_mask][biomass_increased_mask], 
                                          biomass_offset]
                prev_energy = initial_tensor[new_x[direction_mask][biomass_increased_mask], 
                                            new_y[direction_mask][biomass_increased_mask], 
                                            energy_offset]
                
                dest_biomass = world_tensor[new_x[direction_mask][biomass_increased_mask], 
                                            new_y[direction_mask][biomass_increased_mask], 
                                            biomass_offset]
                dest_energy = world_tensor[new_x[direction_mask][biomass_increased_mask], 
                                        new_y[direction_mask][biomass_increased_mask], 
                                        energy_offset]

                moved_biomass = total_biomass_moved[direction_mask][biomass_increased_mask]
                moved_energy = total_energy_moved[direction_mask][biomass_increased_mask]

                numerator = prev_biomass * prev_energy + moved_biomass * moved_energy
                denominator = dest_biomass + 1e-6  # Add epsilon to prevent division by zero

                total_energy = numerator / denominator
                total_energy = torch.clamp(total_energy, 0, const.MAX_ENERGY)

                world_tensor[new_x[direction_mask][biomass_increased_mask], 
                            new_y[direction_mask][biomass_increased_mask], 
                            energy_offset] = total_energy

    for prey_species, prey_info in const.EATING_MAP[species_key].items():
        prey_biomass_offset = const.SPECIES_MAP[prey_species]["biomass_offset"]
        prey_biomass = world_tensor[x_batch, y_batch, prey_biomass_offset]
        eat_amounts = initial_biomass * eat
        eat_amount = torch.min(prey_biomass, eat_amounts)
        world_tensor[x_batch, y_batch, prey_biomass_offset] -= eat_amount
        
        reward_scaling_factor = torch.where(eat_amounts > 0, eat_amount / eat_amounts, torch.tensor(0.0))
        # Clamp scaling factor between 0 and 1 to ensure it's in a valid range
        reward_scaling_factor = torch.clamp(reward_scaling_factor, 0, 1)
        energy_reward = reward_scaling_factor * prey_info["nutrition_amount"] * eat
        new_energy = world_tensor[x_batch, y_batch, species_properties["energy_offset"]] + energy_reward
        world_tensor[x_batch, y_batch, species_properties["energy_offset"]] = torch.clamp(new_energy, max=const.MAX_ENERGY)


    for species in const.SPECIES_MAP.keys():
        biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
        world_tensor[:, :, biomass_offset] = torch.clamp(world_tensor[:, :, biomass_offset], min=0, max=const.SPECIES_MAP[species]["max_biomass_in_cell"])

    biomass = world_tensor[:, :, biomass_offset]

    low_biomass_mask = biomass < species_properties["min_biomass_in_cell"]
    world_tensor[:, :, biomass_offset][low_biomass_mask] = 0

    zero_biomass_mask = world_tensor[x_batch, y_batch, biomass_offset] == 0
    world_tensor[x_batch[zero_biomass_mask], y_batch[zero_biomass_mask], energy_offset] = 0

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
        properties = const.SPECIES_MAP[species]
        biomass_offset = properties["biomass_offset"]
        biomass = world_tensor[:, :, biomass_offset].sum()
        if biomass < (properties["starting_biomass"] * const.MIN_PERCENT_ALIVE) or biomass > (properties["starting_biomass"] * const.MAX_PERCENT_ALIVE):
            return False

    return True