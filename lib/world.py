from enum import Enum
import opensimplex
import math
from random import random
import torch.nn as nn
import torch
import lib.constants as const

# device mps or cpu
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

class Terrain(Enum):
    LAND = 0
    WATER = 1
    OUT_OF_BOUNDS = 2

class Species(Enum):
    PLANKTON = 0
    ANCHOVY = 1
    COD = 2

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

delta_for_action = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1)
}

def noop(_, __):
    pass

# Define movement shifts for each direction
def shift(tensor, delta):
    delta_x, delta_y = delta
    return torch.roll(torch.roll(tensor, delta_x, dims=0), delta_y, dims=1)

def perform_action(world_tensor, action_values_batch, species_index, positions_tensor, debug_visualize=noop):
    initial_tensor = world_tensor.clone()

    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]

    move_up = action_values_batch[:, Action.UP.value]
    move_down = action_values_batch[:, Action.DOWN.value]
    move_left = action_values_batch[:, Action.LEFT.value]
    move_right = action_values_batch[:, Action.RIGHT.value]
    eat = action_values_batch[:, Action.EAT.value]
    # print(f"move_up: {move_up}, move_down: {move_down}, move_left: {move_left}, move_right: {move_right}, eat: {eat}")

    initial_biomass = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_index]
    initial_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_index]

    # Apply energy loss to all cells
    world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_index] -= const.BASE_ENERGY_COST

    # Apply BASE_BIOMASS_LOSS to all action-taking cells (a constant percentage loss)
    biomass_loss = initial_biomass * const.BASE_BIOMASS_LOSS
    # make it a minimum amount
    # biomass_loss = torch.clamp(biomass_loss, min=1)
    biomass_gain = initial_biomass * const.BIOMASS_GROWTH_RATE
    # # only subtract if the energy is below 50%
    energy_at_positions = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_index]
    # # Create a mask for cells where energy is below 50
    energy_below_50_mask = energy_at_positions < 50
    energy_above = energy_at_positions >= 50
    
    world_tensor[x_batch[energy_below_50_mask],
             y_batch[energy_below_50_mask],
             const.OFFSETS_BIOMASS + species_index] -= biomass_loss[energy_below_50_mask]
    
    # Apply biomass gain to cells where energy is above 50
    world_tensor[x_batch[energy_above],
             y_batch[energy_above],
             const.OFFSETS_BIOMASS + species_index] += biomass_gain[energy_above]


    initial_biomass = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_index]
    initial_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_index]


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
    # print(f"valid_right_mask: {valid_right_mask}, valid_left_mask: {valid_left_mask}, valid_up_mask: {valid_up_mask}, valid_down_mask: {valid_down_mask}")

    total_biomass_moved = torch.zeros_like(initial_biomass)
    total_energy_moved = torch.zeros_like(initial_energy)

    destination_energy = torch.zeros_like(initial_energy)
    if valid_up_mask.any():
        total_biomass_moved[valid_up_mask] += biomass_up[valid_up_mask]
        total_energy_moved[valid_up_mask] += energy_up[valid_up_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_up_mask], y_batch[valid_up_mask] - 1, const.OFFSETS_BIOMASS + species_index] += biomass_up[valid_up_mask]
        destination_energy[valid_up_mask] = world_tensor[x_batch[valid_up_mask], y_batch[valid_up_mask] - 1, const.OFFSETS_ENERGY + species_index]

    if valid_down_mask.any():
        total_biomass_moved[valid_down_mask] += biomass_down[valid_down_mask]
        total_energy_moved[valid_down_mask] += energy_down[valid_down_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_down_mask], y_batch[valid_down_mask] + 1, const.OFFSETS_BIOMASS + species_index] += biomass_down[valid_down_mask]
        destination_energy[valid_down_mask] = world_tensor[x_batch[valid_down_mask], y_batch[valid_down_mask] + 1, const.OFFSETS_ENERGY + species_index]

    if valid_left_mask.any():
        total_biomass_moved[valid_left_mask] += biomass_left[valid_left_mask]
        total_energy_moved[valid_left_mask] += energy_left[valid_left_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_left_mask] - 1, y_batch[valid_left_mask], const.OFFSETS_BIOMASS + species_index] += biomass_left[valid_left_mask]
        destination_energy[valid_left_mask] = world_tensor[x_batch[valid_left_mask] - 1, y_batch[valid_left_mask], const.OFFSETS_ENERGY + species_index]

    if valid_right_mask.any():
        total_biomass_moved[valid_right_mask] += biomass_right[valid_right_mask]
        total_energy_moved[valid_right_mask] += energy_right[valid_right_mask] - const.BASE_ENERGY_COST
        world_tensor[x_batch[valid_right_mask] + 1, y_batch[valid_right_mask], const.OFFSETS_BIOMASS + species_index] += biomass_right[valid_right_mask]
        destination_energy[valid_right_mask] = world_tensor[x_batch[valid_right_mask] + 1, y_batch[valid_right_mask], const.OFFSETS_ENERGY + species_index]

    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_index] -= total_biomass_moved

    # Rebalance energy for cells receiving biomass
    for direction_mask, new_x, new_y in [(valid_up_mask, x_batch, y_batch - 1),
                                         (valid_down_mask, x_batch, y_batch + 1),
                                         (valid_left_mask, x_batch - 1, y_batch),
                                         (valid_right_mask, x_batch + 1, y_batch)]:
        if direction_mask.any():
            # Check if biomass in the destination has increased
            biomass_increased_mask = (world_tensor[new_x[direction_mask], new_y[direction_mask], const.OFFSETS_BIOMASS + species_index] >
                                      initial_tensor[new_x[direction_mask], new_y[direction_mask], const.OFFSETS_BIOMASS + species_index])

            if biomass_increased_mask.any():
                previous_biomass = initial_tensor[new_x[direction_mask][biomass_increased_mask], 
                                                  new_y[direction_mask][biomass_increased_mask], 
                                                  const.OFFSETS_BIOMASS + species_index]
                previous_energy = initial_tensor[new_x[direction_mask][biomass_increased_mask], 
                                                 new_y[direction_mask][biomass_increased_mask], 
                                                 const.OFFSETS_ENERGY + species_index]
                
                # Cells where biomass is moving to
                destination_biomass = world_tensor[new_x[direction_mask][biomass_increased_mask], 
                                                   new_y[direction_mask][biomass_increased_mask], 
                                                   const.OFFSETS_BIOMASS + species_index]
                destination_energy = world_tensor[new_x[direction_mask][biomass_increased_mask], 
                                                  new_y[direction_mask][biomass_increased_mask], 
                                                  const.OFFSETS_ENERGY + species_index]

                # Cells that are sending biomass
                moved_biomass = total_biomass_moved[direction_mask][biomass_increased_mask]
                moved_energy = total_energy_moved[direction_mask][biomass_increased_mask]

                non_zero_biomass_mask = destination_biomass != 0

                # Rebalance energy for cells with non-zero biomass
                total_energy = torch.zeros_like(destination_biomass)  # Initialize to zero for all cells
                total_energy[non_zero_biomass_mask] = (
                    (previous_biomass[non_zero_biomass_mask] / destination_biomass[non_zero_biomass_mask]) * previous_energy[non_zero_biomass_mask] +
                    (moved_biomass[non_zero_biomass_mask] / destination_biomass[non_zero_biomass_mask]) * moved_energy[non_zero_biomass_mask]
                )
                total_energy = torch.clamp(total_energy, 0, const.MAX_ENERGY)

                # Ensure matching shapes by reshaping indices
                world_tensor[new_x[direction_mask][biomass_increased_mask], 
                             new_y[direction_mask][biomass_increased_mask], 
                             const.OFFSETS_ENERGY + species_index] = total_energy

    # Eating Logic
    if species_index == Species.COD.value:
        eat_amounts = initial_biomass * eat * const.EAT_AMOUNT_COD
        # COD eats ANCHOVY in the current position
        prey_biomass = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY]
        eat_amount = torch.min(prey_biomass, eat_amounts)
        world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY] -= eat_amount # Reduce anchovy biomass
        # world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_COD] += eat_amount # Add cod biomass
        # apply cost of eating + gained energy from eating
        energy_gain = eat_amount * const.ENERGY_FROM_BIOMASS

        # Apply the cost of eating + energy gained from eating, make sure the values are scalars
        new_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_COD] + energy_gain
        world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_COD] = torch.clamp(new_energy, max=const.MAX_ENERGY)
    elif species_index == Species.ANCHOVY.value:
        eat_amounts = initial_biomass * eat * const.EAT_AMOUNT_ANCHOVY
        # ANCHOVY eats PLANKTON in the current position
        prey_biomass = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON]
        eat_amount = torch.min(prey_biomass, eat_amounts)  # Fix: ensure this gets scalar
        world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON] -= eat_amount  # Reduce plankton biomass
        # world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY] += eat_amount  # Add anchovy biomass
        # apply cost of eating + gained energy from eating
        energy_gain = eat_amount * const.ENERGY_FROM_BIOMASS
        new_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_ANCHOVY] + energy_gain
        world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_ANCHOVY] = torch.clamp(new_energy, max=const.MAX_ENERGY)
    
    # make sure biomass is not negative
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON] = torch.clamp(world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON], min=0)
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY] = torch.clamp(world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY], min=0)
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_COD] = torch.clamp(world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_COD], min=0)

    # set all cells with 0 biomass to 0 energy
    zero_biomass_mask = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_index] == 0
    world_tensor[x_batch[zero_biomass_mask], y_batch[zero_biomass_mask], const.OFFSETS_ENERGY + species_index] = 0

    return world_tensor

def create_world():
    NOISE_SCALING = 4.5
    opensimplex.seed(int(random() * 100000))

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 9) -> 3 for terrain (3 one-hot) + biomass (3 species) + energy (3 species)
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 9, device=device)

    center_x, center_y = const.WORLD_SIZE // 2, const.WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    total_biomass_cod = const.STARTING_BIOMASS_COD
    total_biomass_anchovy = const.STARTING_BIOMASS_ANCHOVY
    total_biomass_plankton = const.STARTING_BIOMASS_PLANKTON

    noise_sum_cod = 0
    noise_sum_anchovy = 0
    noise_sum_plankton = 0

    initial_energy = const.MAX_ENERGY

    # Iterate over the world grid and initialize cells directly into the tensor
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            distance_factor = distance / max_distance
            noise_value = opensimplex.noise2(x * 0.15, y * 0.15)
            threshold = 0.6 * distance_factor + 0.1 * noise_value

            # Determine if the cell is water or land
            terrain = Terrain.WATER if threshold < 0.5 else Terrain.LAND
            # terrain = Terrain.WATER

            # Set terrain one-hot encoding
            terrain_encoding = [0, 1, 0] if terrain == Terrain.WATER else [1, 0, 0]
            world_tensor[x, y, :3] = torch.tensor(terrain_encoding, device=device)

            if terrain == Terrain.WATER:
                # Calculate noise values for species clustering
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                # Apply thresholds to zero-out biomass in certain regions
                if noise_cod < 0.4: noise_cod = 0
                if noise_anchovy < 0.3: noise_anchovy = 0
                if noise_plankton < 0.35: noise_plankton = 0

                # Scale noise to create steeper clusters
                noise_cod = noise_cod ** NOISE_SCALING
                noise_anchovy = noise_anchovy ** NOISE_SCALING
                noise_plankton = noise_plankton ** NOISE_SCALING

                # Sum noise for normalization
                noise_sum_cod += noise_cod
                noise_sum_anchovy += noise_anchovy
                noise_sum_plankton += noise_plankton

    # Second pass: distribute biomass across water cells based on noise values
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_tensor[x, y, Terrain.WATER.value] == 1:
                # Recalculate noise values for biomass
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                if noise_cod < 0.4: noise_cod = 0
                if noise_anchovy < 0.3: noise_anchovy = 0
                if noise_plankton < 0.35: noise_plankton = 0

                noise_cod = noise_cod ** NOISE_SCALING
                noise_anchovy = noise_anchovy ** NOISE_SCALING
                noise_plankton = noise_plankton ** NOISE_SCALING

                # Distribute biomass proportionally to the noise sums
                if noise_sum_cod > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_COD] = (noise_cod / noise_sum_cod) * total_biomass_cod
                    world_tensor[x, y, const.OFFSETS_ENERGY_COD] = initial_energy
                if noise_sum_anchovy > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY] = (noise_anchovy / noise_sum_anchovy) * total_biomass_anchovy
                    world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY] = initial_energy
                if noise_sum_plankton > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_PLANKTON] = (noise_plankton / noise_sum_plankton) * total_biomass_plankton
                    world_tensor[x, y, const.OFFSETS_ENERGY_PLANKTON] = initial_energy

    return world_tensor


def create_static_world():
    NOISE_SCALING = 4.5
    opensimplex.seed(1)

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 6) -> 6 for terrain (3 one-hot) + biomass (3 species) + energy (3 species)
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 9, device=device)

    center_x, center_y = const.WORLD_SIZE // 2, const.WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    total_biomass_cod = const.STARTING_BIOMASS_COD
    total_biomass_anchovy = const.STARTING_BIOMASS_ANCHOVY
    total_biomass_plankton = const.STARTING_BIOMASS_PLANKTON

    noise_sum_cod = 0
    noise_sum_anchovy = 0
    noise_sum_plankton = 0

    initial_energy = const.MAX_ENERGY * 0.5

    # Iterate over the world grid and initialize cells directly into the tensor
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            distance_factor = distance / max_distance
            noise_value = opensimplex.noise2(x * 0.15, y * 0.15)
            threshold = 0.6 * distance_factor + 0.1 * noise_value

            # Determine if the cell is water or land
            terrain = Terrain.WATER if threshold < 0.5 else Terrain.LAND
            # terrain = Terrain.WATER

            # Set terrain one-hot encoding
            terrain_encoding = [0, 1, 0] if terrain == Terrain.WATER else [1, 0, 0]
            world_tensor[x, y, :3] = torch.tensor(terrain_encoding, device=device)

            if terrain == Terrain.WATER:
                # Calculate noise values for species clustering
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                # Apply thresholds to zero-out biomass in certain regions
                if noise_cod < 0.4: noise_cod = 0
                if noise_anchovy < 0.3: noise_anchovy = 0
                if noise_plankton < 0.35: noise_plankton = 0

                # Scale noise to create steeper clusters
                noise_cod = noise_cod ** NOISE_SCALING
                # noise_anchovy = noise_anchovy ** NOISE_SCALING
                noise_plankton = noise_plankton ** NOISE_SCALING

                # Sum noise for normalization
                noise_sum_cod += noise_cod
                # noise_sum_anchovy += noise_anchovy
                noise_sum_plankton += noise_plankton

    # Second pass: distribute biomass across water cells based on noise values
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_tensor[x, y, Terrain.WATER.value] == 1:
                # Recalculate noise values for biomass
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                if noise_cod < 0.4: noise_cod = 0
                if noise_anchovy < 0.3: noise_anchovy = 0
                if noise_plankton < 0.35: noise_plankton = 0

                noise_cod = noise_cod ** NOISE_SCALING
                noise_anchovy = noise_anchovy ** NOISE_SCALING
                noise_plankton = noise_plankton ** NOISE_SCALING

                # Distribute biomass proportionally to the noise sums
                if noise_sum_cod > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_COD] = (noise_cod / noise_sum_cod) * total_biomass_cod
                    world_tensor[x, y, const.OFFSETS_ENERGY_COD] = initial_energy
                if noise_sum_anchovy > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY] = (noise_anchovy / noise_sum_anchovy) * total_biomass_anchovy
                    world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY] = initial_energy
                if noise_sum_plankton > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_PLANKTON] = (noise_plankton / noise_sum_plankton) * total_biomass_plankton
                    world_tensor[x, y, const.OFFSETS_ENERGY_PLANKTON] = initial_energy

    # put all anchovy in right center of the world
    world_tensor[const.WORLD_SIZE - 1, const.WORLD_SIZE // 2, const.OFFSETS_BIOMASS_ANCHOVY] = total_biomass_anchovy
    world_tensor[const.WORLD_SIZE - 1, const.WORLD_SIZE // 2, const.OFFSETS_ENERGY_ANCHOVY] = initial_energy

    return world_tensor

def world_is_alive(world_tensor):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, 6) where:
    - The first 3 values represent the terrain (one-hot encoded).
    - The last 3 values represent the biomass of plankton, anchovy, and cod, respectively.
    """
    # Extract biomass for cod, anchovy, and plankton
    cod_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_COD].sum()  # Biomass for cod
    anchovy_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_ANCHOVY].sum()  # Biomass for anchovy
    plankton_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON].sum()  # Biomass for plankton

    # print(f"Total biomass: Cod={cod_biomass}, Anchovy={anchovy_biomass}, Plankton={plankton_biomass}")
    # Check if the total biomass for any species is below the survival threshold
    if cod_biomass < (const.STARTING_BIOMASS_COD * const.MIN_PERCENT_ALIVE):
        return False
    if anchovy_biomass < (const.STARTING_BIOMASS_ANCHOVY * const.MIN_PERCENT_ALIVE):
        return False
    if plankton_biomass < (const.STARTING_BIOMASS_PLANKTON * const.MIN_PERCENT_ALIVE):
        return False
    

    return True