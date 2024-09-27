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

ENERGY_FROM_BIOMASS = 0.1
MAX_ENERGY = 100.0

def noop(_, __):
    pass

def perform_action(world_tensor, action_values_batch, species_batch, positions_tensor, debug_visualize=noop):
    """
    Perform batched actions on the world tensor, allowing for continuous action values and movement.
    Prevents out-of-bounds movements and only moves biomass to valid water cells.
    Applies a constant BASE_BIOMASS_LOSS to each cell that takes an action.
    """
    # Unpack positions (x and y coordinates)
    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]
    # print(f"x_batch: {x_batch}, y_batch: {y_batch}")

    # Extract the movement and eating actions from action values
    move_up = action_values_batch[:, Action.UP.value]    # % to move up
    move_down = action_values_batch[:, Action.DOWN.value]  # % to move down
    move_left = action_values_batch[:, Action.LEFT.value]  # % to move left
    move_right = action_values_batch[:, Action.RIGHT.value] # % to move right
    eat = action_values_batch[:, Action.EAT.value]       # % to eat
    # print(f"eat: {eat}, move_up: {move_up}, move_down: {move_down}, move_left: {move_left}, move_right: {move_right}")

    # Get current biomass of the species at each position
    current_biomass = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_batch]
    current_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_batch]

    # Movement: Calculate how much biomass should move to each direction
    biomass_up = current_biomass * move_up
    biomass_down = current_biomass * move_down
    biomass_left = current_biomass * move_left
    biomass_right = current_biomass * move_right

    # Energy movement: Calculate how much energy should be lost for each movement
    energy_loss_move_up = (current_energy * move_up) - const.BASE_ENERGY_COST * move_up 
    energy_loss_move_down = (current_energy * move_down) - const.BASE_ENERGY_COST * move_down
    energy_loss_move_left = (current_energy * move_left) - const.BASE_ENERGY_COST * move_left
    energy_loss_move_right = (current_energy * move_right) - const.BASE_ENERGY_COST * move_right

    # apply base energy loss to all cells
    base_energy_loss = const.BASE_ENERGY_COST * torch.ones_like(current_energy)

    MOVEMENT_ENERGY_THRESHOLD = 5.0

    # Valid movement masks
    valid_up_mask = (x_batch > 0) & (world_tensor[x_batch - 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1) & (current_energy >= MOVEMENT_ENERGY_THRESHOLD)
    valid_down_mask = (x_batch < const.WORLD_SIZE - 1) & (world_tensor[x_batch + 1, y_batch, const.OFFSETS_TERRAIN_WATER] == 1) & (current_energy >= MOVEMENT_ENERGY_THRESHOLD)
    valid_left_mask = (y_batch > 0) & (world_tensor[x_batch, y_batch - 1, const.OFFSETS_TERRAIN_WATER] == 1) & (current_energy >= MOVEMENT_ENERGY_THRESHOLD)
    valid_right_mask = (y_batch < const.WORLD_SIZE - 1) & (world_tensor[x_batch, y_batch + 1, const.OFFSETS_TERRAIN_WATER] == 1) & (current_energy >= MOVEMENT_ENERGY_THRESHOLD)

    # Subtract biomass only for valid movements
    total_valid_move = torch.zeros_like(current_biomass)

    total_energy_loss = base_energy_loss

    # Upward movement (only move if target is within bounds and water)
    if valid_up_mask.any():
        total_valid_move[valid_up_mask] += biomass_up[valid_up_mask]
        total_energy_loss[valid_up_mask] += energy_loss_move_up[valid_up_mask]
        world_tensor[x_batch[valid_up_mask] - 1, y_batch[valid_up_mask], species_batch[valid_up_mask] + const.OFFSETS_BIOMASS] += biomass_up[valid_up_mask]

    # Downward movement (only move if target is within bounds and water)
    if valid_down_mask.any():
        total_valid_move[valid_down_mask] += biomass_down[valid_down_mask]
        total_energy_loss[valid_down_mask] += energy_loss_move_down[valid_down_mask]
        world_tensor[x_batch[valid_down_mask] + 1, y_batch[valid_down_mask], species_batch[valid_down_mask] + const.OFFSETS_BIOMASS] += biomass_down[valid_down_mask]

    # Leftward movement (only move if target is within bounds and water)
    if valid_left_mask.any():
        total_valid_move[valid_left_mask] += biomass_left[valid_left_mask]
        total_energy_loss[valid_left_mask] += energy_loss_move_left[valid_left_mask]
        world_tensor[x_batch[valid_left_mask], y_batch[valid_left_mask] - 1, species_batch[valid_left_mask] + const.OFFSETS_BIOMASS] += biomass_left[valid_left_mask]

    # Rightward movement (only move if target is within bounds and water)
    if valid_right_mask.any():
        total_valid_move[valid_right_mask] += biomass_right[valid_right_mask]
        total_energy_loss[valid_right_mask] += energy_loss_move_right[valid_right_mask]
        world_tensor[x_batch[valid_right_mask], y_batch[valid_right_mask] + 1, species_batch[valid_right_mask] + const.OFFSETS_BIOMASS] += biomass_right[valid_right_mask]

    did_not_move_mask = ~(valid_up_mask | valid_down_mask | valid_left_mask | valid_right_mask)
    did_not_move = torch.zeros_like(current_energy)
    did_not_move[did_not_move_mask] = current_energy[did_not_move_mask]

    total_energy_loss[did_not_move_mask] += did_not_move[did_not_move_mask] - const.BASE_ENERGY_COST

    # print(f"total_valid_move: {total_valid_move}, total_energy_loss: {total_energy_loss}")

    # Subtract the total valid movement from the current cell
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_batch] -= total_valid_move

    # Subtract the energy loss from the current cell
    world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_batch] -= total_energy_loss

    # Apply BASE_BIOMASS_LOSS to all action-taking cells (a constant percentage loss)
    biomass_loss = current_biomass * const.BASE_BIOMASS_LOSS
    # only subtract if the energy is below 50%
    energy_at_positions = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY + species_batch]
    # Create a mask for cells where energy is below 50
    energy_below_50_mask = energy_at_positions < 50
    
    world_tensor[x_batch[energy_below_50_mask], 
             y_batch[energy_below_50_mask], 
             const.OFFSETS_BIOMASS + species_batch[energy_below_50_mask]] -= biomass_loss[energy_below_50_mask]

    # Eating Logic
    eat_amounts = current_biomass * eat * const.EAT_AMOUNT

    for i, species in enumerate(species_batch):
        if species == Species.COD.value:
            # COD eats ANCHOVY in the current position
            prey_biomass = world_tensor[x_batch[i], y_batch[i], const.OFFSETS_BIOMASS_ANCHOVY]
            eat_amount = torch.min(prey_biomass, eat_amounts[i])  # Fix: ensure that this gets scalar
            world_tensor[x_batch[i], y_batch[i], const.OFFSETS_BIOMASS_ANCHOVY] -= eat_amount  # Reduce anchovy biomass
            world_tensor[x_batch[i], y_batch[i], const.OFFSETS_BIOMASS_COD] += eat_amount  # Add cod biomass
            # apply cost of eating + gained energy from eating
            energy_gain = eat_amount * ENERGY_FROM_BIOMASS

            # Apply the cost of eating + energy gained from eating, make sure the values are scalars
            new_energy = world_tensor[x_batch[i], y_batch[i], const.OFFSETS_ENERGY_COD] + energy_gain
            world_tensor[x_batch[i], y_batch[i], const.OFFSETS_ENERGY_COD] = torch.clamp(new_energy, max=MAX_ENERGY)


        elif species == Species.ANCHOVY.value:
            # ANCHOVY eats PLANKTON in the current position
            prey_biomass = world_tensor[x_batch[i], y_batch[i], const.OFFSETS_BIOMASS_PLANKTON]
            eat_amount = torch.min(prey_biomass, eat_amounts[i])  # Fix: ensure this gets scalar
            world_tensor[x_batch[i], y_batch[i], const.OFFSETS_BIOMASS_PLANKTON] -= eat_amount  # Reduce plankton biomass
            world_tensor[x_batch[i], y_batch[i], const.OFFSETS_BIOMASS_ANCHOVY] += eat_amount  # Add anchovy biomass
            # apply cost of eating + gained energy from eating
            energy_gain = eat_amount * ENERGY_FROM_BIOMASS
            new_energy = world_tensor[x_batch[i], y_batch[i], const.OFFSETS_ENERGY_ANCHOVY] + energy_gain
            world_tensor[x_batch[i], y_batch[i], const.OFFSETS_ENERGY_ANCHOVY] = torch.clamp(new_energy, max=MAX_ENERGY)

    return world_tensor

def create_world():
    NOISE_SCALING = 4.5
    opensimplex.seed(int(random() * 100000))

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 6) -> 6 for terrain (3 one-hot) + biomass (3 species)
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 9, device=device)

    center_x, center_y = const.WORLD_SIZE // 2, const.WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    total_biomass_cod = const.STARTING_BIOMASS_COD
    total_biomass_anchovy = const.STARTING_BIOMASS_ANCHOVY
    total_biomass_plankton = const.STARTING_BIOMASS_PLANKTON

    noise_sum_cod = 0
    noise_sum_anchovy = 0
    noise_sum_plankton = 0

    initial_energy = MAX_ENERGY

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

    print(f"Total biomass: Cod={cod_biomass}, Anchovy={anchovy_biomass}, Plankton={plankton_biomass}")
    # Check if the total biomass for any species is below the survival threshold
    if cod_biomass < (const.STARTING_BIOMASS_COD * const.MIN_PERCENT_ALIVE):
        return False
    if anchovy_biomass < (const.STARTING_BIOMASS_ANCHOVY * const.MIN_PERCENT_ALIVE):
        return False
    if plankton_biomass < (const.STARTING_BIOMASS_PLANKTON * const.MIN_PERCENT_ALIVE):
        return False
    

    return True