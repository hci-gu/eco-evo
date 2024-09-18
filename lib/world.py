from enum import Enum
import opensimplex
import math
from random import random
import torch.nn as nn
import torch
from lib.constants import BASE_BIOMASS_LOSS, PLANKTON_GROWTH_RATE, MAX_PLANKTON_IN_CELL, WORLD_SIZE, STARTING_BIOMASS_ANCHOVY, STARTING_BIOMASS_COD, STARTING_BIOMASS_PLANKTON, MIN_PERCENT_ALIVE, EAT_AMOUNT

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


def perform_action(world_tensor, action_values_batch, species_batch, positions_tensor):
    """
    Perform batched actions on the world tensor, allowing for continuous action values and movement.
    Prevents out-of-bounds movements and only moves biomass to valid water cells.
    Applies a constant BASE_BIOMASS_LOSS to each cell that takes an action.
    """
    # Unpack positions (x and y coordinates)
    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]

    # Extract the movement and eating actions from action values
    move_up = action_values_batch[:, Action.UP.value]    # % to move up
    move_down = action_values_batch[:, Action.DOWN.value]  # % to move down
    move_left = action_values_batch[:, Action.LEFT.value]  # % to move left
    move_right = action_values_batch[:, Action.RIGHT.value] # % to move right
    eat = action_values_batch[:, Action.EAT.value]       # % to eat

    # Get current biomass of the species at each position
    current_biomass = world_tensor[x_batch, y_batch, species_batch + 3]

    # Movement: Calculate how much biomass should move to each direction
    biomass_up = current_biomass * move_up
    biomass_down = current_biomass * move_down
    biomass_left = current_biomass * move_left
    biomass_right = current_biomass * move_right

    # Valid movement masks
    valid_up_mask = (x_batch > 0) & (world_tensor[x_batch - 1, y_batch, Terrain.WATER.value] == 1)
    valid_down_mask = (x_batch < WORLD_SIZE - 1) & (world_tensor[x_batch + 1, y_batch, Terrain.WATER.value] == 1)
    valid_left_mask = (y_batch > 0) & (world_tensor[x_batch, y_batch - 1, Terrain.WATER.value] == 1)
    valid_right_mask = (y_batch < WORLD_SIZE - 1) & (world_tensor[x_batch, y_batch + 1, Terrain.WATER.value] == 1)

    # Subtract biomass only for valid movements
    total_valid_move = torch.zeros_like(current_biomass)

    # Upward movement (only move if target is within bounds and water)
    if valid_up_mask.any():
        total_valid_move[valid_up_mask] += biomass_up[valid_up_mask]
        world_tensor[x_batch[valid_up_mask] - 1, y_batch[valid_up_mask], species_batch[valid_up_mask] + 3] += biomass_up[valid_up_mask]

    # Downward movement (only move if target is within bounds and water)
    if valid_down_mask.any():
        total_valid_move[valid_down_mask] += biomass_down[valid_down_mask]
        world_tensor[x_batch[valid_down_mask] + 1, y_batch[valid_down_mask], species_batch[valid_down_mask] + 3] += biomass_down[valid_down_mask]

    # Leftward movement (only move if target is within bounds and water)
    if valid_left_mask.any():
        total_valid_move[valid_left_mask] += biomass_left[valid_left_mask]
        world_tensor[x_batch[valid_left_mask], y_batch[valid_left_mask] - 1, species_batch[valid_left_mask] + 3] += biomass_left[valid_left_mask]

    # Rightward movement (only move if target is within bounds and water)
    if valid_right_mask.any():
        total_valid_move[valid_right_mask] += biomass_right[valid_right_mask]
        world_tensor[x_batch[valid_right_mask], y_batch[valid_right_mask] + 1, species_batch[valid_right_mask] + 3] += biomass_right[valid_right_mask]

    # Subtract the total valid movement from the current cell
    world_tensor[x_batch, y_batch, species_batch + 3] -= total_valid_move

    # Apply BASE_BIOMASS_LOSS to all action-taking cells (a constant percentage loss)
    biomass_loss = current_biomass * BASE_BIOMASS_LOSS
    world_tensor[x_batch, y_batch, species_batch + 3] -= biomass_loss

    # Eating Logic
    eat_amounts = current_biomass * eat * EAT_AMOUNT

    for i, species in enumerate(species_batch):
        if species == Species.COD.value:
            # COD eats ANCHOVY in the current position
            prey_biomass = world_tensor[x_batch[i], y_batch[i], Species.ANCHOVY.value + 3]
            eat_amount = torch.min(prey_biomass, eat_amounts[i])  # Fix: ensure that this gets scalar
            world_tensor[x_batch[i], y_batch[i], Species.ANCHOVY.value + 3] -= eat_amount  # Reduce anchovy biomass
            world_tensor[x_batch[i], y_batch[i], Species.COD.value + 3] += eat_amount  # Add cod biomass

        elif species == Species.ANCHOVY.value:
            # ANCHOVY eats PLANKTON in the current position
            prey_biomass = world_tensor[x_batch[i], y_batch[i], Species.PLANKTON.value + 3]
            eat_amount = torch.min(prey_biomass, eat_amounts[i])  # Fix: ensure this gets scalar
            world_tensor[x_batch[i], y_batch[i], Species.PLANKTON.value + 3] -= eat_amount  # Reduce plankton biomass
            world_tensor[x_batch[i], y_batch[i], Species.ANCHOVY.value + 3] += eat_amount  # Add anchovy biomass

    return world_tensor



def create_world():
    NOISE_SCALING = 4.5
    opensimplex.seed(int(random() * 100000))

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 6) -> 6 for terrain (3 one-hot) + biomass (3 species)
    world_tensor = torch.zeros(WORLD_SIZE, WORLD_SIZE, 6, device=device)

    center_x, center_y = WORLD_SIZE // 2, WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    total_biomass_cod = STARTING_BIOMASS_COD
    total_biomass_anchovy = STARTING_BIOMASS_ANCHOVY
    total_biomass_plankton = STARTING_BIOMASS_PLANKTON

    noise_sum_cod = 0
    noise_sum_anchovy = 0
    noise_sum_plankton = 0

    # Iterate over the world grid and initialize cells directly into the tensor
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
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
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
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
                    world_tensor[x, y, 5] = (noise_cod / noise_sum_cod) * total_biomass_cod
                if noise_sum_anchovy > 0:
                    world_tensor[x, y, 4] = (noise_anchovy / noise_sum_anchovy) * total_biomass_anchovy
                if noise_sum_plankton > 0:
                    world_tensor[x, y, 3] = (noise_plankton / noise_sum_plankton) * total_biomass_plankton

                # world_tensor[x, y, 5] = 0
                # world_tensor[x, y, 4] = total_biomass_anchovy
                # world_tensor[x, y, 3] = total_biomass_plankton

    return world_tensor


def world_is_alive(world_tensor):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, 6) where:
    - The first 3 values represent the terrain (one-hot encoded).
    - The last 3 values represent the biomass of plankton, anchovy, and cod, respectively.
    """
    # Extract biomass for cod, anchovy, and plankton
    cod_biomass = world_tensor[:, :, 5].sum()  # Biomass for cod
    anchovy_biomass = world_tensor[:, :, 4].sum()  # Biomass for anchovy
    plankton_biomass = world_tensor[:, :, 3].sum()  # Biomass for plankton

    print(f"Total biomass: Cod={cod_biomass}, Anchovy={anchovy_biomass}, Plankton={plankton_biomass}")
    # Check if the total biomass for any species is below the survival threshold
    if cod_biomass < (STARTING_BIOMASS_COD * MIN_PERCENT_ALIVE):
        return False
    if anchovy_biomass < (STARTING_BIOMASS_ANCHOVY * MIN_PERCENT_ALIVE):
        return False
    if plankton_biomass < (STARTING_BIOMASS_PLANKTON * MIN_PERCENT_ALIVE):
        return False
    

    return True