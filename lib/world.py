from enum import Enum
from noise import pnoise2  # Perlin noise function
import opensimplex
import math
import random
import torch.nn as nn
import numpy as np
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

def reset_plankton_cluster():
    """
    Resets the plankton cluster by deleting its stored position and movement attributes.
    """
    attributes = ['position', 'dx', 'dy', 'radius']
    for attr in attributes:
        if hasattr(move_plankton_cluster, attr):
            delattr(move_plankton_cluster, attr)



def move_plankton_cluster(world_tensor):
    """
    Moves a circular cluster of plankton around the world.
    The cluster moves with constrained random movement, changing direction when it hits the edge.
    """
    # Remove all existing plankton biomass
    world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON] = 0

    # Get the actual size of the world from the tensor
    world_size_x, world_size_y = world_tensor.shape[:2]

    # Initialize cluster properties if they don't exist
    if not hasattr(move_plankton_cluster, 'position'):
        # Start at a random position within the world bounds, considering the radius
        radius = 4
        move_plankton_cluster.position = [
            world_size_x - radius,
            world_size_y / 2
        ]
        # Random initial movement direction
        speed = 1  # Adjust speed as needed
        angle = random.uniform(0, 2 * math.pi)
        dx = speed * math.cos(angle)
        dy = speed * math.sin(angle)
        move_plankton_cluster.dx = dx
        move_plankton_cluster.dy = dy
        move_plankton_cluster.radius = radius

    # Unpack position and movement components
    x_pos, y_pos = move_plankton_cluster.position
    dx = move_plankton_cluster.dx
    dy = move_plankton_cluster.dy
    radius = move_plankton_cluster.radius

    # Introduce slight random variations to dx and dy
    random_factor = 0.1  # Adjust as needed for randomness intensity
    dx += random.uniform(-random_factor, random_factor)
    dy += random.uniform(-random_factor, random_factor)

    # Normalize the movement vector to maintain constant speed
    speed = 1  # Adjust speed as needed
    vector_length = math.sqrt(dx**2 + dy**2)
    if vector_length != 0:
        dx = (dx / vector_length) * speed
        dy = (dy / vector_length) * speed
    else:
        # If vector_length is zero, choose a new random direction
        angle = random.uniform(0, 2 * math.pi)
        dx = speed * math.cos(angle)
        dy = speed * math.sin(angle)

    # Tentatively move the cluster position
    new_x_pos = x_pos + dx
    new_y_pos = y_pos + dy

    # Check for collisions with the world edges and adjust position and direction
    hit_edge = False
    if new_x_pos - radius < 0 or new_x_pos + radius >= world_size_x:
        # Reflect dx
        dx = -dx
        new_x_pos = x_pos + dx
        hit_edge = True
    if new_y_pos - radius < 0 or new_y_pos + radius >= world_size_y:
        # Reflect dy
        dy = -dy
        new_y_pos = y_pos + dy
        hit_edge = True

    # Update movement components after adjusting for edges
    move_plankton_cluster.dx = dx
    move_plankton_cluster.dy = dy

    # Ensure the new position keeps the cluster within bounds
    new_x_pos = max(radius, min(new_x_pos, world_size_x - radius))
    new_y_pos = max(radius, min(new_y_pos, world_size_y - radius))

    # Update position
    move_plankton_cluster.position = [new_x_pos, new_y_pos]

    # Generate the circular cluster at the new position
    xx, yy = torch.meshgrid(
        torch.arange(world_size_x, device=world_tensor.device),
        torch.arange(world_size_y, device=world_tensor.device),
        indexing='ij'  # Specify indexing to avoid warnings
    )
    # Calculate distance from the center of the cluster
    distances = torch.sqrt((xx - new_x_pos) ** 2 + (yy - new_y_pos) ** 2)
    # Create a mask for cells within the radius
    cluster_mask = distances <= radius

    # Ensure plankton is only placed in water cells
    water_mask = world_tensor[:, :, Terrain.WATER.value] == 1
    valid_mask = cluster_mask & water_mask

    # Check if there are valid water cells; if not, adjust position
    num_valid_cells = valid_mask.sum().item()
    if num_valid_cells == 0:
        # Try moving the cluster back to the previous position
        new_x_pos = x_pos
        new_y_pos = y_pos
        move_plankton_cluster.position = [new_x_pos, new_y_pos]
        # Recalculate distances and valid_mask
        distances = torch.sqrt((xx - new_x_pos) ** 2 + (yy - new_y_pos) ** 2)
        cluster_mask = distances <= radius
        valid_mask = cluster_mask & water_mask
        num_valid_cells = valid_mask.sum().item()
        if num_valid_cells == 0:
            # If still no valid cells, reset the cluster
            reset_plankton_cluster()
            print("Plankton cluster reset due to no valid water cells.")
            return world_tensor

    # Get indices of valid cells
    cluster_indices = torch.nonzero(valid_mask)

    # Distribute total biomass equally among the cells
    total_biomass_plankton = const.STARTING_BIOMASS_PLANKTON
    biomass_per_cell = total_biomass_plankton / num_valid_cells

    # Assign biomass to the cells
    world_tensor[cluster_indices[:, 0], cluster_indices[:, 1], const.OFFSETS_BIOMASS_PLANKTON] = biomass_per_cell
    return world_tensor

def respawn_plankton(world_tensor):
    """
    Removes all plankton from the world and creates a cluster at the next side in a clockwise order.
    """
    # Sides in clockwise order
    sides = ['east', 'south', 'west', 'north', ]

    # Initialize the current side index if it doesn't exist
    if not hasattr(respawn_plankton, 'current_side_index'):
        respawn_plankton.current_side_index = 0

    # Select the current side
    side = sides[respawn_plankton.current_side_index]

    # Update the index for the next call (cycle through the sides)
    respawn_plankton.current_side_index = (respawn_plankton.current_side_index + 1) % len(sides)

    # Remove all current plankton biomass
    world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON] = 0

    # Define cluster size (number of rows/columns from the edge)
    cluster_size = math.floor(const.WORLD_SIZE / 10) + 1  # Adjust as needed

    world_size = const.WORLD_SIZE

    # Determine the cluster area based on the side
    if side == 'north':
        x_start, x_end = 0, cluster_size
        y_start, y_end = 0, world_size
    elif side == 'south':
        x_start, x_end = world_size - cluster_size, world_size
        y_start, y_end = 0, world_size
    elif side == 'west':
        x_start, x_end = 0, world_size
        y_start, y_end = 0, cluster_size
    elif side == 'east':
        x_start, x_end = 0, world_size
        y_start, y_end = world_size - cluster_size, world_size

    # Get the mask of water cells in the cluster area
    water_layer = world_tensor[:, :, Terrain.WATER.value]
    cluster_mask = torch.zeros_like(water_layer, dtype=torch.bool)
    cluster_mask[x_start:x_end, y_start:y_end] = True
    water_mask = cluster_mask & (water_layer == 1)

    # Get indices of water cells in the cluster area
    cluster_indices = torch.nonzero(water_mask)

    num_cells = cluster_indices.size(0)
    if num_cells == 0:
        # If no water cells in the cluster area, then we can't place plankton
        print(f"No water cells found on the {side} side for plankton respawn.")
        return world_tensor

    # Distribute total biomass equally among the cells
    total_biomass_plankton = const.STARTING_BIOMASS_PLANKTON
    biomass_per_cell = total_biomass_plankton / num_cells

    # Assign biomass to the cells
    world_tensor[cluster_indices[:, 0], cluster_indices[:, 1], const.OFFSETS_BIOMASS_PLANKTON] = biomass_per_cell

    print(f"Plankton respawned at the {side} side of the world.")
    return world_tensor

def move_plankton_based_on_current(world_tensor, world_data):
    # Get the size of the world grid
    world_size_x, world_size_y = world_tensor.shape[0], world_tensor.shape[1]

    # Extract the plankton biomass layer
    plankton_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON]

    # Extract the current angles
    current_angles = world_data[:, :, 0]

    # Initialize a buffer to accumulate new biomass positions
    new_biomass = torch.zeros_like(plankton_biomass)

    # Get indices of all cells (ensure they are of type torch.long)
    x_indices = torch.arange(world_size_x, dtype=torch.long, device=world_tensor.device).view(-1, 1).expand(-1, world_size_y)
    y_indices = torch.arange(world_size_y, dtype=torch.long, device=world_tensor.device).view(1, -1).expand(world_size_x, -1)

    # Flatten the arrays for vectorized operations
    x_indices_flat = x_indices.flatten()
    y_indices_flat = y_indices.flatten()
    plankton_biomass_flat = plankton_biomass.flatten()
    current_angles_flat = current_angles.flatten()

    # Filter cells that have plankton biomass greater than zero
    mask = plankton_biomass_flat > 0
    x_indices_flat = x_indices_flat[mask]
    y_indices_flat = y_indices_flat[mask]
    plankton_biomass_flat = plankton_biomass_flat[mask]
    current_angles_flat = current_angles_flat[mask]

    # Define possible movement directions (dx, dy) and their corresponding angles
    movement_directions = [
        (1, 0),    # East
        (1, 1),    # Southeast
        (0, 1),    # South
        (-1, 1),   # Southwest
        (-1, 0),   # West
        (-1, -1),  # Northwest
        (0, -1),   # North
        (1, -1),   # Northeast
    ]
    movement_angles = torch.tensor([
        0,                # East (0 degrees)
        math.pi / 4,      # Southeast (45 degrees)
        math.pi / 2,      # South (90 degrees)
        3 * math.pi / 4,  # Southwest (135 degrees)
        math.pi,          # West (180 degrees)
        -3 * math.pi / 4, # Northwest (-135 degrees)
        -math.pi / 2,     # North (-90 degrees)
        -math.pi / 4,     # Northeast (-45 degrees)
    ], device=world_tensor.device)

    # For each plankton cell, attempt to move in the intended direction or the closest valid alternative
    for i in range(len(x_indices_flat)):
        x = x_indices_flat[i]
        y = y_indices_flat[i]
        biomass = plankton_biomass_flat[i]
        angle = current_angles_flat[i]

        # Calculate the difference between the current angle and the movement angles
        angle_diffs = torch.remainder(movement_angles - angle + math.pi, 2 * math.pi) - math.pi
        angle_diffs = torch.abs(angle_diffs)

        # Sort movement directions based on the smallest angle difference (closest to intended direction)
        sorted_indices = torch.argsort(angle_diffs)

        moved = False
        for idx in sorted_indices:
            dx, dy = movement_directions[idx]
            new_x = x + dx
            new_y = y + dy

            # Check if new position is within bounds
            if 0 <= new_x < world_size_x and 0 <= new_y < world_size_y:
                # Check if the terrain at the new position is water
                if world_tensor[new_x, new_y, const.OFFSETS_TERRAIN_WATER] == 1:
                    # Move the biomass to the new position
                    new_biomass[new_x, new_y] += biomass
                    moved = True
                    break  # Exit the loop after successful movement

        if not moved:
            # If no valid movement was found, keep the biomass in the original position
            new_biomass[x, y] += biomass

    # Update the plankton biomass layer in world_tensor
    world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON] = new_biomass

    return world_tensor

def spawn_plankton(world_tensor, world_data):
    plankton_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON]
    plankton_flag = world_data[:, :, 1]  # Cells marked with 1 are initial plankton locations
    plankton_counter = world_data[:, :, 2]  # Plankton respawn counter


    # Create a mask for the cells that are initial plankton locations
    plankton_cells = plankton_flag == 1

    # For cells where there is existing plankton biomass, increase it
    existing_plankton_mask = (plankton_biomass > 0) & plankton_cells

    # For cells where there is no plankton biomass, but the plankton flag is set, add base amount
    empty_plankton_mask = (plankton_biomass == 0) & plankton_cells

    # Increase biomass in existing plankton cells
    if existing_plankton_mask.any():
        increased_biomass = plankton_biomass[existing_plankton_mask] * (1 + const.PLANKTON_GROWTH_RATE)
        world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON][existing_plankton_mask] = torch.clamp(
            increased_biomass, max=const.MAX_PLANKTON_IN_CELL
        )
        world_data[:, :, 2][existing_plankton_mask] = const.PLANKTON_RESPAWN_DELAY

    # Add base biomass to empty plankton cells
    if empty_plankton_mask.any():
        # Decrease the counter by 1
        plankton_counter[empty_plankton_mask] -= 1
        # Ensure the counter doesn't go below zero
        world_data[:, :, 2][empty_plankton_mask] = torch.clamp(plankton_counter[empty_plankton_mask], min=0)
        # Check which counters have reached zero
        respawn_mask = (world_data[:, :, 2] == 0) & empty_plankton_mask
        if respawn_mask.any():
            # Reset the counter to 100 and spawn plankton
            world_data[:, :, 2][respawn_mask] = const.PLANKTON_RESPAWN_DELAY
            world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON][respawn_mask] = const.BASE_PLANKTON_SPAWN_RATE

def perform_action_optimized(world_tensor, action_values_batch, species_index, positions_tensor):
    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]

    # Extract action values
    actions = {
        Action.UP: action_values_batch[:, Action.UP.value],
        Action.DOWN: action_values_batch[:, Action.DOWN.value],
        Action.LEFT: action_values_batch[:, Action.LEFT.value],
        Action.RIGHT: action_values_batch[:, Action.RIGHT.value],
        Action.EAT: action_values_batch[:, Action.EAT.value]
    }

    biomass_offset = const.OFFSETS_BIOMASS + species_index
    energy_offset = const.OFFSETS_ENERGY + species_index

    # Initial biomass and energy
    initial_biomass = world_tensor[x_batch, y_batch, biomass_offset]
    initial_energy = world_tensor[x_batch, y_batch, energy_offset]

    # Apply base energy cost
    world_tensor[x_batch, y_batch, energy_offset] -= const.BASE_ENERGY_COST

    # Energy after base cost
    energy_after_cost = world_tensor[x_batch, y_batch, energy_offset]

    # Biomass gain or loss based on energy level
    biomass_change = torch.where(
        energy_after_cost >= 50,
        initial_biomass * const.BIOMASS_GROWTH_RATE,  # Biomass gain
        -initial_biomass * const.BASE_BIOMASS_LOSS   # Biomass loss
    )
    world_tensor[x_batch, y_batch, biomass_offset] += biomass_change

    # Recompute biomass and energy after change
    updated_biomass = world_tensor[x_batch, y_batch, biomass_offset]
    updated_energy = world_tensor[x_batch, y_batch, energy_offset]

    # Movement computations
    # Movement directions and their corresponding delta positions
    directions = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0)
    }

    total_biomass_moved = torch.zeros_like(initial_biomass)
    total_energy_moved = torch.zeros_like(initial_energy)

    for action, (dx, dy) in directions.items():
        move_mask = actions[action] > 0
        if move_mask.any():
            # Compute new positions
            new_x = x_batch[move_mask] + dx
            new_y = y_batch[move_mask] + dy

            # Check boundaries and terrain
            valid_mask = (
                (new_x >= 0) & (new_x < world_tensor.shape[0]) &
                (new_y >= 0) & (new_y < world_tensor.shape[1]) &
                (world_tensor[new_x, new_y, const.OFFSETS_TERRAIN_WATER] == 1)
            )

            if valid_mask.any():
                idx = torch.nonzero(move_mask)[valid_mask]
                move_indices = idx.squeeze()

                move_biomass = updated_biomass[move_indices] * actions[action][move_mask][valid_mask]
                move_energy = updated_energy[move_indices] * actions[action][move_mask][valid_mask] - const.BASE_ENERGY_COST

                # Update source cells
                total_biomass_moved[move_indices] += move_biomass
                total_energy_moved[move_indices] += move_energy

                # Update destination cells
                dest_x = new_x[valid_mask]
                dest_y = new_y[valid_mask]
                world_tensor[dest_x, dest_y, biomass_offset] += move_biomass
                world_tensor[dest_x, dest_y, energy_offset] += move_energy

    # Subtract total biomass moved from source cells
    world_tensor[x_batch, y_batch, biomass_offset] -= total_biomass_moved

    # Clamping biomass to non-negative values
    world_tensor[:, :, biomass_offset].clamp_(min=0)

    # Eating Logic
    if species_index in [Species.COD.value, Species.ANCHOVY.value]:
        eat_action = actions[Action.EAT]
        eat_mask = eat_action > 0
        if eat_mask.any():
            prey_species_offset = {
                Species.COD.value: const.OFFSETS_BIOMASS_ANCHOVY,
                Species.ANCHOVY.value: const.OFFSETS_BIOMASS_PLANKTON
            }[species_index]

            eat_amount = updated_biomass[eat_mask] * eat_action[eat_mask] * const.EAT_AMOUNT[species_index]
            prey_biomass = world_tensor[x_batch[eat_mask], y_batch[eat_mask], prey_species_offset]
            actual_eat_amount = torch.min(prey_biomass, eat_amount)

            # Update prey biomass
            world_tensor[x_batch[eat_mask], y_batch[eat_mask], prey_species_offset] -= actual_eat_amount

            # Update predator energy
            energy_gain = (actual_eat_amount / eat_amount) * const.ENERGY_REWARD_FOR_EATING * eat_action[eat_mask]
            world_tensor[x_batch[eat_mask], y_batch[eat_mask], energy_offset] += energy_gain
            # Clamp energy to maximum
            world_tensor[x_batch[eat_mask], y_batch[eat_mask], energy_offset].clamp_(max=const.MAX_ENERGY)

    # Set energy to zero where biomass is zero
    zero_biomass_mask = world_tensor[:, :, biomass_offset] == 0
    world_tensor[:, :, energy_offset][zero_biomass_mask] = 0

    return world_tensor

def perform_action(world_tensor, action_values_batch, species_index, positions_tensor):
    initial_tensor = world_tensor.clone()

    x_batch = positions_tensor[:, 0]
    y_batch = positions_tensor[:, 1]

    move_up = action_values_batch[:, Action.UP.value]
    move_down = action_values_batch[:, Action.DOWN.value]
    move_left = action_values_batch[:, Action.LEFT.value]
    move_right = action_values_batch[:, Action.RIGHT.value]
    eat = action_values_batch[:, Action.EAT.value]

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

        reward_scaling_factor = torch.where(eat_amounts > 0, eat_amount / eat_amounts, torch.tensor(0.0, device=eat_amount.device))
        # Clamp scaling factor between 0 and 1 to ensure it's in a valid range
        reward_scaling_factor = torch.clamp(reward_scaling_factor, 0, 1)
        energy_reward = reward_scaling_factor * const.ENERGY_REWARD_FOR_EATING * eat
        new_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_COD] + energy_reward
        world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_COD] = torch.clamp(new_energy, max=const.MAX_ENERGY)
    elif species_index == Species.ANCHOVY.value:
        eat_amounts = initial_biomass * eat * const.EAT_AMOUNT_ANCHOVY
        # ANCHOVY eats PLANKTON in the current position
        prey_biomass = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON]
        eat_amount = torch.min(prey_biomass, eat_amounts)  # Fix: ensure this gets scalar
        world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON] -= eat_amount  # Reduce plankton biomass

        reward_scaling_factor = torch.where(eat_amounts > 0, eat_amount / eat_amounts, torch.tensor(0.0, device=eat_amount.device))
        # Clamp scaling factor between 0 and 1 to ensure it's in a valid range
        reward_scaling_factor = torch.clamp(reward_scaling_factor, 0, 1)

        energy_reward = reward_scaling_factor * const.ENERGY_REWARD_FOR_EATING * eat
        new_energy = world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_ANCHOVY] + energy_reward
        world_tensor[x_batch, y_batch, const.OFFSETS_ENERGY_ANCHOVY] = torch.clamp(new_energy, max=const.MAX_ENERGY)
    
    # make sure biomass is not negative
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON] = torch.clamp(world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_PLANKTON], min=0)
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY] = torch.clamp(world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_ANCHOVY], min=0)
    world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_COD] = torch.clamp(world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS_COD], min=0)

    biomass = world_tensor[:, :, const.OFFSETS_BIOMASS + species_index]

    low_biomass_mask = biomass < const.MIN_BIOMASS_IN_CELL[species_index]
    world_tensor[:, :, const.OFFSETS_BIOMASS + species_index][low_biomass_mask] = 0

    # set all cells with 0 biomass to 0 energy
    zero_biomass_mask = world_tensor[x_batch, y_batch, const.OFFSETS_BIOMASS + species_index] == 0
    world_tensor[x_batch[zero_biomass_mask], y_batch[zero_biomass_mask], const.OFFSETS_ENERGY + species_index] = 0

    # make sure biomass in a cell cannot be more than max for the species
    if species_index == Species.ANCHOVY.value:
        # ignore x_batch, y_batch, apply to all
        world_tensor[:, :, const.OFFSETS_BIOMASS_ANCHOVY] = torch.clamp(world_tensor[:, :, const.OFFSETS_BIOMASS_ANCHOVY], max=const.MAX_ANCHOVY_IN_CELL)

    return world_tensor

def create_world(static=False):
    NOISE_SCALING = 4.5
    seed = 1 if static else int(random.random() * 100000)
    opensimplex.seed(seed)

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 9) -> 3 for terrain (3 one-hot) + biomass (3 species) + energy (3 species)
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 9, device=device)
    world_data = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 3, device=device)

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
                # Calculate the angle of the current based on the cell's position relative to the center
                # This ensures a clockwise rotation direction
                dx = x - center_x
                dy = y - center_y
                current_angle = math.atan2(dy, dx) + math.pi / 2

                # Store the current angle in the tensor
                world_data[x, y, 0] = current_angle  # Add current angle

                # Calculate noise values for species clustering
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                # Apply thresholds to zero-out biomass in certain regions
                if noise_cod < 0.8: noise_cod = 0
                if noise_anchovy < 0.6: noise_anchovy = 0
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

                # print(f"noise_cod: {noise_cod}, noise_anchovy: {noise_anchovy}, noise_plankton: {noise_plankton}")

                if noise_cod < 0.8: noise_cod = 0
                if noise_anchovy < 0.6: noise_anchovy = 0
                if noise_plankton < 0.35: 
                    noise_plankton = 0
                    world_data[x, y, 1] = 0
                    world_data[x, y, 2] = 0

                noise_cod = noise_cod ** NOISE_SCALING
                noise_anchovy = noise_anchovy ** NOISE_SCALING
                noise_plankton = noise_plankton ** NOISE_SCALING

                # Distribute biomass proportionally to the noise sums
                if noise_cod > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_COD] = (noise_cod / noise_sum_cod) * total_biomass_cod
                    world_tensor[x, y, const.OFFSETS_ENERGY_COD] = initial_energy
                if noise_anchovy > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY] = (noise_anchovy / noise_sum_anchovy) * total_biomass_anchovy
                    world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY] = initial_energy
                if noise_plankton > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_PLANKTON] = (noise_plankton / noise_sum_plankton) * total_biomass_plankton
                    world_tensor[x, y, const.OFFSETS_ENERGY_PLANKTON] = initial_energy
                    world_data[x, y, 1] = 1  # Add plankton cluster flag
                    world_data[x, y, 2] = const.PLANKTON_RESPAWN_DELAY # Add plankton respawn counter
                    
    return world_tensor, world_data


def world_is_alive(world_tensor):
    """
    Checks if the world is still "alive" by verifying the biomass of the species in the world tensor.
    The world tensor has shape (WORLD_SIZE, WORLD_SIZE, 6) where:
    - The first 3 values represent the terrain (one-hot encoded).
    - The last 3 values represent the biomass of plankton, anchovy, and cod, respectively.
    """
    cod_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_COD].sum()  # Biomass for cod
    anchovy_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_ANCHOVY].sum()  # Biomass for anchovy
    plankton_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS_PLANKTON].sum()  # Biomass for plankton

    # Check if the total biomass for any species is below the survival threshold or above the maximum threshold
    if cod_biomass < (const.STARTING_BIOMASS_COD * const.MIN_PERCENT_ALIVE) or cod_biomass > (const.STARTING_BIOMASS_COD * const.MAX_PERCENT_ALIVE):
        return False
    if anchovy_biomass < (const.STARTING_BIOMASS_ANCHOVY * const.MIN_PERCENT_ALIVE) or anchovy_biomass > (const.STARTING_BIOMASS_ANCHOVY * const.MAX_PERCENT_ALIVE):
        return False
    if plankton_biomass < (const.STARTING_BIOMASS_PLANKTON * const.MIN_PERCENT_ALIVE) or plankton_biomass > (const.STARTING_BIOMASS_PLANKTON * const.MAX_PERCENT_ALIVE):
        return False

    return True