import math
import random
import lib.constants as const
import torch
import numpy as np
from lib.constants import Terrain, Action

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
        torch.arange(world_size_x),
        torch.arange(world_size_y),
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
    total_biomass_plankton = const.SPECIES_MAP["plankton"]["starting_biomass"]
    biomass_per_cell = total_biomass_plankton / num_cells

    # Assign biomass to the cells
    world_tensor[cluster_indices[:, 0], cluster_indices[:, 1], const.OFFSETS_BIOMASS + const.SPECIES_MAP["plankton"]["index"]] = biomass_per_cell

    print(f"Plankton respawned at the {side} side of the world.")
    return world_tensor

def move_plankton_based_on_current(world_tensor, world_data):
    # Get the size of the world grid
    world_size_x, world_size_y = world_tensor.shape[0], world_tensor.shape[1]

    # Extract the plankton biomass layer
    plankton_biomass = world_tensor[:, :, const.OFFSETS_BIOMASS + const.SPECIES_MAP["plankton"]["index"]]

    # Extract the current angles
    current_angles = world_data[:, :, 0]

    # Initialize a buffer to accumulate new biomass positions
    new_biomass = torch.zeros_like(plankton_biomass)

    # Get indices of all cells (ensure they are of type torch.long)
    x_indices = torch.arange(world_size_x, dtype=torch.long).view(-1, 1).expand(-1, world_size_y)
    y_indices = torch.arange(world_size_y, dtype=torch.long).view(1, -1).expand(world_size_x, -1)

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
    ])

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
    world_tensor[:, :, const.OFFSETS_BIOMASS + const.SPECIES_MAP["plankton"]["index"]] = new_biomass

    return world_tensor

def plankton_growth(P, P_max, P_threshold, k):
    """
    Logistic growth function using NumPy with improved numerical stability.
    Computes logistic growth in float64 to avoid overflow issues, then casts back to float32.
    """
    # Convert inputs to float64 for stability.
    P = np.asarray(P, dtype=np.float64)
    P_max = np.float64(P_max)
    P_threshold = np.float64(P_threshold)
    k = np.float64(k)
    
    # Compute the exponent input and clip it.
    exp_input = np.clip(-k * (P - P_threshold), -100, 100)
    result = P_max / (1 + np.exp(exp_input))
    return result.astype(np.float32)


def spawn_plankton(world, world_data):
    """
    Updates the world (a NumPy array) by applying logistic growth to existing plankton and
    spawning new plankton when appropriate.
    
    Parameters:
      - world: NumPy array of shape (WORLD_SIZE, WORLD_SIZE, TOTAL_TENSOR_VALUES)
      - world_data: NumPy array of shape (WORLD_SIZE, WORLD_SIZE, 5)
      
    Returns:
      The updated world array.
    """
    # Determine the channel index for plankton biomass.
    plankton_index = const.OFFSETS_BIOMASS + const.SPECIES_MAP["plankton"]["index"]
    # Extract the current plankton biomass and related flags/counters.
    plankton_biomass = world[:, :, plankton_index]
    plankton_flag = world_data[:, :, 1]      # Cells with 1 are initial plankton locations.
    plankton_counter = world_data[:, :, 2]     # Plankton respawn counter.
    
    # Create a mask for cells that are designated as plankton cells.
    plankton_cells = (plankton_flag == 1)
    
    # --- Existing plankton cells: apply logistic growth ---
    existing_plankton_mask = (plankton_biomass > 0) & plankton_cells
    if np.any(existing_plankton_mask):
        P_max = const.SPECIES_MAP["plankton"]["max_in_cell"]
        P_threshold = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["growth_threshold"]
        k = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["growth_rate_constant"]
        
        # Calculate updated biomass using logistic growth.
        updated_biomass = plankton_growth(plankton_biomass[existing_plankton_mask], P_max, P_threshold, k)
        # Clamp the values to P_max.
        updated_biomass = np.clip(updated_biomass, None, P_max)
        
        # Safely update the biomass channel.
        channel = world[:, :, plankton_index].copy()
        channel[existing_plankton_mask] = updated_biomass
        world[:, :, plankton_index] = channel
        
        # Reset the respawn counter for these cells.
        counter_channel = world_data[:, :, 2].copy()
        counter_channel[existing_plankton_mask] = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["respawn_delay"]
        world_data[:, :, 2] = counter_channel

    # --- Empty plankton cells: decrement counter and spawn if counter reaches zero ---
    empty_plankton_mask = (plankton_biomass == 0) & plankton_cells
    if np.any(empty_plankton_mask):
        # Decrease the counter by 1 in these cells.
        world_data[:, :, 2][empty_plankton_mask] -= 1
        # Ensure the counter doesn't drop below zero.
        world_data[:, :, 2][empty_plankton_mask] = np.clip(world_data[:, :, 2][empty_plankton_mask], 0, None)
        
        # Identify cells where the respawn counter has reached zero.
        respawn_mask = (world_data[:, :, 2] == 0) & empty_plankton_mask
        if np.any(respawn_mask):
            # Reset the counter and spawn new plankton biomass.
            world_data[:, :, 2][respawn_mask] = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["respawn_delay"]
            channel = world[:, :, plankton_index].copy()
            channel[respawn_mask] = const.SPECIES_MAP["plankton"]["max_in_cell"] * 0.1
            world[:, :, plankton_index] = channel

    return world