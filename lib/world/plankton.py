import math
import random
import numpy as np

from lib.config.settings import Settings
from lib.model import MODEL_OFFSETS
from lib.config.species import SpeciesMap
from lib.world.map import Terrain

def plankton_growth(current, growth_rate=100, max_biomass=5000):
    current = np.asarray(current)
    growth = growth_rate * (1 - current / max_biomass)
    growth = np.maximum(growth, 0)  # Prevent negative growth
    updated = current + growth
    return np.minimum(updated, max_biomass)


def spawn_plankton(species_map: SpeciesMap, world, world_data):
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
    biomass_offset = MODEL_OFFSETS["plankton"]["biomass"]
    # Extract the current plankton biomass and related flags/counters.
    plankton_biomass = world[:, :, biomass_offset]
    plankton_flag = world_data[:, :, 1]      # Cells with 1 are initial plankton locations.
    
    # Create a mask for cells that are designated as plankton cells.
    plankton_cells = (plankton_flag == 1)
    
    # --- Existing plankton cells: apply logistic growth ---
    existing_plankton_mask = (plankton_biomass > 0) & plankton_cells
    if np.any(existing_plankton_mask):
        P_max = species_map["plankton"].max_biomass_in_cell
        k = species_map["plankton"].hardcoded_rules["growth_rate_constant"]
        
        # print("pgrowth", plankton_biomass[existing_plankton_mask])
        updated_biomass = plankton_growth(plankton_biomass[existing_plankton_mask], k, P_max)
        
        # Safely update the biomass channel.
        # channel = world[:, :, biomass_offset].copy()
        world[existing_plankton_mask, biomass_offset] = updated_biomass
        # world[:, :, biomass_offset] = channel

        world_data[:, :, 2][existing_plankton_mask] = species_map["plankton"].hardcoded_rules["respawn_delay"]

    # --- Empty plankton cells: decrement counter and spawn if counter reaches zero ---
    empty_plankton_mask = (plankton_biomass == 0) & plankton_cells
    if np.any(empty_plankton_mask):
        world_data[:, :, 2][empty_plankton_mask] -= 1
        
        respawn_mask = (world_data[:, :, 2] == 0) & empty_plankton_mask
        if np.any(respawn_mask):
            # Reset the counter and spawn new plankton biomass.
            world_data[:, :, 2][respawn_mask] = species_map["plankton"].hardcoded_rules["respawn_delay"]
            world[:, :, biomass_offset][respawn_mask] = species_map["plankton"].max_biomass_in_cell * 0.25

    return world

def randomwalk_plankton(settings: Settings, world_array, world_data):
    """
    Moves all plankton cells to a randomly chosen adjacent water cell that does not
    already have plankton. This function examines each cell in world_data flagged as
    having plankton (world_data[:, :, 1] == 1) and moves its biomass, energy, and timer
    from that cell to a random adjacent water cell if that cell is free of plankton.
    If no adjacent free cell is found, the plankton stays in place.
    
    Note:
      - This function does not consider any timers or delays—it simply moves all plankton.
      - It relies on the "plankton" entry in const.SPECIES_MAP to determine the correct
        channel indices for biomass, energy, and respawn timer.
      - The destination cell must be within bounds, be water, and have no plankton.
    """

    biomass_offset = MODEL_OFFSETS["plankton"]["biomass"]
    energy_offset = MODEL_OFFSETS["plankton"]["energy"]

    world_size = settings.world_size

    # Prepare new arrays for the updated plankton values.
    new_biomass = np.copy(world_array[:, :, biomass_offset])
    new_energy = np.copy(world_array[:, :, energy_offset])
    new_flag = np.copy(world_data[:, :, 1])  # Plankton cluster flag.
    new_timer = np.copy(world_data[:, :, 2])  # Respawn timer.

    # Clear out existing plankton markings from the new arrays.
    new_biomass[:, :] = 0
    new_energy[:, :] = 0
    new_flag[:, :] = 0
    new_timer[:, :] = 0

    # Define neighbor offsets (8-connected grid).
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

    # Process each cell in the original grid that is flagged as plankton.
    for x in range(world_size):
        for y in range(world_size):
            if world_data[x, y, 1] == 1:
                # Shuffle the neighbor list to choose a random order.
                offsets = neighbor_offsets.copy()
                random.shuffle(offsets)
                moved = False

                # Try moving to one of the neighboring cells.
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    # Check bounds.
                    if nx < 0 or nx >= world_size or ny < 0 or ny >= world_size:
                        continue
                    # Check if destination is water.
                    if world_array[nx, ny, Terrain.WATER.value] != 1:
                        continue
                    # Ensure the destination does not already have plankton.
                    if new_flag[nx, ny] == 1:
                        continue

                    # Valid move found: transfer biomass, energy, and timer.
                    new_biomass[nx, ny] = world_array[x, y, biomass_offset]
                    new_energy[nx, ny] = world_array[x, y, energy_offset]
                    new_flag[nx, ny] = 1
                    new_timer[nx, ny] = world_data[x, y, 2]
                    moved = True
                    break

                # If no valid move was found, leave it in place (if not already filled).
                if not moved and new_flag[x, y] == 0:
                    new_biomass[x, y] = world_array[x, y, biomass_offset]
                    new_energy[x, y] = world_array[x, y, energy_offset]
                    new_flag[x, y] = 1
                    new_timer[x, y] = world_data[x, y, 2]

    # Write the updated values back to the original arrays.
    world_array[:, :, biomass_offset] = new_biomass
    world_array[:, :, energy_offset] = new_energy
    world_data[:, :, 1] = new_flag
    world_data[:, :, 2] = new_timer