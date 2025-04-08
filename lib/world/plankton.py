import math
import random
import lib.constants as const
import numpy as np
from lib.constants import Terrain, Action

def plankton_growth(current, growth_rate=100, max_biomass=5000):
    current = np.asarray(current)
    growth = growth_rate * (1 - current / max_biomass)
    growth = np.maximum(growth, 0)  # Prevent negative growth
    updated = current + growth
    return np.minimum(updated, max_biomass)


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
    biomass_offset = const.SPECIES_MAP["plankton"]["biomass_offset"]
    # Extract the current plankton biomass and related flags/counters.
    plankton_biomass = world[:, :, biomass_offset]
    plankton_flag = world_data[:, :, 1]      # Cells with 1 are initial plankton locations.
    
    # Create a mask for cells that are designated as plankton cells.
    plankton_cells = (plankton_flag == 1)
    
    # --- Existing plankton cells: apply logistic growth ---
    existing_plankton_mask = (plankton_biomass > 0) & plankton_cells
    if np.any(existing_plankton_mask):
        P_max = const.SPECIES_MAP["plankton"]["max_biomass_in_cell"]
        k = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["growth_rate_constant"]
        
        # print("pgrowth", plankton_biomass[existing_plankton_mask])
        updated_biomass = plankton_growth(plankton_biomass[existing_plankton_mask], k, P_max)
        
        # Safely update the biomass channel.
        # channel = world[:, :, biomass_offset].copy()
        world[existing_plankton_mask, biomass_offset] = updated_biomass
        # world[:, :, biomass_offset] = channel
        
        world_data[:, :, 2] = const.SPECIES_MAP["plankton"]["hardcoded_rules"]["respawn_delay"]

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
            channel = world[:, :, biomass_offset].copy()
            channel[respawn_mask] = const.SPECIES_MAP["plankton"]["max_biomass_in_cell"] * 0.1
            world[:, :, biomass_offset] = channel

    return world