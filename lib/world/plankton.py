import math
import random
import lib.constants as const
import numpy as np
from lib.constants import Terrain, Action

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