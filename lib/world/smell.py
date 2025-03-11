import numpy as np
from scipy.signal import convolve2d
import lib.constants as const

# Define the diffusion kernel as a NumPy array.
diffusion_kernel = np.array([
    [1/16, 1/8, 1/16],
    [1/8,  1/4, 1/8],
    [1/16, 1/8, 1/16]
], dtype=np.float32)

def diffuse_smell(world):
    """
    Diffuse the smell channels in the world array using 2D convolution.
    This function applies the diffusion kernel to each smell channel.
    """
    # Determine the smell channels from the species properties.
    smell_channels = [properties["smell_offset"] for properties in const.SPECIES_MAP.values()]
    
    # For each smell channel, perform a 2D convolution.
    for channel in smell_channels:
        # Use 'same' mode to preserve the input shape and 'symm' boundary for symmetric padding.
        diffused = convolve2d(world[:, :, channel], diffusion_kernel, mode='same', boundary='symm')
        world[:, :, channel] = diffused

def update_smell(world):
    """
    Updates the smell in the world:
      - First, it decays the existing smell.
      - Then, it adds new smell proportional to the biomass in each cell.
      - Finally, it diffuses the resulting smell.
    """
    # Decay existing smell for each species.
    for species, properties in const.SPECIES_MAP.items():
        channel = properties["smell_offset"]
        world[:, :, channel] *= (1 - const.SMELL_DECAY_RATE)

    # Add emitted smell from the current biomass.
    for species, properties in const.SPECIES_MAP.items():
        biomass_channel = properties["biomass_offset"]
        smell_channel = properties["smell_offset"]
        world[:, :, smell_channel] += world[:, :, biomass_channel] * const.SMELL_EMISSION_RATE

    # Diffuse the updated smell.
    diffuse_smell(world)
