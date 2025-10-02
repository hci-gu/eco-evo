import numpy as np
from scipy.signal import convolve2d
from lib.config.settings import Settings
from lib.config.const import SPECIES
from lib.model import MODEL_OFFSETS

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
    smell_channels = [props["smell"] for props in MODEL_OFFSETS.values() if "smell" in props]

    # For each smell channel, perform a 2D convolution.
    for channel in smell_channels:
        # Use 'same' mode to preserve the input shape and 'symm' boundary for symmetric padding.
        diffused = convolve2d(world[:, :, channel], diffusion_kernel, mode='same', boundary='symm')
        world[:, :, channel] = diffused

def update_smell(settings: Settings, world):
    """
    Updates the smell in the world:
      - First, it decays the existing smell.
      - Then, it adds new smell proportional to the biomass in each cell.
      - Finally, it diffuses the resulting smell.
    """
    # Decay existing smell for each species.
    for species in SPECIES:
        smell_channel = MODEL_OFFSETS[species]["smell"]
        world[:, :, smell_channel] *= (1 - settings.smell_decay)

    # Add emitted smell from the current biomass.
    for species in SPECIES:
        biomass_channel = MODEL_OFFSETS[species]["biomass"]
        smell_channel = MODEL_OFFSETS[species]["smell"]
        world[:, :, smell_channel] += world[:, :, biomass_channel] * settings.smell_emission_rate

    # Diffuse the updated smell.
    diffuse_smell(world)
