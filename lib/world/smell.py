import torch
import torch.nn.functional as F
import lib.constants as const

device = torch.device("cpu")

diffusion_kernel = torch.tensor([
    [1/16, 1/8, 1/16],
    [1/8,  1/4, 1/8],
    [1/16, 1/8, 1/16]
], device=device)

def diffuse_smell(world):
    # Prepare the smell channels for convolution
    # Shape: [Batch_Size=1, Channels=3, Height, Width]
    smell = world[:, :, const.OFFSETS_SMELL_PLANKTON:const.OFFSETS_SMELL_COD+1].permute(2, 0, 1).unsqueeze(0)
    
    # Expand the kernel to match the number of channels
    kernel = diffusion_kernel.expand(smell.size(1), 1, 3, 3)
    
    # Apply convolution with padding to keep the same size
    smell_diffused = F.conv2d(smell, kernel, padding=1, groups=smell.size(1))
    
    # Update the smell channels in the world tensor
    world[:, :, const.OFFSETS_SMELL_PLANKTON:const.OFFSETS_SMELL_COD+1] = smell_diffused.squeeze(0).permute(1, 2, 0)

def update_smell(world):
    # Decay existing smell
    world[:, :, const.OFFSETS_SMELL_PLANKTON] *= (1 - const.SMELL_DECAY_RATE)
    world[:, :, const.OFFSETS_SMELL_ANCHOVY] *= (1 - const.SMELL_DECAY_RATE)
    world[:, :, const.OFFSETS_SMELL_COD] *= (1 - const.SMELL_DECAY_RATE)

    # Add emitted smell from current biomass
    world[:, :, const.OFFSETS_SMELL_PLANKTON] += world[:, :, const.OFFSETS_BIOMASS_PLANKTON] * const.SMELL_EMISSION_RATE
    world[:, :, const.OFFSETS_SMELL_ANCHOVY] += world[:, :, const.OFFSETS_BIOMASS_ANCHOVY] * const.SMELL_EMISSION_RATE
    world[:, :, const.OFFSETS_SMELL_COD] += world[:, :, const.OFFSETS_BIOMASS_COD] * const.SMELL_EMISSION_RATE

    diffuse_smell(world)
