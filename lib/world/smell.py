import torch
import torch.nn.functional as F
import lib.constants as const

device = torch.device(const.DEVICE)

diffusion_kernel = torch.tensor([
    [1/16, 1/8, 1/16],
    [1/8,  1/4, 1/8],
    [1/16, 1/8, 1/16]
])

def diffuse_smell(world):
    # Prepare the smell channels for convolution
    smell_channels = [properties["smell_offset"] for properties in const.SPECIES_MAP.values()]
    smell = world[:, :, smell_channels].permute(2, 0, 1).unsqueeze(0)
    
    # Expand the kernel to match the number of channels
    kernel = diffusion_kernel.expand(smell.size(1), 1, 3, 3)
    
    # Apply convolution with padding to keep the same size
    smell_diffused = F.conv2d(smell, kernel, padding=1, groups=smell.size(1))
    
    # Update the smell channels in the world tensor
    world[:, :, smell_channels] = smell_diffused.squeeze(0).permute(1, 2, 0)

def update_smell(world):
    # Decay existing smell
    for species, properties in const.SPECIES_MAP.items():
        world[:, :, properties["smell_offset"]] *= (1 - const.SMELL_DECAY_RATE)

    # Add emitted smell from current biomass
    for species, properties in const.SPECIES_MAP.items():
        world[:, :, properties["smell_offset"]] += world[:, :, properties["biomass_offset"]] * const.SMELL_EMISSION_RATE

    diffuse_smell(world)

