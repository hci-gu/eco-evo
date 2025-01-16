import opensimplex
import math
import random
import torch
import torch.nn.functional as F
import lib.constants as const
from lib.constants import Terrain
from PIL import Image  # Add this import

device = torch.device(const.DEVICE)

palette = {
    "water": (65, 155, 223),
    "trees": (57, 125, 73),
    "grass": (136, 176, 83),
    "flooded_vegetation": (122, 135, 198),
    "crops": (228, 150, 53),
    "shrub": (223, 195, 90),
    "built": (196, 40, 27),
    "bare": (165, 155, 143),
    "snow": (179, 159, 225),
}

def read_map_from_file(folder_path):
    image = Image.open(folder_path + '/map.png')
    image = image.resize((const.WORLD_SIZE, const.WORLD_SIZE), resample=Image.NEAREST)
    pixels = image.load()

    depth_image = Image.open(folder_path + '/depth.png')
    depth_image = depth_image.resize((const.WORLD_SIZE, const.WORLD_SIZE), resample=Image.NEAREST)
    depth_image = depth_image.convert('L')
    depth_pixels = depth_image.load()
    
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, const.TOTAL_TENSOR_VALUES)
    world_data = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 4)

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            color = pixels[x, y][:3]
            depth_value = depth_pixels[x, y] / 255
            world_data[x, y, 3] = depth_value
            if color == palette["water"]:
                world_tensor[x, y, :3] = torch.tensor([0, 1, 0])
            else:
                world_tensor[x, y, :3] = torch.tensor([1, 0, 0])

    add_species_to_map(world_tensor, world_data)

    return world_tensor, world_data

def add_species_to_map(world_tensor, world_data):
    noise_sums = {species: 0 for species in const.SPECIES_MAP.keys()}

    # First pass: calculate noise sums
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_tensor[x, y, Terrain.WATER.value] == 1:
                for species, properties in const.SPECIES_MAP.items():
                    noise = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2
                    if noise < 0.35: noise = 0
                    noise = noise ** const.NOISE_SCALING
                    noise_sums[species] += noise

    # Second pass: distribute biomass across water cells based on noise values
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_tensor[x, y, Terrain.WATER.value] == 1:
                for species, properties in const.SPECIES_MAP.items():
                    noise = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2
                    if noise < 0.35: 
                        noise = 0
                        world_data[x, y, 1] = 0
                        world_data[x, y, 2] = 0
                    noise = noise ** const.NOISE_SCALING
                    if noise > 0:
                        world_tensor[x, y, properties["biomass_offset"]] = (noise / noise_sums[species]) * properties["starting_biomass"]
                        if properties["hardcoded_logic"]:
                            world_data[x, y, 1] = 1  # Add plankton cluster flag
                            world_data[x, y, 2] = properties["hardcoded_rules"]["respawn_delay"]  # Add plankton respawn counter

    for species, properties in const.SPECIES_MAP.items():
        world_tensor[:, :, properties["smell_offset"]] = 0

def create_map_from_noise(static=False):
    seed = 1 if static else int(random.random() * 100000)
    opensimplex.seed(seed)

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 12) -> 3 for terrain (3 one-hot) + biomass (3 species) + energy (3 species) + smell (3 species)
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, const.TOTAL_TENSOR_VALUES)
    world_data = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 3)

    center_x, center_y = const.WORLD_SIZE // 2, const.WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    # Iterate over the world grid and initialize cells directly into the tensor
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            distance_factor = distance / max_distance
            noise_value = opensimplex.noise2(x * 0.15, y * 0.15)
            threshold = 0.6 * distance_factor + 0.1 * noise_value

            # Determine if the cell is water or land
            terrain = Terrain.WATER if threshold < 0.5 else Terrain.LAND
            
            # Set terrain one-hot encoding
            terrain_encoding = [0, 1, 0] if terrain == Terrain.WATER else [1, 0, 0]
            world_tensor[x, y, :3] = torch.tensor(terrain_encoding)

            if terrain == Terrain.WATER:
                # Calculate the angle of the current based on the cell's position relative to the center
                # This ensures a clockwise rotation direction
                dx = x - center_x
                dy = y - center_y
                current_angle = math.atan2(dy, dx) + math.pi / 2

                # Store the current angle in the tensor
                world_data[x, y, 0] = current_angle  # Add current angle

    add_species_to_map(world_tensor, world_data)

    return world_tensor, world_data