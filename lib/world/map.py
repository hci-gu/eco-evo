import opensimplex
import math
import random
import torch
import torch.nn.functional as F
import lib.constants as const
from lib.constants import Terrain
from PIL import Image  # Add this import

device = torch.device("cpu")

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

def read_map_from_file(file_path):
    print(f"Reading map from file: {file_path}")
    image = Image.open(file_path)
    image = image.resize((const.WORLD_SIZE, const.WORLD_SIZE), resample=Image.NEAREST)
    pixels = image.load()
    
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 12, device=device)
    world_data = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 3, device=device)

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            # world_tensor[x, y, :3] = torch.tensor([0, 1, 0], device=device)  # Water
            color = pixels[x, y][:3]
            print(color)
            if color == palette["water"]:
                world_tensor[x, y, :3] = torch.tensor([0, 1, 0], device=device)
            else:
                world_tensor[x, y, :3] = torch.tensor([1, 0, 0], device=device)

    add_species_to_map(world_tensor, world_data)

    return world_tensor, world_data

def add_species_to_map(world_tensor, world_data):
    total_biomass_cod = const.STARTING_BIOMASS_COD
    total_biomass_anchovy = const.STARTING_BIOMASS_ANCHOVY
    total_biomass_plankton = const.STARTING_BIOMASS_PLANKTON

    noise_sum_cod = 0
    noise_sum_anchovy = 0
    noise_sum_plankton = 0

    # First pass: calculate noise sums
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_tensor[x, y, Terrain.WATER.value] == 1:
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                if noise_cod < 0.8: noise_cod = 0
                if noise_anchovy < 0.6: noise_anchovy = 0
                if noise_plankton < 0.35: noise_plankton = 0

                noise_cod = noise_cod ** const.NOISE_SCALING
                noise_anchovy = noise_anchovy ** const.NOISE_SCALING
                noise_plankton = noise_plankton ** const.NOISE_SCALING

                noise_sum_cod += noise_cod
                noise_sum_anchovy += noise_anchovy
                noise_sum_plankton += noise_plankton

    # Second pass: distribute biomass across water cells based on noise values
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_tensor[x, y, Terrain.WATER.value] == 1:
                noise_cod = (opensimplex.noise2(x * 0.3, y * 0.3) + 1) / 2
                noise_anchovy = (opensimplex.noise2(x * 0.2, y * 0.2) + 1) / 2
                noise_plankton = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2

                if noise_cod < 0.8: noise_cod = 0
                if noise_anchovy < 0.6: noise_anchovy = 0
                if noise_plankton < 0.35: 
                    noise_plankton = 0
                    world_data[x, y, 1] = 0
                    world_data[x, y, 2] = 0

                noise_cod = noise_cod ** const.NOISE_SCALING
                noise_anchovy = noise_anchovy ** const.NOISE_SCALING
                noise_plankton = noise_plankton ** const.NOISE_SCALING

                if noise_cod > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_COD] = (noise_cod / noise_sum_cod) * total_biomass_cod
                    world_tensor[x, y, const.OFFSETS_ENERGY_COD] = const.MAX_ENERGY
                if noise_anchovy > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY] = (noise_anchovy / noise_sum_anchovy) * total_biomass_anchovy
                    world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY] = const.MAX_ENERGY
                if noise_plankton > 0:
                    world_tensor[x, y, const.OFFSETS_BIOMASS_PLANKTON] = (noise_plankton / noise_sum_plankton) * total_biomass_plankton
                    world_tensor[x, y, const.OFFSETS_ENERGY_PLANKTON] = const.MAX_ENERGY
                    world_data[x, y, 1] = 1  # Add plankton cluster flag
                    world_data[x, y, 2] = const.PLANKTON_RESPAWN_DELAY # Add plankton respawn counter

    world_tensor[:, :, const.OFFSETS_SMELL_PLANKTON] = 0
    world_tensor[:, :, const.OFFSETS_SMELL_ANCHOVY] = 0
    world_tensor[:, :, const.OFFSETS_SMELL_COD] = 0

def create_map_from_noise(static=False):
    seed = 1 if static else int(random.random() * 100000)
    opensimplex.seed(seed)

    # Create a tensor to represent the entire world
    # Tensor dimensions: (WORLD_SIZE, WORLD_SIZE, 12) -> 3 for terrain (3 one-hot) + biomass (3 species) + energy (3 species) + smell (3 species)
    world_tensor = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 12, device=device)
    world_data = torch.zeros(const.WORLD_SIZE, const.WORLD_SIZE, 3, device=device)

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

    add_species_to_map(world_tensor, world_data)

    return world_tensor, world_data