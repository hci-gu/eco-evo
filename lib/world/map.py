import opensimplex
import math
import random
import numpy as np
import lib.constants as const
from lib.constants import Terrain
from PIL import Image

# Define a palette for the map colors.
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

def smooth_skewed_random():
    if const.FIXED_BIOMASS:
        return 1

    """Returns a value between 0.25 and 4, skewed toward 0.5-2.0 with smooth falloff, using gamma sampling."""
    alpha = 2.5
    beta_param = 4.5

    x = random.gammavariate(alpha, 1.0)
    y = random.gammavariate(beta_param, 1.0)

    beta_sample = x / (x + y)
    return beta_sample * (4 - 0.25) + 0.25

def add_species_to_map(world_array, world_data):
    opensimplex.seed(int(random.random() * 100000))
    # Calculate total noise sums for each species.
    noise_sums = {species: 0 for species in const.SPECIES_MAP.keys()}
    starting_biomasses = {species: 0 for species in const.SPECIES_MAP.keys()}
    for species, properties in const.SPECIES_MAP.items():
        properties["starting_biomass"] = properties["original_starting_biomass"] 
        starting_biomasses[species] = properties["starting_biomass"] * smooth_skewed_random()
        properties["starting_biomass"] = starting_biomasses[species]
    world_data[:, :, 4] = 0

    # First pass: accumulate noise values.
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_array[x, y, Terrain.WATER.value] == 1:
                for species, properties in const.SPECIES_MAP.items():
                    noise = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2.0
                    if noise < 0.35:
                        noise = 0
                    noise = noise ** const.NOISE_SCALING
                    noise_sums[species] += noise

    # Second pass: distribute biomass based on noise values.
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_array[x, y, Terrain.WATER.value] == 1:
                for species, properties in const.SPECIES_MAP.items():
                    noise = (opensimplex.noise2(x * 0.1, y * 0.1) + 1) / 2.0
                    if noise < 0.35:
                        noise = 0
                        world_data[x, y, 1] = 0
                        world_data[x, y, 2] = 0
                    noise = noise ** const.NOISE_SCALING
                    if noise > 0:
                        # Distribute biomass proportional to noise.
                        world_array[x, y, properties["biomass_offset"]] = (noise / noise_sums[species]) * starting_biomasses[species]
                        world_array[x, y, properties["energy_offset"]] = const.MAX_ENERGY
                        if properties["hardcoded_logic"]:
                            world_data[x, y, 1] = 1  # Mark plankton cluster flag.
                            world_data[x, y, 2] = properties["hardcoded_rules"]["respawn_delay"]  # Set plankton respawn delay.

    # print average biomass in each cell that is not empty per species
    # for species, properties in const.SPECIES_MAP.items():
    #     biomass_offset = properties["biomass_offset"]
    #     biomass = world_array[:, :, biomass_offset]
    #     non_empty_cells = np.count_nonzero(biomass > 0)
    #     if non_empty_cells > 0:
    #         avg_biomass = np.sum(biomass) / non_empty_cells
    #         #print(f"Average biomass for {species}: {avg_biomass:.2f}")


    # Set smell channels to 0.
    for species, properties in const.SPECIES_MAP.items():
        world_array[:, :, properties["smell_offset"]] = 0

    return starting_biomasses

def read_map_from_file(folder_path):
    image = Image.open(folder_path + '/map.png')
    image = image.resize((const.WORLD_SIZE, const.WORLD_SIZE), resample=Image.NEAREST)
    pixels = image.load()

    depth_image = Image.open(folder_path + '/depth.png')
    depth_image = depth_image.resize((const.WORLD_SIZE, const.WORLD_SIZE), resample=Image.NEAREST)
    depth_image = depth_image.convert('L')
    depth_pixels = depth_image.load()
    
    # Create numpy arrays for the world map and extra world data.
    world_array = np.zeros((const.WORLD_SIZE, const.WORLD_SIZE, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((const.WORLD_SIZE, const.WORLD_SIZE, 5), dtype=np.float32)

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            color = pixels[x, y][:3]
            depth_value = depth_pixels[x, y] / 255.0
            world_data[x, y, 3] = depth_value
            if color == palette["water"]:
                world_array[x, y, :3] = np.array([0, 1, 0], dtype=np.float32)
            else:
                world_array[x, y, :3] = np.array([1, 0, 0], dtype=np.float32)

    starting_biomasses = add_species_to_map(world_array, world_data)

    return np.ascontiguousarray(world_array), np.ascontiguousarray(world_data), starting_biomasses

def create_map_from_noise(static=False):
    seed = 1 if static else int(random.random() * 100000)
    opensimplex.seed(seed)

    world_array = np.zeros((const.WORLD_SIZE, const.WORLD_SIZE, const.TOTAL_TENSOR_VALUES), dtype=np.float32)
    world_data = np.zeros((const.WORLD_SIZE, const.WORLD_SIZE, 5), dtype=np.float32)

    center_x, center_y = const.WORLD_SIZE // 2, const.WORLD_SIZE // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            distance_factor = distance / max_distance
            noise_value = opensimplex.noise2(x * 0.15, y * 0.15)
            threshold = 0.6 * distance_factor + 0.1 * noise_value

            terrain = Terrain.WATER if threshold < 0.5 else Terrain.LAND
            terrain_encoding = [0, 1, 0] if terrain == Terrain.WATER else [1, 0, 0]
            world_array[x, y, :3] = np.array(terrain_encoding, dtype=np.float32)

            if terrain == Terrain.WATER:
                dx = x - center_x
                dy = y - center_y
                current_angle = math.atan2(dy, dx) + math.pi / 2
                world_data[x, y, 0] = current_angle  # Store the current angle.

    starting_biomasses = add_species_to_map(world_array, world_data)

    return np.ascontiguousarray(world_array), np.ascontiguousarray(world_data), starting_biomasses
