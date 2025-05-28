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

    """Returns a value between 0.75 and 2.0, skewed toward 0.5-2.0 with smooth falloff, using gamma sampling."""
    alpha = 2.5
    beta_param = 4.5

    x = random.gammavariate(alpha, 1.0)
    y = random.gammavariate(beta_param, 1.0)

    beta_sample = x / (x + y)
    return beta_sample * (2.0 - 0.75) + 0.75

def add_species_to_map(world_array, world_data, seed = None):
    if seed is None:
        seed = int(random.random() * 100000)
    rng = np.random.default_rng(seed)
    opensimplex.seed(seed)
    # Calculate total noise sums for each species.
    noise_sums = {species: 0 for species in const.SPECIES_MAP.keys()}
    starting_biomasses = {species: 0 for species in const.SPECIES_MAP.keys()}
    for species, properties in const.SPECIES_MAP.items():
        properties["starting_biomass"] = properties["original_starting_biomass"] 
        starting_biomasses[species] = properties["starting_biomass"] * smooth_skewed_random()
        properties["starting_biomass"] = starting_biomasses[species]

    world_data[:, :, 4] = 0

    species_noise_offsets = {}
    for species, properties in const.SPECIES_MAP.items():
        species_noise_offsets[species] = rng.integers(0, 100000)

    # First pass: accumulate noise values.
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_array[x, y, Terrain.WATER.value] == 1:
                for species, properties in const.SPECIES_MAP.items():
                    noise = (opensimplex.noise2((x + species_noise_offsets[species]) * 0.1, (y + species_noise_offsets[species]) * 0.1) + 1) / 2.0
                    if noise < properties["noise_threshold"]:
                        noise = 0
                    noise = noise ** properties["noise_scaling"]
                    noise_sums[species] += noise


    # Second pass: distribute biomass based on noise values.
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if world_array[x, y, Terrain.WATER.value] == 1:
                for species, properties in const.SPECIES_MAP.items():
                    noise = (opensimplex.noise2((x + species_noise_offsets[species]) * 0.1, (y + species_noise_offsets[species]) * 0.1) + 1) / 2.0
                    if noise < properties["noise_threshold"]:
                        noise = 0
                        world_data[x, y, 1] = 0
                        world_data[x, y, 2] = 0
                    noise = noise ** properties["noise_scaling"]
                    if noise > 0:
                        # Distribute biomass proportional to noise.
                        world_array[x, y, properties["biomass_offset"]] = (noise / noise_sums[species]) * starting_biomasses[species]
                        # world_array[x, y, properties["energy_offset"]] = 60 + random.random() * 40
                        world_array[x, y, properties["energy_offset"]] = const.MAX_ENERGY
                        if properties["hardcoded_logic"]:
                            world_data[x, y, 1] = 1  # Mark plankton cluster flag.
                            world_data[x, y, 2] = properties["hardcoded_rules"]["respawn_delay"]  # Set plankton respawn delay.
    
    # min cells should be 10% coverage of the world
    MIN_CELLS_WITH_BIOMASS = int(const.WORLD_SIZE * const.WORLD_SIZE * 0.1)
    for species, properties in const.SPECIES_MAP.items():
        biomass_offset = properties["biomass_offset"]
        energy_offset = properties["energy_offset"]
        hardcoded = properties.get("hardcoded_logic", False)

        biomass_layer = world_array[:, :, biomass_offset]
        total_biomass = np.sum(biomass_layer)

        existing_mask = biomass_layer > 0
        num_existing = np.count_nonzero(existing_mask)

        if num_existing < MIN_CELLS_WITH_BIOMASS:
            # We need to add (MIN - existing) more cells
            deficit = MIN_CELLS_WITH_BIOMASS - num_existing

            water_mask = (world_array[:, :, Terrain.WATER.value] == 1)
            empty_mask = (biomass_layer == 0)
            eligible_positions = np.argwhere(water_mask & empty_mask)

            if eligible_positions.shape[0] < deficit:
                print(f"Warning: Not enough eligible cells for {species}. Needed {deficit}, found {eligible_positions.shape[0]}")
                deficit = eligible_positions.shape[0]

            chosen_indices = eligible_positions[rng.choice(eligible_positions.shape[0], size=deficit, replace=False)]

            # Combine existing biomass locations with the newly chosen ones
            existing_indices = np.argwhere(existing_mask)
            all_indices = np.vstack([existing_indices, chosen_indices])

            # Redistribute biomass evenly (or add noise if you prefer)
            biomass_per_cell = total_biomass / all_indices.shape[0]

            # Zero out the old biomass for this species
            biomass_layer[:, :] = 0

            for x, y in all_indices:
                world_array[x, y, biomass_offset] = biomass_per_cell
                world_array[x, y, energy_offset] = const.MAX_ENERGY
                if hardcoded:
                    world_data[x, y, 1] = 1
                    world_data[x, y, 2] = properties["hardcoded_rules"]["respawn_delay"]

        # get existing again
        biomass_layer = world_array[:, :, biomass_offset]
        existing_mask = (biomass_layer > 0)
        num_existing = np.sum(existing_mask)

    # print average biomass in each cell that is not empty per species
    # for species, properties in const.SPECIES_MAP.items():
    #     biomass_offset = properties["biomass_offset"]
    #     biomass = world_array[:, :, biomass_offset]
    #     print(f"Species: {species}, total biomass: {np.sum(biomass):.2f}")

    #     non_empty_cells = np.count_nonzero(biomass > 0)
    #     if non_empty_cells > 0:
    #         avg_biomass = np.sum(biomass) / non_empty_cells
    #         print(f"Average biomass for {species}: {avg_biomass:.2f}")

    # Set smell channels to 0.
    for species, properties in const.SPECIES_MAP.items():
        world_array[:, :, properties["smell_offset"]] = 0

    return starting_biomasses

def add_species_to_map_even(world_array, world_data, seed=None):
    if seed is None:
        seed = int(random.random() * 100000)
    rng = np.random.default_rng(seed)

    world_data[:, :, 4] = 0  # Reset (e.g., plankton cluster marker layer)
    MIN_CELLS_WITH_BIOMASS = int(const.WORLD_SIZE * const.WORLD_SIZE * 0.15)
    tiny_world = const.WORLD_SIZE <= 9
    if tiny_world:
        MIN_CELLS_WITH_BIOMASS = const.WORLD_SIZE * const.WORLD_SIZE
    COD_COVERAGE_FACTOR = 0.5
    PLANKTON_COVERAGE_FACTOR = 2.5

    starting_biomasses = {}

    # Prepare masks
    water_mask = world_array[:, :, Terrain.WATER.value] == 1

    # First, place plankton (base species)
    for species, properties in const.SPECIES_MAP.items():
        properties["starting_biomass"] = properties["original_starting_biomass"]
        starting_biomass = properties["starting_biomass"] * smooth_skewed_random()
        starting_biomasses[species] = starting_biomass

        biomass_offset = properties["biomass_offset"]
        energy_offset = properties["energy_offset"]
        hardcoded = properties.get("hardcoded_logic", False)

        # Determine eligible placement cells
        eligible_mask = water_mask.copy()
        if not tiny_world and species in ("herring", "sprat"):
            for other_species, other_props in const.SPECIES_MAP.items():
                if other_props.get("hardcoded_logic", False):  # e.g., plankton
                    eligible_mask &= world_array[:, :, other_props["biomass_offset"]] == 0
        
        eligible_positions = np.argwhere(eligible_mask)
        if species == "cod":
            num_points = max(int(MIN_CELLS_WITH_BIOMASS * COD_COVERAGE_FACTOR), 1)
        elif species == "plankton":
            num_points = max(int(MIN_CELLS_WITH_BIOMASS * PLANKTON_COVERAGE_FACTOR), 1)
        else:
            num_points = max(MIN_CELLS_WITH_BIOMASS, 1)
        num_points = min(num_points, eligible_positions.shape[0])

        if eligible_positions.shape[0] < num_points:
            num_points = eligible_positions.shape[0]

        chosen = eligible_positions[rng.choice(eligible_positions.shape[0], size=num_points, replace=False)]
        biomass_per_cell = starting_biomass / num_points

        for x, y in chosen:
            world_array[x, y, biomass_offset] = biomass_per_cell
            world_array[x, y, energy_offset] = const.MAX_ENERGY
            # world_array[x, y, energy_offset] = const.MAX_ENERGY * rng.uniform(0, 1)
            if hardcoded:
                world_data[x, y, 1] = 1
                world_data[x, y, 2] = properties["hardcoded_rules"]["respawn_delay"]

    # Reset smell channels
    for species, properties in const.SPECIES_MAP.items():
        world_array[:, :, properties["smell_offset"]] = 0

    return starting_biomasses

def read_map_from_file(folder_path, seed=None):
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

    if const.WORLD_SIZE <= 9:
        for x in range(const.WORLD_SIZE):
            for y in range(const.WORLD_SIZE):
                world_array[x, y, :3] = np.array([0, 1, 0], dtype=np.float32)
                world_data[x, y, 3] = 1
    else:
        for x in range(const.WORLD_SIZE):
            for y in range(const.WORLD_SIZE):
                color = pixels[x, y][:3]
                depth_value = depth_pixels[x, y] / 255.0
                world_data[x, y, 3] = depth_value
                if color == palette["water"]:
                    world_array[x, y, :3] = np.array([0, 1, 0], dtype=np.float32)
                else:
                    world_array[x, y, :3] = np.array([1, 0, 0], dtype=np.float32)

    # starting_biomasses = add_species_to_map(world_array, world_data, seed=seed)
    starting_biomasses = add_species_to_map_even(world_array, world_data, seed=seed)

    return np.ascontiguousarray(world_array), np.ascontiguousarray(world_data), starting_biomasses

def create_map_from_noise(static=False, seed_value=None):
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
