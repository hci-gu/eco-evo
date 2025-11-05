import json
from queue import Queue, Empty
import pygame
import math
from lib.config.settings import Settings
from lib.config.const import SPECIES
from lib.model import MODEL_OFFSETS
from lib.world import Terrain
import lib.config.const as const
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import imageio
import random
import time
from matplotlib.ticker import MaxNLocator

# Visualization settings
WORLD_WIDTH = 500
WORLD_HEIGHT = 500
GRAPH_WIDTH = 500
GEN_GRAPH_HEIGHT = 200

SCREEN_WIDTH = WORLD_WIDTH + GRAPH_WIDTH
SCREEN_HEIGHT = WORLD_HEIGHT + GEN_GRAPH_HEIGHT

SPECIES_COLORS = {
    "plankton": [100, 220, 100],
    "herring": [220, 60, 60],
    "sprat": [240, 140, 60],
    "cod": [40, 40, 40]
}

# Cache for terrain surface and graph surfaces
terrain_surface_cache = None
biomass_graph_cache = None
energy_graph_cache = None
generation_graph_cache = None

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ecosystem simulation")
    return screen

def shutdown_pygame():
    pygame.quit()

def interpolate_color(value, color1, color2):
    return tuple(
        int(color1[i] + (color2[i] - color1[i]) * value)
        for i in range(3)
    )

# Cache and draw the terrain only once to optimize performance
def draw_terrain(settings: Settings, world_tensor, world_data, display_current=False):
    global terrain_surface_cache
    if terrain_surface_cache is not None:
        return terrain_surface_cache

    terrain_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))

    CELL_SIZE = math.floor(WORLD_HEIGHT / settings.world_size)
    for x in range(settings.world_size):
        for y in range(settings.world_size):
            # if (x == 0 or x == const.WORLD_SIZE - 1) or (y == 0 or y == const.WORLD_SIZE - 1):
            #     continue

            # Extract terrain information from the tensor (one-hot encoded: first three values)
            terrain = world_tensor[x, y, :3]
            depth_value = world_data[x, y, 3]
            
            # Determine if the cell is water or land based on the one-hot encoding
            if terrain[Terrain.WATER.value] == 1:
                shallow_color = (15, 79, 230)  # Light Blue
                deep_color = (0, 0, 44)         # Dark Blue
                color = interpolate_color(1 - depth_value, shallow_color, deep_color)
            else:
                color = (84, 140, 47)  # Green for land
                # add some noise to the land color
                color = tuple(int(c + (random.random() - 0.5) * 20) for c in color)

            # Draw the terrain
            pygame.draw.rect(terrain_surface, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # If the display_current flag is True and the cell is water, draw an arrow for the current
            if display_current and terrain[Terrain.WATER.value] == 1:
                # Extract the current angle from the world tensor (index 9)
                current_angle = world_data[x, y, 0]

                # Compute the arrow direction based on the current angle
                arrow_length = CELL_SIZE // 2
                end_x = x * CELL_SIZE + CELL_SIZE // 2 + int(arrow_length * math.cos(current_angle))
                end_y = y * CELL_SIZE + CELL_SIZE // 2 + int(arrow_length * math.sin(current_angle))

                # Draw a simple arrow representing the current direction
                pygame.draw.line(terrain_surface, (255, 255, 255), 
                                 (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 
                                 (end_x, end_y), 2)
    
    # Cache the terrain surface for reuse
    terrain_surface_cache = terrain_surface
    return terrain_surface

# Plot and cache generations graph
def plot_generations(settings: Settings, generations_data):
    global generation_graph_cache

    plt.figure(figsize=(16, 12))  # Create a new figure for the plot

    # Check if generations_data[0] is an object (dict) or an array (list)
    if isinstance(generations_data[0], dict):
        # Multiple species case
        species_list = list(generations_data[0].keys())
    else:
        # Single species case
        species_list = ["Single Species"]

    # Initialize dictionaries to store statistical measures for each species over generations
    species_stats = {}
    for sp in species_list:
        species_stats[sp] = {
            "average": [],
            "median": [],
            "bottom": [],
            "top": []
        }

    # Assign a unique color for each species using a colormap
    colors = plt.cm.tab10.colors  # A set of 10 distinct colors
    species_colors = {sp: colors[i % len(colors)] for i, sp in enumerate(species_list)}

    # Plot each generation's fitness data for each species
    for generation_index, generation in enumerate(generations_data):
        for sp in species_list:
            fitness_values = []
            if sp == "Single Species":
                fitness_values = generation
            else:
                fitness_values = generation[sp]
            # Add jitter to x-coordinates for better visualization
            jitter = np.random.normal(0, 0.05, size=len(fitness_values))
            x_values = generation_index + jitter

            # Plot scatter points for the species (only label on the first generation)
            plt.scatter(x_values, fitness_values, alpha=0.6, color=species_colors[sp],
                        label=sp if generation_index == 0 else "")

            # Compute statistical measures
            avg_fitness = np.mean(fitness_values)
            med_fitness = np.median(fitness_values)
            bottom_percentile = np.percentile(fitness_values, 25)
            top_percentile = np.percentile(fitness_values, 75)

            # Append statistics for this generation
            species_stats[sp]["average"].append(avg_fitness)
            species_stats[sp]["median"].append(med_fitness)
            species_stats[sp]["bottom"].append(bottom_percentile)
            species_stats[sp]["top"].append(top_percentile)

    # x-axis values corresponding to generations
    generations = range(len(generations_data))

    # Plot the average fitness line and fill between for each species
    for sp in species_list:
        plt.plot(generations, species_stats[sp]["average"], color=species_colors[sp],
                 label=f"{sp} Avg", linewidth=2)
        plt.fill_between(generations, species_stats[sp]["bottom"], species_stats[sp]["top"],
                         color=species_colors[sp], alpha=0.2,
                         label=f"{sp} 25th-75th Percentile")

    # Configure axes and title
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness of Agents Over Generations')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))  # Prevent overcrowding of x-axis ticks

    plt.legend()
    plt.tight_layout()

    # Save the plot as an image and update the cache
    plt.savefig(f'{settings.folder}/fitness_plot.png')
    plt.close()

    generation_graph_cache = pygame.image.load(f'{settings.folder}/fitness_plot.png')


# Plot and cache biomass graph
def plot_biomass(agents_data):
    global biomass_graph_cache

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(5, 5))

    # Count total number of plots to get enough colors
    total_plots = sum(len(evals) for evals in agents_data.values())

    # Get a colormap to differentiate agents and evals
    colors = cm.get_cmap('tab20', total_plots)

    # Plot each agent's data
    legend_patches = []
    idx = 0  # Color index
    for agent_index, evals in agents_data.items():
        for eval_index, _ in evals.items():
            data = evals[eval_index]
            plt.plot(data['steps'], data['cod_alive'], label=f'Agent {agent_index} Eval {eval_index} COD', color="black")
            plt.plot(data['steps'], data['herring_alive'], label=f'Agent {agent_index} Eval {eval_index} HERRING', color="red", linestyle='-')
            plt.plot(data['steps'], data['sprat_alive'], label=f'Agent {agent_index} Eval {eval_index} sprat', color="orange", linestyle='-')
            plt.plot(data['steps'], data['plankton_alive'], label=f'Agent {agent_index} Eval {eval_index} PLANKTON', color="green", linestyle='-')
            
            # plt.plot(data['steps'], data['cod_energy'], label=f'Agent {agent_index} Eval {eval_index} COD', color="black", linestyle='-.')
            # plt.plot(data['steps'], data['herring_energy'], label=f'Agent {agent_index} Eval {eval_index} HERRING', color="red", linestyle='-.')
            # plt.plot(data['steps'], data['sprat_energy'], label=f'Agent {agent_index} Eval {eval_index} sprat', color="orange", linestyle='-.')
            # plt.plot(data['steps'], data['plankton_energy'], label=f'Agent {agent_index} Eval {eval_index} PLANKTON', color="green", linestyle='-.')
            idx += 1

    # for each species
    for species in SPECIES:
        color = SPECIES_COLORS[species]
        legend_patches.append(mpatches.Patch(color=[c/255 for c in color], label=species.capitalize()))

    plt.xlabel('Steps')
    plt.ylabel('Number of Species Alive')
    plt.title('Species Population Over Time')

    # Show the custom legend (ignoring linestyles or species)
    plt.legend(handles=legend_patches, title="Species", loc='upper right')

    plt.tight_layout()
    plt.savefig('world_graph.png')
    plt.close()

    # Load the saved plot image using Pygame and cache it
    biomass_graph_cache = pygame.image.load('world_graph.png')

# def draw_actions_values(screen, world_data):
#     # world_data = world_data[1:-1, 1:-1]
#     font = pygame.font.SysFont('Arial', 10)  # Adjust as needed

#     max_value = world_data[:, :, 4].max()

#     for x in range(const.WORLD_SIZE):
#         for y in range(const.WORLD_SIZE):
#             cell_center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
#             number_of_actions = int(world_data[x, y, 4])

#             if number_of_actions > 0:
#                 # color the text from red to green (if value is close to max its green, if its close to 0 its red)
#                 closeness_to_max = number_of_actions / max_value
#                 red_color = (255, 0, 0)
#                 green_color = (0, 255, 0)
#                 color = interpolate_color(closeness_to_max, red_color, green_color)
#                 text = font.render(str(number_of_actions), True, color)

#                 screen.blit(text, (cell_center[0] - 5, cell_center[1] - 5))


counter = 0
def draw_world(settings: Settings,screen, world_tensor, world_data):
    global counter
    # Remove padding if present
    world_tensor = world_tensor[1:-1, 1:-1]
    world_data = world_data[1:-1, 1:-1]

    screen.fill((255, 255, 255))

    terrain_surface = draw_terrain(settings, world_tensor, world_data, False)
    screen.blit(terrain_surface, (0, 0))  # Draw the terrain once

    CELL_SIZE = math.floor(WORLD_HEIGHT / settings.world_size)

    # Calculate maximum biomass for each species
    max_biomass_values = {}
    for species in SPECIES:
        max_val = world_tensor[:, :, MODEL_OFFSETS[species]["biomass"]].max().item()
        max_biomass_values[species] = max_val if max_val > 0 else 1

    # Define radius constraints
    # We allow circles to be large enough to exceed cell boundaries.
    # Keep a moderate max radius, but not too large.
    min_radius = 1
    max_radius = CELL_SIZE // 2.5
    min_radius_plankton = 1
    max_radius_plankton = CELL_SIZE // 3

    # Small offsets to differentiate species within a cell
    offsets = {
        "plankton": (-CELL_SIZE // 6, -CELL_SIZE // 6),
        "herring": (CELL_SIZE // 6, -CELL_SIZE // 6),
        "sprat": (-CELL_SIZE // 6, CELL_SIZE // 6),
        "cod": (CELL_SIZE // 6, CELL_SIZE // 6)
    }

    # We'll create a large surface for each circle to avoid cropping.
    # For safety, let's make it 3x cell size.
    LARGE_SURFACE_SIZE = CELL_SIZE * 3
    half_ls = LARGE_SURFACE_SIZE // 2  # Half of large surface size
    cell_center_offset = (CELL_SIZE // 2, CELL_SIZE // 2)

    for x in range(settings.world_size):
        for y in range(settings.world_size):
            # Extract biomass for each species at the current cell
            species_biomass = {
                species: world_tensor[x, y, MODEL_OFFSETS[species]["biomass"]].item() 
                for species in SPECIES
            }

            # Calculate the cell center on the main screen
            cell_center = (
                x * CELL_SIZE + cell_center_offset[0],
                y * CELL_SIZE + cell_center_offset[1]
            )

            # Draw each species in the cell
            for species, biomass in species_biomass.items():
                if biomass > 0:
                    # Determine min and max radius for the species
                    if species == "plankton":
                        min_r = min_radius_plankton
                        max_r = max_radius_plankton
                    else:
                        min_r = min_radius
                        max_r = max_radius

                    max_biomass = max_biomass_values[species]
                    # Use sqrt scaling again or linear if preferred
                    radius = min_r + (max_r - min_r) * (math.sqrt(biomass / max_biomass))
                    radius = int(radius)

                    offset_x, offset_y = offsets[species]

                    # Create a large transparent surface
                    circle_surface = pygame.Surface((LARGE_SURFACE_SIZE, LARGE_SURFACE_SIZE), pygame.SRCALPHA)
                    circle_surface = circle_surface.convert_alpha()
                    circle_surface.fill((0, 0, 0, 0))  # fully transparent background

                    # Draw the circle at the center of this large surface
                    # The circle center on the large surface:
                    circle_center = (half_ls + offset_x, half_ls + offset_y)
                    color = SPECIES_COLORS[species]

                    # Add partial transparency so overlapping species are visible
                    # For example, alpha = 180 for semi-transparency
                    # lerp opacity from 0 to 255 based on radius
                    opacity = int(255 * (radius / max_radius))
                    circle_color = (color[0], color[1], color[2], opacity)

                    pygame.draw.circle(circle_surface, circle_color, circle_center, radius)

                    # Now blit the large surface onto the screen
                    # We must align large surface center with the cell center
                    # Since large surface center is at half_ls, half_ls,
                    # top-left corner should be cell_center - (half_ls, half_ls)
                    blit_pos = (cell_center[0] - half_ls, cell_center[1] - half_ls)
                    screen.blit(circle_surface, blit_pos)

    # draw_actions_values(screen, world_data)

    # Draw cached biomass and energy graphs if they exist
    if biomass_graph_cache:
        screen.blit(biomass_graph_cache, (WORLD_WIDTH, 0))
    if energy_graph_cache:
        screen.blit(energy_graph_cache, (WORLD_WIDTH, 0))

    pygame.display.flip()

