import json
from queue import Queue, Empty
import pygame
import math
from lib.world import Terrain
import lib.constants as const
from lib.constants import SPECIES_MAP  # Add this import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MaxNLocator

# Visualization settings
WORLD_WIDTH = 500
WORLD_HEIGHT = 500
GRAPH_WIDTH = 500
GEN_GRAPH_HEIGHT = 200
CELL_SIZE = math.floor(WORLD_HEIGHT / const.WORLD_SIZE)

SCREEN_WIDTH = WORLD_WIDTH + GRAPH_WIDTH
SCREEN_HEIGHT = WORLD_HEIGHT + GEN_GRAPH_HEIGHT

# Cache for terrain surface and graph surfaces
terrain_surface_cache = None
biomass_graph_cache = None
energy_graph_cache = None
generation_graph_cache = None

visualization_queue = Queue()

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ecosystem simulation")
    return screen

# Cache and draw the terrain only once to optimize performance
def draw_terrain(world_tensor, world_data, display_current=False):
    global terrain_surface_cache
    if terrain_surface_cache is not None:
        return terrain_surface_cache

    terrain_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
    
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):

            # Extract terrain information from the tensor (one-hot encoded: first three values)
            terrain = world_tensor[x, y, :3]
            
            # Determine if the cell is water or land based on the one-hot encoding
            if terrain[Terrain.WATER.value] == 1:
                color = (0, 0, 255)  # Blue for water
            else:
                color = (0, 255, 0)  # Green for land

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
def plot_generations(generations_data):
    global generation_graph_cache

    plt.figure(figsize=(16, 12))  # Create a new figure for the plot

    # Lists to store statistical measures for each generation
    average_fitness = []
    median_fitness = []
    top_percentiles = []
    bottom_percentiles = []

    percentiles = [25, 75]  # Define the percentiles to calculate

    # Plot each generation's fitness
    for generation, fitness_values in enumerate(generations_data):
        # Add jitter to x-coordinates to improve visualization
        jitter = np.random.normal(0, 0.05, size=len(fitness_values))  # Adjust the standard deviation as needed
        x_values = generation + jitter

        plt.scatter(x_values, fitness_values, alpha=0.6, color='green', label='Fitness' if generation == 0 else "")
        
        # Calculate and store statistical measures
        avg_fitness = np.mean(fitness_values)
        med_fitness = np.median(fitness_values)
        bottom_percentile = np.percentile(fitness_values, percentiles[0])
        top_percentile = np.percentile(fitness_values, percentiles[1])

        average_fitness.append(avg_fitness)
        median_fitness.append(med_fitness)
        bottom_percentiles.append(bottom_percentile)
        top_percentiles.append(top_percentile)

    # Plot the average fitness line
    plt.plot(range(len(average_fitness)), average_fitness, color='red', label='Average Fitness', linewidth=2)

    # Plot the median fitness line
    plt.plot(range(len(median_fitness)), median_fitness, color='blue', label='Median Fitness', linewidth=2)

    # Plot the top and bottom percentiles as a shaded area
    plt.fill_between(range(len(average_fitness)), bottom_percentiles, top_percentiles, color='gray', alpha=0.2, label=f'{percentiles[0]}th to {percentiles[1]}th Percentile')

    # Adjust x-axis ticks to prevent overcrowding
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness of Agents Over Generations')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))  # Adjust 'nbins' as needed

    plt.legend()
    plt.tight_layout()

    # Save plot as an image
    plt.savefig(f'{const.CURRENT_FOLDER}/fitness_plot.png')
    plt.close()

    # Load the saved plot image using Pygame and cache it
    generation_graph_cache = pygame.image.load(f'{const.CURRENT_FOLDER}/fitness_plot.png')

# Plot and cache biomass graph
def plot_biomass(agents_data):
    global biomass_graph_cache

    plt.figure(figsize=(5, 5))

    # Count total number of plots to get enough colors
    total_plots = sum(len(evals) for evals in agents_data.values())

    # Get a colormap to differentiate agents and evals
    colors = cm.get_cmap('tab20', total_plots)

    # Plot each agent's data
    legend_patches = []
    idx = 0  # Color index
    for agent_index, evals in agents_data.items():
        for eval_index, data in evals.items():
            plt.plot(data['steps'], data['cod_alive'], label=f'Agent {agent_index} Eval {eval_index} COD', color="black")
            plt.plot(data['steps'], data['anchovy_alive'], label=f'Agent {agent_index} Eval {eval_index} ANCHOVY', color="red", linestyle='--')
            plt.plot(data['steps'], data['plankton_alive'], label=f'Agent {agent_index} Eval {eval_index} PLANKTON', color="green", linestyle=':')
            legend_patches.append(mpatches.Patch(color=colors(idx), label=f'Agent {agent_index} Eval {eval_index}'))
            idx += 1

    plt.xlabel('Steps')
    plt.ylabel('Number of Species Alive')
    plt.title('Species Population Over Time')

    # Show the custom legend (ignoring linestyles or species)
    plt.legend(handles=legend_patches, title="Agent Evaluations", loc='upper right')

    plt.tight_layout()
    plt.savefig('world_graph.png')
    plt.close()

    # Load the saved plot image using Pygame and cache it
    biomass_graph_cache = pygame.image.load('world_graph.png')

# Redraw the world with species biomass
def draw_world(screen, world_tensor, world_data):
    # Remove padding if present
    world_tensor = world_tensor[1:-1, 1:-1]

    screen.fill((255, 255, 255))

    terrain_surface = draw_terrain(world_tensor, world_data, False)
    screen.blit(terrain_surface, (0, 0))  # Blit the terrain surface once

    # Create a new surface for smell visualization
    smell_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)

    # Calculate maximum smell values for normalization
    max_smell_values = {species: world_tensor[:, :, properties["smell_offset"]].max().item() / 2 for species, properties in SPECIES_MAP.items()}

    # Loop over the world grid for smell visualization
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            # Get smell values for each species
            smell_values = {species: min(world_tensor[x, y, properties["smell_offset"]].item() / max_smell_values[species], 1.0) for species, properties in SPECIES_MAP.items()}

            # Define colors for smells (RGBA with alpha for transparency)
            smell_colors = {
                species: (0, 255, 0, int(smell_values[species] * 150)) if species == "plankton" else
                         (255, 0, 0, int(smell_values[species] * 150)) if species == "anchovy" else
                         (0, 0, 0, int(smell_values[species] * 150)) for species in SPECIES_MAP.keys()
            }

            # Draw smell overlay
            for species, color in smell_colors.items():
                if smell_values[species] > 0:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    smell_surface.fill(color, rect)

    # Blit the smell surface onto the main screen
    screen.blit(smell_surface, (0, 0))


    # Calculate maximum biomass for each species
    max_biomass_values = {species: world_tensor[:, :, properties["biomass_offset"]].max().item() / 2 for species, properties in SPECIES_MAP.items()}

    # Define minimum and maximum radius for the circles
    min_radius = 1
    max_radius = CELL_SIZE // 2  # Subtracting 2 to prevent overlap with cell borders

    min_radius_plankton = 0
    max_radius_plankton = CELL_SIZE // 4

    # Loop over the world grid
    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            # Define offsets for species visualization within the cell
            offsets = {
                species: (-CELL_SIZE // 4, -CELL_SIZE // 4) if species == "plankton" else
                         (CELL_SIZE // 4, -CELL_SIZE // 4) if species == "anchovy" else
                         (0, CELL_SIZE // 4) for species in SPECIES_MAP.keys()
            }

            # Extract biomass for each species at the current cell
            species_biomass = {species: world_tensor[x, y, properties["biomass_offset"]].item() for species, properties in SPECIES_MAP.items()}

            # Draw biomass for each species
            for species, biomass in species_biomass.items():
                if biomass > 0:
                    # Determine min and max radius for the species
                    if species == "plankton":
                        min_r = min_radius_plankton
                        max_r = max_radius_plankton
                    else:
                        min_r = min_radius
                        max_r = max_radius

                    # Calculate the radius proportionally
                    max_biomass = max_biomass_values[species]
                    if max_biomass > 0:
                        radius = min_r + int((max_r - min_r) * (biomass / max_biomass))
                    else:
                        radius = min_r  # Default to minimum radius if max_biomass is zero

                    # Ensure radius is within bounds
                    radius = max(min_r, min(radius, max_r))

                    # Calculate circle center based on offsets
                    offset_x, offset_y = offsets[species]
                    circle_center = (
                        x * CELL_SIZE + CELL_SIZE // 2 + offset_x,
                        y * CELL_SIZE + CELL_SIZE // 2 + offset_y
                    )

                    # Draw the circle representing the species biomass
                    color = (0, 255, 0) if species == "plankton" else (255, 0, 0) if species == "anchovy" else (0, 0, 0)
                    pygame.draw.circle(screen, color, circle_center, radius)

    # Draw cached biomass graph
    if biomass_graph_cache:
        screen.blit(biomass_graph_cache, (WORLD_WIDTH, 0))  # Adjust the position as needed
    if energy_graph_cache:
        screen.blit(energy_graph_cache, (WORLD_WIDTH, 0))

    # Perform screen update only once at the end
    pygame.display.flip()

def draw_world_detailed(screen, world_tensor):
    # remove padding
    world_tensor = world_tensor[1:-1, 1:-1]

    """
    Visualize the world based on the tensor representation, displaying both biomass and energy.
    The world_tensor has shape (WORLD_SIZE, WORLD_SIZE, N) where:
    - The first 3 values represent the terrain (one-hot encoded).
    - The next 3 values represent the biomass of plankton, anchovy, and cod, respectively.
    - The next 3 values represent the energy of plankton, anchovy, and cod, respectively.
    """
    screen.fill((255, 255, 255))

    terrain_surface = draw_terrain(world_tensor)
    screen.blit(terrain_surface, (0, 0))  # Blit the terrain surface once

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            # Define horizontal positions for circles and bars within the cell
            species_positions = {
                species: (-CELL_SIZE // 3, -CELL_SIZE // 4) if species == "plankton" else
                         (0, -CELL_SIZE // 4) if species == "anchovy" else
                         (CELL_SIZE // 3, -CELL_SIZE // 4) for species in SPECIES_MAP.keys()
            }
            
            # Extract biomass and energy information for each species
            species_biomass = {species: world_tensor[x, y, properties["biomass_offset"]].item() for species, properties in SPECIES_MAP.items()}
            species_energy = {species: world_tensor[x, y, properties["energy_offset"]].item() for species, properties in SPECIES_MAP.items()}

            # Dictionary to map species to their biomass, energy, and colors
            species_data = {
                species: (species_biomass[species], species_energy[species], (0, 255, 0), (0, 200, 0)) if species == "plankton" else
                         (species_biomass[species], species_energy[species], (255, 0, 0), (200, 0, 0)) if species == "anchovy" else
                         (species_biomass[species], species_energy[species], (0, 0, 0), (50, 50, 50)) for species in SPECIES_MAP.keys()
            }

            # Draw biomass circles in a row at the top
            for species, (biomass, energy, biomass_color, energy_color) in species_data.items():
                offset_x, offset_y = species_positions[species]
                center_x = x * CELL_SIZE + CELL_SIZE // 2 + offset_x
                center_y = y * CELL_SIZE + CELL_SIZE // 2 + offset_y

                # Draw biomass as circles (aligned in a row at the top)
                if biomass > 0:
                    radius = min(CELL_SIZE // 6, int(math.sqrt(biomass) * 0.75))  # Smaller circles to fit in a row
                    pygame.draw.circle(screen, biomass_color, (center_x, center_y), radius)

                # Draw energy bars in a row below the circles
                if energy > 0:
                    bar_height = int(CELL_SIZE * 0.1)  # Set bar height smaller to fit in the cell
                    max_bar_width = CELL_SIZE // 3     # Max width per species to fit three bars in a row
                    # set bar width based on energy level ( 100 max width, and 0 no width )
                    bar_width = max(1, min(int(max_bar_width * (energy / 100)), max_bar_width))
                    bar_rect = pygame.Rect(center_x - bar_width // 2, center_y + CELL_SIZE // 4, bar_width, bar_height)
                    pygame.draw.rect(screen, energy_color, bar_rect)

    # Draw cached biomass graph
    if biomass_graph_cache:
        screen.blit(biomass_graph_cache, (WORLD_WIDTH, 0))  # Adjust the position as needed
    if energy_graph_cache:
        screen.blit(energy_graph_cache, (WORLD_WIDTH, 0))

    # Perform screen update only once at the end
    pygame.display.flip()
