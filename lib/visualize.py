import pygame
import math
from lib.world import Terrain, Species
import lib.constants as const
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np

# Visualization settings
WORLD_WIDTH = 500
WORLD_HEIGHT = 500
GRAPH_WIDTH = 500
GEN_GRAPH_HEIGHT = 200
CELL_SIZE = math.floor(WORLD_HEIGHT / const.WORLD_SIZE)

SCREEN_WIDTH = WORLD_WIDTH + GRAPH_WIDTH
SCREEN_HEIGHT = WORLD_HEIGHT + GEN_GRAPH_HEIGHT

# Agent and generation data for visualization
agents_data = {}
generations_data = []

cod_alive = []
anchovy_alive = []
plankton_alive = []
steps_list = []

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

screen = init_pygame()

def reset_terrain_cache():
    global terrain_surface_cache
    terrain_surface_cache = None

# Cache and draw the terrain only once to optimize performance
def draw_terrain(world_tensor):
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
    
    # Cache the terrain surface for reuse
    terrain_surface_cache = terrain_surface
    return terrain_surface

# Update generation data
def update_generations_data():
    fitness_values = [max(agent['steps']) for (key, agent) in agents_data.items()]
    generations_data.append(fitness_values)

def reset_plot():
    update_generations_data()
    reset_terrain_cache()
    agents_data.clear()

# Plot and cache generations graph
def plot_generations():
    global generation_graph_cache

    plt.figure(figsize=(16, 12))  # Create a new figure for the plot

    # List to store the average fitness for each generation
    average_fitness = []

    # Plot each generation's fitness
    for generation, fitness_values in enumerate(generations_data):
        plt.scatter([generation] * len(fitness_values), fitness_values, alpha=0.6, label='Fitness' if generation == 0 else "")
        
        # Calculate and store the average fitness for this generation
        avg_fitness = np.mean(fitness_values)
        average_fitness.append(avg_fitness)

    # Plot the average fitness line
    plt.plot(range(len(average_fitness)), average_fitness, color='red', label='Average Fitness', linewidth=2)
    plt.xticks(range(len(generations_data)))  # Set x-axis ticks to integers

    # Set plot labels and title
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness of Agents Over Generations')
    plt.legend()
    plt.tight_layout()

    # Save plot as an image
    plt.savefig(f'{const.CURRENT_FOLDER}/fitness_plot.png')
    plt.close()

    # Load the saved plot image using Pygame and cache it
    generation_graph_cache = pygame.image.load(f'{const.CURRENT_FOLDER}/fitness_plot.png')

# Plot and cache biomass graph
def plot_biomass():
    global biomass_graph_cache

    plt.figure(figsize=(5, 5))

    # Get a colormap to differentiate agents
    colors = cm.get_cmap('tab10', len(agents_data))

    # Plot each agent's data
    legend_patches = []
    for idx, (agent, data) in enumerate(agents_data.items()):
        plt.plot(data['steps'], data['cod_alive'], label=f'Agent {agent} COD', color=colors(idx))
        plt.plot(data['steps'], data['anchovy_alive'], label=f'Agent {agent} ANCHOVY', color=colors(idx), linestyle='--')
        plt.plot(data['steps'], data['plankton_alive'], label=f'Agent {agent} PLANKTON', color=colors(idx), linestyle=':')
        legend_patches.append(mpatches.Patch(color=colors(idx), label=f'Agent {agent}'))
    
    plt.xlabel('Steps')
    plt.ylabel('Number of Species Alive')
    plt.title('Species Population Over Time')

    # Show the custom legend (ignoring linestyles or species)
    plt.legend(handles=legend_patches, title="Agent Colors", loc='upper right')

    plt.tight_layout()
    plt.savefig('world_graph.png')
    plt.close()

    # Load the saved plot image using Pygame and cache it
    biomass_graph_cache = pygame.image.load('world_graph.png')

# Plot and cache biomass graph
def plot_energy():
    global energy_graph_cache

    plt.figure(figsize=(5, 5))

    # Get a colormap to differentiate agents
    colors = cm.get_cmap('tab10', len(agents_data))

    # Plot each agent's data
    legend_patches = []
    for idx, (agent, data) in enumerate(agents_data.items()):
        plt.plot(data['steps'], data['energy_cod'], label=f'Agent {agent} COD', color=colors(idx))
        plt.plot(data['steps'], data['energy_anchovy'], label=f'Agent {agent} ANCHOVY', color=colors(idx), linestyle='--')
        plt.plot(data['steps'], data['energy_plankton'], label=f'Agent {agent} PLANKTON', color=colors(idx), linestyle=':')
        legend_patches.append(mpatches.Patch(color=colors(idx), label=f'Agent {agent}'))
    
    plt.xlabel('Steps')
    plt.ylabel('Total energy of species')
    plt.title('Species Population Over Time')

    # Show the custom legend (ignoring linestyles or species)
    plt.legend(handles=legend_patches, title="Agent Colors", loc='upper right')

    plt.tight_layout()
    plt.savefig('energy_graph.png')
    plt.close()

    # Load the saved plot image using Pygame and cache it
    energy_graph_cache = pygame.image.load('energy_graph.png')

# Redraw the world with species biomass
def draw_world(world_tensor):
    # remove padding
    world_tensor = world_tensor[1:-1, 1:-1]

    """
    Visualize the world based on the tensor representation.
    The world_tensor has shape (WORLD_SIZE, WORLD_SIZE, 6) where:
    - The first 3 values represent the terrain (one-hot encoded).
    - The last 3 values represent the biomass of plankton, anchovy, and cod, respectively.
    """
    screen.fill((255, 255, 255))

    terrain_surface = draw_terrain(world_tensor)
    screen.blit(terrain_surface, (0, 0))  # Blit the terrain surface once

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            # x = x + 1
            # y = y + 1
            # Initialize an offset position for each species in the same cell
            offsets = {
                Species.PLANKTON: (-CELL_SIZE // 4, -CELL_SIZE // 4),
                Species.ANCHOVY: (CELL_SIZE // 4, -CELL_SIZE // 4),
                Species.COD: (0, CELL_SIZE // 4)
            }
            
            # Extract biomass information for each species
            plankton_biomass = world_tensor[x, y, const.OFFSETS_BIOMASS_PLANKTON].item()  # Biomass for plankton
            anchovy_biomass = world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY].item()  # Biomass for anchovy
            cod_biomass = world_tensor[x, y, const.OFFSETS_BIOMASS_COD].item()  # Biomass for cod

            anchovy_energy = world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY].item()  # Biomass for anchovy
            cod_energy = world_tensor[x, y, const.OFFSETS_ENERGY_COD].item()  # Biomass for cod

            # Dictionary to map species to their biomass and color
            # anchovy_opacity = min(255, max(0, int(anchovy_energy * 2.55)))
            # cod_opacity = min(255, max(0, int(cod_energy * 2.55)))
            species_biomass = {
                Species.PLANKTON: (plankton_biomass, (0, 255, 0)),  # Green for plankton
                Species.ANCHOVY: (anchovy_biomass, (255, 0, 0)), # Red for anchovy
                Species.COD: (cod_biomass, (0, 0, 0)) # Black for cod
            }
            
            # Draw biomass for each species
            for species, (biomass, color) in species_biomass.items():
                if biomass > 0:
                    # Calculate the circle center and radius based on biomass
                    offset_x, offset_y = offsets[species]
                    circle_center = (x * CELL_SIZE + CELL_SIZE // 2 + offset_x, y * CELL_SIZE + CELL_SIZE // 2 + offset_y)
                    radius = min(CELL_SIZE // 2, int(math.sqrt(biomass) * 0.75))

                    # Draw the circle representing species biomass with opacity
                    pygame.draw.circle(screen, color, circle_center, radius)

    # Draw cached biomass graph
    if biomass_graph_cache:
        screen.blit(biomass_graph_cache, (WORLD_WIDTH, 0))  # Adjust the position as needed
    if energy_graph_cache:
        screen.blit(energy_graph_cache, (WORLD_WIDTH, 0))

    # Perform screen update only once at the end
    pygame.display.flip()

def draw_world_mask(world_tensor, mask):
    print(mask)
    # mask is a tensor of the same size as the world_tensor with 1s and 0s
    # create a red border around the cells with 1s in the mask

    # remove padding
    world_tensor = world_tensor[1:-1, 1:-1]

    for x in range(const.WORLD_SIZE):
        for y in range(const.WORLD_SIZE):
            if mask[x, y] == 1:
                pygame.draw.rect(screen, (255, 0, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)


def draw_world_detailed(world_tensor):
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
                Species.PLANKTON: (-CELL_SIZE // 3, -CELL_SIZE // 4),  # Left circle
                Species.ANCHOVY: (0, -CELL_SIZE // 4),                 # Center circle
                Species.COD: (CELL_SIZE // 3, -CELL_SIZE // 4)         # Right circle
            }
            
            # Extract biomass and energy information for each species
            plankton_biomass = world_tensor[x, y, const.OFFSETS_BIOMASS_PLANKTON].item()  # Biomass for plankton
            anchovy_biomass = world_tensor[x, y, const.OFFSETS_BIOMASS_ANCHOVY].item()  # Biomass for anchovy
            cod_biomass = world_tensor[x, y, const.OFFSETS_BIOMASS_COD].item()  # Biomass for cod

            # plankton_energy = world_tensor[x, y, const.OFFSETS_ENERGY_PLANKTON].item()  # Energy for plankton
            anchovy_energy = world_tensor[x, y, const.OFFSETS_ENERGY_ANCHOVY].item()  # Energy for anchovy
            cod_energy = world_tensor[x, y, const.OFFSETS_ENERGY_COD].item()  # Energy for cod

            # Dictionary to map species to their biomass, energy, and colors
            species_data = {
                # Species.PLANKTON: (plankton_biomass, plankton_energy, (0, 255, 0), (0, 200, 0)),  # Green for plankton biomass, darker green for energy
                Species.ANCHOVY: (anchovy_biomass, anchovy_energy, (255, 0, 0), (200, 0, 0)),    # Red for anchovy biomass, darker red for energy
                Species.COD: (cod_biomass, cod_energy, (0, 0, 0), (50, 50, 50))                 # Black for cod biomass, gray for energy
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

# Main visualization function, optimized with reduced plotting frequency
visualization_runs = 0
def visualize(world, agent_index, step):
    global visualization_runs
    if agent_index not in agents_data:
        # Initialize data for this agent if not already present
        agents_data[agent_index] = {
            'steps': [],
            'cod_alive': [],
            'anchovy_alive': [],
            'plankton_alive': [],
            'energy_plankton': [],
            'energy_anchovy': [],
            'energy_cod': []
        }

    # Append current step and biomass data for this agent
    agents_data[agent_index]['steps'].append(step)
    agents_data[agent_index]['plankton_alive'].append(world[:, :, const.OFFSETS_BIOMASS_PLANKTON].sum())
    agents_data[agent_index]['anchovy_alive'].append(world[:, :, const.OFFSETS_BIOMASS_ANCHOVY].sum())
    agents_data[agent_index]['cod_alive'].append(world[:, :, const.OFFSETS_BIOMASS_COD].sum())
    agents_data[agent_index]['energy_plankton'].append(world[:, :, const.OFFSETS_ENERGY_PLANKTON].sum())
    agents_data[agent_index]['energy_anchovy'].append(world[:, :, const.OFFSETS_ENERGY_ANCHOVY].sum())
    agents_data[agent_index]['energy_cod'].append(world[:, :, const.OFFSETS_ENERGY_COD].sum())

    # print(f"Agent: {agent_index}, Steps: {step}, Cod: {agents_data[agent_index]['cod_alive'][-1]}, "
    #       f"Anchovy: {agents_data[agent_index]['anchovy_alive'][-1]}, Plankton: {agents_data[agent_index]['plankton_alive'][-1]}")
    # print energy levels
    # print(f"Energy - Cod: {agents_data[agent_index]['energy_cod'][-1]}, "
    #       f"Anchovy: {agents_data[agent_index]['energy_anchovy'][-1]}, Plankton: {agents_data[agent_index]['energy_plankton'][-1]}")
    
    # Redraw the world
    if visualization_runs % 100 == 0:
        draw_world(world)
    # draw_world_detailed(world)

    # Only plot biomass and generations every 50 steps to optimize performance
    if visualization_runs % 100 == 0:
        plot_biomass()
    # plot_energy()
    
    if step == 0:
        plot_generations()
    
    visualization_runs += 1

# Quit pygame safely
def quit_pygame():
    pygame.quit()

def reset_visualization():
    reset_plot()
    reset_terrain_cache()
    global cod_alive
    global anchovy_alive
    global plankton_alive
    global steps_list
    global visualization_runs
    global agents_data
    global generations_data
    global terrain_surface_cache
    global biomass_graph_cache
    global energy_graph_cache
    global generation_graph_cache

    # Agent and generation data for visualization
    agents_data = {}
    generations_data = []
    visualization_runs = 0

    cod_alive = []
    anchovy_alive = []
    plankton_alive = []
    steps_list = []

    # Cache for terrain surface and graph surfaces
    terrain_surface_cache = None
    biomass_graph_cache = None
    energy_graph_cache = None
    generation_graph_cache = None