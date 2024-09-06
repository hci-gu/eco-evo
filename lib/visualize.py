import pygame
import math
from lib.world import Terrain, Species
from lib.constants import WORLD_SIZE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np

WORLD_WIDTH = 500
WORLD_HEIGHT = 500
GRAPH_WIDTH = 500
GEN_GRAPH_HEIGHT = 200
CELL_SIZE = math.floor(WORLD_HEIGHT / WORLD_SIZE)

SCREEN_WIDTH = WORLD_WIDTH + GRAPH_WIDTH
SCREEN_HEIGHT = WORLD_HEIGHT + GEN_GRAPH_HEIGHT

agents_data = {}
generations_data = []

cod_alive = []
anchovy_alive = []
plankton_alive = []
steps_list = []

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ecosystem simulation")
    return screen

screen = init_pygame()

def draw_terrain(world):
    terrain_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
            cell = world[x * WORLD_SIZE + y]
            color = (0, 0, 255) if cell.terrain == Terrain.WATER else (0, 255, 0)
            pygame.draw.rect(terrain_surface, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    return terrain_surface

def initial_draw(world):
    screen.fill((255, 255, 255))
    terrain_surface = draw_terrain(screen, world)
    screen.blit(terrain_surface, (WORLD_WIDTH, 0))
    pygame.display.flip()

def update_generations_data():
    fitness_values = [max(agent['steps']) for (key, agent) in agents_data.items()]
    generations_data.append(fitness_values)

def reset_plot():
    update_generations_data()
    agents_data.clear()

def plot_generations():
    plt.figure(figsize=(9, 2))  # Create a new figure for the plot

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

    # Set plot labels and title
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness of Agents Over Generations')
    plt.legend()
    plt.tight_layout()

    # Save plot as an image
    plt.savefig('fitness_plot.png')
    plt.close()

    # Load the saved plot image using Pygame
    fitness_image = pygame.image.load('fitness_plot.png')
    
    # Display the plot at the bottom of the screen
    image_rect = fitness_image.get_rect(center=(SCREEN_WIDTH // 2, WORLD_HEIGHT + 90))
    screen.blit(fitness_image, image_rect)
    pygame.display.update()
    
def plot_biomass(agent_index, world, step):
    if agent_index not in agents_data:
        # Initialize data for this agent if not already present
        agents_data[agent_index] = {
            'steps': [],
            'cod_alive': [],
            'anchovy_alive': [],
            'plankton_alive': []
        }

    # Append current step and biomass data for this agent
    agents_data[agent_index]['steps'].append(step)
    agents_data[agent_index]['cod_alive'].append(sum([cell.biomass[Species.COD] for cell in world]))
    agents_data[agent_index]['anchovy_alive'].append(sum([cell.biomass[Species.ANCHOVY] for cell in world]))
    agents_data[agent_index]['plankton_alive'].append(sum([cell.biomass[Species.PLANKTON] for cell in world]))

    print(f"Agent: {agent_index}, Steps: {step}, Cod: {agents_data[agent_index]['cod_alive'][-1]}, "
          f"Anchovy: {agents_data[agent_index]['anchovy_alive'][-1]}, Plankton: {agents_data[agent_index]['plankton_alive'][-1]}")

    
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

    # Display the updated graph with pygame
    graph_image = pygame.image.load('world_graph.png')
    screen.blit(graph_image, (WORLD_WIDTH, 0))  # Adjust the position as needed

def draw_world(world):
    screen.fill((255, 255, 255))

    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
            cell = world[x * WORLD_SIZE + y]
            
            # Draw terrain color first
            color = (18, 53, 163) if cell.terrain == Terrain.WATER else (112, 180, 40)
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            # Initialize an offset position for each species in the same cell
            offsets = {
                Species.PLANKTON: (-CELL_SIZE // 4, -CELL_SIZE // 4),
                Species.ANCHOVY: (CELL_SIZE // 4, -CELL_SIZE // 4),
                Species.COD: (0, CELL_SIZE // 4)
            }
            
            # Draw biomass for each species
            for species in Species:
                biomass = cell.biomass

                if biomass[species] > 0:
                    color = {
                        Species.PLANKTON: (0, 255, 0),  # Green for plankton
                        Species.ANCHOVY: (255, 0, 0),  # Red for anchovy
                        Species.COD: (0, 0, 0)  # Black for cod
                    }[species]
                    
                    # Use offsets to separate circles within the same cell
                    offset_x, offset_y = offsets[species]
                    circle_center = (x * CELL_SIZE + CELL_SIZE // 2 + offset_x, y * CELL_SIZE + CELL_SIZE // 2 + offset_y)
                    radius = min(CELL_SIZE // 2, int(math.sqrt(biomass[species]) * 0.4))
                    
                    # Draw the circle representing species biomass
                    pygame.draw.circle(screen, color, circle_center, radius)
            
def quit_pygame():
    pygame.quit()



