import json
import pygame
import lib.constants as const
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

json_file = "./results/single_agent_single_out_random_plankton_behavscore_6/generations_data.json"
output_file = "generations_data.png"

def plot_generations(generations_data):
    global generation_graph_cache

    plt.figure(figsize=(8, 6))  # Create a new figure for the plot

    # Check if generations_data[0] is an object (dict) or an array (list)
    if isinstance(generations_data[0], dict):
        # Multiple species case
        species_list = list(generations_data[0].keys())
    else:
        # Single species case
        species_list = ["Genome"]

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
            if sp == "Genome":
                fitness_values = generation
            else:
                fitness_values = generation[sp]
            # Add jitter to x-coordinates for better visualization
            jitter = np.random.normal(0, 0.05, size=len(fitness_values))
            x_values = generation_index + jitter

            # Plot scatter points for the species (only label on the first generation)
            plt.scatter(x_values, fitness_values, alpha=0.3, color=species_colors[sp],
                        label=sp if generation_index == 0 else "", s=10)

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
                 label=f"Generation average", linewidth=2)
        plt.fill_between(generations, species_stats[sp]["bottom"], species_stats[sp]["top"],
                         color=species_colors[sp], alpha=0.4,
                         label=f"Generation 25th-75th Percentile")

    # Configure axes and title
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness of Agents Over Generations')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))  # Prevent overcrowding of x-axis ticks

    plt.legend()
    plt.tight_layout()

    # Save the plot as an image and update the cache
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    # read generations data from a JSON file
    with open(json_file, 'r') as f:
        generations_data = json.load(f)

        plot_generations(generations_data)