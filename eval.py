import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import lib.constants as const
from lib.runners.petting_zoo import PettingZooRunner

# Enable interactive mode
plt.ion()

def get_runner():
    # folder = "results/knowledge_lab/agents"
    folder = "results/petting_zoo_biomass_fitness_log_5/agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[2].split(".")[0]), reverse=True)
    species = {}
    for f in files:
        s = f.split("_")[1].split(".")[0]
        s = s[1:] if s[0] == "$" else s
        if s == "spat":
            s = "sprat"
        if s not in species:
            species[s] = f

    model_paths = []
    for s, f in species.items():
        model_paths.append({ 'path': os.path.join(folder, f), 'species': s })

    runner = PettingZooRunner()
    return runner, model_paths

def update_initial_biomass():
    csv_data = pd.read_csv("data.csv")

    inital_cod = csv_data['cod'].iloc[0]
    inital_herring = csv_data['herring'].iloc[0]
    inital_sprat = csv_data['sprat'].iloc[0]

    const.FIXED_BIOMASS = True
    const.SPECIES_MAP["cod"]["original_starting_biomass"] = inital_cod
    const.SPECIES_MAP["herring"]["original_starting_biomass"] = inital_herring
    const.SPECIES_MAP["sprat"]["original_starting_biomass"] = inital_sprat
    const.MIN_PERCENT_ALIVE = 0
    const.MAX_PERCENT_ALIVE = 9999




def render_plot(biomass_data):
    # Load the CSV data
    csv_data = pd.read_csv("data.csv")
    csv_data['year'] = csv_data['year'].astype(int)
    csv_data = csv_data.sort_values('year')

    # Define a consistent color mapping for each species using the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    species_colors = {
        'cod': default_colors[0],
        'herring': default_colors[1],
        'sprat': default_colors[2]
    }

    # Clear the current figure
    plt.clf()

    # Plot CSV data for each species using the defined colors
    plt.plot(csv_data['year'], csv_data['cod'], marker='o', color=species_colors['cod'], label='Cod (CSV)')
    plt.plot(csv_data['year'], csv_data['herring'], marker='o', color=species_colors['herring'], label='Herring (CSV)')
    plt.plot(csv_data['year'], csv_data['sprat'], marker='o', color=species_colors['sprat'], label='Sprat (CSV)')

    # If we have biomass data from the agents, convert to a DataFrame and plot it using the same colors
    if biomass_data:
        agent_df = pd.DataFrame(biomass_data)
        agent_df = agent_df.sort_values('year')
        plt.plot(agent_df['year'], agent_df['cod'], marker='x', linestyle='--', color=species_colors['cod'], label='Cod (Agent)')
        plt.plot(agent_df['year'], agent_df['herring'], marker='x', linestyle='--', color=species_colors['herring'], label='Herring (Agent)')
        plt.plot(agent_df['year'], agent_df['sprat'], marker='x', linestyle='--', color=species_colors['sprat'], label='Sprat (Agent)')

    # Set labels, title, and legend
    plt.xlabel('Year')
    plt.ylabel('Biomass')
    plt.title('Biomass Trends for Cod, Herring, and Sprat (1993-2020)')
    plt.legend()
    plt.grid(True)
    plt.xticks(csv_data['year'], rotation=45)
    plt.tight_layout()

    # Update the plot without blocking execution
    plt.draw()
    plt.pause(0.001)

years_rendered = -1
biomass_data = []
starting_year = 1993

def eval_callback(world, fitness):
    global years_rendered
    global biomass_data

    days = fitness * const.DAYS_PER_STEP
    years = int(round(days / 365))
    # print(f"Fitness: {fitness}, Days: {days}, Years: {years}")

    if years > years_rendered:
        years_rendered = years

        cod_biomass = world[..., const.SPECIES_MAP["cod"]["biomass_offset"]].sum()
        herring_biomass = world[..., const.SPECIES_MAP["herring"]["biomass_offset"]].sum()
        sprat_biomass = world[..., const.SPECIES_MAP["sprat"]["biomass_offset"]].sum()

        # Append the new agent data with the adjusted year
        print(fitness, {
            'year': starting_year + years,
            'cod': cod_biomass,
            'herring': herring_biomass,
            'sprat': sprat_biomass
        })
        biomass_data.append({
            'year': starting_year + years,
            'cod': cod_biomass,
            'herring': herring_biomass,
            'sprat': sprat_biomass
        })

        if starting_year + years > 2020:
            # If the year exceeds 2020, stop the simulation
            print("Reached the end of the simulation period.")
            return False
        
        render_plot(biomass_data)

if __name__ == "__main__":
    update_initial_biomass()
    runner, model_paths = get_runner()
    runner.evaluate(model_paths, eval_callback)
    print("Finished evaluation.")
    time.sleep(5000)
