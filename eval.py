import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import lib.constants as const
import random
from lib.runners.petting_zoo import PettingZooRunner
from lib.runners.petting_zoo_single import PettingZooRunnerSingle

# Enable interactive mode
plt.ion()

def get_runner_single():
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".")[0]), reverse=True)
    print(files[0])
    path = os.path.join(folder, files[0])
    runner = PettingZooRunnerSingle()

    return runner, path

def get_runner():
    # folder = "results/knowledge_lab/agents"
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
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
    # calculate max steps, we simulate 1 year in 365 days and we run for 27 years
    # 27 years * 365 days = 9855 days
    days = 27 * 365
    const.MAX_STEPS = int(days / const.DAYS_PER_STEP) + 10
    print(f"MAX_STEPS: {const.MAX_STEPS}")


# def render_plot(biomass_data):
#     # Load the CSV data
#     csv_data = pd.read_csv("data.csv")
#     csv_data['year'] = csv_data['year'].astype(int)
#     csv_data = csv_data.sort_values('year')

#     # Define a consistent color mapping for each species using the default color cycle
#     default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     species_colors = {
#         'cod': default_colors[0],
#         'herring': default_colors[1],
#         'sprat': default_colors[2]
#     }

#     # Clear the current figure
#     plt.clf()

#     # Plot CSV data for each species using the defined colors
#     plt.plot(csv_data['year'], csv_data['cod'], marker='o', color=species_colors['cod'], label='Cod (CSV)')
#     plt.plot(csv_data['year'], csv_data['herring'], marker='o', color=species_colors['herring'], label='Herring (CSV)')
#     plt.plot(csv_data['year'], csv_data['sprat'], marker='o', color=species_colors['sprat'], label='Sprat (CSV)')

#     # If we have biomass data from the agents, convert to a DataFrame and plot it using the same colors
#     if biomass_data:
#         agent_df = pd.DataFrame(biomass_data)
#         agent_df = agent_df.sort_values('year')
#         plt.plot(agent_df['year'], agent_df['cod'], marker='x', linestyle='--', color=species_colors['cod'], label='Cod (Agent)')
#         plt.plot(agent_df['year'], agent_df['herring'], marker='x', linestyle='--', color=species_colors['herring'], label='Herring (Agent)')
#         plt.plot(agent_df['year'], agent_df['sprat'], marker='x', linestyle='--', color=species_colors['sprat'], label='Sprat (Agent)')

#     # Set labels, title, and legend
#     plt.xlabel('Year')
#     plt.ylabel('Biomass')
#     plt.title('Biomass Trends for Cod, Herring, and Sprat (1993-2020)')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(csv_data['year'], rotation=45)
#     plt.tight_layout()

#     # Update the plot without blocking execution
#     plt.draw()
#     plt.pause(0.001)

csv_data = pd.read_csv("data.csv")

def render_plot(biomass_data):
    global csv_data
    # Load the CSV data
    csv_data['year'] = csv_data['year'].astype(int)
    csv_data = csv_data.sort_values('year')

    # Define a consistent color mapping for each species
    species_colors = {
        'cod': const.SPECIES_MAP['cod']['visualization']['color_ones'],
        'herring': const.SPECIES_MAP['herring']['visualization']['color_ones'],
        'sprat': const.SPECIES_MAP['sprat']['visualization']['color_ones'],
        'plankton': const.SPECIES_MAP['plankton']['visualization']['color_ones']
    }

    # Clear the current figure
    plt.clf()

    # Plot CSV data
    plt.plot(csv_data['year'], csv_data['cod'], marker='o', color=species_colors['cod'], label='Cod (Actual)')
    plt.plot(csv_data['year'], csv_data['herring'], marker='o', color=species_colors['herring'], label='Herring (Actual)')
    plt.plot(csv_data['year'], csv_data['sprat'], marker='o', color=species_colors['sprat'], label='Sprat (Actual)')

    # Aggregate biomass_data over multiple runs
    if biomass_data:
        all_records = []
        for eval_idx, records in biomass_data.items():
            all_records.extend(records)

        if all_records:
            agent_df = pd.DataFrame(all_records)
            agent_df = agent_df.groupby('year').mean().reset_index()

            # Plot averaged agent data
            plt.plot(agent_df['year'], agent_df['cod'], marker='x', linestyle='--', color=species_colors['cod'], label='Cod (Simulated Avg)')
            plt.plot(agent_df['year'], agent_df['herring'], marker='x', linestyle='--', color=species_colors['herring'], label='Herring (Simulated Avg)')
            plt.plot(agent_df['year'], agent_df['sprat'], marker='x', linestyle='--', color=species_colors['sprat'], label='Sprat (Simulated Avg)')

    # Set labels, title, and legend
    plt.xlabel('Year')
    plt.ylabel('Biomass (million tonnes)')
    plt.title('Biomass Trends for Cod, Herring, and Sprat (1993-2020)')
    plt.legend()
    plt.grid(True)
    plt.xticks(csv_data['year'], rotation=45)
    formatter = FuncFormatter(lambda x, pos: f'{x*1e-6:.1f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()

    # Update the plot without blocking execution
    plt.draw()
    # save to file
    plt.savefig("biomass_trends.png")
    # save the biomass data to a csv file
    if biomass_data:
        all_records = []
        for eval_idx, records in biomass_data.items():
            all_records.extend(records)
        if all_records:
            df = pd.DataFrame(all_records)
            df.to_csv("evaluation_biomass_data.csv", index=False)
    plt.pause(0.001)

years_rendered = -1
current_eval = 0
biomass_data = {}
starting_year = 1993

def update_fishing_pressure(year):
    global csv_data

    for species in const.SPECIES_MAP:
        if species == "plankton":
            continue
        # Uncomment the next line to set a specific fishing pressure for each species
        key = f"fishing_{species}"
        value = csv_data.loc[csv_data['year'] == year, key].values[0]

        const.update_fishing_for_species(species, value)

def eval_callback(world, fitness):
    global years_rendered
    global biomass_data

    if current_eval not in biomass_data:
        biomass_data[current_eval] = []

    days = fitness * const.DAYS_PER_STEP
    years = int(round(days / 365))

    if years > years_rendered:
        year = starting_year + years
        update_fishing_pressure(year)

        years_rendered = years

        cod_biomass = world[..., const.SPECIES_MAP["cod"]["biomass_offset"]].sum()
        herring_biomass = world[..., const.SPECIES_MAP["herring"]["biomass_offset"]].sum()
        sprat_biomass = world[..., const.SPECIES_MAP["sprat"]["biomass_offset"]].sum()

        # Append the new agent data with the adjusted year
        biomass_data[current_eval].append({
            'year': year,
            'cod': cod_biomass,
            'herring': herring_biomass,
            'sprat': sprat_biomass
        })

        print(f"Eval {current_eval} - Year: {year}, Cod: {cod_biomass}, Herring: {herring_biomass}, Sprat: {sprat_biomass}")
        render_plot(biomass_data)
        
        if year > 2020:
            # If the year exceeds 2020, stop the simulation
            print("Reached the end of the simulation period.")
            return False
        

if __name__ == "__main__":
    runner, model_paths = get_runner_single()
    num_evals = 10
    for i in range(num_evals):
        update_initial_biomass()
        update_fishing_pressure(starting_year)
        years_rendered = -1
        runner.evaluate(model_paths, eval_callback, i)
        current_eval += 1

    print("Finished evaluation.")
    time.sleep(5000)
