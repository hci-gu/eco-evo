import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import lib.constants as const
import numpy as np
import random
from lib.runners.petting_zoo import PettingZooRunner
from lib.runners.petting_zoo_single import PettingZooRunnerSingle

# Enable interactive mode
plt.ion()

fishing_levels = [
    0,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    6,
    7,
    8,
    9,
    10
]
# fishing_levels = [
#     0,
#     1,
#     2,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
#     9,
#     10,
#     15,
#     20
# ]

output_name = f"fishing_pressure_biomass"
csv_output_file = f"{output_name}.csv"

def update_initial_biomass(year=0):
    csv_data = pd.read_csv("data.csv")
    inital_cod = csv_data['cod'].iloc[year]
    inital_herring = csv_data['herring'].iloc[year]
    inital_sprat = csv_data['sprat'].iloc[year]

    const.FIXED_BIOMASS = True
    const.SPECIES_MAP["cod"]["original_starting_biomass"] = inital_cod
    const.SPECIES_MAP["herring"]["original_starting_biomass"] = inital_herring
    const.SPECIES_MAP["sprat"]["original_starting_biomass"] = inital_sprat
    print(f"Year: {year}, Cod: {inital_cod}, Herring: {inital_herring}, Sprat: {inital_sprat}")
    const.MIN_PERCENT_ALIVE = 0.01
    const.MAX_PERCENT_ALIVE = 9999

def get_runner_single():
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".")[0]), reverse=True)
    print(files[0])
    path = os.path.join(folder, files[0])
    runner = PettingZooRunnerSingle()
    return runner, path

def render_plot_avg_with_band(csv_file):
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. Nothing to plot.")
        return

    pressures = sorted(df['fishing_pressure'].unique())
    norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
    cmap = cm.get_cmap('RdYlGn_r')  # reversed so green = low, red = high

    for species in ['cod', 'herring', 'sprat']:
        plt.clf()
        fig, ax = plt.subplots()

        for pressure in pressures:
            group = df[df['fishing_pressure'] == pressure].groupby('step')
            mean_vals = group[species].mean()
            min_vals = group[species].min()
            max_vals = group[species].max()
            steps = mean_vals.index

            color = cmap(norm(pressure))
            label = f"F={pressure}"
            ax.plot(steps, mean_vals, label=label, color=color, alpha=0.9)
            ax.fill_between(steps, min_vals, max_vals, color=color, alpha=0.2)

        ax.set_xlabel('Years')
        ax.set_ylabel('Biomass')
        ax.set_title(f'{species.capitalize()} Biomass per Step (Fishing Pressure Gradient)')
        ax.legend(loc='upper right', fontsize='x-small', title="Fishing Pressure")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_name}_{species}_avg_band_by_pressure.png")
        plt.draw()
        plt.pause(0.001)

def render_combined_log_biomass_plot(csv_file, output_name="output"):
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. Nothing to plot.")
        return

    pressures = sorted(df['fishing_pressure'].unique())
    norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
    cmap = cm.get_cmap('RdYlGn_r')

    plt.clf()
    fig, ax = plt.subplots()

    steps_in_a_year = 365

    for pressure in pressures:
        group = df[df['fishing_pressure'] == pressure].copy()
        group['log_total'] = np.log10(group['cod']) + np.log10(group['herring']) + np.log10(group['sprat'])
        
        grouped = group.groupby('step')
        mean_vals = grouped['log_total'].mean()
        min_vals = grouped['log_total'].min()
        max_vals = grouped['log_total'].max()
        days = mean_vals.index * const.DAYS_PER_STEP
        years = days / steps_in_a_year

        color = cmap(norm(pressure))
        label = f"F={pressure:.2f}%"
        ax.plot(years, mean_vals, label=label, color=color, alpha=0.9)
        ax.fill_between(years, min_vals, max_vals, color=color, alpha=0.2)

    ax.set_xlabel('Years')
    ax.set_ylabel('Biodiversity')
    ax.set_title('Biodiversity by Fishing Pressure')
    ax.legend(loc='upper right', fontsize='x-small', title="Fishing Pressure")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_name}_combined_log_biomass.png")
    plt.draw()
    plt.pause(0.001)

# ---- Simulation + CSV Saving ----

def run_simulation_and_save_csv():
    runner, model_paths = get_runner_single()
    num_evals = 5
    current_fishing_pressure = -1

    all_records = []
    eval_counter = 0

    def eval_callback(world, step):
        nonlocal eval_counter
        cod_biomass = world[..., const.SPECIES_MAP["cod"]["biomass_offset"]].sum()
        herring_biomass = world[..., const.SPECIES_MAP["herring"]["biomass_offset"]].sum()
        sprat_biomass = world[..., const.SPECIES_MAP["sprat"]["biomass_offset"]].sum()
        plankton_biomass = world[..., const.SPECIES_MAP["plankton"]["biomass_offset"]].sum()

        all_records.append({
            'fishing_pressure': current_fishing_pressure,
            'eval': eval_counter,
            'step': step,
            'cod': cod_biomass,
            'herring': herring_biomass,
            'sprat': sprat_biomass,
            'plankton': plankton_biomass
        })

    for pressure in fishing_levels:
        fishing_percent = const.update_fishing_scaler(pressure)
        daily_fishing_percent = fishing_percent / const.DAYS_PER_STEP
        current_fishing_pressure = daily_fishing_percent * 100
        for i in range(num_evals):
            print(f"Running simulations for fishing pressure scalar = {pressure}, eval = {i}")
            fitness, ep_length = runner.evaluate(model_paths, eval_callback, eval_counter)
            if fitness < const.MAX_STEPS:
                print("WE FAILED!!!!")
            eval_counter += 1

    pd.DataFrame(all_records).to_csv(csv_output_file, index=False)
    print(f"Saved simulation results to {csv_output_file}")

if __name__ == "__main__":
    # Toggle between running simulation or just rendering the plot
    RUN_SIMULATION = False

    if RUN_SIMULATION:
        run_simulation_and_save_csv()
        render_combined_log_biomass_plot(csv_output_file)
        print("Finished evaluation.")
        time.sleep(5)
    else:

        render_combined_log_biomass_plot(csv_output_file)
        time.sleep(15)
