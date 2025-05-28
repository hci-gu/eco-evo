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

output_name = f"evaluated_run_long_world_size{const.WORLD_SIZE}"
csv_output_path = f"{output_name}.csv"

def get_runner_single():
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".")[0]), reverse=True)
    print(files[0])
    path = os.path.join(folder, files[0])
    runner = PettingZooRunnerSingle()

    return runner, path

csv_data = pd.read_csv("data.csv")

def render_plot(csv_path):
    species_colors = {
        'cod': const.SPECIES_MAP['cod']['visualization']['color_ones'],
        'herring': const.SPECIES_MAP['herring']['visualization']['color_ones'],
        'sprat': const.SPECIES_MAP['sprat']['visualization']['color_ones'],
        'plankton': const.SPECIES_MAP['plankton']['visualization']['color_ones']
    }

    plt.clf()

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.groupby('step').mean().reset_index()

    # Convert steps to years
    df['year'] = df['step'] * const.DAYS_PER_STEP / 365.0

    # Plot averaged agent data
    plt.plot(df['year'], df['cod'], linestyle='-', color=species_colors['cod'], label='Cod')
    plt.plot(df['year'], df['herring'], linestyle='-', color=species_colors['herring'], label='Herring')
    plt.plot(df['year'], df['sprat'], linestyle='-', color=species_colors['sprat'], label='Sprat')
    plt.plot(df['year'], df['plankton'], linestyle='-', color=species_colors['plankton'], label='Plankton')

    plt.xlabel('Years')
    plt.ylabel('Biomass (million tonnes)')
    plt.title('Biomass for Cod, Herring, and Sprat')
    plt.legend()
    plt.grid(True)

    formatter = FuncFormatter(lambda x, pos: f'{x*1e-6:.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(f"{output_name}.png")
    plt.draw()
    plt.pause(0.001)

current_eval = 0
biomass_data = {}

def eval_callback(world, step):
    global biomass_data

    if current_eval not in biomass_data:
        biomass_data[current_eval] = []

    cod_biomass = world[..., const.SPECIES_MAP["cod"]["biomass_offset"]].sum()
    herring_biomass = world[..., const.SPECIES_MAP["herring"]["biomass_offset"]].sum()
    sprat_biomass = world[..., const.SPECIES_MAP["sprat"]["biomass_offset"]].sum()
    plankton_biomass = world[..., const.SPECIES_MAP["plankton"]["biomass_offset"]].sum()

    # Append the new agent data with the adjusted year
    biomass_data[current_eval].append({
        'step': step,
        'cod': cod_biomass,
        'herring': herring_biomass,
        'sprat': sprat_biomass,
        'plankton': plankton_biomass
    })

    # print(f"Eval {current_eval} - Year: {year}, Cod: {cod_biomass}, Herring: {herring_biomass}, Sprat: {sprat_biomass}")
    if step % 500 == 0:
        print("progress", step)
    #     render_plot(biomass_data)
        

if __name__ == "__main__":
    runner, model_paths = get_runner_single()
    num_evals = 1
    # for i in range(num_evals):
    #     years_rendered = -1
    #     runner.evaluate(model_paths, eval_callback, i)
    #     current_eval += 1

    # # Save biomass data to CSV
    # all_records = []
    # for records in biomass_data.values():
    #     all_records.extend(records)
    # df = pd.DataFrame(all_records)
    # df.to_csv(csv_output_path, index=False)
    # print(f"Saved biomass data to {csv_output_path}")

    render_plot(csv_output_path)
    print("Finished evaluation.")
    time.sleep(5000)
