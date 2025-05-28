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

output_name = f"evaluated_run_averages_world_size{const.WORLD_SIZE}"
csv_output_file = f"{output_name}_per_run.csv"

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

def render_plot_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. Nothing to plot.")
        return

    species_colors = {
        'cod': const.SPECIES_MAP['cod']['visualization']['color_ones'],
        'herring': const.SPECIES_MAP['herring']['visualization']['color_ones'],
        'sprat': const.SPECIES_MAP['sprat']['visualization']['color_ones'],
        'plankton': const.SPECIES_MAP['plankton']['visualization']['color_ones']
    }

    plt.clf()
    first = True
    for eval_id, group in df.groupby("eval"):
        plt.plot(group['step'], group['cod'], linestyle='-', color=species_colors['cod'], alpha=0.5,
                 label='Cod' if first else "")
        plt.plot(group['step'], group['herring'], linestyle='-', color=species_colors['herring'], alpha=0.5,
                 label='Herring' if first else "")
        plt.plot(group['step'], group['sprat'], linestyle='-', color=species_colors['sprat'], alpha=0.5,
                 label='Sprat' if first else "")
        plt.plot(group['step'], group['plankton'], linestyle='-', color=species_colors['plankton'], alpha=0.5,
                 label='Plankton' if first else "")
        first = False

    plt.xlabel('Steps')
    plt.ylabel('Biomass')
    plt.title('Biomass Trends per Run')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_name}_per_run.png")
    plt.draw()
    plt.pause(0.001)

def render_plot_avg_with_band(csv_file):
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. Nothing to plot.")
        return

    species_colors = {
        'cod': const.SPECIES_MAP['cod']['visualization']['color_ones'],
        'herring': const.SPECIES_MAP['herring']['visualization']['color_ones'],
        'sprat': const.SPECIES_MAP['sprat']['visualization']['color_ones'],
        'plankton': const.SPECIES_MAP['plankton']['visualization']['color_ones']
    }

    plt.clf()
    grouped = df.groupby("step")

    for species in ['cod', 'herring', 'sprat', 'plankton']:
        mean_vals = grouped[species].mean()
        min_vals = grouped[species].min()
        max_vals = grouped[species].max()
        # steps = mean_vals.index
        days = mean_vals.index * const.DAYS_PER_STEP
        years = days / 365

        # Plot mean line
        plt.plot(years, mean_vals, label=species.capitalize(), color=species_colors[species])

        # Fill between min and max
        plt.fill_between(years, min_vals, max_vals, color=species_colors[species], alpha=0.2)

    plt.xlabel('Years')
    plt.ylabel('Biomass (million tonnes)')
    plt.title('Average Biomass per Step with Min/Max Bands')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    formatter = FuncFormatter(lambda x, pos: f'{x*1e-6:.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(f"{output_name}_avg_band.png")
    plt.draw()
    plt.pause(0.001)

# ---- Simulation + CSV Saving ----

def run_simulation_and_save_csv():
    runner, model_paths = get_runner_single()
    num_evals = 3
    years = 10
    all_records = []
    eval_counter = 0

    def eval_callback(world, step):
        nonlocal eval_counter
        cod_biomass = world[..., const.SPECIES_MAP["cod"]["biomass_offset"]].sum()
        herring_biomass = world[..., const.SPECIES_MAP["herring"]["biomass_offset"]].sum()
        sprat_biomass = world[..., const.SPECIES_MAP["sprat"]["biomass_offset"]].sum()
        plankton_biomass = world[..., const.SPECIES_MAP["plankton"]["biomass_offset"]].sum()

        all_records.append({
            'eval': eval_counter,
            'step': step,
            'cod': cod_biomass,
            'herring': herring_biomass,
            'sprat': sprat_biomass,
            'plankton': plankton_biomass
        })

    for year in range(years):
        update_initial_biomass(year)
        for _ in range(num_evals):
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
        render_plot_from_csv(csv_output_file)
        print("Finished evaluation.")
        time.sleep(5)
    else:
        print(output_name)
        render_plot_avg_with_band(csv_output_file)
