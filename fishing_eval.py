import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import lib.constants as const
from lib.runners.petting_zoo_single import PettingZooRunnerSingle

# Define simulation parameters
fishing_levels = [round(x * 0.1, 2) for x in range(1, 10)]  # 0.1 to 0.9
simulations_per_level = 2
extinction_threshold = 1e-3  # Biomass threshold for extinction

# Collect results
results = []

def set_fishing_pressure(scalar):
    for species in const.SPECIES_MAP.values():
        species['fishing_mortality_rate'] = 100 * scalar

def run_episode(runner, model_path, scalar):
    set_fishing_pressure(scalar)
    fitness, ep_length = runner.evaluate(model_path)
    return fitness

def get_runner_single():
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".")[0]), reverse=True)
    path = os.path.join(folder, files[0])
    runner = PettingZooRunnerSingle()
    return runner, path

def main():
    runner, path = get_runner_single()
    for pressure in fishing_levels:
        print(f"Running simulations for fishing pressure scalar = {pressure}")
        for _ in range(simulations_per_level):
            steps_survived = run_episode(runner, path, pressure)
            results.append({'fishing_pressure': pressure, 'episode_length': steps_survived})
        print(f"Completed simulations for fishing pressure scalar = {pressure}")
        print(f"Average steps survived for fishing pressure {pressure}: {np.mean([r['episode_length'] for r in results if r['fishing_pressure'] == pressure])}")
    
    # Plot the results
    # Plot the results
    df = pd.DataFrame(results)
    df['fishing_pressure'] = df['fishing_pressure'].astype(float)
    df['episode_length'] = df['episode_length'].astype(int)

    plt.figure(figsize=(10, 6))
    sns.boxenplot(data=df, x='fishing_pressure', y='episode_length', palette="husl")
    plt.title("Time until first species goes extinct")
    plt.xlabel("Fishing Pressure (as % of original)")
    plt.ylabel("Episode length")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
