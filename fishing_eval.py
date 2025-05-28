import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import lib.constants as const
from lib.runners.petting_zoo_single import PettingZooRunnerSingle

# Define simulation parameters
fishing_levels = [
    0,
    0.5,
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
]  # 0.1 to 0.9
simulations_per_level = 5

# Collect results
results = []

def set_fishing_pressure(scalar):
    const.update_fishing_scaler(scalar)

def noop(a, b):
    pass

def run_episode(runner, model_path, scalar, index):
    set_fishing_pressure(scalar)
    fitness, ep_length = runner.evaluate(model_path, noop, index)
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
        for i in range(simulations_per_level):
            steps_survived = run_episode(runner, path, pressure, i)
            results.append({'fishing_pressure': pressure, 'episode_length': steps_survived})
        print(f"Completed simulations for fishing pressure scalar = {pressure}")
        print(f"Average steps survived for fishing pressure {pressure}: {np.mean([r['episode_length'] for r in results if r['fishing_pressure'] == pressure])}")
    
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
    plt.savefig("fishing_pressure_results.png")
    plt.show()

if __name__ == "__main__":
    main()
