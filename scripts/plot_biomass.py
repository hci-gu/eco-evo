import os
import sys

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from lib.runners.pbm import PBMRunner
from lib.config.settings import Settings

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "biomass.png")

def get_model_path():
    folder = "results/2025-11-12_PBM/agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".npy")[0]), reverse=True)
    
    # Take only the highest fitness model
    best_model = files[0]
    model_path = os.path.join(folder, best_model)
    return model_path

if __name__ == "__main__":

    model_path = get_model_path()

    settings = Settings()
    runner = PBMRunner(settings)

    info_history = []

    def callback(info, _fitness, _is_done):
        if info is None:
            return
        info_history.append(info)

    runner.evaluate(model_path, callback=callback)

    if not info_history:
        raise RuntimeError("No biomass information received from evaluation.")

    base_env = runner.env.unwrapped
    n_fg = getattr(base_env, "n_fg", None)
    grid_shape = getattr(base_env, "grid_shape", None)
    if n_fg is None or grid_shape is None:
        raise AttributeError("Unable to determine functional group metadata from environment.")

    n_cells = grid_shape[0] * grid_shape[1]
    pop_sizes = np.array([
        [step.get(f"info--population/FG_{fg_idx}", 0.0) for fg_idx in range(n_fg)]
        for step in info_history
    ], dtype=float)
    total_biomass = pop_sizes * n_cells
    steps = np.arange(total_biomass.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    for fg_idx in range(total_biomass.shape[1]):
        ax.plot(steps, total_biomass[:, fg_idx], label=f"FG{fg_idx}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total biomass (individuals)")
    ax.set_title("Total biomass per functional group")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)) or ".", exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=250)
    plt.close(fig)

    final_values = total_biomass[-1]
    for fg_idx, value in enumerate(final_values):
        print(f"FG{fg_idx}: final total biomass = {value:.2f}")
    print(f"Biomass plot saved to {OUTPUT_FILE}")
