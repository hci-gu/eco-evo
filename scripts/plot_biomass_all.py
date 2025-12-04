"""
Plot biomass comparison across all runner types (PBM, PettingZoo, RL).

This script runs one evaluation for each available runner and creates
comparison plots of total biomass over time for each functional group.

Functional Groups:
    FG_0: Plankton (primary producers)
    FG_1: Sprat (small fish)
    FG_2: Herring (medium fish)
    FG_3: Cod (large fish / apex predator)
"""

import os
import sys
import warnings

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from lib.config.settings import Settings
from lib.model import MODEL_OFFSETS

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "biomass_all.png")

# Functional group names for legend
FG_NAMES = {
    0: "Plankton",
    1: "Sprat",
    2: "Herring",
    3: "Cod"
}

# Species to FG mapping (for PettingZoo runner)
SPECIES_TO_FG = {
    'plankton': 0,
    'sprat': 1,
    'herring': 2,
    'cod': 3
}


# =============================================================================
# Model Path Discovery
# =============================================================================

def get_pbm_model_path():
    """
    Find the best PBM model based on fitness score.
    PBM models: {generation}_{fitness}.npy.npz
    """
    folder = os.path.join(PROJECT_ROOT, "results", "2025-11-12_PBM", "agents")
    if not os.path.exists(folder):
        return None
    
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz") and "$" not in f]
    if not files:
        return None
    
    # Sort by fitness (second part of filename) descending
    files.sort(key=lambda f: float(f.split("_")[1].split(".npy")[0]), reverse=True)
    return os.path.join(folder, files[0])


def get_pettingzoo_model_paths():
    """
    Find the best PettingZoo models for each species.
    PettingZoo models: {generation}_${species}_{fitness}.npy.npz
    Returns list of dicts: [{'species': 'cod', 'path': '...'}, ...]
    """
    folder = os.path.join(PROJECT_ROOT, "results", "2025-10-23_1", "agents")
    if not os.path.exists(folder):
        return None
    
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz") and "$" in f]
    if not files:
        return None
    
    # Group by species and find best fitness for each
    species_best = {}
    for f in files:
        # Parse: {gen}_${species}_{fitness}.npy.npz
        parts = f.split("_$")
        if len(parts) < 2:
            continue
        species_fitness = parts[1]  # e.g. "cod_936.039306640625.npy.npz"
        species_parts = species_fitness.split("_")
        species = species_parts[0]
        fitness = float(species_parts[1].split(".npy")[0])
        
        if species not in species_best or fitness > species_best[species][1]:
            species_best[species] = (f, fitness)
    
    if not species_best:
        return None
    
    return [
        {'species': species, 'path': os.path.join(folder, filename)}
        for species, (filename, _) in species_best.items()
    ]


def get_rl_model_path():
    """
    Find the RL model (.zip file).
    """
    folder = os.path.join(PROJECT_ROOT, "results", "test_rl")
    if not os.path.exists(folder):
        return None
    
    files = [f for f in os.listdir(folder) if f.endswith(".zip")]
    if not files:
        return None
    
    return os.path.join(folder, files[0])


# =============================================================================
# Runner Adapters
# =============================================================================

def run_pbm_evaluation(settings):
    """
    Run PBM evaluation and return biomass time series.
    Returns: (steps, n_fg) numpy array of total biomass
    """
    from lib.runners.pbm import PBMRunner
    
    model_path = get_pbm_model_path()
    if model_path is None:
        warnings.warn("No PBM model found, skipping PBM runner")
        return None
    
    print(f"Running PBM evaluation with model: {os.path.basename(model_path)}")
    runner = PBMRunner(settings)
    
    info_history = []
    
    def callback(info, _fitness, _is_done):
        if info is not None:
            info_history.append(info)
    
    runner.evaluate(model_path, callback=callback)
    
    if not info_history:
        warnings.warn("No biomass data collected from PBM runner")
        return None
    
    # Extract metadata from environment
    base_env = runner.env.unwrapped
    n_fg = getattr(base_env, "n_fg", None)
    grid_shape = getattr(base_env, "grid_shape", None)
    
    if n_fg is None or grid_shape is None:
        warnings.warn("Unable to determine environment metadata from PBM runner")
        return None
    
    n_cells = grid_shape[0] * grid_shape[1]
    
    # Build biomass array: population density * n_cells = total biomass
    pop_sizes = np.array([
        [step.get(f"info--population/FG_{fg_idx}", 0.0) for fg_idx in range(n_fg)]
        for step in info_history
    ], dtype=float)
    
    total_biomass = pop_sizes * n_cells
    return total_biomass


def run_pettingzoo_evaluation(settings):
    """
    Run PettingZoo evaluation and return biomass time series.
    Returns: (steps, n_fg) numpy array of total biomass
    """
    from lib.runners.petting_zoo import PettingZooRunner
    from lib.model import Model
    
    model_paths = get_pettingzoo_model_paths()
    if model_paths is None:
        warnings.warn("No PettingZoo models found, skipping PettingZoo runner")
        return None
    
    print(f"Running PettingZoo evaluation with {len(model_paths)} species models")
    runner = PettingZooRunner(settings)
    
    # Load models for each species
    candidates = {}
    for mp in model_paths:
        # Load weights from .npz file and create model with chromosome
        weights = np.load(mp['path'])
        chromosome = {key: weights[key] for key in weights.files}
        model = Model(chromosome=chromosome)
        candidates[mp['species']] = model
    
    # Make sure we have all required species
    required_species = ['cod', 'herring', 'sprat']
    for species in required_species:
        if species not in candidates:
            warnings.warn(f"Missing model for species '{species}', skipping PettingZoo runner")
            return None
    
    biomass_history = []
    
    def callback(world, _fitness, _is_done):
        if world is None:
            return
        # Extract biomass for each species from world array
        # World has padding, so we need to access inner cells
        pad = 1
        inner_world = world[pad:-pad, pad:-pad]
        
        biomass = {}
        for species, fg_idx in SPECIES_TO_FG.items():
            if species in MODEL_OFFSETS:
                biomass_channel = MODEL_OFFSETS[species]["biomass"]
                total = inner_world[..., biomass_channel].sum()
                biomass[fg_idx] = total
        
        biomass_history.append(biomass)
    
    runner.run(candidates, callback=callback)
    
    if not biomass_history:
        warnings.warn("No biomass data collected from PettingZoo runner")
        return None
    
    # Convert to numpy array
    n_fg = len(FG_NAMES)
    total_biomass = np.array([
        [step.get(fg_idx, 0.0) for fg_idx in range(n_fg)]
        for step in biomass_history
    ], dtype=float)
    
    return total_biomass


def run_rl_evaluation(settings):
    """
    Run RL evaluation and return biomass time series.
    Returns: (steps, n_fg) numpy array of total biomass
    """
    from lib.runners.rl_runner import RLRunner
    
    model_path = get_rl_model_path()
    if model_path is None:
        warnings.warn("No RL model found, skipping RL runner")
        return None
    
    print(f"Running RL evaluation with model: {os.path.basename(model_path)}")
    runner = RLRunner(settings)
    runner.load(model_path)
    
    info_history = []
    
    def callback(info, _fitness, _is_done):
        if info is not None:
            info_history.append(info)
    
    runner.evaluate(steps=settings.max_steps, callback=callback)
    
    if not info_history:
        warnings.warn("No biomass data collected from RL runner")
        return None
    
    # Convert to numpy array
    n_fg = len(FG_NAMES)
    total_biomass = np.array([
        [step.get(f"info--population/FG_{fg_idx}", 0.0) for fg_idx in range(n_fg)]
        for step in info_history
    ], dtype=float)
    
    # Note: RL runner returns density, multiply by grid size for total
    grid_size = settings.world_size * settings.world_size
    total_biomass = total_biomass * grid_size
    
    return total_biomass


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(results, output_file):
    """
    Create comparison plot of biomass over time for each runner.
    
    Args:
        results: dict mapping runner name to (steps, n_fg) biomass array
        output_file: path to save the figure
    """
    active_results = {k: v for k, v in results.items() if v is not None}
    
    if not active_results:
        print("No results to plot!")
        return
    
    n_runners = len(active_results)
    n_fg = len(FG_NAMES)
    
    # Create figure with subplots: one column per runner
    fig, axes = plt.subplots(1, n_runners, figsize=(6 * n_runners, 5), squeeze=False)
    axes = axes[0]  # Flatten to 1D
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_fg))
    
    for ax, (runner_name, biomass) in zip(axes, active_results.items()):
        steps = np.arange(biomass.shape[0])
        
        for fg_idx in range(biomass.shape[1]):
            ax.plot(
                steps, 
                biomass[:, fg_idx], 
                label=f"FG_{fg_idx} ({FG_NAMES.get(fg_idx, 'Unknown')})",
                color=colors[fg_idx],
                linewidth=1.5
            )
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Biomass")
        ax.set_title(f"{runner_name} Runner")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Biomass Comparison Across Runners", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    fig.savefig(output_file, dpi=250, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison plot saved to {output_file}")


def print_final_summary(results):
    """Print final biomass values for each runner."""
    print("\n" + "=" * 60)
    print("Final Biomass Summary")
    print("=" * 60)
    
    for runner_name, biomass in results.items():
        if biomass is None:
            print(f"\n{runner_name}: No data (skipped)")
            continue
        
        print(f"\n{runner_name}:")
        final_values = biomass[-1]
        for fg_idx, value in enumerate(final_values):
            fg_name = FG_NAMES.get(fg_idx, f"FG_{fg_idx}")
            print(f"  FG_{fg_idx} ({fg_name}): {value:.2f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    settings = Settings()
    
    print("Starting biomass comparison evaluation...")
    print("=" * 60)
    
    results = {}
    
    # Run PBM
    print("\n[1/3] PBM Runner")
    print("-" * 40)
    results["PBM"] = run_pbm_evaluation(settings)
    
    # Run PettingZoo
    print("\n[2/3] PettingZoo Runner")
    print("-" * 40)
    results["PettingZoo"] = run_pettingzoo_evaluation(settings)
    
    # Run RL
    print("\n[3/3] RL Runner")
    print("-" * 40)
    results["RL"] = run_rl_evaluation(settings)
    
    # Create comparison plot
    print("\n" + "=" * 60)
    print("Creating comparison plot...")
    plot_comparison(results, OUTPUT_FILE)
    
    # Print summary
    print_final_summary(results)
