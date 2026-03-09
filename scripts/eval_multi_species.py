import os
import sys

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from lib.config.settings import Settings
from lib.environments.petting_zoo import env as petting_zoo_env
from lib.config.species import build_species_map
from lib.model import MODEL_OFFSETS, OUTPUT_SIZE
import lib.config.const as const

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_multi_species():
    print("Setting up multi-species evaluation...")
    settings = Settings(
        world_size=12,
        num_agents=1, # Not used directly in this context as we use world_size for grid
        max_steps=5000,
        folder="results/multi_species_train"
    )
    
    species_list = ["cod", "herring", "sprat"]
    models = {}
    
    # Load models
    print("Loading models...")
    for sp in species_list:
        model_path = os.path.join(settings.folder, f"model_{sp}_best.zip")
        if os.path.exists(model_path):
            print(f"Loading {sp} model from {model_path}")
            models[sp] = PPO.load(model_path)
        else:
            print(f"Warning: Model for {sp} not found at {model_path}. Agents will be random.")
            models[sp] = None

    # Initialize environment
    species_map = build_species_map(settings)
    env = petting_zoo_env(settings, species_map, render_mode="human")
    env.reset()
    
    # Tracking
    biomass_history = {sp: [] for sp in species_list}
    steps = 0
    max_steps = settings.max_steps * 4 # Same as truncation limit
    
    print("Starting simulation...")
    
    # We need to track when a "step" (one round of all agents) happens for plotting consistency
    # But AEC cycles through agents. We can just plot every N moves or just append every move.
    # Appending every move might be too granular/noisy if we plot total biomass.
    # Let's append every time we cycle through all agents? Or just every 100 moves?
    # Actually, let's just append every time `env.num_moves` increments, but that might be per agent.
    # `env.num_moves` in petting_zoo.py seems to increment every step.
    
    # Let's track biomass every time we complete a full cycle of agents roughly, 
    # or just every X steps.
    
    while steps < max_steps:
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("Environment terminated or truncated.")
            break
            
        agent = env.agent_selection
        
        # Record biomass periodically (e.g., every time we process a specific agent or just every N steps)
        # Let's do it every time we process the first agent in our list to approximate a "tick"
        # Or simpler: just every 10 steps.
        for sp in species_list:
            # Calculate total biomass for this species
            # Channel index
            ch = MODEL_OFFSETS[sp]["biomass"]
            total_b = np.sum(env.world[..., ch])
            biomass_history[sp].append(total_b)
        
        if agent == "plankton":
            # So we just need to call step with a dummy action.
            empty_action = np.zeros((settings.world_size, settings.world_size, OUTPUT_SIZE), dtype=np.float32)
            env.step(empty_action)
        else:
            raw_obs = env.observations[agent]
            num_envs = raw_obs.shape[0]
            flat_obs = raw_obs.reshape(num_envs, -1)
            
            # Predict
            actions, _ = models[agent].predict(flat_obs, deterministic=False)
            
            # Convert to grid
            actions_grid = np.zeros((settings.world_size, settings.world_size, OUTPUT_SIZE), dtype=np.float32)
            rows, cols = np.unravel_index(np.arange(num_envs), (settings.world_size, settings.world_size))
            actions_grid[rows, cols, actions] = 1.0
            
            env.step(actions_grid)
        
        steps += 1
        
    print(f"Simulation finished after {steps} steps.")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for sp in species_list:
        plt.plot(biomass_history[sp], label=sp)
        
    plt.title("Multi-Species Biomass Over Time (Evaluation)")
    plt.xlabel("Time (x10 steps)")
    plt.ylabel("Total Biomass")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, "eval_multi_species_biomass.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Check survival
    # We can define survival as "last step where biomass > 0"
    print("\nSurvival Analysis:")
    for sp in species_list:
        # Find last index where biomass > 1.0 (threshold)
        alive_indices = [i for i, b in enumerate(biomass_history[sp]) if b > 1.0]
        if alive_indices:
            last_alive = alive_indices[-1] * 10 # scale back to steps
            print(f"{sp}: Survived until step {last_alive}")
        else:
            print(f"{sp}: Extinct immediately or never present.")

if __name__ == "__main__":
    eval_multi_species()
