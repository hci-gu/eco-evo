import os
import sys

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.environments.petting_zoo import env as petting_zoo_env
from lib.heuristics import get_heuristic_action
from lib.model import OUTPUT_SIZE

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def eval_heuristic():
    print("Setting up heuristic evaluation...")
    settings = Settings(
        world_size=10,
        num_agents=1,
        max_steps=1000,
        folder="results/eval_heuristic"
    )
    
    if not os.path.exists(settings.folder):
        os.makedirs(settings.folder)
        
    species_map = build_species_map(settings)
    env = petting_zoo_env(settings, species_map, render_mode="none")
    
    episodes = 10
    survival_times = []
    
    print(f"Running {episodes} episodes with heuristic policy...")
    
    for ep in range(episodes):
        env.reset()
        steps = 0
        max_steps = settings.max_steps * 4
        
        for _ in range(max_steps):
            # Get current agent
            agent = env.agent_selection
            
            if env.terminations[agent] or env.truncations[agent]:
                # Dead agent step
                env.step(None) # Action doesn't matter
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break
                continue
            
            # Determine action
            if agent == "plankton":
                # Plankton logic handled by env usually, but we need to pass something
                action = np.zeros((settings.world_size, settings.world_size, OUTPUT_SIZE), dtype=np.float32)
            else:
                # Use heuristic for EVERYONE
                action = get_heuristic_action(agent, env.unwrapped.world, species_map, settings)
            
            env.step(action)
            
            # Count steps only once per full cycle or just raw steps?
            # Let's count raw steps for now, or maybe survival of a specific species?
            # The user wants to see "how long we survive for".
            # Let's track if "cod" is alive.
            
            if agent == "cod":
                steps += 1
                if env.terminations["cod"]:
                    break
        
        survival_times.append(steps)
        print(f"Episode {ep+1}: Survival Steps = {steps}")
        
    avg_survival = np.mean(survival_times)
    print(f"Average Survival Time: {avg_survival:.2f} steps")

if __name__ == "__main__":
    eval_heuristic()
