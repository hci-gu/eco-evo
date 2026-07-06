import numpy as np
import matplotlib.pyplot as plt
from lib.config.config_loader import setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment

def run_sim(seed=None, ticks=40):
    if seed is not None:
        np.random.seed(seed)
    fgs = setup_full_mareld_mvp()
    grid_config = {'width': 60, 'height': 60, 'cell_size': 1000.0, 'tick_duration': 6.0}
    env = EcosystemEnvironment(grid_config, fgs)
    
    history = {fid: [] for fid in env.fgs}
    for t in range(ticks):
        observation = env.get_observation()
        actions = env.policy_controller.forward(observation)
        env.step(actions)
        for fid, fg in env.fgs.items():
            history[fid].append(fg.biomass.sum())
    return history

def main():
    print("Running reference seed 1...")
    hist_noll = run_sim(seed=1)
    
    print("Running reference seed 2...")
    hist_proj = run_sim(seed=2)
    
    # Compare specific species, e.g., porpoise (tumlare)
    species = ['porpoises', 'gadoids', 'pelagic_fish']
    
    plt.figure(figsize=(12, 8))
    for i, sp in enumerate(species):
        plt.subplot(len(species), 1, i+1)
        plt.plot(hist_noll[sp], label='Reference seed 1')
        plt.plot(hist_proj[sp], label='Reference seed 2')
        plt.title(f'Biomass Development: {sp}')
        plt.ylabel('Tons')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('results/scenario_comparison.png')
    print("Comparison plot saved to results/scenario_comparison.png")

if __name__ == "__main__":
    main()
