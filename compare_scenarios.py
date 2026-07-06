import numpy as np
import matplotlib.pyplot as plt
from lib.config.config_loader import setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment

def run_sim(with_noise=False, ticks=40):
    fgs = setup_full_mareld_mvp()
    grid_config = {'width': 60, 'height': 60, 'cell_size': 1000.0, 'tick_duration': 6.0}
    env = EcosystemEnvironment(grid_config, fgs)
    
    if with_noise:
        # Load windfarm noise proxy
        try:
            env.grid.load_map_from_png('windfarm_noise', 'vindparker.png', scale=1.0)
        except:
            env.grid.add_map('windfarm_noise', np.ones((60, 60)) * 0.5)
    else:
        env.grid.add_map('windfarm_noise', np.zeros((60, 60)))
        
    env.grid.add_map('bottom_trawling', np.zeros((60, 60)))
    env.grid.add_map('pelagic_trawling', np.zeros((60, 60)))
    env.grid.add_map('rotor', np.zeros((60, 60)))
    
    history = {fid: [] for fid in env.fgs}
    for t in range(ticks):
        observation = env.get_observation()
        actions = env.policy_controller.forward(observation)
        env.step(actions)
        for fid, fg in env.fgs.items():
            history[fid].append(fg.biomass.sum())
    return history

def main():
    print("Running Nollalternativ...")
    hist_noll = run_sim(with_noise=False)
    
    print("Running Projektalternativ...")
    hist_proj = run_sim(with_noise=True)
    
    # Compare specific species, e.g., porpoise (tumlare)
    species = ['porpoises', 'gadoids', 'pelagic_fish']
    
    plt.figure(figsize=(12, 8))
    for i, sp in enumerate(species):
        plt.subplot(len(species), 1, i+1)
        plt.plot(hist_noll[sp], label='Noll (Reference)')
        plt.plot(hist_proj[sp], label='Projekt (Noise)')
        plt.title(f'Biomass Development: {sp}')
        plt.ylabel('Tons')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('results/scenario_comparison.png')
    print("Comparison plot saved to results/scenario_comparison.png")

if __name__ == "__main__":
    main()
