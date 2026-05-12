import torch
import numpy as np
import os
import argparse
import re
from lib.config.config_loader import setup_full_mareld_mvp, load_project_config
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.trainer import ARSTrainer

# Global variable to store project path for env_builder
PROJECT_PATH = None
GRID_WIDTH = 60
GRID_HEIGHT = 60

def parse_grid_arg(value):
    """Parses a --grid argument on the form n*m (or nxm). Both dims must be >= 3."""
    if value is None:
        return None
    m = re.match(r'^\s*(\d+)\s*[\*xX]\s*(\d+)\s*$', value)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid format: '{value}'. Expected n*m (e.g. 30*30)."
        )
    n, k = int(m.group(1)), int(m.group(2))
    if n < 3 or k < 3:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid '{value}': both dimensions must be >= 3."
        )
    return n, k

def env_builder():
    """
    Creates a new instance of the ecosystem for each rollout.
    This is required by the ARS algorithm to evaluate different perturbations (deltas).
    """
    grid_size = (GRID_HEIGHT, GRID_WIDTH)
    if PROJECT_PATH:
        fgs, impact_vars = load_project_config(PROJECT_PATH, grid_size=grid_size)
    else:
        fgs = setup_full_mareld_mvp(grid_size=grid_size)
        impact_vars = ['windfarm_noise']
        
    grid_config = {
        'width': GRID_WIDTH,
        'height': GRID_HEIGHT,
        'cell_size': 1000.0,
        'tick_duration': 6.0
    }
    # Interactions are loaded from FG parameters internally in the MVP version
    env = EcosystemEnvironment(grid_config, fgs, {})
    
    # Add necessary map layers as empty dummies for training
    for iv in impact_vars:
        env.grid.add_map(iv, np.zeros((GRID_HEIGHT, GRID_WIDTH)))
        
    return env

def get_dynamic_policy_params(fgs):
    """
    Calculates policy network dimensions dynamically based on the state of the environment.
    """
    params = {}
    n_fgs = len(fgs)
    # Input: Biomass(self), Energy(self), Biomass(all others), Noise(1)
    # Total number of layers in the observation is n_fgs + 2
    in_dim = n_fgs + 2
    
    for fg_id, fg in fgs.items():
        if fg.is_decision_maker:
            # Output: Move(4) + Rest(1) + Eat(N_prey)
            menu_size = len(fg.params.get('menu', []))
            out_dim = 5 + menu_size
            params[fg_id] = (in_dim, out_dim)
    return params

def main():
    parser = argparse.ArgumentParser(description="Mareld Ecosystem Simulator - Training Module")
    parser.add_argument("--species", nargs="+", default=["pelagic_fish"], 
                        help="Which functional groups to train (e.g., pelagic_fish gadoids). Use 'all' for all decision makers.")
    parser.add_argument("--iterations", type=int, default=15, help="Number of ARS iterations per species (default: 15).")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate (default: 0.03).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Exploration noise (default: 0.1).")
    parser.add_argument("--rollouts", type=int, default=4, help="Number of time steps per evaluation (default: 4).")
    parser.add_argument("--project", type=str, help="Path to project file (.yaml)")
    parser.add_argument("--grid", type=parse_grid_arg, default=None,
                        help="Grid dimensions as n*m (e.g. 30*30). Both dimensions must be >= 3. Default: 60*60.")
    
    args = parser.parse_args()

    if args.grid is not None:
        global GRID_WIDTH, GRID_HEIGHT
        GRID_WIDTH, GRID_HEIGHT = args.grid
        print(f"Grid size set to {GRID_WIDTH} x {GRID_HEIGHT}.")

    if not args.project:
        print("\nError: No project file specified.")
        print("You must specify a project file with the --project flag to start training.")
        print("Example: python3 train.py --project mareld2.yaml\n")
        return

    # Set project globally so env_builder can find it
    global PROJECT_PATH
    PROJECT_PATH = args.project

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Initialize a temporary environment to fetch functional group metadata
    temp_env = env_builder()
    policy_params = get_dynamic_policy_params(temp_env.fgs)
    
    # Handle species selection
    requested_species = args.species
    if "all" in requested_species:
        target_species = list(policy_params.keys())
    else:
        target_species = [s for s in requested_species if s in policy_params]
        
    if not target_species:
        available = ", ".join(policy_params.keys())
        print(f"Error: No valid trainable species found. Available: {available}")
        return

    print(f"==========================================")
    print(f"      MARELD TRAINING SESSION             ")
    print(f"==========================================")
    print(f"Target Species: {', '.join(target_species)}")
    print(f"Method:         ARS (Augmented Random Search)")
    print(f"Iterations:     {args.iterations} per species")
    print(f"Learning Rate:  {args.lr}")
    print(f"Sigma:          {args.sigma}")
    print(f"------------------------------------------")

    # Create the trainer with all relevant policy dimensions
    trainer = ARSTrainer(env_builder, policy_params, sigma=args.sigma, lr=args.lr, n_deltas=8)
    
    for species in target_species:
        print(f"\n>>> Starting training for: {species.upper()}")
        print(f"    Input dim:  {policy_params[species][0]}")
        print(f"    Output dim: {policy_params[species][1]}")
        
        for i in range(args.iterations):
            # n_rollouts: how many time steps (ticks) each test run lasts
            avg_reward = trainer.train_step(species, n_rollouts=args.rollouts)
            print(f"    Iter {i+1:2d}/{args.iterations} | Avg Reward: {avg_reward:10.6f}")

        # Save the trained model
        save_path = f"results/policy_{species}.pth"
        torch.save(trainer.policies[species].state_dict(), save_path)
        print(f"    Training complete. Model saved to: {save_path}")

    print(f"\n==========================================")
    print(f"Training completed for all selected groups.")
    print(f"==========================================")

if __name__ == "__main__":
    main()
