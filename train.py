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

def _auto_workers(n_deltas):
    """Pick a sensible default worker count.

    Rules:
    - Cap at the number of tasks per train_step (2 * n_deltas); more workers
      give no speedup since pool.map is synchronous per step.
    - Leave 1 core for the parent/OS to reduce context-switch overhead.
    - Prefer sched_getaffinity (respects cgroups/taskset) when available.
    - Return 1 (sequential) on tiny machines or when target < 2.
    """
    n_tasks = 2 * n_deltas
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    target = min(n_tasks, max(1, cores - 1))
    return target if target >= 2 else 1

def env_builder(seed=None):
    """
    Creates a new instance of the ecosystem for each rollout.
    This is required by the ARS algorithm to evaluate different perturbations (deltas).

    If `seed` is given, the initial spatial biomass distribution is deterministic.
    This enables Common Random Numbers in ARS (matching +delta and -delta worlds).
    """
    grid_size = (GRID_HEIGHT, GRID_WIDTH)
    if PROJECT_PATH:
        fgs, impact_vars = load_project_config(PROJECT_PATH, grid_size=grid_size, seed=seed)
    else:
        fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed)
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
    
    # Output is now uniform across all decision makers: Move(4) + Rest(1) + Eat(N_fgs).
    # Eat-slots are indexed by a globally sorted FG list; slots outside the
    # predator's menu are permanently masked to 0 by the environment.
    out_dim = 5 + n_fgs
    for fg_id, fg in fgs.items():
        if fg.is_decision_maker:
            params[fg_id] = (in_dim, out_dim)
    return params

def main():
    parser = argparse.ArgumentParser(description="Mareld Ecosystem Simulator - Training Module")
    parser.add_argument("--species", nargs="+", default=["all"],
                        help="Which functional groups to train (e.g., pelagic_fish gadoids). Use 'all' for all decision makers (default: all).")
    parser.add_argument("--iter-per-gen", "--iter_per_gen", dest="iter_per_gen", type=int, default=20,
                        help="ARS iterations per species per generation (default: 20).")
    parser.add_argument("--generations", type=str, default="inf",
                        help="Number of co-evolution generations (outer round-robin loop over species). "
                             "Use 'inf' (default) to run until interrupted with Ctrl+C.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate (default: 0.03).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Exploration noise (default: 0.1).")
    parser.add_argument("--n_eval_ticks", type=int, default=2, help="Number of time steps (ticks) per evaluation rollout (default: 2).")
    parser.add_argument("--project", type=str, help="Path to project file (.yaml)")
    parser.add_argument("--grid", type=parse_grid_arg, default=None,
                        help="Grid dimensions as n*m (e.g. 30*30). Both dimensions must be >= 3. Default: 60*60.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for delta_b (biomass log-ratio) in fitness (default: 1.0).")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for delta_r (energy log-ratio) in fitness (default: 1.0).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel rollout workers. 0 = auto (default, based on CPU cores and n_deltas), "
                             "1 = sequential, N = use N workers.")
    parser.add_argument("--n_deltas", type=int, default=10,
                        help="Number of ARS perturbation directions per iteration (default: 10).")
    
    args = parser.parse_args()

    # Parse --generations: accept 'inf' or a positive integer.
    gen_raw = str(args.generations).strip().lower()
    if gen_raw in ("inf", "infinity", "infinite", "-1"):
        generations_is_inf = True
        generations_value = None
    else:
        try:
            generations_value = int(gen_raw)
            if generations_value < 1:
                raise ValueError
        except ValueError:
            print(f"Error: --generations must be a positive integer or 'inf' (got: {args.generations!r}).")
            return
        generations_is_inf = False

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

    # Build a map of parser defaults so we can flag which values are user-specified.
    _parser_defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    def _mark(name, value):
        default_val = _parser_defaults.get(name, None)
        is_default = (value == default_val)
        return f"{value} {'(default)' if is_default else '(user)'}"

    # Resolve worker count (0 = auto) — done here so we can include it in the summary.
    n_deltas = args.n_deltas
    if args.workers > 0:
        n_workers = args.workers
        workers_origin = "explicit"
    else:
        n_workers = _auto_workers(n_deltas)
        workers_origin = "auto"

    grid_str = f"{GRID_WIDTH}x{GRID_HEIGHT}"
    grid_is_default = (args.grid is None)
    gen_display = "inf (Ctrl+C to stop)" if generations_is_inf else str(generations_value)
    gen_is_default = (str(args.generations).strip().lower() == str(_parser_defaults.get("generations", "")).strip().lower())
    species_is_default = (args.species == _parser_defaults.get("species"))

    print(f"==========================================")
    print(f"      MARELD TRAINING SESSION             ")
    print(f"==========================================")
    print(f"Project:        {args.project} (user)")
    print(f"Grid:           {grid_str} {'(default)' if grid_is_default else '(user)'}")
    print(f"Target Species: {', '.join(target_species)} {'(default: all)' if species_is_default else '(user)'}")
    print(f"Method:         ARS (Augmented Random Search)")
    print(f"Generations:    {gen_display} {'(default)' if gen_is_default else '(user)'}")
    print(f"Iter/Gen:       {_mark('iter_per_gen', args.iter_per_gen)} per species per generation")
    print(f"N Eval Ticks:   {_mark('n_eval_ticks', args.n_eval_ticks)} ticks per rollout")
    print(f"Learning Rate:  {_mark('lr', args.lr)}")
    print(f"Sigma:          {_mark('sigma', args.sigma)}")
    print(f"Alpha (delta_b):{_mark('alpha', args.alpha)}")
    print(f"Beta  (delta_r):{_mark('beta', args.beta)}")
    print(f"N Deltas:       {_mark('n_deltas', args.n_deltas)}")
    if args.workers > 0:
        print(f"Workers:        {n_workers} (user, explicit)")
    else:
        print(f"Workers:        {n_workers} (default: auto, resolved from {os.cpu_count()} CPUs and n_deltas={n_deltas})")
    print(f"------------------------------------------")

    # Create the trainer with all relevant policy dimensions
    trainer = ARSTrainer(env_builder, policy_params, sigma=args.sigma, lr=args.lr, n_deltas=n_deltas,
                         n_workers=n_workers, alpha=args.alpha, beta=args.beta)
    
    import itertools
    gen_iter = itertools.count() if generations_is_inf else range(generations_value)
    gen_label_total = "inf" if generations_is_inf else str(generations_value)

    try:
        for gen in gen_iter:
            print(f"\n========== Generation {gen+1}/{gen_label_total} ==========")
            for species in target_species:
                print(f"\n>>> Training: {species.upper()} (gen {gen+1}/{gen_label_total})")
                print(f"    Input dim:  {policy_params[species][0]}")
                print(f"    Output dim: {policy_params[species][1]}")

                for i in range(args.iter_per_gen):
                    # n_eval_ticks: how many time steps (ticks) each test run lasts
                    avg_reward = trainer.train_step(species, n_eval_ticks=args.n_eval_ticks)
                    print(f"    Iter {i+1:2d}/{args.iter_per_gen} | Avg Reward: {avg_reward:10.6f}")

                # Save checkpoint after each generation so progress is preserved.
                save_path = f"results/policy_{species}.pth"
                torch.save(trainer.policies[species].state_dict(), save_path)
                print(f"    Checkpoint saved to: {save_path}")
    except KeyboardInterrupt:
        print(f"\n\n[Interrupted by user] Stopping training after current step.")
        # Terminate workers immediately so they don't keep computing while we
        # save checkpoints; otherwise pool.map can still hold references.
        try:
            trainer.close()
        except Exception:
            pass
        for species in target_species:
            save_path = f"results/policy_{species}.pth"
            try:
                torch.save(trainer.policies[species].state_dict(), save_path)
                print(f"    Final checkpoint saved to: {save_path}")
            except Exception as e:
                print(f"    Could not save checkpoint for {species}: {e}")
        print(f"\n==========================================")
        print(f"Training interrupted; partial progress saved.")
        print(f"==========================================")
        return

    print(f"\n==========================================")
    print(f"Training completed for all selected groups.")
    print(f"==========================================")
    trainer.close()

if __name__ == "__main__":
    main()
