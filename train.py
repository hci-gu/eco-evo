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
def _sample_impact_maps(impact_vars, impact_ranges, grid_size, seed=None):
    """Sample one impact field per active impact variable.

    Each cell is drawn i.i.d. uniformly from the impact's configured
    ``[value_min, value_max]`` range. Returns a dict ``{impact_id: np.ndarray}``.
    """
    H, W = grid_size
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    maps = {}
    for iv in impact_vars:
        vmin, vmax = impact_ranges.get(iv, (0.0, 0.0))
        if vmax > vmin:
            field = rng.uniform(vmin, vmax, size=(H, W))
        else:
            field = np.full((H, W), float(vmin))
        maps[iv] = field.astype(np.float32)
    return maps

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

def _save_checkpoint(trainer, fg_id, path):
    """Save policy weights AND running obs-normalisation stats (if any).

    Format: {'state_dict': ..., 'obs_stats': {'mean': ..., 'var': ..., 'count': N}}.
    For backwards compatibility, code loading older .pth files (a bare
    state_dict) should still work — torch.save preserves dict structure here.
    """
    payload = {'state_dict': trainer.policies[fg_id].state_dict()}
    if getattr(trainer, 'obs_stats', None) and fg_id in trainer.obs_stats:
        st = trainer.obs_stats[fg_id]
        payload['obs_stats'] = {
            'mean': st['mean'].astype(np.float32),
            'var': st['var'].astype(np.float32),
            'count': int(st['count']),
        }
    torch.save(payload, path)


def _load_checkpoint(trainer, fg_id, path):
    """Load policy weights (and obs-normalisation stats if present) for ``fg_id``.

    Supports two on-disk formats:
      * New: ``{'state_dict': ..., 'obs_stats': {'mean','var','count'}}``
      * Legacy: a bare ``state_dict`` produced by an older training run.
    Returns True on success, False if the file is unreadable / incompatible.
    """
    try:
        payload = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"    [resume] Failed to read {path}: {e}")
        return False

    if isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
        obs_stats = payload.get('obs_stats')
    else:
        state_dict = payload
        obs_stats = None

    try:
        trainer.policies[fg_id].load_state_dict(state_dict)
    except Exception as e:
        print(f"    [resume] Could not load weights for {fg_id} from {path}: {e}")
        return False

    if obs_stats is not None and getattr(trainer, 'obs_stats', None) is not None:
        try:
            mean = np.asarray(obs_stats['mean'], dtype=np.float64)
            var = np.asarray(obs_stats['var'], dtype=np.float64)
            count = int(obs_stats['count'])
            trainer.obs_stats[fg_id] = {'mean': mean, 'var': var, 'count': count}
        except Exception as e:
            print(f"    [resume] Loaded weights but failed to restore obs_stats for {fg_id}: {e}")
    return True


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

class _EnvBuilder:
    """Picklable env_builder callable used by the training loop.

    When ``impact_maps_snapshot`` is provided, every env produced by this
    callable installs *those exact* impact fields (gemensamma per
    generation). When ``None``, fresh maps are sampled per call from the
    project's impact ranges — used only for the initial temp_env probe
    before the training loop starts.

    Implemented as a top-level class (not a closure) so that the
    multiprocessing 'spawn' start method can pickle it when it gets
    passed to worker processes via ``trainer.env_builder``.
    """

    def __init__(self, impact_maps_snapshot=None):
        self.impact_maps_snapshot = impact_maps_snapshot

    def __call__(self, seed=None):
        grid_size = (GRID_HEIGHT, GRID_WIDTH)
        impact_ranges = {}
        if PROJECT_PATH:
            fgs, impact_vars, impact_ranges, observable_impact_vars = load_project_config(
                PROJECT_PATH, grid_size=grid_size, seed=seed)
        else:
            fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed)
            impact_vars = ['windfarm_noise']
            observable_impact_vars = ['windfarm_noise']

        grid_config = {
            'width': GRID_WIDTH,
            'height': GRID_HEIGHT,
            'cell_size': 1000.0,
            'tick_duration': 6.0,
        }
        env = EcosystemEnvironment(grid_config, fgs, {},
                                   observable_impact_vars=observable_impact_vars)

        if self.impact_maps_snapshot is not None:
            for iv in impact_vars:
                field = self.impact_maps_snapshot.get(iv)
                if field is None:
                    field = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
                env.grid.add_map(iv, field)
        else:
            sampled = _sample_impact_maps(
                impact_vars, impact_ranges,
                (GRID_HEIGHT, GRID_WIDTH), seed=seed)
            for iv in impact_vars:
                env.grid.add_map(iv, sampled.get(
                    iv, np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)))
        return env


def _make_env_builder(impact_maps_snapshot=None):
    """Factory kept for call-site compatibility; returns a picklable
    ``_EnvBuilder`` instance."""
    return _EnvBuilder(impact_maps_snapshot)


# Default module-level env_builder: fresh impact maps per call. Used for
# the initial temp_env probe before the main loop installs a generation-
# specific env_builder via _make_env_builder(maps).
env_builder = _make_env_builder(None)

def get_dynamic_policy_params(fgs, n_observable_impacts=0):
    """
    Calculates policy network dimensions dynamically based on the state of the environment.

    ``n_observable_impacts`` is the number of impact_ids flagged as observable
    in the project (one observation channel per id). When zero, the policy
    sees only biomass + energy + other-FG channels.
    """
    params = {}
    n_fgs = len(fgs)
    # Per cell the policy sees the von Neumann neighbourhood (center + N/E/S/W)
    # as prescribed by Method.pdf. The center contributes
    #   center_dim = n_fgs + 1 + n_observable_impacts
    # (B_own, E_own, B_others(N-1), observable impacts). Each of the four
    # neighbours contributes the same set *minus* E_own:
    #   nbr_dim    = n_fgs     + n_observable_impacts
    # Total input dimension:
    n_obs = int(n_observable_impacts)
    center_dim = n_fgs + 1 + n_obs
    nbr_dim = n_fgs + n_obs
    in_dim = center_dim + 4 * nbr_dim
    
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
    parser.add_argument("--n_eval_ticks", type=int, default=15, help="Number of time steps (ticks) per evaluation rollout (default: 15).")
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
    parser.add_argument("--top_deltas", type=int, default=None,
                        help="ARS-V2 top-b: keep only the best b delta pairs (sorted by "
                             "max(r_pos, r_neg)) when forming the gradient. Default: n_deltas // 2. "
                             "Set equal to --n_deltas to disable truncation.")
    parser.add_argument("--no_obs_normalize", action="store_true",
                        help="Disable ARS-V2 running observation normalisation (mean/std).")
    parser.add_argument("--entropy_coef", type=float, default=0.1,
                        help="Entropy bonus weight in fitness: fitness += entropy_coef * H(pi)/H_max. "
                             "Motverkar att softmax-policyn kollapsar till en konstant action. "
                             "0.0 = av (default: 0.1).")
    parser.add_argument("--argmax_penalty", type=float, default=0.3,
                        help="Argmax-penalty weight: fitness -= argmax_penalty * max_argmax_frac, "
                             "där max_argmax_frac = max(move,rest,eat)-fraktion över aktiva celler. "
                             "Straffar direkt degenererade en-action-policyer. 0.0 = av (default: 0.3).")
    parser.add_argument("--temp_start", type=float, default=3.0,
                        help="Softmax-temperatur vid generation 1. Hög T -> jämnare softmax -> "
                             "tvingad utforskning. Annealas linjärt till --temp_end. Default: 3.0.")
    parser.add_argument("--temp_end", type=float, default=1.0,
                        help="Softmax-temperatur vid sista generationen (default: 1.0).")
    parser.add_argument("--integral_reward", action="store_true", default=True,
                        help="Använd medel-biomassa/medel-energy över hela rolloutet istället "
                             "för slutvärdet i fitness-beräkningen. Ger \"ät alltid\" negativ "
                             "gradient i sig själv via byteskollaps under rolloutet. Default: True.")
    parser.add_argument("--no_integral_reward", dest="integral_reward", action="store_false",
                        help="Stäng av integral-reward, använd klassisk slutvärde-fitness.")
    parser.add_argument("--temp_anneal_gens", type=int, default=10,
                        help="Antal generationer över vilka temperaturen annealas linjärt "
                             "från --temp_start till --temp_end. Efter detta håller den --temp_end. "
                             "Default: 10.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previously saved checkpoints in results/policy_<fg>.pth "
                             "for ALL decision makers (not only the trained ones). Missing "
                             "checkpoints are silently skipped and start from random init.")

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
    policy_params = get_dynamic_policy_params(
        temp_env.fgs,
        n_observable_impacts=len(getattr(temp_env, 'observable_impact_vars', []) or []),
    )
    
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
    # Resolve top_deltas (None -> n_deltas // 2)
    if args.top_deltas is None:
        top_deltas_resolved = max(1, n_deltas // 2)
        top_origin = f"default: n_deltas // 2 = {top_deltas_resolved}"
    else:
        top_deltas_resolved = max(1, min(args.top_deltas, n_deltas))
        top_origin = "user"
    print(f"Top Deltas:     {top_deltas_resolved} ({top_origin})")
    obs_norm_enabled = not args.no_obs_normalize
    print(f"Obs Normalize:  {obs_norm_enabled} {'(default)' if not args.no_obs_normalize else '(user, disabled)'}")
    if args.workers > 0:
        print(f"Workers:        {n_workers} (user, explicit)")
    else:
        print(f"Workers:        {n_workers} (default: auto, resolved from {os.cpu_count()} CPUs and n_deltas={n_deltas})")
    print(f"------------------------------------------")

    # Prompt user to confirm parameters before starting training.
    # Enter (empty) or 'y' continues; 'n' aborts. Loops on invalid input.
    try:
        while True:
            resp = input("Continue with these parameters? [Y/n]: ").strip().lower()
            if resp in ("", "y", "yes"):
                break
            if resp in ("n", "no"):
                print("Aborted by user.")
                return
            print("Please answer 'y' or 'n' (or press Enter for default 'y').")
    except EOFError:
        # No interactive stdin available — proceed with defaults.
        pass

    # Create the trainer with all relevant policy dimensions
    trainer = ARSTrainer(env_builder, policy_params, sigma=args.sigma, lr=args.lr, n_deltas=n_deltas,
                         n_workers=n_workers, alpha=args.alpha, beta=args.beta,
                         obs_normalize=obs_norm_enabled, top_deltas=top_deltas_resolved,
                         entropy_coef=args.entropy_coef, argmax_penalty=args.argmax_penalty,
                         integral_reward=args.integral_reward)

    # Optionally resume from previously saved checkpoints. We always load for
    # ALL decision makers (not just the target species) so that single-species
    # runs co-evolve against previously trained policies rather than random ones.
    if args.resume:
        print(f"Resume:         enabled (loading checkpoints for all DMs)")
        loaded, missing = [], []
        for fg_id in policy_params:
            ckpt = f"results/policy_{fg_id}.pth"
            if os.path.exists(ckpt):
                if _load_checkpoint(trainer, fg_id, ckpt):
                    loaded.append(fg_id)
                else:
                    missing.append(fg_id)
            else:
                missing.append(fg_id)
        if loaded:
            print(f"    Loaded checkpoints: {', '.join(loaded)}")
        if missing:
            print(f"    No checkpoint (random init): {', '.join(missing)}")
        print(f"------------------------------------------")
    
    # Discover the project's active impact variables + their value ranges
    # once. These drive the per-generation impact map sampling below.
    if PROJECT_PATH:
        _, _impact_vars_global, _impact_ranges_global, _ = load_project_config(
            PROJECT_PATH, grid_size=(GRID_HEIGHT, GRID_WIDTH), seed=0)
    else:
        _impact_vars_global = ['windfarm_noise']
        _impact_ranges_global = {}

    def _install_generation_maps(gen_idx):
        """Sample one shared set of impact maps for the whole generation
        and install it via a fresh env_builder closure. Rebuilds the
        worker pool (if any) so all workers see the new maps."""
        gen_seed = int(np.random.randint(1, 2**31 - 1))
        maps = _sample_impact_maps(_impact_vars_global, _impact_ranges_global,
                                   (GRID_HEIGHT, GRID_WIDTH), seed=gen_seed)
        new_builder = _make_env_builder(maps)
        trainer.env_builder = new_builder
        # Rebuild worker pool so spawn-workers receive the updated builder.
        if trainer._pool is not None:
            try:
                trainer._pool.terminate()
                trainer._pool.join()
            except Exception:
                pass
            import multiprocessing as _mp
            from lib.runners.parallel_worker import _worker_init
            ctx = _mp.get_context('spawn')
            trainer._pool = ctx.Pool(
                processes=trainer.n_workers,
                initializer=_worker_init,
                initargs=(new_builder, trainer.policy_params),
            )
        if maps:
            summary = ", ".join(
                f"{k}=U[{_impact_ranges_global.get(k,(0,0))[0]:g},"
                f"{_impact_ranges_global.get(k,(0,0))[1]:g}]"
                for k in maps)
            print(f"    Impact maps (shared this gen): {summary}")

    import itertools
    gen_iter = itertools.count() if generations_is_inf else range(generations_value)
    gen_label_total = "inf" if generations_is_inf else str(generations_value)

    try:
        for gen in gen_iter:
            # Linear softmax-temperature annealing from temp_start -> temp_end
            # over the first temp_anneal_gens generations.
            anneal_n = max(1, int(args.temp_anneal_gens))
            frac = min(1.0, gen / max(1, anneal_n - 1)) if anneal_n > 1 else 1.0
            T = float(args.temp_start + (args.temp_end - args.temp_start) * frac)
            trainer.softmax_temperature = T
            print(f"\n========== Generation {gen+1}/{gen_label_total} (T={T:.3f}) ==========")
            _install_generation_maps(gen)
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
                _save_checkpoint(trainer, species, save_path)
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
                _save_checkpoint(trainer, species, save_path)
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
