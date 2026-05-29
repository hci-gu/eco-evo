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
    project's impact ranges - used only for the initial temp_env probe
    before the training loop starts.

    STEP 2 (multi-world averaging scaffold): the builder also carries
    the (impact_seed, spawn_seed) pair that defines one *world* from the
    iteration's WorldList. ``with_world(impact_seed, spawn_seed)``
    produces a fresh sibling builder bound to a different world, sharing
    the same impact_vars / ranges / grid / project. This is what the
    M>1 rollout loop in train_step will consume in Step 3; for now only
    world index 0 is honoured by the legacy single-rollout path, so
    semantics remain effectively unchanged.

    Implemented as a top-level class (not a closure) so that the
    multiprocessing 'spawn' start method can pickle it when it gets
    passed to worker processes via ``trainer.env_builder``.
    """

    def __init__(self, impact_maps_snapshot=None, grid_size=None,
                 project_path=None, spawn_seed=None,
                 impact_vars=None, impact_ranges=None, impact_seed=None):
        self.impact_maps_snapshot = impact_maps_snapshot
        # Step 2: keep the raw impact recipe so ``with_world`` can
        # re-sample maps for a different impact_seed without needing
        # access to module globals (which are reset in spawn-workers).
        self.impact_vars = list(impact_vars) if impact_vars is not None else None
        self.impact_ranges = dict(impact_ranges) if impact_ranges is not None else None
        self.impact_seed = impact_seed
        # Per-generation spawn seed: when set, every env built by this
        # callable produces *identical* biomass maps for all FGs (shared
        # across deltas/workers), analogous to ``impact_maps_snapshot``.
        # ``None`` falls back to per-rollout sampling (legacy).
        self.spawn_seed = spawn_seed
        # Bake grid dims into the instance so worker processes (which
        # reimport train.py via 'spawn' and would otherwise see the
        # module-level 60x60 defaults) build env with the correct shape.
        # Falls back to the module globals when not supplied.
        if grid_size is None:
            grid_size = (GRID_HEIGHT, GRID_WIDTH)
        self.grid_height = int(grid_size[0])
        self.grid_width = int(grid_size[1])
        # Bake project path into the instance as well. Spawn-workers reimport
        # train.py, which resets the module-level ``PROJECT_PATH`` global to
        # ``None``, causing the worker to fall back to ``setup_full_mareld_mvp``
        # (different FG set + a hardcoded observable impact) and producing an
        # obs-dim mismatch with the parent's policy_params -> crash inside
        # the policy forward (RuntimeError: mat1 and mat2 shapes cannot be
        # multiplied). Falls back to the module global when not supplied.
        self.project_path = project_path if project_path is not None else PROJECT_PATH

    def with_world(self, impact_seed, spawn_seed):
        """Return a fresh sibling builder bound to a different world.

        Used by the multi-world averaging path (Step 3): the trainer
        picks world ``m`` from the WorldList and obtains a builder whose
        impact_maps_snapshot is freshly sampled from ``impact_seed`` and
        whose spawn layout is locked by ``spawn_seed``. All other state
        (grid, project, impact recipe) is inherited. Returns a new
        picklable ``_EnvBuilder`` instance; the original is untouched.
        """
        if self.impact_vars is None or self.impact_ranges is None:
            raise RuntimeError(
                "_EnvBuilder.with_world requires impact_vars/impact_ranges "
                "to be baked in (use _make_env_builder with the project's "
                "impact recipe).")
        maps = _sample_impact_maps(
            self.impact_vars, self.impact_ranges,
            (self.grid_height, self.grid_width), seed=int(impact_seed))
        return _EnvBuilder(
            impact_maps_snapshot=maps,
            grid_size=(self.grid_height, self.grid_width),
            project_path=self.project_path,
            spawn_seed=int(spawn_seed),
            impact_vars=self.impact_vars,
            impact_ranges=self.impact_ranges,
            impact_seed=int(impact_seed),
        )

    def __call__(self, seed=None):
        H, W = self.grid_height, self.grid_width
        grid_size = (H, W)
        impact_ranges = {}
        if self.project_path:
            fgs, impact_vars, impact_ranges, observable_impact_vars = load_project_config(
                self.project_path, grid_size=grid_size, seed=seed,
                spawn_seed=self.spawn_seed)
        else:
            fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed,
                                        spawn_seed=self.spawn_seed)
            impact_vars = ['windfarm_noise']
            observable_impact_vars = ['windfarm_noise']

        grid_config = {
            'width': W,
            'height': H,
            'cell_size': 1000.0,
            'tick_duration': 6.0,
        }
        env = EcosystemEnvironment(grid_config, fgs, {},
                                   observable_impact_vars=observable_impact_vars)

        if self.impact_maps_snapshot is not None:
            for iv in impact_vars:
                field = self.impact_maps_snapshot.get(iv)
                if field is None:
                    field = np.zeros((H, W), dtype=np.float32)
                # Defensive: if a snapshot field has the wrong shape
                # (e.g. parent sampled at a different grid before the
                # builder was rebuilt), fall back to zeros instead of
                # crashing inside env.grid.add_map.
                elif field.shape != (H, W):
                    field = np.zeros((H, W), dtype=np.float32)
                env.grid.add_map(iv, field)
        else:
            sampled = _sample_impact_maps(
                impact_vars, impact_ranges,
                (H, W), seed=seed)
            for iv in impact_vars:
                env.grid.add_map(iv, sampled.get(
                    iv, np.zeros((H, W), dtype=np.float32)))
        return env


def _make_env_builder(impact_maps_snapshot=None, grid_size=None,
                      project_path=None, spawn_seed=None,
                      impact_vars=None, impact_ranges=None, impact_seed=None):
    """Factory kept for call-site compatibility; returns a picklable
    ``_EnvBuilder`` instance with an explicit ``grid_size`` and
    ``project_path`` baked in so 'spawn' workers don't fall back to the
    module-level defaults (which are reset to ``None``/60x60 inside the
    worker after reimport).

    ``spawn_seed`` (optional): when set, all envs built by the returned
    callable share the same biomass spawn layout (per FG). Used by
    ``_install_generation_maps`` to lock the spawn pattern for one
    generation, just like ``impact_maps_snapshot`` does for impacts.
    """
    return _EnvBuilder(impact_maps_snapshot, grid_size=grid_size,
                      project_path=project_path, spawn_seed=spawn_seed,
                      impact_vars=impact_vars, impact_ranges=impact_ranges,
                      impact_seed=impact_seed)


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
                             "Counteracts softmax-policy collapse to a constant action. "
                             "0.0 = off (default: 0.1).")
    parser.add_argument("--argmax_penalty", type=float, default=0.3,
                        help="Argmax-penalty weight: fitness -= argmax_penalty * max_argmax_frac, "
                             "where max_argmax_frac = max(move,rest,eat) fraction across active cells. "
                             "Directly penalises degenerate single-action policies. 0.0 = off (default: 0.3).")
    parser.add_argument("--temp_start", type=float, default=3.0,
                        help="Softmax temperature at generation 1. High T -> flatter softmax -> "
                             "forced exploration. Linearly annealed to --temp_end. Default: 3.0.")
    parser.add_argument("--temp_end", type=float, default=1.0,
                        help="Softmax temperature at the last generation (default: 1.0).")
    parser.add_argument("--integral_reward", action="store_true", default=True,
                        help="Use mean biomass / mean energy over the whole rollout instead "
                             "of the final value in the fitness computation. Gives \"eat always\" a "
                             "negative gradient on its own via prey collapse during the rollout. Default: True.")
    parser.add_argument("--no_integral_reward", dest="integral_reward", action="store_false",
                        help="Disable integral reward, use classic final-value fitness.")
    parser.add_argument("--temp_anneal_gens", type=int, default=10,
                        help="Number of generations over which the temperature is annealed linearly "
                             "from --temp_start to --temp_end. After that it stays at --temp_end. "
                             "Default: 10.")
    parser.add_argument("--coevolution", action="store_true", default=True,
                        help="Enable co-evolution (track C in konvergensproblem.txt): "
                             "train ALL target_species simultaneously in a shared rollout per "
                             "delta pair instead of round-robin. This creates negative feedback "
                             "between predator/prey policies (predator 'eat always' -> prey collapse "
                             "-> predator reward drops in the SAME rollout) that round-robin cannot see. "
                             "Per iteration n_deltas * 2 shared rollouts are evaluated, and each "
                             "species' weights are ARS-updated independently with its own reward vector. "
                             "Default: True. Use --no_coevolution to fall back to round-robin.")
    parser.add_argument("--no_coevolution", dest="coevolution", action="store_false",
                        help="Disable co-evolution and fall back to round-robin training (one species at a time).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previously saved checkpoints in results/policy_<fg>.pth "
                             "for ALL decision makers (not only the trained ones). Missing "
                             "checkpoints are silently skipped and start from random init.")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="default",
                        help="Name of the run. Checkpoints are saved to results/<run-name>/policy_<fg>.pth. "
                             "The same name can be used at inference via inference.py --run-name <name>. "
                             "Default: 'default'.")
    parser.add_argument("--uniform_bias_init", action="store_true", default=False,
                        help="Enable uniform-bias init on the output layer: bias=0 + "
                             "weights*0.01 so that softmax starts ~uniform at gen 1. "
                             "OFF by default in line with konvergensproblem.txt - use only "
                             "if a specific species (e.g. seals) is locked in a saturated "
                             "action attractor already at gen 1.")
    # --- Multi-world averaging per delta (variance reduction) ---
    # Implemented across STEPS 1-3 of the ARS-evolution plan in
    # mareld_resume.txt: CLI + seed bookkeeping (STEP 1), per-world
    # env builders via _EnvBuilder.with_world (STEP 2), and M-averaging
    # in ARSTrainer.train_step / train_step_coevolution (STEP 3). With
    # M=1 (default) the legacy single-world tuple-task path is used
    # byte-identically to pre-STEP-3 behaviour.
    parser.add_argument("--rollouts_per_delta", type=int, default=1, metavar="M",
                        help="Number of independent worlds (impact + spawn snapshots) "
                             "averaged per delta evaluation. M=1 reproduces the "
                             "legacy single-world path exactly. Typical useful "
                             "range: 3-5. Cost scales linearly: 2*n_deltas*M "
                             "rollouts per iteration. Within an iteration, +delta "
                             "and -delta share the SAME M worlds (CRN coupling) "
                             "so Var(r_pos - r_neg) drops faster than 1/M. "
                             "Default: 1.")
    parser.add_argument("--worlds_refresh", choices=["generation", "iteration"],
                        default="iteration",
                        help="When to resample the M worlds. 'generation' = lock M worlds "
                             "for the whole generation. 'iteration' = resample at the "
                             "start of every ARS iteration (recommended; avoids "
                             "over-fitting the gradient to a fixed world set). "
                             "Default: iteration.")
    parser.add_argument("--rollouts_per_delta_schedule", type=str, default=None,
                        metavar="SPEC",
                        help="Optional schedule, e.g. '1@0,3@10,5@50' meaning M=1 for "
                             "gen 0-9, M=3 for gen 10-49, M=5 from gen 50 onwards. "
                             "Overrides --rollouts_per_delta when given.")
    parser.add_argument("--profile", type=str, default=None,
                        choices=["sanity", "info", "deep"],
                        help="Preset hyperparameter profile for co-evolution: "
                             "'sanity' (~15 min quick check), 'info' (~1-2h standard run), "
                             "'deep' (~6-10h publication quality). Explicitly given CLI flags "
                             "OVERRIDE the profile values - the profile only fills in values "
                             "not specified on the command line.")

    args = parser.parse_args()

    # --- Profile application -----------------------------------------------
    # Each profile defines a fixed hyperparameter recipe. If --profile is set,
    # the profile fills in values for any flag NOT explicitly given on the
    # command line. Explicitly given CLI flags always take precedence over
    # the profile (explicit > profile > parser default).
    # The profiles NOW run without argmax-penalty, without entropy-bonus,
    # without softmax-temperature annealing (T=1.0 at both ends) and without
    # uniform-bias init - in line with konvergensproblem.txt's conclusion
    # to let the biological rules drive behaviour instead of
    # action-distribution hacks. The flags remain and can be enabled
    # manually without --profile.
    PROFILES = {
        "sanity": {
            "coevolution": True,
            "generations": "10",
            "iter_per_gen": 15,
            "n_deltas": 16,
            "n_eval_ticks": 100,
            "temp_anneal_gens": 8,
            "argmax_penalty": 0.0,
            "entropy_coef": 0.0,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "uniform_bias_init": False,
            "rollouts_per_delta": 3,
        },
        "info": {
            "coevolution": True,
            "generations": "10",
            "iter_per_gen": 20,
            "n_deltas": 16,
            "n_eval_ticks": 150,
            "temp_anneal_gens": 30,
            "argmax_penalty": 0.0,
            "entropy_coef": 0.0,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "uniform_bias_init": False,
            "rollouts_per_delta": 3,
        },
        "deep": {
            "coevolution": True,
            "generations": "80",
            "iter_per_gen": 20,
            "n_deltas": 20,
            "n_eval_ticks": 200,
            "temp_anneal_gens": 60,
            "argmax_penalty": 0.0,
            "entropy_coef": 0.0,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "uniform_bias_init": False,
            "rollouts_per_delta": 3,
        },
    }
    if args.profile is not None:
        prof = PROFILES[args.profile]
        # Determine which args were explicitly supplied on the command line by
        # re-parsing with all defaults set to a sentinel.
        import sys as _sys
        _sentinel = object()
        _sentinel_parser = argparse.ArgumentParser(add_help=False)
        for a in parser._actions:
            if a.dest == "help" or not a.option_strings:
                continue
            kwargs = {"dest": a.dest, "default": _sentinel}
            if isinstance(a, argparse._StoreTrueAction):
                kwargs["action"] = "store_const"; kwargs["const"] = True
            elif isinstance(a, argparse._StoreFalseAction):
                kwargs["action"] = "store_const"; kwargs["const"] = False
            else:
                kwargs["nargs"] = a.nargs
                kwargs["type"] = a.type
                kwargs["choices"] = a.choices
            _sentinel_parser.add_argument(*a.option_strings, **kwargs)
        _ns, _ = _sentinel_parser.parse_known_args()
        explicit = {k: v for k, v in vars(_ns).items() if v is not _sentinel}

        # Explicit CLI flags take precedence: only apply profile values for
        # keys that the user did NOT specify on the command line. Track which
        # profile values were skipped due to an explicit override so the user
        # gets clear feedback.
        applied = {}
        overridden = []
        for key, prof_val in prof.items():
            if key in explicit:
                # User-specified value wins; do not touch args.<key>.
                if explicit[key] != prof_val:
                    overridden.append((key, explicit[key], prof_val))
            else:
                setattr(args, key, prof_val)
                applied[key] = prof_val

        print(f"==========================================")
        print(f"  PROFILE ACTIVE: --profile {args.profile}")
        print(f"==========================================")
        if applied:
            print(f"Profile values applied (no explicit CLI override):")
            for key, prof_val in applied.items():
                print(f"  --{key.replace('_','-')} = {prof_val}")
        else:
            print(f"Profile '{args.profile}': every profile key was "
                  f"overridden by an explicit CLI flag.")
        if overridden:
            print(f"\n[i] EXPLICIT CLI FLAGS OVERRIDE PROFILE - the following "
                  f"profile values were NOT applied because you specified them "
                  f"explicitly on the command line:")
            for key, user_val, prof_val in overridden:
                print(f"  --{key.replace('_','-')}: using your value {user_val!r} "
                      f"(profile '{args.profile}' would have used {prof_val!r})")
        print(f"------------------------------------------")
    # -----------------------------------------------------------------------

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

    # Ensure run directory exists: results/<run-name>/
    run_dir = os.path.join('results', args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Initialize a temporary environment to fetch functional group metadata.
    # Build a fresh env_builder here (rather than reusing the module-level
    # one) so PROJECT_PATH set above is baked into the instance, ensuring
    # spawn-workers later receive the correct project path.
    env_builder_local = _make_env_builder(
        None, grid_size=(GRID_HEIGHT, GRID_WIDTH), project_path=PROJECT_PATH)
    temp_env = env_builder_local()
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
    if args.rollouts_per_delta_schedule:
        print(f"Rollouts/Delta: schedule={args.rollouts_per_delta_schedule} "
              f"(refresh={args.worlds_refresh}) (user)")
    else:
        print(f"Rollouts/Delta: {_mark('rollouts_per_delta', args.rollouts_per_delta)} "
              f"(refresh={args.worlds_refresh})")
    obs_norm_enabled = not args.no_obs_normalize
    print(f"Obs Normalize:  {obs_norm_enabled} {'(default)' if not args.no_obs_normalize else '(user, disabled)'}")
    print(f"Co-evolution:   {args.coevolution} {'(default: on)' if args.coevolution else '(user, disabled -> round-robin)'}")
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
    trainer = ARSTrainer(env_builder_local, policy_params, sigma=args.sigma, lr=args.lr, n_deltas=n_deltas,
                         n_workers=n_workers, alpha=args.alpha, beta=args.beta,
                         obs_normalize=obs_norm_enabled, top_deltas=top_deltas_resolved,
                         entropy_coef=args.entropy_coef, argmax_penalty=args.argmax_penalty,
                         integral_reward=args.integral_reward,
                         uniform_bias_init=args.uniform_bias_init)

    # Optionally resume from previously saved checkpoints. We always load for
    # ALL decision makers (not just the target species) so that single-species
    # runs co-evolve against previously trained policies rather than random ones.
    if args.resume:
        print(f"Resume:         enabled (loading checkpoints for all DMs)")
        loaded, missing = [], []
        for fg_id in policy_params:
            ckpt = os.path.join(run_dir, f"policy_{fg_id}.pth")
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

    # -----------------------------------------------------------------
    # STEP 1: Multi-world averaging scaffold (CLI + seed bookkeeping)
    # -----------------------------------------------------------------
    # ``world_rng`` is the single source of randomness for the WorldList
    # (impact_seed, spawn_seed) pairs introduced by --rollouts_per_delta.
    # It is seeded independently from numpy's global RNG so future M>1
    # work can reproduce world sequences without disturbing the rest of
    # training. STEP 1 only generates and logs WorldList entries; the
    # actual M>1 averaging in train_step is not yet active and only the
    # first world (index 0) drives the legacy single-rollout path below.
    _world_rng = np.random.default_rng()

    def _parse_M_schedule(spec):
        """Parse '1@0,3@10,5@50' -> sorted [(gen, M), ...].
        Returns None when spec is None/empty."""
        if not spec:
            return None
        out = []
        for chunk in str(spec).split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '@' not in chunk:
                raise ValueError(
                    f"--rollouts_per_delta_schedule: invalid entry {chunk!r}, "
                    f"expected 'M@gen' (e.g. '3@10').")
            m_str, gen_str = chunk.split('@', 1)
            out.append((int(gen_str), int(m_str)))
        out.sort(key=lambda x: x[0])
        if not out or out[0][0] != 0:
            raise ValueError(
                "--rollouts_per_delta_schedule must include an entry at gen 0 "
                "(e.g. '1@0,3@10').")
        return out

    _M_schedule = _parse_M_schedule(args.rollouts_per_delta_schedule)

    def _resolve_M(gen_idx):
        """Return the active M for this generation, honouring the schedule
        when given, otherwise the static --rollouts_per_delta value."""
        if _M_schedule is None:
            return max(1, int(args.rollouts_per_delta))
        M = _M_schedule[0][1]
        for g_thresh, m_val in _M_schedule:
            if gen_idx >= g_thresh:
                M = m_val
        return max(1, int(M))

    def _sample_world_list(M):
        """Draw M independent (impact_seed, spawn_seed) pairs from the
        dedicated world_rng. Each seed is a positive int32."""
        return [
            (int(_world_rng.integers(1, 2**31 - 1)),
             int(_world_rng.integers(1, 2**31 - 1)))
            for _ in range(M)
        ]

    # Cache of the current iteration's WorldList. Updated by
    # ``_refresh_world_list_if_needed`` according to --worlds_refresh.
    # STEP 1: read-only consumer is ``_install_generation_maps`` which
    # picks world[0] for the legacy single-world snapshot, preserving
    # exact byte-for-byte behaviour when M=1.
    _current_world_list = []  # list[tuple[int, int]]
    _last_world_gen = [-1]    # mutable holder so closure can write to it

    def _refresh_world_list_if_needed(gen_idx, iter_idx):
        """Refresh ``_current_world_list`` according to the configured
        policy. iteration -> always refresh; generation -> only on the
        first iter of a new generation. Returns True iff the list was
        (re)sampled in this call (Step 2 uses this to decide whether to
        re-install the env builder / rebuild the worker pool)."""
        nonlocal _current_world_list
        M = _resolve_M(gen_idx)
        need_refresh = (
            not _current_world_list
            or args.worlds_refresh == "iteration"
            or _last_world_gen[0] != gen_idx
        )
        if need_refresh:
            _current_world_list = _sample_world_list(M)
            _last_world_gen[0] = gen_idx
            # STEP 3: publish the WorldList to the trainer so train_step /
            # train_step_coevolution can average per delta over M worlds.
            # Empty / len==1 -> trainer keeps the legacy single-world path
            # (uses self.env_builder verbatim, no with_world() call).
            try:
                trainer.world_list = list(_current_world_list)
            except NameError:
                # trainer not yet constructed (first refresh happens before
                # the training loop builds it). The very first refresh in
                # main() is invoked AFTER trainer is set up, so this branch
                # is defensive-only.
                pass
            if M > 1 or args.rollouts_per_delta_schedule is not None:
                pairs = ", ".join(
                    f"(imp={imp},sp={sp})" for imp, sp in _current_world_list)
                print(f"    Worlds (M={M}, refresh={args.worlds_refresh}): {pairs}")
        return need_refresh

    def _install_generation_worlds(gen_idx):
        """Install ``trainer.env_builder`` from world index 0 of the
        current WorldList. Rebuilds the worker pool (if any) so all
        workers see the new builder.

        STEP 2: impact/spawn seeds now come from ``_current_world_list[0]``
        (sampled by ``_refresh_world_list_if_needed`` from the dedicated
        ``_world_rng``), instead of numpy's global RNG. The legacy
        single-rollout path still consumes only world[0]; the M>1
        averaging loop in train_step will be wired in Step 3.
        """
        assert _current_world_list, (
            "_install_generation_worlds called before "
            "_refresh_world_list_if_needed populated the WorldList.")
        impact_seed, spawn_seed = _current_world_list[0]
        maps = _sample_impact_maps(_impact_vars_global, _impact_ranges_global,
                                   (GRID_HEIGHT, GRID_WIDTH), seed=impact_seed)
        new_builder = _make_env_builder(
            maps,
            grid_size=(GRID_HEIGHT, GRID_WIDTH),
            project_path=PROJECT_PATH,
            spawn_seed=spawn_seed,
            impact_vars=_impact_vars_global,
            impact_ranges=_impact_ranges_global,
            impact_seed=impact_seed,
        )
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
                initargs=(new_builder, trainer.policy_params,
                          trainer.uniform_bias_init),
            )
        if maps:
            summary = ", ".join(
                f"{k}=U[{_impact_ranges_global.get(k,(0,0))[0]:g},"
                f"{_impact_ranges_global.get(k,(0,0))[1]:g}]"
                for k in maps)
            print(f"    Impact maps (shared this gen, seed={impact_seed}): {summary}")

    # Backwards-compatible alias: older code/log searches expect this name.
    _install_generation_maps = _install_generation_worlds

    import itertools
    import time as _time
    gen_iter = itertools.count() if generations_is_inf else range(generations_value)
    gen_label_total = "inf" if generations_is_inf else str(generations_value)
    _training_start_ts = _time.monotonic()

    def _fmt_elapsed(seconds: float) -> str:
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    try:
        for gen in gen_iter:
            # Linear softmax-temperature annealing from temp_start -> temp_end
            # over the first temp_anneal_gens generations.
            anneal_n = max(1, int(args.temp_anneal_gens))
            frac = min(1.0, gen / max(1, anneal_n - 1)) if anneal_n > 1 else 1.0
            T = float(args.temp_start + (args.temp_end - args.temp_start) * frac)
            trainer.softmax_temperature = T
            _elapsed = _fmt_elapsed(_time.monotonic() - _training_start_ts)
            print(f"\n========== Generation {gen+1}/{gen_label_total} (T={T:.3f}) ({_elapsed}) ==========")
            # Step 2: WorldList must be populated before world[0] can be
            # installed. ``_refresh_world_list_if_needed(gen, 0)`` returns
            # True on the first iter of each generation (both policies),
            # so we always install at least once per generation here. The
            # per-iter loops below may refresh again (iteration policy)
            # and re-install via ``_maybe_reinstall_worlds``.
            _refresh_world_list_if_needed(gen, 0)
            _install_generation_worlds(gen)
            if args.coevolution:
                # Co-evolution: all species are trained simultaneously per iteration
                # in a SHARED rollout. No inner round-robin loop.
                print(f"\n>>> Co-evolving: {', '.join(s.upper() for s in target_species)} "
                      f"(gen {gen+1}/{gen_label_total})")
                for species in target_species:
                    print(f"    {species}: in={policy_params[species][0]} "
                          f"out={policy_params[species][1]}")
                for i in range(args.iter_per_gen):
                    if i > 0 and _refresh_world_list_if_needed(gen, i):
                        _install_generation_worlds(gen)
                    means = trainer.train_step_coevolution(
                        target_species, n_eval_ticks=args.n_eval_ticks)
                    summary = " | ".join(
                        f"{fid}={means[fid]:+.4f}" for fid in target_species)
                    print(f"    Iter {i+1:2d}/{args.iter_per_gen} | {summary}")
                # Save checkpoints for all co-trained species.
                for species in target_species:
                    save_path = os.path.join(run_dir, f"policy_{species}.pth")
                    _save_checkpoint(trainer, species, save_path)
                    print(f"    Checkpoint saved to: {save_path}")
            else:
                for species in target_species:
                    print(f"\n>>> Training: {species.upper()} (gen {gen+1}/{gen_label_total})")
                    print(f"    Input dim:  {policy_params[species][0]}")
                    print(f"    Output dim: {policy_params[species][1]}")

                    for i in range(args.iter_per_gen):
                        if i > 0 and _refresh_world_list_if_needed(gen, i):
                            _install_generation_worlds(gen)
                        # n_eval_ticks: how many time steps (ticks) each test run lasts
                        avg_reward = trainer.train_step(species, n_eval_ticks=args.n_eval_ticks)
                        print(f"    Iter {i+1:2d}/{args.iter_per_gen} | Avg Reward: {avg_reward:10.6f}")

                    # Save checkpoint after each generation so progress is preserved.
                    save_path = os.path.join(run_dir, f"policy_{species}.pth")
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
            save_path = os.path.join(run_dir, f"policy_{species}.pth")
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
