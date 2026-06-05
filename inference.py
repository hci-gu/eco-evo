"""Inference runner for trained Mareld policies.

Usage:
    python3 inference.py [flags]

This is a draft / scaffold meant to be extended. It implements the inference
pipeline sketched out previously and takes the known pitfalls into account:

  1. Loads policy checkpoints saved by ``train.py`` (`results/policy_<fg>.pth`),
     which carry both ``state_dict`` and (optionally) ``obs_stats`` per FG.
  2. Installs the frozen running mean/var on ``env.obs_mean`` / ``env.obs_var``
     so observations at inference time are normalised identically to training.
  3. Handles dm_ids ordering mismatch between checkpoint and current env via
     an explicit per-DM lookup (rather than positional indexing).
  4. Falls back to ``mean=0, var=1`` for DMs whose checkpoint has ``count==0``
     or is missing — and warns loudly so the user can tell something is off.
  5. Tolerates both the new checkpoint format ({'state_dict': ..., 'obs_stats': ...})
     and the legacy bare ``state_dict`` format.
  6. Does NOT update obs_stats during inference (stats are frozen).
  7. ``sigma_f`` reward normalisation is irrelevant at inference and is not
     touched — no ARS updates are performed.

Future flags can be added under the argparse block; this draft already
provides hooks for project file, grid size, number of ticks, checkpoint
directory, output path, and seed.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

from lib.config.config_loader import load_project_config, setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.policy import PolicyNetwork


def parse_grid_arg(value):
    m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", value)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid format: '{value}'. Expected n*m (e.g. 60*60)."
        )
    n, k = int(m.group(1)), int(m.group(2))
    if n < 3 or k < 3:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid '{value}': both dimensions must be >= 3."
        )
    return n, k


def _load_inference_map_paths(project_path):
    """Return ``{impact_id: absolute_path}`` from the project's ``inference.impact_maps``.

    Returns an empty dict when the project has no such section. Relative paths
    are resolved against the project YAML file's directory.
    """
    if not project_path:
        return {}
    try:
        import yaml as _yaml
        with open(project_path, 'r') as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return {}
    section = (data.get('inference') or {}).get('impact_maps') or {}
    if not isinstance(section, dict):
        return {}
    base = os.path.dirname(os.path.abspath(project_path))
    resolved = {}
    for iid, p in section.items():
        if not isinstance(p, str) or not p:
            continue
        resolved[iid] = p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
    return resolved


def _load_impact_map_npz(path, impact_id, H, W, verbose=True):
    """Load a single (H, W) impact map from ``path[impact_id]`` with nearest-neighbor resampling.

    Returns a ``float32`` array of shape (H, W), or ``None`` on failure (caller
    should fall back to a zero field).
    """
    if not os.path.isfile(path):
        if verbose:
            print(f"  [warn] Impact map for '{impact_id}' not found at {path}; using zero field.")
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            arr = None
            # Priority 1: exact impact_id match (back-compat with archives
            # whose key happens to be named after the impact).
            if impact_id in keys:
                cand = np.asarray(data[impact_id])
                if cand.ndim == 2:
                    arr = cand
            # Priority 2: the first 2-D array in the archive. This makes
            # impact-map .npz files key-agnostic — any 2-D array works,
            # regardless of what it is called inside the archive. Mirrors
            # the loader used by fgconfig's Inference tab.
            if arr is None:
                for k in keys:
                    cand = np.asarray(data[k])
                    if cand.ndim == 2:
                        arr = cand
                        if verbose and k != impact_id:
                            print(f"  [info] Using array '{k}' from "
                                  f"{os.path.basename(path)} as impact map "
                                  f"for '{impact_id}' (key-agnostic load).")
                        break
            if arr is None:
                if verbose:
                    print(f"  [warn] No 2-D array found in "
                          f"{os.path.basename(path)} (keys: {keys}); "
                          f"using zero field for '{impact_id}'.")
                return None
    except Exception as e:
        if verbose:
            print(f"  [warn] Could not read '{impact_id}' from {path}: {e}; using zero field.")
        return None
    arr = arr.astype(np.float32, copy=False)
    Hs, Ws = arr.shape
    if (Hs, Ws) != (H, W):
        # Nearest-neighbor resample to the requested grid.
        if Hs == 0 or Ws == 0:
            return None
        row_idx = (np.arange(H) * Hs / H).astype(np.int64)
        col_idx = (np.arange(W) * Ws / W).astype(np.int64)
        arr = arr[row_idx[:, None], col_idx[None, :]]
        if verbose:
            print(f"  [info] Resampled '{impact_id}' from {(Hs, Ws)} to {(H, W)} (nearest).")
    return arr.astype(np.float32, copy=False)


def build_env(project_path, grid_size, seed=None, verbose=True,
              apply_natural_mortality=False):
    """Construct a fresh EcosystemEnvironment for inference."""
    H, W = grid_size
    impact_ranges = {}
    if project_path:
        fgs, impact_vars, impact_ranges, observable_impact_vars = load_project_config(
            project_path, grid_size=grid_size, seed=seed, mode='inference')
    else:
        fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed)
        impact_vars = ['windfarm_noise']
        observable_impact_vars = ['windfarm_noise']

    grid_config = {
        'width': W,
        'height': H,
        'cell_size': 1000.0,
        'tick_duration': 6.0,
    }
    env = EcosystemEnvironment(grid_config, fgs, {},
                               observable_impact_vars=observable_impact_vars,
                               apply_natural_mortality=apply_natural_mortality)
    # Impact maps are read from .npz files configured in the project's
    # ``inference.impact_maps`` section (set via the fgconfig Inference tab).
    # Missing entries — or files that fail validation — are treated as
    # all-zero fields. Biomass / energy maps are still randomly spawned
    # from the configured initial biomass values.
    map_paths = _load_inference_map_paths(project_path)
    for iv in impact_vars:
        field = None
        if iv in map_paths:
            field = _load_impact_map_npz(map_paths[iv], iv, H, W, verbose=verbose)
        if field is None:
            field = np.zeros((H, W), dtype=np.float32)
        env.grid.add_map(iv, field)
    return env


def load_policies_and_stats(env, checkpoint_dir, verbose=True):
    """Load all DM policy checkpoints from ``checkpoint_dir``.

    Returns (policies_dict, obs_mean, obs_var) where obs_mean/obs_var are
    aligned with env.dm_ids ordering. Falls back to (mean=0, var=1) for
    DMs without usable stats.
    """
    env._build_static_caches()
    dm_ids = list(env.dm_ids)
    N_all = env.N_all
    N_dm = env.N_dm
    n_obs_imp = len(getattr(env, 'observable_impact_vars', []) or [])
    # Per cell the policy sees the von Neumann neighbourhood (center + N/E/S/W)
    # per Method.pdf. Center: B_own, E_own, B_others(N_all-1), observable
    # impacts. Each neighbour: same minus E_own. Total:
    center_dim = 2 + (N_all - 1) + n_obs_imp
    nbr_dim = 1 + (N_all - 1) + n_obs_imp
    D = center_dim + 4 * nbr_dim
    in_dim = D
    out_dim = 5 + N_all

    mean = np.zeros((N_dm, D), dtype=np.float32)
    var = np.ones((N_dm, D), dtype=np.float32)

    policies = {}
    for i, fid in enumerate(dm_ids):
        ckpt_path = os.path.join(checkpoint_dir, f"policy_{fid}.pth")
        if not os.path.exists(ckpt_path):
            if verbose:
                print(f"  [warn] No checkpoint for '{fid}' at {ckpt_path}; "
                      f"using untrained policy and mean=0/var=1 (likely degenerate).")
            policies[fid] = PolicyNetwork(in_dim, out_dim)
            continue

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Support both formats: bare state_dict (legacy) and new dict payload.
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            os_stats = ckpt.get('obs_stats')
        else:
            sd = ckpt
            os_stats = None

        net = PolicyNetwork(in_dim, out_dim)
        net.load_state_dict(sd)
        net.eval()
        policies[fid] = net

        if os_stats is not None and int(os_stats.get('count', 0)) > 0:
            m = np.asarray(os_stats['mean'], dtype=np.float32)
            v = np.asarray(os_stats['var'], dtype=np.float32)
            if m.shape == (D,) and v.shape == (D,):
                mean[i] = m
                var[i] = v
                if verbose:
                    print(f"  [ok]   Loaded '{fid}' (obs_stats count={int(os_stats['count'])}).")
            else:
                if verbose:
                    print(f"  [warn] '{fid}' obs_stats shape {m.shape} != ({D},); "
                          f"using mean=0/var=1.")
        else:
            if verbose:
                print(f"  [warn] '{fid}' has no usable obs_stats; using mean=0/var=1. "
                      f"Observations will be clipped to [-10, 10] without scaling.")

    return policies, mean, var


def run_inference(env, policies, obs_mean, obs_var, n_ticks, verbose=True,
                  viz=None):
    """Install policies + frozen stats and step the environment ``n_ticks`` times.

    If ``viz`` is a :class:`lib.viz.LiveVisualizer`, biomass heatmaps and a
    rolling per-FG total-biomass plot are updated every tick. The visualiser
    is allowed to abort the run early by returning False from ``pump_events``;
    in that case the partial history collected so far is returned.
    """
    env.policies = dict(policies)
    env.obs_mean = obs_mean
    env.obs_var = obs_var
    # Rebuild batched weight tensors used by the fast inference path.
    env._rebuild_batched_weights()

    history = {fid: [] for fid in env.fgs}
    for t in range(n_ticks):
        try:
            env.step()
        except KeyboardInterrupt:
            if verbose:
                print(f"\n    interrupted at tick {t+1}/{n_ticks}; "
                      f"returning partial history.")
            break
        for fid, fg in env.fgs.items():
            history[fid].append(float(fg.biomass.sum()))
        if verbose and (t % max(1, n_ticks // 10) == 0):
            print(f"    tick {t+1}/{n_ticks}")
        if viz is not None:
            try:
                viz.update_biomass(env.fgs, tick=t)
                for fid, h in history.items():
                    viz.update_reward(fid, h[-1], step=t)
                if not viz.pump_events():
                    if verbose:
                        print("    [viz] window closed; stopping early.")
                    break
            except KeyboardInterrupt:
                if verbose:
                    print(f"\n    interrupted at tick {t+1}/{n_ticks} "
                          f"(during viz update); returning partial history.")
                break
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Mareld Ecosystem Simulator - Inference Runner (draft)."
    )
    parser.add_argument("--project", type=str, default=None,
                        help="Path to project YAML file. If omitted, uses setup_full_mareld_mvp.")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Directory containing policy_<fg>.pth checkpoints. "
                             "If omitted, results/<run-name>/ is used (see --run-name).")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="default",
                        help="Name of the run whose checkpoints to load (results/<run-name>/). "
                             "Default: 'default'. Ignored if --checkpoints is given explicitly.")
    parser.add_argument("--grid", type=parse_grid_arg, default=(60, 60),
                        help="Grid dimensions as n*m (default: 60*60).")
    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks to run (default: 100).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic initial biomass distribution.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save per-FG biomass history as .npz.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output.")
    parser.add_argument("--mortality", choices=["on", "off"], default="off",
                        help="Toggle the artificial (density-independent) natural "
                             "mortality term applied to decision-maker FGs each tick. "
                             "Default: off.")
    parser.add_argument("--visual", action="store_true",
                        help="Open a live pygame window with per-FG biomass heatmaps "
                             "and a rolling total-biomass plot. Requires pygame; if "
                             "unavailable the flag is silently ignored.")

    args = parser.parse_args()
    verbose = not args.quiet
    if args.checkpoints is None:
        args.checkpoints = os.path.join("results", args.run_name)

    if verbose:
        print("==========================================")
        print("      MARELD INFERENCE SESSION            ")
        print("==========================================")
        print(f"Project:      {args.project or '(built-in MVP)'}")
        print(f"Run name:     {args.run_name}")
        print(f"Checkpoints:  {args.checkpoints}")
        print(f"Grid:         {args.grid[0]}x{args.grid[1]}")
        print(f"Ticks:        {args.ticks}")
        print(f"Seed:         {args.seed}")
        print(f"Mortality:    {args.mortality}")
        print("------------------------------------------")

    if not os.path.isdir(args.checkpoints):
        print(f"Error: checkpoint directory does not exist: {args.checkpoints}", file=sys.stderr)
        return 1

    env = build_env(args.project, args.grid, seed=args.seed, verbose=verbose,
                    apply_natural_mortality=(args.mortality == "on"))
    if verbose:
        print(f"Loading policies for DMs: {[fid for fid in env.fgs if env.fgs[fid].is_decision_maker]}")
    policies, mean, var = load_policies_and_stats(env, args.checkpoints, verbose=verbose)

    viz = None
    if args.visual:
        try:
            from lib.viz import LiveVisualizer
            viz = LiveVisualizer(fg_ids=list(env.fgs.keys()),
                                 grid_shape=args.grid,
                                 mode="inference")
        except Exception as e:
            print(f"[viz] failed to start visualiser: {e!r}", file=sys.stderr)
            viz = None

    if verbose:
        print(f"Running {args.ticks} ticks...")
    try:
        history = run_inference(env, policies, mean, var, args.ticks,
                                verbose=verbose, viz=viz)
    finally:
        if viz is not None:
            viz.close()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        np.savez(args.output, **{fid: np.asarray(h, dtype=np.float64) for fid, h in history.items()})
        if verbose:
            print(f"Saved biomass history to {args.output}")

    if verbose:
        print("\nFinal totals (tonnes):")
        for fid, h in history.items():
            print(f"  {fid:30s}  start={h[0]:12.3f}   end={h[-1]:12.3f}")
        print("==========================================")
        print("Inference complete.")
        print("==========================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
