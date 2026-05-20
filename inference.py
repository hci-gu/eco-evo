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


def build_env(project_path, grid_size, seed=None):
    """Construct a fresh EcosystemEnvironment for inference."""
    H, W = grid_size
    if project_path:
        fgs, impact_vars = load_project_config(project_path, grid_size=grid_size, seed=seed, mode='inference')
    else:
        fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed)
        impact_vars = ['windfarm_noise']

    grid_config = {
        'width': W,
        'height': H,
        'cell_size': 1000.0,
        'tick_duration': 6.0,
    }
    env = EcosystemEnvironment(grid_config, fgs, {})
    for iv in impact_vars:
        env.grid.add_map(iv, np.zeros((H, W)))
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
    D = 2 + (N_all - 1) + 1
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


def run_inference(env, policies, obs_mean, obs_var, n_ticks, verbose=True):
    """Install policies + frozen stats and step the environment ``n_ticks`` times."""
    env.policies = dict(policies)
    env.obs_mean = obs_mean
    env.obs_var = obs_var
    # Rebuild batched weight tensors used by the fast inference path.
    env._rebuild_batched_weights()

    history = {fid: [] for fid in env.fgs}
    for t in range(n_ticks):
        env.step()
        for fid, fg in env.fgs.items():
            history[fid].append(float(fg.biomass.sum()))
        if verbose and (t % max(1, n_ticks // 10) == 0):
            print(f"    tick {t+1}/{n_ticks}")
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Mareld Ecosystem Simulator - Inference Runner (draft)."
    )
    parser.add_argument("--project", type=str, default=None,
                        help="Path to project YAML file. If omitted, uses setup_full_mareld_mvp.")
    parser.add_argument("--checkpoints", type=str, default="results",
                        help="Directory containing policy_<fg>.pth checkpoints (default: results).")
    parser.add_argument("--grid", type=parse_grid_arg, default=(60, 60),
                        help="Grid dimensions as n*m (default: 60*60).")
    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks to run (default: 100).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic initial biomass distribution.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save per-FG biomass history as .npz.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output.")

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("==========================================")
        print("      MARELD INFERENCE SESSION            ")
        print("==========================================")
        print(f"Project:      {args.project or '(built-in MVP)'}")
        print(f"Checkpoints:  {args.checkpoints}")
        print(f"Grid:         {args.grid[0]}x{args.grid[1]}")
        print(f"Ticks:        {args.ticks}")
        print(f"Seed:         {args.seed}")
        print("------------------------------------------")

    if not os.path.isdir(args.checkpoints):
        print(f"Error: checkpoint directory does not exist: {args.checkpoints}", file=sys.stderr)
        return 1

    env = build_env(args.project, args.grid, seed=args.seed)
    if verbose:
        print(f"Loading policies for DMs: {[fid for fid in env.fgs if env.fgs[fid].is_decision_maker]}")
    policies, mean, var = load_policies_and_stats(env, args.checkpoints, verbose=verbose)

    if verbose:
        print(f"Running {args.ticks} ticks...")
    history = run_inference(env, policies, mean, var, args.ticks, verbose=verbose)

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
