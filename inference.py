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


class RandomPolicy:
    """Drop-in replacement for :class:`PolicyNetwork` that returns uniform
    pre-softmax logits (zeros) for every cell. After the env applies its
    action mask + softmax this yields a uniform distribution over the
    currently-legal actions — i.e. random actions sampled fresh every tick.

    Only the methods used by :meth:`EcosystemEnvironment.step` are
    implemented (no ``parameters()``, no ``state_dict()``); the batched
    forward path is intentionally bypassed by setting
    ``env._batched_ready = False`` on the rnd env, forcing the per-DM
    Python loop that calls ``get_action_logits_torch``.
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def get_action_logits_torch(self, flat_state):
        n = flat_state.shape[0]
        return torch.zeros((n, self.out_dim), dtype=torch.float32)

    def get_action_probs_torch(self, flat_state):
        n = flat_state.shape[0]
        return torch.full((n, self.out_dim), 1.0 / self.out_dim,
                          dtype=torch.float32)

    def eval(self):
        return self


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
              apply_natural_mortality=False, allowed_mask=None):
    """Construct a fresh EcosystemEnvironment for inference."""
    H, W = grid_size
    accessibility = None
    if allowed_mask is not None:
        accessibility = np.asarray(allowed_mask).reshape(H, W).astype(np.float32)
    impact_ranges = {}
    if project_path:
        fgs, impact_vars, impact_ranges, observable_impact_vars = load_project_config(
            project_path, grid_size=grid_size, seed=seed, mode='inference',
            allowed_mask=accessibility)
    else:
        fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed,
                                    allowed_mask=accessibility)
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
    if accessibility is not None:
        env.grid.add_map('accessibility', accessibility)
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


class PolicyCheckpointMismatchError(Exception):
    """Raised when the checkpoints on disk do not match the project's DM set
    or have an incompatible architecture / input dimension."""


def _infer_arch_from_state_dict(sd):
    """Inspect a PolicyNetwork state_dict and return (in_dim, out_dim,
    hidden_dim, hidden_layers). The PolicyNetwork stores its layers inside
    an ``nn.Sequential`` named ``net`` with alternating Linear+Sigmoid and
    a final Linear; weight keys look like ``net.0.weight``, ``net.2.weight``,
    ..., ``net.<2k>.weight``. Returns None on unrecognised layouts.
    """
    try:
        linear_keys = sorted(
            (int(k.split('.')[1]), k) for k in sd.keys()
            if k.startswith('net.') and k.endswith('.weight')
        )
        if not linear_keys:
            return None
        shapes = [tuple(sd[k].shape) for _, k in linear_keys]
        in_dim = shapes[0][1]
        out_dim = shapes[-1][0]
        hidden_dim = shapes[0][0] if len(shapes) > 1 else None
        hidden_layers = len(shapes) - 1
        return in_dim, out_dim, hidden_dim, hidden_layers
    except Exception:
        return None


def load_policies_and_stats(env, checkpoint_dir, verbose=True):
    """Load all DM policy checkpoints from ``checkpoint_dir``.

    Returns (policies_dict, obs_mean, obs_var) where obs_mean/obs_var are
    aligned with env.dm_ids ordering. Falls back to (mean=0, var=1) for
    DMs without usable stats.

    Raises :class:`PolicyCheckpointMismatchError` if the set of
    ``policy_<fg>.pth`` files in ``checkpoint_dir`` does not match the
    project's decision makers, or if any checkpoint has an incompatible
    architecture (e.g. trained on a different grid / FG set / hidden size).
    """
    env._build_static_caches()
    dm_ids = list(env.dm_ids)
    N_all = env.N_all
    N_dm = env.N_dm
    # With the Observability matrix each DM has its own compact ``in_dim``
    # (see Section 4 / Section 41 in mareld_resume.txt). ``env.max_in_dim``
    # is the padded width used for the batched bmm tensor and is what
    # ``obs_stats`` arrays must match. Per-DM checkpoints are saved with
    # their own compact ``per_dm_in_dim[i]``, which we cross-check below.
    D = int(env.max_in_dim)
    per_dm_in_dim = list(env.per_dm_in_dim)
    out_dim = 5 + N_all

    # ---- Cross-check checkpoint files against project's DM set ----------
    # Find every ``policy_<id>.pth`` file currently in the checkpoint dir.
    found_ids = set()
    try:
        for name in os.listdir(checkpoint_dir):
            m = re.fullmatch(r"policy_(.+)\.pth", name)
            if m:
                found_ids.add(m.group(1))
    except OSError as e:
        raise PolicyCheckpointMismatchError(
            f"Could not list checkpoint directory '{checkpoint_dir}': {e}"
        )
    expected_ids = set(dm_ids)
    missing = sorted(expected_ids - found_ids)
    extra = sorted(found_ids - expected_ids)
    if missing or extra:
        lines = [
            f"Policy checkpoints in '{checkpoint_dir}' do not match the "
            f"project's decision makers."
        ]
        lines.append(f"  Expected ({len(expected_ids)}): "
                     f"{sorted(expected_ids)}")
        lines.append(f"  Found    ({len(found_ids)}): "
                     f"{sorted(found_ids)}")
        if missing:
            lines.append(f"  Missing checkpoint(s): {missing}")
        if extra:
            lines.append(f"  Unexpected checkpoint(s): {extra}")
        lines.append("  Hint: make sure --run-name / --checkpoints points at "
                     "a run trained with the same project file, or retrain.")
        raise PolicyCheckpointMismatchError("\n".join(lines))

    mean = np.zeros((N_dm, D), dtype=np.float32)
    var = np.ones((N_dm, D), dtype=np.float32)

    policies = {}
    for i, fid in enumerate(dm_ids):
        ckpt_path = os.path.join(checkpoint_dir, f"policy_{fid}.pth")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Support both formats: bare state_dict (legacy) and new dict payload.
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            os_stats = ckpt.get('obs_stats')
        else:
            sd = ckpt
            os_stats = None

        # Cross-check architecture against the env that we're about to run
        # inference on. Mismatch usually means the checkpoint was trained
        # with a different project (different FG count -> different in_dim
        # / out_dim) or with a different --policynetwork.
        arch = _infer_arch_from_state_dict(sd)
        if arch is None:
            raise PolicyCheckpointMismatchError(
                f"Checkpoint '{ckpt_path}' has an unrecognised layout "
                f"(state_dict keys: {list(sd.keys())[:6]}...)."
            )
        ck_in, ck_out, ck_hidden, ck_layers = arch
        # Per-DM expected in_dim: the Observability matrix gives each DM
        # its own compact input layout (Section 4 / Section 41). The
        # checkpoint was saved with that compact dim, not with the
        # padded ``max_in_dim``.
        expected_in = int(per_dm_in_dim[i])
        if ck_in != expected_in or ck_out != out_dim:
            obs_list = env.fgs[fid].params.get('observes')
            if obs_list is None:
                obs_desc = ("legacy 'see all other FGs' (no _observes_ "
                            "entries for this DM)")
            else:
                others = [o for o in obs_list if o != fid and o in env.fgs]
                obs_desc = (f"{len(others)} other FG(s): {sorted(others)}")
            raise PolicyCheckpointMismatchError(
                f"Checkpoint '{ckpt_path}' has incompatible input/output "
                f"dimensions: file has (in={ck_in}, out={ck_out}) but the "
                f"current project expects (in={expected_in}, out={out_dim}).\n"
                f"  Current Observability config for '{fid}': observes "
                f"{obs_desc}.\n"
                f"  Hint: this checkpoint was trained against a different "
                f"Observability matrix, FG set, or observable impacts. "
                f"Adjust the Observability matrix in fgconfig so '{fid}' "
                f"yields in_dim={ck_in}, or retrain."
            )

        net = PolicyNetwork(expected_in, out_dim,
                            hidden_dim=ck_hidden if ck_hidden else 30,
                            hidden_layers=max(1, ck_layers))
        try:
            net.load_state_dict(sd)
        except Exception as e:
            raise PolicyCheckpointMismatchError(
                f"Could not load weights from '{ckpt_path}' into a "
                f"PolicyNetwork(in={expected_in}, out={out_dim}, "
                f"hidden_dim={ck_hidden}, hidden_layers={ck_layers}): {e}"
            )
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
                  viz=None, rnd_env=None):
    """Install policies + frozen stats and step the environment ``n_ticks`` times.

    If ``viz`` is a :class:`lib.viz.LiveVisualizer`, biomass heatmaps and a
    rolling per-FG total-biomass plot are updated every tick. The visualiser
    is allowed to abort the run early by returning False from ``pump_events``;
    in that case the partial history collected so far is returned.
    """
    env.policies = dict(policies)
    env.obs_mean = obs_mean
    env.obs_var = obs_var
    env.collect_obs_stats = False
    env.collect_action_stats = False
    # Rebuild batched weight tensors used by the fast inference path.
    env._rebuild_batched_weights()

    # Install uniform-random policies on the parallel baseline env, if any.
    if rnd_env is not None:
        rnd_env._build_static_caches()
        out_dim_rnd = 5 + rnd_env.N_all
        in_dim_rnd = env.obs_mean.shape[1] if env.obs_mean.ndim == 2 else 0
        rnd_env.policies = {
            fid: RandomPolicy(in_dim_rnd, out_dim_rnd) for fid in rnd_env.dm_ids
        }
        # Frozen mean=0 / var=1 (we want true uniform actions, no obs
        # normalisation pulling the masked logits anywhere). Logits are
        # constant zero anyway so this is pure bookkeeping.
        rnd_env.obs_mean = np.zeros_like(env.obs_mean)
        rnd_env.obs_var = np.ones_like(env.obs_var)
        rnd_env.collect_obs_stats = False
        rnd_env.collect_action_stats = False
        # Force the per-DM Python path so RandomPolicy.get_action_logits_torch
        # is actually called (the batched path needs stacked Linear weights).
        rnd_env._batched_ready = False

    history = {fid: [] for fid in env.fgs}
    energy_history = {fid: [] for fid in env.fgs}
    rnd_history = {fid: [] for fid in (rnd_env.fgs if rnd_env is not None else {})}
    rnd_energy_history = {fid: [] for fid in (rnd_env.fgs if rnd_env is not None else {})}
    for t in range(n_ticks):
        try:
            env.step()
            if rnd_env is not None:
                rnd_env.step()
        except KeyboardInterrupt:
            if verbose:
                print(f"\n    interrupted at tick {t+1}/{n_ticks}; "
                      f"returning partial history.")
            break
        for fid, fg in env.fgs.items():
            history[fid].append(float(fg.biomass.sum()))
            er = getattr(fg, 'energy_reserve', None)
            energy_history[fid].append(float(er.sum()) if er is not None else 0.0)
        if rnd_env is not None:
            for fid, fg in rnd_env.fgs.items():
                rnd_history[fid].append(float(fg.biomass.sum()))
                er = getattr(fg, 'energy_reserve', None)
                rnd_energy_history[fid].append(
                    float(er.sum()) if er is not None else 0.0)
        # Per-FG biomass/energy ratio (current / initial) as a percentage.
        # The first tick is by definition 100 %. ``history[fid][0]`` /
        # ``energy_history[fid][0]`` are the post-tick-0 baselines shown in
        # the live plot's tabs.
        pct_bio = {}
        pct_eng = {}
        for fid in history.keys():
            b0 = history[fid][0] if history[fid] else 0.0
            pct_bio[fid] = (100.0 * history[fid][-1] / b0) if b0 > 0.0 else 0.0
            e0 = energy_history[fid][0] if energy_history[fid] else 0.0
            pct_eng[fid] = (100.0 * energy_history[fid][-1] / e0) if e0 > 0.0 else 0.0
        if verbose and (t % max(1, n_ticks // 10) == 0):
            print(f"    tick {t+1}/{n_ticks}")
        if viz is not None:
            try:
                viz.update_biomass(env.fgs, tick=t)
                for fid in history.keys():
                    viz.update_series("biomass", fid, pct_bio[fid], step=t)
                    viz.update_series("energy", fid, pct_eng[fid], step=t)
                if rnd_env is not None:
                    for fid in rnd_history.keys():
                        b0 = rnd_history[fid][0] if rnd_history[fid] else 0.0
                        e0 = rnd_energy_history[fid][0] if rnd_energy_history[fid] else 0.0
                        pb = (100.0 * rnd_history[fid][-1] / b0) if b0 > 0.0 else 0.0
                        pe = (100.0 * rnd_energy_history[fid][-1] / e0) if e0 > 0.0 else 0.0
                        viz.update_series("biomass", fid + "_rnd", pb, step=t)
                        viz.update_series("energy", fid + "_rnd", pe, step=t)
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
    parser.add_argument("--biomass-ledger", "--biomass_ledger", dest="biomass_ledger",
                        type=str, default=None,
                        help="Optional path to save an opt-in biomass-flow ledger "
                             "as .npz. A same-stem .csv event table is also written.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output.")
    parser.add_argument("--mortality", choices=["on", "off"], default="off",
                        help="Toggle the artificial (density-independent) natural "
                             "mortality term applied to decision-maker FGs each tick. "
                             "Default: off.")
    parser.add_argument("--rnd-baseline", "--rnd_baseline", dest="rnd_baseline",
                        action="store_true",
                        help="Run a parallel rollout where each DM acts uniformly "
                             "at random (mask-respecting) and overlay its biomass%% "
                             "/ energy%% in the live plot as a baseline. Legend "
                             "entries are suffixed with '_rnd'. No heatmaps for "
                             "the random agents.")
    parser.add_argument("--visual", action="store_true",
                        help="Open a live pygame window with per-FG biomass heatmaps "
                             "and a rolling total-biomass plot. Requires pygame; if "
                             "unavailable the flag is silently ignored.")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="Optional torch CPU thread count. Small inference "
                             "networks can be faster with fewer threads, e.g. 4.")

    args = parser.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
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
    try:
        policies, mean, var = load_policies_and_stats(env, args.checkpoints, verbose=verbose)
    except PolicyCheckpointMismatchError as e:
        print("Error: policy checkpoint mismatch.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1

    if args.biomass_ledger:
        from lib.diagnostics.biomass_ledger import BiomassLedger
        env.biomass_ledger = BiomassLedger.from_env(env)
        if verbose:
            print(f"Biomass ledger: {args.biomass_ledger}")

    # Parallel random-action baseline environment. Built with the same
    # project + seed as the trained env so initial biomass / impacts /
    # spawn layout match exactly; the only difference is the policies
    # (uniform random) installed inside run_inference.
    rnd_env = None
    if args.rnd_baseline:
        rnd_env = build_env(args.project, args.grid, seed=args.seed,
                            verbose=False,
                            apply_natural_mortality=(args.mortality == "on"))

    viz = None
    if args.visual:
        try:
            from lib.viz import LiveVisualizer
            extra = ([fid + "_rnd" for fid in env.fgs.keys()]
                     if rnd_env is not None else None)
            ndm_ids = [fid for fid, fg in env.fgs.items()
                       if not getattr(fg, 'is_decision_maker', False)]
            viz = LiveVisualizer(fg_ids=list(env.fgs.keys()),
                                 grid_shape=args.grid,
                                 mode="inference",
                                 extra_plot_ids=extra,
                                 ndm_ids=ndm_ids or None)
        except Exception as e:
            print(f"[viz] failed to start visualiser: {e!r}", file=sys.stderr)
            viz = None

    if verbose:
        print(f"Running {args.ticks} ticks...")
    interrupted = False
    try:
        history = run_inference(env, policies, mean, var, args.ticks,
                                verbose=verbose, viz=viz, rnd_env=rnd_env)
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\nInterrupted by user (Ctrl+C).")
        history = {fid: [0.0] for fid in env.fgs}
    finally:
        if viz is not None and not interrupted:
            # Keep the final frame on screen until the user closes the
            # window (or hits Q/ESC). Ctrl+C in the terminal aborts the
            # wait and proceeds to close immediately.
            try:
                viz.wait_for_close(banner="inference finished — close window to exit (Q/ESC)")
            except KeyboardInterrupt:
                if verbose:
                    print("\nInterrupted by user (Ctrl+C); closing window.")
        if viz is not None:
            viz.close()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        np.savez(args.output, **{fid: np.asarray(h, dtype=np.float64) for fid, h in history.items()})
        if verbose:
            print(f"Saved biomass history to {args.output}")

    if args.biomass_ledger and getattr(env, 'biomass_ledger', None) is not None:
        csv_path = env.biomass_ledger.save(args.biomass_ledger)
        if verbose:
            print(f"Saved biomass ledger to {args.biomass_ledger}")
            print(f"Saved biomass ledger events to {csv_path}")

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
