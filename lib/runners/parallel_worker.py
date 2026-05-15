"""Worker module for parallel ARS rollouts.

Each worker process holds:
  - a local set of PolicyNetworks (built from policy_params at init)
  - a local env_builder callable

For every evaluation task, the parent sends a flat weight dict
(fg_id -> 1D numpy array). The worker copies those weights into its local
policies, builds a fresh environment, runs n_ticks, and returns the fitness.

This avoids re-pickling PolicyNetwork instances on every task and keeps
the per-task payload to plain numpy arrays.
"""
import numpy as np
import torch

from lib.runners.policy import PolicyNetwork

# Module-level globals, populated by _worker_init in each worker process.
_ENV_BUILDER = None
_POLICIES = None  # dict[str, PolicyNetwork]


def _set_weights_flat(policy, flat_weights):
    idx = 0
    flat = torch.from_numpy(flat_weights)
    for p in policy.parameters():
        n = p.data.numel()
        p.data.copy_(flat[idx:idx + n].view(p.size()))
        idx += n


def _worker_init(env_builder, policy_params):
    global _ENV_BUILDER, _POLICIES
    # Single-threaded BLAS/torch in workers; parallelism comes from the pool.
    # Must be set before importing numpy heavy ops; torch we throttle explicitly.
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _ENV_BUILDER = env_builder
    _POLICIES = {}
    seed = (os.getpid() * 2654435761) & 0xFFFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed & 0x7FFFFFFF)
    for fg_id, (in_dim, out_dim) in policy_params.items():
        _POLICIES[fg_id] = PolicyNetwork(in_dim, out_dim)


def _evaluate_task(task):
    """task = (fg_to_train, weights_dict, n_ticks)
    weights_dict: dict[fg_id, np.ndarray] of flat weights for every policy.
    Returns: float fitness.
    """
    fg_to_train, weights_dict, n_ticks = task
    # Sync policy weights
    for fg_id, w in weights_dict.items():
        if fg_id in _POLICIES:
            _set_weights_flat(_POLICIES[fg_id], w)

    env = _ENV_BUILDER()
    env.policies = _POLICIES

    b0 = env.fgs[fg_to_train].biomass.sum()
    r0 = env.fgs[fg_to_train].energy_reserve.sum()

    for _ in range(n_ticks):
        env.step()

    bh = env.fgs[fg_to_train].biomass.sum()
    rh = env.fgs[fg_to_train].energy_reserve.sum()

    eps_b = max(1e-6 * b0, 1e-9)
    eps_r = max(1e-6 * r0, 1e-9)
    delta_b = np.log((bh + eps_b) / (b0 + eps_b))
    delta_r = np.log((rh + eps_r) / (r0 + eps_r))
    return float(delta_b + delta_r)
