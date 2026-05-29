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


def _worker_init(env_builder, policy_params, uniform_bias_init=False):
    global _ENV_BUILDER, _POLICIES, _UNIFORM_BIAS_INIT
    _UNIFORM_BIAS_INIT = bool(uniform_bias_init)
    # Ignore SIGINT in workers so Ctrl+C is handled solely by the parent.
    # Without this, every worker raises KeyboardInterrupt and spams tracebacks.
    import signal
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
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
        _POLICIES[fg_id] = PolicyNetwork(in_dim, out_dim,
                                         uniform_bias_init=_UNIFORM_BIAS_INIT)


def _evaluate_coevo_task(task):
    """Co-evolution task: evaluate ONE shared rollout and return fitness for
    EVERY species in `fg_list`. All policies are perturbed simultaneously,
    so each species sees the others' perturbed behaviour. This creates the
    feedback loop (predator "ät alltid" -> prey collapse -> predator reward
    drops within the same rollout) that the single-species ARS round-robin
    can never see.

    task = (fg_list, weights_dict, n_ticks, alpha, beta, seed, obs_pack,
            entropy_coef, argmax_penalty, softmax_temperature,
            integral_reward)

    STEP 3 (multi-world averaging): task may also be a dict with the same
    fields plus an optional ``env_builder`` override (one of the M world
    builders for this iteration). When the override is present it is
    used in place of ``_ENV_BUILDER``; otherwise the worker-local default
    is used so legacy single-world callers keep working unchanged.

    Returns:
        (fitness_dict: {fg_id -> float},
         samples: None | (sum (N_dm,D) f64, sumsq (N_dm,D) f64, count int),
         act_diag_dict: {fg_id -> act_diag} | None)
    """
    builder_override = None
    if isinstance(task, dict):
        fg_list = task['fg_list']
        weights_dict = task['weights_dict']
        n_ticks = task['n_ticks']
        alpha = task['alpha']
        beta = task['beta']
        seed = task.get('seed')
        obs_pack = task.get('obs_pack')
        entropy_coef = task.get('entropy_coef', 0.0)
        argmax_penalty = task.get('argmax_penalty', 0.0)
        softmax_temperature = task.get('softmax_temperature', 1.0)
        integral_reward = task.get('integral_reward', False)
        builder_override = task.get('env_builder')
    else:
        (fg_list, weights_dict, n_ticks, alpha, beta, seed, obs_pack,
         entropy_coef, argmax_penalty, softmax_temperature, integral_reward) = task
    _ = (entropy_coef, argmax_penalty)

    # Sync ALL policy weights
    for fg_id, w in weights_dict.items():
        if fg_id in _POLICIES:
            _set_weights_flat(_POLICIES[fg_id], w)

    builder = builder_override if builder_override is not None else _ENV_BUILDER
    env = builder(seed=seed) if seed is not None else builder()
    env.policies = _POLICIES
    env.softmax_temperature = float(softmax_temperature)

    if obs_pack is not None:
        env._build_static_caches()
        dm_ids = obs_pack['dm_ids']
        mean = obs_pack['mean']
        var = obs_pack['var']
        if dm_ids == env.dm_ids:
            env.obs_mean = mean
            env.obs_var = var
        else:
            idx = [dm_ids.index(fid) for fid in env.dm_ids]
            env.obs_mean = mean[idx]
            env.obs_var = var[idx]

    # Initial state per evaluated species.
    b0 = {fid: float(env.fgs[fid].biomass.sum()) for fid in fg_list}
    r0 = {fid: float(env.fgs[fid].energy_reserve.sum()) for fid in fg_list}

    if integral_reward and n_ticks > 0:
        b_sum = {fid: 0.0 for fid in fg_list}
        r_sum = {fid: 0.0 for fid in fg_list}
        for _ in range(n_ticks):
            env.step()
            for fid in fg_list:
                b_sum[fid] += float(env.fgs[fid].biomass.sum())
                r_sum[fid] += float(env.fgs[fid].energy_reserve.sum())
        bh = {fid: b_sum[fid] / n_ticks for fid in fg_list}
        rh = {fid: r_sum[fid] / n_ticks for fid in fg_list}
    else:
        for _ in range(n_ticks):
            env.step()
        bh = {fid: float(env.fgs[fid].biomass.sum()) for fid in fg_list}
        rh = {fid: float(env.fgs[fid].energy_reserve.sum()) for fid in fg_list}

    fitness = {}
    for fid in fg_list:
        eps_b = max(1e-6 * b0[fid], 1e-9)
        eps_r = max(1e-6 * r0[fid], 1e-9)
        delta_b = np.log((bh[fid] + eps_b) / (b0[fid] + eps_b))
        delta_r = np.log((rh[fid] + eps_r) / (r0[fid] + eps_r))
        fitness[fid] = float(alpha * delta_b + beta * delta_r)

    samples = None
    if obs_pack is not None and getattr(env, '_obs_sum', None) is not None:
        samples = (env._obs_sum.copy(), env._obs_sumsq.copy(), env._obs_count)

    act_diag_dict = None
    if getattr(env, '_action_entropy_sum', None) is not None and env._action_entropy_count > 0:
        act_diag_dict = {}
        cnt = env._action_entropy_count
        for fid in fg_list:
            try:
                i = env.dm_ids.index(fid)
                act_diag_dict[fid] = {
                    'entropy': float(env._action_entropy_sum[i] / cnt),
                    'max_entropy': float(env._action_max_entropy),
                    'move_frac': float(env._action_move_frac[i] / cnt),
                    'rest_frac': float(env._action_rest_frac[i] / cnt),
                    'eat_frac': float(env._action_eat_frac[i] / cnt),
                }
            except (ValueError, AttributeError):
                pass
        if not act_diag_dict:
            act_diag_dict = None

    return (fitness, samples, act_diag_dict)


def _evaluate_task(task):
    """task = (fg_to_train, weights_dict, n_ticks, alpha, beta, seed, obs_pack)

    obs_pack: optional dict {'dm_ids': [...], 'mean': (N_dm,D) fp32,
              'var': (N_dm,D) fp32} for ARS-V2 obs normalisation.
              When provided, env.obs_mean/obs_var are installed and the env
              accumulates raw-obs stats which are returned with the fitness.

    Returns:
        (fitness: float,
         samples: None | (sum (N_dm,D) f64, sumsq (N_dm,D) f64, count int))
    """
    seed = None
    obs_pack = None
    entropy_coef = 0.0
    argmax_penalty = 0.0
    softmax_temperature = 1.0
    integral_reward = False
    builder_override = None
    if isinstance(task, dict):
        # STEP 3 dict-format task: same fields as tuple-format plus an
        # optional ``env_builder`` override carrying a per-world builder
        # for the multi-world averaging path. Falls back to _ENV_BUILDER
        # when not provided so single-world callers stay unchanged.
        fg_to_train = task['fg_to_train']
        weights_dict = task['weights_dict']
        n_ticks = task['n_ticks']
        alpha = task.get('alpha', 1.0)
        beta = task.get('beta', 1.0)
        seed = task.get('seed')
        obs_pack = task.get('obs_pack')
        entropy_coef = task.get('entropy_coef', 0.0)
        argmax_penalty = task.get('argmax_penalty', 0.0)
        softmax_temperature = task.get('softmax_temperature', 1.0)
        integral_reward = task.get('integral_reward', False)
        builder_override = task.get('env_builder')
    elif len(task) == 11:
        (fg_to_train, weights_dict, n_ticks, alpha, beta, seed, obs_pack,
         entropy_coef, argmax_penalty, softmax_temperature, integral_reward) = task
    elif len(task) == 10:
        (fg_to_train, weights_dict, n_ticks, alpha, beta, seed, obs_pack,
         entropy_coef, argmax_penalty, softmax_temperature) = task
    elif len(task) == 9:
        (fg_to_train, weights_dict, n_ticks, alpha, beta, seed, obs_pack,
         entropy_coef, argmax_penalty) = task
    elif len(task) == 8:
        fg_to_train, weights_dict, n_ticks, alpha, beta, seed, obs_pack, entropy_coef = task
    elif len(task) == 7:
        fg_to_train, weights_dict, n_ticks, alpha, beta, seed, obs_pack = task
    elif len(task) == 6:
        fg_to_train, weights_dict, n_ticks, alpha, beta, seed = task
    elif len(task) == 5:
        fg_to_train, weights_dict, n_ticks, alpha, beta = task
    else:
        fg_to_train, weights_dict, n_ticks = task
        alpha, beta = 1.0, 1.0

    # Sync policy weights
    for fg_id, w in weights_dict.items():
        if fg_id in _POLICIES:
            _set_weights_flat(_POLICIES[fg_id], w)

    builder = builder_override if builder_override is not None else _ENV_BUILDER
    env = builder(seed=seed) if seed is not None else builder()
    env.policies = _POLICIES
    env.softmax_temperature = float(softmax_temperature)

    # Install obs-normalisation stats if provided.
    if obs_pack is not None:
        env._build_static_caches()
        dm_ids = obs_pack['dm_ids']
        mean = obs_pack['mean']
        var = obs_pack['var']
        if dm_ids == env.dm_ids:
            env.obs_mean = mean
            env.obs_var = var
        else:
            idx = [dm_ids.index(fid) for fid in env.dm_ids]
            env.obs_mean = mean[idx]
            env.obs_var = var[idx]

    b0 = env.fgs[fg_to_train].biomass.sum()
    r0 = env.fgs[fg_to_train].energy_reserve.sum()

    # Integral-reward: medel över alla ticks istället för slutvärde.
    if integral_reward and n_ticks > 0:
        b_sum = 0.0; r_sum = 0.0
        for _ in range(n_ticks):
            env.step()
            b_sum += float(env.fgs[fg_to_train].biomass.sum())
            r_sum += float(env.fgs[fg_to_train].energy_reserve.sum())
        bh = b_sum / n_ticks
        rh = r_sum / n_ticks
    else:
        for _ in range(n_ticks):
            env.step()
        bh = env.fgs[fg_to_train].biomass.sum()
        rh = env.fgs[fg_to_train].energy_reserve.sum()

    eps_b = max(1e-6 * b0, 1e-9)
    eps_r = max(1e-6 * r0, 1e-9)
    delta_b = np.log((bh + eps_b) / (b0 + eps_b))
    delta_r = np.log((rh + eps_r) / (r0 + eps_r))
    # Return *raw* ecological fitness only. Entropy bonus and argmax-penalty
    # are applied in the trainer *after* z-score normalisation of the
    # ecological component across the 2*n_deltas batch, so that the
    # bonus/penalty (which live on the [0,1] scale) have comparable weight
    # for all species regardless of the absolute |delta_b+delta_r| magnitude.
    fitness = float(alpha * delta_b + beta * delta_r)
    # entropy_coef / argmax_penalty arguments are accepted for backward
    # compatibility with task-tuple length 8/9 but are intentionally unused
    # here; the trainer applies them post hoc via act_diag.
    _ = (entropy_coef, argmax_penalty)

    samples = None
    if obs_pack is not None and getattr(env, '_obs_sum', None) is not None:
        samples = (env._obs_sum.copy(), env._obs_sumsq.copy(), env._obs_count)
    act_diag = None
    if getattr(env, '_action_entropy_sum', None) is not None and env._action_entropy_count > 0:
        try:
            i = env.dm_ids.index(fg_to_train)
            cnt = env._action_entropy_count
            act_diag = {
                'entropy': float(env._action_entropy_sum[i] / cnt),
                'max_entropy': float(env._action_max_entropy),
                'move_frac': float(env._action_move_frac[i] / cnt),
                'rest_frac': float(env._action_rest_frac[i] / cnt),
                'eat_frac': float(env._action_eat_frac[i] / cnt),
            }
        except (ValueError, AttributeError):
            act_diag = None
    return (fitness, samples, act_diag)
