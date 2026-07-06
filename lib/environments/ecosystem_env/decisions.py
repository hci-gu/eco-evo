import numpy as np
from lib.environments.ecosystem_env import observations
from lib.environments.ecosystem_env.constants import EAT_START, MOVE_SLICE, REST_INDEX
from lib.environments.ecosystem_env.state import ActionProbabilities, ObservationBatch


def accumulate_observation_stats(env, obs_np):
    dim = obs_np.shape[-1]
    flat = obs_np.reshape(env.N_dm, -1, dim)
    nsamp = flat.shape[1]
    sample_sum = flat.sum(axis=1, dtype=np.float64)
    sample_sumsq = (flat.astype(np.float64) ** 2).sum(axis=1)

    if not hasattr(env, "_obs_sum") or env._obs_sum is None:
        env._obs_sum = np.zeros((env.N_dm, dim), dtype=np.float64)
        env._obs_sumsq = np.zeros((env.N_dm, dim), dtype=np.float64)
        env._obs_count = 0

    env._obs_sum += sample_sum
    env._obs_sumsq += sample_sumsq
    env._obs_count += nsamp


def normalize_observations(env, obs_np):
    dim = obs_np.shape[-1]
    if not (
        getattr(env, "obs_mean", None) is not None
        and getattr(env, "obs_var", None) is not None
        and env.obs_mean.shape == (env.N_dm, dim)
    ):
        return obs_np

    mean = env.obs_mean.astype(env.dtype, copy=False)
    var = env.obs_var.astype(env.dtype, copy=False)
    std = np.sqrt(np.maximum(var, np.float32(1e-2)))
    obs_np = (obs_np - mean[:, None, :]) / std[:, None, :]
    return np.clip(obs_np, -10.0, 10.0).astype(env.dtype, copy=False)


def build_action_mask(env, num_actions):
    H, W = env.H, env.W
    full_mask = np.ones((env.N_dm, num_actions, H, W), dtype=env.dtype)
    full_mask[:, MOVE_SLICE] = env.move_mask

    biomass_all = np.stack(
        [env.fgs[fid].biomass for fid in env.global_fg_order],
        axis=0,
    )
    prey_present = (biomass_all > 0).astype(env.dtype)
    full_mask[:, EAT_START:EAT_START + env.N_all] = (
        env.eat_static_mask[:, :, None, None] * prey_present[None, :, :, :]
    )

    cannot_move = env.dm_v <= 0
    if np.any(cannot_move):
        full_mask[cannot_move, MOVE_SLICE] = 0.0

    any_valid = full_mask.sum(axis=1, keepdims=True) > 0
    if not np.all(any_valid):
        full_mask[:, REST_INDEX:REST_INDEX + 1] = np.where(
            any_valid,
            full_mask[:, REST_INDEX:REST_INDEX + 1],
            np.float32(1.0),
        )
    return full_mask


def subthreshold_mask(env):
    if not np.any(env.dm_min_split > 0):
        return None
    biomass_dm = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    return (
        (biomass_dm > 0)
        & (biomass_dm < env.dm_min_split[:, None, None])
    )


def masked_softmax(env, logits, full_mask):
    logits = logits.transpose(0, 2, 1).reshape(
        env.N_dm, logits.shape[-1], env.H, env.W
    ).astype(env.dtype, copy=False)

    add_mask = np.where(
        full_mask > 0,
        np.float32(0.0),
        np.float32(-1e9),
    ).astype(env.dtype, copy=False)
    logits = logits + add_mask

    temperature = float(getattr(env, "softmax_temperature", 1.0) or 1.0)
    if temperature != 1.0:
        logits = logits / np.float32(temperature)

    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def collapse_subthreshold_actions(probs, subthreshold_mask):
    if subthreshold_mask is None or not np.any(subthreshold_mask):
        return probs

    argmax_idx = np.argmax(probs, axis=1)
    one_hot = np.zeros_like(probs)
    dm_idx, h_idx, w_idx = np.where(subthreshold_mask)
    action_idx = argmax_idx[dm_idx, h_idx, w_idx]
    one_hot[dm_idx, action_idx, h_idx, w_idx] = np.float32(1.0)
    return np.where(subthreshold_mask[:, None, :, :], one_hot, probs)


def update_action_diagnostics(env, probs):
    eps = np.float32(1e-12)
    ent_cell = -np.sum(probs * np.log(probs + eps), axis=1)
    biomass_dm = np.stack([env.fgs[fid].biomass for fid in env.dm_ids], axis=0)
    active = (biomass_dm > 0).astype(env.dtype)
    active_sum = active.sum(axis=(1, 2))
    ent_mean = np.where(
        active_sum > 0,
        (ent_cell * active).sum(axis=(1, 2)) / np.maximum(active_sum, 1.0),
        ent_cell.mean(axis=(1, 2)),
    )

    move_mass = probs[:, MOVE_SLICE].sum(axis=1)
    rest_mass = probs[:, REST_INDEX]
    eat_mass = probs[:, EAT_START:EAT_START + env.N_all].sum(axis=1)
    max_entropy = float(np.log(probs.shape[1]))

    if not hasattr(env, "_action_entropy_sum") or env._action_entropy_sum is None:
        env._action_entropy_sum = np.zeros(env.N_dm, dtype=np.float64)
        env._action_entropy_count = 0
        env._action_active_ticks = np.zeros(env.N_dm, dtype=np.int64)
        env._action_move_frac = np.zeros(env.N_dm, dtype=np.float64)
        env._action_rest_frac = np.zeros(env.N_dm, dtype=np.float64)
        env._action_eat_frac = np.zeros(env.N_dm, dtype=np.float64)
        env._action_max_entropy = max_entropy

    env._action_entropy_count += 1
    for i in range(env.N_dm):
        mask_i = active[i] > 0
        n_cells = int(mask_i.sum())
        if n_cells == 0:
            continue
        env._action_active_ticks[i] += 1
        env._action_entropy_sum[i] += float(ent_mean[i])
        inv = 1.0 / float(n_cells)
        env._action_move_frac[i] += float(move_mass[i][mask_i].sum()) * inv
        env._action_rest_frac[i] += float(rest_mass[i][mask_i].sum()) * inv
        env._action_eat_frac[i] += float(eat_mass[i][mask_i].sum()) * inv


def update_hidden_state(env, actions):
    env.prev_hidden_frac = np.zeros(
        (env.N_all, env.H, env.W), dtype=env.dtype)
    for i, _fid in enumerate(env.dm_ids):
        j = int(env.dm_index_in_all[i])
        env.prev_hidden_frac[j] = actions.rest[i].astype(
            env.dtype, copy=False)


def build_observation(env):
    if not env._static_built:
        env.build_static_caches()

    obs_np = observations.build_observation_batch(env)
    raw_obs = obs_np.copy()
    accumulate_observation_stats(env, raw_obs)
    features = normalize_observations(env, obs_np)

    if env._batched_ready and features.shape[-1] == env._in_dim:
        num_actions = env._out_dim
    else:
        num_actions = EAT_START + env.N_all

    return ObservationBatch(
        features=features,
        raw_features=raw_obs,
        action_mask=build_action_mask(env, num_actions),
        subthreshold_mask=subthreshold_mask(env),
        dm_ids=list(env.dm_ids),
        global_fg_order=list(env.global_fg_order),
    )


def actions_from_probabilities(env, probs, subthreshold):
    probs = collapse_subthreshold_actions(probs, subthreshold)
    update_action_diagnostics(env, probs)
    return ActionProbabilities(
        move=probs[:, MOVE_SLICE],
        rest=probs[:, REST_INDEX],
        eat=probs[:, EAT_START:EAT_START + env.N_all],
    )
