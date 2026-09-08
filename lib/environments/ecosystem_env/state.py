from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from lib.environments.ecosystem_env import grid_masks, interactions


@dataclass
class EcosystemCache:
    H: int
    W: int
    dtype: np.dtype
    dm_ids: List[str]
    global_fg_order: List[str]
    dm_index_in_all: np.ndarray
    move_mask: np.ndarray
    edge_immigration_weights: np.ndarray
    eat_static_mask: np.ndarray
    max_intake: np.ndarray
    energy_gain: np.ndarray
    handling_time: np.ndarray
    has_holling2: bool
    type3_pred_mask: np.ndarray
    has_holling3: bool
    dm_v: np.ndarray
    dm_cost_move: np.ndarray
    dm_cost_eat: np.ndarray
    dm_cost_rest: np.ndarray
    dm_resting_metabolism: np.ndarray
    dm_visibility_floor: np.ndarray
    all_visibility_floor: np.ndarray
    dm_max_energy_reserve: np.ndarray
    dm_min_split: np.ndarray
    obs_others_idx: List[np.ndarray]
    per_dm_in_dim: np.ndarray
    max_in_dim: int


@dataclass
class ActionProbabilities:
    move: np.ndarray
    rest: np.ndarray
    eat: np.ndarray


@dataclass
class ActionSettlement:
    stationary_biomass: np.ndarray
    stationary_reserve: np.ndarray
    moving_biomass: np.ndarray
    moving_reserve: np.ndarray


@dataclass
class ObservationBatch:
    features: np.ndarray
    raw_features: np.ndarray
    action_mask: np.ndarray
    subthreshold_mask: Optional[np.ndarray]
    dm_ids: List[str]
    global_fg_order: List[str]


@dataclass
class PolicyBatch:
    ready: bool = False
    weights: Optional[List[torch.Tensor]] = None
    biases: Optional[List[torch.Tensor]] = None
    in_dim: int = 0
    out_dim: int = 0


@dataclass
class Diagnostics:
    loss_predation: Dict[str, float]
    loss_starvation: Dict[str, float]
    intake_by_pred_prey: Dict[str, Dict[str, float]]


def build_static_caches(env):
    H, W = env.grid.height, env.grid.width
    env.H, env.W = H, W
    env.dtype = np.float32

    env.dm_ids = [
        fid for fid in env.global_fg_order
        if env.fgs[fid].is_decision_maker
    ]
    env.N_dm = len(env.dm_ids)
    env.N_all = len(env.global_fg_order)
    env.dm_index_in_all = np.array(
        [env.global_fg_order.index(fid) for fid in env.dm_ids],
        dtype=np.int64,
    )

    for fg in env.fgs.values():
        if fg.biomass is not None and fg.biomass.dtype != env.dtype:
            fg.biomass = fg.biomass.astype(env.dtype)
        if fg.energy_reserve is not None and fg.energy_reserve.dtype != env.dtype:
            fg.energy_reserve = fg.energy_reserve.astype(env.dtype)
        if fg.temp_energy_gains is not None and fg.temp_energy_gains.dtype != env.dtype:
            fg.temp_energy_gains = fg.temp_energy_gains.astype(env.dtype)

    env.move_mask = grid_masks.build_movement_mask(
        env.grid, env.migration, env.dtype)
    env._edge_imm_weights = grid_masks.build_edge_immigration_weights(
        env.grid, env.dtype)

    eat_static, max_intake, energy_gain, handling_time = (
        interactions.build_interaction_matrices(env)
    )
    env.eat_static_mask = eat_static
    env.max_intake_mat = max_intake
    env.energy_gain_mat = energy_gain
    env.handling_time_mat = handling_time
    env._has_holling2 = bool(np.any(handling_time > 0.0))
    # Specialists (exactly one active prey) use Type III; generalists
    # retain Type II because their policy already handles prey switching.
    n_prey_per_predator = eat_static.sum(axis=1)
    env._type3_pred_mask = (
        (n_prey_per_predator == 1)
        .astype(env.dtype)
        .reshape(env.N_dm, 1, 1, 1)
    )
    env._has_holling3 = bool(np.any(env._type3_pred_mask > 0.0))

    env.dm_v = np.array(
        [float(np.clip(env.fgs[fid].speed, 0.0, 1.0)) for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env.dm_cost_move = np.array(
        [env.fgs[fid].params.get("movement_cost", 3.0) for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env.dm_cost_eat = np.array(
        [env.fgs[fid].params.get("feeding_cost", 3.0) for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env.dm_cost_rest = np.array(
        [env.fgs[fid].params.get("resting_cost", 1.0) for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env.dm_resting_metabolism = np.array(
        [env.fgs[fid].resting_metabolism for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env.dm_visibility_floor = np.array(
        [float(getattr(env.fgs[fid], "visibility_floor", 0.0))
         for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env._all_visibility_floor = np.zeros(env.N_all, dtype=env.dtype)
    for i, _fid in enumerate(env.dm_ids):
        j = int(env.dm_index_in_all[i])
        env._all_visibility_floor[j] = env.dm_visibility_floor[i]

    env.dm_max_energy_reserve = np.array(
        [env.fgs[fid].max_energy_reserve for fid in env.dm_ids],
        dtype=env.dtype,
    )

    env.dm_min_split = np.array(
        [getattr(env.fgs[fid], "min_split_biomass", 0.0) for fid in env.dm_ids],
        dtype=env.dtype,
    )

    env.obs_others_idx = []
    for i, fid in enumerate(env.dm_ids):
        own_idx = int(env.dm_index_in_all[i])
        observes = env.fgs[fid].params.get("observes")
        if observes is None:
            idx = [j for j in range(env.N_all) if j != own_idx]
        else:
            observed_set = set(observes)
            idx = [
                j for j, other_id in enumerate(env.global_fg_order)
                if j != own_idx and other_id in observed_set
            ]
        env.obs_others_idx.append(np.asarray(idx, dtype=np.int64))

    env.per_dm_in_dim = np.zeros(env.N_dm, dtype=np.int64)
    for i in range(env.N_dm):
        k_i = int(env.obs_others_idx[i].shape[0])
        center_dim_i = 2 + k_i
        nbr_dim_i = 1 + k_i
        env.per_dm_in_dim[i] = center_dim_i + 4 * nbr_dim_i
    env.max_in_dim = int(env.per_dm_in_dim.max())

    env.prev_hidden_frac = np.zeros((env.N_all, H, W), dtype=env.dtype)

    env.cache = EcosystemCache(
        H=H,
        W=W,
        dtype=env.dtype,
        dm_ids=list(env.dm_ids),
        global_fg_order=list(env.global_fg_order),
        dm_index_in_all=env.dm_index_in_all,
        move_mask=env.move_mask,
        edge_immigration_weights=env._edge_imm_weights,
        eat_static_mask=env.eat_static_mask,
        max_intake=env.max_intake_mat,
        energy_gain=env.energy_gain_mat,
        handling_time=env.handling_time_mat,
        has_holling2=env._has_holling2,
        type3_pred_mask=env._type3_pred_mask,
        has_holling3=env._has_holling3,
        dm_v=env.dm_v,
        dm_cost_move=env.dm_cost_move,
        dm_cost_eat=env.dm_cost_eat,
        dm_cost_rest=env.dm_cost_rest,
        dm_resting_metabolism=env.dm_resting_metabolism,
        dm_visibility_floor=env.dm_visibility_floor,
        all_visibility_floor=env._all_visibility_floor,
        dm_max_energy_reserve=env.dm_max_energy_reserve,
        dm_min_split=env.dm_min_split,
        obs_others_idx=env.obs_others_idx,
        per_dm_in_dim=env.per_dm_in_dim,
        max_in_dim=env.max_in_dim,
    )

    from lib.environments.ecosystem_env import policies
    policies.rebuild_batched_weights(env)
    env._static_built = True
