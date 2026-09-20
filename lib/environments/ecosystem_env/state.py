from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from lib.environments.ecosystem_env import grid_masks, impacts, interactions


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
    dm_interference: np.ndarray
    has_interference: bool
    dm_v: np.ndarray
    dm_cost_move: np.ndarray
    dm_cost_eat: np.ndarray
    dm_cost_rest: np.ndarray
    dm_resting_metabolism: np.ndarray
    dm_visibility_floor: np.ndarray
    all_visibility_floor: np.ndarray
    vis_floor_mat: np.ndarray
    has_pair_vis_floor: bool
    dm_max_energy_reserve: np.ndarray
    dm_impact_tables: List[list]
    dm_min_split: np.ndarray
    dm_split_thr: np.ndarray
    dm_split_thr_any: bool
    obs_others_idx: List[np.ndarray]
    obs_vis_floor: List[np.ndarray]
    n_obs_imp: int
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

    matrices = interactions.build_interaction_matrices(env)
    eat_static = matrices.eat_static
    handling_time = matrices.handling_time
    env.eat_static_mask = eat_static
    env.max_intake_mat = matrices.max_intake
    env.energy_gain_mat = matrices.energy_gain
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

    # Beddington-DeAngelis interference coefficient w_X [1/ton], per DM.
    # Broadcast shape (N_dm, 1, 1, 1) so it multiplies the predator's own
    # per-cell biomass in ``predation.apply_predation``. All-zero (the
    # default) keeps the pure Holling code path and is bit-identical.
    env.dm_interference = np.array(
        [float(getattr(env.fgs[fid], "interference", 0.0) or 0.0)
         for fid in env.dm_ids],
        dtype=env.dtype,
    ).reshape(env.N_dm, 1, 1, 1)
    env._has_interference = bool(np.any(env.dm_interference > 0.0))

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

    # (N_dm, N_all) pair-resolved visibility floor. Base = the prey FG's
    # own floor (column broadcast), overridden where the interaction
    # definition carries an explicit ``visibility_floor``.
    # ``_has_pair_vis_floor`` is False when every override equals the
    # column default, in which case both the predation and the
    # observation path take the cheaper legacy vector code path with
    # bit-identical results.
    vis_floor_base = np.tile(env._all_visibility_floor[None, :], (env.N_dm, 1))
    env.vis_floor_mat = np.where(
        matrices.vis_floor_has, matrices.vis_floor_over, vis_floor_base
    ).astype(env.dtype, copy=False)
    env._has_pair_vis_floor = bool(
        np.any(env.vis_floor_mat != vis_floor_base))

    env.dm_max_energy_reserve = np.array(
        [env.fgs[fid].max_energy_reserve for fid in env.dm_ids],
        dtype=env.dtype,
    )

    # Per-DM cached impact tables for ALL impacts the FG is affected by.
    # Consumed by ``movement.apply_energy_costs`` (energy_factor -> extra
    # metabolic cost) and ``impacts.apply_impact_mortality``
    # (biomass_factor -> mortality).
    env.dm_impact_tables = impacts.build_dm_impact_tables(env)

    # Per-DM minimum biomass for splitting via movement. 0 = no threshold.
    env.dm_min_split = np.array(
        [getattr(env.fgs[fid], "min_split_biomass", 0.0) for fid in env.dm_ids],
        dtype=env.dtype,
    )

    # Per-DM extinction threshold thr = extinction_threshold_factor *
    # min_split_biomass, i.e. the exact bound the extinction sweep in
    # ``population_change.apply_extinction_threshold`` works with. Used by
    # ``movement.suppress_subthreshold_splits`` so a move action never
    # splits biomass into cells that would immediately be zeroed. 0 = off,
    # matching the extinction sweep's own opt-out.
    env._dm_split_thr = np.array(
        [(getattr(env.fgs[fid], "min_split_biomass", 0.0) or 0.0)
         * (getattr(env.fgs[fid], "extinction_threshold_factor", 0.0) or 0.0)
         for fid in env.dm_ids],
        dtype=env.dtype,
    )
    env._dm_split_thr_any = bool(np.any(env._dm_split_thr > 0.0))

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

    # Observation-side visibility floor, per DM, aligned with the compact
    # ``obs_others_idx`` layout (shape (k_i,)). Row i of ``vis_floor_mat``
    # already resolves to the prey FG's own floor for non-prey columns, so
    # "what I can see" and "what I can eat" stay consistent per observer.
    env._obs_vis_floor = [
        env.vis_floor_mat[i, env.obs_others_idx[i]]
        for i in range(env.N_dm)
    ]

    # Number of impact observation channels. Stored here so observation
    # dim calculations stay consistent in one place.
    n_obs_imp = len(env.observable_impact_vars)
    # Per-DM input dimension. Each DM sees:
    #   center : [B_own, E_own, B_obs_others (k_i), impacts (n_obs_imp)]
    #   neighbour (N/E/S/W) : [B_own, B_obs_others (k_i), impacts (n_obs_imp)]
    # so center_dim_i = 2 + k_i + n_obs_imp and nbr_dim_i = 1 + k_i + n_obs_imp.
    env.per_dm_in_dim = np.zeros(env.N_dm, dtype=np.int64)
    for i in range(env.N_dm):
        k_i = int(env.obs_others_idx[i].shape[0])
        center_dim_i = 2 + k_i + n_obs_imp
        nbr_dim_i = 1 + k_i + n_obs_imp
        env.per_dm_in_dim[i] = center_dim_i + 4 * nbr_dim_i
    env.max_in_dim = int(env.per_dm_in_dim.max()) if env.N_dm > 0 else 0
    env.n_obs_imp = n_obs_imp

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
        dm_interference=env.dm_interference,
        has_interference=env._has_interference,
        dm_v=env.dm_v,
        dm_cost_move=env.dm_cost_move,
        dm_cost_eat=env.dm_cost_eat,
        dm_cost_rest=env.dm_cost_rest,
        dm_resting_metabolism=env.dm_resting_metabolism,
        dm_visibility_floor=env.dm_visibility_floor,
        all_visibility_floor=env._all_visibility_floor,
        vis_floor_mat=env.vis_floor_mat,
        has_pair_vis_floor=env._has_pair_vis_floor,
        dm_max_energy_reserve=env.dm_max_energy_reserve,
        dm_impact_tables=env.dm_impact_tables,
        dm_min_split=env.dm_min_split,
        dm_split_thr=env._dm_split_thr,
        dm_split_thr_any=env._dm_split_thr_any,
        obs_others_idx=env.obs_others_idx,
        obs_vis_floor=env._obs_vis_floor,
        n_obs_imp=env.n_obs_imp,
        per_dm_in_dim=env.per_dm_in_dim,
        max_in_dim=env.max_in_dim,
    )

    from lib.environments.ecosystem_env import policies
    policies.rebuild_batched_weights(env)
    grid_masks.apply_accessibility_biomass_mask(env)
    env._static_built = True
