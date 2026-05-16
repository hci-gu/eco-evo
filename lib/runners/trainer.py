import torch
import numpy as np
import multiprocessing as mp
from lib.runners.policy import PolicyNetwork
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.parallel_worker import _worker_init, _evaluate_task

class ARSTrainer:
    """ARS trainer with optional ARS-V2 extensions:
      - obs_normalize: running mean/std per-DM, per-obs-dim, frozen during a
        rollout; global stats updated across rollouts via Welford parallel
        merge. Sequential and worker paths both feed samples back.
      - top_deltas: keep only the top-b delta pairs (sorted by max(r+, r-))
        when forming the ARS gradient. b defaults to n_deltas (no truncation).
      Common Random Numbers and alpha/beta fitness weighting are inherited
      from the original implementation.
    """
    def __init__(self, env_builder, policy_params, sigma=0.1, lr=0.02, n_deltas=8, n_workers=1,
                 alpha=1.0, beta=1.0, obs_normalize=True, top_deltas=None):
        self.env_builder = env_builder
        self.policy_params = policy_params
        self.sigma = sigma
        self.lr = lr
        self.n_deltas = n_deltas
        self.n_workers = max(1, int(n_workers))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.obs_normalize = bool(obs_normalize)
        # Default: use all deltas (no truncation). When set, must be in [1, n_deltas].
        if top_deltas is None:
            self.top_deltas = self.n_deltas
        else:
            self.top_deltas = max(1, min(int(top_deltas), self.n_deltas))

        # Initialize policies (parent-side; workers hold their own copies)
        self.policies = {}
        for fg_id, (in_dim, out_dim) in policy_params.items():
            self.policies[fg_id] = PolicyNetwork(in_dim, out_dim)

        # Observation running statistics, lazily-shaped on first task return.
        # Shape per fg: mean (D,), var (D,), count int.
        # During a train_step, we materialise an (N_dm, D) view aligned with
        # env.dm_ids ordering and hand it to env/workers.
        self.obs_stats = {}  # fg_id -> dict(mean, var, count)

        # Lazy-initialized worker pool
        self._pool = None
        if self.n_workers > 1:
            ctx = mp.get_context('spawn')
            self._pool = ctx.Pool(
                processes=self.n_workers,
                initializer=_worker_init,
                initargs=(env_builder, policy_params),
            )

    def close(self):
        if self._pool is not None:
            try:
                self._pool.terminate()
            except Exception:
                pass
            try:
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- obs stats helpers ----
    def _get_dm_ids_and_dim(self):
        """Build a fresh env once to discover dm_ids and obs-dim."""
        env = self.env_builder()
        env.policies = self.policies
        # Trigger lazy build by running a single forward pass through
        # _build_static_caches without stepping the simulation.
        env._build_static_caches()
        n_all = env.N_all
        D = 2 + (n_all - 1) + 1
        return list(env.dm_ids), D

    def _ensure_obs_stats(self, dm_ids, D):
        for fid in dm_ids:
            if fid not in self.obs_stats:
                self.obs_stats[fid] = {
                    'mean': np.zeros(D, dtype=np.float64),
                    'var': np.ones(D, dtype=np.float64),
                    'count': 0,
                }

    def _build_obs_arrays(self, dm_ids, D):
        """Stack per-DM mean/var arrays into (N_dm, D) for the env."""
        mean = np.stack([self.obs_stats[fid]['mean'] for fid in dm_ids], axis=0).astype(np.float32)
        var = np.stack([self.obs_stats[fid]['var'] for fid in dm_ids], axis=0).astype(np.float32)
        return mean, var

    def _merge_obs_stats(self, dm_ids, sum_arr, sumsq_arr, count):
        """Welford parallel-merge running stats with one rollout's samples.
        sum_arr/sumsq_arr: (N_dm, D) float64. count: int (samples per DM)."""
        if count <= 0:
            return
        for i, fid in enumerate(dm_ids):
            st = self.obs_stats[fid]
            n_a = st['count']
            n_b = count
            mean_a = st['mean']
            mean_b = sum_arr[i] / n_b
            # M2_b = sumsq - n_b * mean_b**2
            M2_b = sumsq_arr[i] - n_b * (mean_b ** 2)
            if n_a == 0:
                st['mean'] = mean_b
                # population variance (consistent with M2/n_b)
                st['var'] = np.maximum(M2_b / max(1, n_b), 1e-8)
                st['count'] = n_b
                continue
            delta = mean_b - mean_a
            n = n_a + n_b
            new_mean = mean_a + delta * (n_b / n)
            M2_a = st['var'] * n_a
            M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b / n)
            st['mean'] = new_mean
            st['var'] = np.maximum(M2 / n, 1e-8)
            st['count'] = n

    def train_step(self, fg_to_train, n_eval_ticks=2):
        policy = self.policies[fg_to_train]
        weights = self._get_weights(policy)

        deltas = [torch.randn_like(weights) for _ in range(self.n_deltas)]

        # CRN seeds per delta pair
        pair_seeds = [int(np.random.randint(1, 2**31 - 1)) for _ in range(self.n_deltas)]

        # Prepare observation stats (only if normalisation is enabled).
        obs_mean = obs_var = None
        dm_ids_for_norm = None
        D = None
        if self.obs_normalize:
            dm_ids_for_norm, D = self._get_dm_ids_and_dim()
            self._ensure_obs_stats(dm_ids_for_norm, D)
            obs_mean, obs_var = self._build_obs_arrays(dm_ids_for_norm, D)

        # Accumulators for sample stats from this train_step.
        agg_sum = None      # (N_dm, D) float64
        agg_sumsq = None
        agg_count = 0       # samples per DM accumulated across all rollouts

        if self._pool is None:
            rewards_pos = []
            rewards_neg = []
            for i, delta in enumerate(deltas):
                s = pair_seeds[i]
                self._set_weights(policy, weights + self.sigma * delta)
                r, samples = self._evaluate(fg_to_train, n_eval_ticks, seed=s,
                                            obs_mean=obs_mean, obs_var=obs_var,
                                            dm_ids_for_norm=dm_ids_for_norm)
                rewards_pos.append(r)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    if agg_sum is None:
                        agg_sum = s_sum.copy(); agg_sumsq = s_sumsq.copy()
                    else:
                        agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt

                self._set_weights(policy, weights - self.sigma * delta)
                r, samples = self._evaluate(fg_to_train, n_eval_ticks, seed=s,
                                            obs_mean=obs_mean, obs_var=obs_var,
                                            dm_ids_for_norm=dm_ids_for_norm)
                rewards_neg.append(r)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt
            rewards_pos = np.array(rewards_pos)
            rewards_neg = np.array(rewards_neg)
        else:
            base_weights = {
                fg_id: self._get_weights(p).numpy().copy()
                for fg_id, p in self.policies.items()
            }
            base_train = weights.numpy()

            # Pack obs stats once (shared across all tasks in this iteration).
            obs_pack = None
            if self.obs_normalize:
                obs_pack = {
                    'dm_ids': dm_ids_for_norm,
                    'mean': obs_mean,
                    'var': obs_var,
                }

            tasks = []
            for i, delta in enumerate(deltas):
                d = delta.numpy()
                w_pos = dict(base_weights)
                w_pos[fg_to_train] = base_train + self.sigma * d
                tasks.append((fg_to_train, w_pos, n_eval_ticks, self.alpha, self.beta,
                              pair_seeds[i], obs_pack))
            for i, delta in enumerate(deltas):
                d = delta.numpy()
                w_neg = dict(base_weights)
                w_neg[fg_to_train] = base_train - self.sigma * d
                tasks.append((fg_to_train, w_neg, n_eval_ticks, self.alpha, self.beta,
                              pair_seeds[i], obs_pack))

            results = self._pool.map(_evaluate_task, tasks)
            # results: list of (fitness, samples_or_None)
            rewards = []
            for fit, samples in results:
                rewards.append(fit)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    if agg_sum is None:
                        agg_sum = s_sum.copy(); agg_sumsq = s_sumsq.copy()
                    else:
                        agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt
            rewards_pos = np.array(rewards[:self.n_deltas])
            rewards_neg = np.array(rewards[self.n_deltas:])

            self._set_weights(policy, weights)

        # Merge sample stats into running stats (Welford parallel).
        if self.obs_normalize and agg_count > 0 and dm_ids_for_norm is not None:
            self._merge_obs_stats(dm_ids_for_norm, agg_sum, agg_sumsq, agg_count)

        # ARS update (with optional top-b deltas selection).
        # Score per pair = max(|r_pos|, |r_neg|)  -> standard ARS-V2 criterion
        # uses max(r_pos[i], r_neg[i]); we follow that exactly.
        pair_scores = np.maximum(rewards_pos, rewards_neg)
        b = self.top_deltas
        if b < self.n_deltas:
            top_idx = np.argsort(-pair_scores)[:b]
        else:
            top_idx = np.arange(self.n_deltas)

        # sigma_f computed over the selected pairs' rewards (both halves), per
        # ARS-V2 spec. Using only top-b reduces noise from low-reward pairs.
        sel_rewards = np.concatenate([rewards_pos[top_idx], rewards_neg[top_idx]])
        sigma_f = np.std(sel_rewards) + 1e-8

        step = np.zeros_like(weights.numpy())
        for i in top_idx:
            step += (rewards_pos[i] - rewards_neg[i]) * deltas[i].numpy()
        new_weights = weights.numpy() + (self.lr / (b * sigma_f)) * step
        self._set_weights(policy, torch.from_numpy(new_weights))

        return float(np.mean(rewards_pos + rewards_neg))

    def _get_weights(self, policy):
        return torch.cat([p.data.view(-1) for p in policy.parameters()])

    def _set_weights(self, policy, weights):
        idx = 0
        for p in policy.parameters():
            p_size = p.data.numel()
            p.data.copy_(weights[idx:idx + p_size].view(p.size()))
            idx += p_size

    def _evaluate(self, fg_id, n_ticks, seed=None,
                  obs_mean=None, obs_var=None, dm_ids_for_norm=None):
        env = self.env_builder(seed=seed) if seed is not None else self.env_builder()
        env.policies = self.policies

        # Install obs-normalisation stats (frozen during this rollout).
        if obs_mean is not None and obs_var is not None:
            # Reorder if env.dm_ids differs from dm_ids_for_norm; in practice
            # both come from the same config, but guard against future changes.
            env._build_static_caches()
            if dm_ids_for_norm == env.dm_ids:
                env.obs_mean = obs_mean
                env.obs_var = obs_var
            else:
                idx = [dm_ids_for_norm.index(fid) for fid in env.dm_ids]
                env.obs_mean = obs_mean[idx]
                env.obs_var = obs_var[idx]

        b0 = env.fgs[fg_id].biomass.sum()
        r0 = env.fgs[fg_id].energy_reserve.sum()

        for _ in range(n_ticks):
            env.step()

        bh = env.fgs[fg_id].biomass.sum()
        rh = env.fgs[fg_id].energy_reserve.sum()

        eps_b = max(1e-6 * b0, 1e-9)
        eps_r = max(1e-6 * r0, 1e-9)

        delta_b = np.log((bh + eps_b) / (b0 + eps_b))
        delta_r = np.log((rh + eps_r) / (r0 + eps_r))
        fitness = self.alpha * delta_b + self.beta * delta_r

        samples = None
        if obs_mean is not None and getattr(env, '_obs_sum', None) is not None:
            samples = (env._obs_sum.copy(), env._obs_sumsq.copy(), env._obs_count)
        return fitness, samples
