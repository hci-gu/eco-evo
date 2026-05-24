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
                 alpha=1.0, beta=1.0, obs_normalize=True, top_deltas=None, entropy_coef=0.0,
                 argmax_penalty=0.0, integral_reward=False):
        self.env_builder = env_builder
        self.policy_params = policy_params
        self.sigma = sigma
        self.lr = lr
        self.n_deltas = n_deltas
        self.n_workers = max(1, int(n_workers))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.obs_normalize = bool(obs_normalize)
        # Entropy bonus in fitness: fitness += entropy_coef * H(pi)/H_max.
        # Belönar policyer som bibehåller utforskning -> motverkar att
        # softmax kollapsar till en konstant action ("ät alltid" /
        # "rest alltid") under de första generationerna.
        self.entropy_coef = float(entropy_coef)
        # Argmax-penalty: fitness -= argmax_penalty * max_argmax_frac, där
        # max_argmax_frac = max(move_frac, rest_frac, eat_frac) över aktiva
        # celler. Straffar direkt degenererade en-action-policyer ("ät
        # alltid" / "rest alltid"); kompletterar entropy-bonus som bara
        # mäter softmax-spridning, inte argmax-degeneration.
        self.argmax_penalty = float(argmax_penalty)
        # Softmax-temperatur, sätts externt per generation (linjär annealing
        # från temp_start -> temp_end). Hög T -> jämnare softmax -> tvingar
        # utforskning. Skiljer sig från entropy-bonus genom att påverka
        # action-distributionen *direkt* istället för fitness.
        self.softmax_temperature = 1.0
        # Integral-baserad reward: använd medel-biomassa/medel-energy över
        # hela rolloutet istället för slutvärdet. Tanken: "ät alltid" som
        # leder till byteskollaps under rolloutet får sänkt reward eftersom
        # genomsnittet sjunker, även om slutvärdet hade varit jämförbart.
        self.integral_reward = bool(integral_reward)
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
        # Action-entropy accumulators (averaged across rollouts).
        act_entropy_sum = 0.0
        act_move_sum = 0.0
        act_rest_sum = 0.0
        act_eat_sum = 0.0
        act_n = 0
        act_max_entropy = None

        # Per-rollout act_diag, kept aligned with rewards_pos / rewards_neg
        # so we can apply entropy bonus / argmax-penalty per rollout *after*
        # z-score normalisation of the ecological component.
        act_pos = []
        act_neg = []
        if self._pool is None:
            rewards_pos = []
            rewards_neg = []
            for i, delta in enumerate(deltas):
                s = pair_seeds[i]
                self._set_weights(policy, weights + self.sigma * delta)
                r, samples, act = self._evaluate(fg_to_train, n_eval_ticks, seed=s,
                                            obs_mean=obs_mean, obs_var=obs_var,
                                            dm_ids_for_norm=dm_ids_for_norm)
                rewards_pos.append(r); act_pos.append(act)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    if agg_sum is None:
                        agg_sum = s_sum.copy(); agg_sumsq = s_sumsq.copy()
                    else:
                        agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt
                if act is not None:
                    act_entropy_sum += act['entropy']; act_move_sum += act['move_frac']
                    act_rest_sum += act['rest_frac']; act_eat_sum += act['eat_frac']
                    act_max_entropy = act['max_entropy']; act_n += 1

                self._set_weights(policy, weights - self.sigma * delta)
                r, samples, act = self._evaluate(fg_to_train, n_eval_ticks, seed=s,
                                            obs_mean=obs_mean, obs_var=obs_var,
                                            dm_ids_for_norm=dm_ids_for_norm)
                rewards_neg.append(r); act_neg.append(act)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt
                if act is not None:
                    act_entropy_sum += act['entropy']; act_move_sum += act['move_frac']
                    act_rest_sum += act['rest_frac']; act_eat_sum += act['eat_frac']
                    act_max_entropy = act['max_entropy']; act_n += 1
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
                              pair_seeds[i], obs_pack, self.entropy_coef, self.argmax_penalty,
                              self.softmax_temperature, self.integral_reward))
            for i, delta in enumerate(deltas):
                d = delta.numpy()
                w_neg = dict(base_weights)
                w_neg[fg_to_train] = base_train - self.sigma * d
                tasks.append((fg_to_train, w_neg, n_eval_ticks, self.alpha, self.beta,
                              pair_seeds[i], obs_pack, self.entropy_coef, self.argmax_penalty,
                              self.softmax_temperature, self.integral_reward))

            results = self._pool.map(_evaluate_task, tasks)
            # results: list of (fitness, samples_or_None, act_diag_or_None)
            rewards = []
            acts_all = []
            for item in results:
                if len(item) == 3:
                    fit, samples, act = item
                else:
                    fit, samples = item; act = None
                rewards.append(fit); acts_all.append(act)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    if agg_sum is None:
                        agg_sum = s_sum.copy(); agg_sumsq = s_sumsq.copy()
                    else:
                        agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt
                if act is not None:
                    act_entropy_sum += act['entropy']; act_move_sum += act['move_frac']
                    act_rest_sum += act['rest_frac']; act_eat_sum += act['eat_frac']
                    act_max_entropy = act['max_entropy']; act_n += 1
            rewards_pos = np.array(rewards[:self.n_deltas])
            rewards_neg = np.array(rewards[self.n_deltas:])
            act_pos = acts_all[:self.n_deltas]
            act_neg = acts_all[self.n_deltas:]

            self._set_weights(policy, weights)

        # --- Z-score normalisation of ecological component, then apply
        # entropy bonus and argmax-penalty on the normalised scale. Both
        # bonus and penalty live on [0,1] (H/H_max and max action-frac),
        # so a single coefficient now has comparable weight across all
        # species regardless of |delta_b + delta_r| magnitude.
        if self.entropy_coef != 0.0 or self.argmax_penalty != 0.0:
            eco_all = np.concatenate([rewards_pos, rewards_neg])
            eco_mean = float(np.mean(eco_all))
            eco_std = float(np.std(eco_all)) + 1e-8

            def _modify(r_arr, act_arr):
                out = np.empty_like(r_arr, dtype=np.float64)
                for k in range(len(r_arr)):
                    rn = (float(r_arr[k]) - eco_mean) / eco_std
                    a = act_arr[k] if k < len(act_arr) else None
                    bonus = 0.0
                    if a is not None:
                        if self.entropy_coef != 0.0:
                            Hm = a['max_entropy'] or 1.0
                            bonus += self.entropy_coef * (a['entropy'] / Hm)
                        if self.argmax_penalty != 0.0:
                            mf = max(a['move_frac'], a['rest_frac'], a['eat_frac'])
                            bonus -= self.argmax_penalty * mf
                    out[k] = rn + bonus
                return out

            rewards_pos = _modify(rewards_pos, act_pos)
            rewards_neg = _modify(rewards_neg, act_neg)

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

        # --- Diagnostics for convergence debugging ---
        all_rewards = np.concatenate([rewards_pos, rewards_neg])
        r_mean = float(np.mean(all_rewards))
        r_std = float(np.std(all_rewards))
        rel = r_std / (abs(r_mean) + 1e-12)
        diag = {
            'reward_mean': r_mean,
            'reward_std': r_std,
            'reward_rel_std': rel,
            'sigma_f': float(sigma_f),
        }
        if self.obs_normalize and dm_ids_for_norm is not None:
            # obs_var stats for fg_to_train
            if fg_to_train in self.obs_stats:
                v = self.obs_stats[fg_to_train]['var']
                diag['obs_var_min'] = float(np.min(v))
                diag['obs_var_max'] = float(np.max(v))
                diag['obs_var_mean'] = float(np.mean(v))
                diag['obs_var_n_near_zero'] = int(np.sum(v < 1e-6))
                diag['obs_var_dim'] = int(v.shape[-1])
                diag['obs_count'] = int(self.obs_stats[fg_to_train]['count'])
        # Action-entropy aggregate across rollouts in this iteration.
        act_str = ""
        if act_n > 0:
            H_mean = act_entropy_sum / act_n
            H_max = act_max_entropy or 1.0
            mv = act_move_sum / act_n; rs = act_rest_sum / act_n; et = act_eat_sum / act_n
            act_str = (f" H_act={H_mean:.3f}/{H_max:.3f} ({H_mean/H_max:.0%})"
                       f" mv/rs/et={mv:.2f}/{rs:.2f}/{et:.2f}")
        # Print compact line
        print(f"  [diag {fg_to_train}] r_mean={r_mean:+.4e} r_std={r_std:.4e} "
              f"rel_std={rel:.3%} sigma_f={sigma_f:.4e}"
              + (f" obs_var[min={diag.get('obs_var_min',float('nan')):.2e},"
                 f" mean={diag.get('obs_var_mean',float('nan')):.2e},"
                 f" max={diag.get('obs_var_max',float('nan')):.2e},"
                 f" near0={diag.get('obs_var_n_near_zero',0)}/{diag.get('obs_var_dim',0)},"
                 f" cnt={diag.get('obs_count',0)}]"
                 if 'obs_var_min' in diag else "")
              + act_str)

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
        env.softmax_temperature = float(self.softmax_temperature)

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

        # Integral-reward: ackumulera bh/rh per tick och dela med antal
        # ticks i slutet. Annars: använd bara slutvärdet (klassisk).
        if self.integral_reward and n_ticks > 0:
            b_sum = 0.0; r_sum = 0.0
            for _ in range(n_ticks):
                env.step()
                b_sum += float(env.fgs[fg_id].biomass.sum())
                r_sum += float(env.fgs[fg_id].energy_reserve.sum())
            bh = b_sum / n_ticks
            rh = r_sum / n_ticks
        else:
            for _ in range(n_ticks):
                env.step()
            bh = env.fgs[fg_id].biomass.sum()
            rh = env.fgs[fg_id].energy_reserve.sum()

        eps_b = max(1e-6 * b0, 1e-9)
        eps_r = max(1e-6 * r0, 1e-9)

        delta_b = np.log((bh + eps_b) / (b0 + eps_b))
        delta_r = np.log((rh + eps_r) / (r0 + eps_r))
        # Raw ecological fitness. Entropy bonus / argmax-penalty are applied
        # in train_step *after* z-score normalisation of the ecological
        # component across the 2*n_deltas batch (principled scaling).
        fitness = self.alpha * delta_b + self.beta * delta_r

        samples = None
        if obs_mean is not None and getattr(env, '_obs_sum', None) is not None:
            samples = (env._obs_sum.copy(), env._obs_sumsq.copy(), env._obs_count)
        # Action-entropy diagnostics for fg_to_train (averaged across ticks).
        act_diag = None
        if getattr(env, '_action_entropy_sum', None) is not None and env._action_entropy_count > 0:
            try:
                i = env.dm_ids.index(fg_id)
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
        return fitness, samples, act_diag
