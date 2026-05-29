import torch
import numpy as np
import multiprocessing as mp
from lib.runners.policy import PolicyNetwork
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.parallel_worker import _worker_init, _evaluate_task, _evaluate_coevo_task

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
                 argmax_penalty=0.0, integral_reward=True, uniform_bias_init=False):
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
        # Rewards policies that maintain exploration -> counteracts softmax
        # collapse to a constant action ("eat always" / "rest always")
        # during the first generations.
        self.entropy_coef = float(entropy_coef)
        # Argmax-penalty: fitness -= argmax_penalty * max_argmax_frac, where
        # max_argmax_frac = max(move_frac, rest_frac, eat_frac) across active
        # cells. Directly penalises degenerate single-action policies ("eat
        # always" / "rest always"); complements entropy-bonus which only
        # measures softmax spread, not argmax degeneration.
        self.argmax_penalty = float(argmax_penalty)
        # Softmax temperature, set externally per generation (linear annealing
        # from temp_start -> temp_end). High T -> flatter softmax -> forces
        # exploration. Differs from entropy-bonus by affecting the
        # action distribution *directly* instead of fitness.
        self.softmax_temperature = 1.0
        # Integral-based reward: use mean biomass / mean energy over the
        # whole rollout instead of the final value. Idea: "eat always" which
        # leads to prey collapse during the rollout gets lower reward because
        # the average drops, even if the final value would have been comparable.
        self.integral_reward = bool(integral_reward)
        # Default: use all deltas (no truncation). When set, must be in [1, n_deltas].
        if top_deltas is None:
            self.top_deltas = self.n_deltas
        else:
            self.top_deltas = max(1, min(int(top_deltas), self.n_deltas))

        # STEP 3 (multi-world averaging): per-iteration world list, set
        # externally by train.py. Each entry is (impact_seed, spawn_seed)
        # describing one of the M independent worlds that every delta-pair
        # is evaluated against. Empty list -> legacy single-world path
        # (M=1) using ``self.env_builder`` exactly as before. When
        # len(world_list) > 1, the rollout loop averages fitness/act over
        # the M worlds, using ``self.env_builder.with_world(...)`` to
        # build per-world sibling builders.
        self.world_list = []

        # Initialize policies (parent-side; workers hold their own copies)
        self.uniform_bias_init = bool(uniform_bias_init)
        self.policies = {}
        for fg_id, (in_dim, out_dim) in policy_params.items():
            self.policies[fg_id] = PolicyNetwork(in_dim, out_dim,
                                                 uniform_bias_init=self.uniform_bias_init)

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
                initargs=(env_builder, policy_params, self.uniform_bias_init),
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

    def _resolve_world_builders(self):
        """STEP 3 helper: return the list of (per-world) env builders for
        this iteration.

        - Empty / len==1 ``world_list``: returns ``[self.env_builder]``;
          the M=1 legacy path used by both train_step and
          train_step_coevolution. No ``with_world`` call is made so the
          single-world builder installed by train.py
          ``_install_generation_worlds`` is used verbatim (and worker
          tasks can omit the override entirely).
        - len > 1: builds M sibling builders via
          ``self.env_builder.with_world(impact_seed, spawn_seed)``,
          one per world in ``self.world_list``.

        Returns:
            list[callable] of length M.
        """
        wl = list(self.world_list) if self.world_list else []
        if len(wl) <= 1:
            return [self.env_builder]
        builders = []
        for (imp_s, sp_s) in wl:
            builders.append(self.env_builder.with_world(int(imp_s), int(sp_s)))
        return builders

    def train_step(self, fg_to_train, n_eval_ticks=2):
        policy = self.policies[fg_to_train]
        weights = self._get_weights(policy)

        deltas = [torch.randn_like(weights) for _ in range(self.n_deltas)]

        # CRN seeds per delta pair
        pair_seeds = [int(np.random.randint(1, 2**31 - 1)) for _ in range(self.n_deltas)]

        # STEP 3: per-iteration world builders (M=1 legacy or M>1 multi-world).
        world_builders = self._resolve_world_builders()
        M = len(world_builders)

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
        # STEP 3: when M>1, ``act_pos[i]`` holds the *averaged* act_diag
        # across the M worlds for delta i; ``rewards_pos[i]`` is the
        # M-averaged fitness. CRN: world m for +delta and -delta share
        # the exact same (impact_seed, spawn_seed, pair_seeds[i]).
        act_pos = []
        act_neg = []

        def _avg_act_diags(diags):
            """Average a list of act_diag dicts (skipping Nones).
            Returns None if all are None."""
            valid = [d for d in diags if d is not None]
            if not valid:
                return None
            keys = ('entropy', 'move_frac', 'rest_frac', 'eat_frac')
            out = {k: float(np.mean([d[k] for d in valid])) for k in keys}
            out['max_entropy'] = valid[0]['max_entropy']
            return out

        if self._pool is None:
            rewards_pos = []
            rewards_neg = []
            for i, delta in enumerate(deltas):
                s = pair_seeds[i]
                # Per-world averaging within this delta pair.
                rp_world = []; rn_world = []
                ap_world = []; an_world = []
                for m in range(M):
                    builder_m = world_builders[m]
                    self._set_weights(policy, weights + self.sigma * delta)
                    r, samples, act = self._evaluate(
                        fg_to_train, n_eval_ticks, seed=s,
                        obs_mean=obs_mean, obs_var=obs_var,
                        dm_ids_for_norm=dm_ids_for_norm,
                        env_builder=builder_m)
                    rp_world.append(r); ap_world.append(act)
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
                    r, samples, act = self._evaluate(
                        fg_to_train, n_eval_ticks, seed=s,
                        obs_mean=obs_mean, obs_var=obs_var,
                        dm_ids_for_norm=dm_ids_for_norm,
                        env_builder=builder_m)
                    rn_world.append(r); an_world.append(act)
                    if samples is not None:
                        s_sum, s_sumsq, s_cnt = samples
                        agg_sum += s_sum; agg_sumsq += s_sumsq
                        agg_count += s_cnt
                    if act is not None:
                        act_entropy_sum += act['entropy']; act_move_sum += act['move_frac']
                        act_rest_sum += act['rest_frac']; act_eat_sum += act['eat_frac']
                        act_max_entropy = act['max_entropy']; act_n += 1
                rewards_pos.append(float(np.mean(rp_world)))
                rewards_neg.append(float(np.mean(rn_world)))
                act_pos.append(_avg_act_diags(ap_world))
                act_neg.append(_avg_act_diags(an_world))
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

            # STEP 3: when M>1, emit 2*n_deltas*M tasks; for each (i, sign, m)
            # the worker uses builder_m via the dict-task override. For M=1
            # we keep the original tuple-task format (no override needed)
            # so the worker stays on its byte-identical legacy code path.
            tasks = []
            if M == 1:
                for sign in (+1, -1):
                    for i, delta in enumerate(deltas):
                        d = delta.numpy()
                        w = dict(base_weights)
                        w[fg_to_train] = base_train + sign * self.sigma * d
                        tasks.append((fg_to_train, w, n_eval_ticks, self.alpha, self.beta,
                                      pair_seeds[i], obs_pack, self.entropy_coef,
                                      self.argmax_penalty, self.softmax_temperature,
                                      self.integral_reward))
            else:
                for sign in (+1, -1):
                    for i, delta in enumerate(deltas):
                        d = delta.numpy()
                        w = dict(base_weights)
                        w[fg_to_train] = base_train + sign * self.sigma * d
                        for m in range(M):
                            tasks.append({
                                'fg_to_train': fg_to_train,
                                'weights_dict': w,
                                'n_ticks': n_eval_ticks,
                                'alpha': self.alpha,
                                'beta': self.beta,
                                'seed': pair_seeds[i],
                                'obs_pack': obs_pack,
                                'entropy_coef': self.entropy_coef,
                                'argmax_penalty': self.argmax_penalty,
                                'softmax_temperature': self.softmax_temperature,
                                'integral_reward': self.integral_reward,
                                'env_builder': world_builders[m],
                            })

            results = self._pool.map(_evaluate_task, tasks)
            # Unpack & accumulate obs / act stats over ALL rollouts.
            rewards_flat = []
            acts_flat = []
            for item in results:
                if len(item) == 3:
                    fit, samples, act = item
                else:
                    fit, samples = item; act = None
                rewards_flat.append(fit); acts_flat.append(act)
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

            # Reshape: task layout is [sign(+), sign(-)] outer, then for
            # each sign: i in [0..n_deltas), then m in [0..M).
            n = self.n_deltas
            pos_block = rewards_flat[:n * M]
            neg_block = rewards_flat[n * M:]
            act_pos_block = acts_flat[:n * M]
            act_neg_block = acts_flat[n * M:]
            rewards_pos = np.array([
                float(np.mean(pos_block[i * M:(i + 1) * M])) for i in range(n)
            ])
            rewards_neg = np.array([
                float(np.mean(neg_block[i * M:(i + 1) * M])) for i in range(n)
            ])
            act_pos = [_avg_act_diags(act_pos_block[i * M:(i + 1) * M]) for i in range(n)]
            act_neg = [_avg_act_diags(act_neg_block[i * M:(i + 1) * M]) for i in range(n)]

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
                  obs_mean=None, obs_var=None, dm_ids_for_norm=None,
                  env_builder=None):
        # STEP 3: optional per-world builder override. Falls back to
        # self.env_builder so legacy callers (M=1 path) are unchanged.
        builder = env_builder if env_builder is not None else self.env_builder
        env = builder(seed=seed) if seed is not None else builder()
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

        # Integral-reward: accumulate bh/rh per tick and divide by tick
        # count at the end. Otherwise: use only the final value (classic).
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

    # ================================================================
    # Co-evolution (track C in konvergensproblem.txt)
    # ----------------------------------------------------------------
    # Trains ALL decision makers simultaneously: per delta pair each
    # species' weights are perturbed with an independent delta (shared
    # pair_seed), and a SHARED rollout returns fitness for every species.
    # Thus the predator policy sees the prey collapse its "eat always"
    # causes -- and the prey policy sees the predation pressure. The
    # negative feedback missing in round-robin training now materialises
    # as part of the gradient.
    # ================================================================
    def _evaluate_coevo(self, fg_list, n_ticks, seed=None,
                        obs_mean=None, obs_var=None, dm_ids_for_norm=None,
                        env_builder=None):
        """Sequential co-evolution rollout: returns fitness per species
        from a single shared simulation.

        STEP 3: ``env_builder`` is an optional per-world builder override
        used by the multi-world averaging path; default None means use
        ``self.env_builder`` (M=1 legacy)."""
        builder = env_builder if env_builder is not None else self.env_builder
        env = builder(seed=seed) if seed is not None else builder()
        env.policies = self.policies
        env.softmax_temperature = float(self.softmax_temperature)

        if obs_mean is not None and obs_var is not None:
            env._build_static_caches()
            if dm_ids_for_norm == env.dm_ids:
                env.obs_mean = obs_mean
                env.obs_var = obs_var
            else:
                idx = [dm_ids_for_norm.index(fid) for fid in env.dm_ids]
                env.obs_mean = obs_mean[idx]
                env.obs_var = obs_var[idx]

        b0 = {fid: float(env.fgs[fid].biomass.sum()) for fid in fg_list}
        r0 = {fid: float(env.fgs[fid].energy_reserve.sum()) for fid in fg_list}

        if self.integral_reward and n_ticks > 0:
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
            fitness[fid] = float(self.alpha * delta_b + self.beta * delta_r)

        samples = None
        if obs_mean is not None and getattr(env, '_obs_sum', None) is not None:
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

        return fitness, samples, act_diag_dict

    def train_step_coevolution(self, target_species, n_eval_ticks=15):
        """Co-evolution training step (track C).

        Perturbs ALL species in `target_species` simultaneously per delta
        pair and runs a shared rollout. Each species receives its own
        fitness from the same simulation, after which the ARS update is
        performed independently per species with the reward batch from
        this iteration.

        Args:
            target_species: list[str] -- species to train simultaneously.
            n_eval_ticks: int -- ticks per rollout.

        Returns:
            dict[fg_id -> float] -- mean reward per species over the batch.
        """
        # Snapshot of current weights and per-species deltas.
        base_weights = {}     # fg_id -> torch.Tensor (1D)
        deltas = {}           # fg_id -> list[torch.Tensor]
        for fid in target_species:
            w = self._get_weights(self.policies[fid])
            base_weights[fid] = w
            deltas[fid] = [torch.randn_like(w) for _ in range(self.n_deltas)]

        # CRN seeds per delta pair (same rollout seed for +/- for a
        # given i -> identical impact / initial fields).
        pair_seeds = [int(np.random.randint(1, 2**31 - 1)) for _ in range(self.n_deltas)]

        # STEP 3: per-iteration world builders (M=1 legacy or M>1 multi-world).
        world_builders = self._resolve_world_builders()
        M = len(world_builders)

        # Obs-stats setup (same as in train_step).
        obs_mean = obs_var = None
        dm_ids_for_norm = None
        D = None
        if self.obs_normalize:
            dm_ids_for_norm, D = self._get_dm_ids_and_dim()
            self._ensure_obs_stats(dm_ids_for_norm, D)
            obs_mean, obs_var = self._build_obs_arrays(dm_ids_for_norm, D)

        agg_sum = None; agg_sumsq = None; agg_count = 0

        # Weights for UNTRAINED species (non decision makers + species
        # outside target_species but with a policy) -- used as baseline
        # in all rollouts.
        non_target_weights = {
            fg_id: self._get_weights(p).numpy().copy()
            for fg_id, p in self.policies.items()
            if fg_id not in target_species
        }

        # rewards_pos[i] / rewards_neg[i] = dict fg_id -> float
        # act_pos[i] / act_neg[i]         = dict fg_id -> act_diag
        rewards_pos = [None] * self.n_deltas
        rewards_neg = [None] * self.n_deltas
        act_pos = [None] * self.n_deltas
        act_neg = [None] * self.n_deltas

        def _build_weights_dict(sign, idx):
            """Build a weight dict for all policies, perturbed per species (delta-pair idx)."""
            wd = dict(non_target_weights)
            for fid in target_species:
                wd[fid] = base_weights[fid].numpy() + sign * self.sigma * deltas[fid][idx].numpy()
            return wd

        def _avg_fit_dicts(fits):
            """Average a list of {fg_id -> float} dicts."""
            keys = fits[0].keys()
            return {k: float(np.mean([d[k] for d in fits])) for k in keys}

        def _avg_act_dicts(acts):
            """Average a list of {fg_id -> act_diag} dicts (skipping
            Nones / missing keys). Returns None if all are None."""
            valid = [d for d in acts if d is not None]
            if not valid:
                return None
            keys = ('entropy', 'move_frac', 'rest_frac', 'eat_frac')
            out = {}
            for fid in valid[0].keys():
                rows = [d[fid] for d in valid if fid in d]
                if not rows:
                    continue
                a = {k: float(np.mean([r[k] for r in rows])) for k in keys}
                a['max_entropy'] = rows[0]['max_entropy']
                out[fid] = a
            return out if out else None

        if self._pool is None:
            # Sequential path.
            for i in range(self.n_deltas):
                s = pair_seeds[i]
                for sign, store_r, store_a in (
                    (+1.0, rewards_pos, act_pos),
                    (-1.0, rewards_neg, act_neg),
                ):
                    # Set weights on local policies.
                    for fid in target_species:
                        w = base_weights[fid] + sign * self.sigma * deltas[fid][i]
                        self._set_weights(self.policies[fid], w)
                    # STEP 3: per-world averaging within this delta pair.
                    fit_world = []; act_world = []
                    for m in range(M):
                        fit, samples, act_d = self._evaluate_coevo(
                            target_species, n_eval_ticks, seed=s,
                            obs_mean=obs_mean, obs_var=obs_var,
                            dm_ids_for_norm=dm_ids_for_norm,
                            env_builder=world_builders[m],
                        )
                        fit_world.append(fit); act_world.append(act_d)
                        if samples is not None:
                            s_sum, s_sumsq, s_cnt = samples
                            if agg_sum is None:
                                agg_sum = s_sum.copy(); agg_sumsq = s_sumsq.copy()
                            else:
                                agg_sum += s_sum; agg_sumsq += s_sumsq
                            agg_count += s_cnt
                    store_r[i] = _avg_fit_dicts(fit_world)
                    store_a[i] = _avg_act_dicts(act_world)
            # Restore baseline weights before ARS update.
            for fid in target_species:
                self._set_weights(self.policies[fid], base_weights[fid])
        else:
            # Parallel path.
            obs_pack = None
            if self.obs_normalize:
                obs_pack = {'dm_ids': dm_ids_for_norm,
                            'mean': obs_mean, 'var': obs_var}
            tasks = []
            if M == 1:
                # M=1 legacy: keep tuple-format tasks (byte-identical to pre-STEP-3 path).
                for i in range(self.n_deltas):
                    wd_pos = _build_weights_dict(+1.0, i)
                    tasks.append((list(target_species), wd_pos, n_eval_ticks,
                                  self.alpha, self.beta, pair_seeds[i], obs_pack,
                                  self.entropy_coef, self.argmax_penalty,
                                  self.softmax_temperature, self.integral_reward))
                for i in range(self.n_deltas):
                    wd_neg = _build_weights_dict(-1.0, i)
                    tasks.append((list(target_species), wd_neg, n_eval_ticks,
                                  self.alpha, self.beta, pair_seeds[i], obs_pack,
                                  self.entropy_coef, self.argmax_penalty,
                                  self.softmax_temperature, self.integral_reward))
            else:
                # M>1: dict-tasks with per-world env_builder override. Layout:
                # outer = sign (+, -), then i in [0..n_deltas), then m in [0..M).
                for sign in (+1.0, -1.0):
                    for i in range(self.n_deltas):
                        wd = _build_weights_dict(sign, i)
                        for m in range(M):
                            tasks.append({
                                'fg_list': list(target_species),
                                'weights_dict': wd,
                                'n_ticks': n_eval_ticks,
                                'alpha': self.alpha,
                                'beta': self.beta,
                                'seed': pair_seeds[i],
                                'obs_pack': obs_pack,
                                'entropy_coef': self.entropy_coef,
                                'argmax_penalty': self.argmax_penalty,
                                'softmax_temperature': self.softmax_temperature,
                                'integral_reward': self.integral_reward,
                                'env_builder': world_builders[m],
                            })
            results = self._pool.map(_evaluate_coevo_task, tasks)
            # Drain samples for obs-stats accumulation regardless of M.
            fits_flat = []
            acts_flat = []
            for (fit, samples, act_d) in results:
                fits_flat.append(fit); acts_flat.append(act_d)
                if samples is not None:
                    s_sum, s_sumsq, s_cnt = samples
                    if agg_sum is None:
                        agg_sum = s_sum.copy(); agg_sumsq = s_sumsq.copy()
                    else:
                        agg_sum += s_sum; agg_sumsq += s_sumsq
                    agg_count += s_cnt
            if M == 1:
                # Same layout as before: first n_deltas = +sign, next n_deltas = -sign.
                for k in range(self.n_deltas):
                    rewards_pos[k] = fits_flat[k]
                    act_pos[k] = acts_flat[k]
                    rewards_neg[k] = fits_flat[k + self.n_deltas]
                    act_neg[k] = acts_flat[k + self.n_deltas]
            else:
                n = self.n_deltas
                pos_block = fits_flat[:n * M]
                neg_block = fits_flat[n * M:]
                ap_block = acts_flat[:n * M]
                an_block = acts_flat[n * M:]
                for i in range(n):
                    rewards_pos[i] = _avg_fit_dicts(pos_block[i * M:(i + 1) * M])
                    rewards_neg[i] = _avg_fit_dicts(neg_block[i * M:(i + 1) * M])
                    act_pos[i] = _avg_act_dicts(ap_block[i * M:(i + 1) * M])
                    act_neg[i] = _avg_act_dicts(an_block[i * M:(i + 1) * M])

        # Merge obs-stats (shared across all species in the same rollout).
        if self.obs_normalize and agg_count > 0 and dm_ids_for_norm is not None:
            self._merge_obs_stats(dm_ids_for_norm, agg_sum, agg_sumsq, agg_count)

        # ARS update per species with its own reward vector.
        out_means = {}
        for fid in target_species:
            r_pos = np.array([rewards_pos[i][fid] for i in range(self.n_deltas)], dtype=np.float64)
            r_neg = np.array([rewards_neg[i][fid] for i in range(self.n_deltas)], dtype=np.float64)

            # Z-score + entropy/argmax bonus per species (same logic as train_step).
            if self.entropy_coef != 0.0 or self.argmax_penalty != 0.0:
                eco_all = np.concatenate([r_pos, r_neg])
                eco_mean = float(np.mean(eco_all))
                eco_std = float(np.std(eco_all)) + 1e-8

                def _mod(r_arr, act_arr):
                    out = np.empty_like(r_arr, dtype=np.float64)
                    for k in range(len(r_arr)):
                        rn = (float(r_arr[k]) - eco_mean) / eco_std
                        a = act_arr[k].get(fid) if act_arr[k] is not None else None
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

                r_pos = _mod(r_pos, act_pos)
                r_neg = _mod(r_neg, act_neg)

            # ARS update (top-b).
            pair_scores = np.maximum(r_pos, r_neg)
            b = self.top_deltas
            if b < self.n_deltas:
                top_idx = np.argsort(-pair_scores)[:b]
            else:
                top_idx = np.arange(self.n_deltas)
            sel_rewards = np.concatenate([r_pos[top_idx], r_neg[top_idx]])
            sigma_f = np.std(sel_rewards) + 1e-8

            w0 = base_weights[fid].numpy()
            step = np.zeros_like(w0)
            for i in top_idx:
                step += (r_pos[i] - r_neg[i]) * deltas[fid][i].numpy()
            new_w = w0 + (self.lr / (b * sigma_f)) * step
            self._set_weights(self.policies[fid], torch.from_numpy(new_w))

            r_mean = float(np.mean(np.concatenate([r_pos, r_neg])))
            r_std = float(np.std(np.concatenate([r_pos, r_neg])))
            rel = r_std / (abs(r_mean) + 1e-12)
            out_means[fid] = r_mean

            # Per-species action entropy from this iteration.
            act_str = ""
            H_sum = 0.0; mv_sum = 0.0; rs_sum = 0.0; et_sum = 0.0; n_act = 0
            H_max_seen = None
            for arr in (act_pos, act_neg):
                for d in arr:
                    if d is None or fid not in d:
                        continue
                    a = d[fid]
                    H_sum += a['entropy']; mv_sum += a['move_frac']
                    rs_sum += a['rest_frac']; et_sum += a['eat_frac']
                    H_max_seen = a['max_entropy']; n_act += 1
            if n_act > 0:
                Hm = H_max_seen or 1.0
                Hmean = H_sum / n_act
                mv = mv_sum / n_act; rs = rs_sum / n_act; et = et_sum / n_act
                act_str = (f" H_act={Hmean:.3f}/{Hm:.3f} ({Hmean/Hm:.0%})"
                           f" mv/rs/et={mv:.2f}/{rs:.2f}/{et:.2f}")

            print(f"  [coevo {fid}] r_mean={r_mean:+.4e} r_std={r_std:.4e} "
                  f"rel_std={rel:.3%} sigma_f={sigma_f:.4e}" + act_str)

        return out_means
