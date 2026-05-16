import torch
import numpy as np
import multiprocessing as mp
from lib.runners.policy import PolicyNetwork
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.parallel_worker import _worker_init, _evaluate_task

class ARSTrainer:
    def __init__(self, env_builder, policy_params, sigma=0.1, lr=0.02, n_deltas=8, n_workers=1,
                 alpha=1.0, beta=1.0):
        self.env_builder = env_builder
        self.policy_params = policy_params
        self.sigma = sigma
        self.lr = lr
        self.n_deltas = n_deltas
        self.n_workers = max(1, int(n_workers))
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Initialize policies (parent-side; workers hold their own copies)
        self.policies = {}
        for fg_id, (in_dim, out_dim) in policy_params.items():
            self.policies[fg_id] = PolicyNetwork(in_dim, out_dim)

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
            # terminate() is fast and won't hang if workers are still busy or
            # if the parent is shutting down after Ctrl+C. close()+join() can
            # block indefinitely on a busy pool.
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

    def train_step(self, fg_to_train, n_eval_ticks=2):
        policy = self.policies[fg_to_train]
        weights = self._get_weights(policy)

        deltas = [torch.randn_like(weights) for _ in range(self.n_deltas)]

        # Common Random Numbers: one seed per delta pair shared by +/- rollouts.
        # Different seeds across pairs preserve variance over the batch.
        pair_seeds = [int(np.random.randint(1, 2**31 - 1)) for _ in range(self.n_deltas)]

        if self._pool is None:
            # Sequential path (preserves original behaviour, now with CRN).
            rewards_pos = []
            rewards_neg = []
            for i, delta in enumerate(deltas):
                s = pair_seeds[i]
                self._set_weights(policy, weights + self.sigma * delta)
                rewards_pos.append(self._evaluate(fg_to_train, n_eval_ticks, seed=s))
                self._set_weights(policy, weights - self.sigma * delta)
                rewards_neg.append(self._evaluate(fg_to_train, n_eval_ticks, seed=s))
            rewards_pos = np.array(rewards_pos)
            rewards_neg = np.array(rewards_neg)
        else:
            # Parallel path: snapshot all non-trained policy weights once,
            # then submit 2*n_deltas tasks differing only in fg_to_train weights.
            base_weights = {
                fg_id: self._get_weights(p).numpy().copy()
                for fg_id, p in self.policies.items()
            }
            base_train = weights.numpy()

            tasks = []
            for i, delta in enumerate(deltas):
                d = delta.numpy()
                w_pos = dict(base_weights)
                w_pos[fg_to_train] = base_train + self.sigma * d
                tasks.append((fg_to_train, w_pos, n_eval_ticks, self.alpha, self.beta, pair_seeds[i]))
            for i, delta in enumerate(deltas):
                d = delta.numpy()
                w_neg = dict(base_weights)
                w_neg[fg_to_train] = base_train - self.sigma * d
                tasks.append((fg_to_train, w_neg, n_eval_ticks, self.alpha, self.beta, pair_seeds[i]))

            results = self._pool.map(_evaluate_task, tasks)
            rewards_pos = np.array(results[:self.n_deltas])
            rewards_neg = np.array(results[self.n_deltas:])

            # Restore parent-side trained policy to base weights.
            self._set_weights(policy, weights)

        # ARS update
        sigma_f = np.std(np.concatenate([rewards_pos, rewards_neg])) + 1e-8
        step = np.zeros_like(weights.numpy())
        for i in range(self.n_deltas):
            step += (rewards_pos[i] - rewards_neg[i]) * deltas[i].numpy()
        new_weights = weights.numpy() + (self.lr / (self.n_deltas * sigma_f)) * step
        self._set_weights(policy, torch.from_numpy(new_weights))

        return np.mean(rewards_pos + rewards_neg)

    def _get_weights(self, policy):
        return torch.cat([p.data.view(-1) for p in policy.parameters()])

    def _set_weights(self, policy, weights):
        idx = 0
        for p in policy.parameters():
            p_size = p.data.numel()
            p.data.copy_(weights[idx:idx + p_size].view(p.size()))
            idx += p_size

    def _evaluate(self, fg_id, n_ticks, seed=None):
        env = self.env_builder(seed=seed) if seed is not None else self.env_builder()
        env.policies = self.policies

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

        return self.alpha * delta_b + self.beta * delta_r
