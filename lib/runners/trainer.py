import torch
import numpy as np
import multiprocessing as mp
from lib.runners.policy import PolicyNetwork
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.parallel_worker import _worker_init, _evaluate_task

class ARSTrainer:
    def __init__(self, env_builder, policy_params, sigma=0.1, lr=0.02, n_deltas=8, n_workers=1):
        self.env_builder = env_builder
        self.policy_params = policy_params
        self.sigma = sigma
        self.lr = lr
        self.n_deltas = n_deltas
        self.n_workers = max(1, int(n_workers))

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
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def train_step(self, fg_to_train, n_rollouts=2):
        policy = self.policies[fg_to_train]
        weights = self._get_weights(policy)

        deltas = [torch.randn_like(weights) for _ in range(self.n_deltas)]

        if self._pool is None:
            # Sequential path (preserves original behaviour)
            rewards_pos = []
            rewards_neg = []
            for delta in deltas:
                self._set_weights(policy, weights + self.sigma * delta)
                rewards_pos.append(self._evaluate(fg_to_train, n_rollouts))
                self._set_weights(policy, weights - self.sigma * delta)
                rewards_neg.append(self._evaluate(fg_to_train, n_rollouts))
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
            for delta in deltas:
                d = delta.numpy()
                w_pos = dict(base_weights)
                w_pos[fg_to_train] = base_train + self.sigma * d
                tasks.append((fg_to_train, w_pos, n_rollouts))
            for delta in deltas:
                d = delta.numpy()
                w_neg = dict(base_weights)
                w_neg[fg_to_train] = base_train - self.sigma * d
                tasks.append((fg_to_train, w_neg, n_rollouts))

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

    def _evaluate(self, fg_id, n_ticks):
        env = self.env_builder()
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

        return delta_b + delta_r
