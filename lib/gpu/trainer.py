"""ARS co-evolution with GPU world generation, rollouts, and weight updates."""

import math
from pathlib import Path

import torch

from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.random import fold_in, normal
from lib.gpu.rollout import RolloutRunner
from lib.gpu.spawn import WorldSpawner


def ars_update(weights, deltas, positive, negative, lr, top_deltas):
    """One species' reference ARS-V2 update; all arguments stay on device."""
    if top_deltas == deltas.shape[0]:
        index = torch.arange(top_deltas, device=weights.device)
    else:
        index = torch.maximum(positive, negative).argsort(dim=0, descending=True, stable=True)[:top_deltas]
    selected = torch.cat((positive[index], negative[index]))
    sigma_f = selected.std(correction=0) + 1e-8
    difference = (positive[index] - negative[index]).to(deltas.dtype)
    step = (difference[:, None] * deltas[index]).sum(0)
    updated = weights + ((lr / (top_deltas * sigma_f)) * step).to(weights.dtype)
    return updated, sigma_f


class TensorARSTrainer:
    def __init__(self, spec, device="cuda", *, n_deltas=10, worlds=1,
                 top_deltas=None, sigma=0.1, lr=0.03, seed=0,
                 hidden_dim=30, hidden_layers=2, activation="sig",
                 uniform_bias_init=False, obs_normalize=True,
                 integral_reward=True, legacy_reward=False, alpha=1.0, beta=1.0,
                 survival_bonus=0.0, survival_threshold=0.01,
                 entropy_coef=0.0, argmax_penalty=0.0,
                 execution="cuda-graph", graph_ticks=32, pairs_per_batch=None):
        if n_deltas < 1 or worlds < 1:
            raise ValueError("n_deltas and worlds must be positive")
        if sigma <= 0 or not math.isfinite(sigma) or lr < 0 or not math.isfinite(lr):
            raise ValueError("sigma must be positive and lr nonnegative, both finite")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build on the NVIDIA machine; use --device cpu --execution eager for validation.")
        if device.type not in ("cuda", "cpu"):
            raise ValueError("This backend supports CUDA and CPU validation")
        self.spec = spec
        self.model = TensorEcosystem(spec.env, device)
        self.bank = PolicyBank(self.model, hidden_dim, hidden_layers, activation,
                               uniform_bias_init, seed)
        self.theta = self.bank.flat_weights()
        self.spawner = WorldSpawner(spec, self.model)
        self.n_deltas, self.worlds = int(n_deltas), int(worlds)
        self.top_deltas = max(1, n_deltas // 2) if top_deltas is None else int(top_deltas)
        if not 1 <= self.top_deltas <= n_deltas:
            raise ValueError("top_deltas must lie between 1 and n_deltas")
        self.pairs_per_batch = n_deltas if pairs_per_batch is None else int(pairs_per_batch)
        if self.pairs_per_batch < 1:
            raise ValueError("pairs_per_batch must be positive")
        self.pairs_per_batch = min(self.pairs_per_batch, n_deltas)
        self.sigma, self.lr = float(sigma), float(lr)
        self.entropy_coef, self.argmax_penalty = float(entropy_coef), float(argmax_penalty)
        self.obs_normalize = bool(obs_normalize)
        self.seed = self.model.tensor(seed, torch.int64)
        self.iteration = self.model.tensor(0, torch.int64)
        self.temperature = self.model.tensor(1.0)
        self.iterations_completed = 0
        self.last_world_epoch = None
        self.obs_mean = torch.zeros((self.model.D, self.model.F), dtype=torch.float64, device=device)
        self.obs_var = torch.ones_like(self.obs_mean)
        self.obs_count = torch.zeros(self.model.D, dtype=torch.int64, device=device)
        self.runner_options = dict(obs_normalize=obs_normalize, integral_reward=integral_reward,
                                   legacy_reward=legacy_reward, alpha=alpha, beta=beta,
                                   survival_bonus=survival_bonus, survival_threshold=survival_threshold,
                                   execution=execution, graph_ticks=graph_ticks)
        self.architecture = dict(hidden_dim=hidden_dim, hidden_layers=hidden_layers,
                                 activation=self.bank.activation)
        self.rewards = torch.zeros((2, n_deltas, self.model.D), dtype=torch.float64, device=device)
        self.actions = torch.zeros((*self.rewards.shape, 4), dtype=torch.float64, device=device)
        self.last_metrics = {}
        self._allocate_worlds()

    def _allocate_worlds(self):
        if hasattr(self, "runner"):
            self.runner.close()
            del self.runner
        m = self.model
        self.world_index = torch.arange(self.worlds, device=m.device, dtype=torch.int64)
        self.world_biomass = torch.zeros((self.worlds, m.G, m.C), device=m.device)
        self.pair_index = torch.arange(self.n_deltas, device=m.device, dtype=torch.int64)
        self.runner = RolloutRunner(m, self.bank, 2 * self.pairs_per_batch, self.worlds, **self.runner_options)
        self.chunks = []
        for start in range(0, self.n_deltas, self.pairs_per_batch):
            count = min(self.pairs_per_batch, self.n_deltas - start)
            pairs = torch.arange(start, start + self.pairs_per_batch, device=m.device).clamp_max(self.n_deltas - 1)
            candidates = torch.cat((pairs, pairs + self.n_deltas))
            self.chunks.append((start, count, pairs, candidates))
        self.last_world_epoch = None

    def set_world_count(self, worlds):
        if worlds < 1:
            raise ValueError("worlds must be positive")
        if worlds != self.worlds:
            self.worlds = int(worlds)
            self._allocate_worlds()

    def set_temperature(self, temperature):
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("Temperature must be positive and finite")
        self.temperature.fill_(temperature)

    def refresh_worlds(self, epoch=None):
        if epoch is not None and epoch == self.last_world_epoch:
            return
        epoch_key = self.iteration if epoch is None else epoch
        keys = fold_in(fold_in(self.seed.expand(self.worlds), 20001), self.world_index)
        keys = fold_in(keys, epoch_key)
        self.world_biomass.copy_(self.spawner.biomass(keys))
        self.last_world_epoch = epoch

    def make_deltas(self):
        keys = fold_in(fold_in(self.seed.expand(self.n_deltas), self.iteration), self.pair_index)
        return [normal(keys, (w.numel(),), 30001 + d) for d, w in enumerate(self.theta)]

    def close(self):
        self.runner.close()

    def _merge_observations(self, sums, squares, count):
        old_count = self.obs_count[:, None]
        n = old_count + count
        batch_mean = sums / count
        batch_m2 = (squares - count * batch_mean.square()).clamp_min(0)
        delta = batch_mean - self.obs_mean
        mean = self.obs_mean + delta * (count / n)
        m2 = self.obs_var * old_count + batch_m2 + delta.square() * (old_count * count / n)
        self.obs_mean.copy_(mean)
        self.obs_var.copy_((m2 / n).clamp_min(1e-8))
        self.obs_count.add_(count)

    @torch.no_grad()
    def train_step(self, target_species=None, n_eval_ticks=15, world_epoch=None, deltas=None):
        if n_eval_ticks < 1:
            raise ValueError("n_eval_ticks must be positive")
        targets = self.model.dm_ids if target_species is None else tuple(target_species)
        if not targets or len(set(targets)) != len(targets) or any(f not in self.model.dm_ids for f in targets):
            raise ValueError("Targets must be unique active decision makers")
        selected = set(targets)
        self.refresh_worlds(world_epoch)
        deltas = self.make_deltas() if deltas is None else deltas
        if len(deltas) != len(self.theta):
            raise ValueError("Provide one perturbation tensor per decision maker")
        candidates = []
        for fid, w, delta in zip(self.model.dm_ids, self.theta, deltas):
            if delta.shape != (self.n_deltas, w.numel()):
                raise ValueError("Perturbation shape does not match policy")
            if fid in selected:
                candidates.append(torch.cat((w + self.sigma * delta, w - self.sigma * delta)))
            else:
                candidates.append(w[None].expand(2 * self.n_deltas, -1))
        sums, squares = torch.zeros_like(self.obs_mean), torch.zeros_like(self.obs_mean)
        runner, m = self.runner, self.model
        biomass_mean = torch.zeros(m.G, device=m.device, dtype=torch.float64)
        for start, count, pairs, indices in self.chunks:
            runner.set_weights(self.bank.pack([w[indices] for w in candidates]))
            # Both signs see exactly the same initial fields and future noise.
            keys = fold_in(fold_in(self.seed.expand(pairs.numel()), self.iteration), pairs)
            keys = fold_in(keys[:, None], self.world_index[None] + 40001)
            keys = torch.cat((keys, keys), 0).flatten()
            b = self.world_biomass[None].expand(2 * self.pairs_per_batch, -1, -1, -1).reshape(runner.E, m.G, m.C)
            r = self.spawner.reserves(b, keys)
            phase = self.spawner.phases(keys)
            runner.reset(b, r, phase, keys, n_eval_ticks, self.obs_mean, self.obs_var, self.temperature)
            runner.run(n_eval_ticks)
            rewards, actions = runner.results(n_eval_ticks)
            self.rewards[:, start:start + count].copy_(rewards.reshape(2, self.pairs_per_batch, self.worlds, m.D)[:, :count].mean(2))
            self.actions[:, start:start + count].copy_(actions.reshape(2, self.pairs_per_batch, self.worlds, m.D, 4)[:, :count].mean(2))
            if self.obs_normalize:
                sums.add_(runner.obs_sum.reshape(2, self.pairs_per_batch, self.worlds, m.D, m.F)[:, :count].sum((0, 1, 2)))
                squares.add_(runner.obs_sumsq.reshape(2, self.pairs_per_batch, self.worlds, m.D, m.F)[:, :count].sum((0, 1, 2)))
            biomass_mean.add_(runner.biomass_sum.reshape(2, self.pairs_per_batch, self.worlds, m.G)[:, :count].sum((0, 1, 2)))
        if self.obs_normalize:
            self._merge_observations(sums, squares, 2 * self.n_deltas * self.worlds * m.C * n_eval_ticks)
        reward = self.rewards
        if self.entropy_coef != 0 or self.argmax_penalty != 0:
            reward = (reward - reward.mean((0, 1), keepdim=True)) / (reward.std((0, 1), correction=0, keepdim=True) + 1e-8)
            reward = reward + self.entropy_coef * self.actions[..., 0] / math.log(m.A)
            reward = reward - self.argmax_penalty * self.actions[..., 1:].amax(-1)
        sigma_f = torch.zeros(m.D, dtype=torch.float64, device=m.device)
        for d, fid in enumerate(m.dm_ids):
            if fid in selected:
                updated, sd = ars_update(self.theta[d], deltas[d], reward[0, :, d], reward[1, :, d], self.lr, self.top_deltas)
                self.theta[d].copy_(updated)
                sigma_f[d].copy_(sd)
        self.last_metrics = {
            "reward_mean": reward.mean((0, 1)), "reward_std": reward.std((0, 1), correction=0),
            "ecological_reward_mean": self.rewards.mean((0, 1)),
            "actions": self.actions.mean((0, 1)), "sigma_f": sigma_f,
            "mean_biomass": biomass_mean / (2 * self.n_deltas * self.worlds * n_eval_ticks),
        }
        self.iteration.add_(1)
        self.iterations_completed += 1
        return self.last_metrics

    def metrics(self):
        """One compact readback at a requested logging boundary."""
        names = tuple(self.last_metrics)
        sizes = [self.last_metrics[k].numel() for k in names]
        values = torch.cat([self.last_metrics[k].flatten() for k in names]).cpu()
        output, offset = {}, 0
        for name, size in zip(names, sizes):
            output[name] = values[offset:offset + size].reshape(self.last_metrics[name].shape).tolist()
            offset += size
        return output

    def state_dict(self):
        """Explicit checkpoint boundary. Returned tensors own their CPU data."""
        cpu = lambda t: t.detach().cpu().clone()
        return dict(format_version=1, ids=self.model.ids, dm_ids=self.model.dm_ids,
                    grid=(self.model.H, self.model.W), in_dims=self.model.in_dims,
                    architecture=self.architecture, theta=[cpu(w) for w in self.theta],
                    obs_mean=cpu(self.obs_mean), obs_var=cpu(self.obs_var), obs_count=cpu(self.obs_count),
                    seed=cpu(self.seed), iteration=cpu(self.iteration),
                    iterations_completed=self.iterations_completed, temperature=cpu(self.temperature),
                    worlds=self.worlds, last_world_epoch=self.last_world_epoch,
                    world_biomass=cpu(self.world_biomass))

    def load_state_dict(self, state):
        for name, expected in (("ids", self.model.ids), ("dm_ids", self.model.dm_ids),
                               ("grid", (self.model.H, self.model.W)), ("in_dims", self.model.in_dims)):
            if tuple(state[name]) != tuple(expected):
                raise ValueError("Checkpoint has incompatible " + name)
        if state["architecture"] != self.architecture or state.get("format_version") != 1:
            raise ValueError("Checkpoint format or architecture is incompatible")
        if len(state["theta"]) != len(self.theta):
            raise ValueError("Checkpoint policy count is incompatible")
        for target, source in zip(self.theta, state["theta"]):
            if target.shape != source.shape:
                raise ValueError("Checkpoint policy shape is incompatible")
            target.copy_(source)
        for name in ("obs_mean", "obs_var", "obs_count", "seed", "iteration", "temperature"):
            destination = getattr(self, name)
            if destination.shape != state[name].shape:
                raise ValueError("Checkpoint has incompatible " + name)
            destination.copy_(state[name])
        self.iterations_completed = int(state["iterations_completed"])
        if state["worlds"] == self.worlds:
            self.world_biomass.copy_(state["world_biomass"])
            self.last_world_epoch = state["last_world_epoch"]
        else:
            self.last_world_epoch = None

    @torch.no_grad()
    def export_policies(self, directory):
        """Export policy_<species>.pth files consumed by inference.py/train.py."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bank.install(self.theta)
        for d, fid in enumerate(self.model.dm_ids):
            payload = {"state_dict": {k: v.detach().cpu() for k, v in self.bank.policies[fid].state_dict().items()},
                       "architecture": self.architecture}
            if self.obs_normalize:
                payload["obs_stats"] = dict(mean=self.obs_mean[d].float().cpu().numpy(),
                                             var=self.obs_var[d].float().cpu().numpy(),
                                             count=int(self.obs_count[d].cpu()))
            path = directory / ("policy_" + fid + ".pth")
            temporary = path.with_suffix(".pth.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

    @torch.no_grad()
    def import_policies(self, directory):
        """Warm-start from existing CPU policies, with their normalization."""
        loaded = []
        for d, fid in enumerate(self.model.dm_ids):
            path = Path(directory) / ("policy_" + fid + ".pth")
            if not path.exists():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("state_dict", payload)
            self.bank.policies[fid].load_state_dict(state)
            stats = payload.get("obs_stats")
            if stats is not None:
                mean = self.model.tensor(stats["mean"], torch.float64)
                var = self.model.tensor(stats["var"], torch.float64)
                if mean.numel() not in (self.model.in_dims[d], self.model.F) or var.shape != mean.shape:
                    raise ValueError("Observation statistics have incompatible width for " + fid)
                self.obs_mean[d].zero_()
                self.obs_var[d].fill_(1)
                self.obs_mean[d, :mean.numel()].copy_(mean)
                self.obs_var[d, :var.numel()].copy_(var)
                self.obs_count[d].fill_(int(stats["count"]))
            loaded.append(fid)
        for target, source in zip(self.theta, self.bank.flat_weights()):
            target.copy_(source)
        if not loaded:
            raise ValueError("No compatible policy files found in " + str(directory))
        return loaded
