"""Persistent rollout buffers and optional compiled/CUDA-graph execution."""

import torch

from lib.gpu.random import fold_in, uniform


class RolloutRunner:
    def __init__(self, model, bank, candidates, worlds, *, obs_normalize=True,
                 integral_reward=True, legacy_reward=False, alpha=1.0, beta=1.0,
                 survival_bonus=0.0, survival_threshold=0.01,
                 execution="cuda-graph", graph_ticks=32):
        if execution not in ("eager", "compile", "cuda-graph", "compile-graph"):
            raise ValueError("Unknown execution mode: " + execution)
        if "graph" in execution and model.device.type != "cuda":
            raise ValueError("CUDA Graph execution requires a CUDA device; use eager on CPU")
        if graph_ticks < 1:
            raise ValueError("graph_ticks must be positive")
        self.model, self.bank = model, bank
        self.candidates, self.worlds = candidates, worlds
        self.E = candidates * worlds
        self.normalize = obs_normalize
        self.integral_reward, self.legacy_reward = integral_reward, legacy_reward
        self.alpha, self.beta = alpha, beta
        self.survival_bonus, self.survival_threshold = survival_bonus, survival_threshold
        self.execution, self.graph_ticks = execution, graph_ticks
        self.graphs = {}
        self.prepared = False
        self.state_buffers = []
        def buffer(name, shape, dtype=torch.float32):
            value = torch.zeros(shape, dtype=dtype, device=model.device)
            setattr(self, name, value)
            self.state_buffers.append(value)
            return value
        for name in ("biomass", "reserve", "hidden"):
            buffer(name, (self.E, model.G, model.C))
        buffer("phase", (self.E, model.G, 1))
        buffer("keys", (self.E,), torch.int64)
        buffer("tick", (), torch.int64)
        buffer("horizon", (), torch.int64)
        buffer("temperature", ())
        buffer("obs_mean", (model.D, model.F))
        buffer("obs_var", (model.D, model.F))
        self.obs_var.fill_(1)
        for name in ("b0", "r0", "e0", "epsilon", "log_energy", "survived"):
            buffer(name, (self.E, model.D), torch.float64)
        for name in ("biomass_sum", "reserve_sum"):
            buffer(name, (self.E, model.G), torch.float64)
        for name in ("obs_sum", "obs_sumsq"):
            buffer(name, (self.E, model.D, model.F), torch.float64)
        buffer("action_sum", (self.E, model.D, 4), torch.float64)
        buffer("active_ticks", (self.E, model.D), torch.float64)
        self.weights, self.biases = bank.pack([
            w[None].expand(candidates, -1).clone() for w in bank.flat_weights()])
        self._execute_tick = self._tick
        if execution.startswith("compile"):
            # Graph ownership is explicit here. Avoid nesting Inductor's
            # automatic graph pools inside the manual CUDA captures below.
            self._execute_tick = torch.compile(self._tick, fullgraph=True, dynamic=False,
                                               options={"triton.cudagraphs": False})

    def set_weights(self, packed):
        for destination, source in zip(self.weights + self.biases, packed[0] + packed[1]):
            destination.copy_(source)

    def reset(self, biomass, reserve, phase, keys, ticks, mean, var, temperature):
        if ticks < 1:
            raise ValueError("Rollouts must contain at least one tick")
        self.biomass.copy_(biomass)
        self.reserve.copy_(reserve)
        self.phase.copy_(phase)
        self.keys.copy_(keys)
        self.tick.zero_()
        self.horizon.fill_(ticks)
        self.temperature.copy_(temperature)
        self.obs_mean.copy_(mean)
        self.obs_var.copy_(var)
        for value in (self.hidden, self.log_energy, self.biomass_sum, self.reserve_sum,
                      self.obs_sum, self.obs_sumsq, self.action_sum, self.active_ticks):
            value.zero_()
        self.b0.copy_(biomass[:, self.model.dm_index].sum(-1).double())
        self.r0.copy_(reserve[:, self.model.dm_index].sum(-1).double())
        self.e0.copy_(self.b0 * self.model.energy_content[:, self.model.dm_index].double() + self.r0)
        self.epsilon.copy_((1e-6 * self.e0).clamp_min(1e-9))
        self.survived.fill_(ticks)

    def _tick(self):
        m = self.model
        obs = m.observations(self.biomass, self.reserve, self.hidden)
        if self.normalize:
            raw = obs.double()
            self.obs_sum.add_(raw.sum(2))
            self.obs_sumsq.add_((raw * raw).sum(2))
            obs = ((obs - self.obs_mean[None, :, None]) /
                   self.obs_var.clamp_min(1e-2).sqrt()[None, :, None]).clamp(-10, 10)
        grouped = obs.reshape(self.candidates, self.worlds, m.D, m.C, m.F)
        grouped = grouped.permute(0, 2, 1, 3, 4).reshape(self.candidates, m.D, self.worlds * m.C, m.F)
        logits = self.bank.forward(grouped, self.weights, self.biases)
        logits = logits.reshape(self.candidates, m.D, self.worlds, m.C, m.A)
        logits = logits.permute(0, 2, 1, 3, 4).reshape(self.E, m.D, m.C, m.A)
        actions = m.action_probabilities(logits, self.biomass, self.temperature)
        statistics, active = m.action_statistics(self.biomass, actions)
        self.action_sum.add_(statistics)
        self.active_ticks.add_(active)
        if m.has_seeding:
            noise = uniform(fold_in(self.keys, self.tick), (m.G, m.C), 601)
            multiplier = torch.pow(10.0, 2.0 * noise - 1.0)
        else:
            multiplier = torch.zeros_like(self.biomass)
        b, r, hidden, _, _ = m.step(self.biomass, self.reserve, actions,
                                  self.tick, self.phase, multiplier)
        totals_b, totals_r = b.sum(-1).double(), r.sum(-1).double()
        bd, rd = totals_b[:, m.dm_index], totals_r[:, m.dm_index]
        energy = bd * m.energy_content[:, m.dm_index].double() + rd
        self.log_energy.add_(torch.log((energy + self.epsilon) / (self.e0 + self.epsilon)))
        self.biomass_sum.add_(totals_b)
        self.reserve_sum.add_(totals_r)
        died = (self.survived == self.horizon) & (bd < self.survival_threshold * self.b0)
        self.survived.copy_(torch.where(died, self.tick, self.survived))
        self.biomass.copy_(b)
        self.reserve.copy_(r)
        self.hidden.copy_(hidden)
        self.tick.add_(1)

    @torch.no_grad()
    def prepare(self):
        if self.prepared:
            return
        snapshot = [v.clone() for v in self.state_buffers]
        if self.model.device.type == "cuda":
            with torch.cuda.device(self.model.device):
                stream = torch.cuda.Stream(device=self.model.device)
                stream.wait_stream(torch.cuda.current_stream(self.model.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._execute_tick()
                torch.cuda.current_stream(self.model.device).wait_stream(stream)
                if "graph" in self.execution:
                    for length in sorted({1, self.graph_ticks}):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            for _ in range(length):
                                self._execute_tick()
                        self.graphs[length] = graph
                torch.cuda.current_stream(self.model.device).wait_stream(stream)
        elif self.execution == "compile":
            self._execute_tick()
        for value, saved in zip(self.state_buffers, snapshot):
            value.copy_(saved)
        self.prepared = True

    @torch.no_grad()
    def run(self, ticks):
        self.prepare()
        if self.graphs:
            blocks, remainder = divmod(ticks, self.graph_ticks)
            for _ in range(blocks):
                self.graphs[self.graph_ticks].replay()
            for _ in range(remainder):
                self.graphs[1].replay()
        else:
            for _ in range(ticks):
                self._execute_tick()

    def results(self, ticks):
        m = self.model
        if self.integral_reward:
            b = self.biomass_sum[:, m.dm_index] / ticks
            r = self.reserve_sum[:, m.dm_index] / ticks
            log_energy = self.log_energy / ticks
        else:
            b = self.biomass[:, m.dm_index].sum(-1).double()
            r = self.reserve[:, m.dm_index].sum(-1).double()
            energy = b * m.energy_content[:, m.dm_index].double() + r
            log_energy = torch.log((energy + self.epsilon) / (self.e0 + self.epsilon))
        if self.legacy_reward:
            eps_b, eps_r = (1e-6 * self.b0).clamp_min(1e-9), (1e-6 * self.r0).clamp_min(1e-9)
            reward = (self.alpha * torch.log((b + eps_b) / (self.b0 + eps_b)) +
                      self.beta * torch.log((r + eps_r) / (self.r0 + eps_r)) +
                      self.survival_bonus * self.survived / ticks)
        else:
            reward = log_energy
        action = self.action_sum / self.active_ticks.clamp_min(1)[:, :, None]
        return reward, action

    def snapshot(self, index=0):
        """Explicit, occasional host boundary; never called during training."""
        return {"biomass": self.biomass[index].reshape(self.model.G, self.model.H, self.model.W).cpu().numpy().copy(),
                "reserve": self.reserve[index].reshape(self.model.G, self.model.H, self.model.W).cpu().numpy().copy()}

    def close(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        self.graphs.clear()
        # Break the bound-method/compiled-closure reference cycle so old graph
        # pools can be released before allocating a larger world batch.
        self._execute_tick = None
