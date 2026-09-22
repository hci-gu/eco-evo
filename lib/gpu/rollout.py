"""Persistent rollout buffers and optional compiled/CUDA-graph execution."""

import torch

from lib.environments.ecosystem_env.source_tracking import LocalRewardConfig
from lib.environments.ecosystem_env import debug_food
from lib.gpu.random import fold_in, uniform
from lib.runners.population_stability import validate_reward
from lib.runners.survival_reward import TensorSurvivalScore, validate_survival_reward


class RolloutRunner:
    def __init__(self, model, bank, candidates, worlds, *, obs_normalize=True,
                 integral_reward=True, legacy_reward=False, alpha=1.0, beta=1.0,
                 survival_bonus=0.0, survival_threshold=0.01,
                 execution="cuda-graph", graph_ticks=32, population_stability=None,
                 local_reward=None, survival_reward=None):
        validate_survival_reward(survival_reward, integral_reward, legacy_reward,
                                 population_stability, local_reward)
        self.survival_reward = survival_reward
        self.survival_score = None
        # ``--local_reward``: the per-cell source-tracked reward, mirrored
        # from the reference implementation in
        # lib/environments/ecosystem_env/source_tracking.py. Every setting
        # is static for the lifetime of the runner, so the tick stays a
        # single fixed-shape graph.
        self.local_reward = LocalRewardConfig.from_dict(local_reward)
        validate_reward(population_stability, integral_reward, legacy_reward,
                        self.local_reward)
        self.population_stability = population_stability
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
        self.food_blob_motion = None
        if model.food_blobs is not None and model.boundary == "torus":
            self.food_blob_motion = debug_food.TorusBlobMotion(
                model.H, model.W, torch.zeros((self.E, 1), dtype=torch.int64, device=model.device),
                model.food_blobs)
            self.state_buffers.extend(self.food_blob_motion.buffers)
        def buffer(name, shape, dtype=torch.float32):
            value = torch.zeros(shape, dtype=dtype, device=model.device)
            setattr(self, name, value)
            self.state_buffers.append(value)
            return value
        for name in ("biomass", "reserve", "hidden"):
            buffer(name, (self.E, model.G, model.C))
        buffer("phase", (self.E, model.G, 1))
        buffer("keys", (self.E,), torch.int64)
        if model.food_blobs is not None:
            for name in ("food_blob_b0", "food_blob_r0"):
                buffer(name, (self.E, model.food_blob_index.numel(), 1))
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
        buffer("failed", (self.E,), torch.bool)
        buffer("valid_ticks", (self.E,), torch.int64)
        buffer("stability_sum", (self.E, model.D), torch.float64)
        buffer("local_sum", (self.E, model.D), torch.float64)
        buffer("local_occupied", (self.E, model.D), torch.float64)
        self.local_min_energy = (
            model.local_min_energy(self.local_reward.min_energy_factor)
            if self.local_reward is not None else None)
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
        if self.survival_reward is not None:
            # Warmup and capture execute extra ticks before restoring state.
            capacity = ticks + self.graph_ticks + 4
            if self.survival_score is None or self.survival_score.prefix.shape[0] <= capacity:
                if self.model.device.type == "cuda":
                    torch.cuda.synchronize(self.model.device)
                self.graphs.clear()
                self.prepared = False
                if self.survival_score is not None:
                    old = {id(v) for v in self.survival_score.buffers}
                    self.state_buffers = [v for v in self.state_buffers if id(v) not in old]
                self.survival_score = TensorSurvivalScore(
                    (self.E, self.model.D), capacity, self.model.device, self.survival_reward)
                self.state_buffers.extend(self.survival_score.buffers)
        self.biomass.copy_(biomass)
        self.reserve.copy_(reserve)
        self.phase.copy_(phase)
        self.keys.copy_(keys)
        if self.model.food_blobs is not None:
            if self.food_blob_motion is not None:
                self.food_blob_motion.reset(keys[:, None])
            self.food_blob_b0.copy_(biomass[:, self.model.food_blob_index].sum(-1, keepdim=True))
            self.food_blob_r0.copy_(reserve[:, self.model.food_blob_index].sum(-1, keepdim=True))
            b, r = debug_food.apply_tensor(self.model, biomass, reserve, 0, keys,
                                           (self.food_blob_b0, self.food_blob_r0), self.food_blob_motion)
            self.biomass.copy_(b)
            self.reserve.copy_(r)
        self.tick.zero_()
        self.horizon.fill_(ticks)
        self.temperature.copy_(temperature)
        self.obs_mean.copy_(mean)
        self.obs_var.copy_(var)
        for value in (self.hidden, self.log_energy, self.biomass_sum, self.reserve_sum,
                      self.obs_sum, self.obs_sumsq, self.action_sum, self.active_ticks):
            value.zero_()
        self.b0.copy_(biomass[:, self.model.dm_index].sum(-1).double())
        if self.survival_score is not None:
            self.survival_score.reset(self.b0)
        self.r0.copy_(reserve[:, self.model.dm_index].sum(-1).double())
        self.e0.copy_(self.b0 * self.model.energy_content[:, self.model.dm_index].double() + self.r0)
        self.epsilon.copy_((1e-6 * self.e0).clamp_min(1e-9))
        self.survived.fill_(ticks)
        for value in (self.failed, self.valid_ticks, self.stability_sum,
                      self.local_sum, self.local_occupied):
            value.zero_()

    @property
    def occupancy(self):
        """Mean participating cells per tick, per world and DM (diagnostic)."""
        return self.local_occupied / self.horizon.clamp_min(1)

    @property
    def observed_ticks(self):
        """Ticks each world contributed observations for.

        Derived rather than accumulated: a world is alive at the start of a
        tick until it breaches, so it observes its valid ticks plus the single
        failing tick. Counting it inside ``_tick`` would have to read
        ``failed`` in the same traced graph that updates it, and a compiled
        graph then reads the already updated flag and loses the failing tick.
        """
        if self.population_stability is None:
            return self.horizon.reshape(1).expand(self.E)
        return self.valid_ticks + self.failed.long()

    def _tick(self):
        m = self.model
        if self.population_stability is not None:
            alive = ~self.failed
        obs = m.observations(self.biomass, self.reserve, self.hidden)
        if self.normalize:
            raw = obs.double()
            if self.population_stability is not None:
                raw = torch.where(alive[:, None, None, None], raw, 0.0)
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
        if self.population_stability is not None:
            statistics = torch.where(alive[:, None, None], statistics, 0.0)
            active = torch.where(alive[:, None], active, 0)
        self.action_sum.add_(statistics)
        self.active_ticks.add_(active)
        if m.has_seeding:
            noise = uniform(fold_in(self.keys, self.tick), (m.G, m.C), 601)
            multiplier = torch.pow(10.0, 2.0 * noise - 1.0)
        else:
            multiplier = torch.zeros_like(self.biomass)
        current_args = {"current_keys": self.keys} if m.currents is not None or m.food_blobs is not None else {}
        if m.food_blobs is not None:
            current_args["food_blob_totals"] = (self.food_blob_b0, self.food_blob_r0)
            if self.food_blob_motion is not None:
                current_args["food_blob_motion"] = self.food_blob_motion
        if self.local_reward is not None:
            b, r, hidden, _, _, local = m.step(self.biomass, self.reserve, actions,
                                               self.tick, self.phase, multiplier,
                                               track_source=True, **current_args)
            self._accumulate_local_reward(*local)
        else:
            b, r, hidden, _, _ = m.step(self.biomass, self.reserve, actions,
                                      self.tick, self.phase, multiplier, **current_args)
        totals_b, totals_r = b.sum(-1).double(), r.sum(-1).double()
        bd, rd = totals_b[:, m.dm_index], totals_r[:, m.dm_index]
        energy = bd * m.energy_content[:, m.dm_index].double() + rd
        if self.survival_score is not None:
            self.survival_score.step(bd, rd, self.b0, m.max_reserve[:, m.dm_index, 0], self.tick)
        if self.population_stability is not None:
            c = self.population_stability
            present = self.b0 > 0
            ratio = torch.where(present, bd / self.b0.clamp_min(1e-30), 1.0)
            breach = (present & ((ratio < c.lower) | (ratio >= c.upper))).any(-1)
            breach = breach | (~torch.isfinite(energy) | ~torch.isfinite(bd) | (energy < 0)).any(-1)
            valid = alive & ~breach
            low = ((c.warning_lower - ratio) / (c.warning_lower - c.lower)).clamp(0, 1)
            high = ((ratio - c.warning_upper) / (c.upper - c.warning_upper)).clamp(0, 1)
            warning = torch.where(present, torch.maximum(low, high), 0.0).amax(-1)
            base = torch.log((energy + self.epsilon) / (self.e0 + self.epsilon))
            tick_reward = base.clamp(c.energy_floor, c.energy_cap) - warning[:, None]
            self.stability_sum.add_(torch.where(valid[:, None], tick_reward, 0.0))
            self.valid_ticks.add_(valid)
            # Absorb failed worlds: no subsequent state, normalization, or
            # action-statistic changes. Fixed GPU graph shapes are retained.
            b = torch.where(alive[:, None, None], b, self.biomass)
            r = torch.where(alive[:, None, None], r, self.reserve)
            hidden = torch.where(alive[:, None, None], hidden, self.hidden)
            totals_b, totals_r = b.sum(-1).double(), r.sum(-1).double()
            # Update the flag only after the last use of ``alive``: a compiled
            # graph may fuse ``~failed`` into a later kernel, which would then
            # read the already mutated buffer and drop the failing tick from
            # ``observed_ticks`` (the observation-normalisation sample count).
            self.failed.logical_or_(breach)
        self.log_energy.add_(torch.log((energy + self.epsilon) / (self.e0 + self.epsilon)))
        self.biomass_sum.add_(totals_b)
        self.reserve_sum.add_(totals_r)
        died = (self.survived == self.horizon) & (bd < self.survival_threshold * self.b0)
        self.survived.copy_(torch.where(died, self.tick, self.survived))
        self.biomass.copy_(b)
        self.reserve.copy_(r)
        self.hidden.copy_(hidden)
        self.tick.add_(1)

    def _accumulate_local_reward(self, start, tracked, frac_in):
        """Aggregate this tick's per-cell ratios into ``local_sum``.

        ``A`` is scaled by ``frac_in`` so outflow the tracker cannot
        follow (across the grid border) leaves the denominator instead of
        counting as a loss, and only cells that held a viable population
        at the start of the tick take part. Float64 throughout: the raw
        quotient of two float32 energies is the one place where the
        reference and this engine would otherwise disagree visibly.
        """
        config = self.local_reward
        a = start.double()
        a_eff = a * frac_in.double()
        active = (a >= self.local_min_energy) & (a_eff > 0.0)
        ratio = torch.where(active, tracked.double() / a_eff.clamp_min(1e-300), 1.0)
        ratio = ratio.clamp(config.clip_lo, config.clip_hi)
        term = ratio.log() if config.metric == "log" else ratio
        if config.theta == 0.0:
            weight = active.double()
        else:
            weight = torch.where(active, a_eff.clamp_min(0.0).pow(config.theta), 0.0)
        weighted = (weight * term).sum(-1)
        if config.norm == "mean":
            total = weight.sum(-1)
            weighted = torch.where(total > 0.0, weighted / total.clamp_min(1e-300), 0.0)
        elif config.norm == "grid":
            # Constant denominator (the cell count), so the cell count is
            # orthogonal to the fitness instead of being a gradient in
            # either direction. Static python scalar: the tick still
            # traces as one fixed-shape graph.
            weighted = weighted / float(self.model.C)
        self.local_sum.add_(weighted)
        self.local_occupied.add_(active.double().sum(-1))

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
        if self.survival_score is not None:
            return (self.survival_score.results(ticks),
                    self.action_sum / self.active_ticks.clamp_min(1)[:, :, None])
        if self.local_reward is not None:
            # Already a per-tick aggregate, so integral_reward does not
            # apply; the fitness is its mean over the rollout.
            reward = self.local_sum / ticks
            action = self.action_sum / self.active_ticks.clamp_min(1)[:, :, None]
            return reward, action
        if self.population_stability is not None:
            tail = (ticks - self.valid_ticks)[:, None]
            reward = (self.stability_sum + tail * self.population_stability.failure_reward) / ticks
            action = self.action_sum / self.active_ticks.clamp_min(1)[:, :, None]
            return reward, action
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
