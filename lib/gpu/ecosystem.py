"""Ecological equations on fixed-shape tensors.

State fields are [ecosystem, functional_group, cell]. Actions are
[ecosystem, decision_maker, action, cell]. Only construction/fixture import
touch NumPy; every numerical method supports CPU tensors and CUDA tensors.
"""

import math

import numpy as np
import torch


class TensorEcosystem:
    def __init__(self, env, device="cuda"):
        self.device = torch.device(device)
        self.H, self.W = env.H, env.W
        self.C = self.H * self.W
        self.G, self.D = env.N_all, env.N_dm
        if self.D == 0:
            raise ValueError("At least one decision maker is required")
        self.A, self.F = 5 + self.G, env.max_in_dim
        self.ids, self.dm_ids = tuple(env.global_fg_order), tuple(env.dm_ids)
        self.in_dims = tuple(int(n) for n in env.per_dm_in_dim)
        self.dm_positions = tuple(self.ids.index(fid) for fid in self.dm_ids)
        self.ndm_positions = tuple(i for i in range(self.G) if i not in self.dm_positions)
        self.migration = env.migration
        self.holling = env._has_holling2 or env._has_holling3
        tensor = self.tensor
        self.dm_index = tensor(self.dm_positions, torch.long)
        self.is_dm = tensor([i in self.dm_positions for i in range(self.G)], torch.bool)[None, :, None]
        self.move_mask = tensor(env.move_mask.reshape(4, self.C))
        self.eat_mask = tensor(env.eat_static_mask)
        self.intake_rate = tensor(env.max_intake_mat)[None, :, :, None]
        self.energy_gain = tensor(env.energy_gain_mat)[None, :, :, None]
        self.handling = tensor(env.handling_time_mat)[None, :, :, None]
        self.type3 = tensor(env._type3_pred_mask.reshape(self.D))[None, :, None, None]
        self.visibility = tensor(env._all_visibility_floor)[None, :, None]
        self.velocity = tensor(env.dm_v)[None, :, None, None]
        self.metabolism = tensor(env.dm_resting_metabolism)[None, :, None]
        self.cost_rest = tensor(env.dm_cost_rest)[None, :, None]
        self.cost_eat = tensor(env.dm_cost_eat)[None, :, None]
        self.cost_move = tensor(env.dm_cost_move)[None, :, None]
        self.can_move = tensor(env.dm_v > 0, torch.bool)[None, :, None, None]
        self.min_split = tensor(env.dm_min_split)[None, :, None]
        self.action_indices = torch.arange(self.A, device=self.device)[None, None, :, None]

        groups = [env.fgs[fid] for fid in self.ids]
        def parameter(attr, default=0):
            return tensor([getattr(fg, attr, default) for fg in groups])[None, :, None]
        self.max_reserve = parameter("max_energy_reserve")
        self.growth = parameter("growth_rate")
        self.starve = parameter("starve_rate")
        self.starve = torch.where(self.starve > 0, self.starve, self.growth)
        self.maintenance = parameter("maintenance_level")
        self.seed_rate = parameter("seed_rate").clamp_min(0)
        self.has_seeding = any(fg.seed_rate > 0 and not fg.is_decision_maker for fg in groups)
        self.amplitude = parameter("seasonal_amplitude")
        self.period = parameter("seasonal_period")
        self.seasonal = (self.period > 0) & (self.amplitude != 0)
        self.capacity = tensor([fg.params.get("max_carrying_capacity", 100.0) for fg in groups])[None, :, None]
        self.energy_content = tensor([fg.params.get("energy_content", 0.0) or 0.0 for fg in groups])[None, :]
        self.threshold = tensor([max(0.0, fg.min_split_biomass * fg.extinction_threshold_factor) for fg in groups])[None, :, None]
        self.mortality_keep = tensor([
            max(0.0, 1.0 - max(0.0, fg.natural_mortality))
            if env.apply_natural_mortality and fg.is_decision_maker else 1.0
            for fg in groups
        ])[None, :, None]
        self._build_indices(env)

    def tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _build_indices(self, env):
        yy, xx = np.indices((self.H, self.W))
        coords = ((yy - 1, xx), (yy, xx + 1), (yy + 1, xx), (yy, xx - 1))
        neighbors, valid = [], []
        for y, x in coords:
            valid.append(((y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)).ravel())
            neighbors.append((np.clip(y, 0, self.H - 1) * self.W + np.clip(x, 0, self.W - 1)).ravel())
        self.neighbors = self.tensor(np.array(neighbors), torch.long)
        self.neighbor_valid = self.tensor(np.array(valid))
        self.outside = 1.0 - self.neighbor_valid

        # Pack the CPU observation contract directly into [D, C, F], avoiding
        # per-species tensor stacking and transposes during every tick.
        obs_indices = np.zeros((self.D, self.C, self.F), dtype=np.int64)
        obs_valid = np.zeros_like(obs_indices, dtype=np.float32)
        cells = np.arange(self.C)
        for d, own in enumerate(self.dm_positions):
            others = list(env.obs_others_idx[d])
            channels = [own, 2 * self.G + own] + [self.G + j for j in others]
            for k, channel in enumerate(channels):
                obs_indices[d, :, k] = channel * self.C + cells
                obs_valid[d, :, k] = 1.0
            offset = len(channels)
            for direction in range(4):
                for channel in [own] + [self.G + j for j in others]:
                    obs_indices[d, :, offset] = channel * self.C + neighbors[direction]
                    obs_valid[d, :, offset] = valid[direction]
                    offset += 1
        self.obs_indices = self.tensor(obs_indices, torch.long)
        self.obs_valid = self.tensor(obs_valid)

        # Match the reference's edge tie order, including NumPy's argsort
        # behavior. This is immutable metadata, computed only once.
        edge_weights = env._edge_imm_weights.ravel()
        edge_indices = np.flatnonzero(edge_weights > 0)
        order = np.argsort(-edge_weights[edge_indices])
        edge_indices = edge_indices[order]
        self.edge_indices = self.tensor(edge_indices, torch.long)
        self.edge_weights = self.tensor(edge_weights[edge_indices])
        self.edge_cumsum = self.edge_weights.cumsum(0)
        self.edge_ranks = torch.arange(1, len(edge_indices) + 1, device=self.device)

    def import_state(self, environments):
        """Explicit host boundary for reference fixtures or initial uploads."""
        b = np.stack([np.stack([e.fgs[f].biomass for f in self.ids]) for e in environments])
        r = np.stack([np.stack([e.fgs[f].energy_reserve for f in self.ids]) for e in environments])
        h = np.stack([e.prev_hidden_frac for e in environments])
        phase = np.array([[e._season_phase[f] for f in self.ids] for e in environments])
        return (self.tensor(b).flatten(2), self.tensor(r).flatten(2),
                self.tensor(h).flatten(2), self.tensor(phase)[:, :, None])

    def energy_level(self, biomass, reserve):
        safe_b = torch.where(biomass > 1e-9, biomass, 1.0)
        safe_max = torch.where(self.max_reserve != 0, self.max_reserve, 1.0)
        return torch.where((biomass > 1e-9) & (self.max_reserve != 0),
                           reserve / safe_b / safe_max, 0.0)

    def observations(self, biomass, reserve, hidden):
        visible = biomass * (1.0 - hidden * (1.0 - self.visibility))
        bank = torch.cat((biomass, visible, self.energy_level(biomass, reserve)), dim=1)
        return bank.flatten(1)[:, self.obs_indices] * self.obs_valid

    def action_mask(self, biomass):
        batch = biomass.shape[0]
        move = (self.move_mask[None, None] > 0) & self.can_move
        move = move.expand(batch, self.D, 4, self.C)
        rest = torch.ones((batch, self.D, 1, self.C), device=self.device, dtype=torch.bool)
        eat = (self.eat_mask[None, :, :, None] > 0) & (biomass[:, None] > 0)
        return torch.cat((move, rest, eat), dim=2)

    def action_probabilities(self, logits, biomass, temperature):
        # logits [E,D,C,A]; the -1e9 mask and its temperature treatment match
        # the NumPy implementation, including argmax's first-index tie rule.
        logits = logits.transpose(2, 3)
        masked = logits + torch.where(self.action_mask(biomass), 0.0, -1e9)
        probs = torch.softmax(masked / temperature, dim=2)
        b_dm = biomass[:, self.dm_index]
        small = (b_dm > 0) & (b_dm < self.min_split)
        # one_hot performs value checks on some Torch backends/releases. A
        # fixed arange comparison has no scalar extraction or dynamic output.
        one_hot = (self.action_indices == probs.argmax(dim=2)[:, :, None]).to(probs.dtype)
        return torch.where(small[:, :, None], one_hot, probs)

    def action_statistics(self, biomass, actions):
        active = (biomass[:, self.dm_index] > 0).to(actions.dtype)
        count = active.sum(-1)
        denom = count.clamp_min(1)
        entropy = -(actions * torch.log(actions + 1e-12)).sum(2)
        move = actions[:, :, :4].sum(2)
        rest = actions[:, :, 4]
        eat = actions[:, :, 5:].sum(2)
        stats = torch.stack([(v * active).sum(-1) / denom for v in (entropy, move, rest, eat)], dim=-1)
        return stats.to(torch.float64), (count > 0).to(torch.float64)

    def predation(self, biomass, reserve, actions):
        b_dm = biomass[:, self.dm_index]
        hunger = (1.0 - self.energy_level(biomass, reserve)[:, self.dm_index] / 0.8).clamp_min(0)
        hidden = torch.zeros_like(biomass).index_copy(1, self.dm_index, actions[:, :, 4])
        visible = biomass * (1.0 - hidden * (1.0 - self.visibility))
        if self.holling:
            prey = visible[:, None]
            squared = prey * prey
            type2 = self.intake_rate * prey / (1.0 + self.intake_rate * self.handling * prey)
            type3 = self.intake_rate * squared / (1.0 + self.intake_rate * self.handling * squared)
            rate = self.type3 * type3 + (1.0 - self.type3) * type2
        else:
            rate = self.intake_rate
        demand = b_dm[:, :, None] * actions[:, :, 5:] * rate * hunger[:, :, None]
        total = demand.sum(1)
        scale = torch.where(total > visible, visible / (total + 1e-9), 1.0)
        actual = demand * scale[:, None]
        gains = (actual * self.energy_gain).sum(2)
        intake = actual.sum(1)
        reduction = torch.where(biomass > 1e-9, (biomass - intake) / (biomass + 1e-9), 0.0)
        return biomass - intake, reserve * reduction, gains, hidden, actual

    def immigration(self, b_emig, r_emig):
        weights = self.edge_weights[None, None]
        threshold = self.threshold[:, self.dm_index]
        per_cell = b_emig[:, :, None] * weights / self.edge_cumsum.clamp_min(1e-30)
        valid = (per_cell >= threshold) & (threshold > 0) & (b_emig[:, :, None] > 0)
        keep = torch.where(valid, self.edge_ranks, 0).amax(-1, keepdim=True)
        # No viable prefix means retain the reference's ordinary distribution.
        mask = (self.edge_ranks <= keep) | (keep == 0)
        weights = weights * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
        shape = (b_emig.shape[0], self.D, self.C)
        b = torch.zeros(shape, device=self.device).index_copy(2, self.edge_indices, b_emig[:, :, None] * weights)
        r = torch.zeros(shape, device=self.device).index_copy(2, self.edge_indices, r_emig[:, :, None] * weights)
        return b, r

    def movement(self, biomass, reserve, gains, actions):
        b, r = biomass[:, self.dm_index], reserve[:, self.dm_index]
        move, rest = actions[:, :, :4], actions[:, :, 4]
        eat = actions[:, :, 5:].sum(2)
        move_fraction = move.sum(2)
        rest_r = (r * rest - b * rest * self.metabolism * self.cost_rest).clamp_min(0)
        eat_r = (r * eat - b * eat * self.metabolism * self.cost_eat).clamp_min(0) + gains
        move_r = (r * move_fraction - b * move_fraction * self.metabolism * self.cost_move).clamp_min(0)
        safe = torch.where(move_fraction > 0, move_fraction, 1.0)
        moving_r = move / safe[:, :, None] * move_r[:, :, None]
        moving_b = move * b[:, :, None]
        b_out, r_out = moving_b * self.velocity, moving_r * self.velocity
        b_total = b * rest + b * eat + (moving_b - b_out).sum(2)
        r_total = rest_r + eat_r + (moving_r - r_out).sum(2)
        for direction in range(4):
            source = (direction + 2) % 4
            index, valid = self.neighbors[source], self.neighbor_valid[source]
            b_total = b_total + b_out[:, :, direction, index] * valid
            r_total = r_total + r_out[:, :, direction, index] * valid
        if self.migration:
            b_emig = (b_out * self.outside).sum((2, 3))
            r_emig = (r_out * self.outside).sum((2, 3))
            b_imm, r_imm = self.immigration(b_emig, r_emig)
            b_total, r_total = b_total + b_imm, r_total + r_imm
        r_total = torch.minimum(r_total.clamp_min(0), b_total * self.max_reserve[:, self.dm_index])
        return biomass.index_copy(1, self.dm_index, b_total), reserve.index_copy(1, self.dm_index, r_total)

    def population(self, biomass, reserve, tick, phase, seed_multiplier):
        b = biomass * self.mortality_keep
        r = reserve * self.mortality_keep
        surplus = self.energy_level(b, r) - self.maintenance
        delta = b * torch.where(surplus >= 0, self.growth, self.starve) * surplus
        loss = -torch.minimum(delta, torch.zeros_like(delta))
        starve_loss = torch.where(self.is_dm, torch.minimum(loss, b), 0.0)
        reduction = torch.where(loss > 0, (b - loss) / (b + 1e-9), 1.0).clamp(0, 1)
        dm_b, dm_r = (b + delta).clamp_min(0), r * reduction

        season = 1.0 + self.amplitude * torch.sin(2.0 * math.pi * (tick + phase) / self.period.clamp_min(1e-30))
        rate = self.growth * torch.where(self.seasonal, season, 1.0)
        ndm_delta = rate * biomass * (1.0 - biomass / (self.capacity + 1e-9))
        ndm_delta = ndm_delta + self.seed_rate * self.capacity * seed_multiplier
        ndm_b = torch.minimum((biomass + ndm_delta).clamp_min(0), self.capacity)
        b = torch.where(self.is_dm, dm_b, ndm_b)
        r = torch.where(self.is_dm, dm_r, reserve)
        b = torch.where(b < self.threshold, 0.0, b)
        r = torch.where(b <= 0, 0.0, r)
        return b, r, starve_loss

    def step(self, biomass, reserve, actions, tick, phase, seed_multiplier):
        b, r, gains, hidden, intake = self.predation(biomass, reserve, actions)
        b, r = self.movement(b, r, gains, actions)
        b, r, starve_loss = self.population(b, r, tick, phase, seed_multiplier)
        return b, r, hidden, intake, starve_loss
