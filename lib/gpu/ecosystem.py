"""Ecological equations on fixed-shape tensors.

State fields are [ecosystem, functional_group, cell]. Actions are
[ecosystem, decision_maker, action, cell]. Only construction/fixture import
touch NumPy; every numerical method supports CPU tensors and CUDA tensors.
"""

import math

import numpy as np
import torch
from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC
from lib.environments.ecosystem_env import debug_food
from lib.environments.ecosystem_env.currents import direction_fractions
from lib.world.energy_balance import resolve_satiation_scale

# Relative harvest cap, imported rather than repeated so the two engines
# cannot drift apart again (Fix 2 / Section 73: an absolute epsilon in the
# denominator left a residue below float32 precision and drove overharvested
# cells to exactly 0.0, an absorbing state).
HARVEST_FRACTION = float(MAX_HARVEST_FRAC)


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
        self.boundary = env.boundary
        self.currents = env.currents
        self.food_blobs = env.food_blobs
        self.current_world_seed = env.current_world_seed
        self.holling = env._has_holling2 or env._has_holling3
        tensor = self.tensor
        self.food_blob_index = tensor([self.ids.index(fid) for fid in debug_food.selected_ids(env)], torch.long)
        habitat = env.grid.get_map("accessibility")
        self.food_blob_allowed = tensor(
            np.ones(self.C, dtype=bool) if habitat is None else (habitat > 0).ravel(), torch.bool)
        if self.food_blobs is not None and not bool(self.food_blob_allowed.any()):
            raise ValueError("Food blobs need at least one accessible cell")
        self.dm_index = tensor(self.dm_positions, torch.long)
        self.ndm_index = tensor(self.ndm_positions, torch.long)
        self.current_response = tensor([
            0.0 if self.ids[i] in debug_food.selected_ids(env) else env.fgs[self.ids[i]].current_response
            for i in self.ndm_positions
        ])[None, :, None, None]
        self.is_dm = tensor([i in self.dm_positions for i in range(self.G)], torch.bool)[None, :, None]
        self.move_mask = tensor(env.move_mask.reshape(4, self.C))
        self.eat_mask = tensor(env.eat_static_mask)
        self.intake_rate = tensor(env.max_intake_mat)[None, :, :, None]
        self.energy_gain = tensor(env.energy_gain_mat)[None, :, :, None]
        self.handling = tensor(env.handling_time_mat)[None, :, :, None]
        self.type3 = tensor(env._type3_pred_mask.reshape(self.D))[None, :, None, None]
        # Beddington-DeAngelis interference w_X [1/ton], per DM. Mirrors
        # the reference's ``dm_interference`` / ``_has_interference``; the
        # all-zero default keeps the pure Holling path bit-identical.
        self.interference = tensor(
            env.dm_interference.reshape(self.D))[None, :, None, None]
        self.has_interference = bool((self.interference > 0).any())
        self.visibility = tensor(env._all_visibility_floor)[None, :, None]
        # Per-(predator, prey) detection floor. ``None`` keeps the cheaper
        # shared-vector path, which is bit-identical whenever every override
        # equals the prey's own floor - the same switch the reference makes
        # with ``_has_pair_vis_floor``. Detection is a property of the PAIR:
        # porpoise biosonar is unaffected by the crypsis that hides herring
        # from seabirds.
        self.pair_visibility = (
            tensor(env.vis_floor_mat)[None, :, :, None]
            if getattr(env, "_has_pair_vis_floor", False) else None)
        # Per-FG satiation scale for h = max(0, 1 - s/scale). Missing values
        # fall back to HUNGER_SATIATION_SCALE through the shared resolver, so
        # a library override (porpoises: 0.9) is honoured here as well.
        self.satiation = tensor([
            resolve_satiation_scale(
                getattr(env.fgs[fid], "satiation_scale", None))
            for fid in self.dm_ids])[None, :, None]
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
        # Same product, per DM and broadcast over the direction axis of
        # ``b_out``, for the sub-threshold split suppression in
        # ``movement``. Mirrors the reference's ``_dm_split_thr`` /
        # ``_dm_split_thr_any`` (see state.py); 0 opts out, exactly as the
        # extinction sweep itself does.
        self.split_thr = self.threshold[:, self.dm_index][:, :, :, None]
        self.split_thr_any = bool((self.threshold[:, self.dm_index] > 0).any())
        # ``--mortality_multiplier`` scales every FG's rate, exactly as
        # the reference does in ``population_change``; the default 1.0
        # keeps this vector bit-identical.
        mortality_scale = max(0.0, float(getattr(env, "mortality_multiplier", 1.0)))
        self.mortality_keep = tensor([
            max(0.0, 1.0 - max(0.0, fg.natural_mortality) * mortality_scale)
            if env.apply_natural_mortality and fg.is_decision_maker else 1.0
            for fg in groups
        ])[None, :, None]
        self._build_indices(env)

    def tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _build_indices(self, env):
        yy, xx = np.indices((self.H, self.W))
        self.current_x = self.tensor(xx.ravel())
        self.current_y = self.tensor(yy.ravel())
        coords = ((yy - 1, xx), (yy, xx + 1), (yy + 1, xx), (yy, xx - 1))
        neighbors, valid = [], []
        for y, x in coords:
            if self.boundary == "torus":
                valid.append(np.ones(self.C, dtype=bool))
                neighbors.append(((y % self.H) * self.W + x % self.W).ravel())
            else:
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
        # Channel layout of the bank assembled in ``observations``:
        #   [biomass (G)] [visible] [energy_level (G)]
        # where the visible block is one vector of G channels in the shared
        # case and one block of G channels PER OBSERVER when a pair floor
        # exists, because then "what I can see" depends on who is looking.
        pair = self.pair_visibility is not None
        visible_base = self.G
        energy_base = self.G + (self.D * self.G if pair else self.G)
        for d, own in enumerate(self.dm_positions):
            others = list(env.obs_others_idx[d])
            visible = [visible_base + (d * self.G + j if pair else j)
                       for j in others]
            channels = [own, energy_base + own] + visible
            for k, channel in enumerate(channels):
                obs_indices[d, :, k] = channel * self.C + cells
                obs_valid[d, :, k] = 1.0
            offset = len(channels)
            for direction in range(4):
                for channel in [own] + visible:
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
        if self.pair_visibility is None:
            visible = biomass * (1.0 - hidden * (1.0 - self.visibility))
        else:
            # [E, D, G, C] -> D consecutive blocks of G channels, indexed by
            # ``_build_indices`` so every observer reads its own row.
            visible = (biomass[:, None] * (1.0 - hidden[:, None]
                                           * (1.0 - self.pair_visibility))).flatten(1, 2)
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
        hunger = (1.0 - self.energy_level(biomass, reserve)[:, self.dm_index]
                  / self.satiation).clamp_min(0)
        hidden = torch.zeros_like(biomass).index_copy(1, self.dm_index, actions[:, :, 4])
        pair_visible = None
        if self.pair_visibility is None:
            visible = biomass * (1.0 - hidden * (1.0 - self.visibility))
        else:
            pair_visible = biomass[:, None] * (1.0 - hidden[:, None]
                                              * (1.0 - self.pair_visibility))
            # Shared availability cap: total removal from prey j is bounded
            # by what the BEST-detecting predator of j can see. Rows that do
            # not prey on j are excluded so they cannot inflate the cap.
            visible = torch.where(self.eat_mask[None, :, :, None] > 0.0,
                                  pair_visible, 0.0).amax(1)
        # Interference term I = w_X * B_X(c): the predator's own biomass in
        # the cell, intraspecific only, shaped (B, D, 1, C) so it broadcasts
        # over the prey axis exactly as the reference does.
        inter = self.interference * b_dm[:, :, None] if self.has_interference else 0.0
        if self.holling:
            prey = pair_visible if pair_visible is not None else visible[:, None]
            squared = prey * prey
            type2 = self.intake_rate * prey / (1.0 + self.intake_rate * self.handling * prey + inter)
            type3 = self.intake_rate * squared / (1.0 + self.intake_rate * self.handling * squared + inter)
            rate = self.type3 * type3 + (1.0 - self.type3) * type2
        elif self.has_interference:
            rate = self.intake_rate / (1.0 + inter)
        else:
            rate = self.intake_rate
        demand = b_dm[:, :, None] * actions[:, :, 5:] * rate * hunger[:, :, None]
        if pair_visible is not None:
            # Per-predator bound: no predator may demand more than the
            # fraction of the prey IT can detect, even when a better-
            # detecting predator raised the shared cap.
            demand = torch.minimum(demand, pair_visible)
        total = demand.sum(1)
        harvest = visible * HARVEST_FRACTION
        scale = torch.where(total > harvest, harvest / total.clamp_min(1e-30), 1.0)
        actual = demand * scale[:, None]
        gains = (actual * self.energy_gain).sum(2)
        intake = actual.sum(1)
        reduction = torch.where(biomass > 1e-9, (biomass - intake) / (biomass + 1e-9), 0.0)
        return biomass - intake, reserve * reduction, gains, hidden, actual

    def immigration(self, b_emig, r_emig, group_index=None):
        """Shared boundary redistribution for active and passive movement."""
        weights = self.edge_weights[None, None]
        threshold = self.threshold[:, self.dm_index if group_index is None else group_index]
        per_cell = b_emig[:, :, None] * weights / self.edge_cumsum.clamp_min(1e-30)
        valid = (per_cell >= threshold) & (threshold > 0) & (b_emig[:, :, None] > 0)
        keep = torch.where(valid, self.edge_ranks, 0).amax(-1, keepdim=True)
        # No viable prefix means retain the reference's ordinary distribution.
        mask = (self.edge_ranks <= keep) | (keep == 0)
        weights = weights * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
        shape = (b_emig.shape[0], b_emig.shape[1], self.C)
        b = torch.zeros(shape, device=self.device).index_copy(2, self.edge_indices, b_emig[:, :, None] * weights)
        r = torch.zeros(shape, device=self.device).index_copy(2, self.edge_indices, r_emig[:, :, None] * weights)
        return b, r

    def movement(self, biomass, reserve, gains, actions, track=False):
        """Move, split and settle; ``track`` also returns the source flow.

        With ``track`` the third return value is
        ``(b_stay, b_out, b_total)``, the biomass that remained in each
        source cell, what left it per direction, and the resulting
        per-cell biomass. The local reward (``--local_reward``) needs all
        three to attribute the destination cells' end-of-tick energy back
        to the cell the population started in.
        """
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
        b_stay = b * rest + b * eat + (moving_b - b_out).sum(2)
        b_total = b_stay
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
        b_cancel = None
        if self.split_thr_any:
            b_total, r_total, b_cancel = self.suppress_splits(b_out, r_out, b_total, r_total)
        r_total = torch.minimum(r_total.clamp_min(0), b_total * self.max_reserve[:, self.dm_index])
        settled = (biomass.index_copy(1, self.dm_index, b_total),
                   reserve.index_copy(1, self.dm_index, r_total))
        if not track:
            return settled[0], settled[1], None
        if b_cancel is not None:
            # Cancelled splits stayed home, so the tracked flow has to
            # move them from the outflow back into the stay term; only
            # then does it still sum to ``b_total``.
            b_stay = b_stay + b_cancel.sum(2)
            b_out = b_out - b_cancel
        return settled[0], settled[1], (b_stay, b_out, b_total)

    def suppress_splits(self, b_out, r_out, b_total, r_total):
        """Cancel move outflows that would land in a sub-threshold cell.

        Section 69, mirrored from
        ``lib/environments/ecosystem_env/movement.py``. ``population``
        ends the tick by zeroing every cell below ``thr =
        extinction_threshold_factor * min_split_biomass``, so a splitting
        decision maker bleeds biomass through that sweep. An outflow whose
        DESTINATION would still be below ``thr`` after receiving it is
        therefore undone: biomass and reserve stay in the source cell.

        Monotone-safe -- sources only gain, and the cells that lose inflow
        were going to be zeroed anyway -- so the number of sub-threshold
        cells cannot grow and no viable cell is made non-viable.

        Off-grid directions count as unblocked, leaving the migration
        emigration/immigration path (with its own top-k concentration)
        alone. Returns ``(b_total, r_total, b_cancel)``.
        """
        # Tentative destination totals, aligned on the SOURCE cell:
        # ``neighbors[d][c]`` is where a direction-d move from c lands.
        dest = b_total[:, :, self.neighbors]
        dest = torch.where(self.neighbor_valid[None, None] > 0, dest, float("inf"))
        blocked = (b_out > 0.0) & (dest < self.split_thr)
        b_cancel = torch.where(blocked, b_out, 0.0)
        r_cancel = torch.where(blocked, r_out, 0.0)
        b_total = b_total + b_cancel.sum(2)
        r_total = r_total + r_cancel.sum(2)
        for direction in range(4):
            # Exactly the settle loop's gather with the sign flipped.
            source = (direction + 2) % 4
            index, valid = self.neighbors[source], self.neighbor_valid[source]
            b_total = b_total - b_cancel[:, :, direction, index] * valid
            r_total = r_total - r_cancel[:, :, direction, index] * valid
        # float32 round-off on the +/- pair can leave tiny negatives.
        return b_total.clamp_min(0), r_total.clamp_min(0), b_cancel

    def local_energy(self, biomass, reserve):
        """``Q(c) = B(c)*energy_content + R(c)`` per DM, shape [E, D, C]."""
        b, r = biomass[:, self.dm_index], reserve[:, self.dm_index]
        return b * self.energy_content[:, self.dm_index][:, :, None] + r

    def tracked_energy(self, flow, biomass, reserve):
        """End-of-tick energy attributed back to the source cell.

        ``tracked[e, i, c]`` is ``B(c, t+1)`` of the local reward: the
        share of the plus-shaped destination set ``{c, N, E, S, W}`` that
        the population starting in ``c`` is entitled to. Everything after
        the movement is cell-wise multiplicative, so splitting a
        destination cell's energy in proportion to the biomass each
        source delivered is an identity, not an approximation.

        ``frac_in`` is the fraction of the source cell's post-movement
        biomass that stayed on the grid. Outflow across the border has no
        in-grid destination, so it is removed from the denominator (``A``
        is scaled by ``frac_in``) instead of being booked as a loss.
        """
        b_stay, b_out, b_total = flow
        unit = torch.where(b_total > 0.0, self.local_energy(biomass, reserve)
                           / b_total.clamp_min(1e-30), 0.0)
        tracked = b_stay * unit
        b_move_in = torch.zeros_like(b_stay)
        for direction in range(4):
            # ``neighbors[direction]`` is the cell a direction-d move from
            # the source lands in, the inverse of the settle loop above.
            index, valid = self.neighbors[direction], self.neighbor_valid[direction]
            out = b_out[:, :, direction] * valid
            tracked = tracked + out * unit[:, :, index]
            b_move_in = b_move_in + out
        b_source = b_stay + b_out.sum(2)
        frac_in = torch.where(b_source > 0.0,
                              (b_stay + b_move_in) / b_source.clamp_min(1e-30), 1.0)
        return tracked, frac_in

    def local_min_energy(self, factor=1.0):
        """Lower bound on ``A(c, t)`` for a cell to take part, [1, D, 1].

        ``factor * extinction_threshold * energy_content``, i.e. the cell
        held a viable population at the start of the tick. Mirrors
        ``source_tracking._min_energy``, which reads the reference's
        ``_dm_split_thr`` -- the same product stored in ``threshold``.
        """
        floor = (self.threshold[:, self.dm_index]
                 * self.energy_content[:, self.dm_index][:, :, None])
        return (floor.double() * float(factor)).clamp_min(1e-12)

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

    def advect(self, biomass, reserve, tick, keys=None):
        if self.currents is None or self.currents.strength == 0 or not self.ndm_positions:
            return biomass, reserve
        # ``torch.full`` rather than ``as_tensor``: a host scalar copied to the
        # device is not permitted while a CUDA graph is capturing.
        tick = (tick.to(torch.int64) if isinstance(tick, torch.Tensor) else
                torch.full((), int(tick), dtype=torch.int64, device=self.device))
        if keys is None:
            keys = torch.full((biomass.shape[0],), self.current_world_seed,
                              dtype=torch.int64, device=self.device)
        fractions = torch.stack(direction_fractions(
            tick, keys[:, None], self.currents, self.current_x, self.current_y,
            (self.H, self.W) if self.boundary == "torus" else None), dim=1)
        fractions = (fractions[:, None] * self.current_response
                     * self.move_mask[None, None])
        b, r = biomass[:, self.ndm_index], reserve[:, self.ndm_index]
        b_out, r_out = b[:, :, None] * fractions, r[:, :, None] * fractions
        b_total, r_total = b - b_out.sum(2), r - r_out.sum(2)
        for direction in range(4):
            source = (direction + 2) % 4
            index, valid = self.neighbors[source], self.neighbor_valid[source]
            b_total = b_total + b_out[:, :, direction, index] * valid
            r_total = r_total + r_out[:, :, direction, index] * valid
        if self.migration:
            b_emig = (b_out * self.outside).sum((2, 3))
            r_emig = (r_out * self.outside).sum((2, 3))
            b_imm, r_imm = self.immigration(b_emig, r_emig, self.ndm_index)
            b_total, r_total = b_total + b_imm, r_total + r_imm
        return biomass.index_copy(1, self.ndm_index, b_total), reserve.index_copy(1, self.ndm_index, r_total)

    def step(self, biomass, reserve, actions, tick, phase, seed_multiplier,
             current_keys=None, track_source=False, food_blob_totals=None, food_blob_motion=None):
        """One tick. ``track_source`` appends the local-reward tracking.

        With ``track_source`` a sixth value ``(start, tracked, frac_in)``
        is returned: ``start`` is ``A(c, t)``, the cell energy the policy
        observed (before predation, the first mortality of the tick), and
        ``tracked``/``frac_in`` come from ``tracked_energy`` once the
        end-of-tick state is final.
        """
        start = self.local_energy(biomass, reserve) if track_source else None
        if self.food_blobs is not None and food_blob_totals is None:
            food_blob_totals = (biomass[:, self.food_blob_index].sum(-1, keepdim=True),
                                reserve[:, self.food_blob_index].sum(-1, keepdim=True))
        b, r, gains, hidden, intake = self.predation(biomass, reserve, actions)
        b, r, flow = self.movement(b, r, gains, actions, track=track_source)
        b, r = self.advect(b, r, tick, current_keys)
        b, r, starve_loss = self.population(b, r, tick, phase, seed_multiplier)
        if food_blob_motion is not None:
            food_blob_motion.advance()
        b, r = debug_food.apply_tensor(self, b, r, tick + 1, current_keys,
                                       food_blob_totals, food_blob_motion)
        if not track_source:
            return b, r, hidden, intake, starve_loss
        tracked, frac_in = self.tracked_energy(flow, b, r)
        return b, r, hidden, intake, starve_loss, (start, tracked, frac_in)
