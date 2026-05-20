import numpy as np
import torch
from lib.world.grid import Grid
from lib.world.functional_group import FunctionalGroup

class EcosystemEnvironment:
    def __init__(self, grid_config, functional_groups, interactions, policies=None,
                 observable_impact_vars=None):
        self.grid = Grid(**grid_config)
        self.fgs = functional_groups  # Dictionary: id -> FunctionalGroup
        self.interactions = interactions
        self.policies = policies or {} # Dictionary: id -> PolicyNetwork
        # Ordered list of impact_ids that are exposed to the policy network as
        # observation channels. One observation layer is appended per id (in
        # this order) after biomass / energy / other-FG channels. Callers that
        # don't specify it default to no observable impacts.
        self.observable_impact_vars = list(observable_impact_vars or [])
        self.tick_count = 0
        # Stable global ordering of all functional groups. Used to give every
        # decision maker a uniform action space: 5 + N_all_fgs outputs, where
        # eat-slot i refers to global_fg_order[i]. Slots not on the predator's
        # menu (or with predation matrix entry preys_on=False) are permanently
        # masked out per cell.
        self.global_fg_order = sorted(self.fgs.keys())

        # Cached static structures (built lazily on first step)
        self._static_built = False

    # ---------- Static caches (built once) ----------
    def _build_static_caches(self):
        H, W = self.grid.height, self.grid.width
        self.H, self.W = H, W
        self.dtype = np.float32

        self.dm_ids = [fid for fid in self.global_fg_order if self.fgs[fid].is_decision_maker]
        self.N_dm = len(self.dm_ids)
        self.N_all = len(self.global_fg_order)
        self.dm_index_in_all = np.array(
            [self.global_fg_order.index(fid) for fid in self.dm_ids], dtype=np.int64
        )

        # Convert all biomass/energy fields to float32 for consistency
        for fg in self.fgs.values():
            if fg.biomass is not None and fg.biomass.dtype != self.dtype:
                fg.biomass = fg.biomass.astype(self.dtype)
            if fg.energy_reserve is not None and fg.energy_reserve.dtype != self.dtype:
                fg.energy_reserve = fg.energy_reserve.astype(self.dtype)
            if fg.temp_energy_gains is not None and fg.temp_energy_gains.dtype != self.dtype:
                fg.temp_energy_gains = fg.temp_energy_gains.astype(self.dtype)

        # Accessibility-based movement mask: constant across ticks.
        access_map = self.grid.get_map('accessibility')
        move_mask = np.ones((4, H, W), dtype=self.dtype)
        move_mask[0, 0, :] = 0.0
        move_mask[2, -1, :] = 0.0
        move_mask[1, :, -1] = 0.0
        move_mask[3, :, 0] = 0.0
        if access_map is not None:
            am = access_map.astype(self.dtype)
            inacc_n = np.ones((H, W), dtype=self.dtype); inacc_n[1:, :] = am[:-1, :]
            inacc_s = np.ones((H, W), dtype=self.dtype); inacc_s[:-1, :] = am[1:, :]
            inacc_e = np.ones((H, W), dtype=self.dtype); inacc_e[:, :-1] = am[:, 1:]
            inacc_w = np.ones((H, W), dtype=self.dtype); inacc_w[:, 1:] = am[:, :-1]
            move_mask[0] *= (inacc_n > 0).astype(self.dtype)
            move_mask[2] *= (inacc_s > 0).astype(self.dtype)
            move_mask[1] *= (inacc_e > 0).astype(self.dtype)
            move_mask[3] *= (inacc_w > 0).astype(self.dtype)
        self.move_mask = move_mask  # (4, H, W)

        # Static eat-mask + intake/gain matrices (N_dm, N_all)
        eat_static = np.zeros((self.N_dm, self.N_all), dtype=self.dtype)
        max_intake = np.zeros((self.N_dm, self.N_all), dtype=self.dtype)
        energy_gain = np.zeros((self.N_dm, self.N_all), dtype=self.dtype)
        for i, pred_id in enumerate(self.dm_ids):
            pred_fg = self.fgs[pred_id]
            menu = pred_fg.params.get('menu', [])
            interaction = pred_fg.params.get('interaction', {})
            for j, prey_id in enumerate(self.global_fg_order):
                if prey_id not in menu:
                    continue
                inter_id = f'{pred_id}_preys_on_{prey_id}'
                inter_def = interaction.get(inter_id, {})
                if not inter_def.get('preys_on', True):
                    continue
                eat_static[i, j] = 1.0
                max_intake[i, j] = float(inter_def.get('max_intake_rate', 0.0))
                energy_gain[i, j] = float(inter_def.get('energy_gain', 0.0))
        self.eat_static_mask = eat_static
        self.max_intake_mat = max_intake
        self.energy_gain_mat = energy_gain

        # Per-DM cached scalar params
        self.dm_v = np.array([float(np.clip(self.fgs[fid].speed, 0.0, 1.0)) for fid in self.dm_ids], dtype=self.dtype)
        self.dm_cost_move = np.array([self.fgs[fid].params.get('movement_cost', 3.0) for fid in self.dm_ids], dtype=self.dtype)
        self.dm_cost_eat  = np.array([self.fgs[fid].params.get('feeding_cost', 3.0) for fid in self.dm_ids], dtype=self.dtype)
        self.dm_cost_rest = np.array([self.fgs[fid].params.get('resting_cost', 1.0) for fid in self.dm_ids], dtype=self.dtype)
        self.dm_resting_metabolism = np.array([self.fgs[fid].resting_metabolism for fid in self.dm_ids], dtype=self.dtype)
        self.dm_max_energy_reserve = np.array([self.fgs[fid].max_energy_reserve for fid in self.dm_ids], dtype=self.dtype)
        # Per-DM cached impact tables for ALL impacts the FG is affected by.
        # Each entry is a list of (impact_id, (xs, bf, ef)) tuples; impacts
        # without `impact_affects=true` or with an empty/invalid table are
        # omitted. Used by both _apply_movement (energy_factor → extra energy
        # cost) and _apply_growth_and_impact (biomass_factor → mortality,
        # energy_factor → reserve loss).
        self.dm_impact_tables = []
        for fid in self.dm_ids:
            impacts = self.fgs[fid].params.get('impact', {}) or {}
            entries = []
            for imp_id, imp_def in impacts.items():
                tbl = self._extract_impact_table(imp_def)
                if tbl is not None:
                    entries.append((imp_id, tbl))
            self.dm_impact_tables.append(entries)
        # Per-DM minimum biomass for splitting via movement. 0 = no threshold.
        self.dm_min_split = np.array(
            [getattr(self.fgs[fid], 'min_split_biomass', 0.0) for fid in self.dm_ids],
            dtype=self.dtype,
        )

        self._rebuild_batched_weights()
        self._static_built = True

    def _rebuild_batched_weights(self):
        """Stack per-DM PolicyNetwork weights into batched tensors for bmm-based forward."""
        self._batched_ready = False
        if self.N_dm == 0:
            return
        if not all(fid in self.policies for fid in self.dm_ids):
            return
        try:
            first = self.policies[self.dm_ids[0]]
            layers = [m for m in first.net if isinstance(m, torch.nn.Linear)]
            if len(layers) != 3:
                return
            in_dim = layers[0].in_features
            hid = layers[0].out_features
            out_dim = layers[2].out_features
            W1 = torch.empty(self.N_dm, in_dim, hid)
            b1 = torch.empty(self.N_dm, hid)
            W2 = torch.empty(self.N_dm, hid, hid)
            b2 = torch.empty(self.N_dm, hid)
            W3 = torch.empty(self.N_dm, hid, out_dim)
            b3 = torch.empty(self.N_dm, out_dim)
            for i, fid in enumerate(self.dm_ids):
                net = self.policies[fid].net
                lin = [m for m in net if isinstance(m, torch.nn.Linear)]
                if (lin[0].in_features != in_dim or lin[0].out_features != hid
                        or lin[1].in_features != hid or lin[1].out_features != hid
                        or lin[2].in_features != hid or lin[2].out_features != out_dim):
                    return
                W1[i] = lin[0].weight.detach().t()
                b1[i] = lin[0].bias.detach()
                W2[i] = lin[1].weight.detach().t()
                b2[i] = lin[1].bias.detach()
                W3[i] = lin[2].weight.detach().t()
                b3[i] = lin[2].bias.detach()
            self._W1, self._b1 = W1, b1
            self._W2, self._b2 = W2, b2
            self._W3, self._b3 = W3, b3
            self._in_dim = in_dim
            self._out_dim = out_dim
            self._batched_ready = True
        except Exception:
            self._batched_ready = False

    # ---------- Step ----------
    def step(self):
        if not self._static_built:
            self._build_static_caches()
        # Policy weights are stacked once in _build_static_caches. ARS creates a
        # fresh env per rollout (via env_builder) so weights are captured at the
        # start. If policies are reassigned/mutated on a long-lived env, call
        # env._rebuild_batched_weights() explicitly.

        shuffled_ids = list(self.fgs.keys())
        np.random.shuffle(shuffled_ids)
        self.ordered_fg_ids = shuffled_ids

        self._calculate_decisions()
        # Method.pdf §steg 1–6: direct impact mortality m_X^Impact is applied
        # *before* predation, so prey biomass available to predators already
        # reflects impact losses for this tick.
        self._apply_impact_mortality()
        self._apply_predation()
        self._apply_movement()
        self._apply_growth()

        self.tick_count += 1

    # ---------- Observation builder (batched across all DMs) ----------
    def _build_observation_batch(self):
        """Returns observation array of shape (N_dm, H*W, D), dtype float32."""
        H, W = self.H, self.W
        N_all = self.N_all
        B_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0).astype(self.dtype, copy=False)
        # One channel per impact_id flagged as observable in the project.
        # Missing maps default to zero fields so the channel layout is stable
        # even when an impact map hasn't been installed yet.
        impact_layers = []
        for iid in self.observable_impact_vars:
            m = self.grid.get_map(iid)
            if m is None:
                m = np.zeros((H, W), dtype=self.dtype)
            else:
                m = m.astype(self.dtype, copy=False)
            impact_layers.append(m)
        n_obs_imp = len(impact_layers)

        D = 2 + (N_all - 1) + n_obs_imp  # B_own, E_own, others, observable impacts
        obs = np.empty((self.N_dm, D, H, W), dtype=self.dtype)

        for i, pred_id in enumerate(self.dm_ids):
            pred_fg = self.fgs[pred_id]
            obs[i, 0] = pred_fg.biomass
            obs[i, 1] = pred_fg.energy_level.astype(self.dtype, copy=False)
            j_idx = int(self.dm_index_in_all[i])
            if j_idx > 0:
                obs[i, 2:2 + j_idx] = B_all[:j_idx]
            if j_idx < N_all - 1:
                obs[i, 2 + j_idx:2 + (N_all - 1)] = B_all[j_idx + 1:]
            for k, layer in enumerate(impact_layers):
                obs[i, 2 + (N_all - 1) + k] = layer

        return obs.transpose(0, 2, 3, 1).reshape(self.N_dm, H * W, D)

    # ---------- Batched policy inference ----------
    def _batched_policy_forward(self, obs_batch):
        """obs_batch: torch tensor (N_dm, N, D) -> probs (N_dm, N, out_dim)."""
        h = torch.sigmoid(torch.bmm(obs_batch, self._W1) + self._b1.unsqueeze(1))
        h = torch.sigmoid(torch.bmm(h, self._W2) + self._b2.unsqueeze(1))
        logits = torch.bmm(h, self._W3) + self._b3.unsqueeze(1)
        return torch.softmax(logits, dim=-1)

    def _calculate_decisions(self):
        H, W = self.H, self.W
        if self.N_dm == 0:
            self.pi = {fid: None for fid in self.fgs}
            return

        obs_np = self._build_observation_batch()
        D = obs_np.shape[-1]

        # Accumulate raw-obs statistics (sum, sumsq, count) per DM, per dim.
        # These are returned to the trainer for a Welford parallel merge so the
        # global running mean/var converges across rollouts (and across workers
        # in the parallel path).
        # Shapes: obs_np is (N_dm, H*W, D).
        flat = obs_np.reshape(self.N_dm, -1, D)
        nsamp = flat.shape[1]
        sample_sum = flat.sum(axis=1, dtype=np.float64)              # (N_dm, D)
        sample_sumsq = (flat.astype(np.float64) ** 2).sum(axis=1)    # (N_dm, D)
        if not hasattr(self, '_obs_sum') or self._obs_sum is None:
            self._obs_sum = np.zeros((self.N_dm, D), dtype=np.float64)
            self._obs_sumsq = np.zeros((self.N_dm, D), dtype=np.float64)
            self._obs_count = 0
        self._obs_sum += sample_sum
        self._obs_sumsq += sample_sumsq
        self._obs_count += nsamp

        # ARS-V2 observation normalisation: subtract running mean, divide by
        # running std, clip to [-10, 10]. Stats are *frozen* during a rollout
        # (set by the trainer via env.obs_mean / env.obs_var before stepping)
        # so +delta and -delta rollouts see the same normalisation.
        if (getattr(self, 'obs_mean', None) is not None
                and getattr(self, 'obs_var', None) is not None
                and self.obs_mean.shape == (self.N_dm, D)):
            mean = self.obs_mean.astype(self.dtype, copy=False)
            var = self.obs_var.astype(self.dtype, copy=False)
            std = np.sqrt(var + np.float32(1e-8))
            obs_np = (obs_np - mean[:, None, :]) / std[:, None, :]
            obs_np = np.clip(obs_np, -10.0, 10.0).astype(self.dtype, copy=False)

        obs_t = torch.from_numpy(obs_np)

        if self._batched_ready and obs_np.shape[-1] == self._in_dim:
            with torch.no_grad():
                probs_t = self._batched_policy_forward(obs_t)
            probs = probs_t.numpy()
            num_actions = self._out_dim
        else:
            num_actions = 5 + self.N_all
            probs = np.empty((self.N_dm, H * W, num_actions), dtype=self.dtype)
            for i, fid in enumerate(self.dm_ids):
                if fid in self.policies:
                    p = self.policies[fid].get_action_probs_torch(obs_t[i]).numpy()
                else:
                    p = np.full((H * W, num_actions), 1.0 / num_actions, dtype=self.dtype)
                probs[i] = p

        probs = probs.transpose(0, 2, 1).reshape(self.N_dm, num_actions, H, W).astype(self.dtype, copy=False)

        # Build full mask (N_dm, num_actions, H, W)
        full_mask = np.ones_like(probs)
        full_mask[:, 0:4] = self.move_mask

        B_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0)
        prey_present = (B_all > 0).astype(self.dtype)
        full_mask[:, 5:5 + self.N_all] = self.eat_static_mask[:, :, None, None] * prey_present[None, :, :, :]

        # Indivisible-weight threshold: mask out move actions in cells where
        # the DM's biomass is below its minimum split mass. 0 = continuous
        # (no threshold). The existing zero-total fallback below routes such
        # cells to rest/eat via re-normalisation.
        if np.any(self.dm_min_split > 0):
            B_dm = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)  # (N_dm, H, W)
            can_split = (B_dm >= self.dm_min_split[:, None, None]).astype(self.dtype)
            full_mask[:, 0:4] *= can_split[:, None, :, :]

        # Disallow move actions entirely for DMs that cannot move
        # (movement_speed <= 0, e.g. zooplankton per spec). The zero-total
        # fallback below re-routes the freed probability mass to rest/eat.
        cannot_move = (self.dm_v <= 0)  # (N_dm,)
        if np.any(cannot_move):
            full_mask[cannot_move, 0:4] = 0.0

        masked = probs * full_mask
        total = masked.sum(axis=1, keepdims=True)
        zero_total = total <= 1e-12
        if np.any(zero_total):
            rest_slice = masked[:, 4:5]
            rest_slice = np.where(zero_total, np.float32(1.0), rest_slice)
            masked[:, 4:5] = rest_slice
            total = masked.sum(axis=1, keepdims=True)
        probs = masked / total

        self.pi_move = probs[:, 0:4]
        self.pi_rest = probs[:, 4]
        self.pi_eat  = probs[:, 5:5 + self.N_all]

        # Keep self.pi for non-DM consumers (always None entries here)
        self.pi = {fid: None for fid in self.fgs if not self.fgs[fid].is_decision_maker}

    # ---------- Predation (fully vectorized) ----------
    def _apply_predation(self):
        if self.N_dm == 0:
            return

        B_pred = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)  # (N_dm, H, W)
        hunger = np.stack([self.fgs[fid].get_hunger().astype(self.dtype, copy=False) for fid in self.dm_ids], axis=0)
        B_prey_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0)

        D = (
            B_pred[:, None, :, :]
            * self.pi_eat
            * self.max_intake_mat[:, :, None, None]
            * hunger[:, None, :, :]
        )

        total_demand = D.sum(axis=0)  # (N_all, H, W)
        scale = np.where(
            total_demand > B_prey_all,
            B_prey_all / (total_demand + np.float32(1e-9)),
            np.float32(1.0),
        ).astype(self.dtype, copy=False)
        actual = D * scale[None, :, :, :]

        gains = (actual * self.energy_gain_mat[:, :, None, None]).sum(axis=1)  # (N_dm, H, W)
        for i, fid in enumerate(self.dm_ids):
            self.fgs[fid].temp_energy_gains = gains[i]

        total_intake = actual.sum(axis=0)  # (N_all, H, W)
        eps = np.float32(1e-9)
        for j, prey_id in enumerate(self.global_fg_order):
            prey_fg = self.fgs[prey_id]
            B_old = prey_fg.biomass
            intake_j = total_intake[j]
            reduction = np.where(B_old > eps, (B_old - intake_j) / (B_old + eps), np.float32(0.0))
            prey_fg.energy_reserve = (prey_fg.energy_reserve * reduction).astype(self.dtype, copy=False)
            prey_fg.biomass = (B_old - intake_j).astype(self.dtype, copy=False)

    # ---------- Movement (vectorized + slice-assign) ----------
    def _apply_movement(self):
        if self.N_dm == 0:
            return

        H, W = self.H, self.W

        B = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)
        R = np.stack([self.fgs[fid].energy_reserve for fid in self.dm_ids], axis=0)
        TG = np.stack([self.fgs[fid].temp_energy_gains for fid in self.dm_ids], axis=0)

        # Per-DM extra energy cost from all impacts the FG is affected by
        # (sum of energy_factor lookups over every impact with a valid table).
        # The factor (1 + Σ energy_factor) scales resting / feeding / movement
        # metabolic costs.
        impact_energy = np.zeros((self.N_dm, H, W), dtype=self.dtype)
        for i, entries in enumerate(self.dm_impact_tables):
            for imp_id, table in entries:
                map_data = self.grid.get_map(imp_id)
                if map_data is None:
                    continue
                _, ef_map = self._interp_impact(table, map_data.astype(self.dtype, copy=False))
                impact_energy[i] += ef_map
        cost_factor = np.float32(1.0) + impact_energy
        rm = self.dm_resting_metabolism[:, None, None]
        cost_rest = self.dm_cost_rest[:, None, None]
        cost_eat = self.dm_cost_eat[:, None, None]
        cost_move = self.dm_cost_move[:, None, None]

        # Rest
        pi_rest = self.pi_rest
        m_rest = B * pi_rest * rm * cost_rest * cost_factor
        r_rest = np.maximum(0, R * pi_rest - m_rest)
        b_rest = B * pi_rest

        # Eat: pi_eat sums then uses shared scale (sum_j max(0, R*pi_j - B*pi_j*scale) = sum_j pi_j*max(0, R - B*scale))
        pi_eat_sum = self.pi_eat.sum(axis=1)
        scale_eat = rm * cost_eat * cost_factor
        r_eat_tot = pi_eat_sum * np.maximum(0, R - B * scale_eat) + TG
        b_eat_tot = B * pi_eat_sum

        # Move (4 directions)
        pi_move = self.pi_move
        scale_move = rm * cost_move * cost_factor
        r_after_move_meta = np.maximum(0, R - B * scale_move)
        r_move_choices = pi_move * r_after_move_meta[:, None, :, :]
        b_move_choices = pi_move * B[:, None, :, :]

        flux_b = self.dm_v[:, None, None, None]  # (N_dm,1,1,1)
        b_out = b_move_choices * flux_b
        r_out = r_move_choices * flux_b
        b_keep_from_move = b_move_choices - b_out
        r_keep_from_move = r_move_choices - r_out

        b_stay = b_rest + b_eat_tot + b_keep_from_move.sum(axis=1)
        r_stay = r_rest + r_eat_tot + r_keep_from_move.sum(axis=1)

        # Slice-assign neighbour transfers. Direction order: 0=N(-1,0), 1=E(0,+1), 2=S(+1,0), 3=W(0,-1)
        b_total_in = b_stay.copy()
        r_total_in = r_stay.copy()
        # N: dest[y] += source[y+1] -> dest[:-1] += source[1:]
        b_total_in[:, :-1, :] += b_out[:, 0, 1:, :]
        r_total_in[:, :-1, :] += r_out[:, 0, 1:, :]
        # E: dest[:, 1:] += source[:, :-1]
        b_total_in[:, :, 1:]  += b_out[:, 1, :, :-1]
        r_total_in[:, :, 1:]  += r_out[:, 1, :, :-1]
        # S: dest[1:] += source[:-1]
        b_total_in[:, 1:, :]  += b_out[:, 2, :-1, :]
        r_total_in[:, 1:, :]  += r_out[:, 2, :-1, :]
        # W: dest[:, :-1] += source[:, 1:]
        b_total_in[:, :, :-1] += b_out[:, 3, :, 1:]
        r_total_in[:, :, :-1] += r_out[:, 3, :, 1:]

        new_B = b_total_in
        max_R = new_B * self.dm_max_energy_reserve[:, None, None]
        new_R = np.clip(r_total_in, 0.0, max_R)

        for i, fid in enumerate(self.dm_ids):
            self.fgs[fid].biomass = new_B[i].astype(self.dtype, copy=False)
            self.fgs[fid].energy_reserve = new_R[i].astype(self.dtype, copy=False)

    @staticmethod
    def _extract_impact_table(impact_def):
        """Return (xs, biomass_factors, energy_factors) as float32 arrays sorted
        by xs, or None if the FG is not affected by this impact or the table is
        missing/empty."""
        if not impact_def:
            return None
        if not impact_def.get('impact_affects', False):
            return None
        table = impact_def.get('impact_table') or []
        if not table:
            return None
        try:
            rows = sorted(
                ((float(r['value']), float(r['biomass_factor']), float(r['energy_factor']))
                 for r in table),
                key=lambda t: t[0],
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not rows:
            return None
        xs = np.array([r[0] for r in rows], dtype=np.float32)
        bf = np.array([r[1] for r in rows], dtype=np.float32)
        ef = np.array([r[2] for r in rows], dtype=np.float32)
        return xs, bf, ef

    @staticmethod
    def _interp_impact(table, x):
        """Linear interpolation against an impact table. Outside the support,
        the nearest endpoint value is used (no extrapolation). `x` may be a
        scalar or ndarray; returns (biomass_factor, energy_factor) with the
        same shape as `x` (or scalars if `x` is scalar)."""
        xs, bf, ef = table
        x_arr = np.asarray(x, dtype=np.float32)
        b = np.interp(x_arr, xs, bf, left=bf[0], right=bf[-1])
        e = np.interp(x_arr, xs, ef, left=ef[0], right=ef[-1])
        return b.astype(np.float32, copy=False), e.astype(np.float32, copy=False)

    def _compute_impact_mortality(self, fg):
        """Return total impact-induced biomass loss m_X^Impact for the given FG
        as a per-cell array, or None if the FG has no active impact tables."""
        if 'impact' not in fg.params:
            return None
        total_mortality_impact = None
        for impact_id, impact_def in fg.params['impact'].items():
            map_data = self.grid.get_map(impact_id)
            if map_data is None:
                continue
            table = self._extract_impact_table(impact_def)
            if table is None:
                continue
            bf_map, _ = self._interp_impact(table, map_data)
            contribution = fg.biomass * bf_map
            if total_mortality_impact is None:
                total_mortality_impact = contribution
            else:
                total_mortality_impact = total_mortality_impact + contribution
        return total_mortality_impact

    def _apply_impact_mortality(self):
        """Method.pdf §steg 1–6: apply m_X^Impact before predation. Only
        decision-maker FGs carry impact tables in the current model."""
        for fg_id in self.ordered_fg_ids:
            fg = self.fgs[fg_id]
            if not fg.is_decision_maker:
                continue
            total_mortality_impact = self._compute_impact_mortality(fg)
            if total_mortality_impact is None:
                continue

            loss_mask = total_mortality_impact > 0
            reduction = np.ones_like(fg.biomass)
            reduction[loss_mask] = (
                (fg.biomass[loss_mask] - total_mortality_impact[loss_mask])
                / (fg.biomass[loss_mask] + 1e-9)
            )
            reduction = np.clip(reduction, 0.0, 1.0)

            fg.energy_reserve = (fg.energy_reserve * reduction).astype(self.dtype, copy=False)
            fg.biomass = np.maximum(
                0.0, fg.biomass - total_mortality_impact
            ).astype(self.dtype, copy=False)

    def _apply_growth(self):
        for fg_id in self.ordered_fg_ids:
            fg = self.fgs[fg_id]
            if not fg.is_decision_maker:
                cc = fg.params.get('max_carrying_capacity', 100.0)
                mg = fg.growth_rate
                growth = mg * fg.biomass * (1.0 - fg.biomass / (cc + 1e-9))
                fg.biomass = np.clip(fg.biomass + growth, 0.0, cc).astype(self.dtype, copy=False)
            else:
                s_x = fg.energy_level
                u_x = fg.maintenance_level
                q_x = s_x - u_x

                growth = fg.biomass * fg.growth_rate * q_x

                # Handle negative growth (shrinkage) as additional biomass loss
                # that also drains energy reserve proportionally.
                negative_growth = np.minimum(0.0, growth)
                total_loss = -negative_growth

                loss_mask = total_loss > 0
                reduction = np.ones_like(fg.biomass)
                reduction[loss_mask] = (
                    (fg.biomass[loss_mask] - total_loss[loss_mask])
                    / (fg.biomass[loss_mask] + 1e-9)
                )
                reduction = np.clip(reduction, 0.0, 1.0)

                fg.energy_reserve = (fg.energy_reserve * reduction).astype(self.dtype, copy=False)
                fg.biomass = np.maximum(0.0, fg.biomass + growth).astype(self.dtype, copy=False)
