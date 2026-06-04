import numpy as np
import torch
from lib.world.grid import Grid
from lib.world.functional_group import FunctionalGroup

class EcosystemEnvironment:
    def __init__(self, grid_config, functional_groups, interactions, policies=None,
                 observable_impact_vars=None, apply_natural_mortality=True):
        self.grid = Grid(**grid_config)
        # När False appliceras inte den artificiella (densitetsoberoende)
        # natural_mortality-termen i _apply_growth. Default True = legacy.
        self.apply_natural_mortality = bool(apply_natural_mortality)
        self.fgs = functional_groups  # Dictionary: id -> FunctionalGroup
        self.interactions = interactions
        self.policies = policies or {} # Dictionary: id -> PolicyNetwork
        self.collect_action_diagnostics = True
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
        # Holling Type II handlingstid per (predator, prey). 0 = ren Type I
        # (linjär respons, legacy-beteende). >0 ger mättad respons:
        # intake = a*P / (1 + a*h*P) per predator-enhet.
        #
        # KALIBRERINGSANMÄRKNING: enligt Method.pdf är `max_intake_rate`
        # (a = I_XY) ett **fysiologiskt 6h-tak** för intag uttryckt som ton
        # byte per ton predator per tick — dvs maximalt fysiologiskt möjligt
        # intag vid mättad bytestillgång, INTE ett genomsnittligt fältintag.
        # För canonical Holling Type II ska därför `1/h` (asymptotiskt tak)
        # sammanfalla med `a`, vilket innebär att `handling_time = 1/a`.
        # För zoo→phyto: a=0.4 → h=2.5 (se fg_library.yaml). Då dämpar Type II
        # intaget mjukt upp mot fysiologiska maxet vid hög P, vilket är hela
        # poängen med Fix 2. Att sätta h << 1/a (t.ex. 0.1) ger 1/h >> a och
        # gör mättnaden effektivt avstängd vid biologiskt relevanta densiteter.
        # Litteraturens individnivå-snitt (Frost 1972, Mauchline 1998,
        # Kiørboe 2011) ligger på ~0.015–0.05 ton/(ton·tick) men avser
        # realiserade snitt, inte fysiologiska tak — `a` ska kalibreras mot
        # GER/gut-throughput-max, inte mot sustained grazing-snitt.
        handling_time = np.zeros((self.N_dm, self.N_all), dtype=self.dtype)
        for i, pred_id in enumerate(self.dm_ids):
            pred_fg = self.fgs[pred_id]
            menu = pred_fg.params.get('menu', [])
            interaction = pred_fg.params.get('interaction', {})
            # Predatorns generella maxintag per tick (ton byte / ton predator).
            # Tidigare hämtades värdet per (predator, prey) via I_XY-matrisen
            # i interaction_definitions; nu är det en egenskap hos predator-FG.
            pred_max_intake = float(pred_fg.params.get('max_intake_rate', 0.0))
            for j, prey_id in enumerate(self.global_fg_order):
                if prey_id not in menu:
                    continue
                inter_id = f'{pred_id}_preys_on_{prey_id}'
                inter_def = interaction.get(inter_id, {})
                if not inter_def.get('preys_on', True):
                    continue
                eat_static[i, j] = 1.0
                max_intake[i, j] = pred_max_intake
                # Assimilationsfaktor [0, 1]: bytets energiinnehåll multipliceras
                # med denna faktor vid energiupptag. Saknas i YAML → default 1.0
                # (full assimilation, bakåtkompatibelt).
                assim = inter_def.get('assimilation_factor', 1.0)
                try:
                    assim_f = float(assim) if assim not in (None, "") else 1.0
                except (TypeError, ValueError):
                    assim_f = 1.0
                if assim_f < 0.0:
                    assim_f = 0.0
                elif assim_f > 1.0:
                    assim_f = 1.0
                energy_gain[i, j] = float(inter_def.get('energy_gain', 0.0)) * assim_f
                handling_time[i, j] = float(inter_def.get('handling_time', 0.0))
        self.eat_static_mask = eat_static
        self.max_intake_mat = max_intake
        self.energy_gain_mat = energy_gain
        self.handling_time_mat = handling_time
        self._has_holling2 = bool(np.any(handling_time > 0.0))

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
        """Returns observation array of shape (N_dm, H*W, D), dtype float32.

        Per cell the policy sees the von Neumann neighbourhood (center + 4
        neighbours N, E, S, W) as prescribed by Method.pdf. Layout per cell:

          center : [B_own, E_own, B_others (N_all-1), impacts (n_obs_imp)]
          N      : [B_own,        B_others (N_all-1), impacts (n_obs_imp)]
          E      : same as N
          S      : same as N
          W      : same as N

        ``E_own`` (energy fill ratio s_X) is only included for the center cell;
        neighbour energy levels are intentionally omitted. Out-of-bounds
        neighbour values are zero-padded. Neighbour order is N, E, S, W
        matching the AccessN/E/S/W convention in Method.pdf.
        """
        H, W = self.H, self.W
        N_all = self.N_all
        B_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0).astype(self.dtype, copy=False)
        impact_layers = []
        for iid in self.observable_impact_vars:
            m = self.grid.get_map(iid)
            if m is None:
                m = np.zeros((H, W), dtype=self.dtype)
            else:
                m = m.astype(self.dtype, copy=False)
            impact_layers.append(m)
        n_obs_imp = len(impact_layers)

        center_dim = 2 + (N_all - 1) + n_obs_imp      # incl. E_own
        nbr_dim = 1 + (N_all - 1) + n_obs_imp         # excl. E_own
        D = center_dim + 4 * nbr_dim
        obs = np.zeros((self.N_dm, D, H, W), dtype=self.dtype)

        # Shift every layer once per tick. Previously each DM rebuilt the same
        # shifted "other FG" and impact layers repeatedly.
        shifted_B = np.zeros((4, N_all, H, W), dtype=self.dtype)
        shifted_B[0, :, 1:, :] = B_all[:, :-1, :]   # N
        shifted_B[1, :, :, :-1] = B_all[:, :, 1:]   # E
        shifted_B[2, :, :-1, :] = B_all[:, 1:, :]   # S
        shifted_B[3, :, :, 1:] = B_all[:, :, :-1]   # W
        if n_obs_imp:
            impacts = np.stack(impact_layers, axis=0).astype(self.dtype, copy=False)
            shifted_impacts = np.zeros((4, n_obs_imp, H, W), dtype=self.dtype)
            shifted_impacts[0, :, 1:, :] = impacts[:, :-1, :]
            shifted_impacts[1, :, :, :-1] = impacts[:, :, 1:]
            shifted_impacts[2, :, :-1, :] = impacts[:, 1:, :]
            shifted_impacts[3, :, :, 1:] = impacts[:, :, :-1]
        else:
            shifted_impacts = None

        for i, pred_id in enumerate(self.dm_ids):
            pred_fg = self.fgs[pred_id]
            B_own = pred_fg.biomass.astype(self.dtype, copy=False)
            E_own = pred_fg.energy_level.astype(self.dtype, copy=False)
            j_idx = int(self.dm_index_in_all[i])
            # Build (N_all - 1) "other FG" biomass stack in stable order.
            if N_all > 1:
                B_others = np.concatenate([B_all[:j_idx], B_all[j_idx + 1:]], axis=0)
            else:
                B_others = np.zeros((0, H, W), dtype=self.dtype)

            # --- Center ---
            obs[i, 0] = B_own
            obs[i, 1] = E_own
            if N_all > 1:
                obs[i, 2:2 + (N_all - 1)] = B_others
            for k, layer in enumerate(impact_layers):
                obs[i, 2 + (N_all - 1) + k] = layer

            # --- Neighbours N, E, S, W ---
            for d_idx in range(4):
                base = center_dim + d_idx * nbr_dim
                obs[i, base] = shifted_B[d_idx, j_idx]
                if N_all > 1:
                    obs[i, base + 1:base + 1 + (N_all - 1)] = np.concatenate(
                        [shifted_B[d_idx, :j_idx], shifted_B[d_idx, j_idx + 1:]],
                        axis=0,
                    )
                if shifted_impacts is not None:
                    obs[i, base + 1 + (N_all - 1):base + 1 + (N_all - 1) + n_obs_imp] = shifted_impacts[d_idx]

        return obs.transpose(0, 2, 3, 1).reshape(self.N_dm, H * W, D)

    # ---------- Batched policy inference ----------
    def _batched_policy_forward(self, obs_batch, return_logits=False):
        """obs_batch: torch tensor (N_dm, N, D) -> probs or logits (N_dm, N, out_dim)."""
        h = torch.sigmoid(torch.bmm(obs_batch, self._W1) + self._b1.unsqueeze(1))
        h = torch.sigmoid(torch.bmm(h, self._W2) + self._b2.unsqueeze(1))
        logits = torch.bmm(h, self._W3) + self._b3.unsqueeze(1)
        if return_logits:
            return logits
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
            # Var-golv: skydda mot lågvarianskanaler (obs_var spänner ~8
            # storleksordningar i mareld2; kanaler med var ~ 1e-5 får
            # annars ~300x förstärkning -> clip till +/-10 -> konstant
            # input -> noll gradient). Golvet 1e-2 håller std >= 0.1 och
            # låter välbeteende-kanaler (var >> 1e-2) passera oförändrade.
            std = np.sqrt(np.maximum(var, np.float32(1e-2)))
            obs_np = (obs_np - mean[:, None, :]) / std[:, None, :]
            obs_np = np.clip(obs_np, -10.0, 10.0).astype(self.dtype, copy=False)

        obs_t = torch.from_numpy(obs_np)

        # --- Build action validity mask BEFORE softmax ---
        # Shape: (N_dm, num_actions, H, W). 1 = valid, 0 = invalid.
        # Applying the mask pre-softmax (via additive -inf on logits) means
        # invalid actions get zero probability AND zero gradient -- so ARS
        # cannot mistakenly reinforce e.g. "eat where no prey exists".
        if self._batched_ready and obs_np.shape[-1] == self._in_dim:
            num_actions = self._out_dim
        else:
            num_actions = 5 + self.N_all

        full_mask = np.ones((self.N_dm, num_actions, H, W), dtype=self.dtype)
        full_mask[:, 0:4] = self.move_mask

        B_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0)
        prey_present = (B_all > 0).astype(self.dtype)
        full_mask[:, 5:5 + self.N_all] = self.eat_static_mask[:, :, None, None] * prey_present[None, :, :, :]

        # Sub-threshold cells (B < min_split_biomass) are NOT mask-restricted
        # to {rest, eat} any more — they keep access to all 11 actions in the
        # logits, but their post-softmax distribution is collapsed to a
        # one-hot argmax later in this function so the whole sub-threshold
        # group acts as a single unit (no splitting). See ``sub_thr_mask``
        # below for the deterministic collapse.
        if np.any(self.dm_min_split > 0):
            B_dm = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)
            sub_thr_mask = (B_dm > 0) & (B_dm < self.dm_min_split[:, None, None])
        else:
            sub_thr_mask = None

        cannot_move = (self.dm_v <= 0)
        if np.any(cannot_move):
            full_mask[cannot_move, 0:4] = 0.0

        # Ensure each (DM, cell) has at least one valid action; otherwise
        # force rest=1 in mask so the cell has a defined distribution.
        # full_mask layout: (N_dm, num_actions, H, W); reshape to put actions last.
        mask_dch = full_mask  # already (N_dm, A, H, W)
        any_valid = mask_dch.sum(axis=1, keepdims=True) > 0  # (N_dm, 1, H, W)
        if not np.all(any_valid):
            # Where no valid action -> set rest (index 4) to 1.
            mask_dch[:, 4:5] = np.where(any_valid, mask_dch[:, 4:5], np.float32(1.0))
            full_mask = mask_dch

        # Compute logits and apply mask additively (-inf where invalid).
        if self._batched_ready and obs_np.shape[-1] == self._in_dim:
            with torch.no_grad():
                logits_t = self._batched_policy_forward(obs_t, return_logits=True)  # (N_dm, H*W, A)
            logits = logits_t.numpy()
        else:
            logits = np.empty((self.N_dm, H * W, num_actions), dtype=self.dtype)
            for i, fid in enumerate(self.dm_ids):
                if fid in self.policies:
                    lg = self.policies[fid].get_action_logits_torch(obs_t[i]).numpy()
                else:
                    lg = np.zeros((H * W, num_actions), dtype=self.dtype)
                logits[i] = lg

        # logits: (N_dm, H*W, A) -> reshape to (N_dm, A, H, W).
        logits = logits.transpose(0, 2, 1).reshape(self.N_dm, num_actions, H, W).astype(self.dtype, copy=False)

        # Additive mask: 0 -> -inf, 1 -> 0.
        neg_inf = np.float32(-1e9)
        add_mask = np.where(full_mask > 0, np.float32(0.0), neg_inf).astype(self.dtype, copy=False)
        logits = logits + add_mask

        # Temperatur-annealing: hög T -> jämnare softmax (utforskning),
        # T=1 -> normal. Sätts av trainer via env.softmax_temperature.
        T = float(getattr(self, 'softmax_temperature', 1.0) or 1.0)
        if T != 1.0:
            logits = logits / np.float32(T)

        # Stable softmax along action axis.
        logits_max = np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits - logits_max)
        probs = e / np.sum(e, axis=1, keepdims=True)

        # Sub-threshold cells: collapse the action distribution to a one-hot
        # argmax so the whole group performs a single action (variant 2 of
        # the spawn/sub-threshold spec). The argmax is taken over the full
        # 11-action distribution including move directions, allowing small
        # groups to migrate as a unit instead of being forced into rest/eat.
        if sub_thr_mask is not None and np.any(sub_thr_mask):
            # (N_dm, H, W) -> broadcast over the action axis.
            argmax_idx = np.argmax(probs, axis=1)  # (N_dm, H, W)
            one_hot = np.zeros_like(probs)
            d_idx, h_idx, w_idx = np.where(sub_thr_mask)
            a_idx = argmax_idx[d_idx, h_idx, w_idx]
            one_hot[d_idx, a_idx, h_idx, w_idx] = np.float32(1.0)
            # Replace probs only in sub-threshold cells.
            mask3 = sub_thr_mask[:, None, :, :]
            probs = np.where(mask3, one_hot, probs)

        self.pi_move = probs[:, 0:4]
        self.pi_rest = probs[:, 4]
        self.pi_eat  = probs[:, 5:5 + self.N_all]

        if not getattr(self, 'collect_action_diagnostics', True):
            self.pi = {fid: None for fid in self.fgs if not self.fgs[fid].is_decision_maker}
            return

        # --- Action-entropy diagnostics ---
        # Compute per-DM mean Shannon entropy H(pi) averaged over cells with
        # any biomass for that DM (so empty cells, where the action choice is
        # irrelevant, don't dominate the mean). Logged via env._action_entropy.
        # probs shape: (N_dm, num_actions, H, W); already mask-normalised.
        eps = np.float32(1e-12)
        ent_cell = -np.sum(probs * np.log(probs + eps), axis=1)  # (N_dm, H, W)
        B_dm = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)
        active = (B_dm > 0).astype(self.dtype)  # (N_dm, H, W)
        active_sum = active.sum(axis=(1, 2))    # (N_dm,)
        ent_mean = np.where(
            active_sum > 0,
            (ent_cell * active).sum(axis=(1, 2)) / np.maximum(active_sum, 1.0),
            ent_cell.mean(axis=(1, 2)),
        )
        # Soft action-mass distribution (mean over active cells) for the
        # fraction of the population's action mass going to each category.
        # This reflects what the predation/movement modules actually use
        # (the soft pi_move / pi_rest / pi_eat distributions), unlike a
        # winner-takes-all argmax bookkeeping which can be misleading when
        # the distribution is spread out.
        n_act = probs.shape[1]
        max_entropy = float(np.log(n_act))
        # Per-cell category masses: move = sum over 4 move dirs,
        # rest = pi_rest, eat = sum over all prey eat-nodes.
        move_mass = probs[:, 0:4].sum(axis=1)            # (N_dm, H, W)
        rest_mass = probs[:, 4]                          # (N_dm, H, W)
        eat_mass  = probs[:, 5:5 + self.N_all].sum(axis=1)  # (N_dm, H, W)
        if not hasattr(self, '_action_entropy_sum') or self._action_entropy_sum is None:
            self._action_entropy_sum = np.zeros(self.N_dm, dtype=np.float64)
            self._action_entropy_count = 0
            self._action_active_ticks = np.zeros(self.N_dm, dtype=np.int64)
            self._action_move_frac = np.zeros(self.N_dm, dtype=np.float64)
            self._action_rest_frac = np.zeros(self.N_dm, dtype=np.float64)
            self._action_eat_frac = np.zeros(self.N_dm, dtype=np.float64)
            self._action_max_entropy = max_entropy
        self._action_entropy_count += 1
        # Per-DM accumulation: only count ticks where the FG has any biomass,
        # so mv+rs+et==1 per DM (since pi_move+pi_rest+pi_eat == 1 per cell)
        # and H_act is conditional entropy given the FG is alive somewhere.
        # _action_entropy_count is kept as a global tick counter for backward
        # compatibility.
        for i in range(self.N_dm):
            mask_i = active[i] > 0
            n_cells = int(mask_i.sum())
            if n_cells == 0:
                continue
            self._action_active_ticks[i] += 1
            self._action_entropy_sum[i] += float(ent_mean[i])
            inv = 1.0 / float(n_cells)
            self._action_move_frac[i] += float(move_mass[i][mask_i].sum()) * inv
            self._action_rest_frac[i] += float(rest_mass[i][mask_i].sum()) * inv
            self._action_eat_frac[i]  += float(eat_mass[i][mask_i].sum())  * inv

        # Keep self.pi for non-DM consumers (always None entries here)
        self.pi = {fid: None for fid in self.fgs if not self.fgs[fid].is_decision_maker}

    # ---------- Predation (fully vectorized) ----------
    def _apply_predation(self):
        if self.N_dm == 0:
            return

        B_pred = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)  # (N_dm, H, W)
        hunger = np.stack([self.fgs[fid].get_hunger().astype(self.dtype, copy=False) for fid in self.dm_ids], axis=0)
        B_prey_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0)

        a = self.max_intake_mat[:, :, None, None]  # (N_dm, N_all, 1, 1)
        if getattr(self, '_has_holling2', False):
            # Fix 2: Holling Type II. Effektiv attack-rate deflateras lokalt
            # av handling-tid * lokal byte-biomassa, så intake per predator
            # mättas i stället för att skena vid hög P.
            h = self.handling_time_mat[:, :, None, None]
            Bp = B_prey_all[None, :, :, :]
            a_eff = a / (1.0 + a * h * Bp)
        else:
            a_eff = a

        D = (
            B_pred[:, None, :, :]
            * self.pi_eat
            * a_eff
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
                # Fix 1: rekolonisations-floor. Tillsätter en konstant andel
                # av cc per tick i alla celler så NDM aldrig kan utrotas
                # globalt (dvalceller / inflöde). seed_rate=0 ⇒ legacy.
                seed_rate = float(getattr(fg, 'seed_rate', 0.0) or 0.0)
                if seed_rate > 0.0:
                    growth = growth + np.float32(seed_rate * cc)
                fg.biomass = np.clip(fg.biomass + growth, 0.0, cc).astype(self.dtype, copy=False)
            else:
                # Fix 3: densitetsoberoende naturlig mortalitet (senescens,
                # sjukdom, hidden predation). Appliceras före growth-termen
                # så den drar ner biomassan även när q_x > 0. natural_mortality=0
                # ⇒ legacy-beteende (ren energi-driven dynamik).
                nm = float(getattr(fg, 'natural_mortality', 0.0) or 0.0)
                if nm > 0.0 and self.apply_natural_mortality:
                    keep = np.float32(max(0.0, 1.0 - nm))
                    fg.energy_reserve = (fg.energy_reserve * keep).astype(self.dtype, copy=False)
                    fg.biomass = (fg.biomass * keep).astype(self.dtype, copy=False)

                s_x = fg.energy_level
                u_x = fg.maintenance_level
                q_x = s_x - u_x

                starvation_mortality = getattr(fg, 'starvation_mortality', None)
                if starvation_mortality is None:
                    growth = fg.biomass * fg.growth_rate * q_x
                    # Legacy: negative growth doubles as starvation loss.
                    negative_growth = np.minimum(0.0, growth)
                    total_loss = -negative_growth
                    biomass_delta = growth
                else:
                    # New model: growth_rate controls fed growth only, while
                    # starvation_mortality controls biomass loss below
                    # maintenance. At s_x=0, the full per-tick rate applies;
                    # at s_x=u_x, starvation loss is zero.
                    growth = fg.biomass * fg.growth_rate * np.maximum(0.0, q_x)
                    sm = np.float32(max(0.0, float(starvation_mortality)))
                    if u_x > 0.0 and sm > 0.0:
                        deficit = np.clip((u_x - s_x) / (u_x + 1e-9), 0.0, 1.0)
                        total_loss = fg.biomass * sm * deficit
                    else:
                        total_loss = np.zeros_like(fg.biomass)
                    biomass_delta = growth - total_loss

                loss_mask = total_loss > 0
                reduction = np.ones_like(fg.biomass)
                reduction[loss_mask] = (
                    (fg.biomass[loss_mask] - total_loss[loss_mask])
                    / (fg.biomass[loss_mask] + 1e-9)
                )
                reduction = np.clip(reduction, 0.0, 1.0)

                fg.energy_reserve = (fg.energy_reserve * reduction).astype(self.dtype, copy=False)
                fg.biomass = np.maximum(0.0, fg.biomass + biomass_delta).astype(self.dtype, copy=False)
