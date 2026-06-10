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

        # Säsongsmodulering av NDM growth_rate. Per-FG opt-in via
        # `seasonal_amplitude` (fraktion av r) och `seasonal_period` (ticks).
        # r_eff(t) = r * (1 + AMP * sin(2π * (tick + phase) / PERIOD)).
        # Slumpad startfas per FG och env-instans så olika rollouter ser
        # olika säsongsfas. AMP=0 eller PERIOD<=0 ⇒ ingen modulering.
        self._season_phase = {
            fid: float(np.random.uniform(0.0, max(1.0, float(getattr(fg, 'seasonal_period', 0.0) or 0.0))))
            for fid, fg in self.fgs.items()
        }

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

        # Observability: per-DM list of j-indices (into ``global_fg_order``)
        # of OTHER FGs that this DM observes. The DM's own slot is handled
        # separately (always part of the observation via B_own / E_own) and
        # is NOT included in obs_others_idx. Non-observed FGs are
        # COMPLETELY REMOVED from this DM's input space — there is no
        # input slot allocated for them at all. This shrinks the per-DM
        # observation dimension. Because the batched bmm forward pass
        # requires a common D over all DMs, the observation tensor is
        # padded to ``self.max_in_dim`` and W1 weight columns for the
        # padding slots are zero (set in _rebuild_batched_weights), so
        # padded slots have no effect on any DM's network.
        #
        # When a DM's ``observes`` param is None (legacy projects without
        # an observability matrix) the DM sees ALL other FGs.
        self.obs_others_idx = []  # list[np.ndarray[int]], len N_dm
        for i, fid in enumerate(self.dm_ids):
            j_own = int(self.dm_index_in_all[i])
            observes = self.fgs[fid].params.get('observes')
            if observes is None:
                # Legacy: see all other FGs in stable global order.
                idx = [j for j in range(self.N_all) if j != j_own]
            else:
                observed_set = set(observes)
                idx = [j for j, other_id in enumerate(self.global_fg_order)
                       if j != j_own and other_id in observed_set]
            self.obs_others_idx.append(np.asarray(idx, dtype=np.int64))

        # Number of impact observation channels (matches train.py's
        # get_dynamic_policy_params formula). Stored here so observation
        # dim calculations stay consistent in one place.
        n_obs_imp = len(self.observable_impact_vars)
        # Per-DM input dimension. Each DM sees:
        #   center : [B_own, E_own, B_obs_others (k_i), impacts (n_obs_imp)]
        #   neighbour (N/E/S/W) : [B_own, B_obs_others (k_i), impacts (n_obs_imp)]
        # so center_dim_i = 2 + k_i + n_obs_imp and nbr_dim_i = 1 + k_i + n_obs_imp.
        self.per_dm_in_dim = np.zeros(self.N_dm, dtype=np.int64)
        for i in range(self.N_dm):
            k_i = int(self.obs_others_idx[i].shape[0])
            center_dim_i = 2 + k_i + n_obs_imp
            nbr_dim_i = 1 + k_i + n_obs_imp
            self.per_dm_in_dim[i] = center_dim_i + 4 * nbr_dim_i
        self.max_in_dim = int(self.per_dm_in_dim.max()) if self.N_dm > 0 else 0
        self.n_obs_imp = n_obs_imp

        # Hide-action support: ``Rest`` is now also a hide action. The
        # fraction of each DM's biomass that chose rest in cell (h, w)
        # last tick is invisible to every OTHER FG observing it and is
        # also protected from predation this tick (set in
        # _apply_predation using the *current* tick's pi_rest).
        #
        # ``prev_hidden_frac`` shape: (N_all, H, W). Row j corresponds to
        # global_fg_order[j]. NDMs have no action policy and therefore
        # never hide -> their rows stay 0. The observation builder uses
        # this to attenuate B_obs_others for other DMs; B_own stays full.
        self.prev_hidden_frac = np.zeros((self.N_all, self.H, self.W), dtype=self.dtype)

        self._rebuild_batched_weights()
        self._apply_accessibility_biomass_mask()
        self._static_built = True

    def _apply_accessibility_biomass_mask(self):
        """Keep biomass/energy off inaccessible cells when an accessibility map exists."""
        access_map = self.grid.get_map('accessibility')
        if access_map is None:
            return
        habitat = (access_map > 0).astype(self.dtype, copy=False)
        for fg in self.fgs.values():
            if fg.biomass is not None:
                fg.biomass = (fg.biomass * habitat).astype(self.dtype, copy=False)
            if fg.energy_reserve is not None:
                fg.energy_reserve = (fg.energy_reserve * habitat).astype(self.dtype, copy=False)
            if fg.temp_energy_gains is not None:
                fg.temp_energy_gains = (fg.temp_energy_gains * habitat).astype(self.dtype, copy=False)

    def _rebuild_batched_weights(self):
        """Stack per-DM PolicyNetwork weights into batched tensors for bmm-based forward.

        With variable per-DM in_dim (driven by the Observability matrix —
        non-observed FGs are removed from the input space entirely), each
        policy may have a different ``layers[0].in_features``. To keep the
        batched ``torch.bmm`` forward path we pad every W1 row-block to
        ``max_in_dim = max_i(in_dim_i)`` with zero columns at the bottom.
        The observation tensor is then padded to ``max_in_dim`` in
        _build_observation_batch (padding values are 0), so the padding
        slots contribute exactly 0 to each DM's hidden activation —
        semantically identical to having no input slot at all.

        Hidden width and out_dim must be uniform across DMs (they are, by
        construction in train.py). The *number* of hidden layers is
        arbitrary: we stack each Linear layer into its own batched
        (N_dm, in, out) weight tensor + (N_dm, out) bias, stored as lists
        ``self._Ws`` / ``self._bs`` in net order. The forward pass applies
        Sigmoid after every layer except the last (matching PolicyNetwork).
        """
        self._batched_ready = False
        self._Ws = None
        self._bs = None
        if self.N_dm == 0:
            return
        if not all(fid in self.policies for fid in self.dm_ids):
            return
        try:
            first = self.policies[self.dm_ids[0]]
            layers0 = [m for m in first.net if isinstance(m, torch.nn.Linear)]
            n_layers = len(layers0)
            if n_layers < 2:
                # Need at least one hidden + one output layer.
                return
            # Reference shapes from the first DM's net (excluding in_features
            # of layer 0, which varies per DM and is padded to max_in_dim).
            ref_shapes = [(lin.in_features, lin.out_features) for lin in layers0]
            out_dim = ref_shapes[-1][1]
            max_in_dim = int(self.max_in_dim)
            # Sanity: each policy must match ref_shapes except layer 0's
            # in_features, which must equal this DM's per_dm_in_dim[i].
            for i, fid in enumerate(self.dm_ids):
                lin = [m for m in self.policies[fid].net if isinstance(m, torch.nn.Linear)]
                if len(lin) != n_layers:
                    return
                if lin[0].in_features != int(self.per_dm_in_dim[i]):
                    return
                if lin[0].out_features != ref_shapes[0][1]:
                    return
                for k in range(1, n_layers):
                    if (lin[k].in_features != ref_shapes[k][0]
                            or lin[k].out_features != ref_shapes[k][1]):
                        return
            # Allocate batched tensors per layer.
            Ws = []
            bs = []
            for k, (in_k, out_k) in enumerate(ref_shapes):
                if k == 0:
                    W = torch.zeros(self.N_dm, max_in_dim, out_k)
                else:
                    W = torch.empty(self.N_dm, in_k, out_k)
                b = torch.empty(self.N_dm, out_k)
                Ws.append(W)
                bs.append(b)
            for i, fid in enumerate(self.dm_ids):
                lin = [m for m in self.policies[fid].net if isinstance(m, torch.nn.Linear)]
                in_dim_i = int(self.per_dm_in_dim[i])
                # Layer 0: pad W1 rows to max_in_dim (top in_dim_i rows used).
                Ws[0][i, :in_dim_i, :] = lin[0].weight.detach().t()
                bs[0][i] = lin[0].bias.detach()
                for k in range(1, n_layers):
                    Ws[k][i] = lin[k].weight.detach().t()
                    bs[k][i] = lin[k].bias.detach()
            self._Ws = Ws
            self._bs = bs
            # Legacy aliases (kept so external callers / debugging that read
            # _W1/_W2/_W3 still work for the default 2-hidden architecture).
            if n_layers == 3:
                self._W1, self._b1 = Ws[0], bs[0]
                self._W2, self._b2 = Ws[1], bs[1]
                self._W3, self._b3 = Ws[2], bs[2]
            self._in_dim = max_in_dim
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
        self._apply_accessibility_biomass_mask()

        self.tick_count += 1

    # ---------- Observation builder (batched across all DMs) ----------
    def _build_observation_batch(self):
        """Returns observation array of shape (N_dm, H*W, max_D), float32.

        Per cell each DM sees its von Neumann neighbourhood (center + 4
        neighbours N, E, S, W). Layout per DM ``i`` with ``k_i`` observed
        other FGs (from the Observability matrix) and ``n_obs_imp`` impact
        layers:

          center  : [B_own, E_own, B_obs_others (k_i), impacts (n_obs_imp)]
          N/E/S/W : [B_own,        B_obs_others (k_i), impacts (n_obs_imp)]

        ``E_own`` (energy fill ratio s_X) is only included for the center
        cell; neighbour energy levels are intentionally omitted. Non-observed
        FGs are COMPLETELY REMOVED from the layout (no slot allocated) —
        this is the semantic difference from the previous "0-mask" approach.

        Because ``torch.bmm`` requires a common D over all DMs, each DM's
        compact layout is written into the top ``per_dm_in_dim[i]`` slots
        of a ``(N_dm, max_in_dim, H, W)`` tensor; the remaining padding
        slots stay 0. W1's bottom rows are also 0 (see
        _rebuild_batched_weights), so padding contributes nothing.

        Out-of-bounds neighbour values are zero-padded. Neighbour order
        is N, E, S, W matching the AccessN/E/S/W convention in
        Method.pdf.
        """
        H, W = self.H, self.W
        B_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order],
                         axis=0).astype(self.dtype, copy=False)
        # Visible biomass: the rest-action is a hide action, so the
        # fraction of each FG that rested last tick is invisible to
        # observers. ``prev_hidden_frac`` is initialised to 0 (first tick
        # -> everything visible) and updated at the end of
        # _calculate_decisions. Self-observation (B_own) intentionally
        # uses the *full* biomass (the FG always sees its full mass,
        # hidden or not).
        B_visible_all = B_all * (np.float32(1.0) - self.prev_hidden_frac)
        impact_layers = []
        for iid in self.observable_impact_vars:
            m = self.grid.get_map(iid)
            if m is None:
                m = np.zeros((H, W), dtype=self.dtype)
            else:
                m = m.astype(self.dtype, copy=False)
            impact_layers.append(m)
        n_obs_imp = len(impact_layers)

        max_D = int(self.max_in_dim)
        obs = np.zeros((self.N_dm, max_D, H, W), dtype=self.dtype)

        # Helper: shift a 2-D field by one cell in the given direction with
        # zero padding for out-of-bounds neighbours.
        def _shift(field, direction):
            out = np.zeros_like(field)
            if direction == 'N':       # neighbour to the north of (y,x) is (y-1,x)
                out[1:, :] = field[:-1, :]
            elif direction == 'S':
                out[:-1, :] = field[1:, :]
            elif direction == 'E':
                out[:, :-1] = field[:, 1:]
            elif direction == 'W':
                out[:, 1:] = field[:, :-1]
            return out

        # Pre-shift impact layers once (shared across all DMs).
        impact_shifts = {d: [_shift(layer, d) for layer in impact_layers]
                         for d in ('N', 'E', 'S', 'W')}

        for i, pred_id in enumerate(self.dm_ids):
            pred_fg = self.fgs[pred_id]
            B_own = pred_fg.biomass.astype(self.dtype, copy=False)
            E_own = pred_fg.energy_level.astype(self.dtype, copy=False)
            obs_idx = self.obs_others_idx[i]  # j-indices into global_fg_order
            k_i = int(obs_idx.shape[0])

            # Compact "observed others" stack (k_i, H, W) using fancy
            # index. Uses VISIBLE biomass (hidden/rested fraction of
            # each observed FG is removed). B_own (this DM) is handled
            # separately below and uses the full biomass.
            if k_i > 0:
                B_obs = B_visible_all[obs_idx]
            else:
                B_obs = np.zeros((0, H, W), dtype=self.dtype)

            center_dim_i = 2 + k_i + n_obs_imp
            nbr_dim_i = 1 + k_i + n_obs_imp

            # --- Center ---
            obs[i, 0] = B_own
            obs[i, 1] = E_own
            if k_i > 0:
                obs[i, 2:2 + k_i] = B_obs
            for kk, layer in enumerate(impact_layers):
                obs[i, 2 + k_i + kk] = layer

            # --- Neighbours N, E, S, W ---
            for d_idx, direction in enumerate(('N', 'E', 'S', 'W')):
                base = center_dim_i + d_idx * nbr_dim_i
                obs[i, base] = _shift(B_own, direction)
                if k_i > 0:
                    for kk in range(k_i):
                        obs[i, base + 1 + kk] = _shift(B_obs[kk], direction)
                for kk, layer_shift in enumerate(impact_shifts[direction]):
                    obs[i, base + 1 + k_i + kk] = layer_shift

        return obs.transpose(0, 2, 3, 1).reshape(self.N_dm, H * W, max_D)

    # ---------- Batched policy inference ----------
    def _batched_policy_forward(self, obs_batch, return_logits=False):
        """obs_batch: torch tensor (N_dm, N, D) -> probs or logits (N_dm, N, out_dim).

        Generalised over arbitrary number of hidden layers. Mirrors
        PolicyNetwork.forward: Sigmoid after every Linear except the last,
        which produces raw logits.
        """
        Ws = self._Ws
        bs = self._bs
        n_layers = len(Ws)
        h = obs_batch
        for k in range(n_layers - 1):
            h = torch.sigmoid(torch.bmm(h, Ws[k]) + bs[k].unsqueeze(1))
        logits = torch.bmm(h, Ws[-1]) + bs[-1].unsqueeze(1)
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
                    # obs_t is padded along the feature axis to max_in_dim so
                    # that all DMs share a common (N_dm, H*W, max_in_dim)
                    # tensor for the batched fast-path. The per-DM policies,
                    # however, have layers[0].in_features == per_dm_in_dim[i]
                    # (the un-padded width). Slice the top per_dm_in_dim[i]
                    # feature slots — that's exactly where _build_observation_batch
                    # wrote this DM's real features; the rest are zero padding.
                    d_i = int(self.per_dm_in_dim[i])
                    lg = self.policies[fid].get_action_logits_torch(obs_t[i, :, :d_i]).numpy()
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

        # --- Update prev_hidden_frac for NEXT tick's observation. ---
        # ``Rest`` is a hide action: the fraction of each DM that chose
        # rest in cell (h, w) this tick is invisible to other FGs in
        # their next observation. Stored on the (N_all, H, W) grid;
        # NDM rows (no policy / no pi_rest) remain 0 -> always fully
        # visible. Predation in the current tick uses ``self.pi_rest``
        # directly (see _apply_predation), independent of this cache.
        self.prev_hidden_frac = np.zeros((self.N_all, self.H, self.W), dtype=self.dtype)
        for i, fid in enumerate(self.dm_ids):
            j = int(self.dm_index_in_all[i])
            self.prev_hidden_frac[j] = self.pi_rest[i].astype(self.dtype, copy=False)

    # ---------- Predation (fully vectorized) ----------
    def _apply_predation(self):
        if self.N_dm == 0:
            return

        B_pred = np.stack([self.fgs[fid].biomass for fid in self.dm_ids], axis=0)  # (N_dm, H, W)
        hunger = np.stack([self.fgs[fid].get_hunger().astype(self.dtype, copy=False) for fid in self.dm_ids], axis=0)
        B_prey_all = np.stack([self.fgs[fid].biomass for fid in self.global_fg_order], axis=0)

        # ``Rest`` is a hide action: the fraction of each DM-prey that
        # chose rest THIS tick is fully protected from predation. Build
        # a (N_all, H, W) hidden-fraction array using the current tick's
        # pi_rest for DMs (NDMs stay 0 -> always fully edible). Predators
        # can only attack the visible fraction. Note: this uses the
        # CURRENT tick's pi_rest (just computed in _calculate_decisions),
        # while the observation that drove those decisions used the
        # PREVIOUS tick's pi_rest (prev_hidden_frac) -- semantically:
        # observers see what was hidden last tick, while protection is
        # determined by the active hide-choice this tick.
        hidden_frac_now = np.zeros((self.N_all, self.H, self.W), dtype=self.dtype)
        for i, fid in enumerate(self.dm_ids):
            j = int(self.dm_index_in_all[i])
            hidden_frac_now[j] = self.pi_rest[i].astype(self.dtype, copy=False)
        visible_frac = np.float32(1.0) - hidden_frac_now
        B_prey_visible = B_prey_all * visible_frac

        a = self.max_intake_mat[:, :, None, None]  # (N_dm, N_all, 1, 1)
        if getattr(self, '_has_holling2', False):
            # Fix 2: Holling Type II. Effektiv attack-rate deflateras lokalt
            # av handling-tid * lokal byte-biomassa, så intake per predator
            # mättas i stället för att skena vid hög P.
            h = self.handling_time_mat[:, :, None, None]
            # Holling-II saturation uses the *visible* prey biomass:
            # hidden prey is functionally inaccessible this tick, so it
            # neither contributes to attack-rate saturation nor to total
            # available intake.
            Bp = B_prey_visible[None, :, :, :]
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
        # Demand is capped at the *visible* prey biomass: the hidden
        # (rested) fraction is protected entirely this tick.
        scale = np.where(
            total_demand > B_prey_visible,
            B_prey_visible / (total_demand + np.float32(1e-9)),
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
                # Säsongsmodulering: opt-in per NDM via FG-fälten
                # seasonal_amplitude och seasonal_period (se fgconfig).
                # r_eff = r * (1 + AMP * sin(2π * (tick + phase) / PERIOD)).
                # AMP=0 eller PERIOD<=0 ⇒ legacy (ingen modulering).
                amp = float(getattr(fg, 'seasonal_amplitude', 0.0) or 0.0)
                period = float(getattr(fg, 'seasonal_period', 0.0) or 0.0)
                if amp != 0.0 and period > 0.0:
                    phase = self._season_phase.get(fg_id, 0.0)
                    phase_t = (self.tick_count + phase) / period
                    season = 1.0 + amp * float(np.sin(2.0 * np.pi * phase_t))
                    mg = mg * season
                growth = mg * fg.biomass * (1.0 - fg.biomass / (cc + 1e-9))
                # Fix 1: rekolonisations-floor. Tillsätter en konstant andel
                # av cc per tick i alla celler så NDM aldrig kan utrotas
                # globalt (dvalceller / inflöde). seed_rate=0 ⇒ legacy.
                seed_rate = float(getattr(fg, 'seed_rate', 0.0) or 0.0)
                if seed_rate > 0.0:
                    # Per-cell, per-tick log-uniform multiplikator i [0.1, 10.0]
                    # modellerar lokal variabilitet i rekolonisation (dvalceller,
                    # advektion, sporpulser). Log-uniform => lika sannolikt att
                    # minska med faktor 10 som att öka med faktor 10; geometriskt
                    # medelvärde = 1.0 (symmetrisk runt ren konstant seed_rate).
                    seed_mult = np.power(
                        10.0,
                        np.random.uniform(-1.0, 1.0, size=fg.biomass.shape),
                    ).astype(np.float32, copy=False)
                    growth = growth + np.float32(seed_rate * cc) * seed_mult
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
