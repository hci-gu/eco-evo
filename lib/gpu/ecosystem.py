"""Ecological equations on fixed-shape tensors.

State fields are [ecosystem, functional_group, cell]. Actions are
[ecosystem, decision_maker, action, cell]. Only construction/fixture import
touch NumPy; every numerical method supports CPU tensors and CUDA tensors.
"""

import numpy as np
import torch
from lib.environments.ecosystem_env.constants import MAX_HARVEST_FRAC
from lib.environments.ecosystem_env.currents import direction_fractions
from lib.environments.ecosystem_env.impacts import extract_impact_table, interp_impact
from lib.world import nutrients

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
        self.currents = env.currents
        self.current_world_seed = env.current_world_seed
        self.holling = env._has_holling2 or env._has_holling3
        tensor = self.tensor
        self.dm_index = tensor(self.dm_positions, torch.long)
        self.ndm_index = tensor(self.ndm_positions, torch.long)
        # Drifting is per group, not per kind: a decision maker with
        # current_response > 0 is advected too, on top of the move its
        # policy chose. FunctionalGroup defaults the response so a
        # library without the key keeps the old decision-makers-never-
        # drift behaviour, which is why this set is usually the NDMs.
        self.drift_positions = tuple(
            i for i in range(self.G)
            if env.fgs[self.ids[i]].current_response > 0)
        self.drift_index = tensor(self.drift_positions, torch.long)
        self.current_response = tensor([
            env.fgs[self.ids[i]].current_response for i in self.drift_positions
        ])[None, :, None, None]
        self.is_dm = tensor([i in self.dm_positions for i in range(self.G)], torch.bool)[None, :, None]
        self.move_mask = tensor(env.move_mask.reshape(4, self.C))
        self.eat_mask = tensor(env.eat_static_mask)
        self.intake_rate = tensor(env.max_intake_mat)[None, :, :, None]
        self.energy_gain = tensor(env.energy_gain_mat)[None, :, :, None]
        # Prey whose reserve is eaten with it (section 140): the static
        # quality holds the reference fill, the tick adds the difference
        # to the cell's reserve per tonne.
        self.reserve_food = bool(getattr(env, "_has_reserve_food", False))
        if self.reserve_food:
            self.assimilation = tensor(env.assimilation_mat)[None, :, :, None]
            self.reserve_food_mask = tensor(env.prey_reserve_food)[None, :, None]
            self.reserve_reference = tensor(
                env.prey_reserve_fill * env.prey_reserve_max)[None, :, None]
        self.handling = tensor(env.handling_time_mat)[None, :, :, None]
        # Per-pair Holling type mask (section 141): [1, D, G, 1].
        self.type3 = tensor(env._type3_pred_mask.reshape(self.D, self.G))[None, :, :, None]
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
        # Section 208: per-pair accessible fraction of the prey (None =
        # every pair reaches all of it), mirror of
        # ``interactions.apply_pair_access``.
        self.pair_access = (
            tensor(env.access_mat)[None, :, :, None]
            if getattr(env, "_has_pair_access", False) else None)
        # Daylight calendar (section 137). ``light_obs`` is F per year
        # tick (the extra observation channel), ``light_mult`` the
        # (T, D, G) attack-rate multiplier, None when no pair is
        # modulated. Both are indexed per world by ``light_index`` =
        # (start + tick) % light_period, exactly as the reference's
        # ``interactions.light_index``.
        self.daylight = bool(getattr(env, "_has_daylight", False))
        self.light_pairs = bool(getattr(env, "_has_light_pairs", False))
        # The calendar also exists without the observation channel when
        # only producers are light-limited; ``calendar`` is the switch
        # for the year-tick index, ``daylight`` for the channel.
        self.calendar = getattr(env, "daylight", None) is not None
        self.light_period = int(env.light_period) if self.calendar else 1
        self.light_random_start = bool(
            self.calendar and env.daylight.get("random_start", False))
        self.light_fixed_start = int(env.light_start) if self.calendar else 0
        # Light-limited growth of producers (section 138): (T, G, 1)
        # multiplier on growth_rate, exactly 1 for every other group.
        self.growth_light_on = bool(getattr(env, "_has_growth_light", False))
        if self.growth_light_on:
            table = np.ones((self.light_period, self.G), dtype=np.float32)
            for j, fid in enumerate(self.ids):
                if fid in env.growth_light:
                    table[:, j] = env.growth_light[fid]
            self.growth_light = tensor(table)[:, :, None]
        else:
            self.growth_light = None
        # Nutrient pools of producers (section 172): the pool rides in the
        # producer's reserve slot. (T, G, 1) tables, exactly inert (keep 1,
        # supply 0) for every other group; ``nutrient`` masks the groups.
        self.nutrients_on = bool(getattr(env, "_has_nutrients", False))
        if self.nutrients_on:
            keep = np.ones((self.light_period, self.G), dtype=np.float32)
            supply = np.zeros((self.light_period, self.G), dtype=np.float32)
            half = np.ones((self.light_period, self.G), dtype=np.float32)
            regen = np.zeros(self.G, dtype=np.float32)
            stress_max = np.zeros(self.G, dtype=np.float32)
            stress_shape = np.ones(self.G, dtype=np.float32)
            stress_regen = np.zeros(self.G, dtype=np.float32)
            initial = np.zeros((self.light_period, self.G), dtype=np.float32)
            for j, fid in enumerate(self.ids):
                if fid not in env.nutrient_tables:
                    continue
                tables = env.nutrient_tables[fid]
                keep[:, j], supply[:, j], half[:, j] = (
                    tables["keep"], tables["supply"], tables["half"])
                regen[j] = env.nutrient_regeneration[fid]
                if fid in getattr(env, "nutrient_stress", {}):
                    stress_max[j], stress_shape[j] = env.nutrient_stress[fid]
                stress_regen[j] = getattr(
                    env, "nutrient_stress_regeneration", {}).get(fid, 0.0)
                initial[:, j] = [nutrients.initial_pool(
                    env.nutrient_config, env.nutrient_yield[fid], t,
                    getattr(env, "nutrient_layer", {}).get(fid, "mixed"))
                    for t in range(self.light_period)]
            self.nutrient = tensor(np.array(
                [fid in env.nutrient_tables for fid in self.ids]),
                torch.bool)[None, :, None]
            self.nutrient_keep = tensor(keep)[:, :, None]
            self.nutrient_supply = tensor(supply)[:, :, None]
            self.nutrient_half = tensor(half)[:, :, None]
            self.nutrient_regen = tensor(regen)[None, :, None]
            # Section 175: stress loss L_s c (1 - f)/(f + c); L_s 0 = none.
            self.stress_max = tensor(stress_max)[None, :, None]
            self.stress_shape = tensor(stress_shape)[None, :, None]
            self.stress_regen = tensor(stress_regen)[None, :, None]
            self.nutrient_initial = tensor(initial)[:, :, None]
        # Two-layer water column (section 180), mirror of
        # ``population_change.apply_layer_exchange``: (T, 1) tables of the
        # linear map between the mixed and the subsurface producer.
        ex = getattr(env, "layer_exchange", None)
        if ex:
            self.layer = {"m": self.ids.index(ex["mixed"]),
                          "d": self.ids.index(ex["subsurface"])}
            for key in ("n_mm", "n_md", "n_dm", "n_dd", "n_s",
                        "b_mm", "b_md", "b_dm", "b_dd"):
                self.layer[key] = tensor(np.asarray(ex[key], dtype=np.float32))[:, None]
        else:
            self.layer = None
        # Grazing non decision makers (section 177), mirror of
        # ``population_change.apply_ndm_grazing``: one entry per pair, in
        # the reference's order, with (T, 1) temperature / scale tables.
        self.ndm_grazers = []
        for entry in getattr(env, "ndm_grazers", []):
            period = self.light_period
            def table(values):
                if values is None:
                    return None
                return tensor(np.asarray(values, dtype=np.float32))[:, None]
            self.ndm_grazers.append({
                "g": self.ids.index(entry["grazer"]),
                "p": self.ids.index(entry["prey"]),
                "a": float(entry["a"]), "h": float(entry["h"]),
                "threshold": float(entry.get("threshold", 0.0)),
                "edibility": (None if entry.get("edibility") is None else
                              (float(entry["edibility"][0]),
                               table(entry["edibility"][1]))),
                "gge": float(entry["gge"]),
                "pool": (float(entry.get("pool", entry["regen"] * (1.0 - entry["gge"])))
                         if entry["prey"] in getattr(env, "nutrient_regeneration", {})
                         else 0.0),
                "temperature": table(entry["temperature"]),
                "scale": table(entry["scale"])})
        # Section 202, mirror of ``regeneration``: a share of each group's
        # M1 back to the mixed / subsurface pool, split by where it fed.
        self.regen_on = bool(getattr(env, "_has_regeneration", False))
        if self.regen_on:
            R = env.regen
            self.regen_share = tensor([R["shares"].get(f, 0.0) for f in self.ids])[None, :, None]
            self.regen_m = self.ids.index(R["mixed"])
            self.regen_d = None if R["subsurface"] is None else self.ids.index(R["subsurface"])
            self.regen_fallback = tensor(np.asarray(R["fallback"], dtype=np.float32))
            self.regen_grazers = [self.ids.index(g) for g in R["grazers"]]
            self.regen_groups = [self.ids.index(f) for f in self.ids if f in R["shares"]]
            # Host-side (row, group) pairs of the regenerating DMs: no
            # device readback inside a captured tick.
            self.regen_dm_rows = [(i, self.ids.index(f)) for i, f in enumerate(env.dm_ids)
                                  if f in R["shares"]]
        self.light_obs = tensor(env.light_obs_table) if self.daylight else None
        self.light_mult = (tensor(env.light_mult_table)[:, :, :, None]
                           if self.light_pairs else None)
        # Water temperature (section 143): (T, D) Q10 multiplier on the
        # metabolism, indexed by the same year tick as the light tables.
        self.temperature_on = bool(getattr(env, "_has_temperature", False))
        self.metabolism_temp = (tensor(env.metabolism_temp_table)
                                if self.temperature_on else None)
        # Section 181: (T, D) Q10 multiplier on each DM's whole intake.
        self.ingestion_temp = (
            tensor(env.ingestion_temp_table)
            if getattr(env, "_has_ingestion_temperature", False) else None)
        # Section 183: (T, G, 1) Q10 multiplier on each DM's growth rate,
        # 1 for every other group.
        self.growth_temp = None
        if getattr(env, "_has_growth_temperature", False):
            table = np.ones((self.light_period, self.G), dtype=np.float32)
            for i, fid in enumerate(env.dm_ids):
                table[:, self.ids.index(fid)] = env.growth_temp_table[:, i]
            self.growth_temp = tensor(table)[:, :, None]
        self.velocity = tensor(env.dm_v)[None, :, None, None]
        self.metabolism = tensor(env.dm_resting_metabolism)[None, :, None]
        self.cost_rest = tensor(env.dm_cost_rest)[None, :, None]
        self.cost_eat = tensor(env.dm_cost_eat)[None, :, None]
        self.cost_move = tensor(env.dm_cost_move)[None, :, None]
        self.can_move = tensor(env.dm_v > 0, torch.bool)[None, :, None, None]
        self.min_split = tensor(env.dm_min_split)[None, :, None]
        # Section 159 mirror of ``decisions.merge_subunit_fractions``.
        self.unit_splits = (bool(getattr(env, "unit_splits", True))
                            and bool((env.dm_min_split > 0).any()))
        self.fraction_indices = torch.arange(5, device=self.device)[None, None, :, None]
        self.action_indices = torch.arange(self.A, device=self.device)[None, None, :, None]

        groups = [env.fgs[fid] for fid in self.ids]
        def parameter(attr, default=0):
            return tensor([getattr(fg, attr, default) for fg in groups])[None, :, None]
        self.max_reserve = parameter("max_energy_reserve")
        self.growth = parameter("growth_rate")
        self.starve = parameter("starve_rate")
        self.starve = torch.where(self.starve > 0, self.starve, self.growth)
        # Section 180: a subsurface producer without the layer is inert.
        inert = getattr(env, "inert_producers", ())
        if inert:
            self.growth = torch.where(tensor(
                [fid in inert for fid in self.ids], torch.bool)[None, :, None],
                0.0, self.growth)
        self.maintenance = parameter("maintenance_level")
        # Section 184: income-funded growth (``movement.income_growth``).
        # kappa per DM in dm order; such a DM's reserve-funded branch is
        # off in ``population``.
        self.income_on = bool(getattr(env, "_has_income_growth", False))
        if self.income_on:
            self.income_kappa = tensor(env.dm_income_kappa)[None, :, None]
            self.income_off_growth = tensor(
                [fid in env.dm_ids and env.dm_income_kappa[env.dm_ids.index(fid)] > 0
                 for fid in self.ids], torch.bool)[None, :, None]
        # Section 185: regional logistic loss c * grid-mean B per decision
        # maker (``population_change``; c = 0 for every other group).
        self.capacity_on = bool(getattr(env, "_has_regional_capacity", False))
        if self.capacity_on:
            self.capacity_rate = tensor(
                [float(env.dm_capacity_rate[env.dm_ids.index(fid)])
                 if fid in env.dm_ids else 0.0 for fid in self.ids])[None, :, None]
        self.seed_rate = parameter("seed_rate").clamp_min(0)
        self.has_seeding = any(fg.seed_rate > 0 and not fg.is_decision_maker for fg in groups)
        self.capacity = tensor([fg.params.get("max_carrying_capacity", 100.0) for fg in groups])[None, :, None]
        if any(getattr(fg, "capacity_map", None) is not None for fg in groups):
            # Section 207: a depth-habitat group carries K per cell.
            self.capacity = tensor(np.stack([
                np.full(self.C, fg.params.get("max_carrying_capacity", 100.0), dtype=np.float32)
                if getattr(fg, "capacity_map", None) is None
                else np.asarray(fg.capacity_map, dtype=np.float32).reshape(-1)
                for fg in groups]))[None]
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
        # ``--mass-balance`` mirror of the reference's
        # ``_charge_growth_to_reserve``; False keeps ``population``
        # bit-identical to the pre-change kernel.
        self.mass_balance = bool(getattr(env, "mass_balance", False))
        mortality_scale = max(0.0, float(getattr(env, "mortality_multiplier", 1.0)))
        # Every group: a non decision maker's M1 (section 173) scales its
        # biomass before growth, never its reserve slot.
        self.mortality_keep = tensor([
            max(0.0, 1.0 - max(0.0, fg.natural_mortality) * mortality_scale)
            if env.apply_natural_mortality else 1.0
            for fg in groups
        ])[None, :, None]
        # Exposure-weighted M1 (section 139), mirroring
        # ``natural_mortality.components``: per group [1, G, 1] shares,
        # depth-risk ratio, floor and references, and a (T, G, 1) visual
        # light multiplier. Groups without it keep ``mortality_keep``.
        exposure = getattr(env, "m1_exposure", {}) or {}
        self.m1_exposure_on = bool(exposure) and env.apply_natural_mortality
        if self.m1_exposure_on:
            def column(key, default):
                return tensor([float(exposure[f][key]) if f in exposure
                               else default for f in self.ids])[None, :, None]
            self.m1_flag = tensor([f in exposure for f in self.ids],
                                  torch.bool)[None, :, None]
            self.m1_rate = 1.0 - self.mortality_keep
            self.m1_visual = column("visual", 0.0)
            self.m1_tactile = column("tactile", 0.0)
            self.m1_rho = column("rho", 1.0)
            self.m1_floor = column("floor", 0.0)
            self.m1_v_ref = column("v_ref", 1.0)
            self.m1_d_ref = column("d_ref", 1.0)
            light = np.ones((self.light_period, self.G), dtype=np.float32)
            for j, fid in enumerate(self.ids):
                table = exposure.get(fid, {}).get("light")
                if table is not None:
                    light[:, j] = table
            self.m1_light = tensor(light)[:, :, None]
            self.m1_light_on = any(exposure[f].get("light") is not None
                                   for f in exposure)
        # Impact pressure (section 161). Mortality tables for every group
        # (non decision makers too: bottom trawling on the benthos), cost
        # tables for the DMs only (the reference's ``dm_impact_tables``;
        # a group without actions has no cost to scale). A field belongs
        # to a world, so ``impact_fields`` turns the fields into per-cell
        # rates once per world on the host and the tick only multiplies.
        # No table keeps the tick bit-identical.
        self.impact_loss_tables = []
        for fid in self.ids:
            entries = []
            for impact_id, block in (env.fgs[fid].params.get("impact") or {}).items():
                table = extract_impact_table(block)
                if table is not None:
                    entries.append((impact_id, table))
            self.impact_loss_tables.append(entries)
        self.impact_tables = [list(entries) for entries in env.dm_impact_tables]
        self.impact_ids = tuple(sorted(
            {impact_id for entries in self.impact_loss_tables + self.impact_tables
             for impact_id, _ in entries}))
        self.has_impacts = bool(self.impact_ids)
        # Observable impacts (section 161): raw field values read into the
        # compact layout after the observed groups, in the own cell and in
        # each neighbour, exactly as ``observations.build_observation_batch``.
        self.observed_impacts = tuple(env.observable_impact_vars)
        self.n_obs_imp = len(self.observed_impacts)
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
        #   [biomass (G)] [visible] [energy_level (G)] [light] [impacts]
        pair = self.pair_visibility is not None
        visible_base = self.G
        energy_base = self.G + (self.D * self.G if pair else self.G)
        impact_base = energy_base + self.G + (1 if self.daylight else 0)
        impacts = [impact_base + k for k in range(self.n_obs_imp)]
        for d, own in enumerate(self.dm_positions):
            others = list(env.obs_others_idx[d])
            visible = [visible_base + (d * self.G + j if pair else j)
                       for j in others]
            channels = [own, energy_base + own] + visible + impacts
            for k, channel in enumerate(channels):
                obs_indices[d, :, k] = channel * self.C + cells
                obs_valid[d, :, k] = 1.0
            offset = len(channels)
            for direction in range(4):
                for channel in [own] + visible + impacts:
                    obs_indices[d, :, offset] = channel * self.C + neighbors[direction]
                    obs_valid[d, :, offset] = valid[direction]
                    offset += 1
            if self.daylight:
                # One broadcast light channel after the energy block, read
                # into the last slot of the DM's compact layout.
                obs_indices[d, :, offset] = (energy_base + self.G) * self.C + cells
                obs_valid[d, :, offset] = 1.0
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
        return (self.tensor(b).flatten(2), self.tensor(r).flatten(2),
                self.tensor(h).flatten(2))

    def impact_fields(self, maps):
        """Per-world impact loss rate [N, G, C] and cost surcharge [N, D, C].

        ``maps`` holds one ``{impact_id: (H, W) field}`` per world. Each
        group sums, over the impacts it is affected by, its tables'
        ``biomass_factor`` (the fraction of standing biomass removed per
        tick, ``impacts.compute_impact_mortality``); each DM also sums the
        ``energy_factor`` (the extra multiple of every action cost,
        ``impacts.impact_energy_cost_factor``). A missing field is skipped,
        as there. Host side, once per world, never inside a tick.
        """
        loss = np.zeros((len(maps), self.G, self.C), dtype=np.float32)
        cost = np.zeros((len(maps), self.D, self.C), dtype=np.float32)
        for w, fields in enumerate(maps):
            for rows, out, column in ((self.impact_loss_tables, loss, 0),
                                      (self.impact_tables, cost, 1)):
                for i, entries in enumerate(rows):
                    for impact_id, table in entries:
                        field = fields.get(impact_id)
                        if field is None:
                            continue
                        factors = interp_impact(
                            table, np.asarray(field, dtype=np.float32).reshape(-1))
                        out[w, i] += factors[column]
        return self.tensor(loss), self.tensor(cost)

    def impact_observations(self, maps):
        """Observable impact fields per world, [N, n_obs_imp, C].

        Raw values, a missing field as zeros, in ``observable_impact_vars``
        order (``impacts.observable_impact_layers``).
        """
        out = np.zeros((len(maps), self.n_obs_imp, self.C), dtype=np.float32)
        for w, fields in enumerate(maps):
            for k, impact_id in enumerate(self.observed_impacts):
                field = fields.get(impact_id)
                if field is not None:
                    out[w, k] = np.asarray(field, dtype=np.float32).reshape(-1)
        return self.tensor(out)

    def import_impact_observations(self, environments):
        """Observable impact fields of reference envs (fixture upload)."""
        return self.impact_observations([
            {impact_id: env.grid.get_map(impact_id) for impact_id in self.observed_impacts
             if env.grid.get_map(impact_id) is not None}
            for env in environments])

    def import_impacts(self, environments):
        """Impact fields of reference envs (fixture upload)."""
        maps = []
        for env in environments:
            fields = {}
            for impact_id in self.impact_ids:
                field = env.grid.get_map(impact_id)
                if field is not None:
                    fields[impact_id] = field
            maps.append(fields)
        return self.impact_fields(maps)

    def impact_mortality(self, biomass, reserve, loss_rate):
        """Direct impact mortality, before predation (Method.pdf 1-6).

        Mirrors ``impacts.apply_impact_mortality``: the loss is
        ``B * loss_rate`` per cell of every group ([E, G, C]), the reserve
        shrinks with the biomass, and the cell cannot go below zero.
        """
        loss = biomass * loss_rate
        reduction = torch.where(loss > 0, ((biomass - loss) / (biomass + 1e-9)).clamp(0.0, 1.0), 1.0)
        if self.nutrients_on:
            # A producer's nutrient pool is not removed with it (172).
            reduction = torch.where(self.nutrient, 1.0, reduction)
        return (biomass - loss).clamp_min(0), reserve * reduction

    def energy_level(self, biomass, reserve):
        safe_b = torch.where(biomass > 1e-9, biomass, 1.0)
        safe_max = torch.where(self.max_reserve != 0, self.max_reserve, 1.0)
        return torch.where((biomass > 1e-9) & (self.max_reserve != 0),
                           reserve / safe_b / safe_max, 0.0)

    def initial_nutrients(self, reserve, light_start):
        """``reserve`` with every producer's pool set to the surface
        climatology of each world's start tick (section 172); the CPU
        environment does the same in ``initialise_nutrient_pools``."""
        if not self.nutrients_on:
            return reserve
        pools = self.nutrient_initial[light_start.reshape(-1)]   # [E, G, 1]
        return torch.where(self.nutrient, pools.expand_as(reserve), reserve)

    def light_starts(self, environments):
        """Per-world year tick of the reference envs (fixture upload)."""
        return self.tensor([getattr(e, "light_start", 0) for e in environments],
                           torch.long)

    def light_index(self, light_start, tick):
        """Year tick per world, or None without the daylight calendar."""
        if not self.calendar:
            return None
        if light_start is None:
            light_start = torch.zeros((), dtype=torch.long, device=self.device)
        return torch.remainder(light_start + tick, self.light_period)

    def observations(self, biomass, reserve, hidden, light_index=None,
                     impact_obs=None):
        """Compact per-DM observations [E, D, C, F].

        ``impact_obs`` [E, n_obs_imp, C] holds the observable impact fields
        (``impact_observations``); None reads them as zero, as the
        reference does for a missing map.
        """
        if self.pair_visibility is None:
            visible = biomass * (1.0 - hidden * (1.0 - self.visibility))
        else:
            # [E, D, G, C] -> D consecutive blocks of G channels, indexed by
            # ``_build_indices`` so every observer reads its own row.
            visible = (biomass[:, None] * (1.0 - hidden[:, None]
                                           * (1.0 - self.pair_visibility))).flatten(1, 2)
        parts = [biomass, visible, self.energy_level(biomass, reserve)]
        if self.daylight:
            if light_index is None:
                raise ValueError("light_index is required with the daylight calendar")
            light = self.light_obs[light_index].reshape(-1, 1, 1)
            parts.append(light.expand(biomass.shape[0], 1, self.C))
        if self.n_obs_imp:
            if impact_obs is None:
                impact_obs = torch.zeros((biomass.shape[0], self.n_obs_imp, self.C),
                                         device=self.device)
            parts.append(impact_obs)
        bank = torch.cat(parts, dim=1)
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
        probs = torch.where(small[:, :, None], one_hot, probs)
        if self.unit_splits:
            probs = self.merge_subunit_fractions(probs, b_dm)
        return probs

    def merge_subunit_fractions(self, probs, b_dm):
        # A cell holding >= one unit never hands off less than one unit:
        # sub-unit spatial fractions (stay = rest + eat, and the four
        # moves) go to the fractions holding one, proportionally; with
        # none such the largest takes the cell (first index on ties).
        stay = probs[:, :, 4:].sum(2, keepdim=True)
        fractions = torch.cat((probs[:, :, :4], stay), dim=2)
        unit = self.min_split[:, :, None]
        keep = fractions * b_dm[:, :, None] >= unit
        top = self.fraction_indices == fractions.argmax(dim=2)[:, :, None]
        keep = torch.where(keep.any(dim=2, keepdim=True), keep, top)
        kept = fractions * keep.to(probs.dtype)
        kept = kept / kept.sum(2, keepdim=True).clamp_min(1e-30)
        scale = kept[:, :, 4:] / stay.clamp_min(1e-30)
        merged = torch.cat((kept[:, :, :4], probs[:, :, 4:] * scale), dim=2)
        active = (self.min_split > 0) & (b_dm >= self.min_split)
        return torch.where(active[:, :, None], merged, probs)

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

    def predation(self, biomass, reserve, actions, light_index=None):
        b_dm = biomass[:, self.dm_index]
        # h = min(1, max(0, (1 - s) / (1 - u))), as FunctionalGroup.get_hunger
        # (section 133); u >= 1 closes the window.
        u_dm = self.maintenance[:, self.dm_index]
        hunger = torch.where(
            u_dm < 1.0,
            ((1.0 - self.energy_level(biomass, reserve)[:, self.dm_index])
             / (1.0 - u_dm).clamp_min(1e-12)).clamp(0.0, 1.0),
            torch.zeros_like(u_dm))
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
        if self.pair_access is not None:
            base = pair_visible if pair_visible is not None else visible[:, None]
            pair_visible = base * self.pair_access
            visible = torch.where(self.eat_mask[None, :, :, None] > 0.0,
                                  pair_visible, 0.0).amax(1)
        # Interference term I = w_X * B_X(c): the predator's own biomass in
        # the cell, intraspecific only, shaped (B, D, 1, C) so it broadcasts
        # over the prey axis exactly as the reference does.
        inter = self.interference * b_dm[:, :, None] if self.has_interference else 0.0
        # Attack rate of this tick: [1, D, G, 1], or [B, D, G, 1] when the
        # daylight calendar modulates a pair (section 137).
        a = self.intake_rate
        if self.light_pairs:
            if light_index is None:
                raise ValueError("light_index is required with the daylight calendar")
            a = a * self.light_mult[light_index.reshape(-1)]
        if self.holling:
            prey = pair_visible if pair_visible is not None else visible[:, None]
            squared = prey * prey
            type2 = a * prey / (1.0 + a * self.handling * prey + inter)
            type3 = a * squared / (1.0 + a * self.handling * squared + inter)
            rate = self.type3 * type3 + (1.0 - self.type3) * type2
        elif self.has_interference:
            rate = a / (1.0 + inter)
        else:
            rate = a
        if self.ingestion_temp is not None:
            if light_index is None:
                raise ValueError("light_index is required with ingestion_q10")
            rate = rate * self.ingestion_temp[light_index.reshape(-1)][:, :, None, None]
        eat = actions[:, :, 5:]
        demand = b_dm[:, :, None] * eat * rate * hunger[:, :, None]
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
        if self.reserve_food:
            density = torch.where(biomass > 1e-9,
                                  reserve / biomass.clamp_min(1e-9), 0.0)
            offset = (density - self.reserve_reference) * self.reserve_food_mask
            gains = gains + (actual * self.assimilation
                             * offset[:, None]).sum(2)
        intake = actual.sum(1)
        reduction = torch.where(biomass > 1e-9, (biomass - intake) / (biomass + 1e-9), 0.0)
        reserve_after = reserve * reduction
        if self.nutrients_on:
            # Grazing leaves a producer's nutrient pool in place and
            # excretes a fraction of what it removed back into it (172).
            reserve_after = torch.where(
                self.nutrient, reserve + self.nutrient_regen * intake,
                reserve_after)
        return biomass - intake, reserve_after, gains, hidden, actual

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

    def movement(self, biomass, reserve, gains, actions, track=False,
                 light_index=None, impact_cost=None):
        """Move, split and settle; ``track`` also returns the source flow.

        With ``track`` the third return value is
        ``(b_stay, b_out, b_total)``, the biomass that remained in each
        source cell, what left it per direction, and the resulting
        per-cell biomass. The local reward (``--local_reward``) needs all
        three to attribute the destination cells' end-of-tick energy back
        to the cell the population started in.

        ``light_index`` is the year tick per world; the water temperature
        (section 143) scales every action cost by its Q10 multiplier.
        ``impact_cost`` [E, D, C] is the impacts' summed ``energy_factor``,
        which scales them by ``1 + impact_cost`` per cell (section 161).
        """
        b, r = biomass[:, self.dm_index], reserve[:, self.dm_index]
        metabolism = self.metabolism
        if self.temperature_on:
            if light_index is None:
                raise ValueError("light_index is required with the water "
                                 "temperature")
            metabolism = (metabolism
                          * self.metabolism_temp[light_index.reshape(-1)][:, :, None])
        if impact_cost is not None:
            metabolism = metabolism * (1.0 + impact_cost)
        move, rest = actions[:, :, :4], actions[:, :, 4]
        eat = actions[:, :, 5:].sum(2)
        move_fraction = move.sum(2)
        rest_r = (r * rest - b * rest * metabolism * self.cost_rest).clamp_min(0)
        eat_r = (r * eat - b * eat * metabolism * self.cost_eat).clamp_min(0) + gains
        move_r = (r * move_fraction - b * move_fraction * metabolism * self.cost_move).clamp_min(0)
        safe = torch.where(move_fraction > 0, move_fraction, 1.0)
        moving_r = move / safe[:, :, None] * move_r[:, :, None]
        moving_b = move * b[:, :, None]
        b_out, r_out = moving_b * self.velocity, moving_r * self.velocity
        if self.migration and self.split_thr_any:
            # What is taken out of the off-grid outflow lands in the stay
            # terms below by construction (moving minus outflow).
            b_out, r_out = self.cancel_emigration(b_out, r_out)
        stay_b, stay_r = b * rest + b * eat, rest_r + eat_r
        if self.income_on:
            stay_b, stay_r = self.income_growth(b, r, gains, metabolism, rest,
                                                eat, move_fraction, stay_b, stay_r)
        b_stay = stay_b + (moving_b - b_out).sum(2)
        b_total = b_stay
        r_total = stay_r + (moving_r - r_out).sum(2)
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
            b_total, r_total, residue, b_push = self.follow_residues(
                b_out - b_cancel, b_total, r_total)
        r_total = torch.minimum(r_total.clamp_min(0), b_total * self.max_reserve[:, self.dm_index])
        settled = (biomass.index_copy(1, self.dm_index, b_total),
                   reserve.index_copy(1, self.dm_index, r_total))
        if not track:
            return settled[0], settled[1], None
        if b_cancel is not None:
            # Cancelled splits stayed home and followed residues left, so
            # the tracked flow moves the former from the outflow back into
            # the stay term and the latter the other way; only then does
            # it still sum to ``b_total``.
            b_stay = torch.where(residue, 0.0, b_stay + b_cancel.sum(2))
            b_out = b_out - b_cancel + b_push
        return settled[0], settled[1], (b_stay, b_out, b_total)

    def income_growth(self, b, r, gains, metabolism, rest, eat, move_fraction,
                      stay_b, stay_r):
        """Section 184 mirror of ``movement.income_growth``: per cell
        G = min(kappa I - C, B g (s - u) ec), at most the settling reserve,
        moved from the stationary reserve into G / ec new tonnes."""
        cost = (b * rest * metabolism * self.cost_rest
                + b * eat * metabolism * self.cost_eat
                + b * move_fraction * metabolism * self.cost_move)
        growth = self.growth[:, self.dm_index]
        maintenance = self.maintenance[:, self.dm_index]
        ec = self.energy_content[:, self.dm_index][:, :, None]
        capacity = b * self.max_reserve[:, self.dm_index]
        fill = torch.where(capacity > 0, r / torch.where(capacity > 0, capacity, 1.0), 0.0)
        ceiling = b * growth * (fill - maintenance).clamp_min(0) * ec
        paid = torch.minimum((self.income_kappa * gains - cost).clamp_min(0), ceiling)
        paid = torch.minimum(paid, stay_r.clamp_min(0))
        paid = torch.where((self.income_kappa > 0) & (ec > 0), paid, 0.0)
        return (stay_b + paid / torch.where(ec > 0, ec, 1.0), stay_r - paid)

    def cancel_emigration(self, b_out, r_out):
        """Keep a DM's off-grid outflow home when its pool is below thr.

        Section 156, mirrored from
        ``movement.cancel_subthreshold_emigration``: a pool below ``thr``
        has no viable edge cell, so ``immigration`` would spread it over
        the whole edge and ``population`` would sweep all of it. Returns
        ``(b_out, r_out)`` with those outflows removed; the caller's stay
        terms (moving minus outflow) pick them up.
        """
        b_emig = (b_out * self.outside).sum((2, 3))
        small = (b_emig > 0) & (b_emig < self.split_thr[:, :, 0, 0])
        back = self.outside * small[:, :, None, None]
        return b_out - b_out * back, r_out - r_out * back

    def follow_residues(self, b_out, b_total, r_total):
        """Move a sub-threshold residue along its remaining outflows.

        Section 156, mirrored from
        ``movement.follow_subthreshold_residues``. ``b_out`` is the
        outflow left after ``suppress_splits``. A cell with ``0 < b_total
        < thr`` that still sends biomass to an in-grid neighbour hands
        all of its biomass and reserve to those neighbours, in proportion
        to what it sends each. Returns ``(b_total, r_total, residue,
        b_push)`` with ``b_push`` the extra outflow per direction.
        """
        in_grid = b_out * self.neighbor_valid
        sent = in_grid.sum(2)
        residue = (b_total > 0) & (b_total < self.split_thr[:, :, :, 0]) & (sent > 0)
        share = in_grid / torch.where(residue, sent, 1.0)[:, :, None]
        b_push = share * torch.where(residue, b_total, 0.0)[:, :, None]
        r_push = share * torch.where(residue, r_total, 0.0)[:, :, None]
        b_total = torch.where(residue, 0.0, b_total)
        r_total = torch.where(residue, 0.0, r_total)
        for direction in range(4):
            source = (direction + 2) % 4
            index, valid = self.neighbors[source], self.neighbor_valid[source]
            b_total = b_total + b_push[:, :, direction, index] * valid
            r_total = r_total + r_push[:, :, direction, index] * valid
        return b_total, r_total, residue, b_push

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

    def tracked_energy(self, flow, biomass, reserve, drift=None):
        """End-of-tick energy attributed back to the source cell.

        ``tracked[e, i, c]`` is ``B(c, t+1)`` of the local reward: the
        share of the plus-shaped destination set ``{c, N, E, S, W}`` that
        the population starting in ``c`` is entitled to. Everything after
        the movement is cell-wise multiplicative, so splitting a
        destination cell's energy in proportion to the biomass each
        source delivered is an identity, not an approximation.

        ``drift`` is ``advect(..., track=True)``'s third value. The
        current runs after the movement and is linear and uniform within
        a cell, so it composes with the move shares rather than breaking
        them: each destination's unit value is first pushed back through
        the field's shares. Mirrors ``source_tracking``.

        ``frac_in`` is the fraction of the source cell's post-movement
        biomass that stayed on the grid. Outflow across the border has no
        in-grid destination, so it is removed from the denominator (``A``
        is scaled by ``frac_in``) instead of being booked as a loss.
        """
        b_stay, b_out, b_total = flow
        b_final = b_total if drift is None else drift[1]
        unit = torch.where(b_final > 0.0, self.local_energy(biomass, reserve)
                           / b_final.clamp_min(1e-30), 0.0)
        if drift is not None:
            phi = drift[0]
            carried = torch.zeros_like(unit)
            for direction in range(4):
                index, valid = self.neighbors[direction], self.neighbor_valid[direction]
                carried = carried + phi[:, :, direction] * valid * unit[:, :, index]
            unit = (1.0 - phi.sum(2)) * unit + carried
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

    def mortality_keep_field(self, hidden, light_index=None):
        """Per-cell survival of M1 this tick: [B, G, C] or [1, G, 1]."""
        if not self.m1_exposure_on:
            return self.mortality_keep
        light = 1.0
        if self.m1_light_on:
            if light_index is None:
                raise ValueError("light_index is required with visual M1")
            light = self.m1_light[light_index.reshape(-1)]
        visible = (1.0 - hidden * (1.0 - self.m1_floor)) * light
        deep = (1.0 - hidden) + self.m1_rho * hidden
        factor = ((1.0 - self.m1_visual - self.m1_tactile)
                  + self.m1_visual * visible / self.m1_v_ref
                  + self.m1_tactile * deep / self.m1_d_ref)
        keep = 1.0 - (self.m1_rate * factor).clamp(0.0, 1.0)
        return torch.where(self.m1_flag, keep, self.mortality_keep)

    def ndm_grazing(self, biomass, reserve, light_index=None):
        """Section 177: grazing non decision makers, before the rest of
        the population step (mirror of ``apply_ndm_grazing``)."""
        if not self.ndm_grazers:
            return biomass, reserve, []
        biomass, reserve = biomass.clone(), reserve.clone()
        t = None if light_index is None else light_index.reshape(-1)
        totals = []
        for entry in self.ndm_grazers:
            g, p = entry["g"], entry["p"]
            prey, grazer = biomass[:, p], biomass[:, g]
            a, threshold = entry["a"], entry["threshold"]
            if entry["scale"] is not None:
                a = a * entry["scale"][t]
                threshold = threshold / entry["scale"][t]
            food = prey
            if entry["edibility"] is not None:
                if t is None:
                    raise ValueError("light_index is required with edibility")
                floor, half = entry["edibility"]
                pool = reserve[:, p].clamp_min(0)
                food = prey * (floor + (1.0 - floor) * pool / (pool + half[t]))
            if entry["threshold"] > 0.0:
                food = food * food / (food + threshold)
            rate = a * food / (1.0 + a * entry["h"] * food)
            if entry["temperature"] is not None:
                rate = rate * entry["temperature"][t]
            intake = torch.minimum(grazer * rate, HARVEST_FRACTION * prey
                                   ).clamp_min(0)
            biomass[:, p] = prey - intake
            biomass[:, g] = grazer + entry["gge"] * intake
            if entry["pool"]:
                reserve[:, p] = reserve[:, p] + entry["pool"] * intake
            totals.append(intake.sum(-1))
        return biomass, reserve, totals

    def regen_fraction(self, dm_feed, ndm_totals, light_index, batch):
        """Mixed-layer share [B, G] of each group's feeding this tick
        (mirror of ``regeneration.fraction``; 1 without the subsurface
        layer, the thickness share for a group that fed on neither)."""
        B = batch
        if self.regen_d is None:
            return torch.ones((B, self.G), device=self.device)
        fallback = self.regen_fallback[light_index.reshape(-1)].expand(B)
        f = fallback[:, None].expand(B, self.G).clone()
        m, d = self.regen_m, self.regen_d
        for g in self.regen_grazers:
            num = torch.zeros(B, device=self.device)
            den = torch.zeros(B, device=self.device)
            for entry, total in zip(self.ndm_grazers, ndm_totals):
                if entry["g"] != g or entry["p"] not in (m, d):
                    continue
                den = den + total
                if entry["p"] == m:
                    num = num + total
            f[:, g] = torch.where(den > 0, num / den.clamp_min(1e-30), fallback)
        if dm_feed is not None:
            phi_cols = [(m, None), (d, None)] + [(g, g) for g in self.regen_grazers]
            for i, gi in self.regen_dm_rows:
                num = torch.zeros(B, device=self.device)
                den = torch.zeros(B, device=self.device)
                for col, grazer in phi_cols:
                    v = dm_feed[:, i, col]
                    phi = (1.0 if col == m else 0.0) if grazer is None else f[:, grazer]
                    num = num + v * phi
                    den = den + v
                f[:, gi] = torch.where(den > 0, num / den.clamp_min(1e-30), fallback)
        return f

    def layer_exchange(self, biomass, reserve, light_index=None):
        """Section 180: water and algae between the mixed layer and the
        subsurface layer (mirror of ``apply_layer_exchange``)."""
        if self.layer is None:
            return biomass, reserve
        if light_index is None:
            raise ValueError("light_index is required with the subsurface layer")
        L = self.layer
        t = light_index.reshape(-1)
        m, d = L["m"], L["d"]
        # Slices are read (copied) before either column is written.
        n, nd = reserve[:, m].clone(), reserve[:, d].clone()
        b, bd = biomass[:, m].clone(), biomass[:, d].clone()
        biomass, reserve = biomass.clone(), reserve.clone()
        reserve[:, m] = L["n_mm"][t] * n + L["n_md"][t] * nd
        reserve[:, d] = L["n_dm"][t] * n + L["n_dd"][t] * nd + L["n_s"][t]
        biomass[:, m] = L["b_mm"][t] * b + L["b_md"][t] * bd
        biomass[:, d] = L["b_dm"][t] * b + L["b_dd"][t] * bd
        return biomass, reserve

    def population(self, biomass, reserve, seed_multiplier, light_index=None,
                   hidden=None, dm_feed=None):
        """``dm_feed`` [B, D, G]: this tick's intake per pair, for the layer
        split of the regenerated M1 (section 202; None = no DM fed)."""
        biomass, reserve = self.layer_exchange(biomass, reserve, light_index)
        biomass, reserve, ndm_totals = self.ndm_grazing(biomass, reserve, light_index)
        if self.m1_exposure_on:
            if hidden is None:
                raise ValueError("hidden is required with exposure-weighted M1")
            keep = self.mortality_keep_field(hidden, light_index)
        else:
            keep = self.mortality_keep
        b = biomass * keep
        r = reserve * keep
        if self.regen_on:
            if light_index is None:
                raise ValueError("light_index is required with regeneration")
            m1_back = self.regen_share * (biomass - b)
            share_m = self.regen_fraction(dm_feed, ndm_totals, light_index,
                                          biomass.shape[0])
        if self.capacity_on:
            capacity_keep = (1.0 - self.capacity_rate
                             * b.mean(dim=-1, keepdim=True)).clamp_min(0)
            b = b * capacity_keep
            r = r * capacity_keep
        surplus = self.energy_level(b, r) - self.maintenance
        growth = self.growth
        if self.growth_temp is not None:
            if light_index is None:
                raise ValueError("light_index is required with growth_q10")
            growth = growth * self.growth_temp[light_index.reshape(-1)]
        if self.income_on:
            growth = torch.where(self.income_off_growth, 0.0, growth)
        delta = b * torch.where(surplus >= 0, growth, self.starve) * surplus
        if self.mass_balance:
            # Growth buys biomass out of the reserve standing above the
            # maintenance line, and the reserve is debited for it. Same
            # algebra as ``population_change._charge_growth_to_reserve``;
            # see that docstring and section 116.
            ec = self.energy_content[:, :, None]
            available = (r - self.maintenance * self.max_reserve * b).clamp_min(0)
            wish = delta.clamp_min(0)
            gain = torch.where(ec > 0,
                               torch.minimum(wish, available / ec.clamp_min(1e-30)),
                               wish)
            r = (r - torch.where(self.is_dm & (ec > 0), gain * ec,
                                 torch.zeros_like(gain))).clamp_min(0)
            delta = torch.where(delta > 0, gain, delta)
        loss = -torch.minimum(delta, torch.zeros_like(delta))
        starve_loss = torch.where(self.is_dm, torch.minimum(loss, b), 0.0)
        reduction = torch.where(loss > 0, (b - loss) / (b + 1e-9), 1.0).clamp(0, 1)
        dm_b, dm_r = (b + delta).clamp_min(0), r * reduction

        rate = self.growth
        if self.growth_light_on:
            if light_index is None:
                raise ValueError("light_index is required with light-limited growth")
            rate = rate * self.growth_light[light_index.reshape(-1)]
        # Section 173: the M1 of a non decision maker acts before its
        # growth, on the biomass only (``keep`` is 1 for an NDM without it).
        nb = biomass * keep
        ndm_delta = rate * nb * (1.0 - nb / (self.capacity + 1e-9))
        ndm_r = reserve
        if self.nutrients_on:
            if light_index is None:
                raise ValueError("light_index is required with nutrient pools")
            # Section 172, mirror of ``population_change.
            # _nutrient_limited_delta``: supply the pool, grow on it, pay.
            t = light_index.reshape(-1)
            pool = (reserve * self.nutrient_keep[t]
                    + self.nutrient_supply[t]).clamp_min(0)
            limitation = pool / (pool + self.nutrient_half[t])
            stress = (self.stress_max * self.stress_shape * (1.0 - limitation)
                      / (limitation + self.stress_shape))
            lost = torch.where(self.nutrient, nb * stress, 0.0)
            nb = nb - lost
            # Section 177: the lysed share returns to the pool.
            pool = pool + self.stress_regen * lost
            wish = rate * nb * limitation
            limited = torch.minimum(torch.minimum(wish, pool),
                                    (self.capacity - nb).clamp_min(0)
                                    ).clamp_min(0)
            ndm_delta = torch.where(self.nutrient, limited, ndm_delta)
            ndm_r = torch.where(self.nutrient, pool - limited, reserve)
        ndm_delta = ndm_delta + self.seed_rate * self.capacity * seed_multiplier
        ndm_b = torch.minimum((nb + ndm_delta).clamp_min(0), self.capacity)
        b = torch.where(self.is_dm, dm_b, ndm_b)
        r = torch.where(self.is_dm, dm_r, ndm_r)
        b = torch.where(b < self.threshold, 0.0, b)
        empty = b <= 0
        if self.nutrients_on:
            empty = empty & ~self.nutrient
        r = torch.where(empty, 0.0, r)
        if self.regen_on:
            # Section 202: after every group's growth, as the reference.
            to_mixed = (m1_back * share_m[:, :, None]).sum(1)
            r = r.clone()
            r[:, self.regen_m] = r[:, self.regen_m] + to_mixed
            if self.regen_d is not None:
                r[:, self.regen_d] = r[:, self.regen_d] + (m1_back.sum(1) - to_mixed)
        return b, r, starve_loss

    def advect(self, biomass, reserve, tick, keys=None, track=False):
        """Drift every responding group; ``track`` also returns the shares.

        With ``track`` the third return value is ``(phi, b_after)`` for
        the decision makers: the per-direction share the field carried
        out of each cell and the biomass left standing there. The local
        reward needs both, because a drifting decision maker no longer
        sits where the movement tracker recorded it.
        """
        if self.currents is None or self.currents.strength == 0 or not self.drift_positions:
            return (biomass, reserve, None) if track else (biomass, reserve)
        # ``torch.full`` rather than ``as_tensor``: a host scalar copied to the
        # device is not permitted while a CUDA graph is capturing.
        tick = (tick.to(torch.int64) if isinstance(tick, torch.Tensor) else
                torch.full((), int(tick), dtype=torch.int64, device=self.device))
        if keys is None:
            keys = torch.full((biomass.shape[0],), self.current_world_seed,
                              dtype=torch.int64, device=self.device)
        fractions = torch.stack(direction_fractions(
            tick, keys[:, None], self.currents, self.current_x, self.current_y), dim=1)
        fractions = (fractions[:, None] * self.current_response
                     * self.move_mask[None, None])
        b, r = biomass[:, self.drift_index], reserve[:, self.drift_index]
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
            b_imm, r_imm = self.immigration(b_emig, r_emig, self.drift_index)
            b_total, r_total = b_total + b_imm, r_total + r_imm
        out_b = biomass.index_copy(1, self.drift_index, b_total)
        out_r = reserve.index_copy(1, self.drift_index, r_total)
        if not track:
            return out_b, out_r
        # Re-index the shares on the decision makers, zero for the ones
        # the field does not carry, so the tracker sees one fixed shape.
        phi = torch.zeros((biomass.shape[0], self.G) + fractions.shape[2:],
                          device=self.device, dtype=fractions.dtype)
        phi = phi.index_copy(1, self.drift_index, fractions)
        return out_b, out_r, (phi[:, self.dm_index], out_b[:, self.dm_index])

    def step(self, biomass, reserve, actions, tick, seed_multiplier,
             current_keys=None, track_source=False, light_index=None,
             impacts=None):
        """One tick. ``track_source`` appends the local-reward tracking.

        ``impacts`` is ``(loss_rate, cost)`` from ``impact_fields``, or
        None for a world without impact tables (section 161).

        With ``track_source`` a sixth value ``(start, tracked, frac_in)``
        is returned: ``start`` is ``A(c, t)``, the cell energy the policy
        observed (before the impact mortality and predation), and
        ``tracked``/``frac_in`` come from ``tracked_energy`` once the
        end-of-tick state is final.
        """
        start = self.local_energy(biomass, reserve) if track_source else None
        impact_cost = None
        if impacts is not None:
            biomass, reserve = self.impact_mortality(biomass, reserve, impacts[0])
            impact_cost = impacts[1]
        b, r, gains, hidden, intake = self.predation(biomass, reserve, actions,
                                                     light_index)
        b, r, flow = self.movement(b, r, gains, actions, track=track_source,
                                   light_index=light_index, impact_cost=impact_cost)
        if track_source:
            b, r, drift = self.advect(b, r, tick, current_keys, track=True)
        else:
            b, r = self.advect(b, r, tick, current_keys)
        b, r, starve_loss = self.population(
            b, r, seed_multiplier, light_index, hidden,
            dm_feed=intake.sum(-1) if self.regen_on else None)
        if not track_source:
            return b, r, hidden, intake, starve_loss
        tracked, frac_in = self.tracked_energy(flow, b, r, drift)
        return b, r, hidden, intake, starve_loss, (start, tracked, frac_in)
