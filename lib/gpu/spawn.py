"""World generation and allocation entirely on the tensor device."""

import math

import numpy as np
import torch

from lib.gpu.device import wide_dtype
from lib.gpu.random import fold_in, normal, uniform
from lib.spawn.strategies import _apply_grid_scaling


def distribute_with_floor(weights, total, floor):
    """Batched equivalent of the reference's greedy prefix allocator.

    weights: [world, cell], total: [world], floor: nonnegative scalar.
    Zero-weight worlds and totals below the floor retain reference behavior.
    """
    wide = wide_dtype(weights.device)
    weights = weights.clamp_min(0).to(wide)
    total = total.to(wide).clamp_min(0)
    sums = weights.sum(-1, keepdim=True)
    if floor <= 0:
        return weights * total[:, None] / sums.clamp_min(1e-30)
    sorted_w, order = torch.sort(weights, dim=-1, descending=True, stable=True)
    prefix = sorted_w.cumsum(-1)
    rank = torch.arange(1, weights.shape[-1] + 1, device=weights.device)
    max_count = torch.floor(total[:, None] / floor)
    valid = ((sorted_w > 0) & (rank <= max_count) &
             (total[:, None] * sorted_w / prefix.clamp_min(1e-30) >= floor))
    n = torch.where(valid, rank, 0).amax(-1, keepdim=True).clamp_min(1)
    chosen = sorted_w * (rank <= n)
    allocation = chosen * total[:, None] / chosen.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.zeros_like(weights).scatter(1, order, allocation)


def legacy_clusters(keys, total, floor, cells):
    """Legacy stick breaking, computed with prefix products instead of a
    host-controlled while loop. Cell order is sampled without replacement.
    Returns [world, eligible_cell]; the caller maps into the full grid.
    """
    wide = wide_dtype(total.device)
    total = total.to(wide).clamp_min(0)
    draws = uniform(keys, (cells,), 71, wide)
    order = uniform(keys, (cells,), 72, wide).argsort(dim=-1, stable=True)
    if floor <= 0:
        return draws / (draws.sum(-1, keepdim=True) + 1e-9) * total[:, None]
    log_a = torch.log1p(-draws)
    log_p = torch.cat((torch.zeros_like(log_a[:, :1]), log_a.cumsum(-1)[:, :-1]), dim=-1)
    # Values after a stick has been exhausted are masked out. Bounding these
    # intermediates prevents overflow for grids with thousands of cells.
    inv_p = torch.exp((-log_p).clamp_max(600))
    prefix = inv_p.cumsum(-1)
    previous = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), -1)
    active = prefix <= (total[:, None] / floor)
    remaining = torch.exp(log_p.clamp_min(-600)) * (total[:, None] - floor * previous)
    amounts = torch.where(active, floor + draws * (remaining - floor), 0.0)
    n = active.sum(-1).clamp_min(1)
    leftover = (total - amounts.sum(-1)).clamp_min(0)
    pick = torch.floor(uniform(keys, (), 73, wide) * n).long()
    amounts = amounts.scatter_add(1, pick[:, None], leftover[:, None])
    return torch.zeros_like(amounts).scatter(1, order, amounts)


def spawn_order(order, configs):
    """Stable topological order of ``order`` so each env_driven group comes
    after the groups its refs name (the loader's Kahn sort; a cycle keeps
    the rest in the given order, the unmet refs ignored as on the CPU)."""
    order = list(order)
    present = set(order)
    deps = {}
    for fid in order:
        cfg = configs.get(fid) or {}
        refs = (cfg.get("refs") or []) if str(cfg.get("mode", "")).strip().lower() == "env_driven" else []
        deps[fid] = [r.get("name") for r in refs
                     if isinstance(r, dict) and r.get("name") in present and r.get("name") != fid]
    done, out, remaining = set(), [], order
    while remaining:
        rest = []
        for fid in remaining:
            if all(d in done for d in deps[fid]):
                out.append(fid)
                done.add(fid)
            else:
                rest.append(fid)
        if len(rest) == len(remaining):
            out.extend(rest)
            break
        remaining = rest
    return tuple(out)


def closed_edge_rows(env, ids):
    """Group indices (into ``ids``) of the decision makers that can move
    and whose every edge move is masked (section 266.16; with --migration
    off every mobile decision maker). Section 308.6's edge starts act on
    these."""
    mask = getattr(env, "dm_move_mask", None)
    if mask is None:
        mask = np.broadcast_to(env.move_mask, (len(env.dm_ids),) + tuple(env.move_mask.shape))
    speed = np.asarray(getattr(env, "dm_v", np.ones(len(env.dm_ids))))
    rows = []
    for i, fid in enumerate(env.dm_ids):
        m = np.asarray(mask[i])
        closed = (m[0, 0, :].max() == 0 and m[2, -1, :].max() == 0
                  and m[3, :, 0].max() == 0 and m[1, :, -1].max() == 0)
        if closed and speed[i] > 0:
            rows.append(list(ids).index(fid))
    return rows


# Section 308.9: spawn modes a random choice may draw. env_driven follows
# other groups' fields (flat under today's templates; a drawn reference
# could form a cycle) and equilibrium has no template.
RANDOM_SPAWN_MODES = ("uniform", "perlin", "colony")


def random_spawn_menu(templates):
    """The project's ``spawn_templates`` ({mode: {name: params}}) as the
    menu of a random spawn choice: [(name, spawn block)] for every
    template of a ``RANDOM_SPAWN_MODES`` mode, sorted by mode and name."""
    menu = []
    for mode in RANDOM_SPAWN_MODES:
        for name in sorted((templates or {}).get(mode, {}) or {}):
            params = templates[mode][name]
            if isinstance(params, dict):
                menu.append((str(name), {"mode": mode, **params}))
    return menu


def load_spawn_templates(project_path):
    """The manifest's ``spawn_templates`` tree, {} without one."""
    if not project_path:
        return {}
    from lib.config.config_loader import load_config
    data = load_config(project_path) or {}
    tpls = data.get("spawn_templates")
    return {str(m): dict(v) for m, v in tpls.items() if isinstance(v, dict)} if isinstance(tpls, dict) else {}


class WorldSpawner:
    def __init__(self, spec, model):
        self.spec, self.model = spec, model
        self.device = model.device
        self.wide = model.wide
        self.H, self.W, self.C = model.H, model.W, model.C
        mask = np.ones((self.H, self.W), dtype=bool) if spec.allowed_mask is None else np.asarray(spec.allowed_mask, dtype=bool).reshape(self.H, self.W)
        self.allowed = model.tensor(mask.reshape(-1), torch.bool)
        eligible = np.flatnonzero(mask)
        self.eligible = model.tensor(eligible if len(eligible) else np.arange(self.C), torch.long)
        self.n_eligible = self.eligible.numel()
        self.yy, self.xx = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=self.wide),
            torch.arange(self.W, device=self.device, dtype=self.wide), indexing="ij")
        fy = torch.fft.fftfreq(self.H, device=self.device, dtype=self.wide)[:, None]
        fx = torch.fft.fftfreq(self.W, device=self.device, dtype=self.wide)[None, :]
        self.frequency2 = fy * fy + fx * fx
        self.filters = {}
        # The grid factor the CPU spawn applies to n_colonies (section 258).
        self.biomass_scale = float(getattr(spec, "biomass_scale", 1.0))
        self.floors = {f: spec.env.fgs[f].spawn_floor_units * spec.env.fgs[f].min_split_biomass
                       for f in model.ids}
        # Section 207: the depth map as a reference field and the
        # equilibrium patterns, fixed per project (no randomness).
        depth = getattr(spec, "depth", None)
        self.static_fields = ({} if depth is None else
                              {"depth": model.tensor(np.asarray(depth).reshape(1, -1), self.wide)})
        self.equilibrium = {}
        for fid in model.ids:
            pattern = getattr(spec.env.fgs[fid], "equilibrium_map", None)
            if pattern is not None:
                self.equilibrium[fid] = model.tensor(
                    np.asarray(pattern, dtype=np.float64).reshape(1, self.H, self.W), self.wide)
        # Constants and FFT filters are allocated during startup, never during
        # a captured tick. Reset itself can run eagerly on CUDA.
        for fid in spec.spawn_order:
            self._prepare(spec.spawn[fid] or {})
        # The viewer's spawn dropdowns: {fid: spawn block}
        # replacing the project's block on the training worlds.
        self.overrides = {}
        self.order = tuple(spec.spawn_order)
        # Section 308.9: random spawn choice per world (set_random).
        self.random_share = 0.0
        self.random_menu = []
        self.random_groups = ()
        self.last_choices = None

    def _prepare(self, cfg):
        mode = str(cfg.get("mode", "")).strip().lower()
        if mode == "perlin":
            scale = float(cfg.get("scale", 10))
            lacunarity = float(cfg.get("lacunarity", 2))
            if lacunarity <= 0:
                raise ValueError("Spawn lacunarity must be positive")
            for _ in range(max(1, int(cfg.get("octaves", 4)))):
                self._filter(max(1.0, scale))
                scale /= lacunarity
        if mode == "env_driven":
            for ref in cfg.get("refs", []) or []:
                if ref.get("transform") == "gauss_smooth":
                    self._filter(max(1.0, float(ref.get("sigma", 2))))
            if float(cfg.get("noise_amp", 0)) > 0:
                self._filter(float(cfg.get("noise_scale", 5)))

    def config(self, fid):
        """The spawn block a training world uses for ``fid``."""
        return self.overrides.get(fid, self.spec.spawn[fid])

    def set_overrides(self, configs):
        """Spawn the training worlds with ``configs`` ({fid: spawn block},
        ``inference.spawn_override_configs``) in place of the project's
        blocks; {} restores them. Returns True when anything changed (the
        caller then re-draws its worlds)."""
        configs = {fid: dict(cfg) for fid, cfg in (configs or {}).items()
                   if fid in self.spec.spawn and isinstance(cfg, dict)}
        if configs == self.overrides:
            return False
        for cfg in configs.values():
            self._prepare(cfg)
        self.overrides = configs
        self.order = spawn_order(self.spec.spawn_order,
                                 {fid: self.config(fid) or {} for fid in self.spec.spawn_order})
        return True

    def set_random(self, share, groups, menu):
        """Random spawn choice (section 308.9): in a share ``share`` of the
        worlds each group of ``groups`` (ids) draws, independently of the
        others, one start pattern uniformly from its options - its own
        spawn block (``config``: a dropdown's or the project's) and each
        ``menu`` entry ([(name, block)], ``random_spawn_menu``) that is not
        the same block. Other worlds and groups keep their own block.
        Returns True when anything changed."""
        share = float(share)
        if not 0.0 <= share <= 1.0:
            raise ValueError("random spawn share must be in [0, 1]")
        groups = tuple(f for f in self.model.ids if f in set(groups or ()))
        menu = [(str(n), dict(c)) for n, c in (menu or [])]
        if (share, groups, menu) == (self.random_share, self.random_groups, self.random_menu):
            return False
        for _name, cfg in menu:
            self._prepare(cfg)
        self.random_share, self.random_groups, self.random_menu = share, groups, menu
        return True

    def random_options(self, fid):
        """[(name, block)] a random choice of ``fid`` draws from; the first
        is its own block (name "own")."""
        own = self.config(fid) or {}
        out = [("own", own)]
        for name, cfg in self.random_menu:
            if dict(cfg) != dict(own):
                out.append((name, cfg))
        return out

    def _filter(self, sigma):
        if sigma not in self.filters:
            self.filters[sigma] = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * self.frequency2)
        return self.filters[sigma]

    def smooth(self, field, sigma):
        return torch.fft.ifft2(torch.fft.fft2(field) * self.filters[sigma]).real

    def finalize(self, field):
        w = field.flatten(1).clamp_min(0) * self.allowed
        sums = w.sum(-1, keepdim=True)
        fallback = self.allowed.to(w.dtype)
        fallback = torch.where(fallback.sum() > 0, fallback, torch.ones_like(fallback))
        fallback = fallback / fallback.sum()
        return torch.where(sums > 0, w / sums.clamp_min(1e-30), fallback)

    @staticmethod
    def relative_floor(field, frac):
        frac = min(1.0, max(0.0, float(frac)))
        if frac == 0:
            return field
        floor = field.amax((-1, -2), keepdim=True) * frac
        return torch.where(field > 0, torch.maximum(field, floor), field)

    def weights(self, cfg, keys, fields, fid=None):
        mode = str(cfg.get("mode", "uniform")).strip().lower()
        batch = keys.numel()
        shape = (self.H, self.W)
        if mode == "equilibrium":
            # Mirror of strategies.weights_equilibrium: the precomputed
            # B*(c), uniform without one.
            pattern = self.equilibrium.get(fid)
            if pattern is None:
                pattern = torch.ones((1, *shape), dtype=self.wide, device=self.device)
            return self.finalize(pattern.expand(batch, *shape))
        if mode == "perlin":
            field = torch.zeros((batch, *shape), dtype=self.wide, device=self.device)
            scale = float(cfg.get("scale", 10))
            amp = 1.0
            for octave in range(max(1, int(cfg.get("octaves", 4)))):
                field = field + amp * self.smooth(normal(keys, shape, 100 + octave, self.wide), max(1.0, scale))
                amp *= float(cfg.get("persistence", 0.5))
                scale /= float(cfg.get("lacunarity", 2))
            low, high = field.amin((-1, -2), keepdim=True), field.amax((-1, -2), keepdim=True)
            field = (field - low) / (high - low).clamp_min(1e-30)
            field = (field - float(cfg.get("threshold", 0))).clamp_min(0)
            return self.finalize(self.relative_floor(field, cfg.get("min_frac_of_max", 0)))
        if mode == "colony":
            scaled = _apply_grid_scaling(cfg, {"biomass_scale": self.biomass_scale})
            n = max(0, min(int(scaled.get("n_colonies", 3)), self.n_eligible))
            sigma = max(1e-6, float(cfg.get("sigma_cells", 2)))
            scores = uniform(keys, (self.n_eligible,), 201, self.wide)
            centers = self.eligible[scores.argsort(dim=-1, stable=True)[:, :n]]
            jitter = str(cfg.get("amplitude_mode", "uniform")).lower() == "jitter"
            lo, hi = sorted([min(1.0, max(0.0, float(cfg.get(k, default)))) for k, default in (("amplitude_min", 0), ("amplitude_max", 1))])
            amplitudes = lo + (hi - lo) * uniform(keys, (n,), 202, self.wide)
            field = torch.zeros((batch, *shape), device=self.device, dtype=self.wide)
            # n is a static configuration value. Iterating over centers avoids
            # allocating an M*n*H*W temporary for large colony counts.
            for i in range(n):
                cy, cx = (centers[:, i] // self.W)[:, None, None], (centers[:, i] % self.W)[:, None, None]
                bulge = torch.exp(-((self.yy - cy) ** 2 + (self.xx - cx) ** 2) / (2 * sigma * sigma))
                if jitter:
                    field = torch.maximum(field, amplitudes[:, i, None, None] * bulge)
                else:
                    field = field + bulge
            if jitter:
                return field.flatten(1).clamp(0, 1) * self.allowed
            return self.finalize(field)
        if mode == "env_driven":
            out = torch.zeros((batch, *shape), device=self.device, dtype=self.wide)
            for ref in cfg.get("refs", []) or []:
                name = ref.get("name")
                if name in fields:
                    field = fields[name]
                elif name in self.static_fields:
                    field = self.static_fields[name].expand(batch, -1)
                else:
                    continue
                field = field.reshape(batch, *shape)
                transform = ref.get("transform", "linear")
                maximum = field.amax((-1, -2), keepdim=True)
                if transform == "exp":
                    field = torch.exp(field - maximum)
                elif transform == "invert":
                    field = torch.where(maximum > 0, 1 - field / maximum.clamp_min(1e-30), 1.0)
                elif transform == "gauss_smooth":
                    field = self.smooth(field, max(1.0, float(ref.get("sigma", 2))))
                out = out + float(ref.get("weight", 1)) * field
            noise_amp = float(cfg.get("noise_amp", 0))
            if noise_amp > 0:
                out = out + noise_amp * self.smooth(normal(keys, shape, 301, self.wide), float(cfg.get("noise_scale", 5)))
            out = (out - float(cfg.get("floor", 0))).clamp_min(0)
            return self.finalize(self.relative_floor(out, cfg.get("min_frac_of_max", 0)))
        # The CPU registry also falls back to uniform for unknown strategies.
        return self.finalize(uniform(keys, shape, 401, self.wide))

    def biomass(self, world_keys):
        """Start biomass [world, G, C]. With a random spawn choice on
        (``set_random``) ``last_choices`` holds {fid: [world] option index
        into ``random_options(fid)``, -1 = a world without the choice}."""
        fields, choices = {}, {}
        for fid in self.order:
            index = self.model.ids.index(fid)
            keys = fold_in(world_keys, index + 1000)
            lo, hi = self.spec.ranges[fid]
            u = uniform(keys, (), 0, self.wide)
            if lo > 0 and hi > lo:
                # Log-uniform, rounded to whole tonnes (section 200; the CPU
                # loader's _sample_total_biomass draws the same law).
                total = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                total = torch.round(total).clamp(lo, hi)
            else:
                total = torch.floor(u * (hi - lo + 1)) + lo
            if self.random_share > 0.0 and fid in self.random_groups:
                options = self.random_options(fid)
                drawn = [self._field(cfg, keys, total, fields, fid, world_keys.numel())
                         for _name, cfg in options]
                ck = fold_in(fold_in(world_keys, 92001), 100 + index)
                on = uniform(fold_in(world_keys, 92001), (), 0, self.wide) < self.random_share
                pick = torch.floor(uniform(ck, (), 0, self.wide) * len(options)).long()
                pick = torch.where(on, pick.clamp_(0, len(options) - 1), torch.zeros_like(pick))
                fields[fid] = torch.stack(drawn, 1).gather(
                    1, pick[:, None, None].expand(-1, 1, self.C))[:, 0]
                choices[fid] = torch.where(on, pick, torch.full_like(pick, -1))
            else:
                fields[fid] = self._field(self.config(fid), keys, total, fields, fid,
                                          world_keys.numel())
        self.last_choices = choices
        return torch.stack([fields[f] for f in self.model.ids], dim=1).float()

    def _field(self, cfg, keys, total, fields, fid, batch):
        """One group's start field [world, cell] under spawn block ``cfg``."""
        if cfg and str(cfg.get("mode", "uniform")).strip():
            if cfg.get("seed") is not None:
                keys = torch.full_like(keys, int(cfg["seed"]))
            weights = self.weights(cfg, keys, fields, fid)
            return distribute_with_floor(weights * self.allowed, total, self.floors[fid])
        allocated = legacy_clusters(keys, total, self.floors[fid], self.n_eligible)
        return torch.zeros((batch, self.C), dtype=self.wide, device=self.device).index_copy(1, self.eligible, allocated)

    def reserves(self, biomass, pair_keys, light_start=None):
        """Start reserve per world: a fill drawn around u (section 43) or,
        for a group with ``start_fill``, around its field fill on the
        world's start tick ``light_start`` (section 247); 0.7 / the centre
        without randomisation."""
        m = self.model
        centre = None
        if m.start_fill_on:
            if light_start is None or not m.calendar:
                index = torch.zeros(biomass.shape[0], dtype=torch.long, device=biomass.device)
            else:
                index = torch.remainder(light_start.to(torch.long), m.light_period)
            centre = m.start_fill[index][:, :, None]
            known = torch.isfinite(centre)
        if self.spec.randomize_energy:
            lo = (m.maintenance - 0.3).clamp_min(0)
            hi = (m.maintenance + 0.3).clamp_max(1)
            if centre is not None:
                lo = torch.where(known, (centre - 0.3).clamp_min(0), lo)
                hi = torch.where(known, (centre + 0.3).clamp_max(1), hi)
            ratios = lo + (hi - lo) * uniform(pair_keys, (m.G, self.C), 501)
        else:
            ratios = 0.7 if centre is None else torch.where(known, centre, torch.full_like(centre, 0.7))
        reserve = biomass * ratios * self.model.max_reserve
        if self.model.refuge_on:
            # Section 223.12: a refuge group's accessible part starts at
            # its equilibrium share, as FunctionalGroup.initialize_state.
            reserve = torch.where(self.model.refuge_mask,
                                  biomass * self.model.refuge_phi, reserve)
        return reserve

    def depletion(self, world_keys, share):
        """Depleted benthic patches per world (section 223.12): a factor
        [world, 1, cell], 1 outside the patches.

        A share ``share`` of the worlds gets patches: smooth noise
        (sigma 3 cells) above its (1 - coverage) quantile, coverage
        uniform in 0.05-0.30 of the cells, every patch cell at one level
        per world, log-uniform 0.01-0.32 of what it held. Training-world
        design (the fish meet emptied beds and can learn to leave them),
        not biology; keyed by the world, so every candidate on a world
        sees the same patches.
        """
        batch = world_keys.numel()
        keys = fold_in(world_keys, 90001)
        on = uniform(keys, (), 0, self.wide) < float(share)
        coverage = 0.05 + 0.25 * uniform(keys, (), 1, self.wide)
        level = torch.pow(10.0, -2.0 + 1.5 * uniform(keys, (), 2, self.wide))
        self._filter(3.0)
        noise = self.smooth(normal(keys, (self.H, self.W), 3, self.wide), 3.0).reshape(batch, -1)
        index = ((1.0 - coverage) * (noise.shape[1] - 1)).long()
        threshold = noise.sort(dim=1).values.gather(1, index[:, None])
        patch = (noise > threshold) & on[:, None]
        factor = torch.where(patch, level[:, None], torch.ones_like(noise))
        return factor[:, None, :].float()

    # Section 308.6: the eight edge-start places, (side or corner) - the
    # band's distance is to that side's cells or that corner's cell.
    EDGE_PLACES = ("n", "e", "s", "w", "nw", "ne", "sw", "se")

    def edge_distances(self):
        """[8, H, W] distance in cells to each place of ``EDGE_PLACES``
        (a side as its whole line of cells, a corner as its cell)."""
        if not hasattr(self, "_edge_distances"):
            H, W, yy, xx = self.H, self.W, self.yy, self.xx
            sides = [yy, (W - 1) - xx, (H - 1) - yy, xx]
            corners = [torch.hypot(yy - r, xx - c)
                       for r, c in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1))]
            self._edge_distances = torch.stack(sides + corners)
        return self._edge_distances

    def edge_points(self, place, position):
        """(row, col) of each world's edge point: a corner's cell, or for a
        side the point at ``position`` (0-1) along it, from the west / north
        end (row 0 is north). ``place`` [world] indexes ``EDGE_PLACES``."""
        H1, W1 = float(self.H - 1), float(self.W - 1)
        pr, pc = position * H1, position * W1
        zero, h, w = torch.zeros_like(pr), torch.full_like(pr, H1), torch.full_like(pr, W1)
        rows = torch.stack([zero, pr, h, pr, zero, zero, h, h], -1)
        cols = torch.stack([pc, w, pc, zero, zero, w, zero, w], -1)
        pick = place[:, None]
        return rows.gather(-1, pick)[:, 0], cols.gather(-1, pick)[:, 0]

    def edge_starts(self, biomass, world_keys, share, rows, group_share=0.5,
                    length=(0.7, 2.0), mix=(0.75, 1.0), prey_share=0.5,
                    prey_level=(0.3, 0.8)):
        """Edge starts of the training worlds (sections 308.6, 308.8).

        A share ``share`` of the worlds is an edge world. In it each group
        of ``rows`` (the decision makers whose edges are closed, section
        266.16) is moved, with probability ``group_share`` and independently
        of the others, towards a point on the grid edge: one of the eight
        places (four sides, four corners), a side's point at a position
        drawn uniformly along it (``edge_points``). B' = (1 - m) B + m T w,
        w = exp(-d / lambda) on the allowed cells normalised to 1, d the
        distance in cells to the point, lambda log-uniform in ``length``
        (0.7 - 2 cells), m uniform in ``mix`` (0.75 - 1), T the group's
        drawn total (kept): on 30 x 30 the benthivores' top cell has a
        median ~230 t and a third of the draws reach 300 t (fix110's corner
        clump: 400 - 1000 t a cell; section 308.8). In a share ``prey_share`` of the edge worlds the
        benthic prey under the patches is drawn down too: a factor
        1 - (1 - level) exp(-d / lambda) (the lowest over the moved groups),
        level uniform in ``prey_level`` - the state of fix110's ten-year
        corner (prey 0.6-0.8 of the other shallow beds). Training-world
        design: a policy that never meets the wall cannot learn to turn
        there (fix110 benthivores, section 308). Keyed by the world, so
        every candidate on a world sees the same start.

        Returns (biomass [world, G, C], prey factor [world, 1, C], place
        [world, len(rows)] as an index into ``EDGE_PLACES``, -1 = none).
        """
        batch = world_keys.numel()
        keys = fold_in(world_keys, 91001)
        on = uniform(keys, (), 0, self.wide) < float(share)
        yy, xx = self.yy.reshape(1, -1), self.xx.reshape(1, -1)
        allowed = self.allowed.to(self.wide)
        out = biomass.clone()
        factor = torch.ones((batch, 1, self.C), dtype=self.wide, device=self.device)
        prey = on & (uniform(keys, (), 1, self.wide) < float(prey_share))
        level = prey_level[0] + (prey_level[1] - prey_level[0]) * uniform(keys, (), 2, self.wide)
        places = torch.full((batch, len(rows)), -1, dtype=torch.long, device=self.device)
        log_lo, log_hi = math.log(float(length[0])), math.log(float(length[1]))
        for k, g in enumerate(rows):
            gk = fold_in(keys, 100 + int(g))
            moved = on & (uniform(gk, (), 0, self.wide) < float(group_share))
            place = torch.floor(uniform(gk, (), 1, self.wide) * 8).long().clamp_(0, 7)
            lam = torch.exp(log_lo + (log_hi - log_lo) * uniform(gk, (), 2, self.wide))
            m = mix[0] + (mix[1] - mix[0]) * uniform(gk, (), 3, self.wide)
            r0, c0 = self.edge_points(place, uniform(gk, (), 4, self.wide))
            band = torch.exp(-torch.hypot(yy - r0[:, None], xx - c0[:, None]) / lam[:, None])
            w = band * allowed
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
            b = biomass[:, g].to(self.wide)
            total = b.sum(-1, keepdim=True)
            new = (1 - m[:, None]) * b + m[:, None] * total * w
            out[:, g] = torch.where(moved[:, None], new, b).to(out.dtype)
            drawn = 1 - (1 - level[:, None]) * band
            factor = torch.minimum(factor, torch.where((moved & prey)[:, None], drawn,
                                                       torch.ones_like(drawn))[:, None, :])
            places[:, k] = torch.where(moved, place, places[:, k])
        return out, factor.float(), places

    def light_starts(self, world_keys):
        """Year tick each world starts at (daylight calendar, section 137).

        One start per key. The trainer passes the world keys, so both ARS
        signs and every delta pair evaluated on the same world see the
        same season (CRN, section 150). A fixed start day is used as is.
        """
        m = self.model
        if m.calendar and m.light_random_start:
            draw = uniform(world_keys, (), 503) * m.light_period
            return draw.long().clamp_(0, m.light_period - 1)
        return torch.full((world_keys.numel(),), m.light_fixed_start,
                          dtype=torch.long, device=world_keys.device)
