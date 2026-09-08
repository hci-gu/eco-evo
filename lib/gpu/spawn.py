"""World generation and allocation entirely on the tensor device."""

import math

import numpy as np
import torch

from lib.gpu.random import fold_in, normal, uniform


def distribute_with_floor(weights, total, floor):
    """Batched equivalent of the reference's greedy prefix allocator.

    weights: [world, cell], total: [world], floor: nonnegative scalar.
    Zero-weight worlds and totals below the floor retain reference behavior.
    """
    weights = weights.clamp_min(0).to(torch.float64)
    total = total.to(torch.float64).clamp_min(0)
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
    total = total.to(torch.float64).clamp_min(0)
    draws = uniform(keys, (cells,), 71, torch.float64)
    order = uniform(keys, (cells,), 72, torch.float64).argsort(dim=-1, stable=True)
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
    pick = torch.floor(uniform(keys, (), 73, torch.float64) * n).long()
    amounts = amounts.scatter_add(1, pick[:, None], leftover[:, None])
    return torch.zeros_like(amounts).scatter(1, order, amounts)


class WorldSpawner:
    def __init__(self, spec, model):
        self.spec, self.model = spec, model
        self.device = model.device
        self.H, self.W, self.C = model.H, model.W, model.C
        mask = np.ones((self.H, self.W), dtype=bool) if spec.allowed_mask is None else np.asarray(spec.allowed_mask, dtype=bool).reshape(self.H, self.W)
        self.allowed = model.tensor(mask.reshape(-1), torch.bool)
        eligible = np.flatnonzero(mask)
        self.eligible = model.tensor(eligible if len(eligible) else np.arange(self.C), torch.long)
        self.n_eligible = self.eligible.numel()
        self.yy, self.xx = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=torch.float64),
            torch.arange(self.W, device=self.device, dtype=torch.float64), indexing="ij")
        fy = torch.fft.fftfreq(self.H, device=self.device, dtype=torch.float64)[:, None]
        fx = torch.fft.fftfreq(self.W, device=self.device, dtype=torch.float64)[None, :]
        self.frequency2 = fy * fy + fx * fx
        self.filters = {}
        self.floors = {f: 5.0 * spec.env.fgs[f].min_split_biomass for f in model.ids}
        # Constants and FFT filters are allocated during startup, never during
        # a captured tick. Reset itself can run eagerly on CUDA.
        for fid in spec.spawn_order:
            cfg = spec.spawn[fid] or {}
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

    def weights(self, cfg, keys, fields):
        mode = str(cfg.get("mode", "uniform")).strip().lower()
        batch = keys.numel()
        shape = (self.H, self.W)
        if mode == "perlin":
            field = torch.zeros((batch, *shape), dtype=torch.float64, device=self.device)
            scale = float(cfg.get("scale", 10))
            amp = 1.0
            for octave in range(max(1, int(cfg.get("octaves", 4)))):
                field = field + amp * self.smooth(normal(keys, shape, 100 + octave, torch.float64), max(1.0, scale))
                amp *= float(cfg.get("persistence", 0.5))
                scale /= float(cfg.get("lacunarity", 2))
            low, high = field.amin((-1, -2), keepdim=True), field.amax((-1, -2), keepdim=True)
            field = (field - low) / (high - low).clamp_min(1e-30)
            field = (field - float(cfg.get("threshold", 0))).clamp_min(0)
            return self.finalize(self.relative_floor(field, cfg.get("min_frac_of_max", 0)))
        if mode == "colony":
            n = max(0, min(int(cfg.get("n_colonies", 3)), self.n_eligible))
            sigma = max(1e-6, float(cfg.get("sigma_cells", 2)))
            scores = uniform(keys, (self.n_eligible,), 201, torch.float64)
            centers = self.eligible[scores.argsort(dim=-1, stable=True)[:, :n]]
            jitter = str(cfg.get("amplitude_mode", "uniform")).lower() == "jitter"
            lo, hi = sorted([min(1.0, max(0.0, float(cfg.get(k, default)))) for k, default in (("amplitude_min", 0), ("amplitude_max", 1))])
            amplitudes = lo + (hi - lo) * uniform(keys, (n,), 202, torch.float64)
            field = torch.zeros((batch, *shape), device=self.device, dtype=torch.float64)
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
            out = torch.zeros((batch, *shape), device=self.device, dtype=torch.float64)
            for ref in cfg.get("refs", []) or []:
                name = ref.get("name")
                if str(name).lower() == "depth" or name not in fields:
                    continue
                field = fields[name].reshape(batch, *shape)
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
                out = out + noise_amp * self.smooth(normal(keys, shape, 301, torch.float64), float(cfg.get("noise_scale", 5)))
            out = (out - float(cfg.get("floor", 0))).clamp_min(0)
            return self.finalize(self.relative_floor(out, cfg.get("min_frac_of_max", 0)))
        # The CPU registry also falls back to uniform for unknown strategies.
        return self.finalize(uniform(keys, shape, 401, torch.float64))

    def biomass(self, world_keys):
        fields = {}
        for fid in self.spec.spawn_order:
            index = self.model.ids.index(fid)
            keys = fold_in(world_keys, index + 1000)
            lo, hi = self.spec.ranges[fid]
            total = torch.floor(uniform(keys, (), 0, torch.float64) * (hi - lo + 1)) + lo
            cfg = self.spec.spawn[fid]
            if cfg and str(cfg.get("mode", "uniform")).strip():
                if cfg.get("seed") is not None:
                    keys = torch.full_like(keys, int(cfg["seed"]))
                weights = self.weights(cfg, keys, fields)
                fields[fid] = distribute_with_floor(weights * self.allowed, total, self.floors[fid])
            else:
                allocated = legacy_clusters(keys, total, self.floors[fid], self.n_eligible)
                fields[fid] = torch.zeros((world_keys.numel(), self.C), dtype=torch.float64, device=self.device).index_copy(1, self.eligible, allocated)
        return torch.stack([fields[f] for f in self.model.ids], dim=1).float()

    def reserves(self, biomass, pair_keys):
        if self.spec.randomize_energy:
            lo = (self.model.maintenance - 0.3).clamp_min(0)
            hi = (self.model.maintenance + 0.3).clamp_max(1)
            ratios = lo + (hi - lo) * uniform(pair_keys, (self.model.G, self.C), 501)
        else:
            ratios = 0.7
        return biomass * ratios * self.model.max_reserve

    def phases(self, pair_keys):
        return uniform(pair_keys, (self.model.G, 1), 502) * self.model.period.clamp_min(1)
