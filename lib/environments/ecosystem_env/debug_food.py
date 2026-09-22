"""Disposable, opt-in food-following experiment; not an ecological model.

Phytoplankton and benthos share a compact moving blob, replenished to their
rollout-start biomass/reserve after every tick. Predators still gain energy.
Keep experiment-specific logic here so the small engine hooks are removable.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


FOOD_IDS = ("phytoplankton", "benthic_community")


@dataclass(frozen=True)
class FoodBlobConfig:
    speed: float = 0.1
    radius: float = 0.0
    seed: int = 0
    segment_min: int = 40
    segment_max: int = 160

    def __post_init__(self):
        if not math.isfinite(self.speed) or self.speed < 0:
            raise ValueError("Food blob speed must be finite and nonnegative")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Food blob radius must be finite and nonnegative")
        if self.radius != 0 and self.radius < 1:
            raise ValueError("Food blob radius must be zero (automatic) or at least one cell")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Food blob seed must be an integer in [0, 2**32)")
        if (not isinstance(self.segment_min, int) or not isinstance(self.segment_max, int)
                or not 1 <= self.segment_min <= self.segment_max):
            raise ValueError("Food blob segment ticks must satisfy 1 <= MIN <= MAX")

    def metadata(self):
        return asdict(self)


def add_food_blob_arguments(parser):
    parser.add_argument("--debug-food-blobs", action="store_true",
                        help="Debug only: replenish phytoplankton/benthos in one moving food blob")
    parser.add_argument("--food-blob-speed", type=float, default=0.1,
                        help="Blob centre speed in cells/tick; 0 is a stationary control (default: 0.1)")
    parser.add_argument("--food-blob-radius", type=float, default=0.0,
                        help="Blob radius in cells; 0 uses one sixth of the shorter grid side")
    parser.add_argument("--food-blob-seed", type=int, default=0,
                        help="Trajectory seed, combined with the rollout seed (default: 0)")
    parser.add_argument("--food-blob-segment-ticks", type=int, nargs=2, default=(40, 160),
                        metavar=("MIN", "MAX"),
                        help="Torus: random segment duration range, inclusive (default: 40 160)")


def food_blob_options(args):
    config = FoodBlobConfig(args.food_blob_speed, args.food_blob_radius, args.food_blob_seed,
                            *args.food_blob_segment_ticks)
    return config if args.debug_food_blobs else None


def selected_ids(env):
    if env.food_blobs is None:
        return ()
    return tuple(fid for fid in FOOD_IDS
                 if fid in env.fgs and not env.fgs[fid].is_decision_maker)


def _trajectory_uniform(seed, stream):
    """Stateless 32-bit mixing shared by NumPy and Torch, without RNG state."""
    value = (seed + stream * 0x9E3779B9) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    value = (value ^ (value >> 16)) >> 8
    value = value.to(torch.float32) if isinstance(value, torch.Tensor) else value.astype(np.float32)
    return value * (1.0 / 16777216.0)


def blob_center(height, width, tick, keys, config):
    """Seeded start and heading, reflected at boundaries at constant speed.

    Each rollout samples a different start and velocity, not merely a phase
    offset on one shared diagonal path. The heading stays constant between
    bounces. A seed identifies a reproducible trajectory, including for ARS
    pairs, replay, chunked execution and CUDA graph capture.
    """
    tensor = isinstance(keys, torch.Tensor)
    radius = config.radius or max(1.0, min(height, width) / 6.0)
    if tensor:
        seed = (keys.to(torch.int64) + config.seed) & 0xFFFFFFFF
        time = tick.to(torch.float32)
        absolute, sqrt, where = torch.abs, torch.sqrt, torch.where
    else:
        seed = (np.array(keys, dtype=np.int64, ndmin=1) + config.seed) & 0xFFFFFFFF
        time = np.array(tick, dtype=np.float32, ndmin=1)
        absolute, sqrt, where = np.abs, np.sqrt, np.where
    px, py = _trajectory_uniform(seed, 1), _trajectory_uniform(seed, 2)
    vx = 2 * _trajectory_uniform(seed, 3) - 1
    vy = 2 * _trajectory_uniform(seed, 4) - 1
    # A degenerate zero vector gets a unit heading, without a host branch.
    length2 = vx * vx + vy * vy
    vx = where(length2 > 0, vx, 1.0)
    length = sqrt(where(length2 > 0, length2, 1.0))
    vx, vy = vx * (config.speed / length), vy * (config.speed / length)

    def bounce(size, phase, velocity):
        margin = min(radius, (size - 1) / 2)
        span = size - 1 - 2 * margin
        if span <= 0:
            return phase * 0 + margin
        travel = (phase * (2 * span) + time * velocity) % (2 * span)
        return margin + span - absolute(travel - span)

    return bounce(width, px, vx), bounce(height, py, vy)


class TorusBlobMotion:
    """Small reproducible per-world state, with no trajectory table or readback.

    New headings and integer durations are keyed by world and segment number.
    All buffers have fixed addresses for CUDA graph replay; the runner includes
    them in its warmup/capture snapshots. Positions use float64 to avoid long
    rollout drift; the biomass footprint still uses the ecosystem's float32.
    """
    def __init__(self, height, width, keys, config):
        self.height, self.width, self.config = height, width, config
        self.tensor = isinstance(keys, torch.Tensor)
        self.buffers = []
        for name in ("seed", "segment", "remaining", "x", "y", "vx", "vy"):
            integer = name in ("seed", "segment", "remaining")
            if self.tensor:
                value = torch.zeros_like(keys, dtype=torch.int64 if integer else torch.float64)
            else:
                value = np.zeros_like(np.asarray(keys), dtype=np.int64 if integer else np.float64)
            setattr(self, name, value)
            self.buffers.append(value)
        self.reset(keys)

    def _put(self, target, value):
        if self.tensor:
            target.copy_(value)
        else:
            target[...] = value

    def _parameters(self):
        seed = self.seed ^ ((self.segment * 0x9E3779B9) & 0xFFFFFFFF)
        angle = _trajectory_uniform(seed, 3)
        angle = angle.double() if self.tensor else angle.astype(np.float64)
        angle = angle * (2 * math.pi)
        cos, sin = (torch.cos, torch.sin) if self.tensor else (np.cos, np.sin)
        duration = _trajectory_uniform(seed, 5) * (self.config.segment_max - self.config.segment_min + 1)
        duration = duration.long() if self.tensor else duration.astype(np.int64)
        return (cos(angle) * self.config.speed, sin(angle) * self.config.speed,
                duration + self.config.segment_min)

    def reset(self, keys):
        self._put(self.seed, (keys + self.config.seed) & 0xFFFFFFFF)
        self._put(self.segment, self.segment * 0)
        self._put(self.x, _trajectory_uniform(self.seed, 1) * self.width)
        self._put(self.y, _trajectory_uniform(self.seed, 2) * self.height)
        vx, vy, duration = self._parameters()
        self._put(self.vx, vx)
        self._put(self.vy, vy)
        self._put(self.remaining, duration)

    def advance(self):
        expired = self.remaining == 0
        self._put(self.segment, self.segment + expired)
        vx, vy, duration = self._parameters()
        where = torch.where if self.tensor else np.where
        self._put(self.vx, where(expired, vx, self.vx))
        self._put(self.vy, where(expired, vy, self.vy))
        self._put(self.remaining, where(expired, duration, self.remaining) - 1)
        self._put(self.x, (self.x + self.vx) % self.width)
        self._put(self.y, (self.y + self.vy) % self.height)

    def center(self):
        if self.tensor:
            return self.x.float(), self.y.float()
        return self.x.astype(np.float32), self.y.astype(np.float32)


def blob_weights(x, y, height, width, tick, keys, config, allowed,
                 boundary="bounded", center=None):
    """Normalised compact blob; NumPy [cell] or Torch [world, cell].

    Bounded mode reflects analytically; torus mode receives a centre from
    TorusBlobMotion and uses shortest periodic distances across both seams.
    Obstacles clip the footprint; if entirely blocked use the nearest allowed
    cell(s). This debug trajectory does not navigate around internal obstacles.
    """
    tensor = isinstance(x, torch.Tensor)
    radius = config.radius or max(1.0, min(height, width) / 6.0)
    if boundary == "torus" and center is None:
        raise ValueError("Torus food placement requires a TorusBlobMotion centre")
    cx, cy = center if center is not None else blob_center(height, width, tick, keys, config)
    dx, dy = x - cx, y - cy
    if boundary == "torus":
        dx = (dx + width / 2) % width - width / 2
        dy = (dy + height / 2) % height - height / 2
    distance2 = dx ** 2 + dy ** 2
    if tensor:
        raw = (1 - distance2 / radius**2).clamp_min(0).square() * allowed
        total = raw.sum(-1, keepdim=True)
        distance = torch.where(allowed, distance2, float("inf"))
        nearest = (distance == distance.amin(-1, keepdim=True)).to(raw.dtype) * allowed
        return torch.where(total > 0, raw / total.clamp_min(1e-30),
                           nearest / nearest.sum(-1, keepdim=True).clamp_min(1))
    raw = np.maximum(0, 1 - distance2 / radius**2) ** 2 * allowed
    if raw.sum() > 0:
        return (raw / raw.sum()).astype(np.float32)
    distance = np.where(allowed, distance2, np.inf)
    nearest = (distance == distance.min()) & allowed
    return nearest.astype(np.float32) / max(1, nearest.sum())


def reset(env):
    """Capture totals after spawning or viewer biomass overrides, then pose t=0."""
    if env.food_blobs is None:
        return
    ids = selected_ids(env)
    if not ids:
        raise ValueError("--debug-food-blobs requires phytoplankton or benthic_community as a non-decision maker")
    env._food_blob_totals = {
        fid: (float(env.fgs[fid].biomass.sum()), float(env.fgs[fid].energy_reserve.sum()))
        for fid in ids
    }
    if env.boundary == "torus":
        env._food_blob_motion = TorusBlobMotion(
            env.H, env.W, np.array([env.current_world_seed], dtype=np.int64), env.food_blobs)
        env._food_blob_tick = 0
    apply(env, env.tick_count)


def apply(env, tick):
    if env.food_blobs is None:
        return
    y, x = np.indices((env.H, env.W), dtype=np.float32)
    habitat = env.grid.get_map("accessibility")
    allowed = np.ones((env.H, env.W), dtype=bool) if habitat is None else habitat > 0
    if not allowed.any():
        raise ValueError("Food blobs need at least one accessible cell")
    center = None
    if env.boundary == "torus":
        if tick < env._food_blob_tick:
            env._food_blob_motion.reset(np.array([env.current_world_seed], dtype=np.int64))
            env._food_blob_tick = 0
        while env._food_blob_tick < tick:
            env._food_blob_motion.advance()
            env._food_blob_tick += 1
        center = env._food_blob_motion.center()
    weights = blob_weights(x.ravel(), y.ravel(), env.H, env.W, tick,
                           env.current_world_seed, env.food_blobs, allowed.ravel(),
                           env.boundary, center).reshape(env.H, env.W)
    for fid, (biomass, reserve) in env._food_blob_totals.items():
        env.fgs[fid].biomass = (weights * biomass).astype(env.dtype)
        env.fgs[fid].energy_reserve = (weights * reserve).astype(env.dtype)


def apply_tensor(model, biomass, reserve, tick, keys, totals=None, motion=None):
    """Device-only placement; optional fixed totals belong to a rollout runner."""
    if model.food_blobs is None:
        return biomass, reserve
    if keys is None:
        keys = torch.full((biomass.shape[0],), model.current_world_seed,
                          dtype=torch.int64, device=model.device)
    if not isinstance(tick, torch.Tensor):
        tick = torch.full((), tick, dtype=torch.int64, device=model.device)
    if totals is None:
        totals = (biomass[:, model.food_blob_index].sum(-1, keepdim=True),
                  reserve[:, model.food_blob_index].sum(-1, keepdim=True))
    weights = blob_weights(model.current_x, model.current_y, model.H, model.W,
                           tick, keys[:, None], model.food_blobs, model.food_blob_allowed,
                           model.boundary, motion.center() if motion is not None else None)
    return (biomass.index_copy(1, model.food_blob_index, totals[0] * weights[:, None]),
            reserve.index_copy(1, model.food_blob_index, totals[1] * weights[:, None]))
