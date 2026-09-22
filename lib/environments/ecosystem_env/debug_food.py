"""Disposable, opt-in food-following experiment; not an ecological model.

Phytoplankton and benthos share a compact bouncing blob, replenished to their
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

    def __post_init__(self):
        if not math.isfinite(self.speed) or self.speed < 0:
            raise ValueError("Food blob speed must be finite and nonnegative")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Food blob radius must be finite and nonnegative")
        if self.radius != 0 and self.radius < 1:
            raise ValueError("Food blob radius must be zero (automatic) or at least one cell")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Food blob seed must be an integer in [0, 2**32)")

    def metadata(self):
        return asdict(self)


def add_food_blob_arguments(parser):
    parser.add_argument("--debug-food-blobs", action="store_true",
                        help="Debug only: replenish phytoplankton/benthos in one bouncing food blob")
    parser.add_argument("--food-blob-speed", type=float, default=0.1,
                        help="Blob centre speed in cells/tick; 0 is a stationary control (default: 0.1)")
    parser.add_argument("--food-blob-radius", type=float, default=0.0,
                        help="Blob radius in cells; 0 uses one sixth of the shorter grid side")
    parser.add_argument("--food-blob-seed", type=int, default=0,
                        help="Trajectory seed, combined with the rollout seed (default: 0)")


def food_blob_options(args):
    config = FoodBlobConfig(args.food_blob_speed, args.food_blob_radius, args.food_blob_seed)
    return config if args.debug_food_blobs else None


def selected_ids(env):
    if env.food_blobs is None:
        return ()
    return tuple(fid for fid in FOOD_IDS
                 if fid in env.fgs and not env.fgs[fid].is_decision_maker)


def blob_weights(x, y, height, width, tick, keys, config, allowed):
    """Normalised compact blob; NumPy [cell] or Torch [world, cell].

    Reflect the centre analytically at the rectangle's edges. No evolving
    velocity state or random draws, so resets and ARS pairs agree exactly.
    Obstacles clip the footprint; if entirely blocked use the nearest allowed
    cell(s). This debug trajectory does not navigate around internal obstacles.
    """
    tensor = isinstance(x, torch.Tensor)
    radius = config.radius or max(1.0, min(height, width) / 6.0)
    seed = (keys + config.seed) % (2**32)
    px, py = seed % 997, (seed * 37 + 137) % 991
    if tensor:
        px, py = px.to(torch.float32) / 997, py.to(torch.float32) / 991
        time = tick.to(torch.float32)
        absolute = torch.abs
    else:
        px = np.array(px, dtype=np.float32, ndmin=1) / 997
        py = np.array(py, dtype=np.float32, ndmin=1) / 991
        time = np.array(tick, dtype=np.float32, ndmin=1)
        absolute = np.abs

    def bounce(size, phase, velocity):
        margin = min(radius, (size - 1) / 2)
        span = size - 1 - 2 * margin
        if span <= 0:
            return phase * 0 + margin
        travel = (phase * (2 * span) + time * velocity) % (2 * span)
        return margin + span - absolute(travel - span)

    cx, cy = bounce(width, px, config.speed * 0.8), bounce(height, py, config.speed * 0.6)
    distance2 = (x - cx) ** 2 + (y - cy) ** 2
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
    apply(env, env.tick_count)


def apply(env, tick):
    if env.food_blobs is None:
        return
    y, x = np.indices((env.H, env.W), dtype=np.float32)
    habitat = env.grid.get_map("accessibility")
    allowed = np.ones((env.H, env.W), dtype=bool) if habitat is None else habitat > 0
    if not allowed.any():
        raise ValueError("Food blobs need at least one accessible cell")
    weights = blob_weights(x.ravel(), y.ravel(), env.H, env.W, tick,
                           env.current_world_seed, env.food_blobs, allowed.ravel()).reshape(env.H, env.W)
    for fid, (biomass, reserve) in env._food_blob_totals.items():
        env.fgs[fid].biomass = (weights * biomass).astype(env.dtype)
        env.fgs[fid].energy_reserve = (weights * reserve).astype(env.dtype)


def apply_tensor(model, biomass, reserve, tick, keys, totals=None):
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
                           tick, keys[:, None], model.food_blobs, model.food_blob_allowed)
    return (biomass.index_copy(1, model.food_blob_index, totals[0] * weights[:, None]),
            reserve.index_copy(1, model.food_blob_index, totals[1] * weights[:, None]))
