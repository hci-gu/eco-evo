"""Seeded, gently changing bulk currents for non-decision makers."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from lib.environments.ecosystem_env.grid_masks import build_movement_mask
from lib.environments.ecosystem_env.movement import _transfer_to_neighbours


@dataclass(frozen=True)
class CurrentConfig:
    strength: float = 0.1
    period: int = 20
    seed: int = 0

    def __post_init__(self):
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Current strength must be finite and between 0 and 1")
        if not isinstance(self.period, int) or self.period < 1:
            raise ValueError("Current period must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Current seed must be an integer in [0, 2**32)")

    def metadata(self):
        return asdict(self)


def add_current_arguments(parser):
    parser.add_argument("--currents", choices=("on", "off"), default="off",
                        help="Random bulk drift of non-decision makers (default: off)")
    parser.add_argument("--current-strength", "--current_strength", type=float, default=0.1,
                        help="Maximum biomass fraction transported per tick (default: 0.1)")
    parser.add_argument("--current-period", "--current_period", type=int, default=20,
                        help="Ticks between random flow targets, interpolated smoothly (default: 20)")
    parser.add_argument("--current-seed", "--current_seed", type=int, default=0,
                        help="Independent current seed, combined with the rollout seed (default: 0)")


def current_options(args):
    config = CurrentConfig(args.current_strength, args.current_period, args.current_seed)
    return config if args.currents == "on" else None


def _xor(left, right):
    """Bitwise xor for Python scalars, NumPy arrays and Torch tensors.

    ``tensor ^ python_int`` is traced by Dynamo as ``aten::bitwise_xor.Tensor``,
    which rejects the Python scalar and breaks ``torch.compile(fullgraph=True)``.
    Promoting the scalar keeps a single code path for every backend.
    """
    for operand in (left, right):
        if isinstance(operand, torch.Tensor):
            return torch.bitwise_xor(
                torch.as_tensor(left, dtype=operand.dtype, device=operand.device),
                torch.as_tensor(right, dtype=operand.dtype, device=operand.device))
    return left ^ right


def _abs(value):
    """``abs`` that Dynamo can trace: the builtin is unsupported on tensors."""
    if isinstance(value, torch.Tensor):
        return torch.abs(value)
    return abs(value)


def _hash32(value):
    # Same integer operations for Python scalars and device-side Torch tensors.
    value = value & 0xFFFFFFFF
    value = (_xor(value, value >> 16) * 0x7FEB352D) & 0xFFFFFFFF
    value = (_xor(value, value >> 15) * 0x846CA68B) & 0xFFFFFFFF
    return _xor(value, value >> 16) & 0xFFFFFFFF


def direction_fractions(tick, world_seed, config):
    """N/E/S/W outflow fractions, shared by species in one world.

    Stateless randomness keeps ARS signs paired and never consumes the ecology's
    random stream. Supports scalar CPU ticks/seeds and batched Torch tensors.
    Linear interpolation changes direction gradually rather than jittering.
    """
    epoch = tick // config.period
    blend = (tick % config.period) / config.period
    def vector(epoch):
        key = _hash32(_xor(_xor(world_seed, config.seed),
                           _hash32(epoch + 0x9E3779B9)))
        x = ((_hash32(_xor(key, 0xA341316C)) >> 9) + 0.5) / 8388608.0 * 2 - 1
        y = ((_hash32(_xor(key, 0xC8013EA4)) >> 9) + 0.5) / 8388608.0 * 2 - 1
        return x, y
    x0, y0 = vector(epoch)
    x1, y1 = vector(epoch + 1)
    x = ((1 - blend) * x0 + blend * x1) * (config.strength / 2)
    y = ((1 - blend) * y0 + blend * y1) * (config.strength / 2)
    return ((_abs(y) - y) / 2, (_abs(x) + x) / 2,
            (_abs(y) + y) / 2, (_abs(x) - x) / 2)


def apply_currents(env):
    config = env.currents
    if config is None or config.strength == 0:
        return
    ids = [f for f in env.global_fg_order if not env.fgs[f].is_decision_maker]
    if not ids:
        return
    fractions = np.asarray(direction_fractions(env.tick_count, env.current_world_seed, config),
                           dtype=env.dtype)[:, None, None]
    # Closed edges even when active swimmers use emigration/immigration.
    fractions = fractions * build_movement_mask(env.grid, False, env.dtype)
    biomass = np.stack([env.fgs[f].biomass for f in ids])
    reserve = np.stack([env.fgs[f].energy_reserve for f in ids])
    out_b, out_r = biomass[:, None] * fractions, reserve[:, None] * fractions
    b, r = _transfer_to_neighbours(biomass - out_b.sum(1), reserve - out_r.sum(1), out_b, out_r)
    for i, fid in enumerate(ids):
        env.fgs[fid].biomass = b[i].astype(env.dtype, copy=False)
        env.fgs[fid].energy_reserve = r[i].astype(env.dtype, copy=False)
