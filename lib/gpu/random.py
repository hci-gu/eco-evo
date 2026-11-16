"""Counter-based random fields, independent of batching and execution order.

Integer hashing stays on device. Keys identify worlds/pairs; counters identify
samples/ticks. Both ARS signs use the same keys. This intentionally does not
reproduce NumPy's random stream; injected fields support numerical parity tests.
"""

import math

import torch


def hash32(x):
    x = x & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


def fold_in(keys, value):
    return hash32(keys ^ hash32(value + torch.zeros_like(keys) + 0x9E3779B9))


def uniform(keys, shape, stream=0, dtype=torch.float32):
    """Return [len(keys), *shape], with an open (0, 1) interval."""
    count = math.prod(shape)
    counters = torch.arange(count, device=keys.device, dtype=torch.int64)
    bits = hash32(fold_in(keys, stream).reshape(-1, 1) ^ hash32(counters + 1))
    # 23 bits ensure neither endpoint rounds to 0/1 in FP32.
    result = ((bits >> 9).to(dtype) + 0.5) * (1.0 / 8388608.0)
    return result.reshape(keys.numel(), *shape)


def uniform_at(keys, counters):
    """Uniforms at explicit integer ``counters`` under per-row ``keys``.

    ``keys`` [E] (already folded with tick and stream), ``counters`` any
    int64 shape; returns [E, *counters.shape] in float32, bit-identical to
    ``lib.environments.ecosystem_env.encounter.uniform_at`` (section 275).
    """
    bits = hash32(keys.reshape(-1, *([1] * counters.dim())) ^ hash32(counters + 1)[None])
    return ((bits >> 9).to(torch.float32) + 0.5) * (1.0 / 8388608.0)


_C2_I32 = 0x846CA68B - (1 << 32)      # the second constant as a signed int32


def hash32_i32(x):
    """``hash32`` on int32 tensors (half the memory traffic): the multiplies
    wrap modulo 2^32 and each right shift is masked to a logical shift, so
    the bits equal ``hash32`` of the same 32-bit value."""
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * 0x7FEB352D
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * _C2_I32
    return x ^ ((x >> 16) & 0xFFFF)


def uniform_bits_i32(x):
    """Uniforms in (0, 1) from int32 ``x`` = key ^ hash32(counter + 1), as
    ``uniform_at`` builds them (section 275)."""
    bits = hash32_i32(x)
    return (((bits >> 9) & 0x7FFFFF).to(torch.float32) + 0.5) * (1.0 / 8388608.0)


def normal(keys, shape, stream=0, dtype=torch.float32):
    u = uniform(keys, shape, stream * 2 + 1, dtype)
    v = uniform(keys, shape, stream * 2 + 2, dtype)
    return torch.sqrt(-2.0 * torch.log(u)) * torch.cos(2.0 * math.pi * v)
