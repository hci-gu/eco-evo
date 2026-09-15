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


def normal(keys, shape, stream=0, dtype=torch.float32):
    u = uniform(keys, shape, stream * 2 + 1, dtype)
    v = uniform(keys, shape, stream * 2 + 2, dtype)
    return torch.sqrt(-2.0 * torch.log(u)) * torch.cos(2.0 * math.pi * v)
