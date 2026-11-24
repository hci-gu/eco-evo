"""Device selection for the tensor engine: CUDA, Apple MPS or CPU.

MPS (Apple silicon) has no float64. The engine's accumulators (reward sums,
ledgers, books, spawn fields) are float64 on CUDA and CPU and float32 on
MPS: ``wide_dtype``. MPS has no CUDA graphs either, so it runs eager.
"""

import os

import torch


def mps_available():
    """MPS is usable; MARELD_DISABLE_MPS=1 hides it (as CUDA_VISIBLE_DEVICES=""
    hides CUDA), e.g. to force the CPU engine or in the tests."""
    if os.environ.get("MARELD_DISABLE_MPS", "") not in ("", "0"):
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def gpu_available():
    """True when torch sees a CUDA or an MPS device."""
    return torch.cuda.is_available() or mps_available()


def default_gpu_device():
    """``cuda`` when available, else ``mps``, else None."""
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return None


def resolve_device(name):
    """``auto`` -> the default GPU (else cpu); other names are checked."""
    if name in (None, "auto"):
        return torch.device(default_gpu_device() or "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: torch sees no CUDA device. "
                           "Use --device mps on Apple silicon or --device cpu.")
    if device.type == "mps" and not mps_available():
        raise RuntimeError("MPS is unavailable: needs Apple silicon and a "
                           "native arm64 torch build.")
    if device.type not in ("cuda", "mps", "cpu"):
        raise ValueError("The tensor engine supports cuda, mps and cpu")
    return device


def wide_dtype(device):
    """Accumulator dtype: float64, or float32 on MPS (no float64 there)."""
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def default_execution(device):
    """CUDA graphs on CUDA, eager elsewhere."""
    return "cuda-graph" if torch.device(device).type == "cuda" else "eager"


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_event(device):
    """A recorded event on ``device`` with ``query()``, or None on CPU."""
    device = torch.device(device)
    if device.type == "cuda":
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event
    if device.type == "mps":
        event = torch.mps.event.Event()
        event.record()
        return event
    return None
