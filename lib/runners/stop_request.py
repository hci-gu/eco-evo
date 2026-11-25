"""A cooperative stop for the training entry points (section 307).

train_gpu.py installs one process-wide check for the run's lifetime; the
loops that can run long between its own checks - the tensor engine's
rollouts (``lib.gpu.rollout.RolloutRunner.run``) and the viewer probe's
CPU tick loop (``train._probe_biomass``) - read it and raise
``StopRequested``. Without a check (the default, every other caller)
nothing changes.
"""


class StopRequested(BaseException):
    """A training entry point asked the running loop to stop (Ctrl+C, SIGTERM).

    A BaseException, so neither the viewer's ``except Exception`` nor a
    probe's ``except KeyboardInterrupt`` turns it into a partial result:
    it reaches the entry point, which saves the last completed update.
    """


_stop_check = None


def set_stop_check(check):
    """Install ``check`` (no arguments, True = stop); None removes it."""
    global _stop_check
    _stop_check = check


def stop_check():
    """The installed check, or None."""
    return _stop_check


def raise_if_stopped():
    """Raise ``StopRequested`` when the installed check fires."""
    check = _stop_check
    if check is not None and check():
        raise StopRequested
