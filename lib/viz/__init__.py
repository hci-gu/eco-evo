"""Live visualisation utilities for Mareld.

The :class:`LiveVisualizer` in :mod:`lib.viz.pygame_viz` is imported lazily
so that ``import lib.viz`` is cheap and does not pull in pygame unless the
caller actually opts in via ``--visual``.
"""

__all__ = ["LiveVisualizer"]


def __getattr__(name):  # pragma: no cover - trivial lazy import shim
    if name == "LiveVisualizer":
        from .pygame_viz import LiveVisualizer
        return LiveVisualizer
    raise AttributeError(name)
