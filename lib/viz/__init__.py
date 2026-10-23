"""Live visualisation utilities for Mareld.

The :class:`LiveVisualizer` in :mod:`lib.viz.pygame_viz` is imported lazily
so that ``import lib.viz`` is cheap and does not pull in pygame unless the
caller actually opts in via ``--visual``.
"""

__all__ = ["LiveVisualizer", "HEATMAP_NORMS", "add_heatmap_norm_argument"]

# Heatmap normalisation at start (section 197): "rolling" takes the
# colour reference from the frame being drawn, "fixed" from the start of
# the rollout. The N key and the status-bar button still toggle it.
HEATMAP_NORMS = ("rolling", "fixed")


def add_heatmap_norm_argument(parser):
    parser.add_argument(
        "--norm", choices=HEATMAP_NORMS, default="rolling",
        help="With --visual: heatmap normalisation at start. 'rolling' "
             "(default): each frame's own maximum, so the pattern stays "
             "readable however small the numbers get; 'fixed': the "
             "rollout-start maximum, so brightness compares across frames. "
             "The N key or the status-bar button toggles it while running.")


def __getattr__(name):  # pragma: no cover - trivial lazy import shim
    if name == "LiveVisualizer":
        from .pygame_viz import LiveVisualizer
        return LiveVisualizer
    raise AttributeError(name)
