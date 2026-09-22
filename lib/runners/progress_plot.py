"""Shared progress smoothing and headless images; raw history is never changed."""

import numpy as np


def running_average(values, window=5):
    """Trailing mean including the current sample, with shorter initial windows."""
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    count = counts[end] - counts[start]
    return np.divide(sums[end] - sums[start], count,
                     out=np.full(len(values), np.nan), where=count > 0)


def plot_series(ax, x, y, label, window=5, color=None):
    line, = ax.plot(x, running_average(y, window), label=label, color=color, linewidth=1.8)
    ax.scatter(x, y, color=line.get_color(), alpha=.5, s=12)


def save_progress_plot(records, config, destination, window=5, iterations_per_generation=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    species = sorted({fid for r in records for fid in r["survival_ticks"]})
    divisor = iterations_per_generation or 1
    for fid in species:
        x = [r["step"] / divisor for r in records]
        y = [r["survival_ticks"].get(fid) for r in records]
        plot_series(ax, x, y, fid, window)
    ax.set(xlabel="Generation" if iterations_per_generation else "Completed training updates",
           ylabel="Viable ticks", ylim=(0, config["ticks"] * 1.02),
           title=f"Biomass band {config['lower']:g}-{config['upper']:g} x start | "
                 f"dots: evaluations; lines: trailing {window}-evaluation mean")
    ax.grid(alpha=.2)
    if species:
        ax.legend()
    temporary = destination.with_suffix(".png.tmp")
    fig.savefig(temporary, format="png", dpi=140)
    temporary.replace(destination)
