"""Read the paper scenario runs (``lib/scenarios/runner.py``) into pandas.

Used by the marimo notebooks in ``notebooks/paper/``; needs the ``paper``
dependency group (``uv sync --group paper``).

    runs = find_runs("results/paper")                 # one row per run
    yr = yearly(runs[runs.scenario == "fishing_constant"])
    # columns: scenario, variant, sweep_name, sweep_value, seed, year,
    #          group, biomass (annual mean t), catch (t in that year),
    #          catch_<impact> ...
"""

import glob
import json
import os

import numpy as np

# Groups whose impact loss is a fishery catch (the benthos' is the trawl's
# kill on the bed, Nephrops' its catch plus kill; porpoises' gillnet
# bycatch rides on the bottom-trawl field).
FISH = ["planktivorous_fish", "benthivorous_fish", "whiting", "predatory_fish", "mackerel"]
MAMMALS = ["porpoises", "seals"]
BENTHOS = ["benthic_infauna", "mobile_epifauna", "norway_lobster"]
PLANKTON = ["phytoplankton", "subsurface_phytoplankton", "microzooplankton",
            "zooplankton", "krill"]

LABELS = {
    "planktivorous_fish": "Planktivorous fish",
    "benthivorous_fish": "Benthivorous fish",
    "whiting": "Whiting",
    "predatory_fish": "Water-column predators",
    "mackerel": "Mackerel",
    "porpoises": "Harbour porpoise",
    "seals": "Harbour seal",
    "benthic_infauna": "Infauna",
    "mobile_epifauna": "Epifauna",
    "norway_lobster": "Norway lobster",
    "phytoplankton": "Phytoplankton (mixed layer)",
    "subsurface_phytoplankton": "Phytoplankton (subsurface)",
    "microzooplankton": "Microzooplankton",
    "zooplankton": "Zooplankton",
    "krill": "Krill",
}

# The dataviz reference palette (validated light-mode categorical order)
# and its blue ramp for magnitudes (a sweep value).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def group_colors(groups):
    """Fixed colour per group (by its place in FISH + MAMMALS + ...)."""
    order = FISH + MAMMALS
    return {g: CATEGORICAL[order.index(g) % len(CATEGORICAL)] if g in order
            else INK_2 for g in groups}


def ramp(n):
    """``n`` light-to-dark steps of the blue ramp (a magnitude scale)."""
    if n <= 1:
        return [BLUE_RAMP[5]]
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def style(ax, ylabel=None, xlabel=None):
    """Recessive axes for a paper figure."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    return ax


def find_runs(root="results/paper"):
    """One row per finished run under ``root``."""
    import pandas as pd
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(root, "*", "*", "seed_*", "meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        sweep_value = meta.get("sweep_value")
        rows.append({
            "scenario": meta["scenario"],
            "variant": meta["variant"],
            "sweep_name": (",".join(meta["sweep_name"])
                           if isinstance(meta.get("sweep_name"), list) else meta.get("sweep_name")),
            "sweep_value": (tuple(sweep_value) if isinstance(sweep_value, list) else sweep_value),
            "seed": meta["seed"],
            "years": meta["years"],
            "grid": "x".join(str(v) for v in meta["grid"]),
            "run_name": meta["run_name"],
            "git": meta.get("git"),
            "path": os.path.dirname(meta_path),
        })
    return pd.DataFrame(rows, columns=["scenario", "variant", "sweep_name", "sweep_value",
                                       "seed", "years", "grid", "run_name", "git", "path"])


def load(path):
    """A run's ``timeseries.npz`` as a dict of arrays (lists for ids)."""
    with np.load(os.path.join(path, "timeseries.npz")) as d:
        out = {k: d[k] for k in d.files}
    for key in ("group_ids", "impact_ids", "loss_causes", "decision_makers"):
        out[key] = [str(v) for v in out[key]]
    return out


def load_spatial(path):
    p = os.path.join(path, "spatial.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as d:
        return {"t_days": d["t_days"], "biomass": d["biomass"].astype(np.float32),
                "group_ids": [str(v) for v in d["group_ids"]]}


def yearly_one(ts):
    """Long frame of one run: annual mean biomass and that year's catch."""
    import pandas as pd
    t = ts["t_days"]
    year = np.floor((t - 1e-9) / 365.0).astype(int)    # row 0 (t = 0) -> -1
    n_years = int(year.max()) + 1 if len(year) > 1 else 0
    rows = []
    groups = ts["group_ids"]
    impacts = ts["impact_ids"]
    # Cumulative books at each year's end (the last row of that year).
    ends = [np.flatnonzero(year == y)[-1] for y in range(n_years)]
    catch_end = ts["catch"][ends]                      # [Y, I, G]
    catch_year = np.diff(np.concatenate([np.zeros_like(catch_end[:1]), catch_end]), axis=0)
    loss_end = ts["losses"][ends]
    loss_year = np.diff(np.concatenate([np.zeros_like(loss_end[:1]), loss_end]), axis=0)
    causes = ts["loss_causes"]
    for y in range(n_years):
        sel = year == y
        b = ts["biomass"][sel].mean(0)
        r = ts["reserve"][sel].mean(0)
        for g, fid in enumerate(groups):
            row = {"year": y + 1, "group": fid, "biomass": float(b[g]),
                   "reserve": float(r[g]),
                   "catch": float(catch_year[y, :, g].sum())}
            for i, iid in enumerate(impacts):
                row[f"catch_{iid}"] = float(catch_year[y, i, g])
            for c, cause in enumerate(causes):
                row[f"loss_{cause}"] = float(loss_year[y, c, g])
            rows.append(row)
    return pd.DataFrame(rows)


def yearly(runs):
    """``yearly_one`` for every row of ``runs`` (a ``find_runs`` frame)."""
    import pandas as pd
    frames = []
    for run in runs.itertuples():
        df = yearly_one(load(run.path))
        for col in ("scenario", "variant", "sweep_name", "sweep_value", "seed"):
            # A list keeps a tuple sweep value (two axes) whole per row.
            df[col] = [getattr(run, col)] * len(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def daily(runs, groups=None, every_days=1):
    """Long frame of the recorded time series: t_years, group, biomass."""
    import pandas as pd
    frames = []
    for run in runs.itertuples():
        ts = load(run.path)
        step = max(1, int(round(every_days / max(1e-9, ts["t_days"][1] - ts["t_days"][0])))) \
            if len(ts["t_days"]) > 1 else 1
        idx = np.arange(0, len(ts["t_days"]), step)
        for g, fid in enumerate(ts["group_ids"]):
            if groups is not None and fid not in groups:
                continue
            frames.append(pd.DataFrame({
                "t_years": ts["t_days"][idx] / 365.0, "group": fid,
                "biomass": ts["biomass"][idx, g], "reserve": ts["reserve"][idx, g],
                "scenario": run.scenario, "variant": run.variant,
                "sweep_value": [run.sweep_value] * len(idx), "seed": run.seed}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def seed_band(df, by, value):
    """Mean, min and max of ``value`` over seeds, grouped by ``by``."""
    g = df.groupby(by, dropna=False)[value]
    out = g.agg(["mean", "min", "max", "count"]).reset_index()
    return out


def final_mean(yr, years=10, value="biomass"):
    """Mean of ``value`` over each run's last ``years`` years, per group."""
    last = yr.groupby(["scenario", "variant", "seed"])["year"].transform("max")
    tail = yr[yr["year"] > last - years]
    keys = ["scenario", "variant", "sweep_name", "sweep_value", "seed", "group"]
    return tail.groupby(keys, dropna=False)[value].mean().reset_index()


def save(fig, name, directory):
    """``<directory>/<name>.pdf`` and ``.png`` (300 dpi)."""
    os.makedirs(directory, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(directory, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    return [os.path.join(directory, f"{name}.{ext}") for ext in ("pdf", "png")]


def diverging_cmap():
    """Orange - neutral grey - blue (two hues, a grey midpoint)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("paper_div", ["#eb6834", "#f0efec", "#2a78d6"])


def save_by_default():
    """The notebooks' "Save figures" box starts ticked with PAPER_SAVE=1
    (tools/paper/run_all.py runs them headless that way)."""
    return os.environ.get("PAPER_SAVE", "") == "1"
