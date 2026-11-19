import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = Path(mo.notebook_dir()).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from lib.scenarios import results as R

    RESULTS = Path(os.environ.get("PAPER_RESULTS", ROOT / "results" / "paper"))
    FIGURES = ROOT / "figures" / "paper"
    return FIGURES, R, RESULTS, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Today's fishing vs no fishing

    `current_fishing` keeps today's trawl effort (EMODnet maps) for the whole run;
    `no_fishing` sets both trawl fields to zero after a warm-up with today's fishing.
    Lines are the seed mean, bands the seed range.
    """)
    return


@app.cell
def _(R, RESULTS, mo):
    runs = R.find_runs(RESULTS)
    runs = runs[runs.scenario.isin(["current_fishing", "no_fishing"])]
    mo.stop(runs.empty, mo.md(f"No runs of `current_fishing` / `no_fishing` under `{RESULTS}`."))
    save = mo.ui.checkbox(label="Save figures to figures/paper/", value=R.save_by_default())
    mo.vstack([mo.ui.table(runs.drop(columns=["path"]), selection=None), save])
    return runs, save


@app.cell
def _(R, runs):
    groups = R.FISH + R.MAMMALS + R.BENTHOS
    series = R.daily(runs, groups=groups, every_days=5)
    start = series[series.t_years == 0].set_index(["scenario", "seed", "group"])["biomass"]
    series["rel"] = series["biomass"] / series.set_index(
        ["scenario", "seed", "group"]).index.map(start).values
    yr = R.yearly(runs)
    return groups, series, yr


@app.cell
def _(FIGURES, R, groups, np, plt, save, series):
    _colors = {"current_fishing": R.CATEGORICAL[0], "no_fishing": R.CATEGORICAL[1]}
    _labels = {"current_fishing": "Today's fishing", "no_fishing": "No fishing"}
    _n = len(groups)
    _cols = 5
    _rows = int(np.ceil(_n / _cols))
    fig_traj, _axes = plt.subplots(_rows, _cols, figsize=(12, 2.4 * _rows), sharex=True)
    for _ax, _g in zip(_axes.flat, groups):
        for _sc, _df in series[series.group == _g].groupby("scenario"):
            _b = _df.groupby("t_years")["rel"].agg(["mean", "min", "max"])
            _ax.fill_between(_b.index, _b["min"], _b["max"], color=_colors[_sc], alpha=0.18, lw=0)
            _ax.plot(_b.index, _b["mean"], color=_colors[_sc], lw=1.6, label=_labels[_sc])
        _ax.axhline(1.0, color=R.INK_2, lw=0.6, ls=":")
        _ax.set_title(R.LABELS.get(_g, _g), fontsize=9, color=R.INK)
        R.style(_ax)
    for _ax in _axes.flat[_n:]:
        _ax.set_visible(False)
    for _ax in _axes[-1]:
        _ax.set_xlabel("Years", fontsize=9)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("B / B(0)", fontsize=9)
    _axes.flat[0].legend(frameon=False, fontsize=8)
    fig_traj.tight_layout()
    if save.value:
        R.save(fig_traj, "01_biomass_today_vs_none", FIGURES)
    fig_traj
    return


@app.cell
def _(R, mo, yr):
    _fm = R.final_mean(yr, years=10)
    _m = _fm.groupby(["scenario", "group"])["biomass"].mean().unstack("scenario")
    if {"no_fishing", "current_fishing"} <= set(_m.columns):
        _m["no / today"] = _m["no_fishing"] / _m["current_fishing"]
    _catch = (yr[(yr.scenario == "current_fishing") & yr.group.isin(R.FISH + R.BENTHOS)]
              .groupby(["group", "seed"])["catch"].mean().groupby("group").mean())
    _m["annual catch today (t)"] = _catch
    _m.index = [R.LABELS.get(g, g) for g in _m.index]
    mo.vstack([mo.md("**Last 10 years: mean biomass (t) and the annual catch under today's fishing**"),
               mo.ui.table(_m.round(3).reset_index(names="group"), selection=None)])
    return


if __name__ == "__main__":
    app.run()
