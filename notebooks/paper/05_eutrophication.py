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
    # Eutrophication

    `eutrophication`: the new-nitrogen source (the 50 m nitrate reservoir that feeds
    the subsurface layer) x **n**. Today's fishing throughout.
    """)
    return


@app.cell
def _(R, RESULTS, mo):
    runs = R.find_runs(RESULTS)
    runs = runs[runs.scenario == "eutrophication"]
    mo.stop(runs.empty, mo.md(f"No `eutrophication` runs under `{RESULTS}`."))
    yr = R.yearly(runs)
    yr["n"] = yr.sweep_value.astype(float)
    save = mo.ui.checkbox(label="Save figures to figures/paper/", value=R.save_by_default())
    save
    return runs, save, yr


@app.cell
def _(FIGURES, R, plt, save, yr):
    _groups = R.PLANKTON + R.FISH
    _fm = R.final_mean(yr, years=5)
    _fm["n"] = _fm.sweep_value.astype(float)
    _ref = _fm[_fm.n == 1.0].groupby("group")["biomass"].mean()
    _fm["rel"] = _fm.biomass / _fm.group.map(_ref)
    fig_resp, _axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for _ax, _set, _title in ((_axes[0], R.PLANKTON, "Plankton"), (_axes[1], R.FISH, "Fish")):
        for _i, _g in enumerate(_set):
            _d = _fm[_fm.group == _g].groupby("n")["rel"].agg(["mean", "min", "max"])
            _ax.fill_between(_d.index, _d["min"], _d["max"], color=R.CATEGORICAL[_i], alpha=0.12, lw=0)
            _ax.plot(_d.index, _d["mean"], color=R.CATEGORICAL[_i], lw=1.8, marker="o", ms=4,
                     label=R.LABELS[_g])
        _ax.axhline(1.0, color=R.INK_2, lw=0.6, ls=":")
        _ax.set_title(_title, fontsize=10, loc="left")
        R.style(_ax, xlabel="Nitrate supply, n x today's")
        _ax.legend(frameon=False, fontsize=8)
    _axes[0].set_ylabel("Biomass, last 5 y, relative to n = 1", fontsize=9)
    fig_resp.tight_layout()
    if save.value:
        R.save(fig_resp, "05_eutrophication_response", FIGURES)
    fig_resp
    return


@app.cell
def _(FIGURES, R, np, plt, runs, save):
    _groups = ["phytoplankton", "subsurface_phytoplankton", "zooplankton"]
    _d = R.daily(runs, groups=_groups, every_days=1)
    _d["n"] = _d.sweep_value.astype(float)
    _last = np.floor(_d.t_years.max())
    _d = _d[_d.t_years >= _last - 1]
    _ns = sorted(_d.n.unique())
    _cols = R.ramp(len(_ns))
    fig_season, _axes = plt.subplots(1, len(_groups), figsize=(12, 3))
    for _ax, _g in zip(_axes, _groups):
        for _n, _c in zip(_ns, _cols):
            _m = _d[(_d.group == _g) & (_d.n == _n)].groupby("t_years")["biomass"].mean()
            _ax.plot((_m.index - (_last - 1)) * 12, _m.values / 1e3, color=_c, lw=1.4,
                     label=f"n = {_n:g}")
        _ax.set_title(R.LABELS[_g], fontsize=9)
        R.style(_ax, xlabel="Months into the last full year")
    _axes[0].set_ylabel("Biomass (kt)", fontsize=9)
    _axes[-1].legend(frameon=False, fontsize=7)
    fig_season.tight_layout()
    if save.value:
        R.save(fig_season, "05_eutrophication_season", FIGURES)
    fig_season
    return


if __name__ == "__main__":
    app.run()
