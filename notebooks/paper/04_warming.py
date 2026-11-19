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
    # Warming

    `warming`: every temperature layer + **dT** from the start, relative to today's
    climate: each group's Q10 multiplies its resting metabolism, ingestion, growth
    (zooplankton) and grazing (microzooplankton) by q10^(dT/10). Porpoises and seals
    (endotherms), the producers, nutrients and stratification are not warmed.
    Today's fishing throughout.
    """)
    return


@app.cell
def _(R, RESULTS, mo):
    runs = R.find_runs(RESULTS)
    runs = runs[runs.scenario == "warming"]
    mo.stop(runs.empty, mo.md(f"No `warming` runs under `{RESULTS}`."))
    yr = R.yearly(runs)
    yr["dT"] = yr.sweep_value.astype(float)
    window = mo.ui.slider(1, 20, value=10, label="Final window (years)")
    save = mo.ui.checkbox(label="Save figures to figures/paper/", value=R.save_by_default())
    mo.hstack([window, save])
    return save, window, yr


@app.cell
def _(FIGURES, R, plt, save, window, yr):
    _groups = R.FISH + R.MAMMALS
    _fm = R.final_mean(yr, years=window.value)
    _fm["dT"] = _fm.sweep_value.astype(float)
    _ref = _fm[_fm.dT == 0.0].groupby("group")["biomass"].mean()
    _fm["rel"] = _fm.biomass / _fm.group.map(_ref)
    fig_resp, _ax = plt.subplots(figsize=(6.5, 4))
    _colors = R.group_colors(_groups)
    for _g in _groups:
        _d = _fm[_fm.group == _g].groupby("dT")["rel"].agg(["mean", "min", "max"])
        _ax.fill_between(_d.index, _d["min"], _d["max"], color=_colors[_g], alpha=0.12, lw=0)
        _ax.plot(_d.index, _d["mean"], color=_colors[_g], lw=1.8, marker="o", ms=4,
                 label=R.LABELS[_g])
    _ax.axhline(1.0, color=R.INK_2, lw=0.6, ls=":")
    R.style(_ax, ylabel=f"Biomass, last {window.value} y, relative to dT = 0",
            xlabel="Warming dT (deg C)")
    _ax.legend(frameon=False, fontsize=8)
    fig_resp.tight_layout()
    if save.value:
        R.save(fig_resp, "04_warming_response", FIGURES)
    fig_resp
    return


@app.cell
def _(FIGURES, R, np, plt, save, yr):
    _groups = ["zooplankton", "krill"] + R.FISH
    _ds = sorted(yr.dT.unique())
    _cols = R.ramp(len(_ds))
    fig_ts, _axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True)
    for _ax, _g in zip(_axes.flat, _groups):
        for _dt, _c in zip(_ds, _cols):
            _m = yr[(yr.group == _g) & (yr.dT == _dt)].groupby("year")["biomass"].mean()
            _ax.plot(_m.index, _m.values / 1e3, color=_c, lw=1.4, label=f"+{_dt:g} C")
        _ax.set_title(R.LABELS[_g], fontsize=9)
        R.style(_ax)
    for _ax in _axes.flat[len(_groups):]:
        _ax.set_visible(False)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("Biomass (kt)", fontsize=9)
    for _ax in _axes[-1]:
        _ax.set_xlabel("Year", fontsize=9)
    _axes.flat[0].legend(frameon=False, fontsize=7)
    fig_ts.tight_layout()
    if save.value:
        R.save(fig_ts, "04_warming_timeseries", FIGURES)
    fig_ts
    return


@app.cell
def _(R, mo, window, yr):
    _fm = R.final_mean(yr[yr.group.isin(R.FISH)], years=window.value, value="reserve")
    _fm["dT"] = _fm.sweep_value.astype(float)
    _t = _fm.groupby(["group", "dT"])["reserve"].mean().unstack("dT")
    _t = _t.div(_t[0.0], axis=0) if 0.0 in _t.columns else _t
    _t.index = [R.LABELS[g] for g in _t.index]
    mo.vstack([mo.md("**Energy reserve (MJ), last window, relative to dT = 0**"),
               mo.ui.table(_t.round(3).reset_index(names="group"), selection=None)])
    return


if __name__ == "__main__":
    app.run()
