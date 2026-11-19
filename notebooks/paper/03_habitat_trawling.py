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
    # Habitat degradation by bottom trawling

    `habitat_trawling`: the bottom-trawl field x **b** (pelagic trawling at today's level).
    In the model bottom trawling is a mortality on the bed (infauna, exposed epifauna,
    Norway lobster) and on the demersal fish; it does not lower the beds' carrying capacity.
    """)
    return


@app.cell
def _(R, RESULTS, mo):
    runs = R.find_runs(RESULTS)
    runs = runs[runs.scenario == "habitat_trawling"]
    mo.stop(runs.empty, mo.md(f"No `habitat_trawling` runs under `{RESULTS}`."))
    yr = R.yearly(runs)
    window = mo.ui.slider(1, 20, value=10, label="Final window (years)")
    save = mo.ui.checkbox(label="Save figures to figures/paper/", value=R.save_by_default())
    mo.hstack([window, save])
    return runs, save, window, yr


@app.cell
def _(FIGURES, R, plt, save, window, yr):
    _groups = R.BENTHOS + ["benthivorous_fish", "whiting", "predatory_fish"]
    _fm = R.final_mean(yr, years=window.value)
    _fm["b"] = _fm.sweep_value.astype(float)
    _ref = _fm[_fm.b == 0.0].groupby("group")["biomass"].mean()
    _fm["rel"] = _fm.biomass / _fm.group.map(_ref)
    fig_resp, _ax = plt.subplots(figsize=(6.5, 4))
    _colors = R.CATEGORICAL
    for _i, _g in enumerate(_groups):
        _d = _fm[_fm.group == _g].groupby("b")["rel"].agg(["mean", "min", "max"])
        _ax.fill_between(_d.index, _d["min"], _d["max"], color=_colors[_i], alpha=0.12, lw=0)
        _ax.plot(_d.index, _d["mean"], color=_colors[_i], lw=1.8, marker="o", ms=4,
                 label=R.LABELS[_g])
    _ax.axvline(1.0, color=R.INK_2, lw=0.6, ls=":")
    _ax.text(1.02, 0.02, "today", transform=_ax.get_xaxis_transform(), fontsize=8, color=R.INK_2)
    R.style(_ax, ylabel=f"Biomass, last {window.value} y, relative to no trawling",
            xlabel="Bottom-trawl effort, b x today's")
    _ax.legend(frameon=False, fontsize=8)
    fig_resp.tight_layout()
    if save.value:
        R.save(fig_resp, "03_trawling_response", FIGURES)
    fig_resp
    return


@app.cell
def _(FIGURES, R, np, plt, save, yr):
    _groups = R.BENTHOS + ["benthivorous_fish"]
    _bs = sorted(yr.sweep_value.astype(float).unique())
    _cols = R.ramp(len(_bs))
    fig_ts, _axes = plt.subplots(1, len(_groups), figsize=(12, 2.8), sharex=True)
    for _ax, _g in zip(np.atleast_1d(_axes), _groups):
        for _b, _c in zip(_bs, _cols):
            _d = yr[(yr.group == _g) & (yr.sweep_value.astype(float) == _b)]
            _m = _d.groupby("year")["biomass"].mean()
            _ax.plot(_m.index, _m.values / 1e3, color=_c, lw=1.4, label=f"b = {_b:g}")
        _ax.set_title(R.LABELS[_g], fontsize=9)
        R.style(_ax, xlabel="Year")
    np.atleast_1d(_axes)[0].set_ylabel("Biomass (kt)", fontsize=9)
    np.atleast_1d(_axes)[-1].legend(frameon=False, fontsize=7)
    fig_ts.tight_layout()
    if save.value:
        R.save(fig_ts, "03_trawling_timeseries", FIGURES)
    fig_ts
    return


@app.cell
def _(FIGURES, R, mo, np, plt, runs, save):
    _picks = []
    for _b in sorted(runs.sweep_value.astype(float).unique()):
        _r = runs[(runs.sweep_value.astype(float) == _b)].sort_values("seed").iloc[0]
        _sp = R.load_spatial(_r.path)
        if _sp is not None:
            _picks.append((_b, _sp))
    mo.stop(not _picks, mo.md("No spatial snapshots (`spatial_every_days`)."))
    _g = "benthic_infauna"
    _gi = _picks[0][1]["group_ids"].index(_g)
    _vmax = max(float(sp["biomass"][-1, _gi].max()) for _, sp in _picks)
    fig_map, _axes = plt.subplots(1, len(_picks), figsize=(2.6 * len(_picks), 2.8))
    for _ax, (_b, _sp) in zip(np.atleast_1d(_axes), _picks):
        _im = _ax.imshow(_sp["biomass"][-1, _gi], cmap="Blues", vmin=0, vmax=_vmax)
        _ax.set_title(f"b = {_b:g}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    fig_map.colorbar(_im, ax=_axes, shrink=0.8, label="Infauna, t per cell (last snapshot)")
    if save.value:
        R.save(fig_map, "03_infauna_maps", FIGURES)
    fig_map
    return


if __name__ == "__main__":
    app.run()
