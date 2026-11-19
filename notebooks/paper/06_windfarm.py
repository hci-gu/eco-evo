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
    return FIGURES, R, RESULTS, ROOT, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Accumulated effects of the Poseidon wind farm

    `windfarm_poseidon`, 2 x 2: **noise** (the porpoises' energy cost; the policies do
    not observe it) and a **ban** on bottom and pelagic trawling inside the farm (the
    effort is removed). "Additive" is the sum of the two single effects; the gap to
    `noise_ban` is the interaction. The maps show the mean change per cell (both vs
    baseline) over the final window. **The footprint is a placeholder.**
    """)
    return


@app.cell
def _(R, RESULTS, ROOT, mo, np, plt):
    runs = R.find_runs(RESULTS)
    runs = runs[runs.scenario == "windfarm_poseidon"]
    mo.stop(runs.empty, mo.md(f"No `windfarm_poseidon` runs under `{RESULTS}`."))
    yr = R.yearly(runs)
    with np.load(ROOT / "maps" / "poseidon_placeholder.npz") as _f:
        footprint = {"mask": _f["mask"], "noise": _f["windfarm_noise"]}
    fig_site, _axes = plt.subplots(1, 2, figsize=(8, 3.4))
    _im = _axes[0].imshow(footprint["noise"], cmap="Blues")
    fig_site.colorbar(_im, ax=_axes[0], shrink=0.8, label="Noise (dB, placeholder)")
    _axes[0].contour(footprint["mask"], levels=[0.5], colors=[R.INK], linewidths=0.8)
    _axes[0].set_title("Noise field and footprint", fontsize=9)
    with np.load(ROOT / "maps" / "fishing_bottom_otter_trawls.npz") as _f:
        _trawl = _f["bottom_trawling"]
    _im2 = _axes[1].imshow(_trawl, cmap="Blues")
    _axes[1].contour(footprint["mask"], levels=[0.5], colors=[R.INK], linewidths=0.8)
    fig_site.colorbar(_im2, ax=_axes[1], shrink=0.8, label="Bottom-trawl effort (MW-h/km2/y)")
    _axes[1].set_title("Today's bottom trawling", fontsize=9)
    for _ax in _axes:
        _ax.set_xticks([])
        _ax.set_yticks([])
    window = mo.ui.slider(1, 20, value=10, label="Final window (years)")
    save = mo.ui.checkbox(label="Save figures to figures/paper/", value=R.save_by_default())
    mo.vstack([fig_site, mo.hstack([window, save])])
    return fig_site, footprint, runs, save, window, yr


@app.cell
def _(FIGURES, R, fig_site, np, plt, save, window, yr):
    _groups = R.FISH + R.MAMMALS + R.BENTHOS
    _fm = R.final_mean(yr, years=window.value)
    _m = _fm.groupby(["variant", "group"])["biomass"].mean().unstack("variant")
    _base = _m["baseline"]
    _eff = {v: _m[v] / _base - 1.0 for v in ("noise", "ban", "noise_ban") if v in _m}
    if "noise" in _eff and "ban" in _eff:
        _eff["additive"] = _eff["noise"] + _eff["ban"]
    _order = [v for v in ("noise", "ban", "noise_ban", "additive") if v in _eff]
    _labels = {"noise": "Noise", "ban": "Trawling ban", "noise_ban": "Both",
               "additive": "Sum of single effects"}
    fig_eff, _ax = plt.subplots(figsize=(10, 3.8))
    _x = np.arange(len(_groups))
    _w = 0.8 / len(_order)
    for _i, _v in enumerate(_order):
        _vals = [100 * _eff[_v].get(_g, np.nan) for _g in _groups]
        _kw = dict(color=R.CATEGORICAL[_i], width=_w * 0.9)
        if _v == "additive":
            _kw = dict(facecolor="none", edgecolor=R.INK_2, hatch="///", width=_w * 0.9, lw=0.6)
        _ax.bar(_x + (_i - (len(_order) - 1) / 2) * _w, _vals, label=_labels[_v], **_kw)
    _ax.axhline(0, color=R.INK_2, lw=0.8)
    _ax.set_xticks(_x, [R.LABELS[g] for g in _groups], rotation=30, ha="right", fontsize=8)
    R.style(_ax, ylabel=f"Change in biomass, last {window.value} y (%)")
    _ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig_eff.tight_layout()
    if save.value:
        R.save(fig_eff, "06_windfarm_effects", FIGURES)
        R.save(fig_site, "06_windfarm_site", FIGURES)
    fig_eff
    return


@app.cell
def _(FIGURES, R, footprint, mo, np, plt, runs, save, window):
    _groups = ["benthivorous_fish", "benthic_infauna", "norway_lobster", "porpoises"]

    def _mean_map(variant):
        _maps = []
        for _r in runs[runs.variant == variant].itertuples():
            _sp = R.load_spatial(_r.path)
            if _sp is None:
                continue
            _sel = _sp["t_days"] >= _sp["t_days"].max() - 365 * window.value
            _maps.append(_sp["biomass"][_sel].mean(0))
            _ids = _sp["group_ids"]
        return (np.mean(_maps, 0), _ids) if _maps else (None, None)

    _base, _ids = _mean_map("baseline")
    _both, _ = _mean_map("noise_ban")
    mo.stop(_base is None or _both is None, mo.md("No spatial snapshots."))
    fig_map, _axes = plt.subplots(1, len(_groups), figsize=(3 * len(_groups), 3))
    for _ax, _g in zip(_axes, _groups):
        _gi = _ids.index(_g)
        # Absolute change (t per km2 cell): a relative change blows up on
        # the cells the group hardly uses.
        _d = _both[_gi] - _base[_gi]
        _lim = max(1e-9, float(np.nanpercentile(np.abs(_d), 99)))
        _im = _ax.imshow(_d, cmap=R.diverging_cmap(), vmin=-_lim, vmax=_lim)
        _mask = footprint["mask"]
        if _mask.shape == _d.shape:
            _ax.contour(_mask, levels=[0.5], colors=[R.INK], linewidths=0.8)
        _ax.set_title(R.LABELS[_g], fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
        fig_map.colorbar(_im, ax=_ax, shrink=0.75, label="t per cell, both - baseline")
    fig_map.tight_layout()
    if save.value:
        R.save(fig_map, "06_windfarm_maps", FIGURES)
    fig_map
    return


if __name__ == "__main__":
    app.run()
