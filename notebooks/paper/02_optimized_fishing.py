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
    # Optimised fishing: rebuild, then harvest

    * `fishing_constant`: today's fishing field x **k** for the whole run.
    * `fishing_rebuild`: x **r** for the first 5 years (rebuilding), then x **k**.

    Both warm up with today's fishing. The catch is the engine's booked impact loss
    of the fish groups (and, separately, Norway lobster). The box is an open
    population: recruits are imported and planktivores, the water column and
    mackerel exchange with fixed outside densities, so the yield curve is the
    box's local catch, not the stocks'.
    """)
    return


@app.cell
def _(R, RESULTS, mo):
    runs = R.find_runs(RESULTS)
    runs = runs[runs.scenario.isin(["fishing_constant", "fishing_rebuild"])]
    mo.stop(runs.empty, mo.md(f"No `fishing_constant` / `fishing_rebuild` runs under `{RESULTS}`."))
    yr = R.yearly(runs)

    def _strategy(row):
        if row.scenario == "fishing_constant":
            return "constant", float(row.sweep_value)
        r, k = row.sweep_value
        return f"rebuild r={r:g}", float(k)

    _sk = [_strategy(_row) for _row in runs.itertuples()]
    runs = runs.assign(strategy=[_s for _s, _ in _sk], k=[_k for _, _k in _sk])
    yr = yr.merge(runs[["scenario", "variant", "seed", "strategy", "k"]],
                  on=["scenario", "variant", "seed"])
    catch_group = mo.ui.dropdown(
        options={"All fish": "fish", **{R.LABELS[g]: g for g in R.FISH},
                 R.LABELS["norway_lobster"]: "norway_lobster"},
        value="All fish", label="Catch of")
    save = mo.ui.checkbox(label="Save figures to figures/paper/")
    mo.hstack([catch_group, save])
    return catch_group, runs, save, yr


@app.cell
def _(R, catch_group, yr):
    _sel = R.FISH if catch_group.value == "fish" else [catch_group.value]
    per_run = (yr[yr.group.isin(_sel)]
               .groupby(["strategy", "k", "seed", "year"])["catch"].sum().reset_index())
    cumulative = (per_run.groupby(["strategy", "k", "seed"])["catch"].sum()
                  .groupby(["strategy", "k"]).agg(["mean", "min", "max"]).reset_index())
    return cumulative, per_run


@app.cell
def _(FIGURES, R, catch_group, cumulative, plt, runs, save):
    _order = sorted(cumulative.strategy.unique(), key=lambda s: (s != "constant", s))
    fig_yield, _ax = plt.subplots(figsize=(6.5, 4))
    for _i, _s in enumerate(_order):
        _d = cumulative[cumulative.strategy == _s].sort_values("k")
        _c = R.CATEGORICAL[_i]
        _ax.fill_between(_d.k, _d["min"] / 1e3, _d["max"] / 1e3, color=_c, alpha=0.15, lw=0)
        _ax.plot(_d.k, _d["mean"] / 1e3, color=_c, lw=2, marker="o", ms=4, label=_s)
    _today = cumulative[(cumulative.strategy == "constant") & (cumulative.k == 1.0)]
    if not _today.empty:
        _ax.scatter([1.0], _today["mean"] / 1e3, s=80, facecolor="none", edgecolor=R.INK,
                    zorder=5, label="today's fishing")
    R.style(_ax, ylabel=f"Cumulative catch over {runs.years.max():.0f} years (kt)",
            xlabel="Effort after rebuilding, k x today's")
    _ax.set_title(f"Catch: {catch_group.selected_key}", fontsize=10, color=R.INK, loc="left")
    _ax.legend(frameon=False, fontsize=8)
    fig_yield.tight_layout()
    if save.value:
        R.save(fig_yield, f"02_cumulative_catch_{catch_group.value}", FIGURES)
    fig_yield
    return


@app.cell
def _(mo, runs):
    _ks = sorted(runs.k.unique())
    k_pick = mo.ui.dropdown(options=[f"{k:g}" for k in _ks],
                            value=f"{1.0 if 1.0 in _ks else _ks[0]:g}", label="k")
    k_pick
    return (k_pick,)


@app.cell
def _(FIGURES, R, catch_group, k_pick, per_run, plt, save):
    _k = float(k_pick.value)
    _d = per_run[per_run.k == _k]
    _order = sorted(_d.strategy.unique(), key=lambda s: (s != "constant", s))
    fig_annual, _ax = plt.subplots(figsize=(6.5, 3.5))
    for _i, _s in enumerate(sorted(per_run.strategy.unique(), key=lambda s: (s != "constant", s))):
        if _s not in _order:
            continue
        _b = _d[_d.strategy == _s].groupby("year")["catch"].agg(["mean", "min", "max"])
        _ax.fill_between(_b.index, _b["min"], _b["max"], color=R.CATEGORICAL[_i], alpha=0.15, lw=0)
        _ax.plot(_b.index, _b["mean"], color=R.CATEGORICAL[_i], lw=1.6, label=_s)
    _ax.axvline(5.5, color=R.INK_2, lw=0.6, ls=":")
    R.style(_ax, ylabel="Catch per year (t)", xlabel="Year")
    _ax.set_title(f"{catch_group.selected_key}, k = {_k:g}", fontsize=10, color=R.INK, loc="left")
    _ax.legend(frameon=False, fontsize=8)
    fig_annual.tight_layout()
    if save.value:
        R.save(fig_annual, f"02_annual_catch_k{_k:g}_{catch_group.value}", FIGURES)
    fig_annual
    return


@app.cell
def _(FIGURES, R, k_pick, np, plt, save, yr):
    _k = float(k_pick.value)
    _d = yr[(yr.k == _k) & yr.group.isin(R.FISH)]
    _order = sorted(yr.strategy.unique(), key=lambda s: (s != "constant", s))
    fig_bio, _axes = plt.subplots(1, len(R.FISH), figsize=(13, 2.6), sharex=True)
    for _ax, _g in zip(np.atleast_1d(_axes), R.FISH):
        for _i, _s in enumerate(_order):
            _b = _d[(_d.group == _g) & (_d.strategy == _s)].groupby("year")["biomass"].mean()
            if not _b.empty:
                _ax.plot(_b.index, _b.values / 1e3, color=R.CATEGORICAL[_i], lw=1.4, label=_s)
        _ax.set_title(R.LABELS[_g], fontsize=9)
        R.style(_ax, xlabel="Year")
    np.atleast_1d(_axes)[0].set_ylabel("Biomass (kt)", fontsize=9)
    np.atleast_1d(_axes)[0].legend(frameon=False, fontsize=7)
    fig_bio.tight_layout()
    if save.value:
        R.save(fig_bio, f"02_biomass_k{_k:g}", FIGURES)
    fig_bio
    return


@app.cell
def _(cumulative, mo):
    _best = cumulative.loc[cumulative.groupby("strategy")["mean"].idxmax()]
    mo.vstack([mo.md("**Best k per strategy (mean cumulative catch, t)**"),
               mo.ui.table(_best.round(1), selection=None)])
    return


if __name__ == "__main__":
    app.run()
