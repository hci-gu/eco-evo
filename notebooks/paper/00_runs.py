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

    ROOT = Path(mo.notebook_dir()).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from lib.scenarios import results as R
    from lib.scenarios.spec import load_scenario

    RESULTS = Path(os.environ.get("PAPER_RESULTS", ROOT / "results" / "paper"))
    return R, RESULTS, ROOT, load_scenario, mo, plt


@app.cell
def _(R, RESULTS, ROOT, load_scenario, mo):
    import json as _json
    _rows = []
    for _path in sorted((ROOT / "scenarios" / "paper").glob("*.yaml")):
        for _v in load_scenario(str(_path)):
            for _seed in _v.seeds:
                _meta = RESULTS / _v.scenario / _v.name / f"seed_{_seed}" / "meta.json"
                _done = _meta.exists()
                _m = _json.loads(_meta.read_text()) if _done else {}
                _rows.append({"scenario": _v.scenario, "variant": _v.name, "seed": _seed,
                              "done": _done, "years": _m.get("years"),
                              "minutes": (_m.get("seconds") or 0) / 60 if _done else None,
                              "grid": "x".join(map(str, _m.get("grid", []))) or None})
    import pandas as _pd
    status = _pd.DataFrame(_rows)
    _summary = status.groupby("scenario").agg(runs=("done", "size"), done=("done", "sum"),
                                              minutes=("minutes", "mean")).reset_index()
    mo.vstack([mo.md(f"# Paper runs\n\nResults root: `{RESULTS}`"),
               mo.ui.table(_summary.round(1), selection=None),
               mo.accordion({"All runs": mo.ui.table(status, selection=None)})])
    return (status,)


@app.cell
def _(R, RESULTS, mo, plt):
    _runs = R.find_runs(RESULTS)
    _runs = _runs[_runs.scenario.isin(["current_fishing", "no_fishing"])]
    mo.stop(_runs.empty, mo.md("No `current_fishing` / `no_fishing` runs yet."))
    _yr = R.yearly(_runs)
    _groups = R.FISH + R.MAMMALS
    fig_stab, _axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    _colors = R.group_colors(_groups)
    for _ax, _sc in zip(_axes, ["current_fishing", "no_fishing"]):
        _d = _yr[_yr.scenario == _sc]
        for _g in _groups:
            _m = _d[_d.group == _g].groupby("year")["biomass"].mean()
            if _m.empty:
                continue
            _ax.plot(_m.index, _m.values / _m.values[0], color=_colors[_g], lw=1.4,
                     label=R.LABELS[_g])
        _ax.axhline(1.0, color=R.INK_2, lw=0.6, ls=":")
        _ax.set_title(_sc, fontsize=10, loc="left")
        R.style(_ax, xlabel="Year")
    _axes[0].set_ylabel("Annual mean biomass / year 1", fontsize=9)
    _axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig_stab.tight_layout()
    mo.vstack([mo.md("**Long-horizon check**: the policies were trained on one-year "
                     "episodes; the groups should neither drift away nor collapse."),
               fig_stab])
    return


if __name__ == "__main__":
    app.run()
