#!/usr/bin/env python
# baseline_prev_year.py
"""
Baseline: “carry-forward” predictor

For each start year *y*, forecast biomass in year *y + N* as exactly the
observed biomass in year *y*.

Configuration
-------------
YEARS_AHEAD   Horizon *N* (in years) for the naïve baseline.
              Edit the constant near the top of the file.

Outputs
-------
* CSV     : "{N}_year_ahead_predictions_baseline.csv"
* Console : per-year % errors + overall MAPE / RMSE (same metrics as RL run)
"""
# ---------------------------------------------------------------------------
# 0. USER-EDITABLE PARAMETER
# ---------------------------------------------------------------------------
YEARS_AHEAD = 5          # ← set to any positive integer

# ---------------------------------------------------------------------------
# 1. Imports & constants
# ---------------------------------------------------------------------------
import math
import numpy as np
import pandas as pd
from pathlib import Path

SPECIES     = ["cod", "herring", "sprat"]
DATA_FILE   = "data.csv"
OUT_CSV     = f"{YEARS_AHEAD}_year_ahead_predictions_baseline.csv"

# ---------------------------------------------------------------------------
# 2. Helper
# ---------------------------------------------------------------------------
def mape(y_true, y_pred):
    """Mean-Absolute-Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---------- 3.1  Load data ------------------------------------------------
    df = (
        pd.read_csv(DATA_FILE)
          .sort_values("year")
          .reset_index(drop=True)
          .set_index("year")
    )

    # Determine all start-years y such that y+N exists in the data
    eval_years = range(df.index.min(), df.index.max() - YEARS_AHEAD)

    # ---------- 3.2  Baseline predictions ------------------------------------
    records = []
    for y in eval_years:
        target_year = y + YEARS_AHEAD
        row_y       = df.loc[y]
        row_target  = df.loc[target_year]

        rec = {"year": target_year}
        for sp in SPECIES:
            pred_val = row_y[sp]          # naïve carry-forward
            true_val = row_target[sp]
            rec[f"{sp}_pred"]    = pred_val
            rec[f"{sp}_true"]    = true_val
            rec[f"{sp}_abs_err"] = abs(pred_val - true_val)
            rec[f"{sp}_pct_err"] = (
                abs(pred_val - true_val) / true_val * 100
                if true_val != 0 else np.nan
            )
        records.append(rec)

    res = pd.DataFrame(records).reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)

    # ---------- 3.3  Metrics & pretty print ----------------------------------
    print(f"\nPer-year absolute % error (baseline, {YEARS_AHEAD}-year ahead)")
    print(
        res[["year"] + [f"{sp}_pct_err" for sp in SPECIES]]
        .to_string(index=False)
    )

    avg_mape = {sp: mape(res[f"{sp}_true"], res[f"{sp}_pred"]) for sp in SPECIES}
    avg_rmse = {
        sp: math.sqrt(((res[f"{sp}_pred"] - res[f"{sp}_true"]) ** 2).mean())
        for sp in SPECIES
    }

    print(f"\nAverage error (horizon = {YEARS_AHEAD} year"
          f"{'s' if YEARS_AHEAD > 1 else ''})")
    for sp in SPECIES:
        print(f"{sp.capitalize():<8}  MAPE: {avg_mape[sp]:6.2f}%   "
              f"RMSE: {avg_rmse[sp]:,.0f}")

    overall_mape = np.mean(list(avg_mape.values()))
    print(f"\nAverage of MAPE across species: {overall_mape:.2f}%")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
