import os
import math
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import lib.constants as const
from lib.runners.petting_zoo_single import PettingZooRunnerSingle
import optuna

DATA_FILE          = "data.csv"
EVAL_YEAR_PAIRS    = []
SPECIES            = ["cod", "herring", "sprat"]
DAYS_PER_YEAR      = 365
STEPS_PER_YEAR     = math.ceil(DAYS_PER_YEAR / const.DAYS_PER_STEP)
STEPS_TOTAL        = STEPS_PER_YEAR

def set_starting_biomass(row: pd.Series) -> None:
    """Overwrite initial biomass for one evaluation start year."""
    const.FIXED_BIOMASS = True
    for sp in SPECIES:
        const.SPECIES_MAP[sp]["original_starting_biomass"] = row[sp]

    const.MIN_PERCENT_ALIVE = 0.1
    const.MAX_PERCENT_ALIVE = 3
    const.MAX_STEPS         = STEPS_TOTAL + 1      # N-year run

def set_fishing_pressure(row: pd.Series) -> None:
    """Apply year-specific fishing mortality for the whole horizon."""
    for sp in SPECIES:
        const.update_fishing_for_species(sp, row[f"fishing_{sp}"])

# ----------------  Error functions --------------------------------------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def noop():
    pass

def predict_years(eval_function = noop) -> float:
    df = pd.read_csv(DATA_FILE).sort_values("year").reset_index(drop=True)

    # build list of consecutive year pairs from the CSV
    years = df["year"].astype(int).tolist()
    EVAL_YEAR_PAIRS = [(years[i], years[i + 1]) for i in range(len(years) - 1)]

    pred_records = []

    for (y1, y2) in EVAL_YEAR_PAIRS:
        starting_point  = df.loc[df.year == y1].squeeze()
        target          = df.loc[df.year == y2].squeeze()

        # --- 3.1  Configure constants for this starting year ----------------
        set_starting_biomass(starting_point)
        set_fishing_pressure(starting_point)

        # --- 3.2  Run the model for exactly N simulated years --------------
        final_biomass, steps_seen = {}, 0

        def _callback(world, fitness, terminated=False):
            nonlocal steps_seen, final_biomass
            print(f" Step {steps_seen+1}/{STEPS_TOTAL} - Fitness: {fitness:.2f}", end='\r')
            steps_seen += 1
            if steps_seen >= STEPS_TOTAL or terminated:
                for sp in SPECIES:
                    offset = const.SPECIES_MAP[sp]["biomass_offset"]
                    final_biomass[sp] = world[..., offset].sum()
                return False   # stop the simulation
            return True        # continue

        print("before eval")
        eval_function(_callback)
        print("after eval")

        # --- 3.3  Store predictions & errors -------------------------------
        record = {
            "year": y1,
            **{f"{sp}_pred": final_biomass[sp] for sp in SPECIES},
            **{f"{sp}_true": target[sp]    for sp in SPECIES},
        }
        for sp in SPECIES:
            record[f"{sp}_abs_err"] = abs(
                record[f"{sp}_pred"] - record[f"{sp}_true"]
            )
            record[f"{sp}_pct_err"] = (
                (record[f"{sp}_abs_err"] / record[f"{sp}_true"]) * 100.0
                if record[f"{sp}_true"] != 0 else np.nan
            )
        pred_records.append(record)
        print(f"[{y1} -> {y2}] finished")

    # ----------------------------------------------------------------------
    # 4. Results: DataFrame & error statistics
    # ----------------------------------------------------------------------
    results = pd.DataFrame(pred_records)
    csv_name = f"year_ahead_predictions.csv"
    results.to_csv(csv_name, index=False)

    print("\nPer-year absolute % error")
    print(
        results[["year"] + [f"{sp}_pct_err" for sp in SPECIES]]
        .to_string(index=False)
    )

    avg_mape = {
        sp: mape(results[f"{sp}_true"], results[f"{sp}_pred"])
        for sp in SPECIES
    }
    avg_rmse = {
        sp: math.sqrt(
            ((results[f"{sp}_pred"] - results[f"{sp}_true"]) ** 2).mean()
        )
        for sp in SPECIES
    }

    for sp in SPECIES:
        print(
            f"{sp.capitalize():<8s}  MAPE: {avg_mape[sp]:6.2f}%   "
            f"RMSE: {avg_rmse[sp]:,.0f}"
        )
    overall_mape = np.mean(list(avg_mape.values()))
    print(f"\nAverage of MAPE across species: {overall_mape:.2f}%")

    return overall_mape