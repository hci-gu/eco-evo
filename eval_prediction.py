"""
evaluate_one_year_ahead.py
---------------------------------
One-year-ahead biomass‐prediction evaluation
( feed y-year data → predict y+1 ) for
  • cod
  • herring
  • sprat

Assumptions
-----------
* Historical observations and fishing mortalities live in `data.csv`
  (columns: year, cod, herring, sprat,
            fishing_cod, fishing_herring, fishing_sprat).

* A *single-species* agent saved as the most recent
  `.npy.npz` file inside
  results/single_agent_single_out_random_plankton_behavscore_6/agents

* PettingZooRunnerSingle exposes `.evaluate(model_path, callback_fn)`
  in which the callback receives `(world, fitness)` each step.

* `const.DAYS_PER_STEP` gives the simulated number of real days/step.

Outputs
-------
* PNG plot:  three panels showing actual vs. predicted biomass (2007-2020)
* CSV:       yearly predictions & errors (`one_year_ahead_predictions.csv`)
* Console:   per-year error table + 2007-2020 average MAPE & RMSE
"""

import os
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import lib.constants as const
from lib.runners.petting_zoo_single import PettingZooRunnerSingle
import optuna


# --------------------------------------------------------------------------
# 1. Utility helpers
# --------------------------------------------------------------------------
DATA_FILE          = "data.csv"
AGENT_FOLDER       = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
EVAL_YEARS         = range(1993, 2003)          # feed 2006-2019 → predict 2007-2020
SPECIES            = ["cod", "herring", "sprat"]
DAYS_PER_YEAR      = 365
STEPS_PER_YEAR     = math.ceil(DAYS_PER_YEAR / const.DAYS_PER_STEP)

def get_runner_single():
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".")[0]), reverse=True)
    print(files[0])
    path = os.path.join(folder, files[0])
    runner = PettingZooRunnerSingle()

    return runner, path

# ---------------- Constants update helpers ------------------------------------
def set_starting_biomass(row: pd.Series) -> None:
    """Overwrite initial biomass for a single evaluation year."""
    const.FIXED_BIOMASS = True
    for sp in SPECIES:
        const.SPECIES_MAP[sp]["original_starting_biomass"] = row[sp]

    # allow any population size, limit sim length to one year
    const.MIN_PERCENT_ALIVE = 0
    const.MAX_PERCENT_ALIVE = 9_999
    const.MAX_STEPS         = STEPS_PER_YEAR + 1

def set_fishing_pressure(row: pd.Series) -> None:
    """Apply year-specific fishing mortality."""
    for sp in SPECIES:
        const.update_fishing_for_species(sp, row[f"fishing_{sp}"])

# ----------------  Error functions --------------------------------------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

# --------------------------------------------------------------------------
# 2. Evaluation loop
# --------------------------------------------------------------------------
def predict_years() -> None:
    df = pd.read_csv(DATA_FILE).sort_values("year").reset_index(drop=True)

    runner, model_path = get_runner_single()

    # containers
    pred_records = []

    for y in EVAL_YEARS:
        row_y       = df.loc[df.year == y].squeeze()
        row_y_plus1 = df.loc[df.year == y + 1].squeeze()

        # --- 2.1  Configure constants for this starting year -----------------
        set_starting_biomass(row_y)          # biomass at year y
        set_fishing_pressure(row_y)          # fish. mortality during (y → y+1)

        # --- 2.2  Run the model for exactly one simulated year --------------
        final_biomass = {}

        steps_seen    = 0
        def _callback(world, fitness):
            nonlocal steps_seen, final_biomass
            steps_seen += 1
            if steps_seen >= STEPS_PER_YEAR:
                # collect biomass at the end of the simulated year
                for sp in SPECIES:
                    offset = const.SPECIES_MAP[sp]["biomass_offset"]
                    final_biomass[sp] = world[..., offset].sum()
                return False   # stop the simulation
            return True        # continue

        runner.evaluate(model_path, _callback)

        # --- 2.3  Store predictions & errors --------------------------------
        record = {
            "year": y + 1,   # forecast year
            **{f"{sp}_pred": final_biomass[sp]        for sp in SPECIES},
            **{f"{sp}_true": row_y_plus1[sp]          for sp in SPECIES},
        }
        for sp in SPECIES:
            record[f"{sp}_abs_err"] = abs(record[f"{sp}_pred"] - record[f"{sp}_true"])
            record[f"{sp}_pct_err"] = (
                (record[f"{sp}_abs_err"] / record[f"{sp}_true"]) * 100.0
                if record[f"{sp}_true"] != 0 else np.nan
            )
        pred_records.append(record)
        print(f"[{y}->{y+1}] finished")

    # ----------------------------------------------------------------------
    # 3. Results: DataFrame, error statistics
    # ----------------------------------------------------------------------
    results = pd.DataFrame(pred_records)
    results.to_csv("one_year_ahead_predictions.csv", index=False)

    print("\nPer-year absolute % error")
    print(results[["year"] + [f"{sp}_pct_err" for sp in SPECIES]].to_string(index=False))

    avg_mape = {sp: mape(results[f"{sp}_true"], results[f"{sp}_pred"])
                for sp in SPECIES}
    avg_rmse = {sp: math.sqrt(
                    ((results[f"{sp}_pred"] - results[f"{sp}_true"]) ** 2).mean())
                for sp in SPECIES}

    print("\nAverage error 2007-2020")
    for sp in SPECIES:
        print(f"{sp.capitalize():<8s}  MAPE: {avg_mape[sp]:6.2f}%   "
              f"RMSE: {avg_rmse[sp]:,.0f}")

    average_of_mape = np.mean(list(avg_mape.values()))
    return average_of_mape
    fig, axes = plt.subplots(len(SPECIES), 1, figsize=(10, 9), sharex=True)

    species_colors = {              # use same palette as simulation UI
        sp: const.SPECIES_MAP[sp]["visualization"]["color_ones"]
        for sp in SPECIES
    }

    for idx, sp in enumerate(SPECIES):
        ax = axes[idx]
        ax.plot(results["year"], results[f"{sp}_true"],
                label="Actual",  marker="o",
                color=species_colors[sp])
        ax.plot(results["year"], results[f"{sp}_pred"],
                label="Predicted", marker="x", linestyle="--",
                color=species_colors[sp])

        ax.set_ylabel(f"{sp.capitalize()}\n(million t)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _:
                                                   f"{x*1e-6:.1f}"))
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Year")
    plt.suptitle("One-Year-Ahead Biomass Prediction Accuracy (2007–2020)")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig("one_year_ahead_biomass_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    def objective(trial):
        energy_cost = trial.suggest_float("energy_cost", 0.0, 10.0)
        energy_reward = trial.suggest_float("energy_reward", 0.0, 500.0)
        energy_reward_cod = trial.suggest_float("energy_reward_cod", 0.0, 1000.0)
        eat_amount_boost = trial.suggest_float("eat_amount_boost", 0.0, 10.0)

        const.update_energy_params(energy_cost, energy_reward, energy_reward_cod, eat_amount_boost)

        err = predict_years()
        return err
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print(study.best_params)

