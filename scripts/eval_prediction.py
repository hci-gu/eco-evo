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
import sys

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

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
from lib.runners.petting_zoo import PettingZooRunner
from lib.evaluate import predict_years
import optuna

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Utility helpers
# --------------------------------------------------------------------------
DATA_FILE          = "data.csv"
AGENT_FOLDER       = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
EVAL_YEARS         = range(1993, 2006)          # feed 1993-2009 → predict 2010-2020
SPECIES            = ["cod", "herring", "sprat"]
YEARS_AHEAD        = 3                # predict y+1 from y
DAYS_PER_YEAR      = 365
STEPS_PER_YEAR     = math.ceil(DAYS_PER_YEAR / const.DAYS_PER_STEP)
STEPS_TOTAL        = STEPS_PER_YEAR * YEARS_AHEAD

def get_runner_single():
    files = [
        f for f in os.listdir(AGENT_FOLDER) if f.endswith(".npy.npz")
    ]
    files.sort(key=lambda f: float(f.split("_")[1].split(".")[0]), reverse=True)
    model_path = os.path.join(AGENT_FOLDER, files[0])
    runner = PettingZooRunnerSingle()
    return runner, model_path

def get_runner():
    runner = PettingZooRunner()

    folder = "results/2025-09-08_1/agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[2].split(".npy")[0]), reverse=True)
    species = {}
    for f in files:
        s = f.split("_")[1].split(".")[0]
        s = s[1:] if s[0] == "$" else s
        if s == "spat":
            s = "sprat"
        if s not in species:
            species[s] = f

    model_paths = []
    for s, f in species.items():
        model_paths.append({ 'path': os.path.join(folder, f), 'species': s })

    return runner, model_paths

def noop(a, b):
    pass

if __name__ == "__main__":
    runner, model_paths = get_runner()

    for i in range(0, 25):
        def objective(trial):
            energy_cost_sprat     = trial.suggest_float("energy_cost_sprat",   0.0, 10.0)
            energy_reward_sprat   = trial.suggest_float("energy_reward_sprat", 0.0, 1000.0)
            energy_cost_cod       = trial.suggest_float("energy_cost_cod",     0.0, 10.0)
            energy_reward_cod     = trial.suggest_float("energy_reward_cod",   0.0, 1000.0)
            energy_cost_herring   = trial.suggest_float("energy_cost_herring", 0.0, 10.0)
            energy_reward_herring = trial.suggest_float("energy_reward_herring", 0.0, 1000.0)

            for sp in SPECIES:
                const.update_energy_params(
                    sp,
                    energy_cost_sprat   if sp == "sprat"   else
                    energy_cost_cod     if sp == "cod"     else
                    energy_cost_herring,
                    energy_reward_sprat if sp == "sprat"   else
                    energy_reward_cod   if sp == "cod"     else
                    energy_reward_herring,
                )

            def eval_function(callback):
                print("eval function was called", callback)
                runner.evaluate(model_paths, callback)

            err = predict_years(eval_function)
            return err

        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=50)

        out = study.best_params | {"best_value": study.best_value}

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"best_params_{i}.json"), "w") as f:
            json.dump(out, f, indent=4)