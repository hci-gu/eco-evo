#!/usr/bin/env python
"""
Unified script runner for simulation and plotting scripts.

Usage:
    python run_script.py <script_name>
    python run_script.py --list

Available scripts:
    plot_biomass         - Plot biomass per functional group
    plot_fish            - Plot fish growth dynamics
    plot_real_data       - Interactive visualization of real biomass data
    gen_graph            - Generate fitness over generations graph
    biomass_stability    - Test biomass stability under perturbations
    fish_biomass_eval    - Evaluate fish biomass under fishing pressure
    fishing_eval         - Evaluate fishing pressure impact
    eval                 - Main evaluation script
    eval_heuristic       - Evaluate heuristic policy
    eval_multi_species   - Evaluate multi-species model
    eval_prediction      - Prediction evaluation with optimization
    eval_regular         - Regular evaluation run
    eval_unique          - Unique evaluation with averages
"""

import sys
import os
import argparse

# Add project root to path so lib imports work from scripts folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root directory so relative paths work
os.chdir(PROJECT_ROOT)

SCRIPTS = {
    "plot_biomass": "scripts/plot_biomass.py",
    "plot_biomass_all": "scripts/plot_biomass_all.py",
    "plot_fish": "scripts/plot_fish.py",
    "plot_real_data": "scripts/plot_real_data.py",
    "gen_graph": "scripts/gen_graph.py",
    "biomass_stability": "scripts/biomass_stability_test.py",
    "fish_biomass_eval": "scripts/fish_biomass_eval.py",
    "fishing_eval": "scripts/fishing_eval.py",
    "eval": "scripts/eval.py",
    "eval_heuristic": "scripts/eval_heuristic.py",
    "eval_multi_species": "scripts/eval_multi_species.py",
    "eval_prediction": "scripts/eval_prediction.py",
    "eval_regular": "scripts/eval_regular.py",
    "eval_unique": "scripts/eval_unique.py",
}


def list_scripts():
    print("Available scripts:")
    print("-" * 50)
    for name in sorted(SCRIPTS.keys()):
        print(f"  {name}")
    print("-" * 50)
    print(f"\nUsage: python run_script.py <script_name>")


def run_script(script_name: str, extra_args: list):
    if script_name not in SCRIPTS:
        print(f"Error: Unknown script '{script_name}'")
        print(f"Use --list to see available scripts")
        sys.exit(1)

    script_path = os.path.join(PROJECT_ROOT, SCRIPTS[script_name])
    
    if not os.path.exists(script_path):
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)

    print(f"Running: {script_name}")
    print(f"Script path: {script_path}")
    print("-" * 50)

    # Build sys.argv for the target script
    sys.argv = [script_path] + extra_args

    # Execute the script
    with open(script_path, 'r') as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, {'__name__': '__main__', '__file__': script_path})


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation and plotting scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "script_name",
        nargs="?",
        help="Name of the script to run"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available scripts"
    )

    args, extra_args = parser.parse_known_args()

    if args.list or args.script_name is None:
        list_scripts()
        return

    run_script(args.script_name, extra_args)


if __name__ == "__main__":
    main()
