"""
Scripts package for simulation and plotting.
All output files will be placed in results/plots/
"""

import os
import sys

# Add project root to path so lib imports work
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Output directory for all plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")

def get_output_path(filename: str) -> str:
    """Get the full path for an output file in the results/plots directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return os.path.join(PLOTS_DIR, filename)

def ensure_project_root():
    """Change to project root directory for relative paths."""
    os.chdir(PROJECT_ROOT)
