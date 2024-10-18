import argparse
import os
from lib.world import create_world, world_is_alive, Species, Action
from lib.visualize import init_pygame, draw_world, plot_biomass
from lib.model import Model
from lib.constants import override_from_file
from lib.runner import Runner
import time
import random
import math
import torch

def load_config_files(config_folder):
    """Load all config files from the given folder."""
    config_files = []
    for file_name in os.listdir(config_folder):
        if file_name.endswith(".txt"):
            config_files.append(os.path.join(config_folder, file_name))
    return config_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ecosystem simulation with neural networks.")
    parser.add_argument("--config_folder", type=str, required=True,
                        help="Path to the folder containing config files.")
    
    # Parse arguments
    args = parser.parse_args()

    # Load config files
    config_files = load_config_files(args.config_folder)

    print("Running simulation with the following config files:" + str(config_files))

    with torch.no_grad():
        for config_file in config_files:
            override_from_file(config_file)
            time.sleep(1)
            runner = Runner()
            runner.run()

    


