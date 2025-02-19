import argparse
import threading
import os
from lib.data_manager import data_loop, update_generations_data, process_data
from lib.constants import override_from_file
from lib.runners.single_agent_gym import SingleAgentGymRunner
from lib.runners.petting_zoo_reinforcement import MultiAgentRunner
from lib.runners.petting_zoo import PettingZooRunner
import lib.constants as const
from lib.model import Model
import lib.environments.gym
from lib.runner import Runner
import torch
import random
import gymnasium as gym


def load_config_files(config_folder):
    """Load all config files from the given folder."""
    config_files = []
    for file_name in os.listdir(config_folder):
        if file_name.endswith(".txt"):
            config_files.append(os.path.join(config_folder, file_name))
    config_files.sort()
    return config_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ecosystem simulation with neural networks.")
    parser.add_argument("--config_folder", type=str, required=False,
                        help="Path to the folder containing config files.")
    parser.add_argument("--agent_file", type=str, required=False,
                        help="Path to the .pt file containing the agent model.")
    
    # Parse arguments
    args = parser.parse_args()

    # Load config files
    config_files = load_config_files(args.config_folder)

    print("Running simulation with the following config files:" + str(config_files))

    with torch.no_grad():
        for config_file in config_files:
            override_from_file(config_file)
            runner = MultiAgentRunner()
            runner.train()

            # runner = SingleAgentGymRunner()

            # while runner.current_generation < const.GENERATIONS_PER_RUN:
            #     runner.run_generation()
            #     print(f"Generation {runner.current_generation} finished")


    

