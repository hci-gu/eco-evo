import argparse
import os
import sys
import time
import signal
import matplotlib.pyplot as plt
from lib.visualize import shutdown_pygame
from lib.runners.petting_zoo import PettingZooRunner
from lib.runners.pbm import PBMRunner
from lib.config.settings import load_settings, Settings

def evaluate_model():
    folder = "results/2025-11-06_pure_behavscore/agents"
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

    start_time = time.time()
    settings = Settings()
    runner = PettingZooRunner(settings, render_mode="human")
    try:
        runner.evaluate(model_paths)
    finally:
        # Ensure UI resources are closed even on Ctrl+C
        try:
            runner.env.close()
        except Exception:
            pass
        try:
            plt.ioff()
            plt.close('all')
        except Exception:
            pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def evaluate_pbm_model():
    folder = "results/2025-11-12_PBM/agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".npy")[0]), reverse=True)
    
    # Take only the highest fitness model
    best_model = files[0]
    model_path = os.path.join(folder, best_model)

    start_time = time.time()
    settings = Settings()
    runner = PBMRunner(settings)
    try:
        def callback(info, b, c):
            print(info)    

        runner.evaluate(model_path, callback=callback)
    finally:
        # Ensure UI resources are closed even on Ctrl+C
        try:
            runner.env.close()
        except Exception:
            pass
        try:
            plt.ioff()
            plt.close('all')
        except Exception:
            pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

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

    # Handle Ctrl+C cleanly: exit with code 0 to avoid macOS crash dialog
    # try:
    #     evaluate_model()
    #     print("Model evaluation completed.")
    # except KeyboardInterrupt:
    #     print("Interrupted by user. Shutting down cleanly…")
    #     try:
    #         plt.ioff()
    #         plt.close('all')
    #         shutdown_pygame()
    #     except Exception:
    #         pass
    #     sys.exit(0)

    # profiler = cProfile.Profile()
    # profiler.enable()

    # print(const.FISHING_AMOUNTS)
    # time.sleep(10000)
    # total_elapsed_time = 0
    # for i in range(5):
    #     elapsed_time = evaluate_model()
    #     total_elapsed_time += elapsed_time
    #     print(f"Elapsed time for run {i + 1}: {elapsed_time:.2f} seconds")
    # average_elapsed_time = total_elapsed_time / 5
    # print(f"Average elapsed time: {average_elapsed_time:.2f} seconds")

    evaluate_pbm_model()

    # Load config files
    # config_files = load_config_files(args.config_folder)

    # print("Running simulation with the following config files:" + str(config_files))
    # time.sleep(2)

    # for config_file in config_files:
    #     settings = load_settings(config_file)

    #     runner = PBMRunner(settings)
    #     runner.train()
