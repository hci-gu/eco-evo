import argparse
import os
import time
from lib.runners.petting_zoo import PettingZooRunner
from lib.config.settings import load_settings

def evaluate_model():
    folder = "results/back_to_multi_2/agents"
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
    runner = PettingZooRunner(render_mode="human")
    runner.evaluate(model_paths)
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

    # profiler = cProfile.Profile()
    # profiler.enable()

    # evaluate_single_model()
    # print(const.FISHING_AMOUNTS)
    # time.sleep(10000)
    # total_elapsed_time = 0
    # for i in range(5):
    #     elapsed_time = evaluate_model()
    #     total_elapsed_time += elapsed_time
    #     print(f"Elapsed time for run {i + 1}: {elapsed_time:.2f} seconds")
    # average_elapsed_time = total_elapsed_time / 5
    # print(f"Average elapsed time: {average_elapsed_time:.2f} seconds")

    # Load config files
    config_files = load_config_files(args.config_folder)

    print("Running simulation with the following config files:" + str(config_files))

    for config_file in config_files:
        settings = load_settings(config_file)

        runner = PettingZooRunner(settings)
        runner.train()



        


    

