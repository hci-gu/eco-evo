import argparse
import os
import sys
import time
import signal
import re
import matplotlib.pyplot as plt
from lib.visualize import shutdown_pygame
from lib.runners.petting_zoo import PettingZooRunner
from lib.runners.pbm import PBMRunner
from lib.config.settings import load_settings, Settings

SUPPORTED_MODEL_SUFFIXES = (".npy.npz", ".npz")

def _is_supported_model_file(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_MODEL_SUFFIXES)

def _strip_model_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in SUPPORTED_MODEL_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name

def _extract_fitness(name: str) -> float:
    base = _strip_model_suffix(name)
    # Handles patterns like:
    #  12_$sprat__a0_14568.62.npy.npz
    #  12_sprat__a0_14568.62.npy.npz
    match = re.match(r"^\d+_\$?.+_([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$", base)
    if match:
        return float(match.group(1))
    # Fallback: take last parseable numeric token.
    for token in reversed(re.split(r"[^0-9eE\+\-\.]+", base)):
        if not token:
            continue
        try:
            return float(token)
        except ValueError:
            continue
    return float("-inf")

def _infer_species_from_filename(filename: str, known_species: list[str]) -> str | None:
    base = _strip_model_suffix(filename)
    if base.endswith(".npy"):
        base = base[:-4]

    # Prefer explicit $<species>_<fitness> pattern used by current saver.
    match = re.match(r"^\d+_\$(.+?)_[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$", base)
    if match:
        species = match.group(1)
        if species in known_species:
            return species

    # Fallback: longest known species contained in filename.
    lowered = base.lower()
    for sp in sorted(known_species, key=len, reverse=True):
        if sp.lower() in lowered:
            return sp
    return None

def evaluate_model():
    folder = "results/2026-02-11_1/agents"
    start_time = time.time()
    settings = Settings()
    runner = PettingZooRunner(settings, render_mode="human")

    files = [f for f in os.listdir(folder) if _is_supported_model_file(f)]
    files.sort(key=_extract_fitness, reverse=True)
    if not files:
        raise FileNotFoundError(f"No model files found in '{folder}'.")

    required_species = list(runner.species_list)
    species_to_file = {}
    for f in files:
        species = _infer_species_from_filename(f, required_species)
        if species is None:
            continue
        if species not in species_to_file:
            species_to_file[species] = f

    # If only one model exists, use it for all acting species.
    if len(files) == 1:
        shared_path = os.path.join(folder, files[0])
        model_paths = [{"path": shared_path, "species": s} for s in required_species]
    else:
        missing = [s for s in required_species if s not in species_to_file]
        if missing:
            raise ValueError(
                "Missing models for acting species: "
                + ", ".join(missing)
                + f" in folder '{folder}'."
            )
        model_paths = [
            {"path": os.path.join(folder, species_to_file[s]), "species": s}
            for s in required_species
        ]

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
    if not config_folder:
        return []
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

    # evaluate_pbm_model()

    config_files = load_config_files(args.config_folder)
    if not config_files:
        settings = Settings()
        print(
            "No config folder provided; running a default beginner-friendly training run "
            f"with Settings(): fitness_method={settings.fitness_method}, "
            f"age_groups={settings.age_groups}, output={settings.folder}"
        )
        time.sleep(2)
        runner = PettingZooRunner(settings)
        runner.train()
    else:
        print("Running simulation with the following config files:" + str(config_files))
        time.sleep(2)

        for config_file in config_files:
            settings = load_settings(config_file)

            runner = PettingZooRunner(settings)
            runner.train()
