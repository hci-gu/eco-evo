import os
import numpy as np
from lib.model import Model
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.behaviour import run_all_scenarios

def get_model_path():
    folder = "results/pure_behav_1/agents"
    files = [f for f in os.listdir(folder) if f.endswith(".npy.npz")]
    file = files[0]
    print('model', file)
    print('fitness', float(file.split("_")[1].split(".npy")[0]))
    files.sort(key=lambda f: float(f.split("_")[1].split(".npy")[0]), reverse=True)
    print('model', files[0])
    return os.path.join(folder, files[0])

def get_multi_model_paths():
    folder = "results/2025-10-23_4/agents"
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

    return model_paths

# --- Main ---
if __name__ == "__main__":
    # Uncomment the following lines to evaluate single model
    # model_path = get_model_path()
    # print("model_path", model_path)
    # candidate = Model(
    #     chromosome=np.load(model_path)
    # )
    # run_all_scenarios(candidate, "all", True)

    # Uncomment the following lines to evaluate multiple models
    model_paths = get_multi_model_paths()
    print("model_paths", model_paths)
    settings =  Settings()
    species_map = build_species_map(settings)
    for model in model_paths:
        candidate = Model(
            chromosome=np.load(model['path'])
        )
        print("model_path", model['path'])
        run_all_scenarios(settings, species_map, candidate, model['species'], False)
