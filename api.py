from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dataclasses import fields
import base64
import io
import os
import uuid
import json
import numpy as np
from pathlib import Path

from lib.config.settings import Settings
import lib.config.const as sim_const
from lib.model import Model, MODEL_OFFSETS
from lib.runners.petting_zoo import PettingZooRunner

app = Flask(__name__)
CORS(app)

simulations = {}

PROJECT_DIR = Path(__file__).resolve().parent
AGENTS_DIR = PROJECT_DIR / "agents"
AGENTS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODEL_SUFFIXES = (".npy.npz", ".npz")

def get_model_path():
    folder = PROJECT_DIR / "results" / "single_agent_single_out_random_plankton_behavscore_6" / "agents"
    if not folder.exists():
        raise FileNotFoundError(f"Default model folder not found: {folder}")

    files = [p.name for p in folder.iterdir() if p.is_file() and p.name.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".npy")[0]), reverse=True)

    file = files[0]
    model_path = folder / file
    return str(model_path)

def _is_supported_model_file(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_MODEL_SUFFIXES)

def _infer_species_from_filename(filename: str, known_species: list[str]) -> str | None:
    base = filename.lower()
    for suffix in SUPPORTED_MODEL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    tokens = []
    cur = []
    for ch in base:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    for sp in known_species:
        if sp.lower() in tokens or base == sp.lower():
            return sp
    return None

def _load_model_npz(model_path: Path) -> Model:
    if not _is_supported_model_file(model_path.name):
        raise ValueError(f"Unsupported model file: {model_path.name}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with np.load(str(model_path)) as chromosome_npz:
        chromosome = {k: chromosome_npz[k] for k in chromosome_npz.files}
    return Model(chromosome=chromosome)

def _resolve_agent_set_dir(agent_set: str) -> Path:
    base = AGENTS_DIR.resolve()
    candidate = (AGENTS_DIR / agent_set).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("Invalid agent_set path.")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Agent set not found: {agent_set}")
    return candidate

def _summarize_agent_set(agent_set_dir: Path) -> dict:
    model_files = sorted(
        p.name for p in agent_set_dir.iterdir() if p.is_file() and _is_supported_model_file(p.name)
    )
    if not model_files:
        return {
            "name": agent_set_dir.name,
            "kind": "empty",
            "files": [],
            "species": [],
        }

    if len(model_files) == 1:
        return {
            "name": agent_set_dir.name,
            "kind": "single",
            "files": model_files,
            "species": list(sim_const.ACTING_SPECIES),
        }

    species_to_file: dict[str, str] = {}
    for fname in model_files:
        species = _infer_species_from_filename(fname, list(sim_const.SPECIES))
        if species is None:
            continue
        if species in species_to_file:
            raise ValueError(f"Duplicate model for species '{species}' in agent set '{agent_set_dir.name}'.")
        species_to_file[species] = fname

    return {
        "name": agent_set_dir.name,
        "kind": "multi",
        "files": model_files,
        "species": sorted(species_to_file.keys()),
    }

def _load_candidates_from_agent_set(agent_set: str) -> dict:
    agent_set_dir = _resolve_agent_set_dir(agent_set)
    model_paths = sorted(p for p in agent_set_dir.iterdir() if p.is_file() and _is_supported_model_file(p.name))
    if not model_paths:
        raise ValueError(f"Agent set '{agent_set}' contains no supported model files.")

    if len(model_paths) == 1:
        shared = _load_model_npz(model_paths[0])
        return {species: shared for species in sim_const.ACTING_SPECIES}

    # Multi-agent: require one model per acting species, inferred from filename.
    species_to_model: dict[str, Model] = {}
    for p in model_paths:
        species = _infer_species_from_filename(p.name, list(sim_const.ACTING_SPECIES))
        if species is None:
            continue
        if species in species_to_model:
            raise ValueError(f"Duplicate model for species '{species}' in agent set '{agent_set}'.")
        species_to_model[species] = _load_model_npz(p)

    missing = [sp for sp in sim_const.ACTING_SPECIES if sp not in species_to_model]
    if missing:
        raise ValueError(
            f"Agent set '{agent_set}' is missing models for acting species: {', '.join(missing)}."
        )
    return species_to_model

def _snake_to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])

def _coerce(value, target_type):
    if value is None:
        return None
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value

def _settings_from_options(options: dict) -> Settings:
    base = Settings()
    kwargs = {}
    for f in fields(Settings):
        if f.name == "folder":
            continue
        snake = f.name
        camel = _snake_to_camel(snake)
        if snake in options:
            kwargs[snake] = _coerce(options[snake], type(getattr(base, snake)))
        elif camel in options:
            kwargs[snake] = _coerce(options[camel], type(getattr(base, snake)))
    return Settings(**kwargs)

@app.route('/simulate/upload', methods=['POST'])
def upload_files():
    print("Request files:", request.files)
    # Check if both 'texture' and 'depth' files are present in the request
    if 'texture' not in request.files or 'depth' not in request.files:
        return jsonify({"error": "Missing required files"}), 400

    texture_file = request.files['texture']
    depth_file = request.files['depth']

    # Parse options passed in the form data
    options = request.form.get('options')
    # print("Options received:", options)
    options = json.loads(options)

    # Validate filenames
    if texture_file.filename == '' or depth_file.filename == '':
        return jsonify({"error": "One or more files have no filename"}), 400

    # Create a unique folder to store the uploaded files
    simulation_id = str(uuid.uuid4())
    folder_path = os.path.join('maps', simulation_id)
    os.makedirs(folder_path, exist_ok=True)

    # Save the uploaded files
    texture_path = os.path.join(folder_path, 'map.png')
    depth_path = os.path.join(folder_path, 'depth.png')

    try:
        texture_file.save(texture_path)
        depth_file.save(depth_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save files: {str(e)}"}), 500

    simulations[simulation_id] = {
        'options': options,
        'folder_path': folder_path,
    }

    return jsonify({"id": simulation_id}), 200

@app.route('/simulate/agents', methods=['GET'])
def get_agents():
    if not AGENTS_DIR.exists():
        return jsonify([])

    agent_sets = []
    for p in sorted(AGENTS_DIR.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        try:
            agent_sets.append(_summarize_agent_set(p))
        except Exception as e:
            agent_sets.append(
                {
                    "name": p.name,
                    "kind": "error",
                    "error": str(e),
                    "files": [],
                    "species": [],
                }
            )
    return jsonify(agent_sets)

@app.route('/simulate/<simulation_id>', methods=['GET'])
def run_simulation(simulation_id):
    print("Starting simulation:", simulation_id)
    if simulation_id not in simulations:
        return jsonify({"error": "Simulation ID not found"}), 404

    options = dict(simulations[simulation_id].get("options") or {})

    # Query arg overrides (handy for testing without changing the uploader).
    if request.args.get("worldSize") is not None:
        options["world_size"] = request.args.get("worldSize")
    elif request.args.get("world_size") is not None:
        options["world_size"] = request.args.get("world_size")
    if request.args.get("maxSteps") is not None:
        options["max_steps"] = request.args.get("maxSteps")
    elif request.args.get("max_steps") is not None:
        options["max_steps"] = request.args.get("max_steps")

    # Defaults aligned with the storage discussion.
    if "world_size" not in options and "worldSize" not in options:
        options["world_size"] = 50
    if "max_steps" not in options and "maxSteps" not in options:
        options["max_steps"] = 3000

    settings = _settings_from_options(options)
    sample_every = int(
        request.args.get("sampleEvery")
        or request.args.get("sample_every")
        or options.get("sampleEvery")
        or options.get("sample_every")
        or 10
    )
    include_final = _coerce(
        request.args.get("includeFinal") or request.args.get("include_final") or True,
        bool,
    )
    output_format = (request.args.get("format") or "base64").strip().lower()

    map_folder = simulations[simulation_id]["folder_path"]
    print("Using map folder:", map_folder)
    runner = PettingZooRunner(settings=settings, render_mode="none", map_folder=map_folder, build_population=False)
    print("runner created")

    agent_set = (
        request.args.get("agentSet")
        or request.args.get("agent_set")
        or request.args.get("agents")
        or request.args.get("agent")
    )
    print("agent_set:", agent_set)

    try:
        if agent_set:
            candidates = _load_candidates_from_agent_set(agent_set)
        else:
            model_path = request.args.get("modelPath") or request.args.get("model_path") or get_model_path()
            model_path = str(model_path).strip()
            p = Path(model_path)
            if not p.is_absolute():
                p = (PROJECT_DIR / p).resolve()
            shared_model = _load_model_npz(p)
            candidates = {species: shared_model for species in sim_const.ACTING_SPECIES}
    except FileNotFoundError as e:
        print("Error loading models:", e)
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        print("Error loading models:", e)
        return jsonify({"error": str(e)}), 400

    species = list(sim_const.SPECIES)
    biomass_offsets = [MODEL_OFFSETS[s]["biomass"] for s in species]

    sample_steps = []
    samples = []
    sim_step = -1

    def callback(world, _fitness, done):
        nonlocal sim_step
        if not done:
            sim_step += 1
            if sim_step % sample_every != 0:
                return
        elif not include_final:
            return

        if done and len(sample_steps) > 0 and sample_steps[-1] == sim_step:
            return

        grid = world[1:-1, 1:-1, :]
        snapshot = np.stack([grid[:, :, off] for off in biomass_offsets], axis=-1).astype(np.float32, copy=False)
        samples.append(snapshot)
        sample_steps.append(sim_step)

    print("WE GOT HERE")
    fitness, episode_length, reason = runner.run(
        candidates=candidates,
        species_being_evaluated="cod",
        seed=None,
        is_evaluation=True,
        callback=callback,
    )

    if not samples:
        return jsonify({"error": "Simulation produced no samples"}), 500

    biomass = np.stack(samples, axis=0)  # (N, H, W, S)
    steps_arr = np.asarray(sample_steps, dtype=np.int32)

    if output_format == "npz":
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            biomass=biomass,
            steps=steps_arr,
            species=np.asarray(species),
            world_size=np.asarray([settings.world_size], dtype=np.int32),
            sample_every=np.asarray([sample_every], dtype=np.int32),
        )
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="application/octet-stream")

    payload = {
        "simulation_id": simulation_id,
        "world_size": settings.world_size,
        "species": species,
        "sample_every": sample_every,
        "include_final": bool(include_final),
        "dtype": str(biomass.dtype),
        "shape": list(biomass.shape),
        "steps": steps_arr.tolist(),
        "fitness": float(fitness),
        "episode_length": int(episode_length),
        "end_reason": reason,
        "biomass_b64": base64.b64encode(biomass.tobytes(order="C")).decode("ascii"),
    }
    return jsonify(payload)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
