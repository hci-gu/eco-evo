from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import torch
import os
import zipfile
import uuid
from lib.world.update_world import perform_action, world_is_alive
from lib.world.map import read_map_from_file
from lib.runner import Runner
import lib.constants as const

app = Flask(__name__)
CORS(app)

simulations = {}

@app.route('/simulate', methods=['POST'])
def simulate():
    # Check if both 'texture' and 'depth' files are present in the request
    if 'texture' not in request.files or 'depth' not in request.files:
        return jsonify({"error": "Missing required files"}), 400

    texture_file = request.files['texture']
    depth_file = request.files['depth']

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
        'steps': [],
    }

    runner = Runner(map_folder=folder_path)
    def callback(world, world_data, fitness):
        step = {}
        for species, properties in const.SPECIES_MAP.items():
            biomass_offset = properties["biomass_offset"]
            step[f'{species}'] = world[:, :, biomass_offset].sum().item()
        simulations[simulation_id]['steps'].append(step)
    
    runner.simulate(agent_file=const.EVAL_AGENT, visualize=callback)
    
    return jsonify({
        "id": simulation_id,
        "simulation": simulations[simulation_id]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
