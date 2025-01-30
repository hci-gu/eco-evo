from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import os
import uuid
import json
from lib.world.update_world import perform_action, world_is_alive
from lib.world.map import read_map_from_file
from lib.runner import Runner
import lib.constants as const

app = Flask(__name__)
CORS(app)

simulations = {}

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
    options = json.loads(options)
    const.override_from_options(options)

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
        'folder_path': folder_path
    }

    return jsonify({"id": simulation_id}), 200

@app.route('/simulate/<simulation_id>', methods=['GET'])
def stream_simulation(simulation_id):
    print("Starting simulation:", simulation_id)
    if simulation_id not in simulations:
        return jsonify({"error": "Simulation ID not found"}), 404

    def generate_simulation():
        folder_path = simulations[simulation_id]['folder_path']
        runner = Runner(map_folder=folder_path)

        # Buffer for streaming results from callback
        buffer = []

        def callback(world, world_data, fitness):
            step = {
                'data': {},
                'index': fitness
            }
            for species, properties in const.SPECIES_MAP.items():
                biomass_offset = properties["biomass_offset"]
                step['data'][f'{species}'] = world[:, :, biomass_offset].sum().item()
            simulations[simulation_id]['steps'].append(step)
            # Append the result to the buffer
            buffer.append(f"data: {json.dumps(step)}\n\n")

        # Start the simulation in the runner
        runner.simulate(agent_file=const.EVAL_AGENT, visualize=callback)

        # Stream results from the buffer
        while buffer:
            yield buffer.pop(0)

        yield "data: [DONE]\n\n"

    return Response(generate_simulation(), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
