from flask import Flask, request, jsonify, Response
from flask_cors import CORS
# import torch
import os
import uuid
import json
import threading
import queue
from lib.world.update_world import world_is_alive
from lib.world.map import read_map_from_file
# from lib.runner import Runner
from lib.runners.petting_zoo import PettingZooRunner

app = Flask(__name__)
CORS(app)

simulations = {}

def get_model_path():
    folder = "results/single_agent_single_out_random_plankton_behavscore_6/agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".npy.npz")]
    files.sort(key=lambda f: float(f.split("_")[1].split(".npy")[0]), reverse=True)

    file = files[0]
    model_path = os.path.join(folder, file)
    return model_path

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
    # TODO: add this back later
    # os.makedirs(folder_path, exist_ok=True)

    # Save the uploaded files TODO: add this back later
    # texture_path = os.path.join(folder_path, 'map.png')
    # depth_path = os.path.join(folder_path, 'depth.png')

    # try:
    #     texture_file.save(texture_path)
    #     depth_file.save(depth_path)
    # except Exception as e:
    #     return jsonify({"error": f"Failed to save files: {str(e)}"}), 500

    simulations[simulation_id] = {
        'steps': [],
        'options': options,
        'folder_path': folder_path,
    }

    return jsonify({"id": simulation_id}), 200

@app.route('/simulate/agents', methods=['GET'])
def get_agents():
    folder = "agents"
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".pt")]
    
    return jsonify(files)

@app.route('/simulate/<simulation_id>', methods=['GET'])
def stream_simulation(simulation_id):
    print("Starting simulation:", simulation_id)
    if simulation_id not in simulations:
        return jsonify({"error": "Simulation ID not found"}), 404

    def generate_simulation():
        q = queue.Queue()

        def callback(_, day, data):
            step = {
                'data': data,
                'index': day
            }
            simulations[simulation_id]['steps'].append(step)
            q.put(f"data: {json.dumps(step)}\n\n")

        def run_simulation():
            # folder_path = simulations[simulation_id]['folder_path']
            options = simulations[simulation_id]['options']
            runner = PettingZooRunnerSingle()
            runner.evaluate_with_plan(options, get_model_path(), callback)
            q.put("data: [DONE]\n\n")

        threading.Thread(target=run_simulation).start()

        while True:
            try:
                msg = q.get(timeout=1)
                yield msg
                if msg == "data: [DONE]\n\n":
                    break
            except queue.Empty:
                continue  # Prevent hanging if nothing is received yet

    return Response(generate_simulation(), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
