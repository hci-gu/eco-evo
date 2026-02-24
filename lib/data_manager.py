from queue import Empty
from lib.config.settings import Settings
from lib.config.const import SPECIES
from lib.model import MODEL_OFFSETS
import numpy as np
import json

generations_data = []

def queue_data(agent_index, eval_index, step, data_queue, species=None):
    data = {
        'species': species,
        'agent_index': agent_index,
        'eval_index': eval_index,
        'step': step,
    }
    data_queue.put(data)

def save_data_to_file(settings: Settings, generation, agents_data_snapshot):
    agents_file = f'{settings.folder}/agents_data_gen_{generation}.json'
    generations_file = f'{settings.folder}/generations_data.json'
    
    with open(generations_file, 'w') as f:
        json.dump(generations_data, f, indent=4, default=str)

def update_generations_data(settings: Settings, generation, agents_data):
    fitness_values = {}
    fitness_method = str(getattr(settings, "fitness_method", "simple")).strip().lower()

    is_species_map = bool(agents_data) and all(
        isinstance(key, str) and key in SPECIES for key in agents_data.keys()
    )

    def _fitness_for_eval(data):
        # Prefer explicitly recorded fitness from runner/evaluator.
        series = data.get('fitness', [])
        if series:
            return float(series[-1])
        # Backward-compatible fallback for older runs.
        steps = data.get('steps', [])
        if not steps:
            return 0.0
        if fitness_method == "simple":
            return float(steps[-1])
        return float(max(steps))

    if is_species_map:
        for species, _ in agents_data.items():
            fitness_values[species] = []
            for _, evals in agents_data[species].items():
                total_fitness = 0
                for _, data in evals.items():
                    fitness = _fitness_for_eval(data)
                    total_fitness += fitness
                fitness_values[species].append(total_fitness / len(evals))
    else:
        fitness_values = []
        for _, evals in agents_data.items():
            total_fitness = 0
            for _, data in evals.items():
                fitness = _fitness_for_eval(data)
                total_fitness += fitness
            fitness_values.append(total_fitness / len(evals))

    generations_data.append(fitness_values)
    agents_data_snapshot = agents_data.copy()  # Take a snapshot of agents_data
    agents_data.clear()
    save_data_to_file(settings, generation, agents_data_snapshot)  # Save data after clearing agents_data

    return generations_data

def process_data(data, agents_data):
    agent_index = data['agent_index']
    eval_index = data['eval_index']
    step = data['step']
    agent_species = data.get('species', None)

    curr_agents_data = agents_data
    if (agent_species is not None):
        if agent_species not in agents_data:
            agents_data[agent_species] = {}
        curr_agents_data = agents_data[agent_species]

    if agent_index not in curr_agents_data:
        curr_agents_data[agent_index] = {}
    if eval_index not in curr_agents_data[agent_index]:
        curr_agents_data[agent_index][eval_index] = {
            'steps': [],
            'fitness': [],
        }
        for species in SPECIES:
            curr_agents_data[agent_index][eval_index][f'{species}_alive'] = []
            curr_agents_data[agent_index][eval_index][f'{species}_energy'] = []

    curr_agents_data[agent_index][eval_index]['steps'].append(step)
    if "fitness" in data:
        curr_agents_data[agent_index][eval_index]['fitness'].append(float(data["fitness"]))

    # check if world data exists
    if "world" in data:
        world = data['world']
        for species in SPECIES:
            biomass_offset = MODEL_OFFSETS[species]["biomass"]
            energy_offset = MODEL_OFFSETS[species]["energy"]
            cells_with_biomass = world[:, :, biomass_offset] > 0
            average_energy = world[cells_with_biomass, energy_offset].mean() if np.any(cells_with_biomass) else 0
            curr_agents_data[agent_index][eval_index][f'{species}_alive'].append(world[..., biomass_offset].sum())
            curr_agents_data[agent_index][eval_index][f'{species}_energy'].append(average_energy)

    if agent_species is not None:
        agents_data[agent_species] = curr_agents_data
    else:
        agents_data = curr_agents_data

    return agents_data        

def data_loop(data_queue):
    try:
        while True:
            data = data_queue.get_nowait()
            process_data(data)
    except Empty:
        pass
