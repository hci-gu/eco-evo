from queue import Queue, Empty
import numpy as np
# import lib.constants as const
import json

agents_data = {}
generations_data = []

def queue_data(agent_index, eval_index, step, data_queue, species=None):
    data = {
        'species': species,
        'agent_index': agent_index,
        'eval_index': eval_index,
        'step': step,
    }
    data_queue.put(data)

def save_data_to_file(generation, agents_data_snapshot):
    agents_file = f'{const.CURRENT_FOLDER}/agents_data_gen_{generation}.json'
    generations_file = f'{const.CURRENT_FOLDER}/generations_data.json'
    
    # with open(agents_file, 'w') as f:
    #     json.dump(agents_data_snapshot, f, indent=4, default=str)
    with open(generations_file, 'w') as f:
        json.dump(generations_data, f, indent=4, default=str)

def update_generations_data(generation, agents_data=agents_data, generations_data=generations_data):
    fitness_values = {}

    if len(agents_data.items()) == 3:
        # we are working with species map
        for species, _ in agents_data.items():
            fitness_values[species] = []
            for _, evals in agents_data[species].items():
                total_fitness = 0
                for _, data in evals.items():
                    fitness = max(data['steps'])
                    total_fitness += fitness
                fitness_values[species].append(total_fitness / len(evals))
    else:
        fitness_values = []
        for _, evals in agents_data.items():
            total_fitness = 0
            for _, data in evals.items():
                fitness = max(data['steps'])
                total_fitness += fitness
            fitness_values.append(total_fitness / len(evals))

    generations_data.append(fitness_values)
    agents_data_snapshot = agents_data.copy()  # Take a snapshot of agents_data
    agents_data.clear()
    save_data_to_file(generation, agents_data_snapshot)  # Save data after clearing agents_data

    return generations_data

def process_data(data, agents_data=agents_data):
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
        }
        for species in const.SPECIES_MAP.keys():
            curr_agents_data[agent_index][eval_index][f'{species}_alive'] = []
            curr_agents_data[agent_index][eval_index][f'{species}_energy'] = []

    curr_agents_data[agent_index][eval_index]['steps'].append(step)

    # check if world data exists
    if "world" in data:
        world = data['world']
        for species, properties in const.SPECIES_MAP.items():
            biomass_offset = properties["biomass_offset"]
            energy_offset = properties["energy_offset"]
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