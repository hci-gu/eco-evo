from queue import Queue, Empty
import lib.constants as const
import json

agents_data = {}
generations_data = []

def queue_data(agent_index, eval_index, step, data_queue):
    data = {
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

def update_generations_data(generation):
    global agents_data
    fitness_values = []
    for _, evals in agents_data.items():
        total_fitness = 0
        for _, data in evals.items():
            fitness = max(data['steps'])
            total_fitness += fitness
        fitness_values.append(total_fitness / len(evals))

    # fitness not empty
    if len(fitness_values) > 0:
        generations_data.append(fitness_values)
        agents_data_snapshot = agents_data.copy()  # Take a snapshot of agents_data
        agents_data.clear()
        save_data_to_file(generation, agents_data_snapshot)  # Save data after clearing agents_data

    return generations_data

def process_data(data):
    global agents_data
    agent_index = data['agent_index']
    eval_index = data['eval_index']
    step = data['step']

    if agent_index not in agents_data:
        agents_data[agent_index] = {}
    if eval_index not in agents_data[agent_index]:
        agents_data[agent_index][eval_index] = {
            'steps': [],
            'plankton_alive': [],
            'anchovy_alive': [],
            'cod_alive': [],
        }

    agents_data[agent_index][eval_index]['steps'].append(step)

    # check if world data exists
    if ("world" in data):
        world = data['world']
        agents_data[agent_index][eval_index]['plankton_alive'].append(world[:, :, const.OFFSETS_BIOMASS_PLANKTON].sum())
        agents_data[agent_index][eval_index]['anchovy_alive'].append(world[:, :, const.OFFSETS_BIOMASS_ANCHOVY].sum())
        agents_data[agent_index][eval_index]['cod_alive'].append(world[:, :, const.OFFSETS_BIOMASS_COD].sum())

    return agents_data

        

def data_loop(data_queue):
    try:
        while True:
            data = data_queue.get_nowait()
            process_data(data)
    except Empty:
        pass