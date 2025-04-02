import threading
import multiprocessing as mp
import numpy as np
import lib.constants as const
from lib.world import (
    update_smell,
    read_map_from_file,
    spawn_plankton,
    perform_action,
    world_is_alive,
)
from lib.data_manager import queue_data
from lib.model import Model
from lib.evolution import elitism_selection, tournament_selection, crossover, mutation
import time
import random
import copy

def evaluate_agent(agent_dict, world, world_data, agent_index, evaluation_index, data_queue, visualize=None):
    # Create a new agent instance using the provided chromosome (assume Model.forward now works with NumPy).
    agent = Model(chromosome=agent_dict)
    fitness = 0
    world = np.copy(world)  # Clone the padded world array
    species_order = list(const.SPECIES_MAP.keys())
    
    grid_x, grid_y = np.meshgrid(np.arange(const.WORLD_SIZE), np.arange(const.WORLD_SIZE), indexing='ij')
    colors = (grid_x % 3) + 3 * (grid_y % 3)
    max_biomass = np.max(world[..., 3:7])
    max_smell = np.max(world[..., 7:11])
    
    while world_is_alive(world) and fitness < const.MAX_STEPS:
        update_smell(world)
        random.shuffle(species_order)  # Randomize species order

        for species in species_order:
            if species == "plankton":
                spawn_plankton(world, world_data)
                continue

            color_order = list(np.arange(9))
            random.shuffle(color_order)
            for selected_color in color_order:
                # selected_color = random.choice(np.arange(9))
                selected_set_mask = (colors == selected_color)
                selected_positions = np.argwhere(selected_set_mask)  # Shape (n,2)
                selected_positions_padded = selected_positions + 1      # Adjust for padding

                biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
                biomass_values = world[selected_positions_padded[:, 0], selected_positions_padded[:, 1], biomass_offset]
                non_zero_biomass_mask = biomass_values > 0
                selected_positions_padded = selected_positions_padded[non_zero_biomass_mask]

                if selected_positions_padded.shape[0] == 0:
                    continue

                # Extract 3x3 neighborhoods.
                offsets = np.array([
                    [-1, -1], [-1, 0], [-1, 1],
                    [ 0, -1], [ 0, 0], [ 0, 1],
                    [ 1, -1], [ 1, 0], [ 1, 1]
                ])
                neighbor_positions = selected_positions_padded[:, None, :] + offsets[None, :, :]
                # Clip neighbors so indices remain within padded world.
                neighbor_positions[:, :, 0] = np.clip(neighbor_positions[:, :, 0], 0, const.WORLD_SIZE + 1)
                neighbor_positions[:, :, 1] = np.clip(neighbor_positions[:, :, 1], 0, const.WORLD_SIZE + 1)
                n = selected_positions_padded.shape[0]
                neighbor_positions = neighbor_positions.reshape(n * 9, 2)
                neighbor_values = world[neighbor_positions[:, 0], neighbor_positions[:, 1]]
                neighbor_values = neighbor_values.reshape(n, 9, const.TOTAL_TENSOR_VALUES)

                terrain = neighbor_values[..., 0:3]
                biomass = neighbor_values[..., 3:7] / (max_biomass + 1e-8)
                smell   = neighbor_values[..., 7:11] / (max_smell + 1e-8)
                # Concatenate to get a (n, 9, 11) array, then flatten to (n, 99).
                batch_tensor = np.concatenate([terrain, biomass, smell], axis=-1).reshape(n, -1)
                
                # Forward pass through the agent (assumes a NumPy-based forward).
                action_values_batch = agent.forward(batch_tensor, species)
                # Update the world using our NumPy version of perform_action.
                world = perform_action(world, world_data, action_values_batch, species, selected_positions_padded)

        if data_queue is not None:
            queue_data(agent_index, evaluation_index, fitness, data_queue)
        if visualize:
            visualize(world, world_data, fitness)
        fitness += 1
        # Optionally: time.sleep(0.5)
    return (agent_index, evaluation_index, fitness)

def evaluate_agent_wrapper(args):
    return evaluate_agent(*args)

class Runner():
    def __init__(self, map_folder='maps/baltic'):
        self.current_generation = 0
        self.best_fitness = 0
        self.running = False
        # Initialize agents as a list of tuples: (chromosome, fitness).
        self.agents = [(Model().state_dict(), 0) for _ in range(const.NUM_AGENTS)]
        self.best_agent = None
        world, world_data = read_map_from_file(map_folder)  # These now return NumPy arrays.
        # Pad the arrays with 1 cell on each side.
        self.world = np.pad(world, ((1,1), (1,1), (0,0)), mode='constant', constant_values=0)
        self.world_data = np.pad(world_data, ((1,1), (1,1), (0,0)), mode='constant', constant_values=0)
        self.generation_finished = threading.Event()

    def simulate(self, agent_file, visualize):
        # For simulation, assume Model has a load method that returns a numpy-compatible model.
        agent = Model()  # Replace with your loading mechanism, e.g., agent = Model.load(agent_file)
        evaluate_agent(agent.state_dict(), self.world, self.world_data, 0, 0, None, visualize)
    
    def run(self, single_run=False):
        # Prepare tasks: one task per agent evaluation.
        tasks = [
            (agent, self.world, self.world_data, agent_index, evaluation_index, None)
            for agent_index, (agent, _) in enumerate(self.agents)
            for evaluation_index in range(const.AGENT_EVALUATIONS)
        ]
        if const.DEVICE != 'cpu':
            print("Running in main thread (simulated GPU mode)...")
            data_queue = None
            results = []
            for t in tasks:
                results.append(evaluate_agent_wrapper(t))
        else:
            print("Running on CPU with multiprocessing...")
            manager = mp.Manager()
            data_queue = manager.Queue()
            tasks = [
                (agent, self.world, self.world_data, agent_index, evaluation_index, data_queue)
                for agent_index, (agent, _) in enumerate(self.agents)
                for evaluation_index in range(const.AGENT_EVALUATIONS)
            ]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(evaluate_agent_wrapper, tasks)
        self.data_queue = data_queue
        agent_fitnesses = [0] * len(self.agents)
        for result in results:
            agent_index, evaluation_index, fitness = result
            agent_fitnesses[agent_index] += fitness
        for i, fitness in enumerate(agent_fitnesses):
            self.agents[i] = (self.agents[i][0], fitness / const.AGENT_EVALUATIONS)
        self.generation_finished.set()

    def next_generation(self):
        self.current_generation += 1
        
        fittest_agent = max(self.agents, key=lambda x: x[1])
        average_fitness = sum([x[1] for x in self.agents]) / len(self.agents)
        if average_fitness > self.best_fitness:
            self.best_fitness = average_fitness
            print(f"New best fitness: {self.best_fitness}")
            self.best_agent = copy.deepcopy(fittest_agent[0])
            # Save the best model (replace torch.jit.script and torch.jit.save with your own saving mechanism).
            model = Model(chromosome=fittest_agent[0])
            model.save(f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_{self.best_fitness}.npy')
        
        elites = elitism_selection(self.agents, const.ELITISM_SELECTION)
        next_pop = []
        while len(next_pop) < (const.NUM_AGENTS - 2):
            (p1, _), (p2, _) = tournament_selection(elites, 2, const.TOURNAMENT_SELECTION)
            c1_weights, c2_weights = crossover(p1, p2)
            current_mutation_rate = max(const.MIN_MUTATION_RATE, 
                                        const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
            mutation(c1_weights, current_mutation_rate, current_mutation_rate)
            mutation(c2_weights, current_mutation_rate, current_mutation_rate)
            next_pop.append((c1_weights, 0))
            next_pop.append((c2_weights, 0))
        self.agents = next_pop
        self.agents.append((fittest_agent[0], 0))
        self.agents.append((self.best_agent, 0))
        self.agents.append((Model().state_dict(), 0))
        self.generation_finished.clear()
