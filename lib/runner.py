import threading
import torch.multiprocessing as mp
import lib.constants as const
from lib.world import update_smell, read_map_from_file, remove_species_from_fishing, respawn_plankton, reset_plankton_cluster, move_plankton_cluster, move_plankton_based_on_current, spawn_plankton, perform_action, world_is_alive, create_map_from_noise
from lib.data_manager import queue_data
from lib.model import Model
from lib.evolution import elitism_selection, tournament_selection, crossover, mutation
import time
import torch
import random
import copy

device = torch.device(const.DEVICE)

@torch.no_grad()
def evaluate_agent(agent_dict, world, world_data, agent_index, evaluation_index, data_queue, visualize = None):
    agent = Model(chromosome=agent_dict).to(device)
    fitness = 0
    world = world.clone()  # Clone the padded world tensor for each agent
    # reset_plankton_cluster()
    species_order = [species for species in const.SPECIES_MAP.keys()]
    # Create the grid coordinates:
    grid_x, grid_y = torch.meshgrid(
        torch.arange(const.WORLD_SIZE, device=device),
        torch.arange(const.WORLD_SIZE, device=device),
        indexing='ij'
    )

    # Compute a 5-coloring of the grid using the function color = (i + 2*j) mod 5.
    colors = (grid_x + 2 * grid_y) % 5
    max_biomass = world[..., 3:7].max()
    max_smell = world[..., 7:11].max()
    
    while world_is_alive(world) and fitness < const.MAX_STEPS:
        update_smell(world)
        species_order = sorted(species_order, key=lambda x: random.random())

        for species in species_order:
            if species == "plankton":
                spawn_plankton(world, world_data)
                continue

             # Select a color and build a mask directly on the GPU
            selected_color = random.choice([0, 1, 2, 3, 4])
            selected_set_mask = (colors == selected_color)
            selected_positions = selected_set_mask.nonzero(as_tuple=False)
            selected_positions_padded = selected_positions + 1  # adjust if padding

            biomass_offset = const.SPECIES_MAP[species]["biomass_offset"]
            biomass_values = world[selected_positions_padded[:, 0], selected_positions_padded[:, 1], biomass_offset]
            non_zero_biomass_mask = biomass_values > 0
            selected_positions_padded = selected_positions_padded[non_zero_biomass_mask]

            # If no cells have non-zero biomass, skip to the next species
            if selected_positions_padded.size(0) == 0:
                continue

            # Step 4: Extract neighborhoods for selected cells
            offsets = torch.tensor([
                [-1, -1], [-1, 0], [-1, 1],
                [ 0, -1], [ 0, 0], [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1]
            ], device=device)
            neighbor_positions = (selected_positions_padded.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 2)
            neighbor_positions[:, 0].clamp_(0, const.WORLD_SIZE + 1)
            neighbor_positions[:, 1].clamp_(0, const.WORLD_SIZE + 1)

            neighbor_values = world[neighbor_positions[:, 0], neighbor_positions[:, 1]]
            neighbor_values = neighbor_values.view(selected_positions_padded.size(0), 9, const.TOTAL_TENSOR_VALUES)

            terrain = neighbor_values[..., 0:3]
            biomass = neighbor_values[..., 3:7] / (max_biomass + 1e-8)
            smell   = neighbor_values[..., 7:11] / (max_smell + 1e-8)
            batch_tensor = torch.cat([terrain, biomass, smell], dim=-1).view(selected_positions_padded.size(0), -1)

            action_values_batch = agent.forward(batch_tensor, species)
            world = perform_action(world, world_data, action_values_batch, species, selected_positions_padded)

        # Update the display after each step
        if data_queue:
            queue_data(agent_index, evaluation_index, fitness, data_queue)
        if visualize:
            visualize(world, world_data, fitness)
        fitness += 1
        # time.sleep(0.5)

    return (agent_index, evaluation_index, fitness)

def evaluate_agent_wrapper(args):
    return evaluate_agent(*args)

class Runner():
    def __init__(self, map_folder = 'maps/baltic'):
        self.current_generation = 0
        self.best_fitness = 0
        self.running = False
        self.agents = [(Model().state_dict(), 0) for _ in range(const.NUM_AGENTS)]
        self.best_agent = None
        self.starting_world = None
        world, world_data = read_map_from_file(map_folder)
        self.world = torch.nn.functional.pad(world, (0, 0, 1, 1, 1, 1), "constant", 0)
        self.world_data = torch.nn.functional.pad(world_data, (0, 0, 1, 1, 1, 1), "constant", 0)
        self.generation_finished = threading.Event()

    def simulate(self, agent_file, visualize):
        agent = torch.jit.load(agent_file, map_location=device)
        
        evaluate_agent(agent.state_dict(), self.world, self.world_data, 0, 0, None, visualize)

        print("DONE")
    
    def run(self, single_run=False):
        """
        If running on CUDA, we evaluate agents in the main thread.
        Otherwise, we fall back to CPU-based multiprocessing.
        """
        # Prepare tasks for evaluating each agent
        tasks = [
            (agent, self.world, self.world_data, agent_index, evaluation_index, None)
            for agent_index, (agent, _) in enumerate(self.agents)
            for evaluation_index, _ in enumerate(range(const.AGENT_EVALUATIONS))
        ]

        if const.DEVICE != 'cpu':
            print("Running on CUDA/MPS - evaluating agents in the main thread...")
            
            # No multiprocessing: evaluate each task directly on the GPU
            data_queue = None
            results = []
            for t in tasks:
                results.append(evaluate_agent_wrapper(t))            
        else:
            print("Running on CPU - using multiprocessing for agent evaluation...")
            
            # Use CPU multiprocessing
            manager = mp.Manager()
            data_queue = manager.Queue()
            
            # Update tasks so the data_queue is passed in
            tasks = [
                (agent, self.world, self.world_data, agent_index, evaluation_index, data_queue)
                for agent_index, (agent, _) in enumerate(self.agents)
                for evaluation_index, _ in enumerate(range(const.AGENT_EVALUATIONS))
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(evaluate_agent_wrapper, tasks)

        self.data_queue = data_queue

        agent_fitnesses = [0] * len(self.agents)
        for i, result in enumerate(results):
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
            model = torch.jit.script(Model(chromosome=fittest_agent[0]))
            torch.jit.save(model, f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_{self.best_fitness}.pt')

        elites = elitism_selection(self.agents, const.ELITISM_SELECTION)
        next_pop = []

        while len(next_pop) < (const.NUM_AGENTS - 2):
            (p1, _), (p2, _) = tournament_selection(elites, 2, const.TOURNAMENT_SELECTION)

            c1_weights, c2_weights = crossover(p1, p2)

            current_mutation_rate = max(const.MIN_MUTATION_RATE, const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
            mutation(c1_weights, current_mutation_rate, current_mutation_rate)
            mutation(c2_weights, current_mutation_rate, current_mutation_rate)

            next_pop.append((c1_weights, 0))
            next_pop.append((c2_weights, 0))
            
        self.agents = next_pop
        self.agents.append((fittest_agent[0], 0))
        self.agents.append((self.best_agent, 0))
        self.agents.append((Model().state_dict(), 0))

        self.generation_finished.clear()

