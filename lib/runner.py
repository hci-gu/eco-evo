from lib.constants import WORLD_SIZE, NUM_AGENTS, MAX_STEPS
from lib.world import create_world, world_is_alive, Species, Action
from lib.visualize import init_pygame, draw_world, plot_biomass, reset_plot, plot_generations
from lib.model import Model
from lib.evolution import elitism_selection, tournament_selection, crossover, mutation
import time
import torch
import random
import copy

class Runner():
    def __init__(self):
        self.current_generation = 0
        self.best_fitness = 0
        self.running = False
        self.agents = [(Model(), 0) for _ in range(NUM_AGENTS)]
        self.starting_world = None

    def run(self):
        self.starting_world = create_world()
        
        for agent_index, (agent, fitness) in enumerate(self.agents):
            print(f"Running agent {agent_index} of generation {self.current_generation}")
            world = copy.deepcopy(self.starting_world)
            species_order = [Species.PLANKTON, Species.ANCHOVY, Species.COD]

            time_started_gen = time.time()
            while world_is_alive(world) and fitness < MAX_STEPS:
                time_started = time.time()
                species_order = sorted(species_order, key=lambda x: random.random())

                for species in species_order:
                    if species == Species.PLANKTON:
                        for cell in world:
                            cell.plankton_growth()
                        continue

                    batch_tensors = []
                    batch_cells = []
                    available_cells = set(range(WORLD_SIZE * WORLD_SIZE))  # Use a set for available cells
                    
                    while len(available_cells) > 0:
                        cell_index = random.choice(list(available_cells))

                        # get tensor from cell
                        cell = world[cell_index]
                        x, y = cell_index // WORLD_SIZE, cell_index % WORLD_SIZE
                        tensor = cell.toTensor3x3(x, y, world).view(1, -1)
                        
                        batch_tensors.append(tensor)
                        batch_cells.append((cell, species, x, y))

                        # Identify and remove the selected cell and its adjacent cells
                        adjacent_indices = set()
                        adjacent_indices.add(cell_index)  # Current cell
                        
                        # Check and add adjacent cells based on their position
                        if y > 0:  # West
                            adjacent_indices.add(cell_index - 1)
                        if y < (WORLD_SIZE - 1):  # East
                            adjacent_indices.add(cell_index + 1)
                        if x > 0:  # North
                            adjacent_indices.add(cell_index - WORLD_SIZE)
                        if x < (WORLD_SIZE - 1):  # South
                            adjacent_indices.add(cell_index + WORLD_SIZE)
                        
                        # Remove the selected cell and its adjacent cells from the available set
                        available_cells -= adjacent_indices
                    
                    # Convert list of tensors to a batch tensor
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                
                    # Get action probabilities for the whole batch
                    action_values_batch = agent.forward(batch_tensor)

                    # Distribute the results back to the corresponding cells
                    for i, (cell, species, x, y) in enumerate(batch_cells):
                        action_values = action_values_batch[i]
                        # action values is a tensor of size 10 where first 5 are for cod and next 5 are for anchovy
                        # if we are processing cod, we will use first 5 values, else next 5 values
                        action_values = action_values[:5] if species == Species.COD else action_values[5:]
                        for action in Action:
                            action_index = action.value
                            action_probability = action_values[action_index].item()

                            cell.perform_action(species, action, action_probability, x, y, world)
                time_ended = time.time()
                seconds_elapsed = time_ended - time_started

                print(f"Time taken: {seconds_elapsed} seconds")
                if fitness > 0 and (fitness % 25) == 0:
                    time_ended_gen = time.time()
                    seconds_elapsed_gen = time_ended_gen - time_started_gen
                    avg_seconds_per_step = seconds_elapsed_gen / fitness
                    print(f"Time taken for 25 steps: {seconds_elapsed_gen} seconds, Avg time per step: {avg_seconds_per_step} seconds")
                    

                # plot_generations()
                # draw_world(world)
                # plot_biomass(agent_index, world, fitness)

                fitness += 1

        self.next_generation()


    def next_generation(self):
        print('Next generation')
        self.current_generation += 1
        
        fittest_agent = max(self.agents, key=lambda x: x[1])

        elites = elitism_selection(self.agents, 6)
        next_pop = []

        while len(next_pop) < (NUM_AGENTS - 2):
            (p1, _), (p2, _) = tournament_selection(elites, 2, 4)

            p1_weights = p1.get_weights()
            p2_weights = p2.get_weights()

            c1_W0, c2_W0, c1_b0, c2_b0, c1_W1, c2_W1, c1_b1, c2_b1 = crossover(p1_weights, p2_weights)

            c1_params = {'W0': c1_W0, 'b0': c1_b0, 'W1': c1_W1, 'b1': c1_b1}
            c2_params = {'W0': c2_W0, 'b0': c2_b0, 'W1': c2_W1, 'b1': c2_b1}

            mutation(c1_params, 0.05, 0.2)
            mutation(c2_params, 0.05, 0.2)

            c1 = Model(chromosome=c1_params)
            c2 = Model(chromosome=c2_params)

            next_pop.append((c1, 0))
            next_pop.append((c2, 0))
            

        self.agents = next_pop
        self.agents.append(fittest_agent)
        self.agents.append((Model(), 0))
        reset_plot()
        self.run()

