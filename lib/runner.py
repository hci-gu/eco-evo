# from lib.constants import WORLD_SIZE, NUM_AGENTS, MAX_STEPS, PLANKTON_GROWTH_RATE, MAX_PLANKTON_IN_CELL, TOURNAMENT_SELECTION, ELITISM_SELECTION
import lib.constants as const
from lib.world import create_world, respawn_plankton, reset_plankton_cluster, move_plankton_cluster, move_plankton_based_on_current, perform_action, world_is_alive, Species, Action, Terrain
from lib.visualize import visualize, reset_plot, draw_world_mask, reset_visualization
from lib.model import Model
from lib.evolution import elitism_selection, tournament_selection, crossover, mutation
import lib.test as test
import time
import torch
import random
import copy

# device mps or cpu
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

# world_copy = create_world().to(device)

class Runner():
    def __init__(self):
        self.current_generation = 0
        self.best_fitness = 0
        self.running = False
        self.agents = [(Model().to(device), 0) for _ in range(const.NUM_AGENTS)]
        self.best_agent = None
        self.starting_world = None
        reset_visualization()

    def simulate(self, agent_file):
        agent = Model()
        agent.load_state_dict(torch.load(agent_file))
        self.agents = [(agent, 0)]
        self.run(True)
    
    def run(self, single_run=False):
        # Create the world as a tensor from the start
        # self.starting_world = create_world().to(device)
        world, world_data = create_world(True)
        self.starting_world = world.to(device)

        # Pad the world with a 1-cell border of zeros
        padded_world = torch.nn.functional.pad(self.starting_world, (0, 0, 1, 1, 1, 1), "constant", 0)
        world_data = torch.nn.functional.pad(world_data, (0, 0, 1, 1, 1, 1), "constant", 0)

        for agent_index, (agent, fitness) in enumerate(self.agents):
            # print(f"Running agent {agent_index} of generation {self.current_generation}")
            world = padded_world.clone()  # Clone the padded world tensor for each agent
            reset_plankton_cluster()
            species_order = [Species.PLANKTON, Species.ANCHOVY, Species.COD]

            # time_started_gen = time.time()
            while world_is_alive(world) and fitness < const.MAX_STEPS:
                # time_started = time.time()
                species_order = sorted(species_order, key=lambda x: random.random())

                for species in species_order:
                    if species == Species.PLANKTON:
                        # Apply plankton growth to the entire world in a batched operation
                        plankton_mask = (world[:, :, Terrain.WATER.value] == 1)  # 2D mask for water cells
                        # Update only the plankton biomass (channel 3) using the mask
                        world[:, :, const.OFFSETS_BIOMASS_PLANKTON][plankton_mask] = torch.min(world[:, :, const.OFFSETS_BIOMASS_PLANKTON][plankton_mask] * (1 + const.PLANKTON_GROWTH_RATE),
                                                                torch.tensor(const.MAX_PLANKTON_IN_CELL, device=device))
                        if fitness % 20 == 0:
                            move_plankton_cluster(world)
                        #     spawn_plankton_debug(world)
                        #     move_plankton_based_on_current(world, world_data)
                        continue

                    # Collect tensors for cells (batch the inputs)
                    available_cells = set(range(const.WORLD_SIZE * const.WORLD_SIZE)) # All cells are available initially
                    batch_tensors = []
                    batch_positions = []

                    while len(available_cells) > 0:
                        cell_index = random.choice(list(available_cells))
                        x, y = cell_index // const.WORLD_SIZE, cell_index % const.WORLD_SIZE

                        # Adjust for padding by shifting the x and y coordinates by 1
                        padded_x, padded_y = x + 1, y + 1

                        # Get 3x3 tensor from the padded world (no need to check boundaries)
                        tensor = world[padded_x:padded_x + 3, padded_y:padded_y + 3].reshape(1, -1).to(device)

                        # Check if the tensor has valid size (not empty)
                        if tensor.numel() > 0:  # Ensure the tensor is not empty
                            if tensor.numel() != 81:
                                # pad the tensor with zeros
                                tensor = torch.cat([tensor, torch.zeros(1, 81 - tensor.numel(), device=device)], dim=1)
                            batch_tensors.append(tensor)
                            batch_positions.append((padded_x, padded_y))
                        else:
                            print(f"Warning: Skipping empty tensor at position ({padded_x}, {padded_y})")


                        # Remove the selected cell and its adjacent cells from the available set
                        adjacent_indices = set()
                        adjacent_indices.add(cell_index)  # Current cell
                        
                        if y > 0:  # West
                            adjacent_indices.add(cell_index - 1)
                        if y < (const.WORLD_SIZE - 1):  # East
                            adjacent_indices.add(cell_index + 1)
                        if x > 0:  # North
                            adjacent_indices.add(cell_index - const.WORLD_SIZE)
                        if x < (const.WORLD_SIZE - 1):  # South
                            adjacent_indices.add(cell_index + const.WORLD_SIZE)

                        available_cells -= adjacent_indices

                    # If no valid tensors were collected, skip to the next iteration
                    if len(batch_tensors) == 0:
                        print("No valid tensors in this iteration, skipping...")
                        continue

                    # Convert the list of tensors to a batch tensor
                    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

                    # Get action probabilities for the whole batch
                    action_values_batch = agent.forward(batch_tensor, species.value)

                    # Process actions in batch for each species
                    batch_positions = torch.tensor(batch_positions, dtype=torch.long, device=device)
                    species_batch = torch.full((batch_positions.shape[0],), species.value, device=device)

                    # Call perform_action_tensor to handle all actions in batch (this should be a new function)
                    world = perform_action(world, action_values_batch, species.value, batch_positions)

                # Time logging for each step
                # time_ended = time.time()
                # seconds_elapsed = time_ended - time_started
                # print(f"Time taken: {seconds_elapsed} seconds")

                # Update the display after each step
                visualize(world, world_data, agent_index, fitness)

                fitness += 1
                self.agents[agent_index] = (agent, fitness)

        if single_run:
            print('Simulation finished')
            return
        self.next_generation()

    def next_generation(self):
        print('Next generation')
        self.current_generation += 1
        
        fittest_agent = max(self.agents, key=lambda x: x[1])

        average_fitness = sum([x[1] for x in self.agents]) / len(self.agents)
        if average_fitness > self.best_fitness:
            self.best_fitness = average_fitness
            print(f"New best fitness: {self.best_fitness}")
            self.best_agent = copy.deepcopy(fittest_agent[0])
            fittest_agent[0].save_to_file(f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_{self.best_fitness}.pt')

        elites = elitism_selection(self.agents, const.ELITISM_SELECTION)
        next_pop = []

        while len(next_pop) < (const.NUM_AGENTS - 2):
            (p1, _), (p2, _) = tournament_selection(elites, 2, const.TOURNAMENT_SELECTION)

            p1_weights = p1.get_weights()
            p2_weights = p2.get_weights()

            c1_weights, c2_weights = crossover(p1_weights, p2_weights)

            mutation(c1_weights, 0.2, 0.2)
            mutation(c2_weights, 0.2, 0.2)

            c1 = Model(chromosome=c1_weights).to(device)
            c2 = Model(chromosome=c2_weights).to(device)

            next_pop.append((c1, 0))
            next_pop.append((c2, 0))
            

        self.agents = next_pop
        self.agents.append((fittest_agent[0], 0))
        self.agents.append((self.best_agent, 0))
        self.agents.append((Model().to(device), 0))
        reset_plot()

        if (self.current_generation <= const.GENERATIONS_PER_RUN):
            self.run()
        else:
            print('Simulation finished')

    def run_test_case(self):
        const.WORLD_SIZE = 3
        # Create the world as a tensor from the start
        world = test.create_test_world_3x3()

        # Pad the world with a 1-cell border of zeros
        world = torch.nn.functional.pad(world, (0, 0, 1, 1, 1, 1), "constant", 0)

        visualize(world, 0, 0)
        time.sleep(1)

        agent = test.MockModel().to(device)

        moves = [
            ((1, 1), (0, 1)),
            ((1, 1), (0, 1)),
        ]

        cell_index = 4
        x, y = cell_index // const.WORLD_SIZE, cell_index % const.WORLD_SIZE

        # Adjust for padding by shifting the x and y coordinates by 1
        padded_x, padded_y = x + 1, y + 1
        padded_x_2, padded_y_2 = padded_x - 1, padded_y

        # Get 3x3 tensor from the padded world (no need to check boundaries)
        tensor = world[padded_x:padded_x + 3, padded_y:padded_y + 3].reshape(1, -1).to(device)
        tensor2 = world[padded_x_2:padded_x_2 + 3, padded_y_2:padded_y_2 + 3].reshape(1, -1).to(device)

        batch_tensors = []
        batch_tensors.append(tensor)
        batch_tensors.append(tensor2)
        
        batch_positions = [(padded_x, padded_y), (padded_x_2, padded_y_2)]

        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

        # Get action probabilities for the whole batch
        action_values_batch = agent.forward(batch_tensor, Species.ANCHOVY.value)

        # Process actions in batch for each species
        batch_positions = torch.tensor(batch_positions, dtype=torch.long, device=device)
        # species_batch = torch.full((batch_positions.shape[0],), Species.ANCHOVY.value, device=device)

        # Call perform_action_tensor to handle all actions in batch (this should be a new function)
        world = perform_action(world, action_values_batch, Species.ANCHOVY.value, batch_positions, draw_world_mask)

        visualize(world, 0, 0)
        time.sleep(3)

