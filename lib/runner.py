# from lib.constants import WORLD_SIZE, NUM_AGENTS, MAX_STEPS, PLANKTON_GROWTH_RATE, MAX_PLANKTON_IN_CELL, TOURNAMENT_SELECTION, ELITISM_SELECTION
import lib.constants as const
from lib.world import create_world, respawn_plankton, reset_plankton_cluster, move_plankton_cluster, move_plankton_based_on_current, perform_action, perform_action_optimized, world_is_alive, Species, Action, Terrain
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

        for agent_index, (agent, _) in enumerate(self.agents):
            total_fitness = 0
            for evaluation_index in range(const.AGENT_EVALUATIONS):
                fitness = 0

                print(f"Running agent {agent_index} of generation {self.current_generation}, evaluation {evaluation_index}")
                world = padded_world.clone()  # Clone the padded world tensor for each agent
                reset_plankton_cluster()
                species_order = [Species.PLANKTON, Species.ANCHOVY]

                while world_is_alive(world) and fitness < const.MAX_STEPS:
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

                        # Step 1: Generate Independent Sets
                        grid_x, grid_y = torch.meshgrid(
                            torch.arange(const.WORLD_SIZE, device=device),
                            torch.arange(const.WORLD_SIZE, device=device),
                            indexing='ij'
                        )

                        # Create two independent sets using a checkerboard pattern
                        set_a_mask = ((grid_x + grid_y) % 2 == 0)
                        set_b_mask = ~set_a_mask

                        # Step 2: Randomly select one set
                        selected_set_mask = random.choice([set_a_mask, set_b_mask])

                        # Step 3: Get positions of selected cells
                        selected_positions = selected_set_mask.nonzero(as_tuple=False)  # Shape: [Num_Selected_Cells, 2]
                        selected_positions_padded = selected_positions + 1  # Adjust for padding if necessary

                        # Step 4: Extract neighborhoods for selected cells
                        offsets = torch.tensor([
                            [-1, -1], [-1, 0], [-1, 1],
                            [ 0, -1], [ 0, 0], [ 0, 1],
                            [ 1, -1], [ 1, 0], [ 1, 1]
                        ], device=device)  # Shape: [9, 2]

                        neighbor_positions = selected_positions_padded.unsqueeze(1) + offsets.unsqueeze(0)  # Shape: [Num_Selected_Cells, 9, 2]
                        neighbor_positions = neighbor_positions.reshape(-1, 2)

                        # Ensure indices are within bounds
                        neighbor_x = neighbor_positions[:, 0].clamp(0, const.WORLD_SIZE + 1)
                        neighbor_y = neighbor_positions[:, 1].clamp(0, const.WORLD_SIZE + 1)

                        # Extract neighbor values
                        neighbor_values = world[neighbor_x, neighbor_y].view(selected_positions.size(0), -1)

                        # Step 5: Prepare batch input
                        batch_tensor = neighbor_values  # Shape: [Num_Selected_Cells, Channels * 9]

                        # Step 6: Perform neural network forward pass
                        action_values_batch = agent.forward(batch_tensor, species.value)

                        # Step 7: Perform actions
                        world = perform_action(world, action_values_batch, species.value, selected_positions_padded)

                    # Update the display after each step
                    visualize(world, world_data, agent_index, evaluation_index, fitness)

                    fitness += 1
                    total_fitness += 1
                    
                
                average_fitness = total_fitness / const.AGENT_EVALUATIONS
                self.agents[agent_index] = (agent, average_fitness)

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

            current_mutation_rate = max(const.MIN_MUTATION_RATE, const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
            mutation(c1_weights, current_mutation_rate, current_mutation_rate)
            mutation(c2_weights, current_mutation_rate, current_mutation_rate)

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

