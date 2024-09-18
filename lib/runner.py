from lib.constants import WORLD_SIZE, NUM_AGENTS, MAX_STEPS, PLANKTON_GROWTH_RATE, MAX_PLANKTON_IN_CELL
from lib.world import create_world, perform_action, world_is_alive, Species, Action, Terrain
from lib.visualize import visualize, reset_plot
from lib.model import Model
from lib.evolution import elitism_selection, tournament_selection, crossover, mutation
import time
import torch
import random
import copy

# device mps or cpu
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


class Runner():
    def __init__(self):
        self.current_generation = 0
        self.best_fitness = 0
        self.running = False
        self.agents = [(Model().to(device), 0) for _ in range(NUM_AGENTS)]
        self.starting_world = None
    
    def run(self):
        # Create the world as a tensor from the start
        self.starting_world = create_world().to(device)

        # Pad the world with a 1-cell border of zeros
        padded_world = torch.nn.functional.pad(self.starting_world, (0, 0, 1, 1, 1, 1), "constant", 0)

        for agent_index, (agent, fitness) in enumerate(self.agents):
            print(f"Running agent {agent_index} of generation {self.current_generation}")
            world = padded_world.clone()  # Clone the padded world tensor for each agent
            species_order = [Species.PLANKTON, Species.ANCHOVY, Species.COD]

            time_started_gen = time.time()
            while world_is_alive(world) and fitness < MAX_STEPS:
                time_started = time.time()
                species_order = sorted(species_order, key=lambda x: random.random())

                for species in species_order:
                    if species == Species.PLANKTON:
                        # Apply plankton growth to the entire world in a batched operation
                        plankton_mask = (world[:, :, Terrain.WATER.value] == 1)  # 2D mask for water cells
                        # Update only the plankton biomass (channel 3) using the mask
                        world[:, :, 3][plankton_mask] = torch.min(world[:, :, 3][plankton_mask] * (1 + PLANKTON_GROWTH_RATE),
                                                                torch.tensor(MAX_PLANKTON_IN_CELL, device=device))
                        continue

                    # Collect tensors for cells (batch the inputs)
                    available_cells = set(range(WORLD_SIZE * WORLD_SIZE)) # All cells are available initially
                    batch_tensors = []
                    batch_positions = []

                    while len(available_cells) > 0:
                        cell_index = random.choice(list(available_cells))
                        x, y = cell_index // WORLD_SIZE, cell_index % WORLD_SIZE

                        # Adjust for padding by shifting the x and y coordinates by 1
                        padded_x, padded_y = x + 1, y + 1

                        # Get 3x3 tensor from the padded world (no need to check boundaries)
                        tensor = world[padded_x:padded_x + 3, padded_y:padded_y + 3].reshape(1, -1).to(device)

                        # Check if the tensor has valid size (not empty)
                        if tensor.numel() > 0:  # Ensure the tensor is not empty
                            if tensor.numel() != 54:
                                # pad the tensor with zeros
                                tensor = torch.cat([tensor, torch.zeros(1, 54 - tensor.numel(), device=device)], dim=1)
                            batch_tensors.append(tensor)
                            batch_positions.append((padded_x, padded_y))
                        else:
                            print(f"Warning: Skipping empty tensor at position ({padded_x}, {padded_y})")


                        # Remove the selected cell and its adjacent cells from the available set
                        adjacent_indices = set()
                        adjacent_indices.add(cell_index)  # Current cell
                        
                        if y > 0:  # West
                            adjacent_indices.add(cell_index - 1)
                        if y < (WORLD_SIZE - 1):  # East
                            adjacent_indices.add(cell_index + 1)
                        if x > 0:  # North
                            adjacent_indices.add(cell_index - WORLD_SIZE)
                        if x < (WORLD_SIZE - 1):  # South
                            adjacent_indices.add(cell_index + WORLD_SIZE)

                        available_cells -= adjacent_indices

                    # If no valid tensors were collected, skip to the next iteration
                    if len(batch_tensors) == 0:
                        print("No valid tensors in this iteration, skipping...")
                        continue

                    # Convert the list of tensors to a batch tensor
                    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

                    # Get action probabilities for the whole batch
                    action_values_batch = agent.forward(batch_tensor)

                    # Process actions in batch for each species
                    batch_positions = torch.tensor(batch_positions, dtype=torch.long, device=device)
                    species_batch = torch.full((batch_positions.shape[0],), species.value, device=device)

                    # Call perform_action_tensor to handle all actions in batch (this should be a new function)
                    world = perform_action(world, action_values_batch, species_batch, batch_positions)

                # Time logging for each step
                time_ended = time.time()
                seconds_elapsed = time_ended - time_started
                print(f"Time taken: {seconds_elapsed} seconds")

                # Update the display after each step
                visualize(world, agent_index, fitness)

                fitness += 1

        self.next_generation()


    def next_generation(self):
        print('Next generation')
        self.current_generation += 1
        
        fittest_agent = max(self.agents, key=lambda x: x[1])

        elites = elitism_selection(self.agents, 8)
        next_pop = []

        while len(next_pop) < (NUM_AGENTS - 2):
            (p1, _), (p2, _) = tournament_selection(elites, 2, 6)

            p1_weights = p1.get_weights()
            p2_weights = p2.get_weights()

            c1_W0, c2_W0, c1_b0, c2_b0, c1_W1, c2_W1, c1_b1, c2_b1 = crossover(p1_weights, p2_weights)

            c1_params = {'W0': c1_W0, 'b0': c1_b0, 'W1': c1_W1, 'b1': c1_b1}
            c2_params = {'W0': c2_W0, 'b0': c2_b0, 'W1': c2_W1, 'b1': c2_b1}

            mutation(c1_params, 0.1, 0.1)
            mutation(c2_params, 0.1, 0.1)

            c1 = Model(chromosome=c1_params).to(device)
            c2 = Model(chromosome=c2_params).to(device)

            next_pop.append((c1, 0))
            next_pop.append((c2, 0))
            

        self.agents = next_pop
        self.agents.append((fittest_agent[0], 0))
        self.agents.append((Model().to(device), 0))
        reset_plot()
        self.run()

