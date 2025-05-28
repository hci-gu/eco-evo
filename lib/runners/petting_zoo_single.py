import lib.constants as const
from lib.model import Model
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations, plot_biomass
from lib.data_manager import data_loop, update_generations_data, process_data
from lib.behaviour import run_all_scenarios
import numpy as np
import random
import copy
from lib.environments.petting_zoo import env

def noop(a, b):
    pass

model_path = "results/pure_behav_1/agents/80_0.9999999792127552.npy.npz"
base_chromosome = np.load(model_path)
print("model_path", model_path)

class PettingZooRunnerSingle():
    def __init__(self, render_mode="none"):
        # Create the environment (render_mode can be "none" if visualization is not needed)
        self.env = env(render_mode=render_mode)
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        self.current_generation = 0
        self.population = [Model(
            chromosome=base_chromosome
        ) for _ in range(const.NUM_AGENTS)]
        self.best_fitness = -float('inf')
        self.best_agent = None
        self.agent_index = 0
        self.eval_index = 0

    def run(self, candidate, seed = None, is_evaluation = False, callback = noop):
        self.env.reset(seed)
        steps = 0
        episode_length = 0
        fitness = 0
        callback(self.env.world, fitness)
        agents_data = process_data({
            'agent_index': self.agent_index,
            'eval_index': self.eval_index,
            'step': episode_length,
            'world': self.env.world
        }, self.env.plot_data)

        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()) and episode_length < const.MAX_STEPS:
            steps += 1
            agent = self.env.agent_selection
            if agent == "plankton":
                self.env.step(self.empty_action)
            else:
                obs, reward, termination, truncation, info = self.env.last()
                patches = obs  # shape: (num_patches, 3, 3, channels)
                flat_obs = patches.reshape(patches.shape[0], -1)  # shape: (num_patches, 135)

                # Encode species index
                species_index = self.env.possible_agents.index(agent) / len(self.env.possible_agents)
                species_vector = np.full((flat_obs.shape[0], 1), species_index, dtype=np.float32)

                # Concatenate
                obs_with_species = np.concatenate([flat_obs, species_vector], axis=1)  # shape: (num_patches, 136)

                # Pass to model
                action_values = candidate.forward(obs_with_species)
                # action_values = candidate.forward(obs.reshape(-1, 135), agent)
                action_values = action_values.reshape(const.WORLD_SIZE, const.WORLD_SIZE, const.AVAILABLE_ACTIONS)

                self.env.step(action_values)
            
            if steps % 4 == 0:
                episode_length += 1
                fitness += 1
                agents_data = process_data({
                    'agent_index': self.agent_index,
                    'eval_index': self.eval_index,
                    'step': episode_length,
                    'world': self.env.world
                }, self.env.plot_data)
                callback(self.env.world, fitness)
                if self.env.render_mode == "human":
                    plot_biomass(agents_data)
        
        return fitness, episode_length


    def evaluate_population(self):
        """
        Evaluate each model in the population for every species.
        When evaluating a candidate for a species S, for every other species (not S),
        we use the best model from previous generations (if available) to decide their actions.
        Returns a dictionary mapping species to a list of (chromosome, fitness) tuples.
        """
        eval_seeds = [int(random.random() * 100000) for _ in range(const.AGENT_EVALUATIONS)]
        fitnesses = []

        for idx in range(len(self.population)):
            self.agent_index = idx
            evaluation_candidate = self.population[idx]

            evals_fitness = []
            for eval_index in range(const.AGENT_EVALUATIONS):
                self.eval_index = eval_index
                
                fitness, episode_length = self.run(evaluation_candidate, eval_seeds[eval_index])
                while (episode_length == 0):
                    print("something went wrong update this seed")
                    eval_seeds[eval_index] = int(random.random() * 100000)
                    fitness, episode_length = self.run(evaluation_candidate, eval_seeds[eval_index])
                
                evals_fitness.append(fitness)
                print("idx", idx, "eval", eval_index, "fitness", fitness, "episode_length", episode_length)
            
            avg_fitness = sum(evals_fitness) / len(evals_fitness)
            # behaviour_score = run_all_scenarios(evaluation_candidate, False)
            # avg_fitness = avg_fitness * behaviour_score
            
            fitnesses.append((evaluation_candidate.state_dict(), avg_fitness))
            # print("finished eval, behaviour_score:", behaviour_score, ", fitness:", avg_fitness)
            print("finished eval, fitness:", avg_fitness)
        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = {}
        # fittest_agents_for_generation = {}
        fittest_agent = max(fitnesses, key=lambda x: x[1])
        # fittest_agents_for_generation = Model(chromosome=fittest_agent[0])

        current_population = fitnesses

        # Select elites.
        elites = evolution.elitism_selection(current_population, const.ELITISM_SELECTION)
        print("selected elites with fitnesses: ", [x[1] for x in elites])
        next_pop = []

        print("evolve, current mutation rate: ", const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
        while len(next_pop) < const.NUM_AGENTS:
            (p1, _), (p2, _) = evolution.tournament_selection(elites, 2, const.TOURNAMENT_SELECTION)
            c1_weights, c2_weights = evolution.sbx_crossover(p1, p2)
            current_mutation_rate = max(const.MIN_MUTATION_RATE,
                                        const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
            evolution.mutation(c1_weights, current_mutation_rate, current_mutation_rate)
            evolution.mutation(c2_weights, current_mutation_rate, current_mutation_rate)
            next_pop.append(c1_weights)
            next_pop.append(c2_weights)

        # Update best fitness/agent.
        if fittest_agent[1] > self.best_fitness:
            self.best_fitness = fittest_agent[1]
            self.best_agent = fittest_agent[0]
            model = Model(chromosome=fittest_agent[0])
            model.save(f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_{self.best_fitness}.npy')

        # add best agent to the population
        next_pop.append(copy.deepcopy(self.best_agent))
        new_population = [Model(chromosome=chrom) for chrom in next_pop]
        if (self.current_generation < 50):
            new_population.append(Model(
                # chromosome=base_chromosome.copy()
            ))
        
        random.shuffle(new_population)
    
        self.population = new_population

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        print(f"Generation {self.current_generation} completed")
        generations_data = update_generations_data(self.current_generation, self.env.plot_data)
        plot_generations(generations_data)
        self.env.plot_data = {}
        # Evolve the population based on fitnesses.
        self.evolve_population(fitnesses)

    def train(self, generations=const.GENERATIONS_PER_RUN):
        for _ in range(generations):
            self.run_generation()
            # Optionally, save best models or log additional statistics.

    def evaluate(self, model_path = [], callback = noop, seed = 1):
        candidate = Model(
            chromosome=np.load(model_path)
        )

        return self.run(candidate, seed, True, callback)
        
