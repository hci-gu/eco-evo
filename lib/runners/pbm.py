from lib.config import const
from lib.config.settings import Settings
from lib.model import Model
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations
from lib.data_manager import update_generations_data, process_data
import numpy as np
import random
import copy
import time
from lib.environments.pbm_gym import PbmEnv, env_factory

def noop(a, b, c):
    pass

class PBMRunner():
    def __init__(self, settings: Settings, render_mode="none"):
        self.env = env_factory()
        self.settings = settings
        self.env.reset()
        self.current_generation = 0

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.ACTING_SPECIES]
        # Create a population for each species.
        self.population = [Model() for _ in range(self.settings.num_agents)]
        # Track best fitness and best model for each species.
        self.best_fitness = -float('inf')
        self.best_agent = None
        self.agent_index = 0
        self.eval_index = 0
        self.plot_data = {}

    def run(self, candidate, seed = None, is_evaluation = False, callback = noop):
        obs = self.env.reset(seed=seed)
        steps = 0
        episode_length = 0
        fitness = 0
        # callback(self.env.world, fitness, False)

        env_terminated = False
        env_truncated = False

        while not env_terminated and not env_truncated and episode_length < self.settings.max_steps:
            steps += 1
            # action = candidate.forward(obs.reshape(-1, 135))
            # obs, reward, termination, truncation, info = self.env.step(action)
            # action_values = action_values.reshape(self.settings.world_size, self.settings.world_size, 5)
            # mean_action_values = action_values.mean(axis=(0, 1))
            print(obs)
            action = candidate.forward(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            env_terminated = terminated
            env_truncated = truncated
            
            episode_length += 1
            fitness += reward
            if (is_evaluation == False or self.env.render_mode != "none"):
                process_data({
                    'species': None,
                    'agent_index': self.agent_index,
                    'eval_index': self.eval_index,
                    'step': episode_length,
                }, self.plot_data)
            callback(None, fitness, False)
        
        print("end of simulation", fitness, episode_length)
        callback(None, fitness, True)
        return fitness, episode_length, None

    def evaluate_population(self):
        """
        Evaluate each model in the population for every species.
        When evaluating a candidate for a species S, for every other species (not S),
        we use the best model from previous generations (if available) to decide their actions.
        Returns a dictionary mapping species to a list of (chromosome, fitness) tuples.
        """
        eval_seeds = [int(random.random() * 100000) for _ in range(self.settings.agent_evaluations)]
        fitnesses = []

        end_reasons = []
        for idx in range(self.settings.num_agents):
            self.agent_index = idx
            evaluation_candidate = self.population[idx]

            evals_fitness = []
            for eval_index in range(self.settings.agent_evaluations):
                start_time = time.time()
                self.eval_index = eval_index
                
                fitness, episode_length, end_reason = self.run(evaluation_candidate, eval_seeds[eval_index])
                end_reasons.append(end_reason)
                
                evals_fitness.append(fitness)
                end_time = time.time()
                print(f'idx {idx}, eval {eval_index}, fitness {fitness:.1f}, episode_length {episode_length}, steps/sec {episode_length/(end_time - start_time):.2f}')
            
            avg_fitness = sum(evals_fitness) / len(evals_fitness)

            fitnesses.append((evaluation_candidate.state_dict(), avg_fitness))
            print(f'finished eval, fitness: {avg_fitness:.1f}')

        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = []
        fittest_agent = max(fitnesses, key=lambda x: x[1])

        current_population = fitnesses

        # Select elites.
        elites = evolution.elitism_selection(current_population, self.settings.elitism_selection)
        next_pop = []

        next_gen_agents = self.settings.num_agents - 1
        if (self.current_generation < 25):
            next_gen_agents = self.settings.num_agents - 1 - 2

        while len(next_pop) < next_gen_agents:
            (p1, _), (p2, _) = evolution.tournament_selection(elites, 2, self.settings.tournament_selection)
            current_eta = min(10.0, self.settings.sbx_eta * (self.settings.sbx_eta_decay ** self.current_generation))
            c1_weights, c2_weights = evolution.sbx_crossover(p1, p2, current_eta)
            current_mutation_rate = max(self.settings.mutation_rate_min, self.settings.mutation_rate * (self.settings.mutation_rate_decay ** self.current_generation))
            evolution.mutation(c1_weights, current_mutation_rate, current_mutation_rate)
            evolution.mutation(c2_weights, current_mutation_rate, current_mutation_rate)
            next_pop.append(c1_weights)
            next_pop.append(c2_weights)

        # Update best fitness/agent.
        best_for_species = max(current_population, key=lambda x: x[1])
        print("Best for species: ")
        print(best_for_species)
        if best_for_species[1] > self.best_fitness + 0.01:
            self.best_fitness = best_for_species[1]
            self.best_agent = best_for_species[0]
            model = Model(chromosome=self.best_agent)
            model.save(f'{self.settings.folder}/agents/{self.current_generation}_{self.best_fitness}.npy')

        # add best agent to the population
        next_pop.append(self.best_agent)

        new_population = [Model(chromosome=chrom) for chrom in next_pop]
    
        if (self.current_generation < 25):
            new_population.append(Model())
            new_population.append(Model())

        random.shuffle(new_population)

        self.population = new_population

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        print(f"Generation {self.current_generation} complete. Fitnesses: {fitnesses}")
        generations_data = update_generations_data(self.settings, self.current_generation, self.plot_data)
        plot_generations(self.settings, generations_data)
        self.env.plot_data = {}
        # Evolve the population based on fitnesses.
        self.evolve_population(fitnesses)

    def train(self):
        for i in range(self.settings.generations_per_run):
            self.run_generation()
            # if i > 25 and i % 5 == 0:
            #     self.optimize_params()
            # Optionally, save best models or log additional statistics.

    def evaluate(self, model_paths = [], callback = noop):
        candidates = {}
        for model_path in model_paths:
            model = Model(
                chromosome=np.load(model_path['path'])
            )
            candidates[model_path['species']] = model

        return self.run(candidates, "cod", None, True, callback)
